# pi_gemma.py 逐行注解

> 原始文件：`policies/pi_gemma.py`（364 行）
> 这是 pi0 和 pi05 的**共享模型基础设施**。pi0 用标准 GemmaRMSNorm，pi05 用这里的 AdaRMS + 门控残差。

---

## 一、门控残差 `_gated_residual()` (行 54-66)

```python
def _gated_residual(
    x: torch.Tensor | None,    # 残差输入（通常是上一层的输出/原始输入）
    y: torch.Tensor | None,    # 新计算的值（attention 或 MLP 的输出）
    gate: torch.Tensor | None, # ⭐ 门控向量，由 AdaRMS 的 cond 投影得到
) -> torch.Tensor | None:
    """
    门控残差：控制"新的信息 y 有多少可以通过"
    - gate=None  → 标准残差: x + y     （pi0 的行为）
    - gate≠None  → 门控残差: x + y * gate （pi05 的行为）
    """
    if x is None and y is None:           # 两个都没有 → 返回 None
        return None
    if x is None or y is None:            # 只有一个 → 返回有的那个
        return x if x is not None else y
    if gate is None:                      # ⭐ 无 gate → 标准残差
        return x + y
    return x + y * gate                   # ⭐ 有 gate → y 按 gate 缩放
                                          # gate 接近 1 → 新信息完全通过（像标准残差）
                                          # gate 接近 0 → 新信息被屏蔽（保持原值不变）
```

**为什么门控有用？**
> 在去噪过程中，不同时间步需要不同程度的特征更新。早期步（t 大，噪声多）需要大幅更新，晚期步（t 小，接近干净数据）应该微调。gate 让网络能根据当前时间步**自适应控制信息流动**。

---

## 二、条件 LayerNorm 调度 `layernorm_forward()` (行 69-82)

```python
def layernorm_forward(
    layernorm: nn.Module,              # PiGemmaRMSNorm 实例
    x: torch.Tensor,                   # 输入
    cond: torch.Tensor | None = None,  # ⭐ 条件向量（time embedding）
):
    if cond is not None:
        return layernorm(x, cond=cond)  # 条件模式：传入 cond 做自适应调制
    else:
        return layernorm(x)             # 标准模式：退化到普通 RMSNorm
```

---

## 三、⭐ AdaRMS 核心 `PiGemmaRMSNorm` (行 85-133)

这是 pi05 相比 pi0 最核心的架构改进。

### 3.1 初始化 `__init__()`

```python
class PiGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        if cond_dim is not None:
            # ⭐ 自适应模式：用一个线性层把 cond_dim 维的 condition
            #    投影到 dim*3（scale + shift + gate 各 dim 维）
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)  # 零初始化 → 训练初期退化为标准 RMSNorm
        else:
            # 标准模式：只有一个可学习的权重参数（和 GemmaRMSNorm 一样）
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None
```

**`cond_dim` 是什么？**
> 它是动作专家最后一层时间嵌入的维度，通常等于 `action_expert 的 hidden_size`。对于 gemma_300m 就是 1024。每个时间步 t 会产生不同的 condition 向量。

### 3.2 RMS 归一化 `_norm()`

```python
def _norm(self, x):
    # 在 float32 下计算（比 bfloat16 精度更高，和 OpenPI 源码一致）
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)  # 逐样本方差
    normed_inputs = x * torch.rsqrt(var + self.eps)                  # RMS 归一化
    return normed_inputs
```

**RMSNorm vs LayerNorm**：RMSNorm 只做缩放（去均值+除标准差），不做平移（减均值），计算更快，Gemma 全系列都用这个。

### 3.3 前向传播 `forward()` ⭐

```python
def forward(self, x, cond=None) -> tuple[torch.Tensor, torch.Tensor | None]:
    dtype = x.dtype
    normed = self._norm(x)                              # 步骤1: 标准 RMS 归一化

    if cond is None or self.dense is None:
        # ⭐ 无 cond 时 → 标准 RMSNorm（pi0 的行为）
        normed = normed * (1.0 + self.weight.float())
        return normed.type_as(x), None                  # gate=None → 普通残差

    # ⭐ 有 cond 时 → 自适应 AdaRMS（pi05 的行为）
    modulation = self.dense(cond)                        # 步骤2: cond → [scale, shift, gate]
                                                        #        dense 输出 dim*3

    # 如果 x 是 3D (batch, seq, dim)，给 modulation 加 seq 维度
    if len(x.shape) == 3:
        modulation = modulation.unsqueeze(1)             # (B, dim*3) → (B, 1, dim*3)

    scale, shift, gate = modulation.chunk(3, dim=-1)    # 步骤3: 拆分成三组，各 dim 维

    normed = normed * (1 + scale.float()) + shift.float()  # 步骤4: 自适应调制
    #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #         scale 控制"缩放多少"（拉伸/压缩特征）
    #         shift 控制"平移到哪"（改变特征中心）
    #         两者都来自 condition，所以：
    #         - 不同时间步 → 不同的 scale/shift → 不同的归一化行为

    return normed.to(dtype), gate.to(dtype)              # 步骤5: 返回 (调制后的特征, gate)
                                                        #        gate 用于 _gated_residual
```

**完整数据流**：
```
输入 x (噪声动作序列特征)
       │
       ▼
   _norm(x) → 标准 RMS 归一化
       │
       ▼
   cond → dense → [scale, shift, gate]
       │
       ▼
   output = normed * (1+scale) + shift     ← 条件调制
       │
       ▼
   返回 (output, gate)
       │
       ▼
   _gated_residual(residual, output, gate) ← gate 控制残差信息流
```

---

## 四、PiGemma Decoder Layer (行 136-188)

```python
class _PiGemmaDecoderLayerBase(GradientCheckpointingLayer):

    def __init__(self, config, layer_idx):
        self.self_attn = GemmaAttention(...)                       # 标准 Gemma 注意力
        self.mlp = GemmaMLP(...)                                   # 标准 Gemma MLP

        # ⭐ 把标准 GemmaRMSNorm 替换为 PiGemmaRMSNorm
        cond_dim = config.adarms_cond_dim if config.use_adarms else None
        self.input_layernorm = PiGemmaRMSNorm(hidden_size, cond_dim=cond_dim)
        self.post_attention_layernorm = PiGemmaRMSNorm(hidden_size, cond_dim=cond_dim)

    def forward(self, hidden_states, ..., adarms_cond=None):
        # ===== Pre-Attention Norm + Self-Attention =====
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
        #                    ^^^^^^^^^^ 条件归一化，返回 gate
        hidden_states, _ = self.self_attn(hidden_states, ...)
        hidden_states = _gated_residual(residual, hidden_states, gate)
        #               ^^^^^^^^^^^^^^ 门控残差

        # ===== Post-Attention Norm + MLP =====
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _gated_residual(residual, hidden_states, gate)

        return hidden_states
```

**与标准 Gemma DecoderLayer 的区别**：
```
标准:  x → RMSNorm → Attention → + (残差) → RMSNorm → MLP → + (残差)
AdaRMS: x → AdaRMS(cond)→ Attention → + gate*(残差) → AdaRMS(cond)→ MLP → + gate*(残差)
                                  ↑                                    ↑
                              门控调制                              门控调制
```

---

## 五、PiGemmaModel (行 193-320)

```python
class PiGemmaModel(GemmaModel):
    """把 GemmaModel 的 decoder layers 和 final norm 替换为 PiGemma 版本"""

    def __init__(self, config):
        super().__init__(config)
        # 把所有 decoder layer 替换为 PiGemmaDecoderLayer
        self.layers = nn.ModuleList(
            [PiGemmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # 最终 norm 也换成 PiGemmaRMSNorm
        self.norm = PiGemmaRMSNorm(hidden_size, cond_dim=cond_dim)

    def forward(self, ..., adarms_cond=None):
        # 对每一层传递 adarms_cond
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, ..., adarms_cond=adarms_cond)
        # 最终 norm 也用条件调制
        hidden_states, _ = self.norm(hidden_states, adarms_cond)
        return hidden_states
```

---

## 六、组装类 (行 323-362)

```python
class PiGemmaForCausalLM(GemmaForCausalLM):
    """动作专家的基类：使用 PiGemmaModel 作为 decoder backbone"""
    def __init__(self, config):
        super().__init__(config)
        self.model = PiGemmaModel(config)    # 替换为 PiGemma decoder

class PaliGemmaModelWithPiGemma(PaliGemmaModel):
    """PaliGemma VLM，其 language_model 使用 PiGemma decoder"""
    def __init__(self, config):
        super().__init__(config)
        self.language_model = PiGemmaModel(config.text_config)

class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    """最终给 pi05 使用的多模态模型：
       - vision_encoder: SigLIP（标准）
       - language_model: PiGemma decoder（含 AdaRMS + gated_residual）
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = PaliGemmaModelWithPiGemma(config)
```

---

## 总结：pi0 用哪些，pi05 用哪些？

| 组件 | pi0 | pi05 |
|------|-----|------|
| VLM backbone | 标准 `PaliGemmaForConditionalGeneration` | `PaliGemmaForConditionalGenerationWithPiGemma` |
| DecoderLayer | 标准 GemmaDecoderLayer | PiGemmaDecoderLayer |
| Norm | 标准 `GemmaRMSNorm` | ⭐ `PiGemmaRMSNorm` (AdaRMS) |
| 残差连接 | `x + y` | ⭐ `x + y * gate`（门控残差） |
| Condition 路径 | `action_time_mlp` 直接加到特征 | `adarms_cond` 注入每层 norm |
