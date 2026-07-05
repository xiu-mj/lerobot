# modeling_pi0.py 逐行注解

> 原始文件: `policies/pi0/modeling_pi0.py`（1325 行）
> 核心流程: 图像+文本+状态 → PaliGemma VLM → 动作专家 → 流匹配去噪 → 动作

---

## 一、整体架构

```
输入:
  images (2个视角, 224×224) ──→ SigLIP vision encoder ──→ image embeddings
  文本 (tokenized任务描述)    ──→ embedding layer        ──→ text embeddings
  状态 (8维, padded to 32)     ──→ state_proj线性层      ──→ state embeddings

                           ↓  拼接成 prefix_embs  ↓

  PaliGemma VLM (标准 Gemma decoder, NO AdaRMS)
        │
        │  输出 prefix_hidden_states
        │
  ┌─────┴─────────────────────────────────────────┐
  │  动作专家 (标准 Gemma decoder, NO AdaRMS)      │
  │       输入: noise + time_emb + state_emb       │
  │       输出: 预测的速度场 v_t                   │
  └───────────────────────────────────────────────┘
        │
        ▼
  action_out_proj (1024 → 32) → 7维动作(去填充)
```

---

## 二、工具函数

### 2.1 正弦时间嵌入 `create_sinusoidal_pos_embedding()` (行 84-101)

```python
def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period):
    """
    把标量时间步 t ∈ [0.001, 1.0] 编码为 dim 维向量
    用于告诉动作专家"当前处于去噪的哪个阶段"

    原理: 在 [min_period, max_period] 范围内均匀采样 dim/2 个频率
          对每个频率计算 sin(2π * t / period) 和 cos(2π * t / period)
          拼接得到 dim 维嵌入

    例: min_period=4e-3, max_period=4.0
        低频分量(period=4.0):  sin/cos 在大时间尺度上变化 → 感知全局去噪进度
        高频分量(period=4e-3): sin/cos 在小时间尺度上变化 → 感知精细时间信息
    """
```

### 2.2 Beta 时间采样 `sample_beta()` (行 104-109)

```python
def sample_beta(alpha, beta, bsize, device):
    """训练时从 Beta(1.5, 1.0) 分布采样时间步 t
       这样低噪声(t≈0)和高噪声(t≈1)的样本更多
       中间噪声区间的样本较少——因为中间阶段最好学"""
```

### 2.3 2D 注意力掩码 `make_att_2d_masks()` (行 112-141)

```python
def make_att_2d_masks(pad_masks, att_masks):
    """
    构建因果注意力掩码，支持 prefix-LM 模式:
    - 图像/文本 token (att=0): 不产生因果关系（互相可以注意）
      → 形成"prefix"：所有 prefix token 之间双向注意
    - 动作 token (att=1): 因果注意力（只能注意之前的 token）
      → 动作序列内部是因果的（自回归）

    cumsum[:,None,:] <= cumsum[:,:,None]
    确保 att_mask 值更大的 token 不能注意 att_mask 值更小的 token
    """
```

---

## 三、PaliGemmaWithExpertModel (行 339-551)

这是结合 PaliGemma VLM 和动作专家的核心模型类。

### 3.1 初始化关键点

```python
class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, vlm_config, action_expert_config, use_adarms, ...):
        # use_adarms = [False, False]  ← pi0 不使用 AdaRMS
        #   [0] = VLM 是否用 AdaRMS
        #   [1] = 动作专家是否用 AdaRMS

        # 设置 AdaRMS 相关的 config 属性
        vlm_config_hf.text_config.use_adarms = use_adarms[0]       # False
        vlm_config_hf.text_config.adarms_cond_dim = ...            # None

        action_expert_config_hf.use_adarms = use_adarms[1]         # False
        action_expert_config_hf.adarms_cond_dim = ...              # None

        # 关键: 两个都使用 PiGemma 类（但 config 里 use_adarms=False）
        #       所以行为上退化到标准 Gemma
        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None  # 动作专家不需要 token embedding
```

### 3.2 精度管理 `to_bfloat16_for_selected_params()` (行 401-422)

```python
# 视觉路径保持 float32（避免 bf16 精度损失影响图像理解）
# 语言模型部分用 bfloat16（省显存）
params_to_keep_float32 = [
    "vision_tower",
    "multi_modal_projector",
    "input_layernorm",           # norm 层高精度
    "post_attention_layernorm",  # norm 层高精度
    "model.norm",                # 最终 norm 层高精度
]
```

### 3.3 前向传播 `forward()` (行 455-551)

```python
def forward(self, ..., inputs_embeds=[prefix_embs, suffix_embs], adarms_cond=[None, None]):
    """
    三种模式:
    1. inputs_embeds=[prefix, None]  → 只跑 VLM（推理时缓存 prefix KV）
    2. inputs_embeds=[None, suffix]  → 只跑动作专家
    3. inputs_embeds=[prefix, suffix] → 两个一起跑（训练时）

    逐层处理:
    for layer_idx in range(num_layers):
        # VLM 的第 i 层: 处理 prefix_embs
        hidden_states_0, gate = VLM.layers[i].input_layernorm(prefix_embs, cond=None)
        hidden_states_0 = VLM.layers[i].self_attn(hidden_states_0, ...)
        hidden_states_0 = prefix_embs + hidden_states_0  # 标准残差（无 gate）
        ...

        # 动作专家的第 i 层: 处理 suffix_embs
        hidden_states_1, gate = Expert.layers[i].input_layernorm(suffix_embs, cond=None)
        hidden_states_1 = Expert.layers[i].self_attn(hidden_states_1, ...)
        hidden_states_1 = suffix_embs + hidden_states_1  # 标准残差（无 gate）
        ...

    返回: [prefix_output, suffix_output], past_key_values
    """
```

---

## 四、PI0Pytorch (行 554-933)

### 4.1 关键子模块

```python
class PI0Pytorch(nn.Module):
    def __init__(self, config):
        # VLM + 动作专家
        self.paligemma_with_expert = PaliGemmaWithExpertModel(..., use_adarms=[False, False])

        # 动作投影层: max_action_dim(32) ↔ action_expert_width(1024)
        self.action_in_proj = nn.Linear(32, 1024)
        self.action_out_proj = nn.Linear(1024, 32)

        # ⭐ 状态投影层: 连续状态 → embedding 空间
        self.state_proj = nn.Linear(32, 1024)

        # ⭐ 时间+动作融合 MLP: [time_emb | action_emb] → fused embedding
        self.action_time_mlp_in = nn.Linear(2048, 1024)   # 2*1024 → 1024
        self.action_time_mlp_out = nn.Linear(1024, 1024)   # 1024 → 1024
```

### 4.2 embed_prefix (行 645-681) — 编码 VLM 输入

```python
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
    """
    输入处理流程:
    1. 图像 → SigLIP vision encoder → image features
       (SigLIP: 224×224 → patch embedding → Transformer → 2048-dim embedding)
       feature = pooler_output * sqrt(hidden_size)  # 归一化投影

    2. 文本 → language_model.embed_tokens() → token embeddings

    3. 拼接: [image_embs | text_embs]

    4. 构建注意力掩码: prefix token 之间双向注意 (att=0)
    """
```

### 4.3 embed_suffix (行 698-749) — 编码动作专家输入

```python
def embed_suffix(self, state, noisy_actions, timestep):
    """
    输入处理流程:
    1. 时间步 → 正弦位置编码 → time_emb (1024维)

    2. 状态: state_proj(state) → state_emb (1024维)
       连续投影：8维状态 → 线性层 → 1024维空间

    3. 噪声动作: action_in_proj(noisy_actions) → action_emb (1024维)

    4. 融合: time_emb + state_emb + action_emb → action_time_emb
       action_time_mlp(concat(time_emb, action_emb)) → 最终融合

    5. adarms_cond = None  ← ⭐ pi0 不使用条件归一化

    6. 动作 token 使用因果注意力 (att=1)
    """
```

### 4.4 forward (行 751-808) — 训练前向

```python
def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions,
            noise=None, time=None):
    """
    ═══════════════════════════════════════════════════════════════
    流匹配训练流程:
    ═══════════════════════════════════════════════════════════════

    1. 采样噪声和时间:
       noise ~ N(0, I)
       time  ~ Beta(1.5, 1.0) * 0.999 + 0.001

    2. 构造含噪声的动作:
       x_t = t * noise + (1 - t) * actions
       # t=0 → 纯净动作
       # t=1 → 纯噪声
       # t=0.5 → 半噪声半动作

    3. 目标速度场:
       u_t = noise - actions
       # 从噪声指向数据的"速度"，流匹配 ODE 的目标

    4. 编码 prefix (VLM):
       prefix_embs = embed_prefix(images, text_tokens)
       → 通过 PaliGemma VLM 处理

    5. 编码 suffix (动作专家):
       suffix_embs = embed_suffix(state, x_t, time)
       → 动作专家处理

    6. 拼接 prefix + suffix → 构建因果注意力掩码
       prefix token 双向注意（互相可见）
       suffix token 因果注意（只能看之前的）

    7. 通过联合模型 forward:
       [prefix_out, suffix_out] = paligemma_with_expert(inputs_embeds=[prefix, suffix])
       suffix_out = suffix_out[:, -chunk_size:]  # 取出动作部分

    8. 投影回动作空间:
       v_t = action_out_proj(suffix_out)  # 1024 → 32

    9. 损失: MSE(v_t, u_t)
       预测的速度场 vs 真实速度场
    """
```

### 4.5 sample_actions (行 810-933) — 推理去噪

```python
@torch.no_grad()
def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state,
                   noise=None, num_steps=None):
    """
    ═══════════════════════════════════════════════════════════════
    流匹配推理流程 (Euler 法去噪):
    ═══════════════════════════════════════════════════════════════

    1. 先跑一次 prefix (VLM)，缓存 KV:
       prefix_out, past_key_values = paligemma(prefix_embs, use_cache=True)
       # ⭐ 关键优化: prefix 在去噪过程中不变，只需编码一次

    2. 初始化纯噪声:
       x_t = noise ~ N(0, I)
       dt = -1.0 / num_steps  # 每步的时间增量

    3. 迭代去噪 (num_steps 步):
       for step in range(num_steps):
           time = 1.0 + step * dt     # t: 1.0 → 1/num_steps
           v_t = denoise_step(x_t, time, past_key_values, state)
           x_t = x_t + dt * v_t      # Euler 步: x_{t+dt} = x_t + dt * v_t

    ⭐ 为什么 num_inference_steps 影响结果？
       流匹配用 ODE 建模，Euler 法是一阶方法。
       步数太少 → 数值误差大 → 偏离真实解 → 动作不好。
       步数太多 → 理想情况下应该收敛到真实解，但 pi0 在实验中出现
       "步数越多越差"的现象，可能是因为：
       - 每步的模型预测误差会累积
       - 步数多时，中间步骤 t 接近 0.5 的概率更高，
         而模型在中间噪声水平可能训练不充分
    """
```

---

## 五、PI0Policy (行 935-1325)

```python
class PI0Policy(PreTrainedPolicy):
    """LeRobot 的 PI0 OpenPI 策略封装"""

    def __init__(self, config):
        self.model = PI0Pytorch(config)     # 核心模型
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, batch, reduction="mean"):
        """
        batch 包含:
        - observation.images.image (B,3,256,256)
        - observation.images.image2 (B,3,256,256)
        - observation.state (B,8) → padded to (B,32)
        - observation.language_tokens (B, 48)
        - action (B, 50, 7) → padded to (B, 50, 32)
        """
        # 调用 PI0Pytorch.forward → 返回每个时间步的 MSE
        per_step_loss = self.model.forward(images, ..., state, actions)
        return per_step_loss.mean()  # 默认求平均

    def select_action(self, batch):
        """推理入口: 给定观测 → 返回动作序列"""
        # 调用 PI0Pytorch.sample_actions → 去噪后的动作
        actions = self.model.sample_actions(images, ..., state,
                                            num_steps=self.config.num_inference_steps)
        return actions
```

---

## 六、pi0 数据流总览图

```
训练阶段:

  [256×256 RGB]──→SigLIP──→[img_emb]──┐
  [Task Text]  ──→Embed───→[txt_emb]──┤
  [8D State]   ──→state_proj─→[st_emb]─┤
                                        ├─→[prefix_embs]──→PaliGemma VLM──→[prefix_out]
                                        │
  [7D Action]  ──→action_in_proj──→[act_emb]─┐
  [Time t]     ──→sinusoidal──────→[time_emb]┤
                                              ├─→MLP fuse─→[suffix_embs]──→Action Expert──→[v_t]
                                                                                  │
  [Noise] ──→ x_t = t*noise + (1-t)*action ──────────────────────────────────────┤
  [u_t = noise - action] ───────────────────────────────── MSE(v_t, u_t) ←───────┘


推理阶段:

  prefix: (同训练) → PaliGemma VLM → cache KV ─┐
                                                    │
  noise ~ N(0,I) → x_0 = noise                    │
  for step in N steps:                             │
      suffix = embed(x_t, time) + state           │
      v_t = Expert(suffix, cached_KV)             │← 只跑动作专家
      x_{t+dt} = x_t + dt * v_t  (Euler步)       │
  → 最终动作序列
```
