# PI05Config 逐行注解（与 PI0Config 的差异对比）

> 原始文件: `configuration_pi05.py`
> 与 pi0 相同的参数不再重复注解，只标注**差异点**

---

## 核心差异 1: 归一化方式 → QUANTILES

```python
# ⚠️ 与 pi0 的唯一关键差异
normalization_mapping: dict[str, NormalizationMode] = {
    "VISUAL": NormalizationMode.IDENTITY,
    "STATE": NormalizationMode.QUANTILES,    # pi0: NormalizationMode.MEAN_STD
    "ACTION": NormalizationMode.QUANTILES,   # pi0: NormalizationMode.MEAN_STD
}
```

**为什么改成 QUANTILES？**

```
MEAN_STD:
  公式: (x - mean) / std
  问题: 对异常值敏感。如果数据有离群点，mean 和 std 会被拉偏
        长视界任务（libero_10）中，动作范围变化大，MEAN_STD 会把
        大部分数据压缩到一个很小的区间，丢失分辨率

QUANTILES:
  公式: 2 * (x - q01) / (q99 - q01) - 1
  优点: q01 和 q99 只关注 1%~99% 的数据主体
        极端离群值不影响归一化参数
        数据被均匀映射到 [-1, 1]，保留更多分辨率
        对长序列推理尤其重要（这就是 libero_10 +41 点的原因之一）
```

---

## 核心差异 2: Tokenizer 最大长度 → 200

```python
tokenizer_max_length: int = 200              # pi0: 48
```

**为什么 pi05 需要更长的 token 序列？**

pi0 用连续状态投影 (`state_proj` 线性层)，状态直接映射到连续嵌入。pi05 用**离散状态 token**，把状态数值转成文本 token（如 "1.24" → token），需要更多 token 来表示同样的状态向量。200 vs 48 反映了 pi05 用更长的文本 token 序列编码状态。

---

## 模型结构差异（不在 config 里，在 modeling 里）

虽然 config 参数几乎一样，但两个 model 的实现有本质区别：

| 维度 | pi0 modeling | pi05 modeling |
|------|-------------|---------------|
| 状态编码 | `state_proj` 连续线性投影 | 离散 state tokenizer（文本 token） |
| VLM Norm | 标准 GemmaRMSNorm | **AdaRMS**（条件 RMSNorm） |
| 残差连接 | 标准 x + y | **门控残差** x + y * gate |
| Condition 注入 | `action_time_mlp` 直接加 | AdaRMS 的 scale/shift/gate 调制 |
| LoRA | 无 | 训练/推理时可选 LoRA |

---

## 其余参数（与 pi0 完全相同，仅列出）

```python
paligemma_variant: str = "gemma_2b"          # 同 pi0
action_expert_variant: str = "gemma_300m"    # 同 pi0
n_obs_steps: int = 1                         # 同 pi0
chunk_size: int = 50                         # 同 pi0
n_action_steps: int = 50                     # 同 pi0
max_state_dim: int = 32                      # 同 pi0
max_action_dim: int = 32                     # 同 pi0
num_inference_steps: int = 10                # 同 pi0 — 流匹配去噪步数
time_sampling_*: 同 pi0                      # 同 pi0 — 流匹配时间采样
optimizer_lr: float = 2.5e-5                 # 同 pi0
optimizer_betas: (0.9, 0.95)                 # 同 pi0
gradient_checkpointing: bool = False         # 同 pi0
```

**结论**：pi0 和 pi05 共享完全相同的训练超参数和流匹配机制。差异全在模型架构层：归一化方式 + Transformer 内部的 norm/residual 实现。
