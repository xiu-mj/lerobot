# PI0Config 逐行注解

> 原始文件: `configuration_pi0.py`
> 作用: 定义 pi0 流匹配模型的所有超参数，包括架构、训练、推理配置

---

## 装饰器和基类

```python
@PreTrainedConfig.register_subclass("pi0")  # 注册到 draccus 配置系统，CLI 用 --policy.type=pi0 来选中
@dataclass                                   # Python 数据类，自动生成 __init__/__repr__/__eq__
class PI0Config(PreTrainedConfig):           # 继承基类，有 input_features/output_features 等公共字段
```

---

## 模型架构参数

```python
paligemma_variant: str = "gemma_2b"          # VLM 主干模型大小
                                             #   "gemma_2b"  - 2B 参数 PaliGemma（视觉+语言）
                                             #   "gemma_300m" - 300M 参数（轻量版）

action_expert_variant: str = "gemma_300m"    # 动作专家模型大小
                                             #   动作专家是一个独立的 Gemma decoder，负责
                                             #   根据 VLM 编码的多模态特征预测动作序列

dtype: str = "float32"                       # 推理/训练精度
                                             #   "bfloat16" - 省显存，RTX 5880 等新卡首选
                                             #   "float32"  - 高精度，兼容旧硬件
```

---

## 时间维度参数

```python
n_obs_steps: int = 1                        # 输入观测步数 — pi0 只看当前帧，不做时序堆叠

chunk_size: int = 50                        # 预测的动作块大小（一次性预测 50 步动作）
                                            # 对应 OpenPI 源码中的 "action_horizon"

n_action_steps: int = 50                    # 每次执行多少步预测动作
                                            # 必须 ≤ chunk_size
                                            # = chunk_size 时不做 RTC（Real-Time Chunking）
                                            # < chunk_size 时做 RTC，逐块执行
```

> **chunk_size=50 的含义**：LIBERO 是 10 FPS，50 步 = 5 秒。模型一次预测未来 5 秒的所有动作。

---

## 状态/动作维度

```python
max_state_dim: int = 32                     # 状态向量的最大维度
                                            # LIBERO 真实状态是 8 维（eef_pos 3 + axis_angle 3 + gripper_qpos 2）
                                            # 不足 32 维的部分用 0 填充
                                            # 这样不同机器人的不同状态维度可以用同一模型

max_action_dim: int = 32                    # 动作向量的最大维度
                                            # LIBERO 真实动作是 7 维，同样填充到 32
```

---

## 流匹配参数（核心推理机制）

```python
num_inference_steps: int = 10               # ⭐ 推理时的去噪步数，就是实验中的 1/2/4/6/8/10
                                            # 越大 = 越精细 = 越慢
                                            # 训练时不使用（训练直接从数据加噪）

# 以下是流匹配的时间采样超参数，控制训练时噪声尺度的分布
time_sampling_beta_alpha: float = 1.5       # Beta 分布 α 参数
                                            # 控制时间采样倾向于 0（低噪声）还是 1（高噪声）
time_sampling_beta_beta: float = 1.0        # Beta 分布 β 参数
time_sampling_scale: float = 0.999          # 时间缩放
time_sampling_offset: float = 0.001         # 时间偏移，避免 t=0 的奇点

min_period: float = 4e-3                    # 正弦位置嵌入的最小周期
max_period: float = 4.0                     # 正弦位置嵌入的最大周期
                                            # 用于将时间步 t 编码为向量，注入动作专家
```

> **流匹配 vs 扩散**：流匹配直接用 ODE 从噪声→数据，比 DDPM 迭代更高效。训练时：采样 t→加噪→预测速度场。推理时：从纯噪声开始，用 Euler 法迭代 num_inference_steps 步去噪。

---

## 相对动作

```python
use_relative_actions: bool = False           # 是否用相对动作（相对于当前末端执行器位姿）
                                             # LIBERO 数据集中默认是绝对动作
                                             # 相对动作可以提高跨场景泛化

relative_exclude_joints: list[str] = ["gripper"]
                                             # 不参与相对化处理的关节
                                             # 夹爪保持绝对控制（开/合），不需要相对化
```

---

## 归一化配置

```python
normalization_mapping: dict[str, NormalizationMode] = {
    "VISUAL": NormalizationMode.IDENTITY,    # 图像：不做归一化（已经归一化到 [0,1] 或 ImageNet 标准化）
    "STATE": NormalizationMode.MEAN_STD,     # ⭐ 状态：用均值和标准差归一化
    "ACTION": NormalizationMode.MEAN_STD,    # ⭐ 动作：用均值和标准差归一化
}
```

> **pi0 vs pi05 的关键差异就在这里** — pi05 把 STATE 和 ACTION 的 MEAN_STD 改成了 QUANTILES。

---

## 训练配置

```python
gradient_checkpointing: bool = False         # 梯度检查点：用计算换显存
                                             # 开启后显存减半但训练慢 ~20%

compile_model: bool = False                  # torch.compile：JIT 编译加速
compile_mode: str = "max-autotune"           # 编译模式选项

device: str | None = None                    # 设备，None = 自动检测

freeze_vision_encoder: bool = False          # 冻结视觉编码器（微调时只训练动作专家）
train_expert_only: bool = False              # 冻结整个 VLM（更激进的微调策略）
```

---

## 优化器和学习率

```python
optimizer_lr: float = 2.5e-5                 # 峰值学习率，非常小的值
                                             # 因为是从预训练模型微调，大 lr 会破坏预训练权重

optimizer_betas: tuple[float, float] = (0.9, 0.95)
                                             # AdamW β 参数
                                             # β1=0.9（一阶矩衰减）
                                             # β2=0.95（二阶矩衰减，比默认 0.999 更激进）

optimizer_eps: float = 1e-8                  # 数值稳定性
optimizer_weight_decay: float = 0.01         # L2 正则化强度
optimizer_grad_clip_norm: float = 1.0        # 梯度裁剪阈值

scheduler_warmup_steps: int = 1_000          # 学习率预热步数
scheduler_decay_steps: int = 30_000          # 余弦衰减到最低点的步数
scheduler_decay_lr: float = 2.5e-6           # 衰减终点学习率（峰值的 1/10）
```

---

## Tokenizer

```python
tokenizer_max_length: int = 48               # PaliGemma tokenizer 最大长度
                                             # "状态: [1.2, 3.4, ...]" → tokenize → 48 tokens
                                             # pi05 把这个改成了 200（更长的文本描述）
```

## 验证逻辑 (__post_init__)

```python
def __post_init__(self):
    if self.n_action_steps > self.chunk_size:       # 执行步数不能超过预测步数
        raise ValueError(...)
    if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:  # 只支持两种 PaliGemma
        raise ValueError(...)
    if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:  # 动作专家也只支持两种
        raise ValueError(...)
    if self.dtype not in ["bfloat16", "float32"]:   # 只支持两种精度
        raise ValueError(...)
```
