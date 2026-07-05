# normalize_processor.py 逐行注解

> 原始文件: `processor/normalize_processor.py`
> 核心内容: 四种归一化模式 + Normalizer/Unnormalizer 两个 ProcessorStep

---

## 一、归一化基类 `_NormalizationMixin` (行 40-400)

这个 Mixin 是 NormalizerProcessorStep 和 UnnormalizerProcessorStep 共享的核心逻辑。

### 1.1 数据结构

```python
@dataclass
class _NormalizationMixin:
    features: dict[str, PolicyFeature]                      # 特征定义 {"observation.state": PolicyFeature(...)}
    norm_map: dict[FeatureType, NormalizationMode]           # ⭐ 归一化模式映射：STATE→QUANTILES, ACTION→MEAN_STD...
    stats: dict[str, dict[str, Any]] | None = None           # 统计数据 {"observation.state": {"q01": [...], "q99": [...]}}
    eps: float = 1e-8                                        # 防止除零的小常数
    _tensor_stats: dict[str, dict[str, Tensor]]              # stats 的 Tensor 版本（在 GPU 上）
    _stats_explicitly_provided: bool                         # 标记：stats 是否是用户手动提供的
```

### 1.2 构造函数 `__post_init__` (行 103-136)

```python
def __post_init__(self):
    # 1. 标记 stats 是否由用户显式提供（用于 override 保护）
    self._stats_explicitly_provided = self.stats is not None and bool(self.stats)

    # 2. JSON 反序列化容错
    #    从 JSON 加载时，枚举变成字符串，tuple 变成 list
    #    这里把它们恢复成正确的 Python 类型
    if self.features:
        first_val = next(iter(self.features.values()))
        if isinstance(first_val, dict):                     # JSON 反序列化后的 dict
            reconstructed = {}
            for key, ft_dict in self.features.items():
                reconstructed[key] = PolicyFeature(
                    type=FeatureType(ft_dict["type"]),       # "STATE" → FeatureType.STATE
                    shape=tuple(ft_dict["shape"])            # [8] → (8,)
                )
            self.features = reconstructed

    if self.norm_map and all(isinstance(k, str) for k in self.norm_map):
        # "STATE" → FeatureType.STATE, "QUANTILES" → NormalizationMode.QUANTILES
        ...

    # 3. 把 stats 字典转成 GPU 上的 Tensor，一次分配，推理时零开销
    self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
```

### 1.3 设备管理 `to()` (行 138-155)

```python
def to(self, device, dtype):
    self.device = device
    self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
    # 把 stats tensor 搬到 device 上
```

### 1.4 状态字典 `state_dict()` / `load_state_dict()` (行 157-222)

```python
def state_dict(self):
    # 把 Tensor stats 展平保存
    # {"observation.state": {"q01": tensor, "q99": tensor}}
    # → {"observation.state.q01": tensor, "observation.state.q99": tensor}

def load_state_dict(self, state):
    # ⭐ Override 保护机制：
    # 如果 stats 是用户手动提供的（_stats_explicitly_provided=True），
    # 则忽略 state_dict 中的 stats，保留用户提供的值
    # 这允许从新数据集重新计算 stats 而不破坏模型权重
    if self._stats_explicitly_provided:
        return  # 保留用户提供的 stats
    # 否则从 state_dict 加载
```

---

## 二、核心归一化逻辑 `_apply_transform()` (行 281-400)

这是整个文件最重要的函数。支持 4 种归一化模式：

### 2.1 MEAN_STD 模式 (行 328-341)

```python
if norm_mode == NormalizationMode.MEAN_STD:
    mean, std = stats["mean"], stats["std"]
    denom = std + eps                    # 防除零

    # 归一化（数据 → [-∞,∞] 正太化）
    if inverse:
        return tensor * std + mean       # 反归一化：从标准化空间回到原始空间
    return (tensor - mean) / denom       # 归一化：(x - μ) / σ

# 适用场景：数据分布接近正态分布
# 问题：异常值会拉偏 mean/std，导致大部分数据被压缩
```

### 2.2 MIN_MAX 模式 (行 343-363)

```python
if norm_mode == NormalizationMode.MIN_MAX:
    min_val, max_val = stats["min"], stats["max"]
    denom = max_val - min_val
    denom = torch.where(denom == 0, eps, denom)  # 常数特征防除零

    # 归一化（数据 → [-1, 1]）
    if inverse:
        return (tensor + 1) / 2 * denom + min_val  # [-1,1] → [min,max]
    return 2 * (tensor - min_val) / denom - 1       # [min,max] → [-1,1]

# 适用场景：数据有明确物理范围
# 问题：和 MEAN_STD 一样对异常值敏感
```

### 2.3 QUANTILES 模式 (行 365-380) ⭐ pi05 用这个

```python
if norm_mode == NormalizationMode.QUANTILES:
    q01 = stats["q01"]  # 第 1 百分位数
    q99 = stats["q99"]  # 第 99 百分位数
    denom = q99 - q01
    denom = torch.where(denom == 0, eps, denom)

    # 归一化（数据 → [-1, 1]）
    if inverse:
        return (tensor + 1.0) * denom / 2.0 + q01   # [-1,1] → [q01,q99]
    return 2.0 * (tensor - q01) / denom - 1.0        # [q01,q99] → [-1,1]

# ⭐ 关键优势：
#    - 只用 q01（第1百分位）和 q99（第99百分位），忽略最极端的 1% 数据
#    - 数据主体的 98% 被均匀映射到 [-1, 1]
#    - 异常值（<q01 或 >q99）会被映射到 <-1 或 >1，但在合理范围内
#    - 对于长视界任务（动作分布跨度大），保留更多有效分辨率
#
# 为什么 libero_10 上 +41 分？
#   长序列动作的数值范围比短序列大很多。MEAN_STD 被离群值拉偏后，
#   大部分正常动作被挤压到很小的归一化区间（比如 [-0.3, 0.3]），
#   模型在这个压缩空间里预测动作，误差被放大。
#   QUANTILES 保证主体数据始终在 [-1, 1]，模型有完整的预测空间。
```

### 2.4 QUANTILE10 模式 (行 382-397)

```python
if norm_mode == NormalizationMode.QUANTILE10:
    q10 = stats["q10"]
    q90 = stats["q90"]
    # 同 QUANTILES 逻辑，但用 q10/q90（中间 80% 数据）
    # 更保守的异常值过滤
```

---

## 三、NormalizerProcessorStep (行 403-474)

```python
@ProcessorStepRegistry.register(name="normalizer_processor")
class NormalizerProcessorStep(_NormalizationMixin, ProcessorStep):

    def __call__(self, transition):
        # 对 observation 做归一化：原始数据 → 归一化空间
        new_transition[OBSERVATION] = self._normalize_observation(obs, inverse=False)
        # 对 action 做归一化：原始数据 → 归一化空间
        new_transition[ACTION] = self._normalize_action(action, inverse=False)
        return new_transition
```

**在 pi0/pi05 的 preprocessor 流水线中，它在 tokenizer 之后**：
```
观测 → RenameObs → ToBatch → Tokenizer → Device → Normalizer → 模型
```

---

## 四、UnnormalizerProcessorStep (行 477-535)

```python
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):

    def __call__(self, transition):
        # 对 observation 做反归一化：归一化空间 → 原始空间
        new_transition[OBSERVATION] = self._normalize_observation(obs, inverse=True)
        # ⭐ 对 action 做反归一化：模型输出 → 机器人可执行的动作
        new_transition[ACTION] = self._normalize_action(action, inverse=True)
        return new_transition
```

**在 postprocessor 流水线中**：
```
模型输出 → Unnormalizer → AbsoluteActions → 机器人/环境
```

---

## 五、hotswap_stats() (行 538-563)

```python
def hotswap_stats(policy_processor, stats):
    """热替换归一化统计量，不需要重建整个流水线"""
    rp = deepcopy(policy_processor)          # 深拷贝流水线
    for step in rp.steps:
        if isinstance(step, _NormalizationMixin):
            step.stats = stats               # 替换 stats
            step._tensor_stats = to_tensor(stats, ...)  # 重新生成 Tensor
    return rp

# 使用场景：把在 A 数据集上训练的模型适配到 B 数据集
# 不需要重新训练，只需替换归一化统计量
```

---

## 总结：归一化模式选择指南

| 模式 | 何时用 | 优点 | 缺点 |
|------|--------|------|------|
| MEAN_STD | 数据接近正态分布 | 简单，稳定 | 对离群值敏感 |
| MIN_MAX | 数据有明确最小/最大值 | 保证 [-1,1] 范围 | 极端值支配范围 |
| **QUANTILES** | 长尾分布，长序列推理 | 鲁棒，保留分辨率 | 需要计算百分位数 |
| QUANTILE10 | 极度长尾分布 | 最鲁棒 | 丢弃更多边界信息 |
