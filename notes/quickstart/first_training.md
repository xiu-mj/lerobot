# 首次训练指南

本文档介绍如何在 LeRobot 中进行首次训练。

## 环境准备

### 1. 安装 LeRobot

```bash
# 基础安装
pip install lerobot

# 开发环境（推荐）
cd lerobot
uv sync --locked --extra test --extra dev
```

### 2. 验证安装

```bash
lerobot-info
```

## 使用 CLI 训练（推荐方式）

### 基础训练命令

```bash
lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.type=act \
  --output_dir=outputs/train/act_example \
  --job_name=act_example \
  --policy.device=cuda
```

### 完整训练示例（带日志和保存）

```bash
lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.type=act \
  --output_dir=outputs/train/act_example \
  --job_name=act_example \
  --policy.device=cuda \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=my_first_training \
  --policy.repo_id=your_username/act_policy
```

## 关键参数说明

| 参数 | 说明 |
|------|------|
| `--dataset.repo_id` | Hugging Face Hub 上的数据集 ID |
| `--policy.type` | 策略类型：act, diffusion, pi0, groot 等 |
| `--output_dir` | 训练输出目录 |
| `--policy.device` | 运行设备：cuda, cpu, mps |
| `--wandb.enable` | 是否启用 WandB 日志 |
| `--policy.repo_id` | 训练完成后推送到 Hub 的仓库 ID |

## 使用 Python 脚本训练

参考 `examples/tutorial/act/act_training_example.py`：

```python
from pathlib import Path
import torch
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

# 选择数据集
dataset_id = "lerobot/svla_so101_pickplace"
device = torch.device("cuda")  # 或 "cpu"

# 创建策略配置
dataset_metadata = LeRobotDatasetMetadata(dataset_id)
features = dataset_to_policy_features(dataset_metadata.features)

output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

cfg = ACTConfig(input_features=input_features, output_features=output_features)
policy = ACTPolicy(cfg)
preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

policy.train()
policy.to(device)

# 加载数据集
dataset = LeRobotDataset(dataset_id, delta_timestamps={"action": [0]})

# 创建优化器
optimizer = cfg.get_optimizer_preset().build(policy.parameters())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
for batch in dataloader:
    batch = preprocessor(batch)
    loss, _ = policy.forward(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 保存模型
output_directory = Path("outputs/my_policy")
policy.save_pretrained(output_directory)
```

## 训练输出

训练完成后，`output_dir` 目录会包含：

```
outputs/train/act_example/
├── checkpoint
│   ├── last/                    # 最新检查点
│   └── step_1000/              # 指定步数的检查点
├── config.json
├── model.safetensors
└── preprocessor.safetensors
```

## 在仿真环境训练

```bash
# LIBERO 仿真环境
lerobot-train \
  --policy.type=act \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

## 训练技巧

1. **从 ACT 开始**：ACT 训练快速、资源需求低，适合入门
2. **默认参数优先**：ACT 默认超参数对大多数任务效果良好
3. **GPU 内存**：batch_size 从 8 开始，根据显存调整
4. **训练时长**：100k 步大约需要几小时

## 下一步

- [策略复现教程](./tutorials/policy_reproduce.md)
- [仿真指南](./tutorials/simulation_guide.md)
- [API 参考](../api_reference/)
