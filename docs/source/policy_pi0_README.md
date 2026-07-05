# π₀ (pi0)
本仓库是 **π₀** 模型在 Hugging Face 平台的移植版本，基于 Physical Intelligence 团队的 [OpenPI](https://github.com/Physical-Intelligence/openpi) 项目适配开发。
该模型是一款用于**通用机器人控制的视觉-语言-动作（VLA）模型**。

---

## 模型概览

| 特性 | π₀ | π₀.₅ |
| ---- | ------------------------------------------------------ | ----------------------------------------- |
| 时间条件编码 | 通过 `action_time_mlp_*` 将时间与动作拼接 | 使用 `time_mlp_*` 实现 AdaRMS 条件编码 |
| AdaRMS 归一化 | 未使用 | 在动作专家模块中启用 |
| 分词器长度 | 48 个令牌 | 200 个令牌 |
| 离散状态输入 | 否（使用 `state_proj` 层） | 是 |
| 参数规模 | 更大（包含状态嵌入层） | 更小（无状态嵌入层） |

---

## 相对动作
π₀ 支持**相对动作**训练：模型学习机器人当前状态的**相对偏移量**，而非绝对关节坐标。
该机制与 OpenPI 中的 `DeltaActions` 相对动作变换一致，可有效提升模型性能。

### 工作原理
1. **预处理阶段**：将绝对动作转换为相对偏移量
   `相对值 = 动作值 - 状态值`（针对指定关节）
2. 基于相对动作的分布统计数据进行归一化
3. **后处理阶段**：将预测的相对动作还原为绝对动作
   `绝对值 = 相对值 + 状态值`

在 `relative_exclude_joints` 中指定的关节（如夹爪）将**保持绝对动作**，不参与相对转换。

### 配置参数

| 参数名称 | 类型 | 默认值 | 说明 |
| ------------------------- | ----------- | ------------- | ---------------------------------------------------------------- |
| `use_relative_actions` | `布尔值` | `False` | 启用相对动作训练 |
| `relative_exclude_joints` | `字符串列表` | `["gripper"]` | 保持绝对动作的关节名称（子串匹配） |
| `action_feature_names` | `字符串列表` | `None` | 运行时由 `make_policy` 自动从数据集元数据填充 |

### 训练示例
```bash
python -m lerobot.scripts.lerobot_train \
  --policy.type=pi0 \
  --dataset.repo_id=your_org/your_dataset \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]'
```

启用 `use_relative_actions=true` 后，训练脚本会**自动完成以下操作**：
- 从数据集中计算相对动作统计量（分块采样的相对动作）
- 用相对动作统计量替换标准动作统计量，用于归一化
- 在分布式训练中，将统计量同步到所有计算节点

### 为现有数据集重新计算统计量
如需离线预计算相对动作统计量，可使用 `lerobot.datasets` 中的 `recompute_stats` 工具：

```python
from lerobot.datasets import LeRobotDataset, recompute_stats

dataset = LeRobotDataset("your_org/your_dataset")
dataset = recompute_stats(
    dataset,
    relative_action=True,
    relative_exclude_joints=["gripper"],
)
```

---

## 引用说明
如果你使用了本项目，请同时引用 **OpenPI** 与 π₀ 论文：

```bibtex
@misc{openpi2024,
  author       = {Physical Intelligence Lab},
  title        = {OpenPI: PyTorch Implementation of π0 and π0.5 Policies},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}},
  license      = {Apache-2.0}
}

@misc{black2024pi0visionlanguageactionflowmodel,
  title        = {π₀: A Vision-Language-Action Flow Model for General Robot Control},
  author       = {Kevin Black and Noah Brown and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Lucy Xiaoyang Shi and James Tanner and Quan Vuong and Anna Walling and Haohuan Wang and Ury Zhilinsky},
  year         = {2024},
  eprint       = {2410.24164},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  url          = {https://arxiv.org/abs/2410.24164},
}
```

---

## 许可证
本移植版本遵循 **Apache 2.0 开源许可证**，与原始 [OpenPI 仓库](https://github.com/Physical-Intelligence/openpi) 保持一致。

---

### 专业术语对照
- **Vision-Language-Action (VLA)**: 视觉-语言-动作（机器人多模态模型）
- **Relative Actions**: 相对动作
- **Absolute Actions**: 绝对动作
- **AdaRMS**: 自适应均方根归一化（深度学习优化技术）
- **State Embedding**: 状态嵌入
- **Joint**: 机器人关节
- **Gripper**: 机器人夹爪/执行器