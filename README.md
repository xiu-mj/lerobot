我将为您翻译这份LeRobot文档并输出为Markdown格式。这是一份关于Hugging Face机器人学习库的README文档。

```markdown
<p align="center">
  <img alt="LeRobot, Hugging Face机器人库" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![测试](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/latest_deps_tests.yml?query=branch%3Amain)
[![测试](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/docker_publish.yml?query=branch%3Amain)
[![Python版本](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![许可证](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![状态](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![版本](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![贡献者公约](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** 旨在为真实世界机器人提供基于PyTorch的模型、数据集和工具。目标是降低入门门槛，让每个人都能为共享数据集和预训练模型做出贡献并从中受益。

🤗 一个硬件无关的、原生Python的接口，可跨多样化平台标准化控制，从低成本机械臂（SO-100）到人形机器人。

🤗 标准化的、可扩展的LeRobotDataset格式（Parquet + MP4或图像），托管在Hugging Face Hub上，实现大规模机器人数据集的高效存储、流式传输和可视化。

🤗 经过验证可迁移到真实世界的最先进策略，可直接用于训练和部署。

🤗 全面支持开源生态系统，推动物理AI民主化。

## 快速开始

LeRobot可以直接从PyPI安装。

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> 详细的安装指南，请参阅[安装文档](https://huggingface.co/docs/lerobot/installation)。

## 机器人与控制

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2演示">
</div>

LeRobot提供了一个统一的`Robot`类接口，将控制逻辑与硬件细节解耦。它支持广泛的机器人和遥操作设备。

```python
from lerobot.robots.myrobot import MyRobot

# 连接机器人
robot = MyRobot(config=...)
robot.connect()

# 读取观测并发送动作
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**支持的硬件：** SO100、LeKiwi、Koch、HopeJR、OMX、EarthRover、Reachy2、游戏手柄、键盘、手机、OpenARM、Unitree G1。

虽然这些设备已原生集成到LeRobot代码库中，但该库设计为可扩展的。您可以轻松实现Robot接口，以利用LeRobot的数据收集、训练和可视化工具来适配您自己的自定义机器人。

详细的硬件设置指南，请参阅[硬件文档](https://huggingface.co/docs/lerobot/integrate_hardware)。

## LeRobot数据集

为解决机器人领域的数据碎片化问题，我们采用**LeRobotDataset**格式。

- **结构：** 同步的MP4视频（或图像）用于视觉数据，Parquet文件用于状态/动作数据。
- **HF Hub集成：** 在[Hugging Face Hub](https://huggingface.co/lerobot)上探索数千个机器人数据集。
- **工具：** 无缝删除片段、按索引/比例分割、添加/删除特征、合并多个数据集。

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 从Hub加载数据集
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# 访问数据（自动处理视频解码）
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

了解更多信息，请参阅[LeRobotDataset文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)。

## 最先进模型

LeRobot使用纯PyTorch实现了最先进的策略，涵盖模仿学习、强化学习和视觉-语言-动作（VLA）模型，更多模型即将推出。它还为您提供工具来监测和检查训练过程。

<p align="center">
  <img alt="Gr00t架构" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

训练策略只需运行脚本配置：

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| 类别 | 模型 |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **模仿学习** | [ACT](./docs/source/policy_act_README.md)、[Diffusion](./docs/source/policy_diffusion_README.md)、[VQ-BeT](./docs/source/policy_vqbet_README.md)、[多任务DiT策略](./docs/source/policy_multi_task_dit_README.md) |
| **强化学习** | [HIL-SERL](./docs/source/hilserl.mdx)、[TDMPC](./docs/source/policy_tdmpc_README.md) 和 QC-FQL（即将推出） |
| **VLA模型** | [Pi0Fast](./docs/source/pi0fast.mdx)、[Pi0.5](./docs/source/pi05.mdx)、[GR00T N1.5](./docs/source/policy_groot_README.md)、[SmolVLA](./docs/source/policy_smolvla_README.md)、[XVLA](./docs/source/xvla.mdx) |

与硬件类似，您可以轻松实现自己的策略，并利用LeRobot的数据收集、训练和可视化工具，将您的模型分享到HF Hub。

详细的策略设置指南，请参阅[策略文档](https://huggingface.co/docs/lerobot/bring_your_own_policies)。

## 推理与评估

使用统一的评估脚本在仿真或真实硬件上评估您的策略。LeRobot支持标准基准测试，如**LIBERO**、**MetaWorld**等，更多基准即将推出。

```bash
# 在LIBERO基准上评估策略
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

了解如何实现自己的仿真环境或基准测试，并通过HF Hub分发，请参阅[EnvHub文档](https://huggingface.co/docs/lerobot/envhub)。

## 资源

- **[文档](https://huggingface.co/docs/lerobot/index)：** 完整的教程和API指南。
- **[中文教程：LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** 详细的组装、遥操作、数据集、训练、部署文档。由Seed Studio和5位全球黑客马拉松选手验证。
- **[Discord](https://discord.gg/q8Dzzpym3f)：** 加入`LeRobot`服务器与社区讨论。
- **[X](https://x.com/LeRobotHF)：** 在X上关注我们，获取最新动态。
- **[机器人学习教程](https://huggingface.co/spaces/lerobot/robot-learning-tutorial)：** 免费的实践课程，使用LeRobot学习机器人学习。

## 引用

如果您在项目中使用LeRobot，请引用GitHub仓库以感谢持续开发和贡献者：

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

如果您引用我们的研究或学术论文，请同时引用我们的ICLR发表：

<details>
<summary><b>ICLR 2026论文</b></summary>

```bibtex
@inproceedings{cadenelerobot,
  title={LeRobot: An Open-Source Library for End-to-End Robot Learning},
  author={Cadene, Remi and Alibert, Simon and Capuano, Francesco and Aractingi, Michel and Zouitine, Adil and Kooijmans, Pepijn and Choghari, Jade and Russi, Martino and Pascal, Caroline and Palma, Steven and Shukor, Mustafa and Moss, Jess and Soare, Alexander and Aubakirova, Dana and Lhoest, Quentin and Gallou\'edec, Quentin and Wolf, Thomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2602.22818}
}
```

</details>

## 贡献

我们欢迎社区每个人的贡献！要开始使用，请阅读我们的[CONTRIBUTING.md](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md)指南。无论您是添加新功能、改进文档还是修复bug，您的帮助和反馈都弥足珍贵。我们对开源机器人的未来感到非常兴奋，迫不及待地想与您合作迎接下一个阶段——感谢您的支持！

<p align="center">
  <img alt="SO101视频" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>由<a href="https://huggingface.co/lerobot">LeRobot</a>团队在<a href="https://huggingface.co">Hugging Face</a>用❤️打造</sub>
</div>
```