# LeRobot 项目结构

**LeRobot** 是 Hugging Face 开发的一个基于 PyTorch 的开源机器人学习库，用于真实世界机器人的数据集、预训练策略和工具。

## 主要目录结构

```
lerobot/
├── src/lerobot/              # 核心源代码
│   ├── policies/             # 策略模型 (ACT, Diffusion, Pi0, GR00T等)
│   ├── datasets/             # LeRobotDataset 数据集加载
│   ├── robots/               # 机器人硬件抽象层
│   ├── motors/               # 电机控制
│   ├── cameras/              # 相机驱动
│   ├── teleoperators/        # 遥操作设备 (键盘、手柄等)
│   ├── envs/                 # 仿真环境 (Gymnasium)
│   ├── configs/              # 配置系统 (draccus)
│   ├── processor/            # 数据处理流水线
│   ├── transforms/           # 图像/数据变换
│   ├── transport/            # 通信传输层
│   ├── rl/                   # 强化学习算法
│   ├── scripts/              # 内部脚本
│   └── common/               # 通用工具
├── scripts/                  # CLI 入口 (lerobot-train, lerobot-eval等)
├── configs/                  # 策略配置
├── tests/                    # 单元测试
├── examples/                 # 示例教程
├── docs/                     # 文档
├── benchmarks/               # 性能测试
├── docker/                   # Docker配置
└── pyproject.toml            # 项目配置 (uv管理)
```

## 核心模块

| 模块 | 作用 |
|------|------|
| `policies/` | 实现了ACT、Diffusion、VQ-BeT、Pi0、GR00T等策略模型 |
| `datasets/` | LeRobotDataset 格式，支持 HF Hub 数据集流式传输 |
| `robots/` | 统一 Robot 接口，支持 SO100、Unitree G1、Reachy2 等 |
| `envs/` | 仿真环境集成 (LIBERO、MetaWorld) |
| `rl/` | 强化学习算法 (HIL-SERL、TDMPC) |

## 技术栈

- Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate)
- draccus (配置管理) · Gymnasium (仿真环境) · uv (包管理)

## 重点掌握文件

### 核心入口

| 文件 | 说明 |
|------|------|
| `src/lerobot/__init__.py` | 包入口，了解模块组织 |
| `scripts/` | CLI 入口 (lerobot-train, lerobot-eval, lerobot-record) |
| `pyproject.toml` | 依赖和项目配置 |

### 策略模块 (核心)

| 文件 | 说明 |
|------|------|
| `src/lerobot/policies/pretrained.py` | **PreTrainedPolicy 基类**，所有策略的父类 |
| `src/lerobot/policies/factory.py` | 策略工厂，按名称加载 |
| `src/lerobot/policies/act/` | ACT 策略实现 (入门首选) |
| `src/lerobot/policies/pi0/` | Pi0 策略实现 (VLA模型) |

### 数据集模块

| 文件 | 说明 |
|------|------|
| `src/lerobot/datasets/lerobot_dataset.py` | **LeRobotDataset 类**，核心数据结构 |
| `src/lerobot/datasets/lerobot_dataset_metadata.py` | 数据集元数据 |

### 机器人接口

| 文件 | 说明 |
|------|------|
| `src/lerobot/robots/robot.py` | **Robot 基类**，硬件抽象接口 |
| `src/lerobot/robots/myrobot.py` | 自定义机器人的参考实现 |

### 配置系统

| 文件 | 说明 |
|------|------|
| `src/lerobot/configs/train.py` | 训练配置 |
| `src/lerobot/configs/policies.py` | 策略配置基类 |

### 学习建议路径

```
1. pretrained.py → 理解策略基类
2. lerobot_dataset.py → 理解数据加载
3. robot.py → 理解硬件接口
4. 选定一个策略 (如 act/) → 深入理解实现
5. factory.py → 理解如何组合使用
```
