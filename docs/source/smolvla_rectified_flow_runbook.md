# SmolVLA Rectified Flow 改造运行文档

本文档用于在本地只修改代码，再同步到服务器运行 SmolVLA Rectified Flow 训练/评估。大模型权重不需要从本地同步到服务器。

## 1. 本次代码改动

- `policy.flow_objective` 默认改为 `rectified_flow`。
- Rectified Flow 训练路径：
  - `x_t = (1 - t) * noise + t * action`
  - `target = action - noise`
  - `t ~ Uniform(eps, 1 - eps)`，默认 `eps=1e-3`
- Rectified Flow 推理路径：
  - 从 `x_0 = noise` 开始
  - 用 Euler ODE 从 `t=0` 积分到 `t=1`
  - `policy.num_steps` 控制积分步数，默认仍为 `10`
- 保留旧实现用于对照实验：
  - `--policy.flow_objective=flow_matching`
  - `--policy.flow_time_sampling=beta`

## 2. 本地检查

Windows 本地如果没有完整训练环境，至少做语法检查：

```bash
python -m py_compile \
  src/lerobot/policies/smolvla/configuration_smolvla.py \
  src/lerobot/policies/smolvla/modeling_smolvla.py \
  src/lerobot/policies/rtc/modeling_rtc.py \
  tests/policies/smolvla/test_smolvla_rectified_flow.py
```

在服务器完整环境中建议运行：

```bash
uv run pytest tests/policies/smolvla/test_smolvla_rectified_flow.py -q
```

## 3. 同步代码到服务器

推荐用 `git` 同步代码，不提交模型权重。确认 `.gitignore` 已排除本地权重目录、缓存目录和训练输出目录，例如：

```gitignore
outputs/
checkpoints/
pretrained/
*.safetensors
*.bin
*.pt
*.pth
```

提交代码：

```bash
git status
git add src/lerobot/policies/smolvla/configuration_smolvla.py \
        src/lerobot/policies/smolvla/modeling_smolvla.py \
        src/lerobot/policies/rtc/modeling_rtc.py \
        tests/policies/smolvla/test_smolvla_rectified_flow.py \
        docs/source/smolvla_rectified_flow_runbook.md
git commit -m "Add rectified flow objective for SmolVLA"
git push origin <your-branch>
```

服务器拉取：

```bash
cd /path/to/lerobot
git fetch origin
git checkout <your-branch>
git pull --ff-only
```

如果不想走远端仓库，也可以用 `rsync`，但要排除大文件：

```bash
rsync -av --delete \
  --exclude ".git/" \
  --exclude "outputs/" \
  --exclude "checkpoints/" \
  --exclude "pretrained/" \
  --exclude "*.safetensors" \
  --exclude "*.bin" \
  --exclude "*.pt" \
  --exclude "*.pth" \
  ./ user@server:/path/to/lerobot/
```

## 4. 服务器环境准备

```bash
cd /path/to/lerobot
uv sync --locked --extra smolvla --extra test --extra dev
```

如果服务器已经安装好依赖和预训练模型，可以只确认当前环境可导入：

```bash
uv run python -c "from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig; print(SmolVLAConfig().flow_objective)"
```

输出应为：

```text
rectified_flow
```

## 5. Rectified Flow 微调命令

从服务器已有的 SmolVLA 预训练权重继续训练：

```bash
uv run lerobot-train \
  --policy.path=/path/to/server/pretrained/smolvla_base \
  --policy.flow_objective=rectified_flow \
  --policy.flow_time_sampling=uniform \
  --policy.flow_time_eps=0.001 \
  --policy.num_steps=10 \
  --dataset.repo_id=<USER>/<DATASET> \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/train/smolvla_rf
```

如果从配置新建 SmolVLA，而不是从 `policy.path` 读取：

```bash
uv run lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.flow_objective=rectified_flow \
  --policy.flow_time_sampling=uniform \
  --policy.num_steps=10 \
  --dataset.repo_id=<USER>/<DATASET> \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/train/smolvla_rf
```

## 6. 对照实验：旧 Flow Matching

旧行为的关键差异是时间方向相反、目标为 `noise - action`，并使用 Beta 时间采样。用于 ablation：

```bash
uv run lerobot-train \
  --policy.path=/path/to/server/pretrained/smolvla_base \
  --policy.flow_objective=flow_matching \
  --policy.flow_time_sampling=beta \
  --policy.flow_time_beta_alpha=1.5 \
  --policy.flow_time_beta_beta=1.0 \
  --dataset.repo_id=<USER>/<DATASET> \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/train/smolvla_fm_baseline
```

## 7. 推理与评估

使用训练出的 checkpoint：

```bash
uv run lerobot-eval \
  --policy.path=outputs/train/smolvla_rf/checkpoints/last/pretrained_model \
  --env.type=<ENV_TYPE> \
  --eval.n_episodes=50
```

如果你想测试更少/更多 ODE 步数：

```bash
uv run lerobot-eval \
  --policy.path=outputs/train/smolvla_rf/checkpoints/last/pretrained_model \
  --policy.num_steps=6 \
  --env.type=<ENV_TYPE> \
  --eval.n_episodes=50
```

建议先比较 `num_steps=10`、`8`、`6`。Rectified Flow 的目标是让轨迹更直，理想情况下可以用更少步数保持接近效果，但实际取决于数据集和微调时长。

## 8. 注意事项

- 直接把旧 SmolVLA 权重切到 `rectified_flow` 做零样本推理通常不可靠，因为时间 embedding 和速度符号都变了。建议至少做一次微调。
- 如果要严格复现原始 SmolVLA，请显式加上 `--policy.flow_objective=flow_matching --policy.flow_time_sampling=beta`。
- 训练输出和预训练权重只放服务器，不纳入 git。
- 修改后新增的单元测试不依赖 SmolVLA 大权重，适合先在服务器 CI/命令行快速验证。
