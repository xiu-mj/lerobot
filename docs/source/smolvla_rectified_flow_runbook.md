# SmolVLA Rectified Flow 运行手册

这份文档适合你的场景：本地只改代码，然后把代码同步到服务器，在服务器上的 conda 环境里训练和评估。模型权重文件留在服务器，不需要同步回本地。

## 1. 本次改动

- `policy.flow_objective` 默认改成了 `rectified_flow`
- Rectified Flow 训练路径：
  - `x_t = (1 - t) * noise + t * action`
  - `target = action - noise`
  - `t ~ Uniform(eps, 1 - eps)`，默认 `eps=1e-3`
- Rectified Flow 推理路径：
  - 从 `x_0 = noise` 开始
  - 从 `t=0` 积分到 `t=1`
  - `policy.flow_solver` 可选择 `euler`（默认）或 `heun`
  - `policy.num_steps` 控制积分步数，默认还是 `10`
- 保留旧版 flow matching 作为对照：
  - `--policy.flow_objective=flow_matching`
  - `--policy.flow_time_sampling=beta`

## 2. 本地提交与同步

如果你只是想把代码推到 GitHub，然后在服务器拉取，直接走 git 就行。

提交代码：

```bash
git status
git add src/lerobot/policies/smolvla/configuration_smolvla.py \
        src/lerobot/policies/smolvla/modeling_smolvla.py \
        src/lerobot/policies/rtc/modeling_rtc.py \
        tests/policies/smolvla/test_smolvla_rectified_flow.py \
        docs/source/smolvla_rectified_flow_runbook.md \
        docs/source/policy_smolvla_README.md
git commit -m "Add rectified flow objective for SmolVLA"
git push origin main
```

如果你更习惯先拉到服务器再跑，也可以在服务器上执行：

```bash
cd /path/to/lerobot
git fetch origin
git checkout main
git pull --ff-only origin main
```

## 3. 服务器 conda 环境准备

把下面的 `smolvla` 换成你的环境名。

```bash
cd /path/to/lerobot
conda activate smolvla
```

如果这个环境还没装依赖，通常先在环境里执行一次：

```bash
pip install -e ".[smolvla,test,dev]"
```

检查当前环境能否导入配置：

```bash
python -c "from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig; print(SmolVLAConfig().flow_objective)"
```

正常应输出：

```text
rectified_flow
```

## 4. 语法与单测

先做一个轻量语法检查：

```bash
python -m py_compile \
  src/lerobot/policies/smolvla/configuration_smolvla.py \
  src/lerobot/policies/smolvla/modeling_smolvla.py \
  src/lerobot/policies/rtc/modeling_rtc.py \
  tests/policies/smolvla/test_smolvla_rectified_flow.py
```

再跑新增单测：

```bash
python -m pytest tests/policies/smolvla/test_smolvla_rectified_flow.py -q
```

## 5. Rectified Flow 训练命令

从服务器上的 SmolVLA 预训练权重继续训练：

```bash
python -m lerobot.scripts.lerobot_train \
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

如果你是从 `--policy.type=smolvla` 新建训练，而不是从已有 checkpoint 续训：

```bash
python -m lerobot.scripts.lerobot_train \
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

如果你想和原版对比，直接切回旧参数：

```bash
python -m lerobot.scripts.lerobot_train \
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

## 7. 评估命令

训练完成后，拿最后一个 checkpoint 做评估：

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path=outputs/train/smolvla_rf/checkpoints/last/pretrained_model \
  --env.type=<ENV_TYPE> \
  --eval.n_episodes=50
```

如果你想看看更少步数是否还能保持效果：

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path=outputs/train/smolvla_rf/checkpoints/last/pretrained_model \
  --policy.num_steps=6 \
  --env.type=<ENV_TYPE> \
  --eval.n_episodes=50
```

已有的 Rectified Flow checkpoint 不需要重新训练，可以直接切换到 Heun：

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path=outputs/train/smolvla_rf/checkpoints/last/pretrained_model \
  --policy.flow_solver=heun \
  --policy.num_steps=4 \
  --env.type=<ENV_TYPE> \
  --eval.n_episodes=50
```

Heun 每一步会调用模型两次，因此相同步数下推理时间和显存临时开销可能增加。公平比较计算量时，建议比较 Euler `num_steps=8` 与 Heun `num_steps=4`，以及 Euler `num_steps=4` 与 Heun `num_steps=2`。首轮实验可跑 Euler `4/8/10` 和 Heun `2/4/5`。

## 8. 需要注意的地方

- 旧的 SmolVLA 权重直接切到 `rectified_flow` 做零样本推理通常不稳，最好至少微调一下
- 如果要严格复现原来的行为，就显式加上 `--policy.flow_objective=flow_matching --policy.flow_time_sampling=beta`
- 训练输出和预训练权重留在服务器，不要纳入 git
- 我加的单测不依赖大权重，适合先在服务器上快速验证代码没坏
