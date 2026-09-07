# SmolVLA Rectified Flow + Continuous Consistency

本文档说明 `smolvla-rf-consistency` 分支新增了什么、模块在训练和推理的什么位置起作用，
以及如何在 LIBERO 4-suite 上与原始 RF Uniform 做严格对照实验。

实现提交为 `bed1e5d4`，基于纯 RF 提交 `468c4444`。默认
`flow_consistency_enabled=false`，因此关闭新开关时保持原 RF/FM 路径，不改变已有 checkpoint 的行为。

## 1. 研究问题与公平对照

实验只比较：

| 组别 | 训练目标 | 时间采样 | 新增条件 |
|---|---|---|---|
| RF Uniform 基线 | RF loss | uniform | 无 |
| RF Uniform + Continuous Consistency | RF loss + consistency loss | uniform | 相对步长 `delta_t` + EMA teacher |

处理组必须从与基线相同的 `smolvla_base` 开始，以相同数据、seed、batch size、学习率和总步数训练。
不要从 RF 80k checkpoint 继续微调后再与 RF 80k 比较，否则处理组拥有更多训练量，无法把差异归因于一致性目标。

旧 RF checkpoint 可以在该分支直接评估，但它没有训练过 `delta_t` 分支；这不能验证 Continuous
Consistency。验证新模块必须训练新的 checkpoint。

## 2. 算法逻辑

### 2.1 原始 Rectified Flow

给定动作 `a`、高斯噪声 `epsilon` 和 `t ~ Uniform[0, 1]`：

```text
x_t = (1 - t) * epsilon + t * a
u_rf = a - epsilon
L_rf = MSE(v_theta(x_t, t, delta_t=0), u_rf)
```

### 2.2 连续时间一致性目标

默认将 batch 的 25% 用作 consistency 样本，其余 75% 仍计算普通 RF loss。对 consistency 子集：

```text
t       = k / K,  k sampled from {0, ..., K-1}
delta_t ~ Uniform[0, 1]
t_next  = min(t + delta_t, 1)
```

使用同一动作和同一噪声分别构造 `x_t` 与 `x_t_next`。EMA teacher 在后一个时刻预测速度：

```text
v_next = v_ema(x_t_next, t_next, delta_t)
a_hat  = x_t_next + (1 - t_next) * v_next
u_ct   = (a_hat - x_t) / (1 - t)
```

student 在较早的时刻预测能直接指向 teacher 端点的平均速度：

```text
L_ct    = MSE(v_theta(x_t, t, delta_t), stop_gradient(u_ct))
L_total = L_rf + lambda_ct * L_ct
```

这里的目标不是单纯降低训练 loss，而是让一个网络学会根据目标积分间隔 `delta_t` 输出跨时间段的平均
速度，从而更适合 `N=1/2/4` 等少步数推理。

## 3. 代码改动

### 3.1 配置与校验

文件：`src/lerobot/policies/smolvla/configuration_smolvla.py`

新增参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `flow_consistency_enabled` | `false` | 打开连续一致性训练 |
| `flow_consistency_ratio` | `0.25` | batch 中 consistency 样本比例 |
| `flow_consistency_weight` | `1.0` | `L_ct` 的权重 |
| `flow_consistency_timesteps` | `10` | consistency 起点 `t=k/K` 的网格数 |
| `flow_consistency_ema_power` | `0.75` | EMA warm-up 幂指数 |
| `flow_consistency_ema_max_decay` | `0.9999` | EMA decay 上限 |
| `flow_consistency_use_ema_for_inference` | `true` | 评估时默认使用 EMA 模型 |

一致性训练只允许和 `flow_objective=rectified_flow` 组合；当前也禁止同时使用 `compile_model=true`。

### 3.2 `delta_t` 条件分支

文件：`src/lerobot/policies/smolvla/modeling_smolvla.py`

新增 `action_target_dt_mlp`。它先把 `delta_t` 做正弦位置编码，再通过两层 MLP，并以残差形式加入原有
action-time embedding。最后一层以零权重、零偏置初始化，因此刚创建模型时该分支不会立即扰动原 RF
输出。

普通 RF 样本使用 `delta_t=0`；consistency 样本使用训练时抽到的连续 `delta_t`；推理时使用真实 Euler
步长 `delta_t=1/N`。因此训练和推理对“本次速度要跨越多长时间”采用同一条件语义。

### 3.3 混合 batch 与 consistency target

`_continuous_consistency_batch()` 将一个 batch 划分为 RF 子集和 consistency 子集，并调用
`consistency_velocity_target()` 构造 EMA teacher 端点目标。两个子集的逐元素 loss 会分别缩放，使外层
求均值后得到 `L_rf + lambda_ct * L_ct`，而不是按 75%/25% 做简单凸组合。

训练日志新增：

- `loss_rf`：未经子集缩放的普通 RF MSE；
- `loss_consistency`：未经子集缩放的 consistency MSE。

### 3.4 EMA teacher、checkpoint 与推理

打开一致性训练时，`SmolVLAPolicy` 创建冻结的 `ema_model`：

- 每次 optimizer step 后由 `policy.update()` 更新；
- EMA 参数不会进入 optimizer；
- teacher 始终处于 eval 模式，target 在 `torch.no_grad()` 下生成；
- 新 checkpoint 同时保存 student、EMA 和 EMA step，可以恢复训练；
- 从旧 RF checkpoint 初始化时，如果没有 EMA 权重，会自动将 student 同步给 teacher；
- 默认评估 EMA；设置 `flow_consistency_use_ema_for_inference=false` 可评估 student。

由于 checkpoint 保存两套模型权重，文件大小和模型常驻显存会明显增加。服务器应先跑 smoke test 并记录
峰值显存；不要直接启动完整训练。当前实现以单 GPU/DDP 路径为主，未宣称已验证 FSDP。

### 3.5 单元测试

`tests/policies/smolvla/test_smolvla_rectified_flow.py` 新增了：

- teacher 端点速度公式测试；
- EMA decay warm-up 与上限测试；
- consistency 配置合法性测试；
- consistency 只能用于 Rectified Flow 的约束测试。

## 4. 服务器实验任务

### 4.1 拉取与预检

```bash
source /home/caimu/miniconda3/bin/activate lerobot
cd /home/caimu/lerobot

git fetch myfork
git switch smolvla-rf-consistency
git pull --ff-only myfork smolvla-rf-consistency
git log -2 --oneline

python -m pytest tests/policies/smolvla/test_smolvla_rectified_flow.py -q
```

确认代码提交至少包含 `bed1e5d4`。检查基线配置和输入文件：

```bash
BASE_MODEL=/home/caimu/lerobot/lerobot_models/smolvla_base
VLM=/home/caimu/lerobot/lerobot_models/SmolVLM2-500M-Video-Instruct
DATA_ROOT=/home/caimu/lerobot/lerobot_models/lerobot_libero
BASE_RUN=outputs/train/smolvla_libero_4suite_rf_uniform
BASE_CFG=$BASE_RUN/checkpoints/080000/pretrained_model/train_config.json

test -f "$BASE_MODEL/model.safetensors"
test -d "$DATA_ROOT"
test -f "$BASE_CFG"
python -c "import json; c=json.load(open('$BASE_CFG')); print(json.dumps(c, indent=2))"
```

必须以 `BASE_CFG` 为准确认基线确实是：4 suites、`seed=1000`、`batch_size=16`、`steps=100000`、
`save_freq=20000`、学习率 `1e-4`、chunk size 50、RF Uniform。若服务器真实配置与本文不同，保留真实
基线值，只新增 consistency 参数。

### 4.2 两步 smoke training

以下命令复用基线 `train_config.json`，避免丢失 dataset、processor、rename map 等隐含配置：

```bash
SMOKE=outputs/train/smolvla_libero_4suite_rf_uniform_consistency_smoke
test ! -e "$SMOKE"

CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --config_path="$BASE_CFG" \
  --policy.path="$BASE_MODEL" \
  --policy.vlm_model_name="$VLM" \
  --policy.flow_objective=rectified_flow \
  --policy.flow_time_sampling=uniform \
  --policy.flow_consistency_enabled=true \
  --policy.flow_consistency_ratio=0.25 \
  --policy.flow_consistency_weight=1.0 \
  --policy.flow_consistency_timesteps=10 \
  --policy.flow_consistency_ema_power=0.75 \
  --policy.flow_consistency_ema_max_decay=0.9999 \
  --policy.flow_consistency_use_ema_for_inference=true \
  --steps=2 \
  --save_freq=2 \
  --log_freq=1 \
  --wandb.enable=false \
  --output_dir="$SMOKE" \
  --job_name=rf_uniform_consistency_smoke
```

Smoke test 必须检查：

1. forward/backward、EMA update 和保存 checkpoint 均无异常；
2. 日志同时出现有限的 `loss_rf` 与 `loss_consistency`，没有 NaN/Inf；
3. `config.json` 保存了 `flow_consistency_enabled=true`；
4. checkpoint 能重新加载，并能对 `libero_spatial task 0` 完成 1 episode、`num_steps=2` 的评估；
5. 记录峰值 GPU 显存、两步耗时和 `model.safetensors` 大小。

若发生 OOM，先保留日志并把 batch size 从 16 降到 8；不要改变 consistency ratio、loss 权重或学习率。
报告中必须注明 batch size 与基线不同，此时结论只能作为初筛。

### 4.3 正式训练

Smoke 通过后，从 `smolvla_base` 独立训练 100k，不从 RF 80k 续训：

```bash
OUT=outputs/train/smolvla_libero_4suite_rf_uniform_consistency
LOG=logs/rf_uniform_consistency_train.log
mkdir -p logs
test ! -e "$OUT"

nohup env CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --config_path="$BASE_CFG" \
  --policy.path="$BASE_MODEL" \
  --policy.vlm_model_name="$VLM" \
  --policy.flow_objective=rectified_flow \
  --policy.flow_time_sampling=uniform \
  --policy.flow_consistency_enabled=true \
  --policy.flow_consistency_ratio=0.25 \
  --policy.flow_consistency_weight=1.0 \
  --policy.flow_consistency_timesteps=10 \
  --policy.flow_consistency_ema_power=0.75 \
  --policy.flow_consistency_ema_max_decay=0.9999 \
  --policy.flow_consistency_use_ema_for_inference=true \
  --seed=1000 \
  --batch_size=16 \
  --steps=100000 \
  --save_freq=20000 \
  --wandb.enable=true \
  --wandb.mode=offline \
  --output_dir="$OUT" \
  --job_name=smolvla_libero_4suite_rf_uniform_consistency \
  > "$LOG" 2>&1 &

echo $!
```

训练期间监控 `loss_rf`、`loss_consistency`、总 loss、梯度范数、吞吐和显存。不得覆盖原目录
`smolvla_libero_4suite_rf_uniform`。

### 4.4 评估矩阵

先以 `N=2` 对 40k/60k/80k/100k 做 `3 episodes/task` checkpoint 初筛，再对最佳 checkpoint
做 `N={1,2,4,8}` 扫描。这样避免一开始运行完整的 16 组笛卡尔积。固定：

```text
4 suites = libero_spatial, libero_object, libero_goal, libero_10
n_action_steps = 10
flow_solver = euler
seed = 1000
eval.batch_size = 1
EMA inference = true
```

使用下面的函数，已完成且含 `eval_info.json` 的目录必须跳过，不得覆盖：

```bash
CT_RUN=outputs/train/smolvla_libero_4suite_rf_uniform_consistency
EVAL_ROOT=outputs/eval/rf_continuous_consistency
mkdir -p "$EVAL_ROOT/screen_ep3" logs/rf_continuous_consistency

eval_ct() {
  STEP=$1
  N=$2
  EP=$3
  MODE=$4
  CKPT="$CT_RUN/checkpoints/$STEP/pretrained_model"
  OUT="$EVAL_ROOT/${MODE}/ct_${STEP}_s${N}_a10_ep${EP}_seed1000"
  LOG="logs/rf_continuous_consistency/${MODE}_ct_${STEP}_s${N}_ep${EP}.log"
  test -f "$CKPT/model.safetensors" || { echo "missing $CKPT"; return 1; }
  test -f "$OUT/eval_info.json" && { echo "skip $OUT"; return 0; }

  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 lerobot-eval \
    --policy.path="$CKPT" \
    --policy.vlm_model_name="$VLM" \
    --policy.empty_cameras=1 \
    --policy.flow_objective=rectified_flow \
    --policy.flow_time_sampling=uniform \
    --policy.flow_solver=euler \
    --policy.flow_consistency_enabled=true \
    --policy.flow_consistency_use_ema_for_inference=true \
    --policy.n_action_steps=10 \
    --policy.num_steps="$N" \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --env.camera_name_mapping='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}' \
    --eval.batch_size=1 \
    --eval.n_episodes="$EP" \
    --seed=1000 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --output_dir="$OUT" \
    > "$LOG" 2>&1
}

# 阶段 A：checkpoint sweep。
for STEP in 040000 060000 080000 100000; do
  eval_ct "$STEP" 2 3 screen_ep3
done

# 阶段 B：根据阶段 A 的 4-suite 均值选 BEST_STEP，再做 NFE sweep。
BEST_STEP=080000  # 必须用阶段 A 的真实最佳值替换此占位值
for N in 1 2 4 8; do
  eval_ct "$BEST_STEP" "$N" 3 screen_ep3
done
```

输出目录格式为：

```text
outputs/eval/rf_continuous_consistency/screen_ep3/ct_{STEP}_s{N}_a10_ep3_seed1000
```

选择最佳 checkpoint 和最佳 `N` 后，做 `10 episodes/task` 正式评估。至少正式评估：

- CT 最佳 checkpoint：`N=1,2,4,8`；
- 原 RF Uniform 80k：使用同 seed、同协议的已有结果；若结果目录或协议无法核验，则重跑；
- CT 最佳配置再设置 `flow_consistency_use_ema_for_inference=false`，做一次 student/EMA 消融。

正式评估前创建目录，然后仍调用上述函数：

```bash
mkdir -p "$EVAL_ROOT/formal_ep10"
for N in 1 2 4 8; do
  eval_ct "$BEST_STEP" "$N" 10 formal_ep10
done
```

student 消融应复制同一条 `lerobot-eval` 命令，仅将
`--policy.flow_consistency_use_ema_for_inference=false`，输出到独立的 `student_ablation` 目录。

### 4.5 结果判定与交付

主要指标为 4-suite 平均成功率，另外单独报告 `libero_10`。需要给出：

1. checkpoint x NFE 成功率表；
2. 每个 suite 和每个 task 的成功数；
3. RF 基线与 CT 的 paired success/failure 表；
4. EMA vs student 消融；
5. `N=1/2/4/8` 的均值、最佳值和跨 N 波动；
6. 训练时间、峰值显存、checkpoint 大小；
7. 失败、OOM、重跑和配置偏差的完整记录。

现有重复评估表明约 1 percentage point 可能来自评估噪声。3-episode 初筛只能筛选，不能下最终结论。
建议用 10-episode 结果按以下标准解释：

- 最佳成功率提高超过约 1 pp，并在 `libero_10` 或 paired comparison 上一致改善：支持该方向；
- 成功率相当（绝对差不超过约 1 pp），但 NFE 更低或跨 N 明显更稳定：说明一致性训练有计算收益；
- 所有 `N=1/2/4/8` 均落后约 2 pp 以上：当前配置下无效；
- 只改善训练 loss、不改善任务成功率：不能视为方法有效。

最终在服务器生成：

```text
docs/source/RF_CONTINUOUS_CONSISTENCY_REPORT.md
```

报告开头必须记录 Git commit、完整训练配置、checkpoint 哈希、数据集路径、GPU、seed 和所有结果目录。

## 5. 当前实现边界

- 这是将 ManiFlow 风格的连续一致性机制移植到 SmolVLA RF 的实验实现，不代表已经证明有效；
- 当前只实现相对 `delta_t` 条件；未加入 absolute target time、其他 `delta_t` 分布或额外采样器；
- 默认超参数首先用于验证方向，只有主对照有效后才值得消融 ratio、weight、EMA decay；
- 不应同时改时间采样、solver、动作 chunk 或数据配方，否则无法判断收益来自哪里。
