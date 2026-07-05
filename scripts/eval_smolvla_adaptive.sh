#!/bin/bash
# ============================================================
# SmolVLA Adaptive Inference 评估脚本
# 自适应 vs 固定步数 对比
# ============================================================
set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONDA_PYTHON=/home/caimu/miniconda3/envs/lerobot/bin/python
PROJECT_DIR=/home/caimu/lerobot
OUTPUT_BASE=$PROJECT_DIR/outputs/eval/smolvla_adaptive
MODEL_PATH=lerobot/smolvla_base
SEED=1000
EPISODES=10

mkdir -p $OUTPUT_BASE

# ============================================================
# 自适应推理
# ============================================================
echo "========== Adaptive Inference =========="
for suite in libero_spatial libero_object libero_goal libero_10; do
    job_name="adaptive_${suite}"
    logfile="$OUTPUT_BASE/${job_name}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $job_name"

    $CONDA_PYTHON -m lerobot.scripts.lerobot_eval \
        --policy.path="$MODEL_PATH" \
        --env.type=libero \
        --env.task="$suite" \
        --seed=$SEED \
        --eval.batch_size=1 \
        --eval.n_episodes=$EPISODES \
        --policy.empty_cameras=1 \
        --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
        --policy.adaptive_config.enabled=true \
        --policy.adaptive_config.max_steps=10 \
        --policy.adaptive_config.min_steps=2 \
        --policy.adaptive_config.rel_threshold=0.01 \
        --policy.adaptive_config.patience=2 \
        > "$logfile" 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $job_name (exit=$?)"
done

# ============================================================
# 固定 10 步 (baseline)
# ============================================================
echo "========== Fixed 10 Steps =========="
for suite in libero_spatial libero_object libero_goal libero_10; do
    job_name="fixed10_${suite}"
    logfile="$OUTPUT_BASE/${job_name}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $job_name"

    $CONDA_PYTHON -m lerobot.scripts.lerobot_eval \
        --policy.path="$MODEL_PATH" \
        --env.type=libero \
        --env.task="$suite" \
        --seed=$SEED \
        --eval.batch_size=1 \
        --eval.n_episodes=$EPISODES \
        --policy.empty_cameras=1 \
        --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
        --policy.num_inference_steps=10 \
        > "$logfile" 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $job_name (exit=$?)"
done

echo ""
echo "========== ALL DONE =========="
