#!/bin/bash
# ============================================================
# pi0 系列 baseline 评估脚本
# 模型: pi05_libero_finetuned / pi0_libero_finetuned / pi0fast-libero
# 套件: libero_spatial / libero_object / libero_goal / libero_10
# 推理步数: 1, 2, 4, 6, 8, 10 (pi0_fast 不适用)
# ============================================================
set -e

CONDA_PYTHON=/home/caimu/miniconda3/envs/lerobot/bin/python
PROJECT_DIR=/home/caimu/lerobot
OUTPUT_BASE=$PROJECT_DIR/outputs/eval
MODELS_DIR=$PROJECT_DIR/lerobot_models

mkdir -p $OUTPUT_BASE

run_eval() {
    local model_name=$1
    local model_path=$2
    local suite=$3
    local steps=$4
    local extra_flags=$5

    local job_name="${model_name}_${suite}_${steps}step"
    local logfile="$OUTPUT_BASE/${job_name}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $job_name"

    $CONDA_PYTHON -m lerobot.scripts.lerobot_eval \
        --policy.path="$model_path" \
        --env.type=libero \
        --env.task="$suite" \
        --seed=1000 \
        --eval.batch_size=1 \
        --eval.n_episodes=10 \
        $extra_flags \
        > "$logfile" 2>&1

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $job_name (exit=$?)"
}

# ============================================================
# pi05_libero_finetuned  (流匹配，支持 num_inference_steps)
# ============================================================
echo "========== pi05_libero_finetuned =========="
for suite in libero_spatial libero_object libero_goal libero_10; do
    for steps in 1 2 4 6 8 10; do
        run_eval "pi05_libero" "$MODELS_DIR/pi05_libero_finetuned" "$suite" "$steps" \
            "--policy.num_inference_steps=$steps"
    done
done

# ============================================================
# pi0_libero_finetuned  (流匹配，支持 num_inference_steps)
# ============================================================
echo "========== pi0_libero_finetuned =========="
for suite in libero_spatial libero_object libero_goal libero_10; do
    for steps in 1 2 4 6 8 10; do
        run_eval "pi0_libero" "$MODELS_DIR/pi0_libero_finetuned" "$suite" "$steps" \
            "--policy.num_inference_steps=$steps"
    done
done

# ============================================================
# pi0fast-libero  (自回归，不支持 num_inference_steps)
# 每个套件只跑一次
# ============================================================
echo "========== pi0fast-libero =========="
for suite in libero_spatial libero_object libero_goal libero_10; do
    run_eval "pi0fast_libero" "$MODELS_DIR/pi0fast-libero" "$suite" "default" ""
done

echo ""
echo "========== ALL DONE =========="
