# Phase 1 — Rigorous RF Ablation Results

All models trained on `libero_spatial` only, evaluated on 4 LIBERO suites (10 episodes/task).
`seed` = evaluation seed block (not training seed).

## Checkpoint Configurations

| Checkpoint | Path | flow_objective | flow_solver |
|---|---|---|---|
| FM-100k | `outputs/train/smolvla_libero/checkpoints/100000/` | flow_matching (default) | euler (default) |
| RF-100k | `outputs/train/smolvla_rf_v5/checkpoints/100000/` | rectified_flow | euler (default) |
| RF-60k | `outputs/train/smolvla_rf_v5/checkpoints/060000/` | rectified_flow | euler (default) |
| RF-200k | `outputs/train/smolvla_rf_v5/checkpoints/200000/` | rectified_flow | euler (default) |

## Raw Results Index

| Experiment | Model | Steps | Seed | Output Directory |
|---|---|---|---|---|
| E1 | FM 100k | s=1 | 1000 | `phase1/fm100_s1_seed1000/eval_info.json` |
| E1 | FM 100k | s=1 | 2000 | `phase1/fm100_s1_seed2000/eval_info.json` |
| E1 | FM 100k | s=1 | 3000 | `phase1/fm100_s1_seed3000/eval_info.json` |
| E1 | FM 100k | s=2 | 1000 | `../smolvla_standard_100k_steps2/eval_info.json` |
| E1 | FM 100k | s=2 | 2000 | `../smolvla_standard_100k_s2_seed2000/eval_info.json` |
| E1 | FM 100k | s=2 | 3000 | `../smolvla_standard_100k_s2_seed3000/eval_info.json` |
| E1 | FM 100k | s=3 | 1000 | `phase1/fm100_s3_seed1000/eval_info.json` |
| E1 | FM 100k | s=3 | 2000 | `phase1/fm100_s3_seed2000/eval_info.json` |
| E1 | FM 100k | s=3 | 3000 | `phase1/fm100_s3_seed3000/eval_info.json` |
| E1 | RF 100k | s=1 | 1000 | `phase1/rf100_s1_seed1000/eval_info.json` |
| E1 | RF 100k | s=1 | 2000 | `phase1/rf100_s1_seed2000/eval_info.json` |
| E1 | RF 100k | s=1 | 3000 | `phase1/rf100_s1_seed3000/eval_info.json` |
| E1 | RF 100k | s=2 | 1000 | `../smolvla_rf_100k_steps2/eval_info.json` |
| E1 | RF 100k | s=2 | 2000 | `../smolvla_rf_100k_s2_seed2000/eval_info.json` |
| E1 | RF 100k | s=2 | 3000 | `../smolvla_rf_100k_s2_seed3000/eval_info.json` |
| E1 | RF 100k | s=3 | 1000 | `phase1/rf100_s3_seed1000/eval_info.json` |
| E1 | RF 100k | s=3 | 2000 | `phase1/rf100_s3_seed2000/eval_info.json` |
| E1 | RF 100k | s=3 | 3000 | ⬜ pending |
| E2 | RF 60k | s=2 | 1000 | `../smolvla_rf_60k_s2/eval_info.json` |
| E2 | RF 60k | s=2 | 2000 | `phase1/rf60k_s2_seed2000/eval_info.json` |
| E2 | RF 60k | s=2 | 3000 | `phase1/rf60k_s2_seed3000/eval_info.json` |
| E2 | RF 200k | s=2 | 1000 | `../smolvla_rf_200k_steps2/eval_info.json` |
| E2 | RF 200k | s=2 | 2000 | `phase1/rf200k_s2_seed2000/eval_info.json` |
| E2 | RF 200k | s=2 | 3000 | `phase1/rf200k_s2_seed3000/eval_info.json` |

## Experiment 1: s=1 vs s=2 vs s=3 (3 eval seeds each)

### FM 100k

| Steps | seed=1000 | seed=2000 | seed=3000 | **mean ± std** |
|-------|-----------|-----------|-----------|----------------|
| s=1 | 79.0% | 79.0% | 82.2% | **80.1% ± 1.9%** |
| s=2 | 77.2% | 77.2% | 78.0% | **77.5% ± 0.4%** |
| s=3 | 76.5% | 76.2% | 71.8% | **74.8% ± 2.7%** |

### RF 100k

| Steps | seed=1000 | seed=2000 | seed=3000 | **mean ± std** |
|-------|-----------|-----------|-----------|----------------|
| s=1 | 82.8% | 78.8% | 82.2% | **81.2% ± 2.2%** |
| s=2 | 83.0% | 81.8% | 74.8% | **79.8% ± 4.4%** |
| s=3 | 75.5% | 73.5% | ⬜ | **74.5% ± 1.4%** (2 seeds) |

### RF − FM Difference

| Steps | seed=1000 | seed=2000 | seed=3000 |
|-------|-----------|-----------|-----------|
| s=1 | +3.8% | −0.2% | +0.0% |
| s=2 | +5.8% | +4.5% | −3.2% |
| s=3 | −1.0% | −2.8% | N/A |

### Per-Suite Mean (across seeds)

| Suite | FM s=1 | RF s=1 | FM s=2 | RF s=2 | FM s=3 | RF s=3 |
|-------|--------|--------|--------|--------|--------|--------|
| libero_spatial | 76% | 77% | 75% | 76% | 70% | 76% |
| libero_object | 97% | 97% | 89% | 90% | 87% | 82% |
| libero_goal | 90% | 92% | 90% | 93% | 88% | 86% |
| libero_10 | 58% | 58% | 57% | 60% | 54% | 53% |

### E1 Findings

1. **s=1 is the optimal inference step count**, not s=2. FM s=1 (80.1%) > FM s=2 (77.5%). RF s=1 (81.2%) > RF s=2 (79.8%).
2. **RF advantage narrows at s=1 to +1.1%**. The previously claimed +5.8% was specific to s=2 seed=1000. At s=1, RF is neutral on seeds 2000/3000.
3. **FM is more stable**: FM ±0.4–2.7% vs RF ±1.4–4.4%. RF s=2 seed=3000 anomalous drop to 74.8%.
4. **object (86–97%) and goal (86–93%) dominate**; spatial (70–77%) and li10 (53–60%) drive variance.

## Experiment 2: RF Early Stopping Stability (s=2, 3 eval seeds)

| Checkpoint | seed=1000 | seed=2000 | seed=3000 | **mean ± std** |
|------------|-----------|-----------|-----------|----------------|
| 60k | 78.8% | 77.8% | 78.2% | **78.2% ± 0.5%** |
| 100k | 83.0% | 81.8% | 74.8% | **79.8% ± 4.4%** |
| 200k | 77.2% | 77.8% | 80.0% | **78.3% ± 1.5%** |

### E2 Findings

1. **All three checkpoints perform equivalently**: means 78.2–79.8% within error bars. No overtraining degradation, no clear sweet spot.
2. **60k is the most stable** (±0.5%). 100k variance driven by seed=3000 outlier.
3. **60k is sufficient**: training beyond 60k provides no reliable improvement.

## Overall Phase 1 Conclusions

1. **Use s=1 for inference**. It's faster and at least as good as s=2. Two-step inference provides no benefit over single-step for these flow models.
2. **RF marginal benefit over FM is not robust**: at s=1 the advantage is +1.1%, within evaluation noise. The strong RF claims from single-seed s=2 results overstate the case.
3. **60k training steps is sufficient**. No reliable improvement from 60k to 100k or 200k.
4. **FM 100k s=1 is the practical recommendation**: 80.1% mean, ±1.9% std, no special training modifications needed.
5. **Remaining: RF 100k s=3 seed=3000** — low priority, trend is already clear.

## Commands Template

All Phase 1 evals used this template:

```bash
cd /home/caimu/lerobot && MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<GPU> nohup lerobot-eval \
  --policy.path=<CHECKPOINT_PATH> \
  --policy.vlm_model_name=/home/caimu/lerobot/lerobot_models/SmolVLM2-500M-Video-Instruct \
  --policy.empty_cameras=1 --policy.flow_solver=euler --policy.num_steps=<STEPS> \
  [--policy.flow_objective=flow_matching --policy.flow_time_sampling=beta]  # FM only \
  --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --env.camera_name_mapping='{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}' \
  --eval.batch_size=1 --eval.n_episodes=10 --seed=<SEED> \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --output_dir=outputs/eval/phase1/<OUTPUT_DIR>
```
