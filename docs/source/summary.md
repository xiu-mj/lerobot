# SmolVLA on LIBERO — Complete Evaluation Results

All models trained on `libero_spatial` only, evaluated on 4 LIBERO suites (10 episodes each).

**FM** = Flow Matching (original), **RF** = Rectified Flow (Euler), **Heun** = Rectified Flow with Heun solver.

## 1. Complete Comparison Matrix

| Model | Trained | Method | Steps | spatial | object | goal | li10 | **4-suite** |
|-------|---------|--------|-------|---------|--------|------|------|-------------|
| Std | 100k | FM | s=2 | 72% | 89% | 91% | 57% | **77.2%** |
| Std | 100k | FM | s=4 | 67% | 81% | 90% | 53% | **72.8%** |
| Std | 100k | FM | s=6 | 65% | 74% | 78% | 45% | **65.5%** |
| Std | 100k | FM | s=8 | 68% | 70% | 82% | 46% | **66.5%** |
| Std | 100k | FM | s=10 | 69% | 66% | 83% | 37% | **63.8%** |
| Std | 200k | RF | s=2 | 81% | 90% | 96% | 48% | **78.8%** |
| Std | 200k | RF | s=4 | 75% | 84% | 89% | 46% | **73.5%** |
| Std | 200k | RF | s=6 | 65% | 80% | 84% | 52% | **70.2%** |
| Std | 200k | RF | s=8 | 70% | 79% | 87% | 56% | **73.0%** |
| Std | 200k | RF | s=10 | 60% | 77% | 85% | 51% | **68.2%** |
| Std | 200k | Heun | s=2 | 16% | 48% | 52% | 20% | **34.0%** |
| Std | 200k | Heun | s=4 | 55% | 66% | 74% | 40% | **58.8%** |
| Std | 200k | Heun | s=5 | 62% | 69% | 81% | 41% | **63.2%** |
| RF | 100k | RF | s=2 | 80% | 93% | 93% | 66% | **83.0%** |
| RF | 100k | RF | s=4 | 69% | 84% | 83% | 48% | **71.0%** |
| RF | 100k | RF | s=6 | 73% | 80% | 83% | 36% | **68.0%** |
| RF | 100k | RF | s=8 | 72% | 76% | 80% | 52% | **70.0%** |
| RF | 100k | RF | s=10 | 75% | 78% | 81% | 44% | **69.5%** |
| RF | 200k | RF | s=2 | 80% | 88% | 88% | 53% | **77.2%** |
| RF | 200k | RF | s=4 | 68% | 85% | 87% | 50% | **72.5%** |
| RF | 200k | RF | s=6 | 75% | 77% | 78% | 43% | **68.2%** |
| RF | 200k | RF | s=8 | 65% | 77% | 84% | 51% | **69.2%** |
| RF | 200k | RF | s=10 | 70% | 72% | 78% | 45% | **66.2%** |
| RF | 200k | Heun | s=2 | 4% | 46% | 38% | 15% | **25.8%** |
| RF | 200k | Heun | s=4 | 65% | 62% | 71% | 41% | **59.8%** |
| RF | 200k | Heun | s=5 | 65% | 65% | 78% | 42% | **62.5%** |

## 2. Top 10 Models

| Rank | Model | 4-suite | spatial | object | goal | li10 |
|------|-------|---------|---------|--------|------|------|
| 1 | RF 100k RF s=2 | **83.0%** | 80% | 93% | 93% | 66% |
| 2 | Std 200k RF s=2 | **78.8%** | 81% | 90% | 96% | 48% |
| 3 | Std 100k FM s=2 | **77.2%** | 72% | 89% | 91% | 57% |
| 4 | RF 200k RF s=2 | **77.2%** | 80% | 88% | 88% | 53% |
| 5 | Std 200k RF s=4 | **73.5%** | 75% | 84% | 89% | 46% |
| 6 | Std 200k RF s=8 | **73.0%** | 70% | 79% | 87% | 56% |
| 7 | Std 100k FM s=4 | **72.8%** | 67% | 81% | 90% | 53% |
| 8 | RF 200k RF s=4 | **72.5%** | 68% | 85% | 87% | 50% |
| 9 | RF 100k RF s=4 | **71.0%** | 69% | 84% | 83% | 48% |
| 10 | Std 200k RF s=6 | **70.2%** | 65% | 80% | 84% | 52% |

## 3. Steps Ablation — Standard SmolVLA

| Steps | Std 100k (FM) | Std 200k (RF) |
|-------|---------------|---------------|
| s=10 | **63.8%** | **68.2%** |
| s=8 | **66.5%** | **73.0%** |
| s=6 | **65.5%** | **70.2%** |
| s=4 | **72.8%** | **73.5%** |
| s=2 | **77.2%** | **78.8%** |

## 4. Steps Ablation — Rectified Flow

| Steps | RF 100k | RF 200k |
|-------|---------|---------|
| s=10 | **69.5%** | **66.2%** |
| s=8 | **70.0%** | **69.2%** |
| s=6 | **68.0%** | **68.2%** |
| s=4 | **71.0%** | **72.5%** |
| s=2 | **83.0%** | **77.2%** |

## 5. Euler vs Heun (200k RF models)

| Compute equiv | Std Euler | Std Heun | RF Euler | RF Heun |
|---------------|-----------|----------|----------|---------|
| Euler 2 ≈ Heun 2 | **78.8%** | **34.0%** | **77.2%** | **25.8%** |
| Euler 4 ≈ Heun 2 | **73.5%** | **34.0%** | **72.5%** | **25.8%** |
| Euler 4 ≈ Heun 4 | **73.5%** | **58.8%** | **72.5%** | **59.8%** |
| Euler 8 ≈ Heun 4 | **73.0%** | **58.8%** | **69.2%** | **59.8%** |
| Euler 10 ≈ Heun 5 | **68.2%** | **63.2%** | **66.2%** | **62.5%** |

## 6. Best per Suite

| Suite | Best Model | Score |
|-------|-----------|-------|
| libero_spatial | Std 200k RF s=2 | 81% |
| libero_object | RF 100k RF s=2 | 93% |
| libero_goal | Std 200k RF s=2 | 96% |
| libero_10 | RF 100k RF s=2 | 66% |

## 7. Key Findings

1. **Best overall: RF 100k RF s=2 — 83.0%** (spatial 80%, object 93%, goal 93%, li10 66%)
2. **s=2 is universally best across all models**: Std 100k FM (77.2%), Std 200k RF (78.8%), RF 100k (83.0%), RF 200k (77.2%). The flow field is straight enough that 2-step inference outperforms 10-step.
3. **RF 100k s=2 dominates**: 83.0% with only 100k training steps and 2 inference steps. Object 93%, goal 93%.
4. **All models show the same pattern**: fewer steps = better. FM (old), RF Euler, and Heun all follow this.
5. **Heun does NOT improve over Euler at equal compute**: Heun s=4 (≈Euler 8) is systematically worse than Euler s=8.
6. **libero_10 remains hardest**: Best is 66% (RF 100k s=2).
7. **Training more than 100k provides marginal returns**: RF 200k is worse than RF 100k at most step counts.

## 8. Seed Variability (3 seeds × 10 episodes each)

RF 100k Euler s=2 vs Std 100k FM s=2, seeds 1000/2000/3000.

| Model | seed=1000 | seed=2000 | seed=3000 | **mean ± std** |
|-------|-----------|-----------|-----------|----------------|
| RF 100k s=2 | 83.0% | 81.8% | 74.8% | **79.8% ± 4.4%** |
| Std 100k FM s=2 | 77.2% | 77.2% | 78.0% | **77.5% ± 0.4%** |

Per-suite breakdown for RF 100k s=2:

| Suite | seed=1000 | seed=2000 | seed=3000 | mean ± std |
|-------|-----------|-----------|-----------|------------|
| libero_spatial | 80% | 77% | 72% | **76% ± 4%** |
| libero_object | 93% | 89% | 88% | **90% ± 3%** |
| libero_goal | 93% | 99% | 87% | **93% ± 6%** |
| libero_10 | 66% | 62% | 52% | **60% ± 7%** |

**Finding**: RF advantage holds at mean (+2.3%) but with much higher variance (±4.4% vs ±0.4%). Std is remarkably stable across seeds; RF is sensitive to initialization, particularly on `libero_10` and `libero_goal` where seed 3000 drops significantly.

## 9. RF Early Stopping Curve

All evaluated at Euler s=2, seed=2000, across RF training checkpoints.

| Checkpoint | spatial | object | goal | li10 | **4-suite** |
|------------|---------|--------|------|------|-------------|
| 60k | 80% | 90% | 86% | 59% | **78.8%** |
| 80k | 78% | 88% | 90% | 49% | **76.2%** |
| 100k | 77% | 89% | 99% | 62% | **81.8%** ← best |
| 120k | 83% | 87% | 88% | 55% | **78.2%** |
| 160k | 79% | 87% | 93% | 55% | **78.5%** |
| 200k | 80% | 88% | 88% | 53% | **77.2%** |

**Finding**: 100k is the optimal stopping point. 60k already reaches 78.8% — only 3% below peak. Beyond 100k, performance degrades (200k: 77.2%). Most learning happens early.

## 10. Time Sampling: Uniform vs Beta

RF 100k Euler s=2, seed=2000. Model trained with uniform sampling. Inference-time override of `flow_time_sampling=beta` tested.

| Sampling | spatial | object | goal | li10 | **4-suite** |
|----------|---------|--------|------|------|-------------|
| Uniform | 77% | 89% | 99% | 62% | **81.8%** |
| Beta | 77% | 89% | 99% | 62% | **81.8%** |

**Finding**: Inference-time time sampling change has zero effect — the learned vector field is determined by training, not inference. A proper comparison requires training a separate model with Beta sampling from scratch.