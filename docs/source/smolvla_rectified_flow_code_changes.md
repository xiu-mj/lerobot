# SmolVLA Rectified Flow Code Changes

This document records the first-stage SmolVLA Rectified Flow modification in this fork.
It is intended as a compact technical reference before starting the next round of changes.

## Goal

The original SmolVLA action head uses a flow-matching style objective with a noise-to-action
denoising process written in the reverse time convention. This change adds an explicit
Rectified Flow objective while keeping the original flow-matching behavior available for
ablation.

The implemented Rectified Flow path is:

```text
x_t = (1 - t) * noise + t * action
target velocity = action - noise
t ~ Uniform(eps, 1 - eps)
```

At inference time, sampling starts from Gaussian noise at `t=0` and integrates the learned
velocity field to `t=1` with Euler steps.

## Modified Files

### `src/lerobot/policies/smolvla/configuration_smolvla.py`

Added configurable flow settings to `SmolVLAConfig`:

```python
flow_objective: str = "rectified_flow"
flow_time_sampling: str = "uniform"
flow_time_beta_alpha: float = 1.5
flow_time_beta_beta: float = 1.0
flow_time_eps: float = 1e-3
```

Validation was added for:

- `flow_objective in {"rectified_flow", "flow_matching"}`
- `flow_time_sampling in {"uniform", "beta"}`
- `0 <= flow_time_eps < 0.5`

The default is now `rectified_flow`, but the previous behavior can be requested with:

```bash
--policy.flow_objective=flow_matching \
--policy.flow_time_sampling=beta
```

### `src/lerobot/policies/smolvla/modeling_smolvla.py`

Added reusable helpers:

```python
sample_flow_time(...)
flow_training_path(...)
```

`flow_training_path` contains both supported objectives:

```python
if objective == "rectified_flow":
    x_t = (1 - time_expanded) * noise + time_expanded * actions
    u_t = actions - noise
elif objective == "flow_matching":
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
```

Training now uses:

```python
x_t, u_t = flow_training_path(actions, noise, time, self.config.flow_objective)
```

Inference direction now depends on the configured objective:

```python
if self.config.flow_objective == "rectified_flow":
    start_time = 0.0
    dt = 1.0 / num_steps
else:
    start_time = 1.0
    dt = -1.0 / num_steps
```

For Rectified Flow, the final clean action estimate used by RTC is:

```python
x_1 = x_t + (1 - t) * v_t
```

For legacy flow matching, it remains:

```python
x_0 = x_t - t * v_t
```

### `src/lerobot/policies/rtc/modeling_rtc.py`

RTC previously assumed a fixed endpoint formula:

```python
x1_t = x_t - time * v_t
```

That formula is correct for the legacy reverse-time convention, but not for Rectified Flow.
The RTC denoising wrapper now accepts an optional `endpoint_prediction_fn` so policies can
provide their own endpoint estimate:

```python
endpoint_prediction_fn=None
```

If no callback is passed, RTC keeps its old behavior. SmolVLA passes its flow-aware endpoint
function during sampling.

### `tests/policies/smolvla/test_smolvla_rectified_flow.py`

Added lightweight unit tests for:

- default config uses `rectified_flow`
- Rectified Flow interpolation and target direction
- legacy flow-matching interpolation and target direction
- time sampling respects epsilon bounds
- invalid flow objective raises `ValueError`

These tests do not require the large SmolVLA weights.

### Documentation

Added/updated:

- `docs/source/smolvla_rectified_flow_runbook.md`
- `docs/source/policy_smolvla_README.md`
- `docs/source/smolvla_rectified_flow_code_changes.md`

## Main Training Flags

Rectified Flow:

```bash
--policy.flow_objective=rectified_flow \
--policy.flow_time_sampling=uniform \
--policy.flow_time_eps=0.001
```

Legacy flow matching baseline:

```bash
--policy.flow_objective=flow_matching \
--policy.flow_time_sampling=beta \
--policy.flow_time_beta_alpha=1.5 \
--policy.flow_time_beta_beta=1.0
```

## Current Experimental Takeaway

The first LIBERO experiments indicate that the code path is functional and that RF is useful
as a training-efficiency modification:

- `RF 100k s=10`: 69.5 percent 4-suite average
- `Standard 100k s=10`: 63.8 percent 4-suite average
- Delta: +5.7 points at the same training and inference budget

At 200k steps, the best current Standard configuration is still slightly higher:

- `Standard 200k s=4`: 73.5 percent
- `RF 200k s=4`: 72.5 percent

So the conservative conclusion is:

```text
Rectified Flow is a valid and useful modification for SmolVLA, especially for faster
convergence at 100k steps. Current evidence does not yet show it as a uniformly better
final-performance replacement for Standard SmolVLA after inference-step tuning.
```

## Recommended Next Experiments

1. Run RF 100k with `num_steps=4/6/8/2`.
2. Run Standard 100k with `num_steps=4/6/8/2`.
3. Compare `best-s` at the same training budget.
4. Keep `libero_90` separate from the main 4-suite conclusion because all current scores are near zero.

