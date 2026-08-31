# ProbeFlow inference for SmolVLA Rectified Flow

This fork provides a training-free ProbeFlow solver for an existing SmolVLA
Rectified Flow checkpoint. It does not change the training objective or model
weights.

## Method

Set `policy.flow_solver=probe_euler` to replace the fixed Euler schedule with a
lookahead linearity probe:

1. Evaluate the velocity at `t=0`.
2. Take a lookahead step to `t=probeflow_probe_time` and evaluate the velocity again.
3. Compute cosine similarity on the real action dimensions and, by default, only
   on the first `n_action_steps` actions that will be executed.
4. Use two Euler steps for a linear path and increase the schedule for a curved path.

The default candidate set is `{2, 4}` because the existing RF evaluation found
that fixed 8-step inference was slower and less successful. If every sample is
assigned two steps, the output is exactly the same as fixed two-step Euler when
`probeflow_probe_time=0.5`.

For a dense fallback with scheduled step count `N > 2`, the lookahead evaluation
cannot be reused and actual NFE is `N + 1`. Always compare actual NFE and latency,
not only the scheduled step count.

## Configuration

```text
flow_solver=probe_euler
probeflow_probe_time=0.5
probeflow_epsilon=0.008
probeflow_max_steps=4
probeflow_step_increment=2
probeflow_use_action_horizon=true
probeflow_log_every=100
```

`num_steps` is ignored by `probe_euler`. The implementation currently supports
only `flow_objective=rectified_flow` and non-compiled inference.

For a batch larger than one, the largest schedule requested by any sample is
applied to the whole batch to preserve vectorized inference. Use `eval.batch_size=1`
for per-episode adaptive evaluation.

## Evaluation

No training is required. Point evaluation at an existing RF checkpoint:

```bash
python -m lerobot.scripts.lerobot_eval \
  --policy.path=outputs/train/smolvla_libero_4suite_rf_uniform/checkpoints/080000/pretrained_model \
  --policy.flow_objective=rectified_flow \
  --policy.flow_solver=probe_euler \
  --policy.probeflow_probe_time=0.5 \
  --policy.probeflow_epsilon=0.008 \
  --policy.probeflow_max_steps=4 \
  --policy.probeflow_step_increment=2 \
  --policy.n_action_steps=10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --output_dir=outputs/eval/probeflow/rf80k_eps008_n2_4
```

The evaluator log prints cumulative diagnostics every
`probeflow_log_every` action chunks:

```text
ProbeFlow stats: calls=..., mean_similarity=..., mean_steps=..., mean_nfe=..., step_histogram=...
```

The final aggregate is also written to the `probeflow` section of
`eval_info.json`.

Calibrate `probeflow_epsilon` on a validation run rather than copying the paper's
threshold blindly. A useful first sweep is `0.002, 0.004, 0.008, 0.016`, while
keeping the candidate set fixed at `{2, 4}`.
