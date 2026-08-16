# SmolVLA Adaptive Chunk and Adaptive Step

This module adds task-adaptive temporal and generative budgets to the
Rectified Flow SmolVLA action expert.

## Architecture

The image, language, and robot-state prefix embeddings are mask-mean pooled and
passed through a shared difficulty encoder. Two classifier heads predict:

- action horizon `H`, selected from `(10, 20, 50)` by default;
- Rectified Flow inference steps `N`, selected from `(2, 4, 8, 10)` by default.

At inference time, `H` controls the number of action tokens created by the
action expert and `N` controls the number of Euler function evaluations. The
original fixed `chunk_size` and `num_steps` behavior remains unchanged unless
adaptive computation is explicitly enabled.

## Enable the controller

```python
from lerobot.policies.smolvla.adaptive_computation import AdaptiveComputationConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

config = SmolVLAConfig(
    chunk_size=50,
    num_steps=10,
    flow_objective="rectified_flow",
    adaptive_computation=AdaptiveComputationConfig(
        horizon_choices=(10, 20, 50),
        num_steps_choices=(2, 4, 8, 10),
        hidden_dim=256,
        detach_context=True,
    ),
)
```

For the first implementation stage, keep a fixed action horizon and train only
Adaptive Step:

```python
adaptive_computation = AdaptiveComputationConfig(
    adapt_horizon=False,
    adapt_num_steps=True,
    num_steps_choices=(2, 4, 8, 10),
)
```

For an Adaptive Chunk-only ablation, set `adapt_horizon=True` and
`adapt_num_steps=False`.

## Offline label construction

Run an evaluation grid for every task or episode and arrange the success rates
as `(num_items, num_horizons, num_step_choices)`. The helper chooses the
lowest-cost pair that reaches the requested success threshold. If no pair
reaches it, the best-performing pair is selected and budget cost breaks ties.

```python
import torch

from lerobot.policies.smolvla.adaptive_computation import select_optimal_budget_labels

success_rates = torch.tensor(
    [
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[0.3, 0.4], [0.6, 0.7], [0.8, 0.9]],
    ]
)
horizon_labels, step_labels = select_optimal_budget_labels(
    success_rates,
    horizon_choices=(10, 20, 50),
    num_steps_choices=(2, 8),
    success_threshold=0.9,
)
```

The resulting labels in this example are `(10, 2)` and `(50, 8)`.

## Controller training

Add the actual budget values to each training batch using these keys:

```python
batch["adaptive_horizon"] = horizon_labels
batch["adaptive_num_steps"] = step_labels
```

`SmolVLAPolicy.forward` automatically adds the two cross-entropy terms to the
action loss and reports the following metrics:

- `adaptive_horizon_loss`;
- `adaptive_num_steps_loss`;
- `adaptive_horizon_accuracy`;
- `adaptive_num_steps_accuracy`;
- `adaptive_difficulty`.

The default `detach_context=True` trains the controller without changing the
VLM/action-expert representation through the controller loss. Set it to `False`
for joint end-to-end fine-tuning after the controller-only stage is stable.

## Inference and ablations

The controller is used automatically after it is enabled and trained. The last
decision can be inspected with:

```python
actions = policy.predict_action_chunk(batch)
print(policy.model.last_adaptive_decision)
```

Fixed-budget overrides are available for grid search and ablations even when
the adaptive controller is disabled:

```python
actions = policy.predict_action_chunk(
    batch,
    action_horizon=20,
    num_inference_steps=4,
)
```

For batched inference, different samples cannot produce tensors with different
sequence lengths in one call. The default `batch_strategy="max"` therefore uses
the largest predicted `H` and `N` in the batch. Online robot evaluation normally
uses batch size one and receives the exact predicted pair.

## Current compatibility constraints

- Adaptive computation requires dynamic action-token shapes and is not combined
  with `compile_model=True`.
- Adaptive computation is not combined with RTC yet because consecutive chunks
  may have different temporal lengths.
- `horizon_choices` cannot exceed the `chunk_size` used to train the action
  expert.
