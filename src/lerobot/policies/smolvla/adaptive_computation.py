#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-adaptive temporal and generative budget allocation for SmolVLA.

The controller consumes the multimodal prefix tokens already built by SmolVLA
and predicts two coupled budgets:

* an action horizon (the number of action tokens to generate), and
* a flow integration budget (the number of denoising/function evaluations).

The module deliberately keeps label construction separate from controller
training. :func:`select_optimal_budget_labels` converts an offline grid search
over ``(horizon, num_steps)`` into the lowest-cost successful budget labels.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


ADAPTIVE_HORIZON_LABEL = "adaptive_horizon"
ADAPTIVE_NUM_STEPS_LABEL = "adaptive_num_steps"


@dataclass
class AdaptiveComputationConfig:
    """Configuration for the shared difficulty encoder and its two heads."""

    enabled: bool = True
    adapt_horizon: bool = True
    adapt_num_steps: bool = True
    horizon_choices: tuple[int, ...] = (10, 20, 50)
    num_steps_choices: tuple[int, ...] = (2, 4, 8, 10)
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    dropout: float = 0.1
    detach_context: bool = True
    horizon_loss_weight: float = 1.0
    num_steps_loss_weight: float = 1.0
    batch_strategy: Literal["max", "mode"] = "max"

    def __post_init__(self) -> None:
        self.horizon_choices = self._validate_choices(self.horizon_choices, "horizon_choices")
        self.num_steps_choices = self._validate_choices(
            self.num_steps_choices, "num_steps_choices"
        )
        if not self.adapt_horizon and not self.adapt_num_steps:
            raise ValueError("At least one adaptive budget must be enabled.")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")
        if self.num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be positive, got {self.num_hidden_layers}."
            )
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        if self.horizon_loss_weight < 0 or self.num_steps_loss_weight < 0:
            raise ValueError("Adaptive loss weights must be non-negative.")
        if self.batch_strategy not in {"max", "mode"}:
            raise ValueError(
                f"batch_strategy must be 'max' or 'mode', got {self.batch_strategy!r}."
            )

    @staticmethod
    def _validate_choices(choices: tuple[int, ...], name: str) -> tuple[int, ...]:
        values = tuple(int(value) for value in choices)
        if not values:
            raise ValueError(f"{name} must not be empty.")
        if any(value <= 0 for value in values):
            raise ValueError(f"All {name} entries must be positive, got {values}.")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique entries, got {values}.")
        if values != tuple(sorted(values)):
            raise ValueError(f"{name} must be sorted in ascending order, got {values}.")
        return values

    def validate_for_policy(self, max_horizon: int) -> None:
        if self.adapt_horizon and self.horizon_choices[-1] > max_horizon:
            raise ValueError(
                "Adaptive action horizons cannot exceed SmolVLA chunk_size. "
                f"Got max horizon {self.horizon_choices[-1]} and chunk_size {max_horizon}."
            )


@dataclass
class AdaptiveComputationOutput:
    horizon_logits: Tensor
    num_steps_logits: Tensor
    horizon_indices: Tensor
    num_steps_indices: Tensor
    horizons: Tensor
    num_steps: Tensor
    difficulty: Tensor


@dataclass(frozen=True)
class RuntimeBudget:
    action_horizon: int
    num_steps: int


class DifficultyEncoder(nn.Module):
    """Shared representation used by the horizon and flow-step heads."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, context: Tensor) -> Tensor:
        return self.network(context)


def masked_mean_pool(tokens: Tensor, padding_mask: Tensor | None = None) -> Tensor:
    """Pool ``(B, L, D)`` context tokens while ignoring padded positions."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected context tokens with shape (B, L, D), got {tokens.shape}.")
    if padding_mask is None:
        return tokens.mean(dim=1)
    if padding_mask.shape != tokens.shape[:2]:
        raise ValueError(
            "padding_mask must match the first two token dimensions, "
            f"got mask {padding_mask.shape} and tokens {tokens.shape}."
        )
    weights = padding_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1)
    return (tokens * weights).sum(dim=1) / denominator


class AdaptiveComputationController(nn.Module):
    """Predict coupled action-horizon and flow-integration budgets."""

    def __init__(self, input_dim: int, config: AdaptiveComputationConfig) -> None:
        super().__init__()
        self.config = config
        self.difficulty_encoder = DifficultyEncoder(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
        )
        self.horizon_head = nn.Linear(config.hidden_dim, len(config.horizon_choices))
        self.num_steps_head = nn.Linear(config.hidden_dim, len(config.num_steps_choices))
        self.register_buffer(
            "horizon_values", torch.tensor(config.horizon_choices, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "num_steps_values",
            torch.tensor(config.num_steps_choices, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self, context_tokens: Tensor, padding_mask: Tensor | None = None
    ) -> AdaptiveComputationOutput:
        context = masked_mean_pool(context_tokens, padding_mask)
        if self.config.detach_context:
            context = context.detach()
        parameter_dtype = self.horizon_head.weight.dtype
        context = context.to(dtype=parameter_dtype)
        difficulty_features = self.difficulty_encoder(context)
        horizon_logits = self.horizon_head(difficulty_features)
        num_steps_logits = self.num_steps_head(difficulty_features)

        horizon_indices = horizon_logits.argmax(dim=-1)
        num_steps_indices = num_steps_logits.argmax(dim=-1)
        horizons = self.horizon_values[horizon_indices]
        num_steps = self.num_steps_values[num_steps_indices]

        horizon_scale = _normalize_choices(self.horizon_values, horizon_logits.dtype)
        num_steps_scale = _normalize_choices(self.num_steps_values, num_steps_logits.dtype)
        expected_horizon = (horizon_logits.softmax(dim=-1) * horizon_scale).sum(dim=-1)
        expected_num_steps = (num_steps_logits.softmax(dim=-1) * num_steps_scale).sum(dim=-1)
        difficulty = 0.5 * (expected_horizon + expected_num_steps)

        return AdaptiveComputationOutput(
            horizon_logits=horizon_logits,
            num_steps_logits=num_steps_logits,
            horizon_indices=horizon_indices,
            num_steps_indices=num_steps_indices,
            horizons=horizons,
            num_steps=num_steps,
            difficulty=difficulty,
        )

    def supervised_loss(
        self,
        output: AdaptiveComputationOutput,
        horizon_labels: Tensor | None = None,
        num_steps_labels: Tensor | None = None,
        reduction: Literal["none", "mean"] = "mean",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute controller losses from actual budget values, not class indices."""

        loss_terms: dict[str, Tensor] = {}
        total = torch.zeros(
            output.horizon_logits.shape[0],
            device=output.horizon_logits.device,
            dtype=output.horizon_logits.dtype,
        )
        if horizon_labels is not None and self.config.adapt_horizon:
            horizon_targets = budget_values_to_indices(
                horizon_labels, self.horizon_values, "horizon"
            )
            horizon_loss = F.cross_entropy(
                output.horizon_logits, horizon_targets, reduction="none"
            )
            loss_terms["adaptive_horizon_loss"] = horizon_loss
            total = total + self.config.horizon_loss_weight * horizon_loss
        if num_steps_labels is not None and self.config.adapt_num_steps:
            num_steps_targets = budget_values_to_indices(
                num_steps_labels, self.num_steps_values, "num_steps"
            )
            num_steps_loss = F.cross_entropy(
                output.num_steps_logits, num_steps_targets, reduction="none"
            )
            loss_terms["adaptive_num_steps_loss"] = num_steps_loss
            total = total + self.config.num_steps_loss_weight * num_steps_loss
        if not loss_terms:
            raise ValueError("At least one enabled adaptive budget label must be provided.")

        if reduction == "mean":
            return total.mean(), {name: value.mean() for name, value in loss_terms.items()}
        if reduction != "none":
            raise ValueError(f"Unsupported reduction: {reduction!r}.")
        return total, loss_terms


def budget_values_to_indices(labels: Tensor, choices: Tensor, budget_name: str) -> Tensor:
    labels = torch.as_tensor(labels, device=choices.device, dtype=choices.dtype).reshape(-1)
    matches = labels[:, None] == choices[None, :]
    valid = matches.any(dim=-1)
    if not torch.all(valid):
        invalid = labels[~valid].detach().cpu().tolist()
        raise ValueError(
            f"Invalid {budget_name} labels {invalid}; expected values from {choices.cpu().tolist()}."
        )
    return matches.to(dtype=torch.long).argmax(dim=-1)


def resolve_runtime_budget(
    output: AdaptiveComputationOutput | None,
    config: AdaptiveComputationConfig | None,
    default_horizon: int,
    default_num_steps: int,
    action_horizon_override: int | None = None,
    num_steps_override: int | None = None,
) -> RuntimeBudget:
    """Reduce per-sample predictions to one shape/loop budget for a runtime batch."""

    if action_horizon_override is not None:
        action_horizon = int(action_horizon_override)
    elif output is not None and config is not None and config.enabled and config.adapt_horizon:
        action_horizon = _reduce_batch_budget(output.horizons, config.batch_strategy)
    else:
        action_horizon = int(default_horizon)

    if num_steps_override is not None:
        num_steps = int(num_steps_override)
    elif output is not None and config is not None and config.enabled and config.adapt_num_steps:
        num_steps = _reduce_batch_budget(output.num_steps, config.batch_strategy)
    else:
        num_steps = int(default_num_steps)

    if not 0 < action_horizon <= default_horizon:
        raise ValueError(
            f"action_horizon must be in [1, {default_horizon}], got {action_horizon}."
        )
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    return RuntimeBudget(action_horizon=action_horizon, num_steps=num_steps)


def select_optimal_budget_labels(
    success_rates: Tensor,
    horizon_choices: tuple[int, ...],
    num_steps_choices: tuple[int, ...],
    success_threshold: float = 1.0,
    horizon_cost_weight: float = 1.0,
    num_steps_cost_weight: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Select the cheapest successful ``(H, N)`` pair from an offline grid.

    ``success_rates`` must end in ``(len(horizon_choices), len(num_steps_choices))``.
    When no pair reaches ``success_threshold``, the highest-success pair is used,
    with normalized budget cost breaking ties.
    """

    expected_shape = (len(horizon_choices), len(num_steps_choices))
    if tuple(success_rates.shape[-2:]) != expected_shape:
        raise ValueError(
            f"Expected success_rates trailing shape {expected_shape}, got {success_rates.shape}."
        )
    if horizon_cost_weight < 0 or num_steps_cost_weight < 0:
        raise ValueError("Budget cost weights must be non-negative.")

    rates = success_rates.to(dtype=torch.float32)
    flat_rates = rates.reshape(-1, *expected_shape)
    horizons = torch.tensor(horizon_choices, device=rates.device, dtype=torch.float32)
    steps = torch.tensor(num_steps_choices, device=rates.device, dtype=torch.float32)
    horizon_cost = _normalize_choices(horizons, rates.dtype)[:, None]
    step_cost = _normalize_choices(steps, rates.dtype)[None, :]
    cost = horizon_cost_weight * horizon_cost + num_steps_cost_weight * step_cost
    flat_cost = cost.reshape(-1)

    selected_indices = []
    for task_rates in flat_rates:
        flat_task_rates = task_rates.reshape(-1)
        feasible = flat_task_rates >= success_threshold
        if feasible.any():
            candidate_cost = torch.where(feasible, flat_cost, torch.inf)
        else:
            best_success = flat_task_rates.max()
            best_mask = torch.isclose(flat_task_rates, best_success)
            candidate_cost = torch.where(best_mask, flat_cost, torch.inf)
        selected_indices.append(candidate_cost.argmin())

    selected = torch.stack(selected_indices)
    step_count = len(num_steps_choices)
    horizon_indices = torch.div(selected, step_count, rounding_mode="floor")
    num_steps_indices = selected % step_count
    horizon_values = torch.tensor(horizon_choices, device=rates.device, dtype=torch.long)
    num_steps_values = torch.tensor(num_steps_choices, device=rates.device, dtype=torch.long)
    leading_shape = rates.shape[:-2]
    return (
        horizon_values[horizon_indices].reshape(leading_shape),
        num_steps_values[num_steps_indices].reshape(leading_shape),
    )


def _normalize_choices(values: Tensor, dtype: torch.dtype) -> Tensor:
    values = values.to(dtype=dtype)
    minimum = values.min()
    value_range = values.max() - minimum
    if value_range.item() == 0:
        return torch.zeros_like(values)
    return (values - minimum) / value_range


def _reduce_batch_budget(values: Tensor, strategy: Literal["max", "mode"]) -> int:
    if values.numel() == 0:
        raise ValueError("Cannot resolve a runtime budget from an empty batch.")
    if strategy == "max":
        return int(values.max().item())
    if strategy == "mode":
        return int(torch.mode(values.flatten()).values.item())
    raise ValueError(f"Unsupported batch strategy: {strategy!r}.")
