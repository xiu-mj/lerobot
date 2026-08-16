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

from collections import deque
from types import MethodType, SimpleNamespace

import pytest
import torch

from lerobot.policies.smolvla.adaptive_computation import (
    AdaptiveComputationConfig,
    AdaptiveComputationController,
    masked_mean_pool,
    resolve_runtime_budget,
    select_optimal_budget_labels,
)
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION


def test_masked_mean_pool_ignores_padding():
    tokens = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])

    pooled = masked_mean_pool(tokens, mask)

    assert torch.equal(pooled, torch.tensor([[2.0, 3.0]]))


def test_controller_predicts_both_coupled_budgets():
    config = AdaptiveComputationConfig(hidden_dim=8, num_hidden_layers=1, dropout=0)
    controller = AdaptiveComputationController(input_dim=6, config=config)
    with torch.no_grad():
        controller.horizon_head.weight.zero_()
        controller.horizon_head.bias.copy_(torch.tensor([-1.0, 3.0, 0.0]))
        controller.num_steps_head.weight.zero_()
        controller.num_steps_head.bias.copy_(torch.tensor([-1.0, 0.0, 4.0, 1.0]))

    output = controller(torch.randn(2, 5, 6), torch.ones(2, 5, dtype=torch.bool))

    assert output.horizon_logits.shape == (2, 3)
    assert output.num_steps_logits.shape == (2, 4)
    assert output.horizons.tolist() == [20, 20]
    assert output.num_steps.tolist() == [8, 8]
    assert torch.all((0 <= output.difficulty) & (output.difficulty <= 1))


def test_controller_supervised_loss_uses_budget_values_and_backpropagates():
    config = AdaptiveComputationConfig(hidden_dim=8, num_hidden_layers=1, dropout=0)
    controller = AdaptiveComputationController(input_dim=6, config=config)
    output = controller(torch.randn(2, 4, 6))

    loss, terms = controller.supervised_loss(
        output,
        horizon_labels=torch.tensor([10, 50]),
        num_steps_labels=torch.tensor([2, 8]),
    )
    loss.backward()

    assert loss.ndim == 0
    assert set(terms) == {"adaptive_horizon_loss", "adaptive_num_steps_loss"}
    assert controller.horizon_head.weight.grad is not None
    assert controller.num_steps_head.weight.grad is not None

    with pytest.raises(ValueError, match="Invalid horizon labels"):
        controller.supervised_loss(output, horizon_labels=torch.tensor([30, 50]))


def test_runtime_budget_uses_conservative_batch_max_and_supports_overrides():
    config = AdaptiveComputationConfig(hidden_dim=8, num_hidden_layers=1, dropout=0)
    controller = AdaptiveComputationController(input_dim=4, config=config)
    output = controller(torch.randn(2, 3, 4))
    output.horizons = torch.tensor([10, 50])
    output.num_steps = torch.tensor([2, 8])

    budget = resolve_runtime_budget(output, config, default_horizon=50, default_num_steps=10)
    overridden = resolve_runtime_budget(
        output,
        config,
        default_horizon=50,
        default_num_steps=10,
        action_horizon_override=20,
        num_steps_override=4,
    )

    assert budget.action_horizon == 50
    assert budget.num_steps == 8
    assert overridden.action_horizon == 20
    assert overridden.num_steps == 4


def test_offline_grid_search_selects_cheapest_successful_budget():
    # Shape: tasks x horizons x flow steps.
    success_rates = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[0.3, 0.4], [0.6, 0.7], [0.8, 0.9]],
        ]
    )

    horizons, num_steps = select_optimal_budget_labels(
        success_rates,
        horizon_choices=(10, 20, 50),
        num_steps_choices=(2, 8),
        success_threshold=0.9,
    )

    assert horizons.tolist() == [10, 50]
    assert num_steps.tolist() == [2, 8]


def test_smolvla_config_rejects_horizon_larger_than_training_chunk():
    with pytest.raises(ValueError, match="cannot exceed SmolVLA chunk_size"):
        SmolVLAConfig(
            chunk_size=50,
            adaptive_computation=AdaptiveComputationConfig(horizon_choices=(10, 20, 60)),
        )


def test_select_action_forwards_runtime_budget_overrides_to_chunk_generation():
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(rtc_config=None, n_action_steps=10)
    policy._queues = {ACTION: deque(maxlen=10)}
    received_kwargs = {}

    def prepare_batch(self, batch):
        return batch

    def get_action_chunk(self, batch, noise=None, **kwargs):
        received_kwargs.update(kwargs)
        return torch.zeros(1, 10, 7)

    policy._prepare_batch = MethodType(prepare_batch, policy)
    policy._get_action_chunk = MethodType(get_action_chunk, policy)

    action = policy.select_action({}, action_horizon=10, num_inference_steps=2)

    assert action.shape == (1, 7)
    assert received_kwargs == {"action_horizon": 10, "num_inference_steps": 2}
