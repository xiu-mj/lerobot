#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass


@dataclass
class AdaptiveInferenceConfig:
    """Configuration for adaptive inference step selection in flow matching ODE solvers.

    When enabled, the solver monitors the relative change in action predictions
    between consecutive ODE steps and terminates early when the prediction has
    stabilized, avoiding unnecessary denoising steps.

    Usage:
        # In a policy config:
        adaptive_config: AdaptiveInferenceConfig | None = None

        # CLI override:
        --policy.adaptive_config.enabled=true
        --policy.adaptive_config.rel_threshold=0.005
    """

    enabled: bool = False

    # Minimum number of ODE steps to always execute before checking convergence.
    # These warm-up steps ensure the model has seen enough of the trajectory
    # for the convergence signal to be meaningful.
    min_steps: int = 2

    # Maximum number of ODE steps to ever execute (safety cap).
    # Also used to compute the uniform time step: dt = -1.0 / max_steps.
    max_steps: int = 10

    # Threshold on ||x_t - x_{t-1}|| / (||x_{t-1}|| + eps).
    # When the relative action change falls below this value, a step is
    # considered "stable".
    rel_threshold: float = 0.01

    # Number of consecutive stable steps required to trigger early termination.
    # Higher values reduce the risk of premature stopping due to noise.
    patience: int = 2

    def __post_init__(self):
        if self.min_steps < 1:
            raise ValueError(f"min_steps must be >= 1, got {self.min_steps}")
        if self.max_steps < self.min_steps:
            raise ValueError(f"max_steps ({self.max_steps}) must be >= min_steps ({self.min_steps})")
        if self.rel_threshold <= 0:
            raise ValueError(f"rel_threshold must be positive, got {self.rel_threshold}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
