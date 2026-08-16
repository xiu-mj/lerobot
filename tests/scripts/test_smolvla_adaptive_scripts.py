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

import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_adaptive_budget_labels import canonical_language, select_budget  # noqa: E402
from train_smolvla_adaptive_controller import (  # noqa: E402
    split_tasks,
    validate_label_diversity,
)


def test_select_budget_uses_success_tolerance_then_measured_latency():
    task_results = {
        (10, 1): {"success_rate": 2 / 3},
        (10, 2): {"success_rate": 1.0},
        (20, 1): {"success_rate": 1.0},
    }
    latency = {(10, 1): 4.0, (10, 2): 8.0, (20, 1): 6.0}

    selected, selected_latency, maximum, reason = select_budget(
        task_results, latency, success_tolerance=1 / 3
    )

    assert selected == (10, 1)
    assert selected_latency == 4.0
    assert maximum == 1.0
    assert "measured p50 latency" in reason


def test_select_budget_falls_back_to_horizon_times_nfe():
    task_results = {
        (10, 2): {"success_rate": 1.0},
        (20, 1): {"success_rate": 1.0},
        (50, 1): {"success_rate": 0.0},
    }

    selected, selected_latency, _, reason = select_budget(
        task_results, latency={}, success_tolerance=0
    )

    assert selected == (10, 2)
    assert selected_latency == 20.0
    assert "H*N proxy cost" in reason


def test_language_canonicalization_handles_libero_metadata_variants():
    assert canonical_language("Pick-up_the-red_mug!") == "pick up the red mug"


def test_label_diversity_guard_rejects_a_degenerate_controller_target():
    labels = {index: (10, 1) for index in range(40)}

    with pytest.raises(ValueError, match="meaningful joint-controller"):
        validate_label_diversity(labels, allow_degenerate=False)


def test_task_split_holds_out_two_tasks_from_every_suite():
    records = [
        {"suite": suite, "task_index": suite_index * 10 + task_id}
        for suite_index, suite in enumerate(("spatial", "object", "goal", "long"))
        for task_id in range(10)
    ]

    train_tasks, validation_tasks = split_tasks(records, val_tasks_per_suite=2, seed=1000)

    assert len(train_tasks) == 32
    assert len(validation_tasks) == 8
    for suite_index in range(4):
        suite_range = set(range(suite_index * 10, suite_index * 10 + 10))
        assert len(suite_range.intersection(validation_tasks)) == 2
