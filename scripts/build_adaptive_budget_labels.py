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

"""Build per-task adaptive SmolVLA budget labels from the fixed-budget grid."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smolvla_adaptive_common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_GRID_ROOT,
    HORIZON_CHOICES,
    LIBERO_SUITES,
    NUM_STEPS_CHOICES,
    load_json,
    write_json,
)


Budget = tuple[int, int]
TaskKey = tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", default=str(DEFAULT_GRID_ROOT))
    parser.add_argument(
        "--latency-json",
        default="outputs/eval/adaptive_computation/budget_latency.json",
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--output-json",
        default="outputs/adaptive_computation/task_budget_labels.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/adaptive_computation/task_budget_matrix.csv",
    )
    parser.add_argument(
        "--success-tolerance",
        type=float,
        default=1 / 3,
        help="Keep budgets whose success rate is within this value of the task maximum.",
    )
    return parser.parse_args()


def canonical_language(text: str) -> str:
    text = text.strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def read_grid(grid_root: str | Path) -> dict[TaskKey, dict[Budget, dict[str, Any]]]:
    """Read the complete 12 x 40 evaluation grid and validate its rectangular shape."""

    grid_root = Path(grid_root)
    result: dict[TaskKey, dict[Budget, dict[str, Any]]] = {}
    expected_tasks: set[TaskKey] | None = None
    for horizon in HORIZON_CHOICES:
        for num_steps in NUM_STEPS_CHOICES:
            budget = (horizon, num_steps)
            path = grid_root / f"rf80k_h{horizon}_n{num_steps}_a10" / "eval_info.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing grid result for H={horizon}, N={num_steps}: {path}")
            payload = load_json(path)
            per_task = payload.get("per_task")
            if not isinstance(per_task, list):
                raise ValueError(f"{path} does not contain an eval_info-style 'per_task' list.")

            budget_tasks: set[TaskKey] = set()
            for item in per_task:
                key = (str(item["task_group"]), int(item["task_id"]))
                if key in budget_tasks:
                    raise ValueError(f"Duplicate task {key} in {path}")
                successes = item.get("metrics", {}).get("successes")
                if not isinstance(successes, list) or not successes:
                    raise ValueError(f"Missing metrics.successes for task {key} in {path}")
                successes = [bool(value) for value in successes]
                result.setdefault(key, {})[budget] = {
                    "successes": successes,
                    "success_count": sum(successes),
                    "episode_count": len(successes),
                    "success_rate": sum(successes) / len(successes),
                }
                budget_tasks.add(key)

            if expected_tasks is None:
                expected_tasks = budget_tasks
            elif budget_tasks != expected_tasks:
                missing = sorted(expected_tasks - budget_tasks)
                extra = sorted(budget_tasks - expected_tasks)
                raise ValueError(f"Non-rectangular grid at {path}; missing={missing}, extra={extra}")

    expected_budgets = len(HORIZON_CHOICES) * len(NUM_STEPS_CHOICES)
    if expected_tasks is None or len(expected_tasks) != 40:
        raise ValueError(f"Expected exactly 40 LIBERO tasks, found {len(expected_tasks or [])}.")
    incomplete = {key: len(values) for key, values in result.items() if len(values) != expected_budgets}
    if incomplete:
        raise ValueError(f"Tasks do not have all {expected_budgets} budgets: {incomplete}")
    return result


def read_latency(path: str | Path) -> dict[Budget, float]:
    payload = load_json(path)
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Latency file {path} must contain a 'results' list.")
    latency: dict[Budget, float] = {}
    for row in rows:
        horizon = int(row.get("action_horizon", row.get("horizon")))
        num_steps = int(row.get("num_inference_steps", row.get("num_steps")))
        p50 = row.get("p50_ms", row.get("p50"))
        if p50 is not None:
            latency[(horizon, num_steps)] = float(p50)
    return latency


def get_libero_languages(task_keys: set[TaskKey]) -> dict[TaskKey, str]:
    try:
        from libero.libero import benchmark
    except ImportError as error:
        raise RuntimeError(
            "LIBERO is required to map (suite, task_id) to the official task language."
        ) from error

    benchmark_dict = benchmark.get_benchmark_dict()
    result: dict[TaskKey, str] = {}
    for suite_name in LIBERO_SUITES:
        if suite_name not in benchmark_dict:
            raise KeyError(f"LIBERO benchmark registry does not contain {suite_name!r}.")
        suite = benchmark_dict[suite_name]()
        suite_keys = sorted(key for key in task_keys if key[0] == suite_name)
        for key in suite_keys:
            task = suite.get_task(key[1])
            language = getattr(task, "language", None)
            if not language:
                raise ValueError(f"LIBERO task {key} has no language description.")
            result[key] = str(language)
    if set(result) != task_keys:
        raise ValueError(f"Could not map all grid tasks through LIBERO: {sorted(task_keys - set(result))}")
    return result


def read_dataset_task_mapping(dataset_root: str | Path) -> dict[str, tuple[int, str]]:
    import pandas as pd

    path = Path(dataset_root) / "meta" / "tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset task metadata does not exist: {path}")
    frame = pd.read_parquet(path).reset_index()
    if "task_index" not in frame.columns:
        raise ValueError(f"{path} does not contain a task_index column.")
    task_column = "task" if "task" in frame.columns else frame.columns[0]

    mapping: dict[str, tuple[int, str]] = {}
    for row in frame.to_dict(orient="records"):
        language = str(row[task_column])
        canonical = canonical_language(language)
        if canonical in mapping:
            raise ValueError(f"Dataset contains duplicate canonical task language: {language!r}")
        mapping[canonical] = (int(row["task_index"]), language)
    return mapping


def select_budget(
    task_results: dict[Budget, dict[str, Any]],
    latency: dict[Budget, float],
    success_tolerance: float,
) -> tuple[Budget, float, float, str]:
    """Choose the lowest-latency budget within tolerance of the task's best success."""

    if not 0 <= success_tolerance <= 1:
        raise ValueError("success_tolerance must be in [0, 1].")
    maximum = max(row["success_rate"] for row in task_results.values())
    threshold = max(0.0, maximum - success_tolerance)
    candidates = [
        budget
        for budget, row in task_results.items()
        if row["success_rate"] + 1e-12 >= threshold
    ]

    def cost(budget: Budget) -> tuple[float, int, int, int]:
        horizon, num_steps = budget
        return (latency.get(budget, float(horizon * num_steps)), horizon * num_steps, horizon, num_steps)

    selected = min(candidates, key=cost)
    measured = selected in latency
    selected_latency = latency.get(selected, float(selected[0] * selected[1]))
    reason = (
        f"minimum {'measured p50 latency' if measured else 'H*N proxy cost'} among "
        f"budgets with success_rate >= {threshold:.6f} (s_max={maximum:.6f})"
    )
    return selected, selected_latency, maximum, reason


def main() -> None:
    args = parse_args()
    grid = read_grid(args.grid_root)
    latency = read_latency(args.latency_json)
    languages = get_libero_languages(set(grid))
    dataset_mapping = read_dataset_task_mapping(args.dataset_root)

    task_labels = []
    matrix_rows = []
    seen_task_indices: set[int] = set()
    for (suite, task_id), task_results in sorted(grid.items()):
        official_language = languages[(suite, task_id)]
        canonical = canonical_language(official_language)
        if canonical not in dataset_mapping:
            raise KeyError(
                f"LIBERO task {suite}/{task_id} ({official_language!r}) was not found in "
                f"{Path(args.dataset_root) / 'meta/tasks.parquet'}"
            )
        dataset_task_index, dataset_language = dataset_mapping[canonical]
        if dataset_task_index in seen_task_indices:
            raise ValueError(f"Dataset task_index {dataset_task_index} mapped to multiple LIBERO tasks.")
        seen_task_indices.add(dataset_task_index)

        selected, selected_latency, maximum, reason = select_budget(
            task_results, latency, args.success_tolerance
        )
        success_by_budget = {}
        for budget, result in sorted(task_results.items()):
            horizon, num_steps = budget
            measured_latency = latency.get(budget)
            row = {
                "task_index": dataset_task_index,
                "task_name": dataset_language,
                "suite": suite,
                "suite_task_id": task_id,
                "action_horizon": horizon,
                "num_inference_steps": num_steps,
                "success_count": result["success_count"],
                "episode_count": result["episode_count"],
                "success_rate": result["success_rate"],
                "p50_latency_ms": measured_latency,
                "cost_source": "measured_p50" if measured_latency is not None else "horizon_x_nfe",
                "selection_threshold": max(0.0, maximum - args.success_tolerance),
                "selected": budget == selected,
            }
            matrix_rows.append(row)
            success_by_budget[f"h{horizon}_n{num_steps}"] = {
                key: row[key]
                for key in (
                    "success_count",
                    "episode_count",
                    "success_rate",
                    "p50_latency_ms",
                    "cost_source",
                )
            }

        task_labels.append(
            {
                "task_index": dataset_task_index,
                "task_name": dataset_language,
                "libero_language": official_language,
                "suite": suite,
                "suite_task_id": task_id,
                "selected_horizon": selected[0],
                "selected_num_steps": selected[1],
                "latency_ms": selected_latency,
                "max_success_rate": maximum,
                "selection_reason": reason,
                "success_by_budget": success_by_budget,
            }
        )

    if len(task_labels) != 40 or len(seen_task_indices) != 40:
        raise ValueError(
            f"Expected a one-to-one mapping for 40 tasks, got {len(task_labels)} labels and "
            f"{len(seen_task_indices)} task indices."
        )

    task_labels.sort(key=lambda item: item["task_index"])
    distribution = Counter(
        f"h{item['selected_horizon']}_n{item['selected_num_steps']}" for item in task_labels
    )
    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "grid_root": str(args.grid_root),
            "latency_json": str(args.latency_json),
            "dataset_tasks_parquet": str(Path(args.dataset_root) / "meta/tasks.parquet"),
            "horizon_choices": list(HORIZON_CHOICES),
            "num_steps_choices": list(NUM_STEPS_CHOICES),
            "success_tolerance": args.success_tolerance,
            "num_tasks": len(task_labels),
            "label_distribution": dict(sorted(distribution.items())),
        },
        "tasks": task_labels,
    }
    write_json(args.output_json, payload)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)

    print(f"Wrote {len(task_labels)} task labels to {args.output_json}")
    print(f"Wrote the complete {len(matrix_rows)}-row budget matrix to {args.output_csv}")
    print("Selected budget distribution:")
    for budget, count in sorted(distribution.items()):
        print(f"  {budget}: {count}")


if __name__ == "__main__":
    main()
