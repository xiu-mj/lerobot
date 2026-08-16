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

"""Evaluate fixed, adaptive, and per-task-oracle SmolVLA compute budgets."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Any

from smolvla_adaptive_common import (
    DEFAULT_ADAPTIVE_RUN,
    DEFAULT_CHECKPOINT,
    DEFAULT_GRID_ROOT,
    DEFAULT_LABELS_PATH,
    DEFAULT_VLM_PATH,
    LIBERO_SUITES,
    load_json,
    load_policy_and_processors,
    make_libero_env_config,
    make_policy_config,
    resolve_pretrained_dir,
    tensor_to_python,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fixed", "adaptive", "oracle"), required=True)
    parser.add_argument("--h", type=int, default=50, help="Fixed action horizon.")
    parser.add_argument("--n", type=int, default=2, help="Fixed RF inference steps.")
    parser.add_argument("--global-best", action="store_true")
    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument("--step-only", action="store_true")
    adaptive_group.add_argument("--chunk-only", action="store_true")
    adaptive_group.add_argument("--joint", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--adaptive-checkpoint", default=str(DEFAULT_ADAPTIVE_RUN))
    parser.add_argument("--vlm-path", default=str(DEFAULT_VLM_PATH))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--grid-root", default=str(DEFAULT_GRID_ROOT))
    parser.add_argument(
        "--latency-json", default="outputs/eval/adaptive_computation/budget_latency.json"
    )
    parser.add_argument(
        "--output-root", default="outputs/eval/adaptive_computation/ablation_ep3"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--use-async-envs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def validate_mode(args: argparse.Namespace) -> str:
    adaptive_flags = sum((args.step_only, args.chunk_only, args.joint))
    if args.mode == "adaptive" and adaptive_flags != 1:
        raise ValueError("Adaptive mode requires exactly one of --step-only, --chunk-only, or --joint.")
    if args.mode != "adaptive" and adaptive_flags:
        raise ValueError("Adaptive ablation flags can only be used with --mode adaptive.")
    if args.global_best and args.mode != "fixed":
        raise ValueError("--global-best can only be used with --mode fixed.")
    if args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("--episodes and --batch-size must be positive.")
    if args.batch_size > args.episodes:
        raise ValueError("--batch-size cannot exceed --episodes.")

    if args.mode == "fixed" and not args.global_best:
        return f"fixed_h{args.h}_n{args.n}"
    if args.mode == "fixed":
        return "fixed_global_best"
    if args.mode == "oracle":
        return "per_task_oracle"
    if args.step_only:
        return "adaptive_step_only"
    if args.chunk_only:
        return "adaptive_chunk_only"
    return "adaptive_joint"


def read_grid_success(grid_root: str | Path) -> dict[tuple[int, int], float]:
    """Return episode-weighted 4-suite success for every complete grid budget."""

    from build_adaptive_budget_labels import read_grid

    grid = read_grid(grid_root)
    aggregate: dict[tuple[int, int], list[bool]] = defaultdict(list)
    for task_results in grid.values():
        for budget, result in task_results.items():
            aggregate[budget].extend(result["successes"])
    return {budget: sum(values) / len(values) for budget, values in aggregate.items()}


def choose_global_best(grid_root: str | Path, latency_path: str | Path) -> tuple[int, int]:
    from build_adaptive_budget_labels import read_latency

    success = read_grid_success(grid_root)
    latency = read_latency(latency_path) if Path(latency_path).is_file() else {}
    best_success = max(success.values())
    candidates = [budget for budget, value in success.items() if value == best_success]
    return min(
        candidates,
        key=lambda budget: (
            latency.get(budget, float(budget[0] * budget[1])),
            budget[0] * budget[1],
            budget,
        ),
    )


def read_oracle_labels(path: str | Path) -> dict[tuple[str, int], tuple[int, int]]:
    payload = load_json(path)
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list) or len(tasks) != 40:
        raise ValueError(f"{path} must contain exactly 40 task labels.")
    labels = {
        (str(task["suite"]), int(task["suite_task_id"])): (
            int(task["selected_horizon"]),
            int(task["selected_num_steps"]),
        )
        for task in tasks
    }
    if len(labels) != 40:
        raise ValueError(f"{path} does not map 40 unique (suite, task_id) pairs.")
    return labels


class BudgetInstrumentation:
    """Inject runtime budgets into ``select_action`` and log each model invocation."""

    def __init__(self, policy, device: str) -> None:
        self.policy = policy
        self.device = device
        self.original_select_action = policy.select_action
        self.current_task: tuple[str, int] | None = None
        self.action_horizon: int | None = None
        self.num_steps: int | None = None
        self.decisions: list[dict[str, Any]] = []
        policy.select_action = MethodType(self._select_action, policy)

    def set_task(
        self,
        suite: str,
        task_id: int,
        action_horizon: int | None,
        num_steps: int | None,
    ) -> None:
        self.current_task = (suite, task_id)
        self.action_horizon = action_horizon
        self.num_steps = num_steps

    def _select_action(self, policy, batch, noise=None, **kwargs):
        import torch

        is_model_invocation = policy._check_get_actions_condition()
        runtime_kwargs = dict(kwargs)
        if self.action_horizon is not None:
            runtime_kwargs["action_horizon"] = self.action_horizon
        if self.num_steps is not None:
            runtime_kwargs["num_inference_steps"] = self.num_steps

        if is_model_invocation and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        started = time.perf_counter()
        action = self.original_select_action(batch, noise=noise, **runtime_kwargs)
        if is_model_invocation:
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(self.device))
            elapsed_ms = (time.perf_counter() - started) * 1000
            raw_decision = policy.model.last_adaptive_decision or {}
            suite, task_id = self.current_task or ("unknown", -1)
            selected_horizon = int(
                raw_decision.get("selected_horizon", self.action_horizon or policy.config.chunk_size)
            )
            selected_num_steps = int(
                raw_decision.get("selected_num_steps", self.num_steps or policy.config.num_steps)
            )
            difficulty = tensor_to_python(raw_decision.get("difficulty"))
            if isinstance(difficulty, list) and difficulty:
                difficulty_mean = statistics.fmean(float(value) for value in difficulty)
            else:
                difficulty_mean = None
            self.decisions.append(
                {
                    "invocation_index": len(self.decisions),
                    "suite": suite,
                    "task_id": task_id,
                    "selected_horizon": selected_horizon,
                    "selected_num_steps": selected_num_steps,
                    "horizon_x_nfe": selected_horizon * selected_num_steps,
                    "predicted_horizons": tensor_to_python(
                        raw_decision.get("predicted_horizons")
                    ),
                    "predicted_num_steps": tensor_to_python(
                        raw_decision.get("predicted_num_steps")
                    ),
                    "difficulty": difficulty,
                    "difficulty_mean": difficulty_mean,
                    "latency_ms": elapsed_ms,
                }
            )
        return action


def task_runtime_budget(
    args: argparse.Namespace,
    task_key: tuple[str, int],
    oracle_labels: dict[tuple[str, int], tuple[int, int]],
) -> tuple[int | None, int | None]:
    if args.mode == "fixed":
        return args.h, args.n
    if args.mode == "oracle":
        if task_key not in oracle_labels:
            raise KeyError(f"No oracle budget for {task_key}")
        return oracle_labels[task_key]
    if args.step_only:
        return 50, None
    if args.chunk_only:
        return None, 2
    return None, None


def aggregate_eval(per_task: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    group_values: dict[str, dict[str, list]] = defaultdict(
        lambda: {"sum_rewards": [], "max_rewards": [], "successes": []}
    )
    overall = {"sum_rewards": [], "max_rewards": [], "successes": []}
    for task in per_task:
        metrics = task["metrics"]
        group = group_values[task["task_group"]]
        for key in overall:
            group[key].extend(metrics[key])
            overall[key].extend(metrics[key])

    def mean(values: list) -> float:
        return statistics.fmean(float(value) for value in values) if values else float("nan")

    per_group = {
        suite: {
            "avg_sum_reward": mean(values["sum_rewards"]),
            "avg_max_reward": mean(values["max_rewards"]),
            "pc_success": mean(values["successes"]) * 100,
            "n_episodes": len(values["successes"]),
            "video_paths": [],
            "predicted_video_paths": [],
        }
        for suite, values in group_values.items()
    }
    return {
        "per_task": per_task,
        "per_group": per_group,
        "overall": {
            "avg_sum_reward": mean(overall["sum_rewards"]),
            "avg_max_reward": mean(overall["max_rewards"]),
            "pc_success": mean(overall["successes"]) * 100,
            "n_episodes": len(overall["successes"]),
            "eval_s": elapsed_s,
            "eval_ep_s": elapsed_s / max(1, len(overall["successes"])),
            "video_paths": [],
            "predicted_video_paths": [],
        },
    }


def summarize_decisions(decisions: list[dict[str, Any]], peak_memory_mib: float) -> dict[str, Any]:
    if not decisions:
        raise ValueError("No model invocation decisions were recorded.")
    horizons = [row["selected_horizon"] for row in decisions]
    num_steps = [row["selected_num_steps"] for row in decisions]
    costs = [row["horizon_x_nfe"] for row in decisions]
    latencies = [row["latency_ms"] for row in decisions]
    distribution = Counter(f"h{h}_n{n}" for h, n in zip(horizons, num_steps, strict=True))
    difficulties = [row["difficulty_mean"] for row in decisions if row["difficulty_mean"] is not None]
    return {
        "num_model_invocations": len(decisions),
        "mean_action_horizon": statistics.fmean(horizons),
        "mean_nfe": statistics.fmean(num_steps),
        "mean_horizon_x_nfe": statistics.fmean(costs),
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "peak_memory_allocated_mib": peak_memory_mib,
        "budget_distribution": dict(sorted(distribution.items())),
        "difficulty_mean": statistics.fmean(difficulties) if difficulties else None,
    }


def add_baseline_comparison(
    payload: dict[str, Any], baseline_path: Path, current_output: Path
) -> None:
    if not baseline_path.is_file() or baseline_path.resolve() == current_output.resolve():
        return
    baseline = load_json(baseline_path)
    baseline_summary = baseline.get("adaptive_summary", {})
    current_summary = payload["adaptive_summary"]
    comparison = {
        "baseline": str(baseline_path),
        "success_delta_pp": payload["overall"]["pc_success"]
        - baseline["overall"]["pc_success"],
    }
    if baseline_summary.get("latency_p50_ms") is not None:
        comparison["latency_p50_delta_ms"] = (
            current_summary["latency_p50_ms"] - baseline_summary["latency_p50_ms"]
        )
        comparison["latency_p50_delta_percent"] = (
            comparison["latency_p50_delta_ms"] / baseline_summary["latency_p50_ms"] * 100
        )
    payload["comparison_to_fixed_h50_n2"] = comparison


def main() -> None:
    args = parse_args()
    run_name = validate_mode(args)
    if args.global_best:
        args.h, args.n = choose_global_best(args.grid_root, args.latency_json)
        run_name = f"fixed_global_best_h{args.h}_n{args.n}"
        print(f"Global best fixed grid budget: H={args.h}, N={args.n}")

    import torch

    from lerobot.envs import make_env, make_env_pre_post_processors
    from lerobot.scripts.lerobot_eval import run_one
    from lerobot.utils.random_utils import set_seed

    set_seed(args.seed)
    adaptive_mode = args.mode == "adaptive"
    selected_checkpoint = (
        resolve_pretrained_dir(args.adaptive_checkpoint)
        if adaptive_mode
        else resolve_pretrained_dir(args.checkpoint)
    )
    adaptive_config = None
    if adaptive_mode:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        saved_config = SmolVLAConfig.from_pretrained(selected_checkpoint)
        adaptive_config = saved_config.adaptive_computation
        if adaptive_config is None or not adaptive_config.enabled:
            raise ValueError(
                f"Adaptive checkpoint {selected_checkpoint} does not enable adaptive_computation."
            )

    config = make_policy_config(
        selected_checkpoint,
        device=args.device,
        vlm_path=args.vlm_path,
        adaptive_config=adaptive_config,
        n_action_steps=10,
    )
    config.use_amp = args.use_amp
    env_config = make_libero_env_config(LIBERO_SUITES)
    policy, preprocessor, postprocessor = load_policy_and_processors(
        config, env_config, selected_checkpoint
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config, policy_cfg=config
    )
    instrumentation = BudgetInstrumentation(policy, args.device)
    oracle_labels = read_oracle_labels(args.labels) if args.mode == "oracle" else {}

    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval_info.json"
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing evaluation: {output_path}. "
            "Remove it or use another output root."
        )

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    started = time.time()
    per_task: list[dict[str, Any]] = []
    envs = make_env(
        env_config,
        n_envs=args.batch_size,
        use_async_envs=args.use_async_envs,
    )
    autocast = (
        torch.autocast(device_type=torch.device(config.device).type)
        if args.use_amp
        else nullcontext()
    )
    try:
        with torch.no_grad(), autocast:
            for suite, task_envs in envs.items():
                for task_id, env in task_envs.items():
                    task_key = (suite, int(task_id))
                    horizon, num_steps = task_runtime_budget(args, task_key, oracle_labels)
                    instrumentation.set_task(suite, int(task_id), horizon, num_steps)
                    first_decision = len(instrumentation.decisions)
                    try:
                        _, _, metrics = run_one(
                            suite,
                            int(task_id),
                            env,
                            policy=policy,
                            env_preprocessor=env_preprocessor,
                            env_postprocessor=env_postprocessor,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            n_episodes=args.episodes,
                            max_episodes_rendered=0,
                            videos_dir=None,
                            return_episode_data=False,
                            start_seed=args.seed,
                        )
                    finally:
                        env.close()
                    task_success_rate = sum(metrics["successes"]) / len(metrics["successes"])
                    for decision in instrumentation.decisions[first_decision:]:
                        decision["task_success_rate"] = task_success_rate
                    per_task.append(
                        {"task_group": suite, "task_id": int(task_id), "metrics": metrics}
                    )
    finally:
        for task_envs in envs.values():
            for env in task_envs.values():
                try:
                    env.close()
                except Exception:
                    pass

    peak_memory_mib = (
        torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**20
        if torch.cuda.is_available() and args.device.startswith("cuda")
        else 0.0
    )
    payload = aggregate_eval(per_task, time.time() - started)
    payload["configuration"] = {
        "mode": args.mode,
        "run_name": run_name,
        "checkpoint": str(selected_checkpoint),
        "fixed_horizon": args.h if args.mode == "fixed" else None,
        "fixed_num_steps": args.n if args.mode == "fixed" else None,
        "n_action_steps": 10,
        "episodes_per_task": args.episodes,
        "seed": args.seed,
        "suites": list(LIBERO_SUITES),
    }
    payload["adaptive_summary"] = summarize_decisions(
        instrumentation.decisions, peak_memory_mib
    )
    baseline_path = Path(args.output_root) / "fixed_h50_n2" / "eval_info.json"
    add_baseline_comparison(payload, baseline_path, output_path)
    write_json(output_path, payload)
    write_json(output_dir / "decision_summary.json", payload["adaptive_summary"])
    with (output_dir / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for decision in instrumentation.decisions:
            stream.write(json.dumps(decision, ensure_ascii=False) + "\n")

    print(json.dumps(payload["overall"], indent=2))
    print(json.dumps(payload["adaptive_summary"], indent=2))
    print(f"Wrote evaluation and decision logs to {output_dir}")


if __name__ == "__main__":
    main()
