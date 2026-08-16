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

"""Benchmark real latency and CUDA memory for all adaptive SmolVLA budgets."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from smolvla_adaptive_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_VLM_PATH,
    HORIZON_CHOICES,
    NUM_STEPS_CHOICES,
    get_real_libero_batch,
    load_policy_and_processors,
    make_libero_env_config,
    make_policy_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--vlm-path", default=str(DEFAULT_VLM_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--output-prefix",
        default="outputs/eval/adaptive_computation/budget_latency",
        help="Output path without .json/.csv suffixes.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    if args.warmup < 10:
        raise ValueError("--warmup must be at least 10 for a stable CUDA benchmark.")
    if args.repetitions < 100:
        raise ValueError("--repetitions must be at least 100.")

    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device and explicit synchronization.")

    config = make_policy_config(
        args.checkpoint,
        device=args.device,
        vlm_path=args.vlm_path,
        adaptive_config=None,
        n_action_steps=10,
    )
    env_config = make_libero_env_config(("libero_spatial",), [0])
    policy, preprocessor, _ = load_policy_and_processors(config, env_config, args.checkpoint)
    batch = get_real_libero_batch(
        policy, preprocessor, env_config, suite="libero_spatial", task_id=0, seed=args.seed
    )
    autocast = torch.autocast(device_type=device.type) if config.use_amp else nullcontext()

    results: list[dict] = []
    with torch.inference_mode(), autocast:
        for horizon in HORIZON_CHOICES:
            for num_steps in NUM_STEPS_CHOICES:
                for _ in range(args.warmup):
                    policy.predict_action_chunk(
                        batch, action_horizon=horizon, num_inference_steps=num_steps
                    )
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

                durations_ms = []
                for _ in range(args.repetitions):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    policy.predict_action_chunk(
                        batch, action_horizon=horizon, num_inference_steps=num_steps
                    )
                    torch.cuda.synchronize(device)
                    durations_ms.append((time.perf_counter() - started) * 1000)

                row = {
                    "action_horizon": horizon,
                    "num_inference_steps": num_steps,
                    "nfe": num_steps,
                    "horizon_x_nfe": horizon * num_steps,
                    "repetitions": args.repetitions,
                    "mean_ms": statistics.fmean(durations_ms),
                    "p50_ms": percentile(durations_ms, 0.50),
                    "p95_ms": percentile(durations_ms, 0.95),
                    "max_memory_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                    "max_memory_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
                }
                results.append(row)
                print(
                    f"H={horizon:>2}, N={num_steps}: p50={row['p50_ms']:.2f} ms, "
                    f"p95={row['p95_ms']:.2f} ms, max_mem={row['max_memory_allocated_mib']:.1f} MiB"
                )

    output_prefix = Path(args.output_prefix)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "input_suite": "libero_spatial",
            "input_task_id": 0,
            "seed": args.seed,
        },
        "results": results,
    }
    write_json(json_path, payload)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
