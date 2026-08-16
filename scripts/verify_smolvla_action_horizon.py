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

"""Verify that SmolVLA's runtime action horizon changes the generated tensor length."""

from __future__ import annotations

import argparse
from contextlib import nullcontext

from smolvla_adaptive_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_VLM_PATH,
    HORIZON_CHOICES,
    get_real_libero_batch,
    load_policy_and_processors,
    make_libero_env_config,
    make_policy_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--vlm-path", default=str(DEFAULT_VLM_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

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

    autocast = (
        torch.autocast(device_type=torch.device(config.device).type)
        if config.use_amp
        else nullcontext()
    )
    results = []
    with torch.inference_mode(), autocast:
        for horizon in HORIZON_CHOICES:
            actions = policy.predict_action_chunk(
                batch,
                action_horizon=horizon,
                num_inference_steps=args.num_inference_steps,
            )
            actual = int(actions.shape[1])
            if actual != horizon:
                raise AssertionError(
                    f"Runtime action horizon failed: requested H={horizon}, returned length={actual}."
                )
            results.append((horizon, actual, tuple(actions.shape)))

    print("SmolVLA runtime action-horizon verification passed:")
    for requested, actual, shape in results:
        print(f"  H={requested}: length={actual}, shape={shape}")
    print(f"n_action_steps remained fixed at {policy.config.n_action_steps}.")


if __name__ == "__main__":
    main()
