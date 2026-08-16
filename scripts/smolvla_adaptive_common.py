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

"""Shared helpers for the SmolVLA adaptive-computation experiment scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = Path(
    "outputs/train/smolvla_libero_4suite_rf_uniform/checkpoints/080000/pretrained_model"
)
DEFAULT_ADAPTIVE_RUN = Path("outputs/train/smolvla_adaptive_controller_rf80k")
DEFAULT_DATASET_ROOT = Path("/home/caimu/lerobot/lerobot_models/lerobot_libero")
DEFAULT_VLM_PATH = Path("/home/caimu/lerobot/lerobot_models/SmolVLM2-500M-Video-Instruct")
DEFAULT_GRID_ROOT = Path("outputs/eval/adaptive_computation/grid_ep3")
DEFAULT_LABELS_PATH = Path("outputs/adaptive_computation/task_budget_labels.json")

HORIZON_CHOICES = (10, 20, 50)
NUM_STEPS_CHOICES = (1, 2, 4, 8)
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CAMERA_NAME_MAPPING = {
    "agentview_image": "camera1",
    "robot0_eye_in_hand_image": "camera2",
}
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def resolve_pretrained_dir(path: str | Path) -> Path:
    """Resolve either a pretrained-model directory or the latest checkpoint in a run."""

    path = Path(path)
    if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
        return path

    candidates = sorted(path.glob("checkpoints/*/pretrained_model"))
    candidates = [candidate for candidate in candidates if (candidate / "config.json").is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No pretrained model found at {path} or below {path / 'checkpoints'}"
        )
    return candidates[-1]


def make_policy_config(
    checkpoint: str | Path,
    *,
    device: str,
    vlm_path: str | Path,
    adaptive_config=None,
    n_action_steps: int = 10,
):
    """Load a SmolVLA config and apply runtime-only experiment overrides."""

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    checkpoint = resolve_pretrained_dir(checkpoint)
    config = SmolVLAConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = device
    config.vlm_model_name = str(vlm_path)
    config.n_action_steps = n_action_steps
    config.empty_cameras = 1
    config.chunk_size = max(config.chunk_size, max(HORIZON_CHOICES))
    config.compile_model = False
    config.rtc_config = None
    config.adaptive_computation = adaptive_config
    return config


def make_libero_env_config(
    suites: tuple[str, ...] | list[str] = LIBERO_SUITES,
    task_ids: list[int] | None = None,
):
    from lerobot.envs.configs import LiberoEnv

    return LiberoEnv(
        task=",".join(suites),
        task_ids=task_ids,
        camera_name_mapping=dict(CAMERA_NAME_MAPPING),
        max_parallel_tasks=1,
    )


def load_policy_and_processors(config, env_config, checkpoint: str | Path):
    """Load a policy plus the exact checkpoint processors used by evaluation."""

    from lerobot.policies import make_policy, make_pre_post_processors

    policy = make_policy(cfg=config, env_cfg=env_config, rename_map=RENAME_MAP)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(resolve_pretrained_dir(checkpoint)),
        preprocessor_overrides={
            "device_processor": {"device": str(config.device)},
            "rename_observations_processor": {"rename_map": dict(RENAME_MAP)},
        },
    )
    return policy, preprocessor, postprocessor


def get_real_libero_batch(
    policy,
    preprocessor,
    env_config,
    *,
    suite: str = "libero_spatial",
    task_id: int = 0,
    seed: int = 1000,
):
    """Reset one real LIBERO environment and apply the normal evaluation preprocessors."""

    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation

    single_env_config = make_libero_env_config((suite,), [task_id])
    envs = make_env(single_env_config, n_envs=1, use_async_envs=False)
    env = envs[suite][task_id]
    try:
        observation, _ = env.reset(seed=[seed])
        observation = preprocess_observation(observation)
        try:
            observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            observation["task"] = [""]
        env_preprocessor, _ = make_env_pre_post_processors(
            env_cfg=env_config, policy_cfg=policy.config
        )
        return preprocessor(env_preprocessor(observation))
    finally:
        env.close()


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def tensor_to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
