#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Rectified Flow reflow pipeline for SmolVLA.

Implements the reflow procedure from "Flow Straight and Fast" (Liu et al., ICLR 2023).

Usage:
    lerobot-reflow \
        --policy_path outputs/train/smolvla_libero/checkpoints/last/pretrained_model \
        --vlm_model_name /path/to/SmolVLM2-500M-Video-Instruct \
        --dataset_repo_id lerobot/libero \
        --dataset_root /path/to/dataset \
        --output_dir outputs/reflow/smolvla_libero_rf2 \
        --batch_size 8
"""

import argparse
import logging
import time
from pathlib import Path

import torch
from termcolor import colored
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_policy
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Rectified Flow — Generate reflow (noise, action) pairs for 2-Rectified Flow training"
    )
    # Policy
    parser.add_argument("--policy_path", type=str, required=True, help="Path to trained policy checkpoint")
    parser.add_argument("--vlm_model_name", type=str, required=True, help="Path to VLM backbone")
    parser.add_argument("--empty_cameras", type=int, default=1, help="Number of empty camera channels")
    parser.add_argument("--device", type=str, default="cuda")
    # Dataset
    parser.add_argument("--dataset_repo_id", type=str, required=True, help="Dataset HF repo id")
    parser.add_argument("--dataset_root", type=str, required=True, help="Local dataset root path")
    parser.add_argument("--dataset_episodes", type=str, default=None, help="Episode filter e.g. [0,1,2]")
    # Rename
    parser.add_argument("--rename_map", type=str, default="{}", help="JSON rename map for dataset keys")
    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for reflow pairs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--reflow_num_steps", type=int, default=50, help="ODE steps for pair generation")
    parser.add_argument("--seed", type=int, default=1000)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(colored("Rectified Flow — Generating Reflow Pairs", "green", attrs=["bold"]))

    set_seed(args.seed)
    device = get_safe_torch_device(args.device, log=True)

    # Parse rename map
    import json
    rename_map = json.loads(args.rename_map)
    dataset_episodes = json.loads(args.dataset_episodes) if args.dataset_episodes else None

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)

    # Load normalization stats from checkpoint (critical: state must be normalized
    # exactly as during training, else reflow pairs are generated with wrong context)
    from safetensors import safe_open
    from lerobot.utils.constants import OBS_STATE
    preprocessor_stats_path = Path(args.policy_path) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    state_mean = None
    state_std = None
    if preprocessor_stats_path.exists():
        with safe_open(str(preprocessor_stats_path), framework="pt") as f:
            state_mean = f.get_tensor(f"{OBS_STATE}.mean").to(device)
            state_std = f.get_tensor(f"{OBS_STATE}.std").to(device)
        logging.info(f"Loaded state normalization stats from {preprocessor_stats_path}")
    else:
        logging.warning(f"No normalization stats found at {preprocessor_stats_path}, state will NOT be normalized")

    # Build policy config
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.vlm_model_name = args.vlm_model_name
    policy_cfg.empty_cameras = args.empty_cameras
    policy_cfg.device = args.device

    logging.info(f"Loading policy from {args.policy_path}...")
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta, rename_map=rename_map)
    policy.eval()
    policy.to(device)

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=dataset_episodes,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    model = policy.model
    tokenizer = model.vlm_with_expert.processor.tokenizer
    max_token_length = policy.config.tokenizer_max_length

    all_noise = []
    all_actions = []
    total_pairs = 0
    start_time = time.time()

    logging.info(
        f"Generating reflow pairs over {len(dataset)} samples "
        f"(batch_size={args.batch_size}, ode_steps={args.reflow_num_steps})"
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating reflow pairs"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Apply rename_map to match policy's expected feature keys
            for old_key, new_key in rename_map.items():
                if old_key in batch:
                    batch[new_key] = batch.pop(old_key)

            # Tokenize task descriptions into language tokens
            tasks = list(batch["task"])  # list of strings
            task_suffix = "\n"
            tasks_with_newline = [t + task_suffix for t in tasks]
            tokenized = tokenizer(
                tasks_with_newline,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_token_length,
            ).to(device)
            lang_tokens = tokenized["input_ids"]
            lang_masks = tokenized["attention_mask"].bool()

            # Apply state normalization BEFORE padding (normalize raw state, then pad)
            from lerobot.utils.constants import OBS_STATE as _OBS_STATE
            raw_state = batch[_OBS_STATE]
            if raw_state.ndim > 2:
                raw_state = raw_state[:, -1, :]
            if state_mean is not None:
                raw_state = (raw_state - state_mean) / (state_std + 1e-8)
            batch[_OBS_STATE] = raw_state

            # Preprocess batch using policy's prepare methods
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)

            # Generate reflow pairs
            z_0, z_1 = model.generate_reflow_pairs(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                num_sample_steps=args.reflow_num_steps,
            )

            all_noise.append(z_0.cpu())
            all_actions.append(z_1.cpu())
            total_pairs += z_0.shape[0]

    # Save
    all_noise = torch.cat(all_noise, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    save_path = output_dir / "reflow_pairs.pt"
    torch.save({"noise": all_noise, "actions": all_actions}, save_path)

    elapsed = time.time() - start_time
    logging.info(
        colored(
            f"Saved {total_pairs} reflow pairs to {save_path} in {elapsed:.1f}s ({total_pairs/elapsed:.1f} pairs/s)",
            "green",
        )
    )

    # Save model config for Phase 2 training
    policy.save_pretrained(output_dir / "model")
    logging.info(f"Saved model config to {output_dir / 'model'}")


if __name__ == "__main__":
    main()
