#!/usr/bin/env python
"""Generate (noise, action) reflow pairs for 2-Rectified Flow training.

Usage:
    nohup python generate_reflow_pairs.py > logs/reflow_gen.log 2>&1 &
"""

import sys
import torch
from pathlib import Path
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import make_policy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.device_utils import get_safe_torch_device

# ---- Config -----------------------------------------------------------
PHASE1_CKPT = "outputs/train/smolvla_dit_libero/checkpoints/last/pretrained_model"
REFLOW_DIR = "outputs/reflow/smolvla_dit_libero_rf2"
VLM_MODEL = "/home/caimu/lerobot/lerobot_models/SmolVLM2-500M-Video-Instruct"
DATASET_REPO = "lerobot/libero"
DATASET_ROOT = "/home/caimu/lerobot/lerobot_models/lerobot_libero"
BATCH_SIZE = 4        # keep low to avoid OOM during 50-step ODE
NUM_WORKERS = 2
ODE_STEPS = 50
# -----------------------------------------------------------------------

device = get_safe_torch_device("cuda", log=True)

# Dataset metadata
ds_meta = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)

# Load Phase-1 policy
policy_cfg = PreTrainedConfig.from_pretrained(PHASE1_CKPT)
policy_cfg.vlm_model_name = VLM_MODEL
policy_cfg.empty_cameras = 1
policy_cfg.device = "cuda"

policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
policy.eval().to(device)

# Dataset
dataset = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

model = policy.model
tokenizer = model.vlm_with_expert.processor.tokenizer
max_len = policy.config.tokenizer_max_length

all_noise = []
all_actions = []

for batch in tqdm(loader, desc="Generating reflow pairs"):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    tasks = [t + "\n" for t in batch["task"]]
    tok_out = tokenizer(
        tasks, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_len,
    ).to(device)
    lang_tokens = tok_out["input_ids"]
    lang_masks = tok_out["attention_mask"].bool()

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)

    z0, z1 = model.generate_reflow_pairs(
        images, img_masks, lang_tokens, lang_masks, state,
        num_sample_steps=ODE_STEPS,
    )
    all_noise.append(z0.cpu())
    all_actions.append(z1.cpu())

# Save
Path(REFLOW_DIR).mkdir(parents=True, exist_ok=True)
noise_all = torch.cat(all_noise)
actions_all = torch.cat(all_actions)
save_path = Path(REFLOW_DIR) / "reflow_pairs.pt"
torch.save({"noise": noise_all, "actions": actions_all}, save_path)
policy.save_pretrained(Path(REFLOW_DIR) / "model")
print(f"\nSaved {noise_all.shape[0]} reflow pairs to {save_path}")
