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

"""Train only the SmolVLA adaptive budget controller on offline task labels."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from smolvla_adaptive_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_LABELS_PATH,
    DEFAULT_VLM_PATH,
    HORIZON_CHOICES,
    NUM_STEPS_CHOICES,
    RENAME_MAP,
    load_json,
    make_policy_config,
    resolve_pretrained_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--vlm-path", default=str(DEFAULT_VLM_PATH))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset-repo-id", default="lerobot/libero")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument(
        "--output-dir", default="outputs/train/smolvla_adaptive_controller_rf80k"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=250)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--max-val-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--val-tasks-per-suite", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-degenerate-labels",
        action="store_true",
        help="Train even if one head has one class or one joint label covers >=90%% of tasks.",
    )
    return parser.parse_args()


def read_labels(path: str | Path) -> tuple[dict[int, tuple[int, int]], list[dict[str, Any]]]:
    payload = load_json(path)
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list) or len(tasks) != 40:
        raise ValueError(f"{path} must contain exactly 40 task labels under 'tasks'.")

    lookup: dict[int, tuple[int, int]] = {}
    for task in tasks:
        task_index = int(task["task_index"])
        budget = (int(task["selected_horizon"]), int(task["selected_num_steps"]))
        if task_index in lookup:
            raise ValueError(f"Duplicate task_index {task_index} in {path}")
        if budget[0] not in HORIZON_CHOICES or budget[1] not in NUM_STEPS_CHOICES:
            raise ValueError(f"Unsupported label for task_index {task_index}: {budget}")
        lookup[task_index] = budget
    return lookup, tasks


def validate_label_diversity(
    labels: dict[int, tuple[int, int]], *, allow_degenerate: bool
) -> dict[str, dict[str, int]]:
    horizons = Counter(value[0] for value in labels.values())
    num_steps = Counter(value[1] for value in labels.values())
    joint = Counter(f"h{value[0]}_n{value[1]}" for value in labels.values())
    majority_fraction = max(joint.values()) / len(labels)
    degenerate = len(horizons) < 2 or len(num_steps) < 2 or majority_fraction >= 0.9
    if degenerate and not allow_degenerate:
        raise ValueError(
            "Budget labels do not support meaningful joint-controller training: "
            f"horizon={dict(horizons)}, num_steps={dict(num_steps)}, joint={dict(joint)}. "
            "Inspect the grid/label rule first, or explicitly pass --allow-degenerate-labels."
        )
    return {
        "horizon": {str(key): value for key, value in sorted(horizons.items())},
        "num_steps": {str(key): value for key, value in sorted(num_steps.items())},
        "joint": dict(sorted(joint.items())),
    }


def split_tasks(
    task_records: list[dict[str, Any]], val_tasks_per_suite: int, seed: int
) -> tuple[list[int], list[int]]:
    by_suite: dict[str, list[int]] = {}
    for task in task_records:
        by_suite.setdefault(str(task["suite"]), []).append(int(task["task_index"]))
    if len(by_suite) != 4:
        raise ValueError(f"Expected labels from four suites, got {sorted(by_suite)}")

    rng = random.Random(seed)
    validation: list[int] = []
    for suite, task_indices in sorted(by_suite.items()):
        if len(task_indices) <= val_tasks_per_suite:
            raise ValueError(f"Suite {suite} does not have enough tasks for the requested split.")
        validation.extend(rng.sample(sorted(task_indices), val_tasks_per_suite))
    validation = sorted(validation)
    training = sorted(set(task["task_index"] for task in task_records) - set(validation))
    return training, validation


def adaptive_controller_forward(policy, batch: dict[str, Any]):
    """Run the frozen multimodal prefix once, skipping the action expert and RF loss."""

    import torch

    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    with torch.no_grad():
        images, image_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        language_tokens = batch[OBS_LANGUAGE_TOKENS]
        language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embeddings, prefix_padding_mask, prefix_attention_mask = policy.model.embed_prefix(
            images, image_masks, language_tokens, language_masks, state=state
        )
        attention_mask = make_att_2d_masks(prefix_padding_mask, prefix_attention_mask)
        position_ids = torch.cumsum(prefix_padding_mask, dim=1) - 1
        prefix_outputs, _ = policy.model.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=False,
            # Match sample_actions' prefix path. In cross-attention mode this also avoids
            # routing a missing action-expert suffix through cross-attention layers.
            fill_kv_cache=True,
        )
        context = prefix_outputs[0].detach()
    return policy.model.adaptive_controller(context, prefix_padding_mask)


def add_labels_after_preprocessing(
    processed_batch: dict[str, Any], raw_task_indices, label_lookup: dict[int, tuple[int, int]]
) -> None:
    import torch

    indices = [int(value) for value in raw_task_indices.detach().cpu().reshape(-1).tolist()]
    missing = sorted(set(indices) - set(label_lookup))
    if missing:
        raise KeyError(f"The batch contains task_index values without adaptive labels: {missing}")
    device = next(value.device for value in processed_batch.values() if isinstance(value, torch.Tensor))
    processed_batch["adaptive_horizon"] = torch.tensor(
        [label_lookup[index][0] for index in indices], dtype=torch.long, device=device
    )
    processed_batch["adaptive_num_steps"] = torch.tensor(
        [label_lookup[index][1] for index in indices], dtype=torch.long, device=device
    )


def compute_controller_loss(policy, batch):
    output = adaptive_controller_forward(policy, batch)
    loss, terms = policy.model.adaptive_controller.supervised_loss(
        output,
        horizon_labels=batch["adaptive_horizon"],
        num_steps_labels=batch["adaptive_num_steps"],
        reduction="mean",
    )
    horizon_correct = output.horizons == batch["adaptive_horizon"].reshape(-1)
    step_correct = output.num_steps == batch["adaptive_num_steps"].reshape(-1)
    metrics = {
        "loss": float(loss.detach()),
        "horizon_loss": float(terms["adaptive_horizon_loss"].detach()),
        "num_steps_loss": float(terms["adaptive_num_steps_loss"].detach()),
        "horizon_accuracy": float(horizon_correct.float().mean()),
        "num_steps_accuracy": float(step_correct.float().mean()),
        "joint_accuracy": float((horizon_correct & step_correct).float().mean()),
        "difficulty_mean": float(output.difficulty.detach().mean()),
    }
    predictions = list(
        zip(output.horizons.detach().cpu().tolist(), output.num_steps.detach().cpu().tolist(), strict=True)
    )
    return loss, metrics, predictions


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No metric rows to aggregate.")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def save_training_checkpoint(
    policy,
    preprocessor,
    postprocessor,
    output_dir: Path,
    step: int,
    labels_path: Path,
    metadata_paths: list[Path],
) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(
        checkpoint_dir, config_filename="policy_preprocessor.json"
    )
    postprocessor.save_pretrained(
        checkpoint_dir, config_filename="policy_postprocessor.json"
    )
    shutil.copy2(labels_path, checkpoint_dir / "task_budget_labels.json")
    for path in metadata_paths:
        if path.is_file():
            shutil.copy2(path, checkpoint_dir / path.name)
    return checkpoint_dir


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("--steps and --batch-size must be positive.")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty training output directory: {output_dir}"
        )

    import torch

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.adaptive_computation import AdaptiveComputationConfig
    from lerobot.utils.collate import lerobot_collate_fn

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    labels_path = Path(args.labels)
    label_lookup, task_records = read_labels(labels_path)
    label_distribution = validate_label_diversity(
        label_lookup, allow_degenerate=args.allow_degenerate_labels
    )
    train_tasks, val_tasks = split_tasks(task_records, args.val_tasks_per_suite, args.seed)

    adaptive_config = AdaptiveComputationConfig(
        horizon_choices=HORIZON_CHOICES,
        num_steps_choices=NUM_STEPS_CHOICES,
        detach_context=True,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    config = make_policy_config(
        args.checkpoint,
        device=args.device,
        vlm_path=args.vlm_path,
        adaptive_config=adaptive_config,
        n_action_steps=10,
    )
    config.use_amp = args.use_amp
    checkpoint = resolve_pretrained_dir(args.checkpoint)
    dataset_meta = LeRobotDatasetMetadata(
        args.dataset_repo_id, root=args.dataset_root
    )
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=resolve_delta_timestamps(config, dataset_meta),
        return_uint8=True,
    )
    if set(label_lookup) != set(int(value) for value in dataset.meta.tasks["task_index"].tolist()):
        raise ValueError("Label task_index values do not exactly match the dataset task metadata.")

    policy = make_policy(cfg=config, ds_meta=dataset.meta, rename_map=RENAME_MAP)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(config.device)},
            "rename_observations_processor": {"rename_map": dict(RENAME_MAP)},
        },
    )
    for parameter in policy.parameters():
        parameter.requires_grad = False
    controller = policy.model.adaptive_controller
    if controller is None:
        raise RuntimeError("Adaptive controller was not created.")
    for parameter in controller.parameters():
        parameter.requires_grad = True
    policy.eval()
    controller.train()

    task_column = dataset.hf_dataset.data.column("task_index").to_numpy()
    train_indices = [index for index, task_index in enumerate(task_column) if int(task_index) in train_tasks]
    val_indices = [index for index, task_index in enumerate(task_column) if int(task_index) in val_tasks]
    if not train_indices or not val_indices:
        raise ValueError("The task split produced an empty frame split.")

    train_counts = Counter(int(task_column[index]) for index in train_indices)
    weights = [1.0 / train_counts[int(task_column[index])] for index in train_indices]
    sampler_generator = torch.Generator().manual_seed(args.seed)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(train_indices), replacement=True, generator=sampler_generator
    )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=str(config.device).startswith("cuda"),
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(config.device).startswith("cuda"),
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        controller.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    device_type = torch.device(config.device).type
    def autocast_context():
        return torch.autocast(device_type=device_type) if args.use_amp else nullcontext()

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_metrics.jsonl"
    training_config_path = output_dir / "adaptive_training_config.json"
    split_path = output_dir / "task_split.json"
    write_json(
        training_config_path,
        {
            **vars(args),
            "checkpoint": str(checkpoint),
            "adaptive_config": asdict(adaptive_config),
            "label_distribution": label_distribution,
            "num_train_frames": len(train_indices),
            "num_val_frames": len(val_indices),
        },
    )
    write_json(split_path, {"train_task_indices": train_tasks, "val_task_indices": val_tasks})
    shutil.copy2(labels_path, output_dir / "task_budget_labels.json")

    def evaluate() -> tuple[dict[str, float], Counter]:
        controller.eval()
        rows: list[dict[str, float]] = []
        predictions: Counter = Counter()
        with torch.inference_mode(), autocast_context():
            for batch_index, raw_batch in enumerate(val_loader):
                if batch_index >= args.max_val_batches:
                    break
                raw_task_indices = raw_batch["task_index"]
                batch = preprocessor(raw_batch)
                add_labels_after_preprocessing(batch, raw_task_indices, label_lookup)
                _, metrics, batch_predictions = compute_controller_loss(policy, batch)
                rows.append(metrics)
                predictions.update(f"h{horizon}_n{num_steps}" for horizon, num_steps in batch_predictions)
        controller.train()
        return aggregate_metrics(rows), predictions

    train_iterator = iter(train_loader)
    recent_rows: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        try:
            raw_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            raw_batch = next(train_iterator)
        raw_task_indices = raw_batch["task_index"]
        batch = preprocessor(raw_batch)
        add_labels_after_preprocessing(batch, raw_task_indices, label_lookup)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            loss, metrics, _ = compute_controller_loss(policy, batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), args.grad_clip_norm)
        optimizer.step()
        metrics["grad_norm"] = float(grad_norm)
        recent_rows.append(metrics)

        should_log = step == 1 or step % args.log_freq == 0
        should_eval = step == 1 or step % args.eval_freq == 0 or step == args.steps
        record: dict[str, Any] = {"step": step}
        if should_log:
            record["train"] = aggregate_metrics(recent_rows)
            recent_rows.clear()
        if should_eval:
            val_metrics, val_predictions = evaluate()
            record["validation"] = val_metrics
            record["validation_predicted_budget_distribution"] = dict(sorted(val_predictions.items()))
        if len(record) > 1:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False))

        if step % args.save_freq == 0 or step == args.steps:
            saved = save_training_checkpoint(
                policy,
                preprocessor,
                postprocessor,
                output_dir,
                step,
                labels_path,
                [training_config_path, split_path, log_path],
            )
            print(f"Saved controller checkpoint: {saved}")

    print(f"Training complete. Metrics: {log_path}")


if __name__ == "__main__":
    main()
