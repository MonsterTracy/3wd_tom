#!/usr/bin/env python3
"""Evaluate a TWD-ToM checkpoint and print eval_loss plus top2_f1 only.

Typical command:
    PYTHONPATH=. python script/twd_tom/eval.py \
      --config checkpoints/twd_tom_v05/full_game_001_060/config.yaml \
      --checkpoint_path checkpoints/twd_tom_v05/full_game_001_060/checkpoint_best.pt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from script.twd_tom.train import (
    build_model,
    load_config,
    load_jsonl_samples,
    resolve_device,
    resolve_project_path,
    split_samples,
)
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.features import TWDToMFeatureBuilder
from werewolf.models.twd_tom.losses import twd_tom_loss
from werewolf.models.twd_tom.metrics import compute_twd_tom_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a TWD-ToM checkpoint with loss/top2_f1 output."
    )
    parser.add_argument(
        "--config",
        default="configs/twd_tom_train.yaml",
        help="Path to the YAML training config used for the checkpoint.",
    )
    parser.add_argument("--data_path", help="Override data.source_path.")
    parser.add_argument(
        "--checkpoint_dir",
        help="Directory containing checkpoint_best.pt.",
    )
    parser.add_argument(
        "--checkpoint_path",
        help="Explicit checkpoint path. Defaults to checkpoint_best.pt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override data.batch_size.",
    )
    parser.add_argument(
        "--device",
        help="Device override: auto, cuda, cuda:N, mps, or cpu.",
    )
    return parser.parse_args(argv)


def apply_cli_overrides(config: dict, args) -> dict:
    config = {key: value.copy() if isinstance(value, dict) else value
              for key, value in config.items()}
    if args.data_path is not None:
        config["data"]["source_path"] = args.data_path
    if args.checkpoint_dir is not None:
        config["checkpoint"]["output_dir"] = args.checkpoint_dir
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.device is not None:
        config.setdefault("train", {})["device"] = args.device
    return config


def resolve_checkpoint_path(checkpoint_dir, checkpoint_path=None) -> Path:
    if checkpoint_path is not None:
        return resolve_project_path(checkpoint_path)
    return resolve_project_path(checkpoint_dir) / "checkpoint_best.pt"


def build_eval_loader(samples: list[dict], config: dict):
    feature_builder = TWDToMFeatureBuilder(
        max_seq_len=int(config["model"]["max_seq_len"])
    )
    dataset = TWDToMDataset(samples, feature_builder)
    return DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_twd_tom_samples,
    )


def select_eval_samples(samples: list[dict], config: dict) -> list[dict]:
    _, eval_samples = split_samples(
        samples,
        split_by_game_id=bool(config["data"]["split_by_game_id"]),
        val_ratio=float(config["data"]["val_ratio"]),
        seed=int(config["data"]["seed"]),
    )
    if not eval_samples:
        raise ValueError("evaluation split is empty")
    return eval_samples


def move_batch_to_device(batch: dict, device: torch.device):
    return {
        "event_tokens": batch["event_tokens"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "observer_id": batch["observer_id"].to(device),
        "wolf_labels": batch["wolf_labels"].to(device),
    }


def evaluate_model(model, data_loader, device, loss_config) -> dict:
    model.eval()
    loss_total = 0.0
    top2_f1_total = 0.0
    num_examples = 0

    with torch.no_grad():
        for raw_batch in data_loader:
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(
                batch["event_tokens"],
                attention_mask=batch["attention_mask"],
                observer_id=batch["observer_id"],
            )
            losses = twd_tom_loss(
                outputs,
                batch["wolf_labels"],
                attention_mask=batch["attention_mask"],
                supervision_mode=loss_config["supervision_mode"],
                bce_weight=float(loss_config["bce_weight"]),
                cardinality_weight=float(loss_config["cardinality_weight"]),
                num_wolves=float(loss_config["num_wolves"]),
                region_weight=float(loss_config["region_weight"]),
            )
            total_loss = losses["loss"]
            if not torch.isfinite(total_loss).all():
                raise FloatingPointError(
                    f"non-finite eval loss: {total_loss.detach().cpu()}"
                )

            metrics = compute_twd_tom_metrics(
                outputs,
                batch["wolf_labels"],
                attention_mask=batch["attention_mask"],
            )
            batch_size = batch["wolf_labels"].shape[0]
            loss_total += float(total_loss.detach()) * batch_size
            top2_f1_total += float(metrics["top2_f1"]) * batch_size
            num_examples += batch_size

    if num_examples == 0:
        raise ValueError("no evaluation examples")
    return {
        "eval_loss": loss_total / num_examples,
        "top2_f1": top2_f1_total / num_examples,
    }


def load_checkpoint_best(model, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def format_eval_summary(metrics: dict) -> str:
    return (
        f"eval_loss={float(metrics['eval_loss']):.6f} "
        f"top2_f1={float(metrics['top2_f1']):.6f}"
    )


def main(argv=None):
    args = parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args)
    device = resolve_device(args.device or config.get("train", {}).get("device"))

    samples = load_jsonl_samples(config["data"]["source_path"])
    eval_samples = select_eval_samples(samples, config)
    eval_loader = build_eval_loader(eval_samples, config)

    model = build_model(config).to(device)
    checkpoint_path = resolve_checkpoint_path(
        config["checkpoint"]["output_dir"],
        args.checkpoint_path,
    )
    load_checkpoint_best(model, checkpoint_path, device)

    metrics = evaluate_model(model, eval_loader, device, config["loss"])
    print(format_eval_summary(metrics))


if __name__ == "__main__":
    main()
