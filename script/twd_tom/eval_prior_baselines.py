#!/usr/bin/env python3
"""Evaluate no-learning TWD-ToM prior baselines on the eval split.

Typical command:
    PYTHONPATH=. python script/twd_tom/eval_prior_baselines.py \
      --config checkpoints/twd_tom_v05/full_game_001_060/config.yaml \
      --data_path data/twd_tom/debug/game_001_060.jsonl \
      --num_trials 10000
"""

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from script.twd_tom.train import load_config, load_jsonl_samples, split_samples
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.features import TWDToMFeatureBuilder
from werewolf.models.twd_tom.losses import twd_tom_loss
from werewolf.models.twd_tom.metrics import compute_wolf_probability_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate no-learning TWD-ToM prior baselines."
    )
    parser.add_argument(
        "--config",
        default="configs/twd_tom_train.yaml",
        help="Path to the YAML training/evaluation config.",
    )
    parser.add_argument("--data_path", help="Override data.source_path.")
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1000,
        help="Monte Carlo trials for random top-2 tie-break baselines.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed. Defaults to config data.seed.",
    )
    return parser.parse_args(argv)


def apply_cli_overrides(config: dict, args) -> dict:
    config = copy.deepcopy(config)
    if args.data_path is not None:
        config["data"]["source_path"] = args.data_path
    return config


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


def move_batch_to_device(batch: dict, device: torch.device):
    return {
        "attention_mask": batch["attention_mask"].to(device),
        "wolf_labels": batch["wolf_labels"].to(device),
    }


def make_uniform_prior_outputs(batch: dict, loss_config: dict):
    labels = batch["wolf_labels"]
    attention_mask = batch["attention_mask"]
    batch_size = labels.shape[0]
    seq_len = attention_mask.shape[1]
    num_players = labels.shape[-1]
    device = labels.device
    dtype = torch.float32

    prior_prob = float(loss_config.get("num_wolves", 2.0)) / float(num_players)
    prior_prob_tensor = torch.tensor(prior_prob, device=device, dtype=dtype)
    prior_logit = torch.logit(prior_prob_tensor)
    return {
        "wolf_logits": torch.full(
            (batch_size, seq_len, num_players),
            float(prior_logit),
            device=device,
            dtype=dtype,
        ),
        "wolf_prob": torch.full(
            (batch_size, seq_len, num_players),
            prior_prob,
            device=device,
            dtype=dtype,
        ),
        "region_probs": torch.full(
            (batch_size, seq_len, num_players, 3),
            1.0 / 3.0,
            device=device,
            dtype=dtype,
        ),
    }


def random_top2_scores(
    batch_size: int,
    num_players: int,
    generator: torch.Generator,
    device: torch.device,
):
    scores = torch.zeros(
        batch_size,
        num_players,
        dtype=torch.float32,
        device=device,
    )
    for row in range(batch_size):
        chosen = torch.randperm(num_players, generator=generator)[:2]
        scores[row, chosen.to(device)] = 1.0
    return scores


def monte_carlo_top2_f1(
    data_loader,
    device: torch.device,
    num_trials: int,
    seed: int,
) -> float:
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    f1_total = 0.0
    num_examples = 0

    for raw_batch in data_loader:
        batch = move_batch_to_device(raw_batch, device)
        labels = batch["wolf_labels"]
        batch_size = labels.shape[0]
        num_players = labels.shape[-1]
        batch_f1_total = 0.0
        for _ in range(num_trials):
            scores = random_top2_scores(
                batch_size,
                num_players,
                generator,
                device,
            )
            metrics = compute_wolf_probability_metrics(
                scores,
                labels,
                attention_mask=batch["attention_mask"],
            )
            batch_f1_total += float(metrics["top2_f1"])
        f1_total += (batch_f1_total / num_trials) * batch_size
        num_examples += batch_size

    if num_examples == 0:
        raise ValueError("no evaluation examples")
    return f1_total / num_examples


def uniform_prior_eval_loss(
    data_loader,
    device: torch.device,
    loss_config: dict,
) -> float:
    loss_total = 0.0
    num_examples = 0

    for raw_batch in data_loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = make_uniform_prior_outputs(batch, loss_config)
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
                f"non-finite uniform eval loss: {total_loss.detach().cpu()}"
            )
        batch_size = batch["wolf_labels"].shape[0]
        loss_total += float(total_loss.detach()) * batch_size
        num_examples += batch_size

    if num_examples == 0:
        raise ValueError("no evaluation examples")
    return loss_total / num_examples


def evaluate_prior_baselines(
    data_loader,
    device: torch.device,
    loss_config: dict,
    num_trials: int,
    seed: int,
) -> dict:
    return {
        "uniform_eval_loss": uniform_prior_eval_loss(
            data_loader,
            device,
            loss_config,
        ),
        "uniform_top2_f1": monte_carlo_top2_f1(
            data_loader,
            device,
            num_trials,
            seed,
        ),
        "random_top2_f1": monte_carlo_top2_f1(
            data_loader,
            device,
            num_trials,
            seed + 1,
        ),
    }


def format_baseline_summary(metrics: dict) -> str:
    return (
        "uniform_eval_loss="
        f"{float(metrics['uniform_eval_loss']):.6f} "
        "uniform_top2_f1="
        f"{float(metrics['uniform_top2_f1']):.6f}\n"
        "random_top2_f1="
        f"{float(metrics['random_top2_f1']):.6f}"
    )


def main(argv=None):
    args = parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args)
    seed = int(args.seed if args.seed is not None else config["data"]["seed"])
    samples = load_jsonl_samples(config["data"]["source_path"])
    eval_samples = select_eval_samples(samples, config)
    eval_loader = build_eval_loader(eval_samples, config)
    metrics = evaluate_prior_baselines(
        eval_loader,
        torch.device("cpu"),
        config["loss"],
        num_trials=int(args.num_trials),
        seed=seed,
    )
    print(format_baseline_summary(metrics))


if __name__ == "__main__":
    main()
