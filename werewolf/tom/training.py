"""Training and evaluation loops for the 21-class ToM model."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import __version__ as TRANSFORMERS_VERSION

from werewolf.events.encoder import ENCODER_SCHEMA_VERSION, KIND2ID, VALUE2ID
from werewolf.prompt_protocol import checkpoint_prompt_metadata
from werewolf.tom.dataset import ToMDataset
from werewolf.tom.features import collate_features
from werewolf.tom.losses import masked_pair_cross_entropy
from werewolf.tom.metrics import compute_metrics
from werewolf.tom.model import ToMModel, ToMModelConfig
from werewolf.tom.pair_space import WOLF_PAIRS


TRAIN_SCHEMA_VERSION = "train.v1"
TRAIN_FIELDS = {"schema_version", "data", "model", "training"}
DATA_FIELDS = {
    "train_paths", "valid_paths", "task", "mode", "include_first_order_private"
}
TRAINING_FIELDS = {
    "epochs", "batch_size", "learning_rate", "weight_decay", "seed", "device", "output_dir"
}


def validate_train_config(config):
    if not isinstance(config, dict) or set(config) != TRAIN_FIELDS:
        raise ValueError("training config fields do not match train.v1")
    if config["schema_version"] != TRAIN_SCHEMA_VERSION:
        raise ValueError("unsupported training schema_version")
    if not isinstance(config["data"], dict) or set(config["data"]) != DATA_FIELDS:
        raise ValueError("training data fields do not match train.v1")
    for name in ("train_paths", "valid_paths"):
        paths = config["data"][name]
        if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
            raise ValueError(f"data.{name} must be a non-empty list of paths")
    if config["data"]["task"] not in (None, "first_order", "second_order"):
        raise ValueError("data.task is invalid")
    if config["data"]["mode"] not in (
        None, "private_conditioned", "public_only", "wolf_conditioned"
    ):
        raise ValueError("data.mode is invalid")
    if type(config["data"]["include_first_order_private"]) is not bool:
        raise ValueError("include_first_order_private must be boolean")
    ToMModelConfig(**config["model"])
    if not isinstance(config["training"], dict) or set(config["training"]) != TRAINING_FIELDS:
        raise ValueError("training fields do not match train.v1")
    for name in ("epochs", "batch_size", "seed"):
        if type(config["training"][name]) is not int or config["training"][name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    for name in ("learning_rate", "weight_decay"):
        if not isinstance(config["training"][name], (int, float)) or config["training"][name] < 0:
            raise ValueError(f"training.{name} must be non-negative")
    if config["training"]["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("training.device is invalid")
    if not isinstance(config["training"]["output_dir"], str):
        raise ValueError("training.output_dir is required")
    return True


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    logits_parts = []
    label_parts = []
    mask_parts = []
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        batch = _move(batch, device)
        logits = model(batch)
        loss = masked_pair_cross_entropy(logits, batch["labels"], batch["output_mask"])
        count = logits.shape[0]
        total_loss += loss.item() * count
        total_samples += count
        logits_parts.append(logits.cpu())
        label_parts.append(batch["labels"].cpu())
        mask_parts.append(batch["output_mask"].cpu())
    if not total_samples:
        raise ValueError("evaluation loader is empty")
    metrics = compute_metrics(
        torch.cat(logits_parts), torch.cat(label_parts), torch.cat(mask_parts)
    )
    metrics["loss"] = total_loss / total_samples
    metrics["samples"] = total_samples
    return metrics


def train_from_config(config):
    validate_train_config(config)
    training = config["training"]
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = config["data"]
    dataset_kwargs = {
        "task": data["task"],
        "mode": data["mode"],
        "include_first_order_private": data["include_first_order_private"],
    }
    train_dataset = ToMDataset(data["train_paths"], **dataset_kwargs)
    valid_dataset = ToMDataset(data["valid_paths"], **dataset_kwargs)
    overlapping_games = sorted(train_dataset.game_ids & valid_dataset.game_ids)
    if overlapping_games:
        raise ValueError(
            "train and validation data must be split by game_id; "
            f"overlap={overlapping_games}"
        )
    prompt_protocol_metadata = checkpoint_prompt_metadata(
        [train_dataset.prompt_protocol, valid_dataset.prompt_protocol]
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=collate_features,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training["batch_size"],
        shuffle=False,
        collate_fn=collate_features,
    )
    device = resolve_device(training["device"])
    model = ToMModel(ToMModelConfig(**config["model"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    output_dir = Path(training["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = masked_pair_cross_entropy(logits, batch["labels"], batch["output_mask"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * logits.shape[0]
            train_samples += logits.shape[0]
        valid_metrics = evaluate_loader(model, valid_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / train_samples,
            "valid": valid_metrics,
        }
        history.append(epoch_record)
        checkpoint = {
            "schema_version": "model.v2",
            "architecture": model.config.architecture,
            "transformers_version": TRANSFORMERS_VERSION,
            "gpt2_config": model.gpt2_config_metadata(),
            "pair_space": [list(pair) for pair in WOLF_PAIRS],
            "event_encoder": {
                "schema_version": ENCODER_SCHEMA_VERSION,
                "kind_vocabulary": dict(KIND2ID),
                "value_vocabulary": dict(VALUE2ID),
            },
            "prompt_protocol": prompt_protocol_metadata,
            "epoch": epoch,
            "config": config,
            "model_state": model.state_dict(),
            "valid_metrics": valid_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if valid_metrics["loss"] < best_loss:
            best_loss = valid_metrics["loss"]
            torch.save(checkpoint, output_dir / "best.pt")
    report = {"device": str(device), "best_valid_loss": best_loss, "history": history}
    with (output_dir / "history.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    return report
