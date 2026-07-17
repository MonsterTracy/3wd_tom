"""Checkpoint evaluation with overall and task/mode slices."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from werewolf.events.encoder import ENCODER_SCHEMA_VERSION, KIND2ID, VALUE2ID
from werewolf.prompt_protocol import checkpoint_prompt_metadata
from werewolf.tom.dataset import ToMDataset
from werewolf.tom.features import MODE_IDS, TASK_IDS, collate_features
from werewolf.tom.metrics import compute_metrics
from werewolf.tom.model import ToMModel, ToMModelConfig
from werewolf.tom.pair_space import WOLF_PAIRS
from werewolf.tom.training import resolve_device


EVALUATE_SCHEMA_VERSION = "evaluate.v1"
EVALUATE_FIELDS = {
    "schema_version", "checkpoint", "data_paths", "batch_size", "device",
    "include_first_order_private", "output"
}


def validate_evaluate_config(config):
    if not isinstance(config, dict) or set(config) != EVALUATE_FIELDS:
        raise ValueError("evaluation config fields do not match evaluate.v1")
    if config["schema_version"] != EVALUATE_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation schema_version")
    if not isinstance(config["checkpoint"], str) or not config["checkpoint"]:
        raise ValueError("checkpoint is required")
    if not isinstance(config["data_paths"], list) or not config["data_paths"]:
        raise ValueError("data_paths must be a non-empty list")
    if type(config["batch_size"]) is not int or config["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    if config["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("device is invalid")
    if type(config["include_first_order_private"]) is not bool:
        raise ValueError("include_first_order_private must be boolean")
    if not isinstance(config["output"], str) or not config["output"]:
        raise ValueError("output is required")
    return True


@torch.no_grad()
def evaluate_from_config(config):
    validate_evaluate_config(config)
    device = resolve_device(config["device"])
    checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != "model.v1":
        raise ValueError("unsupported checkpoint schema_version")
    if checkpoint.get("pair_space") != [list(pair) for pair in WOLF_PAIRS]:
        raise ValueError("checkpoint pair_space does not match the canonical 21 classes")
    expected_encoder = {
        "schema_version": ENCODER_SCHEMA_VERSION,
        "kind_vocabulary": KIND2ID,
        "value_vocabulary": VALUE2ID,
    }
    if checkpoint.get("event_encoder") != expected_encoder:
        raise ValueError("checkpoint event encoder metadata is incompatible")
    dataset = ToMDataset(
        config["data_paths"],
        include_first_order_private=config["include_first_order_private"],
    )
    expected_prompt_protocol = checkpoint_prompt_metadata(
        [dataset.prompt_protocol]
    )
    if checkpoint.get("prompt_protocol") != expected_prompt_protocol:
        raise ValueError("checkpoint prompt protocol does not match evaluation data")
    model = ToMModel(ToMModelConfig(**checkpoint["config"]["model"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_features,
    )
    logits_parts = []
    labels_parts = []
    masks_parts = []
    task_parts = []
    mode_parts = []
    for batch in loader:
        device_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        logits_parts.append(model(device_batch).cpu())
        labels_parts.append(batch["labels"])
        masks_parts.append(batch["output_mask"])
        task_parts.append(batch["task_id"])
        mode_parts.append(batch["mode_id"])
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    masks = torch.cat(masks_parts)
    task_ids = torch.cat(task_parts)
    mode_ids = torch.cat(mode_parts)
    report = {
        "checkpoint": config["checkpoint"],
        "overall": compute_metrics(logits, labels, masks),
        "by_task": {},
        "by_mode": {},
    }
    for name, identifier in TASK_IDS.items():
        selected = task_ids == identifier
        if selected.any():
            report["by_task"][name] = compute_metrics(
                logits[selected], labels[selected], masks[selected]
            )
    for name, identifier in MODE_IDS.items():
        selected = mode_ids == identifier
        if selected.any():
            report["by_mode"][name] = compute_metrics(
                logits[selected], labels[selected], masks[selected]
            )
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    return report
