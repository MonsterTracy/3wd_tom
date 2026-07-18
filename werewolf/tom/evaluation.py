"""Checkpoint evaluation with overall and task/mode slices."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import __version__ as TRANSFORMERS_VERSION

from werewolf.events.encoder import ENCODER_SCHEMA_VERSION, KIND2ID, VALUE2ID
from werewolf.prompt_protocol import checkpoint_prompt_metadata
from werewolf.tom.dataset import ToMDataset
from werewolf.tom.features import MODE_IDS, TASK_IDS, collate_features
from werewolf.tom.losses import player_marginal_binary_cross_entropy
from werewolf.tom.metrics import (
    compute_metrics,
    compute_player_distribution_metrics,
    pair_probabilities,
    player_marginals,
)
from werewolf.tom.model import ARCHITECTURES, ToMModel, ToMModelConfig
from werewolf.tom.pair_space import NUM_WOLF_PAIRS, PLAYER_IDS, WOLF_PAIRS
from werewolf.tom.training import resolve_device


EVALUATE_SCHEMA_VERSION = "evaluate.v1"
PREDICTION_SCHEMA_VERSION = "tom.prediction.v1"
EVALUATE_FIELDS = {
    "schema_version", "checkpoint", "data_paths", "batch_size", "device",
    "include_first_order_private", "output"
}
PREDICTION_FIELDS = {
    "schema_version",
    "sample_id",
    "game_id",
    "task",
    "mode",
    "observer_id",
    "modeler_id",
    "target_id",
    "day",
    "phase",
    "turn",
    "elicited_label_pair",
    "elicited_label_index",
    "predicted_pair",
    "predicted_pair_index",
    "predicted_pair_probability",
    "valid_pair_count",
    "output_mask",
    "pair_probabilities",
    "player_marginals",
}


class _IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


def _collate_indexed(items):
    indices, features = zip(*items)
    return torch.tensor(indices, dtype=torch.long), collate_features(list(features))


def _temporary_output(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            writer(output)
            output.flush()
            os.fsync(output.fileno())
        return temporary_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_outputs(output_path, predictions_path, report, predictions):
    def write_predictions(output):
        for prediction in predictions:
            if set(prediction) != PREDICTION_FIELDS:
                raise ValueError("prediction fields do not match tom.prediction.v1")
            output.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            output.write("\n")

    def write_report(output):
        json.dump(
            report,
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    predictions_temporary = None
    report_temporary = None
    try:
        predictions_temporary = _temporary_output(
            predictions_path, write_predictions
        )
        report_temporary = _temporary_output(output_path, write_report)
        os.replace(predictions_temporary, predictions_path)
        predictions_temporary = None
        os.replace(report_temporary, output_path)
        report_temporary = None
    finally:
        if predictions_temporary is not None:
            predictions_temporary.unlink(missing_ok=True)
        if report_temporary is not None:
            report_temporary.unlink(missing_ok=True)


def _evaluation_metrics(logits, labels, output_mask):
    return {
        **compute_metrics(logits, labels, output_mask),
        **compute_player_distribution_metrics(logits, labels, output_mask),
        "player_marginal_binary_cross_entropy": (
            player_marginal_binary_cross_entropy(
                logits, labels, output_mask
            ).item()
        ),
    }


def _prediction_record(record, probabilities, marginals, output_mask):
    if probabilities.shape != (NUM_WOLF_PAIRS,):
        raise ValueError("prediction probabilities must contain 21 values")
    if marginals.shape != (len(PLAYER_IDS),):
        raise ValueError("prediction player marginals must contain seven values")
    if output_mask.shape != (NUM_WOLF_PAIRS,) or output_mask.dtype != torch.bool:
        raise ValueError("prediction output mask is invalid")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(marginals).all():
        raise ValueError("prediction probabilities and marginals must be finite")
    if torch.any(probabilities < 0) or torch.any(probabilities > 1):
        raise ValueError("prediction probabilities must be in [0,1]")
    if torch.any(marginals < -1e-6) or torch.any(marginals > 1 + 1e-6):
        raise ValueError("prediction player marginals must be in [0,1]")
    if not torch.isclose(
        probabilities.sum(), probabilities.new_tensor(1.0), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("prediction probabilities must sum to one")
    if not torch.equal(
        probabilities[~output_mask], torch.zeros_like(probabilities[~output_mask])
    ):
        raise ValueError("invalid pair probabilities must be exactly zero")
    if not torch.isclose(
        marginals.sum(), marginals.new_tensor(2.0), atol=1e-6, rtol=1e-6
    ):
        raise ValueError("prediction player marginals must sum to two")
    canonical_marginals = player_marginals(probabilities.unsqueeze(0)).squeeze(0)
    if not torch.allclose(marginals, canonical_marginals, atol=1e-7, rtol=1e-7):
        raise ValueError("prediction player marginals do not match pair probabilities")

    elicited_index = record["label_index"]
    elicited_pair = list(record["label_pair"])
    if elicited_pair != list(WOLF_PAIRS[elicited_index]):
        raise ValueError("elicited label pair and index do not match")
    if not bool(output_mask[elicited_index]):
        raise ValueError("elicited label pair is excluded by the output mask")
    predicted_index = int(probabilities.argmax().item())
    if not bool(output_mask[predicted_index]):
        raise ValueError("predicted pair is excluded by the output mask")
    probability_values = [float(value) for value in probabilities.tolist()]
    marginal_values = [float(value) for value in marginals.tolist()]
    output_mask_values = [bool(value) for value in output_mask.tolist()]
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "sample_id": record["sample_id"],
        "game_id": record["game_id"],
        "task": record["task"],
        "mode": record["mode"],
        "observer_id": record["observer_id"],
        "modeler_id": record["modeler_id"],
        "target_id": record["target_id"],
        "day": record["day"],
        "phase": record["phase"],
        "turn": record["turn"],
        "elicited_label_pair": elicited_pair,
        "elicited_label_index": elicited_index,
        "predicted_pair": list(WOLF_PAIRS[predicted_index]),
        "predicted_pair_index": predicted_index,
        "predicted_pair_probability": probability_values[predicted_index],
        "valid_pair_count": int(output_mask.sum().item()),
        "output_mask": output_mask_values,
        "pair_probabilities": probability_values,
        "player_marginals": marginal_values,
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
    if checkpoint.get("schema_version") != "model.v2":
        raise ValueError("unsupported checkpoint schema_version")
    architecture = checkpoint.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError("checkpoint architecture is not canonical")
    model_config = checkpoint.get("config", {}).get("model")
    if not isinstance(model_config, dict) or model_config.get("architecture") != architecture:
        raise ValueError("checkpoint architecture does not match model config")
    if checkpoint.get("transformers_version") != TRANSFORMERS_VERSION:
        raise ValueError("checkpoint transformers version is incompatible")
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
    model = ToMModel(ToMModelConfig(**model_config))
    if checkpoint.get("gpt2_config") != model.gpt2_config_metadata():
        raise ValueError("checkpoint GPT2Config metadata is incompatible")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    loader = DataLoader(
        _IndexedDataset(dataset),
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=_collate_indexed,
    )
    index_parts = []
    logits_parts = []
    labels_parts = []
    masks_parts = []
    task_parts = []
    mode_parts = []
    for indices, batch in loader:
        device_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        index_parts.append(indices)
        logits_parts.append(model(device_batch).cpu())
        labels_parts.append(batch["labels"])
        masks_parts.append(batch["output_mask"])
        task_parts.append(batch["task_id"])
        mode_parts.append(batch["mode_id"])
    indices = torch.cat(index_parts)
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    masks = torch.cat(masks_parts)
    task_ids = torch.cat(task_parts)
    mode_ids = torch.cat(mode_parts)
    if sorted(indices.tolist()) != list(range(len(dataset))):
        raise ValueError("evaluation sample indices are incomplete or duplicated")
    order = indices.argsort()
    indices = indices[order]
    logits = logits[order]
    labels = labels[order]
    masks = masks[order]
    task_ids = task_ids[order]
    mode_ids = mode_ids[order]
    probabilities = pair_probabilities(logits, masks)
    marginals = player_marginals(probabilities)
    predictions = [
        _prediction_record(
            dataset.records[int(index)],
            probabilities[row],
            marginals[row],
            masks[row],
        )
        for row, index in enumerate(indices.tolist())
    ]
    output_path = Path(config["output"])
    predictions_path = output_path.with_name(
        f"{output_path.stem}.predictions.jsonl"
    )
    report = {
        "checkpoint": config["checkpoint"],
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "pair_space": [list(pair) for pair in WOLF_PAIRS],
        "player_ids": list(PLAYER_IDS),
        "predictions_output": str(predictions_path),
        "overall": _evaluation_metrics(logits, labels, masks),
        "by_task": {},
        "by_mode": {},
    }
    for name, identifier in TASK_IDS.items():
        selected = task_ids == identifier
        if selected.any():
            report["by_task"][name] = _evaluation_metrics(
                logits[selected], labels[selected], masks[selected]
            )
    for name, identifier in MODE_IDS.items():
        selected = mode_ids == identifier
        if selected.any():
            report["by_mode"][name] = _evaluation_metrics(
                logits[selected], labels[selected], masks[selected]
            )
    _write_outputs(output_path, predictions_path, report, predictions)
    return report
