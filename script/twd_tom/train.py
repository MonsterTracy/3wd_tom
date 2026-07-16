#!/usr/bin/env python3
"""Train TWD-ToM models from JSONL samples with loss-only logging.

Typical command:
    PYTHONPATH=. python script/twd_tom/train.py \
      --config configs/twd_tom_train.yaml \
      --data_path data/twd_tom/debug/game_001_060.jsonl \
      --output_dir checkpoints/twd_tom_v05/full_game_001_060
"""

import argparse
import copy
import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError as exc:
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from werewolf.models.twd_tom.backbone import ToMBackboneConfig
from werewolf.models.twd_tom.dataset import (
    TWDToMDataset,
    collate_twd_tom_samples,
)
from werewolf.models.twd_tom.features import TWDToMFeatureBuilder
from werewolf.models.twd_tom.losses import twd_tom_loss
from werewolf.models.twd_tom.model import TWDToMConfig, TWDToMModel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the observer-conditioned TWD-ToM V0.5 model."
    )
    parser.add_argument(
        "--config",
        default="configs/twd_tom_train.yaml",
        help="Path to the YAML training config.",
    )
    parser.add_argument("--data_path", help="Override data.source_path.")
    parser.add_argument("--output_dir", help="Override checkpoint.output_dir.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override data.batch_size.",
    )
    parser.add_argument("--lr", type=float, help="Override train.learning_rate.")
    parser.add_argument(
        "--device",
        help="Device override: auto, cuda, cuda:N, mps, or cpu.",
    )
    return parser.parse_args(argv)


def _require_yaml():
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read the training config. "
            "Install it with: pip install PyYAML"
        ) from _YAML_IMPORT_ERROR


def resolve_project_path(path_value) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path) -> dict:
    _require_yaml()
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML mapping: {config_path}")
    return config


def apply_cli_overrides(config: dict, args) -> dict:
    config = copy.deepcopy(config)
    if args.data_path is not None:
        config["data"]["source_path"] = args.data_path
    if args.output_dir is not None:
        config["checkpoint"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["learning_rate"] = args.lr
    if args.device is not None:
        config["train"]["device"] = args.device
    return config


def load_jsonl_samples(path) -> list[dict]:
    data_path = resolve_project_path(path)
    samples = []
    with data_path.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {data_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(sample, dict):
                raise ValueError(
                    f"sample at {data_path}:{line_number} must be an object"
                )
            samples.append(sample)
    if not samples:
        raise ValueError(f"no samples found in {data_path}")
    return samples


def _sorted_unique(values):
    return sorted(set(values), key=lambda value: str(value))


def _sum_numeric(value) -> float:
    if isinstance(value, (list, tuple)):
        return sum(_sum_numeric(item) for item in value)
    return float(value)


def print_sample_stats(samples: list[dict]):
    game_ids = _sorted_unique(
        sample.get("game_id")
        for sample in samples
        if sample.get("game_id") is not None
    )
    observer_ids = _sorted_unique(
        sample.get("observer_id", sample.get("observer"))
        for sample in samples
        if sample.get("observer_id", sample.get("observer")) is not None
    )
    label_sums = Counter(
        _sum_numeric(sample.get("wolf_labels", []))
        for sample in samples
    )

    print(f"num_samples: {len(samples)}")
    print(f"num_game_ids: {len(game_ids)}")
    print(f"observer_ids: {observer_ids}")
    print(
        "wolf_label_sums: "
        f"{dict(sorted(label_sums.items(), key=lambda item: item[0]))}"
    )


def split_samples_by_game_id(
    samples: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if not samples:
        return [], []
    if any(sample.get("game_id") is None for sample in samples):
        raise ValueError(
            "split_by_game_id requires every sample to have game_id"
        )

    game_ids = _sorted_unique(sample["game_id"] for sample in samples)
    if len(game_ids) == 1:
        warnings.warn(
            "dataset has only one game_id; validation split will be empty",
            RuntimeWarning,
            stacklevel=2,
        )
        return list(samples), []

    shuffled_game_ids = list(game_ids)
    random.Random(seed).shuffle(shuffled_game_ids)
    val_game_count = round(len(shuffled_game_ids) * val_ratio)
    val_game_count = max(1, min(len(shuffled_game_ids) - 1, val_game_count))
    val_game_ids = set(shuffled_game_ids[:val_game_count])

    train_samples = [
        sample
        for sample in samples
        if sample["game_id"] not in val_game_ids
    ]
    val_samples = [
        sample
        for sample in samples
        if sample["game_id"] in val_game_ids
    ]
    return train_samples, val_samples


def split_samples_randomly(
    samples: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if len(samples) <= 1:
        return list(samples), []

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    val_count = round(len(indices) * val_ratio)
    val_count = max(1, min(len(indices) - 1, val_count))
    val_indices = set(indices[:val_count])
    train_samples = [
        sample for index, sample in enumerate(samples) if index not in val_indices
    ]
    val_samples = [
        sample for index, sample in enumerate(samples) if index in val_indices
    ]
    return train_samples, val_samples


def split_samples(
    samples: list[dict],
    split_by_game_id: bool,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if split_by_game_id:
        return split_samples_by_game_id(samples, val_ratio, seed)
    return split_samples_randomly(samples, val_ratio, seed)


def _game_ids(samples: list[dict]):
    return _sorted_unique(
        sample.get("game_id")
        for sample in samples
        if sample.get("game_id") is not None
    )


def build_dataloaders(
    train_samples: list[dict],
    val_samples: list[dict],
    config: dict,
):
    data_config = config["data"]
    feature_builder = TWDToMFeatureBuilder(
        max_seq_len=int(config["model"]["max_seq_len"])
    )
    train_dataset = TWDToMDataset(train_samples, feature_builder)
    val_dataset = TWDToMDataset(val_samples, feature_builder)
    generator = torch.Generator()
    generator.manual_seed(int(data_config["seed"]))
    common_loader_args = {
        "batch_size": int(data_config["batch_size"]),
        "num_workers": int(data_config["num_workers"]),
        "collate_fn": collate_twd_tom_samples,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=bool(data_config["shuffle_train"]),
        generator=generator,
        **common_loader_args,
    )
    val_loader = (
        DataLoader(val_dataset, shuffle=False, **common_loader_args)
        if val_samples
        else None
    )
    return train_loader, val_loader


def build_model(config: dict) -> TWDToMModel:
    model_config = config["model"]
    backbone_config = ToMBackboneConfig(
        backbone_type=model_config.get("backbone_type", "transformer"),
        num_players=int(model_config["num_players"]),
        d_model=int(model_config["d_model"]),
        n_head=int(model_config["n_head"]),
        n_layer=int(model_config["n_layer"]),
        dropout=float(model_config["dropout"]),
        max_seq_len=int(model_config["max_seq_len"]),
        max_day=int(model_config["max_day"]),
        intermediate_size=(
            int(model_config["intermediate_size"])
            if model_config.get("intermediate_size") is not None
            else None
        ),
        rope_theta=float(model_config.get("rope_theta", 10000.0)),
        use_observer_id=bool(model_config.get("use_observer_id", True)),
    )
    return TWDToMModel(TWDToMConfig(tom_config=backbone_config))


def resolve_device(requested=None) -> torch.device:
    requested = None if requested in (None, "", "auto") else requested
    if requested is not None:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                f"requested device {requested!r} is unavailable; using cpu",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        if requested == "mps" and not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            warnings.warn(
                "requested device 'mps' is unavailable; using cpu",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch_to_device(batch: dict, device: torch.device):
    return {
        "event_tokens": batch["event_tokens"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "observer_id": batch["observer_id"].to(device),
        "wolf_labels": batch["wolf_labels"].to(device),
    }


def run_epoch(
    model,
    data_loader,
    device,
    loss_config,
    optimizer=None,
    grad_clip_norm=1.0,
    log_every_steps=0,
    global_step=0,
):
    is_training = optimizer is not None
    model.train(is_training)
    loss_total = 0.0
    num_examples = 0

    for batch_index, raw_batch in enumerate(data_loader, start=1):
        batch = _move_batch_to_device(raw_batch, device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
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
                cardinality_weight=float(
                    loss_config["cardinality_weight"]
                ),
                num_wolves=float(loss_config["num_wolves"]),
                region_weight=float(loss_config["region_weight"]),
            )
            total_loss = losses["loss"]
            if not torch.isfinite(total_loss).all():
                raise FloatingPointError(
                    f"non-finite loss at batch {batch_index}: "
                    f"{total_loss.detach().cpu()}"
                )
            if is_training:
                total_loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        batch_size = batch["wolf_labels"].shape[0]
        current_loss = float(total_loss.detach())
        loss_total += current_loss * batch_size
        num_examples += batch_size

        if is_training:
            global_step += 1
            if log_every_steps > 0 and global_step % log_every_steps == 0:
                print(
                    f"step={global_step} "
                    f"train_loss={current_loss:.6f}"
                )

    average_loss = loss_total / num_examples if num_examples else None
    return average_loss, global_step


def _format_optional_loss(value) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def format_loss_summary(
    epoch: int,
    train_loss,
    eval_loss,
    lowest_eval_loss,
) -> str:
    displayed_lowest = lowest_eval_loss if eval_loss is not None else None
    return (
        f"epoch={epoch} "
        f"train_loss={_format_optional_loss(train_loss)} "
        f"eval_loss={_format_optional_loss(eval_loss)} "
        "lowest_eval_loss="
        f"{_format_optional_loss(displayed_lowest)}"
    )


def build_history_record(
    epoch: int,
    global_step: int,
    train_loss,
    eval_loss,
    lowest_eval_loss,
) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "lowest_eval_loss": lowest_eval_loss,
    }


def get_checkpoint_monitor_value(
    train_loss,
    eval_loss,
    monitor_metric: str,
):
    named_metrics = {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
    }
    return named_metrics.get(monitor_metric)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    best_metric,
    config_dict: dict,
    train_loss,
    eval_loss,
    lowest_eval_loss,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "config_dict": config_dict,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "lowest_eval_loss": lowest_eval_loss,
        },
        path,
    )


def save_training_config(config: dict, output_dir: Path):
    _require_yaml()
    with (output_dir / "config.yaml").open(
        "w",
        encoding="utf-8",
    ) as config_file:
        yaml.safe_dump(
            config,
            config_file,
            sort_keys=False,
            allow_unicode=True,
        )


def save_history(history: list[dict], output_dir: Path):
    with (output_dir / "train_history.json").open(
        "w",
        encoding="utf-8",
    ) as history_file:
        json.dump(history, history_file, ensure_ascii=False, indent=2)


def _is_improved(value: float, best_metric, mode: str) -> bool:
    if best_metric is None:
        return True
    if mode == "max":
        return value > best_metric
    if mode == "min":
        return value < best_metric
    raise ValueError("checkpoint.mode must be 'max' or 'min'")


def set_random_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    args = parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args)
    data_config = config["data"]
    train_config = config["train"]
    checkpoint_config = config["checkpoint"]
    seed = int(data_config["seed"])
    set_random_seed(seed)

    samples = load_jsonl_samples(data_config["source_path"])
    print_sample_stats(samples)
    train_samples, val_samples = split_samples(
        samples,
        split_by_game_id=bool(data_config["split_by_game_id"]),
        val_ratio=float(data_config["val_ratio"]),
        seed=seed,
    )
    print(f"train_games: {_game_ids(train_samples)}")
    print(f"val_games: {_game_ids(val_samples)}")
    print(f"train_samples: {len(train_samples)}")
    print(f"val_samples: {len(val_samples)}")

    output_dir = resolve_project_path(checkpoint_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_training_config(config, output_dir)

    train_loader, val_loader = build_dataloaders(
        train_samples,
        val_samples,
        config,
    )
    requested_device = args.device or train_config.get("device")
    device = resolve_device(requested_device)
    print(f"device: {device}")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    epochs = int(train_config["epochs"])
    eval_every = int(train_config["eval_every_epochs"])
    save_every = int(train_config["save_every_epochs"])
    if epochs <= 0:
        raise ValueError("train.epochs must be positive")
    if eval_every <= 0 or save_every <= 0:
        raise ValueError(
            "eval_every_epochs and save_every_epochs must be positive"
        )

    history = []
    global_step = 0
    best_metric = None
    lowest_eval_loss = None
    monitor_metric = checkpoint_config["monitor_metric"]
    monitor_mode = checkpoint_config["mode"]
    if monitor_mode not in ("max", "min"):
        raise ValueError("checkpoint.mode must be 'max' or 'min'")

    for epoch in range(1, epochs + 1):
        train_loss, global_step = run_epoch(
            model,
            train_loader,
            device,
            config["loss"],
            optimizer=optimizer,
            grad_clip_norm=float(train_config["grad_clip_norm"]),
            log_every_steps=int(train_config["log_every_steps"]),
            global_step=global_step,
        )
        eval_loss = None
        if val_loader is not None and epoch % eval_every == 0:
            with torch.no_grad():
                eval_loss, _ = run_epoch(
                    model,
                    val_loader,
                    device,
                    config["loss"],
                )

        if eval_loss is not None and (
            lowest_eval_loss is None
            or eval_loss < lowest_eval_loss
        ):
            lowest_eval_loss = eval_loss

        print(
            format_loss_summary(
                epoch,
                train_loss,
                eval_loss,
                lowest_eval_loss,
            )
        )

        history.append(
            build_history_record(
                epoch,
                global_step,
                train_loss,
                eval_loss,
                lowest_eval_loss,
            )
        )
        save_history(history, output_dir)

        current_metric = get_checkpoint_monitor_value(
            train_loss,
            eval_loss,
            monitor_metric,
        )
        improved = (
            bool(checkpoint_config["save_best"])
            and current_metric is not None
            and _is_improved(current_metric, best_metric, monitor_mode)
        )
        if improved:
            best_metric = current_metric

        should_save_last = epoch % save_every == 0 or epoch == epochs
        if should_save_last:
            save_checkpoint(
                output_dir / "checkpoint_last.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_metric,
                config,
                train_loss,
                eval_loss,
                lowest_eval_loss,
            )
        if improved:
            save_checkpoint(
                output_dir / "checkpoint_best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                best_metric,
                config,
                train_loss,
                eval_loss,
                lowest_eval_loss,
            )

    print(f"training finished: {output_dir}")


if __name__ == "__main__":
    main()
