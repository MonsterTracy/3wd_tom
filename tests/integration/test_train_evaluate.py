import json
from copy import deepcopy
from pathlib import Path

import torch

from werewolf.events.encoder import ENCODER_SCHEMA_VERSION, KIND2ID, VALUE2ID
from werewolf.tom.evaluation import evaluate_from_config
from werewolf.tom.pair_space import WOLF_PAIRS
from werewolf.tom.training import train_from_config


FIXTURE = Path("tests/fixtures/tom_v1.jsonl")


def test_tiny_train_and_evaluate_smoke(tmp_path):
    first_order = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    second_state = deepcopy(first_order)
    second_state["sample_id"] = "fixture:first:second-state"
    second_state["state_id"] = "fixture:e5"
    second_state["public_state_id"] = "fixture:e5"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    train_path.write_text(
        "\n".join(json.dumps(record) for record in (first_order, second_state)) + "\n",
        encoding="utf-8",
    )
    valid_path.write_text(json.dumps(first_order) + "\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    train_config = {
        "schema_version": "train.v1",
        "data": {
            "train_paths": [str(train_path)],
            "valid_paths": [str(valid_path)],
            "task": "first_order",
            "mode": "private_conditioned",
            "include_first_order_private": True,
        },
        "model": {
            "architecture": "boe_mlp",
            "d_model": 8,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "max_events": 16,
            "max_day": 8,
            "use_target_embedding": True,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "seed": 7,
            "device": "cpu",
            "output_dir": str(output_dir),
        },
    }
    report = train_from_config(train_config)
    assert report["history"][0]["valid"]["samples"] == 1
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["pair_space"] == [list(pair) for pair in WOLF_PAIRS]
    assert checkpoint["event_encoder"] == {
        "schema_version": ENCODER_SCHEMA_VERSION,
        "kind_vocabulary": KIND2ID,
        "value_vocabulary": VALUE2ID,
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation = evaluate_from_config(
        {
            "schema_version": "evaluate.v1",
            "checkpoint": str(output_dir / "best.pt"),
            "data_paths": [str(valid_path)],
            "batch_size": 2,
            "device": "cpu",
            "include_first_order_private": True,
            "output": str(evaluation_path),
        }
    )
    assert set(evaluation["by_task"]) == {"first_order"}
    assert json.loads(evaluation_path.read_text(encoding="utf-8"))["overall"]
