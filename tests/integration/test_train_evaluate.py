import json
from copy import deepcopy
from pathlib import Path

import torch
import pytest

from werewolf.events.encoder import ENCODER_SCHEMA_VERSION, KIND2ID, VALUE2ID
from werewolf.prompt_protocol import (
    checkpoint_prompt_metadata,
    protocol_id_from_references,
)
from werewolf.tom.collection import build_audit_report
from werewolf.tom.evaluation import evaluate_from_config
from werewolf.tom.pair_space import WOLF_PAIRS
from werewolf.tom.training import train_from_config


FIXTURE = Path("tests/fixtures/tom_v1.jsonl")


def _write_dataset_run(tmp_path, records, run_id, *, audit_records=None):
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    samples_path = run_dir / f"{run_id}.samples.jsonl"
    samples_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    audited = records if audit_records is None else audit_records
    audit = build_audit_report(
        audited,
        game_ids=sorted({record["game_id"] for record in audited}),
    )
    (run_dir / f"{run_id}.audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    (run_dir / f"{run_id}.failures.jsonl").touch()
    return samples_path


def test_tiny_train_and_evaluate_smoke(tmp_path):
    first_order = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    second_state = deepcopy(first_order)
    second_state["sample_id"] = "fixture:first:second-state"
    second_state["state_id"] = "fixture:e5"
    second_state["public_state_id"] = "fixture:e5"
    valid_record = deepcopy(first_order)
    valid_record["game_id"] = "fixture-valid"
    train_path = _write_dataset_run(
        tmp_path, [first_order, second_state], "game_001"
    )
    valid_path = _write_dataset_run(tmp_path, [valid_record], "game_002")
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
    assert checkpoint["prompt_protocol"] == checkpoint_prompt_metadata(
        [first_order["prompt_protocol"]]
    )
    assert checkpoint["prompt_protocol"]["prompt_language"] == "zh-CN"
    assert checkpoint["prompt_protocol"]["prompt_protocol_version"] == (
        "prompt_protocol.zh.v6"
    )
    assert checkpoint["prompt_protocol"]["gameplay_prompt_version"] == (
        "gameplay.zh.v4"
    )
    assert checkpoint["prompt_protocol"]["belief_prompt_version"] == (
        "belief.zh.v3"
    )
    assert checkpoint["prompt_protocol"]["parser_prompt_version"] == (
        "parser.zh.v3"
    )
    assert checkpoint["prompt_protocol"]["ruleset"] == (
        first_order["prompt_protocol"]["ruleset"]
    )
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

    mismatched = deepcopy(first_order)
    mismatched["prompt_protocol"]["parser"]["sha256"] = "0" * 64
    references = {
        name: mismatched["prompt_protocol"][name]
        for name in ("gameplay", "belief", "parser")
    }
    mismatched["prompt_protocol"]["protocol_id"] = protocol_id_from_references(
        references
    )
    mismatched_path = _write_dataset_run(
        tmp_path,
        [mismatched],
        "game_003",
        audit_records=[first_order],
    )
    with pytest.raises(ValueError, match="prompt.protocol|canonical prompt"):
        evaluate_from_config(
            {
                "schema_version": "evaluate.v1",
                "checkpoint": str(output_dir / "best.pt"),
                "data_paths": [str(mismatched_path)],
                "batch_size": 2,
                "device": "cpu",
                "include_first_order_private": True,
                "output": str(tmp_path / "mismatch-evaluation.json"),
            }
        )

    ruleset_mismatch = deepcopy(checkpoint)
    ruleset_mismatch["prompt_protocol"]["ruleset"]["sha256"] = "0" * 64
    mismatched_checkpoint = tmp_path / "ruleset-mismatch.pt"
    torch.save(ruleset_mismatch, mismatched_checkpoint)
    with pytest.raises(ValueError, match="prompt protocol"):
        evaluate_from_config(
            {
                "schema_version": "evaluate.v1",
                "checkpoint": str(mismatched_checkpoint),
                "data_paths": [str(valid_path)],
                "batch_size": 2,
                "device": "cpu",
                "include_first_order_private": True,
                "output": str(tmp_path / "ruleset-mismatch-evaluation.json"),
            }
        )


def test_training_rejects_a_game_split_across_train_and_validation(tmp_path):
    record = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    train_path = _write_dataset_run(tmp_path, [record], "game_001")
    valid_path = _write_dataset_run(tmp_path, [record], "game_002")
    config = {
        "schema_version": "train.v1",
        "data": {
            "train_paths": [str(train_path)],
            "valid_paths": [str(valid_path)],
            "task": "first_order",
            "mode": "private_conditioned",
            "include_first_order_private": True,
        },
        "model": {
            "architecture": "boe_mlp", "d_model": 8, "num_layers": 1,
            "num_heads": 2, "dropout": 0.0, "max_events": 16,
            "max_day": 8, "use_target_embedding": True,
        },
        "training": {
            "epochs": 1, "batch_size": 1, "learning_rate": 0.001,
            "weight_decay": 0.0, "seed": 7, "device": "cpu",
            "output_dir": str(tmp_path / "run"),
        },
    }
    with pytest.raises(ValueError, match="split by game_id"):
        train_from_config(config)
