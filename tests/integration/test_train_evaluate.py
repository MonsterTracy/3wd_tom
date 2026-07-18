import json
from copy import deepcopy
from pathlib import Path

import torch
import pytest
import transformers

import werewolf.tom.evaluation as tom_evaluation
import werewolf.tom.training as tom_training
from werewolf.events.encoder import ENCODER_SCHEMA_VERSION, KIND2ID, VALUE2ID
from werewolf.prompt_protocol import (
    checkpoint_prompt_metadata,
    protocol_id_from_references,
)
from werewolf.tom.collection import build_audit_report
from werewolf.tom.evaluation import (
    PREDICTION_FIELDS,
    PREDICTION_SCHEMA_VERSION,
    evaluate_from_config,
    validate_evaluate_config,
)
from werewolf.tom.losses import compute_training_losses
from werewolf.tom.pair_space import PLAYER_IDS, WOLF_PAIRS
from werewolf.tom.training import (
    evaluate_loader,
    train_from_config,
    validate_train_config,
)


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


@pytest.mark.parametrize(
    "architecture,marginal_bce_weight",
    [
        ("gpt2block", 0.0),
        ("gpt2block", 0.25),
        ("gru", 0.25),
        ("boe_mlp", 0.25),
    ],
)
def test_tiny_train_and_evaluate_smoke(
    tmp_path, architecture, marginal_bce_weight, monkeypatch
):
    first_order = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    second_state = deepcopy(first_order)
    second_state["sample_id"] = "fixture:first:second-state"
    second_state["state_id"] = "fixture:e5"
    second_state["public_state_id"] = "fixture:e5"
    valid_record = deepcopy(first_order)
    valid_record["game_id"] = "fixture-valid"
    second_valid_record = deepcopy(valid_record)
    second_valid_record["sample_id"] = "fixture-valid:first:second-state"
    second_valid_record["state_id"] = "fixture-valid:e5"
    second_valid_record["public_state_id"] = "fixture-valid:e5"
    train_path = _write_dataset_run(
        tmp_path, [first_order, second_state], "game_001"
    )
    valid_path = _write_dataset_run(
        tmp_path, [valid_record, second_valid_record], "game_002"
    )
    output_dir = tmp_path / "run"
    train_config = {
        "schema_version": "train.v2",
        "data": {
            "train_paths": [str(train_path)],
            "valid_paths": [str(valid_path)],
            "task": "first_order",
            "mode": "private_conditioned",
            "include_first_order_private": True,
        },
        "model": {
            "architecture": architecture,
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
            "marginal_bce_weight": marginal_bce_weight,
            "seed": 7,
            "device": "cpu",
            "output_dir": str(output_dir),
        },
    }
    report = train_from_config(train_config)
    epoch = report["history"][0]
    assert epoch["valid"]["samples"] == 2
    assert {
        "train_pair_loss",
        "train_marginal_bce",
        "train_total_loss",
        "valid_pair_loss",
        "valid_marginal_bce",
        "valid_total_loss",
    } <= epoch.keys()
    assert report["checkpoint_selection_metric"] == "valid_pair_loss"
    assert report["best_valid_pair_loss"] == epoch["valid_pair_loss"]
    assert report["valid_marginal_bce_at_best"] == epoch["valid_marginal_bce"]
    assert report["valid_total_loss_at_best"] == epoch["valid_total_loss"]
    assert report["best_epoch"] == 1
    if marginal_bce_weight == 0:
        assert epoch["train_total_loss"] == epoch["train_pair_loss"]
        assert epoch["valid_total_loss"] == epoch["valid_pair_loss"]
    assert not {
        "normalized_player_marginal_kl",
        "normalized_player_marginal_cross_entropy",
        "player_marginal_brier",
        "player_top2_recall",
    } & report["history"][0]["valid"].keys()
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["schema_version"] == "model.v2"
    assert checkpoint["config"]["schema_version"] == "train.v2"
    assert (
        checkpoint["config"]["training"]["marginal_bce_weight"]
        == marginal_bce_weight
    )
    assert checkpoint["checkpoint_selection_metric"] == "valid_pair_loss"
    assert checkpoint["valid_pair_loss"] == epoch["valid_pair_loss"]
    assert checkpoint["valid_marginal_bce"] == epoch["valid_marginal_bce"]
    assert checkpoint["valid_total_loss"] == epoch["valid_total_loss"]
    assert checkpoint["architecture"] == architecture
    assert checkpoint["config"]["model"]["architecture"] == architecture
    assert checkpoint["transformers_version"] == transformers.__version__
    if architecture == "gpt2block":
        assert checkpoint["gpt2_config"] == {
            "vocab_size": 1,
            "n_positions": 16,
            "n_ctx": 16,
            "n_embd": 8,
            "n_layer": 1,
            "n_head": 2,
            "n_inner": 32,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "use_cache": False,
            "add_cross_attention": False,
            "bos_token_id": None,
            "eos_token_id": None,
        }
    else:
        assert checkpoint["gpt2_config"] is None
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
    evaluation_config = {
        "schema_version": "evaluate.v1",
        "checkpoint": str(output_dir / "best.pt"),
        "data_paths": [str(valid_path)],
        "batch_size": 2,
        "device": "cpu",
        "include_first_order_private": True,
        "output": str(evaluation_path),
    }
    evaluation = evaluate_from_config(evaluation_config)
    assert set(evaluation["by_task"]) == {"first_order"}
    assert json.loads(evaluation_path.read_text(encoding="utf-8"))["overall"]
    assert evaluation["prediction_schema_version"] == PREDICTION_SCHEMA_VERSION
    assert evaluation["pair_space"] == [list(pair) for pair in WOLF_PAIRS]
    assert evaluation["player_ids"] == list(PLAYER_IDS)
    prediction_path = evaluation_path.with_name("evaluation.predictions.jsonl")
    assert evaluation["predictions_output"] == str(prediction_path)
    prediction_lines = prediction_path.read_text(encoding="utf-8").splitlines()
    predictions = [json.loads(line) for line in prediction_lines]
    assert [row["sample_id"] for row in predictions] == [
        valid_record["sample_id"],
        second_valid_record["sample_id"],
    ]
    for row, source in zip(predictions, [valid_record, second_valid_record]):
        assert set(row) == PREDICTION_FIELDS
        assert row["schema_version"] == PREDICTION_SCHEMA_VERSION
        assert row["elicited_label_pair"] == source["label_pair"]
        assert row["elicited_label_index"] == source["label_index"]
        assert row["predicted_pair"] == list(
            WOLF_PAIRS[row["predicted_pair_index"]]
        )
        assert row["predicted_pair_probability"] == max(
            row["pair_probabilities"]
        )
        assert len(row["pair_probabilities"]) == 21
        assert len(row["player_marginals"]) == 7
        assert sum(row["pair_probabilities"]) == pytest.approx(1.0)
        assert sum(row["player_marginals"]) == pytest.approx(2.0)
        assert row["valid_pair_count"] == sum(source["output_mask"])
        assert row["output_mask"] == source["output_mask"]
        assert len(row["output_mask"]) == 21
        assert row["output_mask"][row["elicited_label_index"]]
        assert row["output_mask"][row["predicted_pair_index"]]
        for pair_index, allowed in enumerate(source["output_mask"]):
            if not allowed:
                assert row["pair_probabilities"][pair_index] == 0.0
        recomputed_marginals = [0.0] * 7
        for probability, pair in zip(row["pair_probabilities"], WOLF_PAIRS):
            for player_id in pair:
                recomputed_marginals[player_id - 1] += probability
        assert row["player_marginals"] == pytest.approx(recomputed_marginals)
        assert not any("actual" in field or "true" in field for field in row)
    assert {
        "negative_log_likelihood",
        "pair_accuracy",
        "pair_top_3_accuracy",
        "pair_brier",
        "player_marginal_mae",
        "normalized_player_marginal_kl",
        "normalized_player_marginal_cross_entropy",
        "player_marginal_brier",
        "player_top2_recall",
        "player_marginal_binary_cross_entropy",
    } <= evaluation["overall"].keys()

    if architecture == "gpt2block" and marginal_bce_weight == 0:
        original_summary = evaluation_path.read_text(encoding="utf-8")
        original_predictions = prediction_path.read_text(encoding="utf-8")
        assert evaluate_from_config(evaluation_config) == evaluation
        assert evaluation_path.read_text(encoding="utf-8") == original_summary
        assert prediction_path.read_text(encoding="utf-8") == original_predictions

        batch_one_path = tmp_path / "batch-one.json"
        batch_one_config = dict(evaluation_config)
        batch_one_config["batch_size"] = 1
        batch_one_config["output"] = str(batch_one_path)
        batch_one = evaluate_from_config(batch_one_config)
        assert batch_one["overall"] == pytest.approx(evaluation["overall"])
        batch_one_predictions = [
            json.loads(line)
            for line in batch_one_path.with_name(
                "batch-one.predictions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        for expected, actual in zip(predictions, batch_one_predictions):
            for field in PREDICTION_FIELDS - {
                "predicted_pair_probability",
                "pair_probabilities",
                "player_marginals",
            }:
                assert actual[field] == expected[field]
            assert actual["predicted_pair_probability"] == pytest.approx(
                expected["predicted_pair_probability"]
            )
            assert actual["pair_probabilities"] == pytest.approx(
                expected["pair_probabilities"]
            )
            assert actual["player_marginals"] == pytest.approx(
                expected["player_marginals"]
            )
        assert prediction_path.read_text(encoding="utf-8") == original_predictions

        invalid_config = dict(evaluation_config)
        invalid_config["predictions_output"] = "not-allowed.jsonl"
        with pytest.raises(ValueError, match="evaluation config fields"):
            validate_evaluate_config(invalid_config)

        failed_output = tmp_path / "failed-evaluation.json"
        failed_config = dict(evaluation_config)
        failed_config["output"] = str(failed_output)
        with monkeypatch.context() as scoped:
            scoped.setattr(
                "werewolf.tom.evaluation.player_marginals",
                lambda probabilities: torch.full(
                    (probabilities.shape[0], 7), float("nan")
                ),
            )
            with pytest.raises(ValueError, match="finite"):
                evaluate_from_config(failed_config)
        assert not failed_output.exists()
        assert not failed_output.with_name("failed-evaluation.predictions.jsonl").exists()

        nonfinite_output = tmp_path / "nonfinite-evaluation.json"
        nonfinite_config = dict(evaluation_config)
        nonfinite_config["output"] = str(nonfinite_output)
        original_prediction_record = tom_evaluation._prediction_record

        def nonfinite_prediction_record(*args):
            row = original_prediction_record(*args)
            row["predicted_pair_probability"] = float("nan")
            return row

        with monkeypatch.context() as scoped:
            scoped.setattr(
                tom_evaluation,
                "_prediction_record",
                nonfinite_prediction_record,
            )
            with pytest.raises(ValueError, match="Out of range float"):
                evaluate_from_config(nonfinite_config)
        assert not nonfinite_output.exists()
        assert not nonfinite_output.with_name(
            "nonfinite-evaluation.predictions.jsonl"
        ).exists()

        legacy_checkpoint = deepcopy(checkpoint)
        legacy_checkpoint["config"]["schema_version"] = "train.v1"
        legacy_checkpoint["config"]["training"].pop("marginal_bce_weight")
        for field in (
            "checkpoint_selection_metric",
            "valid_pair_loss",
            "valid_marginal_bce",
            "valid_total_loss",
        ):
            legacy_checkpoint.pop(field)
        legacy_path = tmp_path / "legacy-model-v2.pt"
        torch.save(legacy_checkpoint, legacy_path)
        legacy_config = dict(evaluation_config)
        legacy_config["checkpoint"] = str(legacy_path)
        legacy_config["output"] = str(tmp_path / "legacy-evaluation.json")
        assert evaluate_from_config(legacy_config)["overall"]["samples"] == 2

        model_v1 = deepcopy(checkpoint)
        model_v1["schema_version"] = "model.v1"
        model_v1_path = tmp_path / "model-v1.pt"
        torch.save(model_v1, model_v1_path)
        with pytest.raises(ValueError, match="unsupported checkpoint schema_version"):
            evaluate_from_config(
                {
                    "schema_version": "evaluate.v1",
                    "checkpoint": str(model_v1_path),
                    "data_paths": [str(valid_path)],
                    "batch_size": 2,
                    "device": "cpu",
                    "include_first_order_private": True,
                    "output": str(tmp_path / "model-v1-evaluation.json"),
                }
            )

        transformer_checkpoint = deepcopy(checkpoint)
        transformer_checkpoint["architecture"] = "transformer"
        transformer_checkpoint["config"]["model"]["architecture"] = "transformer"
        transformer_path = tmp_path / "transformer.pt"
        torch.save(transformer_checkpoint, transformer_path)
        with pytest.raises(ValueError, match="checkpoint architecture"):
            evaluate_from_config(
                {
                    "schema_version": "evaluate.v1",
                    "checkpoint": str(transformer_path),
                    "data_paths": [str(valid_path)],
                    "batch_size": 2,
                    "device": "cpu",
                    "include_first_order_private": True,
                    "output": str(tmp_path / "transformer-evaluation.json"),
                }
            )

        partial_checkpoint = deepcopy(checkpoint)
        partial_checkpoint["model_state"].pop(next(iter(partial_checkpoint["model_state"])))
        partial_path = tmp_path / "partial.pt"
        torch.save(partial_checkpoint, partial_path)
        with pytest.raises(RuntimeError, match="Missing key"):
            evaluate_from_config(
                {
                    "schema_version": "evaluate.v1",
                    "checkpoint": str(partial_path),
                    "data_paths": [str(valid_path)],
                    "batch_size": 2,
                    "device": "cpu",
                    "include_first_order_private": True,
                    "output": str(tmp_path / "partial-evaluation.json"),
                }
            )

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
        "schema_version": "train.v2",
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
            "weight_decay": 0.0, "marginal_bce_weight": 0.0,
            "seed": 7, "device": "cpu",
            "output_dir": str(tmp_path / "run"),
        },
    }
    with pytest.raises(ValueError, match="split by game_id"):
        train_from_config(config)


def _minimal_train_config(tmp_path):
    return {
        "schema_version": "train.v2",
        "data": {
            "train_paths": ["train.samples.jsonl"],
            "valid_paths": ["valid.samples.jsonl"],
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
            "weight_decay": 0.0, "marginal_bce_weight": 0.0,
            "seed": 7, "device": "cpu", "output_dir": str(tmp_path / "run"),
        },
    }


@pytest.mark.parametrize(
    "value", [-0.1, float("nan"), float("inf"), "0.1", None, 0, True]
)
def test_training_rejects_invalid_marginal_bce_weight(tmp_path, value):
    config = _minimal_train_config(tmp_path)
    config["training"]["marginal_bce_weight"] = value
    with pytest.raises(ValueError, match="marginal_bce_weight"):
        validate_train_config(config)


def test_train_v1_is_strictly_rejected(tmp_path):
    config = _minimal_train_config(tmp_path)
    config["schema_version"] = "train.v1"
    with pytest.raises(ValueError, match="schema_version"):
        validate_train_config(config)


def test_best_checkpoint_uses_pair_loss_even_when_total_loss_is_lower(
    tmp_path, monkeypatch
):
    record = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    valid_record = deepcopy(record)
    valid_record["game_id"] = "fixture-valid"
    train_path = _write_dataset_run(tmp_path, [record], "game_001")
    valid_path = _write_dataset_run(tmp_path, [valid_record], "game_002")
    config = _minimal_train_config(tmp_path)
    config["training"]["epochs"] = 2
    config["training"]["marginal_bce_weight"] = 0.5
    config["data"]["train_paths"] = [str(train_path)]
    config["data"]["valid_paths"] = [str(valid_path)]
    results = iter(
        [
            ({"samples": 1}, {"pair_loss": 1.0, "marginal_bce": 2.0, "total_loss": 2.0}),
            ({"samples": 1}, {"pair_loss": 1.1, "marginal_bce": 0.1, "total_loss": 1.15}),
        ]
    )

    monkeypatch.setattr(tom_training, "evaluate_loader", lambda *_args: next(results))
    report = train_from_config(config)
    best = torch.load(
        Path(config["training"]["output_dir"]) / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    last = torch.load(
        Path(config["training"]["output_dir"]) / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert report["best_epoch"] == 1
    assert report["best_valid_pair_loss"] == 1.0
    assert report["history"][1]["valid_total_loss"] < report["history"][0][
        "valid_total_loss"
    ]
    assert best["epoch"] == 1
    assert last["epoch"] == 2


def test_validation_losses_are_weighted_by_sample_count_across_uneven_batches():
    class EchoModel:
        def eval(self):
            return self

        def __call__(self, batch):
            return batch["logits"]

    logits = torch.stack(
        [
            torch.zeros(21),
            torch.linspace(-1, 1, 21),
            torch.linspace(1, -1, 21),
        ]
    )
    labels = torch.tensor([0, 1, 2])
    masks = torch.ones(3, 21, dtype=torch.bool)
    loader = [
        {"logits": logits[:2], "labels": labels[:2], "output_mask": masks[:2]},
        {"logits": logits[2:], "labels": labels[2:], "output_mask": masks[2:]},
    ]
    _, actual = evaluate_loader(EchoModel(), loader, torch.device("cpu"), 0.25)
    expected = compute_training_losses(logits, labels, masks, 0.25)
    for name, loss in expected.items():
        assert actual[name] == pytest.approx(loss.item())
