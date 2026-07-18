import json
import math
from pathlib import Path

import pytest
import torch
import transformers
from torch import nn
from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

from werewolf.events.encoder import EVENT_TOKEN_FIELDS, VOCABULARIES
from werewolf.tom.collection import build_audit_report
from werewolf.tom.dataset import ToMDataset
from werewolf.tom.features import collate_features
from werewolf.tom.losses import (
    compute_training_losses,
    masked_pair_cross_entropy,
    player_marginal_binary_cross_entropy,
)
from werewolf.tom.metrics import (
    compute_metrics,
    compute_player_distribution_metrics,
    pair_probabilities,
    player_marginals,
)
from werewolf.tom.model import ARCHITECTURES, ToMModel, ToMModelConfig


def _dataset_path(tmp_path):
    records = [
        json.loads(line)
        for line in Path("tests/fixtures/tom_v1.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    run_dir = tmp_path / "game_001"
    run_dir.mkdir()
    samples_path = run_dir / "game_001.samples.jsonl"
    samples_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    audit = build_audit_report(records, game_ids=["fixture"])
    (run_dir / "game_001.audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    (run_dir / "game_001.failures.jsonl").touch()
    return samples_path


@pytest.mark.parametrize("architecture", ["gpt2block", "gru", "boe_mlp"])
def test_all_formal_backbones_emit_21_masked_logits(architecture, tmp_path):
    dataset = ToMDataset(_dataset_path(tmp_path))
    batch = collate_features([dataset[0], dataset[1]])
    model = ToMModel(
        ToMModelConfig(
            architecture=architecture, d_model=16, num_layers=1,
            num_heads=4, max_events=16
        )
    )
    logits = model(batch)
    loss = masked_pair_cross_entropy(logits, batch["labels"], batch["output_mask"])
    loss.backward()
    assert logits.shape == (2, 21)
    assert torch.equal(
        logits[~batch["output_mask"]],
        torch.full_like(
            logits[~batch["output_mask"]], torch.finfo(logits.dtype).min
        ),
    )
    assert torch.isfinite(loss)
    assert "pair_accuracy" in compute_metrics(
        logits.detach(), batch["labels"], batch["output_mask"]
    )


def _synthetic_batch(*, lengths=(3, 2), sequence_length=None):
    sequence_length = sequence_length or max(lengths)
    tokens = torch.ones(
        len(lengths), sequence_length, len(EVENT_TOKEN_FIELDS), dtype=torch.long
    )
    event_mask = torch.zeros(len(lengths), sequence_length, dtype=torch.bool)
    for row, length in enumerate(lengths):
        event_mask[row, :length] = True
    return {
        "event_tokens": tokens,
        "event_mask": event_mask,
        "task_id": torch.ones(len(lengths), dtype=torch.long),
        "mode_id": torch.ones(len(lengths), dtype=torch.long),
        "observer_id": torch.ones(len(lengths), dtype=torch.long),
        "modeler_id": torch.zeros(len(lengths), dtype=torch.long),
        "target_id": torch.zeros(len(lengths), dtype=torch.long),
        "output_mask": torch.ones(len(lengths), 21, dtype=torch.bool),
        "labels": torch.zeros(len(lengths), dtype=torch.long),
    }


def _small_model(**overrides):
    values = {
        "architecture": "gpt2block",
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.0,
        "max_events": 8,
        "max_day": 8,
        "use_target_embedding": True,
    }
    values.update(overrides)
    return ToMModel(ToMModelConfig(**values))


def test_gpt2block_is_the_only_canonical_transformer_architecture(monkeypatch):
    assert transformers.__version__ == "5.13.0"
    assert ARCHITECTURES == ("gpt2block", "gru", "boe_mlp")
    assert ToMModelConfig().architecture == "gpt2block"
    with pytest.raises(ValueError, match="unsupported architecture"):
        ToMModelConfig(architecture="transformer")

    def reject_pretrained(*_args, **_kwargs):
        raise AssertionError("from_pretrained must not be called")

    monkeypatch.setattr(GPT2Model, "from_pretrained", reject_pretrained)
    model = _small_model()
    assert isinstance(model.sequence_model, GPT2Model)
    assert len(model.sequence_model.h) == model.config.num_layers
    assert all(isinstance(layer, GPT2Block) for layer in model.sequence_model.h)
    assert model.sequence_model.config.n_embd == model.config.d_model
    assert model.sequence_model.config.n_head == model.config.num_heads
    assert model.sequence_model.config.use_cache is False
    assert model.sequence_model.config.add_cross_attention is False
    assert not hasattr(model.sequence_model, "lm_head")
    assert not hasattr(model, "tokenizer")
    assert not any(isinstance(module, nn.TransformerEncoder) for module in model.modules())


def test_all_fourteen_field_embeddings_are_summed_without_external_gpt_position():
    model = _small_model()
    assert len(EVENT_TOKEN_FIELDS) == 14
    assert len(model.field_embeddings) == len(EVENT_TOKEN_FIELDS)
    for field, vocabulary, embedding in zip(
        EVENT_TOKEN_FIELDS, VOCABULARIES, model.field_embeddings
    ):
        if vocabulary is not None:
            expected_size = max(vocabulary.values()) + 1
        elif field in ("speaker_id", "target_id"):
            expected_size = 8
        else:
            assert field == "day_id"
            expected_size = model.config.max_day
        assert embedding.num_embeddings == expected_size

    tokens = torch.ones(1, 3, len(EVENT_TOKEN_FIELDS), dtype=torch.long)
    expected = torch.stack(
        [
            embedding(tokens[..., index])
            for index, embedding in enumerate(model.field_embeddings)
        ]
    ).sum(dim=0)
    assert torch.equal(model._encode_tokens(tokens), expected)
    assert not hasattr(model, "position_embedding")
    assert model.sequence_model.wpe.num_embeddings == model.config.max_events

    for architecture in ("gru", "boe_mlp"):
        ablation = _small_model(architecture=architecture)
        ablation_fields = torch.stack(
            [
                embedding(tokens[..., index])
                for index, embedding in enumerate(ablation.field_embeddings)
            ]
        ).sum(dim=0)
        positions = torch.arange(tokens.shape[1])
        assert torch.equal(
            ablation._encode_tokens(tokens),
            ablation_fields + ablation.position_embedding(positions)[None, :, :],
        )


def test_gpt2block_is_causal_and_uses_padding_mask():
    torch.manual_seed(3)
    model = _small_model().eval()
    batch = _synthetic_batch(lengths=(3,))
    changed_future = {key: value.clone() for key, value in batch.items()}
    changed_future["event_tokens"][0, 2, 0] = 2
    first = model.sequence_model(
        inputs_embeds=model._encode_tokens(batch["event_tokens"]),
        attention_mask=batch["event_mask"],
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    second = model.sequence_model(
        inputs_embeds=model._encode_tokens(changed_future["event_tokens"]),
        attention_mask=changed_future["event_mask"],
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    assert torch.allclose(first[:, :2], second[:, :2])

    changed_past = {key: value.clone() for key, value in batch.items()}
    changed_past["event_tokens"][0, 0, 0] = 2
    third = model.sequence_model(
        inputs_embeds=model._encode_tokens(changed_past["event_tokens"]),
        attention_mask=changed_past["event_mask"],
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    assert not torch.allclose(first[:, -1], third[:, -1])

    padded = _synthetic_batch(lengths=(2,), sequence_length=4)
    changed_padding = {key: value.clone() for key, value in padded.items()}
    changed_padding["event_tokens"][0, 2:, :] = 2
    assert torch.allclose(model(padded), model(changed_padding))


def test_gpt2block_pools_last_valid_event_and_handles_length_boundaries():
    model = _small_model().eval()
    batch = _synthetic_batch(lengths=(3, 1), sequence_length=4)
    encoded = model._encode_tokens(batch["event_tokens"])
    hidden = model.sequence_model(
        inputs_embeds=encoded,
        attention_mask=batch["event_mask"],
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    expected = hidden[
        torch.arange(hidden.shape[0]), batch["event_mask"].sum(1) - 1
    ]
    assert torch.equal(model._pool(encoded, batch["event_mask"]), expected)
    assert model(batch).shape == (2, 21)
    assert model(_synthetic_batch(lengths=(1,))).shape == (1, 21)
    assert model(_synthetic_batch(lengths=(8,), sequence_length=8)).shape == (1, 21)
    with pytest.raises(ValueError, match="at least one event"):
        model(_synthetic_batch(lengths=(0,), sequence_length=0))


def test_conditioning_and_target_ablation_behavior_remain_explicit():
    torch.manual_seed(11)
    model = _small_model().eval()
    batch = _synthetic_batch(lengths=(2,))
    baseline = model(batch)
    for field in ("task_id", "mode_id", "observer_id", "modeler_id", "target_id"):
        changed = {key: value.clone() for key, value in batch.items()}
        changed[field].fill_(2)
        assert not torch.allclose(baseline, model(changed))

    without_target = _small_model(use_target_embedding=False).eval()
    changed_target = {key: value.clone() for key, value in batch.items()}
    changed_target["target_id"].fill_(3)
    assert torch.allclose(without_target(batch), without_target(changed_target))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_gpt2block_mps_forward_backward_without_fallback():
    model = _small_model().to("mps")
    batch = {key: value.to("mps") for key, value in _synthetic_batch().items()}
    losses = compute_training_losses(
        model(batch), batch["labels"], batch["output_mask"], 0.25
    )
    losses["total_loss"].backward()
    assert all(torch.isfinite(loss).item() for loss in losses.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpt2block_cuda_forward():
    model = _small_model().to("cuda")
    batch = {key: value.to("cuda") for key, value in _synthetic_batch().items()}
    assert model(batch).shape == (2, 21)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_player_marginals_sum_to_two_and_stay_in_unit_interval(dtype):
    probabilities = torch.full((3, 21), 1 / 21, dtype=dtype)
    marginals = player_marginals(probabilities)
    assert marginals.shape == (3, 7)
    assert torch.allclose(
        marginals.sum(1), torch.full((3,), 2.0, dtype=dtype)
    )
    assert torch.allclose(marginals, torch.full((3, 7), 2 / 7, dtype=dtype))
    assert torch.all((marginals >= 0) & (marginals <= 1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_pair_probability_has_two_hot_player_marginals(dtype):
    probabilities = torch.zeros(1, 21, dtype=dtype)
    probabilities[0, 0] = 1
    assert torch.equal(
        player_marginals(probabilities),
        torch.tensor([[1, 1, 0, 0, 0, 0, 0]], dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_evaluation_only_player_distribution_metrics(dtype):
    logits = torch.full((1, 21), -100.0, dtype=dtype)
    logits[0, 0] = 100.0
    labels = torch.tensor([0])
    mask = torch.ones(1, 21, dtype=torch.bool)

    metrics = compute_player_distribution_metrics(logits, labels, mask)

    assert metrics["normalized_player_marginal_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["normalized_player_marginal_cross_entropy"] == pytest.approx(
        math.log(2), abs=1e-6
    )
    assert metrics["player_marginal_brier"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["player_top2_recall"] == pytest.approx(1.0)


def test_uniform_player_distribution_metrics_have_expected_kl_and_cross_entropy():
    logits = torch.zeros(1, 21, dtype=torch.float64)
    labels = torch.tensor([0])
    mask = torch.ones(1, 21, dtype=torch.bool)

    metrics = compute_player_distribution_metrics(logits, labels, mask)

    assert metrics["normalized_player_marginal_cross_entropy"] == pytest.approx(
        math.log(7)
    )
    assert metrics["normalized_player_marginal_kl"] == pytest.approx(
        math.log(7 / 2)
    )


def test_player_brier_and_top2_recall_use_normalized_two_player_distributions():
    logits = torch.full((1, 21), -100.0, dtype=torch.float64)
    logits[0, 1] = 100.0  # predicts [1,3] while the elicited pair is [1,2]
    metrics = compute_player_distribution_metrics(
        logits,
        torch.tensor([0]),
        torch.ones(1, 21, dtype=torch.bool),
    )
    assert metrics["player_marginal_brier"] == pytest.approx(0.5)
    assert metrics["player_top2_recall"] == pytest.approx(0.5)


def test_training_metrics_do_not_include_evaluation_only_player_metrics():
    metrics = compute_metrics(
        torch.zeros(1, 21),
        torch.tensor([0]),
        torch.ones(1, 21, dtype=torch.bool),
    )
    assert set(metrics) == {
        "samples",
        "pair_accuracy",
        "pair_top_3_accuracy",
        "negative_log_likelihood",
        "pair_brier",
        "player_marginal_mae",
    }


def test_loss_rejects_a_label_masked_by_knowledge():
    logits = torch.zeros(1, 21)
    labels = torch.tensor([1])
    mask = torch.zeros(1, 21, dtype=torch.bool)
    mask[:, 0] = True
    with pytest.raises(ValueError, match="excluded"):
        masked_pair_cross_entropy(logits, labels, mask)


def test_zero_marginal_weight_exactly_preserves_pair_loss_and_gradients():
    torch.manual_seed(19)
    mask = torch.ones(2, 21, dtype=torch.bool)
    labels = torch.tensor([0, 8])
    legacy_logits = torch.randn(2, 21, dtype=torch.float64, requires_grad=True)
    combined_logits = legacy_logits.detach().clone().requires_grad_(True)

    legacy_loss = masked_pair_cross_entropy(legacy_logits, labels, mask)
    legacy_loss.backward()
    losses = compute_training_losses(combined_logits, labels, mask, 0.0)
    losses["total_loss"].backward()

    assert torch.equal(losses["pair_loss"], legacy_loss.detach())
    assert torch.equal(losses["total_loss"], legacy_loss.detach())
    assert torch.equal(combined_logits.grad, legacy_logits.grad)


def test_perfect_pair_has_near_zero_pair_and_marginal_losses():
    logits = torch.zeros(1, 21, dtype=torch.float64)
    labels = torch.tensor([8])
    mask = torch.zeros(1, 21, dtype=torch.bool)
    mask[0, 8] = True
    losses = compute_training_losses(logits, labels, mask, 0.5)
    assert losses["pair_loss"].item() == pytest.approx(0.0)
    assert losses["marginal_bce"].item() == pytest.approx(0.0, abs=2e-7)
    assert torch.isfinite(losses["total_loss"])


def test_uniform_pair_distribution_has_exact_expected_marginal_bce():
    marginal_bce = player_marginal_binary_cross_entropy(
        torch.zeros(1, 21, dtype=torch.float64),
        torch.tensor([0]),
        torch.ones(1, 21, dtype=torch.bool),
    )
    assert marginal_bce.item() == pytest.approx(0.5982695885852573)


def test_one_shared_player_has_lower_marginal_bce_than_disjoint_pair():
    label_index = 0  # [1,2]

    def marginal_bce_for(alternative_index):
        logits = torch.zeros(1, 21, dtype=torch.float64)
        mask = torch.zeros(1, 21, dtype=torch.bool)
        mask[0, [label_index, alternative_index]] = True
        logits[0, alternative_index] = math.log(9)
        return player_marginal_binary_cross_entropy(
            logits, torch.tensor([label_index]), mask
        )

    shared = marginal_bce_for(1)  # [1,3]
    disjoint = marginal_bce_for(11)  # [3,4]
    assert shared < disjoint


def test_dynamic_mask_marginal_loss_and_gradients_are_finite():
    torch.manual_seed(23)
    logits = torch.randn(2, 21, requires_grad=True)
    labels = torch.tensor([0, 8])
    mask = torch.zeros(2, 21, dtype=torch.bool)
    mask[0, [0, 1, 5]] = True
    mask[1, [8, 9, 20]] = True
    losses = compute_training_losses(logits, labels, mask, 0.25)
    probabilities = pair_probabilities(logits, mask)
    marginals = player_marginals(probabilities)
    losses["total_loss"].backward()
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))
    assert torch.allclose(marginals.sum(1), torch.full((2,), 2.0))
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("weight", [0.0, 0.1, 0.25, 0.5])
def test_total_loss_uses_exact_marginal_weight_formula(weight):
    losses = compute_training_losses(
        torch.linspace(-1, 1, 21, dtype=torch.float64).unsqueeze(0),
        torch.tensor([0]),
        torch.ones(1, 21, dtype=torch.bool),
        weight,
    )
    assert torch.equal(
        losses["total_loss"],
        losses["pair_loss"] + weight * losses["marginal_bce"],
    )


def test_torch_masked_probabilities_are_normalized_and_exactly_zero_when_invalid():
    logits = torch.linspace(-2, 2, 42).reshape(2, 21)
    mask = torch.zeros(2, 21, dtype=torch.bool)
    mask[0, [0, 5]] = True
    mask[1, [3, 8, 20]] = True
    probabilities = pair_probabilities(logits, mask)
    assert probabilities.shape == (2, 21)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(1), torch.ones(2))
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))
    assert mask.gather(1, probabilities.argmax(dim=1, keepdim=True)).all()
