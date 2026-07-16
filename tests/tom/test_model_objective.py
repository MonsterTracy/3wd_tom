import pytest
import torch

from werewolf.tom.dataset import ToMDataset
from werewolf.tom.features import collate_features
from werewolf.tom.losses import masked_pair_cross_entropy
from werewolf.tom.metrics import compute_metrics, pair_probabilities, player_marginals
from werewolf.tom.model import ToMModel, ToMModelConfig


@pytest.mark.parametrize("architecture", ["boe_mlp", "gru", "transformer"])
def test_all_formal_backbones_emit_21_masked_logits(architecture):
    dataset = ToMDataset("tests/fixtures/tom_v1.jsonl")
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
    assert torch.isfinite(loss)
    assert "pair_accuracy" in compute_metrics(
        logits.detach(), batch["labels"], batch["output_mask"]
    )


def test_player_marginals_sum_to_two():
    probabilities = torch.full((3, 21), 1 / 21)
    assert torch.allclose(player_marginals(probabilities).sum(1), torch.full((3,), 2.0))


def test_loss_rejects_a_label_masked_by_knowledge():
    logits = torch.zeros(1, 21)
    labels = torch.tensor([1])
    mask = torch.zeros(1, 21, dtype=torch.bool)
    mask[:, 0] = True
    with pytest.raises(ValueError, match="excluded"):
        masked_pair_cross_entropy(logits, labels, mask)


def test_torch_masked_probabilities_are_normalized_and_exactly_zero_when_invalid():
    logits = torch.linspace(-2, 2, 42).reshape(2, 21)
    mask = torch.zeros(2, 21, dtype=torch.bool)
    mask[0, [0, 5]] = True
    mask[1, [3, 8, 20]] = True
    probabilities = pair_probabilities(logits, mask)
    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(1), torch.ones(2))
    assert torch.equal(probabilities[~mask], torch.zeros_like(probabilities[~mask]))
