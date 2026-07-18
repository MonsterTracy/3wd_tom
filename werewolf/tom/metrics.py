"""Metrics for pair probabilities and their seven-player marginals."""

import torch

from werewolf.tom.pair_space import NUM_WOLF_PAIRS, WOLF_PAIRS


PAIR_MEMBERSHIP = torch.tensor(
    [[float(player_id in pair) for player_id in range(1, 8)] for pair in WOLF_PAIRS],
    dtype=torch.float32,
)


def pair_probabilities(logits, output_mask):
    if logits.ndim != 2 or logits.shape[1] != NUM_WOLF_PAIRS:
        raise ValueError(f"logits must have shape [batch,{NUM_WOLF_PAIRS}]")
    if output_mask.shape != logits.shape:
        raise ValueError("output_mask shape must match logits")
    if output_mask.dtype != torch.bool or not output_mask.any(dim=1).all():
        raise ValueError("every output mask must be boolean and keep one class")
    masked = logits.masked_fill(~output_mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked, dim=1)


def player_marginals(probabilities):
    membership = PAIR_MEMBERSHIP.to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    return probabilities @ membership


def compute_metrics(logits, labels, output_mask, *, top_k=3):
    probabilities = pair_probabilities(logits, output_mask)
    predictions = probabilities.argmax(dim=1)
    true_pair = PAIR_MEMBERSHIP.to(logits.device)[labels]
    marginals = player_marginals(probabilities)
    true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
    top_k = min(top_k, NUM_WOLF_PAIRS)
    return {
        "samples": logits.shape[0],
        "pair_accuracy": (predictions == labels).float().mean().item(),
        f"pair_top_{top_k}_accuracy": (
            probabilities.topk(top_k, dim=1).indices == labels[:, None]
        ).any(dim=1).float().mean().item(),
        "negative_log_likelihood": (-true_probability.clamp_min(1e-12).log()).mean().item(),
        "pair_brier": (
            probabilities - torch.nn.functional.one_hot(labels, NUM_WOLF_PAIRS)
        ).pow(2).sum(dim=1).mean().item(),
        "player_marginal_mae": (marginals - true_pair).abs().mean().item(),
    }


def compute_player_distribution_metrics(logits, labels, output_mask):
    """Return normalized player metrics used only by offline evaluation."""
    probabilities = pair_probabilities(logits, output_mask)
    marginals = player_marginals(probabilities)
    target_membership = PAIR_MEMBERSHIP.to(
        device=logits.device, dtype=probabilities.dtype
    )[labels.to(logits.device)]
    predicted_distribution = marginals / 2
    target_distribution = target_membership / 2
    log_prediction = predicted_distribution.clamp_min(1e-12).log()
    log_target = target_distribution.clamp_min(1e-12).log()
    top_players = marginals.topk(2, dim=1).indices
    return {
        "normalized_player_marginal_kl": (
            target_distribution * (log_target - log_prediction)
        ).sum(dim=1).mean().item(),
        "normalized_player_marginal_cross_entropy": (
            -(target_distribution * log_prediction).sum(dim=1).mean().item()
        ),
        "player_marginal_brier": (
            (predicted_distribution - target_distribution)
            .pow(2)
            .sum(dim=1)
            .mean()
            .item()
        ),
        "player_top2_recall": (
            target_membership.gather(1, top_players).sum(dim=1).div(2).mean().item()
        ),
    }
