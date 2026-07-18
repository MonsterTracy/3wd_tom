"""Losses for the global pair objective and optional player auxiliary."""

import math
import torch
from torch.nn import functional as F

from werewolf.tom.metrics import PAIR_MEMBERSHIP, pair_probabilities, player_marginals
from werewolf.tom.pair_space import NUM_WOLF_PAIRS


def masked_pair_cross_entropy(logits, labels, output_mask):
    if logits.ndim != 2 or logits.shape[1] != NUM_WOLF_PAIRS:
        raise ValueError(f"logits must have shape [batch,{NUM_WOLF_PAIRS}]")
    if labels.shape != (logits.shape[0],):
        raise ValueError("labels must have shape [batch]")
    if output_mask.shape != logits.shape or output_mask.dtype != torch.bool:
        raise ValueError("output_mask must be a boolean tensor matching logits")
    if not output_mask.any(dim=1).all():
        raise ValueError("every sample must retain at least one output class")
    if not output_mask.gather(1, labels[:, None]).all():
        raise ValueError("a label is excluded by its output mask")
    masked_logits = logits.masked_fill(~output_mask, torch.finfo(logits.dtype).min)
    return F.cross_entropy(masked_logits, labels)


def player_marginal_binary_cross_entropy(logits, labels, output_mask):
    probabilities = pair_probabilities(logits, output_mask)
    marginals = player_marginals(probabilities)
    targets = PAIR_MEMBERSHIP.to(
        device=logits.device, dtype=logits.dtype
    )[labels.to(logits.device)]
    epsilon = max(torch.finfo(logits.dtype).eps, 1e-7)
    safe_marginals = marginals.clamp(epsilon, 1 - epsilon)
    return F.binary_cross_entropy(safe_marginals, targets)


def compute_training_losses(logits, labels, output_mask, marginal_bce_weight):
    if (
        type(marginal_bce_weight) is not float
        or not math.isfinite(marginal_bce_weight)
        or marginal_bce_weight < 0
    ):
        raise ValueError("marginal_bce_weight must be a finite non-negative float")
    pair_loss = masked_pair_cross_entropy(logits, labels, output_mask)
    marginal_bce = player_marginal_binary_cross_entropy(
        logits, labels, output_mask
    )
    return {
        "pair_loss": pair_loss,
        "marginal_bce": marginal_bce,
        "total_loss": pair_loss + marginal_bce_weight * marginal_bce,
    }
