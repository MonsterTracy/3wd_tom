"""Losses for the global 21-class pair objective."""

import torch
from torch.nn import functional as F

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
