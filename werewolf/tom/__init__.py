"""Canonical first- and second-order theory-of-mind components."""

from werewolf.tom.masks import first_order_knowledge_mask, second_order_output_mask
from werewolf.tom.pair_space import WOLF_PAIRS, pair_index
from werewolf.tom.schemas import TOM_SCHEMA_VERSION, validate_sample

__all__ = [
    "TOM_SCHEMA_VERSION",
    "WOLF_PAIRS",
    "first_order_knowledge_mask",
    "pair_index",
    "second_order_output_mask",
    "validate_sample",
]
