"""The fixed 21-class identity space for a seven-player, two-wolf game."""

from itertools import combinations

import numpy as np


PLAYER_IDS = tuple(range(1, 8))
WOLF_PAIRS = tuple(combinations(PLAYER_IDS, 2))
PAIR_TO_INDEX = {pair: index for index, pair in enumerate(WOLF_PAIRS)}
NUM_WOLF_PAIRS = len(WOLF_PAIRS)


def normalize_pair(pair) -> tuple[int, int]:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError("wolf pair must contain exactly two player ids")
    if any(type(player_id) is not int for player_id in pair):
        raise ValueError("wolf pair player ids must be integers")
    normalized = tuple(sorted(pair))
    if normalized[0] == normalized[1]:
        raise ValueError("wolf pair must contain two distinct players")
    if normalized not in PAIR_TO_INDEX:
        raise ValueError("wolf pair player ids must be between 1 and 7")
    return normalized


def pair_index(pair) -> int:
    return PAIR_TO_INDEX[normalize_pair(pair)]


def pair_at(index: int) -> tuple[int, int]:
    if type(index) is not int or not 0 <= index < NUM_WOLF_PAIRS:
        raise ValueError(f"pair index must be between 0 and {NUM_WOLF_PAIRS - 1}")
    return WOLF_PAIRS[index]


def validate_pair_mask(mask) -> np.ndarray:
    values = np.asarray(mask)
    if values.shape != (NUM_WOLF_PAIRS,):
        raise ValueError(f"pair mask must have shape [{NUM_WOLF_PAIRS}]")
    if not np.all(np.logical_or(values == 0, values == 1)):
        raise ValueError("pair mask values must be boolean")
    values = values.astype(bool)
    if not values.any():
        raise ValueError("pair mask must keep at least one class")
    return values


def masked_softmax(logits, mask) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (NUM_WOLF_PAIRS,):
        raise ValueError(f"logits must have shape [{NUM_WOLF_PAIRS}]")
    valid = validate_pair_mask(mask)
    shifted = values[valid] - np.max(values[valid])
    probabilities = np.zeros(NUM_WOLF_PAIRS, dtype=np.float64)
    probabilities[valid] = np.exp(shifted)
    probabilities /= probabilities.sum()
    return probabilities


def pair_probabilities_to_player_marginals(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (NUM_WOLF_PAIRS,):
        raise ValueError(f"probabilities must have shape [{NUM_WOLF_PAIRS}]")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("probabilities must be finite and non-negative")
    total = values.sum()
    if not np.isclose(total, 1.0):
        raise ValueError("pair probabilities must sum to one")
    marginals = np.zeros(len(PLAYER_IDS), dtype=np.float64)
    for probability, pair in zip(values, WOLF_PAIRS):
        for player_id in pair:
            marginals[player_id - 1] += probability
    return marginals
