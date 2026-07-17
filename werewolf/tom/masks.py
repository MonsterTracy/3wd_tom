"""Knowledge masks with separate first- and second-order semantics."""

import numpy as np

from werewolf.tom.pair_space import PLAYER_IDS, WOLF_PAIRS


FIRST_ORDER_ROLES = ("Werewolf", "Seer", "Witch", "Guard", "Villager")
SECOND_ORDER_MODES = ("public_only", "wolf_conditioned")


def _player_id(value, name):
    if type(value) is not int or value not in PLAYER_IDS:
        raise ValueError(f"{name} must be a player id between 1 and 7")
    return value


def _player_set(values, name):
    if values is None:
        return set()
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{name} must be a sequence of player ids")
    normalized = {_player_id(value, name) for value in values}
    return normalized


def first_order_constraints(
    *,
    observer_id: int,
    observer_role: str,
    known_wolves=(),
    known_good=(),
) -> dict[str, tuple[int, ...]]:
    """Return the hard pair constraints visible to one observer."""

    observer_id = _player_id(observer_id, "observer_id")
    if observer_role not in FIRST_ORDER_ROLES:
        raise ValueError(f"unsupported observer_role: {observer_role!r}")
    required_wolves = _player_set(known_wolves, "known_wolves")
    forbidden_wolves = _player_set(known_good, "known_good")
    if required_wolves & forbidden_wolves:
        raise ValueError("a player cannot be both required and forbidden")

    if observer_role == "Werewolf":
        if observer_id not in required_wolves or len(required_wolves) != 2:
            raise ValueError("a wolf observer must know the exact two-player wolf team")
    else:
        if observer_id in required_wolves:
            raise ValueError("a non-wolf observer cannot be a required wolf")
        forbidden_wolves.add(observer_id)

    return {
        "required_wolves": tuple(sorted(required_wolves)),
        "forbidden_wolves": tuple(sorted(forbidden_wolves)),
    }


def first_order_knowledge_mask(
    *,
    observer_id: int,
    observer_role: str,
    known_wolves=(),
    known_good=(),
) -> np.ndarray:
    """Return classes consistent with the observer's actual private knowledge.

    A non-wolf always knows that they are good. A wolf knows the exact two-player
    wolf team, which intentionally creates a one-class mask used only by tests and
    diagnostics; wolves are excluded from the main first-order collection path.
    """

    constraints = first_order_constraints(
        observer_id=observer_id,
        observer_role=observer_role,
        known_wolves=known_wolves,
        known_good=known_good,
    )
    required_wolves = set(constraints["required_wolves"])
    forbidden_wolves = set(constraints["forbidden_wolves"])

    mask = np.asarray(
        [
            required_wolves.issubset(pair) and forbidden_wolves.isdisjoint(pair)
            for pair in WOLF_PAIRS
        ],
        dtype=bool,
    )
    if not mask.any():
        raise ValueError("first-order knowledge is inconsistent with every wolf pair")
    return mask


def second_order_output_mask(*, mode: str, target_id: int) -> np.ndarray:
    """Return the output mask implied by a second-order input condition.

    Public-only prediction has no private identity constraint. A wolf-conditioned
    modeler knows that the selected target is a non-wolf, so pairs containing that
    target are excluded. No target role, target private fact, or true-wolf mask is
    accepted by this API.
    """

    target_id = _player_id(target_id, "target_id")
    if mode not in SECOND_ORDER_MODES:
        raise ValueError(f"unsupported second-order mode: {mode!r}")
    if mode == "public_only":
        return np.ones(len(WOLF_PAIRS), dtype=bool)
    return np.asarray([target_id not in pair for pair in WOLF_PAIRS], dtype=bool)
