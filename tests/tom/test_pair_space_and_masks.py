import numpy as np
import pytest

from werewolf.events.environment_events import role_reveal_event, self_role_event
from werewolf.events.streams import knowledge_for_player
from werewolf.tom.masks import first_order_knowledge_mask, second_order_output_mask
from werewolf.tom.pair_space import (
    WOLF_PAIRS, masked_softmax, pair_index, pair_probabilities_to_player_marginals
)


def test_global_pair_space_is_stable_and_complete():
    assert len(WOLF_PAIRS) == 21
    assert WOLF_PAIRS[0] == (1, 2)
    assert WOLF_PAIRS[-1] == (6, 7)
    assert pair_index((7, 6)) == 20


def test_first_order_mask_uses_private_knowledge_not_alive_state():
    base = first_order_knowledge_mask(observer_id=3, observer_role="Seer")
    informed = first_order_knowledge_mask(
        observer_id=3, observer_role="Seer", known_wolves=[1], known_good=[4]
    )
    assert base.sum() == 15
    assert informed.sum() == 4
    assert all(3 not in pair for pair, keep in zip(WOLF_PAIRS, base) if keep)
    assert all(1 in pair and 4 not in pair for pair, keep in zip(WOLF_PAIRS, informed) if keep)


def test_wolf_first_order_mask_is_degenerate_but_explicit():
    mask = first_order_knowledge_mask(
        observer_id=2, observer_role="Werewolf", known_wolves=[2, 6]
    )
    assert mask.sum() == 1
    assert mask[pair_index((2, 6))]
    with pytest.raises(ValueError, match="exact"):
        first_order_knowledge_mask(observer_id=2, observer_role="Werewolf", known_wolves=[2])


def test_second_order_masks_have_separate_semantics():
    assert second_order_output_mask(mode="public_only", target_id=5).sum() == 21
    conditioned = second_order_output_mask(mode="wolf_conditioned", target_id=5)
    assert conditioned.sum() == 15
    assert all(5 not in pair for pair, keep in zip(WOLF_PAIRS, conditioned) if keep)
    with pytest.raises(TypeError):
        second_order_output_mask(mode="public_only", target_id=5, target_role="Seer")


def test_pair_probabilities_produce_two_expected_wolves():
    probabilities = masked_softmax(np.zeros(21), np.ones(21, dtype=bool))
    marginals = pair_probabilities_to_player_marginals(probabilities)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.isclose(marginals.sum(), 2.0)


def test_public_role_reveal_becomes_hard_first_order_knowledge():
    events = [
        self_role_event(
            event_id="e1", day=0, phase="init", turn=1,
            visible_to=[3], target=3, value="Seer"
        ),
        role_reveal_event(
            event_id="e2", day=1, phase="result", turn=2,
            target=1, value={"role": "Werewolf"}
        ),
    ]
    knowledge = knowledge_for_player(events, 3)
    assert knowledge["known_wolves"] == [1]
    mask = first_order_knowledge_mask(
        observer_id=3, observer_role=knowledge["role"],
        known_wolves=knowledge["known_wolves"], known_good=knowledge["known_good"]
    )
    assert all(1 in pair for pair, keep in zip(WOLF_PAIRS, mask) if keep)
