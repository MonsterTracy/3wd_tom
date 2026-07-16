import json
from copy import deepcopy
from pathlib import Path

import pytest

from werewolf.events.environment_events import (
    check_result_event,
    death_event,
    self_role_event,
    wolf_team_event,
)
from werewolf.tom.dataset import ToMDataset
from werewolf.tom.guess_provider import BeliefGuessProvider, SYSTEM_PROMPT
from werewolf.tom.masks import second_order_output_mask
from werewolf.tom.pair_space import pair_index
from werewolf.tom.schemas import validate_sample, validate_sample_collection


FIXTURE = Path(__file__).parents[1] / "fixtures" / "tom_v1.jsonl"


class SequenceBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.messages = []

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.messages.append(deepcopy(messages))
        return next(self.responses)


def test_guess_provider_repairs_once_without_defaulting():
    backend = SequenceBackend(["bad", '{"wolf_pair":[2,1]}'])
    result = BeliefGuessProvider(backend, "agent-model").elicit(
        player_view="view", output_mask=[True] * 21
    )
    assert result.status == "ok"
    assert result.pair == (1, 2)
    assert result.attempts == 2
    assert len(result.raw_text) == 2
    assert backend.calls == 2
    assert backend.messages[0] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "view"},
    ]
    assert backend.messages[1] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "view"},
        {"role": "assistant", "content": "bad"},
        {
            "role": "user",
            "content": (
                "Your response was invalid. Return only the required JSON "
                "object with a valid pair."
            ),
        },
    ]


def test_guess_provider_preserves_failure_and_does_not_autocorrect():
    mask = [True] + [False] * 20
    backend = SequenceBackend(['{"wolf_pair":[6,7]}', '{"wolf_pair":[6,7]}'])
    result = BeliefGuessProvider(backend, "agent-model").elicit(
        player_view="view", output_mask=mask
    )
    assert result.status == "failed"
    assert result.pair is None
    assert "knowledge mask" in result.error
    assert len(result.raw_text) == 2
    assert result.raw_text == ('{"wolf_pair":[6,7]}', '{"wolf_pair":[6,7]}')
    assert result.attempts == 2
    assert backend.calls == 2
    assert mask[pair_index((1, 2))]


def test_guess_provider_does_not_fabricate_assistant_after_backend_error():
    class ErrorThenSuccessBackend(SequenceBackend):
        def chat(self, messages, **kwargs):
            self.calls += 1
            self.messages.append(deepcopy(messages))
            if self.calls == 1:
                raise RuntimeError("temporary backend error")
            return '{"wolf_pair":[1,2]}'

    backend = ErrorThenSuccessBackend([])
    result = BeliefGuessProvider(backend, "agent-model").elicit(
        player_view="legal player view", output_mask=[True] * 21
    )
    assert result.status == "ok"
    assert result.attempts == 2
    assert backend.calls == 2
    assert backend.messages[0] == backend.messages[1]
    assert backend.messages[1] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "legal player view"},
    ]


def test_guess_provider_has_no_gameplay_side_effects():
    observation = {"player_id": 3, "phase": "day_speech", "view": "legal view"}
    events = [{"event_id": "e1", "content": {"kind": "SPEECH"}}]
    agent_state = {"memory": ["prior"], "actions": 4}
    output_mask = [True] * 21
    before = deepcopy((observation, events, agent_state, output_mask))

    backend = SequenceBackend(['{"wolf_pair":[1,2]}'])
    result = BeliefGuessProvider(backend, "agent-model").elicit(
        player_view=observation["view"], output_mask=output_mask
    )

    assert result.status == "ok"
    assert (observation, events, agent_state, output_mask) == before
    assert backend.calls == 1


def test_dataset_accepts_only_successful_tom_v1(tmp_path):
    dataset = ToMDataset(FIXTURE)
    assert len(dataset) == 2
    assert dataset[0]["output_mask"].shape == (21,)
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps({"schema_version": "legacy.v0"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy samples are rejected|fields"):
        ToMDataset(legacy)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        ToMDataset([FIXTURE, FIXTURE])


def test_schema_rejects_private_leakage_in_public_second_order():
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    leaked = deepcopy(records[1])
    leaked["events"].append(deepcopy(records[0]["events"][1]))
    with pytest.raises(ValueError, match="public-only"):
        validate_sample(leaked)


def test_public_second_order_rejects_any_mask_smaller_than_21():
    public = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[1])
    public["output_mask"][0] = False
    with pytest.raises(ValueError, match="all 21"):
        validate_sample(public)


def _private_event(builder, event_id, *, visible_to, target, value, speaker=0):
    return builder(
        event_id=event_id,
        day=1,
        phase="1_day_speech",
        turn=3,
        visible_to=visible_to,
        target=target,
        value=value,
        speaker=speaker,
    )


def _wolf_conditioned_sample():
    sample = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[1])
    sample["sample_id"] = "fixture:second:wolf:1"
    sample["mode"] = "wolf_conditioned"
    sample["modeler_id"] = 1
    sample["output_mask"] = second_order_output_mask(
        mode="wolf_conditioned", target_id=3
    ).tolist()
    sample["events"].extend(
        [
            _private_event(
                self_role_event,
                "fixture.wolf.self",
                visible_to=[1],
                target=1,
                value="Werewolf",
            ),
            _private_event(
                wolf_team_event,
                "fixture.wolf.team",
                visible_to=[1, 2],
                target=[1, 2],
                value=None,
            ),
        ]
    )
    return sample


@pytest.mark.parametrize(
    ("visible_to", "target"),
    [([3], 3), ([2], 2)],
    ids=["target-only-private", "modeler-unseen-private"],
)
def test_wolf_conditioned_rejects_private_facts_the_modeler_cannot_see(
    visible_to, target
):
    sample = _wolf_conditioned_sample()
    sample["events"].append(
        _private_event(
            self_role_event,
            f"fixture.unseen.{target}",
            visible_to=visible_to,
            target=target,
            value="Villager",
        )
    )
    with pytest.raises(ValueError, match="private target information"):
        validate_sample(sample)


def test_target_private_check_is_never_a_wolf_conditioned_input():
    sample = _wolf_conditioned_sample()
    sample["events"].append(
        _private_event(
            check_result_event,
            "fixture.leaked.check",
            visible_to=[1],
            target=3,
            value="Village",
            speaker=1,
        )
    )
    with pytest.raises(ValueError, match="target private information"):
        validate_sample(sample)


def test_god_view_private_role_fact_is_rejected():
    sample = _wolf_conditioned_sample()
    sample["events"].append(
        _private_event(
            self_role_event,
            "fixture.god-view",
            visible_to=list(range(1, 8)),
            target=3,
            value="Seer",
        )
    )
    with pytest.raises(ValueError, match="god-view"):
        validate_sample(sample)


def test_future_event_after_checkpoint_is_rejected():
    sample = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[1])
    sample["events"][0]["turn"] = sample["turn"] + 1
    with pytest.raises(ValueError, match="future event"):
        validate_sample(sample)


def test_dead_players_remain_valid_identity_pair_classes():
    sample = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[1])
    sample["events"].append(
        death_event(
            event_id="fixture.death",
            day=1,
            phase="1_day_result",
            turn=3,
            target=[1, 2],
            value=None,
        )
    )
    assert validate_sample(sample)
    assert sample["output_mask"][pair_index((1, 2))]


def test_second_order_source_alignment_is_mandatory():
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    misaligned = deepcopy(records)
    misaligned[1]["target_id"] = 4
    with pytest.raises(ValueError, match="target_id does not match"):
        validate_sample_collection(misaligned)

    missing_source = deepcopy(records)
    missing_source[1]["source_first_order_sample_id"] = "missing:first"
    with pytest.raises(ValueError, match="missing source"):
        validate_sample_collection(missing_source)
