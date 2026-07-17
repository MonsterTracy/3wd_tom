from copy import deepcopy

import pytest

from werewolf.events.encoder import (
    EVENT_TOKEN_FIELDS,
    KIND2ID,
    VALUE2ID,
    _content_value,
    encode_event,
    encode_events,
)
from werewolf.events.environment_events import (
    check_result_event,
    death_event,
    exile_event,
    guard_result_event,
    outcome_event,
    private_action_event,
    role_reveal_event,
    self_role_event,
    setting_event,
    speech_event,
    vote_event,
    vote_result_event,
    witch_state_event,
    wolf_team_event,
)
from werewolf.events.schema import make_event, validate_event
from werewolf.events.streams import public_events, visible_events
from werewolf.prompt_protocol import PARSER_PROMPT_SPEC


def test_environment_events_have_fixed_schema_and_visibility():
    public = setting_event(
        event_id="e1", day=0, phase="init", turn=1, value=None
    )
    private = self_role_event(
        event_id="e2", day=0, phase="init", turn=2,
        visible_to=[3], target=3, value="Seer"
    )
    assert validate_event(public)
    assert validate_event(private)
    assert public_events([public, private]) == [public]
    assert visible_events([public, private], 3) == [public, private]
    assert visible_events([public, private], 4) == [public]


def test_event_schema_rejects_extra_fields_and_source_family_mismatch():
    event = setting_event(event_id="e1", day=0, phase="init", turn=1, value=None)
    event["unexpected"] = "old field"
    with pytest.raises(ValueError, match="unknown"):
        validate_event(event)
    bad_metadata = setting_event(
        event_id="bad-metadata", day=0, phase="init", turn=1, value=None
    )
    bad_metadata["metadata"] = []
    with pytest.raises(ValueError, match="metadata"):
        validate_event(bad_metadata)
    with pytest.raises(ValueError, match="environment"):
        make_event(
            event_id="e2", day=0, phase="speech", turn=2,
            source_type="environment", visibility="public", visible_to=range(1, 8),
            speaker=1, event_family="BELIEF_ASSERTION", target=2,
            content={"kind": "CAMP", "value": "Werewolf"}
        )


def test_encoder_uses_only_current_event_fields():
    event = setting_event(event_id="e1", day=0, phase="init", turn=1, value=None)
    tokens = encode_events([event])
    assert len(tokens) == 1
    assert len(tokens[0]) == len(EVENT_TOKEN_FIELDS) == 14


def test_none_unknown_and_padding_have_distinct_encoder_semantics():
    assert VALUE2ID["PAD"] == 0
    assert VALUE2ID["NONE"] != VALUE2ID["PAD"]
    assert VALUE2ID["NONE"] != VALUE2ID["UNKNOWN"]
    event = setting_event(
        event_id="e1", day=0, phase="init", turn=1, value=None
    )
    token = encode_event(event)[0]
    assert token[EVENT_TOKEN_FIELDS.index("value_id")] == VALUE2ID["NONE"]
    assert _content_value("ROLE", "Wizard") == "UNKNOWN"
    assert _content_value("ROLE", None) == "UNKNOWN"
    assert _content_value("SPEECH", None) == "NONE"
    invalid = self_role_event(
        event_id="e2", day=0, phase="init", turn=2,
        visible_to=[3], target=3, value="Seer"
    )
    invalid = deepcopy(invalid)
    invalid["content"]["value"] = "Wizard"
    value_index = EVENT_TOKEN_FIELDS.index("value_id")
    assert encode_event(invalid)[0][value_index] == VALUE2ID["UNKNOWN"]


def _parsed_event(*, event_family, kind, value, qualifier=None):
    return make_event(
        event_id=f"parsed.{kind}",
        utterance_id="u1",
        day=1,
        phase="1_day_speech",
        turn=3,
        source_type="speech_parser",
        visibility="public",
        visible_to=range(1, 8),
        speaker=1,
        event_family=event_family,
        target=2,
        content={"kind": kind, "value": value},
        metadata={
            "parser_protocol": {
                "version": PARSER_PROMPT_SPEC["version"],
                "sha256": PARSER_PROMPT_SPEC["sha256"],
                "model": "test-parser",
                "temperature": 0.0,
                "attempts": 1,
                "status": "ok",
            }
        },
        qualifier=qualifier,
        source_span="player 2",
        parser_confidence=1.0,
    )


def test_schema_normalizes_aliases_and_rejects_uncontrolled_values():
    camp = check_result_event(
        event_id="alias.camp", day=1, phase="night", turn=1,
        visible_to=[3], speaker=3, target=2, value="Village"
    )
    role = self_role_event(
        event_id="alias.role", day=0, phase="init", turn=2,
        visible_to=[3], target=3, value="seer"
    )
    assert camp["content"]["value"] == "Village"
    assert role["content"]["value"] == "Seer"
    value_index = EVENT_TOKEN_FIELDS.index("value_id")
    assert encode_event(camp)[0][value_index] == VALUE2ID["Village"]
    with pytest.raises(ValueError, match="Werewolf/Village"):
        check_result_event(
            event_id="alias.bad-camp", day=1, phase="night", turn=1,
            visible_to=[3], speaker=3, target=2, value="good"
        )
    with pytest.raises(ValueError, match="requires target"):
        check_result_event(
            event_id="alias.no-target", day=1, phase="night", turn=1,
            visible_to=[3], speaker=3, target=None, value="Village"
        )
    with pytest.raises(ValueError, match="only to its seer speaker"):
        check_result_event(
            event_id="alias.leak", day=1, phase="night", turn=1,
            visible_to=[3, 4], speaker=3, target=2, value="Village"
        )
    with pytest.raises(ValueError, match="ROLE|controlled"):
        _parsed_event(
            event_family="BELIEF_ASSERTION", kind="ROLE", value="Wizard"
        )
    with pytest.raises(ValueError, match="SPEECH|NONE"):
        speech_event(
            event_id="bad-speech",
            day=1,
            phase="speech",
            turn=4,
            speaker=1,
            target=1,
            value="free text must stay in source_span",
            source_span="free text must stay in source_span",
        )


def test_parser_families_do_not_duplicate_qualifier_semantics_in_value():
    stance = _parsed_event(
        event_family="SOCIAL_STANCE",
        kind="STANCE",
        value=None,
        qualifier={"polarity": "negative", "strength": "strong"},
    )
    relation = _parsed_event(
        event_family="CLAIM_RESPONSE",
        kind="RELATION",
        value=None,
        qualifier={"relation": "challenge"},
    )
    assert stance["content"]["value"] is None
    assert stance["qualifier"]["polarity"] == "negative"
    assert relation["content"]["value"] is None
    assert relation["qualifier"]["relation"] == "challenge"


def test_all_canonical_environment_event_values_encode_without_unknown():
    common = {"day": 1, "phase": "1_day_result"}
    events = [
        setting_event(event_id="e1", turn=1, value=None, **common),
        speech_event(
            event_id="e2", turn=2, speaker=3, target=3, value=None,
            source_span="raw speech", **common
        ),
        vote_event(event_id="e3", turn=3, speaker=3, target=2, value=None, **common),
        vote_result_event(event_id="e4", turn=4, target=2, value=None, **common),
        exile_event(event_id="e5", turn=5, target=2, value=None, **common),
        death_event(event_id="e6", turn=6, target=[2, 4], value=None, **common),
        role_reveal_event(event_id="e7", turn=7, target=2, value="Werewolf", **common),
        outcome_event(event_id="e8", turn=8, value="Village", **common),
        self_role_event(
            event_id="e9", turn=9, visible_to=[3], target=3, value="Seer", **common
        ),
        wolf_team_event(
            event_id="e10", turn=10, visible_to=[1, 2], target=[1, 2],
            value=None, **common
        ),
        check_result_event(
            event_id="e11", turn=11, visible_to=[3], speaker=3, target=1,
            value="Werewolf", **common
        ),
        witch_state_event(
            event_id="e12", turn=12, visible_to=[4], target=2,
            value="HEAL_AND_POISON_AVAILABLE", **common
        ),
        guard_result_event(
            event_id="e13", turn=13, visible_to=[4], speaker=4, target=2,
            value=None, **common
        ),
        private_action_event(
            event_id="e14", turn=14, visible_to=[1, 2], speaker=1, target=3,
            value="KILL", **common
        ),
    ]
    kind_index = EVENT_TOKEN_FIELDS.index("kind_id")
    value_index = EVENT_TOKEN_FIELDS.index("value_id")
    assert all(event["content"]["kind"] != "SPEECH" or event["source_span"] for event in events)
    assert events[9]["target"] == [1, 2] and events[9]["content"]["value"] is None
    assert events[2]["target"] == [2] and events[2]["content"]["value"] is None
    for event in events:
        for token in encode_event(event):
            assert token[kind_index] != KIND2ID["UNKNOWN"]
            assert token[value_index] != VALUE2ID["UNKNOWN"]
