import pytest

from werewolf.events.encoder import EVENT_TOKEN_FIELDS, encode_events
from werewolf.events.environment_events import self_role_event, setting_event
from werewolf.events.schema import make_event, validate_event
from werewolf.events.streams import public_events, visible_events


def test_environment_events_have_fixed_schema_and_visibility():
    public = setting_event(
        event_id="e1", day=0, phase="init", turn=1, value={"players": 7}
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
    event = setting_event(event_id="e1", day=0, phase="init", turn=1, value={})
    event["unexpected"] = "old field"
    with pytest.raises(ValueError, match="unknown"):
        validate_event(event)
    with pytest.raises(ValueError, match="environment"):
        make_event(
            event_id="e2", day=0, phase="speech", turn=2,
            source_type="environment", visibility="public", visible_to=range(1, 8),
            speaker=1, event_family="BELIEF_ASSERTION", target=2,
            content={"kind": "CAMP", "value": "Werewolf"}
        )


def test_encoder_uses_only_current_event_fields():
    event = setting_event(event_id="e1", day=0, phase="init", turn=1, value={})
    tokens = encode_events([event])
    assert len(tokens) == 1
    assert len(tokens[0]) == len(EVENT_TOKEN_FIELDS) == 14
