"""Deterministic environment event builders; no LLM parsing occurs here."""

from werewolf.events.schema import make_event
from werewolf.game_rules import PLAYER_IDS


ALL_PLAYERS = list(PLAYER_IDS)


def game_event(
    *, event_id, day, phase, turn, kind, value=None, target=None, speaker=0,
    utterance_id=None, source_span=None, qualifier=None, metadata=None,
):
    return make_event(
        event_id=event_id,
        utterance_id=utterance_id,
        day=day,
        phase=phase,
        turn=turn,
        source_type="environment",
        visibility="public",
        visible_to=ALL_PLAYERS,
        speaker=speaker,
        event_family="GAME_EVENT",
        target=target,
        content={"kind": kind, "value": value},
        metadata=metadata,
        qualifier=qualifier,
        source_span=source_span,
    )


def private_fact(
    *, event_id, day, phase, turn, visible_to, kind, value=None, target=None,
    speaker=0, metadata=None,
):
    return make_event(
        event_id=event_id,
        day=day,
        phase=phase,
        turn=turn,
        source_type="environment",
        visibility="private",
        visible_to=visible_to,
        speaker=speaker,
        event_family="PRIVATE_FACT",
        target=target,
        content={"kind": kind, "value": value},
        metadata=metadata,
        qualifier={"certainty": "strong", "evidence_source": "private_fact"},
    )


def setting_event(**kwargs):
    return game_event(kind="SETTING", **kwargs)


def speech_event(**kwargs):
    return game_event(kind="SPEECH", **kwargs)


def vote_event(**kwargs):
    return game_event(kind="VOTE_CAST", **kwargs)


def vote_result_event(**kwargs):
    return game_event(kind="VOTE_RESULT", **kwargs)


def exile_event(**kwargs):
    return game_event(kind="EXILE", **kwargs)


def death_event(**kwargs):
    return game_event(kind="DEATH", **kwargs)


def role_reveal_event(**kwargs):
    return game_event(kind="ROLE_REVEAL", **kwargs)


def outcome_event(**kwargs):
    return game_event(kind="OUTCOME", **kwargs)


def self_role_event(**kwargs):
    return private_fact(kind="SELF_ROLE", **kwargs)


def wolf_team_event(**kwargs):
    return private_fact(kind="WOLF_TEAM", **kwargs)


def check_result_event(**kwargs):
    if kwargs.get("target") is None or kwargs.get("value") not in {
        "Werewolf", "Village"
    }:
        raise ValueError("CHECK_RESULT requires target and Werewolf/Village value")
    if kwargs.get("visible_to") != [kwargs.get("speaker")]:
        raise ValueError("CHECK_RESULT must be visible only to its seer speaker")
    return private_fact(kind="CHECK_RESULT", **kwargs)


def witch_state_event(**kwargs):
    return private_fact(kind="WITCH_STATE", **kwargs)


def guard_result_event(**kwargs):
    return private_fact(kind="GUARD_RESULT", **kwargs)


def private_action_event(**kwargs):
    return private_fact(kind="PRIVATE_ACTION_RESULT", **kwargs)
