"""Visibility-safe public and per-player private event streams."""

from copy import deepcopy
import json

from werewolf.events.schema import event_sort_key, validate_event


def public_events(events) -> list[dict]:
    return [deepcopy(event) for event in events if event["visibility"] == "public"]


def private_events(events, player_id: int) -> list[dict]:
    return [
        deepcopy(event)
        for event in events
        if event["visibility"] == "private" and player_id in event["visible_to"]
    ]


def visible_events(events, player_id: int) -> list[dict]:
    selected = [
        deepcopy(event)
        for event in events
        if event["visibility"] == "public" or player_id in event["visible_to"]
    ]
    return sorted(selected, key=event_sort_key)


def split_for_player(events, player_id: int) -> tuple[list[dict], list[dict]]:
    return public_events(events), private_events(events, player_id)


def validate_stream(events):
    for event in events:
        validate_event(event)
    event_ids = [event["event_id"] for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event_id values must be unique")
    return True


def render_event(event: dict) -> str:
    content = event["content"]
    targets = ",".join(str(value) for value in event["target"]) or "none"
    rendered = (
        f"day={event['day']} phase={event['phase']} speaker={event['speaker']} "
        f"family={event['event_family']} target={targets} "
        f"kind={content['kind']} value={content['value']}"
    )
    if event["source_span"] is not None:
        rendered += " source_span=" + json.dumps(
            event["source_span"], ensure_ascii=False
        )
    return rendered


def render_stream(events) -> str:
    return "\n".join(render_event(event) for event in sorted(events, key=event_sort_key))


def knowledge_for_player(events, player_id: int) -> dict:
    """Extract only hard identity facts visible to one player."""

    role = None
    known_wolves = set()
    known_good = set()
    for event in visible_events(events, player_id):
        kind = event["content"]["kind"]
        value = event["content"]["value"]
        if (
            event["event_family"] == "PRIVATE_FACT"
            and kind == "SELF_ROLE"
            and event["target"] == [player_id]
        ):
            role = value
        elif event["event_family"] == "PRIVATE_FACT" and kind == "WOLF_TEAM":
            known_wolves.update(event["target"])
        elif (
            event["event_family"] == "PRIVATE_FACT"
            and kind == "CHECK_RESULT"
            and event["target"]
        ):
            if value == "Werewolf":
                known_wolves.add(event["target"][0])
            elif value == "Village":
                known_good.add(event["target"][0])
        elif (
            event["event_family"] == "GAME_EVENT"
            and kind == "ROLE_REVEAL"
            and event["target"]
        ):
            if value == "Werewolf":
                known_wolves.add(event["target"][0])
            elif value in ("Seer", "Witch", "Guard", "Villager"):
                known_good.add(event["target"][0])
    return {
        "role": role,
        "known_wolves": sorted(known_wolves),
        "known_good": sorted(known_good),
    }
