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


def partition_events(events, player_id: int | None = None) -> dict[str, list[dict]]:
    """Partition one visibility-safe stream without duplicating any event."""

    partitions = {
        "private_facts": [],
        "public_game_events": [],
        "public_player_claims": [],
    }
    seen = set()
    for event in sorted(events, key=event_sort_key):
        event_id = event["event_id"]
        if event_id in seen:
            raise ValueError(f"event_id values must be unique: {event_id}")
        seen.add(event_id)
        family = event["event_family"]
        if event["visibility"] == "private":
            if player_id is not None and player_id in event["visible_to"]:
                partitions["private_facts"].append(deepcopy(event))
            continue
        if family == "GAME_EVENT" and event["content"]["kind"] != "SPEECH":
            partitions["public_game_events"].append(deepcopy(event))
        else:
            partitions["public_player_claims"].append(deepcopy(event))
    return partitions


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
        f"event_id={event['event_id']} day={event['day']} "
        f"phase={event['phase']} speaker={event['speaker']} "
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


def render_information_partitions(
    events, player_id: int | None = None,
) -> dict[str, str]:
    partitions = partition_events(events, player_id)
    return {
        name: render_stream(selected) if selected else "（无）"
        for name, selected in partitions.items()
    }


def alive_players_from_events(events) -> list[int]:
    alive = set(range(1, 8))
    for event in sorted(events, key=event_sort_key):
        if event["event_family"] != "GAME_EVENT":
            continue
        if event["content"]["kind"] in {"DEATH", "EXILE"}:
            alive.difference_update(event["target"])
    return sorted(alive)


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
