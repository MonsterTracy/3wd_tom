"""Versioned schema for all public and private game events."""

from copy import deepcopy

from werewolf.prompt_protocol import PARSER_PROMPT_SPEC


EVENT_SCHEMA_VERSION = "events.v1"

EVENT_FAMILIES = (
    "BELIEF_ASSERTION",
    "SOCIAL_STANCE",
    "ACTION_POSITION",
    "CLAIM_RESPONSE",
    "GAME_EVENT",
    "PRIVATE_FACT",
)

SOURCE_TYPES = ("environment", "speech_parser")
VISIBILITIES = ("public", "private")

QUALIFIER_ENUMS = {
    "polarity": (None, "positive", "negative", "neutral"),
    "certainty": (None, "weak", "normal", "strong"),
    "stance": (None, "negative", "neutral", "positive"),
    "strength": (None, "weak", "normal", "strong"),
    "commitment": (None, "consider", "intend", "commit"),
    "evidence_source": (
        None,
        "public_history",
        "claimed_private_info",
        "private_fact",
        "unspecified",
    ),
    "relation": (None, "support", "challenge", "question", "retract"),
}

DEFAULT_QUALIFIER = {name: None for name in QUALIFIER_ENUMS}

ROLE_VALUES = ("Werewolf", "Seer", "Witch", "Guard", "Villager")
CAMP_VALUES = ("Werewolf", "Village")
ACTION_VALUES = ("VOTE", "PASS")
PRIVATE_ACTION_VALUES = ("KILL", "WITCH_HEAL", "WITCH_POISON", "WITCH_PASS")
WITCH_STATE_VALUES = (
    "HEAL_AND_POISON_AVAILABLE",
    "HEAL_AVAILABLE",
    "POISON_AVAILABLE",
    "NO_POTIONS_AVAILABLE",
)

CONTENT_VALUES_BY_KIND = {
    "ROLE": ROLE_VALUES,
    "CAMP": CAMP_VALUES,
    "FACT": (None,),
    "STANCE": (None,),
    "ACTION": ACTION_VALUES,
    "RELATION": (None,),
    "SETTING": (None,),
    "SPEECH": (None,),
    "VOTE_CAST": (None,),
    "VOTE_RESULT": (None,),
    "EXILE": (None,),
    "DEATH": (None,),
    "ROLE_REVEAL": ROLE_VALUES,
    "OUTCOME": CAMP_VALUES,
    "SELF_ROLE": ROLE_VALUES,
    "WOLF_TEAM": (None,),
    "CHECK_RESULT": CAMP_VALUES,
    "WITCH_STATE": WITCH_STATE_VALUES,
    "GUARD_RESULT": (None,),
    "PRIVATE_ACTION_RESULT": PRIVATE_ACTION_VALUES,
}

CONTENT_KINDS_BY_FAMILY = {
    "BELIEF_ASSERTION": {"ROLE", "CAMP", "FACT"},
    "SOCIAL_STANCE": {"STANCE"},
    "ACTION_POSITION": {"ACTION"},
    "CLAIM_RESPONSE": {"RELATION"},
    "GAME_EVENT": {
        "SETTING",
        "SPEECH",
        "VOTE_CAST",
        "VOTE_RESULT",
        "EXILE",
        "DEATH",
        "ROLE_REVEAL",
        "OUTCOME",
    },
    "PRIVATE_FACT": {
        "SELF_ROLE",
        "WOLF_TEAM",
        "CHECK_RESULT",
        "WITCH_STATE",
        "GUARD_RESULT",
        "PRIVATE_ACTION_RESULT",
    },
}

_ROLE_ALIASES = {
    "werewolf": "Werewolf",
    "wolf": "Werewolf",
    "seer": "Seer",
    "witch": "Witch",
    "guard": "Guard",
    "villager": "Villager",
}
_CAMP_ALIASES = {
    "werewolf": "Werewolf",
    "wolf": "Werewolf",
    "bad": "Werewolf",
    "wolf_side": "Werewolf",
    "village": "Village",
    "good": "Village",
    "villager_side": "Village",
    "village_side": "Village",
}

REQUIRED_EVENT_FIELDS = (
    "schema_version",
    "event_id",
    "utterance_id",
    "day",
    "phase",
    "turn",
    "source_type",
    "visibility",
    "visible_to",
    "speaker",
    "event_family",
    "target",
    "content",
    "metadata",
    "qualifier",
    "ref_event_id",
    "source_span",
    "parser_confidence",
)
PARSER_PROTOCOL_FIELDS = {
    "version",
    "sha256",
    "model",
    "temperature",
    "attempts",
    "status",
}


def _player_id(value, *, allow_zero=False):
    lower = 0 if allow_zero else 1
    if type(value) is not int or not lower <= value <= 7:
        raise ValueError(f"player id must be between {lower} and 7")
    return value


def normalize_targets(target) -> list[int]:
    if target is None:
        return []
    values = target if isinstance(target, (list, tuple, set)) else [target]
    normalized = []
    for value in values:
        player_id = _player_id(value)
        if player_id not in normalized:
            normalized.append(player_id)
    return normalized


def normalize_qualifier(qualifier=None) -> dict:
    qualifier = {} if qualifier is None else dict(qualifier)
    unknown = set(qualifier) - set(QUALIFIER_ENUMS)
    if unknown:
        raise ValueError(f"unknown qualifier fields: {sorted(unknown)}")
    normalized = dict(DEFAULT_QUALIFIER)
    normalized.update(qualifier)
    for name, allowed in QUALIFIER_ENUMS.items():
        if normalized[name] not in allowed:
            raise ValueError(f"invalid qualifier {name}: {normalized[name]!r}")
    return normalized


def normalize_content(content) -> dict:
    """Normalize one canonical kind/value pair before event construction."""

    if not isinstance(content, dict) or set(content) != {"kind", "value"}:
        raise ValueError("content must contain only kind and value")
    kind = content["kind"]
    if not isinstance(kind, str) or kind not in CONTENT_VALUES_BY_KIND:
        raise ValueError(f"unsupported content.kind: {kind!r}")
    value = content["value"]
    if kind in {"ROLE", "ROLE_REVEAL", "SELF_ROLE"}:
        if not isinstance(value, str):
            raise ValueError(f"{kind} requires a controlled role value")
        value = _ROLE_ALIASES.get(value.strip().lower())
    elif kind in {"CAMP", "OUTCOME", "CHECK_RESULT"}:
        if not isinstance(value, str):
            raise ValueError(f"{kind} requires a controlled camp value")
        value = _CAMP_ALIASES.get(value.strip().lower())
    elif kind in {"ACTION", "PRIVATE_ACTION_RESULT", "WITCH_STATE"}:
        if not isinstance(value, str):
            raise ValueError(f"{kind} requires a controlled scalar value")
        value = value.strip().upper()
    allowed = CONTENT_VALUES_BY_KIND[kind]
    if value not in allowed:
        expected = "NONE" if allowed == (None,) else repr(allowed)
        raise ValueError(f"{kind} value must be controlled: {expected}")
    return {"kind": kind, "value": value}


def make_event(
    *,
    event_id: str,
    day: int,
    phase: str,
    turn: int,
    source_type: str,
    visibility: str,
    visible_to,
    speaker: int,
    event_family: str,
    target=None,
    content: dict,
    metadata=None,
    qualifier=None,
    utterance_id: str | None = None,
    ref_event_id: str | None = None,
    source_span: str | None = None,
    parser_confidence: float | None = None,
) -> dict:
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "utterance_id": utterance_id,
        "day": day,
        "phase": phase,
        "turn": turn,
        "source_type": source_type,
        "visibility": visibility,
        "visible_to": list(visible_to),
        "speaker": speaker,
        "event_family": event_family,
        "target": normalize_targets(target),
        "content": normalize_content(deepcopy(content)),
        "metadata": {} if metadata is None else deepcopy(metadata),
        "qualifier": normalize_qualifier(qualifier),
        "ref_event_id": ref_event_id,
        "source_span": source_span,
        "parser_confidence": parser_confidence,
    }
    validate_event(event)
    return event


def validate_event(event: dict) -> bool:
    if not isinstance(event, dict):
        raise ValueError("event must be a mapping")
    missing = [field for field in REQUIRED_EVENT_FIELDS if field not in event]
    unknown = set(event) - set(REQUIRED_EVENT_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"event fields do not match {EVENT_SCHEMA_VERSION}; "
            f"missing={missing}, unknown={sorted(unknown)}"
        )
    if event["schema_version"] != EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported event schema_version")
    if not isinstance(event["event_id"], str) or not event["event_id"]:
        raise ValueError("event_id is required")
    if event["utterance_id"] is not None and not isinstance(
        event["utterance_id"], str
    ):
        raise ValueError("utterance_id must be text or null")
    if type(event["day"]) is not int or event["day"] < 0:
        raise ValueError("day must be a non-negative integer")
    if not isinstance(event["phase"], str) or not event["phase"]:
        raise ValueError("phase is required")
    if type(event["turn"]) is not int or event["turn"] < 0:
        raise ValueError("turn must be a non-negative integer")
    if event["source_type"] not in SOURCE_TYPES:
        raise ValueError("invalid source_type")
    if event["visibility"] not in VISIBILITIES:
        raise ValueError("invalid visibility")
    visible_to = event["visible_to"]
    if not isinstance(visible_to, list):
        raise ValueError("visible_to must be a list")
    for player_id in visible_to:
        _player_id(player_id)
    if event["visibility"] == "public" and sorted(set(visible_to)) != list(
        range(1, 8)
    ):
        raise ValueError("public events must be visible to all seven players")
    if len(visible_to) != len(set(visible_to)):
        raise ValueError("visible_to cannot contain duplicate players")
    if event["visibility"] == "private" and not visible_to:
        raise ValueError("private events require at least one viewer")
    _player_id(event["speaker"], allow_zero=True)
    if event["event_family"] not in EVENT_FAMILIES:
        raise ValueError("invalid event_family")
    if not isinstance(event["target"], list):
        raise ValueError("target must be a list")
    if normalize_targets(event["target"]) != event["target"]:
        raise ValueError("target must contain unique player ids in source order")
    content = event["content"]
    normalized_content = normalize_content(content)
    if normalized_content != content:
        raise ValueError("content kind/value must already be canonical")
    if content["kind"] not in CONTENT_KINDS_BY_FAMILY[event["event_family"]]:
        raise ValueError("content.kind is not valid for its event_family")
    if content["kind"] == "SPEECH" and (
        event["source_span"] is None
        or not isinstance(event["source_span"], str)
    ):
        raise ValueError("SPEECH keeps its raw text in source_span")
    if content["kind"] == "WOLF_TEAM" and len(event["target"]) != 2:
        raise ValueError("WOLF_TEAM requires the two wolf ids in target")
    if content["kind"] == "CHECK_RESULT":
        if len(event["target"]) != 1 or content["value"] not in CAMP_VALUES:
            raise ValueError("CHECK_RESULT requires one target and a canonical camp")
    if not isinstance(event["metadata"], dict):
        raise ValueError("metadata must be a mapping")
    if not isinstance(event["qualifier"], dict) or set(event["qualifier"]) != set(
        QUALIFIER_ENUMS
    ):
        raise ValueError("qualifier fields do not match the schema")
    normalize_qualifier(event["qualifier"])
    if content["kind"] == "STANCE" and not (
        event["qualifier"]["polarity"] or event["qualifier"]["stance"]
    ):
        raise ValueError("STANCE requires polarity or stance in qualifier")
    if content["kind"] == "RELATION" and event["qualifier"]["relation"] is None:
        raise ValueError("RELATION requires qualifier.relation")
    if event["ref_event_id"] is not None and not isinstance(
        event["ref_event_id"], str
    ):
        raise ValueError("ref_event_id must be text or null")
    if event["source_span"] is not None and not isinstance(event["source_span"], str):
        raise ValueError("source_span must be text or null")
    confidence = event["parser_confidence"]
    if confidence is not None and not (
        isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
    ):
        raise ValueError("parser_confidence must be between 0 and 1")
    if event["source_type"] == "speech_parser":
        if event["event_family"] not in EVENT_FAMILIES[:4]:
            raise ValueError("speech_parser may emit only speech event families")
        if event["visibility"] != "public":
            raise ValueError("speech-derived events must be public")
        if not event["utterance_id"] or not event["source_span"]:
            raise ValueError("speech-derived events require utterance_id and source_span")
        if confidence is None:
            raise ValueError("speech-derived events require parser_confidence")
        if set(event["metadata"]) != {"parser_protocol"}:
            raise ValueError("speech-derived event metadata requires parser_protocol")
        parser_protocol = event["metadata"]["parser_protocol"]
        if not isinstance(parser_protocol, dict) or set(parser_protocol) != PARSER_PROTOCOL_FIELDS:
            raise ValueError("parser_protocol fields do not match the schema")
        if parser_protocol["version"] != PARSER_PROMPT_SPEC["version"]:
            raise ValueError("parser_protocol version is incompatible")
        if parser_protocol["sha256"] != PARSER_PROMPT_SPEC["sha256"]:
            raise ValueError("parser_protocol sha256 is incompatible")
        if not isinstance(parser_protocol["model"], str) or not parser_protocol["model"]:
            raise ValueError("parser_protocol model is required")
        if (
            isinstance(parser_protocol["temperature"], bool)
            or not isinstance(parser_protocol["temperature"], (int, float))
            or parser_protocol["temperature"] != 0.0
        ):
            raise ValueError("parser_protocol temperature must be 0.0")
        if type(parser_protocol["attempts"]) is not int or not 1 <= parser_protocol["attempts"] <= 2:
            raise ValueError("parser_protocol attempts must be one or two")
        if parser_protocol["status"] != "ok":
            raise ValueError("parser_protocol status must be ok for emitted events")
    else:
        if event["event_family"] not in EVENT_FAMILIES[4:]:
            raise ValueError("environment may emit only GAME_EVENT or PRIVATE_FACT")
        expected_visibility = (
            "public" if event["event_family"] == "GAME_EVENT" else "private"
        )
        if event["visibility"] != expected_visibility:
            raise ValueError("environment event visibility does not match its family")
        if confidence is not None:
            raise ValueError("environment events cannot carry parser_confidence")
    return True


def event_sort_key(event: dict):
    return event["turn"], event["event_id"]
