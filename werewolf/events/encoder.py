"""Encode canonical structured events into model-native integer tokens."""

from werewolf.events.schema import CONTENT_VALUES_BY_KIND, event_sort_key


ENCODER_SCHEMA_VERSION = "structured_event.v1_1"


EVENT_TOKEN_FIELDS = (
    "family_id",
    "visibility_id",
    "speaker_id",
    "target_id",
    "kind_id",
    "value_id",
    "polarity_id",
    "certainty_id",
    "stance_id",
    "strength_id",
    "commitment_id",
    "relation_id",
    "phase_id",
    "day_id",
)

FAMILY2ID = {
    "PAD": 0,
    "BELIEF_ASSERTION": 1,
    "SOCIAL_STANCE": 2,
    "ACTION_POSITION": 3,
    "CLAIM_RESPONSE": 4,
    "GAME_EVENT": 5,
    "PRIVATE_FACT": 6,
}
VISIBILITY2ID = {"pad": 0, "public": 1, "private": 2}
KIND2ID = {
    "PAD": 0,
    "UNKNOWN": 1,
    "ROLE": 2,
    "CAMP": 3,
    "FACT": 4,
    "STANCE": 5,
    "ACTION": 6,
    "RELATION": 7,
    "SETTING": 8,
    "SPEECH": 9,
    "VOTE_CAST": 10,
    "VOTE_RESULT": 11,
    "EXILE": 12,
    "DEATH": 13,
    "ROLE_REVEAL": 14,
    "OUTCOME": 15,
    "SELF_ROLE": 16,
    "WOLF_TEAM": 17,
    "CHECK_RESULT": 18,
    "WITCH_STATE": 19,
    "GUARD_RESULT": 20,
    "PRIVATE_ACTION_RESULT": 21,
}
VALUE2ID = {
    "PAD": 0,
    "UNKNOWN": 1,
    "NONE": 2,
    "Werewolf": 3,
    "Seer": 4,
    "Witch": 5,
    "Guard": 6,
    "Villager": 7,
    "Village": 8,
    "VOTE": 9,
    "PASS": 10,
    "KILL": 11,
    "WITCH_HEAL": 12,
    "WITCH_POISON": 13,
    "WITCH_PASS": 14,
    "HEAL_AND_POISON_AVAILABLE": 15,
    "HEAL_AVAILABLE": 16,
    "POISON_AVAILABLE": 17,
    "NO_POTIONS_AVAILABLE": 18,
}
POLARITY2ID = {None: 0, "positive": 1, "negative": 2, "neutral": 3}
CERTAINTY2ID = {None: 0, "weak": 1, "normal": 2, "strong": 3}
STANCE2ID = {None: 0, "negative": 1, "neutral": 2, "positive": 3}
STRENGTH2ID = {None: 0, "weak": 1, "normal": 2, "strong": 3}
COMMITMENT2ID = {None: 0, "consider": 1, "intend": 2, "commit": 3}
RELATION2ID = {
    None: 0,
    "support": 1,
    "challenge": 2,
    "question": 3,
    "retract": 4,
}
PHASE2ID = {
    "pad": 0,
    "init": 1,
    "night": 2,
    "speech": 3,
    "speech_pk": 4,
    "vote": 5,
    "vote_pk": 6,
    "result": 7,
    "end": 8,
}

VOCABULARIES = (
    FAMILY2ID,
    VISIBILITY2ID,
    None,
    None,
    KIND2ID,
    VALUE2ID,
    POLARITY2ID,
    CERTAINTY2ID,
    STANCE2ID,
    STRENGTH2ID,
    COMMITMENT2ID,
    RELATION2ID,
    PHASE2ID,
    None,
)


def _phase_name(phase: str) -> str:
    for name in ("speech_pk", "vote_pk", "speech", "vote"):
        if name in phase:
            return name
    if "night" in phase or "skill" in phase:
        return "night"
    if "result" in phase or "exile" in phase:
        return "result"
    if "end" in phase:
        return "end"
    return "init"


def _content_value(kind, value):
    """Return a canonical scalar, NONE, or the final UNKNOWN fallback."""

    allowed = CONTENT_VALUES_BY_KIND.get(kind)
    if allowed is None:
        return "UNKNOWN"
    if value is None:
        return "NONE" if None in allowed else "UNKNOWN"
    if value in allowed and value in VALUE2ID:
        return value
    return "UNKNOWN"


def encode_event(event: dict) -> list[list[int]]:
    qualifier = event["qualifier"]
    targets = event["target"] or [0]
    base = {
        "family_id": FAMILY2ID[event["event_family"]],
        "visibility_id": VISIBILITY2ID[event["visibility"]],
        "speaker_id": event["speaker"],
        "kind_id": KIND2ID.get(event["content"]["kind"], KIND2ID["UNKNOWN"]),
        "value_id": VALUE2ID.get(
            _content_value(event["content"]["kind"], event["content"]["value"]),
            VALUE2ID["UNKNOWN"],
        ),
        "polarity_id": POLARITY2ID[qualifier["polarity"]],
        "certainty_id": CERTAINTY2ID[qualifier["certainty"]],
        "stance_id": STANCE2ID[qualifier["stance"]],
        "strength_id": STRENGTH2ID[qualifier["strength"]],
        "commitment_id": COMMITMENT2ID[qualifier["commitment"]],
        "relation_id": RELATION2ID[qualifier["relation"]],
        "phase_id": PHASE2ID[_phase_name(event["phase"])],
        "day_id": event["day"],
    }
    tokens = []
    for target_id in targets:
        values = dict(base, target_id=target_id)
        tokens.append([values[field] for field in EVENT_TOKEN_FIELDS])
    return tokens


def encode_events(events) -> list[list[int]]:
    tokens = []
    for event in sorted(events, key=event_sort_key):
        tokens.extend(encode_event(event))
    return tokens
