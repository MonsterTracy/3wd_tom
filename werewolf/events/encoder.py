"""Encode unified structured events into model-native integer tokens."""

from werewolf.events.schema import event_sort_key, validate_event


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
    "Werewolf": 2,
    "Seer": 3,
    "Witch": 4,
    "Guard": 5,
    "Villager": 6,
    "Village": 7,
    "good": 8,
    "bad": 9,
    "VOTE": 10,
    "PASS": 11,
    "support": 12,
    "challenge": 13,
    "question": 14,
    "retract": 15,
    "win": 16,
    "lose": 17,
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


def _content_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("role", "camp", "result", "action", "status"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return "UNKNOWN"


def encode_event(event: dict) -> list[list[int]]:
    validate_event(event)
    qualifier = event["qualifier"]
    targets = event["target"] or [0]
    base = {
        "family_id": FAMILY2ID[event["event_family"]],
        "visibility_id": VISIBILITY2ID[event["visibility"]],
        "speaker_id": event["speaker"],
        "kind_id": KIND2ID.get(event["content"]["kind"], KIND2ID["UNKNOWN"]),
        "value_id": VALUE2ID.get(
            _content_value(event["content"]["value"]), VALUE2ID["UNKNOWN"]
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
