"""Strict, versioned JSON schemas for ToM collection samples."""

from copy import deepcopy

from werewolf.events.schema import validate_event
from werewolf.events.streams import knowledge_for_player
from werewolf.tom.masks import (
    SECOND_ORDER_MODES,
    first_order_knowledge_mask,
    second_order_output_mask,
)
from werewolf.tom.pair_space import NUM_WOLF_PAIRS, normalize_pair, pair_index


TOM_SCHEMA_VERSION = "tom.v1"
TASKS = ("first_order", "second_order")
FIRST_ORDER_MODE = "private_conditioned"
GUESS_STATUSES = ("ok", "failed")

COMMON_FIELDS = {
    "schema_version",
    "sample_id",
    "game_id",
    "task",
    "mode",
    "checkpoint",
    "state_id",
    "day",
    "phase",
    "turn",
    "observer_id",
    "modeler_id",
    "target_id",
    "events",
    "output_mask",
    "label_pair",
    "label_index",
    "guess",
}
GUESS_FIELDS = {"status", "raw_text", "error", "attempts", "model"}


def _nullable_player(value, name):
    if value is not None and (type(value) is not int or not 1 <= value <= 7):
        raise ValueError(f"{name} must be null or a player id between 1 and 7")


def _validate_mask(mask):
    if not isinstance(mask, list) or len(mask) != NUM_WOLF_PAIRS:
        raise ValueError(f"output_mask must have shape [{NUM_WOLF_PAIRS}]")
    if any(type(value) is not bool for value in mask):
        raise ValueError("output_mask values must be booleans")
    if not any(mask):
        raise ValueError("output_mask must keep at least one class")


def validate_sample(sample: dict, *, require_success=True) -> bool:
    if not isinstance(sample, dict):
        raise ValueError("sample must be a mapping")
    unknown = set(sample) - COMMON_FIELDS
    missing = COMMON_FIELDS - set(sample)
    if unknown or missing:
        raise ValueError(
            f"sample fields do not match {TOM_SCHEMA_VERSION}; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if sample["schema_version"] != TOM_SCHEMA_VERSION:
        raise ValueError("unsupported ToM schema_version; legacy samples are rejected")
    for name in ("sample_id", "game_id", "checkpoint", "state_id", "phase"):
        if not isinstance(sample[name], str) or not sample[name]:
            raise ValueError(f"{name} is required")
    if sample["task"] not in TASKS:
        raise ValueError("invalid task")
    if type(sample["day"]) is not int or sample["day"] < 0:
        raise ValueError("day must be a non-negative integer")
    if type(sample["turn"]) is not int or sample["turn"] < 0:
        raise ValueError("turn must be a non-negative integer")
    for name in ("observer_id", "modeler_id", "target_id"):
        _nullable_player(sample[name], name)

    if sample["task"] == "first_order":
        if sample["mode"] != FIRST_ORDER_MODE:
            raise ValueError("first-order samples require private_conditioned mode")
        if sample["observer_id"] is None:
            raise ValueError("first-order samples require observer_id")
        if sample["modeler_id"] is not None or sample["target_id"] is not None:
            raise ValueError("first-order samples cannot set modeler_id or target_id")
    else:
        if sample["mode"] not in SECOND_ORDER_MODES:
            raise ValueError("invalid second-order mode")
        if sample["target_id"] is None:
            raise ValueError("second-order samples require target_id")
        if sample["observer_id"] is not None:
            raise ValueError("second-order samples cannot set observer_id")
        if sample["mode"] == "public_only" and sample["modeler_id"] is not None:
            raise ValueError("public-only samples cannot set modeler_id")
        if sample["mode"] == "wolf_conditioned" and sample["modeler_id"] is None:
            raise ValueError("wolf-conditioned samples require modeler_id")

    if not isinstance(sample["events"], list):
        raise ValueError("events must be a list")
    for event in sample["events"]:
        validate_event(event)
    _validate_mask(sample["output_mask"])

    if sample["task"] == "first_order":
        observer_id = sample["observer_id"]
        if any(
            event["visibility"] == "private" and observer_id not in event["visible_to"]
            for event in sample["events"]
        ):
            raise ValueError("first-order sample contains another player's private event")
        knowledge = knowledge_for_player(sample["events"], observer_id)
        if knowledge["role"] is None:
            raise ValueError("first-order sample requires the observer SELF_ROLE fact")
        if knowledge["role"] == "Werewolf":
            raise ValueError("wolves are excluded from the main first-order dataset")
        expected_mask = first_order_knowledge_mask(
            observer_id=observer_id,
            observer_role=knowledge["role"],
            known_wolves=knowledge["known_wolves"],
            known_good=knowledge["known_good"],
        ).tolist()
        if sample["output_mask"] != expected_mask:
            raise ValueError("first-order output_mask does not match visible hard knowledge")
    elif sample["mode"] == "public_only":
        if any(event["visibility"] != "public" for event in sample["events"]):
            raise ValueError("public-only second-order sample contains a private event")
        if sample["output_mask"] != [True] * NUM_WOLF_PAIRS:
            raise ValueError("public-only second-order output_mask must keep all 21 pairs")
    else:
        modeler_id = sample["modeler_id"]
        if any(
            event["visibility"] == "private" and modeler_id not in event["visible_to"]
            for event in sample["events"]
        ):
            raise ValueError("wolf-conditioned sample contains private target information")
        modeler_knowledge = knowledge_for_player(sample["events"], modeler_id)
        if modeler_knowledge["role"] != "Werewolf":
            raise ValueError("wolf-conditioned modeler requires a Werewolf SELF_ROLE fact")
        expected_mask = second_order_output_mask(
            mode="wolf_conditioned", target_id=sample["target_id"]
        ).tolist()
        if sample["output_mask"] != expected_mask:
            raise ValueError("wolf-conditioned output_mask may exclude only target pairs")

    guess = sample["guess"]
    if not isinstance(guess, dict) or set(guess) != GUESS_FIELDS:
        raise ValueError("guess fields do not match the schema")
    if guess["status"] not in GUESS_STATUSES:
        raise ValueError("invalid guess status")
    if not isinstance(guess["raw_text"], list) or any(
        not isinstance(text, str) for text in guess["raw_text"]
    ):
        raise ValueError("guess.raw_text must be a list of responses")
    if type(guess["attempts"]) is not int or not 1 <= guess["attempts"] <= 2:
        raise ValueError("guess.attempts must be one or two")
    if guess["model"] is not None and not isinstance(guess["model"], str):
        raise ValueError("guess.model must be text or null")
    if guess["error"] is not None and not isinstance(guess["error"], str):
        raise ValueError("guess.error must be text or null")

    if guess["status"] == "ok":
        pair = normalize_pair(sample["label_pair"])
        index = pair_index(pair)
        if sample["label_index"] != index:
            raise ValueError("label_index does not match label_pair")
        if not sample["output_mask"][index]:
            raise ValueError("label_pair is excluded by output_mask")
        if guess["error"] is not None:
            raise ValueError("successful guess cannot carry an error")
    else:
        if sample["label_pair"] is not None or sample["label_index"] is not None:
            raise ValueError("failed guesses cannot carry labels")
        if not guess["error"]:
            raise ValueError("failed guesses require an error")
        if require_success:
            raise ValueError("failed belief elicitation is not a training sample")
    return True


def make_sample(
    *,
    sample_id,
    game_id,
    task,
    mode,
    checkpoint,
    state_id,
    day,
    phase,
    turn,
    events,
    output_mask,
    guess,
    observer_id=None,
    modeler_id=None,
    target_id=None,
) -> dict:
    pair = guess.pair if guess.status == "ok" else None
    sample = {
        "schema_version": TOM_SCHEMA_VERSION,
        "sample_id": sample_id,
        "game_id": game_id,
        "task": task,
        "mode": mode,
        "checkpoint": checkpoint,
        "state_id": state_id,
        "day": day,
        "phase": phase,
        "turn": turn,
        "observer_id": observer_id,
        "modeler_id": modeler_id,
        "target_id": target_id,
        "events": deepcopy(events),
        "output_mask": [bool(value) for value in output_mask],
        "label_pair": list(pair) if pair is not None else None,
        "label_index": pair_index(pair) if pair is not None else None,
        "guess": {
            "status": guess.status,
            "raw_text": list(guess.raw_text),
            "error": guess.error,
            "attempts": guess.attempts,
            "model": guess.model,
        },
    }
    validate_sample(sample, require_success=False)
    return sample
