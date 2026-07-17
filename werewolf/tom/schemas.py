"""Strict, versioned JSON schemas for ToM collection samples."""

from copy import deepcopy
import re

from werewolf.events.schema import validate_event
from werewolf.events.streams import knowledge_for_player
from werewolf.game_rules import canonical_ruleset_metadata
from werewolf.prompt_protocol import (
    PROMPT_LANGUAGE,
    PROMPT_NAMES,
    PROMPT_PROTOCOL_VERSION,
    protocol_id_from_references,
)
from werewolf.tom.masks import (
    SECOND_ORDER_MODES,
    first_order_constraints,
    first_order_knowledge_mask,
    second_order_output_mask,
)
from werewolf.tom.guess_provider import GUESS_ERROR_CODES
from werewolf.tom.pair_space import NUM_WOLF_PAIRS, normalize_pair, pair_index


TOM_SCHEMA_VERSION = "tom.v1_1"
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
    "checkpoint_scope",
    "state_id",
    "public_state_id",
    "source_first_order_sample_id",
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
    "prompt_protocol",
}
GUESS_FIELDS = {
    "status",
    "raw_text",
    "error",
    "attempts",
    "model",
    "first_error_code",
    "final_error_code",
    "required_wolves",
    "forbidden_wolves",
}
PROMPT_PROTOCOL_FIELDS = {
    "protocol_version",
    "language",
    "protocol_id",
    "ruleset",
    "gameplay",
    "belief",
    "parser",
    "runtime",
}
PROMPT_REFERENCE_FIELDS = {"version", "sha256"}
RULESET_REFERENCE_FIELDS = {"id", "version", "sha256"}
RUNTIME_FIELDS = {"gameplay_profiles", "belief_profiles", "parser"}
RUNTIME_MODEL_FIELDS = {"backend", "model", "temperature"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _validate_runtime_model(value, name):
    if not isinstance(value, dict) or set(value) != RUNTIME_MODEL_FIELDS:
        raise ValueError(f"{name} fields do not match the prompt runtime schema")
    for field in ("backend", "model"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{name}.{field} is required")
    if isinstance(value["temperature"], bool) or not isinstance(
        value["temperature"], (int, float)
    ):
        raise ValueError(f"{name}.temperature must be numeric")


def validate_prompt_protocol(prompt_protocol) -> bool:
    if not isinstance(prompt_protocol, dict) or set(prompt_protocol) != PROMPT_PROTOCOL_FIELDS:
        raise ValueError("prompt_protocol fields do not match prompt_protocol.zh.v4")
    if prompt_protocol["protocol_version"] != PROMPT_PROTOCOL_VERSION:
        raise ValueError("unsupported prompt_protocol version")
    if prompt_protocol["language"] != PROMPT_LANGUAGE:
        raise ValueError("prompt_protocol.language must be zh-CN")
    ruleset = prompt_protocol["ruleset"]
    if not isinstance(ruleset, dict) or set(ruleset) != RULESET_REFERENCE_FIELDS:
        raise ValueError("prompt_protocol.ruleset fields are invalid")
    if ruleset != canonical_ruleset_metadata():
        raise ValueError("prompt_protocol.ruleset does not match the canonical ruleset")
    references = {}
    for name in PROMPT_NAMES:
        reference = prompt_protocol[name]
        if not isinstance(reference, dict) or set(reference) != PROMPT_REFERENCE_FIELDS:
            raise ValueError(f"prompt_protocol.{name} fields are invalid")
        if not isinstance(reference["version"], str) or not re.fullmatch(
            rf"{name}\.zh\.v[1-9][0-9]*", reference["version"]
        ):
            raise ValueError(f"prompt_protocol.{name}.version is invalid")
        if not isinstance(reference["sha256"], str) or not SHA256_PATTERN.fullmatch(
            reference["sha256"]
        ):
            raise ValueError(f"prompt_protocol.{name}.sha256 is invalid")
        references[name] = reference
    protocol_id = prompt_protocol["protocol_id"]
    if not isinstance(protocol_id, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", protocol_id
    ):
        raise ValueError("prompt_protocol.protocol_id is invalid")
    expected_id = protocol_id_from_references(
        references,
        ruleset=ruleset,
        protocol_version=prompt_protocol["protocol_version"],
        language=prompt_protocol["language"],
    )
    if protocol_id != expected_id:
        raise ValueError("prompt_protocol.protocol_id does not match prompt specs")

    runtime = prompt_protocol["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_FIELDS:
        raise ValueError("prompt_protocol.runtime fields are invalid")
    gameplay_profiles = runtime["gameplay_profiles"]
    belief_profiles = runtime["belief_profiles"]
    if not isinstance(gameplay_profiles, dict) or not gameplay_profiles:
        raise ValueError("prompt runtime requires gameplay_profiles")
    if not isinstance(belief_profiles, dict) or set(belief_profiles) != set(
        gameplay_profiles
    ):
        raise ValueError("belief_profiles must match gameplay profile names")
    for profile_name, profile in gameplay_profiles.items():
        if not isinstance(profile_name, str) or not profile_name:
            raise ValueError("gameplay profile names must be non-empty text")
        _validate_runtime_model(profile, f"gameplay_profiles.{profile_name}")
        _validate_runtime_model(
            belief_profiles[profile_name], f"belief_profiles.{profile_name}"
        )
    _validate_runtime_model(runtime["parser"], "parser runtime")
    return True


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


def _validate_player_list(values, name):
    if not isinstance(values, list) or values != sorted(set(values)):
        raise ValueError(f"guess.{name} must be a sorted unique list")
    if any(type(value) is not int or not 1 <= value <= 7 for value in values):
        raise ValueError(f"guess.{name} must contain player ids between 1 and 7")


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
    validate_prompt_protocol(sample["prompt_protocol"])
    for name in ("sample_id", "game_id", "checkpoint", "state_id", "phase"):
        if not isinstance(sample[name], str) or not sample[name]:
            raise ValueError(f"{name} is required")
    if sample["task"] not in TASKS:
        raise ValueError("invalid task")
    if sample["checkpoint_scope"] not in ("public", "private"):
        raise ValueError("checkpoint_scope must be public or private")
    if sample["public_state_id"] is not None and (
        not isinstance(sample["public_state_id"], str) or not sample["public_state_id"]
    ):
        raise ValueError("public_state_id must be non-empty text or null")
    if sample["source_first_order_sample_id"] is not None and (
        not isinstance(sample["source_first_order_sample_id"], str)
        or not sample["source_first_order_sample_id"]
    ):
        raise ValueError("source_first_order_sample_id must be non-empty text or null")
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
        if sample["source_first_order_sample_id"] is not None:
            raise ValueError("first-order samples cannot have a source first-order sample")
        if sample["checkpoint_scope"] == "public":
            if sample["public_state_id"] != sample["state_id"]:
                raise ValueError("public first-order state ids must match")
        elif sample["public_state_id"] is not None:
            raise ValueError("private checkpoints cannot set public_state_id")
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
        if sample["checkpoint_scope"] != "public":
            raise ValueError("second-order samples require a public checkpoint")
        if sample["public_state_id"] != sample["state_id"]:
            raise ValueError("second-order public_state_id must match state_id")
        if sample["source_first_order_sample_id"] is None:
            raise ValueError("second-order samples require source_first_order_sample_id")

    if not isinstance(sample["events"], list):
        raise ValueError("events must be a list")
    for event in sample["events"]:
        validate_event(event)
        if event["turn"] > sample["turn"]:
            raise ValueError("sample contains a future event after its checkpoint")
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
        known_wolves = set(modeler_knowledge["known_wolves"])
        if len(known_wolves) != 2 or modeler_id not in known_wolves:
            raise ValueError("wolf-conditioned modeler requires the exact visible wolf team")
        for event in sample["events"]:
            if event["visibility"] != "private":
                continue
            kind = event["content"]["kind"]
            if kind == "SELF_ROLE":
                if event["target"] != [modeler_id] or event["visible_to"] != [modeler_id]:
                    raise ValueError("wolf-conditioned sample contains a god-view role fact")
            elif kind == "WOLF_TEAM":
                if modeler_id not in event["target"] or set(event["visible_to"]) != set(event["target"]):
                    raise ValueError("wolf-conditioned WOLF_TEAM visibility is inconsistent")
            elif kind == "PRIVATE_ACTION_RESULT":
                if (
                    set(event["visible_to"]) != known_wolves
                    or event["speaker"] not in known_wolves
                ):
                    raise ValueError(
                        "wolf-conditioned private action visibility is inconsistent"
                    )
            else:
                raise ValueError("wolf-conditioned sample contains target private information")
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
    for name in ("required_wolves", "forbidden_wolves"):
        _validate_player_list(guess[name], name)
    if set(guess["required_wolves"]) & set(guess["forbidden_wolves"]):
        raise ValueError("guess required and forbidden constraints overlap")
    for name in ("first_error_code", "final_error_code"):
        value = guess[name]
        if value is not None and value not in GUESS_ERROR_CODES:
            raise ValueError(f"guess.{name} is invalid")

    if sample["task"] == "first_order":
        expected_constraints = first_order_constraints(
            observer_id=sample["observer_id"],
            observer_role=knowledge["role"],
            known_wolves=knowledge["known_wolves"],
            known_good=knowledge["known_good"],
        )
        for name in ("required_wolves", "forbidden_wolves"):
            if guess[name] != list(expected_constraints[name]):
                raise ValueError(f"guess.{name} does not match visible hard knowledge")

    if guess["status"] == "ok":
        pair = normalize_pair(sample["label_pair"])
        index = pair_index(pair)
        if sample["label_index"] != index:
            raise ValueError("label_index does not match label_pair")
        if not sample["output_mask"][index]:
            raise ValueError("label_pair is excluded by output_mask")
        if guess["error"] is not None:
            raise ValueError("successful guess cannot carry an error")
        if guess["final_error_code"] is not None:
            raise ValueError("successful guess cannot carry a final error code")
        if guess["attempts"] == 1 and guess["first_error_code"] is not None:
            raise ValueError("first-attempt success cannot carry an error code")
        if guess["attempts"] == 2 and guess["first_error_code"] is None:
            raise ValueError("second-attempt success requires its first error code")
        if not set(guess["required_wolves"]).issubset(pair):
            raise ValueError("successful pair misses a required wolf")
        if set(guess["forbidden_wolves"]) & set(pair):
            raise ValueError("successful pair contains a forbidden player")
    else:
        if sample["label_pair"] is not None or sample["label_index"] is not None:
            raise ValueError("failed guesses cannot carry labels")
        if not guess["error"]:
            raise ValueError("failed guesses require an error")
        if guess["attempts"] == 1:
            if (
                guess["first_error_code"] != "backend_error"
                or guess["final_error_code"] != "backend_error"
            ):
                raise ValueError(
                    "one-attempt failures must be non-retryable backend errors"
                )
        elif (
            guess["first_error_code"] is None
            or guess["final_error_code"] is None
        ):
            raise ValueError("two-attempt failures require both error codes")
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
    checkpoint_scope,
    state_id,
    public_state_id,
    day,
    phase,
    turn,
    events,
    output_mask,
    guess,
    prompt_protocol,
    observer_id=None,
    modeler_id=None,
    target_id=None,
    source_first_order_sample_id=None,
) -> dict:
    pair = guess.pair if guess.status == "ok" else None
    sample = {
        "schema_version": TOM_SCHEMA_VERSION,
        "sample_id": sample_id,
        "game_id": game_id,
        "task": task,
        "mode": mode,
        "checkpoint": checkpoint,
        "checkpoint_scope": checkpoint_scope,
        "state_id": state_id,
        "public_state_id": public_state_id,
        "source_first_order_sample_id": source_first_order_sample_id,
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
            "first_error_code": guess.first_error_code,
            "final_error_code": guess.final_error_code,
            "required_wolves": list(guess.required_wolves),
            "forbidden_wolves": list(guess.forbidden_wolves),
        },
        "prompt_protocol": deepcopy(prompt_protocol),
    }
    validate_sample(sample, require_success=False)
    return sample


def first_order_key(sample):
    return sample["game_id"], sample["state_id"], sample["observer_id"]


def second_order_key(sample):
    key = (
        sample["game_id"],
        sample["public_state_id"],
        sample["mode"],
    )
    if sample["mode"] == "public_only":
        return (*key, sample["target_id"])
    return (*key, sample["modeler_id"], sample["target_id"])


def validate_sample_collection(samples) -> bool:
    """Validate global identities and first-to-second-order label provenance."""

    sample_ids = set()
    first_keys = set()
    second_keys = set()
    first_by_id = {}
    for sample in samples:
        validate_sample(sample, require_success=False)
        sample_id = sample["sample_id"]
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if sample["task"] == "first_order":
            key = first_order_key(sample)
            if key in first_keys:
                raise ValueError(f"duplicate first-order key: {key}")
            first_keys.add(key)
            first_by_id[sample_id] = sample
        else:
            key = second_order_key(sample)
            if key in second_keys:
                raise ValueError(f"duplicate second-order key: {key}")
            second_keys.add(key)

    for sample in samples:
        if sample["task"] != "second_order":
            continue
        source_id = sample["source_first_order_sample_id"]
        source = first_by_id.get(source_id)
        if source is None:
            raise ValueError(f"missing source first-order sample: {source_id}")
        if source["checkpoint_scope"] != "public":
            raise ValueError("private checkpoint cannot source a second-order sample")
        if sample["game_id"] != source["game_id"]:
            raise ValueError("second-order game_id does not match its source")
        if sample["public_state_id"] != source["public_state_id"]:
            raise ValueError("second-order public_state_id does not match its source")
        if sample["target_id"] != source["observer_id"]:
            raise ValueError("second-order target_id does not match its source observer")
        if sample["label_pair"] != source["label_pair"]:
            raise ValueError("second-order label_pair does not match its source")
        if sample["label_index"] != source["label_index"]:
            raise ValueError("second-order label_index does not match its source")
    return True
