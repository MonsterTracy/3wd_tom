"""Checkpoint-driven collection for first- and second-order ToM labels."""

import json
import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from werewolf.events.encoder import (
    EVENT_TOKEN_FIELDS,
    KIND2ID,
    VALUE2ID,
    encode_event,
)
from werewolf.events.streams import (
    knowledge_for_player,
    public_events,
    render_stream,
    visible_events,
)
from werewolf.prompt_protocol import PARSER_PROMPT_SPEC
from werewolf.tom.masks import (
    first_order_constraints,
    first_order_knowledge_mask,
    second_order_output_mask,
)
from werewolf.tom.schemas import (
    FIRST_ORDER_MODE,
    first_order_key,
    make_sample,
    second_order_key,
    validate_prompt_protocol,
    validate_sample,
)


PUBLIC_CHECKPOINTS = {
    "SPEECH": "after_speech",
    "VOTE_CAST": "after_public_vote",
    "VOTE_RESULT": "after_vote_result",
    "EXILE": "after_exile",
    "DEATH": "after_death",
    "ROLE_REVEAL": "after_reveal",
}
PRIVATE_FIRST_ORDER_CHECKPOINTS = {
    "CHECK_RESULT": "after_seer_check",
    "WOLF_TEAM": "after_wolf_team",
    "WITCH_STATE": "after_witch_info",
    "GUARD_RESULT": "after_guard_result",
}
PARSER_FAILURE_RATE_LIMIT = 0.20
BELIEF_SUCCESS_RATE_MINIMUM = 0.95


@dataclass(frozen=True)
class CollectionBatch:
    samples: tuple[dict, ...]
    failures: tuple[dict, ...]


class JsonlSink:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record):
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def _distribution(counter):
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _percentile(values, percentile):
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _safe_unknown_raw_value(value):
    if isinstance(value, dict):
        value = {
            str(key): (
                "<redacted>"
                if any(
                    marker in str(key).lower()
                    for marker in ("api_key", "token", "secret", "password")
                )
                else item
            )
            for key, item in list(value.items())[:20]
        }
    rendered = repr(value)
    if rendered.startswith(("'sk-", '"sk-')):
        return "'<redacted>'"
    return rendered[:160]


def _raw_guess_pair(guess):
    raw_text = guess.get("raw_text") if isinstance(guess, dict) else None
    if not isinstance(raw_text, list) or not raw_text:
        return None
    try:
        payload = json.loads(raw_text[-1])
        values = payload["wolf_pair"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(type(value) is not int or not 1 <= value <= 7 for value in values)
        or values[0] == values[1]
    ):
        return None
    return tuple(sorted(values))


def build_audit_report(
    samples,
    failures=(),
    *,
    game_ids=(),
    max_sequence_length=512,
):
    """Build deterministic collection statistics without changing any records."""

    successful = list(samples)
    records = successful + list(failures)
    sample_id_counts = Counter()
    first_key_counts = Counter()
    second_key_counts = Counter()
    schema_errors = 0
    labels_outside_mask = 0
    public_checkpoints = set()
    private_checkpoints = set()
    games = {str(game_id) for game_id in game_ids}
    first_by_id = {}
    first_records = []
    second_records = []
    prompt_protocol_distribution = Counter()
    gameplay_prompt_versions = set()
    belief_prompt_versions = set()
    parser_prompt_versions = set()
    gameplay_prompt_hashes = set()
    belief_prompt_hashes = set()
    parser_prompt_hashes = set()
    runtime_model_distribution = Counter()
    missing_prompt_protocol_count = 0
    invalid_prompt_protocol_count = 0

    for record in records:
        if not isinstance(record, dict):
            schema_errors += 1
            missing_prompt_protocol_count += 1
            continue
        prompt_protocol = record.get("prompt_protocol")
        if prompt_protocol is None:
            missing_prompt_protocol_count += 1
        else:
            protocol_id = prompt_protocol.get("protocol_id") if isinstance(
                prompt_protocol, dict
            ) else None
            if isinstance(protocol_id, str):
                prompt_protocol_distribution[protocol_id] += 1
            if isinstance(prompt_protocol, dict):
                for name, versions, hashes in (
                    ("gameplay", gameplay_prompt_versions, gameplay_prompt_hashes),
                    ("belief", belief_prompt_versions, belief_prompt_hashes),
                    ("parser", parser_prompt_versions, parser_prompt_hashes),
                ):
                    reference = prompt_protocol.get(name)
                    if isinstance(reference, dict):
                        if isinstance(reference.get("version"), str):
                            versions.add(reference["version"])
                        if isinstance(reference.get("sha256"), str):
                            hashes.add(reference["sha256"])
            try:
                validate_prompt_protocol(prompt_protocol)
            except (KeyError, TypeError, ValueError):
                invalid_prompt_protocol_count += 1
            else:
                runtime = prompt_protocol["runtime"]
                for component in ("gameplay", "belief"):
                    profiles = runtime[f"{component}_profiles"]
                    for profile_name, profile in profiles.items():
                        runtime_model_distribution[
                            f"{component}:{profile_name}:{profile['backend']}:"
                            f"{profile['model']}:temperature={profile['temperature']}"
                        ] += 1
                parser_runtime = runtime["parser"]
                runtime_model_distribution[
                    f"parser:{parser_runtime['backend']}:{parser_runtime['model']}:"
                    f"temperature={parser_runtime['temperature']}"
                ] += 1
        if record.get("game_id") is not None:
            games.add(str(record["game_id"]))
        sample_id = record.get("sample_id")
        if sample_id is not None:
            sample_id_counts[sample_id] += 1
        try:
            validate_sample(record, require_success=False)
        except (TypeError, ValueError, KeyError):
            schema_errors += 1
        label_index = record.get("label_index")
        mask = record.get("output_mask")
        if label_index is not None and (
            type(label_index) is not int
            or not isinstance(mask, list)
            or not 0 <= label_index < len(mask)
            or not mask[label_index]
        ):
            labels_outside_mask += 1
        scope = record.get("checkpoint_scope")
        checkpoint_key = (record.get("game_id"), record.get("state_id"))
        if scope == "public":
            public_checkpoints.add(checkpoint_key)
        elif scope == "private":
            private_checkpoints.add(checkpoint_key)
        try:
            if record.get("task") == "first_order":
                key = first_order_key(record)
                first_key_counts[key] += 1
                first_records.append(record)
                first_by_id.setdefault(record["sample_id"], record)
            elif record.get("task") == "second_order":
                key = second_order_key(record)
                second_key_counts[key] += 1
                second_records.append(record)
        except (KeyError, TypeError):
            pass

    state_alignment_errors = 0
    private_to_second_order_errors = 0
    for second in second_records:
        source = first_by_id.get(second.get("source_first_order_sample_id"))
        if second.get("checkpoint_scope") != "public":
            private_to_second_order_errors += 1
        if source is None:
            state_alignment_errors += 1
            continue
        if source.get("checkpoint_scope") != "public":
            private_to_second_order_errors += 1
        aligned = (
            second.get("game_id") == source.get("game_id")
            and second.get("public_state_id") == source.get("public_state_id")
            and second.get("target_id") == source.get("observer_id")
            and second.get("label_pair") == source.get("label_pair")
            and second.get("label_index") == source.get("label_index")
        )
        if not aligned:
            state_alignment_errors += 1

    unique_first = {}
    for record in first_records:
        try:
            unique_first.setdefault(first_order_key(record), record)
        except KeyError:
            continue
    successful_guesses = sum(
        record.get("guess", {}).get("status") == "ok"
        for record in unique_first.values()
    )
    failed_guesses = sum(
        record.get("guess", {}).get("status") == "failed"
        for record in unique_first.values()
    )
    repair_attempts = sum(
        max(0, record.get("guess", {}).get("attempts", 1) - 1)
        for record in unique_first.values()
    )
    belief_failure_reasons = Counter()
    belief_contains_observer_failures = 0
    belief_missing_required_wolf_failures = 0
    belief_contains_forbidden_player_failures = 0
    belief_invalid_format_failures = 0
    belief_success_constraint_violations = 0
    belief_first_attempt_successes = 0
    belief_repair_successes = 0
    belief_repair_failures = 0
    invalid_format_codes = {
        "invalid_json",
        "not_exactly_two_players",
        "duplicate_players",
        "out_of_range",
    }
    for record in unique_first.values():
        guess = record.get("guess", {})
        status = guess.get("status")
        attempts = guess.get("attempts")
        if status == "ok":
            if attempts == 1:
                belief_first_attempt_successes += 1
            elif attempts == 2:
                belief_repair_successes += 1
            pair = record.get("label_pair") or []
            if (
                not set(guess.get("required_wolves", ())).issubset(pair)
                or set(guess.get("forbidden_wolves", ())) & set(pair)
            ):
                belief_success_constraint_violations += 1
            continue
        if status != "failed":
            continue
        if attempts == 2:
            belief_repair_failures += 1
        error_code = guess.get("final_error_code") or "unknown"
        belief_failure_reasons[error_code] += 1
        pair = _raw_guess_pair(guess)
        required = set(guess.get("required_wolves", ()))
        forbidden = set(guess.get("forbidden_wolves", ()))
        if pair is None:
            belief_invalid_format_failures += 1
        else:
            if record.get("observer_id") in pair:
                belief_contains_observer_failures += 1
            if not required.issubset(pair):
                belief_missing_required_wolf_failures += 1
            if forbidden & set(pair):
                belief_contains_forbidden_player_failures += 1
        if error_code in invalid_format_codes and pair is not None:
            belief_invalid_format_failures += 1
    belief_success_rate = (
        successful_guesses / len(unique_first) if unique_first else 1.0
    )
    belief_repair_success_rate = (
        belief_repair_successes / repair_attempts if repair_attempts else 1.0
    )

    unique_events = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("events"), list):
            continue
        for event in record["events"]:
            if isinstance(event, dict) and isinstance(event.get("event_id"), str):
                unique_events.setdefault(
                    (str(record.get("game_id")), event["event_id"]), event
                )
    speech_events = {
        key: event
        for key, event in unique_events.items()
        if event.get("event_family") == "GAME_EVENT"
        and event.get("content", {}).get("kind") == "SPEECH"
    }
    semantic_events = {
        key: event
        for key, event in unique_events.items()
        if event.get("source_type") == "speech_parser"
    }
    speech_utterances = {
        (game_id, event.get("utterance_id"))
        for (game_id, _), event in speech_events.items()
    }
    semantic_utterances = {
        (game_id, event.get("utterance_id"))
        for (game_id, _), event in semantic_events.items()
    }
    parsed_event_families = Counter(
        event.get("event_family") for event in semantic_events.values()
    )
    parser_statuses = Counter()
    parser_failure_reasons = Counter()
    parser_repair_attempts = 0
    missing_parser_metadata_count = 0
    parser_metadata_fields = {
        "version",
        "sha256",
        "model",
        "temperature",
        "status",
        "attempts",
        "error_code",
        "error",
    }
    valid_parser_metadata = {}
    for key, event in speech_events.items():
        metadata = event.get("metadata", {}).get("parser_result")
        valid = (
            isinstance(metadata, dict)
            and set(metadata) == parser_metadata_fields
            and metadata.get("version") == PARSER_PROMPT_SPEC["version"]
            and metadata.get("sha256") == PARSER_PROMPT_SPEC["sha256"]
            and isinstance(metadata.get("model"), str)
            and bool(metadata.get("model"))
            and metadata.get("temperature") == 0.0
            and metadata.get("status") in {"success", "empty", "failed"}
            and type(metadata.get("attempts")) is int
            and 1 <= metadata["attempts"] <= 2
            and (
                (
                    metadata.get("status") in {"success", "empty"}
                    and metadata.get("error_code") is None
                    and metadata.get("error") is None
                )
                or (
                    metadata.get("status") == "failed"
                    and isinstance(metadata.get("error_code"), str)
                    and bool(metadata.get("error_code"))
                    and isinstance(metadata.get("error"), str)
                    and bool(metadata.get("error"))
                )
            )
        )
        if not valid:
            missing_parser_metadata_count += 1
            continue
        valid_parser_metadata[key] = metadata
        parser_statuses[metadata["status"]] += 1
        parser_repair_attempts += metadata["attempts"] - 1
        if metadata["status"] == "failed":
            parser_failure_reasons[metadata.get("error_code") or "unknown"] += 1
    parser_utterance_mismatch_count = len(speech_events) - len(speech_utterances)
    parser_utterance_mismatch_count += sum(
        utterance not in speech_utterances for utterance in semantic_utterances
    )
    for key, metadata in valid_parser_metadata.items():
        game_id = key[0]
        utterance_id = speech_events[key]["utterance_id"]
        has_semantic = (game_id, utterance_id) in semantic_utterances
        if (metadata["status"] == "success") != has_semantic:
            parser_utterance_mismatch_count += 1
    speech_with_semantic_events = len(speech_utterances & semantic_utterances)

    mask_sizes = Counter()
    pair_labels = Counter()
    event_families = Counter()
    samples_by_day = Counter()
    samples_by_phase = Counter()
    first_by_role = Counter()
    sequence_lengths = []
    unknown_kind_count = 0
    unknown_value_count = 0
    not_applicable_value_count = 0
    semantic_token_count = 0
    unknown_by_event_family = Counter()
    unknown_by_content_kind = Counter()
    unknown_raw_values = Counter()
    kind_index = EVENT_TOKEN_FIELDS.index("kind_id")
    value_index = EVENT_TOKEN_FIELDS.index("value_id")
    for sample in successful:
        mask = sample.get("output_mask", [])
        if isinstance(mask, list):
            mask_sizes[sum(value is True for value in mask)] += 1
        label_pair = sample.get("label_pair")
        if isinstance(label_pair, list) and len(label_pair) == 2:
            pair_labels[f"{label_pair[0]}-{label_pair[1]}"] += 1
        samples_by_day[sample.get("day")] += 1
        samples_by_phase[sample.get("phase")] += 1
        events = sample.get("events", [])
        if not isinstance(events, list):
            events = []
        event_families.update(
            event.get("event_family")
            for event in events
            if isinstance(event, dict) and event.get("event_family") is not None
        )
        tokens = []
        for event in events:
            try:
                event_tokens = encode_event(event)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            tokens.extend(event_tokens)
        sequence_lengths.append(len(tokens))
        if sample.get("task") == "first_order":
            try:
                role = knowledge_for_player(events, sample["observer_id"])["role"]
            except (KeyError, TypeError, ValueError):
                role = "unknown"
            first_by_role[role or "unknown"] += 1

    for record in records:
        if not isinstance(record, dict) or not isinstance(
            record.get("events"), list
        ):
            continue
        for event in record["events"]:
            try:
                event_tokens = encode_event(event)
                family = event.get("event_family", "<missing>")
                content = event.get("content", {})
                content_kind = content.get("kind", "<missing>")
                raw_value = content.get("value")
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            for token in event_tokens:
                semantic_token_count += 1
                if token[kind_index] == KIND2ID["UNKNOWN"]:
                    unknown_kind_count += 1
                    unknown_by_event_family[family] += 1
                    unknown_by_content_kind[content_kind] += 1
                    continue
                if token[value_index] == VALUE2ID["NONE"]:
                    not_applicable_value_count += 1
                    continue
                semantic_token_count += 1
                if token[value_index] == VALUE2ID["UNKNOWN"]:
                    unknown_value_count += 1
                    unknown_by_event_family[family] += 1
                    unknown_by_content_kind[content_kind] += 1
                    unknown_raw_values[_safe_unknown_raw_value(raw_value)] += 1

    sequence_length = {
        "min": min(sequence_lengths, default=0),
        "p50": _percentile(sequence_lengths, 0.50),
        "p90": _percentile(sequence_lengths, 0.90),
        "p95": _percentile(sequence_lengths, 0.95),
        "p99": _percentile(sequence_lengths, 0.99),
        "max": max(sequence_lengths, default=0),
    }
    return {
        "schema_version": "tom.audit.v1_3",
        "games": len(games),
        "public_checkpoints": len(public_checkpoints),
        "private_checkpoints": len(private_checkpoints),
        "unique_belief_elicitations": len(unique_first),
        "successful_guesses": successful_guesses,
        "failed_guesses": failed_guesses,
        "repair_attempts": repair_attempts,
        "belief_failure_reason_distribution": _distribution(belief_failure_reasons),
        "belief_first_attempt_successes": belief_first_attempt_successes,
        "belief_repair_successes": belief_repair_successes,
        "belief_repair_failures": belief_repair_failures,
        "belief_contains_observer_failures": belief_contains_observer_failures,
        "belief_missing_required_wolf_failures": belief_missing_required_wolf_failures,
        "belief_contains_forbidden_player_failures": belief_contains_forbidden_player_failures,
        "belief_invalid_format_failures": belief_invalid_format_failures,
        "belief_success_constraint_violations": belief_success_constraint_violations,
        "belief_success_rate": belief_success_rate,
        "belief_repair_success_rate": belief_repair_success_rate,
        "speech_event_count": len(speech_events),
        "parser_call_count": len(valid_parser_metadata),
        "parser_success_count": parser_statuses["success"],
        "parser_empty_count": parser_statuses["empty"],
        "parser_failure_count": parser_statuses["failed"],
        "parser_repair_attempts": parser_repair_attempts,
        "parsed_semantic_event_count": len(semantic_events),
        "speech_with_semantic_events": speech_with_semantic_events,
        "speech_without_semantic_events": len(speech_events) - speech_with_semantic_events,
        "parsed_event_family_distribution": _distribution(parsed_event_families),
        "parser_failure_reason_distribution": _distribution(parser_failure_reasons),
        "missing_parser_metadata_count": missing_parser_metadata_count,
        "parser_utterance_mismatch_count": parser_utterance_mismatch_count,
        "first_order_samples": sum(
            sample.get("task") == "first_order" for sample in successful
        ),
        "second_order_public_samples": sum(
            sample.get("task") == "second_order" and sample.get("mode") == "public_only"
            for sample in successful
        ),
        "second_order_wolf_samples": sum(
            sample.get("task") == "second_order" and sample.get("mode") == "wolf_conditioned"
            for sample in successful
        ),
        "duplicate_sample_ids": sum(count - 1 for count in sample_id_counts.values() if count > 1),
        "duplicate_first_order_keys": sum(count - 1 for count in first_key_counts.values() if count > 1),
        "duplicate_second_order_keys": sum(count - 1 for count in second_key_counts.values() if count > 1),
        "state_alignment_errors": state_alignment_errors,
        "private_to_second_order_errors": private_to_second_order_errors,
        "schema_errors": schema_errors,
        "labels_outside_mask": labels_outside_mask,
        "mask_size_distribution": _distribution(mask_sizes),
        "pair_label_distribution": _distribution(pair_labels),
        "event_family_distribution": _distribution(event_families),
        "unknown_kind_count": unknown_kind_count,
        "unknown_value_count": unknown_value_count,
        "not_applicable_value_count": not_applicable_value_count,
        "unknown_by_event_family": _distribution(unknown_by_event_family),
        "unknown_by_content_kind": _distribution(unknown_by_content_kind),
        "top_unknown_raw_values": [
            {"value": value, "count": count}
            for value, count in sorted(
                unknown_raw_values.items(), key=lambda item: (-item[1], item[0])
            )[:20]
        ],
        "unknown_token_count": unknown_kind_count + unknown_value_count,
        "unknown_token_ratio": (
            (unknown_kind_count + unknown_value_count) / semantic_token_count
            if semantic_token_count
            else 0.0
        ),
        "sequence_length": sequence_length,
        "truncated_samples": sum(
            length > max_sequence_length for length in sequence_lengths
        ),
        "samples_by_day": _distribution(samples_by_day),
        "samples_by_phase": _distribution(samples_by_phase),
        "first_order_by_observer_role": _distribution(first_by_role),
        "prompt_protocol_ids": sorted(prompt_protocol_distribution),
        "prompt_protocol_distribution": _distribution(prompt_protocol_distribution),
        "gameplay_prompt_versions": sorted(gameplay_prompt_versions),
        "belief_prompt_versions": sorted(belief_prompt_versions),
        "parser_prompt_versions": sorted(parser_prompt_versions),
        "gameplay_prompt_hashes": sorted(gameplay_prompt_hashes),
        "belief_prompt_hashes": sorted(belief_prompt_hashes),
        "parser_prompt_hashes": sorted(parser_prompt_hashes),
        "runtime_model_distribution": _distribution(runtime_model_distribution),
        "missing_prompt_protocol_count": missing_prompt_protocol_count,
        "invalid_prompt_protocol_count": invalid_prompt_protocol_count,
    }


def assert_audit_passes(report):
    fatal_fields = (
        "duplicate_sample_ids",
        "duplicate_first_order_keys",
        "duplicate_second_order_keys",
        "state_alignment_errors",
        "private_to_second_order_errors",
        "schema_errors",
        "labels_outside_mask",
        "unknown_kind_count",
        "unknown_value_count",
        "missing_prompt_protocol_count",
        "invalid_prompt_protocol_count",
        "belief_contains_observer_failures",
        "belief_success_constraint_violations",
        "missing_parser_metadata_count",
        "parser_utterance_mismatch_count",
    )
    failures = {name: report.get(name, 0) for name in fatal_fields if report.get(name, 0)}
    for field in (
        "prompt_protocol_ids",
        "gameplay_prompt_hashes",
        "belief_prompt_hashes",
        "parser_prompt_hashes",
    ):
        values = report.get(field, [])
        if len(values) != 1:
            failures[field] = values
    speech_count = report.get("speech_event_count", 0)
    parser_calls = report.get("parser_call_count", 0)
    parser_outcomes = sum(
        report.get(field, 0)
        for field in ("parser_success_count", "parser_empty_count", "parser_failure_count")
    )
    if speech_count != parser_calls:
        failures["speech_event_count/parser_call_count"] = [
            speech_count,
            parser_calls,
        ]
    if parser_calls != parser_outcomes:
        failures["parser_call_outcome_invariant"] = [parser_calls, parser_outcomes]
    if speech_count > 0 and report.get("parsed_semantic_event_count", 0) == 0:
        failures["parsed_semantic_event_count"] = 0
    parser_failure_count = report.get("parser_failure_count", 0)
    if parser_calls and parser_failure_count / parser_calls > PARSER_FAILURE_RATE_LIMIT:
        failures["parser_failure_rate"] = parser_failure_count / parser_calls
    if (
        report.get("unique_belief_elicitations", 0)
        and report.get("belief_success_rate", 0.0) < BELIEF_SUCCESS_RATE_MINIMUM
    ):
        failures["belief_success_rate"] = report.get("belief_success_rate")
    if (
        report.get("repair_attempts", 0) > 0
        and report.get("belief_repair_success_rate", 0.0) == 0.0
    ):
        failures["belief_repair_success_rate"] = 0.0
    if failures:
        raise RuntimeError(f"collection audit failed: {failures}")
    return True


def checkpoint_for_event(event):
    kind = event["content"]["kind"]
    if event["visibility"] == "public":
        checkpoint = PUBLIC_CHECKPOINTS.get(kind)
        if checkpoint is not None:
            return checkpoint
        if event["event_family"] == "GAME_EVENT" and kind not in {"SETTING", "OUTCOME"}:
            return f"after_{kind.lower()}"
        return None
    return PRIVATE_FIRST_ORDER_CHECKPOINTS.get(kind)


class ToMCollector:
    """Collect labels at state-changing checkpoints without affecting gameplay."""

    def __init__(
        self,
        *,
        game_id,
        roles,
        guess_provider_for,
        prompt_protocol,
        sample_sink=None,
        failure_sink=None,
    ):
        if len(roles) != 7:
            raise ValueError("roles must contain exactly seven entries")
        self.game_id = str(game_id)
        self.roles = tuple(roles)
        self.guess_provider_for = guess_provider_for
        validate_prompt_protocol(prompt_protocol)
        self.prompt_protocol = deepcopy(prompt_protocol)
        self.sample_sink = sample_sink
        self.failure_sink = failure_sink
        self._sample_ids = set()
        self._first_order_keys = set()
        self._second_order_keys = set()

    def collect(self, *, trigger_event, events, alive_players):
        checkpoint = checkpoint_for_event(trigger_event)
        if checkpoint is None:
            return CollectionBatch(samples=(), failures=())
        if trigger_event["visibility"] == "public":
            return self._collect_public(
                trigger_event=trigger_event,
                checkpoint=checkpoint,
                events=events,
                alive_players=alive_players,
            )
        return self._collect_private(
            trigger_event=trigger_event,
            checkpoint=checkpoint,
            events=events,
            alive_players=alive_players,
        )

    def _target_guess(self, target_id, events):
        knowledge = knowledge_for_player(events, target_id)
        role = knowledge["role"]
        if role is None:
            raise ValueError(f"player {target_id} has no SELF_ROLE private fact")
        mask = first_order_knowledge_mask(
            observer_id=target_id,
            observer_role=role,
            known_wolves=knowledge["known_wolves"],
            known_good=knowledge["known_good"],
        )
        constraints = first_order_constraints(
            observer_id=target_id,
            observer_role=role,
            known_wolves=knowledge["known_wolves"],
            known_good=knowledge["known_good"],
        )
        view = render_stream(visible_events(events, target_id))
        return self.guess_provider_for(target_id).elicit(
            observer_id=target_id,
            player_view=view,
            output_mask=mask,
            required_wolves=constraints["required_wolves"],
            forbidden_wolves=constraints["forbidden_wolves"],
        ), mask

    def _base(self, trigger_event, checkpoint):
        state_id = f"{self.game_id}:{trigger_event['event_id']}"
        checkpoint_scope = trigger_event["visibility"]
        return {
            "game_id": self.game_id,
            "checkpoint": checkpoint,
            "checkpoint_scope": checkpoint_scope,
            "state_id": state_id,
            "public_state_id": state_id if checkpoint_scope == "public" else None,
            "day": trigger_event["day"],
            "phase": trigger_event["phase"],
            "turn": trigger_event["turn"],
        }

    def _sample(self, *, suffix, trigger_event, checkpoint, guess, **kwargs):
        base = self._base(trigger_event, checkpoint)
        sample = make_sample(
            sample_id=f"{base['state_id']}:{suffix}",
            guess=guess,
            prompt_protocol=self.prompt_protocol,
            **base,
            **kwargs,
        )
        self._register(sample)
        sink = self.sample_sink if guess.status == "ok" else self.failure_sink
        if sink is not None:
            sink.append(sample)
        return sample

    def _register(self, sample):
        sample_id = sample["sample_id"]
        if sample_id in self._sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        self._sample_ids.add(sample_id)
        if sample["task"] == "first_order":
            key = first_order_key(sample)
            if key in self._first_order_keys:
                raise ValueError(f"duplicate first-order key: {key}")
            self._first_order_keys.add(key)
            return
        key = second_order_key(sample)
        if key in self._second_order_keys:
            raise ValueError(f"duplicate second-order key: {key}")
        self._second_order_keys.add(key)

    def _collect_public(self, *, trigger_event, checkpoint, events, alive_players):
        alive = set(alive_players)
        nonwolf_targets = [
            player_id
            for player_id in range(1, 8)
            if player_id in alive and self.roles[player_id - 1] != "Werewolf"
        ]
        wolf_modelers = [
            player_id
            for player_id, role in enumerate(self.roles, start=1)
            if role == "Werewolf"
        ]
        samples = []
        failures = []
        for target_id in nonwolf_targets:
            guess, first_mask = self._target_guess(target_id, events)
            first_sample = self._sample(
                suffix=f"first:{target_id}",
                trigger_event=trigger_event,
                checkpoint=checkpoint,
                guess=guess,
                task="first_order",
                mode=FIRST_ORDER_MODE,
                observer_id=target_id,
                modeler_id=None,
                target_id=None,
                events=visible_events(events, target_id),
                output_mask=first_mask,
            )
            target_records = [
                first_sample,
                self._sample(
                    suffix=f"second:public:{target_id}",
                    trigger_event=trigger_event,
                    checkpoint=checkpoint,
                    guess=guess,
                    task="second_order",
                    mode="public_only",
                    observer_id=None,
                    modeler_id=None,
                    target_id=target_id,
                    source_first_order_sample_id=first_sample["sample_id"],
                    events=public_events(events),
                    output_mask=second_order_output_mask(
                        mode="public_only", target_id=target_id
                    ),
                ),
            ]
            for modeler_id in wolf_modelers:
                target_records.append(
                    self._sample(
                        suffix=f"second:wolf:{modeler_id}:{target_id}",
                        trigger_event=trigger_event,
                        checkpoint=checkpoint,
                        guess=guess,
                        task="second_order",
                        mode="wolf_conditioned",
                        observer_id=None,
                        modeler_id=modeler_id,
                        target_id=target_id,
                        source_first_order_sample_id=first_sample["sample_id"],
                        events=visible_events(events, modeler_id),
                        output_mask=second_order_output_mask(
                            mode="wolf_conditioned", target_id=target_id
                        ),
                    )
                )
            destination = samples if guess.status == "ok" else failures
            destination.extend(target_records)
        return CollectionBatch(samples=tuple(samples), failures=tuple(failures))

    def _collect_private(self, *, trigger_event, checkpoint, events, alive_players):
        viewers = [
            player_id
            for player_id in trigger_event["visible_to"]
            if player_id in set(alive_players)
            and self.roles[player_id - 1] != "Werewolf"
        ]
        samples = []
        failures = []
        for observer_id in viewers:
            guess, mask = self._target_guess(observer_id, events)
            sample = self._sample(
                suffix=f"first:{observer_id}",
                trigger_event=trigger_event,
                checkpoint=checkpoint,
                guess=guess,
                task="first_order",
                mode=FIRST_ORDER_MODE,
                observer_id=observer_id,
                modeler_id=None,
                target_id=None,
                events=visible_events(events, observer_id),
                output_mask=mask,
            )
            (samples if guess.status == "ok" else failures).append(sample)
        return CollectionBatch(samples=tuple(samples), failures=tuple(failures))
