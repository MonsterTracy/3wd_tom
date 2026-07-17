import json
import random
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from werewolf.envs.werewolf_text_env_v0 import WerewolfTextEnvV0
from werewolf.events.encoder import EVENT_TOKEN_FIELDS, KIND2ID, VALUE2ID, encode_event
from werewolf.events.environment_events import (
    check_result_event,
    self_role_event,
    setting_event,
    speech_event,
    wolf_team_event,
)
from werewolf.events.speech_parser import SpeechEventParser
from werewolf.events.streams import public_events, visible_events
from werewolf.game_rules import (
    LEGAL_TARGET_RULES,
    NUM_PLAYERS,
    NUM_WEREWOLVES,
    PASS_RULES,
    PHASE_ORDER,
    ROLE_DISTRIBUTIONS,
    TIE_RULES,
    VICTORY_RULES,
    canonical_ruleset_metadata,
    ruleset_sha256,
    validate_role_distribution,
)
from werewolf.prompt_protocol import build_prompt_protocol, protocol_id_from_references
from werewolf.tom.collection import (
    JsonlSink,
    ToMCollector,
    assert_audit_passes,
    build_audit_report,
)
from werewolf.tom.guess_provider import GuessResult
from werewolf.tom.features import collate_features, sample_to_features
from werewolf.tom.losses import masked_pair_cross_entropy
from werewolf.tom.model import ToMModel, ToMModelConfig
from werewolf.tom.pair_space import WOLF_PAIRS
from werewolf.tom.schemas import validate_sample_collection


ROLES = ["Werewolf", "Werewolf", "Seer", "Witch", "Villager", "Villager", "Villager"]
FIXTURE = Path("tests/fixtures/tom_v1.jsonl")
TEST_PROMPT_PROTOCOL = build_prompt_protocol(
    {
        "gameplay_profiles": {
            "fake": {"backend": "fake", "model": "fake", "temperature": 0.7}
        },
        "belief_profiles": {
            "fake": {"backend": "fake", "model": "fake", "temperature": 0.0}
        },
        "parser": {"backend": "fake", "model": "fake-parser", "temperature": 0.0},
    }
)


def _audit_report_with_parser_outcomes(*, calls, failures):
    records = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
    ]
    for index in range(calls):
        status = "failed" if index < failures else "empty"
        parser_result = {
            "version": TEST_PROMPT_PROTOCOL["parser"]["version"],
            "sha256": TEST_PROMPT_PROTOCOL["parser"]["sha256"],
            "model": "fake-parser",
            "temperature": 0.0,
            "status": status,
            "attempts": 1,
            "error_code": "schema_validation_error" if status == "failed" else None,
            "error": "fake schema validation failure" if status == "failed" else None,
        }
        speech = speech_event(
            event_id=f"fixture.parser-rate.{index}",
            utterance_id=f"fixture.parser-rate.{index}",
            day=records[0]["day"],
            phase=records[0]["phase"],
            turn=min(record["turn"] for record in records),
            speaker=3,
            target=3,
            value=None,
            source_span="",
            metadata={"parser_result": parser_result},
        )
        for record in records:
            record["events"].append(deepcopy(speech))
    return build_audit_report(records)


class EmptyParserBackend:
    def chat(self, messages, **kwargs):
        return '{"events":[]}'


class ValidGuessProvider:
    def elicit(
        self, *, player_view, output_mask, observer_id,
        required_wolves, forbidden_wolves
    ):
        pair = next(pair for pair, allowed in zip(WOLF_PAIRS, output_mask) if allowed)
        return GuessResult(
            status="ok", pair=pair, raw_text=(f'{{"wolf_pair":{list(pair)}}}',),
            error=None, attempts=1, model="fake",
            first_error_code=None, final_error_code=None,
            required_wolves=tuple(required_wolves),
            forbidden_wolves=tuple(forbidden_wolves),
        )


def _rollout(environment):
    rng = random.Random(11)
    observation = environment.reset(roles=ROLES)
    for _ in range(300):
        if "speech" in observation["phase"]:
            action = (environment.phase, f"player {observation['player_id']} statement")
        else:
            action = rng.choice(observation["valid_actions"])
        observation, _, done, info = environment.step(action)
        if done:
            return info
    raise AssertionError("game did not finish")


def test_full_game_uses_unified_events_and_checkpoint_collection():
    collector = ToMCollector(
        game_id="integration", roles=ROLES,
        guess_provider_for=lambda player_id: ValidGuessProvider(),
        prompt_protocol=TEST_PROMPT_PROTOCOL,
    )
    environment = WerewolfTextEnvV0(
        random_seed=11,
        speech_parser=SpeechEventParser(EmptyParserBackend(), "fake-parser"),
        tom_collector=collector,
    )
    info = _rollout(environment)
    assert info["Werewolf"] in (-1, 1)
    assert all(event["event_family"] in {
        "BELIEF_ASSERTION", "SOCIAL_STANCE", "ACTION_POSITION", "CLAIM_RESPONSE",
        "GAME_EVENT", "PRIVATE_FACT"
    } for event in environment.events)
    assert all(
        event["content"]["kind"] != "CHECK_RESULT"
        or event["content"]["value"] in (None, "Werewolf", "Village")
        for event in environment.events
    )
    kind_index = EVENT_TOKEN_FIELDS.index("kind_id")
    value_index = EVENT_TOKEN_FIELDS.index("value_id")
    assert all(
        token[kind_index] != KIND2ID["UNKNOWN"]
        and token[value_index] != VALUE2ID["UNKNOWN"]
        for event in environment.events
        for token in encode_event(event)
    )

    samples = [sample for batch in environment.collection_batches for sample in batch.samples]
    assert samples
    assert all(
        not (sample["task"] == "first_order" and ROLES[sample["observer_id"] - 1] == "Werewolf")
        for sample in samples
    )
    assert all(
        sum(sample["output_mask"]) == 21
        for sample in samples
        if sample["task"] == "second_order" and sample["mode"] == "public_only"
    )
    assert all(
        sum(sample["output_mask"]) == 15
        for sample in samples
        if sample["task"] == "second_order" and sample["mode"] == "wolf_conditioned"
    )
    second_order_smoke = [
        next(sample for sample in samples if sample["mode"] == mode)
        for mode in ("public_only", "wolf_conditioned")
    ]
    batch = collate_features([sample_to_features(sample) for sample in second_order_smoke])
    model = ToMModel(
        ToMModelConfig(
            architecture="boe_mlp", d_model=8, num_layers=1,
            num_heads=2, max_events=512
        )
    )
    logits = model(batch)
    loss = masked_pair_cross_entropy(logits, batch["labels"], batch["output_mask"])
    assert logits.shape == (2, 21)
    assert torch.isfinite(loss)


def test_guess_failure_does_not_stop_the_game_or_enter_samples(tmp_path):
    class FailureProvider:
        def elicit(self, **kwargs):
            return GuessResult(
                status="failed", pair=None, raw_text=("bad", "bad"),
                error="invalid", attempts=2, model="fake",
                first_error_code="invalid_json", final_error_code="invalid_json",
                required_wolves=tuple(kwargs["required_wolves"]),
                forbidden_wolves=tuple(kwargs["forbidden_wolves"]),
            )

    collector = ToMCollector(
        game_id="failures", roles=ROLES,
        guess_provider_for=lambda player_id: FailureProvider(),
        prompt_protocol=TEST_PROMPT_PROTOCOL,
        failure_sink=JsonlSink(tmp_path / "failures.jsonl"),
    )
    environment = WerewolfTextEnvV0(random_seed=4, tom_collector=collector)
    _rollout(environment)
    assert any(batch.failures for batch in environment.collection_batches)
    assert not any(batch.samples for batch in environment.collection_batches)
    assert all(
        sample["prompt_protocol"] == TEST_PROMPT_PROTOCOL
        for batch in environment.collection_batches
        for sample in batch.failures
    )
    failure_records = [
        json.loads(line)
        for line in (tmp_path / "failures.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert failure_records
    assert all(
        record["prompt_protocol"] == TEST_PROMPT_PROTOCOL
        for record in failure_records
    )


def _event(builder, event_id, turn, **kwargs):
    return builder(
        event_id=event_id,
        day=1,
        phase="1_day_speech",
        turn=turn,
        **kwargs,
    )


def test_public_checkpoint_reuses_one_first_order_guess_with_explicit_source():
    events = [
        _event(setting_event, "e1", 1, value=None),
        _event(self_role_event, "e2", 2, visible_to=[1], target=1, value="Werewolf"),
        _event(self_role_event, "e3", 3, visible_to=[2], target=2, value="Werewolf"),
        _event(self_role_event, "e4", 4, visible_to=[3], target=3, value="Seer"),
        _event(
            wolf_team_event,
            "e5",
            5,
            visible_to=[1, 2],
            target=[1, 2],
            value=None,
        ),
        _event(
            speech_event,
            "e6",
            6,
            speaker=3,
            target=3,
            value=None,
            source_span="statement",
        ),
    ]
    collector = ToMCollector(
        game_id="source",
        roles=ROLES,
        guess_provider_for=lambda player_id: ValidGuessProvider(),
        prompt_protocol=TEST_PROMPT_PROTOCOL,
    )
    batch = collector.collect(
        trigger_event=events[-1], events=events, alive_players=[3]
    )

    assert len(batch.samples) == 4
    assert len({sample["sample_id"] for sample in batch.samples}) == 4
    first = next(sample for sample in batch.samples if sample["task"] == "first_order")
    seconds = [sample for sample in batch.samples if sample["task"] == "second_order"]
    assert first["checkpoint_scope"] == "public"
    assert first["public_state_id"] == first["state_id"]
    assert all(sample["source_first_order_sample_id"] == first["sample_id"] for sample in seconds)
    assert all(sample["public_state_id"] == first["public_state_id"] for sample in seconds)
    assert all(sample["target_id"] == first["observer_id"] for sample in seconds)
    assert all(sample["label_pair"] == first["label_pair"] for sample in seconds)
    assert all(sample["label_index"] == first["label_index"] for sample in seconds)
    assert validate_sample_collection(batch.samples)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        collector.collect(trigger_event=events[-1], events=events, alive_players=[3])


def test_private_checkpoint_never_creates_second_order_sample():
    events = [
        _event(setting_event, "e1", 1, value=None),
        _event(self_role_event, "e2", 2, visible_to=[3], target=3, value="Seer"),
        _event(
            check_result_event,
            "e3",
            3,
            visible_to=[3],
            speaker=3,
            target=1,
            value="Werewolf",
        ),
    ]
    collector = ToMCollector(
        game_id="private",
        roles=ROLES,
        guess_provider_for=lambda player_id: ValidGuessProvider(),
        prompt_protocol=TEST_PROMPT_PROTOCOL,
    )
    batch = collector.collect(
        trigger_event=events[-1], events=events, alive_players=[3]
    )

    assert len(batch.samples) == 1
    assert batch.samples[0]["task"] == "first_order"
    assert batch.samples[0]["checkpoint_scope"] == "private"
    assert batch.samples[0]["public_state_id"] is None
    assert batch.samples[0]["source_first_order_sample_id"] is None


def test_collection_audit_reports_required_counts_and_fatal_gates():
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    report = build_audit_report(records, game_ids=["fixture"])
    required = {
        "schema_version", "games", "public_checkpoints", "private_checkpoints",
        "collection_status", "completed_games", "runtime_failure_count",
        "failed_game_id", "runtime_error_type", "runtime_error_message",
        "unique_belief_elicitations", "successful_guesses", "failed_guesses",
        "repair_attempts", "first_order_samples", "second_order_public_samples",
        "belief_failure_reason_distribution", "belief_first_attempt_successes",
        "belief_repair_successes", "belief_repair_failures",
        "belief_backend_failure_count", "belief_backend_retry_attempts",
        "belief_backend_retry_successes", "belief_backend_retry_failures",
        "belief_contains_observer_failures",
        "belief_missing_required_wolf_failures",
        "belief_contains_forbidden_player_failures",
        "belief_invalid_format_failures", "belief_success_rate",
        "belief_repair_success_rate", "belief_success_constraint_violations",
        "speech_event_count", "parser_call_count", "parser_success_count",
        "parser_empty_count", "parser_failure_count", "parser_failure_rate",
        "parser_repair_attempts",
        "parsed_semantic_event_count", "speech_with_semantic_events",
        "speech_without_semantic_events", "parsed_event_family_distribution",
        "parser_failure_reason_distribution", "missing_parser_metadata_count",
        "parser_utterance_mismatch_count",
        "second_order_wolf_samples", "duplicate_sample_ids",
        "duplicate_first_order_keys", "duplicate_second_order_keys",
        "state_alignment_errors", "private_to_second_order_errors",
        "mask_size_distribution", "pair_label_distribution",
        "event_family_distribution", "unknown_token_count", "unknown_token_ratio",
        "unknown_kind_count", "unknown_value_count", "not_applicable_value_count",
        "unknown_by_event_family", "unknown_by_content_kind",
        "top_unknown_raw_values",
        "sequence_length", "truncated_samples", "samples_by_day", "samples_by_phase",
        "first_order_by_observer_role",
        "prompt_protocol_ids", "prompt_protocol_distribution",
        "gameplay_prompt_versions", "belief_prompt_versions",
        "parser_prompt_versions", "gameplay_prompt_hashes",
        "belief_prompt_hashes", "parser_prompt_hashes",
        "runtime_model_distribution", "missing_prompt_protocol_count",
        "invalid_prompt_protocol_count",
        "ruleset_ids", "ruleset_versions", "ruleset_hashes",
        "missing_ruleset_count", "invalid_ruleset_count",
    }
    assert required <= set(report)
    assert report["schema_version"] == "tom.audit.v1_4"
    assert report["collection_status"] == "complete"
    assert report["completed_games"] == 1
    assert report["runtime_failure_count"] == 0
    assert report["failed_game_id"] is None
    assert report["runtime_error_type"] is None
    assert report["runtime_error_message"] is None
    assert report["games"] == 1
    assert report["public_checkpoints"] == 1
    assert report["unique_belief_elicitations"] == 1
    assert report["successful_guesses"] == 1
    assert report["belief_first_attempt_successes"] == 1
    assert report["belief_backend_failure_count"] == 0
    assert report["belief_backend_retry_attempts"] == 0
    assert report["belief_backend_retry_successes"] == 0
    assert report["belief_backend_retry_failures"] == 0
    assert report["belief_success_rate"] == 1.0
    assert report["belief_repair_success_rate"] == 1.0
    assert report["first_order_samples"] == 1
    assert report["second_order_public_samples"] == 1
    assert report["unknown_kind_count"] == 0
    assert report["unknown_value_count"] == 0
    assert report["unknown_token_count"] == 0
    assert report["unknown_token_ratio"] == 0.0
    assert report["not_applicable_value_count"] > 0
    assert report["top_unknown_raw_values"] == []
    assert report["prompt_protocol_ids"] == [
        records[0]["prompt_protocol"]["protocol_id"]
    ]
    assert report["prompt_protocol_distribution"] == {
        records[0]["prompt_protocol"]["protocol_id"]: 2
    }
    assert report["gameplay_prompt_versions"] == ["gameplay.zh.v4"]
    assert report["belief_prompt_versions"] == ["belief.zh.v3"]
    assert report["parser_prompt_versions"] == ["parser.zh.v3"]
    assert report["ruleset_ids"] == ["werewolf_7p"]
    assert report["ruleset_versions"] == ["werewolf_7p.zh.v1"]
    assert report["ruleset_hashes"] == [
        records[0]["prompt_protocol"]["ruleset"]["sha256"]
    ]
    assert report["missing_ruleset_count"] == 0
    assert report["invalid_ruleset_count"] == 0
    assert report["missing_prompt_protocol_count"] == 0
    assert report["invalid_prompt_protocol_count"] == 0
    assert report["speech_event_count"] == 0
    assert report["parser_call_count"] == 0
    assert report["parser_failure_rate"] is None
    assert assert_audit_passes(report)

    failed_report = build_audit_report(
        records,
        game_ids=["fixture"],
        collection_status="failed",
        completed_games=0,
        failed_game_id="fixture",
        runtime_error_type="RuntimeError",
        runtime_error_message="gameplay action generation failed: invalid_json",
    )
    assert failed_report["runtime_failure_count"] == 1
    with pytest.raises(RuntimeError, match="collection_status"):
        assert_audit_passes(failed_report)

    duplicate_report = build_audit_report(records + [deepcopy(records[0])])
    assert duplicate_report["duplicate_sample_ids"] == 1
    assert duplicate_report["duplicate_first_order_keys"] == 1
    with pytest.raises(RuntimeError, match="duplicate_sample_ids"):
        assert_audit_passes(duplicate_report)

    duplicate_second = deepcopy(records[1])
    duplicate_second["sample_id"] = "fixture:second:duplicate-key"
    duplicate_second_report = build_audit_report(records + [duplicate_second])
    assert duplicate_second_report["duplicate_second_order_keys"] == 1
    with pytest.raises(RuntimeError, match="duplicate_second_order_keys"):
        assert_audit_passes(duplicate_second_report)

    misaligned = deepcopy(records)
    misaligned[1]["target_id"] = 4
    misaligned_report = build_audit_report(misaligned)
    assert misaligned_report["state_alignment_errors"] == 1
    with pytest.raises(RuntimeError, match="state_alignment_errors"):
        assert_audit_passes(misaligned_report)

    masked_label = deepcopy(records)
    masked_label[1]["output_mask"][masked_label[1]["label_index"]] = False
    masked_label_report = build_audit_report(masked_label)
    assert masked_label_report["labels_outside_mask"] == 1
    with pytest.raises(RuntimeError, match="labels_outside_mask"):
        assert_audit_passes(masked_label_report)

    unknown_kind = deepcopy(records)
    unknown_kind[1]["events"][0]["content"] = {
        "kind": "UNREGISTERED_KIND",
        "value": None,
    }
    unknown_kind_report = build_audit_report(unknown_kind)
    assert unknown_kind_report["unknown_kind_count"] == 1
    assert unknown_kind_report["unknown_token_count"] == 1
    with pytest.raises(RuntimeError, match="unknown_kind_count"):
        assert_audit_passes(unknown_kind_report)

    unknown_value = deepcopy(records)
    unknown_value[0]["events"][1]["content"]["value"] = "Wizard"
    unknown_value_report = build_audit_report(unknown_value)
    assert unknown_value_report["unknown_value_count"] == 1
    assert unknown_value_report["unknown_token_count"] == 1
    assert unknown_value_report["top_unknown_raw_values"] == [
        {"value": "'Wizard'", "count": 1}
    ]
    with pytest.raises(RuntimeError, match="unknown_value_count"):
        assert_audit_passes(unknown_value_report)

    missing_protocol = deepcopy(records)
    missing_protocol[0].pop("prompt_protocol")
    missing_protocol_report = build_audit_report(missing_protocol)
    assert missing_protocol_report["missing_prompt_protocol_count"] == 1
    with pytest.raises(RuntimeError, match="missing_prompt_protocol_count"):
        assert_audit_passes(missing_protocol_report)

    forged_ruleset = deepcopy(records)
    forged_ruleset[0]["prompt_protocol"]["ruleset"]["sha256"] = "0" * 64
    forged_references = {
        name: forged_ruleset[0]["prompt_protocol"][name]
        for name in ("gameplay", "belief", "parser")
    }
    forged_ruleset[0]["prompt_protocol"]["protocol_id"] = (
        protocol_id_from_references(
            forged_references,
            ruleset=forged_ruleset[0]["prompt_protocol"]["ruleset"],
        )
    )
    forged_ruleset_report = build_audit_report(forged_ruleset)
    assert forged_ruleset_report["invalid_ruleset_count"] == 1
    with pytest.raises(RuntimeError, match="invalid_ruleset_count"):
        assert_audit_passes(forged_ruleset_report)

    mixed_protocol = deepcopy(records)
    mixed_protocol[1]["prompt_protocol"]["belief"]["sha256"] = "0" * 64
    references = {
        name: mixed_protocol[1]["prompt_protocol"][name]
        for name in ("gameplay", "belief", "parser")
    }
    mixed_protocol[1]["prompt_protocol"]["protocol_id"] = (
        protocol_id_from_references(references)
    )
    mixed_protocol_report = build_audit_report(mixed_protocol)
    assert len(mixed_protocol_report["prompt_protocol_ids"]) == 2
    assert mixed_protocol_report["invalid_prompt_protocol_count"] == 1
    with pytest.raises(RuntimeError, match="prompt_protocol_ids|invalid_prompt_protocol"):
        assert_audit_passes(mixed_protocol_report)


def test_parser_audit_fails_missing_calls_zero_semantics_and_high_failure_rate():
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    speech = speech_event(
        event_id="fixture.speech",
        utterance_id="fixture.speech",
        day=1,
        phase="1_day_speech",
        turn=3,
        speaker=3,
        target=3,
        value=None,
        source_span="statement",
    )
    records[0]["events"].append(deepcopy(speech))
    records[1]["events"].append(deepcopy(speech))
    report = build_audit_report(records)
    assert report["speech_event_count"] == 1
    assert report["parser_call_count"] == 0
    assert report["parsed_semantic_event_count"] == 0
    assert report["missing_parser_metadata_count"] == 1
    with pytest.raises(RuntimeError, match="parser_call_count|parser_metadata"):
        assert_audit_passes(report)

    rate_report = _audit_report_with_parser_outcomes(calls=5, failures=2)
    assert rate_report["parser_failure_rate"] == 0.4
    with pytest.raises(RuntimeError, match="parser_failure_rate"):
        assert_audit_passes(rate_report)


def test_parser_failure_rate_uses_call_count_and_preserves_json_zero():
    no_failures = _audit_report_with_parser_outcomes(calls=16, failures=0)
    assert no_failures["parser_call_count"] == 16
    assert no_failures["parser_failure_count"] == 0
    assert no_failures["parser_failure_rate"] == 0.0
    assert json.loads(json.dumps(no_failures))["parser_failure_rate"] == 0.0
    assert assert_audit_passes(no_failures)

    some_failures = _audit_report_with_parser_outcomes(calls=16, failures=3)
    assert some_failures["parser_failure_rate"] == 0.1875
    assert assert_audit_passes(some_failures)

    at_limit = _audit_report_with_parser_outcomes(calls=5, failures=1)
    assert at_limit["parser_failure_rate"] == 0.2
    assert assert_audit_passes(at_limit)

    no_calls = _audit_report_with_parser_outcomes(calls=0, failures=0)
    assert no_calls["parser_failure_rate"] is None
    assert assert_audit_passes(no_calls)

    invalid_none = deepcopy(no_failures)
    invalid_none["parser_failure_rate"] = None
    with pytest.raises(RuntimeError, match="parser_failure_rate"):
        assert_audit_passes(invalid_none)

    invalid_zero = deepcopy(no_calls)
    invalid_zero["parser_failure_rate"] = 0.0
    with pytest.raises(RuntimeError, match="parser_failure_rate"):
        assert_audit_passes(invalid_zero)


def test_failed_guess_is_audited_and_fails_the_quality_gate():
    failures = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    for record in failures:
        record["label_pair"] = None
        record["label_index"] = None
        record["guess"].update(
            status="failed", raw_text=["bad", "still bad"], error="invalid",
            attempts=2, first_error_code="invalid_json",
            final_error_code="invalid_json"
        )
    report = build_audit_report([], failures)
    assert report["unique_belief_elicitations"] == 1
    assert report["successful_guesses"] == 0
    assert report["failed_guesses"] == 1
    assert report["repair_attempts"] == 1
    assert report["first_order_samples"] == 0
    assert report["belief_invalid_format_failures"] == 1
    assert report["belief_success_rate"] == 0.0
    assert report["belief_repair_success_rate"] == 0.0
    with pytest.raises(RuntimeError, match="belief_success_rate"):
        assert_audit_passes(report)


def test_belief_audit_separates_backend_retries_and_semantic_repairs():
    original = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])

    retry_success = deepcopy(original)
    retry_success["guess"].update(
        raw_text=['{"wolf_pair":[1,2]}'], attempts=2,
        first_error_code="backend_error", final_error_code=None,
    )
    success_report = build_audit_report([retry_success])
    assert success_report["repair_attempts"] == 0
    assert success_report["belief_repair_successes"] == 0
    assert success_report["belief_backend_retry_attempts"] == 1
    assert success_report["belief_backend_retry_successes"] == 1
    assert success_report["belief_backend_retry_failures"] == 0
    assert success_report["belief_backend_failure_count"] == 0
    assert success_report["belief_repair_success_rate"] == 1.0

    retry_failure = deepcopy(original)
    retry_failure["label_pair"] = None
    retry_failure["label_index"] = None
    retry_failure["guess"].update(
        status="failed", raw_text=[], error="safe backend error", attempts=2,
        first_error_code="backend_error", final_error_code="backend_error",
    )
    failure_report = build_audit_report([], [retry_failure])
    assert failure_report["repair_attempts"] == 0
    assert failure_report["belief_repair_failures"] == 0
    assert failure_report["belief_backend_retry_attempts"] == 1
    assert failure_report["belief_backend_retry_successes"] == 0
    assert failure_report["belief_backend_retry_failures"] == 1
    assert failure_report["belief_backend_failure_count"] == 1
    assert failure_report["belief_invalid_format_failures"] == 0
    with pytest.raises(RuntimeError, match="belief_backend_failure_count"):
        assert_audit_passes(failure_report)

    non_retryable = deepcopy(retry_failure)
    non_retryable["guess"]["attempts"] = 1
    non_retryable_report = build_audit_report([], [non_retryable])
    assert non_retryable_report["belief_backend_failure_count"] == 1
    assert non_retryable_report["belief_backend_retry_attempts"] == 0

    semantic_success = deepcopy(original)
    semantic_success["guess"].update(
        raw_text=["bad", '{"wolf_pair":[1,2]}'], attempts=2,
        first_error_code="invalid_json", final_error_code=None,
    )
    semantic_report = build_audit_report([semantic_success])
    assert semantic_report["repair_attempts"] == 1
    assert semantic_report["belief_repair_successes"] == 1
    assert semantic_report["belief_repair_success_rate"] == 1.0
    assert semantic_report["belief_backend_retry_attempts"] == 0


def test_fixed_seven_player_roles_and_both_win_conditions():
    environment = WerewolfTextEnvV0(random_seed=1)
    environment.reset(roles=ROLES)
    assert len(environment.roles) == 7
    assert environment.roles.count("Werewolf") == 2
    assert environment.roles.count("Seer") == 1
    assert environment.roles.count("Witch") == 1
    assert environment.roles.count("Villager") == 3
    assert environment.events[0]["content"] == {"kind": "SETTING", "value": None}
    assert environment.events[0]["metadata"]["roles"]["Werewolf"] == 2
    with pytest.raises(ValueError, match="supported seven-player variant"):
        environment.reset(
            roles=["Werewolf", "Seer", "Seer", "Witch", "Villager", "Villager", "Villager"]
        )

    environment.reset(roles=ROLES)
    environment.alive = {3, 4, 5}
    village_reward, village_done, village_info = environment._is_done()
    assert village_done and village_info == {"Werewolf": -1}
    assert village_reward[2] == environment.village_reward

    environment.alive = {1, 2, 3, 4}
    wolf_reward, wolf_done, wolf_info = environment._is_done()
    assert wolf_done and wolf_info == {"Werewolf": 1}
    assert wolf_reward[0] == environment.werewolf_reward


def test_canonical_ruleset_matches_environment_counts_actions_and_outcomes():
    metadata = canonical_ruleset_metadata()
    assert metadata == canonical_ruleset_metadata()
    assert metadata["sha256"] == ruleset_sha256()
    assert len(metadata["sha256"]) == 64
    assert NUM_PLAYERS == 7
    assert NUM_WEREWOLVES == 2
    assert ROLE_DISTRIBUTIONS["seer_witch"] == {
        "Werewolf": 2, "Seer": 1, "Witch": 1, "Guard": 0,
        "Villager": 3,
    }
    assert ROLE_DISTRIBUTIONS["seer_guard"] == {
        "Werewolf": 2, "Seer": 1, "Witch": 0, "Guard": 1,
        "Villager": 3,
    }
    assert validate_role_distribution(ROLES, "seer_witch")
    assert PHASE_ORDER["seer_witch"][:3] == (
        "skill_wolf", "skill_seer", "skill_witch"
    )
    assert PHASE_ORDER["seer_guard"][:3] == (
        "skill_wolf", "skill_seer", "skill_guard"
    )

    environment = WerewolfTextEnvV0(random_seed=13)
    observation = environment.reset(roles=ROLES)
    assert environment.n_player == NUM_PLAYERS
    assert environment.n_werewolf == NUM_WEREWOLVES
    assert ("kill", 0) in observation["valid_actions"]
    assert PASS_RULES["skill_wolf"]
    assert {target for action, target in observation["valid_actions"] if target} == {
        3, 4, 5, 6, 7
    }
    assert "存活的非狼人" in LEGAL_TARGET_RULES["skill_wolf"]

    environment.step(("kill", 0))
    observation, _, _, _ = environment.step(("kill", 0))
    assert ("check", 0) in observation["valid_actions"]
    assert all(target != 3 for _, target in observation["valid_actions"])
    assert PASS_RULES["skill_seer"]
    assert "PK" in TIE_RULES["first_vote"]
    assert "无人被放逐" in TIE_RULES["pk_vote"]
    assert "存活狼人数量变为零" in VICTORY_RULES["Village"]
    assert "存活普通村民数量变为零" in VICTORY_RULES["Werewolf"]


def test_night_order_seer_witch_death_and_private_visibility():
    environment = WerewolfTextEnvV0(random_seed=2)
    observation = environment.reset(roles=ROLES)
    assert (environment.phase, observation["player_id"]) == ("skill_wolf", 1)
    observation, _, done, _ = environment.step(("kill", 5))
    assert not done and (environment.phase, observation["player_id"]) == ("skill_wolf", 2)
    observation, _, done, _ = environment.step(("kill", 5))
    assert not done and (environment.phase, observation["player_id"]) == ("skill_seer", 3)
    observation, _, done, _ = environment.step(("check", 1))
    assert not done and (environment.phase, observation["player_id"]) == ("skill_witch", 4)
    witch_state = next(
        event
        for event in observation["events"]
        if event["content"]["kind"] == "WITCH_STATE"
    )
    assert witch_state["content"]["value"] == "HEAL_AND_POISON_AVAILABLE"
    assert ("witch_heal", 5) in observation["valid_actions"]
    assert any(action[0] == "witch_poison" for action in observation["valid_actions"])
    observation, _, done, _ = environment.step(("witch_heal", 5))
    assert not done and environment.phase == "speech"
    assert 5 in environment.alive
    assert environment.events[-1]["content"]["kind"] == "DEATH"
    assert environment.events[-1]["target"] == []
    assert environment.events[-1]["content"]["value"] is None

    seer_view = visible_events(environment.events, 3)
    check = next(event for event in seer_view if event["content"]["kind"] == "CHECK_RESULT")
    assert check["target"] == [1]
    assert check["content"]["value"] == "Werewolf"
    assert check["speaker"] == 3
    assert check["visibility"] == "private"
    assert check["visible_to"] == [3]
    assert not any(
        event["content"]["kind"] == "CHECK_RESULT"
        for event in public_events(environment.events)
    )
    wolf_view = visible_events(environment.events, 1)
    nonwolf_view = visible_events(environment.events, 5)
    assert any(event["content"]["kind"] == "WOLF_TEAM" for event in wolf_view)
    assert any(
        event["content"] == {"kind": "PRIVATE_ACTION_RESULT", "value": "KILL"}
        for event in wolf_view
    )
    assert not any(
        event["content"]["kind"] in {"WOLF_TEAM", "PRIVATE_ACTION_RESULT"}
        for event in nonwolf_view
    )
    assert not any(event["content"]["kind"] == "CHECK_RESULT" for event in nonwolf_view)


def test_seer_village_check_has_canonical_target_value_and_private_visibility():
    environment = WerewolfTextEnvV0(random_seed=2)
    environment.reset(roles=ROLES)
    environment.step(("kill", 5))
    environment.step(("kill", 5))
    environment.step(("check", 5))

    check = next(
        event for event in environment.events
        if event["content"]["kind"] == "CHECK_RESULT"
    )
    assert check["target"] == [5]
    assert check["content"] == {"kind": "CHECK_RESULT", "value": "Village"}
    assert check["speaker"] == 3
    assert check["visible_to"] == [3]
    assert all(
        not any(event["content"]["kind"] == "CHECK_RESULT" for event in visible_events(environment.events, player_id))
        for player_id in (1, 2, 4, 5, 6, 7)
    )


def test_seer_pass_witch_self_heal_single_action_and_guard_target_rules():
    witch_environment = WerewolfTextEnvV0(random_seed=7)
    witch_environment.reset(roles=ROLES)
    witch_environment.step(("kill", 4))
    witch_environment.step(("kill", 4))
    witch_environment.step(("check", 0))
    assert not any(
        event["content"]["kind"] == "CHECK_RESULT"
        for event in witch_environment.events
    )
    actions = witch_environment.valid_actions()
    assert ("witch_heal", 4) in actions
    assert any(action == "witch_poison" for action, _ in actions)
    witch_environment.step(("witch_heal", 4))
    assert witch_environment.phase == "speech"
    assert 4 in witch_environment.alive

    guard_roles = [
        "Werewolf", "Werewolf", "Seer", "Guard",
        "Villager", "Villager", "Villager",
    ]
    guard_environment = WerewolfTextEnvV0(
        random_seed=8, n_witch=0, n_guard=1
    )
    guard_environment.reset(roles=guard_roles)
    guard_environment.step(("kill", 0))
    guard_environment.step(("kill", 0))
    guard_environment.step(("check", 0))
    assert ("guard", 4) in guard_environment.valid_actions()
    guard_environment.guard_history = [4]
    assert ("guard", 4) not in guard_environment.valid_actions()
    assert ("guard", 0) in guard_environment.valid_actions()


def test_guard_variant_protects_at_night_and_records_private_result():
    guard_roles = [
        "Werewolf", "Werewolf", "Seer", "Guard", "Villager", "Villager", "Villager"
    ]
    environment = WerewolfTextEnvV0(
        random_seed=3, n_witch=0, n_guard=1
    )
    environment.reset(roles=guard_roles)
    environment.step(("kill", 5))
    environment.step(("kill", 5))
    observation, _, _, _ = environment.step(("check", 1))
    assert (environment.phase, observation["player_id"]) == ("skill_guard", 4)
    observation, _, done, _ = environment.step(("guard", 5))
    assert not done and environment.phase == "speech"
    assert 5 in environment.alive
    guard_view = visible_events(environment.events, 4)
    guard_result = next(
        event for event in guard_view if event["content"]["kind"] == "GUARD_RESULT"
    )
    assert guard_result["content"]["value"] is None
    assert not any(
        event["content"]["kind"] == "GUARD_RESULT"
        for event in visible_events(environment.events, 5)
    )


def _reach_vote(environment):
    environment.reset(roles=ROLES)
    environment.step(("kill", 0))
    environment.step(("kill", 0))
    environment.step(("check", 0))
    environment.step(("witch_pass", 0))
    while environment.phase == "speech":
        environment.step(("speech", "statement"))
    assert environment.phase == "vote"


def test_vote_exile_and_tie_enter_the_expected_public_phases():
    exile_environment = WerewolfTextEnvV0(random_seed=4)
    _reach_vote(exile_environment)
    while exile_environment.phase == "vote":
        exile_environment.step(("vote", 1))
    assert 1 not in exile_environment.alive
    kinds = [event["content"]["kind"] for event in exile_environment.events]
    assert kinds.count("VOTE_CAST") == 7
    assert "VOTE_RESULT" in kinds
    assert "EXILE" in kinds
    assert "ROLE_REVEAL" in kinds

    tie_environment = WerewolfTextEnvV0(random_seed=5)
    _reach_vote(tie_environment)
    for target in (1, 1, 1, 2, 2, 2, 0):
        tie_environment.step(("vote", target))
    assert tie_environment.phase == "speech_pk"
    assert tie_environment.vote_pk_players == [1, 2]
    result = next(
        event
        for event in reversed(tie_environment.events)
        if event["content"]["kind"] == "VOTE_RESULT"
    )
    assert result["target"] == [1, 2]
    assert result["content"]["value"] is None
    assert result["metadata"]["counts"] == {"1": 3, "2": 3}
