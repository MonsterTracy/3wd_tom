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
from werewolf.events.streams import visible_events
from werewolf.tom.collection import (
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


class EmptyParserBackend:
    def chat(self, messages, **kwargs):
        return '{"events":[]}'


class ValidGuessProvider:
    def elicit(self, *, player_view, output_mask):
        pair = next(pair for pair, allowed in zip(WOLF_PAIRS, output_mask) if allowed)
        return GuessResult(
            status="ok", pair=pair, raw_text=(f'{{"wolf_pair":{list(pair)}}}',),
            error=None, attempts=1, model="fake"
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
        guess_provider_for=lambda player_id: ValidGuessProvider()
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


def test_guess_failure_does_not_stop_the_game_or_enter_samples():
    class FailureProvider:
        def elicit(self, **kwargs):
            return GuessResult(
                status="failed", pair=None, raw_text=("bad", "bad"),
                error="invalid", attempts=2, model="fake"
            )

    collector = ToMCollector(
        game_id="failures", roles=ROLES,
        guess_provider_for=lambda player_id: FailureProvider()
    )
    environment = WerewolfTextEnvV0(random_seed=4, tom_collector=collector)
    _rollout(environment)
    assert any(batch.failures for batch in environment.collection_batches)
    assert not any(batch.samples for batch in environment.collection_batches)


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
        "unique_belief_elicitations", "successful_guesses", "failed_guesses",
        "repair_attempts", "first_order_samples", "second_order_public_samples",
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
    }
    assert required <= set(report)
    assert report["games"] == 1
    assert report["public_checkpoints"] == 1
    assert report["unique_belief_elicitations"] == 1
    assert report["successful_guesses"] == 1
    assert report["first_order_samples"] == 1
    assert report["second_order_public_samples"] == 1
    assert report["unknown_kind_count"] == 0
    assert report["unknown_value_count"] == 0
    assert report["unknown_token_count"] == 0
    assert report["unknown_token_ratio"] == 0.0
    assert report["not_applicable_value_count"] > 0
    assert report["top_unknown_raw_values"] == []
    assert assert_audit_passes(report)

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


def test_failed_guess_is_audited_but_is_not_a_fatal_gate():
    failures = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    for record in failures:
        record["label_pair"] = None
        record["label_index"] = None
        record["guess"].update(
            status="failed", raw_text=["bad", "still bad"], error="invalid", attempts=2
        )
    report = build_audit_report([], failures)
    assert report["unique_belief_elicitations"] == 1
    assert report["successful_guesses"] == 0
    assert report["failed_guesses"] == 1
    assert report["repair_attempts"] == 1
    assert report["first_order_samples"] == 0
    assert assert_audit_passes(report)


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
    with pytest.raises(ValueError, match="two wolves"):
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
    assert check["content"]["value"] == "Werewolf"
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
