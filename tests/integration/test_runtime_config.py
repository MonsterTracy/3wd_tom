import json
import re
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pytest
import yaml

from script.tom.collect import collect_from_config, preflight_collection
from werewolf.prompt_protocol import (
    BELIEF_PROMPT_SPEC,
    GAMEPLAY_PROMPT_SPEC,
    PARSER_PROMPT_SPEC,
    protocol_id_from_references,
)
from werewolf.runtime_config import validate_runtime_config


def _config():
    return yaml.safe_load(Path("configs/tom/collect.yaml").read_text(encoding="utf-8"))


def test_canonical_collection_config_is_strict():
    config = _config()
    assert validate_runtime_config(config)
    assert config["backends"]["deepseek"] == {
        "type": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    }
    assert config["parser"] == {"backend": "deepseek", "model": "deepseek-chat"}
    assert config["guess"] == {"backend": "deepseek", "model": "deepseek-chat"}
    legacy = {"backend": {}, "agent_config": {}, "env_config": {}}
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_runtime_config(legacy)


def test_collection_preflight_rejects_missing_key_bad_backend_and_parser_model():
    config = _config()
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        preflight_collection(config, env={})

    bad_backend = deepcopy(config)
    bad_backend["parser"]["backend"] = "missing"
    with pytest.raises(ValueError, match="parser backend"):
        preflight_collection(bad_backend, env={"DEEPSEEK_API_KEY": "fake"})

    missing_model = deepcopy(config)
    missing_model["parser"]["model"] = ""
    with pytest.raises(ValueError, match="parser.model"):
        preflight_collection(missing_model, env={"DEEPSEEK_API_KEY": "fake"})

    missing_gameplay_model = deepcopy(config)
    missing_gameplay_model["agents"]["profiles"]["gameplay"]["model"] = ""
    with pytest.raises(ValueError, match="agent profile gameplay.model"):
        preflight_collection(
            missing_gameplay_model, env={"DEEPSEEK_API_KEY": "fake"}
        )

    nonpositive_games = deepcopy(config)
    nonpositive_games["games"] = 0
    with pytest.raises(ValueError, match="games"):
        preflight_collection(nonpositive_games, env={"DEEPSEEK_API_KEY": "fake"})


def test_collection_preflight_resolves_guess_inheritance_and_override_without_network():
    inherited = _config()
    inherited["guess"] = {"backend": None, "model": None}
    inherited_result = preflight_collection(
        inherited, env={"DEEPSEEK_API_KEY": "fake"}
    )
    assert inherited_result["guess"]["gameplay"] == {
        "backend": "deepseek",
        "model": "deepseek-chat",
    }

    overridden = _config()
    overridden_result = preflight_collection(
        overridden, env={"DEEPSEEK_API_KEY": "fake"}
    )
    assert overridden_result["guess"]["gameplay"] == overridden["guess"]
    assert overridden_result["parser"] == overridden["parser"]
    assert overridden_result["backend_names"] == ["deepseek"]


class DeterministicFakeBackend:
    def __init__(self):
        self.parser_calls = 0

    def chat(self, messages, **kwargs):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system == PARSER_PROMPT_SPEC["text"]:
            self.parser_calls += 1
            if self.parser_calls == 1:
                return '{"events":[]}'
            return (
                '{"events":[{"event_family":"BELIEF_ASSERTION","target":[],'
                '"content":{"kind":"FACT","value":null},"qualifier":{},'
                '"ref_event_id":null,"source_span":"确定性发言",'
                '"parser_confidence":1.0}]}'
            )
        if system == BELIEF_PROMPT_SPEC["text"]:
            view = messages[1]["content"]
            self_role = next(
                line for line in view.splitlines() if "kind=SELF_ROLE" in line
            )
            observer_id = int(re.search(r"target=(\d+)", self_role).group(1))
            if len(messages) == 2:
                other = 1 if observer_id != 1 else 2
                return json.dumps({"wolf_pair": [observer_id, other]})
            known_wolves = set()
            known_good = {observer_id}
            for line in view.splitlines():
                if "kind=CHECK_RESULT" not in line and "kind=ROLE_REVEAL" not in line:
                    continue
                target = re.search(r"target=(\d+)", line)
                if target is None:
                    continue
                destination = known_wolves if "Werewolf" in line else known_good
                destination.add(int(target.group(1)))
            pair = next(
                pair
                for pair in combinations(range(1, 8), 2)
                if known_wolves.issubset(pair) and known_good.isdisjoint(pair)
            )
            return json.dumps({"wolf_pair": pair})
        assert system == GAMEPLAY_PROMPT_SPEC["text"]
        prompt = messages[1]["content"]
        if '{"speech":"你的公开发言"}' in prompt:
            return '{"speech":"确定性发言"}'
        return '{"action_index":1}'


def test_one_fake_game_collects_samples_failures_and_audit_without_network(tmp_path):
    config = _config()
    config["seed"] = 19
    config["output"] = {
        "samples": str(tmp_path / "samples.jsonl"),
        "failures": str(tmp_path / "failures.jsonl"),
        "logs": str(tmp_path / "logs"),
        "overwrite": True,
    }
    result = collect_from_config(
        config,
        games=1,
        backends={"deepseek": DeterministicFakeBackend()},
        env={"DEEPSEEK_API_KEY": "fake-for-test"},
    )

    audit = result["audit"]
    assert len(result["games"]) == 1
    assert result["games"][0]["winner"] in {"Werewolf", "Village"}
    assert audit["games"] == 1
    assert audit["unique_belief_elicitations"] > 0
    assert audit["successful_guesses"] == audit["unique_belief_elicitations"]
    assert audit["failed_guesses"] == 0
    assert audit["belief_first_attempt_successes"] == 0
    assert audit["belief_repair_successes"] == audit["unique_belief_elicitations"]
    assert audit["belief_repair_failures"] == 0
    assert audit["belief_contains_observer_failures"] == 0
    assert audit["belief_missing_required_wolf_failures"] == 0
    assert audit["belief_contains_forbidden_player_failures"] == 0
    assert audit["belief_success_rate"] == 1.0
    assert audit["belief_repair_success_rate"] == 1.0
    assert audit["first_order_samples"] > 0
    assert audit["second_order_public_samples"] > 0
    assert audit["second_order_wolf_samples"] > 0
    assert audit["duplicate_sample_ids"] == 0
    assert audit["state_alignment_errors"] == 0
    assert audit["unknown_kind_count"] == 0
    assert audit["unknown_value_count"] == 0
    assert audit["unknown_token_count"] == 0
    assert audit["unknown_token_ratio"] == 0.0
    assert audit["not_applicable_value_count"] > 0
    assert audit["top_unknown_raw_values"] == []
    assert audit["speech_event_count"] > 0
    assert audit["parser_call_count"] == audit["speech_event_count"]
    assert audit["parser_success_count"] > 0
    assert audit["parser_empty_count"] == 1
    assert audit["parser_failure_count"] == 0
    assert audit["parsed_semantic_event_count"] > 0
    assert audit["speech_with_semantic_events"] > 0
    assert audit["speech_without_semantic_events"] == 1
    assert audit["missing_parser_metadata_count"] == 0
    assert audit["parser_utterance_mismatch_count"] == 0
    samples = [
        json.loads(line)
        for line in Path(config["output"]["samples"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert samples
    protocol_ids = {sample["prompt_protocol"]["protocol_id"] for sample in samples}
    assert len(protocol_ids) == 1
    protocol = samples[0]["prompt_protocol"]
    assert protocol["language"] == "zh-CN"
    references = {
        name: protocol[name] for name in ("gameplay", "belief", "parser")
    }
    assert protocol["protocol_id"] == protocol_id_from_references(references)
    assert audit["prompt_protocol_ids"] == [protocol["protocol_id"]]
    assert audit["prompt_protocol_distribution"] == {
        protocol["protocol_id"]: len(samples)
    }
    assert audit["missing_prompt_protocol_count"] == 0
    assert audit["invalid_prompt_protocol_count"] == 0
    parser_events = [
        event
        for sample in samples
        for event in sample["events"]
        if event["source_type"] == "speech_parser"
    ]
    assert parser_events
    assert all(
        event["metadata"]["parser_protocol"]["version"] == "parser.zh.v1"
        and event["metadata"]["parser_protocol"]["sha256"]
        == PARSER_PROMPT_SPEC["sha256"]
        and event["metadata"]["parser_protocol"]["model"] == "deepseek-chat"
        and event["metadata"]["parser_protocol"]["temperature"] == 0.0
        and event["metadata"]["parser_protocol"]["status"] == "ok"
        for event in parser_events
    )
    unique_events = {
        event["event_id"]: event for sample in samples for event in sample["events"]
    }
    check_results = [
        event for event in unique_events.values()
        if event["content"]["kind"] == "CHECK_RESULT"
    ]
    assert check_results
    assert all(
        len(event["target"]) == 1
        and event["content"]["value"] in {"Werewolf", "Village"}
        and event["visibility"] == "private"
        and event["visible_to"] == [event["speaker"]]
        for event in check_results
    )
    log_records = [
        json.loads(line)
        for path in sorted((tmp_path / "logs" / "game_000019").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert log_records
    assert all(
        record["gameplay_prompt"] == {
            "version": "gameplay.zh.v1",
            "sha256": GAMEPLAY_PROMPT_SPEC["sha256"],
        }
        and record["model"] == "deepseek-chat"
        and record["temperature"] == 0.7
        and record["attempts"] in (1, 2)
        and "action" in record
        for record in log_records
    )
    assert "DEEPSEEK_API_KEY" not in json.dumps(samples)
    assert "fake-for-test" not in json.dumps(samples)
    assert Path(config["output"]["failures"]).exists()
    assert Path(config["output"]["failures"]).read_text(encoding="utf-8") == ""
    assert json.loads(Path(result["audit_path"]).read_text(encoding="utf-8")) == audit
