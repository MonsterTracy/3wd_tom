import json
import re
import subprocess
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pytest
import yaml

import script.tom.collect as collect_module
from script.tom.collect import collect_from_config, preflight_collection
from werewolf.prompt_protocol import (
    BELIEF_SYSTEM_PROMPT,
    BELIEF_PROMPT_SPEC,
    GAMEPLAY_PROMPT_SPEC,
    PARSER_SYSTEM_PROMPT,
    PARSER_PROMPT_SPEC,
    protocol_id_from_references,
)
from werewolf.runtime_config import resolve_collection_output, validate_runtime_config


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


def test_output_dir_overrides_all_configured_paths_and_default_keeps_config(tmp_path):
    config = _config()
    unchanged, default_paths = resolve_collection_output(config)
    assert unchanged == config
    assert default_paths["samples"] == Path(config["output"]["samples"])
    assert default_paths["failures"] == Path(config["output"]["failures"])
    assert default_paths["logs"] == Path(config["output"]["logs"])

    pilot_dir = tmp_path / "pilot_001"
    overridden, paths = resolve_collection_output(config, pilot_dir)
    assert paths == {
        "output_dir": pilot_dir.resolve(),
        "samples": pilot_dir.resolve() / "samples.jsonl",
        "failures": pilot_dir.resolve() / "failures.jsonl",
        "audit": pilot_dir.resolve() / "samples.audit.json",
        "logs": pilot_dir.resolve() / "logs",
    }
    assert overridden["output"] == {
        "samples": str(paths["samples"]),
        "failures": str(paths["failures"]),
        "logs": str(paths["logs"]),
        "overwrite": False,
    }
    assert config == _config()


def test_output_dir_safety_rejects_blank_file_and_existing_dir_before_backend(
    tmp_path, monkeypatch
):
    config = _config()
    loaded = []

    def unexpected_backend_load(_config):
        loaded.append(True)
        raise AssertionError("backend construction must not start")

    monkeypatch.setattr(collect_module, "load_named_backends", unexpected_backend_load)
    with pytest.raises(ValueError, match="non-empty"):
        collect_from_config(
            config, output_dir="", env={"DEEPSEEK_API_KEY": "fake"}
        )

    output_file = tmp_path / "pilot-file"
    output_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="is a file"):
        collect_from_config(
            config, output_dir=output_file,
            env={"DEEPSEEK_API_KEY": "fake"},
        )

    existing_dir = tmp_path / "pilot-existing"
    existing_dir.mkdir()
    (existing_dir / "samples.jsonl").write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        collect_from_config(
            config, output_dir=existing_dir,
            env={"DEEPSEEK_API_KEY": "fake"},
        )
    assert loaded == []


def test_configured_existing_sample_is_rejected_before_backend(tmp_path, monkeypatch):
    config = _config()
    config["output"] = {
        "samples": str(tmp_path / "samples.jsonl"),
        "failures": str(tmp_path / "failures.jsonl"),
        "logs": str(tmp_path / "logs"),
        "overwrite": False,
    }
    Path(config["output"]["samples"]).write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(
        collect_module,
        "load_named_backends",
        lambda _config: pytest.fail("backend construction must not start"),
    )
    with pytest.raises(FileExistsError, match="samples.jsonl"):
        collect_from_config(config, env={"DEEPSEEK_API_KEY": "fake"})


class DeterministicFakeBackend:
    def __init__(self):
        self.parser_calls = 0

    def chat(self, messages, **kwargs):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system == PARSER_SYSTEM_PROMPT:
            self.parser_calls += 1
            if self.parser_calls == 1:
                return '{"events":[]}'
            return (
                '{"events":[{"event_family":"BELIEF_ASSERTION","target":[],'
                '"content":{"kind":"FACT","value":null},"qualifier":{},'
                '"ref_event_id":null,"source_span":"确定性发言",'
                '"parser_confidence":1.0}]}'
            )
        if system == BELIEF_SYSTEM_PROMPT:
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
        assert "【游戏规则】" in system
        assert "【当前角色规则】" in system
        prompt = messages[1]["content"]
        if '{"speech":"..."}' in prompt:
            return '{"speech":"确定性发言"}'
        return '{"action_index":1}'


def test_one_fake_game_collects_samples_failures_and_audit_without_network(tmp_path):
    config = _config()
    config["seed"] = 19
    config["output"] = {
        "samples": str(tmp_path / "configured" / "samples.jsonl"),
        "failures": str(tmp_path / "configured" / "failures.jsonl"),
        "logs": str(tmp_path / "configured-logs"),
        "overwrite": False,
    }
    output_dir = tmp_path / "pilot_001"
    result = collect_from_config(
        config,
        games=1,
        output_dir=output_dir,
        backends={"deepseek": DeterministicFakeBackend()},
        env={"DEEPSEEK_API_KEY": "fake-for-test"},
    )

    audit = result["audit"]
    assert result["output_dir"] == str(output_dir.resolve())
    assert Path(result["audit_path"]) == output_dir / "samples.audit.json"
    assert {
        "samples.jsonl", "failures.jsonl", "samples.audit.json", "logs"
    } <= {path.name for path in output_dir.iterdir()}
    assert not list(output_dir.glob(".samples.audit.json.*.tmp"))
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
        for line in (output_dir / "samples.jsonl").read_text(
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
    assert audit["ruleset_ids"] == [protocol["ruleset"]["id"]]
    assert audit["ruleset_versions"] == [protocol["ruleset"]["version"]]
    assert audit["ruleset_hashes"] == [protocol["ruleset"]["sha256"]]
    assert audit["missing_ruleset_count"] == 0
    assert audit["invalid_ruleset_count"] == 0
    parser_events = [
        event
        for sample in samples
        for event in sample["events"]
        if event["source_type"] == "speech_parser"
    ]
    assert parser_events
    assert all(
        event["metadata"]["parser_protocol"]["version"] == "parser.zh.v2"
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
        for path in sorted((output_dir / "logs" / "game_000019").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert log_records
    assert all(
        record["gameplay_prompt"] == {
            "version": "gameplay.zh.v2",
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
    assert (output_dir / "failures.jsonl").exists()
    assert (output_dir / "failures.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads(Path(result["audit_path"]).read_text(encoding="utf-8")) == audit


def test_audit_fatal_preserves_all_pilot_outputs(tmp_path, monkeypatch):
    config = _config()
    config["seed"] = 19
    output_dir = tmp_path / "pilot_fatal"

    def forced_failure(_audit):
        raise RuntimeError("forced audit failure")

    monkeypatch.setattr(collect_module, "assert_audit_passes", forced_failure)
    with pytest.raises(RuntimeError, match="forced audit failure"):
        collect_from_config(
            config,
            games=1,
            output_dir=output_dir,
            backends={"deepseek": DeterministicFakeBackend()},
            env={"DEEPSEEK_API_KEY": "fake-for-test"},
        )
    for name in ("samples.jsonl", "failures.jsonl", "samples.audit.json"):
        assert (output_dir / name).is_file()


def test_collect_cli_help_and_summary_include_output_dir(tmp_path, monkeypatch, capsys):
    project_root = Path(__file__).parents[2]
    help_result = subprocess.run(
        [sys.executable, "-m", "script.tom.collect", "--help"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--output-dir OUTPUT_DIR" in help_result.stdout

    output_dir = tmp_path / "pilot_summary"
    captured = {}

    def fake_collect(config, *, games, output_dir, **_kwargs):
        captured.update(games=games, output_dir=output_dir)
        return {
            "games": [{}],
            "output_dir": str(Path(output_dir).resolve()),
            "audit_path": str(Path(output_dir).resolve() / "samples.audit.json"),
            "audit": {
                "unique_belief_elicitations": 1,
                "successful_guesses": 1,
                "failed_guesses": 0,
                "first_order_samples": 1,
                "second_order_public_samples": 1,
                "second_order_wolf_samples": 2,
            },
        }

    monkeypatch.setattr(collect_module, "collect_from_config", fake_collect)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect.py", "--config", "configs/tom/collect.yaml",
            "--games", "1", "--output-dir", str(output_dir),
        ],
    )
    collect_module.main()
    summary = json.loads(capsys.readouterr().out)
    assert captured == {"games": 1, "output_dir": str(output_dir)}
    assert summary["output_dir"] == str(output_dir.resolve())
    assert summary["audit_path"] == str(output_dir.resolve() / "samples.audit.json")


def test_generated_pilot_data_is_ignored_but_fixture_is_tracked():
    project_root = Path(__file__).parents[2]
    generated = subprocess.run(
        ["git", "check-ignore", "-v", "data/tom/pilot_test/samples.jsonl"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    fixture = subprocess.run(
        ["git", "check-ignore", "-v", "tests/fixtures/tom_v1.jsonl"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert generated.returncode == 0
    assert "data/tom/**/*.jsonl" in generated.stdout
    assert fixture.returncode == 1
    assert fixture.stdout == ""
