import json
import re
import subprocess
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import script.tom.collect as collect_module
from script.tom.collect import collect_from_config, preflight_collection
from werewolf.backends.base import BackendError
from werewolf.backends.openai_compatible import OpenAICompatibleBackend
from werewolf.prompt_protocol import (
    BELIEF_SYSTEM_PROMPT,
    BELIEF_PROMPT_SPEC,
    GAMEPLAY_PROMPT_SPEC,
    PARSER_SYSTEM_PROMPT,
    PARSER_PROMPT_SPEC,
    protocol_id_from_references,
)
from werewolf.runtime_config import resolve_collection_output, validate_runtime_config
from werewolf.tom.dataset import ToMDataset


def _config():
    return yaml.safe_load(Path("configs/tom/collect.yaml").read_text(encoding="utf-8"))


class FakeCompletions:
    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []

    def create(self, **request):
        self.requests.append(deepcopy(request))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _fake_openai_backend(outcome):
    completions = FakeCompletions(outcome)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return OpenAICompatibleBackend(
        client=client, default_model="deepseek-chat"
    ), completions


def _fake_openai_error(
    name, *, status_code=None, body=None, request_id=None, headers=None,
):
    error_type = type(name, (Exception,), {})
    error = error_type("unsafe SDK repr must not be serialized")
    error.status_code = status_code
    error.body = body
    error.request_id = request_id
    error.response = SimpleNamespace(
        status_code=status_code,
        headers=headers or {},
        json=lambda: body,
    )
    return error


def test_backend_error_keeps_safe_read_only_diagnostics_and_chaining():
    error = BackendError(
        "backend failed\ncompactly",
        retryable=True,
        details={"cause_type": "RateLimitError", "status_code": 429},
    )
    assert error.message == "backend failed compactly"
    assert error.retryable is True
    assert error.details == {
        "cause_type": "RateLimitError", "status_code": 429,
    }
    assert "retryable=true" in str(error)
    assert "cause_type=RateLimitError" in str(error)
    with pytest.raises(TypeError):
        error.safe_details["status_code"] = 200
    with pytest.raises(TypeError, match="safe scalars"):
        BackendError("bad details", details={"exception": ValueError("secret")})

    cause = _fake_openai_error(
        "BadRequestError",
        status_code=400,
        body={"error": {"message": "bad input", "code": "bad_request"}},
    )
    backend, _ = _fake_openai_backend(cause)
    with pytest.raises(BackendError) as captured:
        backend.chat([{"role": "user", "content": "private prompt"}])
    assert captured.value.__cause__ is cause


@pytest.mark.parametrize(
    ("name", "status_code", "expected_retryable"),
    [
        ("BadRequestError", 400, False),
        ("AuthenticationError", 401, False),
        ("PermissionDeniedError", 403, False),
        ("NotFoundError", 404, False),
        ("RateLimitError", 429, True),
        ("InternalServerError", 500, True),
        ("APITimeoutError", None, True),
        ("APIConnectionError", None, True),
    ],
)
def test_openai_backend_classifies_safe_sdk_failures(
    name, status_code, expected_retryable
):
    cause = _fake_openai_error(
        name,
        status_code=status_code,
        body={
            "error": {
                "type": "provider_type",
                "code": "provider_code",
                "param": "response_format",
                "message": "safe provider message",
                "forbidden_extra": "must not survive",
            },
            "request_payload": "must not survive",
        },
        request_id="req-direct-123",
    )
    backend, _ = _fake_openai_backend(cause)
    with pytest.raises(BackendError) as captured:
        backend.chat([{"role": "user", "content": "private prompt"}])

    error = captured.value
    assert error.retryable is expected_retryable
    assert error.safe_details == {
        "backend": "openai_compatible",
        "model": "deepseek-chat",
        "cause_type": name,
        "status_code": status_code,
        "provider_error_type": "provider_type",
        "provider_error_code": "provider_code",
        "provider_error_param": "response_format",
        "request_id": "req-direct-123",
        "safe_message": "safe provider message",
    }
    assert "forbidden_extra" not in str(error)
    assert "request_payload" not in str(error)


def test_openai_backend_extracts_header_request_id_and_redacts_secrets_prompts():
    secret = "sk-deepseek-super-secret-123"
    prompt = "SYSTEM PRIVATE PROMPT CONTENT"
    message = (
        f"DEEPSEEK_API_KEY={secret}\n"
        f"Authorization: Bearer {secret}\n{prompt}"
    )
    cause = _fake_openai_error(
        "BadRequestError",
        status_code=400,
        body={"error": {"message": message, "code": "invalid_request"}},
        headers={
            "X-Request-ID": "req-header-456",
            "Authorization": f"Bearer {secret}",
        },
    )
    backend, _ = _fake_openai_backend(cause)
    with pytest.raises(BackendError) as captured:
        backend.chat([{"role": "system", "content": prompt}])

    rendered = str(captured.value)
    assert captured.value.safe_details["request_id"] == "req-header-456"
    assert captured.value.safe_details["provider_error_code"] == "invalid_request"
    assert secret not in rendered
    assert prompt not in rendered
    assert "Authorization: Bearer" not in rendered
    assert "[REDACTED" in rendered


@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        (SimpleNamespace(choices=[]), "empty_choices"),
        (
            SimpleNamespace(choices=[SimpleNamespace(message=None)]),
            "missing_message",
        ),
        (
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
            ),
            "non_text_content",
        ),
        (
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="  "))]
            ),
            "empty_content",
        ),
    ],
)
def test_openai_backend_rejects_deterministic_response_shapes(
    response, error_code
):
    backend, completions = _fake_openai_backend(response)
    with pytest.raises(BackendError) as captured:
        backend.chat([{"role": "user", "content": "prompt"}])
    assert len(completions.requests) == 1
    assert captured.value.retryable is False
    assert captured.value.safe_details["cause_type"] == "ResponseShapeError"
    assert captured.value.safe_details["provider_error_code"] == error_code


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


@pytest.mark.parametrize("run_id", ["game_001", "game_010", "game_1000"])
def test_run_id_resolves_canonical_data_and_log_paths(tmp_path, run_id):
    config = _config()
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"
    resolved, paths = resolve_collection_output(
        config, run_id=run_id, data_dir=data_root, log_dir=log_root
    )
    assert paths.data_run_dir == data_root.resolve() / run_id
    assert paths.log_run_dir == log_root.resolve() / run_id
    assert paths.samples_path == paths.data_run_dir / f"{run_id}.samples.jsonl"
    assert paths.audit_path == paths.data_run_dir / f"{run_id}.audit.json"
    assert paths.failures_path == paths.data_run_dir / f"{run_id}.failures.jsonl"
    assert paths.game_log_path == paths.log_run_dir / f"{run_id}.game_log.json"
    assert [paths.player_log_path(player_id) for player_id in range(1, 8)] == [
        paths.log_run_dir / f"{run_id}.player_{player_id}.jsonl"
        for player_id in range(1, 8)
    ]
    assert resolved["output"] == {
        "samples": str(paths.samples_path),
        "failures": str(paths.failures_path),
        "logs": str(paths.log_run_dir),
        "overwrite": False,
    }
    assert config == _config()


@pytest.mark.parametrize(
    "run_id", ["game_1", "pilot_001", "001", "../game_001", "game_001/test"]
)
def test_invalid_run_id_is_rejected(run_id):
    config = _config()
    with pytest.raises(ValueError, match="run_id must match"):
        resolve_collection_output(config, run_id=run_id)


@pytest.mark.parametrize("conflict", ["data", "logs", "both"])
def test_run_directory_conflicts_fail_before_backend_or_partial_creation(
    tmp_path, monkeypatch, conflict
):
    config = _config()
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"
    data_run_dir = data_root / "game_001"
    log_run_dir = log_root / "game_001"
    if conflict in {"data", "both"}:
        data_run_dir.mkdir(parents=True)
    if conflict in {"logs", "both"}:
        log_run_dir.mkdir(parents=True)
    loaded = 0

    def unexpected_backend_load(_config):
        nonlocal loaded
        loaded += 1
        raise AssertionError("backend construction must not start")

    monkeypatch.setattr(collect_module, "load_named_backends", unexpected_backend_load)
    with pytest.raises(FileExistsError, match="run directory already exists") as error:
        collect_from_config(
            config,
            run_id="game_001",
            data_dir=data_root,
            log_dir=log_root,
            env={"DEEPSEEK_API_KEY": "fake"},
        )
    assert loaded == 0
    expected_conflicts = {
        "data": [data_run_dir],
        "logs": [log_run_dir],
        "both": [data_run_dir, log_run_dir],
    }[conflict]
    assert all(str(path) in str(error.value) for path in expected_conflicts)
    if conflict == "data":
        assert not log_root.exists()
    if conflict == "logs":
        assert not data_root.exists()


def test_games_other_than_one_fail_before_backend(tmp_path, monkeypatch):
    config = _config()
    monkeypatch.setattr(
        collect_module,
        "load_named_backends",
        lambda _config: pytest.fail("backend construction must not start"),
    )
    with pytest.raises(ValueError, match="one run_id represents exactly one game"):
        collect_from_config(
            config,
            run_id="game_001",
            games=2,
            data_dir=tmp_path / "data",
            log_dir=tmp_path / "logs",
            env={"DEEPSEEK_API_KEY": "fake"},
        )
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "logs").exists()


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


class EmptySpeechFakeBackend(DeterministicFakeBackend):
    def chat(self, messages, **kwargs):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system not in {PARSER_SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT}:
            prompt = messages[1]["content"]
            if '{"speech":"..."}' in prompt:
                return '{"speech":""}'
            match = re.search(
                r"【当前合法动作】\n(\[.*?\])\n\n【动作下标说明】",
                prompt,
                re.DOTALL,
            )
            actions = json.loads(match.group(1))
            return json.dumps({"action_index": 1 if len(actions) > 1 else 0})
        return super().chat(messages, **kwargs)


class InvalidGameplayFakeBackend(DeterministicFakeBackend):
    def chat(self, messages, **kwargs):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system not in {PARSER_SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT}:
            return '{“action_index”:0}'
        return super().chat(messages, **kwargs)


class PlayerIdConfusionFakeBackend(DeterministicFakeBackend):
    def __init__(self, *, repair_succeeds):
        super().__init__()
        self.repair_succeeds = repair_succeeds
        self.awaiting_repair = False
        self.corrected_index = None

    @staticmethod
    def _options(prompt):
        match = re.search(
            r"【当前合法动作】\n(\[.*?\])\n\n【动作下标说明】",
            prompt,
            re.DOTALL,
        )
        return json.loads(match.group(1))

    @staticmethod
    def _index_for_target(options, target_player):
        return next(
            option["option_index"]
            for option in options
            if option["target_player"] == target_player
        )

    def chat(self, messages, **kwargs):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system in {PARSER_SYSTEM_PROMPT, BELIEF_SYSTEM_PROMPT}:
            return super().chat(messages, **kwargs)
        if self.awaiting_repair and len(messages) == 4:
            self.awaiting_repair = False
            if not self.repair_succeeds:
                return '{"action_index":7}'
            return json.dumps({"action_index": self.corrected_index})
        prompt = messages[1]["content"]
        if '{"speech":"..."}' in prompt:
            return '{"speech":"确定性发言"}'
        options = self._options(prompt)
        phase = re.search(r"当前阶段：([^\n]+)", prompt).group(1)
        player_id = int(re.search(r"玩家编号：(\d+)", prompt).group(1))
        if phase == "2_day_vote" and player_id == 1:
            self.corrected_index = self._index_for_target(options, 7)
            self.awaiting_repair = True
            return '{"action_index":7}'
        if "vote" in phase:
            target = (
                6
                if phase == "1_day_vote"
                else (7 if phase == "2_day_vote" else 5)
            )
            return json.dumps(
                {"action_index": self._index_for_target(options, target)}
            )
        no_target = next(
            option for option in options if option["target_player"] is None
        )
        return json.dumps({"action_index": no_target["option_index"]})


def test_one_fake_game_collects_samples_failures_and_audit_without_network(tmp_path):
    config = _config()
    config["seed"] = 19
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"
    result = collect_from_config(
        config,
        games=1,
        run_id="game_001",
        data_dir=data_root,
        log_dir=log_root,
        backends={"deepseek": DeterministicFakeBackend()},
        env={"DEEPSEEK_API_KEY": "fake-for-test"},
    )

    audit = result["audit"]
    data_run_dir = data_root / "game_001"
    log_run_dir = log_root / "game_001"
    samples_path = data_run_dir / "game_001.samples.jsonl"
    failures_path = data_run_dir / "game_001.failures.jsonl"
    audit_path = data_run_dir / "game_001.audit.json"
    assert result["run_id"] == "game_001"
    assert Path(result["data_run_dir"]) == data_run_dir
    assert Path(result["log_run_dir"]) == log_run_dir
    assert Path(result["samples_path"]) == samples_path
    assert Path(result["audit_path"]) == audit_path
    assert {path.name for path in data_run_dir.iterdir()} == {
        "game_001.samples.jsonl",
        "game_001.failures.jsonl",
        "game_001.audit.json",
    }
    assert {path.name for path in log_run_dir.iterdir()} == {
        "game_001.game_log.json",
        *(f"game_001.player_{player_id}.jsonl" for player_id in range(1, 8)),
    }
    assert not any(path.is_dir() for path in data_run_dir.iterdir())
    assert not any(path.is_dir() for path in log_run_dir.iterdir())
    assert not (data_root / "tom").exists()
    assert not list(data_run_dir.glob(".game_001.audit.json.*.tmp"))
    assert len(result["games"]) == 1
    assert result["games"][0]["winner"] in {"Werewolf", "Village"}
    assert audit["schema_version"] == "tom.audit.v1_4"
    assert audit["collection_status"] == "complete"
    assert audit["completed_games"] == 1
    assert audit["runtime_failure_count"] == 0
    assert audit["failed_game_id"] is None
    assert audit["runtime_error_type"] is None
    assert audit["runtime_error_message"] is None
    assert audit["games"] == 1
    assert audit["unique_belief_elicitations"] > 0
    assert audit["successful_guesses"] == audit["unique_belief_elicitations"]
    assert audit["failed_guesses"] == 0
    assert audit["belief_first_attempt_successes"] == 0
    assert audit["belief_repair_successes"] == audit["unique_belief_elicitations"]
    assert audit["belief_repair_failures"] == 0
    assert audit["repair_attempts"] == audit["unique_belief_elicitations"]
    assert audit["belief_backend_failure_count"] == 0
    assert audit["belief_backend_retry_attempts"] == 0
    assert audit["belief_backend_retry_successes"] == 0
    assert audit["belief_backend_retry_failures"] == 0
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
    assert audit["parser_failure_rate"] == 0.0
    assert audit["parsed_semantic_event_count"] > 0
    assert audit["speech_with_semantic_events"] > 0
    assert audit["speech_without_semantic_events"] == 1
    assert audit["missing_parser_metadata_count"] == 0
    assert audit["parser_utterance_mismatch_count"] == 0
    samples = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
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
        event["metadata"]["parser_protocol"]["version"] == "parser.zh.v3"
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
        for path in sorted(log_run_dir.glob("game_001.player_*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert log_records
    assert all(
        record["gameplay_prompt"] == {
            "version": "gameplay.zh.v4",
            "sha256": GAMEPLAY_PROMPT_SPEC["sha256"],
        }
        and record["model"] == "deepseek-chat"
        and record["temperature"] == 0.7
        and record["attempts"] in (1, 2)
        and "action" in record
        and "valid_action_options" in record
        for record in log_records
    )
    assert "DEEPSEEK_API_KEY" not in json.dumps(samples)
    assert "fake-for-test" not in json.dumps(samples)
    assert failures_path.read_text(encoding="utf-8") == ""
    assert json.loads(Path(result["audit_path"]).read_text(encoding="utf-8")) == audit
    assert len(ToMDataset(samples_path)) > 0

    with pytest.raises(FileExistsError, match="run directory already exists"):
        collect_from_config(
            config,
            games=1,
            run_id="game_001",
            data_dir=data_root,
            log_dir=log_root,
            backends={"deepseek": DeterministicFakeBackend()},
            env={"DEEPSEEK_API_KEY": "fake-for-test"},
        )


def test_one_fake_game_preserves_explicit_empty_speech_and_completes_audit(tmp_path):
    config = _config()
    config["seed"] = 21
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"
    backend = EmptySpeechFakeBackend()

    result = collect_from_config(
        config,
        games=1,
        run_id="game_002",
        data_dir=data_root,
        log_dir=log_root,
        backends={"deepseek": backend},
        env={"DEEPSEEK_API_KEY": "fake-for-test"},
    )

    audit = result["audit"]
    assert audit["collection_status"] == "complete"
    assert audit["completed_games"] == 1
    assert audit["runtime_failure_count"] == 0
    assert audit["speech_event_count"] > 0
    assert audit["parser_call_count"] == audit["speech_event_count"]
    assert audit["parser_empty_count"] == audit["speech_event_count"]
    assert audit["parser_success_count"] == 0
    assert audit["parser_failure_count"] == 0
    assert audit["parser_failure_rate"] == 0.0
    assert audit["parser_repair_attempts"] == 0
    assert audit["parsed_semantic_event_count"] == 0
    assert audit["speech_without_semantic_events"] == audit["speech_event_count"]
    assert audit["public_checkpoints"] > 0
    assert audit["first_order_samples"] > 0
    assert backend.parser_calls == 0

    samples = [
        json.loads(line)
        for line in (
            data_root / "game_002" / "game_002.samples.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_speech = {
        event["event_id"]: event
        for sample in samples
        for event in sample["events"]
        if event["source_type"] == "environment"
        and event["content"]["kind"] == "SPEECH"
    }
    assert raw_speech
    assert all(event["source_span"] == "" for event in raw_speech.values())
    assert all(
        event["metadata"]["parser_result"]["status"] == "empty"
        and event["metadata"]["parser_result"]["attempts"] == 1
        for event in raw_speech.values()
    )
    assert not any(
        event["source_type"] == "speech_parser"
        for sample in samples
        for event in sample["events"]
    )


def test_fake_player_id_confusion_repairs_to_option_index_and_completes(tmp_path):
    config = _config()
    backend = PlayerIdConfusionFakeBackend(repair_succeeds=True)
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"

    result = collect_from_config(
        config,
        games=1,
        run_id="game_005",
        data_dir=data_root,
        log_dir=log_root,
        backends={"deepseek": backend},
        env={"DEEPSEEK_API_KEY": "fake-for-test"},
    )

    assert result["audit"]["collection_status"] == "complete"
    assert result["audit"]["runtime_failure_count"] == 0
    player_log = log_root / "game_005" / "game_005.player_1.jsonl"
    records = [
        json.loads(line)
        for line in player_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    repaired = next(record for record in records if record["phase"] == "2_day_vote")
    target_option = next(
        option
        for option in repaired["valid_action_options"]
        if option["target_player"] == 7
    )
    assert target_option["option_index"] != 7
    assert repaired["attempts"] == 2
    assert repaired["responses"] == [
        '{"action_index":7}',
        json.dumps({"action_index": target_option["option_index"]}),
    ]
    assert repaired["error_code"] is None
    assert repaired["action"] == ["vote", 7]
    assert len(ToMDataset(result["samples_path"])) > 0


def test_fake_repeated_player_id_confusion_keeps_partial_failure(tmp_path):
    config = _config()
    backend = PlayerIdConfusionFakeBackend(repair_succeeds=False)
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"

    with pytest.raises(RuntimeError, match="action_index_out_of_range"):
        collect_from_config(
            config,
            games=1,
            run_id="game_006",
            data_dir=data_root,
            log_dir=log_root,
            backends={"deepseek": backend},
            env={"DEEPSEEK_API_KEY": "fake-for-test"},
        )

    data_run_dir = data_root / "game_006"
    audit = json.loads(
        (data_run_dir / "game_006.audit.json").read_text(encoding="utf-8")
    )
    assert audit["collection_status"] == "failed"
    assert audit["runtime_failure_count"] == 1
    assert audit["failed_game_id"] == "game_006"
    player_log = log_root / "game_006" / "game_006.player_1.jsonl"
    records = [
        json.loads(line)
        for line in player_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed = next(record for record in records if record["phase"] == "2_day_vote")
    assert failed["attempts"] == 2
    assert failed["responses"] == ['{"action_index":7}', '{"action_index":7}']
    assert failed["error_code"] == "action_index_out_of_range"
    assert failed["action"] is None
    with pytest.raises(ValueError, match="failed collection audit"):
        ToMDataset(data_run_dir / "game_006.samples.jsonl")


def test_gameplay_failure_writes_partial_audit_and_is_not_trainable(tmp_path):
    config = _config()
    config["seed"] = 23
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"

    with pytest.raises(RuntimeError, match="invalid_json"):
        collect_from_config(
            config,
            games=1,
            run_id="game_003",
            data_dir=data_root,
            log_dir=log_root,
            backends={"deepseek": InvalidGameplayFakeBackend()},
            env={"DEEPSEEK_API_KEY": "fake-secret-must-not-leak"},
        )

    data_run_dir = data_root / "game_003"
    log_run_dir = log_root / "game_003"
    samples_path = data_run_dir / "game_003.samples.jsonl"
    for name in (
        "game_003.samples.jsonl",
        "game_003.failures.jsonl",
        "game_003.audit.json",
    ):
        assert (data_run_dir / name).is_file()
    assert (log_run_dir / "game_003.game_log.json").is_file()
    assert all(
        (log_run_dir / f"game_003.player_{player_id}.jsonl").is_file()
        for player_id in range(1, 8)
    )
    audit = json.loads(
        (data_run_dir / "game_003.audit.json").read_text(encoding="utf-8")
    )
    assert audit["schema_version"] == "tom.audit.v1_4"
    assert audit["collection_status"] == "failed"
    assert audit["completed_games"] == 0
    assert audit["runtime_failure_count"] == 1
    assert audit["failed_game_id"] == "game_003"
    assert audit["runtime_error_type"] == "RuntimeError"
    assert audit["runtime_error_message"] == (
        "gameplay action generation failed: invalid_json"
    )
    assert audit["parser_failure_rate"] is None
    rendered_audit = json.dumps(audit, ensure_ascii=False)
    assert "fake-secret-must-not-leak" not in rendered_audit
    assert "【游戏规则】" not in rendered_audit

    log_records = [
        json.loads(line)
        for path in log_run_dir.glob("game_003.player_*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(log_records) == 1
    assert log_records[0]["attempts"] == 2
    assert log_records[0]["responses"] == [
        '{“action_index”:0}', '{“action_index”:0}'
    ]
    assert log_records[0]["error_code"] == "invalid_json"
    assert log_records[0]["action"] is None
    with pytest.raises(ValueError, match="failed collection audit"):
        ToMDataset(samples_path)


def test_audit_fatal_preserves_all_pilot_outputs(tmp_path, monkeypatch):
    config = _config()
    config["seed"] = 19
    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"

    def forced_failure(_audit):
        raise RuntimeError("forced audit failure")

    monkeypatch.setattr(collect_module, "assert_audit_passes", forced_failure)
    with pytest.raises(RuntimeError, match="forced audit failure"):
        collect_from_config(
            config,
            games=1,
            run_id="game_004",
            data_dir=data_root,
            log_dir=log_root,
            backends={"deepseek": DeterministicFakeBackend()},
            env={"DEEPSEEK_API_KEY": "fake-for-test"},
        )
    for name in (
        "game_004.samples.jsonl",
        "game_004.failures.jsonl",
        "game_004.audit.json",
    ):
        assert (data_root / "game_004" / name).is_file()


def test_collect_cli_requires_run_id_rejects_output_dir_and_reports_paths(
    tmp_path, monkeypatch, capsys
):
    project_root = Path(__file__).parents[2]
    help_result = subprocess.run(
        [sys.executable, "-m", "script.tom.collect", "--help"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--run-id RUN_ID" in help_result.stdout
    assert "--data-dir DATA_DIR" in help_result.stdout
    assert "--log-dir LOG_DIR" in help_result.stdout
    assert "--output-dir" not in help_result.stdout

    missing_run_id = subprocess.run(
        [sys.executable, "-m", "script.tom.collect", "--games", "1"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_run_id.returncode == 2
    assert "--run-id" in missing_run_id.stderr

    legacy_output = subprocess.run(
        [
            sys.executable, "-m", "script.tom.collect", "--games", "1",
            "--run-id", "game_001", "--output-dir", str(tmp_path / "old"),
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert legacy_output.returncode == 2
    assert "unrecognized arguments" in legacy_output.stderr

    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"
    captured = {}

    def fake_collect(config, *, games, run_id, data_dir, log_dir, **_kwargs):
        captured.update(
            games=games, run_id=run_id, data_dir=data_dir, log_dir=log_dir
        )
        return {
            "games": [{}],
            "run_id": run_id,
            "data_run_dir": str(Path(data_dir).resolve() / run_id),
            "log_run_dir": str(Path(log_dir).resolve() / run_id),
            "samples_path": str(
                Path(data_dir).resolve() / run_id / f"{run_id}.samples.jsonl"
            ),
            "audit_path": str(
                Path(data_dir).resolve() / run_id / f"{run_id}.audit.json"
            ),
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
            "--games", "1", "--run-id", "game_001",
            "--data-dir", str(data_root), "--log-dir", str(log_root),
        ],
    )
    collect_module.main()
    summary = json.loads(capsys.readouterr().out)
    assert captured == {
        "games": 1,
        "run_id": "game_001",
        "data_dir": str(data_root),
        "log_dir": str(log_root),
    }
    assert summary["run_id"] == "game_001"
    assert summary["data_run_dir"] == str(data_root.resolve() / "game_001")
    assert summary["log_run_dir"] == str(log_root.resolve() / "game_001")
    assert summary["audit_path"] == str(
        data_root.resolve() / "game_001" / "game_001.audit.json"
    )


def test_generated_game_runs_are_ignored_but_fixed_resources_are_tracked():
    project_root = Path(__file__).parents[2]
    generated_paths = (
        "data/game_001/game_001.samples.jsonl",
        "logs/game_001/game_001.player_1.jsonl",
    )
    generated = [
        subprocess.run(
            ["git", "check-ignore", "-v", path],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        for path in generated_paths
    ]
    fixture = subprocess.run(
        ["git", "check-ignore", "-v", "tests/fixtures/tom_v1.jsonl"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    docs = subprocess.run(
        ["git", "check-ignore", "-v", "data/docs/reference.md"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert all(result.returncode == 0 for result in generated)
    assert "data/game_*/" in generated[0].stdout
    assert "logs/game_*/" in generated[1].stdout
    assert fixture.returncode == 1
    assert fixture.stdout == ""
    assert docs.returncode == 1
    assert docs.stdout == ""
