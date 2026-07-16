import json
import re
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pytest
import yaml

from script.tom.collect import collect_from_config, preflight_collection
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
    def chat(self, messages, **kwargs):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if system.startswith("Extract only explicit"):
            return '{"events":[]}'
        if system.startswith("You report the player's current belief"):
            view = messages[1]["content"]
            self_role = next(
                line for line in view.splitlines() if "kind=SELF_ROLE" in line
            )
            observer_id = int(re.search(r"target=(\d+)", self_role).group(1))
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
        prompt = messages[0]["content"]
        if 'Return exactly {"speech"' in prompt:
            return '{"speech":"deterministic statement"}'
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
    assert Path(config["output"]["samples"]).read_text(encoding="utf-8").strip()
    assert Path(config["output"]["failures"]).exists()
    assert Path(config["output"]["failures"]).read_text(encoding="utf-8") == ""
    assert json.loads(Path(result["audit_path"]).read_text(encoding="utf-8")) == audit
