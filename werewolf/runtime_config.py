"""Strict validation for the sole supported collection runtime schema."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

from werewolf.game_rules import NUM_PLAYERS, NUM_WEREWOLVES, ROLE_DISTRIBUTIONS


RUNTIME_SCHEMA_VERSION = "runtime.v1"
RUN_ID_PATTERN = re.compile(r"^game_[0-9]{3,}$")
BACKEND_FIELDS = {"type", "base_url", "api_key_env", "default_model"}
PROFILE_FIELDS = {"agent_type", "backend", "model", "temperature", "strategy"}
TOP_LEVEL_FIELDS = {
    "schema_version",
    "seed",
    "games",
    "max_steps",
    "backends",
    "parser",
    "guess",
    "agents",
    "environment",
    "output",
}


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    data_root: Path
    log_root: Path
    data_run_dir: Path
    log_run_dir: Path
    samples_path: Path
    audit_path: Path
    failures_path: Path
    game_log_path: Path
    parser_failures_path: Path

    def player_log_path(self, player_id):
        if type(player_id) is not int or not 1 <= player_id <= NUM_PLAYERS:
            raise ValueError(f"player_id must be between 1 and {NUM_PLAYERS}")
        return self.log_run_dir / f"{self.run_id}.player_{player_id}.jsonl"


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _exact_fields(value, expected, name):
    value = _mapping(value, name)
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(f"{name} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value


def validate_runtime_config(config):
    _exact_fields(config, TOP_LEVEL_FIELDS, "runtime config")
    if config["schema_version"] != RUNTIME_SCHEMA_VERSION:
        raise ValueError("unsupported runtime schema_version; legacy configs are rejected")
    for field in ("seed", "games", "max_steps"):
        if type(config[field]) is not int or config[field] < (0 if field == "seed" else 1):
            raise ValueError(f"{field} has an invalid value")

    backends = _mapping(config["backends"], "backends")
    if not backends:
        raise ValueError("at least one backend is required")
    for name, backend in backends.items():
        _text(name, "backend name")
        _exact_fields(backend, BACKEND_FIELDS, f"backend {name}")
        if backend["type"] != "openai_compatible":
            raise ValueError("only openai_compatible backends are supported")
        _text(backend["api_key_env"], f"backend {name}.api_key_env")
        _text(backend["default_model"], f"backend {name}.default_model")
        _text(backend["base_url"], f"backend {name}.base_url")

    parser = config["parser"]
    _exact_fields(parser, {"backend", "model"}, "parser")
    if parser["backend"] not in backends:
        raise ValueError("parser backend does not exist")
    _text(parser["model"], "parser.model")

    guess = config["guess"]
    _exact_fields(guess, {"backend", "model"}, "guess")
    if (guess["backend"] is None) != (guess["model"] is None):
        raise ValueError("guess.backend and guess.model must both inherit or both be set")
    if guess["backend"] is not None:
        if guess["backend"] not in backends:
            raise ValueError("guess backend does not exist")
        _text(guess["model"], "guess.model")

    agents = config["agents"]
    _exact_fields(agents, {"profiles", "werewolf_profile", "village_profile"}, "agents")
    profiles = _mapping(agents["profiles"], "agents.profiles")
    if not profiles:
        raise ValueError("at least one agent profile is required")
    for name, profile in profiles.items():
        _exact_fields(profile, PROFILE_FIELDS, f"agent profile {name}")
        _text(profile["agent_type"], f"agent profile {name}.agent_type")
        if profile["backend"] not in backends:
            raise ValueError(f"agent profile {name} backend does not exist")
        _text(profile["model"], f"agent profile {name}.model")
        if isinstance(profile["temperature"], bool) or not isinstance(
            profile["temperature"], (int, float)
        ):
            raise ValueError(f"agent profile {name}.temperature must be numeric")
        if not isinstance(profile["strategy"], dict):
            raise ValueError(f"agent profile {name}.strategy must be a mapping")
    for assignment in ("werewolf_profile", "village_profile"):
        if agents[assignment] not in profiles:
            raise ValueError(f"agents.{assignment} does not exist")

    environment = config["environment"]
    canonical_roles = ROLE_DISTRIBUTIONS["seer_witch"]
    required_environment = {
        "n_player": NUM_PLAYERS,
        "n_role": sum(count > 0 for count in canonical_roles.values()),
        "n_werewolf": NUM_WEREWOLVES,
        "n_seer": canonical_roles["Seer"],
        "n_villager": canonical_roles["Villager"],
        "n_hunter": 0,
    }
    expected_environment_fields = set(required_environment) | {
        "n_witch", "n_guard", "werewolf_reward", "village_reward"
    }
    _exact_fields(environment, expected_environment_fields, "environment")
    for name, expected in required_environment.items():
        if environment[name] != expected:
            raise ValueError(f"environment.{name} must be {expected}")
    if environment["n_witch"] not in (0, 1) or environment["n_guard"] not in (0, 1):
        raise ValueError("environment witch/guard counts must be zero or one")
    if environment["n_witch"] + environment["n_guard"] != 1:
        raise ValueError("environment requires exactly one witch or guard")

    output = config["output"]
    _exact_fields(output, {"samples", "failures", "logs", "overwrite"}, "output")
    for name in ("samples", "failures", "logs"):
        _text(output[name], f"output.{name}")
    if type(output["overwrite"]) is not bool:
        raise ValueError("output.overwrite must be boolean")
    return True


def normalize_runtime_config(config):
    validate_runtime_config(config)
    return deepcopy(config)


def resolve_guess_config(config, profile):
    """Resolve the optional global guess override against a gameplay profile."""

    guess = config["guess"]
    if guess["backend"] is None:
        return {"backend": profile["backend"], "model": profile["model"]}
    return deepcopy(guess)


def validate_run_id(run_id):
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must match ^game_[0-9]{3,}$")
    return run_id


def _collection_root(path, name):
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError(f"{name} must be a non-empty path")
    return Path(path).expanduser().resolve()


def resolve_collection_output(
    config, *, run_id, data_dir="data", log_dir="logs"
):
    """Resolve the sole one-run/one-game collection layout."""

    resolved = normalize_runtime_config(config)
    run_id = validate_run_id(run_id)
    data_root = _collection_root(data_dir, "data-dir")
    log_root = _collection_root(log_dir, "log-dir")
    if data_root == log_root:
        raise ValueError("data-dir and log-dir must be different directories")
    data_run_dir = data_root / run_id
    log_run_dir = log_root / run_id
    paths = RunPaths(
        run_id=run_id,
        data_root=data_root,
        log_root=log_root,
        data_run_dir=data_run_dir,
        log_run_dir=log_run_dir,
        samples_path=data_run_dir / f"{run_id}.samples.jsonl",
        audit_path=data_run_dir / f"{run_id}.audit.json",
        failures_path=data_run_dir / f"{run_id}.failures.jsonl",
        game_log_path=log_run_dir / f"{run_id}.game_log.json",
        parser_failures_path=(
            log_run_dir / f"{run_id}.parser_failures.json"
        ),
    )
    resolved["output"] = {
        "samples": str(paths.samples_path),
        "failures": str(paths.failures_path),
        "logs": str(paths.log_run_dir),
        "overwrite": False,
    }
    return resolved, paths


def build_prompt_runtime_metadata(config):
    """Describe the configured models separately from stable prompt hashes."""

    validate_runtime_config(config)
    profiles = config["agents"]["profiles"]
    assigned_profiles = sorted(
        {
            config["agents"]["werewolf_profile"],
            config["agents"]["village_profile"],
        }
    )
    gameplay_profiles = {}
    belief_profiles = {}
    for profile_name in assigned_profiles:
        profile = profiles[profile_name]
        guess = resolve_guess_config(config, profile)
        gameplay_profiles[profile_name] = {
            "backend": profile["backend"],
            "model": profile["model"],
            "temperature": profile["temperature"],
        }
        belief_profiles[profile_name] = {
            "backend": guess["backend"],
            "model": guess["model"],
            "temperature": 0.0,
        }
    return {
        "gameplay_profiles": gameplay_profiles,
        "belief_profiles": belief_profiles,
        "parser": {
            "backend": config["parser"]["backend"],
            "model": config["parser"]["model"],
            "temperature": 0.0,
        },
    }
