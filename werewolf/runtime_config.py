"""Strict validation for the sole supported collection runtime schema."""

from copy import deepcopy
from pathlib import Path

from werewolf.game_rules import NUM_PLAYERS, NUM_WEREWOLVES, ROLE_DISTRIBUTIONS


RUNTIME_SCHEMA_VERSION = "runtime.v1"
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


def resolve_collection_output(config, output_dir=None):
    """Resolve the sole collection output layout without changing the schema."""

    resolved = normalize_runtime_config(config)
    if output_dir is None:
        samples = Path(resolved["output"]["samples"])
        return resolved, {
            "output_dir": samples.parent.resolve(),
            "samples": samples,
            "failures": Path(resolved["output"]["failures"]),
            "audit": samples.with_suffix(".audit.json"),
            "logs": Path(resolved["output"]["logs"]),
        }
    if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip():
        raise ValueError("output-dir must be a non-empty path")
    directory = Path(output_dir).expanduser().resolve()
    resolved["output"] = {
        "samples": str(directory / "samples.jsonl"),
        "failures": str(directory / "failures.jsonl"),
        "logs": str(directory / "logs"),
        "overwrite": False,
    }
    return resolved, {
        "output_dir": directory,
        "samples": directory / "samples.jsonl",
        "failures": directory / "failures.jsonl",
        "audit": directory / "samples.audit.json",
        "logs": directory / "logs",
    }


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
