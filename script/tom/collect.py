"""Collect ``tom.v1_1`` samples from complete games."""

import argparse
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

from werewolf.backends import load_named_backends
from werewolf.runtime import build_collection_runtime, rollout, shuffled_roles
from werewolf.runtime_config import (
    resolve_collection_output,
    resolve_guess_config,
    validate_runtime_config,
)
from werewolf.tom.collection import assert_audit_passes, build_audit_report


def _writable_ancestor(path):
    candidate = Path(path).resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _validate_new_output_directory(path):
    path = Path(path)
    if path.exists():
        if path.is_file():
            raise NotADirectoryError(f"output-dir is a file: {path}")
        raise FileExistsError(
            f"output-dir already exists; choose a new directory: {path}"
        )
    ancestor = _writable_ancestor(path.parent)
    if not os.access(ancestor, os.W_OK):
        raise PermissionError(
            f"output-dir parent is not writable via {ancestor}: {path.parent}"
        )


def _create_output_directory(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise OSError(f"failed to create output-dir {path}: {exc}") from exc


def _write_audit_atomically(path, audit):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        json.dump(audit, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        temporary_path = Path(output.name)
    temporary_path.replace(path)


def preflight_collection(config, *, env=None):
    """Validate a production collection run without constructing network clients."""

    validate_runtime_config(config)
    env = os.environ if env is None else env
    profiles = config["agents"]["profiles"]
    assigned_profiles = {
        config["agents"]["werewolf_profile"],
        config["agents"]["village_profile"],
    }
    used_backends = {config["parser"]["backend"]}
    resolved_guesses = {}
    for profile_name in assigned_profiles:
        profile = profiles[profile_name]
        used_backends.add(profile["backend"])
        resolved_guess = resolve_guess_config(config, profile)
        resolved_guesses[profile_name] = resolved_guess
        used_backends.add(resolved_guess["backend"])
    for backend_name in used_backends:
        backend = config["backends"][backend_name]
        if backend["type"] != "openai_compatible":
            raise ValueError("production collection cannot use a mock backend")
        if not backend["base_url"].strip():
            raise ValueError(f"backend {backend_name}.base_url is required")
        key_name = backend["api_key_env"]
        if not isinstance(env.get(key_name), str) or not env[key_name].strip():
            raise ValueError(f"API key environment variable {key_name} is required")
    for output_name in ("samples", "failures", "logs"):
        ancestor = _writable_ancestor(config["output"][output_name])
        if not os.access(ancestor, os.W_OK):
            raise ValueError(
                f"output.{output_name} is not writable via {ancestor}"
            )
    return {
        "backend_names": sorted(used_backends),
        "parser": dict(config["parser"]),
        "gameplay": {
            name: {
                "backend": profiles[name]["backend"],
                "model": profiles[name]["model"],
            }
            for name in sorted(assigned_profiles)
        },
        "guess": {name: resolved_guesses[name] for name in sorted(resolved_guesses)},
    }


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append(None)
    return records


def collect_from_config(
    config, *, games=None, output_dir=None, backends=None, env=None
):
    config = deepcopy(config)
    if games is not None:
        if games < 1:
            raise ValueError("games must be positive")
        config["games"] = games
    config, output_paths = resolve_collection_output(config, output_dir)
    if output_dir is not None:
        _validate_new_output_directory(output_paths["output_dir"])
    preflight = preflight_collection(config, env=env)
    if output_dir is not None:
        _create_output_directory(output_paths["output_dir"])
    samples_path = output_paths["samples"]
    audit_path = output_paths["audit"]
    output_files = [
        samples_path,
        output_paths["failures"],
        audit_path,
    ]
    for path in output_files:
        if path.exists() and not config["output"]["overwrite"]:
            raise FileExistsError(f"refusing to append to existing output: {path}")
        if path.exists():
            path.unlink()
    backends = backends or load_named_backends(config)
    results = []
    for game_index in range(config["games"]):
        game_config = deepcopy(config)
        game_config["seed"] = config["seed"] + game_index
        game_id = f"game_{game_config['seed']:06d}"
        log_directory = Path(game_config["output"]["logs"]) / game_id
        if log_directory.exists() and not game_config["output"]["overwrite"]:
            raise FileExistsError(
                f"refusing to append to existing game logs: {log_directory}"
            )
        if log_directory.exists():
            shutil.rmtree(log_directory)
        roles = shuffled_roles(game_config["environment"], game_config["seed"])
        environment, agents = build_collection_runtime(
            game_config,
            game_id=game_id,
            roles=roles,
            backends=backends,
        )
        result = rollout(
            environment,
            agents,
            roles,
            max_steps=game_config["max_steps"],
        )
        results.append({"game_id": game_id, "roles": roles, **result})
    samples = _read_jsonl(samples_path)
    failures = _read_jsonl(config["output"]["failures"])
    audit = build_audit_report(
        samples,
        failures,
        game_ids=[result["game_id"] for result in results],
    )
    _write_audit_atomically(audit_path, audit)
    assert_audit_passes(audit)
    return {
        "games": results,
        "preflight": preflight,
        "output_dir": str(output_paths["output_dir"]),
        "audit_path": str(audit_path),
        "audit": audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tom/collect.yaml")
    parser.add_argument("--games", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    result = collect_from_config(
        config, games=args.games, output_dir=args.output_dir
    )
    audit = result["audit"]
    print(
        json.dumps(
            {
                "games": len(result["games"]),
                "output_dir": result["output_dir"],
                "audit_path": result["audit_path"],
                "unique_belief_elicitations": audit["unique_belief_elicitations"],
                "successful_guesses": audit["successful_guesses"],
                "failed_guesses": audit["failed_guesses"],
                "training_samples": (
                    audit["first_order_samples"]
                    + audit["second_order_public_samples"]
                    + audit["second_order_wolf_samples"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
