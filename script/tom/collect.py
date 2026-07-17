"""Collect ``tom.v1_1`` samples from complete games."""

import argparse
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

from werewolf.backends import BackendError, load_named_backends
from werewolf.game_rules import NUM_PLAYERS
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


def _validate_new_run_directories(paths):
    for root, name in (
        (paths.data_root, "data-dir"),
        (paths.log_root, "log-dir"),
    ):
        if root.exists() and not root.is_dir():
            raise NotADirectoryError(f"{name} is not a directory: {root}")
    conflicts = [
        path
        for path in (paths.data_run_dir, paths.log_run_dir)
        if path.exists()
    ]
    if conflicts:
        rendered = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(f"run directory already exists: {rendered}")
    for run_dir in (paths.data_run_dir, paths.log_run_dir):
        ancestor = _writable_ancestor(run_dir.parent)
        if not os.access(ancestor, os.W_OK):
            raise PermissionError(
                f"run directory parent is not writable via {ancestor}: "
                f"{run_dir.parent}"
            )


def _create_run_directories(paths):
    roots = tuple(dict.fromkeys((paths.data_root, paths.log_root)))
    try:
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"failed to create collection roots: {exc}") from exc
    created = []
    try:
        for run_dir in (paths.data_run_dir, paths.log_run_dir):
            run_dir.mkdir(exist_ok=False)
            created.append(run_dir)
    except OSError as exc:
        for run_dir in reversed(created):
            run_dir.rmdir()
        raise OSError(f"failed to create run directories: {exc}") from exc


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _initialize_run_files(paths):
    paths.samples_path.touch(exist_ok=False)
    paths.failures_path.touch(exist_ok=False)
    _write_json(paths.game_log_path, [])
    for player_id in range(1, NUM_PLAYERS + 1):
        paths.player_log_path(player_id).touch(exist_ok=False)


def _write_runtime_logs(paths, environment):
    events = [] if environment is None else environment.events
    _write_json(paths.game_log_path, events)
    parser_failures = [] if environment is None else environment.parser_failures
    if parser_failures:
        _write_json(paths.parser_failures_path, parser_failures)


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


def _safe_runtime_error(exc):
    error_type = type(exc).__name__
    message = str(exc)
    if isinstance(exc, BackendError) or message.startswith(
        ("gameplay action generation failed:", "rollout exceeded max_steps=")
    ):
        safe_message = " ".join(message.split())[:1000]
    else:
        safe_message = "collection rollout failed"
    return error_type, safe_message


def collect_from_config(
    config,
    *,
    run_id,
    games=None,
    data_dir="data",
    log_dir="logs",
    backends=None,
    env=None,
):
    config = deepcopy(config)
    if games is not None:
        config["games"] = games
    if config.get("games") != 1:
        raise ValueError(
            "one run_id represents exactly one game; use --games 1"
        )
    config, run_paths = resolve_collection_output(
        config,
        run_id=run_id,
        data_dir=data_dir,
        log_dir=log_dir,
    )
    _validate_new_run_directories(run_paths)
    preflight = preflight_collection(config, env=env)
    _create_run_directories(run_paths)
    _initialize_run_files(run_paths)
    environment = None
    try:
        backends = backends or load_named_backends(config)
        roles = shuffled_roles(config["environment"], config["seed"])
        environment, agents = build_collection_runtime(
            config,
            game_id=run_id,
            roles=roles,
            backends=backends,
        )
        game_result = rollout(
            environment,
            agents,
            roles,
            max_steps=config["max_steps"],
        )
    except Exception as exc:
        _write_runtime_logs(run_paths, environment)
        samples = _read_jsonl(run_paths.samples_path)
        failures = _read_jsonl(run_paths.failures_path)
        error_type, error_message = _safe_runtime_error(exc)
        audit = build_audit_report(
            samples,
            failures,
            game_ids=[run_id],
            collection_status="failed",
            completed_games=0,
            failed_game_id=run_id,
            runtime_error_type=error_type,
            runtime_error_message=error_message,
        )
        _write_audit_atomically(run_paths.audit_path, audit)
        raise
    _write_runtime_logs(run_paths, environment)
    results = [{"game_id": run_id, "roles": roles, **game_result}]
    samples = _read_jsonl(run_paths.samples_path)
    failures = _read_jsonl(run_paths.failures_path)
    audit = build_audit_report(
        samples,
        failures,
        game_ids=[run_id],
    )
    _write_audit_atomically(run_paths.audit_path, audit)
    assert_audit_passes(audit)
    return {
        "games": results,
        "preflight": preflight,
        "run_id": run_id,
        "data_run_dir": str(run_paths.data_run_dir),
        "log_run_dir": str(run_paths.log_run_dir),
        "samples_path": str(run_paths.samples_path),
        "failures_path": str(run_paths.failures_path),
        "audit_path": str(run_paths.audit_path),
        "game_log_path": str(run_paths.game_log_path),
        "audit": audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tom/collect.yaml")
    parser.add_argument("--games", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    result = collect_from_config(
        config,
        games=args.games,
        run_id=args.run_id,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
    )
    audit = result["audit"]
    print(
        json.dumps(
            {
                "games": len(result["games"]),
                "run_id": result["run_id"],
                "data_run_dir": result["data_run_dir"],
                "log_run_dir": result["log_run_dir"],
                "samples_path": result["samples_path"],
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
