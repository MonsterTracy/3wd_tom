"""Collect ``tom.v1`` samples from complete games."""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path

import yaml

from werewolf.backends import load_named_backends
from werewolf.runtime import build_collection_runtime, rollout, shuffled_roles
from werewolf.runtime_config import validate_runtime_config


def collect_from_config(config, *, games=None, backends=None):
    validate_runtime_config(config)
    config = deepcopy(config)
    if games is not None:
        if games < 1:
            raise ValueError("games must be positive")
        config["games"] = games
    output_paths = [Path(config["output"][name]) for name in ("samples", "failures")]
    for path in output_paths:
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
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tom/collect.yaml")
    parser.add_argument("--games", type=int)
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    print(json.dumps(collect_from_config(config, games=args.games), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
