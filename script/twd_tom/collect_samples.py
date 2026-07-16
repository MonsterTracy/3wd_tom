#!/usr/bin/env python3
"""Collect TWD-ToM rollout samples by running multiple Werewolf games.

Typical command:
    PYTHONPATH=. python script/twd_tom/collect_samples.py \
      --num_games 30 \
      --config configs/twd_tom_deepseek_only_debug.yaml \
      --output_dir logs/twd_tom_v05_deepseek_debug/game_new \
      --samples_path data/twd_tom/debug/game_new.jsonl \
      --overwrite
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect TWD-ToM rollout samples by repeatedly running run_random.py."
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=5,
        help="Number of games to collect.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/twd_tom_collect.yaml",
        help="Config file for run_random.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/twd_tom/first5",
        help="Directory for per-game logs.",
    )
    parser.add_argument(
        "--samples_path",
        type=str,
        default="data/twd_tom/first5/train_samples.jsonl",
        help="Shared JSONL sample output path. Existing file will be appended unless --overwrite is set.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Starting game index.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing samples_path before collection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_games <= 0:
        raise ValueError("--num_games must be positive")

    project_root = Path(__file__).resolve().parents[2]
    run_random = project_root / "run_random.py"
    config_path = project_root / args.config
    output_dir = project_root / args.output_dir
    samples_path = project_root / args.samples_path

    if not run_random.exists():
        raise FileNotFoundError(f"run_random.py not found: {run_random}")

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and samples_path.exists():
        samples_path.unlink()

    for offset in range(args.num_games):
        game_idx = args.start_index + offset
        game_dir = output_dir / f"game_{game_idx}"

        print(f"===== collecting game {game_idx} / {args.start_index + args.num_games - 1} =====", flush=True)

        cmd = [
            sys.executable,
            "-u",
            str(run_random),
            "--config",
            str(config_path),
            "--log_save_path",
            str(game_dir),
            "--twd_tom_sample_path",
            str(samples_path),
        ]

        subprocess.run(cmd, cwd=str(project_root), check=True)

    print("===== collection finished =====", flush=True)
    print(f"samples_path: {samples_path}", flush=True)


if __name__ == "__main__":
    main()
