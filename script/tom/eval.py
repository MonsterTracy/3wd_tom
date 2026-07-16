"""Evaluate a trained ToM checkpoint."""

import argparse
import json
from pathlib import Path

import yaml

from werewolf.tom.evaluation import evaluate_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tom/evaluate.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    print(json.dumps(evaluate_from_config(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
