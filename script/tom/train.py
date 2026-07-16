"""Train a 21-class ToM model from a strict YAML config."""

import argparse
import json
from pathlib import Path

import yaml

from werewolf.tom.training import train_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tom/first_order.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    print(json.dumps(train_from_config(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
