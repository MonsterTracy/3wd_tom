"""Dataset loader that accepts only successful ``tom.v1`` JSONL records."""

import json
from pathlib import Path

from torch.utils.data import Dataset

from werewolf.tom.features import sample_to_features
from werewolf.tom.schemas import validate_sample


class ToMDataset(Dataset):
    def __init__(
        self,
        paths,
        *,
        task=None,
        mode=None,
        include_first_order_private=True,
    ):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.records = []
        self.include_first_order_private = include_first_order_private
        for path in paths:
            path = Path(path)
            with path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        validate_sample(record)
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
                    if task is not None and record["task"] != task:
                        continue
                    if mode is not None and record["mode"] != mode:
                        continue
                    self.records.append(record)
        if not self.records:
            raise ValueError("dataset contains no matching successful tom.v1 samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return sample_to_features(
            self.records[index],
            include_first_order_private=self.include_first_order_private,
        )
