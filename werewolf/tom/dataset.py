"""Dataset loader for successful single-protocol ``tom.v1_1`` records."""

from copy import deepcopy
import json
from pathlib import Path

from torch.utils.data import Dataset

from werewolf.tom.features import sample_to_features
from werewolf.tom.schemas import validate_sample, validate_sample_collection


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
        records = []
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
                    records.append(record)
        try:
            validate_sample_collection(records)
        except ValueError as exc:
            raise ValueError(f"dataset identity/alignment error: {exc}") from exc
        protocol_ids = sorted(
            {record["prompt_protocol"]["protocol_id"] for record in records}
        )
        if len(protocol_ids) != 1:
            raise ValueError(
                f"dataset must contain one prompt protocol; found={protocol_ids}"
            )
        self.protocol_id = protocol_ids[0]
        self.prompt_protocol = deepcopy(records[0]["prompt_protocol"])
        self.records = [
            record
            for record in records
            if (task is None or record["task"] == task)
            and (mode is None or record["mode"] == mode)
        ]
        if not self.records:
            raise ValueError("dataset contains no matching successful tom.v1_1 samples")
        self.game_ids = frozenset(record["game_id"] for record in self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return sample_to_features(
            self.records[index],
            include_first_order_private=self.include_first_order_private,
        )
