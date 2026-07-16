import torch
from torch.utils.data import Dataset

from werewolf.models.twd_tom.features import (
    TWDToMFeatureBuilder,
    normalize_observer_id,
)
from werewolf.models.twd_tom.samples import normalize_alive_mask


class TWDToMDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        feature_builder: TWDToMFeatureBuilder | None = None,
    ):
        self.samples = samples
        self.feature_builder = (
            TWDToMFeatureBuilder()
            if feature_builder is None
            else feature_builder
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = self.feature_builder.encode_observation(
            sample["observation"]
        )
        return {
            "event_tokens": features["event_tokens"][0],
            "attention_mask": features["attention_mask"][0],
            "wolf_labels": torch.tensor(
                sample["wolf_labels"],
                dtype=torch.float32,
            ),
            "game_id": sample.get("game_id"),
            "observer_id": normalize_observer_id(sample),
            "phase": sample.get("phase"),
            "alive_mask": torch.tensor(
                normalize_alive_mask(sample.get("alive_mask")),
                dtype=torch.float32,
            ),
        }


def collate_twd_tom_samples(batch):
    batch_size = len(batch)
    max_length = max(
        item["event_tokens"].shape[0]
        for item in batch
    )
    token_width = batch[0]["event_tokens"].shape[1]
    event_tokens = batch[0]["event_tokens"].new_zeros(
        (batch_size, max_length, token_width)
    )
    attention_mask = batch[0]["attention_mask"].new_zeros(
        (batch_size, max_length)
    )

    for index, item in enumerate(batch):
        length = item["event_tokens"].shape[0]
        event_tokens[index, :length] = item["event_tokens"]
        attention_mask[index, :length] = item["attention_mask"]

    return {
        "event_tokens": event_tokens,
        "attention_mask": attention_mask,
        "observer_id": event_tokens.new_tensor(
            [item["observer_id"] for item in batch]
        ),
        "alive_mask": torch.stack(
            [item["alive_mask"] for item in batch]
        ),
        "wolf_labels": torch.stack(
            [item["wolf_labels"] for item in batch]
        ),
        "metadata": {
            "game_id": [item["game_id"] for item in batch],
            "observer_id": [item["observer_id"] for item in batch],
            "phase": [item["phase"] for item in batch],
            "alive_mask": [
                item["alive_mask"].tolist()
                for item in batch
            ],
        },
    }
