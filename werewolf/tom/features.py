"""Feature construction for strict ``tom.v1`` samples."""

import torch

from werewolf.events.encoder import EVENT_TOKEN_FIELDS, encode_events
from werewolf.tom.schemas import validate_sample


TASK_IDS = {"first_order": 1, "second_order": 2}
MODE_IDS = {
    "private_conditioned": 1,
    "public_only": 2,
    "wolf_conditioned": 3,
}


def sample_to_features(sample, *, include_first_order_private=True):
    validate_sample(sample)
    events = sample["events"]
    if sample["task"] == "first_order" and not include_first_order_private:
        events = [event for event in events if event["visibility"] == "public"]
    tokens = encode_events(events)
    if not tokens:
        tokens = [[0] * len(EVENT_TOKEN_FIELDS)]
    return {
        "event_tokens": torch.tensor(tokens, dtype=torch.long),
        "task_id": TASK_IDS[sample["task"]],
        "mode_id": MODE_IDS[sample["mode"]],
        "observer_id": sample["observer_id"] or 0,
        "modeler_id": sample["modeler_id"] or 0,
        "target_id": sample["target_id"] or 0,
        "output_mask": torch.tensor(sample["output_mask"], dtype=torch.bool),
        "label": sample["label_index"],
        "sample_id": sample["sample_id"],
    }


def collate_features(features):
    if not features:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(features)
    max_events = max(item["event_tokens"].shape[0] for item in features)
    width = len(EVENT_TOKEN_FIELDS)
    event_tokens = torch.zeros(batch_size, max_events, width, dtype=torch.long)
    event_mask = torch.zeros(batch_size, max_events, dtype=torch.bool)
    for row, item in enumerate(features):
        length = item["event_tokens"].shape[0]
        event_tokens[row, :length] = item["event_tokens"]
        event_mask[row, :length] = True
    scalar_fields = ("task_id", "mode_id", "observer_id", "modeler_id", "target_id")
    batch = {
        name: torch.tensor([item[name] for item in features], dtype=torch.long)
        for name in scalar_fields
    }
    batch.update(
        {
            "event_tokens": event_tokens,
            "event_mask": event_mask,
            "output_mask": torch.stack([item["output_mask"] for item in features]),
            "labels": torch.tensor([item["label"] for item in features], dtype=torch.long),
            "sample_ids": [item["sample_id"] for item in features],
        }
    )
    return batch
