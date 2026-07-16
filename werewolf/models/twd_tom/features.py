from collections.abc import Callable

import torch

from werewolf.encoding.event_encoder import (
    EVENT_TOKEN_FIELDS,
    encode_observation_game_log,
)


def normalize_observer_id(sample: dict) -> int:
    if not isinstance(sample, dict):
        return 0
    key = "observer_id" if "observer_id" in sample else "observer"
    value = sample.get(key)
    return value if type(value) is int and 1 <= value <= 7 else 0


class TWDToMFeatureBuilder:
    def __init__(
        self,
        event_encoder: Callable[[dict], list[dict]] | None = None,
        max_seq_len: int = 256,
        pad_token_id: int = 0,
        device: torch.device | str | None = None,
    ):
        if (
            not isinstance(max_seq_len, int)
            or isinstance(max_seq_len, bool)
            or max_seq_len <= 0
        ):
            raise ValueError("max_seq_len must be a positive integer")
        if pad_token_id != 0:
            raise ValueError("pad_token_id must be 0 in the first version")

        self.event_encoder = (
            encode_observation_game_log
            if event_encoder is None
            else event_encoder
        )
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.device = device

    def encode_observation(self, observation):
        return self.encode_batch([observation])

    def encode_batch(self, observations):
        sequences = []
        observer_ids = []
        for observation in observations:
            encoded_events = self.event_encoder(observation)
            sequence = [
                [int(event[field]) for field in EVENT_TOKEN_FIELDS]
                for event in encoded_events
            ]
            sequences.append(sequence[-self.max_seq_len :])
            observer_ids.append(normalize_observer_id(observation))

        batch_size = len(sequences)
        sequence_length = max(
            1,
            max((len(sequence) for sequence in sequences), default=0),
        )
        event_tokens = torch.zeros(
            batch_size,
            sequence_length,
            len(EVENT_TOKEN_FIELDS),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.long,
            device=self.device,
        )
        observer_id = torch.tensor(
            observer_ids,
            dtype=torch.long,
            device=self.device,
        )

        for index, sequence in enumerate(sequences):
            if not sequence:
                continue
            sequence_tensor = torch.tensor(
                sequence,
                dtype=torch.long,
                device=self.device,
            )
            event_tokens[index, : len(sequence)] = sequence_tensor
            attention_mask[index, : len(sequence)] = 1

        return {
            "event_tokens": event_tokens,
            "attention_mask": attention_mask,
            "observer_id": observer_id,
        }
