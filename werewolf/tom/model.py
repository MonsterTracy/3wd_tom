"""Compact sequence models for 21-class first- and second-order ToM."""

from dataclasses import dataclass

import torch
from torch import nn

from werewolf.events.encoder import EVENT_TOKEN_FIELDS, VOCABULARIES
from werewolf.tom.pair_space import NUM_WOLF_PAIRS


ARCHITECTURES = ("transformer", "gru", "boe_mlp")


@dataclass(frozen=True)
class ToMModelConfig:
    architecture: str = "transformer"
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_events: int = 512
    max_day: int = 32
    use_target_embedding: bool = True

    def __post_init__(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {self.architecture}")
        if self.d_model <= 0 or self.num_layers <= 0 or self.max_events <= 0:
            raise ValueError("model dimensions must be positive")
        if self.architecture == "transformer" and self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")


def _field_vocab_sizes(config):
    sizes = []
    for field, vocabulary in zip(EVENT_TOKEN_FIELDS, VOCABULARIES):
        if vocabulary is not None:
            sizes.append(max(vocabulary.values()) + 1)
        elif field in ("speaker_id", "target_id"):
            sizes.append(8)
        elif field == "day_id":
            sizes.append(config.max_day)
        else:
            raise RuntimeError(f"no vocabulary size for {field}")
    return sizes


class ToMModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ToMModelConfig()
        d_model = self.config.d_model
        self.field_embeddings = nn.ModuleList(
            [nn.Embedding(size, d_model, padding_idx=0) for size in _field_vocab_sizes(self.config)]
        )
        self.position_embedding = nn.Embedding(self.config.max_events, d_model)
        if self.config.architecture == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=self.config.num_heads,
                dim_feedforward=4 * d_model,
                dropout=self.config.dropout,
                batch_first=True,
                norm_first=False,
            )
            self.sequence_model = nn.TransformerEncoder(layer, self.config.num_layers)
        elif self.config.architecture == "gru":
            self.sequence_model = nn.GRU(
                d_model,
                d_model,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            self.sequence_model = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(2 * d_model, d_model),
            )
        self.task_embedding = nn.Embedding(3, d_model, padding_idx=0)
        self.mode_embedding = nn.Embedding(4, d_model, padding_idx=0)
        self.player_embedding = nn.Embedding(8, d_model, padding_idx=0)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(d_model, NUM_WOLF_PAIRS),
        )

    def _encode_tokens(self, tokens):
        if tokens.shape[-1] != len(EVENT_TOKEN_FIELDS):
            raise ValueError("event token width does not match EVENT_TOKEN_FIELDS")
        if tokens.shape[1] > self.config.max_events:
            raise ValueError("event sequence exceeds max_events")
        for index, (field, embedding) in enumerate(
            zip(EVENT_TOKEN_FIELDS, self.field_embeddings)
        ):
            values = tokens[..., index]
            if values.min() < 0 or values.max() >= embedding.num_embeddings:
                raise ValueError(f"event token for {field} is outside its vocabulary")
        embedded = torch.stack(
            [embedding(tokens[..., index]) for index, embedding in enumerate(self.field_embeddings)]
        ).sum(dim=0)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return embedded + self.position_embedding(positions)[None, :, :]

    def _pool(self, encoded, event_mask):
        if self.config.architecture == "transformer":
            encoded = self.sequence_model(encoded, src_key_padding_mask=~event_mask)
            weights = event_mask.unsqueeze(-1).to(encoded.dtype)
            return (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        if self.config.architecture == "gru":
            encoded, _ = self.sequence_model(encoded)
            last_index = event_mask.sum(1).clamp_min(1) - 1
            return encoded[torch.arange(encoded.shape[0], device=encoded.device), last_index]
        weights = event_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.sequence_model(pooled)

    def forward(self, batch):
        if batch["event_mask"].shape != batch["event_tokens"].shape[:2]:
            raise ValueError("event_mask must match the event sequence")
        if not batch["event_mask"].any(dim=1).all():
            raise ValueError("every sample must contain at least one event token")
        encoded = self._encode_tokens(batch["event_tokens"])
        state = self._pool(encoded, batch["event_mask"])
        state = state + self.task_embedding(batch["task_id"])
        state = state + self.mode_embedding(batch["mode_id"])
        state = state + self.player_embedding(batch["observer_id"])
        state = state + self.player_embedding(batch["modeler_id"])
        if self.config.use_target_embedding:
            state = state + self.player_embedding(batch["target_id"])
        logits = self.output(state)
        output_mask = batch["output_mask"]
        if output_mask.shape != logits.shape:
            raise ValueError("output_mask shape must match pair logits")
        if not output_mask.any(dim=1).all():
            raise ValueError("every sample must retain at least one output class")
        return logits.masked_fill(~output_mask, torch.finfo(logits.dtype).min)
