from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from werewolf.encoding.dialogue_actions import (
    CAMP2ID,
    CERTAINTY2ID,
    EVENT_TYPE2ID,
    PHASE2ID,
    POLARITY2ID,
    PREDICATE2ID,
    ROLE2ID,
)

try:
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block

    HAS_TRANSFORMERS = True
except ImportError:
    GPT2Config = None
    GPT2Block = None
    HAS_TRANSFORMERS = False


def _vocab_size(mapping):
    return max(mapping.values()) + 1


@dataclass
class ToMBackboneConfig:
    num_players: int = 7
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    max_seq_len: int = 256
    max_day: int = 10
    backbone_type: str = "transformer"
    use_observer_id: bool = True
    intermediate_size: int | None = None
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return hidden_states * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("LLaMA head_dim must be even for RoPE")
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        positions = torch.arange(
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.outer(positions, self.inv_freq.to(device))
        cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def _apply_rotary_pos_emb(hidden_states, cos, sin):
    even = hidden_states[..., 0::2]
    odd = hidden_states[..., 1::2]
    rotated = torch.empty_like(hidden_states)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


class LlamaSelfAttention(nn.Module):
    def __init__(self, config: ToMBackboneConfig):
        super().__init__()
        if config.d_model % config.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.num_heads = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            theta=float(config.rope_theta),
        )
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        return hidden_states.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_heads(self, hidden_states):
        batch_size, _, seq_len, _ = hidden_states.shape
        return hidden_states.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))

        cos, sin = self.rotary_emb(
            seq_len,
            hidden_states.device,
            query.dtype,
        )
        query = _apply_rotary_pos_emb(query, cos, sin)
        key = _apply_rotary_pos_emb(key, cos, sin)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(
            causal_mask[None, None, :, :],
            torch.finfo(scores.dtype).min,
        )

        if attention_mask is not None:
            valid_keys = attention_mask.to(
                device=hidden_states.device,
                dtype=torch.bool,
            )[:, None, None, :]
            scores = scores.masked_fill(
                ~valid_keys,
                torch.finfo(scores.dtype).min,
            )

        attn_weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self.o_proj(self._merge_heads(attn_output))
        if attention_mask is not None:
            valid_queries = attention_mask.to(
                device=hidden_states.device,
                dtype=attn_output.dtype,
            )[:, :, None]
            attn_output = attn_output * valid_queries
        return attn_output


class LlamaMLP(nn.Module):
    def __init__(self, config: ToMBackboneConfig):
        super().__init__()
        intermediate_size = (
            int(config.intermediate_size)
            if config.intermediate_size is not None
            else 4 * config.d_model
        )
        self.gate_proj = nn.Linear(
            config.d_model,
            intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.d_model,
            intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            config.d_model,
            bias=False,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(
            hidden_states
        )
        return self.dropout(self.down_proj(hidden_states))


class LlamaDecoderBlock(nn.Module):
    def __init__(self, config: ToMBackboneConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.d_model)
        self.self_attn = LlamaSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.d_model)
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


class GPTNeoXSelfAttention(nn.Module):
    def __init__(self, config: ToMBackboneConfig):
        super().__init__()
        if config.d_model % config.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.num_heads = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.query_key_value = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=True,
        )
        self.dense = nn.Linear(config.d_model, config.d_model, bias=True)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            theta=float(config.rope_theta),
        )
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        return hidden_states.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_heads(self, hidden_states):
        batch_size, _, seq_len, _ = hidden_states.shape
        return hidden_states.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        )

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        cos, sin = self.rotary_emb(
            seq_len,
            hidden_states.device,
            query.dtype,
        )
        query = _apply_rotary_pos_emb(query, cos, sin)
        key = _apply_rotary_pos_emb(key, cos, sin)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(
            causal_mask[None, None, :, :],
            torch.finfo(scores.dtype).min,
        )

        if attention_mask is not None:
            valid_keys = attention_mask.to(
                device=hidden_states.device,
                dtype=torch.bool,
            )[:, None, None, :]
            scores = scores.masked_fill(
                ~valid_keys,
                torch.finfo(scores.dtype).min,
            )

        attn_weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self.dense(self._merge_heads(attn_output))
        if attention_mask is not None:
            valid_queries = attention_mask.to(
                device=hidden_states.device,
                dtype=attn_output.dtype,
            )[:, :, None]
            attn_output = attn_output * valid_queries
        return attn_output


class GPTNeoXMLP(nn.Module):
    def __init__(self, config: ToMBackboneConfig):
        super().__init__()
        intermediate_size = (
            int(config.intermediate_size)
            if config.intermediate_size is not None
            else 4 * config.d_model
        )
        self.dense_h_to_4h = nn.Linear(
            config.d_model,
            intermediate_size,
        )
        self.dense_4h_to_h = nn.Linear(
            intermediate_size,
            config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states):
        hidden_states = F.gelu(self.dense_h_to_4h(hidden_states))
        hidden_states = self.dense_4h_to_h(hidden_states)
        return self.dropout(hidden_states)


class GPTNeoXBlock(nn.Module):
    def __init__(self, config: ToMBackboneConfig):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.d_model)
        self.attention = GPTNeoXSelfAttention(config)
        self.mlp = GPTNeoXMLP(config)

    def forward(self, hidden_states, attention_mask=None):
        normed = self.input_layernorm(hidden_states)
        attention_output = self.attention(
            normed,
            attention_mask=attention_mask,
        )
        mlp_output = self.mlp(normed)
        return hidden_states + attention_output + mlp_output


class ToMBackbone(nn.Module):
    def __init__(self, config: ToMBackboneConfig | None = None):
        super().__init__()
        self.config = config or ToMBackboneConfig()

        self.event_type_embedding = nn.Embedding(
            _vocab_size(EVENT_TYPE2ID), self.config.d_model
        )
        self.speaker_embedding = nn.Embedding(
            self.config.num_players + 1, self.config.d_model
        )
        self.target_embedding = nn.Embedding(
            self.config.num_players + 1, self.config.d_model
        )
        self.observer_emb = nn.Embedding(
            self.config.num_players + 1,
            self.config.d_model,
            padding_idx=0,
        )
        self.predicate_embedding = nn.Embedding(
            _vocab_size(PREDICATE2ID), self.config.d_model
        )
        self.role_embedding = nn.Embedding(
            _vocab_size(ROLE2ID), self.config.d_model
        )
        self.camp_embedding = nn.Embedding(
            _vocab_size(CAMP2ID), self.config.d_model
        )
        self.polarity_embedding = nn.Embedding(
            _vocab_size(POLARITY2ID), self.config.d_model
        )
        self.certainty_embedding = nn.Embedding(
            _vocab_size(CERTAINTY2ID), self.config.d_model
        )
        self.phase_embedding = nn.Embedding(
            _vocab_size(PHASE2ID), self.config.d_model
        )
        self.day_embedding = nn.Embedding(
            self.config.max_day + 1, self.config.d_model
        )
        self.position_embedding = nn.Embedding(
            self.config.max_seq_len, self.config.d_model
        )
        self.layer_norm = nn.LayerNorm(self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)

        if self.config.backbone_type not in (
            "transformer",
            "boe_mlp",
            "gru",
            "llama",
            "gpt_neox",
        ):
            raise ValueError(
                "backbone_type must be 'transformer', 'boe_mlp', "
                "'gru', 'llama', or 'gpt_neox'"
            )

        self.uses_transformers = (
            self.config.backbone_type == "transformer"
            and HAS_TRANSFORMERS
        )
        if self.uses_transformers:
            transformer_config = GPT2Config(
                n_embd=self.config.d_model,
                n_head=self.config.n_head,
                n_layer=self.config.n_layer,
                n_positions=self.config.max_seq_len,
                n_ctx=self.config.max_seq_len,
                resid_pdrop=self.config.dropout,
                embd_pdrop=self.config.dropout,
                attn_pdrop=self.config.dropout,
                use_cache=False,
            )
            if hasattr(transformer_config, "_attn_implementation"):
                try:
                    transformer_config._attn_implementation = "eager"
                except (AttributeError, TypeError):
                    pass
            elif hasattr(transformer_config, "attn_implementation"):
                try:
                    transformer_config.attn_implementation = "eager"
                except (AttributeError, TypeError):
                    pass

            blocks = []
            for layer_idx in range(self.config.n_layer):
                try:
                    block = GPT2Block(
                        transformer_config,
                        layer_idx=layer_idx,
                    )
                except TypeError:
                    block = GPT2Block(transformer_config)
                blocks.append(block)
            self.blocks = nn.ModuleList(blocks)
        elif self.config.backbone_type == "transformer":
            # Fallback for environments without transformers.
            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=self.config.d_model,
                        nhead=self.config.n_head,
                        dropout=self.config.dropout,
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(self.config.n_layer)
                ]
            )
        elif self.config.backbone_type == "llama":
            self.blocks = nn.ModuleList(
                [
                    LlamaDecoderBlock(self.config)
                    for _ in range(self.config.n_layer)
                ]
            )
            self.final_norm = RMSNorm(self.config.d_model)
        elif self.config.backbone_type == "gpt_neox":
            self.blocks = nn.ModuleList(
                [
                    GPTNeoXBlock(self.config)
                    for _ in range(self.config.n_layer)
                ]
            )
            self.final_norm = nn.LayerNorm(self.config.d_model)
        else:
            self.blocks = nn.ModuleList()
            if self.config.backbone_type == "boe_mlp":
                self.boe_mlp = nn.Sequential(
                    nn.Linear(self.config.d_model, self.config.d_model),
                    nn.GELU(),
                    nn.Dropout(self.config.dropout),
                    nn.Linear(self.config.d_model, self.config.d_model),
                    nn.GELU(),
                )
            else:
                self.gru = nn.GRU(
                    input_size=self.config.d_model,
                    hidden_size=self.config.d_model,
                    batch_first=True,
                )

        self.wolf_head = nn.Linear(
            self.config.d_model,
            self.config.num_players,
        )

    def forward(self, event_tokens, attention_mask=None, observer_id=None):
        if event_tokens.ndim != 3:
            raise ValueError("event_tokens must have shape [B, T, 10]")

        batch_size, seq_len, field_count = event_tokens.shape
        if field_count != 10:
            raise ValueError("event_tokens must have 10 fields")
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if attention_mask is not None and attention_mask.shape != (
            batch_size,
            seq_len,
        ):
            raise ValueError("attention_mask must have shape [B, T]")
        if observer_id is None:
            observer_id = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=event_tokens.device,
            )
        elif observer_id.shape != (batch_size,):
            raise ValueError("observer_id must have shape [B]")
        else:
            observer_id = observer_id.to(
                device=event_tokens.device,
                dtype=torch.long,
            )

        positions = torch.arange(
            seq_len,
            device=event_tokens.device,
        ).unsqueeze(0)
        day_ids = event_tokens[..., 9].clamp(max=self.config.max_day)
        observer_hidden = (
            self.observer_emb(observer_id).unsqueeze(1)
            if self.config.use_observer_id
            else 0
        )
        hidden_states = (
            self.event_type_embedding(event_tokens[..., 0])
            + self.speaker_embedding(event_tokens[..., 1])
            + self.target_embedding(event_tokens[..., 2])
            + observer_hidden
            + self.predicate_embedding(event_tokens[..., 3])
            + self.role_embedding(event_tokens[..., 4])
            + self.camp_embedding(event_tokens[..., 5])
            + self.polarity_embedding(event_tokens[..., 6])
            + self.certainty_embedding(event_tokens[..., 7])
            + self.phase_embedding(event_tokens[..., 8])
            + self.day_embedding(day_ids)
            + self.position_embedding(positions)
        )
        hidden_states = self.dropout(self.layer_norm(hidden_states))

        if self.uses_transformers:
            additive_mask = None
            if attention_mask is not None:
                valid_tokens = attention_mask.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )[:, None, None, :]
                additive_mask = (1.0 - valid_tokens) * torch.finfo(
                    hidden_states.dtype
                ).min

            for block in self.blocks:
                block_output = block(
                    hidden_states,
                    attention_mask=additive_mask,
                    use_cache=False,
                )
                hidden_states = (
                    block_output[0]
                    if isinstance(block_output, tuple)
                    else block_output
                )
        elif self.config.backbone_type == "boe_mlp":
            if attention_mask is None:
                valid_tokens = torch.ones(
                    batch_size,
                    seq_len,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            else:
                valid_tokens = attention_mask.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            prefix_sum = torch.cumsum(
                hidden_states * valid_tokens.unsqueeze(-1),
                dim=1,
            )
            prefix_count = torch.cumsum(valid_tokens, dim=1).clamp_min(1.0)
            hidden_states = self.boe_mlp(
                prefix_sum / prefix_count.unsqueeze(-1)
            )
        elif self.config.backbone_type == "gru":
            if attention_mask is None:
                valid_tokens = torch.ones(
                    batch_size,
                    seq_len,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
            else:
                valid_tokens = attention_mask.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            hidden_states, _ = self.gru(
                hidden_states * valid_tokens.unsqueeze(-1)
            )
            hidden_states = hidden_states * valid_tokens.unsqueeze(-1)
        elif self.config.backbone_type == "llama":
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                )
            hidden_states = self.final_norm(hidden_states)
        elif self.config.backbone_type == "gpt_neox":
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                )
            hidden_states = self.final_norm(hidden_states)
        else:
            causal_mask = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                    device=hidden_states.device,
                ),
                diagonal=1,
            )
            padding_mask = None
            if attention_mask is not None:
                padding_mask = attention_mask.to(
                    device=hidden_states.device
                ).eq(0)

            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    src_mask=causal_mask,
                    src_key_padding_mask=padding_mask,
                )

        wolf_logits = self.wolf_head(hidden_states)
        wolf_prob = torch.sigmoid(wolf_logits)
        return {
            "hidden_states": hidden_states,
            "wolf_logits": wolf_logits,
            "wolf_prob": wolf_prob,
        }
