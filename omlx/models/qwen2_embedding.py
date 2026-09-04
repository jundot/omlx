# SPDX-License-Identifier: Apache-2.0
"""
Native Qwen2 embedding adapter for omlx.

Serves Qwen2-decoder embedding models (``architectures=["Qwen2ForCausalLM"]``,
``model_type="qwen2"``) such as ``jinaai/jina-code-embeddings-1.5b`` (causal)
and ``Alibaba-NLP/gte-Qwen2-1.5B-instruct`` (bidirectional), without depending
on mlx-embeddings, which has no ``qwen2`` module and raises
``ValueError("Model type qwen2 not supported.")`` for these checkpoints.

These models pool the *last* (mask-aware) token of the final hidden state and
L2-normalize, matching their SentenceTransformers configs
(``pooling_mode_lasttoken: true`` + a Normalize module). Mean pooling would
silently corrupt every vector, so last-token pooling is load-bearing here; the
``_extract_embeddings_array`` consumer does not normalize, so the returned
``text_embeds`` are already L2-normalized.

Deltas from the Qwen3 embedder: Qwen2 has no QK-norm, applies attention bias on
the q/k/v projections (``o_proj`` unbiased), and derives
``head_dim = hidden_size // num_attention_heads`` (no explicit config field).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base_model import (
    BaseModelArgs,
    BaseModelOutput,
    last_token_pool,
    normalize_embeddings,
)


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen2 embedding model configuration."""

    model_type: str = "qwen2"
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    intermediate_size: int = 8960
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 32768
    vocab_size: int = 151936

    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Attention direction. A plain Qwen2 decoder-embedder (jina-code) is
    # causal. Some Qwen2 embedders run the decoder *bidirectionally* and signal
    # it with ``is_causal: false`` in config (Alibaba's gte-Qwen2 family, which
    # ships a custom bidirectional modeling_qwen.py). Default causal; the config
    # flips it off when needed.
    is_causal: bool = True

    tie_word_embeddings: bool = False

    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    architectures: List[str] = field(
        default_factory=lambda: ["Qwen2ForCausalLM"]
    )

    def __post_init__(self):
        """Derive grouped-query and head dims that Qwen2 leaves implicit."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads


class Qwen2MLP(nn.Module):
    """SwiGLU MLP: SiLU(gate_proj(x)) * up_proj(x) -> down_proj."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    """
    Grouped-query attention for Qwen2.

    Unlike Qwen3 there is no query/key RMSNorm, and the q/k/v projections carry
    a bias (``o_proj`` does not). Grouped-query head expansion is handled by
    ``mx.fast.scaled_dot_product_attention``.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.rotary_emb = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.reshape(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        queries = self.rotary_emb(queries)
        keys = self.rotary_emb(keys)

        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=attention_mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, self.num_heads * self.head_dim
        )
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer (RMSNorm, residual, SwiGLU)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2Model(nn.Module):
    """Qwen2 transformer decoder stack."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _build_attention_mask(
        self,
        attention_mask: Optional[mx.array],
        seq_length: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        """
        Additive (batch, 1, seq, seq) mask combining key padding with the
        causal triangle (skipped for bidirectional embedders).

        Built as a single boolean keep-mask (query i may attend key j iff key j
        is a real token, and — when causal — ``j <= i``) mapped to one finite
        ``finfo.min`` fill. Using the dtype minimum rather than ``-inf`` keeps
        fully-masked rows (the leading pad positions under *left* padding)
        NaN-free: a uniform softmax over equal fills yields finite garbage at
        pad positions that we never pool, instead of ``NaN`` that ``0 * NaN``
        would propagate into real positions at the next layer. A single fill
        also avoids the additive ``2 * finfo.min`` overflow back to ``-inf``.
        """
        if self.config.is_causal:
            keep = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
            keep = keep[None, None]  # (1, 1, seq, seq)
        else:
            keep = mx.ones((1, 1, seq_length, seq_length), dtype=mx.bool_)
        if attention_mask is not None:
            key_keep = (attention_mask != 0)[:, None, None, :]  # (batch, 1, 1, seq)
            keep = keep & key_keep
        return mx.where(keep, 0.0, mx.finfo(dtype).min).astype(dtype)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        _, seq_length = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is not None and attention_mask.ndim != 2:
            # Already an additive (batch, 1, seq, seq) mask; use as-is.
            mask = attention_mask
        else:
            mask = self._build_attention_mask(
                attention_mask, seq_length, hidden_states.dtype
            )

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=mask)

        return self.norm(hidden_states)


class Model(nn.Module):
    """Qwen2 decoder wrapped for embedding generation (last-token + L2)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen2Model(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> BaseModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        elif attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} doesn't match "
                f"input_ids shape {input_ids.shape}"
            )

        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)

        # Mask-aware last-token pool, then L2-normalize. The embedding consumer
        # (_extract_embeddings_array) does not normalize, so text_embeds must be
        # unit-norm already.
        pooled_output = last_token_pool(last_hidden_state, attention_mask)
        text_embeds = normalize_embeddings(pooled_output)

        return BaseModelOutput(
            text_embeds=text_embeds, last_hidden_state=last_hidden_state
        )

    def sanitize(self, weights: dict) -> dict:
        """
        Map a HuggingFace Qwen2ForCausalLM checkpoint onto this module tree.

        Drops the unused LM head (and any precomputed rotary inverse-frequency
        buffers) and normalizes the transformer prefix to ``model.``.
        """
        sanitized_weights = {}
        for key, value in weights.items():
            if "lm_head.weight" in key:
                continue
            if "rotary_emb.inv_freq" in key:
                continue

            if key.startswith("transformer."):
                new_key = key.replace("transformer.", "model.", 1)
            elif not key.startswith("model.") and "." in key:
                new_key = f"model.{key}"
            else:
                new_key = key

            sanitized_weights[new_key] = value

        return sanitized_weights
