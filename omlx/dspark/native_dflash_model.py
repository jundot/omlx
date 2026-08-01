"""DFlash block-diffusion drafter (MLX) — vendored from z-lab/dflash (MIT).

Upstream: https://github.com/z-lab/dflash  (file: dflash/model_mlx.py)
Paper:    DFlash: Block Diffusion for Flash Speculative Decoding — Chen et al.,
          arXiv:2602.06036.

MIT License. Copyright (c) 2026 Z Lab.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

Only the drafter model primitives are adapted here. oMLX's native Scheduler owns
generation, verification, cache rollback, streaming and statistics; z-lab's server
and generation loop are intentionally not included.

Architecture (differs from the DSpark drafter in ``model.py``): a Qwen3-style backbone
(silu MLP, separate v_proj, per-head q/k RMSNorm, default RoPE, sliding-window attention
on some layers) that **reuses the target model's embed_tokens + lm_head** (tied), consumes
a multi-layer fused target-hidden context via EAGLE-style KV injection, and predicts a
whole block of mask positions in a single parallel (block-diffusion) pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    target_hidden_size: int | None = None
    draft_vocab_size: int | None = None
    markov_rank: int = 0
    confidence_head_with_markov: bool = False
    enable_confidence_head: bool = False
    mask_token_id: int = 0
    rope_scaling: dict[str, Any] | None = None
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    sliding_window: int | None = None
    final_logit_softcapping: float | None = None


def _build_rope(head_dim, rope_theta, max_position_embeddings, rope_scaling):
    return initialize_rope(
        dims=head_dim,
        base=rope_theta,
        traditional=False,
        scaling_config=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim**-0.5
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.q_proj = nn.Linear(dim, n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        batch_size, block_len, _ = x.shape
        context_len = x_ctx.shape[1]
        if self.is_sliding:
            keep_ctx = self.sliding_window - 1
            if keep_ctx < context_len:
                skip = context_len - keep_ctx
                x_ctx = x_ctx[:, skip:]
                context_len = x_ctx.shape[1]
                cache.offset += skip
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(
            queries.reshape(batch_size, block_len, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(
            ctx_keys.reshape(batch_size, context_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(
            batch_size, context_len, self.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(
            prop_keys.reshape(batch_size, block_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(
            batch_size, block_len, self.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=cache.offset + context_len)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + context_len)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        ctx_len = keys.shape[2]
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        mask = None
        if self.is_sliding:
            mask = (
                "causal"
                if ctx_len + block_len <= self.sliding_window
                else create_causal_mask(
                    block_len, offset=ctx_len, window_size=self.sliding_window
                )
            )
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(
            output.transpose(0, 2, 1, 3).reshape(batch_size, block_len, -1)
        )

    def prefill_context(self, x_ctx, rope, cache, offset: int) -> None:
        """Append target context without running a proposal block."""
        batch_size, context_len, _ = x_ctx.shape
        keys = self.k_norm(
            self.k_proj(x_ctx).reshape(batch_size, context_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = (
            self.v_proj(x_ctx)
            .reshape(batch_size, context_len, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        keys = rope(keys, offset=offset)
        cache.update_and_fetch(keys, values)

    def propose(self, x, rope, cache):
        """Attend a proposal block to the already-prefilled target context."""
        batch_size, block_len, _ = x.shape
        queries = self.q_norm(
            self.q_proj(x).reshape(batch_size, block_len, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            self.k_proj(x).reshape(batch_size, block_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = (
            self.v_proj(x)
            .reshape(batch_size, block_len, self.n_kv_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        offset = int(getattr(cache, "offset", 0))
        queries = rope(queries, offset=offset)
        keys = rope(keys, offset=offset)
        ctx_keys, ctx_values = cache.state
        all_keys = mx.concatenate([ctx_keys, keys], axis=2)
        all_values = mx.concatenate([ctx_values, values], axis=2)
        output = mx.fast.scaled_dot_product_attention(
            queries, all_keys, all_values, scale=self.scale, mask=None
        )
        return self.o_proj(
            output.transpose(0, 2, 1, 3).reshape(batch_size, block_len, -1)
        )


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        if not self.config.layer_types:
            self.config.layer_types = (
                "full_attention",
            ) * self.config.num_hidden_layers
        target_hidden = config.target_hidden_size or config.hidden_size
        concat_dim = len(config.target_layer_ids) * target_hidden
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            DFlashDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(
            config.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
            config.rope_scaling,
        )
        self.embed_tokens = None
        self.lm_head = None
        self.embed_scale = 1.0

    def bind(self, target_model):
        """Wire the drafter to the *target's* embed_tokens + lm_head (DFlash reuses them)."""
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self.embed_tokens = inner.embed_tokens
        self.embed_scale = getattr(
            self.embed_tokens, "embed_scale", getattr(inner, "embed_scale", 1.0)
        )
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self.embed_tokens.as_linear
        )
        return self

    def make_cache(self):
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                if self.config.sliding_window is None:
                    raise ValueError(
                        "Draft config must define sliding_window for sliding_attention layers."
                    )
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window - 1, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def __call__(self, inputs, target_hidden, cache, logits_start: int = 0):
        h = self.embed_tokens(inputs) * self.embed_scale
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        if logits_start:
            h = h[:, logits_start:]
        logits = self.lm_head(self.norm(h))
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits

    def prefill_context(self, target_hidden, cache, offset: int = 0) -> None:
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            layer.self_attn.prefill_context(h_ctx, self.rope, c, offset)

    def propose(self, inputs, cache):
        h = self.embed_tokens(inputs) * self.embed_scale
        for layer, c in zip(self.layers, cache):
            residual = h
            h = residual + layer.self_attn.propose(
                layer.input_layernorm(h), self.rope, c
            )
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self.lm_head(self.norm(h))
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / cap) * cap
        return logits


class DFlashVanillaMarkov(nn.Module):
    """Rank-reduced previous-token correction used by hybrid draft heads."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def step_bias(self, token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.markov_w1(token_ids))


class DFlashMarkovDraftModel(DFlashDraftModel):
    """DFlash backbone with a strict VanillaMarkov proposal head."""

    def __init__(self, config: DFlashConfig, markov_rank: int):
        super().__init__(config)
        self.markov_head = DFlashVanillaMarkov(config.vocab_size, markov_rank)


class _CompactMarkov(nn.Module):
    def __init__(self, vocab_size: int, draft_vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, draft_vocab_size, bias=False)

    def step_bias(self, token_ids: mx.array) -> mx.array:
        return self.markov_w2(self.markov_w1(token_ids))


class _ConfidenceHead(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.proj = nn.Linear(input_size, 1, bias=True)


class SpeculatorsDraftModel(DFlashDraftModel):
    """Speculators DFlash+Markov model with a compact proposal vocabulary."""

    def __init__(self, config: DFlashConfig):
        super().__init__(config)
        draft_vocab = int(config.draft_vocab_size or config.vocab_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, draft_vocab, bias=False)
        self.d2t = mx.zeros((draft_vocab,), dtype=mx.int32)
        self.t2d = mx.zeros((config.vocab_size,), dtype=mx.int32)
        self.markov_head = (
            _CompactMarkov(config.vocab_size, draft_vocab, int(config.markov_rank))
            if config.markov_rank > 0
            else None
        )
        if config.enable_confidence_head:
            confidence_input = config.hidden_size
            if config.confidence_head_with_markov:
                confidence_input += int(config.markov_rank)
            self.confidence_head = _ConfidenceHead(confidence_input)

    @property
    def draft_to_target(self):
        return self.d2t

    def bind(self, target_model):
        """Speculators owns its vocabulary modules; only validate target shape."""
        del target_model
        return self
