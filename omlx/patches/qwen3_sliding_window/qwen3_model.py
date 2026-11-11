# SPDX-License-Identifier: Apache-2.0
"""Qwen3 sliding-window attention support, layered onto the pinned mlx-lm dep.

mlx-lm's pinned Qwen3 ModelArgs/Qwen3Model have no concept of per-layer
attention type, so a Qwen3 checkpoint with layer_types and sliding_window
(e.g. jinaai/jina-reranker-v3.5-mlx: 28 layers alternating sliding/full)
gets every layer run as full attention regardless of the config.

mlx-lm already implements this identically for other model families
in this same pinned version (e.g. mlx_lm.models.gpt_oss).
The window_size parameter in mlx_lm.models.base.create_attention_mask
is pre-existing general-purpose, already used by gemma3_text, gemma4_text,
cohere2, exaone4, olmo3, llama, ministral3, and more. This ports that pattern
to Qwen3.

For any config without layer_types, it falls back to all "full_attention" and
create_attention_mask(..., window_size=None) returns the same "causal"
string it would with no window_size argument, so every Qwen3 model that
doesn't declare layer_types/sliding_window has the same behavior to
before this patch.

See jundot/omlx#2422.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.models import qwen3 as _qwen3
from mlx_lm.models.base import create_attention_mask, create_causal_mask


@dataclass
class ModelArgs(_qwen3.ModelArgs):
    layer_types: list[str] | None = None
    sliding_window: int | None = None


def _sliding_window_attention(attn, x: mx.array, window: int) -> mx.array:
    """Cache-free mlx-lm Qwen3 attention, one window-sized query chunk at a time.

    Each chunk reads only the keys inside its window, so memory grows with the
    input length instead of its square.
    """
    batch, length, _ = x.shape
    queries = attn.q_norm(attn.q_proj(x).reshape(batch, length, attn.n_heads, -1))
    keys = attn.k_norm(attn.k_proj(x).reshape(batch, length, attn.n_kv_heads, -1))
    values = attn.v_proj(x).reshape(batch, length, attn.n_kv_heads, -1)
    queries = attn.rope(queries.transpose(0, 2, 1, 3))
    keys = attn.rope(keys.transpose(0, 2, 1, 3))
    values = values.transpose(0, 2, 1, 3)

    chunks = []
    for start in range(0, length, window):
        end = min(start + window, length)
        key_start = max(0, start - window + 1)
        mask = create_causal_mask(end - start, start - key_start, window_size=window)
        chunks.append(
            mx.fast.scaled_dot_product_attention(
                queries[:, :, start:end],
                keys[:, :, key_start:end],
                values[:, :, key_start:end],
                scale=attn.scale,
                mask=mask,
            )
        )
    output = (
        mx.concatenate(chunks, axis=2).transpose(0, 2, 1, 3).reshape(batch, length, -1)
    )
    return attn.o_proj(output)


class Qwen3Model(_qwen3.Qwen3Model):
    def __init__(self, args: ModelArgs):
        super().__init__(args)

        layer_types = args.layer_types or ["full_attention"] * args.num_hidden_layers
        if len(layer_types) != args.num_hidden_layers:
            raise ValueError(
                f"len(layer_types)={len(layer_types)} != "
                f"num_hidden_layers={args.num_hidden_layers}"
            )
        # Boolean list: True -> sliding_attention, False -> full_attention
        self.is_sliding: list[bool] = [lt == "sliding_attention" for lt in layer_types]
        self.window_size: int | None = args.sliding_window

    def __call__(
        self,
        inputs,
        cache: list[Any] | None = None,
        input_embeddings=None,
    ):
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings

        if cache is None:
            cache = [None] * len(self.layers)

        # Build both masks once, dispatch per layer. Mirrors the reference
        # jina-reranker-v3.5-mlx Qwen3Model.__call__ and mlx_lm.models.gpt_oss.
        # Without a cache, a sliding layer longer than its window runs chunked
        # instead, because its full mask would grow with the square of the input.
        full_mask = create_attention_mask(h, cache[0])
        chunked = (
            cache[0] is None
            and self.window_size is not None
            and h.shape[1] > self.window_size
        )
        swa_mask = (
            None
            if chunked
            else create_attention_mask(h, cache[0], window_size=self.window_size)
        )

        for is_sliding, layer, c in zip(self.is_sliding, self.layers, cache):
            if is_sliding and chunked:
                x = layer.input_layernorm(h)
                h = h + _sliding_window_attention(layer.self_attn, x, self.window_size)
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            else:
                h = layer(h, mask=(swa_mask if is_sliding else full_mask), cache=c)

        return self.norm(h)
