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

from mlx_lm.models import qwen3 as _qwen3
from mlx_lm.models.base import create_attention_mask


@dataclass
class ModelArgs(_qwen3.ModelArgs):
    layer_types: list[str] | None = None
    sliding_window: int | None = None


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
        full_mask = create_attention_mask(h, cache[0])
        swa_mask = create_attention_mask(h, cache[0], window_size=self.window_size)

        for is_sliding, layer, c in zip(self.is_sliding, self.layers, cache):
            h = layer(h, mask=(swa_mask if is_sliding else full_mask), cache=c)

        return self.norm(h)
