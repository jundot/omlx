# Copyright (c) 2025 Apple Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 ``glm_moe_dsa`` model for the pinned mlx-lm runtime.

Vendored from ml-explore/mlx-lm#1410 so oMLX can load GLM-5.2 checkpoints
while the pinned mlx-lm still exposes ``glm_moe_dsa`` as a bare
DeepSeek-V3.2 subclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import CacheList, KVCache
from .deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from .deepseek_v32 import Model as DSV32Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: int | None
    n_routed_experts: int | None
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    topk_method: str
    scoring_func: str
    norm_topk_prob: bool
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    moe_layer_freq: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_parameters: dict
    attention_bias: bool
    rope_scaling: dict | None = None
    rope_theta: float | None = None
    indexer_types: list[str] | None = None
    index_topk_pattern: Any | None = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]

        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                if isinstance(pattern, str):
                    self.indexer_types = [
                        {"F": "full", "S": "shared"}[c] for c in pattern
                    ]
                else:
                    self.indexer_types = list(pattern)
            else:
                freq = max(int(self.index_topk_freq), 1)
                offset = int(self.index_skip_topk_offset)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]
        else:
            self.indexer_types = list(self.indexer_types)

        if len(self.indexer_types) != self.num_hidden_layers:
            raise ValueError(
                "`indexer_types` must have one entry per hidden layer, "
                f"got {len(self.indexer_types)} for {self.num_hidden_layers} layers."
            )
        invalid = sorted(set(self.indexer_types) - {"full", "shared"})
        if invalid:
            raise ValueError(f"Unsupported GLM MoE DSA indexer types: {invalid}")
        if self.indexer_types and self.indexer_types[0] != "full":
            raise ValueError("The first GLM MoE DSA layer must be a full indexer layer.")


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config)
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        prev_topk_indices: mx.array | None = None,
    ):
        batch_size, seq_len, _hidden_dim = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(
            batch_size, seq_len, self.num_heads, self.q_head_dim
        ).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(
            batch_size, seq_len, 1, self.qk_rope_head_dim
        ).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        if self.indexer is not None:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        else:
            topk_indices = prev_topk_indices

        if topk_indices is not None:
            if seq_len == 1:
                idx = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                shape = list(topk_indices.shape)
                shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        if self.indexer is not None and cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if seq_len == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if seq_len == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output), topk_indices


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
        prev_topk_indices: mx.array | None = None,
    ):
        r, topk_indices = self.self_attn(
            self.input_layernorm(x), mask, cache, prev_topk_indices
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, topk_indices


class GlmMoeDsaModel(DeepseekV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.layers = [
            GlmMoeDsaDecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        cache: Any | None = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk_indices
            )

        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
