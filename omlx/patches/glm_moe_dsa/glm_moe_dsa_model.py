# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlx.core as mx

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import CacheList, KVCache
from .kernels import fast as glm_fast
from .deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
)
from .deepseek_v32 import Model as DSV32Model
from .sparse_mla import (
    exact_block_token_attention,
    q8_vup_flat,
    sparse_mla_attention,
)


def _native_sparse_mla_default_min_k() -> str:
    """Match the MLX fork's default sparse MLA threshold when native is present."""
    return "11264" if glm_fast.has("glm_dsa_sparse_mla_attention") else str(2**63 - 1)


# Decode-sized multi-row forwards (MTP draft verify and head folds run
# L = depth+1 <= 8 rows). The min_k threshold above marks where the sparse
# kernel beats the *prefill* paths; below it, small L would otherwise fall
# through to the fallback that materializes full-cache multi-head K/V
# (embed_q/unembed_out over every cached position) — the absorbed latent
# path and the sparse kernel are both far cheaper at these shapes.
_ABSORBED_DECODE_MAX_L = 8


def _parse_topk_state(topk_state):
    topk_indices = topk_state
    prefix_rows = 0
    if isinstance(topk_state, tuple):
        if len(topk_state) == 3:
            topk_indices, _, prefix_rows = topk_state
        else:
            topk_indices, _ = topk_state
    return topk_indices, prefix_rows


def _apply_sparse_topk_mask(
    mask: Optional[mx.array],
    topk_indices: Optional[mx.array],
    topk_prefix_rows: int,
    *,
    key_length: int,
    query_length: int,
) -> Optional[mx.array]:
    if topk_indices is None or query_length <= 1:
        return mask

    topk_rows = topk_indices.shape[2]
    if topk_rows == query_length:
        shape = list(topk_indices.shape)
        shape[-1] = key_length
        sparse_mask = mx.zeros(shape, dtype=mx.bool_)
        sparse_mask = mx.put_along_axis(
            sparse_mask, topk_indices, mx.array(True), axis=-1
        )
    elif topk_prefix_rows > 0 and topk_rows + topk_prefix_rows == query_length:
        prefix_shape = list(topk_indices.shape)
        prefix_shape[2] = topk_prefix_rows
        prefix_shape[-1] = key_length
        slots = mx.arange(key_length, dtype=mx.uint32).reshape(
            1, 1, 1, key_length
        )
        lengths = mx.arange(topk_prefix_rows, dtype=mx.uint32).reshape(
            1, 1, topk_prefix_rows, 1
        ) + mx.array(key_length - query_length + 1, dtype=mx.uint32)
        prefix_mask = mx.broadcast_to(slots < lengths, prefix_shape)

        suffix_shape = list(topk_indices.shape)
        suffix_shape[-1] = key_length
        suffix_mask = mx.zeros(suffix_shape, dtype=mx.bool_)
        suffix_mask = mx.put_along_axis(
            suffix_mask, topk_indices, mx.array(True), axis=-1
        )
        sparse_mask = mx.concatenate([prefix_mask, suffix_mask], axis=2)
    else:
        return mask

    if mask is not None:
        sparse_mask = sparse_mask & mask
    return sparse_mask


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
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
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
    rope_parameters: Dict
    attention_bias: bool
    rope_scaling: Dict = None
    rope_theta: Optional[float] = None
    indexer_rope_interleave: bool = True
    indexer_types: Optional[List[str]] = None
    index_topk_pattern: Optional[Any] = None
    index_topk_freq: int = 1
    index_skip_topk_offset: int = 2
    quantization: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.rope_scaling = self.rope_parameters
        self.rope_theta = self.rope_parameters["rope_theta"]

        config_indexer_types = self.indexer_types

        if self.indexer_types is None:
            if self.index_topk_pattern is not None:
                pattern = self.index_topk_pattern
                if isinstance(pattern, str):
                    if len(pattern) != self.num_hidden_layers:
                        raise ValueError(
                            "index_topk_pattern length must match "
                            f"num_hidden_layers ({len(pattern)} != "
                            f"{self.num_hidden_layers})."
                        )
                    pattern_types = [{"F": "full", "S": "shared"}[c] for c in pattern]
                else:
                    pattern_types = list(pattern)
                if config_indexer_types is not None:
                    self.indexer_types = [
                        "full" if base == "full" and selected == "full" else "shared"
                        for base, selected in zip(config_indexer_types, pattern_types)
                    ]
                else:
                    self.indexer_types = pattern_types
            else:
                freq = max(self.index_topk_freq, 1)
                offset = self.index_skip_topk_offset
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]


class GlmMoeDsaAttention(DeepseekV32Attention):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
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
            topk_state = self.indexer(x, qr, mask, cache=cache[1])
        else:
            topk_state = prev_topk_indices

        if L == 1:
            topk_indices, _ = _parse_topk_state(topk_state)
            if topk_indices is not None:
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

            # Ensure the indexer cache is evaluated even if the topk_indices are unused
            # to keep the graph from getting too large.
            if self.indexer is not None and cache is not None and cache[0] is not None:
                cache[0].keys = mx.depends(
                    cache[0].keys, (cache[1].keys, cache[1].values)
                )

            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )
            q_nope = self.embed_q(q_nope)
            output = scaled_dot_product_attention(
                q_nope,
                kv_latent,
                kv_latent,
                cache=cache,
                scale=self.scale,
                mask=pe_scores,
            )
            output = self.unembed_out(output)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output), topk_state

        topk_indices, topk_prefix_rows = _parse_topk_state(topk_state)

        # Ensure the indexer cache is evaluated even if the topk_indices are unused
        # to keep the graph from getting too large
        if self.indexer is not None and cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        # Decode-shape multi-row forwards (MTP draft verify and head folds):
        # generalize the L == 1 per-row topk gather instead of dispatching
        # the native sparse-MLA kernel, whose prefill-shaped grid runs ~8x
        # slower than this at tiny L (4.2 ms vs 0.5 ms per layer at K≈4.7k).
        # Each row attends to its own gathered top-k in latent space, so the
        # cost is independent of context length. Indices are causally valid
        # per row by construction (same contract the L == 1 path relies on).
        if (
            topk_indices is not None
            and 1 < L <= _ABSORBED_DECODE_MAX_L
            and B == 1
            and topk_indices.shape[2] == L
            and topk_prefix_rows == 0
        ):
            idx = topk_indices[0, 0]  # (L, topk)
            kv_rows = kv_latent[0, 0][idx]  # (L, topk, latent)
            pe_rows = k_pe[0, 0][idx]  # (L, topk, rope)
            q_lat = self.embed_q(q_nope)  # (1, H, L, latent)
            qg = q_lat.transpose(0, 2, 1, 3)[0][:, :, None]  # (L, H, 1, latent)
            qp = q_pe.transpose(0, 2, 1, 3)[0][:, :, None]  # (L, H, 1, rope)
            pe_scores = (qp * self.scale) @ pe_rows[:, None].swapaxes(-1, -2)
            # The native indexer emits causally valid indices, but the
            # argpartition fallback can select future rows; mask gathered
            # keys past each row's own absolute position.
            row_pos = mx.arange(
                kv_latent.shape[2] - L, kv_latent.shape[2], dtype=idx.dtype
            )
            valid = idx <= row_pos[:, None]  # (L, topk)
            pe_scores = mx.where(
                valid[:, None, None, :],
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )
            output = scaled_dot_product_attention(
                qg,
                kv_rows[:, None],
                kv_rows[:, None],
                cache=cache,
                scale=self.scale,
                mask=pe_scores,
            )  # (L, H, 1, latent)
            output = output[:, :, 0].transpose(1, 0, 2)[None]  # (1, H, L, latent)
            output = self.unembed_out(output)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output), topk_state

        direct_sparse_mla_min_k = int(
            _native_sparse_mla_default_min_k()
        )
        native_sparse_mla_shape = (
            topk_indices is not None
            and self.num_heads in (32, 64)  # 32 = tensor-sharded half of the 64 MLA heads
            and q_pe.shape[-1] == 64
            and kv_latent.shape[-1] == 512
            and k_pe.shape[-1] == 64
            and topk_indices.shape[-1] == 2048
        )
        direct_sparse_mla_requested = (
            native_sparse_mla_shape
            and L > 1
            and kv_latent.shape[2] >= direct_sparse_mla_min_k
        )
        if direct_sparse_mla_requested:
            fast_topk_indices = glm_fast.has("dsa_topk_indices")
            causal_prefix_indices = fast_topk_indices
            q_latent = self.embed_q(q_nope)
            output = sparse_mla_attention(
                q_latent,
                q_pe,
                kv_latent,
                k_pe,
                topk_indices,
                self.scale,
                topk_valid_prefix=fast_topk_indices,
                causal_prefix_indices=causal_prefix_indices,
                causal_prefix_rows=topk_prefix_rows,
            )
            if output is not None:
                output_flat = q8_vup_flat(
                    output, self.unembed_out, key_length=kv_latent.shape[2]
                )
                if output_flat is None:
                    output = self.unembed_out(output)
                    output_flat = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                output = output_flat
                return self.o_proj(output), topk_state

        if topk_indices is not None and L > 8:
            k = self.embed_q(kv_latent, transpose=False)
            k_pe_heads = mx.broadcast_to(k_pe, k.shape[:-1] + k_pe.shape[-1:])
            q = mx.concatenate([q_nope, q_pe], axis=-1)
            k = mx.concatenate([k, k_pe_heads], axis=-1)
            v = self.unembed_out(kv_latent)
            fast_topk_indices = glm_fast.has("dsa_topk_indices")
            output = exact_block_token_attention(
                q,
                k,
                v,
                topk_indices,
                self.scale,
                q_block_size=32,
                k_block_size=8,
                causal_prefix_indices=fast_topk_indices,
                causal_prefix_rows=topk_prefix_rows,
            )
            if output is not None:
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output), topk_state

        mask = _apply_sparse_topk_mask(
            mask,
            topk_indices,
            topk_prefix_rows,
            key_length=kv_latent.shape[2],
            query_length=L,
        )

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        absorbed = L <= _ABSORBED_DECODE_MAX_L
        if absorbed:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
        )
        if absorbed:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), topk_state


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GlmMoeDsaAttention(config, layer_idx)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
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
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h, cache[0][0] if cache[0] else None, return_array=True
        )

        # Receive from the previous process in the pipeline
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        prev_topk_indices = None
        for i in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + i](
                h, mask, cache[i], prev_topk_indices
            )

        # Send to the next process in the pipeline
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1][0].keys = mx.depends(cache[-1][0].keys, h)

        # Broadcast h while keeping it in the graph
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(DSV32Model):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        self.model = GlmMoeDsaModel(config)

    def sanitize(self, weights):
        weights = super().sanitize(weights)
        skip_prefixes = [
            f"model.layers.{i}.self_attn.indexer."
            for i, layer in enumerate(self.model.layers)
            if getattr(layer.self_attn, "skip_topk", False)
        ]
        if skip_prefixes:
            weights = {
                k: v
                for k, v in weights.items()
                if not any(k.startswith(prefix) for prefix in skip_prefixes)
            }
        return weights

    def make_cache(self):
        # Shared layers run no indexer, so they get no indexer KVCache.
        caches = []
        for layer in self.layers:
            if getattr(layer.self_attn, "skip_topk", False):
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
