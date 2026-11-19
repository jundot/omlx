# SPDX-License-Identifier: Apache-2.0
# Reference architecture: Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team.
"""Native Hy4 Flash text decoder (no vision or speculative draft layers).

Equations follow transformers' HYV4FlashTextModel at c2ec850510a3. Parameter
paths follow the released checkpoint, *before* transformers' internal renames.
A request has its own attention caches and small Engram token history; model
parameters and the external, read-only table provider can be shared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention, RMSNorm
from .engram import EngramLayer
from .experts import build_mlp


@dataclass
class ModelArgs:
    # Require the checkpoint's dimensions and layer schedules. In particular,
    # do not default a missing config to the multi-hundred-GB public shape.
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]
    mlp_layer_types: list[str]
    intermediate_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    swa_num_attention_heads: int
    swa_qk_nope_head_dim: int
    swa_qk_rope_head_dim: int
    swa_v_head_dim: int
    rope_parameters: dict
    compressed_indexer_compress_ratio: int
    compressed_indexer_num_attention_heads: int
    compressed_indexer_head_dim: int
    compressed_indexer_group_topk: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    routed_expert_hidden_size: int | None

    model_type: str = "hy_v4_flash"
    max_position_embeddings: int = 1048576
    sliding_window: int = 512
    attention_bias: bool = False
    learnable_sink_init: float = 0.0
    compressed_indexer_candidate_topk: int | None = None
    use_compressed_indexer: bool = True
    compressed_indexer_key_pooling: str = "gated"
    compressed_indexer_skip_fine_select: bool = True
    compressed_indexer_reuse_fine_indexer: bool = False
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.006
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_magnitude: float = 2.0
    enable_ihc: bool = True
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    routed_scaling_factor: float = 3.998
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    moe_router_use_sigmoid: bool = True
    moe_router_enable_expert_bias: bool = True
    tie_word_embeddings: bool = False
    enable_lm_head_fp32: bool = True
    engram_layer_ids: list[int] | None = None
    engram_layer_head_vocab_sizes: dict | None = None
    engram_pad_id: int = 120002
    engram_seed: int = 0
    engram_max_ngram_size: int = 3
    engram_n_head_per_ngram: int = 8
    engram_n_embed_per_ngram: int = 768
    engram_fp8_embedding: bool = True
    engram_lookup_table_size: int = 120818
    omlx_oqh: dict | None = None

    @classmethod
    def from_dict(cls, config: dict) -> ModelArgs:
        # The public VLM nests these fields, the converted text checkpoint
        # keeps them at the root. Neither case requires loading a vision model.
        text = config.get("text_config", config)
        args = {f.name: text[f.name] for f in fields(cls) if f.name in text}
        if "omlx_oqh" in config:
            args["omlx_oqh"] = config["omlx_oqh"]
        return cls(**args)

    def __post_init__(self):
        if self.model_type not in ("hy_v4_flash", "hy_v4_flash_text"):
            raise ValueError(
                f"Expected a Hy4 Flash text config, got {self.model_type!r}"
            )
        if self.num_hidden_layers < 1 or min(self.vocab_size, self.hidden_size) < 1:
            raise ValueError("Invalid Hy4 Flash model dimensions")
        if not self.enable_ihc or self.hc_mult < 1:
            raise ValueError("Hy4 Flash requires independent hyper-connections")
        for schedule, supported in (
            (self.mlp_layer_types, {"dense", "sparse"}),
            (
                self.layer_types,
                {"deepseek_sparse_attention", "indexed_attention", "sliding_attention"},
            ),
        ):
            if len(schedule) != self.num_hidden_layers or set(schedule) - supported:
                raise ValueError(f"Invalid per-layer schedule: {schedule!r}")
        if self.compressed_indexer_candidate_topk is None:
            self.compressed_indexer_candidate_topk = (
                self.compressed_indexer_compress_ratio
                * self.compressed_indexer_group_topk
            )
        if "sparse" in self.mlp_layer_types and not (
            self.moe_router_use_sigmoid and self.moe_router_enable_expert_bias
        ):
            raise ValueError(
                "Expected the public sigmoid router with expert correction bias"
            )
        ids = self.engram_layer_ids or []
        if len(set(ids)) != len(ids) or any(
            not 1 <= i <= self.num_hidden_layers for i in ids
        ):
            raise ValueError(
                "Engram layer IDs must be distinct one-based decoder indices"
            )
        if ids and self.engram_max_ngram_size < 2:
            raise ValueError("Engram requires at least bigrams")


def _hc_logits(streams, fn, rms_eps):
    flat = streams.flatten(-2).astype(mx.float32)
    inv_rms = mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + rms_eps)
    fn = fn.astype(mx.float32)
    # MLX's multi-row float32 GEMM can use TF32 on M5. Do not round the
    # checkpoint's FP32 gates. With only 4/8 output channels we can use
    # full-precision GEMVs without broadcasting [B,T,C,C*D] temporaries.
    if flat.size == flat.shape[-1]:
        mixes = flat @ fn.T  # single-token decode is already a GEMV
    else:
        mixes = mx.stack([flat @ row for row in fn], axis=-1)
    return mixes * inv_rms


class _HyperConnectionGates(nn.Module):
    def __init__(self, args):
        super().__init__()
        n = args.hc_mult
        self.hc_fn = mx.random.normal(
            (2 * n, n * args.hidden_size), scale=args.initializer_range
        )
        self.hc_base = mx.concatenate(
            (mx.full((n,), -math.log(max(1, n - 1))), mx.zeros((n,)))
        )
        self.hc_scale = mx.full((2,), 0.01)
        self._rms_eps = args.rms_norm_eps
        self._eps = args.hc_eps
        self._post_magnitude = args.hc_magnitude

    def __call__(self, streams):
        logits = _hc_logits(streams, self.hc_fn, self._rms_eps)
        pre, post = mx.split(logits, 2, axis=-1)
        pre_b, post_b = mx.split(self.hc_base.astype(mx.float32), 2)
        scales = self.hc_scale.astype(mx.float32)
        pre = mx.sigmoid(pre * scales[0] + pre_b) + self._eps
        post = self._post_magnitude * mx.sigmoid(post * scales[1] + post_b) + self._eps
        mixed = mx.sum(pre[..., None] * streams.astype(mx.float32), axis=-2)
        return post, mixed.astype(streams.dtype)


class HyperConnection(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Public weights are nested under {hc_attn_layer,hc_mlp_layer}.hc_pre.
        self.hc_pre = _HyperConnectionGates(args)

    def __call__(self, streams):
        return self.hc_pre(streams)

    @staticmethod
    def combine(streams, post, update):
        update = post[..., None] * update.astype(mx.float32)[..., None, :]
        return (streams.astype(mx.float32) + update).astype(streams.dtype)


class HyperHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        n = args.hc_mult
        self.hc_head_fn = mx.random.normal(
            (n, n * args.hidden_size), scale=args.initializer_range
        )
        self.hc_head_scale = mx.full((1,), 0.01)
        self.hc_head_base = mx.full((n,), -math.log(max(1, n - 1)))
        self._rms_eps = args.rms_norm_eps
        self._eps = args.hc_eps

    def __call__(self, streams):
        logits = _hc_logits(streams, self.hc_head_fn, self._rms_eps)
        gates = (
            mx.sigmoid(
                logits * self.hc_head_scale.astype(mx.float32)
                + self.hc_head_base.astype(mx.float32)
            )
            + self._eps
        )
        return mx.sum(gates[..., None] * streams.astype(mx.float32), axis=-2).astype(
            streams.dtype
        )


class DecoderLayer(nn.Module):
    def __init__(self, args, block):
        super().__init__()
        self.self_attn = Attention(args, block)
        self.mlp = build_mlp(args, block)
        self.hc_attn_layer = HyperConnection(args)
        self.hc_mlp_layer = HyperConnection(args)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # The config and hash seeds are one-based; model paths are zero-based.
        self.engram_layer = (
            EngramLayer(args, block)
            if block + 1 in (args.engram_layer_ids or ())
            else None
        )

    def __call__(self, streams, ngrams, cache):
        if self.engram_layer is not None:
            streams = streams + self.engram_layer(streams, ngrams)
        post, x = self.hc_attn_layer(streams)
        x = self.self_attn(self.input_layernorm(x), cache=cache)
        streams = self.hc_attn_layer.combine(streams, post, x)
        post, x = self.hc_mlp_layer(streams)
        x = self.mlp(self.post_attention_layernorm(x))
        return self.hc_mlp_layer.combine(streams, post, x)


@dataclass
class RequestCache:
    """Non-serializable state for one request (equal-length, unpadded rows)."""

    attention: list
    # Last max_ngram_size - 1 IDs per row in chronological order, left-padded.
    history: mx.array | None = None
    offset: int = 0
    batch_size: int | None = None


def _ngram_inputs(input_ids, history, order, pad_id):
    """[current, previous, ...] at every position; share across all Engrams."""
    batch, length = input_ids.shape
    width = order - 1
    if history is None:
        history = mx.full((batch, width), pad_id, dtype=input_ids.dtype)
    tokens = mx.concatenate((history, input_ids), axis=1)
    shifted = [tokens[:, start : start + length] for start in range(width, -1, -1)]
    # Gather, rather than retaining a slice/view of the entire prefill buffer.
    next_history = mx.take(tokens, mx.arange(length, length + width), axis=1)
    return mx.stack(shifted, axis=-1), next_history


class TextModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hc_head = HyperHead(args)
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._hc_mult = args.hc_mult

    def __call__(self, ids, ngrams, cache):
        x = self.embed_tokens(ids)
        streams = mx.broadcast_to(
            x[:, :, None, :], (*x.shape[:2], self._hc_mult, x.shape[-1])
        )
        for i, layer in enumerate(self.layers):
            streams = layer(
                streams, ngrams, None if cache is None else cache.attention[i]
            )
        return self.norm(self.hc_head(streams))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def set_engram_tables(self, provider):
        """Bind external row reads *after* loading the per-layer lookup maps."""
        for layer in self.layers:
            if layer.engram_layer is not None:
                layer.engram_layer.bind_tables(provider)

    def make_cache(self) -> RequestCache:
        return RequestCache([layer.self_attn.make_cache() for layer in self.layers])

    def __call__(
        self, input_ids, cache: RequestCache | None = None, *, logits_to_keep: int = 0
    ):
        if (
            input_ids.ndim != 2
            or min(input_ids.shape) < 1
            or not mx.issubdtype(input_ids.dtype, mx.integer)
        ):
            raise ValueError("Expected nonempty integer token IDs [batch, sequence]")
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a non-negative integer")
        batch, length = input_ids.shape
        if cache is not None:
            if not isinstance(cache, RequestCache) or len(cache.attention) != len(
                self.layers
            ):
                raise ValueError(
                    "Expected a request cache created by model.make_cache()"
                )
            if cache.batch_size is not None and cache.batch_size != batch:
                raise ValueError(
                    "Cannot change the batch size of an active request cache"
                )
            if (
                cache.offset
                and self.args.engram_layer_ids
                and (
                    cache.history is None
                    or cache.history.shape
                    != (batch, self.args.engram_max_ngram_size - 1)
                )
            ):
                raise ValueError(
                    "Incomplete Engram history; discard this request cache"
                )
            if any(c.offset != cache.offset for c in cache.attention):
                raise ValueError(
                    "Inconsistent attention cache offsets; discard this request cache"
                )
        if (
            length + (cache.offset if cache is not None else 0)
            > self.args.max_position_embeddings
        ):
            raise ValueError("Sequence exceeds the maximum position embeddings")

        ngrams = history = None
        if self.args.engram_layer_ids:
            ngrams, history = _ngram_inputs(
                input_ids,
                None if cache is None else cache.history,
                self.args.engram_max_ngram_size,
                self.args.engram_pad_id,
            )
        prep = getattr(self, "_engram_preparer", None)
        if prep is not None and ngrams is not None:
            with prep.batch(ngrams) as prepared:
                hidden = self.model(input_ids, prepared, cache)
        else:
            hidden = self.model(input_ids, ngrams, cache)
        if cache is not None:
            cache.history = history
            cache.offset += length
            cache.batch_size = batch
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        weight = (
            self.model.embed_tokens.weight
            if self.args.tie_word_embeddings
            else self.lm_head.weight
        )
        # HF keeps the LM head in fp32 despite its BF16 on-disk representation.
        dtype = mx.float32 if self.args.enable_lm_head_fp32 else weight.dtype
        return hidden.astype(dtype) @ weight.astype(dtype).T
