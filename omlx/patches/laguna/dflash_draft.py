# SPDX-License-Identifier: Apache-2.0
"""DFlashLagunaDraftModel — the poolside/Laguna-XS-2.1-DFlash-NVFP4 drafter.

The laguna DFlash drafter is a 5-layer gated-delta-net (the same
architecture as the 40-layer target) with an Eagle-style aux head: each of
the 5 aux target hidden states (target_layer_ids [1,13,25,33,39]) is
RMS-normed by its own ``aux_hidden_norms[k]``, concatenated to 10240,
projected by ``fc [2048, 10240]`` -> 2048, then ``hidden_norm``. The 5 trunk
layers use fused ``qkv_proj`` (q + k + v in one [10240, 2048] projection)
with per-head ``g_proj`` gating (the laguna attention shape), distinct from
dflash's generic separate q/k/v DFlashAttention.

Implements the same DFlashDraftModel contract (project_target_hidden /
forward_projected_context / advance_projected_context_cache) so the dflash
engine can drive it unchanged.
"""
from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.rope_utils import initialize_rope

from mlx_lm.models.qwen3_next import Qwen3NextMLP


class LagunaDraftArgs:
    """Parsed laguna drafter config (mirrors DFlashDraftModelArgs fields)."""

    @classmethod
    def from_dict(cls, config: dict) -> "LagunaDraftArgs":
        return cls(config)

    def __init__(self, config: dict):
        self.model_type = str(config.get("model_type", "laguna"))
        self.hidden_size = int(config["hidden_size"])
        self.num_hidden_layers = int(config["num_hidden_layers"])
        self.intermediate_size = int(config["intermediate_size"])
        self.num_attention_heads = int(config["num_attention_heads"])
        self.num_key_value_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.rms_norm_eps = float(config.get("rms_norm_eps", 1e-6))
        self.vocab_size = int(config["vocab_size"])
        self.max_position_embeddings = int(config.get("max_position_embeddings", 262144))
        self.rope_theta = float(config.get("rope_theta", 500000.0))
        self.attention_bias = bool(config.get("attention_bias", False))
        layer_types = config.get("layer_types") or []
        self.layer_types = tuple(layer_types)
        self.sliding_window = int(config.get("sliding_window", 0) or 0)
        df = config.get("dflash_config") or {}
        self.block_size = int(df.get("block_size", 16) or 16)
        self.mask_token_id = int(df.get("mask_token_id", 12) or 12)
        self.num_target_layers = int(df.get("num_target_layers", 40) or 40)
        self.target_layer_ids = [int(i) for i in df.get("target_layer_ids") or []]
        self.gating = config.get("gating", "per-head")
        self.gate_per_head = self.gating == "per-head"


class LagunaDraftAttention(nn.Module):
    """Laguna-architecture draft attention: fused qkv_proj + per-head gate.

    Mirrors the drafter's fused ``qkv_proj [10240, 2048]`` (q 8192 + k 1024 +
    v 1024 for heads=64, kv_heads=8, head_dim=128), q/k RMSNorm, per-head
    ``g_proj`` gate, and the o_proj. The cache path mirrors dflash's
    FullContextDraftKVCache contract (target-context keys/values).
    """

    def __init__(self, args: "LagunaDraftArgs", layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.n_k_heads = self.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        dim = args.hidden_size
        self.qkv_proj = nn.Linear(
            dim, self.n_heads * self.head_dim + 2 * self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        gate_dim = self.n_heads if args.gate_per_head else self.n_heads * self.head_dim
        self.g_proj = nn.Linear(dim, gate_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.sliding_window = int(getattr(args, "sliding_window", 0) or 0)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
        )

    def _split_qkv(self, qkv: mx.array):
        q_split = self.n_heads * self.head_dim
        k_end = q_split + self.n_kv_heads * self.head_dim
        return qkv[..., :q_split], qkv[..., q_split:k_end], qkv[..., k_end:]

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        queries, keys, values = self._split_qkv(qkv)
        queries = self.q_norm(
            queries.reshape(B, L, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        gate = self.g_proj(hidden_states)

        ctx_len = int(target_hidden.shape[1])
        # the generic DFlashAttention computes its OWN context k/v from the
        # (projected) target hidden in-call, then appends the noise block
        # k/v and attends with the causal block mask. Mirror that exactly
        # with the fused qkv_proj.
        B2, T2, _ = target_hidden.shape
        cqkv = self.qkv_proj(target_hidden)
        _, context_keys, context_values = self._split_qkv(cqkv)
        context_keys = self.k_norm(
            context_keys.reshape(B2, T2, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        context_values = context_values.reshape(
            B2, T2, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        cache_offset = int(getattr(cache, "offset", 0) or 0)
        query_offset = cache_offset + ctx_len
        queries = self.rope(queries, offset=query_offset)
        context_keys = self.rope(context_keys, offset=cache_offset)
        noise_keys = self.rope(keys, offset=query_offset)
        noise_values = values

        if cache is not None:
            from dflash_mlx.model import FullContextDraftKVCache

            if isinstance(cache, FullContextDraftKVCache):
                all_keys = mx.concatenate([context_keys, noise_keys], axis=-2)
                all_values = mx.concatenate([context_values, noise_values], axis=-2)
                keys2, values2 = cache.fetch_with_block(noise_keys, noise_values)
                _ = all_keys, all_values
                out = scaled_dot_product_attention(
                    queries, keys2, values2, cache=None, scale=self.scale, mask=None)
            else:
                keys2 = mx.concatenate([context_keys, noise_keys], axis=-2)
                values2 = mx.concatenate([context_values, noise_values], axis=-2)
                full_key_len = query_offset + int(keys2.shape[-2]) if False else query_offset + int(noise_keys.shape[-2])
                from mlx_lm.models.base import create_causal_mask

                mask = create_causal_mask(
                    int(noise_keys.shape[-2]),
                    offset=query_offset,
                    window_size=int(self.sliding_window) if self.sliding_window else None,
                )
                out = scaled_dot_product_attention(
                    queries, keys2, values2, cache=None, scale=self.scale, mask=mask)
        else:
            keys2 = mx.concatenate([context_keys, noise_keys], axis=-2)
            values2 = mx.concatenate([context_values, noise_values], axis=-2)
            out = scaled_dot_product_attention(
                queries, keys2, values2, cache=None, scale=self.scale, mask=None)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        if gate.shape[-1] == self.n_heads:
            g = mx.sigmoid(gate.astype(mx.float32)).astype(self.head_dim and out.dtype)
            gated = out.reshape(B, L, self.n_heads, self.head_dim) * \
                g.reshape(B, L, self.n_heads, 1)
            out = gated.reshape(B, L, -1)
        else:
            out = out * mx.sigmoid(gate.astype(mx.float32)).astype(out.dtype)
        return self.o_proj(out)

    def append_projected_context_cache(
        self,
        *,
        target_hidden: mx.array,
        cache: Any,
    ) -> None:
        import os
        if os.environ.get("LAGUNA_DRAFT_DEBUG"):
            print(f"[laguna-draft] append ctx: target_hidden {tuple(target_hidden.shape)} cache {type(cache).__name__} keys {getattr(getattr(cache,'keys',None),'shape',None)}", flush=True)
        from dflash_mlx.model import ContextOnlyDraftKVCache

        if not isinstance(cache, ContextOnlyDraftKVCache):
            raise TypeError("draft context advance requires a DFlash draft KV cache")
        ctx_len = int(target_hidden.shape[1])
        if ctx_len <= 0:
            return
        # the context here is the ALREADY-projected draft context [B, ctx, H]
        # (the engine's feature_store projects via project_target_hidden).
        B = int(target_hidden.shape[0])
        qkv = self.qkv_proj(target_hidden)
        _, context_keys, context_values = self._split_qkv(qkv)
        selected = int(context_keys.shape[1])
        context_keys = self.k_norm(
            context_keys.reshape(B, selected, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        context_values = context_values.reshape(
            B, selected, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        context_keys = self.rope(context_keys, offset=int(cache.offset))
        cache.append_context(
            context_keys, context_values, ctx_len,
            positions=mx.arange(int(cache.offset), int(cache.offset) + ctx_len,
                                dtype=mx.int32),
            advance_positions=ctx_len,
        )


class LagunaDraftDecoderLayer(nn.Module):
    def __init__(self, args: "LagunaDraftArgs", layer_idx: int):
        super().__init__()
        self.self_attn = LagunaDraftAttention(args, layer_idx)
        self.mlp = Qwen3NextMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden=target_hidden, cache=cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def advance_projected_context_cache(
        self,
        *,
        target_hidden: mx.array,
        cache: Any,
    ) -> None:
        self.self_attn.append_projected_context_cache(
            target_hidden=target_hidden, cache=cache
        )


class DFlashLagunaDraftModel(nn.Module):
    """The poolside laguna DFlash drafter (5-layer laguna + Eagle aux head)."""

    def __init__(self, args: "LagunaDraftArgs"):
        super().__init__()
        self.args = args
        self.model_type = "laguna_draft"
        self.layers = [
            LagunaDraftDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.target_layer_ids = args.target_layer_ids or [
            (i + 1) * args.num_target_layers // (args.num_hidden_layers + 1)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Eagle aux: per-target-layer RMS norms + the concat projection.
        self.aux_hidden_norms = [
            nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            for _ in self.target_layer_ids
        ]
        self.fc = nn.Linear(
            len(self.target_layer_ids) * args.hidden_size, args.hidden_size, bias=False
        )
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.block_size = int(args.block_size)
        self.mask_token_id = int(args.mask_token_id)
        self.embed_scale = 1.0

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        text_model = target_ops.text_model(target_model)
        self.embed_scale = getattr(text_model, "embed_scale", 1.0)

    def project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        """Norm each aux-layer hidden, concat to 10240, fc -> 2048, hidden_norm.

        ``target_hidden`` is [B, T, num_target_layers * hidden] (the captured
        "layer_id+1" hiddens stacked on the last axis).
        """
        if target_hidden.ndim == 3:
            hs = target_hidden
        else:
            hs = target_hidden[..., None] if False else target_hidden
        # split the last axis into the per-layer hiddens
        hidden = self.args.hidden_size
        rows = int(hs.shape[-1]) // hidden
        if rows == 1:
            pieces = [hs]
        else:
            pieces = [hs[..., r * hidden : (r + 1) * hidden] for r in range(rows)]
        normed = []
        for k, piece in enumerate(pieces):
            if k < len(self.aux_hidden_norms):
                normed.append(self.aux_hidden_norms[k](piece))
            else:
                normed.append(piece)
        cat = mx.concatenate(normed, axis=-1)
        return self.hidden_norm(self.fc(cat))

    def forward_projected_context(
        self,
        *,
        noise_embedding: mx.array,
        draft_context: mx.array,
        cache: Optional[list[Any]] = None,
    ) -> mx.array:
        hidden_states = noise_embedding * self.embed_scale
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden_states = layer(
                hidden_states,
                target_hidden=draft_context,
                cache=layer_cache,
            )
        return self.norm(hidden_states)

    def advance_projected_context_cache(
        self,
        *,
        draft_context: mx.array,
        cache: list[Any],
    ) -> None:
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            layer.advance_projected_context_cache(
                target_hidden=draft_context, cache=layer_cache
            )

    def __call__(
        self,
        *,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: Optional[list[Any]] = None,
    ) -> mx.array:
        return self.forward_projected_context(
            noise_embedding=noise_embedding,
            draft_context=self.project_target_hidden(target_hidden),
            cache=cache,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return weights
