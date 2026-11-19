# SPDX-License-Identifier: Apache-2.0
"""HY V4 Flash text attention (unpadded text, shared absolute offset per batch).

Reference: huggingface/transformers c2ec8505, HYV4FlashAttention and
HYV4FlashCompressedIndexer. Keep the *checkpoint* names, including
linear_gate and learnable_sink_param (gate_proj and sinks in Transformers).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, RotatingKVCache


class RMSNorm(nn.RMSNorm):
    def __call__(self, x):
        # HF rounds the unit-RMS activation before applying the learned scale.
        return mx.fast.rms_norm(x, None, self.eps) * self.weight


class _TextRoPE:
    """On text, all three interleaved M-RoPE axes have the same position."""

    def __init__(self, dim, params, max_positions):
        if dim <= 0 or dim % 2:
            raise ValueError("HY V4 Flash needs an even, positive rotary width")
        base = params["rope_theta"]
        inv_freq = 1 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.amplitude = 1.0
        kind = params["rope_type"]
        if kind != "default":
            # The pinned checkpoint uses default RoPE on both attention kinds.
            raise ValueError(f"Unsupported HY V4 Flash RoPE type: {kind!r}")
        self.inv_freq = inv_freq

    def phases(self, slots, dtype):
        theta = slots.astype(mx.float32)[:, None] * self.inv_freq
        # HF rounds its table to activation precision before the fp32 rotation.
        return tuple(
            (trig(theta) * self.amplitude)
            .astype(dtype)
            .astype(mx.float32)[None, :, None, :]
            for trig in (mx.cos, mx.sin)
        )


def _rotate(x, phases):
    """Rotate adjacent pairs in the trailing channels of [batch, time, heads, dim]."""
    cos, sin = phases
    width = 2 * cos.shape[-1]
    r = x[..., -width:].astype(mx.float32)
    even, odd = r[..., ::2], r[..., 1::2]
    r = mx.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1)
    r = r.flatten(-2).astype(x.dtype)
    return mx.concatenate((x[..., :-width], r), axis=-1) if width < x.shape[-1] else r


class CompressedIndexer(nn.Module):
    """Recall groups of actual token slots, not pooled attention values."""

    def __init__(self, args):
        super().__init__()
        self.compress_ratio = args.compressed_indexer_compress_ratio
        self.group_topk = args.compressed_indexer_group_topk
        self.num_heads = args.compressed_indexer_num_attention_heads
        self.head_dim = args.compressed_indexer_head_dim
        self.budget = self.compress_ratio * self.group_topk
        if (
            min(self.compress_ratio, self.group_topk, self.num_heads) <= 0
            or self.head_dim < args.qk_rope_head_dim
        ):
            raise ValueError("Invalid compressed indexer dimensions or recall budget")
        budget = getattr(args, "compressed_indexer_candidate_topk", None)
        if budget is not None and self.budget != budget:
            raise ValueError(
                "Compressed recall budget must be group_topk * compress_ratio"
            )
        if (
            not getattr(args, "use_compressed_indexer", True)
            or getattr(args, "compressed_indexer_key_pooling", "gated") != "gated"
            or not getattr(args, "compressed_indexer_skip_fine_select", True)
            or getattr(args, "compressed_indexer_reuse_fine_indexer", False)
        ):
            raise ValueError(
                "Only the public gated, group-only compressed indexer is supported"
            )

        h, d = self.num_heads, self.head_dim
        self.query_projection = nn.Linear(args.q_lora_rank, h * d, bias=False)
        self.head_weight_projection = nn.Linear(args.hidden_size, h, bias=False)
        self.token_key_projection = nn.Linear(args.hidden_size, d, bias=False)
        self.token_key_norm = RMSNorm(d, eps=args.rms_norm_eps)
        self.key_value_projection = nn.Linear(d, d, bias=False)
        self.key_gate_projection = nn.Linear(d, d, bias=False)
        self.key_norm = RMSNorm(d, eps=args.rms_norm_eps)
        self.intra_group_ape = mx.zeros((self.compress_ratio, d))

    def encode_slots(self, x, cache, offset, rope, rope_dim):
        """Put each complete group's pooled key in its last token slot.

        The other slots retain pre-pool token keys. This lets an unfinished
        group span calls, including after a trim/restore, without replaying the
        pooling for past groups or maintaining untracked Python side state.
        """
        tokens = self.token_key_norm(self.token_key_projection(x))
        ratio = self.compress_ratio
        tail = offset % ratio
        count = (tail + x.shape[1]) // ratio
        if not count:
            return tokens
        if tail:
            previous = cache.state[1][:, 0, offset - tail : offset, rope_dim:]
            grouped = mx.concatenate((previous, tokens), axis=1)
        else:
            grouped = tokens
        grouped = grouped[:, : count * ratio].reshape(
            x.shape[0], count, ratio, self.head_dim
        )
        # Both projections, the intra-group APE and the pooling operate in fp32.
        grouped = grouped.astype(mx.float32)
        gates = self.key_gate_projection(grouped) + self.intra_group_ape
        pooled = (
            self.key_value_projection(grouped) * mx.softmax(gates, axis=2, precise=True)
        ).sum(axis=2)
        pooled = self.key_norm(pooled).astype(tokens.dtype)
        starts = offset - tail + ratio * mx.arange(count)
        pooled = _rotate(pooled[:, :, None], rope.phases(starts, tokens.dtype))[:, :, 0]
        tokens[:, ratio - tail - 1 :: ratio] = pooled
        return tokens

    def recall(self, x, q_resid, groups, slots, phases):
        """Return [batch, queries, budget] slots; callers mask the own-group tail."""
        batch, length = x.shape[:2]
        own = slots // self.compress_ratio
        own = mx.broadcast_to(own[None, :, None], (batch, length, 1))
        if self.group_topk > 1:
            q = self.query_projection(q_resid).reshape(
                batch, length, self.num_heads, self.head_dim
            )
            q = _rotate(q, phases).astype(mx.float32)
            scores = mx.maximum(
                q @ groups[:, None].astype(mx.float32).swapaxes(-1, -2), 0
            )
            weights = (
                self.head_weight_projection(x).astype(mx.float32)
                * (self.num_heads * self.head_dim) ** -0.5
            )
            logits = (weights[:, :, None] @ scores).squeeze(2)
            logits = mx.where(mx.arange(groups.shape[1]) < own, logits, -mx.inf)
            # Stable tie-break by group slot, independent of causal chunk length.
            complete = mx.argsort(-logits, axis=-1)[..., : self.group_topk - 1]
            recalled = mx.concatenate((complete.astype(mx.int32), own), axis=-1)
        else:
            recalled = own
        return (
            recalled[..., None] * self.compress_ratio + mx.arange(self.compress_ratio)
        ).flatten(-2)


class Attention(nn.Module):
    def __init__(self, args, zero_based_block):
        super().__init__()
        kind = args.layer_types[zero_based_block]
        if kind not in (
            "sliding_attention",
            "deepseek_sparse_attention",
            "indexed_attention",
        ):
            raise ValueError(f"Unsupported HY V4 Flash attention type: {kind!r}")
        sliding = kind == "sliding_attention"
        prefix = "swa_" if sliding else ""
        self.window = args.sliding_window if sliding else None
        if sliding and self.window <= 0:
            raise ValueError("sliding_window must be positive")
        self.num_heads = getattr(args, prefix + "num_attention_heads")
        self.nope_dim = getattr(args, prefix + "qk_nope_head_dim")
        self.rope_dim = getattr(args, prefix + "qk_rope_head_dim")
        self.v_dim = getattr(args, prefix + "v_head_dim")
        self.qk_dim = self.nope_dim + self.rope_dim
        self.rank = args.kv_lora_rank
        self.scale = self.qk_dim**-0.5
        self._rope = _TextRoPE(
            self.rope_dim,
            args.rope_parameters[kind],
            getattr(args, "max_position_embeddings", None),
        )

        h, hidden, bias = self.num_heads, args.hidden_size, args.attention_bias
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=bias)
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(args.q_lora_rank, h * self.qk_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.rank + self.rope_dim, bias=bias
        )
        self.kv_a_layernorm = RMSNorm(self.rank, eps=args.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.rank, h * (self.nope_dim + self.v_dim), bias=False
        )
        self.learnable_sink_param = mx.full(
            (h,), args.learnable_sink_init, dtype=mx.float32
        )
        self.linear_gate = nn.Linear(hidden, h * self.v_dim, bias=False)
        self.o_proj = nn.Linear(h * self.v_dim, hidden, bias=bias)
        self.post_proj_norm = RMSNorm(hidden, eps=args.rms_norm_eps)
        self.compressed_indexer = None if sliding else CompressedIndexer(args)

    def make_cache(self):
        if self.window is not None:
            return RotatingKVCache(max_size=self.window, keep=0)
        return KVCache()

    def _aggregate(self, q, latent, k_rot, allowed, sink):
        keys = mx.concatenate((latent, k_rot), axis=-1)
        logits = (q @ keys.swapaxes(-1, -2)) * self.scale
        logits = mx.where(allowed, logits, -mx.inf)
        # A sink is one extra per-head logit with a zero value, not a KV token.
        sink = mx.broadcast_to(sink, logits.shape[:-1] + (1,))
        probs = mx.softmax(
            mx.concatenate((logits, sink), axis=-1), axis=-1, precise=True
        )
        return probs[..., :-1].astype(latent.dtype) @ latent

    def _contiguous(self, q, latent, k_rot, start, key_base):
        # A windowed panel never constructs a full history by history mask.
        out = []
        for i in range(0, q.shape[1], 128):
            row = q[:, i : i + 128].swapaxes(1, 2)
            lo = max(0, start + i - self.window + 1 - key_base) if self.window else 0
            hi = start + i + row.shape[2] - key_base
            slots = mx.arange(key_base + lo, key_base + hi)
            pos = mx.arange(start + i, start + i + row.shape[2])[:, None]
            allowed = slots <= pos
            if self.window:
                allowed = allowed & (slots > pos - self.window)
            y = self._aggregate(
                row,
                latent[:, None, lo:hi],
                k_rot[:, None, lo:hi],
                allowed,
                self.learnable_sink_param[None, :, None, None],
            )
            out.append(y.swapaxes(1, 2))
        return out

    def _sparse(self, q, x, q_resid, latent, k_rot, encoded, start):
        indexer = self.compressed_indexer
        batch, length = x.shape[:2]
        groups = encoded[:, indexer.compress_ratio - 1 :: indexer.compress_ratio]
        # Cap the [B, queries, index-heads, groups] temporary as history grows.
        panel = max(
            1, min(16, (1 << 20) // (batch * indexer.num_heads * groups.shape[1]))
        )
        out = []
        for i in range(0, length, panel):
            j = min(i + panel, length)
            slots = mx.arange(start + i, start + j)
            closed = (start + j - 1) // indexer.compress_ratio
            ids = indexer.recall(
                x[:, i:j],
                q_resid[:, i:j],
                groups[:, :closed],
                slots,
                self._rope.phases(slots, x.dtype),
            )
            allowed = ids <= slots[None, :, None]
            # Out-of-range slots in an incomplete group are masked, not
            # duplicate copies of its last token. The other groups are distinct.
            safe = mx.minimum(ids, latent.shape[1] - 1)
            b = mx.arange(batch)[:, None, None]
            out.append(
                self._aggregate(
                    q[:, i:j],
                    latent[b, safe],
                    k_rot[b, safe],
                    allowed[:, :, None],
                    self.learnable_sink_param[None, None, :, None],
                )
            )
        return out

    def __call__(self, x, cache=None):
        batch, length, _ = x.shape
        if not batch or not length:
            raise ValueError("Attention needs at least one unpadded text token")
        if cache is not None:
            if self.window is None and not isinstance(cache, KVCache):
                raise ValueError(
                    "Indexed attention requires the full latent history; use make_cache()"
                )
            if self.window is not None and not (
                isinstance(cache, RotatingKVCache)
                and cache.max_size == self.window
                and cache.keep == 0
            ):
                raise ValueError(
                    "Sliding attention needs its configured window; use make_cache()"
                )
        start = cache.offset if cache is not None else 0
        if not isinstance(start, int):
            raise ValueError(
                "HY V4 Flash attention requires equal, unpadded text offsets"
            )
        slots = mx.arange(start, start + length)
        phases = self._rope.phases(slots, x.dtype)
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).reshape(batch, length, self.num_heads, self.qk_dim)
        q_nope, q_rot = mx.split(q, [self.nope_dim], axis=-1)
        kv, k_rot = mx.split(self.kv_a_proj_with_mqa(x), [self.rank], axis=-1)
        latent = self.kv_a_layernorm(kv)
        k_rot = _rotate(k_rot[:, :, None], phases)[:, :, 0]

        # Absorb the public (unquantized) K expansion into Q, and postpone
        # the V expansion until after reduction. Never expand all past latents
        # across heads, including for sparse multi-token prefill.
        w = self.kv_b_proj.weight.reshape(
            self.num_heads, self.nope_dim + self.v_dim, self.rank
        )
        wk, wv = mx.split(w, [self.nope_dim], axis=1)
        q_abs = (q_nope.swapaxes(1, 2) @ wk).swapaxes(1, 2)
        q = mx.concatenate((q_abs, _rotate(q_rot, phases)), axis=-1)

        indexer = self.compressed_indexer
        sparse_at = (
            length if indexer is None else min(length, max(0, indexer.budget - start))
        )
        if indexer is not None and (cache is not None or sparse_at < length):
            encoded = indexer.encode_slots(x, cache, start, self._rope, self.rope_dim)
            k_rot = mx.concatenate((k_rot, encoded), axis=-1)
        if cache is not None:
            latent, k_rot = cache.update_and_fetch(latent[:, None], k_rot[:, None])
            latent, k_rot = latent[:, 0], k_rot[:, 0]
        encoded = k_rot[..., self.rope_dim :]
        k_rot = k_rot[..., : self.rope_dim]
        key_base = start + length - latent.shape[1]
        panels = self._contiguous(q[:, :sparse_at], latent, k_rot, start, key_base)
        if sparse_at < length:
            panels.extend(
                self._sparse(
                    q[:, sparse_at:],
                    x[:, sparse_at:],
                    q_resid[:, sparse_at:],
                    latent,
                    k_rot,
                    encoded,
                    start + sparse_at,
                )
            )
        reduced = panels[0] if len(panels) == 1 else mx.concatenate(panels, axis=1)
        values = (reduced.swapaxes(1, 2) @ wv.swapaxes(-1, -2)).swapaxes(1, 2)
        gated = values.flatten(2) * mx.sigmoid(self.linear_gate(x))
        return self.post_proj_norm(self.o_proj(gated))
