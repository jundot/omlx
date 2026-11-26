# SPDX-License-Identifier: Apache-2.0
"""Layers shared by Hy4 and Hy4-flash: iHC, gated MLA with sinks, MoE, DSA.

Parameter names follow the released checkpoints (``linear_gate``,
``learnable_sink_param``, ``hc_attn_layer.hc_pre.hc_fn``), not the renamed
transformers modules. ``kv_b_proj`` is split into ``embed_q`` / ``unembed_out``
at load time, as in mlx-lm's DeepSeek-V3.2.
"""

from __future__ import annotations

import functools
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.mla import MultiLinear
from mlx_vlm.models.base import create_attention_mask

from ...custom_kernels.attention import fast as attn_fast
from ...custom_kernels.nax import is_nax_available
from ...oqh.modules import exact_rows as oqh_exact_rows
from ..deepseek_v4.switch_layers import SwitchGLU, _sort_threshold
from ..glm_moe_dsa.sparse_mla_nax import sparse_mla_attention_nax
from . import exact, indexing, verify_dispatch
from . import kernels as hc_kernels
from .config import TextConfig
from .flash import CompressedIndexer
from .rope import TrailingRoPE, rotate
from .verify_moe import routed_experts
from .verify_sparse import attend as attend_sparse_verify

_ROW_CHUNK = 256
# Up to this many query rows, attention gathers each row's own keys instead of
# running the prefill-shaped kernels (draft verify blocks, short chunks).
_VERIFY_ROWS = 16


@functools.cache
def _nax_host() -> bool:
    return bool(is_nax_available())


def _nax_attention(queries, keys, values, scale, mask, sinks):
    """MLX's fused tensor-unit attention for the 256-wide heads, or None.

    On M5 MLX has a split-head-dim kernel for 256-wide heads that takes the
    boolean mask and the sinks; it must be requested explicitly for shorter
    query blocks. Older MLX builds without ``force_fused`` keep the block-mask
    kernel.
    """
    try:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
            force_fused=True,
        )
    except (TypeError, ValueError):
        return None


def _cache_length(cache) -> int | None:
    """Key slots of a one-row cache before this call (physical for batch caches)."""
    if cache is None:
        return None
    if hasattr(cache, "left_padding"):
        return int(cache._idx)
    return int(cache.offset)


def masked_min(dtype) -> mx.array:
    return mx.array(mx.finfo(dtype).min, dtype)


def _sliding_shared_values(q, kv, mask, sinks, scale):
    """Apply the stock masked D512 softmax, then multiply the shared V once.

    MLX's D512 fallback treats the value product as one matmul per query
    head. Here every head uses the same single KV bank, so the value product
    can use the flattened head and query rows of one matmul.
    """
    _, heads, rows, channels = q.shape
    width = kv.shape[2]
    logits = (q * scale) @ kv.swapaxes(-1, -2) + mask
    sink_column = mx.broadcast_to(sinks[None, :, None, None], (1, heads, rows, 1))
    weights = mx.softmax(
        mx.concatenate((sink_column, logits), axis=-1), axis=-1, precise=True
    )[..., 1:]
    return (weights.reshape(1, heads * rows, width) @ kv[0]).reshape(
        1, heads, rows, channels
    )


def rope_offset(cache) -> Any:
    if cache is None:
        return 0
    return cache.offset


class _FusedWeight:
    """The fp32 mix weight transposed for the fused kernel, rebuilt on reload."""

    def __init__(self):
        self._source = None
        self._value = None

    def get(self, weight: mx.array) -> mx.array:
        if self._source is not weight:
            self._value = mx.contiguous(weight.astype(mx.float32).T)
            self._source = weight
        return self._value


class HyperConnectionPre(nn.Module):
    """Identity hyper-connection gates (DeepSeek-V4 HC without the comb matrix).

    The gates are computed in fp32 from the RMS-normalized flattened streams.
    """

    def __init__(self, args: TextConfig):
        super().__init__()
        n = args.hc_mult
        self.hc_fn = mx.zeros((2 * n, n * args.hidden_size), dtype=mx.float32)
        self.hc_base = mx.zeros((2 * n,), dtype=mx.float32)
        self.hc_scale = mx.ones((2,), dtype=mx.float32)
        self._n = n
        self._eps = args.hc_eps
        self._norm_eps = args.rms_norm_eps
        self._magnitude = args.hc_magnitude
        self._fused = _FusedWeight()

    def __call__(self, streams: mx.array) -> tuple[mx.array, mx.array]:
        if not hc_kernels.available():
            return self.reference(streams)
        return self._mix(streams, self._fused.get(self.hc_fn))

    def _mix(self, streams, transposed):
        return hc_kernels.hc_pre(
            streams,
            transposed,
            self.hc_base.astype(mx.float32),
            self.hc_scale.astype(mx.float32),
            self._norm_eps,
            self._eps,
            self._magnitude,
        )

    def reference(self, streams: mx.array) -> tuple[mx.array, mx.array]:
        flat = streams.astype(mx.float32).flatten(-2)
        inv_rms = mx.rsqrt(
            mx.mean(flat * flat, axis=-1, keepdims=True) + self._norm_eps
        )
        mixes = (flat @ self.hc_fn.astype(mx.float32).T) * inv_rms
        pre, post = mx.split(mixes, 2, axis=-1)
        pre_base, post_base = mx.split(self.hc_base.astype(mx.float32), 2)
        scale = self.hc_scale.astype(mx.float32)
        pre = mx.sigmoid(pre * scale[0] + pre_base) + self._eps
        post = self._magnitude * mx.sigmoid(post * scale[1] + post_base) + self._eps
        collapsed = (pre[..., None] * streams.astype(mx.float32)).sum(axis=-2)
        return collapsed.astype(streams.dtype), post


class HyperConnection(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.hc_pre = HyperConnectionPre(args)

    def __call__(self, streams: mx.array) -> tuple[mx.array, mx.array]:
        return self.hc_pre(streams)


@mx.compile
def hc_expand(update: mx.array, streams: mx.array, post: mx.array) -> mx.array:
    """Add one sublayer output to every stream, scaled by its post gate."""
    out = (
        streams.astype(mx.float32)
        + post[..., None] * update.astype(mx.float32)[..., None, :]
    )
    return out.astype(streams.dtype)


class HyperHead(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        n = args.hc_mult
        self.hc_head_fn = mx.zeros((n, n * args.hidden_size), dtype=mx.float32)
        self.hc_head_base = mx.zeros((n,), dtype=mx.float32)
        self.hc_head_scale = mx.ones((1,), dtype=mx.float32)
        self._eps = args.hc_eps
        self._norm_eps = args.rms_norm_eps
        self._fused = _FusedWeight()

    def __call__(self, streams: mx.array) -> mx.array:
        if not hc_kernels.available():
            return self.reference(streams)
        collapsed, _ = hc_kernels.hc_pre(
            streams,
            self._fused.get(self.hc_head_fn),
            self.hc_head_base.astype(mx.float32),
            self.hc_head_scale.astype(mx.float32),
            self._norm_eps,
            self._eps,
            None,
        )
        return collapsed

    def reference(self, streams: mx.array) -> mx.array:
        flat = streams.astype(mx.float32).flatten(-2)
        inv_rms = mx.rsqrt(
            mx.mean(flat * flat, axis=-1, keepdims=True) + self._norm_eps
        )
        mixes = (flat @ self.hc_head_fn.astype(mx.float32).T) * inv_rms
        pre = (
            mx.sigmoid(
                mixes * self.hc_head_scale.astype(mx.float32)
                + self.hc_head_base.astype(mx.float32)
            )
            + self._eps
        )
        out = (pre[..., None] * streams.astype(mx.float32)).sum(axis=-2)
        return out.astype(streams.dtype)


class MLP(nn.Module):
    """Dense SwiGLU (dense layer and shared expert). Hy4-flash's dense layer
    closes with ``post_fc2_norm`` (the sandwich norm)."""

    def __init__(self, args: TextConfig, intermediate: int, post_norm: bool = False):
        super().__init__()
        d = args.hidden_size
        self.gate_proj = nn.Linear(d, intermediate, bias=False)
        self.up_proj = nn.Linear(d, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, d, bias=False)
        if post_norm:
            self.post_fc2_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        act = nn.silu(exact.linear(self.gate_proj, x)) * exact.linear(self.up_proj, x)
        y = exact.linear(self.down_proj, act)
        if "post_fc2_norm" in self:
            y = self.post_fc2_norm(y)
        return y


@mx.compile
def _clamped_swiglu(up: mx.array, gate: mx.array, limit: mx.array) -> mx.array:
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


class ClampedSwiGLU(nn.Module):
    """SwiGLU with the routed experts' clamp: gate <= limit, |up| <= limit."""

    def __init__(self, limit: float):
        super().__init__()
        # oQh's fused expert activation reads the same clamp from ``limit``.
        self.limit = float(limit) if limit and limit > 0 else None

    def __call__(self, up: mx.array, gate: mx.array) -> mx.array:
        if self.limit is None:
            return nn.silu(gate) * up
        return _clamped_swiglu(up, gate, mx.array(self.limit, dtype=up.dtype))


class Router(nn.Module):
    """Sigmoid router with expert-bias correction (scores stay unbiased)."""

    def __init__(self, args: TextConfig):
        super().__init__()
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros(
            (args.n_routed_experts,), dtype=mx.float32
        )
        self._top_k = args.num_experts_per_tok
        self._n_group = args.n_group
        self._topk_group = args.topk_group
        self._norm = args.norm_topk_prob
        self._scale = args.routed_scaling_factor

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        weight_t = self.weight.astype(mx.float32).T
        if exact.block(x):
            logits = exact.rowwise(lambda r: r.astype(mx.float32) @ weight_t, x)
        else:
            logits = x.astype(mx.float32) @ weight_t
        scores = mx.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)
        if self._n_group > 1:
            grouped = mx.unflatten(choice, -1, (self._n_group, -1))
            best = mx.topk(grouped, 2, axis=-1).sum(axis=-1, keepdims=True)
            drop = self._n_group - self._topk_group
            groups = mx.argpartition(best, kth=drop - 1, axis=-2)[..., :drop, :]
            grouped = mx.put_along_axis(grouped, groups, mx.array(-mx.inf), axis=-2)
            choice = mx.flatten(grouped, -2, -1)
        k = self._top_k
        inds = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self._norm:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, weights * self._scale


class MoE(nn.Module):
    """Routed experts plus one shared expert.

    Hy4-flash runs the routed experts in a latent width
    (``routed_expert_down_proj`` / ``routed_expert_up_proj``) and closes the
    block with ``combine_output_norm``.
    """

    def __init__(self, args: TextConfig):
        super().__init__()
        d = args.hidden_size
        width = args.routed_expert_hidden_size if args.is_flash else None
        self.gate = Router(args)
        if width:
            self.routed_expert_down_proj = nn.Linear(d, width, bias=False)
            self.routed_expert_up_proj = nn.Linear(width, d, bias=False)
        self.switch_mlp = SwitchGLU(
            width or d,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=ClampedSwiGLU(args.swiglu_limit),
        )
        self.switch_mlp._split_affine_prefill = True
        self.shared_experts = MLP(
            args, args.moe_intermediate_size * args.n_shared_experts
        )
        if args.is_flash:
            self.combine_output_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        inds, weights = self.gate(x)
        routed = x
        if "routed_expert_down_proj" in self:
            routed = exact.linear(self.routed_expert_down_proj, x)
        if exact.block(x):
            y = self._exact_experts(routed, inds, weights)
        else:
            y = None
            if verify_dispatch.eligible(routed.shape[0]):
                activation = getattr(self.switch_mlp, "activation", None)
                if type(activation) is ClampedSwiGLU and activation.limit is not None:
                    y = routed_experts(
                        self.switch_mlp, routed, inds, weights, activation.limit
                    )
            if y is None:
                y = self.switch_mlp(routed, inds, scores=weights, weighted_sum=True)
            if y.ndim == routed.ndim + 1:
                y = (y * weights[..., None].astype(y.dtype)).sum(axis=-2)
        if "routed_expert_up_proj" in self:
            y = exact.linear(self.routed_expert_up_proj, y)
        y = y + self.shared_experts(x)
        if "combine_output_norm" in self:
            y = self.combine_output_norm(y)
        return y.reshape(shape)

    def _exact_experts(self, routed, inds, weights):
        """Routed experts of a verify block with one-token decode arithmetic.

        Decode's top-k routes stay below the sort threshold, so they run the
        unsorted gather; rows are grouped to stay below it as well.
        """
        switch = self.switch_mlp
        k = inds.shape[-1]
        if hasattr(switch, "oqh_experts"):
            with oqh_exact_rows():
                parts = [switch(routed, inds, scores=weights, weighted_sum=True)]
        else:
            group = max(1, (_sort_threshold(*switch.projections()) - 1) // k)
            parts = [
                switch(routed[i : i + group], inds[i : i + group], scores=weights)
                for i in range(0, routed.shape[0], group)
            ]
        y = mx.concatenate(parts) if len(parts) > 1 else parts[0]
        return (y * weights[..., None].astype(y.dtype)).sum(axis=-2)


class Indexer(nn.Module):
    """Hy4 DSA indexer: head-weighted ReLU scores over cached index keys.

    Unlike DeepSeek-V3.2 it has no Hadamard rotation and rotates the trailing
    ``qk_rope_head_dim`` channels. ``k_norm`` is a LayerNorm with bias.
    """

    def __init__(self, args: TextConfig):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.topk = args.index_topk
        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=args.rms_norm_eps)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.rope = TrailingRoPE(args.qk_rope_head_dim, args.rope_theta("sparse"))
        self._weight_scale = self.n_heads**-0.5 * self.head_dim**-0.5

    def __call__(
        self, x, qr, mask, cache, offset=0, positions=None
    ) -> mx.array | list | None:
        if exact.block(x):
            # Per-row selections (None below the top-k) of a verify block.
            folded = self._exact_select(x, qr, mask, cache, offset, positions)
            if folded is not None:
                return folded
            length = x.shape[1]
            mask = as_4d_mask(mask)
            keys = 0 if mask is None else mask.shape[-1]
            return [
                self(
                    x[:, i : i + 1],
                    qr[:, i : i + 1],
                    exact.row_mask(mask, i, keys - length + i + 1),
                    cache,
                    offset + i,
                    None if positions is None else positions.row(i),
                )
                for i in range(length)
            ]
        b, s, _ = x.shape
        q = self.wq_b(qr).reshape(b, s, self.n_heads, self.head_dim).swapaxes(1, 2)
        q = rotate(self.rope, q, offset, positions)
        k = self.k_norm(self.wk(x).astype(mx.float32)).astype(x.dtype)
        k = rotate(self.rope, k.reshape(b, 1, s, self.head_dim), offset, positions)
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((b, 1, s, 0), dtype=k.dtype))
        total = k.shape[2]
        if total <= self.topk:
            return None
        w = self.weights_proj(x.astype(mx.float32)).astype(mx.float32)
        w = w * self._weight_scale
        native = self._native_select(q, k, w, mask, total)
        if native is not None:
            return native
        w = w.swapaxes(-1, -2)[..., None]
        kf = k.astype(mx.float32).swapaxes(-1, -2)
        out = []
        for start in range(0, s, _ROW_CHUNK):
            stop = min(s, start + _ROW_CHUNK)
            scores = mx.maximum(q[:, :, start:stop].astype(mx.float32) @ kf, 0)
            scores = (scores * w[:, :, start:stop]).sum(axis=1, keepdims=True)
            if mask is not None:
                scores = mx.where(mask[..., start:stop, :], scores, -mx.inf)
            chosen = mx.argpartition(-scores, kth=self.topk - 1, axis=-1)
            chosen = chosen[..., : self.topk]
            # A row with fewer visible keys than topk also picks masked ones
            # (padding or future slots); point those past the cache instead.
            picked = mx.take_along_axis(scores, chosen, axis=-1)
            out.append(mx.where(picked > -mx.inf, chosen, total).astype(mx.int32))
        return out[0] if len(out) == 1 else mx.concatenate(out, axis=2)

    def _exact_select(self, x, qr, mask, cache, offset, positions) -> list | None:
        """A verify block's per-row selections, scored together.

        Only while every row would decode sparse on the decode-score kernel;
        otherwise each row runs the one-row path in turn.
        """
        length = x.shape[1]
        before = _cache_length(cache)
        if before is None or before < self.topk:
            return None
        q = exact.linear(self.wq_b, qr)
        if not indexing.decode_rows_ready(self.n_heads, before + 1, q.dtype):
            return None
        q = q.reshape(1, length, self.n_heads, self.head_dim).swapaxes(1, 2)
        q = rotate(self.rope, q, offset, positions)
        k = self.k_norm(exact.linear(self.wk, x).astype(mx.float32)).astype(x.dtype)
        k = rotate(self.rope, k.reshape(1, 1, length, self.head_dim), offset, positions)
        k, _ = cache.update_and_fetch(k, mx.zeros((1, 1, length, 0), dtype=k.dtype))
        w = exact.linear(self.weights_proj, x.astype(mx.float32)).astype(mx.float32)
        totals = [before + i + 1 for i in range(length)]
        return indexing.decode_rows_select(
            q, k, w * self._weight_scale, totals, as_4d_mask(mask), self.topk
        )

    def _native_select(self, q, k, w, mask, total) -> mx.array | None:
        """The same selection on the sparse_index kernels (bf16 scores)."""
        scores = indexing.scores(q, k, w, causal=True)
        if scores is None:
            return None
        if mask is not None:
            scores = mx.where(mask, scores, -mx.inf)
        chosen = indexing.topk(scores, self.topk)
        if chosen is None:
            return None
        picked = mx.take_along_axis(scores, chosen, axis=-1)
        return mx.where(picked > -mx.inf, chosen, total)


def sparse_mask(indices: mx.array, total: int) -> mx.array:
    """Boolean [B, 1, L, total] mask of the selected key slots.

    Indices may point past ``total`` (an unfinished group). They land in a
    spare column that is dropped, so they never clear a selected slot.
    """
    b, _, length, _ = indices.shape
    target = mx.minimum(indices, total)
    hits = mx.zeros((b, 1, length, total + 1), dtype=mx.bool_)
    hits = mx.put_along_axis(hits, target, mx.array(True), axis=-1)
    return hits[..., :total]


_TOKEN_BITS = 16
_QUERY_BLOCK = 32


def block_masks(mask: mx.array, length: int, total: int):
    """Pack a boolean [..., length, total] mask for block-mask attention.

    Returns the [B, 1, ceil(length / 32), ceil(total / 16)] block mask and the
    [B, 1, length, ceil(total / 16)] per-row 16-key bitmask.
    """
    mask = as_4d_mask(mask)
    mask = mx.broadcast_to(mask, mask.shape[:2] + (length, total))
    pad_k = (-total) % _TOKEN_BITS
    if pad_k:
        mask = mx.pad(mask, [(0, 0)] * 3 + [(0, pad_k)])
    weights = mx.left_shift(
        mx.array(1, dtype=mx.uint32), mx.arange(_TOKEN_BITS, dtype=mx.uint32)
    )
    bits = (
        mask.reshape(*mask.shape[:3], -1, _TOKEN_BITS).astype(mx.uint32) * weights
    ).sum(axis=-1)
    bits = bits.astype(mx.uint32)
    pad_q = (-length) % _QUERY_BLOCK
    rows = mx.pad(bits, [(0, 0), (0, 0), (0, pad_q), (0, 0)]) if pad_q else bits
    blocks = rows.reshape(*rows.shape[:2], -1, _QUERY_BLOCK, rows.shape[-1])
    return mx.any(blocks != 0, axis=3), bits


def as_4d_mask(mask: mx.array | None) -> mx.array | None:
    if mask is None or mask.ndim == 4:
        return mask
    return mask.reshape((1,) * (4 - mask.ndim) + mask.shape)


class Attention(nn.Module):
    """Gated MLA with learned per-head sinks.

    Sparse layers attend to the top-k slots their indexer (or the previous full
    layer, on Hy4's shared layers) selects; sliding layers (Hy4-flash) attend to
    the last ``sliding_window`` keys.
    """

    def __init__(self, args: TextConfig, layer: int):
        super().__init__()
        self.kind = args.attention_kinds[layer]
        heads, nope, rope, value = args.head_dims(layer)
        self.num_heads = heads
        self.nope = nope
        self.rope_dim = rope
        self.v_head_dim = value
        self.rank = args.kv_lora_rank
        self.scale = (nope + rope) ** -0.5
        self.window = args.sliding_window if self.kind == "sliding" else None
        d, bias = args.hidden_size, args.attention_bias
        self.q_a_proj = nn.Linear(d, args.q_lora_rank, bias=bias)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(args.q_lora_rank, heads * (nope + rope), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d, self.rank + rope, bias=bias)
        self.kv_a_layernorm = nn.RMSNorm(self.rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(nope, self.rank, heads)
        self.unembed_out = MultiLinear(self.rank, value, heads)
        self.linear_gate = nn.Linear(d, heads * value, bias=False)
        self.o_proj = nn.Linear(heads * value, d, bias=bias)
        self.learnable_sink_param = mx.full(
            (heads,), args.learnable_sink_init, dtype=mx.float32
        )
        if args.is_flash:
            self.post_proj_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(rope, traditional=True, base=args.rope_theta(self.kind))
        if self.kind == "sparse":
            if args.is_flash:
                self.compressed_indexer = CompressedIndexer(args)
            elif args.indexer_types[layer] == "full":
                self.indexer = Indexer(args)
        self._capture_topk = False
        self._captured_topk = None

    @property
    def index_module(self) -> nn.Module | None:
        for name in ("indexer", "compressed_indexer"):
            if name in self:
                return self[name]
        return None

    def _caches(self, cache):
        """(latent cache, indexer cache) of this layer."""
        if cache is None:
            return None, None
        if self.kind == "sliding":
            return cache, None
        return cache[0], (cache[1] if len(cache.caches) > 1 else None)

    def _project_inputs(self, x):
        b, length, _ = x.shape
        gate = mx.sigmoid(exact.linear(self.linear_gate, x)).reshape(
            b, length, self.num_heads, self.v_head_dim
        )
        qr = self.q_a_layernorm(exact.linear(self.q_a_proj, x))
        q = exact.linear(self.q_b_proj, qr)
        q = q.reshape(b, length, self.num_heads, -1).swapaxes(1, 2)
        q_nope, q_pe = mx.split(q, [self.nope], axis=-1)
        latent, k_pe = mx.split(
            exact.linear(self.kv_a_proj_with_mqa, x), [self.rank], axis=-1
        )
        latent = self.kv_a_layernorm(latent)[:, None]
        k_pe = k_pe.reshape(b, 1, length, self.rope_dim)
        return gate, qr, q_nope, q_pe, latent, k_pe

    def _prepare_cache(self, x, projected, mask, cache, prev_topk, positions):
        kv_cache, index_cache = self._caches(cache)
        gate, qr, q_nope, q_pe, latent, k_pe = projected
        offset = rope_offset(kv_cache)
        q_pe = rotate(self.rope, q_pe, offset, positions)
        k_pe = rotate(self.rope, k_pe, offset, positions)
        # oMLX's BatchKVCache advances its offset array in place, so every use
        # of ``offset`` must come before the latent cache update.
        topk = None
        indexer = self.index_module
        if indexer is not None:
            topk = indexer(x, qr, mask, index_cache, offset=offset, positions=positions)
        elif self.kind == "sparse":
            topk = prev_topk
        if self._capture_topk:
            # DFly reuses this layer's selection for the last committed row.
            self._captured_topk = topk
        sinks = self.learnable_sink_param.astype(x.dtype)
        return kv_cache, topk, gate, q_nope, q_pe, latent, k_pe, sinks

    def __call__(
        self,
        x,
        mask,
        cache=None,
        prev_topk=None,
        positions=None,
        *,
        return_values=False,
    ):
        b, length = x.shape[:2]
        kv_cache, topk, gate, q_nope, q_pe, latent, k_pe, sinks = self._prepare_cache(
            x, self._project_inputs(x), mask, cache, prev_topk, positions
        )
        if exact.block(x):
            out = self._exact_rows(
                q_nope, q_pe, latent, k_pe, topk, mask, kv_cache, sinks
            )
            if return_values:
                return (out, gate), topk
            out = out.swapaxes(1, 2) * gate
            out = exact.linear(self.o_proj, out.reshape(b, length, -1))
            if "post_proj_norm" in self:
                out = self.post_proj_norm(out)
            return out, topk
        if kv_cache is not None:
            latent, k_pe = kv_cache.update_and_fetch(latent, k_pe)

        if length == 1:
            out = self._decode(q_nope, q_pe, latent, k_pe, topk, mask, sinks)
        elif length <= _VERIFY_ROWS:
            out = self._verify(q_nope, q_pe, latent, k_pe, topk, mask, sinks)
        else:
            out = self._prefill(q_nope, q_pe, latent, k_pe, topk, mask, sinks)
        if return_values:
            return (out, gate), topk
        return self._project_output(out, gate), topk

    def _project_output(self, values, gate):
        b, length = gate.shape[:2]
        out = values.swapaxes(1, 2) * gate
        out = self.o_proj(out.reshape(b, length, -1))
        if "post_proj_norm" in self:
            out = self.post_proj_norm(out)
        return out

    def _exact_rows(self, q_nope, q_pe, latent, k_pe, topk, mask, kv_cache, sinks):
        """One-token decode steps for the rows of an exact verify block.

        ``topk`` holds each row's own selection. A sliding window takes the
        rows one at a time, so its ring has decode's physical layout; each row
        keeps a copy of its step's view and the mask decode built for it.
        """
        length = q_nope.shape[2]
        views = []
        if self.kind == "sliding":
            batched = hasattr(kv_cache, "left_padding")
            for i in range(length):
                row_mask = None
                if batched:
                    row_mask = create_attention_mask(
                        latent[:, 0, i : i + 1],
                        kv_cache,
                        window_size=self.window,
                        return_array=True,
                    )
                lat, pe = kv_cache.update_and_fetch(
                    latent[..., i : i + 1, :], k_pe[..., i : i + 1, :]
                )
                views.append((lat + 0, pe + 0, row_mask))
        else:
            lat, pe = kv_cache.update_and_fetch(latent, k_pe)
            total = lat.shape[2]
            for i in range(length):
                end = total - length + i + 1
                row_mask = exact.row_mask(as_4d_mask(mask), i, end)
                views.append((lat[..., :end, :], pe[..., :end, :], row_mask))
        rows = []
        for i, (lat, pe, row_mask) in enumerate(views):
            row_topk = topk[i] if isinstance(topk, list) else None
            rows.append(
                self._decode(
                    q_nope[:, :, i : i + 1],
                    q_pe[:, :, i : i + 1],
                    lat,
                    pe,
                    row_topk,
                    row_mask,
                    sinks,
                )
            )
        return mx.concatenate(rows, axis=2)

    def _decode(self, q_nope, q_pe, latent, k_pe, topk, mask, sinks):
        if topk is not None:
            total = latent.shape[2]
            index = topk[:, :, 0, :, None]
            valid = topk[:, :, 0, :] < total
            safe = mx.minimum(index, total - 1)
            latent = mx.take_along_axis(
                latent, mx.broadcast_to(safe, safe.shape[:-1] + (latent.shape[-1],)), 2
            )
            k_pe = mx.take_along_axis(
                k_pe, mx.broadcast_to(safe, safe.shape[:-1] + (k_pe.shape[-1],)), 2
            )
            keep = valid[:, :, None, :]
            mask = as_4d_mask(mask)
            if mask is not None:
                mask = mx.broadcast_to(
                    mask, (topk.shape[0],) + mask.shape[1:3] + (total,)
                )
                keep = keep & mx.take_along_axis(
                    mask, mx.minimum(topk[:, :, :1, :], total - 1), axis=-1
                )
            mask = keep
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, masked_min(pe_scores.dtype))
        q_latent = self.embed_q(q_nope)
        out = mx.fast.scaled_dot_product_attention(
            q_latent, latent, latent, scale=self.scale, mask=pe_scores, sinks=sinks
        )
        return self.unembed_out(out)

    def _verify(self, q_nope, q_pe, latent, k_pe, topk, mask, sinks):
        """Absorbed attention for a few rows (draft verify, short chunks).

        Each row attends only to its own selection, folded into the batch
        axis, so the cost does not grow with the context.
        """
        b, heads, length, _ = q_nope.shape
        total = latent.shape[2]
        q_latent = self.embed_q(q_nope)
        rows = mx.arange(total - length, total)
        mask = as_4d_mask(mask)
        if (
            self.kind == "sparse"
            and verify_dispatch.eligible(length)
            and topk is not None
            and b == 1
            and heads == 32
            and topk.shape[-1] == 2048
            and q_pe.shape[-1] == k_pe.shape[-1] == 64
            and q_latent.shape[-1] == latent.shape[-1] == 512
            and q_latent.dtype
            == q_pe.dtype
            == latent.dtype
            == k_pe.dtype
            == sinks.dtype
            and latent.dtype in (mx.bfloat16, mx.float16)
        ):
            return self.unembed_out(
                attend_sparse_verify(
                    q_latent, q_pe, latent, k_pe, topk, mask, sinks, self.scale
                )
            )
        if topk is None:
            keep = rows[:, None] >= mx.arange(total)
            if mask is not None:
                keep = keep & mask
            scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            scores = mx.where(keep, scores, masked_min(scores.dtype))
            if (
                self.kind == "sliding"
                and b == 1
                and heads == 64
                and 512 <= total < 512 + length
                and q_latent.shape[-1] == 512
                and latent.shape == (1, 1, total, 512)
                and q_latent.dtype == latent.dtype == scores.dtype == sinks.dtype
                and latent.dtype == mx.bfloat16
                and verify_dispatch.eligible(length)
            ):
                return self.unembed_out(
                    _sliding_shared_values(q_latent, latent, scores, sinks, self.scale)
                )
            out = mx.fast.scaled_dot_product_attention(
                q_latent, latent, latent, scale=self.scale, mask=scores, sinks=sinks
            )
            return self.unembed_out(out)
        index = topk[:, 0]
        keep = (index < total) & (index <= rows[:, None])
        safe = mx.minimum(index, total - 1)
        if mask is not None:
            mask = mx.broadcast_to(mask, (b, 1, length, total))
            keep = keep & mx.take_along_axis(mask[:, 0], safe, axis=-1)
        width = index.shape[-1]
        gather = safe.reshape(b, length * width, 1)
        lat = mx.take_along_axis(latent[:, 0], gather, axis=1)
        pe = mx.take_along_axis(k_pe[:, 0], gather, axis=1)
        lat = lat.reshape(b * length, 1, width, -1)
        pe = pe.reshape(b * length, 1, width, -1)
        q_pe = q_pe.swapaxes(1, 2).reshape(b * length, heads, 1, -1)
        q_latent = q_latent.swapaxes(1, 2).reshape(b * length, heads, 1, -1)
        scores = (q_pe * self.scale) @ pe.swapaxes(-1, -2)
        keep = keep.reshape(b * length, 1, 1, width)
        scores = mx.where(keep, scores, masked_min(scores.dtype))
        out = mx.fast.scaled_dot_product_attention(
            q_latent, lat, lat, scale=self.scale, mask=scores, sinks=sinks
        )
        out = out.reshape(b, length, heads, -1).swapaxes(1, 2)
        return self.unembed_out(out)

    def _sparse_kernel(self, q_nope, q_pe, latent, k_pe, topk):
        """Absorbed sparse MLA over the selected slots on the native kernels.

        Out-of-range and future slots are skipped by the kernels; the indexers
        send masked picks past the cache, so batch padding never enters. M5
        hosts take the tensor-unit kernel (one sequence) first.
        """
        if not (
            self.num_heads in (32, 64)
            and self.rank == 512
            and self.rope_dim == 64
            and topk.shape[-1] == 2048
            and q_nope.dtype in (mx.float16, mx.bfloat16)
        ):
            return None
        q_latent = self.embed_q(q_nope).astype(q_nope.dtype)
        if _nax_host() and q_nope.shape[0] == 1:
            out = sparse_mla_attention_nax(
                q_latent,
                latent.astype(q_nope.dtype),
                topk.astype(mx.int32),
                self.scale,
                q_pe=q_pe,
                k_pe=k_pe,
                sinks=self.learnable_sink_param,
            )
            if out is not None:
                return self.unembed_out(out)
        if not attn_fast.NATIVE_AVAILABLE:
            return None
        out = attn_fast.sparse_mla_attention(
            q_latent,
            q_pe,
            latent.astype(q_nope.dtype),
            k_pe.astype(q_nope.dtype),
            topk.astype(mx.uint32),
            self.scale,
            sinks=self.learnable_sink_param,
        )
        return self.unembed_out(out)

    def _prefill(self, q_nope, q_pe, latent, k_pe, topk, mask, sinks):
        if topk is not None:
            out = self._sparse_kernel(q_nope, q_pe, latent, k_pe, topk)
            if out is not None:
                return out
        total = latent.shape[2]
        if mask is None:
            length = q_nope.shape[2]
            mask = mx.arange(total - length, total)[:, None] >= mx.arange(total)
        if topk is not None:
            mask = mask & sparse_mask(topk, total)
        keys = mx.concatenate(
            [
                self.embed_q(latent, transpose=False),
                mx.broadcast_to(
                    k_pe, k_pe.shape[:1] + (self.num_heads,) + k_pe.shape[2:]
                ),
            ],
            axis=-1,
        )
        values = self.unembed_out(latent)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        # Sliding windows keep the block-mask kernel on M5 too: it skips the
        # blocks outside the window (measured 7.4 ms against 9.7 ms fused).
        if (
            self.kind != "sliding"
            and queries.dtype in (mx.float16, mx.bfloat16)
            and _nax_host()
        ):
            out = _nax_attention(
                queries,
                keys.astype(queries.dtype),
                values.astype(queries.dtype),
                self.scale,
                mask,
                sinks,
            )
            if out is not None:
                return out
        if (
            attn_fast.NATIVE_AVAILABLE
            and queries.shape[-1] == 256
            and values.shape[-1] == 256
            and queries.dtype in (mx.float16, mx.bfloat16)
        ):
            block_mask, token_mask = block_masks(mask, queries.shape[2], total)
            return attn_fast.block_mask_attention(
                queries,
                keys.astype(queries.dtype),
                values.astype(queries.dtype),
                block_mask,
                token_mask,
                self.scale,
                causal=True,
                sinks=self.learnable_sink_param,
            )
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask, sinks=sinks
        )
