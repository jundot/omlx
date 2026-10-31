# SPDX-License-Identifier: Apache-2.0
"""Fused MoE routing and combine for one-token Flint decode.

Flint's MoE block (mlx-vlm ``models/flint``) routes with a sigmoid over 256
expert logits, adds a selection-only bias, takes the top 6 and renormalizes
the selected sigmoid scores. At batch-one decode that chain is about eight
tiny-tensor launches per layer (cast, sigmoid, bias add, argpartition,
slice, take, sum, divide), and the combine after the experts (cast, score
multiply, k-sum, cast, shared add, residual add) about six more. Across 47
MoE layers these launches are pure overhead: measured ~1.7 ms of an
~11 ms decode step on M4 Max for a router whose weights are 1 MB a layer.

This patch runs one-token MoE layers as:

1. the gate linear (unchanged),
2. one router launch: precise sigmoid (MLX's ``Sigmoid`` formula), bias add,
   top-k by repeated threadgroup argmax, renormalization in fp32,
3. the routed experts and shared expert (unchanged modules),
4. one combine launch: fp32 score-weighted k-sum rounded to the activation
   dtype, plus the shared expert, plus the residual, with the stock
   path's roundings in its order.

The selected expert set matches the composed path except on exact fp32
ties of ``sigmoid + bias``. Those are common: the gate logits are bf16
and the shipped bias is zero, so ties resolve to the highest expert index,
mlx argpartition's pinned behavior at this shape. The
k-sums run in selection order rather than argpartition's, so outputs can
differ from the composed path by reduction-order rounding. Prefill and
multi-row calls keep the original body.
"""

from __future__ import annotations

import logging
from functools import cache

import mlx.core as mx

from .flint_fp16_activation import mlp_parts, routed_scale

logger = logging.getLogger(__name__)

_ROUTER_THREADS = 256
_MAX_EXPERTS = 1024
_MAX_TOP_K = 16

_ROUTER_SOURCE = """
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int NSG = TG / 32;

    threadgroup float score[E];
    threadgroup float choice[E];
    threadgroup float red_v[NSG];
    threadgroup int red_i[NSG];
    threadgroup int sel[K];

    for (int e = int(t); e < E; e += TG) {
        // MLX's Sigmoid op, in fp32 as the reference casts before it.
        float x = float(logits[e]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float s = (x < 0.0f) ? y : 1.0f - y;
        score[e] = s;
        choice[e] = s + float(bias[e]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < K; ++k) {
        float bv = -INFINITY;
        int bi = -1;
        for (int e = int(t); e < E; e += TG) {
            float v = choice[e];
            if (v >= bv) { bv = v; bi = e; }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_xor(bv, off);
            int oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red_v[sg] = bv; red_i[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            bv = lane < uint(NSG) ? red_v[lane] : -INFINITY;
            bi = lane < uint(NSG) ? red_i[lane] : -1;
            for (ushort off = 16; off > 0; off >>= 1) {
                float ov = simd_shuffle_xor(bv, off);
                int oi = simd_shuffle_xor(bi, off);
                if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) {
                sel[k] = bi;
                choice[bi] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t == 0) {
        float total = 0.0f;
        for (int k = 0; k < K; ++k) total += score[sel[k]];
        total += 1e-20f;
        for (int k = 0; k < K; ++k) {
            indices[k] = uint(sel[k]);
            weights[k] = score[sel[k]] / total;
        }
    }
"""

_COMBINE_SOURCE = """
    const uint d = thread_position_in_grid.x;
    if (d >= uint(D)) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += float(routed[k * D + d]) * weights[k];
    }
    T moe = static_cast<T>(acc);
    T mlp = static_cast<T>(float(moe) + float(shared[d]));
    out[d] = static_cast<T>(float(residual[d]) + float(mlp));
"""

# float16 range-safe SwiGLU (flint_fp16_activation): the routed and shared
# outputs carry a scale, undone with the roundings of the separate multiplies.
_SCALED_COMBINE_SOURCE = """
    const uint d = thread_position_in_grid.x;
    if (d >= uint(D)) return;
    const float rs = float(routed_scale[0]);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        T r = static_cast<T>(float(routed[k * D + d]) * rs);
        acc += float(r) * weights[k];
    }
    T moe = static_cast<T>(acc);
    T sh = static_cast<T>(float(shared[d]) * float(shared_scale[0]));
    T mlp = static_cast<T>(float(moe) + float(sh));
    out[d] = static_cast<T>(float(residual[d]) + float(mlp));
"""


@cache
def _router_kernel():
    return mx.fast.metal_kernel(
        name="omlx_flint_router_topk",
        input_names=["logits", "bias"],
        output_names=["indices", "weights"],
        source=_ROUTER_SOURCE,
    )


@cache
def _combine_kernel():
    return mx.fast.metal_kernel(
        name="omlx_flint_moe_combine",
        input_names=["routed", "weights", "shared", "residual"],
        output_names=["out"],
        source=_COMBINE_SOURCE,
    )


@cache
def _scaled_combine_kernel():
    return mx.fast.metal_kernel(
        name="omlx_flint_moe_scaled_combine",
        input_names=[
            "routed",
            "weights",
            "shared",
            "residual",
            "routed_scale",
            "shared_scale",
        ],
        output_names=["out"],
        source=_SCALED_COMBINE_SOURCE,
    )


def fused_router(logits: mx.array, bias: mx.array, top_k: int):
    """Top-k sigmoid routing of one row: ``(indices uint32 [K], weights f32 [K])``."""
    num_experts = logits.shape[-1]
    indices, weights = _router_kernel()(
        inputs=[logits.reshape(num_experts), bias],
        template=[("E", num_experts), ("K", top_k), ("TG", _ROUTER_THREADS)],
        grid=(_ROUTER_THREADS, 1, 1),
        threadgroup=(_ROUTER_THREADS, 1, 1),
        output_shapes=[(top_k,), (top_k,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, weights


def fused_combine(
    routed, weights, shared, residual, routed_scale=None, shared_scale=None
):
    """``residual + ((routed * weights).sum(-2) + shared)`` with stock roundings.

    ``routed_scale`` / ``shared_scale`` (float16 range-safe SwiGLU) multiply
    the routed and shared outputs first; both or neither.
    """
    dim = residual.shape[-1]
    top_k = weights.shape[-1]
    kernel, inputs = _combine_kernel(), [routed, weights, shared, residual]
    if routed_scale is not None:
        kernel = _scaled_combine_kernel()
        inputs += [routed_scale, shared_scale]
    return kernel(
        inputs=inputs,
        template=[("T", residual.dtype), ("D", dim), ("K", top_k)],
        grid=(dim, 1, 1),
        threadgroup=(min(dim, 256), 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[residual.dtype],
    )[0]


def _one_token(x) -> bool:
    return x.ndim == 3 and x.shape[0] * x.shape[1] == 1


def _moe_eligible(mlp, x) -> bool:
    return (
        x.dtype in (mx.bfloat16, mx.float16)
        and getattr(mlp, "shared_expert", None) is not None
        and mlp.gate.weight.shape[0] <= _MAX_EXPERTS
        and mlp.top_k <= _MAX_TOP_K
    )


def apply_flint_decode_patch() -> bool:
    """Route one-token Flint MoE layers through the fused router and combine.

    Wraps ``DecoderLayer.__call__`` so the residual add folds into the
    combine launch; dense-MLP layers and multi-row calls keep the original
    body.
    """
    if not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.flint import language as flint
    except ImportError:
        return False
    cls = flint.DecoderLayer
    if getattr(cls, "_omlx_flint_decode", False):
        return True
    moe_cls = flint.SparseMoeBlock
    orig_call = cls.__call__

    def patched_call(self, x, mask=None, cache=None):
        moe = self.mlp
        if not (_one_token(x) and isinstance(moe, moe_cls) and _moe_eligible(moe, x)):
            return orig_call(self, x, mask, cache)
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        y = self.post_attention_layernorm(h)
        indices, weights = fused_router(
            moe.gate(y), moe.expert_bias.astype(mx.float32), moe.top_k
        )
        routed = moe.switch_mlp(y, indices.reshape(1, 1, moe.top_k))
        shared, shared_scale = mlp_parts(moe.shared_expert, y)
        scale = routed_scale(moe)
        if (scale is None) != (shared_scale is None):
            routed = routed if scale is None else routed * scale
            shared = shared if shared_scale is None else shared * shared_scale
            scale = shared_scale = None
        return fused_combine(routed, weights, shared, h, scale, shared_scale)

    cls.__call__ = patched_call
    cls._omlx_flint_decode = True
    cls._omlx_flint_original_call = orig_call
    logger.info("Flint fused MoE decode patch applied")
    return True
