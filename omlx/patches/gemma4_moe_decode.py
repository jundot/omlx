# SPDX-License-Identifier: Apache-2.0
"""Fused MoE routing and combine for Gemma 4 decode and speculative verify rows.

Gemma 4's MoE router (26B-A4B: 128 experts, top 8) and its combine take about
eight small launches per layer per step. For up to ``_MAX_ROWS`` token rows this
patch runs each as one launch; the logits projection and the experts are
unchanged.

1. Router: top-k by repeated threadgroup argmax, then the softmax over the kept
   logits and the ``per_expert_scale`` product with every rounding the stock
   ops make (exp, the two four-wide partial sums, the reciprocal and the
   products, all in the activation dtype).
2. Combine: the k products summed in order, rounding to the activation dtype
   after each add, as ``(w * y).sum(-2)`` does.

At these widths ``argpartition`` is a stable ascending sort, so among logits
tied at the k-th place it keeps the highest index. The kernel selects and
orders the same way, so indices, weights and output are bit-identical to the
composed path. Prefill chunks keep the stock ops.
"""

from __future__ import annotations

import logging
from functools import cache

import mlx.core as mx

logger = logging.getLogger(__name__)

_ROUTER_THREADS = 128
_MAX_EXPERTS = 1024
# The stock softmax sums four logits per lane; two lanes add in either order,
# so up to eight kept experts reproduce it exactly.
_MAX_TOP_K = 8
_MAX_ROWS = 16

_ROUTER_SOURCE = """
    const uint t = thread_position_in_threadgroup.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int NSG = TG / 32;

    threadgroup float logit[E];
    threadgroup float red_v[NSG];
    threadgroup int red_i[NSG];
    threadgroup int sel[K];
    threadgroup float sel_v[K];

    for (int e = int(t); e < E; e += TG) logit[e] = float(scores[r * E + e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < K; ++k) {
        float bv = -INFINITY;
        int bi = -1;
        for (int e = int(t); e < E; e += TG) {
            float v = logit[e];
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
                // NaN logits never win a comparison: stay in bounds.
                bi = bi < 0 ? 0 : bi;
                sel[k] = bi;
                sel_v[k] = bv;
                logit[bi] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t == 0) {
        // Stock order: ascending in (logit, index), i.e. selection reversed.
        // Stock softmax: x - max and exp rounded to T, four-wide partial sums
        // accumulated in T, lanes added, reciprocal rounded to T, then scaled
        // by per_expert_scale and rounded again.
        const float m = sel_v[0];
        T ex[K];
        for (int j = 0; j < K; ++j) {
            T diff = static_cast<T>(sel_v[K - 1 - j] - m);
            ex[j] = static_cast<T>(metal::fast::exp(float(diff)));
        }
        T lane_sum[2] = {static_cast<T>(0.0f), static_cast<T>(0.0f)};
        for (int j = 0; j < K; ++j)
            lane_sum[j / 4] = static_cast<T>(float(lane_sum[j / 4]) + float(ex[j]));
        T total = static_cast<T>(float(lane_sum[0]) + float(lane_sum[1]));
        T inv = static_cast<T>(1.0f / float(total));
        for (int j = 0; j < K; ++j) {
            const int e = sel[K - 1 - j];
            T p = static_cast<T>(float(ex[j]) * float(inv));
            indices[r * K + j] = uint(e);
            weights[r * K + j] = static_cast<T>(float(p) * float(expert_scale[e]));
        }
    }
"""

_COMBINE_SOURCE = """
    const uint d = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    if (d >= uint(D)) return;
    T acc = static_cast<T>(0.0f);
    for (int k = 0; k < K; ++k) {
        T product = static_cast<T>(float(weights[r * K + k]) * float(routed[(r * K + k) * D + d]));
        acc = static_cast<T>(float(acc) + float(product));
    }
    out[r * D + d] = acc;
"""


@cache
def _router_kernel():
    return mx.fast.metal_kernel(
        name="omlx_gemma4_router_topk",
        input_names=["scores", "expert_scale"],
        output_names=["indices", "weights"],
        source=_ROUTER_SOURCE,
    )


@cache
def _combine_kernel():
    return mx.fast.metal_kernel(
        name="omlx_gemma4_moe_combine",
        input_names=["routed", "weights"],
        output_names=["out"],
        source=_COMBINE_SOURCE,
    )


def fused_router(scores: mx.array, expert_scale: mx.array, top_k: int):
    """Top-k routing of every row of ``scores (..., E)``.

    Returns ``(indices uint32, weights)`` shaped ``scores.shape[:-1] + (K,)``,
    ``weights`` in the dtype of ``scores``: the softmax over the kept logits
    times ``expert_scale`` of the kept experts.
    """
    num_experts = scores.shape[-1]
    rows = scores.size // num_experts
    lead = tuple(scores.shape[:-1])
    indices, weights = _router_kernel()(
        inputs=[scores.reshape(rows, num_experts), expert_scale],
        template=[
            ("E", num_experts),
            ("K", top_k),
            ("TG", _ROUTER_THREADS),
            ("T", scores.dtype),
        ],
        grid=(_ROUTER_THREADS, rows, 1),
        threadgroup=(_ROUTER_THREADS, 1, 1),
        output_shapes=[lead + (top_k,), lead + (top_k,)],
        output_dtypes=[mx.uint32, scores.dtype],
    )
    return indices, weights


def fused_combine(routed: mx.array, weights: mx.array) -> mx.array:
    """``(weights[..., None] * routed).sum(-2)``, bit-identical.

    ``routed`` is ``(..., K, D)``, ``weights`` ``(..., K)``.
    """
    dim = routed.shape[-1]
    top_k = weights.shape[-1]
    rows = weights.size // top_k
    return _combine_kernel()(
        inputs=[routed, weights],
        template=[("T", routed.dtype), ("D", dim), ("K", top_k)],
        grid=(dim, rows, 1),
        threadgroup=(min(dim, 256), 1, 1),
        output_shapes=[tuple(weights.shape[:-1]) + (dim,)],
        output_dtypes=[routed.dtype],
    )[0]


def _fused_rows(x) -> bool:
    return x.ndim == 3 and 1 <= x.shape[0] * x.shape[1] <= _MAX_ROWS


def _router_eligible(router, x) -> bool:
    return (
        _fused_rows(x)
        and x.dtype in (mx.bfloat16, mx.float16)
        and router.per_expert_scale.dtype == x.dtype
        and router.per_expert_scale.shape[0] <= _MAX_EXPERTS
        and router.config.top_k_experts <= _MAX_TOP_K
    )


def _patch_classes(router_cls, experts_cls) -> None:
    orig_router = router_cls.__call__
    orig_experts = experts_cls.__call__

    def router_call(self, x):
        if not _router_eligible(self, x):
            return orig_router(self, x)
        # ``scale * root_size`` is constant per layer; compute it once.
        if getattr(self, "_omlx_norm_src", None) is not self.scale:
            self._omlx_norm_weight = self.scale * self._root_size
            self._omlx_norm_src = self.scale
        x = mx.fast.rms_norm(x, self._omlx_norm_weight, self.eps)
        return fused_router(
            self.proj(x), self.per_expert_scale, self.config.top_k_experts
        )

    def experts_call(self, x, top_k_indices, top_k_weights):
        if not (_fused_rows(top_k_weights) and top_k_weights.dtype == x.dtype):
            return orig_experts(self, x, top_k_indices, top_k_weights)
        return fused_combine(self.switch_glu(x, top_k_indices), top_k_weights)

    router_cls.__call__ = router_call
    experts_cls.__call__ = experts_call
    router_cls._omlx_gemma4_moe_decode = True
    router_cls._omlx_gemma4_original_call = orig_router
    experts_cls._omlx_gemma4_original_call = orig_experts


def apply_gemma4_moe_decode_patch() -> bool:
    """Route one-token and verify-block Gemma 4 MoE layers through the fused
    router and combine. Returns whether a Gemma 4 implementation was patched."""
    if not mx.metal.is_available():
        return False
    patched = False
    for module_path in (
        "mlx_vlm.models.gemma4.language",
        "mlx_lm.models.gemma4_text",
    ):
        try:
            module = __import__(module_path, fromlist=["Router", "Experts"])
            router_cls, experts_cls = module.Router, module.Experts
        except (ImportError, AttributeError):
            continue
        if not getattr(router_cls, "_omlx_gemma4_moe_decode", False):
            _patch_classes(router_cls, experts_cls)
        patched = True
    if patched:
        logger.info("Gemma 4 fused MoE decode patch applied")
    return patched
