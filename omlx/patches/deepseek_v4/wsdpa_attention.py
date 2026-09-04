# SPDX-License-Identifier: Apache-2.0
"""Fused windowed + pooled-sparse prefill attention for DeepSeek-V4.

During chunked prefill the stock path runs full N^2 scaled_dot_product_attention
with a materialized boolean mask, even though each query can only see
  local rows j: (p - window, p]          (causal + sliding window)
  pool rows  i: i < (p + 1) // ratio     (compressed KV, ratio in {4, 128})
plus a per-head attention sink. At ratio-4 chunk boundaries that is ~4x the
necessary FLOPs (measured 39-48 ms per layer-call at L=2048, head_dim=512).

This module computes the identical math with a single fused Metal kernel that
visits only visible rows (fp32 online softmax in fixed row order, sink in the
denominator). Any setup/runtime failure permanently falls back to the stock
path; OMLX_DSV4_WSDPA=0 disables the kernel outright.
"""

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("OMLX_DSV4_WSDPA", "1") == "1"
_kernel = None
_broken = False
_ready = False
_topk_ready = False


def _wsdpa_route_enabled(*, topk: bool = False) -> bool:
    if _broken or not _ENABLED:
        return False
    return not topk or _TOPK_ENABLED


def wsdpa_prefill_route_active(*, topk: bool = False) -> bool:
    """Whether the matching WSDPA route is proven active in this process.

    Stay conservative until the matching kernel output has evaluated once.
    Setup/dispatch failures set ``_broken`` before returning to stock attention,
    and this live predicate then makes the memory profile price that fallback.
    """
    if not _wsdpa_route_enabled(topk=topk):
        return False
    return _topk_ready if topk else _ready


_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_SOURCE = """
    // q:      [H, L, D]  bf16 (contiguous)
    // kv:     [S, D]     bf16 (local rows, linear layout)
    // pooled: [P, D]     bf16 (compressed rows; dummy row when P == 0)
    // sinks:  [H]        bf16
    // params: int32 [6] = {offset, window, ratio, P, S, L}
    // scalep: fp32  [1] = scale
    // out:    [H, L, D]  bf16
    //
    // grid (16, L): gx = head-group (4 heads), gy = query row.
    // threadgroup 128 threads = 4 simdgroups, one head per simdgroup.
    // No threadgroup staging: rows are read directly (L2-resident at these
    // sizes); zero barriers measurably beats staged variants (13.7 vs 17.3ms).
    const uint t    = threadgroup_position_in_grid.y;
    const uint tid  = thread_index_in_threadgroup;
    const uint head = threadgroup_position_in_grid.x * 4 + (tid / 32);
    const uint lane = tid % 32;

    constant int *prm = (constant int *)&params[0];
    const int offset = prm[0];
    const int window = prm[1];
    const int ratio  = prm[2];
    const int P      = prm[3];
    const int S      = prm[4];
    const int L      = prm[5];
    const float scale = ((constant float *)&scalep[0])[0];

    const uint D = 512;
    const uint p = (uint)offset + t;

    device const bfloat4 *qh = (device const bfloat4 *)(q + ((uint64_t)head * L + t) * D);
    float4 qa0 = float4(qh[lane +  0]);
    float4 qa1 = float4(qh[lane + 32]);
    float4 qa2 = float4(qh[lane + 64]);
    float4 qa3 = float4(qh[lane + 96]);

    float m = -INFINITY, s = 0.0f;
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    device const bfloat4 *kvr = (device const bfloat4 *)kv;
    device const bfloat4 *pol = (device const bfloat4 *)pooled;

    // ---- local rows (p-window, p], then pooled prefix i < (p+1)/ratio
    // The local cache is a RotatingKVCache that trims to the last S rows, so
    // buffer row 0 sits at absolute position base = offset + L - S. Translate
    // absolute window bounds into buffer indices.
    const int base = offset + L - S;
    const int j0 = max(base, (int)p - window + 1) - base;
    const int j1 = (int)p - base;
    const int pvis = min((int)((p + 1) / (uint)ratio), P);
    const int n_local = j1 - j0 + 1;
    const int total = n_local + pvis;

    for (int r = 0; r < total; r++) {
        const bool is_pool = r >= n_local;
        const uint idx = is_pool ? (uint)(r - n_local) : (uint)(j0 + r);
        device const bfloat4 *row = (is_pool ? pol : kvr) + (uint64_t)idx * 128;
        float4 k0 = float4(row[lane + 0]), k1 = float4(row[lane + 32]);
        float4 k2 = float4(row[lane + 64]), k3 = float4(row[lane + 96]);
        float d = dot(qa0, k0) + dot(qa1, k1) + dot(qa2, k2) + dot(qa3, k3);
        d = simd_sum(d) * scale;
        float mn = max(m, d);
        float c = (m == -INFINITY) ? 0.0f : exp(m - mn);
        float w = exp(d - mn);
        m = mn; s = s * c + w;
        a0 = a0 * c + w * k0; a1 = a1 * c + w * k1;
        a2 = a2 * c + w * k2; a3 = a3 * c + w * k3;
    }

    // ---- attention sink (denominator only)
    s += exp(float(((device const bfloat *)sinks)[head]) - m);

    const float inv = (s == 0.0f) ? 0.0f : 1.0f / s;
    device bfloat4 *oh = (device bfloat4 *)(out + ((uint64_t)head * L + t) * D);
    oh[lane +  0] = bfloat4(a0 * inv);
    oh[lane + 32] = bfloat4(a1 * inv);
    oh[lane + 64] = bfloat4(a2 * inv);
    oh[lane + 96] = bfloat4(a3 * inv);
"""


def _get_kernel():
    global _kernel, _broken
    if not _wsdpa_route_enabled():
        return None
    if _kernel is None:
        try:
            _kernel = mx.fast.metal_kernel(
                name="dsv4_wsdpa",
                input_names=["q", "kv", "pooled", "sinks", "params", "scalep"],
                output_names=["out"],
                source=_SOURCE,
                header=_HEADER,
            )
        except Exception:
            _broken = True
            logger.warning(
                "DSV4 wsdpa kernel setup failed; using stock SDPA", exc_info=True
            )
            return None
    return _kernel


_KERNEL_TOPK = None
_TOPK_ENABLED = os.environ.get("OMLX_DSV4_WSDPA_TOPK", "1") == "1"

_SOURCE_TOPK = """
    // q:      [H, L, D]  bf16 (contiguous)
    // kv:     [S, D]     bf16 (local rows, linear absolute layout)
    // pooled: [P, D]     bf16 (compressed rows)
    // topk:   [L, K]     uint32 (temporally sorted pooled indices)
    // sinks:  [H]        bf16
    // params: int32 [7] = {offset, window, ratio, P, S, L, K}
    // scalep: fp32  [1] = scale
    // out:    [H, L, D]  bf16
    const uint t    = threadgroup_position_in_grid.y;
    const uint tid  = thread_index_in_threadgroup;
    const uint head = threadgroup_position_in_grid.x * 4 + (tid / 32);
    const uint lane = tid % 32;

    constant int *prm = (constant int *)&params[0];
    const int offset = prm[0];
    const int window = prm[1];
    const int ratio  = prm[2];
    const int P      = prm[3];
    const int S      = prm[4];
    const int L      = prm[5];
    const int K      = prm[6];
    const float scale = ((constant float *)&scalep[0])[0];

    const uint D = 512;
    const uint p = (uint)offset + t;

    device const bfloat4 *qh = (device const bfloat4 *)(q + ((uint64_t)head * L + t) * D);
    float4 qa0 = float4(qh[lane +  0]);
    float4 qa1 = float4(qh[lane + 32]);
    float4 qa2 = float4(qh[lane + 64]);
    float4 qa3 = float4(qh[lane + 96]);

    float m = -INFINITY, s = 0.0f;
    float4 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    device const bfloat4 *kvr = (device const bfloat4 *)kv;
    device const bfloat4 *pol = (device const bfloat4 *)pooled;

    const int base = offset + L - S;
    const int j0 = max(base, (int)p - window + 1) - base;
    const int j1 = (int)p - base;
    const int pvis = min((int)((p + 1) / (uint)ratio), P);
    const int n_local = j1 - j0 + 1;

    for (int r = 0; r < n_local; r++) {
        const uint idx = (uint)(j0 + r);
        device const bfloat4 *row = kvr + (uint64_t)idx * 128;
        float4 k0 = float4(row[lane + 0]), k1 = float4(row[lane + 32]);
        float4 k2 = float4(row[lane + 64]), k3 = float4(row[lane + 96]);
        float d = dot(qa0, k0) + dot(qa1, k1) + dot(qa2, k2) + dot(qa3, k3);
        d = simd_sum(d) * scale;
        float mn = max(m, d);
        float c = (m == -INFINITY) ? 0.0f : exp(m - mn);
        float w = exp(d - mn);
        m = mn; s = s * c + w;
        a0 = a0 * c + w * k0; a1 = a1 * c + w * k1;
        a2 = a2 * c + w * k2; a3 = a3 * c + w * k3;
    }

    device const uint *tk = (device const uint *)topk;
    const uint64_t tk_base = (uint64_t)t * (uint)K;
    for (int r = 0; r < K; r++) {
        const uint idx = tk[tk_base + (uint)r];
        if ((int)idx >= pvis) break;  // indexer topk rows are temporally sorted
        device const bfloat4 *row = pol + (uint64_t)idx * 128;
        float4 k0 = float4(row[lane + 0]), k1 = float4(row[lane + 32]);
        float4 k2 = float4(row[lane + 64]), k3 = float4(row[lane + 96]);
        float d = dot(qa0, k0) + dot(qa1, k1) + dot(qa2, k2) + dot(qa3, k3);
        d = simd_sum(d) * scale;
        float mn = max(m, d);
        float c = (m == -INFINITY) ? 0.0f : exp(m - mn);
        float w = exp(d - mn);
        m = mn; s = s * c + w;
        a0 = a0 * c + w * k0; a1 = a1 * c + w * k1;
        a2 = a2 * c + w * k2; a3 = a3 * c + w * k3;
    }

    s += exp(float(((device const bfloat *)sinks)[head]) - m);

    const float inv = (s == 0.0f) ? 0.0f : 1.0f / s;
    device bfloat4 *oh = (device bfloat4 *)(out + ((uint64_t)head * L + t) * D);
    oh[lane +  0] = bfloat4(a0 * inv);
    oh[lane + 32] = bfloat4(a1 * inv);
    oh[lane + 64] = bfloat4(a2 * inv);
    oh[lane + 96] = bfloat4(a3 * inv);
"""


def _get_topk_kernel():
    global _KERNEL_TOPK, _broken
    if not _wsdpa_route_enabled(topk=True):
        return None
    if _KERNEL_TOPK is None:
        try:
            _KERNEL_TOPK = mx.fast.metal_kernel(
                name="dsv4_wsdpa_topk",
                input_names=["q", "kv", "pooled", "topk", "sinks", "params", "scalep"],
                output_names=["out"],
                source=_SOURCE_TOPK,
                header=_HEADER,
            )
        except Exception:
            _broken = True
            logger.warning(
                "DSV4 wsdpa topk kernel setup failed; using stock sparse attention",
                exc_info=True,
            )
            return None
    return _KERNEL_TOPK


def wsdpa_prefill(
    q: mx.array,
    kv: mx.array,
    pooled: mx.array | None,
    sinks: mx.array,
    scale: float,
    offset: int,
    window: int,
    ratio: int,
) -> mx.array | None:
    """Fused window+pool attention; returns [B, H, L, D] or None (fallback).

    q: [B, H, L, D] (B == 1), kv: [B, 1, S, D], pooled: [B, P, D] or None.
    Only valid for prefill shapes (L > 1); callers keep decode paths.
    """
    global _broken, _ready
    if isinstance(offset, mx.array):
        return None  # should not happen in prefill; stay safe
    if (
        q.dtype != mx.bfloat16
        or q.shape[0] != 1
        or q.shape[1] != 64
        or q.shape[3] != 512
    ):
        return None
    kfn = _get_kernel()
    if kfn is None:
        return None
    try:
        heads, q_len, head_dim = q.shape[1], q.shape[2], q.shape[3]
        kv_len = kv.shape[2]
        pooled_len = 0 if pooled is None else pooled.shape[1]
        qc = mx.contiguous(q[0])  # [H, L, D]
        kvc = mx.contiguous(kv[0, 0])  # [S, D]
        pol = (
            mx.contiguous(pooled[0])
            if pooled_len
            else mx.zeros((1, head_dim), dtype=q.dtype)  # constant-space guard
        )
        params = mx.array(
            [offset, window, ratio, pooled_len, kv_len, q_len], dtype=mx.int32
        )
        scalep = mx.array([scale], dtype=mx.float32)
        out = kfn(
            inputs=[qc, kvc, pol, sinks, params, scalep],
            grid=(16 * 128, q_len, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(heads, q_len, head_dim)],
            output_dtypes=[mx.bfloat16],
        )[0]
        if not _ready:
            # MLX is lazy: construction alone cannot prove the route works.
            # Evaluate the first dispatch inside this fallback boundary.
            mx.eval(out)
            _ready = True
        return out[None]
    except Exception:
        _broken = True
        logger.warning("DSV4 wsdpa kernel disabled; using stock SDPA", exc_info=True)
        return None


def wsdpa_topk_prefill(
    q: mx.array,
    kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    sinks: mx.array,
    scale: float,
    offset: int,
    window: int,
    ratio: int,
) -> mx.array | None:
    """Fused window + indexer-topk pooled attention for ratio-4 prefill.

    Requires the Indexer's temporally sorted topk rows. Returns None to keep the
    native/stock path whenever shapes or dtypes are not the exact prefill case.
    """
    global _broken, _topk_ready
    if isinstance(offset, mx.array) or not _wsdpa_route_enabled(topk=True):
        return None
    if (
        q.dtype != mx.bfloat16
        or q.shape[0] != 1
        or q.shape[1] != 64
        or q.shape[3] != 512
        or topk.dtype != mx.uint32
        or topk.ndim != 3
        or topk.shape[0] != 1
        or topk.shape[1] != q.shape[2]
        or pooled is None
        or pooled.shape[1] == 0
    ):
        return None
    kfn = _get_topk_kernel()
    if kfn is None:
        return None
    try:
        heads, q_len, head_dim = q.shape[1], q.shape[2], q.shape[3]
        params = mx.array(
            [
                offset,
                window,
                ratio,
                pooled.shape[1],
                kv.shape[2],
                q_len,
                topk.shape[2],
            ],
            dtype=mx.int32,
        )
        scalep = mx.array([scale], dtype=mx.float32)
        out = kfn(
            inputs=[
                mx.contiguous(q[0]),
                mx.contiguous(kv[0, 0]),
                mx.contiguous(pooled[0]),
                mx.contiguous(topk[0]),
                sinks,
                params,
                scalep,
            ],
            grid=(16 * 128, q_len, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(heads, q_len, head_dim)],
            output_dtypes=[mx.bfloat16],
        )[0]
        if not _topk_ready:
            # Keep the low-memory profile disabled until lazy execution proves
            # that this independent top-k route is usable.
            mx.eval(out)
            _topk_ready = True
        return out[None]
    except Exception:
        _broken = True
        logger.warning(
            "DSV4 wsdpa topk kernel disabled; using stock sparse attention",
            exc_info=True,
        )
        return None
