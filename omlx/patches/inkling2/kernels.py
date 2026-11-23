# SPDX-License-Identifier: Apache-2.0
"""Attention and attention-residual primitives for Inkling 2."""

from __future__ import annotations

import logging

import mlx.core as mx

from ...custom_kernels.nax import is_nax_available
from ..qwen35_verify_qmm import is_row_exact_armed
from . import verify_attention

logger = logging.getLogger(__name__)

# Upper bound on one additive mask block (elements) in the general attention
# path; queries are split into sub-blocks so the mask stays below it.
_MASK_BUDGET = 1 << 29


_ATTN_RES_TG = 256
_attn_res_kernels: dict = {}
# Fused kernels whose threadgroup this GPU accepts, by variant. A pipeline's
# thread limit depends on its register use, so it is found by launching each
# variant once (M1 Pro accepts 576 threads for some attn_res variants and only
# 384-512 for others).
_launch_ok: dict = {}


def _fits_threadgroup(key, launch) -> bool:
    """Run ``launch`` (a tiny launch of the variant) once per ``key``; False
    when Metal rejects its threadgroup size, so the caller falls back."""
    ok = _launch_ok.get(key)
    if ok is None:
        try:
            mx.eval(launch())
            ok = True
        except ValueError as exc:
            if "threads per threadgroup" not in str(exc):
                raise
            logger.info(
                "Inkling 2 fused kernel %s exceeds this GPU's threadgroup "
                "limit; using the MLX ops",
                key,
            )
            ok = False
        _launch_ok[key] = ok
    return ok


def _attn_res_kernel(blocks: int, queries: int):
    """Fused depth mixing for ``blocks`` completed blocks plus the partial sum.

    One threadgroup per token row: dot products of every source with every
    query, softmax over sources, then the weighted sum, without stacking the
    sources.
    """
    key = (blocks, queries)
    kernel = _attn_res_kernels.get(key)
    if kernel is not None:
        return kernel
    srcs = [f"b{i}" for i in range(blocks)] + ["p"]
    loads = "\n".join(
        f"        float s{i} = {name}[base + h];" for i, name in enumerate(srcs)
    )
    dots = "\n".join(
        f"        acc[{i} * Q + qi] += s{i} * qv;" for i in range(len(srcs))
    )
    mix = " + ".join(f"w[{i} * Q + qi] * s{i}" for i in range(len(srcs)))
    source = f"""
    constexpr uint NS = {len(srcs)};
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = (size_t)row * H;
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            float qv = query[qi * H + h];
{dots}
        }}
    }}
    threadgroup float red[NS * Q][TG / 32];
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0) red[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float t = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) t += red[i][j];
        w[i] = t * scale[0];
    }}
    for (uint qi = 0; qi < Q; ++qi) {{
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) m = max(m, w[n * Q + qi]);
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            w[n * Q + qi] = metal::exp(w[n * Q + qi] - m);
            z += w[n * Q + qi];
        }}
        for (uint n = 0; n < NS; ++n) w[n * Q + qi] /= z;
    }}
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            out[((size_t)row * Q + qi) * H + h] = {mix};
        }}
    }}
"""
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_attn_res_{blocks}_{queries}",
        input_names=[*srcs, "query", "scale"],
        output_names=["out"],
        source=source,
    )
    _attn_res_kernels[key] = kernel
    return kernel


def attn_res_mix(blocks: list, partial_n: mx.array, query: mx.array, scale: float):
    """Mix unit-RMS depth sources with learned queries.

    ``blocks`` holds [B, L, H] FP32 sources, ``partial_n`` is the current
    partial sum (also unit-RMS) and ``query`` is [Q, H]. Returns [B, L, Q, H].
    """
    if mx.default_device() != mx.gpu:
        src = mx.stack([*blocks, partial_n], axis=-2)
        logits = (src @ query.T) * scale
        w = mx.softmax(logits, axis=-2, precise=True)
        return w.swapaxes(-1, -2) @ src
    b, length, hidden = partial_n.shape
    nq = query.shape[0]
    rows = b * length
    kernel = _attn_res_kernel(len(blocks), nq)
    out = kernel(
        inputs=[
            *[x.astype(mx.float32) for x in blocks],
            partial_n.astype(mx.float32),
            query.astype(mx.float32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("H", hidden), ("Q", nq), ("TG", _ATTN_RES_TG)],
        grid=(rows * _ATTN_RES_TG, 1, 1),
        threadgroup=(_ATTN_RES_TG, 1, 1),
        output_shapes=[(b, length, nq, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    return out


# The fused kernels below repeat the arithmetic of the pinned MLX's rms_norm
# and reduce kernels, so outputs keep their bits. Rerun
# test_inkling2_fused_layer_kernels_keep_the_mlx_bits after an MLX pin change.
_RMS_INV_HEADER = """
inline float omlx_inkling2_rms_inv(
    float acc,
    threadgroup float* local_sums,
    threadgroup float* inv_mean,
    uint lane,
    uint sg,
    float width,
    float eps) {
    acc = simd_sum(acc);
    if (sg == 0) {
        local_sums[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        local_sums[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) {
            inv_mean[0] = metal::precise::rsqrt(acc / width + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inv_mean[0];
}
"""
_attn_res_fused_kernels: dict = {}


def _rms_threads(hidden: int) -> int:
    """Threadgroup size of MLX's single-row rms_norm: 4 values per thread."""
    return (((hidden + 3) // 4 + 31) // 32) * 32


def _attn_res_fused_kernel(
    blocks: int, queries: int, norm: bool, f32: bool, pout: bool
):
    """``attn_res_mix`` with its input unit-RMS and, given ``norm``, its output
    RMSNorm and cast in one kernel."""
    key = (blocks, queries, norm, f32, pout)
    kernel = _attn_res_fused_kernels.get(key)
    if kernel is not None:
        return kernel
    nsrc = blocks + 1
    srcs = [f"b{i}" for i in range(blocks)]

    def loads(index: str) -> str:
        lines = [
            f"        float s{i} = {name}[base + {index}];"
            for i, name in enumerate(srcs)
        ]
        lines.append(f"        float s{blocks} = ps[{index}];")
        return "\n".join(lines)

    f32_store = (
        "                    out32[((size_t)qi * gsize + row) * H + h] = float(o);"
        if f32
        else ""
    )
    p_store = "                    pn[base + h] = pv;" if pout else ""
    dots = "\n".join(
        f"            acc[{i} * Q + qi] += s{i} * qv;" for i in range(nsrc)
    )
    mix = " + ".join(f"w[{i} * Q + qi] * s{i}" for i in range(nsrc))
    if norm:
        output = f"""
    for (uint qi = 0; qi < Q; ++qi) {{
        float mv[4];
        float macc = 0.0f;
        if (tid < TGR) {{
            for (uint i = 0; i < 4; ++i) {{
                const uint h = tid * 4 + i;
                if (h < H) {{
{loads("h")}
                    mv[i] = {mix};
                }} else {{
                    mv[i] = 0.0f;
                }}
                macc = metal::fma(mv[i], mv[i], macc);
            }}
        }}
        const float inv = omlx_inkling2_rms_inv(macc, local_sums, inv_mean, lane, sg, width, eps[1]);
        if (tid < TGR) {{
            for (uint i = 0; i < 4; ++i) {{
                const uint h = tid * 4 + i;
                if (h < H) {{
                    const T o = T(norm_w[qi * H + h] * (mv[i] * inv));
                    out[((size_t)qi * gsize + row) * H + h] = o;
{f32_store}
                }}
            }}
        }}
    }}
"""
    else:
        output = f"""
    for (uint h = tid; h < H; h += threads_per_threadgroup.x) {{
{loads("h")}
        for (uint qi = 0; qi < Q; ++qi) {{
            out[((size_t)row * Q + qi) * H + h] = {mix};
        }}
    }}
"""
    source = f"""
    constexpr uint NS = {nsrc};
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const size_t base = (size_t)row * H;
    const size_t gsize = (size_t)partial_shape[0] * partial_shape[1];
    // Runtime width: rms_norm divides by its axis size argument.
    const float width = float(partial_shape[partial_ndim - 1]);
    threadgroup float ps[H];
    threadgroup float local_sums[32];
    threadgroup float inv_mean[1];
    {{
        float xv[4];
        float acc = 0.0f;
        if (tid < TGR) {{
            for (uint i = 0; i < 4; ++i) {{
                const uint h = tid * 4 + i;
                xv[i] = h < H ? float(partial[base + h]) : 0.0f;
                acc = metal::fma(xv[i], xv[i], acc);
            }}
        }}
        const float inv = omlx_inkling2_rms_inv(acc, local_sums, inv_mean, lane, sg, width, eps[0]);
        if (tid < TGR) {{
            for (uint i = 0; i < 4; ++i) {{
                const uint h = tid * 4 + i;
                if (h < H) {{
                    const float pv = 1.0f * (xv[i] * inv);
                    ps[h] = pv;
{p_store}
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    if (tid < TGM) {{
        for (uint h = tid; h < H; h += TGM) {{
{loads("h")}
            for (uint qi = 0; qi < Q; ++qi) {{
                float qv = query[qi * H + h];
{dots}
            }}
        }}
    }}
    threadgroup float red[NS * Q][TGM / 32];
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0 && sg < TGM / 32) red[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float t = 0.0f;
        for (uint j = 0; j < TGM / 32; ++j) t += red[i][j];
        w[i] = t * scale[0];
    }}
    for (uint qi = 0; qi < Q; ++qi) {{
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) m = max(m, w[n * Q + qi]);
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            w[n * Q + qi] = metal::exp(w[n * Q + qi] - m);
            z += w[n * Q + qi];
        }}
        for (uint n = 0; n < NS; ++n) w[n * Q + qi] /= z;
    }}
{output}
"""
    inputs = [*srcs, "partial", "query", "scale", "eps"]
    if norm:
        inputs.append("norm_w")
    outputs = ["out"]
    if norm and f32:
        outputs.append("out32")
    if pout:
        outputs.append("pn")
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_attn_res_fused_{blocks}_{queries}_{int(norm)}{int(f32)}{int(pout)}",
        input_names=inputs,
        output_names=outputs,
        source=source,
        header=_RMS_INV_HEADER,
    )
    _attn_res_fused_kernels[key] = kernel
    return kernel


def attn_res_fused(
    blocks: list,
    partial: mx.array,
    query: mx.array,
    scale: float,
    *,
    unit_eps: float,
    norm_weight: mx.array | None = None,
    norm_eps: float = 0.0,
    out_dtype=None,
    want_f32: bool = False,
    want_partial: bool = False,
    _probe: bool = False,
):
    """``attn_res_mix(blocks, unit_rms(partial), ...)``, then per query
    ``rms_norm(mix, norm_weight).astype(out_dtype)`` when ``norm_weight`` is
    given, as one kernel; None when unsupported. Returns the mix ([B, L, Q, H]
    FP32, or [Q, B, L, H] when normalized), its FP32 copy if ``want_f32`` and
    the unit-RMS partial if ``want_partial``."""
    b, length, hidden = partial.shape
    nq = query.shape[0]
    norm = norm_weight is not None
    tgr = _rms_threads(hidden)
    threads = max(tgr, _ATTN_RES_TG)
    if (
        mx.default_device() != mx.gpu
        or threads > 1024
        or hidden * 4 > 16384
        or any(x.dtype != mx.float32 for x in blocks)
        or query.dtype != mx.float32
        or (norm and norm_weight.dtype != mx.float32)
    ):
        return None
    f32 = norm and want_f32
    variant = ("attn_res", len(blocks), nq, norm, f32, want_partial, hidden)
    variant += (partial.dtype, out_dtype)
    if not _probe and not _fits_threadgroup(
        variant,
        lambda: attn_res_fused(
            [mx.zeros((1, 1, hidden), dtype=mx.float32) for _ in blocks],
            mx.zeros((1, 1, hidden), dtype=partial.dtype),
            mx.zeros(query.shape, dtype=mx.float32),
            scale,
            unit_eps=unit_eps,
            norm_weight=mx.ones(norm_weight.shape, dtype=mx.float32) if norm else None,
            norm_eps=norm_eps,
            out_dtype=out_dtype,
            want_f32=want_f32,
            want_partial=want_partial,
            _probe=True,
        ),
    ):
        return None
    kernel = _attn_res_fused_kernel(len(blocks), nq, norm, f32, want_partial)
    rows = b * length
    inputs = [
        *blocks,
        partial,
        query,
        mx.array([scale], dtype=mx.float32),
        mx.array([unit_eps, norm_eps], dtype=mx.float32),
    ]
    out_type = (out_dtype or partial.dtype) if norm else mx.float32
    shape = (nq, b, length, hidden) if norm else (b, length, nq, hidden)
    shapes = [shape]
    dtypes = [out_type]
    if norm:
        inputs.append(norm_weight)
    if f32:
        shapes.append(shape)
        dtypes.append(mx.float32)
    if want_partial:
        shapes.append((b, length, hidden))
        dtypes.append(mx.float32)
    template = [
        ("H", hidden),
        ("Q", nq),
        ("TGM", _ATTN_RES_TG),
        ("TGR", tgr),
    ]
    if norm:
        template.append(("T", out_type))
    return tuple(
        kernel(
            inputs=inputs,
            template=template,
            grid=(rows * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=shapes,
            output_dtypes=dtypes,
        )
    )


# Elementwise steps keep one statement per op, as the compiled graph does, so
# nothing is contracted into a fused multiply-add.


_LOOP_IN_SOURCE = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    const uint j = threadgroup_position_in_grid.y;
    const uint i = j / 2;
    const int rows = x_shape[0] * x_shape[1];
    const float width = float(x_shape[3]);
    const device float* src = (j % 2 == 0)
        ? x + ((size_t)row * x_shape[2] + (XQ ? i : 0)) * H
        : m + ((size_t)row * 4 + M0 + i) * H;
    threadgroup float local_sums[32];
    threadgroup float inv_mean[1];
    float xv[4];
    float acc = 0.0f;
    if (tid < TGR) {
        for (uint e = 0; e < 4; ++e) {
            const uint h = tid * 4 + e;
            xv[e] = h < H ? src[h] : 0.0f;
            acc = metal::fma(xv[e], xv[e], acc);
        }
    }
    const float inv = omlx_inkling2_rms_inv(acc, local_sums, inv_mean, lane, sg, width, 1e-6f);
    if (tid < TGR) {
        device float* dst = z + ((size_t)j * rows + row) * H;
        for (uint e = 0; e < 4; ++e) {
            const uint h = tid * 4 + e;
            if (h < H) {
                const float u = 1.0f * (xv[e] * inv);
                dst[h] = float(bfloat16_t(u));
            }
        }
    }
"""

# MLX's small column reduce adds zero to each value, then the second to the first.
_LOOP_LOW_SOURCE = """
    const uint idx = thread_position_in_grid.x;
    const uint n = low_shape[0] / 2;
    const uint per = low_shape[1] * low_shape[2];
    if (idx >= n * per) {
        return;
    }
    const uint i = idx / per;
    const uint rest = idx % per;
    const float a0 = low[(size_t)(2 * i) * per + rest] + 0.0f;
    const float a1 = low[(size_t)(2 * i + 1) * per + rest] + 0.0f;
    const float s = a1 + a0;
    out[idx] = float(bfloat16_t(s));
"""

_LOOP_OUT_SOURCE = """
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    const uint i = threadgroup_position_in_grid.y;
    const int rows = x_shape[0] * x_shape[1];
    const float width = float(x_shape[3]);
    const device float* xs = x + ((size_t)row * x_shape[2] + (XQ ? i : 0)) * H;
    const device float* ms = m + ((size_t)row * 4 + M0 + i) * H;
    const device float* gs = gates + ((size_t)i * rows + row) * 2 * H;
    threadgroup float local_sums[32];
    threadgroup float inv_mean[1];
    float ov[4];
    float acc = 0.0f;
    if (tid < TGR) {
        for (uint e = 0; e < 4; ++e) {
            const uint h = tid * 4 + e;
            if (h < H) {
                const float gain = metal::precise::tanh(gs[h]);
                const float forget = metal::precise::tanh(gs[H + h]);
                const float keep = 1.0f - forget;
                const float kept = keep * xs[h];
                const float added = gain * ms[h];
                ov[e] = kept + added;
            } else {
                ov[e] = 0.0f;
            }
            acc = metal::fma(ov[e], ov[e], acc);
        }
    }
    const float inv = omlx_inkling2_rms_inv(acc, local_sums, inv_mean, lane, sg, width, eps[0]);
    if (tid < TGR) {
        const device float* wr = norm_w + (NW > 1 ? i : 0) * H;
        for (uint e = 0; e < 4; ++e) {
            const uint h = tid * 4 + e;
            if (h < H) {
                const T o = T(wr[h] * (ov[e] * inv));
                out[((size_t)i * rows + row) * H + h] = o;
{f32_store}
            }
        }
    }
"""
_loop_kernels: dict = {}


def _loop_kernel(kind: str, f32: bool = False):
    key = (kind, f32)
    kernel = _loop_kernels.get(key)
    if kernel is not None:
        return kernel
    if kind == "in":
        names, outs, source = ["x", "m"], ["z"], _LOOP_IN_SOURCE
    elif kind == "low":
        names, outs, source = ["low"], ["out"], _LOOP_LOW_SOURCE
    else:
        store = (
            "                out32[((size_t)i * rows + row) * H + h] = float(o);"
            if f32
            else ""
        )
        names = ["gates", "x", "m", "norm_w", "eps"]
        outs = ["out", "out32"] if f32 else ["out"]
        source = _LOOP_OUT_SOURCE.replace("{f32_store}", store)
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_loop_{kind}{int(f32)}",
        input_names=names,
        output_names=outs,
        source=source,
        header=_RMS_INV_HEADER if kind != "low" else "",
    )
    _loop_kernels[key] = kernel
    return kernel


def loop_messages_supported(mixed: mx.array, messages: mx.array) -> bool:
    hidden = mixed.shape[-1]
    return (
        mx.default_device() == mx.gpu
        and _rms_threads(hidden) <= 1024
        and mixed.dtype == mx.float32
        and messages.dtype == mx.float32
        and messages.shape[2] == 4
    )


def loop_message_inputs(mixed, messages, m0: int, n: int):
    """[2n, rows, H] FP32: for each controlled input i, the BF16-rounded unit
    RMS of ``mixed[:, :, i or 0]`` then of ``messages[:, :, m0 + i]``."""
    b, length, qx, hidden = mixed.shape
    rows = b * length
    tgr = _rms_threads(hidden)
    return _loop_kernel("in")(
        inputs=[mixed, messages],
        template=[("H", hidden), ("TGR", tgr), ("M0", m0), ("XQ", qx > 1)],
        grid=(rows * tgr, 2 * n, 1),
        threadgroup=(tgr, 1, 1),
        output_shapes=[(2 * n, rows, hidden)],
        output_dtypes=[mx.float32],
    )[0]


def loop_message_low(low: mx.array):
    """``low.reshape(n, 2, rows, r).sum(1)`` rounded to BF16, as FP32 [n, rows, r]."""
    two_n, rows, rank = low.shape
    total = two_n // 2 * rows * rank
    return _loop_kernel("low")(
        inputs=[low],
        grid=(total, 1, 1),
        threadgroup=(min(total, 256), 1, 1),
        output_shapes=[(two_n // 2, rows, rank)],
        output_dtypes=[mx.float32],
    )[0]


def loop_message_outputs(
    gates, mixed, messages, m0, n, norm_w, norm_eps, dtype, want_f32
):
    """The controlled inputs ``(1 - tanh(f)) * x + tanh(g) * message`` after
    their RMSNorm (FP32 weights [NW, H]) and cast to ``dtype``, [n, B, L, H];
    with ``want_f32`` also their FP32 copy."""
    b, length, qx, hidden = mixed.shape
    rows = b * length
    tgr = _rms_threads(hidden)
    shapes = [(n, b, length, hidden)]
    dtypes = [dtype]
    if want_f32:
        shapes.append((n, b, length, hidden))
        dtypes.append(mx.float32)
    return _loop_kernel("out", want_f32)(
        inputs=[gates, mixed, messages, norm_w, mx.array([norm_eps], dtype=mx.float32)],
        template=[
            ("T", dtype),
            ("H", hidden),
            ("TGR", tgr),
            ("M0", m0),
            ("XQ", qx > 1),
            ("NW", norm_w.shape[0]),
        ],
        grid=(rows * tgr, n, 1),
        threadgroup=(tgr, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )


def _decode_attention(q, k, v, rel, sink, scale, window):
    s = k.shape[2]
    lo = max(0, s - window) if window else 0
    if lo:
        k = k[:, :, lo:]
        v = v[:, :, lo:]
    sv = s - lo
    r = rel.shape[-1]
    n = min(r, sv)
    hq = q.shape[1]
    tail = rel[0, 0, :, :n][:, ::-1]
    if n < sv:
        tail = mx.concatenate([mx.zeros((hq, sv - n), dtype=tail.dtype), tail], -1)
    mask = tail.reshape(1, hq, 1, sv).astype(q.dtype)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sink[0, 0].astype(q.dtype)
    )


# Flash attention with the relative-bias band and per-query sinks, for
# prefill-sized BF16 or FP16 query blocks (FP16 after the M1/M2 activation
# recast). One threadgroup runs 32 queries of one head
# in four simdgroups (8 rows each) over 16-key tiles staged in half. Scores
# live in the log2 domain; each row's online softmax starts from its sink
# logit, so the zero-value sink is exact without an extra key.
_FLASH_SOURCE = """
    constexpr int BQ = 32;
    constexpr int BK = 16;
    constexpr int NT = 128;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int qt = threadgroup_position_in_grid.x;
    const int h = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int lp = dims[2 + b];
    const int hk = h / (HQ / HKV);
    const int q0 = qt * BQ;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int a_row = S - L + row;
    const float qscale = scale[0] * LOG2E;

    // Q, K and V are FP16, or BF16 values within half's range (L2-clipped
    // q/k, unit-RMS v), so staging them in half is exact; scores accumulate
    // in FP32 and the softmax scale is applied there. K is stored transposed
    // and both tiles are padded by 16 bytes per row against bank conflicts.
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + row * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = row_ok ? float(sink[(b * L + row) * HQ + h]) * LOG2E : 0.0f;
    float l = 1.0f;
    const device T* relp = rel + ((size_t)(b * L + (row_ok ? row : 0)) * HQ + h) * R;

    const int a_first = S - L + q0;
    const int a_last = S - L + min(q0 + BQ, L) - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int k_begin = (lo / BK) * BK;
    const device T* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1;

    for (int k0 = k_begin; k0 <= a_last; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            // 8 BF16 or FP16 values (16 bytes) per load.
            for (int c = tid; c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < S) {
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                if (IS_BF16) {
                    for (int j = 0; j < 4; ++j) {
                        kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                        kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                        vd[2 * j] = half(as_type<float>(vw[j] << 16));
                        vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                    }
                } else {
                    // FP16 activations (M1/M2 recast): the words already hold halves.
                    for (int j = 0; j < 4; ++j) {
                        const half2 k2 = as_type<half2>(kw[j]);
                        const half2 v2 = as_type<half2>(vw[j]);
                        kd[(2 * j) * LDK] = k2.x;
                        kd[(2 * j + 1) * LDK] = k2.y;
                        vd[2 * j] = v2.x;
                        vd[2 * j + 1] = v2.y;
                    }
                }
            }
        } else {
            for (int c = tid; c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < S) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = a_row - key;
                const bool vis = row_ok && d >= 0 && key >= lp
                    && (WINDOW == 0 || d < WINDOW);
                float s = Sf[kb].thread_elements()[e] * qscale;
                if (vis) {
                    if (d < R) {
                        s += float(relp[d]) * LOG2E;
                    }
                } else {
                    s = -INFINITY;
                }
                Sf[kb].thread_elements()[e] = s;
                rmax = max(rmax, s);
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(Sf[kb].thread_elements()[e] - m_new);
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        const float inv = 1.0f / l;
        device T* op = out + ((size_t)(b * L + row) * HQ + h) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(Of[db].thread_elements()[0] * inv);
            op[db * 8 + fn + 1] = T(Of[db].thread_elements()[1] * inv);
        }
    }
"""
_flash_kernel = None
# Shorter query blocks take the decode kernel when batched or left-padded,
# else SDPA.
_FLASH_MIN_QUERIES = 32


# FlashDecoding for batched or left-padded decode-sized query blocks (decode
# steps and verify): the bias band, the window and the sink are applied in the kernel,
# so no mask is built and K/V are read in place. One threadgroup reduces one
# chunk of CHUNK backward distances of one (row, position, KV head) to a
# partial, and a second kernel folds the partials into the sink, nearest chunk
# first. The tiling depends only on how many keys a query sees, so a batched
# verify row has the bits of the same token's batched decode step.
_DECODE_PART_SOURCE = """
    constexpr int BK = 16;
    constexpr int NT = NSG * 32;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int chunk = threadgroup_position_in_grid.x;
    const int gy = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int NS = dims[2];
    const int lp = dims[3 + b];
    const int pos = gy % L;
    const int hk = gy / L;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    // The G query heads of this KV head are the rows of the tile.
    const int g = int(sg) * 8 + fm;
    const bool row_ok = g < G;
    const int h = hk * G + (row_ok ? g : 0);
    // Index of the query's own key, and the number of keys it sees.
    const int a = S - L + pos;
    int nk = a + 1 - lp;
    if (WINDOW > 0) {
        nk = min(nk, WINDOW);
    }
    const int d_begin = chunk * CHUNK;
    const int d_end = min(d_begin + CHUNK, nk);
    const float qscale = scale[0] * LOG2E;

    // As in the prefill kernel: tiles staged in half (exact for FP16 and for
    // the BF16 values attention sees), K transposed, both padded.
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + pos * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = 0.0f;
    float l = 0.0f;
    const device T* relp = rel + ((size_t)(b * L + pos) * HQ + h) * R;
    const device T* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    // The vector load reads 16 bytes at a 16-byte address: a view whose rows
    // or base sit elsewhere takes the scalar loop.
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1
        && (k_strides[2] & 7) == 0 && (v_strides[2] & 7) == 0
        && (uintptr_t(kb_ptr) & 15) == 0 && (uintptr_t(vb_ptr) & 15) == 0;

    // Slot r of a tile is the key at distance d0 + r.
    for (int d0 = d_begin; d0 < d_end; d0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            for (int c = tid; c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (d0 + r < d_end) {
                    const int key = a - d0 - r;
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                if (IS_BF16) {
                    for (int j = 0; j < 4; ++j) {
                        kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                        kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                        vd[2 * j] = half(as_type<float>(vw[j] << 16));
                        vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                    }
                } else {
                    for (int j = 0; j < 4; ++j) {
                        const half2 k2 = as_type<half2>(kw[j]);
                        const half2 v2 = as_type<half2>(vw[j]);
                        kd[(2 * j) * LDK] = k2.x;
                        kd[(2 * j + 1) * LDK] = k2.y;
                        vd[2 * j] = v2.x;
                        vd[2 * j + 1] = v2.y;
                    }
                }
            }
        } else {
            for (int c = tid; c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                float kv = 0.0f;
                float vv = 0.0f;
                if (d0 + r < d_end) {
                    const int key = a - d0 - r;
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(sg) * 8 >= G) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        // Every tile holds a visible key (d0 < d_end), so rmax is a score.
        float sc[BK / 4];
        float rmax = -FLT_MAX;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int d = d0 + kb * 8 + fn + e;
                float s = Sf[kb].thread_elements()[e] * qscale;
                if (d < R) {
                    s += float(relp[d]) * LOG2E;
                }
                sc[kb * 2 + e] = s;
                if (d < d_end) {
                    rmax = max(rmax, s);
                }
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const bool first = d0 == d_begin;
        const float m_new = first ? rmax : max(m, rmax);
        const float factor = first ? 0.0f : precise::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int d = d0 + kb * 8 + fn + e;
                const float p = d < d_end
                    ? precise::exp2(sc[kb * 2 + e] - m_new) : 0.0f;
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        // A chunk past the query's keys leaves l == 0 and is skipped.
        const size_t slot = (((size_t)b * HQ + h) * L + pos) * NS + chunk;
        if (fn == 0) {
            part_m[slot] = m;
            part_l[slot] = l;
        }
        device float* op = part_o + slot * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = Of[db].thread_elements()[0];
            op[db * 8 + fn + 1] = Of[db].thread_elements()[1];
        }
    }
"""

# One simdgroup per query: the online softmax over its chunk partials,
# starting from the zero-value sink. L and NS are read at run time, so new
# block lengths and cache sizes do not compile new kernels.
_DECODE_FOLD_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    const int L = fdims[0];
    const int NS = fdims[1];
    const uint lane = thread_index_in_simdgroup;
    const int row = threadgroup_position_in_grid.x;
    const int pos = row % L;
    const int h = (row / L) % HQ;
    const int b = row / (L * HQ);
    constexpr int PER = (D + 31) / 32;
    float m = float(sink[b * sink_strides[0] + pos * sink_strides[1]
                         + h * sink_strides[2]]) * LOG2E;
    float l = 1.0f;
    float o[PER];
    for (int i = 0; i < PER; ++i) {
        o[i] = 0.0f;
    }
    const size_t base = (size_t)row * NS;
    for (int c = 0; c < NS; ++c) {
        const float lc = part_l[base + c];
        if (!(lc > 0.0f)) {
            continue;
        }
        const float mc = part_m[base + c];
        const float m_new = max(m, mc);
        const float f0 = precise::exp2(m - m_new);
        const float f1 = precise::exp2(mc - m_new);
        const float l0 = l * f0;
        const float l1 = lc * f1;
        l = l0 + l1;
        m = m_new;
        const device float* po = part_o + (base + c) * D;
        for (int i = 0; i < PER; ++i) {
            const int e = int(lane) + 32 * i;
            if (e < D) {
                const float o0 = o[i] * f0;
                const float o1 = po[e] * f1;
                o[i] = o0 + o1;
            }
        }
    }
    const float inv = 1.0f / l;
    for (int i = 0; i < PER; ++i) {
        const int e = int(lane) + 32 * i;
        if (e < D) {
            out[(size_t)row * D + e] = T(o[i] * inv);
        }
    }
"""
_decode_kernels = None
# Distances per chunk for full and sliding layers. They stay fixed per layer
# because a row's bits depend on its chunk layout.
_DECODE_CHUNK = 1024
_DECODE_SLIDING_CHUNK = 128
# Simdgroups staging one tile (the first ones also run the queries).
_DECODE_STAGERS = 2


def _use_flash_decode(q, k) -> bool:
    """BF16/FP16 queries on the GPU whose head layout fits the decode kernel."""
    return (
        q.dtype in (mx.bfloat16, mx.float16)
        and mx.default_device() == mx.gpu
        and q.shape[-1] % 8 == 0
        # Bytes of the staged K and V tiles, within Metal's 32 KB of
        # threadgroup memory (head dimensions up to 400).
        and 80 * q.shape[-1] + 256 <= 32768
        and q.shape[1] % k.shape[1] == 0
    )


def gqa_flash_decode_attention(q, k, v, rel, sink, *, scale, window, left_padding=None):
    """``banded_sink_attention`` for batched or left-padded decode-sized
    BF16/FP16 query blocks. Each row has the bits of the same row run alone."""
    global _decode_kernels
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    g = hq // hkv
    if _decode_kernels is None:
        _decode_kernels = (
            mx.fast.metal_kernel(
                name="omlx_inkling2_flash_decode_part",
                input_names=["q", "k", "v", "rel", "dims", "scale"],
                output_names=["part_o", "part_m", "part_l"],
                source=_DECODE_PART_SOURCE,
                ensure_row_contiguous=False,
            ),
            mx.fast.metal_kernel(
                name="omlx_inkling2_flash_decode_fold",
                input_names=["part_o", "part_m", "part_l", "sink", "fdims"],
                output_names=["out"],
                source=_DECODE_FOLD_SOURCE,
                ensure_row_contiguous=False,
            ),
        )
    part, fold = _decode_kernels
    reach = min(s, window) if window else s
    chunk = _DECODE_SLIDING_CHUNK if window else _DECODE_CHUNK
    ns = max(1, -(-reach // chunk))
    nsg = max(-(-g // 8), _DECODE_STAGERS)
    dims = mx.array([length, s, ns] + [0] * b, dtype=mx.int32)
    if left_padding is not None:
        dims = mx.concatenate([dims[:3], left_padding.astype(mx.int32)])
    part_o, part_m, part_l = part(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            dims,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("IS_BF16", q.dtype == mx.bfloat16),
            ("D", d),
            ("HQ", hq),
            ("G", g),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
            ("CHUNK", chunk),
            ("NSG", nsg),
        ],
        grid=(ns * nsg * 32, hkv * length, b),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[
            (b, hq, length, ns, d),
            (b, hq, length, ns),
            (b, hq, length, ns),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return fold(
        inputs=[part_o, part_m, part_l, sink, mx.array([length, ns], dtype=mx.int32)],
        template=[("T", q.dtype), ("D", d), ("HQ", hq)],
        grid=(b * hq * length * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b, hq, length, d)],
        output_dtypes=[q.dtype],
    )[0]


def _use_flash(length: int, window: int) -> bool:
    """Prefill blocks take the flash kernel on the GPU.

    On hosts with tensor units (NAX) MLX's masked SDPA runs on them, which
    this kernel does not; there only sliding layers, whose cost is dominated
    by building the mask, switch to it.
    """
    if length < _FLASH_MIN_QUERIES or mx.default_device() != mx.gpu:
        return False
    return bool(window) or not is_nax_available()


def prefill_mask_elements() -> int | None:
    """Mask elements one full-attention prefill call holds, None on flash."""
    return None if _use_flash(_FLASH_MIN_QUERIES, 0) else _MASK_BUDGET


def flash_banded_attention(q, k, v, rel, sink, *, scale, window, left_padding=None):
    """``banded_sink_attention`` for BF16/FP16 prefill blocks; same arguments and result."""
    global _flash_kernel
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    if _flash_kernel is None:
        _flash_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_flash_banded",
            input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
            output_names=["out"],
            source=_FLASH_SOURCE,
            ensure_row_contiguous=False,
        )
    lp = (
        left_padding.astype(mx.int32)
        if left_padding is not None
        else mx.zeros((b,), dtype=mx.int32)
    )
    dims = mx.concatenate([mx.array([length, s], dtype=mx.int32), lp])
    out = _flash_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sink.astype(mx.float32)),
            dims,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("IS_BF16", q.dtype == mx.bfloat16),
            ("D", d),
            ("HQ", hq),
            ("HKV", hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
        ],
        grid=(((length + 31) // 32) * 128, hq, b),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, length, hq, d)],
        output_dtypes=[q.dtype],
    )[0]
    return out.transpose(0, 2, 1, 3)


def banded_sink_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sink: mx.array,
    *,
    scale: float,
    window: int = 0,
    left_padding=None,
) -> mx.array:
    """Attention with a relative bias band and per-query sink logits.

    q: [B, Hq, L, D]; k, v: [B, Hkv, S, D] (queries are the last L keys);
    rel: [B, L, Hq, R] bias by backward distance d (applied for d < R);
    sink: [B, L, Hq] logits of a zero-value sink. ``window`` > 0 limits
    keys to d < window (sliding layers use window == R).
    """
    b, length = q.shape[0], q.shape[2]
    if (
        (b > 1 or left_padding is not None)
        and length < _FLASH_MIN_QUERIES
        and _use_flash_decode(q, k)
    ):
        # Batches, and a row left alone in a batch cache: the kernel reads
        # left_padding itself instead of masking every cached key.
        return gqa_flash_decode_attention(
            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding
        )
    if b == 1 and length == 1 and left_padding is None:
        return _decode_attention(q, k, v, rel, sink, scale, window)
    if b == 1 and left_padding is None and is_row_exact_armed():
        # Row-exact verify: each row is the decode step of its token, over the
        # keys up to its own position.
        fused = verify_attention.rows_attention(
            q, k, v, rel, sink, scale=scale, window=window
        )
        if fused is not None:
            return fused
        # The rows are joined as [B, L, H, D], so the output projection reads
        # them without a transpose copy.
        s = k.shape[2]
        rows = [
            _decode_attention(
                q[:, :, r : r + 1],
                k[:, :, : s - length + 1 + r],
                v[:, :, : s - length + 1 + r],
                rel[:, r : r + 1],
                sink[:, r : r + 1],
                scale,
                window,
            ).transpose(0, 2, 1, 3)
            for r in range(length)
        ]
        return mx.concatenate(rows, axis=1).transpose(0, 2, 1, 3)
    if q.dtype in (mx.bfloat16, mx.float16) and _use_flash(length, window):
        return flash_banded_attention(
            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding
        )
    return _masked_attention(q, k, v, rel, sink, scale, window, left_padding)


def _masked_attention(q, k, v, rel, sink, scale, window, left_padding):
    """MLX SDPA over an additive mask that carries the bias band and sinks."""
    b, hq, length, d = q.shape
    s = k.shape[2]
    r = rel.shape[-1]
    if window:
        assert window == r, "sliding layers expect rel_extent == window"
    lo = max(0, s - length - window + 1) if window else 0
    hkv = k.shape[1]
    # A zero key/value row carries the sink: its logit comes from the mask.
    zero = mx.zeros((b, hkv, 1, d), dtype=k.dtype)
    kz = mx.concatenate([zero, k[:, :, lo:]], axis=2)
    vz = mx.concatenate([zero, v[:, :, lo:]], axis=2)
    nk = s - lo
    qb = max(16, min(length, _MASK_BUDGET // max(1, b * hq * (nk + 1))))
    neg = mx.array(-mx.inf, dtype=q.dtype)
    lp_ok = None
    if left_padding is not None:
        lp_ok = (lo + mx.arange(nk))[None, :] >= left_padding[:, None]
        lp_ok = mx.concatenate([mx.ones((b, 1), dtype=mx.bool_), lp_ok], axis=1)
        lp_ok = lp_ok[:, None, None, :]

    outs = []
    for i0 in range(0, length, qb):
        i1 = min(length, i0 + qb)
        nq = i1 - i0
        a0 = s - length + i0
        hi = s - length + i1
        f = max(lo, a0 - r + 1)
        near = hi - f
        dist = (a0 - f) + mx.arange(nq)[:, None] - mx.arange(near)[None, :]
        relq = rel[:, i0:i1].transpose(0, 2, 1, 3)
        gidx = mx.broadcast_to(mx.clip(dist, 0, r - 1), (b, hq, nq, near))
        bias = mx.take_along_axis(relq, gidx, axis=-1)
        vis = dist >= 0
        if window:
            vis = vis & (dist < window)
        near_mask = mx.where(vis, mx.where(dist < r, bias, 0), neg)
        parts = [sink[:, i0:i1].transpose(0, 2, 1)[..., None].astype(q.dtype)]
        if f > lo:
            fill = neg if window else mx.array(0, dtype=q.dtype)
            parts.append(mx.broadcast_to(fill, (b, hq, nq, f - lo)))
        parts.append(near_mask.astype(q.dtype))
        if hi < s:
            parts.append(mx.broadcast_to(neg, (b, hq, nq, s - hi)))
        mask = mx.concatenate(parts, axis=-1)
        if lp_ok is not None:
            mask = mx.where(lp_ok, mask, neg)
        out = mx.fast.scaled_dot_product_attention(
            q[:, :, i0:i1], kz, vz, scale=scale, mask=mask
        )
        if qb < length:
            # A lazy graph would hold every block's mask at once (12 GiB at
            # 64K keys); evaluated block by block, one mask is alive.
            mx.eval(out)
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


# One head of one token per 32-thread threadgroup. Clip sums add rounded
# squares (MLX squares in a separate kernel), 4 adjacent values per lane then
# simd_sum, as MLX's row reduce; d_rel rows are summed in order. v follows
# rms_norm, whose squares are fused multiply-adds.
_QKV_POST_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint task = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const int L = q_shape[1];
    const int b = row / L;
    const int t = row % L;
    threadgroup float local_sums[32];
    threadgroup float inv_mean[1];
    if (task < HQ + HKV) {
        const bool is_q = task < HQ;
        const int head = is_q ? task : task - HQ;
        const int col = head * D + lane * 4;
        float xf[4];
        if (is_q) {
            const device T* src = q + b * q_strides[0] + t * q_strides[1];
            for (int i = 0; i < 4; ++i) {
                xf[i] = float(src[(col + i) * q_strides[2]]) * float(q_scale[col + i]);
            }
        } else {
            const device T* src = k + b * k_strides[0] + t * k_strides[1];
            for (int i = 0; i < 4; ++i) {
                xf[i] = float(src[(col + i) * k_strides[2]]);
            }
        }
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const float sq = xf[i] * xf[i];
            acc = sq + acc;
        }
        const float norm = metal::precise::sqrt(simd_sum(acc));
        const float f = metal::min(qk_clip[0] / (norm + 1e-6f), 1.0f);
        const int heads = is_q ? HQ : HKV;
        device T* dst = (is_q ? qo : ko) + ((size_t)row * heads + head) * D + lane * 4;
        for (int i = 0; i < 4; ++i) {
            T o = T(xf[i] * f);
            if (HAS_TAU && is_q) {
                o = T(float(o) * tau[b * tau_strides[0] + t * tau_strides[1]]);
            }
            dst[i] = o;
        }
    } else if (task < HQ + 2 * HKV) {
        const int head = task - HQ - HKV;
        const int col = head * D + lane * 4;
        const device float* src = v + b * v_strides[0] + t * v_strides[1];
        float xv[4];
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            xv[i] = src[(col + i) * v_strides[2]];
            acc = metal::fma(xv[i], xv[i], acc);
        }
        const float inv = omlx_inkling2_rms_inv(
            acc, local_sums, inv_mean, lane, 0, float(D), 1e-6f);
        device T* dst = vo + ((size_t)row * HKV + head) * D + lane * 4;
        for (int i = 0; i < 4; ++i) {
            dst[i] = T(1.0f * (xv[i] * inv));
        }
    } else {
        const int head = (task - HQ - 2 * HKV) * 32 + lane;
        if (head >= HQ) {
            return;
        }
        const device T* src = r + b * r_strides[0] + t * r_strides[1];
        float xf[R];
        float acc = 0.0f;
        for (int j = 0; j < R; ++j) {
            xf[j] = float(src[(head * R + j) * r_strides[2]]) * float(r_scale[head * R + j]);
            const float sq = xf[j] * xf[j];
            acc = sq + acc;
        }
        const float norm = metal::precise::sqrt(acc);
        const float f = metal::min(r_clip[0] / (norm + 1e-6f), 1.0f);
        device T* dst = ro + ((size_t)row * HQ + head) * R;
        for (int j = 0; j < R; ++j) {
            T o = T(xf[j] * f);
            if (HAS_TAU) {
                o = T(float(o) * tau[b * tau_strides[0] + t * tau_strides[1]]);
            }
            dst[j] = o;
        }
    }
"""
_qkv_post_kernel = None


def qkv_post(q, k, v, r, q_scale, r_scale, qk_clip, r_clip, tau, heads, kv_heads):
    """``Inkling2Attention.project``'s clips, v unit-RMS and log scaling as one
    kernel, or None when unsupported. Returns q, k, v as [B, L, H, D] and r as
    [B, L, Hq, R] in q's dtype."""
    global _qkv_post_kernel
    b, length, dq = q.shape
    dkv = k.shape[-1]
    head_dim = dq // heads
    d_rel = r.shape[-1] // heads
    if (
        mx.default_device() != mx.gpu
        or head_dim != 128
        or dkv != kv_heads * head_dim
        or not 0 < d_rel <= 64
        or k.dtype != q.dtype
        or r.dtype != q.dtype
        or v.dtype != mx.float32
    ):
        return None
    if _qkv_post_kernel is None:
        _qkv_post_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_qkv_post",
            input_names=[
                "q",
                "k",
                "v",
                "r",
                "q_scale",
                "r_scale",
                "qk_clip",
                "r_clip",
                "tau",
            ],
            output_names=["qo", "ko", "vo", "ro"],
            source=_QKV_POST_SOURCE,
            header=_RMS_INV_HEADER,
            ensure_row_contiguous=False,
        )
    has_tau = tau is not None
    # 0-d arrays would be bound by value; the kernel indexes them.
    qk_clip = qk_clip.reshape(1)
    r_clip = r_clip.reshape(1)
    if has_tau:
        tau = mx.broadcast_to(tau.reshape(tau.shape[0], length), (b, length))
    else:
        tau = qk_clip.reshape(1, 1)
    dtype = q.dtype
    tasks = heads + 2 * kv_heads + (heads + 31) // 32
    return _qkv_post_kernel(
        inputs=[q, k, v, r, q_scale, r_scale, qk_clip, r_clip, tau],
        template=[
            ("T", dtype),
            ("HQ", heads),
            ("HKV", kv_heads),
            ("D", head_dim),
            ("R", d_rel),
            ("HAS_TAU", has_tau),
        ],
        grid=(tasks * 32, b * length, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (b, length, heads, head_dim),
            (b, length, kv_heads, head_dim),
            (b, length, kv_heads, head_dim),
            (b, length, heads, d_rel),
        ],
        output_dtypes=[dtype] * 4,
    )


_ROUTER_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    constexpr uint PER = E / 32;
    float sc[PER];
    float key[PER];
    for (uint j = 0; j < PER; ++j) {
        uint e = lane + 32 * j;
        float s = 1.0f / (1.0f + metal::exp(-logits[row * E + e]));
        sc[j] = s;
        key[j] = s + bias[e];
    }
    float picked[K];
    uint chosen[K];
    float total = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint bj = 0;
        for (uint j = 0; j < PER; ++j) {
            if (key[j] > best) {
                best = key[j];
                bj = j;
            }
        }
        float top = simd_max(best);
        uint mine = (best == top) ? (lane + 32 * bj) : 0xFFFFFFFFu;
        uint win = simd_min(mine);
        float s = (lane == win % 32) ? sc[win / 32] : 0.0f;
        s = simd_sum(s);
        if (lane == win % 32) key[win / 32] = -INFINITY;
        picked[k] = s;
        chosen[k] = win;
        total += s;
    }
    if (lane == 0) {
        float norm = route_scale[0] / (total + 1e-20f);
        for (uint k = 0; k < K; ++k) {
            inds[row * K + k] = chosen[k];
            weights[row * K + k] = picked[k] * norm;
        }
    }
"""
_router_kernel = None


def route_topk(logits: mx.array, bias: mx.array, top_k: int, route_scale: float):
    """Sigmoid scores, top-k by ``score + bias``, renormalized unbiased weights.

    ``logits``: [N, E] FP32. Returns ([N, K] uint32 indices, [N, K] FP32).
    """
    global _router_kernel
    rows, experts = logits.shape
    if mx.default_device() != mx.gpu or experts % 32:
        scores = mx.sigmoid(logits)
        inds = mx.argpartition(-(scores + bias), kth=top_k - 1, axis=-1)[..., :top_k]
        w = mx.take_along_axis(scores, inds, axis=-1)
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20) * route_scale
        return inds, w
    if _router_kernel is None:
        _router_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_route_topk",
            input_names=["logits", "bias", "route_scale"],
            output_names=["inds", "weights"],
            source=_ROUTER_SOURCE,
        )
    inds, weights = _router_kernel(
        inputs=[
            logits.astype(mx.float32),
            bias.astype(mx.float32),
            mx.array([route_scale], dtype=mx.float32),
        ],
        template=[("E", experts), ("K", top_k)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, weights


_SHORT_CONV_HEADER = """
// Row i of the padded window [state, x], x rounded to T; the state has p rows.
template <typename T, typename S, typename X>
inline float omlx_inkling2_padded(
    const device S* srow,
    int64_t ss,
    const device X* xrow,
    int64_t xs,
    int p,
    int i) {
    return i < p ? float(srow[i * ss]) : float(T(xrow[(i - p) * xs]));
}

// x is the experts' weighted sum, added as MLX's small column reduce does.
// The weights may be bound in the constant address space (fewer than 8).
template <typename T, typename S, typename Z, int KE, typename WP>
inline float omlx_inkling2_padded_moe(
    const device S* srow,
    int64_t ss,
    const device Z* zcol,
    WP wt,
    int C,
    int p,
    int i) {
    if (i < p) {
        return float(srow[i * ss]);
    }
    const int t = i - p;
    const device Z* z = zcol + (size_t)t * KE * C;
    const auto w = wt + (size_t)t * KE;
    float acc = 0.0f;
    for (int e = 0; e < KE; ++e) {
        const float prod = float(z[e * C]) * w[e];
        const float term = prod + 0.0f;
        acc = e == 0 ? term : term + acc;
    }
    return float(T(acc));
}
"""

# One thread per (channel, padded row, batch row) of the window [state, x].
# Rows below K - 1 only copy the state; the others also compute their token.
# Each product and sum is its own statement, in the order and precision of
# short_conv_reference: a fused multiply-add would change the bits.
_SHORT_CONV_SOURCE = """
    constexpr int P = K - 1;
    const int c = thread_position_in_grid.x;
    const int y = thread_position_in_grid.y;
    const int b = thread_position_in_grid.z;
    const int L = x_shape[1];
    const int C = x_shape[2];
    const device S* srow = state + b * state_strides[0] + c * state_strides[2];
    const device X* xrow = x + b * x_strides[0] + c * x_strides[2];
    const int64_t ss = state_strides[1];
    const int64_t xs = x_strides[1];
    const float own = omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, y);
    xp[((size_t)b * (P + L) + y) * C + c] = own;
    if (y < P) {
        return;
    }
    const device W* wrow = w + c * w_strides[0];
    const int t = y - P;
    float acc = float(wrow[0]) * omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, t);
    for (int j = 1; j < K; ++j) {
        const float tap = float(wrow[j * w_strides[1]]);
        const float term = tap * omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, t + j);
        acc = acc + term;
    }
    // Rounded to the weight dtype before the residual add, as the reference.
    const float conv = float(W(acc));
    const float res = conv + own;
    const float o = res * scale[0];
    // The result in T, then the caller's residual add and cast.
    float v = float(T(o));
    if (HAS_RESIDUAL) {
        v = float(residual[b * residual_strides[0] + t * residual_strides[1]
                           + c * residual_strides[2]]) + v;
    }
    out[((size_t)b * L + t) * C + c] = O(v);
"""
_short_conv_kernel = None


def short_conv_reference(x, w, state, scale, residual=None, out_dtype=None, dtype=None):
    """``short_conv`` as MLX ops (CPU path and test reference)."""
    x = x.astype(dtype or x.dtype)
    xf = x.astype(mx.float32)
    k = w.shape[-1]
    xp = mx.concatenate([state, xf], axis=1)
    length = xf.shape[1]
    out = w[:, 0] * xp[:, 0:length]
    for j in range(1, k):
        out = out + w[:, j] * xp[:, j : j + length]
    # The reference rounds the depthwise conv to the weight dtype before the
    # residual add.
    out = (out.astype(w.dtype).astype(mx.float32) + xf) * scale
    out = out.astype(x.dtype)
    if residual is not None:
        out = residual.astype(mx.float32) + out.astype(mx.float32)
    return out.astype(out_dtype or x.dtype), xp


def short_conv(
    x: mx.array,
    w: mx.array,
    state: mx.array,
    scale: float,
    residual: mx.array | None = None,
    out_dtype=None,
    dtype=None,
):
    """Depthwise causal conv with residual (taps: ``w[:, K-1]`` is the current
    token), as one kernel.

    ``x``: [B, L, C], read as ``dtype`` (the activation dtype, default
    ``x.dtype``); ``w``: [C, K]; ``state``: [B, K-1, C], the previous K-1
    inputs. The output is ``(x + conv) * scale`` in ``dtype``; given a
    ``residual`` [B, L, C], it is then added to the residual in FP32. Either
    way it is returned cast to ``out_dtype`` (default ``dtype``), with the
    FP32 padded window ``[state, x]``, whose last K-1 rows are the next
    state. Strided inputs are read in place.
    """
    global _short_conv_kernel
    dtype = dtype or x.dtype
    out_dtype = out_dtype or dtype
    if mx.default_device() != mx.gpu:
        return short_conv_reference(x, w, state, scale, residual, out_dtype, dtype)
    if _short_conv_kernel is None:
        _short_conv_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_short_conv",
            input_names=["x", "w", "state", "residual", "scale"],
            output_names=["out", "xp"],
            header=_SHORT_CONV_HEADER,
            source=_SHORT_CONV_SOURCE,
            ensure_row_contiguous=False,
        )
    b, length, channels = x.shape
    k = w.shape[-1]
    has_residual = residual is not None
    # Without a residual, x is bound in its place and never read.
    residual = residual if has_residual else x
    out, xp = _short_conv_kernel(
        inputs=[x, w, state, residual, mx.array([scale], dtype=mx.float32)],
        template=[
            ("T", dtype),
            ("X", x.dtype),
            ("W", w.dtype),
            ("S", state.dtype),
            ("O", out_dtype),
            ("K", k),
            ("HAS_RESIDUAL", has_residual),
        ],
        grid=(channels, length + k - 1, b),
        threadgroup=(min(channels, 256), 1, 1),
        output_shapes=[(b, length, channels), (b, length + k - 1, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )
    return out, xp


_SHORT_CONV_MOE_SOURCE = (
    _SHORT_CONV_SOURCE.replace("const int C = x_shape[2];", "const int C = x_shape[3];")
    .replace(
        "const device X* xrow = x + b * x_strides[0] + c * x_strides[2];",
        "const device X* zcol = x + (size_t)b * L * KE * C + c;",
    )
    .replace(
        "const int64_t xs = x_strides[1];",
        "const auto wt = moe_w + (size_t)b * L * KE;",
    )
    .replace(
        "omlx_inkling2_padded<T>(srow, ss, xrow, xs, P,",
        "omlx_inkling2_padded_moe<T, S, X, KE>(srow, ss, zcol, wt, C, P,",
    )
)
_short_conv_moe_kernel = None
# The weighted sum is MLX's small column reduce only while one threadgroup row
# holds each expert.
_MOE_SUM_MAX_EXPERTS = 8


def moe_short_conv(
    z: mx.array,
    weights: mx.array,
    w: mx.array,
    state: mx.array,
    scale: float,
    residual: mx.array,
    dtype,
):
    """``short_conv`` of the expert sum ``moe_sum(z, weights)`` with the sum done
    in the kernel, or None when unsupported."""
    global _short_conv_moe_kernel
    b, length, experts, channels = z.shape
    if mx.default_device() != mx.gpu or experts > _MOE_SUM_MAX_EXPERTS:
        return None
    if weights.dtype != mx.float32:
        return None
    if _short_conv_moe_kernel is None:
        _short_conv_moe_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_short_conv_moe",
            input_names=["x", "w", "state", "residual", "scale", "moe_w"],
            output_names=["out", "xp"],
            header=_SHORT_CONV_HEADER,
            source=_SHORT_CONV_MOE_SOURCE,
            ensure_row_contiguous=False,
        )
    k = w.shape[-1]
    return _short_conv_moe_kernel(
        inputs=[
            mx.contiguous(z),
            w,
            state,
            residual,
            mx.array([scale], dtype=mx.float32),
            mx.contiguous(weights),
        ],
        template=[
            ("T", dtype),
            ("X", z.dtype),
            ("W", w.dtype),
            ("S", state.dtype),
            ("O", dtype),
            ("K", k),
            ("KE", experts),
            ("HAS_RESIDUAL", True),
        ],
        grid=(channels, length + k - 1, b),
        threadgroup=(min(channels, 256), 1, 1),
        output_shapes=[(b, length, channels), (b, length + k - 1, channels)],
        output_dtypes=[dtype, mx.float32],
    )


# The full-attention k/v conv: per-token taps static_w[lag] + dyn[head, lag],
# laid out like _SHORT_CONV_SOURCE, with the sum over lags newest first and
# no rounding before the residual add, as dynamic_conv_reference.
_DYNAMIC_CONV_SOURCE = """
    constexpr int P = K - 1;
    const int c = thread_position_in_grid.x;
    const int y = thread_position_in_grid.y;
    const int b = thread_position_in_grid.z;
    const int L = x_shape[1];
    const int C = x_shape[2];
    const device S* srow = state + b * state_strides[0] + c * state_strides[2];
    const device T* xrow = x + b * x_strides[0] + c * x_strides[2];
    const int64_t ss = state_strides[1];
    const int64_t xs = x_strides[1];
    const float own = omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, y);
    xp[((size_t)b * (P + L) + y) * C + c] = own;
    if (y < P) {
        return;
    }
    const int t = y - P;
    const int head = c / (C / H);
    const auto drow = dyn + b * dyn_strides[0] + t * dyn_strides[1]
        + head * K * dyn_strides[2];
    float acc = 0.0f;
    for (int lag = 0; lag < K; ++lag) {
        const float sw = float(static_w[lag * static_w_strides[0]
                                        + c * static_w_strides[1]]);
        const float tap = sw + float(drow[lag * dyn_strides[2]]);
        const float term = omlx_inkling2_padded<T>(srow, ss, xrow, xs, P, y - lag)
            * tap;
        acc = lag == 0 ? term : acc + term;
    }
    const float res = acc + own;
    const float o = res * scale[0];
    out[((size_t)b * L + t) * C + c] = O(float(T(o)));
"""
_dynamic_conv_kernel = None


def dynamic_conv_reference(x, dyn, static_w, state, heads, scale, out_dtype=None):
    """``dynamic_conv`` as MLX ops (CPU path and test reference)."""
    xf = x.astype(mx.float32)
    b, length, c = xf.shape
    k = static_w.shape[0]
    xp = mx.concatenate([state, xf], axis=1)
    dyn = dyn.astype(mx.float32).reshape(b, length, heads, 1, k)
    out = None
    for lag in range(k):
        start = k - 1 - lag
        window = xp[:, start : start + length].reshape(b, length, heads, c // heads)
        # A slice, not static_w[lag]: integer indexing is a gather kernel.
        tap = static_w[lag : lag + 1].reshape(heads, c // heads) + dyn[..., lag]
        term = window * tap
        out = term if out is None else out + term
    out = (out.reshape(b, length, c) + xf) * scale
    return out.astype(x.dtype).astype(out_dtype or x.dtype), xp


def dynamic_conv(
    x: mx.array,
    dyn: mx.array,
    static_w: mx.array,
    state: mx.array,
    heads: int,
    scale: float,
    out_dtype=None,
):
    """Depthwise causal conv with per-token taps ``static_w[lag] + dyn[head,
    lag]`` and residual, as one kernel (full-attention k and v).

    ``x``: [B, L, C]; ``dyn``: [B, L, heads * K]; ``static_w``: [K, C], lag 0
    the current token; ``state``: [B, K-1, C]. Returns ``(x + conv) * scale``
    rounded to ``x``'s dtype, cast to ``out_dtype`` (default ``x.dtype``),
    and the FP32 padded window ``[state, x]``.
    """
    global _dynamic_conv_kernel
    out_dtype = out_dtype or x.dtype
    if mx.default_device() != mx.gpu:
        return dynamic_conv_reference(x, dyn, static_w, state, heads, scale, out_dtype)
    if _dynamic_conv_kernel is None:
        _dynamic_conv_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_dynamic_conv",
            input_names=["x", "dyn", "static_w", "state", "scale"],
            output_names=["out", "xp"],
            header=_SHORT_CONV_HEADER,
            source=_DYNAMIC_CONV_SOURCE,
            ensure_row_contiguous=False,
        )
    b, length, channels = x.shape
    k = static_w.shape[0]
    out, xp = _dynamic_conv_kernel(
        inputs=[x, dyn, static_w, state, mx.array([scale], dtype=mx.float32)],
        template=[
            ("T", x.dtype),
            ("S", state.dtype),
            ("O", out_dtype),
            ("K", k),
            ("H", heads),
        ],
        grid=(channels, length + k - 1, b),
        threadgroup=(min(channels, 256), 1, 1),
        output_shapes=[(b, length, channels), (b, length + k - 1, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )
    return out, xp
