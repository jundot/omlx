# SPDX-License-Identifier: Apache-2.0
"""Fused single-token kernels for Inkling 2 decode.

The eager layer body spends most of a decode step on short dependent Metal
launches: the four depthwise convolutions, the q/k/r L2 clips, the loop
controller mixing and the MoE epilogue each expand into a dozen tiny
elementwise kernels. At batch-one decode every launch costs about as much as
the arithmetic it performs, so this module folds those groups into one launch
each while keeping the reference arithmetic (rounding points included).

Every kernel here is a drop-in for an existing expression; ``model.py`` only
takes these paths for a single unpadded row and falls back to the eager body
otherwise. Set ``OMLX_INKLING2_FAST_DECODE=0`` to disable.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

_ENABLED = os.environ.get("OMLX_INKLING2_FAST_DECODE", "1") != "0"
# Ablation hook: comma-separated fusion names to keep on the eager path.
_OFF = {p for p in os.environ.get("OMLX_FD_OFF", "").split(",") if p}

_kernels: dict = {}
logger = logging.getLogger(__name__)


_LOGGED = False


def enabled() -> bool:
    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.info(
            "Inkling 2 fused decode: %s (parts off: %s)",
            "on" if _ENABLED else "off",
            sorted(_OFF) or "none",
        )
    return _ENABLED and mx.default_device() == mx.gpu


def part(name: str) -> bool:
    return _ENABLED and name not in _OFF and mx.default_device() == mx.gpu


def _rup(a: int, m: int) -> int:
    return ((a + m - 1) // m) * m


def _kernel(name: str, source: str, inputs, outputs):
    kernel = _kernels.get(name)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name, input_names=inputs, output_names=outputs, source=source
        )
        _kernels[name] = kernel
    return kernel


# ---------------------------------------------------------------- convs ----

_CONV_SRC = r"""
    uint c = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (c >= C || b >= B) return;
    for (uint i = 0; i < L; ++i) {
        float acc = 0.0f;
        for (uint kk = 0; kk < K; ++kk) {
            int r = (int)(i + kk) - (int)(K - 1);
            float v;
            if (r < 0) {
                v = state[((size_t)b * (K - 1) + (uint)(r + int(K - 1))) * C + c];
            } else {
                v = (float)x[((size_t)b * L + (uint)r) * C + c];
            }
            acc += (float)w[c * K + kk] * v;
        }
        TW conv_r = (TW)acc;
        float o = (((float)conv_r) + (float)x[((size_t)b * L + i) * C + c]) * scale[0];
        out[((size_t)b * L + i) * C + c] = (T)o;
    }
    for (uint s = 0; s < K - 1; ++s) {
        int r = (int)(L + s) - (int)(K - 1);
        nstate[((size_t)b * (K - 1) + s) * C + c] = (r < 0)
            ? state[((size_t)b * (K - 1) + (uint)(r + int(K - 1))) * C + c]
            : (float)x[((size_t)b * L + (uint)r) * C + c];
    }
    // The padded input window [state, x] that the reference keeps as its
    // cache entry; MTP rollback re-cuts conv state from it.
    for (uint r = 0; r < K - 1 + L; ++r) {
        window[((size_t)b * (K - 1 + L) + r) * C + c] = (r < K - 1)
            ? state[((size_t)b * (K - 1) + r) * C + c]
            : (float)x[((size_t)b * L + (r - (K - 1))) * C + c];
    }
"""


def conv_short(x: mx.array, w: mx.array, state: mx.array, scale: float):
    """Depthwise causal conv for a short window (decode or MTP verify).

    Mirrors ``model._static_conv``: the window is ``[state (K-1 rows), x]``,
    the conv result is rounded to the weight dtype before the residual add,
    and the product by ``scale`` is rounded to ``x``. Returns ``(out, state)``
    with the next state already trimmed to ``K - 1`` rows.
    """
    kernel = _kernel(
        "omlx_inkling2_conv_short",
        _CONV_SRC,
        ["x", "w", "state", "scale"],
        ["out", "nstate", "window"],
    )
    b, length, c = x.shape
    k = w.shape[-1]
    out, nstate, window = kernel(
        inputs=[x, w, state, mx.array([scale], dtype=mx.float32)],
        template=[
            ("T", x.dtype),
            ("TW", w.dtype),
            ("B", b),
            ("L", length),
            ("C", c),
            ("K", k),
        ],
        grid=(_rup(c, 64), b, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(b, length, c), (b, k - 1, c), (b, k - 1 + length, c)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )
    return out, nstate, window


# ---------------------------------------------------------------- clips ----

_CLIP_SRC = r"""
    uint d = thread_position_in_grid.x;      // position within a head row
    uint hz = thread_position_in_grid.y;     // head
    uint row = thread_position_in_grid.z;    // b * L + l
    size_t off = ((size_t)row * H + hz) * D;
    bool live = (d < D) && (hz < H);
    float v = live ? (float)x[off + d] : 0.0f;
    if (live && HAS_SCALE != 0) v *= scale[hz * D + d];
    float total = simd_sum(v * v);
    threadgroup float red[4];
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    if (lane == 0) red[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = red[0] + red[1] + red[2] + red[3];
    if (live) {
        float g = min(threshold[0] / (sqrt(total) + 1e-6f), 1.0f);
        out[off + d] = (T)(v * g);
    }
"""

_UNIT_RMS_SRC = r"""
    uint d = thread_position_in_grid.x;
    uint hz = thread_position_in_grid.y;
    uint row = thread_position_in_grid.z;
    size_t off = ((size_t)row * H + hz) * D;
    bool live = (d < D) && (hz < H);
    float v = live ? (float)x[off + d] : 0.0f;
    float total = simd_sum(v * v);
    threadgroup float red[4];
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    if (lane == 0) red[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = (red[0] + red[1] + red[2] + red[3]) / (float)D;
    if (live) out[off + d] = (T)(v / sqrt(var + eps[0]));
"""


def l2_clip_heads(x: mx.array, scale, threshold: float):
    """``model._l2_clip`` for ``[rows, heads, dim]`` rows in one launch."""
    kernel = _kernel(
        "omlx_inkling2_l2_clip", _CLIP_SRC, ["x", "scale", "threshold"], ["out"]
    )
    rows, heads, dim = x.shape
    has_scale = getattr(scale, "ndim", 0) == 2
    sc = (
        scale.astype(mx.float32)
        if has_scale
        else mx.array([float(scale.reshape(-1)[0].item())], dtype=mx.float32)
    )
    return kernel(
        inputs=[x, sc, mx.array([threshold], dtype=mx.float32)],
        template=[
            ("T", x.dtype),
            ("H", heads),
            ("D", dim),
            ("HAS_SCALE", int(has_scale)),
        ],
        grid=(_rup(dim, 128), heads, rows),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def unit_rms_heads(x: mx.array):
    """``model._unit_rms`` over the last axis of ``[rows, heads, dim]``."""
    kernel = _kernel(
        "omlx_inkling2_unit_rms_heads", _UNIT_RMS_SRC, ["x", "eps"], ["out"]
    )
    rows, heads, dim = x.shape
    return kernel(
        inputs=[x.astype(mx.float32), mx.array([1e-6], dtype=mx.float32)],
        template=[("T", x.dtype), ("H", heads), ("D", dim)],
        grid=(_rup(dim, 128), heads, rows),
        threadgroup=(128, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


# ------------------------------------------------------------ MoE epilogue ----

_SWIGLU_SRC = r"""
    uint i = thread_position_in_grid.x;     // intermediate index
    uint kk = thread_position_in_grid.y;    // selected expert slot
    uint n = thread_position_in_grid.z;     // token row
    if (i >= I || kk >= K) return;
    float s = s13[n * K + kk];
    size_t base = ((size_t)n * K + kk) * (2u * I) + 2u * i;
    T g0 = (T)((float)y[base] * s);
    T u0 = (T)((float)y[base + 1] * s);
    float g = (float)g0;
    float u = (float)u0;
    float og = tanh(clamp(g / gc[0], -10.0f, 10.0f)) * gc[0];
    float ou = tanh(clamp(u / uc[0], -10.0f, 10.0f)) * uc[0];
    // metal::exp(-x) returns NaN for x well below zero under fast math, and a
    // negative logit that large makes sigmoid exactly 0/1 in the activation
    // dtype anyway, so clamp before the exponential.
    float act = og * (1.0f / (1.0f + metal::exp(-clamp(g, -88.0f, 88.0f)))) * ou;
    out[(size_t)n * K * I + (size_t)kk * I + i] = (T)act;
"""

_COMBINE_SRC = r"""
    uint h = thread_position_in_grid.x;     // hidden index
    uint n = thread_position_in_grid.y;     // token row
    if (h >= H) return;
    float acc = 0.0f;
    for (uint kk = 0; kk < K; ++kk) {
        float z = (float)zin[(size_t)n * K * H + (size_t)kk * H + h];
        acc += z * (wk[n * K + kk] * s2[n * K + kk]);
    }
    if (HAS_SHARED != 0) acc += (float)sh[(size_t)n * H + h];
    out[(size_t)n * H + h] = (T)acc;
"""


def swiglu_scaled(y: mx.array, s13: mx.array, gcap: float, ucap: float):
    """Routed w13 output -> activated experts in one launch."""
    kernel = _kernel(
        "omlx_inkling2_swiglu_scaled", _SWIGLU_SRC, ["y", "s13", "gc", "uc"], ["out"]
    )
    n, k, two_i = y.shape
    i = two_i // 2
    return kernel(
        inputs=[
            y,
            s13.astype(mx.float32),
            mx.array([gcap], dtype=mx.float32),
            mx.array([ucap], dtype=mx.float32),
        ],
        template=[("T", y.dtype), ("I", i), ("K", k)],
        grid=(_rup(i, 128), k, n),
        threadgroup=(128, 1, 1),
        output_shapes=[(n, k, i)],
        output_dtypes=[y.dtype],
    )[0]


def moe_combine(z: mx.array, s2: mx.array, weights: mx.array, shared=None):
    """Weighted k-sum (+ shared expert) of the per-expert outputs, one launch."""
    kernel = _kernel(
        "omlx_inkling2_moe_combine", _COMBINE_SRC, ["zin", "s2", "wk", "sh"], ["out"]
    )
    n, k, h = z.shape
    has_shared = shared is not None
    sh = shared if has_shared else mx.zeros((n, h), dtype=z.dtype)
    return kernel(
        inputs=[z, s2.astype(mx.float32), weights.astype(mx.float32), sh],
        template=[("T", z.dtype), ("H", h), ("K", k), ("HAS_SHARED", int(has_shared))],
        grid=(_rup(h, 256), n, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, h)],
        output_dtypes=[z.dtype],
    )[0]


# ------------------------------------------------------- loop controllers ----

_MSG_PREP_SRC = r"""
    uint tid = thread_position_in_grid.x;    // 0..127
    uint row = thread_position_in_grid.y;    // (n * rowc) + r
    uint which = thread_position_in_grid.z;  // 0 = x, 1 = message
    uint n = row / ROWC;
    uint rw = row - n * ROWC;
    const device T* src = (which == 0) ? x : msg;
    size_t base = (size_t)row * H;
    float acc = 0.0f;
    for (uint h = tid; h < H; h += 128u) {
        float v = (float)src[base + h];
        acc += v * v;
    }
    float total = simd_sum(acc);
    threadgroup float red[4];
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    if (lane == 0) red[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = (red[0] + red[1] + red[2] + red[3]) / (float)H;
    float inv = 1.0f / sqrt(var + 1e-6f);
    size_t obase = (size_t)((2u * n + which) * ROWC + rw) * H;
    for (uint h = tid; h < H; h += 128u) {
        z[obase + h] = (T)((float)src[base + h] * inv);
    }
"""

_MSG_LOW_SRC = r"""
    uint r = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    if (r >= R) return;
    uint rw = row % ROWS;
    size_t a = ((size_t)(2u * row - rw) * R + r);
    size_t b = ((size_t)(2u * row - rw + ROWS) * R + r);
    float v = halves[a] + halves[b];
    low[(size_t)row * R + r] = (T)v;   // T is bf16: the reference's bf16 round-trip
"""

_MSG_APPLY_SRC = r"""
    uint h = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;   // flat row in [0, N*ROWS)
    if (h >= H) return;
    size_t gbase = (size_t)row * (2u * H);
    float gain = tanh(clamp(gates[gbase + h], -10.0f, 10.0f));
    float forget = tanh(clamp(gates[gbase + H + h], -10.0f, 10.0f));
    float xv = (float)x[(size_t)row * H + h];
    float mv = (float)msg[(size_t)row * H + h];
    out[(size_t)row * H + h] = (T)((1.0f - forget) * xv + gain * mv);
"""


def messages_prep(x: mx.array, message: mx.array):
    """Unit-RMS both controller inputs, rounded like the reference's z round-trip.

    ``x``/``message``: [N, rows, H]. Returns [2N*rows, H] in the activation
    dtype (the caller promotes it back to FP32 for the down matmul).
    """
    kernel = _kernel("omlx_inkling2_msg_prep", _MSG_PREP_SRC, ["x", "msg"], ["z"])
    n, rows, h = x.shape
    return kernel(
        inputs=[x, message],
        template=[("T", x.dtype), ("H", h), ("ROWC", rows)],
        grid=(128, n * rows, 2),
        threadgroup=(128, 1, 1),
        output_shapes=[(2 * n * rows, h)],
        output_dtypes=[x.dtype],
    )[0]


def messages_low(halves: mx.array, rows: int):
    """Sum the x/message halves of the down projection and round to bf16."""
    kernel = _kernel("omlx_inkling2_msg_low", _MSG_LOW_SRC, ["halves"], ["low"])
    rank = halves.shape[-1]
    halves = halves.reshape(-1, rank)
    total = halves.shape[0] // 2
    return kernel(
        inputs=[halves.astype(mx.float32)],
        template=[("T", mx.bfloat16), ("ROWS", rows), ("R", rank)],
        grid=(_rup(rank, 64), total, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(total, rank)],
        output_dtypes=[mx.bfloat16],
    )[0].astype(mx.float32)


def messages_apply(x: mx.array, message: mx.array, gates: mx.array):
    """``(1 - tanh(forget)) * x + tanh(gain) * message`` in one launch."""
    kernel = _kernel(
        "omlx_inkling2_msg_apply", _MSG_APPLY_SRC, ["x", "msg", "gates"], ["out"]
    )
    n, rows, h = x.shape
    return kernel(
        inputs=[
            x.astype(mx.float32),
            message.astype(mx.float32),
            gates.astype(mx.float32),
        ],
        template=[("T", mx.float32), ("H", h)],
        grid=(_rup(h, 128), n * rows, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(n, rows, h)],
        output_dtypes=[mx.float32],
    )[0]


# ------------------------------------------------------------- attention ----

# ``S`` stays a runtime buffer: templating it compiled a new pipeline on every
# decode step (the KV length grows by one per token).
_MASK_SRC = r"""
    uint j = thread_position_in_grid.x;     // key offset, 0 = oldest in window
    uint hz = thread_position_in_grid.y;
    uint S = dims[0];
    if (j >= S || hz >= H) return;
    uint d = S - 1 - j;                     // backward distance of this key
    float v = (d < R) ? (float)rel[(size_t)hz * R + d] : 0.0f;
    out[(size_t)hz * S + j] = (T)v;
"""


def decode_mask(rel: mx.array, s: int):
    """Additive decode mask ``[1, H, 1, S]`` from the relative bias row."""
    kernel = _kernel("omlx_inkling2_decode_mask", _MASK_SRC, ["rel", "dims"], ["out"])
    _, _, heads, r = rel.shape
    return kernel(
        inputs=[rel.reshape(heads, r), mx.array([s], dtype=mx.int32)],
        template=[("T", rel.dtype), ("H", heads), ("R", r)],
        grid=(_rup(s, 128), heads, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(1, heads, 1, s)],
        output_dtypes=[rel.dtype],
    )[0]


# ------------------------------------------------------- NVFP4 expert GEMV ----
#
# Decode runs each selected expert as a 1-row GEMV. MLX's ``gather_qmm`` pays a
# launch per selected expert and the epilogue adds two more launches, so one
# layer's routed experts cost ~160 us. These two kernels do the same arithmetic
# — E2M1 codes times E4M3 block scales times the per-expert tensor scale,
# accumulated in FP32 — in one launch for gate+up+SwiGLU and one for
# down+combine. Used only for a single row (plain decode); everything else keeps
# ``gather_qmm``.

# E2M1 codes 0..15 (sign bit 3) as floats; a device-buffer table beats the
# branchless arithmetic form by ~4x in the inner loop.
# Metal's fast-math tanh/exp return NaN well outside the linear range (tanh at
# 250 already), and every activation in these kernels saturates long before
# that. Clamping the argument keeps the reference result while staying finite.
_E2M1_LUT = mx.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=mx.float32,
)

_NVFP4_INLINE = r"""
    // E4M3 block scale (bias 7, subnormals at m/8 * 2^-6). The E2M1 codes are
    // looked up in a 16-entry table passed by the caller: the branchless
    // arithmetic form compiles to a long select chain and is 4x slower here.
    #define NVFP4_E4M3(b) (((b) & 0x7Fu) == 0u ? 0.0f : \
        ((1.0f + (float)((b) & 0x7u) * 0.125f) \
         * metal::exp2((float)(((b) >> 3) & 0xFu) - 7.0f) \
         * (((b) & 0x80u) ? -1.0f : 1.0f)))
"""

_GATE_UP_SRC = r"""
    uint tg = threadgroup_position_in_grid.x;
    uint slot = tg / IBLOCKS;
    uint ib = tg - slot * IBLOCKS;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint i = ib * 4u + sg;
    if (slot >= TOPK || i >= INTER) return;
    uint e = idx[slot];
    float wscale = s13[slot];
    size_t rowg = ((size_t)e * (2u * INTER) + 2u * i) * WORDS;
    size_t rowu = rowg + WORDS;
    size_t scg = ((size_t)e * (2u * INTER) + 2u * i) * GROUPS;
    size_t scu = scg + GROUPS;
    float accg = 0.0f, accu = 0.0f;
    for (uint u = lane; u < WORDS; u += 32u) {
        float sg_ = NVFP4_E4M3(scales[scg + (u >> 1)]);
        float su_ = NVFP4_E4M3(scales[scu + (u >> 1)]);
        uint wg = w13[rowg + u];
        uint wu = w13[rowu + u];
        uint xb = u * 8u;
        for (uint t = 0; t < 8u; ++t) {
            float xv = (float)x[xb + t];
            accg += (xv * sg_) * lut[(wg >> (4u * t)) & 0xFu];
            accu += (xv * su_) * lut[(wu >> (4u * t)) & 0xFu];
        }
    }
    accg = simd_sum(accg);
    accu = simd_sum(accu);
    if (lane == 0u) {
        DT g = (DT)(accg * wscale);
        DT u = (DT)(accu * wscale);
        float gf = (float)g;
        float uf = (float)u;
        float og = tanh(clamp(gf / gc[0], -10.0f, 10.0f)) * gc[0];
        float ou = tanh(clamp(uf / uc[0], -10.0f, 10.0f)) * uc[0];
        // See the swiglu kernel: clamp before the exponential, because
        // metal::exp(-x) is NaN for large x under fast math.
        act[(size_t)slot * INTER + i] =
            (DT)(og * (1.0f / (1.0f + metal::exp(-clamp(gf, -88.0f, 88.0f)))) * ou);
    }
"""

_DOWN_SRC = r"""
    // One threadgroup covers 4 output rows x TOPK slots (one warp per pair),
    // so every expert's weight rows are read in parallel. Warps of one row
    // reduce through threadgroup memory and apply the score-weighted sum.
    uint lane = thread_index_in_simdgroup;
    uint wid = thread_index_in_threadgroup / 32u;
    uint row_off = wid / TOPK;
    uint slot = wid - row_off * TOPK;
    uint row = threadgroup_position_in_grid.x * 4u + row_off;
    float acc = 0.0f;
    if (row < HID) {
        uint e = idx[slot];
        size_t base = ((size_t)e * HID + row) * WORDS;
        size_t sbase = ((size_t)e * HID + row) * GROUPS;
        const device DT* arow = act + (size_t)slot * INTER;
        for (uint u = lane; u < WORDS; u += 32u) {
            float s = NVFP4_E4M3(scales[sbase + (u >> 1)]);
            uint wv = w2[base + u];
            uint ab = u * 8u;
            for (uint t = 0; t < 8u; ++t) {
                acc += ((float)arow[ab + t] * s) * lut[(wv >> (4u * t)) & 0xFu];
            }
        }
        acc = simd_sum(acc) * (wts[slot] * s2[slot]);
    }
    threadgroup float partial[4 * TOPK];
    if (lane == 0u) partial[wid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (wid < 4u && lane == 0u) {
        uint out_row = threadgroup_position_in_grid.x * 4u + wid;
        if (out_row < HID) {
            float total = 0.0f;
            for (uint t = 0u; t < TOPK; ++t) total += partial[wid * TOPK + t];
            out[out_row] = (DT)total;
        }
    }
"""


def moe_gate_up(x, experts, flat_idx, top_k):
    """Routed gate+up with the softcapped SwiGLU epilogue, one launch."""
    w13 = experts.w13
    hidden = w13.weight.shape[2] * 8
    inter = experts.w2.weight.shape[2] * 8
    blocks = (inter + 3) // 4
    kernel = _kernel(
        "omlx_inkling2_nvfp4_gate_up",
        _NVFP4_INLINE + _GATE_UP_SRC,
        ["x", "w13", "scales", "idx", "s13", "gc", "uc", "lut"],
        ["act"],
    )
    return kernel(
        inputs=[
            x.reshape(-1),
            w13.weight,
            w13.scales,
            flat_idx.astype(mx.uint32),
            w13.post_scale(flat_idx).astype(mx.float32),
            mx.array([float(experts._gcap.item())], dtype=mx.float32),
            mx.array([float(experts._ucap.item())], dtype=mx.float32),
            _E2M1_LUT,
        ],
        template=[
            ("DT", x.dtype),
            ("INTER", inter),
            ("TOPK", top_k),
            ("WORDS", hidden // 8),
            ("GROUPS", hidden // 16),
            ("IBLOCKS", blocks),
        ],
        grid=(top_k * blocks * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(top_k, inter)],
        output_dtypes=[x.dtype],
    )[0]


def moe_down(act, experts, flat_idx, weights):
    """Routed down projection + score-weighted combine, one launch."""
    w2 = experts.w2
    hidden = w2.weight.shape[1]
    inter = w2.weight.shape[2] * 8
    top_k = flat_idx.shape[0]
    kernel = _kernel(
        "omlx_inkling2_nvfp4_down",
        _NVFP4_INLINE + _DOWN_SRC,
        ["act", "w2", "scales", "idx", "wts", "s2", "lut"],
        ["out"],
    )
    return kernel(
        inputs=[
            act.reshape(-1),
            w2.weight,
            w2.scales,
            flat_idx.astype(mx.uint32),
            weights.reshape(-1).astype(mx.float32),
            w2.post_scale(flat_idx).astype(mx.float32),
            _E2M1_LUT,
        ],
        template=[
            ("DT", act.dtype),
            ("HID", hidden),
            ("INTER", inter),
            ("TOPK", top_k),
            ("WORDS", inter // 8),
            ("GROUPS", inter // 16),
        ],
        grid=(((hidden + 3) // 4) * (4 * top_k * 32), 1, 1),
        threadgroup=(4 * top_k * 32, 1, 1),
        output_shapes=[(hidden,)],
        output_dtypes=[act.dtype],
    )[0]


def moe_decode_nvfp4(x, experts, inds, weights):
    """One-token NVFP4 routed experts: gate+up+SwiGLU, then down+combine."""
    flat_idx = inds.reshape(-1)
    act = moe_gate_up(x, experts, flat_idx, inds.shape[1])
    out = moe_down(act, experts, flat_idx, weights)
    return out.reshape(1, -1)


def nvfp4_decode_eligible(experts) -> bool:
    w13, w2 = experts.w13, experts.w2
    return (
        w13.mode == "nvfp4"
        and w13.bits == 4
        and w13.group_size == 16
        and w2.mode == "nvfp4"
        and w2.bits == 4
        and w2.group_size == 16
    )
