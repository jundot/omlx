# SPDX-License-Identifier: Apache-2.0
"""Exact fused decode kernels for MiniMax-M3.1-Flash (minimax_m3_vl).

A one-token decode step is a long chain of dependent launches, many of them
small. The kernels here fuse chains of them while reproducing the replaced
arithmetic bit for bit:

* ``rms_norm``: ``MiniMaxRMSNorm`` for calls of up to eight rows. MLX runs a
  BF16 input against the FP32 ``1 + w`` as cast, FP32 norm and cast; this is
  one launch with rms_single_row / rms_looped's reduction order.
* ``head_norm_rope``: the attention's q, k, index-q and index-k head norms in
  FP32, ``mx.fast.rope`` on the FP32 result and the cast to the model dtype,
  for every head of up to eight rows in one launch instead of a norm, a RoPE
  (a copy and a rotation when only part of the head rotates) and a cast per
  tensor. Each head is a simdgroup with ``rms_norm``'s single-row arithmetic;
  RoPE keeps rope.metal's: inv_freq = exp2(-(j / (dims / 2)) * log2(base))
  with log2(base) rounded as MLX's host code does (log2f), fast::cos /
  fast::sin, and the position as rope_single (signed offset) or rope
  (unsigned sum) converts it.
* ``moe_router_logits`` and ``moe_router_select``: the MoE router's FP32
  logits (MLX's one-row gemv for every row) and the expert selection after
  them (sigmoid, correction bias, the stable argpartition order, the FP32
  normalization with its 1e-20, the routed scaling and the down-scale fold)
  in two launches instead of ten.
* ``moe_experts``: one token's routed NVFP4 experts and MXFP8 shared expert
  in two launches instead of eleven, the weight rows of both read in the same
  dispatch: fp4_expert_rows' and MLX's MXFP8 row arithmetic, moe_rows'
  scaled SwiGLU-OAI epilogue, ``_combine_routed`` and the shared add, each
  value rounded where the replaced kernels round it.

Exactness rules: every reduction replays the order of the kernel it replaces,
every intermediate is rounded where the replaced code materializes it, and a
product is never contracted into an add that consumed it in a different
kernel (separate statements). ``tests/test_minimax_m3_sparse_attention_patch.py``
checks each kernel bitwise against the ops it replaces.
"""

from __future__ import annotations

import ctypes
import struct
from functools import cache

import mlx.core as mx

MAX_ROWS = 8
# Rows per simdgroup of the one-token expert launches (fastest on an M3 Ultra).
_GATE_UP_RPS = 2
_DOWN_RPS = 1

_RMS_N_READS = 4  # MLX RMS_N_READS
_RMS_LOOPED_LIMIT = 4096  # MLX RMS_LOOPED_LIMIT
_RMS_LOOPED_THREADS = 1024  # rms_looped threadgroup (maxTotalThreadsPerThreadgroup)


def _float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _rows(x: mx.array) -> int:
    return x.size // x.shape[-1] if x.ndim else 0


def _rms_threads(dims: int) -> int:
    if dims > _RMS_LOOPED_LIMIT:
        return _RMS_LOOPED_THREADS
    needed = (dims + _RMS_N_READS - 1) // _RMS_N_READS
    return 32 * ((needed + 31) // 32)


# Sum of squares and reduction of MLX rms_single_row / rms_looped on FP32
# data; TG threads per row, D values, x read as T and widened (exact).
_RMS_BODY = r"""
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    float acc = 0;
    for (uint r = 0; r < D; r += TG * 4) {
        if (r + lid * 4 + 4 <= D) {
            for (int i = 0; i < 4; i++) {
                float xi = static_cast<float>(xr[r + lid * 4 + i]);
                acc += xi * xi;
            }
        } else {
            for (int i = 0; i < 4; i++) {
                if ((r + lid * 4 + i) < D) {
                    float xi = static_cast<float>(xr[r + lid * 4 + i]);
                    acc += xi * xi;
                }
            }
        }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_RMS_SOURCE = (
    r"""
    const uint lid = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const float eps = as_type<float>(uint(EPS_BITS));
    const device T* xr = x + size_t(row) * D;
    device OutT* yr = y + size_t(row) * D;
"""
    + _RMS_BODY
    + r"""
    for (uint r = 0; r < D; r += TG * 4) {
        for (int i = 0; i < 4; i++) {
            const uint c = r + lid * 4 + i;
            if (c < D) {
                const float o = scale[c] * (static_cast<float>(xr[c]) * local_inv_mean[0]);
                yr[c] = static_cast<OutT>(o);
            }
        }
    }
"""
)


@cache
def _rms_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_rms_norm",
        input_names=["x", "scale"],
        output_names=["y"],
        source=_RMS_SOURCE,
    )


def rms_norm(x: mx.array, scale: mx.array, eps: float, out_dtype) -> mx.array | None:
    """``mx.fast.rms_norm(x, scale, eps).astype(out_dtype)`` in one launch for
    a half-precision ``x`` of up to eight rows and an FP32 ``scale``, or None
    when not covered."""
    dims = x.shape[-1]
    if not (
        x.dtype in (mx.bfloat16, mx.float16)
        and out_dtype in (x.dtype, mx.float32)
        and scale.dtype == mx.float32
        and scale.shape == (dims,)
        and dims % _RMS_N_READS == 0
        and 1 <= _rows(x) <= MAX_ROWS
        and mx.metal.is_available()
    ):
        return None
    threads = _rms_threads(dims)
    return _rms_kernel()(
        inputs=[x, scale],
        template=[
            ("T", x.dtype),
            ("OutT", out_dtype),
            ("D", dims),
            ("TG", threads),
            ("EPS_BITS", _float_bits(eps)),
        ],
        grid=(threads * _rows(x), 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[out_dtype],
    )[0]


@cache
def _log2f(value: float) -> float:
    """``std::log2`` of a float, as MLX's RoPE takes it of ``base`` on the host."""
    libm = ctypes.CDLL(None)
    libm.log2f.restype = ctypes.c_float
    libm.log2f.argtypes = [ctypes.c_float]
    return libm.log2f(value)


# One simdgroup per head of HD = 128 values (4 per lane): the norm of
# rms_single_row in FP32, then rope.metal on the first ROPE values of the FP32
# result (j and j + ROPE / 2 exchanged across lanes), values past ROPE kept,
# and one rounding to T. Threadgroup t is head t % H{n} of row t / H{n}
# (row = b * LROWS + l); tensor i holds heads [H{i}, H{i+1}), read as
# [B, L, heads, HD] and written as [B, heads, L, HD] like mx.fast.rope.
def _head_norm_rope_source(ntensors: int) -> str:
    pick = "\n".join(
        f"    {'if' if i == 0 else 'else if'} (head < H{i + 1}) {{\n"
        f"        const uint n = H{i + 1} - H{i}, h = head - H{i};\n"
        f"        xr = x{i} + (size_t(row) * n + h) * HD;\n"
        f"        yr = y{i} + ((size_t(b) * n + h) * LROWS + l) * HD;\n"
        f"        sc = scale{i};\n"
        f"    }}"
        for i in range(ntensors)
    )
    return (
        f"""
    const uint lid = thread_index_in_simdgroup;
    const uint simd_lane_id = lid;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x / H{ntensors};
    const uint head = threadgroup_position_in_grid.x - row * H{ntensors};
    const uint b = row / LROWS;
    const uint l = row % LROWS;
    const float eps = as_type<float>(uint(EPS_BITS));
    const device T* xr;
    device T* yr;
    const device float* sc;
{pick}
    else return;
    constexpr uint D = HD;
    constexpr uint TG = 32;
"""
        + _RMS_BODY
        + r"""
    float v[4];
    for (int i = 0; i < 4; i++) {
        const uint c = lid * 4 + i;
        v[i] = sc[c] * (static_cast<float>(xr[c]) * local_inv_mean[0]);
    }
    constexpr uint HALF = ROPE / 2;
    constexpr uint PARTNER = HALF / 4;  // lane ^ PARTNER holds value j +- HALF
    float other[4];
    for (int i = 0; i < 4; i++) {
        other[i] = simd_shuffle_xor(v[i], PARTNER);
    }
    if (lid * 4 < ROPE) {
        // rope_single converts the int offset; rope the uint sum pos.y + offset.
        const int start = offset[b * OFFSET_STRIDE];
        const float position =
            SINGLE ? static_cast<float>(start) : static_cast<float>(l + start);
        const float L = as_type<float>(uint(SCALE_BITS)) * position;
        const float base = as_type<float>(uint(BASE_BITS));
        const bool first = lid * 4 < HALF;
        for (int i = 0; i < 4; i++) {
            const uint j = (lid * 4 + i) % HALF;
            const float d = static_cast<float>(j) / static_cast<float>(HALF);
            const float inv_freq = metal::exp2(-d * base);
            const float theta = L * inv_freq;
            const float costheta = metal::fast::cos(theta);
            const float sintheta = metal::fast::sin(theta);
            const float x1 = first ? v[i] : other[i];
            const float x2 = first ? other[i] : v[i];
            v[i] = first ? (x1 * costheta - x2 * sintheta) : (x1 * sintheta + x2 * costheta);
        }
    }
    for (int i = 0; i < 4; i++) {
        yr[lid * 4 + i] = static_cast<T>(v[i]);
    }
"""
    )


@cache
def _head_norm_rope_kernel(ntensors: int):
    return mx.fast.metal_kernel(
        name=f"minimax_m3_head_norm_rope_{ntensors}",
        input_names=[f"x{i}" for i in range(ntensors)]
        + [f"scale{i}" for i in range(ntensors)]
        + ["offset"],
        output_names=[f"y{i}" for i in range(ntensors)],
        source=_head_norm_rope_source(ntensors),
    )


def head_norm_rope(
    xs, scales, eps: float, offset, dims: int, base: float, scale: float
) -> list | None:
    """For each ``[B, L, heads, 128]`` tensor ``x`` of ``xs`` and its FP32 norm
    scale, in one launch::

        y = mx.fast.rms_norm(x, scale, eps).astype(mx.float32)
        mx.fast.rope(y.transpose(0, 2, 1, 3), dims, traditional=False,
                     base=base, scale=scale, offset=offset).astype(x.dtype)

    for up to eight rows (B * L), or None when not covered.
    """
    x = xs[0]
    if x.ndim != 4 or x.dtype not in (mx.bfloat16, mx.float16):
        return None
    batch, length, _, head_dim = x.shape
    if isinstance(offset, mx.array) and (
        offset.ndim > 1
        or offset.size not in (1, batch)
        or not mx.issubdtype(offset.dtype, mx.integer)
    ):
        return None
    if not (
        head_dim == 128
        and 1 <= batch * length <= MAX_ROWS
        and dims in (8, 16, 32, 64, 128)  # the RoPE partner is a lane xor
        and all(
            y.ndim == 4
            and y.shape[:2] == (batch, length)
            and y.shape[-1] == head_dim
            and y.dtype == x.dtype
            for y in xs
        )
        and all(s.dtype == mx.float32 and s.shape == (head_dim,) for s in scales)
        and mx.metal.is_available()
    ):
        return None
    # A 0-d kernel input is passed by value; the kernel indexes a buffer.
    if not isinstance(offset, mx.array):
        offset = mx.array([offset], dtype=mx.int32)
    elif offset.ndim == 0 or offset.dtype != mx.int32:
        offset = offset.reshape(-1).astype(mx.int32)
    starts = [0]
    for y in xs:
        starts.append(starts[-1] + y.shape[2])
    template = [
        ("T", x.dtype),
        ("HD", head_dim),
        ("ROPE", dims),
        ("LROWS", length),
        ("OFFSET_STRIDE", int(offset.size > 1)),
        ("SINGLE", int(length == 1 and offset.size == 1)),
        ("EPS_BITS", _float_bits(eps)),
        ("BASE_BITS", _float_bits(_log2f(base))),
        ("SCALE_BITS", _float_bits(scale)),
    ] + [(f"H{i}", s) for i, s in enumerate(starts)]
    return _head_norm_rope_kernel(len(xs))(
        inputs=[*xs, *scales, offset],
        template=template,
        grid=(32 * starts[-1] * batch * length, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, y.shape[2], length, head_dim) for y in xs],
        output_dtypes=[x.dtype] * len(xs),
    )


# Router logits: MLX's gemv for an FP32 matrix with K >= 16 * E (GEMVKernel
# BM 1, BN 8, SM 1, SN 32, TM 4, TN 4). Threadgroup t computes the four
# logits [4 t, 4 t + 4) of row t / (E / 4): simdgroup s takes 4 values per lane
# at 128 s + 4 lane of every 1024-value strip of K, the lanes are reduced with
# a shuffle-down tree, then simdgroups 1..7 are added to simdgroup 0's sums in
# order. x is read as T and widened (exact).
_ROUTER_LOGITS_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sgn = simdgroup_index_in_threadgroup;
    const int tg = int(threadgroup_position_in_grid.x);
    const int r = tg / (E / 4);
    const int out_row = (tg - r * (E / 4)) * 4;
    threadgroup float tgp[8 * 4];
    float result[4] = {0, 0, 0, 0};
    int bn = (32 * int(sgn) + int(lane)) * 4;
    const device float* mat = w + size_t(out_row) * K;
    const device T* xr = x + size_t(r) * K;
    for (int i = 0; i < K / 1024; i++) {
        float v_coeff[4];
        for (int tn = 0; tn < 4; tn++) {
            v_coeff[tn] = static_cast<float>(xr[bn + tn]);
        }
        for (int tm = 0; tm < 4; tm++) {
            float inter[4];
            for (int tn = 0; tn < 4; tn++) {
                inter[tn] = mat[tm * K + bn + tn];
            }
            for (int tn = 0; tn < 4; tn++) {
                result[tm] += inter[tn] * v_coeff[tn];
            }
        }
        bn += 1024;
    }
    for (int tm = 0; tm < 4; tm++) {
        for (ushort sn = 16; sn >= 1; sn >>= 1) {
            result[tm] += simd_shuffle_down(result[tm], sn);
        }
    }
    if (lane == 0) {
        for (int tm = 0; tm < 4; tm++) {
            tgp[sgn * 4 + tm] = result[tm];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgn == 0 && lane == 0) {
        for (int s = 1; s < 8; s++) {
            for (int tm = 0; tm < 4; tm++) {
                result[tm] += tgp[s * 4 + tm];
            }
        }
        for (int tm = 0; tm < 4; tm++) {
            logits[r * E + out_row + tm] = result[tm];
        }
    }
"""

# _minimax_moe_select for sigmoid scores, then the down-scale fold; one
# threadgroup per row, one thread per expert. Sigmoid is MLX's functor; an
# expert's rank is its place in MLX's stable argpartition order (biased score
# descending, ties to the lower index); the first TOPK experts' scores are
# summed in MLX's order (total = value + total), + 1e-20, divided, scaled and
# times the expert's down scale, each step rounded in FP32 like the ops.
_ROUTER_SELECT_SOURCE = r"""
    const uint e = thread_position_in_threadgroup.x;
    const uint r = threadgroup_position_in_grid.x;
    threadgroup float score[E];
    threadgroup float biased[E];
    threadgroup uint chosen[TOPK];
    const float g = logits[r * E + e];
    const float y = 1 / (1 + metal::precise::exp(metal::abs(g)));
    const float s = (g < 0) ? y : 1 - y;
    const float b = s + bias[e];
    score[e] = s;
    biased[e] = b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = 0;
    for (uint i = 0; i < E; i++) {
        const float bi = biased[i];
        rank += (bi > b || (bi == b && i < e)) ? 1 : 0;
    }
    if (rank < TOPK) {
        chosen[rank] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
        float w[TOPK];
        float total = 0;
        for (int t = 0; t < TOPK; t++) {
            w[t] = score[chosen[t]];
            total = w[t] + total;
        }
        const float denom = total + 1e-20f;
        const float scaling = as_type<float>(uint(SCALING_BITS));
        for (int t = 0; t < TOPK; t++) {
            const float normed = w[t] / denom;
            const float scaled = normed * scaling;
            inds[r * TOPK + t] = chosen[t];
            weights[r * TOPK + t] = DOWN_SCALE ? scaled * down_scale[chosen[t]] : scaled;
        }
    }
"""


@cache
def _router_logits_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_moe_router_logits",
        input_names=["x", "w"],
        output_names=["logits"],
        source=_ROUTER_LOGITS_SOURCE,
    )


@cache
def _router_select_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_moe_router_select",
        input_names=["logits", "bias", "down_scale"],
        output_names=["inds", "weights"],
        source=_ROUTER_SELECT_SOURCE,
    )


def moe_router_logits(x: mx.array, weight: mx.array) -> mx.array | None:
    """``x.astype(mx.float32) @ weight.T`` for an FP32 router ``weight`` and up
    to eight rows, each row as MLX's one-row matrix-vector product computes
    it, or None when not covered."""
    hidden = x.shape[-1]
    rows = _rows(x)
    experts = weight.shape[0]
    if not (
        weight.ndim == 2
        and weight.dtype == mx.float32
        and weight.shape[1] == hidden
        and x.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and hidden % 1024 == 0
        and hidden >= 16 * experts  # MLX's gemv uses BM 1, BN 8 here
        and experts >= 4
        and experts % 4 == 0
        and 1 <= rows <= MAX_ROWS
        and mx.metal.is_available()
    ):
        return None
    return _router_logits_kernel()(
        inputs=[x, weight],
        template=[("T", x.dtype), ("K", hidden), ("E", experts)],
        grid=(256 * rows * (experts // 4), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*x.shape[:-1], experts)],
        output_dtypes=[mx.float32],
    )[0]


def moe_router_select(
    logits: mx.array,
    bias: mx.array,
    top_k: int,
    scaling: float,
    down_scale: mx.array | None = None,
) -> tuple[mx.array, mx.array] | None:
    """``_minimax_moe_select(logits, bias, top_k, scaling, "sigmoid")`` and,
    with ``down_scale``, the weights times ``down_scale[inds]``, in one launch
    for up to eight rows of FP32 logits, or None when not covered."""
    experts = logits.shape[-1]
    rows = _rows(logits)
    if not (
        logits.dtype == mx.float32
        and bias.dtype == mx.float32
        and bias.shape == (experts,)
        and (
            down_scale is None
            or (down_scale.dtype == mx.float32 and down_scale.shape == (experts,))
        )
        and 1 <= top_k <= 16
        and top_k <= experts <= 1024
        and 1 <= rows <= MAX_ROWS
        and mx.metal.is_available()
    ):
        return None
    shape = (*logits.shape[:-1], top_k)
    return tuple(
        _router_select_kernel()(
            inputs=[logits, bias, bias if down_scale is None else down_scale],
            template=[
                ("E", experts),
                ("TOPK", top_k),
                ("SCALING_BITS", _float_bits(scaling)),
                ("DOWN_SCALE", down_scale is not None),
            ],
            grid=(experts * rows, 1, 1),
            threadgroup=(experts, 1, 1),
            output_shapes=[shape, shape],
            output_dtypes=[mx.uint32, mx.float32],
        )
    )


# One token's rows of MLX fp_qmv_fast. NVFP4 as fp4_expert_rows reads them:
# lane l takes values [512 b + 16 l, +16) of block b, fp4_qdot16 per block.
# MXFP8 as MLX's qdot: lane l takes [256 b + 8 l, +8), an 8-term dot in order
# times its 32-value group's E8M0 scale. An E4M3 code's bits placed in a
# half's sign, exponent and mantissa give code * 2^-8 (two codes per mask and
# shift); widened and times 256 that is exactly the value MLX decodes.
_EXPERTS_HEADER = r"""
inline void mm3_fp8x4(uint p, thread float* v) {
    const float2 a = float2(as_type<half2>(((p & 0x00800080u) << 8) | ((p & 0x007F007Fu) << 7)));
    const uint q = p >> 8;
    const float2 b = float2(as_type<half2>(((q & 0x00800080u) << 8) | ((q & 0x007F007Fu) << 7)));
    v[0] = a.x * 256.0f;
    v[1] = b.x * 256.0f;
    v[2] = a.y * 256.0f;
    v[3] = b.y * 256.0f;
}

inline float mm3_mxfp8_dot8(uint2 w, thread const float* x, float scale) {
    float v[8];
    mm3_fp8x4(w.x, v);
    mm3_fp8x4(w.y, v + 4);
    float accum = 0;
    for (int i = 0; i < 8; i++) {
        accum += x[i] * v[i];
    }
    return scale * accum;
}

template <typename T>
inline T mm3_scaled_swiglu_oai(float g, float u, float gate_scale, float up_scale,
                               float alpha, float limit, float beta) {
    float x_glu = float(static_cast<T>(g)) * gate_scale;
    x_glu = fp4_minimum(x_glu, limit);
    float x_linear = float(static_cast<T>(u)) * up_scale;
    x_linear = fp4_minimum(fp4_maximum(x_linear, -limit), limit);
    const float sig = fp4_sigmoid(alpha * x_glu);
    return static_cast<T>(x_glu * sig * (x_linear + beta));
}
"""

# Simdgroup s computes the gate and up rows [j, j + RPS) of slot t: slots
# 0..TOPK-1 are the routed experts (NVFP4, with their gate and up scales),
# slot TOPK is the shared expert (MXFP8, scale 1). Each row keeps its own
# accumulator, updated block by block in order.
_GATE_UP_SWIGLU_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const int s = int(threadgroup_position_in_grid.y) * 2 + int(simdgroup_index_in_threadgroup);
    const int t = s / (I / RPS);
    const int j = (s - t * (I / RPS)) * RPS;
    if (t > TOPK) {
        return;
    }
    float g[RPS];
    float u[RPS];
    for (int r = 0; r < RPS; r++) {
        g[r] = 0;
        u[r] = 0;
    }
    float gate_scale_t = 1.0f;
    float up_scale_t = 1.0f;
    if (t < TOPK) {
        const uint e = inds[t];
        const size_t row = size_t(e) * I + j;
        const device uint32_t* gw = gate_w + row * (H / 8) + lane * 2;
        const device uint32_t* uw = up_w + row * (H / 8) + lane * 2;
        const device uint8_t* gs = gate_s + row * (H / 16) + lane;
        const device uint8_t* us = up_s + row * (H / 16) + lane;
        for (int b = 0; b < H / 512; b++) {
            float xt[16];
            for (int i = 0; i < 16; i++) {
                xt[i] = float(x[b * 512 + lane * 16 + i]);
            }
            for (int r = 0; r < RPS; r++) {
                const uint2 wg = *(const device uint2*)(gw + r * (H / 8) + b * 64);
                const uint2 wu = *(const device uint2*)(uw + r * (H / 8) + b * 64);
                g[r] += fp4_qdot16(wg, xt, fp4_scale_e4m3(gs[r * (H / 16) + b * 32]));
                u[r] += fp4_qdot16(wu, xt, fp4_scale_e4m3(us[r * (H / 16) + b * 32]));
            }
        }
        gate_scale_t = gate_scale[e];
        up_scale_t = up_scale[e];
    } else {
        const device uint8_t* gw = (const device uint8_t*)sh_gate_w + size_t(j) * H + lane * 8;
        const device uint8_t* uw = (const device uint8_t*)sh_up_w + size_t(j) * H + lane * 8;
        const device uint8_t* gs = sh_gate_s + size_t(j) * (H / 32) + lane / 4;
        const device uint8_t* us = sh_up_s + size_t(j) * (H / 32) + lane / 4;
        for (int k = 0; k < H; k += 256) {
            float xt[8];
            for (int i = 0; i < 8; i++) {
                xt[i] = float(x[k + lane * 8 + i]);
            }
            for (int r = 0; r < RPS; r++) {
                const uint2 wg = *(const device uint2*)(gw + r * H + k);
                const uint2 wu = *(const device uint2*)(uw + r * H + k);
                g[r] += mm3_mxfp8_dot8(wg, xt, fp4_scale_e8m0(gs[r * (H / 32) + k / 32]));
                u[r] += mm3_mxfp8_dot8(wu, xt, fp4_scale_e8m0(us[r * (H / 32) + k / 32]));
            }
        }
    }
    for (int r = 0; r < RPS; r++) {
        const float gr = simd_sum(g[r]);
        const float ur = simd_sum(u[r]);
        if (lane == 0) {
            act[t * I + j + r] = mm3_scaled_swiglu_oai<T>(
                gr, ur, gate_scale_t, up_scale_t, params[0], params[1], params[2]);
        }
    }
"""

# Simdgroup s computes the output columns [j, j + RPS): every routed expert's
# down row (rounded to T as fp4_expert_rows.down writes it) and the shared
# expert's, then _combine_routed (FP32 products summed over the experts in
# order, rounded) and the add of the rounded shared row.
_DOWN_COMBINE_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const int j = (int(threadgroup_position_in_grid.y) * 2 + int(simdgroup_index_in_threadgroup)) * RPS;
    if (j >= H) {
        return;
    }
    uint e[TOPK];
    for (int t = 0; t < TOPK; t++) {
        e[t] = inds[t];
    }
    float y[TOPK][RPS];
    float sh[RPS];
    for (int r = 0; r < RPS; r++) {
        for (int t = 0; t < TOPK; t++) {
            y[t][r] = 0;
        }
        sh[r] = 0;
    }
    const device uint8_t* sw = (const device uint8_t*)sh_down_w + size_t(j) * I + lane * 8;
    const device uint8_t* ss = sh_down_s + size_t(j) * (I / 32) + lane / 4;
    for (int b = 0; b < I / 512; b++) {
        for (int t = 0; t < TOPK; t++) {
            float at[16];
            for (int i = 0; i < 16; i++) {
                at[i] = float(act[t * I + b * 512 + lane * 16 + i]);
            }
            const size_t row = size_t(e[t]) * H + j;
            for (int r = 0; r < RPS; r++) {
                const uint2 w = *(const device uint2*)(down_w + (row + r) * (I / 8) + lane * 2 + b * 64);
                y[t][r] += fp4_qdot16(w, at, fp4_scale_e4m3(down_s[(row + r) * (I / 16) + lane + b * 32]));
            }
        }
        for (int k = b * 512; k < b * 512 + 512; k += 256) {
            float xt[8];
            for (int i = 0; i < 8; i++) {
                xt[i] = float(act[TOPK * I + k + lane * 8 + i]);
            }
            for (int r = 0; r < RPS; r++) {
                const uint2 w = *(const device uint2*)(sw + r * I + k);
                sh[r] += mm3_mxfp8_dot8(w, xt, fp4_scale_e8m0(ss[r * (I / 32) + k / 32]));
            }
        }
    }
    for (int r = 0; r < RPS; r++) {
        float routed[TOPK];
        for (int t = 0; t < TOPK; t++) {
            routed[t] = simd_sum(y[t][r]);
        }
        const float shared_sum = simd_sum(sh[r]);
        if (lane == 0) {
            float total = 0.0f;
            for (int t = 0; t < TOPK; t++) {
                const float weighted = float(static_cast<T>(routed[t])) * weights[t];
                total = total + weighted;
            }
            const float combined = float(static_cast<T>(total));
            out[j + r] = static_cast<T>(combined + float(static_cast<T>(shared_sum)));
        }
    }
"""


@cache
def _experts_header() -> str:
    from omlx.patches import fp4_expert_rows

    return fp4_expert_rows._HEADER + _EXPERTS_HEADER


@cache
def _gate_up_swiglu_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_moe_gate_up_swiglu",
        input_names=[
            "x",
            "inds",
            "gate_w",
            "gate_s",
            "gate_scale",
            "up_w",
            "up_s",
            "up_scale",
            "sh_gate_w",
            "sh_gate_s",
            "sh_up_w",
            "sh_up_s",
            "params",
        ],
        output_names=["act"],
        header=_experts_header(),
        source=_GATE_UP_SWIGLU_SOURCE,
    )


@cache
def _down_combine_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_moe_down_combine",
        input_names=[
            "act",
            "inds",
            "weights",
            "down_w",
            "down_s",
            "sh_down_w",
            "sh_down_s",
        ],
        output_names=["out"],
        header=_experts_header(),
        source=_DOWN_COMBINE_SOURCE,
    )


@cache
def _swiglu_params(alpha: float, limit: float, beta: float) -> mx.array:
    return mx.array([alpha, limit, beta], dtype=mx.float32)


def _mxfp8(linear, k: int) -> bool:
    return (
        getattr(linear, "mode", None) == "mxfp8"
        and getattr(linear, "bits", None) == 8
        and getattr(linear, "group_size", None) == 32
        and linear.get("biases") is None
        and linear.get("bias") is None
        and linear.weight.dtype == mx.uint32
        and linear.weight.shape[-1] * 4 == k
        and k % 256 == 0
    )


def moe_experts(x, inds, weights, switch, shared) -> mx.array | None:
    """One token's ``_combine_routed(switch(x, inds), weights) + shared(x)`` in
    two launches, or None when not covered.

    ``switch`` is a ``MiniMaxScaledSwitchGLU`` with NVFP4 banks, ``shared`` a
    ``MiniMaxMLP`` with MXFP8 projections, ``weights`` the FP32 routing
    weights with the down scales folded in. Every weight row of both is read
    in the same dispatch: gate, up and the activation in the first, down, the
    combine and the shared add in the second.
    """
    from .moe_rows import _nvfp4

    hidden = x.shape[-1]
    top_k = inds.size
    if shared is None or getattr(switch, "gate_scale", None) is None:
        return None
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    inter = down.weight.shape[-1] * 8
    act_fn, shared_act = switch.activation, shared.act_fn
    params = (act_fn.alpha, act_fn.limit, act_fn.beta)
    if not (
        x.size == hidden
        and x.dtype in (mx.bfloat16, mx.float16)
        and inds.dtype == mx.uint32
        and weights.dtype == mx.float32
        and weights.size == top_k
        # _combine_routed's sum is MLX's col_reduce_small, one expert per
        # thread up to eight, so it adds the experts in order.
        and 1 <= top_k <= 8
        and _nvfp4(gate, hidden)
        and _nvfp4(up, hidden)
        and _nvfp4(down, inter)
        and gate.weight.shape[1] == up.weight.shape[1] == inter
        and down.weight.shape[1] == hidden
        and _mxfp8(shared.gate_proj, hidden)
        and _mxfp8(shared.up_proj, hidden)
        and _mxfp8(shared.down_proj, inter)
        and shared.gate_proj.weight.shape[0] == shared.up_proj.weight.shape[0] == inter
        and shared.down_proj.weight.shape[0] == hidden
        and (shared_act.alpha, shared_act.limit, shared_act.beta) == params
        and mx.metal.is_available()
    ):
        return None
    template = [("T", x.dtype), ("H", hidden), ("I", inter), ("TOPK", top_k)]
    act = _gate_up_swiglu_kernel()(
        inputs=[
            x,
            inds,
            gate.weight,
            gate.scales,
            switch.gate_scale,
            up.weight,
            up.scales,
            switch.up_scale,
            shared.gate_proj.weight,
            shared.gate_proj.scales,
            shared.up_proj.weight,
            shared.up_proj.scales,
            _swiglu_params(*params),
        ],
        template=template + [("RPS", _GATE_UP_RPS)],
        grid=(32, (top_k + 1) * inter // _GATE_UP_RPS, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(top_k + 1, inter)],
        output_dtypes=[x.dtype],
    )[0]
    return _down_combine_kernel()(
        inputs=[
            act,
            inds,
            weights,
            down.weight,
            down.scales,
            shared.down_proj.weight,
            shared.down_proj.scales,
        ],
        template=template + [("RPS", _DOWN_RPS)],
        grid=(32, hidden // _DOWN_RPS, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


# MLX fp_qmv_fast for one MXFP8 row (lane l takes values [256 b + 8 l, +8) of
# block b, an 8-term dot times its group's E8M0 scale, blocks summed in order,
# then simd_sum), RPS output columns per simdgroup and NSG simdgroups per
# threadgroup: the column arithmetic does not depend on the tiling, which
# here keeps more weight loads in flight than MLX's 4 x 2.
_MXFP8_ONE_ROW_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const int n0 = (int(threadgroup_position_in_grid.y) * NSG + int(simdgroup_index_in_threadgroup)) * RPS;
    if (n0 >= N) {
        return;
    }
    const device uint8_t* wl = (const device uint8_t*)w + size_t(n0) * K + lane * 8;
    const device uint8_t* sl = s + size_t(n0) * (K / 32) + lane / 4;
    const device T* xl = x + lane * 8;
    float acc[RPS];
    for (int r = 0; r < RPS; r++) {
        acc[r] = 0;
    }
    for (int k = 0; k < K; k += 256) {
        float xt[8];
        for (int i = 0; i < 8; i++) {
            xt[i] = float(xl[k + i]);
        }
        for (int r = 0; r < RPS; r++) {
            const uint2 wv = *(const device uint2*)(wl + r * K + k);
            acc[r] += mm3_mxfp8_dot8(wv, xt, fp4_scale_e8m0(sl[r * (K / 32) + k / 32]));
        }
    }
    for (int r = 0; r < RPS; r++) {
        const float v = simd_sum(acc[r]);
        if (lane == 0) {
            y[n0 + r] = static_cast<T>(v);
        }
    }
"""


@cache
def _mxfp8_one_row_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_mxfp8_one_row",
        input_names=["x", "w", "s"],
        output_names=["y"],
        header=_experts_header(),
        source=_MXFP8_ONE_ROW_SOURCE,
    )


def mxfp8_one_row(x: mx.array, weight: mx.array, scales: mx.array) -> mx.array | None:
    """``x @ dequantized(weight).T`` for one row of an MXFP8 projection with
    MLX ``fp_qmv_fast``'s arithmetic, or None when not covered."""
    k = x.shape[-1]
    n = weight.shape[0]
    nsg = 8 if n >= 8192 else 4
    if not (
        x.size == k
        and x.dtype in (mx.bfloat16, mx.float16)
        and weight.dtype == mx.uint32
        and weight.shape == (n, k // 4)
        and scales.dtype == mx.uint8
        and scales.shape == (n, k // 32)
        and k % 256 == 0
        and n % (4 * nsg) == 0
        and mx.metal.is_available()
    ):
        return None
    return _mxfp8_one_row_kernel()(
        inputs=[x, weight, scales],
        template=[("T", x.dtype), ("K", k), ("N", n), ("RPS", 4), ("NSG", nsg)],
        grid=(32, n // 4, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]
