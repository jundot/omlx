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
