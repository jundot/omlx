# SPDX-License-Identifier: Apache-2.0
"""Exact fused decode kernels for MiniMax-M3.1-Flash (minimax_m3_vl).

A one-token decode step is a long chain of dependent launches, many of them
small. The kernels here fuse chains of them while reproducing the replaced
arithmetic bit for bit:

* ``rms_norm``: ``MiniMaxRMSNorm`` for calls of up to eight rows. MLX runs a
  BF16 input against the FP32 ``1 + w`` as cast, FP32 norm and cast; this is
  one launch with rms_single_row / rms_looped's reduction order.

Exactness rules: every reduction replays the order of the kernel it replaces,
every intermediate is rounded where the replaced code materializes it, and a
product is never contracted into an add that consumed it in a different
kernel (separate statements). ``tests/test_minimax_m3_sparse_attention_patch.py``
checks each kernel bitwise against the ops it replaces.
"""

from __future__ import annotations

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
