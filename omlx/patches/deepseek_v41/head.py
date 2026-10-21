# SPDX-License-Identifier: MIT
"""Vocabulary projection with BF16 storage and FP32 accumulation/output."""

from functools import cache

import mlx.core as mx

# Each output row: KL lanes stride the K dimension four elements at a time and reduce with
# simd_shuffle_down. All NQ query rows share each weight load (the weight matrix is read once
# per call); every query keeps that same per-row accumulation order, so a row's logits do
# not depend on how many rows are projected with it.
_SOURCE = r"""
    const uint row = thread_position_in_grid.x / KL;
    if (row >= N) return;
    const uint lane = thread_index_in_simdgroup % KL;
    float sum[NQ];
    for (int q = 0; q < NQ; ++q) sum[q] = 0;
    for (uint k = lane * 4; k < K; k += KL * 4) {
        const size_t offset = size_t(row) * K + k;
        const float4 a = float4(w[offset], w[offset + 1],
                                w[offset + 2], w[offset + 3]);
        for (int q = 0; q < NQ; ++q) {
            const float4 b = float4(x[q * K + k], x[q * K + k + 1], x[q * K + k + 2], x[q * K + k + 3]);
            sum[q] += dot(a, b);
        }
    }
    for (uint offset = KL / 2; offset > 0; offset /= 2) {
        for (int q = 0; q < NQ; ++q) sum[q] += simd_shuffle_down(sum[q], offset);
    }
    if (lane == 0) for (int q = 0; q < NQ; ++q) y[q * N + row] = sum[q];
"""
MAX_ROWS = 8


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="v41_bf16_head_fp32_output",
        input_names=["x", "w"],
        output_names=["y"],
        source=_SOURCE,
    )


def project_logits(x, weight):
    """Keep the full logits contract for prefill and multi-token verification."""
    rows, width = weight.shape
    if (
        weight.dtype != mx.bfloat16
        or x.shape[-1] != width
        or not 1 <= x.size // width <= MAX_ROWS
        or width % 4
        or width < 256
        or rows < 4096
        or mx.default_device() == mx.cpu
    ):
        return x.astype(mx.float32) @ weight.astype(mx.float32).T
    lanes = 16
    return _kernel()(
        inputs=[x.astype(mx.float32), weight],
        template=[("N", rows), ("K", width), ("KL", lanes), ("NQ", x.size // width)],
        grid=(((rows * lanes + 63) // 64) * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], rows)],
        output_dtypes=[mx.float32],
    )[0]
