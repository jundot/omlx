# SPDX-License-Identifier: MIT
"""Combine sorted expert outputs without a full unpermuted intermediate."""

from functools import cache

import mlx.core as mx

_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= ROWS * D) return;
    const uint row = i / D, d = i % D;
    float sum = 0.0f;
    for (uint j = 0; j < K; j++) {
        sum += float(x[size_t(order[row * K + j]) * D + d]);
    }
    y[i] = T(sum + float(shared[i]));
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="v41_unpermute_reduce",
        input_names=["x", "order", "shared"],
        output_names=["y"],
        source=_SOURCE,
    )


def combine_sorted_experts(routed, inverse, shared):
    """Return the fused six-expert BF16 sum, or None for other layouts."""
    if (
        mx.default_device() != mx.gpu
        or shared.ndim != 3
        or not shared.size
        or routed.dtype != mx.bfloat16
        or shared.dtype != mx.bfloat16
        or inverse.ndim != 1
        or inverse.dtype != mx.uint32
        or routed.shape != (inverse.size, 1, shared.shape[-1])
        or inverse.size != 6 * shared.shape[0] * shared.shape[1]
    ):
        return None
    width = shared.shape[-1]
    rows = shared.size // width
    return _kernel()(
        inputs=[routed, inverse, shared],
        template=[("T", shared.dtype), ("ROWS", rows), ("D", width), ("K", 6)],
        grid=(shared.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shared.shape],
        output_dtypes=[shared.dtype],
    )[0]
