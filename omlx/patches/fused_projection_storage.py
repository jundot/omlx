# SPDX-License-Identifier: Apache-2.0
"""Share quantized projection operands with their fused decode cache."""

from __future__ import annotations

import functools

import mlx.core as mx


def share_fused_projection_storage(linears):
    """Replace source operands with row views of an existing fused bank.

    Quantization groups run along input columns, so splitting output rows
    preserves the packed values and scale/bias groups. Materialize the views
    before rebuilding the identity key used by both projection consumers.
    """
    first = linears[0]
    cached = first._fused_decode_linears
    if cached is getattr(first, "_omlx_shared_fused_decode_linears", None):
        return cached
    _, weights, scales, biases, split_indices = cached
    bounds = [0, *split_indices, weights.shape[0]]
    for linear, lo, hi in zip(linears, bounds, bounds[1:]):
        linear.weight = weights[lo:hi]
        linear.scales = scales[lo:hi]
        linear.biases = biases[lo:hi]
    mx.eval([(item.weight, item.scales, item.biases) for item in linears])
    key = tuple((id(item.weight), id(item.scales), id(item.biases)) for item in linears)
    shared = (key, weights, scales, biases, split_indices)
    first._fused_decode_linears = shared
    first._omlx_shared_fused_decode_linears = shared
    return shared


def apply_fused_projection_storage_patch() -> bool:
    """Compact mlx-vlm's shared cache after it builds eligible operands."""
    from mlx_vlm.speculative.ops import linear as ops

    original = ops._decode_quantized_linears_fused
    if getattr(original, "_omlx_shared_projection_storage", False):
        return False

    @functools.wraps(original)
    def fused(linears, x):
        result = original(linears, x)
        if result is not None:
            share_fused_projection_storage(linears)
        return result

    fused._omlx_shared_projection_storage = True
    ops._decode_quantized_linears_fused = fused
    return True
