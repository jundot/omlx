# SPDX-License-Identifier: Apache-2.0
"""Blockwise FP8 weight conversion for Qwen3.8 checkpoints."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

FP8_BLOCK_SIZE = 128


def dequantize_fp8_weights(
    weights: dict[str, Any],
    dtype: Any = mx.bfloat16,
    *,
    copy_weights: bool = True,
) -> dict[str, Any]:
    """Expand Qwen3.8 block-FP8 tensors using their inverse-scale grids."""
    weights = dict(weights) if copy_weights else weights
    scale_suffix = ".weight_scale_inv"

    for scale_key in [key for key in weights if key.endswith(scale_suffix)]:
        weight_key = scale_key.removesuffix("_scale_inv")
        if weight_key not in weights:
            raise ValueError(f"Missing FP8 weight for {scale_key}.")

        weight = weights[weight_key]
        scale = weights.pop(scale_key)
        if weight.ndim != 2:
            raise ValueError(f"Expected a matrix for {weight_key}, got {weight.shape}.")

        rows, columns = weight.shape
        expected_scale_shape = (
            (rows + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE,
            (columns + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE,
        )
        if scale.shape != expected_scale_shape:
            raise ValueError(
                f"Invalid FP8 scale shape for {weight_key}: expected "
                f"{expected_scale_shape}, got {scale.shape}."
            )

        padded_rows = expected_scale_shape[0] * FP8_BLOCK_SIZE
        padded_columns = expected_scale_shape[1] * FP8_BLOCK_SIZE
        decoded = mx.from_fp8(weight, dtype=mx.float32)
        decoded = mx.pad(
            decoded,
            ((0, padded_rows - rows), (0, padded_columns - columns)),
        )
        decoded = decoded.reshape(
            expected_scale_shape[0],
            FP8_BLOCK_SIZE,
            expected_scale_shape[1],
            FP8_BLOCK_SIZE,
        )
        decoded = decoded * scale.astype(mx.float32)[:, None, :, None]
        decoded = decoded.reshape(padded_rows, padded_columns)[:rows, :columns]
        decoded = decoded.astype(dtype)
        mx.eval(decoded)
        weights[weight_key] = decoded

    return weights
