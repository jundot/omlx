"""Model-neutral affine qmm routing for GLM-5.3 projections."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _native_qmm(linear: nn.QuantizedLinear, x: mx.array):
    bits = int(getattr(linear, "bits", 0) or 0)
    group_size = int(getattr(linear, "group_size", 0) or 0)
    if bits not in (2, 4, 5, 6, 8) or group_size not in (64, 128):
        return None
    if getattr(linear, "mode", None) != "affine" or "bias" in linear:
        return None
    min_tokens = 1024 if bits == 8 else 128
    if x.ndim < 2 or x.shape[-2] < min_tokens:
        return None

    try:
        from omlx.patches.qwen35_q4_mlp import _is_supported_affine_linear
        from omlx.custom_kernels.qwen35_prefill import fast

        name = f"qwen35_q{bits}_affine_qmm_t"
        if not fast.has_symbol(name) or not _is_supported_affine_linear(linear, x):
            return None
        return getattr(fast, name)(
            x,
            linear.weight,
            linear.scales,
            linear.biases,
            8,
            group_size,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def linear_forward(linear: nn.Module, x: mx.array) -> mx.array:
    """Route GLM-5.3 projections through standard mlx quantized matmul.

    The oMLX native qwen35 affine-qmm tile (``qwen35_q{bits}_affine_qmm_t``)
    returns incorrect results for glm5_next 8-bit affine projections once the
    token count reaches its 1024-token threshold, corrupting long-context
    attention and breaking head/middle retrieval (jundot/omlx#3248). Standard
    mlx quantized matmul is correct at every length, so use it unconditionally.
    """
    return linear(x)


def fused_quantized_matmul(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    bits: int,
    group_size: int,
) -> mx.array:
    """Route a concatenated projection through the same native affine tile."""
    min_tokens = 1024 if bits == 8 else 128
    if (
        x.ndim >= 2
        and x.shape[-2] >= min_tokens
        and bits in (2, 4, 5, 6, 8)
        and group_size in (64, 128)
        and weight.ndim == scales.ndim == biases.ndim == 2
        and weight.dtype == mx.uint32
        and scales.dtype == x.dtype
        and biases.dtype == x.dtype
        and weight.shape[0] % 64 == 0
        and x.shape[-1] % 64 == 0
    ):
        try:
            from omlx.custom_kernels.qwen35_prefill import fast

            name = f"qwen35_q{bits}_affine_qmm_t"
            if fast.has_symbol(name) and fast.qmm_supports_group_size(group_size):
                return getattr(fast, name)(
                    x,
                    weight,
                    scales,
                    biases,
                    8,
                    group_size,
                )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )
