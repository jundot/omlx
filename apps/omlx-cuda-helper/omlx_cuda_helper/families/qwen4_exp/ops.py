# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access and the small ops the Qwen3.8 Flash-Next blocks share."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from ... import checkpoint
from ...tensors import unpack


class Linear:
    """A bias-free dense linear layer.

    The host rounds each output once from a float32 sum. The 16-bit product
    of the helper's MLX build does the same (``tensors.check_backend``).
    """

    def __init__(self, weight: mx.array):
        self.weight = weight

    def parameters(self) -> list[mx.array]:
        return [self.weight]

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.T


class Weights:
    """Lazy checkpoint tensors under one name prefix, with their quantization."""

    def __init__(self, arrays: dict[str, mx.array], config: dict[str, Any], root: str):
        self.arrays, self.config, self.root = arrays, config, root

    def at(self, path: str) -> Weights:
        return Weights(self.arrays, self.config, f"{self.root}{path}.")

    def __contains__(self, name: str) -> bool:
        return self.root + name in self.arrays

    def __getitem__(self, name: str) -> mx.array:
        return self.arrays[self.root + name]

    def quantization(self, name: str) -> dict[str, Any] | None:
        return checkpoint.quantization(self.config, self.root + name)

    def float32(self, name: str) -> mx.array:
        """A linear weight in float32; a quantized one is not rounded."""
        weight = self[f"{name}.weight"]
        if f"{name}.scales" in self:
            weight = unpack(
                weight,
                self[f"{name}.scales"],
                self[f"{name}.biases"],
                self.quantization(name),
                mx.float32,
            )
        return weight.astype(mx.float32)

    def linear(self, name: str) -> Linear:
        """A dense linear layer in the dtype the checkpoint computes in."""
        dtype = self[f"{name}.weight"].dtype
        if f"{name}.scales" in self:
            dtype = self[f"{name}.scales"].dtype
        return Linear(self.float32(name).astype(dtype))


def unit_scale(weight: mx.array) -> mx.array:
    """Scale of a norm whose checkpoint weight is centered at zero."""
    return 1.0 + weight.astype(mx.float32)


def norm(x: mx.array, scale: mx.array, eps: float) -> mx.array:
    """RMS norm with a float32 scale, rounded once to the dtype of ``x``."""
    return mx.fast.rms_norm(x, scale, eps).astype(x.dtype)


def stream_norm(
    x: mx.array, scale: mx.array, eps: float, square_sums: Any = None
) -> mx.array:
    """RMS norm of each residual stream; ``scale`` is [streams, hidden] float32.

    ``square_sums(x, streams)`` may supply the float32 sum of squares of each
    stream; the rest is the same float32 arithmetic, rounded once.
    """
    streams, hidden = scale.shape
    y = x.reshape(*x.shape[:-1], streams, hidden)
    if square_sums is None:
        y = mx.fast.rms_norm(y.astype(mx.float32), None, eps) * scale
        return y.reshape(x.shape).astype(x.dtype)
    inverse = mx.rsqrt(square_sums(x, streams) / hidden + eps)
    return _scaled(y, inverse, scale).reshape(x.shape)


@mx.compile
def _scaled(y: mx.array, inverse: mx.array, scale: mx.array) -> mx.array:
    return (y.astype(mx.float32) * inverse * scale).astype(y.dtype)


def row_sum(x: mx.array) -> mx.array:
    """``x.sum(-1, keepdims=True)`` rounded as MLX's Metal reduce rounds it.

    That reduce adds each four neighbours in the input dtype, then 32 such sums
    and then the groups in float32, rounding after each stage.
    """
    width = x.shape[-1]
    if x.dtype == mx.float32 or width % 128 or width > 4096:
        return x.sum(axis=-1, keepdims=True)
    lanes = x.reshape(*x.shape[:-1], width // 4, 4)
    partial = lanes[..., 0] + lanes[..., 1] + lanes[..., 2] + lanes[..., 3]
    groups = partial.reshape(*x.shape[:-1], width // 128, 32)
    groups = groups.astype(mx.float32).sum(axis=-1).astype(x.dtype)
    return groups.astype(mx.float32).sum(axis=-1, keepdims=True).astype(x.dtype)


def stacked_sum(x: mx.array) -> mx.array:
    """``x.sum(-2)`` over fewer than 32 rows, rounded as on the host.

    MLX's Metal reduce adds the rows eight apart first and then those eight
    partial sums in order, all in the input dtype.
    """
    rows = x.shape[-2]
    if x.dtype == mx.float32 or rows >= 32:
        return x.sum(axis=-2)
    partial = []
    for first in range(min(rows, 8)):
        total = x[..., first, :]
        for row in range(first + 8, rows, 8):
            total = x[..., row, :] + total
        partial.append(total)
    total = partial[0]
    for part in partial[1:]:
        total = part + total
    return total


def depthwise_conv(x: mx.array, taps: mx.array, dilation: int) -> mx.array:
    """Causal depthwise conv of [1, rows, C] with float32 ``taps`` [C, K].

    The taps add up in float32 in order and round once, like the host conv.
    """
    rows = x.shape[1] - (taps.shape[1] - 1) * dilation
    acc = x[:, :rows].astype(mx.float32) * taps[:, 0]
    for i in range(1, taps.shape[1]):
        lo = i * dilation
        acc = acc + x[:, lo : lo + rows].astype(mx.float32) * taps[:, i]
    return acc.astype(x.dtype)
