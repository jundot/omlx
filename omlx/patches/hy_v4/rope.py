# SPDX-License-Identifier: Apache-2.0
"""Adjacent-pair (GPT-J) RoPE helpers for Hy4.

Hy4 rotates channel pairs (2i, 2i+1). On text, its interleaved M-RoPE reads
the same position on all three axes, which is ordinary RoPE.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class TrailingRoPE(nn.Module):
    """RoPE on the trailing ``dims`` channels of each head.

    Hy4's indexers rotate the last ``qk_rope_head_dim`` channels, the opposite
    end from DeepSeek-V3.2's indexer.
    """

    def __init__(self, dims: int, base: float):
        super().__init__()
        self.dims = dims
        self.base = base

    def __call__(self, x: mx.array, offset=0) -> mx.array:
        head = x.shape[-1] - self.dims
        rotated = mx.fast.rope(
            x[..., head:],
            self.dims,
            traditional=True,
            base=self.base,
            scale=1.0,
            offset=offset,
        )
        if head == 0:
            return rotated
        return mx.concatenate([x[..., :head], rotated], axis=-1)


def rope_at(x: mx.array, positions: mx.array, dims: int, base: float) -> mx.array:
    """RoPE of the trailing ``dims`` channels at explicit positions.

    ``x`` is [..., N, D]; ``positions`` holds one position per row N and may
    carry leading batch dimensions.
    """
    half = dims // 2
    inv_freq = base ** (-mx.arange(0, dims, 2, dtype=mx.float32) / dims)
    theta = positions.astype(mx.float32)[..., None] * inv_freq
    cos, sin = mx.cos(theta), mx.sin(theta)
    head = x.shape[-1] - dims
    r = x[..., head:].astype(mx.float32).reshape(*x.shape[:-1], half, 2)
    even, odd = r[..., 0], r[..., 1]
    rotated = mx.stack([even * cos - odd * sin, odd * cos + even * sin], axis=-1)
    rotated = rotated.reshape(*x.shape[:-1], dims).astype(x.dtype)
    if head == 0:
        return rotated
    return mx.concatenate([x[..., :head], rotated], axis=-1)
