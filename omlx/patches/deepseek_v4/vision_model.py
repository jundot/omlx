# SPDX-License-Identifier: Apache-2.0
"""Small MLX port of the public DeepSeek-V4 Vision ViT and aligner."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    hpos = mx.broadcast_to(mx.arange(n_h)[:, None], (n_h, n_w))
    wpos = mx.broadcast_to(mx.arange(n_w)[None], (n_h, n_w))
    positions = mx.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1)
    frequencies = positions.astype(mx.float32) * inv
    frequencies = frequencies.reshape(frequencies.shape[0], -1)
    return mx.cos(frequencies)[:, None], mx.sin(frequencies)[:, None]


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    dtype = x.dtype
    x = x.astype(mx.float32)
    first, second = mx.split(x, 2, axis=-1)
    return mx.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    ).astype(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, config):
        patch = config.vision_patch_size
        self.proj = nn.Linear(3 * patch * patch, config.vision_dim, bias=True)

    def __call__(self, x):
        return self.proj(x.reshape(x.shape[0], -1))


class Attention(nn.Module):
    def __init__(self, config):
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim, bias=True)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim, bias=True)

    def __call__(self, x, cos, sin):
        count = x.shape[0]
        q, k, v = (
            value.reshape(count, self.n_heads, self.head_dim)
            for value in mx.split(self.wqkv(x), 3, axis=-1)
        )
        q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)
        q, k, v = (value.transpose(1, 0, 2)[None] for value in (q, k, v))
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5
        )
        return self.wo(output[0].transpose(1, 0, 2).reshape(count, -1))


class MLP(nn.Module):
    def __init__(self, config):
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)

    def __call__(self, x):
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, config):
        self.norm1 = nn.RMSNorm(config.vision_dim, eps=1e-6)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.vision_dim, eps=1e-6)
        self.mlp = MLP(config)

    def __call__(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """Full bidirectional attention over one image with 2-D RoPE."""

    def __init__(self, config):
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = PatchEmbed(config)
        self.blocks = [Block(config) for _ in range(config.vision_n_layers)]
        self.norm = nn.RMSNorm(config.vision_dim, eps=1e-6)

    def __call__(self, patches, n_h: int, n_w: int):
        x = self.patch_embed(patches)
        cos, sin = _vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, config):
        self.downsample_ratio = config.vision_downsample_ratio
        input_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(input_dim, config.hidden_size, bias=True)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, x, n_h: int, n_w: int):
        ratio = self.downsample_ratio
        x = x.reshape(n_h, n_w, -1)
        x = mx.pad(x, ((0, -n_h % ratio), (0, -n_w % ratio), (0, 0)))
        out_h, out_w = x.shape[0] // ratio, x.shape[1] // ratio
        x = x.reshape(out_h, ratio, out_w, ratio, x.shape[-1])
        # Match torch.unfold on channel-first input: C, kernel-H, kernel-W.
        x = x.transpose(0, 2, 4, 1, 3).reshape(out_h * out_w, -1)
        return self.w2(nn.gelu(self.w1(x)))
