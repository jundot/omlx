# SPDX-License-Identifier: MIT
# Adapted from Blaizzy/mlx-vlm PR #2136 (7ec3d07e), MIT license.
import math

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig

VISION_NORM_EPS = 1e-6


def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float
) -> tuple[mx.array, mx.array]:
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    hpos = mx.broadcast_to(mx.arange(n_h)[:, None], (n_h, n_w))
    wpos = mx.broadcast_to(mx.arange(n_w)[None, :], (n_h, n_w))
    positions = mx.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1)
    freqs = (positions.astype(mx.float32) * inv_freq).reshape(n_h * n_w, -1)
    return mx.cos(freqs)[:, None, :], mx.sin(freqs)[:, None, :]


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    dtype = x.dtype
    x1, x2 = mx.split(x.astype(mx.float32), 2, axis=-1)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(
        dtype
    )


class PatchEmbed(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        patch_dim = 3 * config.vision_patch_size**2
        self.proj = nn.Linear(patch_dim, config.vision_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x.reshape(x.shape[0], -1))


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.scale = self.head_dim**-0.5
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        n = x.shape[0]
        q, k, v = (
            value.reshape(n, self.n_heads, self.head_dim)
            for value in mx.split(self.wqkv(x), 3, axis=-1)
        )
        q = apply_rotary(q, cos, sin).transpose(1, 0, 2)
        k = apply_rotary(k, cos, sin).transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        output = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=self.scale
        )[0]
        return self.wo(output.transpose(1, 0, 2).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(nn.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.vision_dim, eps=VISION_NORM_EPS)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(config.vision_dim, eps=VISION_NORM_EPS)
        self.ffn = MLP(config)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.ffn(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT with full attention over one image and 2D RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        head_dim = config.vision_dim // config.vision_n_heads
        if head_dim % 4:
            raise ValueError("DeepSeek-V4.1 vision head dimension must divide by 4")
        self.rope_dim = head_dim // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = PatchEmbed(config)
        self.blocks = [Block(config) for _ in range(config.vision_n_layers)]
        self.norm = nn.RMSNorm(config.vision_dim, eps=VISION_NORM_EPS)

    def __call__(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        if patches.shape[0] != n_h * n_w:
            raise ValueError(
                f"Expected {n_h * n_w} vision patches, got {patches.shape[0]}"
            )
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.dim)
        self.w2 = nn.Linear(config.dim, config.dim)
        self.gelu = nn.GELU()

    def __call__(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        ratio = self.downsample_ratio
        dim = x.shape[-1]
        pad_h = -n_h % ratio
        pad_w = -n_w % ratio
        x = x.reshape(n_h, n_w, dim)
        if pad_h or pad_w:
            x = mx.pad(x, ((0, pad_h), (0, pad_w), (0, 0)))

        out_h = math.ceil(n_h / ratio)
        out_w = math.ceil(n_w / ratio)
        x = x.reshape(out_h, ratio, out_w, ratio, dim)
        x = x.transpose(0, 2, 4, 1, 3).reshape(out_h * out_w, dim * ratio**2)
        return self.w2(self.gelu(self.w1(x)))


__all__ = ["Aligner", "ViT", "apply_rotary", "get_vision_cos_sin"]
