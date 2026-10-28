# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..lfm3.language import MLP


def grid_positions(height, width, merge):
    coords = np.stack(
        np.meshgrid(np.arange(height), np.arange(width), indexing="ij"), axis=-1
    )
    return mx.array(
        coords.reshape(height // merge, merge, width // merge, merge, 2)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 2)
    )


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = nn.RMSNorm(self.head_dim, eps=config.norm_eps)

    def __call__(self, x, cos, sin):
        n = x.shape[0]
        q = self.q_layernorm(self.q_proj(x).reshape(n, self.n_heads, self.head_dim))
        k = self.k_layernorm(self.k_proj(x).reshape(n, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(n, self.n_kv_heads, self.head_dim)

        def rotate(t):
            dtype = t.dtype
            t = t.astype(mx.float32)
            a, b = mx.split(t, 2, axis=-1)
            return (
                t * cos[:, None] + mx.concatenate([-b, a], axis=-1) * sin[:, None]
            ).astype(dtype)

        q, k = rotate(q), rotate(k)
        out = mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None],
            scale=self.scale,
        )
        return self.out_proj(out[0].transpose(1, 0, 2).reshape(n, -1))


class VisionConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.merge = config.spatial_merge_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.spatial_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size,
            config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(self, x, height, width):
        b, c, v = mx.split(self.in_proj(x), 3, axis=-1)
        m = self.merge
        grid = (
            (b * v)
            .reshape(height // m, width // m, m, m, -1)
            .transpose(0, 2, 1, 3, 4)
            .reshape(1, height, width, -1)
        )
        grid = self.spatial_conv(grid)
        out = (
            grid.reshape(height // m, m, width // m, m, -1)
            .transpose(0, 2, 1, 3, 4)
            .reshape(height * width, -1)
        )
        return self.out_proj(c * out)


class VisionLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.is_attention_layer = config.layer_types[index] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = VisionAttention(config)
        else:
            self.conv = VisionConv(config)
        self.feed_forward = MLP(config)
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x, height, width, cos, sin):
        h = self.operator_norm(x)
        x = x + (
            self.self_attn(h, cos, sin)
            if self.is_attention_layer
            else self.conv(h, height, width)
        )
        return x + self.feed_forward(self.ffn_norm(x))


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.patchifier = nn.Linear(
            config.in_channels * config.temporal_patch_size * config.patch_size**2,
            config.hidden_size,
            bias=False,
        )
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.layers = [VisionLayer(config, i) for i in range(config.num_hidden_layers)]

    def __call__(self, pixel_values, grid_thw):
        frames = []
        offset = 0
        config = self.config
        spatial_dim = config.hidden_size // config.num_attention_heads // 2
        inv_freq = config.rope_parameters["rope_theta"] ** (
            -mx.arange(0, spatial_dim, 2, dtype=mx.float32) / spatial_dim
        )
        for temporal, height, width in np.asarray(grid_thw).tolist():
            coords = grid_positions(height, width, config.spatial_merge_size)
            angles = (coords[..., None] * inv_freq).reshape(height * width, -1)
            angles = mx.concatenate([angles, angles], axis=-1)
            cos, sin = mx.cos(angles), mx.sin(angles)
            for _ in range(temporal):
                x = self.embedding_norm(
                    self.patchifier(
                        pixel_values[offset : offset + height * width].astype(
                            self.embedding_norm.weight.dtype
                        )
                    )
                )
                for layer in self.layers:
                    x = layer(x, height, width, cos, sin)
                frames.append(x)
                offset += height * width
        if offset != pixel_values.shape[0]:
            raise ValueError("Pixel patch count does not match grid_thw")
        return mx.concatenate(frames, axis=0)
