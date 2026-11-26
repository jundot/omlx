# SPDX-License-Identifier: Apache-2.0
"""Hy4 / Hy4-flash vision tower and projector.

Pre-LayerNorm ViT over 16x16 patches with learned position embeddings that
are bilinearly resampled to each image's grid, followed by the
``rmsnorm_mlp_adapter`` projector: RMSNorm, a 2x2 strided conv, GELU, a 1x1
projection to the text width, and one learned newline vector appended to every
merged row. Each image attends only to itself.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def grid_rows(grid_thw) -> list[tuple[int, int, int]]:
    """``[t, h, w]`` rows of ``grid_thw`` as Python ints."""
    if isinstance(grid_thw, mx.array):
        grid_thw = grid_thw.tolist()
    return [(int(t), int(h), int(w)) for t, h, w in np.asarray(grid_thw).tolist()]


def _bilinear_axis(out_size: int, in_size: int, scale_factor: float):
    """Source indices and weights of torch's bilinear resize on one axis.

    The checkpoints were trained with ``scale_factor=(size + 0.1) / in_size``,
    which shifts the sampling grid against a plain size-to-size resize, so the
    axis is computed here exactly as ``F.interpolate`` does (align_corners off,
    coordinates clamped at the first sample).
    """
    scale = np.float32(1.0 / scale_factor)
    src = scale * (np.arange(out_size, dtype=np.float32) + np.float32(0.5))
    src = np.maximum(src - np.float32(0.5), np.float32(0.0))
    low = np.minimum(src.astype(np.int64), in_size - 1)
    high = np.minimum(low + 1, in_size - 1)
    frac = np.minimum(src - low.astype(np.float32), np.float32(1.0))
    return low, high, frac


class PatchEmbedding(nn.Module):
    """Patch projection plus resampled learned position embeddings.

    The patch conv is applied as a matrix product over the flattened
    (channel, row, column) patch: its stride equals its kernel.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        patch = config.patch_size
        self.edge = config.max_image_size // patch
        self.patch_embedding = nn.Linear(
            config.num_channels * patch * patch, config.hidden_size, bias=True
        )
        # Row 0 is the class-token slot; the reference reads only rows 1 onward.
        self.position_embedding = nn.Embedding(self.edge**2 + 1, config.hidden_size)

    def position_grid(self, height: int, width: int) -> mx.array:
        """[height * width, hidden] float32 positions of one image grid, resampled
        from the learned square table (up or down)."""
        table = self.position_embedding.weight[1:].astype(mx.float32)
        table = table.reshape(self.edge, self.edge, -1)
        rows = []
        for out_size, scale in (
            (height, (height + 0.1) / self.edge),
            (width, (width + 0.1) / self.edge),
        ):
            low, high, frac = _bilinear_axis(out_size, self.edge, scale)
            rows.append((mx.array(low), mx.array(high), mx.array(frac)))
        (h0, h1, hf), (w0, w1, wf) = rows
        along_h = table[h0] * (1 - hf)[:, None, None] + table[h1] * hf[:, None, None]
        out = (
            along_h[:, w0] * (1 - wf)[None, :, None]
            + along_h[:, w1] * wf[None, :, None]
        )
        return out.reshape(height * width, -1)

    def __call__(self, pixel_values: mx.array, grids) -> mx.array:
        x = self.patch_embedding(pixel_values.astype(self.patch_embedding.weight.dtype))
        positions = [self.position_grid(h, w) for _, h, w in grids]
        return x + mx.concatenate(positions, axis=0).astype(x.dtype)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.scale = (d // self.num_heads) ** -0.5
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.o_proj = nn.Linear(d, d, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        n, d = x.shape
        shape = (1, n, self.num_heads, d // self.num_heads)
        q, k, v = (
            proj(x).reshape(shape).swapaxes(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.o_proj(out.swapaxes(1, 2).reshape(n, d))


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.dense_4h_to_h = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.dense_4h_to_h(nn.gelu(self.dense_h_to_4h(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        d, eps = config.hidden_size, config.rms_norm_eps
        self.input_layernorm = nn.LayerNorm(d, eps=eps)
        self.self_attn = VisionAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(d, eps=eps)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class Projector(nn.Module):
    """Merge 2x2 patch blocks into text-width tokens; one newline per row."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        d, m = config.hidden_size, config.spatial_merge_size
        self.merge = m
        self.norm = nn.RMSNorm(d, eps=config.rms_norm_eps)
        self.proj_conv = nn.Conv2d(d, d * m * m, kernel_size=m, stride=m, bias=True)
        self.proj_out = nn.Linear(d * m * m, config.out_hidden_size, bias=True)
        self.image_newline = mx.zeros((config.out_hidden_size,))

    def __call__(self, x: mx.array, height: int, width: int) -> mx.array:
        if height % self.merge or width % self.merge:
            raise ValueError(f"Image grid {height}x{width} is not mergeable")
        x = self.norm(x).reshape(1, height, width, -1)
        x = self.proj_out(nn.gelu(self.proj_conv(x)))
        newline = mx.broadcast_to(
            self.image_newline.astype(x.dtype), (1, x.shape[1], 1, x.shape[-1])
        )
        return mx.concatenate([x, newline], axis=2).reshape(-1, x.shape[-1])


class VisionModel(nn.Module):
    """Image tower: [patches, C*P*P] pixels and their grids -> text features."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.embeddings = PatchEmbedding(config)
        self.layers = [VisionBlock(config) for _ in range(config.num_hidden_layers)]
        self.perceive = Projector(config)

    @property
    def dtype(self):
        return self.embeddings.patch_embedding.weight.dtype

    def __call__(self, pixel_values: mx.array, grid_thw) -> mx.array:
        """[sum of image tokens, out_hidden_size] features, images in order.

        An image of ``h x w`` patches yields ``(h / 2) * (w / 2 + 1)`` tokens
        (a newline vector ends every merged row).
        """
        grids = grid_rows(grid_thw)
        if any(t != 1 for t, _, _ in grids):
            raise ValueError("Hy4 vision tower supports still images only")
        if sum(h * w for _, h, w in grids) != pixel_values.shape[0]:
            raise ValueError("pixel_values does not match image_grid_thw")
        x = self.embeddings(pixel_values, grids)
        starts = np.cumsum([0] + [h * w for _, h, w in grids])
        # Images are independent, so each one runs through the layers alone.
        outputs = []
        for (_, h, w), lo, hi in zip(grids, starts[:-1], starts[1:]):
            y = x[int(lo) : int(hi)]
            for layer in self.layers:
                y = layer(y)
            outputs.append(self.perceive(y, h, w))
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
