# SPDX-License-Identifier: Apache-2.0
"""MLX vision encoder for MiMo V2.6 omnimodal checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    out_hidden_size: int = 4096
    num_heads: int = 32
    num_key_value_heads: int = 8
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_chans: int = 3
    spatial_merge_size: int = 2
    hidden_act: str = "silu"
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [0, 9, 18, 27])
    use_sink: bool = True
    visual_token_window_size: int = 64
    vit_window_attn_types: list[int] = field(default_factory=lambda: [-1] * 28)
    qk_channels: int = 64
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> VisionConfig:
        aliases = {"in_channels": "in_chans"}
        fields = cls.__dataclass_fields__
        normalized = {aliases.get(key, key): value for key, value in values.items()}
        return cls(**{key: value for key, value in normalized.items() if key in fields})


def _rotate_half(x: mx.array) -> mx.array:
    midpoint = x.shape[-1] // 2
    return mx.concatenate((-x[..., midpoint:], x[..., :midpoint]), axis=-1)


def _apply_rotary(q: mx.array, k: mx.array, frequencies: mx.array):
    cos = mx.cos(frequencies).astype(mx.float32)[:, None, :]
    sin = mx.sin(frequencies).astype(mx.float32)[:, None, :]
    q_dtype, k_dtype = q.dtype, k.dtype
    q_float, k_float = q.astype(mx.float32), k.astype(mx.float32)
    q = (q_float * cos + _rotate_half(q_float) * sin).astype(q_dtype)
    k = (k_float * cos + _rotate_half(k_float) * sin).astype(k_dtype)
    return q, k


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, sequence_length: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta
            ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        return mx.outer(mx.arange(sequence_length, dtype=mx.float32), inv_freq)


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_chans
        self.hidden_size = config.hidden_size
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            config.in_chans,
            config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)
        return self.proj(hidden_states).reshape(-1, self.hidden_size)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig, *, use_sink: bool):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.qk_channels
        self.scale = self.head_dim**-0.5
        self.window_size = config.visual_token_window_size
        qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv = nn.Linear(config.hidden_size, qkv_dim, bias=True)
        self.proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=True)
        self.sinks = mx.zeros((self.num_heads,)) if use_sink else None

    def _mask(self, length: int, dtype: mx.Dtype, full_attention: bool):
        mask = None
        if not full_attention and self.window_size > 0:
            positions = mx.arange(length)
            outside = mx.abs(positions[:, None] - positions[None, :]) > self.window_size
            mask = mx.where(
                outside,
                mx.array(float("-inf"), dtype=dtype),
                mx.array(0, dtype=dtype),
            )

        # MiMo's vision sink is a learned bias on the first real key, not the
        # extra softmax-denominator sink implemented by MLX's ``sinks=`` API.
        if not full_attention and self.sinks is not None:
            first_key = (mx.arange(length) == 0).astype(dtype)
            sink_bias = (
                self.sinks.astype(dtype)[None, :, None, None]
                * first_key[None, None, None, :]
            )
            mask = sink_bias if mask is None else mask[None, None, :, :] + sink_bias
        return mask

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        frequencies: mx.array,
        *,
        full_attention: bool,
    ) -> mx.array:
        sequence_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q = qkv[:, :q_size].reshape(sequence_length, self.num_heads, self.head_dim)
        k = qkv[:, q_size : q_size + kv_size].reshape(
            sequence_length, self.num_kv_heads, self.head_dim
        )
        v = qkv[:, q_size + kv_size :].reshape(
            sequence_length, self.num_kv_heads, self.head_dim
        )
        q, k = _apply_rotary(q, k, frequencies)

        split_points = [int(value) for value in cu_seqlens[1:-1].tolist()]
        q_chunks = mx.split(q, split_points, axis=0)
        k_chunks = mx.split(k, split_points, axis=0)
        v_chunks = mx.split(v, split_points, axis=0)
        outputs = []
        for q_chunk, k_chunk, v_chunk in zip(q_chunks, k_chunks, v_chunks):
            if self.num_kv_groups > 1:
                k_chunk = mx.repeat(k_chunk, self.num_kv_groups, axis=1)
                v_chunk = mx.repeat(v_chunk, self.num_kv_groups, axis=1)
            q_chunk = q_chunk.transpose(1, 0, 2)[None, ...]
            k_chunk = k_chunk.transpose(1, 0, 2)[None, ...]
            v_chunk = v_chunk.transpose(1, 0, 2)[None, ...]
            output = mx.fast.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                scale=self.scale,
                mask=self._mask(q_chunk.shape[2], q_chunk.dtype, full_attention),
            )
            outputs.append(output[0].transpose(1, 0, 2))
        hidden_states = mx.concatenate(outputs, axis=0).reshape(sequence_length, -1)
        return self.proj(hidden_states)


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig, *, use_sink: bool):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = VisionAttention(config, use_sink=use_sink)
        self.mlp = VisionMLP(config)

    def __call__(self, hidden_states, cu_seqlens, frequencies, *, full_attention):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens,
            frequencies,
            full_attention=full_attention,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        merged_size = config.hidden_size * config.spatial_merge_size**2
        self.merged_size = merged_size
        # The released checkpoint omits these three zero-valued PyTorch bias
        # tensors. Disabling them makes strict sidecar loading match exactly
        # while preserving the reference computation.
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=1e-6, bias=False)
        self.mlp = [
            nn.Linear(merged_size, merged_size, bias=False),
            nn.GELU(),
            nn.Linear(merged_size, config.out_hidden_size, bias=False),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ln_q(x).reshape(-1, self.merged_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class VisionModel(nn.Module):
    """MiMo's GQA vision transformer, including row/column attention order."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.fullatt_block_indexes = set(config.fullatt_block_indexes)
        self.vit_window_attn_types = config.vit_window_attn_types
        self.patch_embed = PatchEmbed(config)
        self.rotary_pos_emb = VisionRotaryEmbedding(config.qk_channels // 2)
        self.blocks = [
            VisionBlock(
                config,
                use_sink=config.use_sink and index not in self.fullatt_block_indexes,
            )
            for index in range(config.depth)
        ]
        self.merger = PatchMerger(config)

    def _apply_index(self, tensor: mx.array, index: mx.array) -> mx.array:
        grouped = tensor.reshape(-1, self.spatial_merge_unit, tensor.shape[-1])
        return grouped[index].reshape(-1, tensor.shape[-1])

    def _window_index(self, grid_thw: mx.array, *, column: bool) -> mx.array:
        indexes = []
        offset = 0
        for grid_t, grid_h, grid_w in grid_thw.tolist():
            height = int(grid_h) // self.spatial_merge_size
            width = int(grid_w) // self.spatial_merge_size
            index = mx.arange(int(grid_t) * height * width).reshape(
                int(grid_t), height, width
            )
            if column:
                index = index.transpose(0, 2, 1)
            indexes.append(index.reshape(-1) + offset)
            offset += int(grid_t) * height * width
        return mx.concatenate(indexes, axis=0).astype(mx.int32)

    def _rotary_frequencies(self, grid_thw: mx.array) -> mx.array:
        positions = []
        maximum = 0
        merge = self.spatial_merge_size
        for grid_t, grid_h, grid_w in grid_thw.tolist():
            t, h, w = int(grid_t), int(grid_h), int(grid_w)
            maximum = max(maximum, h, w)
            h_ids = mx.repeat(mx.arange(h)[:, None], w, axis=1)
            w_ids = mx.repeat(mx.arange(w)[None, :], h, axis=0)
            h_ids = h_ids.reshape(h // merge, merge, w // merge, merge)
            w_ids = w_ids.reshape(h // merge, merge, w // merge, merge)
            h_ids = h_ids.transpose(0, 2, 1, 3).reshape(-1)
            w_ids = w_ids.transpose(0, 2, 1, 3).reshape(-1)
            positions.append(mx.tile(mx.stack((h_ids, w_ids), axis=-1), (t, 1)))
        position_ids = mx.concatenate(positions, axis=0).astype(mx.int32)
        frequencies = self.rotary_pos_emb(maximum)[position_ids].reshape(
            position_ids.shape[0], -1
        )
        return mx.concatenate((frequencies, frequencies), axis=-1)

    @staticmethod
    def _cu_seqlens(grid_thw: mx.array) -> mx.array:
        lengths = []
        for grid_t, grid_h, grid_w in grid_thw.tolist():
            lengths.extend([int(grid_h) * int(grid_w)] * int(grid_t))
        return mx.pad(
            mx.cumsum(mx.array(lengths, dtype=mx.int32)),
            (1, 0),
            constant_values=0,
        )

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array, **_kwargs):
        dtype = self.patch_embed.proj.weight.dtype
        hidden_states = self.patch_embed(pixel_values.astype(dtype))
        frequencies = self._rotary_frequencies(grid_thw)
        column_index = self._window_index(grid_thw, column=True)
        reverse_column_index = mx.argsort(column_index)
        row_frequencies = frequencies
        column_frequencies = self._apply_index(frequencies, column_index)
        cu_seqlens = self._cu_seqlens(grid_thw)

        previous_type = -1
        for index, block in enumerate(self.blocks):
            attention_type = self.vit_window_attn_types[index]
            if attention_type == 1 and previous_type != 1:
                hidden_states = self._apply_index(hidden_states, column_index)
            elif attention_type != 1 and previous_type == 1:
                hidden_states = self._apply_index(hidden_states, reverse_column_index)
            frequencies = column_frequencies if attention_type == 1 else row_frequencies
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                frequencies,
                full_attention=index in self.fullatt_block_indexes,
            )
            previous_type = attention_type
        if previous_type == 1:
            hidden_states = self._apply_index(hidden_states, reverse_column_index)
        return self.merger(hidden_states)

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("visual."):
                key = key[len("visual.") :]
            if key.endswith("patch_embed.proj.weight") and value.ndim == 5:
                # PyTorch: OIzyx; MLX Conv3d: OzyxI.
                value = value.transpose(0, 2, 3, 4, 1)
            sanitized[key] = value
        return sanitized
