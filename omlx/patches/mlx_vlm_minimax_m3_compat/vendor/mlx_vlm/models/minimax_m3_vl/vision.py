from itertools import accumulate
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import ensure_fused_sdpa
from .config import VisionConfig


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_vision_rope(x, freqs):
    orig_dtype = x.dtype
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rot_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x_rot = x_rot * cos + _rotate_half(x_rot) * sin
    return mx.concatenate([x_rot, x_pass], axis=-1).astype(orig_dtype)


def _axis_freq(coords, dim, theta):
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    return mx.outer(coords.astype(mx.float32), inv_freq)


class MiniMaxVisionPatchEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.patch_dim = (
            config.num_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size
        )
        self.weight = mx.zeros(
            (
                config.hidden_size,
                config.num_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            )
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        original_shape = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, self.patch_dim)
        weight = self.weight.reshape(self.hidden_size, self.patch_dim)
        hidden_states = pixel_values @ weight.T
        return hidden_states.reshape(*original_shape[:-1], self.hidden_size)


class MiniMaxVisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embedding = MiniMaxVisionPatchEmbedding(config)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        return self.patch_embedding(pixel_values)


class MiniMaxVisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: Optional[mx.array] = None,
    ) -> mx.array:
        seq_length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        if rotary_pos_emb is not None:
            q = _apply_vision_rope(mx.expand_dims(q, axis=0), rotary_pos_emb)[0]
            k = _apply_vision_rope(mx.expand_dims(k, axis=0), rotary_pos_emb)[0]

        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]

        splits = [
            mx.split(tensor, cu_seqlens[1:-1].tolist(), axis=2) for tensor in (q, k, v)
        ]
        outputs = [
            ensure_fused_sdpa(q_i, k_i, v_i, self.scale)
            for q_i, k_i, v_i in zip(*splits)
        ]
        output = mx.concatenate(outputs, axis=2)
        output = output[0].transpose(1, 0, 2).reshape(seq_length, -1)
        return self.out_proj(output)


class MiniMaxVisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.hidden_act = config.hidden_act

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        if self.hidden_act == "quick_gelu":
            hidden_states = hidden_states * mx.sigmoid(1.702 * hidden_states)
        elif self.hidden_act == "silu":
            hidden_states = nn.silu(hidden_states)
        else:
            hidden_states = nn.gelu(hidden_states)
        return self.fc2(hidden_states)


class MiniMaxVisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = MiniMaxVisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MiniMaxVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), cu_seqlens, rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class MiniMaxVisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [
            MiniMaxVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        rotary_pos_emb: mx.array,
        output_hidden_states: bool = False,
    ) -> mx.array:
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, rotary_pos_emb)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        if output_hidden_states:
            return hidden_states, tuple(all_hidden_states)
        return hidden_states


class MiniMaxVisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = MiniMaxVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = MiniMaxVisionEncoder(config)

    def _rotary_pos_emb(self, grid_thw: mx.array) -> mx.array:
        merge_size = self.config.spatial_merge_size
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        rope_dims = 2 * (head_dim // 2)
        axis_dim = 2 * ((rope_dims // 3) // 2)

        pos_embeds = []
        for t, h, w in self._segment_grid_thw(grid_thw):
            t, h, w = int(t), int(h), int(w)
            merged_h, merged_w = h // merge_size, w // merge_size
            t_idx = mx.arange(t)
            h_block = mx.arange(merged_h)
            w_block = mx.arange(merged_w)
            intra_h = mx.arange(merge_size)
            intra_w = mx.arange(merge_size)

            tt = mx.broadcast_to(
                t_idx[:, None, None, None, None],
                (t, merged_h, merged_w, merge_size, merge_size),
            )
            hh = (
                h_block[None, :, None, None, None] * merge_size
                + intra_h[None, None, None, :, None]
            )
            hh = mx.broadcast_to(hh, (t, merged_h, merged_w, merge_size, merge_size))
            ww = (
                w_block[None, None, :, None, None] * merge_size
                + intra_w[None, None, None, None, :]
            )
            ww = mx.broadcast_to(ww, (t, merged_h, merged_w, merge_size, merge_size))

            freqs = [
                _axis_freq(tt.reshape(-1), axis_dim, self.config.rope_theta),
                _axis_freq(hh.reshape(-1), axis_dim, self.config.rope_theta),
                _axis_freq(ww.reshape(-1), axis_dim, self.config.rope_theta),
            ]
            freqs = mx.concatenate(freqs, axis=-1)
            pos_embeds.append(mx.concatenate([freqs, freqs], axis=-1))

        return mx.concatenate(pos_embeds, axis=0)

    def _segment_grid_thw(self, grid_thw: mx.array):
        max_frames = self.config.vision_segment_max_frames
        segments = []
        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            if max_frames is None or t <= max_frames:
                segments.append((t, h, w))
                continue
            for start in range(0, t, max_frames):
                segments.append((min(max_frames, t - start), h, w))
        return segments

    def __call__(
        self,
        pixel_values: mx.array,
        grid_thw: mx.array,
        output_hidden_states: bool = False,
    ) -> mx.array:
        hidden_states = self.embeddings(pixel_values).reshape(
            -1, self.config.hidden_size
        )
        hidden_states = self.pre_layrnorm(hidden_states)
        rotary_pos_emb = self._rotary_pos_emb(grid_thw)

        seqlens = [int(t * h * w) for t, h, w in self._segment_grid_thw(grid_thw)]
        cu_seqlens = mx.array([0] + list(accumulate(seqlens)), dtype=mx.int32)
        return self.encoder(
            hidden_states,
            cu_seqlens,
            rotary_pos_emb,
            output_hidden_states=output_hidden_states,
        )


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = MiniMaxVisionTransformer(config)

    def __call__(
        self,
        pixel_values: mx.array,
        grid_thw: mx.array,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        return self.vision_model(
            pixel_values, grid_thw, output_hidden_states=output_hidden_states
        )
