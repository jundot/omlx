from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig
from .linear import linear_forward


def check_array_shape(arr):
    shape = arr.shape

    if len(shape) == 4:
        out_channels, kH, KW, _ = shape
        return (out_channels >= kH) and (out_channels >= KW) and (kH == KW)
    elif len(shape) == 5:
        out_channels, kT, kH, KW, _ = shape
        return (out_channels >= kH) and (out_channels >= KW) and (kH == KW)
    else:
        return False


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> tuple:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    cos = mx.expand_dims(cos, axis=-2).astype(mx.float32)
    sin = mx.expand_dims(sin, axis=-2).astype(mx.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.astype(orig_q_dtype)
    k_embed = k_embed.astype(orig_k_dtype)
    return q_embed, k_embed


class Glm5NextVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, seqlen: int) -> mx.array:
        inv_freq = 1.0 / (
            self.theta ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        seq = mx.arange(seqlen, dtype=inv_freq.dtype)
        freqs = mx.outer(seq, inv_freq)
        return freqs


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ).moveaxis(1, 4)

        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(-1, self.embed_dim)
        return hidden_states


class Glm5NextVisionPatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_act: str,
        swiglu_limit: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = linear_forward(self.proj, x)
        x = nn.gelu(self.post_projection_norm(x))
        gate = linear_forward(self.gate_proj, x)
        up = linear_forward(self.up_proj, x)
        if self.swiglu_limit is not None:
            gate = mx.clip(gate, a_min=None, a_max=self.swiglu_limit)
            up = mx.clip(up, a_min=-self.swiglu_limit, a_max=self.swiglu_limit)
        return linear_forward(self.down_proj, nn.silu(gate) * up)


class Glm5NextVisionAttention(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(
            config.hidden_size, config.hidden_size * 3, bias=config.attention_bias
        )
        self.proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        cu_seqlens: mx.array,
        position_embeddings: tuple,
    ) -> mx.array:
        seq_length = hidden_states.shape[0]

        qkv = linear_forward(self.qkv, hidden_states)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, -1)
        qkv = qkv.transpose(1, 0, 2, 3)
        q, k, v = mx.split(qkv, 3, axis=0)
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(1, 0, 2)[None, ...]
        k = k.transpose(1, 0, 2)[None, ...]
        v = v.transpose(1, 0, 2)[None, ...]

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        split_indices = []
        cumsum = 0
        for i, length in enumerate(lengths[:-1]):
            cumsum += length
            split_indices.append(cumsum)

        q_splits = mx.split(q, split_indices, axis=2)
        k_splits = mx.split(k, split_indices, axis=2)
        v_splits = mx.split(v, split_indices, axis=2)

        attn_outputs = []
        for q_chunk, k_chunk, v_chunk in zip(q_splits, k_splits, v_splits):
            output = mx.fast.scaled_dot_product_attention(
                q_chunk, k_chunk, v_chunk, scale=self.scale, mask=None
            )
            attn_outputs.append(output)

        attn_output = mx.concatenate(attn_outputs, axis=2)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(seq_length, -1)
        attn_output = linear_forward(self.proj, attn_output)
        return attn_output


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.attention_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        gate = linear_forward(self.gate_proj, x)
        up = linear_forward(self.up_proj, x)
        if self.swiglu_limit is not None:
            gate = mx.clip(gate, a_min=None, a_max=self.swiglu_limit)
            up = mx.clip(up, a_min=-self.swiglu_limit, a_max=self.swiglu_limit)
        return linear_forward(self.down_proj, nn.silu(gate) * up)


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config)

    def __call__(
        self, hidden_states: mx.array, cu_seqlens: mx.array, position_embeddings: tuple
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size

        self.patch_embed = Glm5NextVisionPatchEmbed(config)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(head_dim // 2)

        self.blocks = [Glm5NextVisionBlock(config) for _ in range(config.depth)]

        self.merger = Glm5NextVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.projection_intermediate_size,
            hidden_act=config.hidden_act,
            swiglu_limit=config.swiglu_limit,
        )

        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
            bias=True,
        )

        self.post_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def rot_pos_emb(self, grid_thw: mx.array):
        pos_ids = []

        for t, h, w in grid_thw.tolist():
            hpos_ids = mx.expand_dims(mx.arange(h), 1)
            hpos_ids = mx.repeat(hpos_ids, w, axis=1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = mx.expand_dims(mx.arange(w), 0)
            wpos_ids = mx.repeat(wpos_ids, h, axis=0)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_pos_ids = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_ids.append(mx.tile(stacked_pos_ids, (t, 1)))

        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = mx.max(grid_thw[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size.item())
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].reshape(pos_ids.shape[0], -1)

        emb = mx.concatenate((rotary_pos_emb, rotary_pos_emb), axis=-1)
        return (mx.cos(emb), mx.sin(emb)), pos_ids

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings, _ = self.rot_pos_emb(grid_thw)

        seq_lens = grid_thw[:, 1] * grid_thw[:, 2]
        repeats = grid_thw[:, 0]
        repeated_values = []
        for seq_len, repeat_count in zip(seq_lens.tolist(), repeats.tolist()):
            repeated_values.extend([seq_len] * repeat_count)

        cu_seqlens = mx.array(repeated_values).cumsum(axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), constant_values=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.post_layernorm(hidden_states)

        hidden_states = hidden_states.reshape(
            -1,
            self.spatial_merge_size,
            self.spatial_merge_size,
            hidden_states.shape[-1],
        )
        hidden_states = self.downsample(hidden_states).reshape(
            -1, self.config.out_hidden_size
        )

        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            elif "patch_embed.proj.weight" in k or "downsample.weight" in k:
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    if v.ndim == 5:
                        sanitized_weights[k] = v.transpose(0, 2, 3, 4, 1)
                    elif v.ndim == 4:
                        sanitized_weights[k] = v.transpose(0, 2, 3, 1)
                    else:
                        sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights
