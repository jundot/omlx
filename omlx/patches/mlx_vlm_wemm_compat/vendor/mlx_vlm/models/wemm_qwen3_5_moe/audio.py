from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import AudioConfig, QFormerConfig


def _whisper_sinusoids(
    length: int, channels: int, max_timescale: float = 10000.0
) -> mx.array:
    if channels % 2 != 0:
        raise ValueError(f"channels must be even, got {channels}")
    log_inc = math.log(max_timescale) / (channels // 2 - 1)
    inv = mx.exp(-log_inc * mx.arange(channels // 2, dtype=mx.float32))
    t = mx.arange(length, dtype=mx.float32)[:, None] * inv[None, :]
    return mx.concatenate([mx.sin(t), mx.cos(t)], axis=1)


class WeMMWhisperAttention(nn.Module):
    """Whisper encoder multi-head attention with separate q/k/v/out projections."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        bsz, seq_len, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(hidden_states)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(hidden_states)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.embed_dim)
        return self.out_proj(out)


class WeMMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        d_model = config.d_model
        self.self_attn = WeMMWhisperAttention(
            embed_dim=d_model, num_heads=config.encoder_attention_heads
        )
        self.self_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.fc1 = nn.Linear(d_model, config.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, d_model, bias=True)
        self.final_layer_norm = nn.LayerNorm(d_model, eps=1e-5)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(self.self_attn_layer_norm(hidden_states))
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = nn.gelu(self.fc1(self.final_layer_norm(hidden_states)))
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class WeMMQFormerAttention(nn.Module):
    """Post-LN BERT self/cross-attention block used in the WeMM audio Q-Former."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_dim: int,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Linear(hidden_size, hidden_size, bias=True)
        self.key = nn.Linear(kv_dim, hidden_size, bias=True)
        self.value = nn.Linear(kv_dim, hidden_size, bias=True)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(
        self, hidden_states: mx.array, context: Optional[mx.array] = None
    ) -> mx.array:
        ctx = hidden_states if context is None else context
        bsz, q_len, _ = hidden_states.shape
        kv_len = ctx.shape[1]

        q = (
            self.query(hidden_states)
            .reshape(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.key(ctx)
            .reshape(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.value(ctx)
            .reshape(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        out = self.dense(out)
        return self.LayerNorm(out + hidden_states)


class WeMMQFormerLayer(nn.Module):
    def __init__(self, config: QFormerConfig):
        super().__init__()
        eps = config.layer_norm_eps
        self.attention = WeMMQFormerAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.hidden_size,
            eps=eps,
        )
        self.crossattention = WeMMQFormerAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.encoder_width,
            eps=eps,
        )
        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.output_dense = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )
        self.output_LayerNorm = nn.LayerNorm(config.hidden_size, eps=eps)

    def __call__(
        self, query_states: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        q = self.attention(query_states)
        q = self.crossattention(q, encoder_hidden_states)
        ffn = self.output_dense(nn.gelu(self.intermediate_dense(q)))
        return self.output_LayerNorm(ffn + q)


class WeMMAudioTower(nn.Module):
    """WeMM Audio Tower: Whisper Encoder -> Temporal Windowing -> 2-layer Q-Former -> Projection."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        qfc = config.qformer_config
        d_model = config.d_model

        self.second_per_window = float(config.second_per_window)
        self.second_stride = float(config.second_stride)
        self.window_frames = round(1500 * self.second_per_window / 30.0)
        self.stride_frames = max(1, round(1500 * self.second_stride / 30.0))

        self.conv1 = nn.Conv1d(config.num_mel_bins, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=2, padding=1
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, d_model)
        self.embed_positions.weight = _whisper_sinusoids(
            config.max_source_positions, d_model
        )

        self.layers = [
            WeMMWhisperEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-5)

        self.q_tokens = mx.zeros((1, qfc.query_length, qfc.hidden_size))
        self.qformer_embeddings_ln = nn.LayerNorm(
            qfc.hidden_size, eps=qfc.layer_norm_eps
        )
        self.qformer_layers = [
            WeMMQFormerLayer(qfc) for _ in range(qfc.num_hidden_layers)
        ]
        self.audio_proj = nn.Linear(qfc.hidden_size, qfc.out_hidden_size, bias=True)

    def _encode_batch(self, mel_features: mx.array) -> mx.array:
        # mel_features: (B, 128, 3000) -> MLX Conv1d expects (B, 3000, 128)
        target_dtype = self.conv1.weight.dtype
        x = mel_features.astype(target_dtype)
        if x.ndim == 2:
            x = x[None, :, :]
        x = x.transpose(0, 2, 1)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        x = x + self.embed_positions.weight[: x.shape[1], :].astype(x.dtype)
        for layer in self.layers:
            x = layer(x)
        speech_embeds = self.layer_norm(x)  # (B, 1500, 1280)

        bsz, seq_len, channels = speech_embeds.shape
        kernel = self.window_frames
        stride = self.stride_frames
        if kernel == stride and seq_len % kernel == 0:
            num_windows = seq_len // kernel
            windows = speech_embeds.reshape(bsz * num_windows, kernel, channels)
        else:
            starts = range(0, seq_len - kernel + 1, stride)
            stacked = mx.stack(
                [speech_embeds[:, s : s + kernel, :] for s in starts], axis=1
            )
            windows = stacked.reshape(-1, kernel, channels)

        q = mx.broadcast_to(
            self.q_tokens.astype(windows.dtype),
            (windows.shape[0], self.q_tokens.shape[1], self.q_tokens.shape[2]),
        )
        q = self.qformer_embeddings_ln(q)
        for qlayer in self.qformer_layers:
            q = qlayer(q, windows)
        projected = self.audio_proj(q)
        return projected.reshape(bsz, -1, projected.shape[-1])

    def __call__(self, input_features: mx.array, batch_size: int = 8) -> mx.array:
        if input_features.ndim == 2:
            input_features = input_features[None, :, :]
        total = int(input_features.shape[0])
        if total <= batch_size:
            return self._encode_batch(input_features)
        chunks = []
        for start in range(0, total, batch_size):
            chunk = self._encode_batch(input_features[start : start + batch_size])
            mx.eval(chunk)
            chunks.append(chunk)
        return mx.concatenate(chunks, axis=0)
