# SPDX-License-Identifier: Apache-2.0
"""MLX audio tokenizer and audio bridge for MiMo V2.6."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class AudioTokenizerAttention(nn.Module):
    def __init__(self, dim: int, heads: int, *, causal: bool, window_size: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.causal = causal
        self.window_size = window_size
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        batch, length, dim = x.shape
        q = (
            self.q_proj(x)
            .reshape(batch, length, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(batch, length, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(batch, length, self.heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        mask = None
        if self.causal or self.window_size > 0:
            row = mx.arange(length)[:, None]
            col = mx.arange(length)[None, :]
            blocked = mx.zeros((length, length), dtype=mx.bool_)
            if self.causal:
                blocked = mx.logical_or(blocked, col > row)
            if self.window_size > 0:
                blocked = mx.logical_or(blocked, mx.abs(row - col) > self.window_size)
            mask = mx.where(
                blocked, mx.array(-1e9, dtype=q.dtype), mx.array(0, dtype=q.dtype)
            )
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, dim))


class AudioTokenizerLayer(nn.Module):
    def __init__(
        self, dim: int, heads: int, ffn_dim: int, *, causal: bool, window_size: int
    ):
        super().__init__()
        self.self_attn = AudioTokenizerAttention(
            dim, heads, causal=causal, window_size=window_size
        )
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.self_attn(self.self_attn_layer_norm(x), cos, sin)
        return x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))


class _Codebook(nn.Module):
    def __init__(self, size: int, dim: int):
        super().__init__()
        self.embed = mx.zeros((size, dim))


class _VQLayer(nn.Module):
    def __init__(self, size: int, dim: int):
        super().__init__()
        self._codebook = _Codebook(size, dim)


class _VQ(nn.Module):
    def __init__(self, sizes: list[int], dim: int):
        super().__init__()
        self.layers = [_VQLayer(size, dim) for size in sizes]


class _Quantizer(nn.Module):
    def __init__(self, sizes: list[int], dim: int):
        super().__init__()
        self.vq = _VQ(sizes, dim)

    def encode(self, x: mx.array) -> mx.array:
        residual = x.astype(mx.float32)
        indices = []
        for layer in self.vq.layers:
            embed = layer._codebook.embed.astype(mx.float32)
            # Squared distance without materializing [T, codebook, dim].
            score = 2.0 * (residual @ embed.T) - mx.sum(embed * embed, axis=1)[None, :]
            index = mx.argmax(score, axis=-1)
            indices.append(index)
            residual = residual - embed[index]
        return mx.stack(indices, axis=-1)


class AudioTokenizerEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        dim = int(config["d_model"])
        self.config = SimpleNamespace(**config)
        self.head_dim = dim // int(config["encoder_attention_heads"])
        self.conv1 = nn.Conv1d(
            int(config["n_mels"]), dim, int(config["kernel_size"]), padding=1
        )
        self.conv2 = nn.Conv1d(
            dim,
            dim,
            int(config["kernel_size"]),
            stride=int(config["stride_size"]),
            padding=1,
        )
        hybrid = bool(config.get("hybrid_attention", False))
        swa_per_block = int(config.get("swa_per_block", 2))
        local_window = int(config.get("encoder_attn_window_size", [-1])[0])
        self.layers = [
            AudioTokenizerLayer(
                dim,
                int(config["encoder_attention_heads"]),
                int(config["encoder_ffn_dim"]),
                causal=bool(config.get("encoder_causal", False)),
                window_size=(
                    local_window
                    if hybrid and index % swa_per_block < swa_per_block - 1
                    else -1
                ),
            )
            for index in range(int(config["encoder_layers"]))
        ]
        self.layer_norm = nn.LayerNorm(dim)
        avg = int(config["avg_pooler"])
        self.down_sample_layer = (
            [nn.Conv1d(dim, dim, avg, stride=avg, bias=False), nn.GELU()]
            if avg != 1
            else None
        )
        self.down_sample_norm = nn.LayerNorm(dim) if avg != 1 else None
        sizes = list(config["codebook_size"])
        if len(sizes) < int(config["num_quantizers"]):
            sizes.extend([sizes[-1]] * (int(config["num_quantizers"]) - len(sizes)))
        self.quantizer = _Quantizer(sizes, dim)

    def _rope(self, length: int, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        inv = 1.0 / (
            float(self.config.rope_theta)
            ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
        )
        freqs = mx.arange(length, dtype=mx.float32)[:, None] * inv[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)

    def get_output_length(self, mel_len: int) -> int:
        length = (mel_len + 2 - int(self.config.kernel_size)) // int(
            self.config.stride_size
        ) + 1
        avg = int(self.config.avg_pooler)
        return (length + avg - 1) // avg

    def encode(self, mel: mx.array) -> mx.array:
        # Input mel is [n_mels, time]; MLX Conv1d consumes [batch, time, channels].
        x = mel.T[None, :, :]
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        cos, sin = self._rope(x.shape[1], x.dtype)
        skip = None
        skip_index = self.config.encoder_skip_layer_id
        for index, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            if skip_index is not None and index == int(skip_index) - 1:
                skip = x
        if skip is not None:
            x = x + skip
        x = self.layer_norm(x)
        if self.down_sample_layer is not None:
            avg = int(self.config.avg_pooler)
            if x.shape[1] % avg:
                x = mx.pad(x, [(0, 0), (0, avg - x.shape[1] % avg), (0, 0)])
            x = self.down_sample_layer[0](x)
            x = nn.gelu(x)
            x = self.down_sample_norm(x)
        return self.quantizer.encode(x[0])


class MiMoAudioTokenizer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = SimpleNamespace(**config)
        self.encoder = AudioTokenizerEncoder(config)

    @classmethod
    def load(cls, root: str | Path) -> MiMoAudioTokenizer:
        root = Path(root)
        config = json.loads((root / "config.json").read_text())
        model = cls(config)
        weights = mx.load(str(root / "model.safetensors"))
        keep = {}
        for key, value in weights.items():
            if not key.startswith("encoder."):
                continue
            if ".quantizer." in key and not key.endswith("._codebook.embed"):
                continue
            if key.endswith(
                ("conv1.weight", "conv2.weight", "down_sample_layer.0.weight")
            ):
                # PyTorch Conv1d OIK -> MLX OKI.
                value = value.transpose(0, 2, 1)
            keep[key] = value
        model.load_weights(list(keep.items()), strict=False)
        mx.eval(model.parameters())
        model.eval()
        return model


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = [
            nn.Linear(4096, 16384, bias=False),
            nn.GELU(),
            nn.Linear(16384, 4096, bias=False),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp[2](nn.gelu(self.mlp[0](x)))


class _AudioEncoderContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_local_transformer = _LocalTransformer()
        self.projection = _Projection()


class _LocalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        from mlx_lm.models.qwen2 import ModelArgs, TransformerBlock

        args = ModelArgs(
            model_type="qwen2",
            hidden_size=1024,
            num_hidden_layers=6,
            intermediate_size=4096,
            num_attention_heads=16,
            num_key_value_heads=16,
            rms_norm_eps=1e-6,
            vocab_size=1,
            max_position_embeddings=1048576,
            rope_theta=640000.0,
            tie_word_embeddings=False,
        )
        self.layers = [TransformerBlock(args) for _ in range(6)]
        self.norm = nn.RMSNorm(1024, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MiMoAudioBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.speech_embeddings = [nn.Embedding(1280, 1024) for _ in range(20)]
        self.audio_encoder = _AudioEncoderContainer()

    def __call__(self, grouped_codes: mx.array) -> mx.array:
        # grouped_codes: [audio tokens, group=4, channels=20]
        x = mx.zeros(
            (grouped_codes.shape[0], grouped_codes.shape[1], 1024),
            dtype=self.speech_embeddings[0].weight.dtype,
        )
        for channel, embedding in enumerate(self.speech_embeddings):
            x = x + embedding(grouped_codes[:, :, channel])
        x = self.audio_encoder.input_local_transformer(x)
        return self.audio_encoder.projection(x.reshape(x.shape[0], -1))

    @classmethod
    def load(cls, path: str | Path) -> MiMoAudioBridge:
        model = cls()
        weights = mx.load(str(path))
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        model.eval()
        return model


def mel_spectrogram(audio: np.ndarray, config: Any) -> mx.array:
    """Match MiMo-Audio-Tokenizer's torchaudio MelSpectrogram + natural log."""
    from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

    n_fft = int(config.nfft)
    sample_rate = int(config.sampling_rate)
    filters = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=int(config.n_mels),
        min_frequency=float(config.fmin),
        max_frequency=float(config.fmax)
        if config.fmax is not None
        else sample_rate / 2,
        sampling_rate=sample_rate,
        norm=None,
        mel_scale="htk",
    )
    window = window_function(int(config.window_size), "hann", periodic=True)
    mel = spectrogram(
        np.asarray(audio, dtype=np.float32),
        window,
        frame_length=int(config.window_size),
        hop_length=int(config.hop_length),
        fft_length=n_fft,
        power=1.0,
        center=True,
        pad_mode="reflect",
        mel_filters=filters,
        log_mel=None,
    )
    return mx.array(np.log(np.clip(mel, 1e-7, None)).astype(np.float32))


def group_audio_codes(
    codes: mx.array, group_size: int = 4, zero_index: int = 1024
) -> mx.array:
    if codes.ndim != 2 or codes.shape[1] != 20:
        raise ValueError(f"MiMo audio codes must have shape [T, 20], got {codes.shape}")
    remainder = codes.shape[0] % group_size
    if remainder:
        codes = mx.pad(
            codes, [(0, group_size - remainder), (0, 0)], constant_values=zero_index
        )
    return codes.reshape(-1, group_size, 20)


class MiMoAudioProcessor:
    """Add MiMo audio-code generation to the existing Qwen vision processor."""

    supports_multiple_audio = True

    def __init__(self, base: Any, tokenizer_root: str | Path):
        self.base = base
        self.tokenizer = base.tokenizer
        self.image_processor = base.image_processor
        self.video_processor = getattr(base, "video_processor", None)
        self.detokenizer = getattr(base, "detokenizer", None)
        self.feature_extractor = SimpleNamespace(sampling_rate=24000)
        self._tokenizer_root = Path(tokenizer_root)
        self._audio_tokenizer = None

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def _load_audio_tokenizer(self) -> MiMoAudioTokenizer:
        if self._audio_tokenizer is None:
            self._audio_tokenizer = MiMoAudioTokenizer.load(self._tokenizer_root)
        return self._audio_tokenizer

    def __call__(
        self,
        text,
        images=None,
        audios=None,
        padding=True,
        return_tensors="mlx",
        add_special_tokens=False,
        **kwargs,
    ):
        if not audios:
            return self.base(
                text=text,
                images=images,
                padding=padding,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )
        tokenizer = self._load_audio_tokenizer()
        grouped = []
        counts = []
        for audio in audios:
            mel = mel_spectrogram(np.asarray(audio), tokenizer.config)
            codes = tokenizer.encoder.encode(mel)
            current = group_audio_codes(codes)
            mx.eval(current)
            grouped.append(current)
            counts.append(current.shape[0])
        prompts = [text] if isinstance(text, str) else list(text)
        if len(prompts) != 1:
            raise ValueError("MiMo audio currently supports one prompt per request")
        expanded = prompts[0]
        marker = "<|audio_pad|>"
        parts = expanded.split(marker)
        if len(parts) - 1 != len(counts):
            raise ValueError(
                "MiMo audio prompt placeholder count does not match its audio inputs"
            )
        expanded = parts[0] + "".join(
            marker * count + parts[index + 1] for index, count in enumerate(counts)
        )
        result = self.base(
            text=[expanded],
            images=images,
            padding=padding,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        result = dict(result)
        result["audio_codes"] = mx.concatenate(grouped, axis=0)
        return result
