# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the Liquid codec reference implementation.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ..base import BaseModelConfig
from ..lfm3.config import ModelConfig as Lfm3Config
from ..lfm3.language import Lfm3Model


@dataclass
class CodecConfig(BaseModelConfig):
    encoder_dim: int = 512
    quantizer_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [8, 7, 6, 6])
    n_q: int = 12
    stft_n_fft: int = 1280
    stft_hop_length: int = 320
    stft_win_length: int = 1280
    stft_frame_stack: int = 6
    decoder_window: int = 30
    sampling_rate: int = 24000

    def __post_init__(self):
        if self.stft_win_length != self.stft_n_fft:
            raise ValueError("Liquid codec requires stft_win_length == stft_n_fft")
        if self.stft_n_fft % self.stft_hop_length:
            raise ValueError("Liquid codec requires stft_n_fft % stft_hop_length == 0")

    @property
    def codebook_size(self):
        return math.prod(self.fsq_levels)

    def lfm_config(self):
        # The backbone shape is fixed by the reference config class, not config.json.
        return Lfm3Config(
            hidden_size=self.encoder_dim,
            intermediate_size=2304,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=1,
            layer_types=[
                "sliding_attention" if i in (2, 4, 6) else "conv" for i in range(8)
            ],
            rope_parameters={"sliding_attention": {"rope_theta": 1e6}},
            sliding_window=self.decoder_window,
            tie_word_embeddings=False,
        )


class Quantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        levels = config.fsq_levels
        self.levels = levels
        self.basis = [math.prod(levels[:i]) for i in range(len(levels))]
        self.project_out = nn.Linear(
            config.n_q * len(levels), config.quantizer_dim, bias=True
        )

    def __call__(self, codes):
        # FSQ dequantization: [B, n_q, T] codes -> [B, T, n_q * len(levels)].
        levels = mx.array(self.levels)
        half = (levels // 2).astype(mx.float32)
        digits = (codes.transpose(0, 2, 1)[..., None] // mx.array(self.basis)) % levels
        latents = (digits.astype(mx.float32) - half) / half
        latents = latents.reshape(codes.shape[0], codes.shape[2], -1)
        return self.project_out(latents.astype(self.project_out.weight.dtype))


class ISTFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_fft = config.stft_n_fft
        self.hop = config.stft_hop_length
        self.window = mx.zeros((config.stft_win_length,), dtype=mx.float32)

    def _overlap_add(self, frames):
        batch, n, _ = frames.shape
        k = self.n_fft // self.hop
        blocks = frames.reshape(batch, n, k, self.hop)
        out = sum(
            mx.pad(blocks[:, :, j], [(0, 0), (j, k - 1 - j), (0, 0)]) for j in range(k)
        )
        return out.reshape(batch, -1)

    def __call__(self, spec):
        # "same" padding: trim (win - hop) / 2 from both ends and divide by the window envelope.
        window = self.window.astype(mx.float32)
        frames = mx.fft.irfft(spec, n=self.n_fft, axis=-1) * window
        audio = self._overlap_add(frames)
        envelope = self._overlap_add(
            mx.broadcast_to(window * window, (1, spec.shape[1], window.shape[0]))
        )
        trim = (self.n_fft - self.hop) // 2
        return audio[:, trim:-trim] / envelope[:, trim:-trim]


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stack = config.stft_frame_stack
        self.lfm = Lfm3Model(config.lfm_config())
        self.lfm.embed_tokens = None
        self.lin = nn.Linear(config.encoder_dim, config.stft_n_fft + 2)
        self.istft = ISTFT(config)

    def __call__(self, x):
        h = self.lfm(input_embeddings=mx.repeat(x, self.stack, axis=1))
        log_mag, angle = mx.split(self.lin(h).astype(mx.float32), 2, axis=-1)
        mag = mx.exp(log_mag)
        return self.istft(mag * mx.cos(angle) + 1j * (mag * mx.sin(angle)))


class LiquidCodec(nn.Module):
    """Decode-only Liquid codec: FSQ codes [B, n_q, T] -> 24 kHz waveform."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.quantizer = Quantizer(config)
        self.from_quantizer = nn.Linear(config.quantizer_dim, config.encoder_dim)
        self.decoder = Decoder(config)

    def __call__(self, codes):
        return self.decoder(self.from_quantizer(self.quantizer(codes)))

    def decode(self, codes):
        """codes [n_q, T] or [B, n_q, T] in [0, codebook_size) -> float32 [T * 1920] or [B, T * 1920]."""
        codes = mx.array(codes)
        squeeze = codes.ndim == 2
        codes = codes[None] if squeeze else codes
        if codes.ndim != 3 or codes.shape[1] != self.config.n_q:
            raise ValueError(
                f"Expected codes [B, {self.config.n_q}, T], got {codes.shape}"
            )
        # The LM's audio EOS (2016) and -1 padding would silently map to valid latents.
        if codes.size and (codes.min() < 0 or codes.max() >= self.config.codebook_size):
            raise ValueError(
                f"Codes must be in [0, {self.config.codebook_size}); strip EOS frames and padding"
            )
        if codes.shape[2] == 0:
            audio = mx.zeros((codes.shape[0], 0), mx.float32)
        else:
            audio = self(codes.astype(mx.int32))
        return audio[0] if squeeze else audio

    @staticmethod
    def sanitize(weights):
        result = {}
        for name, value in weights.items():
            name = name.replace("quantizer.fsq.", "quantizer.")
            # Decode-only port: drop the encoder half of the codec.
            if not name.startswith(
                ("decoder.", "from_quantizer.", "quantizer.project_out.")
            ):
                continue
            if name.endswith("conv.conv.weight") and value.shape[1] == 1:
                value = value.transpose(0, 2, 1)
            result[name] = value
        return result


def codec_path(model_path):
    model_path = Path(model_path)
    for subfolder in ("audio_tokenizer", "audio_detokenizer"):
        if (model_path / subfolder / "config.json").exists():
            return model_path / subfolder
    raise FileNotFoundError(f"No Liquid codec subfolder found in {model_path}")


def load_codec(model_path, dtype=mx.bfloat16):
    path = codec_path(model_path)
    config = CodecConfig.from_dict(json.loads((path / "config.json").read_text()))
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(file)))
    # The ISTFT window stays fp32; the reference keeps the whole ISTFT in fp32.
    weights = {
        k: v if k == "decoder.istft.window" else v.astype(dtype)
        for k, v in LiquidCodec.sanitize(weights).items()
    }
    model = LiquidCodec(config)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
