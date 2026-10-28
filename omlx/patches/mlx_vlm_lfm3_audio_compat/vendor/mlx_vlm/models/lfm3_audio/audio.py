"""Speech to backbone embeddings, as in the reference ``Lfm3AudioFeatureExtractor``,
``Lfm3AudioEncoder`` and ``Lfm3AudioProjector``.

A 16 kHz mono clip -> log-mel features (128 Slaney bins every 10 ms) -> three
stride-2 convolutions over the (mel, time) plane -> 17-layer LFM2 trunk, causal
with a 32-key window -> projector: one backbone embedding per 8 feature frames.
"""

from dataclasses import dataclass
from typing import Any, Dict, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import BaseModelConfig
from ..cache import create_causal_mask
from ..lfm2.config import ModelConfig as Lfm2Config
from ..lfm2.language import Lfm2DecoderLayer

SAMPLE_RATE, N_FFT, HOP, NUM_MEL_BINS, LOG_FLOOR = 16000, 400, 160, 128, -12.0


def num_frames(num_samples: int) -> int:
    """Feature frames of a clip: it is right-padded to 400 + k * 160 samples."""
    return 1 + max(0, -(-(num_samples - N_FFT) // HOP))


def num_encoder_tokens(num_frames: int) -> int:
    return -(num_frames // -8)


_MEL_FILTERS = None


def log_mel(audio) -> np.ndarray:
    """One mono 16 kHz clip (N,) -> float32 features (128, num_frames(N))."""
    global _MEL_FILTERS
    if _MEL_FILTERS is None:
        from mlx_audio.dsp import mel_filters

        _MEL_FILTERS = np.array(
            mel_filters(
                SAMPLE_RATE,
                N_FFT,
                NUM_MEL_BINS,
                norm="slaney",
                mel_scale="slaney",
                precise=True,
            ),
            dtype=np.float64,
        )
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected one mono clip, got shape {x.shape}")
    T = num_frames(len(x))
    x = np.pad(x.astype(np.float64), (0, N_FFT + (T - 1) * HOP - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    with np.errstate(divide="ignore"):
        mel = np.maximum(np.log10(_MEL_FILTERS @ power.T), LOG_FLOOR)
    return ((mel + 4) / 4).astype(np.float32)


def extract_features(clips):
    """Mono 16 kHz clips -> (input_features (B, 128, T) zero-padded to the longest
    clip, feature_attention_mask (B, T)), both numpy, as the reference returns."""
    features = [log_mel(clip) for clip in clips]
    lengths = [f.shape[-1] for f in features]
    out = np.zeros((len(features), NUM_MEL_BINS, max(lengths)), np.float32)
    mask = np.zeros((len(features), max(lengths)), np.int32)
    for i, f in enumerate(features):
        out[i, :, : lengths[i]] = f
        mask[i, : lengths[i]] = 1
    return out, mask


def _fp32(act, x):
    # torch evaluates activations of bf16 inputs in fp32 and rounds once; a bf16
    # erf alone doubles the projector's bf16 error
    return act(x.astype(mx.float32)).astype(x.dtype)


@dataclass
class AudioEncoderConfig(BaseModelConfig):
    text_config: Union[Dict[str, Any], Lfm2Config]
    conv_channels: int = 256
    hidden_size: int = 768
    num_mel_bins: int = NUM_MEL_BINS
    sliding_window: int = 32

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = Lfm2Config.from_dict(
                {**self.text_config, "block_dim": self.text_config["hidden_size"]}
            )


class Downsampler(nn.Module):
    """Convolutions over NHWC (B, mel, time, C); 8 frames of 128 bins -> 1 vector."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        c = config.conv_channels
        self.conv_in = nn.Conv2d(1, c, 3, stride=2, padding=1)
        self.conv_mid = nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c)
        self.lin_mid = nn.Conv2d(c, c, 1)
        self.conv_end = nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c)
        self.lin_end = nn.Conv2d(c, c, 1)
        self.lin_out = nn.Linear(config.num_mel_bins * c // 8, config.hidden_size)

    def __call__(self, x, lengths):
        def mask(x, n):
            return x * (mx.arange(x.shape[2]) < n[:, None])[:, None, :, None]

        x = mask(x[..., None].astype(self.conv_in.weight.dtype), lengths)
        x = _fp32(nn.silu, self.conv_in(x))
        lengths = (lengths + 1) // 2
        x = mask(x, lengths)
        x = _fp32(nn.silu, self.lin_mid(self.conv_mid(x)))
        lengths = (lengths + 1) // 2
        x = mask(x, lengths)
        x = _fp32(nn.silu, self.lin_end(self.conv_end(x)))
        B, F, T, C = x.shape
        # the reference flattens (channel, mel) channel-major
        return self.lin_out(x.transpose(0, 2, 3, 1).reshape(B, T, C * F))


class Trunk(nn.Module):
    def __init__(self, config: Lfm2Config):
        super().__init__()
        self.layers = [
            Lfm2DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, h, window):
        mask = create_causal_mask(h.shape[1], window_size=window)
        for layer in self.layers:
            h = layer(h, mask if layer.is_attention_layer else None)
        return self.embedding_norm(h)


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.downsample = Downsampler(config)
        self.lfm = Trunk(config.text_config)

    def __call__(self, input_features, lengths):
        """``input_features`` (B, 128, T), ``lengths`` (B,) valid frames per clip ->
        the valid rows of every clip, in order: (sum ceil(lengths / 8), hidden)."""
        lengths = mx.array(lengths)
        x = self.downsample(mx.array(input_features), lengths)
        # padded rows trail each clip and the trunk is causal, so they change nothing
        h = self.lfm(x, self.config.sliding_window)
        n = num_encoder_tokens(np.array(lengths)).tolist()
        return mx.concatenate([h[i, :k] for i, k in enumerate(n)])


class AudioProjector(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features, eps=1e-5)
        self.linear_1 = nn.Linear(in_features, out_features)
        self.linear_2 = nn.Linear(out_features, out_features)

    def __call__(self, x):
        return self.linear_2(_fp32(nn.gelu, self.linear_1(self.layer_norm(x))))


def sanitize(weights):
    """torch conv layouts under ``audio_tower.`` -> MLX; other keys untouched; idempotent."""
    out = {}
    for key, value in weights.items():
        if key.startswith("audio_tower.downsample."):
            # (O, I, kh, kw) -> (O, kh, kw, I); kernels are 3x3 or 1x1
            if value.ndim == 4 and value.shape[2] == value.shape[3]:
                value = value.transpose(0, 2, 3, 1)
            elif value.ndim == 3:  # lin_out Conv1d (O, I, 1) -> Linear
                value = value[..., 0]
        elif key.startswith("audio_tower.lfm.") and key.endswith("conv.conv.weight"):
            if value.shape[-1] > value.shape[1]:
                value = value.transpose(0, 2, 1)
        out[key] = value
    return out
