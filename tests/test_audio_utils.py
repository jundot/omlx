# SPDX-License-Identifier: Apache-2.0
"""Tests for audio utility functions."""

import io
import wave

import mlx.core as mx
import numpy as np

from omlx.engine.audio_utils import audio_to_wav_bytes


def _read_wav(data: bytes):
    """Helper: parse WAV bytes and return (samples_int16, sample_rate)."""
    buf = io.BytesIO(data)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16), sr


class TestAudioToWavBytes:
    def test_numpy_float32(self):
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        wav = audio_to_wav_bytes(audio, 24000)
        samples, sr = _read_wav(wav)
        assert sr == 24000
        assert len(samples) == 5

    def test_mlx_float32(self):
        audio = mx.array([0.0, 0.5, -0.5], dtype=mx.float32)
        wav = audio_to_wav_bytes(audio, 16000)
        samples, sr = _read_wav(wav)
        assert sr == 16000
        assert len(samples) == 3

    def test_mlx_bfloat16(self):
        """bfloat16 arrays (e.g. Kokoro-82M-bf16) must not crash."""
        audio = mx.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=mx.bfloat16)
        wav = audio_to_wav_bytes(audio, 24000)
        samples, sr = _read_wav(wav)
        assert sr == 24000
        assert len(samples) == 5
        # Verify values are reasonable after bf16 → f32 → int16 conversion
        assert samples[0] == 0
        assert samples[3] == 32767  # 1.0 clipped
        assert samples[4] == -32767  # -1.0 clipped

    def test_mlx_float16(self):
        audio = mx.array([0.0, 0.25], dtype=mx.float16)
        wav = audio_to_wav_bytes(audio, 24000)
        samples, _ = _read_wav(wav)
        assert len(samples) == 2

    def test_clipping(self):
        audio = np.array([2.0, -2.0], dtype=np.float32)
        wav = audio_to_wav_bytes(audio, 24000)
        samples, _ = _read_wav(wav)
        assert samples[0] == 32767
        assert samples[1] == -32767
