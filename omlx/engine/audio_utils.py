# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for audio engines (STT, TTS, STS)."""

import io
import wave

import numpy as np


# Default sample rate used when the model does not report one.
DEFAULT_SAMPLE_RATE = 24000


def audio_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    """Convert a float32 audio array to 16-bit PCM WAV bytes.

    Args:
        audio_array: numpy or mlx array of float32 samples in [-1, 1]
        sample_rate: audio sample rate in Hz

    Returns:
        WAV-encoded bytes (RIFF header + PCM data)
    """
    # Ensure we have a numpy array for the wave module
    if not isinstance(audio_array, np.ndarray):
        # NumPy doesn't support bfloat16 — cast to float32 first
        if hasattr(audio_array, "dtype"):
            import mlx.core as mx

            if audio_array.dtype == mx.bfloat16:
                audio_array = audio_array.astype(mx.float32)
        audio_array = np.array(audio_array)

    # Flatten to 1-D (mono)
    audio_array = audio_array.flatten()

    # Clip to [-1, 1] then convert to int16
    audio_array = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_array * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()
