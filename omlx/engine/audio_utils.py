# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for audio engines (STT, TTS, STS)."""

import io
import wave

import numpy as np


# Default sample rate used when the model does not report one.
DEFAULT_SAMPLE_RATE = 24000

# OpenAI-compatible response formats supported by omlx TTS.
# Formats that require ffmpeg are only available if it is installed.
SUPPORTED_RESPONSE_FORMATS = {"wav", "pcm", "mp3", "opus", "flac", "aac"}


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


def _wav_bytes_to_pcm_int16(wav_bytes: bytes) -> bytes:
    """Extract raw 16-bit PCM samples from WAV bytes (strips RIFF header)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.readframes(wf.getnframes())


def _wav_to_format_via_ffmpeg(wav_bytes: bytes, fmt: str) -> bytes:
    """Convert WAV bytes to another format using ffmpeg subprocess.

    Raises RuntimeError if ffmpeg is not found or conversion fails.
    """
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            f"ffmpeg is required to encode audio as '{fmt}' but was not found. "
            "Install ffmpeg (e.g. 'brew install ffmpeg') or request 'wav' format."
        )

    codec_map = {
        "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2"],
        "opus": ["-f", "ogg", "-codec:a", "libopus", "-b:a", "96k"],
        "flac": ["-f", "flac", "-codec:a", "flac"],
        "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"],
    }
    if fmt not in codec_map:
        raise ValueError(f"Unsupported audio format: {fmt!r}")

    ffmpeg_args = codec_map[fmt]
    cmd = [
        "ffmpeg", "-y",
        "-f", "wav", "-i", "pipe:0",
        *ffmpeg_args,
        "pipe:1",
    ]
    result = subprocess.run(cmd, input=wav_bytes, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg encoding to '{fmt}' failed: {result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def convert_wav_to_response_format(
    wav_bytes: bytes,
    response_format: str,
) -> tuple[bytes, str]:
    """Convert WAV bytes to the requested OpenAI-compatible response format.

    Args:
        wav_bytes: Input audio as WAV-encoded bytes.
        response_format: One of 'wav', 'pcm', 'mp3', 'opus', 'flac', 'aac'.

    Returns:
        Tuple of (audio_bytes, media_type).

    Raises:
        ValueError: If response_format is unknown.
        RuntimeError: If the conversion requires ffmpeg and it is not installed.
    """
    fmt = (response_format or "wav").lower()

    if fmt == "wav":
        return wav_bytes, "audio/wav"

    if fmt == "pcm":
        # Raw 16-bit little-endian PCM (OpenAI default: 24 kHz, mono, s16le)
        return _wav_bytes_to_pcm_int16(wav_bytes), "audio/pcm"

    if fmt in ("mp3", "opus", "flac", "aac"):
        audio_bytes = _wav_to_format_via_ffmpeg(wav_bytes, fmt)
        media_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "flac": "audio/flac",
            "aac": "audio/aac",
        }
        return audio_bytes, media_types[fmt]

    raise ValueError(
        f"Unsupported response_format '{fmt}'. "
        f"Supported: {sorted(SUPPORTED_RESPONSE_FORMATS)}"
    )
