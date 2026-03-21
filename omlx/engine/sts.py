# SPDX-License-Identifier: Apache-2.0
"""
STS (Speech-to-Speech) engine for oMLX.

This module provides an engine for audio processing (speech enhancement,
speech-to-speech conversion) using mlx-audio.
Unlike LLM engines, STS engines don't support streaming or chat completion.
mlx-audio is imported lazily inside start() to avoid module-level import errors
when mlx-audio is not installed.

Supported model families:
- DeepFilterNet: speech enhancement / noise removal
- MossFormer2: speech enhancement
- Moshi: full-duplex STS (kyutai/moshiko-mlx-*)
- LFM2.5-Audio: multimodal STS+TTS+STT (mlx-community/LFM2.5-Audio-*)
"""

import asyncio
import gc
import io
import logging
import wave
from typing import Any, Dict

import mlx.core as mx
import numpy as np

from ..engine_core import get_mlx_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

# Default sample rate used when the model does not report one.
_DEFAULT_SAMPLE_RATE = 22050


def _detect_sts_family(model_name: str) -> str:
    """Detect STS model family from model name/path.

    Returns one of: "deepfilternet", "mossformer2", "moshi", "lfm2", "generic"
    """
    name_lower = model_name.lower()
    if "deepfilter" in name_lower:
        return "deepfilternet"
    if "mossformer" in name_lower:
        return "mossformer2"
    if "moshi" in name_lower:
        return "moshi"
    if "lfm" in name_lower:
        return "lfm2"
    return "generic"


def _audio_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    """Convert a float32 audio array to 16-bit PCM WAV bytes.

    Args:
        audio_array: numpy or mlx array of float32 samples in [-1, 1]
        sample_rate: audio sample rate in Hz

    Returns:
        WAV-encoded bytes (RIFF header + PCM data)
    """
    # Ensure we have a numpy array for the wave module
    if not isinstance(audio_array, np.ndarray):
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


class STSEngine(BaseNonStreamingEngine):
    """
    Engine for speech-to-speech / audio processing (Speech-to-Speech).

    This engine wraps mlx-audio STS models and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support streaming or chat
    since audio processing is computed in a single forward pass.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the STS engine.

        Args:
            model_name: HuggingFace model name or local path
            **kwargs: Additional model-specific parameters
        """
        self._model_name = model_name
        self._model = None
        self._family = _detect_sts_family(model_name)
        self._kwargs = kwargs

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    async def start(self) -> None:
        """Start the engine (load model if not loaded).

        Model loading runs on the global MLX executor to avoid Metal
        command buffer races with concurrent BatchGenerator steps.
        mlx-audio is imported here (lazily) to avoid module-level errors
        when the package is not installed.
        """
        if self._model is not None:
            return

        logger.info(f"Starting STS engine: {self._model_name} (family={self._family})")

        try:
            from mlx_audio.sts.utils import load_model as _load_model
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is required for STS inference. "
                "Install it with: pip install mlx-audio"
            ) from exc

        model_name = self._model_name

        def _load_sync():
            return _load_model(model_name)

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        logger.info(f"STS engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping STS engine: {self._model_name}")
        self._model = None

        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info(f"STS engine stopped: {self._model_name}")

    async def process(self, audio_path: str, **kwargs) -> bytes:
        """
        Process an audio file through the STS model.

        For speech enhancement models (DeepFilterNet, MossFormer2), this
        enhances / denoises the audio. For Moshi/LFM2, this runs full
        speech-to-speech inference.

        Args:
            audio_path: Path to the audio file to process
            **kwargs: Additional model-specific parameters

        Returns:
            WAV-encoded bytes (RIFF header + 16-bit mono PCM) of processed audio
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        try:
            from mlx_audio.sts.utils import process as _process
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is required for STS inference. "
                "Install it with: pip install mlx-audio"
            ) from exc

        model = self._model

        def _process_sync():
            result = _process(model=model, audio=str(audio_path), **kwargs)

            # result may be (audio_array, sample_rate) or just an audio array
            if isinstance(result, tuple):
                audio_array, sample_rate = result[0], result[1]
            else:
                audio_array = result
                # Try to get sample_rate from model config
                sample_rate = getattr(
                    getattr(model, "config", None),
                    "sample_rate",
                    _DEFAULT_SAMPLE_RATE,
                )

            return _audio_to_wav_bytes(audio_array, int(sample_rate))

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_mlx_executor(), _process_sync)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "family": self._family,
        }

    def __repr__(self) -> str:
        s = "running" if self._model is not None else "stopped"
        return f"<STSEngine model={self._model_name} family={self._family} {s}>"
