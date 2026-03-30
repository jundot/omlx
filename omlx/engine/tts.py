# SPDX-License-Identifier: Apache-2.0
"""
TTS (Text-to-Speech) engine for oMLX.

This module provides an engine for speech synthesis using mlx-audio.
Unlike LLM engines, TTS engines don't support streaming or chat completion.
mlx-audio is imported lazily inside start() to avoid module-level import errors
when mlx-audio is not installed.
"""

import asyncio
import gc
import logging
from typing import Any, Dict, Optional

import mlx.core as mx
import numpy as np

from ..engine_core import get_mlx_executor
from .audio_utils import DEFAULT_SAMPLE_RATE as _DEFAULT_SAMPLE_RATE
from .audio_utils import audio_to_wav_bytes as _audio_to_wav_bytes
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


class TTSEngine(BaseNonStreamingEngine):
    """
    Engine for speech synthesis (Text-to-Speech).

    This engine wraps mlx-audio TTS models and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support streaming or chat
    since synthesis is computed in a single forward pass.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the TTS engine.

        Args:
            model_name: HuggingFace model name or local path
            **kwargs: Additional model-specific parameters
        """
        self._model_name = model_name
        self._model = None
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

        logger.info(f"Starting TTS engine: {self._model_name}")

        try:
            from mlx_audio.tts.utils import load_model as _load_model
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is required for TTS inference. "
                "Install it with: pip install mlx-audio"
            ) from exc

        model_name = self._model_name

        def _load_sync():
            try:
                return _load_model(model_name, strict=True)
            except ValueError as exc:
                if "Expected shape" not in str(exc):
                    raise
                # mlx-audio bug: sanitize() merges quantization scales into
                # weights before apply_quantization() can detect them, causing
                # shape mismatches for quantized models (e.g. VibeVoice 8-bit).
                # Retry with strict=False so mismatched layers are skipped.
                logger.warning(
                    "Strict weight loading failed for %s (likely quantized "
                    "model with mlx-audio compatibility issue), retrying "
                    "with strict=False: %s", model_name, exc,
                )
                return _load_model(model_name, strict=False)

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        logger.info(f"TTS engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping TTS engine: {self._model_name}")
        self._model = None

        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info(f"TTS engine stopped: {self._model_name}")

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice: Optional voice/speaker identifier
            speed: Speech speed multiplier (1.0 = normal)
            **kwargs: Additional model-specific parameters

        Returns:
            WAV-encoded bytes (RIFF header + 16-bit mono PCM)
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        import time

        logger.info(
            "TTS synthesize: model=%s, text_len=%d, voice=%s, speed=%.1f",
            self._model_name, len(text), voice, speed,
        )

        model = self._model
        t0 = time.monotonic()

        def _synthesize_sync():
            # model.generate() returns an iterable of results,
            # each with .audio (array) and .sample_rate (int).
            gen_kwargs: Dict[str, Any] = {
                "text": text,
                "verbose": False,
            }
            if voice is not None:
                # Route the voice value to the correct generate() parameter.
                # Models like CustomVoice accept a speaker name via 'voice';
                # VoiceDesign-only models accept a description via 'instruct'.
                # When both exist, 'voice' takes priority (CustomVoice).
                import inspect
                gen_params = inspect.signature(model.generate).parameters
                if "voice" in gen_params:
                    gen_kwargs["voice"] = voice
                elif "instruct" in gen_params:
                    gen_kwargs["instruct"] = voice
            if speed != 1.0:
                gen_kwargs["speed"] = speed
            gen_kwargs.update(kwargs)

            results = model.generate(**gen_kwargs)
            audio_chunks = []
            sample_rate = _DEFAULT_SAMPLE_RATE

            for result in results:
                audio_chunks.append(np.array(result.audio))
                if hasattr(result, "sample_rate"):
                    sample_rate = result.sample_rate

            if not audio_chunks:
                raise RuntimeError("TTS model produced no audio output")

            audio = np.concatenate(audio_chunks, axis=0)
            return _audio_to_wav_bytes(audio, int(sample_rate))

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            get_mlx_executor(), _synthesize_sync
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "TTS synthesize done: model=%s, %.2fs, %d bytes output",
            self._model_name, elapsed, len(result),
        )
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<TTSEngine model={self._model_name} status={status}>"
