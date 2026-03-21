# SPDX-License-Identifier: Apache-2.0
"""
STT (Speech-to-Text) engine for oMLX.

This module provides an engine for audio transcription using mlx-audio.
Unlike LLM engines, STT engines don't support streaming or chat completion.
mlx-audio is imported lazily inside start() to avoid module-level import errors
when mlx-audio is not installed.
"""

import asyncio
import gc
import logging
from typing import Any, Dict, Optional

import mlx.core as mx

from ..engine_core import get_mlx_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


class STTEngine(BaseNonStreamingEngine):
    """
    Engine for audio transcription (Speech-to-Text).

    This engine wraps mlx-audio STT models and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support streaming or chat
    since transcription is computed in a single forward pass.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the STT engine.

        Args:
            model_name: HuggingFace model name or local path
            **kwargs: Additional model-specific parameters
        """
        self._model_name = model_name
        self._model = None
        self._processor = None
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

        logger.info(f"Starting STT engine: {self._model_name}")

        try:
            from mlx_audio.stt.utils import load_model as _load_model
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is required for STT inference. "
                "Install it with: pip install mlx-audio"
            ) from exc

        model_name = self._model_name

        def _load_sync():
            model, processor = _load_model(model_name)
            return model, processor

        loop = asyncio.get_running_loop()
        self._model, self._processor = await loop.run_in_executor(
            get_mlx_executor(), _load_sync
        )
        logger.info(f"STT engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping STT engine: {self._model_name}")
        self._model = None
        self._processor = None

        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info(f"STT engine stopped: {self._model_name}")

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language code (e.g. 'en', 'fr')
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary with keys:
                text: Transcribed text
                language: Detected or specified language
                segments: List of timed segments (may be empty)
                duration: Audio duration in seconds
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        try:
            from mlx_audio.stt.utils import transcribe as _transcribe
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is required for STT inference. "
                "Install it with: pip install mlx-audio"
            ) from exc

        model = self._model
        processor = self._processor

        def _transcribe_sync():
            result = _transcribe(
                model=model,
                processor=processor,
                audio=str(audio_path),
                language=language,
                **kwargs,
            )
            # Normalise result to a dict with our expected keys
            if isinstance(result, dict):
                return {
                    "text": result.get("text", ""),
                    "language": result.get("language", language or ""),
                    "segments": result.get("segments", []),
                    "duration": result.get("duration", 0.0),
                }
            # Some versions return a plain string
            return {
                "text": str(result),
                "language": language or "",
                "segments": [],
                "duration": 0.0,
            }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(get_mlx_executor(), _transcribe_sync)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<STTEngine model={self._model_name} status={status}>"
