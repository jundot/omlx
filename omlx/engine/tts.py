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
from collections.abc import AsyncIterator
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
        super().__init__()
        self._model_name = model_name
        self._model = None
        self._kwargs = kwargs

    @staticmethod
    def _audio_array_to_pcm_bytes(audio: Any) -> bytes:
        audio_array = np.array(audio).flatten()
        audio_array = np.clip(audio_array, -1.0, 1.0)
        return (audio_array * 32767).astype(np.int16).tobytes()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    def supports_native_tts_streaming(self) -> bool:
        """Return whether the loaded model exposes model-native audio streaming."""
        if self._model is None:
            return False
        import inspect

        try:
            gen_params = inspect.signature(self._model.generate).parameters
        except (TypeError, ValueError):
            return False
        return "stream" in gen_params and "streaming_interval" in gen_params

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
                'Install it with: pip install "omlx[audio]"'
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
        instructions: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice: Optional voice/speaker identifier
            speed: Speech speed multiplier (1.0 = normal)
            instructions: Optional voice description for instruct-capable models
            ref_audio: Optional path to reference audio file (voice cloning)
            ref_text: Optional transcript of the reference audio
            temperature: Sampling temperature for generation
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty for generation
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            WAV-encoded bytes (RIFF header + 16-bit mono PCM)
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        import time

        logger.info(
            "TTS synthesize: model=%s, text_len=%d, voice=%s, speed=%.1f, ref_audio=%s",
            self._model_name, len(text), voice, speed,
            "yes" if ref_audio else "no",
        )

        model = self._model
        t0 = time.monotonic()

        def _build_generate_kwargs() -> Dict[str, Any]:
            gen_kwargs: Dict[str, Any] = {
                "text": text,
                "verbose": False,
            }
            import inspect
            gen_params = inspect.signature(model.generate).parameters
            if voice is not None:
                # Route voice to the correct generate() kwarg.
                # Models with 'voice' param (CustomVoice, Kokoro) get it as
                # a speaker name. Models with only 'instruct' (non-Qwen TTS)
                # get it as a voice description fallback.
                if "voice" in gen_params:
                    gen_kwargs["voice"] = voice
                elif "instruct" in gen_params:
                    gen_kwargs["instruct"] = voice
            if instructions is not None and "instruct" in gen_params:
                gen_kwargs["instruct"] = instructions
            if speed != 1.0:
                gen_kwargs["speed"] = speed
            if ref_audio is not None and "ref_audio" in gen_params:
                gen_kwargs["ref_audio"] = ref_audio
                gen_kwargs["ref_text"] = ref_text
            # Generation params (only add non-None values)
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            if max_tokens is not None:
                gen_kwargs["max_tokens"] = max_tokens
            gen_kwargs.update(kwargs)
            return gen_kwargs

        def _synthesize_sync():
            # model.generate() returns an iterable of results,
            # each with .audio (array) and .sample_rate (int).
            gen_kwargs = _build_generate_kwargs()

            results = model.generate(**gen_kwargs)

            # Use model.sample_rate if available (e.g. Qwen3-TTS)
            sample_rate = getattr(model, "sample_rate", _DEFAULT_SAMPLE_RATE)
            audio_chunks = []

            for result in results:
                audio_chunks.append(np.array(result.audio))

            if not audio_chunks:
                raise RuntimeError("TTS model produced no audio output")

            audio = np.concatenate(audio_chunks, axis=0)
            return _audio_to_wav_bytes(audio, int(sample_rate))

        activity_id = self._begin_activity(
            "synthesizing speech",
            detail="Synthesizing speech",
            metadata={"text_length": len(text)},
        )
        try:
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
        finally:
            if self._end_activity(activity_id):
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )

    async def stream_synthesize_pcm(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        instructions: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        streaming_interval: float = 0.4,
        **kwargs,
    ) -> AsyncIterator[tuple[int, int, int, bytes]]:
        """Stream synthesized PCM chunks from models that natively support it."""
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")
        if not self.supports_native_tts_streaming():
            raise NotImplementedError("Loaded TTS model does not expose native streaming")

        import inspect
        import time

        logger.info(
            "TTS native stream start: model=%s, text_len=%d, voice=%s, interval=%.2fs",
            self._model_name, len(text), voice, streaming_interval,
        )

        model = self._model
        t0 = time.monotonic()

        def _build_generate_kwargs() -> Dict[str, Any]:
            gen_kwargs: Dict[str, Any] = {
                "text": text,
                "verbose": False,
                "stream": True,
            }
            gen_params = inspect.signature(model.generate).parameters
            if "streaming_interval" in gen_params:
                gen_kwargs["streaming_interval"] = streaming_interval
            if voice is not None:
                if "voice" in gen_params:
                    gen_kwargs["voice"] = voice
                elif "instruct" in gen_params:
                    gen_kwargs["instruct"] = voice
            if instructions is not None and "instruct" in gen_params:
                gen_kwargs["instruct"] = instructions
            if speed != 1.0:
                gen_kwargs["speed"] = speed
            if ref_audio is not None and "ref_audio" in gen_params:
                gen_kwargs["ref_audio"] = ref_audio
                gen_kwargs["ref_text"] = ref_text
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            if max_tokens is not None:
                gen_kwargs["max_tokens"] = max_tokens
            gen_kwargs.update(kwargs)
            return gen_kwargs

        iterator: Any = None
        sentinel = object()
        chunk_count = 0
        total_bytes = 0

        def _next_pcm_chunk():
            nonlocal iterator
            if iterator is None:
                iterator = iter(model.generate(**_build_generate_kwargs()))
            try:
                result = next(iterator)
            except StopIteration:
                return sentinel
            audio = getattr(result, "audio", None)
            if audio is None:
                return None
            sample_rate = int(
                getattr(result, "sample_rate", getattr(model, "sample_rate", _DEFAULT_SAMPLE_RATE))
            )
            return sample_rate, 1, 2, self._audio_array_to_pcm_bytes(audio)

        activity_id = self._begin_activity(
            "streaming speech",
            detail="Streaming speech",
            metadata={"text_length": len(text)},
        )
        try:
            loop = asyncio.get_running_loop()
            while True:
                chunk = await loop.run_in_executor(get_mlx_executor(), _next_pcm_chunk)
                if chunk is sentinel:
                    break
                if chunk is None:
                    continue
                sample_rate, channels, sample_width, pcm_bytes = chunk
                if not pcm_bytes:
                    continue
                chunk_count += 1
                total_bytes += len(pcm_bytes)
                self._update_activity(
                    activity_id,
                    chunk_count=chunk_count,
                    output_bytes=total_bytes,
                )
                yield sample_rate, channels, sample_width, pcm_bytes
        finally:
            if self._end_activity(activity_id):
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
                )
            logger.info(
                "TTS native stream done: model=%s, %.2fs, chunks=%d, pcm_bytes=%d",
                self._model_name, time.monotonic() - t0, chunk_count, total_bytes,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<TTSEngine model={self._model_name} status={status}>"
