# SPDX-License-Identifier: Apache-2.0
"""
STT (Speech-to-Text) engine for oMLX.

This module provides an engine for audio transcription using mlx-audio.
Unlike LLM engines, STT engines don't support chat completion. Transcription
results can be streamed incrementally via transcribe_stream() for models
whose mlx-audio backend supports it.
mlx-audio is imported lazily inside start() to avoid module-level import errors
when mlx-audio is not installed.
"""

import asyncio
import gc
import logging
from collections.abc import AsyncIterator
from typing import Any

import mlx.core as mx

from ..engine_core import get_mlx_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


# Lowercase full-names are needed for Qwen3-ASR-style prompt builders whose
# support_languages list contains names such as "Chinese" and "English".
_ISO_TO_STT_LANG: dict[str, str] = {
    "zh": "chinese",
    "yue": "cantonese",
    "en": "english",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "ko": "korean",
    "ja": "japanese",
}


def _stt_model_expects_language_names(model: Any) -> bool:
    """Return True for STT backends whose language hints are full names."""
    config = getattr(model, "config", None)
    support_languages = getattr(config, "support_languages", None)
    if not support_languages:
        return False
    if isinstance(support_languages, str):
        support_languages = [support_languages]

    supported = {
        str(lang).strip().lower() for lang in support_languages if str(lang).strip()
    }
    return bool(supported & set(_ISO_TO_STT_LANG.values()))


def _normalize_stt_generate_language(
    model: Any,
    language: str | None,
) -> str | None:
    """Normalize API language hints for the specific mlx-audio STT backend."""
    if language is None:
        return None

    normalized = language.strip()
    if not normalized:
        return None

    if _stt_model_expects_language_names(model):
        return _ISO_TO_STT_LANG.get(normalized.lower(), normalized)
    return normalized


# ---------------------------------------------------------------------------
# Error helpers (#800): turn opaque mlx-audio/HF processor failures into
# actionable RuntimeErrors that tell users which file is missing and where
# to find a compatible variant.
# ---------------------------------------------------------------------------


_MISSING_PROCESSOR_HINTS = (
    "preprocessor_config.json",
    "feature extractor",
    "featureextractor",
)


def _looks_like_missing_processor(message: str) -> bool:
    """True if the error text from mlx-audio / HF points at a missing processor."""
    lowered = message.lower()
    return any(h in lowered for h in _MISSING_PROCESSOR_HINTS)


def _missing_processor_hint(model_name: str) -> str:
    return (
        f"STT model '{model_name}' is missing the HuggingFace processor / "
        "feature-extractor configuration (preprocessor_config.json and/or "
        "tokenizer files). MLX-converted repositories sometimes omit these. "
        "Fix: either use an HF-compatible variant of the model or copy "
        "preprocessor_config.json, tokenizer.json and special_tokens_map.json "
        "from the upstream HuggingFace repo into the local model directory."
    )


def _normalize_result_language(raw_lang: Any) -> Any:
    """Normalize the language field returned by mlx-audio backends."""
    if isinstance(raw_lang, list):
        raw_lang = raw_lang[0] if raw_lang else None
    if isinstance(raw_lang, str) and raw_lang.lower() == "none":
        return None
    return raw_lang


def _map_stt_prompt_kwargs(model: Any, prompt: str | None) -> dict[str, str]:
    """Map the OpenAI ``prompt`` field onto the backend's biasing hook.

    Qwen3-ASR-style backends expose ``generate(..., system_prompt=...)`` —
    a trained context-injection mechanism with strong biasing. Whisper-family
    backends expose ``generate(..., initial_prompt=...)`` — a decoder-prefix
    soft prior (~224-token window). Backends with neither hook ignore the
    field; a request must never fail because of ``prompt``.
    """
    if prompt is None or not prompt.strip():
        return {}

    import inspect

    try:
        params = inspect.signature(model.generate).parameters
    except (TypeError, ValueError):
        return {}

    if "system_prompt" in params:
        return {"system_prompt": prompt}
    if "initial_prompt" in params:
        return {"initial_prompt": prompt}

    logger.debug(
        "STT backend %s has no prompt-biasing hook; ignoring 'prompt'",
        type(model).__name__,
    )
    return {}


def _wrap_stt_load_error(model_name: str, exc: Exception) -> Exception:
    """Return a clearer exception for known mlx-audio STT load failures."""
    message = str(exc)
    if _looks_like_missing_processor(message):
        return RuntimeError(
            f"{_missing_processor_hint(model_name)} Original error: {message}"
        )
    return exc


def _validate_stt_processor(model_name: str, model: Any) -> None:
    """Fail fast if a Whisper-family mlx-audio model loaded without a processor."""
    module_name = type(model).__module__ or ""
    is_whisper_like = "whisper" in module_name.lower()
    if not is_whisper_like:
        return
    # mlx-audio Whisper attaches a HF processor to ``_processor``; it's set
    # to None when WhisperProcessor.from_pretrained() failed on load.
    if not hasattr(model, "_processor"):
        return
    if model._processor is not None:
        return
    raise RuntimeError(_missing_processor_hint(model_name))


class STTEngine(BaseNonStreamingEngine):
    """
    Engine for audio transcription (Speech-to-Text).

    This engine wraps mlx-audio STT models and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support chat. transcribe() computes the
    full result in one pass; transcribe_stream() yields incremental chunks
    for mlx-audio backends that expose ``generate(..., stream=True)``.
    """

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the STT engine.

        Args:
            model_name: HuggingFace model name or local path
            **kwargs: Additional model-specific parameters
        """
        super().__init__()
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

        logger.info(f"Starting STT engine: {self._model_name}")

        try:
            from mlx_audio.stt.utils import load_model as _load_model
        except ImportError as exc:
            raise ImportError(
                "mlx-audio is required for STT inference. "
                'Install it with: pip install "omlx[audio]"'
            ) from exc

        model_name = self._model_name

        def _load_sync():
            # load_model returns a single nn.Module, not a tuple
            return _load_model(model_name)

        loop = asyncio.get_running_loop()
        try:
            model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        except Exception as exc:
            # #800: MLX-packaged repos (Qwen3-ASR-*-MLX-*, some mlx-community
            # whisper variants) often omit preprocessor_config.json, which
            # mlx-audio / HuggingFace AutoFeatureExtractor reports with an
            # opaque OSError. Re-raise with an actionable message instead.
            raise _wrap_stt_load_error(model_name, exc) from exc

        # #800: Whisper models in mlx-audio load silently without a
        # HuggingFace processor when preprocessor_config.json is missing
        # (mlx-audio only emits a warning). Fail fast at start so callers
        # see the real problem instead of a downstream "Processor not found"
        # 500 during transcribe.
        _validate_stt_processor(model_name, model)

        self._model = model
        logger.info(f"STT engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping STT engine: {self._model_name}")
        self._model = None

        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info(f"STT engine stopped: {self._model_name}")

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language code (e.g. 'en', 'fr')
            prompt: Optional vocabulary / context biasing text (OpenAI
                ``prompt`` field), mapped onto the backend's biasing hook
                (Qwen3-ASR ``system_prompt``, Whisper ``initial_prompt``);
                ignored by backends without one
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

        import os
        import time

        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        logger.info(
            "STT transcribe: model=%s, file=%s (%d bytes), language=%s",
            self._model_name, os.path.basename(audio_path), file_size, language,
        )

        model = self._model
        t0 = time.monotonic()

        def _normalize_segment(s) -> dict:
            """Convert any segment type to a plain dict."""
            if isinstance(s, dict):
                return s
            # dataclass → asdict
            import dataclasses
            if dataclasses.is_dataclass(s) and not isinstance(s, type):
                return dataclasses.asdict(s)
            # object with __dict__
            if hasattr(s, "__dict__"):
                return vars(s)
            return {"text": str(s)}

        def _transcribe_sync():
            # Call model.generate() directly instead of
            # generate_transcription() which writes files to disk.
            gen_kwargs = dict(kwargs)
            generate_language = _normalize_stt_generate_language(model, language)
            if generate_language is not None:
                gen_kwargs["language"] = generate_language
            gen_kwargs.update(_map_stt_prompt_kwargs(model, prompt))

            result = model.generate(audio_path, **gen_kwargs)

            # result is typically an STTOutput dataclass with:
            # text, segments, language, total_time, etc.
            if hasattr(result, "text"):
                raw_lang = _normalize_result_language(
                    getattr(result, "language", None)
                )
                if raw_lang is None:
                    raw_lang = language

                raw_segs = getattr(result, "segments", None)
                segments = [
                    _normalize_segment(s) for s in raw_segs
                ] if raw_segs else []

                return {
                    "text": result.text or "",
                    "language": raw_lang,
                    "segments": segments,
                    "duration": getattr(
                        result, "total_time", 0.0
                    ),
                }
            # Fallback for unexpected return types
            return {
                "text": str(result),
                "language": language,
                "segments": [],
                "duration": 0.0,
            }

        activity_id = self._begin_activity(
            "transcribing",
            detail="Transcribing",
            metadata={"file_size_bytes": file_size},
        )
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                get_mlx_executor(), _transcribe_sync
            )

            elapsed = time.monotonic() - t0
            text_len = len(result.get("text", ""))
            logger.info(
                "STT transcribe done: model=%s, %.2fs, %d chars output",
                self._model_name, elapsed, text_len,
            )
            return result
        finally:
            await self._finish_activity(activity_id)

    def supports_native_stt_streaming(self) -> bool:
        """True when the loaded model's generate() accepts a ``stream`` flag.

        mlx-audio streaming-capable backends (whisper, parakeet, canary,
        qwen3-asr, ...) all expose ``generate(..., stream: bool)`` that
        returns a generator of incremental results.
        """
        if self._model is None:
            return False
        import inspect

        try:
            params = inspect.signature(self._model.generate).parameters
        except (TypeError, ValueError):
            return False
        return "stream" in params

    async def transcribe_stream(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream transcription chunks as the model decodes them.

        ``prompt`` is the OpenAI vocabulary / context biasing field, mapped
        onto the backend's biasing hook exactly as in transcribe().

        Yields dicts with keys:
            text: Incremental text delta for this chunk
            language: Detected or specified language (may be None)
            prompt_tokens: Cumulative prompt token count (0 if unknown)
            generation_tokens: Cumulative generated token count (0 if unknown)

        Models whose generate() lacks native streaming support fall back to
        one-shot transcribe() and yield a single chunk with the full text.
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        if not self.supports_native_stt_streaming():
            result = await self.transcribe(
                audio_path, language=language, prompt=prompt, **kwargs
            )
            yield {
                "text": result.get("text", ""),
                "language": result.get("language"),
                "prompt_tokens": 0,
                "generation_tokens": 0,
            }
            return

        import os
        import time

        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        logger.info(
            "STT stream transcribe: model=%s, file=%s (%d bytes), language=%s",
            self._model_name, os.path.basename(audio_path), file_size, language,
        )

        model = self._model
        t0 = time.monotonic()

        gen_kwargs = dict(kwargs)
        generate_language = _normalize_stt_generate_language(model, language)
        if generate_language is not None:
            gen_kwargs["language"] = generate_language
        gen_kwargs.update(_map_stt_prompt_kwargs(model, prompt))
        gen_kwargs["stream"] = True

        iterator: Any = None
        sentinel = object()

        def _next_chunk():
            nonlocal iterator
            if iterator is None:
                iterator = iter(model.generate(audio_path, **gen_kwargs))
            try:
                result = next(iterator)
            except StopIteration:
                return sentinel
            if isinstance(result, str):
                return {
                    "text": result,
                    "language": None,
                    "prompt_tokens": 0,
                    "generation_tokens": 0,
                }
            return {
                "text": getattr(result, "text", "") or "",
                "language": _normalize_result_language(
                    getattr(result, "language", None)
                ),
                "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
                "generation_tokens": int(
                    getattr(result, "generation_tokens", 0) or 0
                ),
            }

        activity_id = self._begin_activity(
            "transcribing",
            detail="Streaming transcription",
            metadata={"file_size_bytes": file_size},
        )
        chunk_count = 0
        text_len = 0
        try:
            loop = asyncio.get_running_loop()
            while True:
                chunk = await loop.run_in_executor(get_mlx_executor(), _next_chunk)
                if chunk is sentinel:
                    break
                chunk_count += 1
                text_len += len(chunk["text"])
                yield chunk
        finally:
            await self._finish_activity(activity_id)
            logger.info(
                "STT stream transcribe done: model=%s, %.2fs, chunks=%d, %d chars",
                self._model_name, time.monotonic() - t0, chunk_count, text_len,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<STTEngine model={self._model_name} status={status}>"
