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
from typing import Any

import mlx.core as mx

from ..engine_core import get_mlx_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


# Lowercase full-names work for both Qwen3-ASR (its _build_prompt lowercases
# the supported-language list before lookup) and Whisper (its TO_LANGUAGE_CODE
# normalizer maps lowercase names to ISO codes). Capitalized names would break
# Whisper because `<|Chinese|>` is not a valid language token.
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


def _normalize_stt_generate_language(language: str | None) -> str | None:
    """Map OpenAI-style ISO codes to language names accepted by mlx-audio backends."""
    if language is None:
        return None

    normalized = language.strip()
    if not normalized:
        return None

    return _ISO_TO_STT_LANG.get(normalized.lower(), normalized)


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
        **kwargs,
    ) -> dict[str, Any]:
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

        def _normalize_language(raw_lang):
            """Normalize language field from mlx-audio."""
            if isinstance(raw_lang, list):
                raw_lang = raw_lang[0] if raw_lang else None
            if isinstance(raw_lang, str) and raw_lang.lower() == "none":
                return None
            return raw_lang

        def _transcribe_sync():
            # Call model.generate() directly instead of
            # generate_transcription() which writes files to disk.
            gen_kwargs = dict(kwargs)
            generate_language = _normalize_stt_generate_language(language)
            if generate_language is not None:
                gen_kwargs["language"] = generate_language

            result = model.generate(audio_path, **gen_kwargs)

            # result is typically an STTOutput dataclass with:
            # text, segments, language, total_time, etc.
            if hasattr(result, "text"):
                raw_lang = _normalize_language(
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
            if self._end_activity(activity_id):
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    get_mlx_executor(),
                    lambda: (mx.synchronize(), mx.clear_cache()),
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
