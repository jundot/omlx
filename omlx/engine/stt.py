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


# ---------------------------------------------------------------------------
# Qwen3-ASR extended sampling (frequency_penalty / presence_penalty).
#
# Upstream mlx-audio's Qwen3ASRModel.generate calls make_logits_processors
# with only (repetition_penalty, repetition_context_size); frequency and
# presence are not exposed. mlx_lm.sample_utils.make_logits_processors does
# support both natively, so we replicate the chunk loop here and build the
# full processor stack ourselves. Activated only when one of the extended
# fields is set; otherwise the normal model.generate() path runs unchanged.
# ---------------------------------------------------------------------------

_EXT_SAMPLING_FIELDS = ("frequency_penalty", "presence_penalty")


def _has_ext_sampling(kwargs: dict) -> bool:
    return any(kwargs.get(k) is not None for k in _EXT_SAMPLING_FIELDS)


def _inner_asr_model(top_model: Any) -> Any:
    """Unwrap mlx-audio's Model wrapper to the Qwen3ASRModel inside."""
    return getattr(top_model, "_model", top_model)


def _run_qwen3_asr_extended_sampling(
    model: Any,
    audio_path: str,
    *,
    language: str | None,
    system_prompt: str | None,
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float | None,
    repetition_context_size: int | None,
    frequency_penalty: float | None,
    frequency_context_size: int | None,
    presence_penalty: float | None,
    presence_context_size: int | None,
) -> Any:
    """Run Qwen3-ASR with the full mlx-lm logits-processor stack.

    Returns a SimpleNamespace that duck-types as STTOutput (text, segments,
    language, total_time, prompt_tokens, generation_tokens, total_tokens).
    """
    import time
    import types

    import numpy as np
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import split_audio_into_chunks
    from mlx_audio.stt.utils import load_audio
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    start_time = time.time()

    inner = _inner_asr_model(model)

    audio_input = audio_path
    if isinstance(audio_input, str):
        audio_input = load_audio(audio_input)
    audio_np = (
        np.array(audio_input) if not isinstance(audio_input, np.ndarray) else audio_input
    )

    chunks = split_audio_into_chunks(
        audio_np,
        sr=inner.sample_rate,
        chunk_duration=1200.0,
        min_chunk_duration=1.0,
    )

    sampler = make_sampler(
        temperature if temperature is not None else 0.0,
        top_p if top_p is not None else 1.0,
        min_p if min_p is not None else 0.0,
        min_tokens_to_keep=1,
        top_k=top_k if top_k is not None else 0,
    )

    lp_kwargs: dict = {}
    if repetition_penalty is not None:
        lp_kwargs["repetition_penalty"] = repetition_penalty
        if repetition_context_size is not None:
            lp_kwargs["repetition_context_size"] = repetition_context_size
    if frequency_penalty is not None:
        lp_kwargs["frequency_penalty"] = frequency_penalty
        if frequency_context_size is not None:
            lp_kwargs["frequency_context_size"] = frequency_context_size
    if presence_penalty is not None:
        lp_kwargs["presence_penalty"] = presence_penalty
        if presence_context_size is not None:
            lp_kwargs["presence_context_size"] = presence_context_size
    logits_processors = make_logits_processors(**lp_kwargs) if lp_kwargs else None

    all_texts: list[str] = []
    segments: list[dict] = []
    total_prompt_tokens = 0
    total_generation_tokens = 0
    remaining_tokens = max_tokens if max_tokens is not None else 8192

    cur_language = language
    for chunk_audio, offset_sec in chunks:
        if remaining_tokens <= 0:
            break
        actual_chunk_duration = len(chunk_audio) / inner.sample_rate

        text, prompt_toks, gen_toks = inner._generate_single_chunk(
            chunk_audio,
            max_tokens=remaining_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            language=cur_language,
            prefill_step_size=2048,
            verbose=False,
            system_prompt=system_prompt,
        )

        if cur_language is None:
            cur_language, text = inner.extract_language(text)

        all_texts.append(text)
        total_prompt_tokens += prompt_toks
        total_generation_tokens += gen_toks
        remaining_tokens -= gen_toks

        segments.append({
            "text": text,
            "language": cur_language,
            "start": float(offset_sec),
            "end": float(offset_sec + actual_chunk_duration),
        })

        mx.clear_cache()

    elapsed = time.time() - start_time
    return types.SimpleNamespace(
        text=" ".join(all_texts),
        segments=segments,
        language=[s["language"] for s in segments],
        prompt_tokens=total_prompt_tokens,
        generation_tokens=total_generation_tokens,
        total_tokens=total_prompt_tokens + total_generation_tokens,
        total_time=elapsed,
        prompt_tps=(
            total_prompt_tokens / elapsed if elapsed > 0 else 0.0
        ),
        generation_tps=(
            total_generation_tokens / elapsed if elapsed > 0 else 0.0
        ),
    )


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

        # Apply mlx_audio per-layer quantization override patch (idempotent).
        # Required for mixed-precision STT checkpoints, e.g. Qwen3-ASR
        # audio8/text4 where audio_tower.* is 8-bit but text_model is 4-bit.
        # Without this, mlx_audio's default class_predicate consults
        # Qwen3ASRModel.model_quant_predicate first, which returns False for
        # audio_tower.* and skips quantization, leaving 8-bit packed weights
        # loaded as fp16 and crashing the first encoder matmul.
        from ..patches.mlx_audio_per_layer_quant import (
            apply_mlx_audio_per_layer_quant_patch,
        )
        apply_mlx_audio_per_layer_quant_patch()

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
        text: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file to transcribe
            language: Optional language code (e.g. 'en', 'fr')
            text: Reference transcript for forced-alignment models
                (e.g. Qwen3-ForcedAligner). Ignored by ASR models that
                don't accept a ``text`` kwarg.
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary with keys:
                text: Transcribed text (ASR) or reconstructed from items (aligner)
                language: Detected or specified language
                segments: List of timed segments (may be empty)
                words: For aligner models, list of {word, start, end} dicts
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

            # OpenAI's `prompt` field is forwarded to the backend's prompt-style
            # kwarg if one exists. Different backends use different names:
            #   Whisper        → initial_prompt
            #   Qwen2-Audio    → prompt
            #   Qwen3-ASR      → system_prompt
            #   others         → context / previous_text
            # We inspect the model's generate() signature and pick the first
            # match in priority order. Backends without any prompt-style kwarg
            # drop the field silently.
            user_prompt = gen_kwargs.pop("prompt", None)
            if user_prompt:
                import inspect
                resolved_prompt_kwarg = None
                try:
                    params = inspect.signature(model.generate).parameters
                    for candidate in (
                        "initial_prompt",
                        "prompt",
                        "system_prompt",
                        "context",
                        "previous_text",
                    ):
                        if candidate in params:
                            resolved_prompt_kwarg = candidate
                            break
                except (TypeError, ValueError):
                    resolved_prompt_kwarg = "prompt"
                if resolved_prompt_kwarg is not None:
                    gen_kwargs[resolved_prompt_kwarg] = user_prompt
                else:
                    logger.debug(
                        "STT backend %s.generate has no prompt-style kwarg; "
                        "dropping user prompt (len=%d)",
                        type(model).__name__, len(user_prompt),
                    )

            # Forced-aligner reference text — only inject if caller provided
            # one and the backend accepts it positionally. ASR models like
            # Qwen3-ASR / Qwen2-Audio / Whisper don't take a ``text`` kwarg
            # and would TypeError if we passed text=None blindly.
            if text is not None:
                gen_kwargs["text"] = text

            # Extended-sampling short-circuit. mlx-audio's Qwen3ASRModel.generate
            # builds make_logits_processors with only (repetition_penalty,
            # repetition_context_size) — frequency_penalty / presence_penalty
            # have no path in. When either of these is set we route through a
            # parallel chunk loop that calls _generate_single_chunk directly
            # with the full mlx-lm processor stack. Only Qwen3-ASR is wired
            # for this; other backends (Whisper, Qwen2-Audio, aligner, …) get
            # 400 if the user tries.
            if _has_ext_sampling(gen_kwargs):
                inner_cls = type(_inner_asr_model(model)).__name__
                if inner_cls != "Qwen3ASRModel":
                    raise RuntimeError(
                        f"frequency_penalty / presence_penalty currently only "
                        f"supported on Qwen3-ASR backends; got {inner_cls}."
                    )
                if text is not None:
                    raise RuntimeError(
                        "frequency_penalty / presence_penalty cannot be combined "
                        "with forced-aligner 'text' field."
                    )
                result = _run_qwen3_asr_extended_sampling(
                    model,
                    audio_path,
                    language=gen_kwargs.get("language"),
                    system_prompt=gen_kwargs.get("system_prompt"),
                    max_tokens=gen_kwargs.get("max_tokens"),
                    temperature=gen_kwargs.get("temperature", 0.0),
                    top_p=gen_kwargs.get("top_p", 1.0),
                    top_k=gen_kwargs.get("top_k", 0),
                    min_p=gen_kwargs.get("min_p", 0.0),
                    repetition_penalty=gen_kwargs.get("repetition_penalty"),
                    repetition_context_size=gen_kwargs.get("repetition_context_size"),
                    frequency_penalty=gen_kwargs.get("frequency_penalty"),
                    frequency_context_size=gen_kwargs.get("frequency_context_size"),
                    presence_penalty=gen_kwargs.get("presence_penalty"),
                    presence_context_size=gen_kwargs.get("presence_context_size"),
                )
                # Fall through to the shared STTOutput → dict normalization
                # path below (hasattr(result, "text") branch).
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
                        "duration": getattr(result, "total_time", 0.0),
                    }
                # Shouldn't happen — helper always returns SimpleNamespace.
                raise RuntimeError(
                    "extended-sampling helper returned unexpected shape"
                )

            # Defensive filter: if the backend's generate() signature does
            # not have **kwargs, drop any gen_kwargs whose name isn't an
            # explicit parameter. Qwen3-ASR / Qwen2-Audio / Whisper all
            # have **kwargs (or **decode_options) so this is a no-op for
            # them, but the audio_routes layer now forwards top_p/top_k/
            # min_p/repetition_penalty for sampler tuning — backends like
            # Parakeet that have a strict keyword-only signature would
            # raise TypeError otherwise.
            try:
                import inspect as _inspect
                _params = _inspect.signature(model.generate).parameters
                _has_var_kwargs = any(
                    p.kind == _inspect.Parameter.VAR_KEYWORD
                    for p in _params.values()
                )
                if not _has_var_kwargs:
                    gen_kwargs = {
                        k: v for k, v in gen_kwargs.items() if k in _params
                    }
            except (TypeError, ValueError):
                pass

            result = model.generate(audio_path, **gen_kwargs)

            # Two shapes possible:
            #  - STTOutput (text, segments, language, total_time, ...) for ASR.
            #  - ForcedAlignResult for ForcedAligner. NOTE: at runtime this also
            #    exposes .text and .segments (each as a one-item-per-char dict),
            #    so the cheap ``hasattr(result, "items")`` check is not enough —
            #    we must also verify the items list contains ForcedAlignItem-
            #    shaped objects (with .start_time / .end_time, not plain dicts).
            #  - List[ForcedAlignResult] for batch alignment (we always call
            #    single-audio here, so unwrap a single-element list).
            if isinstance(result, list) and result and hasattr(result[0], "items"):
                result = result[0]

            items_attr = getattr(result, "items", None)
            is_aligner_result = False
            if items_attr is not None and not callable(items_attr):
                try:
                    first = next(iter(items_attr), None)
                except TypeError:
                    first = None
                if first is not None and hasattr(first, "start_time"):
                    is_aligner_result = True

            if is_aligner_result:
                # ForcedAligner output. Wrap as a single segment containing the
                # words array so the API response shape stays the same whether
                # the caller hit an ASR directly or the aligner standalone.
                items = list(items_attr)
                words = [
                    {
                        "word": it.text,
                        "start": float(it.start_time),
                        "end": float(it.end_time),
                    }
                    for it in items
                ]
                reconstructed = "".join(it.text for it in items)
                start = float(min(
                    (it.start_time for it in items), default=0.0
                ))
                end = float(max(
                    (it.end_time for it in items), default=0.0
                ))
                segment = {
                    "text": reconstructed,
                    "language": language,
                    "start": start,
                    "end": end,
                    "words": words,
                }
                return {
                    "text": reconstructed,
                    "language": language,
                    "segments": [segment],
                    "words": words,
                    "duration": end,
                }

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
            await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<STTEngine model={self._model_name} status={status}>"
