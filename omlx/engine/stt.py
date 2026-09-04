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
import threading
from collections.abc import AsyncIterator
from typing import Any

import mlx.core as mx
import numpy as np

from ..engine_core import get_mlx_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

REALTIME_SAMPLE_RATE = 16000


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


def _load_audio_samples(audio_path: str) -> np.ndarray:
    """Load an audio file as 16 kHz mono float32 samples."""
    from mlx_audio.stt.utils import load_audio

    return np.asarray(load_audio(audio_path), dtype=np.float32)


def _split_audio_segments(
    samples: np.ndarray,
    sr: int,
    segment_seconds: float = 30.0,
    search_seconds: float = 5.0,
) -> list[np.ndarray]:
    """Split audio into <= ``segment_seconds`` segments at low-energy points.

    Each hard boundary is moved back to the quietest 50 ms frame within the
    trailing ``search_seconds`` window, so cuts land in pauses instead of
    mid-word. Concatenating the returned segments reproduces the input.
    """
    seg_len = int(segment_seconds * sr)
    if len(samples) <= seg_len:
        return [samples]

    search_len = int(search_seconds * sr)
    frame = int(0.05 * sr)
    segments: list[np.ndarray] = []
    start = 0
    while len(samples) - start > seg_len:
        hard_end = start + seg_len
        search_start = max(start, hard_end - search_len)
        window = samples[search_start:hard_end]
        n_frames = len(window) // frame
        if n_frames > 0:
            frames = window[: n_frames * frame].reshape(n_frames, frame)
            rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
            cut = search_start + int(np.argmin(rms)) * frame + frame // 2
        else:
            cut = hard_end
        segments.append(samples[start:cut])
        start = cut

    tail = samples[start:]
    if len(tail) >= int(0.1 * sr) or not segments:
        segments.append(tail)
    elif len(tail) > 0:
        segments[-1] = np.concatenate([segments[-1], tail])
    return segments


# ---------------------------------------------------------------------------
# Realtime (push-input) transcription sessions
# ---------------------------------------------------------------------------


class _WhisperRealtimeBackend:
    """Push-input adapter over mlx-audio Whisper's AlignAtt StreamingDecoder.

    Buffers incoming 16 kHz float32 samples; every ``CHUNK_SECONDS`` of new
    audio is mel-transformed and fed to ``StreamingDecoder.decode_chunk``,
    which accumulates mel across chunks and returns newly committed tokens.
    All *_sync methods must run on the MLX executor; ``feed`` is thread-safe
    and cheap.

    Two upstream hazards are handled here rather than in mlx-audio:

    - The decoder trims its accumulated mel to a 30 s window but keeps a
      session-cumulative emitted-token list, so once audio exceeds 30 s its
      deltas become arbitrary re-decoded suffixes (duplicated/spliced text).
      The window is therefore rotated (flush + ``decoder.reset()``) before
      the mel cap is ever reached, preferring a quiet chunk as the boundary.
    - Each delta's token slice is decoded independently upstream, which
      splits multi-token CJK characters into replacement chars. Deltas are
      instead diffed from a cumulative decode of all window tokens, and a
      trailing incomplete character is withheld until it completes.
    """

    CHUNK_SECONDS = 1.0
    # Language auto-detection runs on the first decoded chunk; 1 s is too
    # little signal, so the first decode waits for a larger chunk.
    FIRST_CHUNK_SECONDS = 2.0
    # Rotate before the decoder's 30 s mel cap (N_FRAMES) is crossed.
    MAX_WINDOW_SECONDS = 28.0
    # Past this window size, rotate early when a quiet chunk shows up so
    # the boundary lands in a pause instead of mid-word.
    ROTATE_QUIET_SECONDS = 22.0
    QUIET_RMS = 0.01

    def __init__(self, model: Any, language: str | None = None):
        self._model = model
        self._language = language
        self._pinned_language = language is not None
        self._decoder = None
        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self._lock = threading.Lock()
        self._window_samples = 0
        self._window_tokens: list[int] = []
        self._window_text = ""

    def feed(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buffer.append(samples)
            self._buffered += len(samples)

    def _take(self, min_samples: int) -> np.ndarray | None:
        with self._lock:
            if self._buffered < min_samples or not self._buffer:
                return None
            merged = np.concatenate(self._buffer)
            self._buffer = []
            self._buffered = 0
        return merged

    def _decode(self, samples: np.ndarray, is_last: bool) -> str:
        from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram
        from mlx_audio.stt.models.whisper.streaming import (
            StreamingConfig,
            StreamingDecoder,
        )

        mel = log_mel_spectrogram(samples, n_mels=self._model.dims.n_mels)
        if self._decoder is None:
            if self._language is None:
                self._language = self._model._detect_language(mel)
            self._decoder = StreamingDecoder(
                self._model, StreamingConfig(), language=self._language
            )
        result = self._decoder.decode_chunk(mel, is_last=is_last)
        # The decoder's _emitted_tokens holds its full current-window decode
        # and is the authoritative hypothesis. Its per-call result.tokens
        # slice inherits the decoder's own bookkeeping misalignments (it
        # slices a fresh re-decode at a stale offset), so diffing against
        # the full hypothesis is the only way to stay duplication-free.
        window_tokens = getattr(self._decoder, "_emitted_tokens", None)
        if window_tokens is None:
            # Upstream renamed the attribute: fall back to accumulating the
            # per-call slices (weaker, but never crashes).
            self._window_tokens.extend(result.tokens or [])
            window_tokens = self._window_tokens
        return self._emit_stable_delta(list(window_tokens), is_last)

    def _emit_stable_delta(self, window_tokens: list, is_last: bool) -> str:
        """Diff the decoder's full window hypothesis into a text delta.

        Withholds a trailing incomplete character (U+FFFD from a partial
        multi-token CJK sequence) until later tokens complete it. When the
        decoder's full-window re-decode revises already-emitted text (its
        hypothesis is not prefix-stable), the new hypothesis is aligned
        against the tail of the emitted text and only the continuation is
        appended — sent text cannot be retracted, and a position-based
        slice would re-emit old content. Replacement chars are stripped
        from the returned delta (kept in bookkeeping).
        """
        if not window_tokens:
            return ""
        full = self._decoder.tokenizer.decode(window_tokens)
        stable_end = len(full)
        if not is_last:
            while stable_end > 0 and full[stable_end - 1] == "�":
                stable_end -= 1
        stable = full[:stable_end]
        prev = self._window_text
        if stable.startswith(prev):
            delta = stable[len(prev):]
            self._window_text = stable
        else:
            delta = self._align_revision(prev, stable)
            if delta:
                self._window_text = prev + delta
            elif is_last and stable:
                # Closing flush with an unalignable revision: mid-window
                # drops are fine (the hypothesis is still in flux and the
                # flush recovers it), but dropping the settled final
                # hypothesis would lose the whole revised region. Emit
                # everything past the common prefix — worst case is a
                # bounded echo of the divergent tail, never lost speech.
                n = 0
                for a, b in zip(prev, stable):
                    if a != b:
                        break
                    n += 1
                delta = stable[n:]
                self._window_text = prev + delta
        return delta.replace("�", "")

    @staticmethod
    def _align_revision(prev: str, new: str) -> str:
        """Return the continuation of ``new`` past the tail of ``prev``.

        Finds the longest suffix of the emitted text (>= 8 chars, capped at
        200) that occurs in the new hypothesis and returns everything after
        it. Returns "" when no trustworthy alignment exists — dropping a
        revision is safer than duplicating already-emitted text.
        """
        max_k = min(len(prev), len(new), 200)
        for k in range(max_k, 7, -1):
            pos = new.rfind(prev[-k:])
            if pos != -1:
                return new[pos + k:]
        return ""

    def _rotate_window(self) -> None:
        if self._decoder is not None and self._pinned_language:
            self._decoder.reset()
        else:
            # Drop the decoder so the next window re-detects language:
            # rotation boundaries land at pauses, which is exactly where
            # code-switched audio changes language.
            self._decoder = None
            self._language = None
        self._window_samples = 0
        self._window_tokens = []
        self._window_text = ""

    def poll_sync(self) -> list[str]:
        min_seconds = (
            self.CHUNK_SECONDS if self._decoder is not None
            else self.FIRST_CHUNK_SECONDS
        )
        samples = self._take(int(min_seconds * REALTIME_SAMPLE_RATE))
        if samples is None:
            return []

        window_seconds = self._window_samples / REALTIME_SAMPLE_RATE
        chunk_seconds = len(samples) / REALTIME_SAMPLE_RATE
        rms = (
            float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
            if len(samples)
            else 0.0
        )
        rotate = self._decoder is not None and (
            window_seconds + chunk_seconds >= self.MAX_WINDOW_SECONDS
            or (
                window_seconds >= self.ROTATE_QUIET_SECONDS
                and rms < self.QUIET_RMS
            )
        )
        if rotate:
            # Close the window on this chunk (is_last flushes held-back
            # tokens), then reset so the decoder's 30 s mel cap is never
            # crossed with stale emitted-token bookkeeping.
            text = self._decode(samples, is_last=True)
            self._rotate_window()
        else:
            self._window_samples += len(samples)
            text = self._decode(samples, is_last=False)
        return [text] if text else []

    def close_sync(self) -> list[str]:
        samples = self._take(1)
        if samples is None:
            if self._decoder is None:
                return []
            # No pending audio: feed a short silence chunk so decode_chunk
            # runs once more with is_last=True and flushes held-back tokens.
            samples = np.zeros(
                int(0.2 * REALTIME_SAMPLE_RATE), dtype=np.float32
            )
        text = self._decode(samples, is_last=True)
        return [text] if text else []


class _VoxtralRealtimeBackend:
    """Push-input adapter over mlx-audio's VoxtralStreamingSession."""

    def __init__(self, model: Any):
        self._model = model
        self._session = None

    def start_sync(self) -> None:
        self._session = self._model.create_streaming_session()

    def feed(self, samples: np.ndarray) -> None:
        if self._session is not None:
            self._session.feed(samples)

    def poll_sync(self) -> list[str]:
        if self._session is None:
            return []
        return [d for d in self._session.step(max_decode_tokens=8) if d]

    def close_sync(self) -> list[str]:
        if self._session is None:
            return []
        self._session.close()
        out: list[str] = []
        while not self._session.done:
            out.extend(d for d in self._session.step(max_decode_tokens=16) if d)
        return out


class RealtimeTranscriptionSession:
    """Async facade over a push-input STT backend.

    ``feed_pcm16`` accepts raw little-endian Int16 mono PCM at 16 kHz (the
    WebSocket wire format) and queues it without MLX work. ``poll`` and
    ``close`` run the backend's decode work on the shared MLX executor and
    return lists of committed text deltas. ``release`` must always be called
    when the session ends; it clears the engine's single-session slot and
    finishes the activity that protects the engine from LRU eviction.
    """

    def __init__(self, engine: "STTEngine", backend: Any, activity_id: str):
        self._engine = engine
        self._backend = backend
        self._activity_id = activity_id
        self._closed = False

    def feed_pcm16(self, data: bytes) -> None:
        if self._closed or not data:
            return
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self._backend.feed(samples)

    async def poll(self) -> list[str]:
        if self._closed:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_mlx_executor(), self._backend.poll_sync
        )

    async def close(self) -> list[str]:
        if self._closed:
            return []
        self._closed = True
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_mlx_executor(), self._backend.close_sync
        )

    async def release(self) -> None:
        self._closed = True
        await self._engine._release_realtime_session(self._activity_id)


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
        self._realtime_active = False

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

        if hasattr(self._model, "generate_streaming"):
            # Whisper: generate(stream=True) routes to the experimental
            # AlignAtt decoder, which bypasses the whole anti-hallucination
            # stack (temperature fallback, compression-ratio / no-speech
            # checks, condition_on_previous_text), mis-tracks emitted tokens
            # once audio exceeds its 30 s mel window (duplicated/spliced
            # text on long files), locks language from the first second,
            # and re-encodes the window for every 1 s chunk. Stream proper
            # non-streaming decodes segment by segment instead.
            async for chunk in self._transcribe_stream_segmented(
                audio_path, language=language, prompt=prompt, **kwargs
            ):
                yield chunk
            return

        if hasattr(self._model, "stream_generate") and hasattr(
            self._model, "extract_language"
        ):
            # Qwen3-ASR: upstream stream_transcribe decodes every token
            # independently (tokenizer.decode([token])), which splits
            # multi-token CJK characters into U+FFFD replacement chars.
            # Drive stream_generate directly and decode the cumulative
            # token ids so character boundaries stay intact.
            async for chunk in self._transcribe_stream_token_ids(
                audio_path, language=language, prompt=prompt, **kwargs
            ):
                yield chunk
            return

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

    async def _transcribe_stream_segmented(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a long file as sequential full-quality segment decodes.

        Splits audio into <= 30 s segments at low-energy points and runs the
        model's regular generate() (with its full anti-hallucination stack)
        per segment, yielding each segment's text as one delta. Without an
        explicit ``language``, each segment auto-detects — code-switched
        audio comes out per-segment in its own language.
        """
        import os
        import time

        model = self._model
        loop = asyncio.get_running_loop()

        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        logger.info(
            "STT segmented stream transcribe: model=%s, file=%s (%d bytes), "
            "language=%s",
            self._model_name, os.path.basename(audio_path), file_size, language,
        )
        t0 = time.monotonic()

        samples = await loop.run_in_executor(
            get_mlx_executor(), lambda: _load_audio_samples(audio_path)
        )
        segments = _split_audio_segments(samples, REALTIME_SAMPLE_RATE)

        # Segment deltas are plain text; timestamps are meaningless across
        # independently decoded segments.
        kwargs.pop("word_timestamps", None)

        activity_id = self._begin_activity(
            "transcribing",
            detail="Streaming transcription",
            metadata={"file_size_bytes": file_size},
        )
        emitted_any = False
        text_len = 0
        try:
            for segment in segments:
                def _decode_segment(segment=segment):
                    gen_kwargs = dict(kwargs)
                    generate_language = _normalize_stt_generate_language(
                        model, language
                    )
                    if generate_language is not None:
                        gen_kwargs["language"] = generate_language
                    gen_kwargs.update(_map_stt_prompt_kwargs(model, prompt))
                    result = model.generate(segment, **gen_kwargs)
                    text = getattr(result, "text", "") or ""
                    lang = _normalize_result_language(
                        getattr(result, "language", None)
                    )
                    return text, lang

                text, lang = await loop.run_in_executor(
                    get_mlx_executor(), _decode_segment
                )
                if not text:
                    continue
                if emitted_any and not text[:1].isspace():
                    text = " " + text
                emitted_any = True
                text_len += len(text)
                yield {
                    "text": text,
                    "language": lang or language,
                    "prompt_tokens": 0,
                    "generation_tokens": 0,
                }
        finally:
            await self._finish_activity(activity_id)
            logger.info(
                "STT segmented stream done: model=%s, %.2fs, segments=%d, "
                "%d chars",
                self._model_name, time.monotonic() - t0, len(segments), text_len,
            )

    async def _transcribe_stream_token_ids(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """Token-level streaming with cumulative detokenization (Qwen3-ASR).

        Deltas are diffed from a decode of ALL generated token ids, with a
        trailing incomplete character (U+FFFD) withheld until later tokens
        complete it. Language auto-detection consumes the model's
        ``language {name}<asr_text>`` prefix via ``extract_language``.
        """
        import os
        import time

        from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
            split_audio_into_chunks,
        )

        model = self._model
        tokenizer = model._tokenizer
        loop = asyncio.get_running_loop()

        file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        logger.info(
            "STT token stream transcribe: model=%s, file=%s (%d bytes), "
            "language=%s",
            self._model_name, os.path.basename(audio_path), file_size, language,
        )
        t0 = time.monotonic()

        samples = await loop.run_in_executor(
            get_mlx_executor(), lambda: _load_audio_samples(audio_path)
        )
        chunks = split_audio_into_chunks(samples, sr=model.sample_rate)

        gen_language = _normalize_stt_generate_language(model, language)
        system_prompt = _map_stt_prompt_kwargs(model, prompt).get("system_prompt")
        max_tokens = int(kwargs.get("max_tokens") or 8192)

        sentinel = object()
        activity_id = self._begin_activity(
            "transcribing",
            detail="Streaming transcription",
            metadata={"file_size_bytes": file_size},
        )
        total_tokens = 0
        text_len = 0
        try:
            for chunk_audio, _offset in chunks:
                ids: list[int] = []
                emitted = ""
                iterator: Any = None

                def _next_token(chunk_audio=chunk_audio):
                    nonlocal iterator
                    if iterator is None:
                        iterator = model.stream_generate(
                            chunk_audio,
                            max_tokens=max_tokens,
                            language=gen_language,
                            system_prompt=system_prompt,
                        )
                    try:
                        token, _ = next(iterator)
                    except StopIteration:
                        return sentinel
                    return int(token)

                while True:
                    token = await loop.run_in_executor(
                        get_mlx_executor(), _next_token
                    )
                    if token is sentinel:
                        break
                    ids.append(token)
                    full = tokenizer.decode(ids)
                    if (
                        gen_language is None
                        and "<asr_text>" not in full
                        and len(ids) <= 16
                    ):
                        # Auto-detect prefix still streaming in
                        continue
                    lang_out, visible = model.extract_language(full)
                    has_marker = "<asr_text>" in full
                    stable_end = len(visible)
                    while stable_end > 0 and visible[stable_end - 1] == "�":
                        stable_end -= 1
                    stable = visible[:stable_end]
                    if len(stable) <= len(emitted):
                        continue
                    delta = stable[len(emitted):]
                    emitted = stable
                    text_len += len(delta)
                    yield {
                        "text": delta.replace("�", ""),
                        "language": lang_out if has_marker else language,
                        "prompt_tokens": 0,
                        "generation_tokens": 0,
                    }

                total_tokens += len(ids)
                # Flush anything withheld at end of chunk, then report the
                # cumulative token count on an empty chunk (the SSE layer
                # keeps the max seen and skips empty deltas).
                if ids:
                    _, visible = model.extract_language(tokenizer.decode(ids))
                    final_delta = visible[len(emitted):]
                    text_len += len(final_delta)
                    yield {
                        "text": final_delta.replace("�", ""),
                        "language": language,
                        "prompt_tokens": 0,
                        "generation_tokens": total_tokens,
                    }
        finally:
            await self._finish_activity(activity_id)
            logger.info(
                "STT token stream done: model=%s, %.2fs, tokens=%d, %d chars",
                self._model_name, time.monotonic() - t0, total_tokens, text_len,
            )

    def supports_realtime_stt(self) -> bool:
        """True when the loaded model supports push-based realtime decoding.

        Whisper-family models expose ``generate_streaming`` (AlignAtt
        StreamingDecoder); voxtral_realtime exposes
        ``create_streaming_session``. Other STT backends only accept a
        complete audio input.
        """
        if self._model is None:
            return False
        return hasattr(self._model, "create_streaming_session") or hasattr(
            self._model, "generate_streaming"
        )

    async def create_realtime_session(
        self, language: str | None = None
    ) -> RealtimeTranscriptionSession:
        """Create a push-input realtime transcription session.

        Only one realtime session may be active per engine at a time; a
        second concurrent request raises RuntimeError. The caller must
        always await ``session.release()`` when done.
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")
        if not self.supports_realtime_stt():
            raise RuntimeError(
                f"Model '{self._model_name}' does not support realtime "
                "transcription."
            )
        if self._realtime_active:
            raise RuntimeError(
                "A realtime transcription session is already active for "
                f"model '{self._model_name}'."
            )

        model = self._model
        if hasattr(model, "create_streaming_session"):
            backend: Any = _VoxtralRealtimeBackend(model)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(get_mlx_executor(), backend.start_sync)
        else:
            backend = _WhisperRealtimeBackend(model, language=language)

        self._realtime_active = True
        activity_id = self._begin_activity(
            "transcribing", detail="Realtime transcription"
        )
        logger.info("STT realtime session started: model=%s", self._model_name)
        return RealtimeTranscriptionSession(self, backend, activity_id)

    async def _release_realtime_session(self, activity_id: str) -> None:
        self._realtime_active = False
        await self._finish_activity(activity_id)
        logger.info("STT realtime session ended: model=%s", self._model_name)

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<STTEngine model={self._model_name} status={status}>"
