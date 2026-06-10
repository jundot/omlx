# SPDX-License-Identifier: Apache-2.0
"""
Audio API routes for oMLX.

This module provides OpenAI-compatible audio endpoints:
- POST /v1/audio/transcriptions  - Speech-to-Text
- POST /v1/audio/speech          - Text-to-Speech
- POST /v1/audio/process         - Speech-to-Speech / audio processing
"""

import base64
import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from ..engine.audio_utils import wav_bytes_to_pcm_frames, wav_header
from ..server_metrics import get_server_metrics
from .audio_models import AudioSpeechRequest, AudioTranscriptionResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum upload size for audio files (100 MB).
MAX_AUDIO_UPLOAD_BYTES = 100 * 1024 * 1024

# Maximum base64-encoded ref_audio size (~15 MB raw audio, enough for ~60s).
MAX_REF_AUDIO_BASE64_BYTES = 20 * 1024 * 1024

# Default native TTS chunk cadence. Keep this below the mlx-audio default to
# improve TTFT while still letting the model process the full input at once.
DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS = 0.2
MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS = 0.01

# Video container extensions that should be routed through ffmpeg decoding.
# mlx-audio only recognises audio-specific extensions (m4a, aac, ogg, opus),
# so we remap video containers to .m4a before handing off. ffmpeg detects the
# actual format from file content, not the extension.
_VIDEO_CONTAINERS = {".mp4", ".mkv", ".mov", ".m4v", ".webm", ".avi"}


# ---------------------------------------------------------------------------
# Engine pool accessor — patched in tests via omlx.api.audio_routes._get_engine_pool
# ---------------------------------------------------------------------------


def _get_engine_pool():
    """Return the active EnginePool from server state.

    Imported lazily to avoid a circular import at module load time.
    Can be replaced in tests via patch('omlx.api.audio_routes._get_engine_pool').
    """
    # Import here to avoid circular imports at module load
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    """Resolve a model alias to its real model ID.

    Delegates to the same resolve_model_id used by LLM/chat endpoints,
    ensuring audio endpoints handle aliases consistently.
    """
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _get_settings_manager():
    """Return the active ModelSettingsManager from server state, or None.

    Lazy import + defensive guard so the audio router stays usable in tests
    that don't bring up the full server state.
    """
    try:
        from omlx.server import _server_state
    except Exception:
        return None
    return getattr(_server_state, "settings_manager", None)


def _record_audio_request(model_id: str) -> None:
    """Record audio request count without treating bytes/chars as tokens."""
    try:
        get_server_metrics().record_request_complete(
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            model_id=model_id,
        )
    except Exception as exc:
        logger.warning("Failed to record audio metrics for %s: %s", model_id, exc)


async def _read_upload(file: UploadFile) -> bytes:
    """Read an uploaded file in chunks, bailing early if it exceeds the limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio file exceeds maximum allowed size "
                    f"({MAX_AUDIO_UPLOAD_BYTES} bytes)"
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_ref_audio_base64(request: AudioSpeechRequest) -> Optional[bytes]:
    """Validate and decode optional base64 ref_audio from a TTS request."""
    if request.ref_audio is None:
        return None

    if not request.ref_text:
        raise HTTPException(
            status_code=400,
            detail="'ref_text' is required when 'ref_audio' is provided "
            "(must be the transcript of the reference audio)",
        )
    if len(request.ref_audio) > MAX_REF_AUDIO_BASE64_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"ref_audio exceeds maximum allowed size "
                f"({MAX_REF_AUDIO_BASE64_BYTES} bytes base64, "
                f"~60 seconds of audio)"
            ),
        )
    try:
        return base64.b64decode(request.ref_audio, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 encoding in 'ref_audio' field",
        )


def _write_ref_audio_tempfile(audio_bytes: Optional[bytes]) -> Optional[str]:
    """Persist decoded ref audio to a temp file if present."""
    if audio_bytes is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        tmp.write(audio_bytes)
        return tmp.name
    finally:
        tmp.close()


def _cleanup_tempfile(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _resolve_tts_streaming_interval(request: AudioSpeechRequest) -> float:
    """Return a native TTS streaming interval that is safe for mlx-audio."""
    if request.streaming_interval is None:
        return DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS

    interval = request.streaming_interval
    if (
        not math.isfinite(interval)
        or interval < MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "'streaming_interval' must be at least "
                f"{MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS} seconds"
            ),
        )
    return interval


def _split_tts_text(text: str, max_chars: int = 300) -> list[str]:
    """Split TTS input into conservative sentence-like chunks."""
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            flush_current()
            parts = re.split(r"(?<=[,;:，；：])\s*", sentence)
            parts = [p.strip() for p in parts if p and p.strip()]
            buffer = ""
            for part in parts or [sentence]:
                while len(part) > max_chars:
                    if buffer:
                        chunks.append(buffer.strip())
                        buffer = ""
                    chunks.append(part[:max_chars].strip())
                    part = part[max_chars:].strip()
                if not part:
                    continue
                candidate = f"{buffer} {part}".strip() if buffer else part
                if len(candidate) <= max_chars:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(buffer.strip())
                    buffer = part
            if buffer:
                chunks.append(buffer.strip())
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > max_chars:
            flush_current()
            current = sentence
        else:
            current = candidate

    flush_current()
    return chunks or [text]


async def _stream_speech_response(
    engine,
    request: AudioSpeechRequest,
    ref_audio_path: Optional[str],
    streaming_interval: float,
) -> AsyncIterator[bytes]:
    """Stream sentence-level TTS as a single WAV header plus PCM chunks."""
    try:
        if (
            hasattr(engine, "supports_native_tts_streaming")
            and engine.supports_native_tts_streaming()
            and hasattr(engine, "stream_synthesize_pcm")
        ):
            logger.info(
                "TTS native streaming start: model=%s, text_len=%d, voice=%s, language=%s",
                request.model, len(request.input), request.voice, request.language or "auto",
            )
            stream_format: Optional[tuple[int, int, int]] = None
            try:
                async for sample_rate, channels, sample_width, pcm_bytes in engine.stream_synthesize_pcm(
                    request.input,
                    voice=request.voice,
                    language=request.language,
                    speed=request.speed,
                    instructions=request.instructions,
                    ref_audio=ref_audio_path,
                    ref_text=request.ref_text,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    max_tokens=request.max_tokens,
                    streaming_interval=streaming_interval,
                ):
                    fmt = (sample_rate, channels, sample_width)
                    if stream_format is None:
                        stream_format = fmt
                        yield wav_header(
                            sample_rate=sample_rate,
                            channels=channels,
                            sample_width=sample_width,
                        )
                    elif fmt != stream_format:
                        raise RuntimeError(
                            "Inconsistent native streaming PCM format: "
                            f"expected {stream_format}, got {fmt}"
                        )
                    if pcm_bytes:
                        yield pcm_bytes
            except NotImplementedError:
                if stream_format is not None:
                    raise
                logger.info(
                    "TTS native streaming unavailable at runtime; falling back "
                    "to segmented synthesis: model=%s",
                    request.model,
                )
            else:
                return

        segments = _split_tts_text(request.input)
        logger.info(
            "TTS streaming start: model=%s, text_len=%d, segments=%d, voice=%s, language=%s",
            request.model, len(request.input), len(segments), request.voice, request.language or "auto",
        )

        stream_format: Optional[tuple[int, int, int]] = None
        for idx, segment in enumerate(segments, start=1):
            wav_bytes = await engine.synthesize(
                segment,
                voice=request.voice,
                language=request.language,
                speed=request.speed,
                instructions=request.instructions,
                ref_audio=ref_audio_path,
                ref_text=request.ref_text,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                max_tokens=request.max_tokens,
            )
            sample_rate, channels, sample_width, pcm_bytes = wav_bytes_to_pcm_frames(wav_bytes)
            fmt = (sample_rate, channels, sample_width)
            if stream_format is None:
                stream_format = fmt
                yield wav_header(sample_rate=sample_rate, channels=channels, sample_width=sample_width)
            elif fmt != stream_format:
                raise RuntimeError(
                    "Inconsistent WAV format across TTS segments: "
                    f"expected {stream_format}, got {fmt}"
                )
            logger.debug(
                "TTS streaming segment %d/%d: text_len=%d, pcm_bytes=%d",
                idx, len(segments), len(segment), len(pcm_bytes),
            )
            if pcm_bytes:
                yield pcm_bytes
    finally:
        _cleanup_tempfile(ref_audio_path)


async def _stream_with_prefetched_chunk(
    first_chunk: bytes,
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Yield a chunk fetched before response headers, then the rest of the stream."""
    try:
        yield first_chunk
        async for chunk in stream:
            yield chunk
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _sse_event(payload: dict) -> str:
    """Serialize one data-only SSE event, OpenAI transcription-stream style."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_transcription_events(
    engine,
    tmp_path: str,
    model_id: str,
    transcribe_kwargs: dict,
) -> AsyncIterator[str]:
    """Yield OpenAI-compatible transcript.text.* SSE events.

    Owns the uploaded temp file: it is deleted when the stream finishes,
    errors, or is cancelled by a client disconnect.
    """
    full_text: list[str] = []
    prompt_tokens = 0
    generation_tokens = 0
    try:
        async for chunk in engine.transcribe_stream(tmp_path, **transcribe_kwargs):
            # Cumulative totals arrive on the chunks that know them
            # (typically the final one); keep the max seen.
            prompt_tokens = max(
                prompt_tokens, int(chunk.get("prompt_tokens") or 0)
            )
            generation_tokens = max(
                generation_tokens, int(chunk.get("generation_tokens") or 0)
            )
            delta = chunk.get("text") or ""
            if not delta:
                continue
            full_text.append(delta)
            yield _sse_event({"type": "transcript.text.delta", "delta": delta})

        done: dict = {"type": "transcript.text.done", "text": "".join(full_text)}
        if prompt_tokens or generation_tokens:
            done["usage"] = {
                "type": "tokens",
                "input_tokens": prompt_tokens,
                "output_tokens": generation_tokens,
                "total_tokens": prompt_tokens + generation_tokens,
            }
        yield _sse_event(done)
        _record_audio_request(model_id)
    finally:
        _cleanup_tempfile(tmp_path)


@router.post("/v1/audio/transcriptions", response_model=AudioTranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    stream: bool = Form(False),
    max_tokens: Optional[int] = Form(None),
    word_timestamps: bool = Form(False),
):
    """OpenAI-compatible audio transcription endpoint (Speech-to-Text).

    Note: ``response_format`` and ``temperature`` are accepted for OpenAI API
    compatibility but are not yet implemented — they are silently ignored.

    ``stream=true`` switches the response to OpenAI's transcription SSE
    format: ``transcript.text.delta`` events with incremental text followed
    by a final ``transcript.text.done`` event with the full transcription
    (#1066). Models whose mlx-audio backend lacks native streaming still
    respond in SSE format, with the full text arriving in a single delta.

    ``prompt`` follows the OpenAI spec: optional text to guide recognition
    toward specific vocabulary, spellings, or style. Mapped onto the
    backend's biasing hook — Qwen3-ASR receives it as trained context
    injection (``system_prompt``, strong biasing), Whisper models as a
    decoder-prefix soft prior (``initial_prompt``, ~224-token window).
    Backends without a biasing hook ignore it; it never fails a request.

    ``max_tokens`` is an oMLX extension that raises the underlying model's
    output cap. Useful for long audio with models like VibeVoice-ASR whose
    mlx-audio default (8192) truncates ~24 min files. When omitted, the
    model's own default applies.

    ``word_timestamps`` is an oMLX extension that exposes mlx-audio's native
    word-level alignment for Whisper models. When True, each segment in the
    response includes a ``words`` array of
    ``{word, start, end, probability}`` objects. Default False preserves the
    existing response shape for every current caller.
    """
    from omlx.engine.stt import STTEngine
    from omlx.exceptions import ModelNotFoundError

    pool = _get_engine_pool()
    resolved_model = _resolve_model(model)

    # Load the engine via pool (handles model loading and LRU eviction)
    try:
        engine = await pool.get_engine(resolved_model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved_model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(engine, STTEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{resolved_model}' is not a speech-to-text model",
        )

    # Save uploaded file to a temp path so the engine can open it by path.
    # Remap video container extensions to .m4a so mlx-audio routes them
    # through ffmpeg instead of miniaudio (which can't decode containers).
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    if suffix.lower() in _VIDEO_CONTAINERS:
        suffix = ".m4a"
    tmp_path = None
    try:
        content = await _read_upload(file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        # Effective max_tokens precedence: request > per-model setting (if any) >
        # model's own ``generate(max_tokens=...)`` default. The per-model lookup
        # mirrors how chat completions reads ModelSettings.max_tokens for LLMs;
        # for STT, settings.json's ``max_tokens`` (e.g. raised to 65536 for
        # VibeVoice-ASR) becomes the durable default for that model.
        effective_max_tokens = max_tokens
        if effective_max_tokens is None:
            sm = _get_settings_manager()
            if sm is not None:
                try:
                    ms = sm.get_settings(resolved_model)
                    if ms is not None and getattr(ms, "max_tokens", None) is not None:
                        effective_max_tokens = ms.max_tokens
                except Exception:
                    pass

        transcribe_kwargs: dict = {"language": language}
        if prompt is not None:
            transcribe_kwargs["prompt"] = prompt
        if effective_max_tokens is not None:
            transcribe_kwargs["max_tokens"] = effective_max_tokens
        if word_timestamps:
            transcribe_kwargs["word_timestamps"] = True

        if stream:
            # Word timestamps only exist in the JSON segment response;
            # SSE streaming emits plain text deltas (matching OpenAI, which
            # also rejects timestamp granularity with stream=true).
            transcribe_kwargs.pop("word_timestamps", None)
            # The event generator owns tmp_path from here: its finally block
            # deletes the file once the stream completes or errors — the
            # route's finally must not remove it while chunks are pending.
            events = _stream_transcription_events(
                engine, tmp_path, resolved_model, transcribe_kwargs
            )
            tmp_path = None
            first_event = await events.__anext__()
            return StreamingResponse(
                _stream_with_prefetched_chunk(first_event, events),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
            )

        result = await engine.transcribe(tmp_path, **transcribe_kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    _record_audio_request(resolved_model)

    # Build response directly from the dict returned by STTEngine
    segments = result.get("segments") or None

    return AudioTranscriptionResponse(
        text=result.get("text", ""),
        language=result.get("language"),
        duration=result.get("duration"),
        segments=segments,
    )


@router.get("/v1/audio/voices")
async def list_model_voices(model: Optional[str] = None):
    """List built-in speaker/voice names for a TTS model.

    Reads static metadata only — a ``voices/`` directory (Kokoro-style)
    or the speaker table in ``config.json`` (Qwen3-TTS CustomVoice's
    ``talker_config.spk_id``) — so the model does not need to be loaded.
    Returns ``{"model": ..., "voices": [...]}``; an empty list means the
    model has no named speakers (e.g. voice-cloning base models).
    """
    if not model:
        raise HTTPException(
            status_code=400, detail="'model' query parameter is required"
        )
    pool = _get_engine_pool()
    resolved = _resolve_model(model)
    entry = pool.get_entry(resolved)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{resolved}' not found"
        )

    model_dir = Path(entry.model_path)
    voices: list[str] = []
    voices_dir = model_dir / "voices"
    if voices_dir.is_dir():
        voices = sorted({
            f.stem
            for f in voices_dir.iterdir()
            if f.suffix in (".safetensors", ".pt")
        })
    else:
        try:
            config = json.loads((model_dir / "config.json").read_text())
        except (OSError, ValueError):
            config = {}
        talker = config.get("talker_config") or {}
        spk = talker.get("spk_id") or config.get("spk_id") or {}
        if isinstance(spk, dict):
            voices = sorted(spk.keys())
    return {"model": resolved, "voices": voices}


@router.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    """OpenAI-compatible text-to-speech endpoint."""
    from omlx.engine.tts import TTSEngine
    from omlx.exceptions import ModelNotFoundError

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="'input' field must not be empty")
    # Only WAV is produced (no transcoding); reject unsupported response_format
    # with a clear 400 instead of silently returning WAV. (#753, #1013)
    if request.response_format not in (None, "wav"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"response_format '{request.response_format}' is not supported; "
                "only 'wav' is currently available"
            ),
        )
    streaming_interval = DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS
    if request.stream:
        streaming_interval = _resolve_tts_streaming_interval(request)

    audio_bytes = _decode_ref_audio_base64(request)

    pool = _get_engine_pool()
    resolved_model = _resolve_model(request.model)

    try:
        engine = await pool.get_engine(resolved_model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved_model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(engine, TTSEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{resolved_model}' is not a text-to-speech model",
        )

    ref_audio_path = _write_ref_audio_tempfile(audio_bytes)

    if request.stream:
        stream = _stream_speech_response(
            engine,
            request,
            ref_audio_path,
            streaming_interval,
        )
        try:
            first_chunk = await stream.__anext__()
        except StopAsyncIteration as exc:
            raise HTTPException(
                status_code=500,
                detail="TTS streaming produced no audio output",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return StreamingResponse(
            _stream_with_prefetched_chunk(first_chunk, stream),
            media_type="audio/wav",
        )

    try:
        wav_bytes = await engine.synthesize(
            request.input,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
            instructions=request.instructions,
            ref_audio=ref_audio_path,
            ref_text=request.ref_text,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_tokens=request.max_tokens,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _cleanup_tempfile(ref_audio_path)

    _record_audio_request(resolved_model)

    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/v1/audio/process")
async def process_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
):
    """Audio processing endpoint (speech enhancement, source separation, STS).

    Accepts a multipart audio file upload and a model identifier, processes
    the audio through an STS engine (e.g. DeepFilterNet, MossFormer2,
    SAMAudio, LFM2.5-Audio), and returns WAV bytes of the processed audio.
    """
    from omlx.engine.sts import STSEngine
    from omlx.exceptions import ModelNotFoundError

    pool = _get_engine_pool()
    resolved_model = _resolve_model(model)

    # Load the engine via pool (handles model loading and LRU eviction)
    try:
        engine = await pool.get_engine(resolved_model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{resolved_model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(engine, STSEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{resolved_model}' is not a speech-to-speech / audio processing model",
        )

    # Save uploaded file to a temp path so the engine can open it by path.
    # Remap video container extensions to .m4a so mlx-audio routes them
    # through ffmpeg instead of miniaudio (which can't decode containers).
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    if suffix.lower() in _VIDEO_CONTAINERS:
        suffix = ".m4a"
    tmp_path = None
    try:
        content = await _read_upload(file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        wav_bytes = await engine.process(tmp_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    _record_audio_request(resolved_model)

    return Response(content=wav_bytes, media_type="audio/wav")
