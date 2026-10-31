# SPDX-License-Identifier: Apache-2.0
"""
Audio API routes for oMLX.

This module provides OpenAI-compatible audio endpoints:
- POST /v1/audio/transcriptions  - Speech-to-Text
- POST /v1/audio/speech          - Text-to-Speech
- POST /v1/audio/process         - Speech-to-Speech / audio processing
"""

import asyncio
import base64
import contextlib
import io
import json
import logging
import math
import os
import re
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response, StreamingResponse
from PIL import Image

from ..engine.audio_utils import wav_bytes_to_pcm_frames, wav_header
from ..engine.lfm3_speech import SPEECH_MODEL_TYPES, VOICES
from ..engine.realtime_voice import (
    SAMPLE_RATE,
    RealtimeVoiceError,
    RealtimeVoiceSession,
    ResponseResult,
    pcm16_bytes,
)
from ..engine.vlm import (
    VLMBatchedEngine,
    realtime_voice_capabilities,
    realtime_voice_instructions,
    realtime_voice_names,
)
from ..exceptions import ModelNotFoundError
from ..server_metrics import get_server_metrics
from ..utils import gpu_power
from .audio_models import AudioSpeechRequest, AudioTranscriptionResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Separate router for the realtime WebSocket: the main audio router is
# mounted with an HTTP-only auth dependency that cannot resolve
# in a websocket scope (and browsers cannot set an Authorization header on
# WebSocket connections anyway). Auth happens in-band via the first
# {"type": "start"} message instead.
realtime_router = APIRouter()

# Maximum upload size for audio files (100 MB). Used as the default when
# server settings are not initialized.
MAX_AUDIO_UPLOAD_BYTES = 100 * 1024 * 1024

# Maximum base64-encoded ref_audio size (~15 MB raw audio, enough for ~60s).
MAX_REF_AUDIO_BASE64_BYTES = 20 * 1024 * 1024

# Default native TTS chunk cadence. Keep this below the mlx-audio default to
# improve TTFT while still letting the model process the full input at once.
DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS = 0.2
MIN_NATIVE_TTS_STREAMING_INTERVAL_SECONDS = 0.01

# Non-streaming TTS output formats and their Content-Type. wav is the native
# engine output; the others are transcoded in memory by soundfile's bundled
# libsndfile (lame/opus/flac), which the audio extra already ships via
# mlx-audio -> librosa. aac is not offered (libsndfile has no aac encoder).
_SPEECH_RESPONSE_FORMATS = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}

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


def _get_gpu_power_limiter():
    """Return the server's GPU power limiter, or None outside a full server."""
    try:
        from omlx.server import _server_state
    except Exception:
        return None
    return getattr(_server_state, "gpu_power_limiter", None)


def _realtime_gpu_power_limit(requested: int | None, model_id: str) -> int | None:
    """Return the lower of the session's and the model settings' GPU power caps."""
    from omlx.server import _resolve_gpu_power_limit

    return _resolve_gpu_power_limit(
        SimpleNamespace(gpu_power_limit_w=requested), model_id
    )


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


def _max_audio_upload_bytes() -> int:
    """Return the configured audio upload limit, falling back to the default."""
    try:
        from omlx.settings import get_settings

        return get_settings().server.max_audio_upload_bytes()
    except RuntimeError:
        return MAX_AUDIO_UPLOAD_BYTES
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning(
            "Invalid max_audio_upload_size; using %s byte default: %s",
            MAX_AUDIO_UPLOAD_BYTES,
            exc,
        )
        return MAX_AUDIO_UPLOAD_BYTES


async def _read_upload(file: UploadFile) -> bytes:
    """Read an uploaded file in chunks, bailing early if it exceeds the limit."""
    limit = _max_audio_upload_bytes()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio file exceeds maximum allowed size "
                    f"({limit} bytes)"
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
    max_tokens: Optional[int] = Form(None, gt=0),
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
        engine = _lfm3_speech(engine)
    if engine is None:
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


def _lfm3_speech(engine):
    """Return the speech service of an LFM3-Audio or LFM3-Omni engine, or None."""
    return engine.speech if isinstance(engine, VLMBatchedEngine) else None


def _verify_ws_api_key(websocket: WebSocket, api_key: Optional[str]) -> bool:
    """Verify an in-band API key, including the manual inference opt-in.

    WebSocket connections from browsers cannot carry an Authorization
    header, so the key arrives inside the {"type": "start"} message.
    """
    from omlx.server import _server_state, allows_unauthenticated_inference

    if allows_unauthenticated_inference():
        return True
    if _server_state.api_key is None:
        return True
    gs = _server_state.global_settings
    if gs is not None and gs.auth.skip_api_key_verification:
        return True
    if not api_key or not isinstance(api_key, str):
        return False

    from omlx.admin.auth import check_api_key, verify_any_api_key

    sub_keys = gs.auth.sub_keys if gs is not None else []
    return check_api_key(
        websocket,
        lambda: verify_any_api_key(api_key, _server_state.api_key, sub_keys),
    )


@realtime_router.websocket("/v1/audio/transcriptions/realtime")
async def realtime_transcription(websocket: WebSocket) -> None:
    """Push-based realtime transcription over WebSocket.

    Protocol:
      client → {"type": "start", "model": ..., "api_key": ..., "language"?: ...}
      server → {"type": "ready"} or {"type": "error", "detail": ...}
      client → binary frames: 16 kHz mono little-endian Int16 PCM
      server → {"type": "transcript.delta", "delta": ...} as text commits
      client → {"type": "stop"} (or closes the socket to abandon)
      server → remaining deltas, then {"type": "transcript.done", "text": ...}
    """
    await websocket.accept()

    async def _reject(detail: str, close_code: int = 1008) -> None:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "detail": detail})
            await websocket.close(code=close_code)

    try:
        start = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
    except asyncio.TimeoutError:
        await _reject("Handshake timeout")
        return
    except WebSocketDisconnect:
        return
    except Exception:
        await _reject("Invalid handshake message")
        return

    if not isinstance(start, dict) or start.get("type") != "start":
        await _reject("First message must be a {'type': 'start'} object")
        return
    try:
        key_ok = _verify_ws_api_key(websocket, start.get("api_key"))
    except HTTPException as exc:
        await _reject(str(exc.detail))
        return
    if not key_ok:
        await _reject("Invalid API key")
        return
    model_id = start.get("model")
    if not model_id or not isinstance(model_id, str):
        await _reject("Missing 'model' in start message")
        return
    language = start.get("language") or None

    resolved_model = _resolve_model(model_id)
    try:
        pool = _get_engine_pool()
    except HTTPException as exc:
        await _reject(str(exc.detail))
        return
    try:
        engine = await pool.get_engine(resolved_model)
    except Exception as exc:
        await _reject(f"Failed to load model '{model_id}': {exc}")
        return

    from ..engine.stt import STTEngine

    if not isinstance(engine, STTEngine):
        await _reject(
            f"Model '{model_id}' is not a speech-to-text model. "
            "Realtime transcription requires an audio_stt model."
        )
        return
    if not engine.supports_realtime_stt():
        await _reject(
            f"Model '{model_id}' does not support realtime transcription. "
            "Use POST /v1/audio/transcriptions with a complete audio file."
        )
        return
    try:
        session = await engine.create_realtime_session(language=language)
    except RuntimeError as exc:
        await _reject(str(exc))
        return

    stop_event = asyncio.Event()
    client_gone = False

    async def _receiver() -> None:
        nonlocal client_gone
        while not stop_event.is_set():
            try:
                message = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError):
                client_gone = True
                stop_event.set()
                return
            if message.get("type") == "websocket.disconnect":
                client_gone = True
                stop_event.set()
                return
            data = message.get("bytes")
            if data:
                session.feed_pcm16(data)
                continue
            text = message.get("text")
            if text:
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue
                if isinstance(payload, dict) and payload.get("type") == "stop":
                    stop_event.set()
                    return

    full_text: list[str] = []

    async def _send_deltas(deltas: list[str]) -> None:
        for delta in deltas:
            full_text.append(delta)
            await websocket.send_json({"type": "transcript.delta", "delta": delta})

    receiver = asyncio.create_task(_receiver())
    try:
        await websocket.send_json({"type": "ready"})
        while not stop_event.is_set():
            deltas = await session.poll()
            if deltas:
                await _send_deltas(deltas)
            else:
                # No committed text this round — wait briefly for more audio
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=0.1)
        if not client_gone:
            deltas = await session.close()
            await _send_deltas(deltas)
            await websocket.send_json(
                {"type": "transcript.done", "text": "".join(full_text)}
            )
            await websocket.close(code=1000)
        _record_audio_request(resolved_model)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Realtime transcription failed for %s", resolved_model)
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "detail": str(exc)})
            await websocket.close(code=1011)
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver
        await session.release()


REALTIME_IDLE_SECONDS = 600
REALTIME_UNLOAD_POLL_SECONDS = 1.0
MAX_REALTIME_IMAGE_BYTES = 20 * 1024 * 1024
# Larger images only cost prefill time in a spoken reply.
MAX_REALTIME_IMAGE_SIDE = 2048
_REALTIME_KEY_PROTOCOL = "openai-insecure-api-key."


def _realtime_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _realtime_api_key(websocket: WebSocket) -> str | None:
    """Read the key from the Authorization header or the browser subprotocol."""
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    for protocol in websocket.scope.get("subprotocols") or []:
        if protocol.startswith(_REALTIME_KEY_PROTOCOL):
            return protocol[len(_REALTIME_KEY_PROTOCOL) :]
    return None


def _realtime_session_object(
    session_id: str, model: str, session: RealtimeVoiceSession
) -> dict:
    pcm = {"type": "audio/pcm", "rate": SAMPLE_RATE}
    return {
        "object": "realtime.session",
        "type": "realtime",
        "id": session_id,
        "model": model,
        "output_modalities": ["audio"],
        "instructions": session.instructions,
        "tools": [{"type": "function", **tool["function"]} for tool in session.tools],
        "tool_choice": "auto",
        # Not in the OpenAI session object: what this model's backend accepts.
        "capabilities": session.capabilities,
        "audio": {
            "input": {"format": pcm, "turn_detection": None},
            "output": {"format": pcm, "voice": session.voice},
        },
        "max_output_tokens": session.max_output_tokens or "inf",
    }


def _check_realtime_format(value) -> None:
    if value is None or value == "pcm16":
        return
    if (
        isinstance(value, dict)
        and value.get("type") == "audio/pcm"
        and value.get("rate", SAMPLE_RATE) == SAMPLE_RATE
    ):
        return
    raise RealtimeVoiceError("unsupported", "only 24 kHz PCM16 audio is supported")


def _parse_session_update(config: dict) -> dict:
    """Read the supported fields of an OpenAI Realtime session object."""
    audio = config.get("audio") or {}
    audio_in, audio_out = audio.get("input") or {}, audio.get("output") or {}
    # The client commits each turn; the server never detects speech.
    if (
        config.get("turn_detection") is not None
        or audio_in.get("turn_detection") is not None
    ):
        raise RealtimeVoiceError(
            "unsupported",
            "server turn detection is not supported; set turn_detection to null "
            "and commit each turn",
        )
    for value in (
        config.get("input_audio_format"),
        config.get("output_audio_format"),
        audio_in.get("format"),
        audio_out.get("format"),
    ):
        _check_realtime_format(value)
    max_tokens = config.get(
        "max_output_tokens", config.get("max_response_output_tokens")
    )
    if max_tokens == "inf":
        max_tokens = 0
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0
    ):
        raise RealtimeVoiceError(
            "invalid_value", "max_output_tokens must be an integer"
        )
    tool_choice = config.get("tool_choice")
    if tool_choice not in (None, "auto", "none"):
        raise RealtimeVoiceError("unsupported", "tool_choice must be auto or none")
    tools = config.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise RealtimeVoiceError("invalid_value", "tools must be a list")
        tools = [_chat_tool(tool) for tool in tools]
    if tool_choice == "none":
        tools = []
    return {
        "instructions": config.get("instructions"),
        "voice": config.get("voice") or audio_out.get("voice"),
        "max_output_tokens": max_tokens,
        "tools": tools,
    }


def _chat_tool(tool) -> dict:
    """A Realtime function tool in the chat-completions shape the templates read."""
    if (
        not isinstance(tool, dict)
        or tool.get("type") != "function"
        or not isinstance(tool.get("name"), str)
    ):
        raise RealtimeVoiceError(
            "invalid_value", "each tool must be a function tool with a name"
        )
    function = {"name": tool["name"]}
    for key in ("description", "parameters"):
        if key in tool:
            function[key] = tool[key]
    return {"type": "function", "function": function}


def _realtime_images(item: dict) -> list:
    """Decode the input_image parts of a user message item."""
    if item.get("role") != "user":
        raise RealtimeVoiceError("unsupported", "only user messages can be added")
    images = []
    for part in item.get("content") or []:
        if not isinstance(part, dict):
            part = {}
        url = part.get("image_url")
        if part.get("type") != "input_image" or not isinstance(url, str):
            raise RealtimeVoiceError(
                "unsupported", "a user message item can only carry input_image parts"
            )
        header, _, payload = url.partition(",")
        if not header.startswith("data:image/") or ";base64" not in header:
            raise RealtimeVoiceError(
                "invalid_value", "image_url must be a base64 data URL"
            )
        if len(payload) > MAX_REALTIME_IMAGE_BYTES:
            raise RealtimeVoiceError("invalid_value", "the image is too large")
        try:
            image = Image.open(io.BytesIO(base64.b64decode(payload, validate=True)))
            image = image.convert("RGB")
        except Exception as exc:
            raise RealtimeVoiceError(
                "invalid_value", "the image could not be decoded"
            ) from exc
        image.thumbnail((MAX_REALTIME_IMAGE_SIDE, MAX_REALTIME_IMAGE_SIDE))
        images.append(image)
    if not images:
        raise RealtimeVoiceError("invalid_value", "the message item has no image")
    return images


@realtime_router.websocket("/v1/realtime")
async def realtime_voice(websocket: WebSocket, model: str | None = None) -> None:
    """Spoken conversation over WebSocket, following the OpenAI Realtime API.

    The client streams 24 kHz PCM16 with ``input_audio_buffer.append``, ends
    each turn with ``input_audio_buffer.commit`` and asks for the spoken reply
    with ``response.create``. There is no server turn detection; barge-in is
    ``response.cancel`` followed by ``conversation.item.truncate``.
    """
    offered = websocket.scope.get("subprotocols") or []
    await websocket.accept(subprotocol="realtime" if "realtime" in offered else None)
    send_lock = asyncio.Lock()

    async def send(event: dict) -> None:
        event.setdefault("event_id", _realtime_id("event"))
        async with send_lock:
            await websocket.send_json(event)

    async def send_error(code: str, message: str, client_event_id=None) -> None:
        await send(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": code,
                    "message": message,
                    "param": None,
                    "event_id": client_event_id,
                },
            }
        )

    async def reject(code: str, message: str, close_code: int = 1008) -> None:
        with contextlib.suppress(Exception):
            await send_error(code, message)
            await websocket.close(code=close_code)

    try:
        key_ok = _verify_ws_api_key(websocket, _realtime_api_key(websocket))
    except HTTPException as exc:
        await reject("invalid_api_key", str(exc.detail))
        return
    if not key_ok:
        await reject("invalid_api_key", "Invalid API key")
        return
    if not model:
        await reject("missing_model", "Pass the model as the 'model' query parameter")
        return
    raw_limit = websocket.query_params.get("gpu_power_limit_w")
    requested_limit = None
    if raw_limit is not None:
        try:
            requested_limit = int(raw_limit)
        except ValueError:
            requested_limit = 0
        if not gpu_power.MIN_WATTS <= requested_limit <= gpu_power.MAX_WATTS:
            await reject(
                "invalid_value",
                f"gpu_power_limit_w must be an integer from {gpu_power.MIN_WATTS} "
                f"to {gpu_power.MAX_WATTS}",
            )
            return
    resolved_model = _resolve_model(model)
    try:
        pool = _get_engine_pool()
    except HTTPException as exc:
        await reject("server_error", str(exc.detail), 1011)
        return
    try:
        engine = await pool.get_engine(resolved_model, _lease=True)
    except ModelNotFoundError:
        await reject("model_not_found", f"Model '{model}' not found")
        return
    except Exception as exc:
        await reject(
            "model_load_failed", f"Failed to load model '{model}': {exc}", 1011
        )
        return

    session: RealtimeVoiceSession | None = None
    limiter = None
    gpu_limit = None
    try:
        if (
            not isinstance(engine, VLMBatchedEngine)
            or not engine.supports_realtime_voice
        ):
            await reject(
                "unsupported_model", f"Model '{model}' does not support realtime voice"
            )
            return
        try:
            session = await engine.create_realtime_session()
        except RealtimeVoiceError as exc:
            await reject(
                exc.code, str(exc), 1013 if exc.code == "session_limit" else 1008
            )
            return
        # The cap holds for the whole conversation, like a capped chat request.
        gpu_limit = _realtime_gpu_power_limit(requested_limit, resolved_model)
        limiter = _get_gpu_power_limiter() if gpu_limit is not None else None
        if limiter is not None:
            limiter.acquire(gpu_limit)
        session_id = _realtime_id("sess")
        await send(
            {
                "type": "session.created",
                "session": _realtime_session_object(session_id, model, session),
            }
        )
        await _run_realtime_voice(
            websocket,
            send,
            send_error,
            session,
            session_id,
            model,
            resolved_model,
            pool,
        )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Realtime voice failed for %s", resolved_model)
        with contextlib.suppress(Exception):
            await send_error("server_error", str(exc))
            await websocket.close(code=1011)
    finally:
        try:
            if session is not None:
                await session.close()
        finally:
            try:
                if limiter is not None:
                    limiter.release(gpu_limit)
            finally:
                await pool.release_engine(resolved_model)


async def _run_realtime_voice(
    websocket: WebSocket,
    send,
    send_error,
    session: RealtimeVoiceSession,
    session_id: str,
    model: str,
    resolved_model: str,
    pool,
) -> None:
    items: list[str] = []
    last_reply: str | None = None

    def reply_callbacks(response_id: str, item_id: str, announced: asyncio.Event):
        ids = {"response_id": response_id, "item_id": item_id}
        part = {"output_index": 0, "content_index": 0}
        transcript: list[str] = []

        async def send_delta(event: dict) -> None:
            try:
                await send(event)
            except Exception:
                # The client is gone; stop the reply instead of failing it.
                session.cancel()

        async def on_text(delta: str) -> None:
            await announced.wait()
            transcript.append(delta)
            await send_delta(
                {
                    "type": "response.output_audio_transcript.delta",
                    **ids,
                    **part,
                    "delta": delta,
                }
            )

        async def on_audio(audio) -> None:
            await announced.wait()
            delta = base64.b64encode(pcm16_bytes(audio)).decode("ascii")
            await send_delta(
                {"type": "response.output_audio.delta", **ids, **part, "delta": delta}
            )

        async def send_function_calls(result: ResponseResult) -> list[dict]:
            calls = []
            for index, call in enumerate(result.function_calls, start=1):
                item = {
                    "id": _realtime_id("item"),
                    "object": "realtime.item",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                where = {"response_id": response_id, "output_index": index}
                await send(
                    {
                        "type": "response.output_item.added",
                        **where,
                        "item": {**item, "status": "in_progress", "arguments": ""},
                    }
                )
                await send(
                    {
                        "type": "response.function_call_arguments.done",
                        **where,
                        "item_id": item["id"],
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                )
                await send({"type": "response.output_item.done", **where, "item": item})
                items.append(item["id"])
                calls.append(item)
            return calls

        async def on_done(result: ResponseResult) -> None:
            await announced.wait()
            text = "".join(transcript)
            item = {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "completed" if result.status == "completed" else "incomplete",
                "content": [{"type": "output_audio", "transcript": text}],
            }
            if result.error is not None:
                await send_error(result.error.code, str(result.error))
            await send({"type": "response.output_audio.done", **ids, **part})
            await send(
                {
                    "type": "response.output_audio_transcript.done",
                    **ids,
                    **part,
                    "transcript": text,
                }
            )
            await send(
                {
                    "type": "response.output_item.done",
                    "response_id": response_id,
                    "output_index": 0,
                    "item": item,
                }
            )
            calls = await send_function_calls(result)
            await send(
                {
                    "type": "response.done",
                    "response": {
                        "object": "realtime.response",
                        "id": response_id,
                        "status": result.status,
                        "output": [item, *calls],
                        "usage": {
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_slots,
                            "total_tokens": result.input_tokens + result.output_slots,
                        },
                    },
                }
            )
            _record_audio_request(resolved_model)

        return on_text, on_audio, on_done

    async def watch_unload() -> None:
        while True:
            await asyncio.sleep(REALTIME_UNLOAD_POLL_SECONDS)
            reason = pool.get_abort_requested_reason(resolved_model)
            if reason:
                with contextlib.suppress(Exception):
                    await send_error(
                        "model_unloaded", f"The model is being unloaded: {reason}"
                    )
                    await websocket.close(code=1012)
                return

    watcher = asyncio.create_task(watch_unload())
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    websocket.receive_json(), timeout=REALTIME_IDLE_SECONDS
                )
            except TimeoutError:
                await send_error("session_expired", "The session was idle for too long")
                await websocket.close(code=1000)
                return
            except ValueError:
                await send_error("invalid_json", "Events must be JSON objects")
                continue
            if not isinstance(event, dict):
                await send_error("invalid_json", "Events must be JSON objects")
                continue
            kind, client_event_id = event.get("type"), event.get("event_id")
            try:
                if kind == "session.update":
                    session.update(**_parse_session_update(event.get("session") or {}))
                    await send(
                        {
                            "type": "session.updated",
                            "session": _realtime_session_object(
                                session_id, model, session
                            ),
                        }
                    )
                elif kind == "input_audio_buffer.append":
                    session.append(
                        base64.b64decode(event.get("audio") or "", validate=True)
                    )
                elif kind == "input_audio_buffer.clear":
                    session.clear()
                    await send({"type": "input_audio_buffer.cleared"})
                elif kind == "input_audio_buffer.commit":
                    session.commit()
                    item_id = _realtime_id("item")
                    previous = items[-1] if items else None
                    items.append(item_id)
                    await send(
                        {
                            "type": "input_audio_buffer.committed",
                            "item_id": item_id,
                            "previous_item_id": previous,
                        }
                    )
                elif kind == "response.create":
                    response_id, item_id = _realtime_id("resp"), _realtime_id("item")
                    announced = asyncio.Event()
                    session.start_response(
                        *reply_callbacks(response_id, item_id, announced)
                    )
                    items.append(item_id)
                    last_reply = item_id
                    # The reply callbacks wait for these two events; a failed send must
                    # not leave them, and the session close behind them, waiting forever.
                    try:
                        await send(
                            {
                                "type": "response.created",
                                "response": {
                                    "object": "realtime.response",
                                    "id": response_id,
                                    "status": "in_progress",
                                    "output": [],
                                },
                            }
                        )
                        await send(
                            {
                                "type": "response.output_item.added",
                                "response_id": response_id,
                                "output_index": 0,
                                "item": {
                                    "id": item_id,
                                    "object": "realtime.item",
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                            }
                        )
                    finally:
                        announced.set()
                elif kind == "conversation.item.create":
                    item = event.get("item")
                    if not isinstance(item, dict):
                        raise RealtimeVoiceError(
                            "invalid_value", "item must be an object"
                        )
                    added = {
                        "id": item.get("id") or _realtime_id("item"),
                        "object": "realtime.item",
                        "type": item.get("type"),
                    }
                    if item.get("type") == "function_call_output":
                        call_id, output = item.get("call_id"), item.get("output")
                        if not isinstance(call_id, str) or not isinstance(output, str):
                            raise RealtimeVoiceError(
                                "invalid_value",
                                "function_call_output needs call_id and output strings",
                            )
                        session.add_tool_output(call_id, output)
                        added["call_id"] = call_id
                    elif item.get("type") == "message":
                        for image in await asyncio.to_thread(_realtime_images, item):
                            session.add_image(image)
                        added["role"] = "user"
                    else:
                        raise RealtimeVoiceError(
                            "unsupported",
                            "only user image messages and function_call_output items "
                            "can be added",
                        )
                    previous = items[-1] if items else None
                    items.append(added["id"])
                    await send(
                        {
                            "type": "conversation.item.added",
                            "previous_item_id": previous,
                            "item": added,
                        }
                    )
                elif kind == "response.cancel":
                    session.cancel()
                elif kind == "conversation.item.truncate":
                    if event.get("item_id") != last_reply:
                        raise RealtimeVoiceError(
                            "item_not_found",
                            "only the latest assistant reply can be truncated",
                        )
                    audio_end_ms = event.get("audio_end_ms")
                    if not isinstance(audio_end_ms, int) or isinstance(
                        audio_end_ms, bool
                    ):
                        raise RealtimeVoiceError(
                            "invalid_value", "audio_end_ms must be an integer"
                        )
                    await session.truncate(audio_end_ms)
                    await send(
                        {
                            "type": "conversation.item.truncated",
                            "item_id": last_reply,
                            "content_index": event.get("content_index", 0),
                            "audio_end_ms": audio_end_ms,
                        }
                    )
                else:
                    raise RealtimeVoiceError(
                        "unknown_event", f"Unsupported event type {kind!r}"
                    )
            except RealtimeVoiceError as exc:
                await send_error(exc.code, str(exc), client_event_id)
            except ValueError as exc:
                await send_error("invalid_value", str(exc), client_event_id)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher


@router.get("/v1/audio/voices")
async def list_model_voices(model: Optional[str] = None, realtime: bool = False):
    """List built-in speaker/voice names for a TTS model.

    With ``realtime=true`` it lists the voices a /v1/realtime session on the
    model can pick, which can differ from its TTS voices, and adds the
    session's default ``instructions`` and its ``capabilities``.

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
    if realtime:
        try:
            config = json.loads((model_dir / "config.json").read_text())
        except (OSError, ValueError):
            config = {}
        return {
            "model": resolved,
            "voices": realtime_voice_names(config),
            "instructions": realtime_voice_instructions(config),
            "capabilities": realtime_voice_capabilities(config),
        }
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
        if config.get("model_type") in SPEECH_MODEL_TYPES:
            voices = list(VOICES)
        elif isinstance(spk, dict):
            voices = sorted(spk.keys())
    return {"model": resolved, "voices": voices}


def _transcode_speech_output(wav_bytes: bytes, response_format: str) -> bytes:
    """Transcode the engine's native WAV output to response_format in memory."""
    if response_format == "pcm":
        return wav_bytes_to_pcm_frames(wav_bytes)[3]

    # Lazy import like the other optional audio deps: soundfile ships with
    # the audio extra (mlx-audio -> librosa -> soundfile), and this module
    # must stay importable without it (see server.py route registration).
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    buf = io.BytesIO()
    if response_format == "mp3":
        sf.write(buf, data, sample_rate, format="MP3")
    elif response_format == "opus":
        sf.write(buf, data, sample_rate, format="OGG", subtype="OPUS")
    else:
        sf.write(buf, data, sample_rate, format="FLAC")
    return buf.getvalue()


@router.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    """OpenAI-compatible text-to-speech endpoint."""
    from omlx.engine.tts import TTSEngine
    from omlx.exceptions import ModelNotFoundError

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="'input' field must not be empty")
    response_format = request.response_format or "wav"
    if response_format not in _SPEECH_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"response_format '{request.response_format}' is not supported; "
                f"available formats: {', '.join(_SPEECH_RESPONSE_FORMATS)}"
            ),
        )
    streaming_interval = DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS
    if request.stream:
        if response_format != "wav":
            raise HTTPException(
                status_code=400,
                detail="Streaming TTS currently only supports response_format='wav'",
            )
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
        engine = _lfm3_speech(engine)
    if engine is None:
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

    audio_out = wav_bytes
    if response_format != "wav":
        try:
            audio_out = await asyncio.to_thread(
                _transcode_speech_output, wav_bytes, response_format
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=audio_out, media_type=_SPEECH_RESPONSE_FORMATS[response_format]
    )


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
        engine = _lfm3_speech(engine)
    if engine is None:
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
