# SPDX-License-Identifier: Apache-2.0
"""
Audio API routes for oMLX.

This module provides OpenAI-compatible audio endpoints:
- POST /v1/audio/transcriptions  - Speech-to-Text
- POST /v1/audio/speech          - Text-to-Speech
- POST /v1/audio/process         - Speech-to-Speech / audio processing
"""

import logging
import tempfile
import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from .audio_models import AudioSpeechRequest, AudioTranscriptionResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum upload size for audio files (100 MB).
MAX_AUDIO_UPLOAD_BYTES = 100 * 1024 * 1024

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/v1/audio/transcriptions", response_model=AudioTranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """OpenAI-compatible audio transcription endpoint (Speech-to-Text).

    Note: ``response_format`` and ``temperature`` are accepted for OpenAI API
    compatibility but are not yet implemented — they are silently ignored.
    """
    from omlx.engine.stt import STTEngine
    from omlx.exceptions import ModelNotFoundError

    pool = _get_engine_pool()
    model = _resolve_model(model)

    # Load the engine via pool (handles model loading and LRU eviction)
    try:
        engine = await pool.get_engine(model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(engine, STTEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not a speech-to-text model",
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

        result = await engine.transcribe(tmp_path, language=language)
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

    # Build response directly from the dict returned by STTEngine
    segments = result.get("segments") or None

    return AudioTranscriptionResponse(
        text=result.get("text", ""),
        language=result.get("language"),
        duration=result.get("duration"),
        segments=segments,
    )


@router.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    """OpenAI-compatible text-to-speech endpoint."""
    from omlx.engine.tts import TTSEngine
    from omlx.exceptions import ModelNotFoundError

    # Validate input is non-empty
    if not request.input:
        raise HTTPException(status_code=400, detail="'input' field must not be empty")

    pool = _get_engine_pool()
    resolved_model = _resolve_model(request.model)

    # Load the engine via pool
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

    try:
        wav_bytes = await engine.synthesize(
            request.input,
            voice=request.voice,
            speed=request.speed,
            instructions=request.instructions,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
    model = _resolve_model(model)

    # Load the engine via pool (handles model loading and LRU eviction)
    try:
        engine = await pool.get_engine(model)
    except ModelNotFoundError as exc:
        avail = ", ".join(exc.available_models) if exc.available_models else "(none)"
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found. Available: {avail}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(engine, STSEngine):
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not a speech-to-speech / audio processing model",
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

    return Response(content=wav_bytes, media_type="audio/wav")
