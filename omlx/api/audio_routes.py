# SPDX-License-Identifier: Apache-2.0
"""
Audio API routes for oMLX.

This module provides OpenAI-compatible audio endpoints:
- POST /v1/audio/transcriptions  - Speech-to-Text
- POST /v1/audio/speech          - Text-to-Speech
- POST /v1/audio/process         - Speech-to-Speech / audio processing
"""

import base64
import logging
import math
import os
import re
import tempfile
from typing import AsyncIterator, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

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


def _fmt_srt_time(t: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm"""
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_time(t: float) -> str:
    """WebVTT timestamp: HH:MM:SS.mmm"""
    return _fmt_srt_time(t).replace(",", ".")


def _group_words_into_cues(
    words: list[dict],
    max_chars: int = 40,
    max_duration: float = 5.0,
) -> list[dict]:
    """Pack consecutive word-level entries into subtitle cues.

    A cue is closed when any of these holds:
      * accumulated text reaches ``max_chars``
      * cue duration reaches ``max_duration`` seconds
      * the current word ends with a sentence-terminating punctuation
        (Chinese 。！？ or Western .!?)
      * the speaker label changes (one cue per speaker for cleaner
        rendering — applies only when words carry a ``speaker`` field
        from diarization)

    Each input word entry should have ``word`` / ``start`` / ``end`` keys
    (the shape produced by the Whisper word-timestamps path and the
    Qwen3-ForcedAligner auto-chain). The optional ``speaker`` key is
    carried into the resulting cue.
    """
    if not words:
        return []
    cues: list[dict] = []
    cur_text = ""
    cur_start: float | None = None
    cur_end = 0.0
    cur_speaker: str | None = None
    for w in words:
        token = (w.get("word") or "").strip("\n")
        if not token:
            continue
        w_speaker = w.get("speaker")
        # Force-close current cue if the speaker changed mid-stream.
        speaker_change = (
            cur_start is not None
            and cur_speaker is not None
            and w_speaker is not None
            and w_speaker != cur_speaker
        )
        if speaker_change:
            cue = {
                "start": cur_start,
                "end": max(cur_end, cur_start + 0.05),
                "text": cur_text.strip(),
            }
            if cur_speaker:
                cue["speaker"] = cur_speaker
            cues.append(cue)
            cur_text = ""
            cur_start = None
            cur_end = 0.0
            cur_speaker = None
        if cur_start is None:
            cur_start = float(w.get("start", 0.0))
            cur_speaker = w_speaker
        cur_text += token
        cur_end = float(w.get("end", cur_end))
        last = token[-1:] if token else ""
        cue_full = (
            len(cur_text) >= max_chars
            or (cur_end - cur_start) >= max_duration
            or last in "。！？.!?"
        )
        if cue_full:
            cue = {
                "start": cur_start,
                "end": max(cur_end, cur_start + 0.05),
                "text": cur_text.strip(),
            }
            if cur_speaker:
                cue["speaker"] = cur_speaker
            cues.append(cue)
            cur_text = ""
            cur_start = None
            cur_end = 0.0
            cur_speaker = None
    if cur_text.strip() and cur_start is not None:
        cue = {
            "start": cur_start,
            "end": max(cur_end, cur_start + 0.05),
            "text": cur_text.strip(),
        }
        if cur_speaker:
            cue["speaker"] = cur_speaker
        cues.append(cue)
    return cues


def _build_subtitle(result: dict, fmt: str = "srt") -> str:
    """Build SRT or WebVTT body from an STTEngine.transcribe() result.

    Prefers word-level entries (segments[*].words from word_timestamps /
    Forced-Aligner chain). Falls back to segment-level entries when no
    word data is present (one cue per ASR segment).
    """
    segments = result.get("segments") or []
    cues: list[dict] = []
    # 1) Word-level path
    all_words: list[dict] = []
    for seg in segments:
        ws = seg.get("words") or []
        if ws:
            all_words.extend(ws)
    if all_words:
        cues = _group_words_into_cues(all_words)
    # 2) Fall back to segment-level
    if not cues and segments:
        cues = [
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": (s.get("text") or "").strip(),
            }
            for s in segments
            if (s.get("text") or "").strip()
        ]
    # 3) Final fallback: a single cue from the whole text + duration
    if not cues and (result.get("text") or "").strip():
        cues = [{
            "start": 0.0,
            "end": float(result.get("duration") or 0.0),
            "text": result["text"].strip(),
        }]

    fmt = fmt.lower()
    fmt_time = _fmt_vtt_time if fmt == "vtt" else _fmt_srt_time
    lines: list[str] = []
    if fmt == "vtt":
        lines.append("WEBVTT")
        lines.append("")
    for idx, cue in enumerate(cues, start=1):
        speaker = cue.get("speaker")
        text = cue["text"]
        if fmt == "srt":
            lines.append(str(idx))
            lines.append(f"{fmt_time(cue['start'])} --> {fmt_time(cue['end'])}")
            # SRT has no standard speaker tag — most players that show
            # speaker labels look for a "Speaker: text" prefix on the cue.
            lines.append(f"{speaker}: {text}" if speaker else text)
        else:  # vtt
            lines.append(f"{fmt_time(cue['start'])} --> {fmt_time(cue['end'])}")
            # WebVTT has a built-in voice tag: <v Speaker>text</v>.
            lines.append(
                f"<v {speaker}>{text}</v>" if speaker else text
            )
        lines.append("")
    return "\n".join(lines)


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
                "TTS native streaming start: model=%s, text_len=%d, voice=%s",
                request.model, len(request.input), request.voice,
            )
            stream_format: Optional[tuple[int, int, int]] = None
            try:
                async for sample_rate, channels, sample_width, pcm_bytes in engine.stream_synthesize_pcm(
                    request.input,
                    voice=request.voice,
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
            "TTS streaming start: model=%s, text_len=%d, segments=%d, voice=%s",
            request.model, len(request.input), len(segments), request.voice,
        )

        stream_format: Optional[tuple[int, int, int]] = None
        for idx, segment in enumerate(segments, start=1):
            wav_bytes = await engine.synthesize(
                segment,
                voice=request.voice,
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


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    top_p: Optional[float] = Form(None),
    top_k: Optional[int] = Form(None),
    min_p: Optional[float] = Form(None),
    repetition_penalty: Optional[float] = Form(None),
    repetition_context_size: Optional[int] = Form(None),
    frequency_penalty: Optional[float] = Form(None),
    frequency_context_size: Optional[int] = Form(None),
    presence_penalty: Optional[float] = Form(None),
    presence_context_size: Optional[int] = Form(None),
    max_tokens: Optional[int] = Form(None),
    word_timestamps: bool = Form(False),
    n: int = Form(1),
    # ---- Diarization: Phase 1 energy backend for stereo+L/R, Phase 2 pyannote
    # for mono / multi-speaker / conference audio.
    diarize_backend: str = Form("auto"),
    left_speaker: Optional[str] = Form(None),
    right_speaker: Optional[str] = Form(None),
    diarize_threshold: float = Form(1.3),
    speakers: Optional[str] = Form(None),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(2),
    max_speakers: Optional[int] = Form(8),
):
    """OpenAI-compatible audio transcription endpoint (Speech-to-Text).

    ``prompt`` is forwarded to the underlying STT backend as a transcription-
    biasing context (Whisper's ``initial_prompt``, Qwen2-Audio's ``prompt``,
    etc.) — useful for steering domain vocabulary, proper nouns, product
    names or stylistic preferences. Backends without any prompt-style kwarg
    drop it silently.

    ``text`` is a reference transcript for forced-alignment models like
    Qwen3-ForcedAligner. Calling this endpoint with ``model=<aligner>`` +
    ``text=<known transcript>`` returns word/char-level timestamps. Without
    a ``text`` the aligner model returns 400.

    ``word_timestamps`` is an oMLX extension. For Whisper-family models it
    exposes mlx-audio's native word-level alignment (each segment in the
    response gets a ``words`` array of ``{word, start, end, probability}``).
    For Qwen3-ASR and other backends without native word-level alignment,
    if the model has ``aligner_model`` set in its ModelSettings, the server
    automatically runs that aligner (e.g. Qwen3-ForcedAligner-0.6B-4bit)
    on the (audio, ASR transcript) pair and merges the result into
    ``segments[0].words``. Default False preserves the existing response
    shape for every current caller.

    ``temperature``, ``top_p``, ``top_k``, ``min_p``, ``repetition_penalty``,
    ``repetition_context_size`` are forwarded to the backend's sampler when
    non-default. Useful for breaking decoder degenerate loops on long
    silent/repetitive audio — e.g. Qwen3-ASR on mono channels can emit
    hundreds of repeated tokens; ``repetition_penalty=1.2`` with
    ``repetition_context_size=64`` is a typical fix. Qwen3-ASR and
    Qwen2-Audio accept all of these natively; Whisper has ``**decode_options``
    so unknown kwargs are silently ignored.

    ``frequency_penalty`` / ``presence_penalty`` (with optional
    ``frequency_context_size`` / ``presence_context_size``) are oMLX
    extensions for Qwen3-ASR. Upstream mlx-audio's ``Qwen3ASRModel.generate``
    hardcodes ``make_logits_processors`` to only emit a
    ``repetition_penalty`` processor. When either of these is set the
    engine takes an alternate path that builds the full mlx-lm logits-
    processors stack (``repetition`` + ``frequency`` + ``presence``) and
    drives ``_generate_single_chunk`` directly. Other STT backends do
    not yet support these — the request is rejected with 400 if the
    target model is not Qwen3-ASR.

    ``response_format`` supports ``json`` (default), ``verbose_json``,
    ``text``, ``srt`` and ``vtt``. SRT/VTT subtitle output auto-enables
    word_timestamps so the cue pacing isn't a single giant block; cues
    are packed by sentence-ending punctuation, 40-char text limit, or
    5-second duration cap, whichever comes first. When no word-level
    data is available the output falls back to one cue per ASR segment.

    ``n`` must be 1. Multi-candidate transcription is rejected with 400.

    ``max_tokens`` is an oMLX extension that raises the underlying model's
    output cap. Useful for long audio with models like VibeVoice-ASR whose
    mlx-audio default (8192) truncates ~24 min files. When omitted, the
    model's own default applies.

    ``diarize_backend`` selects the speaker-attribution backend:
      * ``energy`` — stereo only; single mix-pass ASR + per-word RMS
        comparison of L/R channels. Cheap, but words spoken simultaneously
        on both channels get a fallback ``[overlap]`` label and may be
        dropped by the mix-down (decoder can't separate two voices in
        one channel). Requires ``left_speaker`` / ``right_speaker``.
      * ``energy_tripass`` — stereo only; runs ASR three times (L mono,
        R mono, mix mono). Mix gives canonical text; L / R give
        deterministic speaker attribution + recover words the mix lost
        to simultaneous speech. 3x ASR latency, no overlap label,
        zero word loss on simultaneous talk. Requires
        ``left_speaker`` / ``right_speaker``.
      * ``pyannote`` — mono / multi-speaker; runs pyannote-audio's
        speaker-diarization-3.1 pipeline. First call lazily downloads and
        loads the ~400 MB model; subsequent calls are free. Requires the
        HuggingFace gated license at hf.co/pyannote/speaker-diarization-3.1
        to have been accepted by the running server's HF_TOKEN account.
      * ``auto`` (default) — energy when stereo + L/R given; pyannote when
        any of the multi-speaker fields (``speakers`` / ``num_speakers`` /
        non-default ``min_speakers``-``max_speakers``) is set; ``none``
        otherwise. The "signal of intent" gate keeps plain transcription
        requests from accidentally loading the pyannote weights.
      * ``none`` — skip diarization entirely.

    ``speakers`` is a comma-separated canonical name list. First pyannote
    label encountered maps to speakers[0], next new label to speakers[1],
    etc. Excess raw labels fall back to anonymous ``speaker_N``.
    ``num_speakers`` constrains pyannote to exactly N speakers;
    ``min_speakers`` / ``max_speakers`` are soft bounds for auto-detect.
    """
    if n != 1:
        raise HTTPException(
            status_code=400,
            detail="n must be 1; multi-candidate transcription is not supported.",
        )
    response_format = (response_format or "json").lower().strip()
    if response_format not in ("json", "verbose_json", "text", "srt", "vtt"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown response_format='{response_format}'. "
                "Supported: json, verbose_json, text, srt, vtt."
            ),
        )
    # Diarization backend validation. Phase 1 implements energy + none + auto
    # (auto routes to energy when stereo+L/R speakers given). pyannote backend
    # is reserved for Phase 2 — reject with 501 if explicitly requested.
    diarize_backend = (diarize_backend or "auto").lower().strip()
    if diarize_backend not in (
        "auto", "energy", "energy_tripass", "pyannote", "none"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown diarize_backend='{diarize_backend}'. "
                "Supported: auto, energy, energy_tripass, pyannote, none."
            ),
        )
    # pyannote backend (Phase 2) is now wired below the ASR call. Its install
    # / license check happens lazily inside omlx.engine.pyannote_diarize on
    # first use — a 501 here would block requests on a misconfigured site
    # before we have a chance to give the actionable error from the engine.
    # SRT/VTT need word-level timestamps to produce useful subtitle pacing —
    # otherwise we'd emit one giant cue per ASR segment. Auto-enable when
    # the caller asked for subtitles unless they explicitly opted out.
    if response_format in ("srt", "vtt") and not word_timestamps:
        word_timestamps = True
    # Energy-mode diarization needs word-level alignment to attribute each
    # word to a channel. Auto-enable when caller asked for energy diarize.
    energy_diarize_requested = (
        diarize_backend == "energy"
        or (diarize_backend == "auto" and left_speaker and right_speaker)
    )
    if energy_diarize_requested and not word_timestamps:
        word_timestamps = True
    # 3-pass energy: explicit opt-in only (3x ASR cost), never auto-routed.
    # Eats the boundary-disambiguation tax of per-channel ASR by also running
    # a mix pass for canonical text, while keeping the per-channel passes for
    # deterministic speaker attribution + recovery of words the mix dropped
    # to simultaneous speech.
    energy_tripass_requested = (diarize_backend == "energy_tripass")
    if energy_tripass_requested and not word_timestamps:
        word_timestamps = True
    # pyannote (Phase 2) routing. Explicit when diarize_backend=pyannote;
    # auto routes here when the caller signalled diarization intent via any
    # of the multi-speaker form fields (speakers / num_speakers / etc) but
    # didn't give the stereo L/R pair that energy mode needs. This mirrors
    # the energy auto-route — "auto + signal of intent" — so that
    # plain transcription requests don't accidentally pull the 400 MB
    # pyannote weights into memory.
    parsed_speakers: Optional[list[str]] = None
    if speakers:
        parsed_speakers = [
            s.strip() for s in speakers.split(",") if s.strip()
        ] or None
    pyannote_intent = bool(
        parsed_speakers
        or num_speakers is not None
        or (
            (min_speakers is not None or max_speakers is not None)
            # min_speakers default is 2, max_speakers default is 8 — those
            # are not "intent" signals on their own. Treat as intent only
            # when explicitly different from the defaults, i.e. user
            # actually overrode something. (Cheap heuristic; future
            # cleanup: switch min/max defaults to None.)
            and (min_speakers != 2 or max_speakers != 8)
        )
    )
    pyannote_diarize_requested = (
        diarize_backend == "pyannote"
        or (
            diarize_backend == "auto"
            and not energy_diarize_requested
            and pyannote_intent
        )
    )
    if pyannote_diarize_requested and not word_timestamps:
        word_timestamps = True
    from omlx.engine.stt import STTEngine
    from omlx.exceptions import ModelNotFoundError

    pool = _get_engine_pool()
    resolved_model = _resolve_model(model)

    # Settings-driven default for the ``auto`` backend's quality/cost tradeoff
    # on stereo + L/R recordings. When the per-model
    # ``default_diarize_quality`` is "high", auto-routes upgrade the cheap
    # 1-pass energy backend to 3-pass energy_tripass. Per-request explicit
    # ``diarize_backend=energy`` / ``=energy_tripass`` always wins (caller
    # already declared intent). Plain transcription requests (no L/R given)
    # are unaffected — quality preference only fires when energy auto-route
    # would already have triggered.
    if (
        diarize_backend == "auto"
        and energy_diarize_requested
        and not energy_tripass_requested
    ):
        try:
            _sm = _get_settings_manager()
            if _sm is not None:
                _ms = _sm.get_settings(resolved_model)
                if (
                    _ms is not None
                    and (getattr(_ms, "default_diarize_quality", None) or "").lower() == "high"
                ):
                    energy_diarize_requested = False
                    energy_tripass_requested = True
                    if not word_timestamps:
                        word_timestamps = True
        except Exception:
            # Settings lookup failure → fall through to standard energy
            # routing rather than 500. The diarize feature itself works
            # without settings; this just optimizes the default.
            pass

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
    mono_tmp_path = None
    tripass_L_path: Optional[str] = None
    tripass_R_path: Optional[str] = None
    stereo_audio_for_diarize = None
    stereo_sr_for_diarize = None
    try:
        content = await _read_upload(file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        # Energy-based diarization preprocessing: load the original audio with
        # soundfile to keep the stereo buffer for per-word RMS analysis later,
        # and feed a mono mix-down to the ASR (preserves the full-context
        # decoding advantage over per-channel ASR — boundary words like
        # "对" / "你/您" disambiguate much better when the model sees both
        # sides of the conversation). Tripass additionally writes per-channel
        # mono temp files for the L / R ASR passes.
        if energy_diarize_requested or energy_tripass_requested:
            if not left_speaker or not right_speaker:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "energy diarization requires both left_speaker and "
                        "right_speaker form fields."
                    ),
                )
            try:
                import soundfile as _sf
                import numpy as _np
                audio_data, audio_sr = _sf.read(tmp_path, dtype="float32")
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read audio for diarization: {exc}",
                ) from exc
            if audio_data.ndim != 2 or audio_data.shape[1] != 2:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "energy diarization requires stereo (2-channel) "
                        f"input; got shape {audio_data.shape}. For mono or "
                        "multi-speaker recordings use diarize_backend=pyannote "
                        "(set speakers / num_speakers to activate auto-routing)."
                    ),
                )
            stereo_audio_for_diarize = audio_data
            stereo_sr_for_diarize = int(audio_sr)
            # Down-mix to mono for ASR with 0.5 scaling to avoid clipping.
            mono = (audio_data[:, 0] + audio_data[:, 1]) * 0.5
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as mtf:
                mono_tmp_path = mtf.name
            _sf.write(mono_tmp_path, mono, audio_sr, subtype="FLOAT")
            tmp_path = mono_tmp_path
            if energy_tripass_requested:
                # Per-channel passes get pure L / R mono (no clipping concern;
                # each is already a single channel). The mix file above
                # serves as the third pass — keep tmp_path pointed at it so
                # the standard transcribe() call uses it as the canonical
                # decode, and L / R are run alongside.
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as ltf:
                    tripass_L_path = ltf.name
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as rtf:
                    tripass_R_path = rtf.name
                _sf.write(
                    tripass_L_path, audio_data[:, 0], audio_sr,
                    subtype="FLOAT",
                )
                _sf.write(
                    tripass_R_path, audio_data[:, 1], audio_sr,
                    subtype="FLOAT",
                )

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
        if effective_max_tokens is not None:
            transcribe_kwargs["max_tokens"] = effective_max_tokens
        if word_timestamps:
            transcribe_kwargs["word_timestamps"] = True
        if prompt:
            transcribe_kwargs["prompt"] = prompt
        # Forward `temperature` only when > 0. Most STT backends interpret
        # temperature=0 as greedy (their default), and several reject the
        # kwarg entirely; passing 0 just adds noise. >0 is the case that
        # matters — used to break decoder degenerate loops (e.g. Qwen3-ASR
        # on silent/quasi-silent mono input emits hundreds of repeated "嗯").
        if temperature is not None and temperature > 0:
            transcribe_kwargs["temperature"] = temperature
        # Sampling tail (Qwen3-ASR / Qwen2-Audio accept all of these natively;
        # Whisper has **decode_options so unknown kwargs are silently dropped).
        # repetition_penalty + repetition_context_size is the typical fix for
        # mono-channel long-silence degeneration on Qwen3-ASR.
        if top_p is not None:
            transcribe_kwargs["top_p"] = top_p
        if top_k is not None:
            transcribe_kwargs["top_k"] = top_k
        if min_p is not None:
            transcribe_kwargs["min_p"] = min_p
        if repetition_penalty is not None:
            transcribe_kwargs["repetition_penalty"] = repetition_penalty
        if repetition_context_size is not None:
            transcribe_kwargs["repetition_context_size"] = repetition_context_size
        # Extended sampling (Qwen3-ASR only — engine validates and rejects
        # otherwise). Upstream Qwen3ASRModel.generate hardcodes
        # make_logits_processors(repetition_penalty=..., repetition_context_size=...)
        # and ignores frequency / presence; the engine takes a parallel
        # _generate_single_chunk loop with the full mlx-lm stack when either
        # of these is set.
        if frequency_penalty is not None:
            transcribe_kwargs["frequency_penalty"] = frequency_penalty
        if frequency_context_size is not None:
            transcribe_kwargs["frequency_context_size"] = frequency_context_size
        if presence_penalty is not None:
            transcribe_kwargs["presence_penalty"] = presence_penalty
        if presence_context_size is not None:
            transcribe_kwargs["presence_context_size"] = presence_context_size
        if text is not None:
            transcribe_kwargs["text"] = text

        # Detect aligner model so we can either run alignment (text given) or
        # reject cleanly (text missing) instead of crashing on a missing
        # positional argument deep inside mlx-audio.
        is_aligner = False
        try:
            inner = getattr(getattr(engine, "_model", None), "_model", None)
            is_aligner = type(inner).__name__ == "ForcedAlignerModel"
        except Exception:
            pass
        if is_aligner and text is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{resolved_model}' is a forced aligner and requires "
                    "a 'text' form field with the reference transcript. To get "
                    "word-level timestamps automatically from audio alone, call "
                    "an ASR model with word_timestamps=true; the server will "
                    "auto-chain its configured aligner."
                ),
            )

        async def _maybe_align_inplace(res: dict, audio_path: str) -> None:
            """Run the ASR's configured forced-aligner on `audio_path` to fill
            res["segments"][0]["words"] if word_timestamps was requested and
            the ASR engine itself didn't emit per-word timestamps. No-op if
            current model is the aligner itself, if no aligner is configured
            via ModelSettings, or if the result already has words. Used by
            both the 1-pass and tripass code paths."""
            if not word_timestamps or is_aligner:
                return
            if not res.get("text"):
                return
            _segs = res.get("segments") or []
            if _segs and _segs[0].get("words"):
                return
            aligner_name = None
            sm = _get_settings_manager()
            if sm is not None:
                try:
                    ms = sm.get_settings(resolved_model)
                    if ms is not None:
                        aligner_name = getattr(ms, "aligner_model", None)
                except Exception:
                    pass
            if not aligner_name:
                return
            try:
                aligner_resolved = _resolve_model(aligner_name)
                aligner_engine = await pool.get_engine(aligner_resolved)
                if not isinstance(aligner_engine, STTEngine):
                    return
                align_kwargs: dict = {"language": language}
                if effective_max_tokens is not None:
                    align_kwargs["max_tokens"] = effective_max_tokens
                align_kwargs["text"] = res["text"]
                align_result = await aligner_engine.transcribe(
                    audio_path, **align_kwargs
                )
                words = align_result.get("words") or []
                if not words:
                    return
                segments = res.get("segments") or []
                if not segments:
                    segments = [{
                        "text": res["text"],
                        "language": res.get("language"),
                        "start": float(min(
                            (w["start"] for w in words), default=0.0
                        )),
                        "end": float(max(
                            (w["end"] for w in words),
                            default=res.get("duration", 0.0),
                        )),
                    }]
                segments[0] = {**segments[0], "words": words}
                res["segments"] = segments
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning(
                    "Aligner companion %s failed for %s: %s — returning "
                    "transcript without word timestamps",
                    aligner_name, resolved_model, exc,
                )

        if energy_tripass_requested:
            # 3-pass: L only, R only, mix (mix = tmp_path). Serial — the
            # underlying MLX engine is single-threaded so asyncio.gather
            # wouldn't actually overlap the compute, only the python
            # bookkeeping. 3x ASR latency is the documented cost. Each pass
            # gets its own aligner companion run so per-pass words[] is
            # populated for the merge.
            result_L = await engine.transcribe(tripass_L_path, **transcribe_kwargs)
            await _maybe_align_inplace(result_L, tripass_L_path)
            result_R = await engine.transcribe(tripass_R_path, **transcribe_kwargs)
            await _maybe_align_inplace(result_R, tripass_R_path)
            result_mix = await engine.transcribe(tmp_path, **transcribe_kwargs)
            await _maybe_align_inplace(result_mix, tmp_path)
            from omlx.engine.diarize import tripass_merge
            result = tripass_merge(
                stereo_audio_for_diarize, stereo_sr_for_diarize,
                result_L, result_R, result_mix,
                left_speaker=left_speaker, right_speaker=right_speaker,
            )
        else:
            result = await engine.transcribe(tmp_path, **transcribe_kwargs)
            await _maybe_align_inplace(result, tmp_path)

        # pyannote diarization (Phase 2). Mono / multi-speaker / conference
        # audio. Must run BEFORE the finally block unlinks tmp_path —
        # pyannote_diarize.diarize_words re-opens the file via soundfile
        # to do its own 8 → 16 kHz resample + mono down-mix. Heavy load
        # (~400 MB pyannote model loaded lazily on first call, gated HF
        # license, requires HF_TOKEN in server env).
        if pyannote_diarize_requested:
            from omlx.engine.pyannote_diarize import diarize_words
            try:
                segments = result.get("segments") or []
                # Flatten all words from all segments so pyannote sees the
                # full timeline once (cheaper than per-segment, and pyannote
                # needs full audio context anyway).
                all_words: list[dict] = []
                for seg in segments:
                    ws = seg.get("words") or []
                    all_words.extend(ws)
                if all_words:
                    # diarize_words mutates each word in place by adding
                    # "speaker". If energy mode replaced tmp_path with a
                    # mono mix, pyannote will run on that mix (which is
                    # fine — single-speaker-per-channel down-mix loses
                    # nothing for diarization); otherwise tmp_path is the
                    # upload as-saved.
                    diarize_words(
                        tmp_path,
                        all_words,
                        speakers=parsed_speakers,
                        num_speakers=num_speakers,
                        min_speakers=(
                            min_speakers if min_speakers != 2 else None
                        ),
                        max_speakers=(
                            max_speakers if max_speakers != 8 else None
                        ),
                    )
                    # Re-attach majority-vote segment-level speaker
                    # (parallel to what energy mode does — so json /
                    # verbose_json consumers see a useful segment.speaker
                    # without drilling into words).
                    for seg in segments:
                        ws = seg.get("words") or []
                        counts: dict[str, int] = {}
                        for w in ws:
                            sp = w.get("speaker")
                            if sp:
                                counts[sp] = counts.get(sp, 0) + 1
                        if counts:
                            seg["speaker"] = max(counts, key=counts.get)
                    result["segments"] = segments
            except (ImportError, RuntimeError) as exc:
                # Surface install / license errors as 503 with the
                # actionable message — the wrapper raises with full
                # HF-license / pip-install instructions, so we just
                # forward.
                raise HTTPException(
                    status_code=503,
                    detail=f"pyannote diarization unavailable: {exc}",
                ) from exc
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception(
                    "pyannote diarization failed for %s", resolved_model,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"pyannote diarization failed: {exc}",
                ) from exc

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
        if (
            mono_tmp_path
            and mono_tmp_path != tmp_path
            and os.path.exists(mono_tmp_path)
        ):
            try:
                os.unlink(mono_tmp_path)
            except OSError:
                pass
        for _extra in (tripass_L_path, tripass_R_path):
            if _extra and os.path.exists(_extra):
                try:
                    os.unlink(_extra)
                except OSError:
                    pass

    # Energy-based diarization: annotate each word with a speaker label
    # based on which stereo channel dominates its time window. We do this
    # AFTER ASR so we get the high-accuracy transcript from the mono mix
    # (full conversational context for the decoder) PLUS speaker labels
    # from the original stereo — without paying for two separate ASR runs.
    if (
        energy_diarize_requested
        and not energy_tripass_requested
        and stereo_audio_for_diarize is not None
    ):
        from omlx.engine.diarize import energy_diarize_words
        segments = result.get("segments") or []
        for seg in segments:
            words = seg.get("words") or []
            if not words:
                continue
            labelled = energy_diarize_words(
                stereo_audio_for_diarize,
                stereo_sr_for_diarize,
                words,
                left_speaker=left_speaker,
                right_speaker=right_speaker,
                diarize_threshold=diarize_threshold,
            )
            seg["words"] = labelled
            # Tag the segment itself with the dominant speaker (majority
            # vote) so callers that don't drill into words still see
            # something useful at the segment level.
            counts: dict[str, int] = {}
            for w in labelled:
                sp = w.get("speaker")
                if sp:
                    counts[sp] = counts.get(sp, 0) + 1
            if counts:
                seg["speaker"] = max(counts, key=counts.get)
        result["segments"] = segments

    _record_audio_request(resolved_model)

    # response_format=text: OpenAI spec returns just the transcript body as
    # text/plain with no envelope, segments, or metadata.
    if response_format == "text":
        return PlainTextResponse(
            content=result.get("text", ""),
            media_type="text/plain; charset=utf-8",
        )

    # SRT / VTT — assemble subtitle cues from segments/words.
    if response_format in ("srt", "vtt"):
        body = _build_subtitle(result, fmt=response_format)
        return PlainTextResponse(
            content=body,
            media_type=("text/srt" if response_format == "srt"
                        else "text/vtt") + "; charset=utf-8",
        )

    # json (default) and verbose_json both return the same shape today —
    # OpenAI's verbose_json is a superset of json with segments/duration,
    # and we always include those when the backend exposes them. Clients
    # asking for plain json simply ignore the extra fields.
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

    if not request.input or not request.input.strip():
        raise HTTPException(status_code=400, detail="'input' field must not be empty")
    streaming_interval = DEFAULT_NATIVE_TTS_STREAMING_INTERVAL_SECONDS
    if request.stream:
        if request.response_format not in (None, "wav"):
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
