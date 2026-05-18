# SPDX-License-Identifier: Apache-2.0
"""pyannote diarize Phase 2: mono / multi-speaker diarization wrapper.

Phase 1 (omlx/engine/diarize.py) handles stereo recordings where each
channel carries one speaker. Phase 2 covers everything else — mono
recordings, conference audio, podcasts with shared microphones —
by running pyannote/speaker-diarization-3.1 and grafting per-word
speaker labels onto the ASR output.

Activation
----------
- Explicit:  diarize_backend='pyannote'
- Auto:      diarize_backend='auto' + mono input
             (Phase 1 stereo+L/R still wins when both are given)

Word-to-speaker attribution
---------------------------
For each ASR word with [start, end]:
  1. Compute overlap duration against every diarization turn.
  2. Pick the turn with the longest overlap. That speaker wins.
  3. If the word lies entirely outside any turn, fall back to the
     nearest turn within ``gap_tolerance`` seconds; otherwise mark
     ``speaker=None``.

This is deterministic — straddle cases resolve by overlap duration,
not by first/last-encountered, so re-running on the same audio gives
the same labels.

Speaker labelling
-----------------
- ``speakers`` given (list[str]): first-encountered pyannote label
  (SPEAKER_00) maps to speakers[0], next new label to speakers[1],
  and so on, in the order pyannote emits them. Excess raw labels fall
  back to anonymous numbering.
- ``num_speakers`` given (int): pyannote constrains to exactly N
  speakers; oMLX renames SPEAKER_xx → speaker_0/1/2/... unless
  ``speakers`` is also given.
- Neither: pyannote auto-detects between ``min_speakers`` and
  ``max_speakers`` (defaults to its pipeline defaults if not set).

Dependencies (installed at the user's site)
-------------------------------------------
- pip install pyannote.audio torch torchaudio  (~400 MB on Apple silicon)
- HuggingFace gated license at
  https://hf.co/pyannote/speaker-diarization-3.1 must be accepted in the
  web UI (one-time, per HF account).
- HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) must be present in the server's
  environment, or passed via the ``hf_token`` argument.

The import of pyannote.audio is deferred to first-call inside
``_get_pipeline``; importing this module is free, so the rest of the
engine stays warm even on machines that have never installed pyannote.
The loaded pipeline is cached in a module-level singleton (guarded by
a Lock for concurrent first-call safety) — the first transcription pays
the ~3-5 s load cost; subsequent calls are free.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
_PIPELINE_SR = 16_000  # pyannote was trained at 16 kHz

_PIPELINE: Any = None
_PIPELINE_LOCK = threading.Lock()


def _resolve_hf_token(explicit: Optional[str]) -> Optional[str]:
    """Look up an HF token: explicit arg, env vars, then HF cached token."""
    if explicit:
        return explicit
    tok = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if tok:
        return tok
    # huggingface_hub Pipeline.from_pretrained auto-resolves token from
    # ~/.cache/huggingface/token; mirror that fallback so `hf auth login`
    # alone is enough (no env var needed in spawned server env).
    try:
        from huggingface_hub import get_token as _hf_get_token
        return _hf_get_token()
    except Exception:
        return None


def _get_pipeline(
    model_id: str = _DEFAULT_MODEL,
    hf_token: Optional[str] = None,
):
    """Return the lazily-loaded pyannote diarization pipeline.

    First call performs the import + download + load. The lock guards
    against two requests racing through ``Pipeline.from_pretrained`` at
    once on a cold start. Subsequent calls go straight through the
    fast-path check at the top.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is required for pyannote diarization backend. "
                "Install: pip install pyannote.audio torch torchaudio "
                "(~400 MB on Apple silicon). Then accept the gated license at "
                "https://hf.co/pyannote/speaker-diarization-3.1 and set "
                "HF_TOKEN in the server's environment."
            ) from exc

        token = _resolve_hf_token(hf_token)
        if not token:
            raise RuntimeError(
                "pyannote diarization requires an HF token. Accept the gated "
                "license at https://hf.co/pyannote/speaker-diarization-3.1, "
                "then set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in the server's "
                "environment, or pass hf_token=... to diarize_words."
            )

        logger.info("Loading pyannote pipeline %s ...", model_id)
        pipeline = Pipeline.from_pretrained(model_id, token=token)
        _PIPELINE = pipeline
        logger.info("pyannote pipeline ready")
        return _PIPELINE


def _ensure_mono(audio_np):
    """Down-mix stereo input to mono. pyannote needs single-channel audio."""
    if audio_np.ndim == 1:
        return audio_np
    if audio_np.ndim == 2:
        return audio_np.mean(axis=1).astype(audio_np.dtype, copy=False)
    raise ValueError(f"Unsupported audio shape: {audio_np.shape}")


def _resample_to_16k(audio_np, src_sr: int):
    """Resample audio to 16 kHz with scipy.signal.resample_poly.

    Handles arbitrary integer source rates via gcd. 8 kHz telephony
    inputs become 16 kHz at up/down = 2/1; 44.1 kHz consumer audio at
    160/441; 48 kHz at 1/3. ``resample_poly`` is exact for rational
    ratios and high-quality enough for diarization (per pyannote
    issue tracker — they recommend it explicitly).
    """
    if src_sr == _PIPELINE_SR:
        return audio_np, _PIPELINE_SR
    from scipy.signal import resample_poly

    g = math.gcd(int(src_sr), _PIPELINE_SR)
    up = _PIPELINE_SR // g
    down = int(src_sr) // g
    resampled = resample_poly(audio_np, up, down)
    return resampled.astype(audio_np.dtype, copy=False), _PIPELINE_SR


def _next_speaker_name(
    raw_label: str,
    label_map: dict[str, str],
    speakers: Optional[list[str]],
    next_idx: list[int],
) -> str:
    """Allocate a stable display name for a pyannote raw label.

    pyannote emits ``SPEAKER_00``, ``SPEAKER_01``, ... in the order it
    discovers speakers. First time we see one, allocate the next slot
    from ``speakers`` if given, otherwise an anonymous ``speaker_N``.
    """
    if raw_label in label_map:
        return label_map[raw_label]
    idx = next_idx[0]
    if speakers and idx < len(speakers):
        name = speakers[idx]
    else:
        name = f"speaker_{idx}"
    label_map[raw_label] = name
    next_idx[0] = idx + 1
    return name


def diarize_words(
    audio_path: str,
    words: list[dict],
    *,
    speakers: Optional[list[str]] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
    gap_tolerance: float = 0.3,
    model_id: str = _DEFAULT_MODEL,
) -> list[dict]:
    """Annotate ``words`` in place with a per-word ``speaker`` field.

    Parameters
    ----------
    audio_path : str
        Path to the audio file. Read here via soundfile so the resample
        + mono-mix can be done independently of whatever path ASR took.
    words : list of dict
        Each entry must have ``start`` and ``end`` (seconds). Mutated in
        place with a new ``speaker`` key. Returned for chaining.
    speakers : list of str, optional
        Canonical display names. First pyannote label encountered maps
        to speakers[0], next new one to speakers[1], etc.
    num_speakers, min_speakers, max_speakers : int, optional
        Forwarded to pyannote pipeline. ``num_speakers`` constrains to
        exactly N; ``min_speakers`` / ``max_speakers`` are soft bounds.
    hf_token : str, optional
        Override the HF_TOKEN env var. Useful for test injection.
    gap_tolerance : float
        How far (seconds) a word can be from the nearest turn before
        we give up and emit ``speaker=None``. 0.3 s is roughly one
        comma's worth of silence.

    Returns
    -------
    list of dict
        The same ``words`` list, each item augmented with ``speaker``.
    """
    import soundfile as sf

    audio_np, sr = sf.read(audio_path, dtype="float32")
    audio_np = _ensure_mono(audio_np)
    audio_np, sr = _resample_to_16k(audio_np, sr)

    pipeline = _get_pipeline(model_id=model_id, hf_token=hf_token)

    import torch
    waveform = torch.from_numpy(audio_np[None, :]).float()

    pipe_kwargs: dict[str, int] = {}
    if num_speakers is not None:
        pipe_kwargs["num_speakers"] = int(num_speakers)
    if min_speakers is not None:
        pipe_kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        pipe_kwargs["max_speakers"] = int(max_speakers)

    diarization = pipeline(
        {"waveform": waveform, "sample_rate": sr}, **pipe_kwargs
    )

    # pyannote 4.x wraps the Annotation in DiarizeOutput.speaker_diarization;
    # 3.x returns the Annotation directly. Accept both.
    annotation = getattr(diarization, "speaker_diarization", diarization)

    # pyannote.core.Annotation.itertracks(yield_label=True) yields
    #   (Segment(start, end), track_label, speaker_label)
    turns: list[tuple[float, float, str]] = [
        (float(seg.start), float(seg.end), spk)
        for seg, _trk, spk in annotation.itertracks(yield_label=True)
    ]

    label_map: dict[str, str] = {}
    next_idx = [0]
    # Pre-allocate display names in the order pyannote first emits each
    # raw label. This guarantees stable speakers[0]→first-heard mapping
    # even if a later word ends up attributed to a label we haven't yet
    # bound (the attribution loop falls back through the same helper).
    for _, _, raw in turns:
        _next_speaker_name(raw, label_map, speakers, next_idx)

    for w in words:
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))
        if we < ws:
            we = ws

        best_overlap = 0.0
        best_label: Optional[str] = None
        nearest_gap_dist = float("inf")
        nearest_gap_label: Optional[str] = None

        for ts, te, lbl in turns:
            overlap = max(0.0, min(we, te) - max(ws, ts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = lbl
            elif overlap == 0.0:
                # word strictly outside this turn — measure distance for
                # the gap fallback. Positive distance only.
                gap = max(ts - we, ws - te, 0.0)
                if gap < nearest_gap_dist:
                    nearest_gap_dist = gap
                    nearest_gap_label = lbl

        if best_label is not None:
            w["speaker"] = _next_speaker_name(
                best_label, label_map, speakers, next_idx
            )
        elif nearest_gap_label is not None and nearest_gap_dist <= gap_tolerance:
            w["speaker"] = _next_speaker_name(
                nearest_gap_label, label_map, speakers, next_idx
            )
        else:
            w["speaker"] = None
    return words


def reset_pipeline_cache() -> None:
    """Drop the cached pipeline. Test/admin hook — next call reloads."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        _PIPELINE = None
