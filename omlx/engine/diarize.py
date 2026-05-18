# SPDX-License-Identifier: Apache-2.0
"""Speaker diarization backends for the STT pipeline.

Two backends are planned:

* ``energy`` — stereo channels carry one speaker each (FreeSWITCH 2-leg
  call recording, podcasts with separate mics). Compares L/R RMS energy
  per word to assign label. Zero model, zero deps beyond numpy. Implemented
  here.

* ``pyannote`` — mono or multi-speaker input where channel-based labelling
  doesn't apply (conference recordings, single-mic interviews). Wraps
  pyannote.audio's speaker-diarization-3.1 pipeline. Phase 2, separate
  session — model download + HF license + 16kHz resample + glue.

Both produce the same output shape: a list of word dicts with a
``speaker`` field populated on each entry. The audio_routes layer then
threads this label into the SRT/VTT formatter and the verbose_json
response.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def energy_diarize_words(
    stereo_audio: np.ndarray,
    sample_rate: int,
    words: list[dict],
    left_speaker: str,
    right_speaker: str,
    diarize_threshold: float = 1.3,
    pad_seconds: float = 0.05,
    overlap_label: str = "overlap",
) -> list[dict]:
    """Label each word with a speaker based on stereo L / R RMS ratio.

    Each input word has ``word`` / ``start`` / ``end`` (seconds). For
    every word we slice the stereo buffer over [start-pad, end+pad],
    compute RMS on each channel, and pick the louder side — unless the
    ratio is below ``diarize_threshold`` (interpreted as cross-talk or
    silent gap), in which case we emit ``overlap_label``.

    Args:
        stereo_audio: shape ``(samples, 2)``, L = column 0, R = column 1.
        sample_rate: original sample rate of the buffer (Hz).
        words: ASR word entries.
        left_speaker / right_speaker: label string for L / R speakers.
        diarize_threshold: ``max(L, R) / min(L, R)`` below this is
            considered indeterminate. Default 1.3.
        pad_seconds: widen the per-word window by this on each side
            before measuring energy. Default 0.05.
        overlap_label: label to assign when the ratio is too close.

    Returns:
        A new list of word dicts with a ``speaker`` field added. The input
        list is not mutated.
    """
    if stereo_audio.ndim != 2 or stereo_audio.shape[1] != 2:
        raise ValueError(
            "energy_diarize requires stereo (samples, 2); got shape "
            f"{stereo_audio.shape}"
        )
    L = stereo_audio[:, 0]
    R = stereo_audio[:, 1]
    n_samples = L.shape[0]
    pad_n = max(1, int(pad_seconds * sample_rate))

    out: list[dict] = []
    for w in words:
        try:
            start_s = float(w.get("start", 0.0))
            end_s = float(w.get("end", start_s))
        except (TypeError, ValueError):
            out.append({**w, "speaker": overlap_label})
            continue

        s = max(0, int(start_s * sample_rate) - pad_n)
        e = min(n_samples, int(end_s * sample_rate) + pad_n)
        if e <= s:
            out.append({**w, "speaker": overlap_label})
            continue

        L_rms = float(np.sqrt(np.mean(L[s:e] ** 2)))
        R_rms = float(np.sqrt(np.mean(R[s:e] ** 2)))
        hi = max(L_rms, R_rms)
        lo = max(min(L_rms, R_rms), 1e-9)
        ratio = hi / lo

        if ratio < diarize_threshold:
            label = overlap_label
        elif L_rms >= R_rms:
            label = left_speaker
        else:
            label = right_speaker
        out.append({**w, "speaker": label})

    return out


def mono_mix_for_asr(stereo_audio: np.ndarray) -> np.ndarray:
    """Down-mix stereo to mono with clip protection.

    Simple ``(L + R) / 2`` averaging. Avoids the +6 dB clipping that
    naive ``L + R`` would cause if both channels are loud at the same
    moment. The diarization layer keeps the original stereo buffer for
    RMS analysis; only the ASR-bound copy is mixed down.
    """
    if stereo_audio.ndim == 1:
        return stereo_audio
    if stereo_audio.ndim != 2 or stereo_audio.shape[1] not in (1, 2):
        raise ValueError(
            f"mono_mix expects (samples,) or (samples, 1|2); got shape "
            f"{stereo_audio.shape}"
        )
    if stereo_audio.shape[1] == 1:
        return stereo_audio[:, 0]
    return (stereo_audio[:, 0] + stereo_audio[:, 1]) * 0.5


def _flatten_words(result: dict) -> list[dict]:
    words: list[dict] = []
    for seg in result.get("segments") or []:
        for w in seg.get("words") or []:
            words.append({
                "word": w.get("word", ""),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", w.get("start", 0.0))),
                "_used": False,
            })
    words.sort(key=lambda w: (w["start"], w["end"]))
    return words


def tripass_merge(
    stereo_audio: np.ndarray,
    sample_rate: int,
    pass_L_result: dict,
    pass_R_result: dict,
    pass_mix_result: dict,
    left_speaker: str,
    right_speaker: str,
    time_tolerance: float = 0.3,
    pad_seconds: float = 0.05,
    rms_recover_threshold: float = 0.55,
) -> dict:
    """Merge per-channel ASR (L, R) + mix-pass ASR into a single transcript.

    Strategy: the mix pass produces the canonical text (best decoder
    context — boundary words like 你/您, 对/兑 disambiguate when the
    decoder sees both sides). The L and R per-channel passes act as
    speaker oracles: a word that appears in pass_L within ``time_tolerance``
    of the mix word IS spoken on L, regardless of how loud R is at that
    moment. This sidesteps the cross-talk RMS confusion the 1-pass energy
    backend hits, and recovers words the mix lost to simultaneous speech.

    Args:
        stereo_audio: shape ``(samples, 2)``, L = column 0, R = column 1.
        sample_rate: original sample rate (Hz).
        pass_L_result / pass_R_result / pass_mix_result: full ASR result
            dicts as returned by ``engine.transcribe()`` on each input.
        left_speaker / right_speaker: label strings.
        time_tolerance: max start-to-start difference (seconds) to call
            a mix word "the same word" as a per-channel word. 0.3 s is
            wide enough to absorb the small timing drift that comes from
            decoding the same utterance on different signals.
        pad_seconds: per-word RMS window pad for the energy fallback.
        rms_recover_threshold: when recovering a per-channel word the
            mix pass dropped, require the originating channel to carry
            at least this fraction of the total per-window RMS energy.
            Filters out the cross-talk-leak ghost transcriptions.

    Returns:
        A merged result dict: ``text`` (time-ordered concatenation),
        ``segments`` (one entry per speaker turn, with ``speaker`` field
        and ``words[].speaker``), ``language`` / ``duration`` copied
        from pass_mix_result.
    """
    if stereo_audio.ndim != 2 or stereo_audio.shape[1] != 2:
        raise ValueError(
            "tripass_merge requires stereo (samples, 2); got shape "
            f"{stereo_audio.shape}"
        )
    L = stereo_audio[:, 0]
    R = stereo_audio[:, 1]
    n_samples = L.shape[0]
    pad_n = max(1, int(pad_seconds * sample_rate))

    def _rms_at(t_start: float, t_end: float) -> tuple[float, float]:
        s = max(0, int(t_start * sample_rate) - pad_n)
        e = min(n_samples, int(t_end * sample_rate) + pad_n)
        if e <= s:
            return 1e-9, 1e-9
        l = float(np.sqrt(np.mean(L[s:e] ** 2)))
        r = float(np.sqrt(np.mean(R[s:e] ** 2)))
        return l, r

    words_L = _flatten_words(pass_L_result)
    words_R = _flatten_words(pass_R_result)
    words_M = _flatten_words(pass_mix_result)

    def _consume_match(
        pool: list[dict], t_start: float, t_end: float, text: str
    ) -> bool:
        """Find an unused pool word with matching text near this time window.
        Marks the matched word as consumed and returns True. Pool must be
        sorted by start ascending."""
        for pw in pool:
            if pw["_used"]:
                continue
            if pw["end"] < t_start - time_tolerance:
                continue
            if pw["start"] > t_end + time_tolerance:
                break
            if pw["word"] == text:
                pw["_used"] = True
                return True
        return False

    merged: list[dict] = []
    for wm in words_M:
        text = wm["word"]
        ts, te = wm["start"], wm["end"]
        in_L = _consume_match(words_L, ts, te, text)
        in_R = _consume_match(words_R, ts, te, text)
        if in_L and not in_R:
            spk = left_speaker
        elif in_R and not in_L:
            spk = right_speaker
        else:
            l_rms, r_rms = _rms_at(ts, te)
            spk = left_speaker if l_rms >= r_rms else right_speaker
        merged.append(
            {"word": text, "start": ts, "end": te, "speaker": spk}
        )

    # Recovery: per-channel words the mix pass dropped (simultaneous-speech
    # casualty). Require originating-channel RMS to dominate, to filter
    # out cross-talk-leak ghost transcriptions.
    for wL in words_L:
        if wL["_used"]:
            continue
        ts, te = wL["start"], wL["end"]
        l_rms, r_rms = _rms_at(ts, te)
        if l_rms / (l_rms + r_rms + 1e-9) >= rms_recover_threshold:
            merged.append(
                {"word": wL["word"], "start": ts, "end": te,
                 "speaker": left_speaker}
            )
    for wR in words_R:
        if wR["_used"]:
            continue
        ts, te = wR["start"], wR["end"]
        l_rms, r_rms = _rms_at(ts, te)
        if r_rms / (l_rms + r_rms + 1e-9) >= rms_recover_threshold:
            merged.append(
                {"word": wR["word"], "start": ts, "end": te,
                 "speaker": right_speaker}
            )

    merged.sort(key=lambda w: (w["start"], w["end"]))

    # Re-segment by speaker turn so downstream (SRT / VTT / segment.text)
    # sees one segment per uninterrupted speaker stretch.
    segments: list[dict] = []
    cur: dict | None = None
    for w in merged:
        if cur is None or w["speaker"] != cur["speaker"]:
            if cur is not None:
                cur["end"] = cur["words"][-1]["end"]
                cur["text"] = "".join(x["word"] for x in cur["words"])
                segments.append(cur)
            cur = {
                "start": w["start"],
                "end": w["end"],
                "speaker": w["speaker"],
                "language": pass_mix_result.get("language"),
                "words": [w],
            }
        else:
            cur["words"].append(w)
    if cur is not None:
        cur["end"] = cur["words"][-1]["end"]
        cur["text"] = "".join(x["word"] for x in cur["words"])
        segments.append(cur)

    full_text = "".join(seg["text"] for seg in segments)

    out = {
        "text": full_text,
        "language": pass_mix_result.get("language"),
        "duration": pass_mix_result.get("duration"),
        "segments": segments,
    }
    return out
