# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for mlx-audio dependency drift."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_RESAMPLE_EXPORT_CHECKED = False


def ensure_mlx_audio_resample_export() -> bool:
    """Ensure mlx-vlm's legacy resampler import path is available.

    mlx-vlm@526c210 imports ``resample_audio`` from ``mlx_audio.utils`` inside
    ``load_audio()``.  mlx-audio@5175326 moved that function to
    ``mlx_audio.stt.utils`` without re-exporting it from the old location.
    """
    global _RESAMPLE_EXPORT_CHECKED

    try:
        import mlx_audio.utils as audio_utils
    except ImportError:
        return False

    if hasattr(audio_utils, "resample_audio"):
        _RESAMPLE_EXPORT_CHECKED = True
        return True

    try:
        from mlx_audio.stt.utils import resample_audio
    except ImportError:
        return False

    audio_utils.resample_audio = resample_audio
    if not _RESAMPLE_EXPORT_CHECKED:
        logger.debug("mlx_audio.utils.resample_audio compatibility export applied")
    _RESAMPLE_EXPORT_CHECKED = True
    return True
