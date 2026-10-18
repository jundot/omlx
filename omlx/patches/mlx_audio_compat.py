# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for mlx-audio dependency drift."""

from __future__ import annotations

import logging
from functools import wraps

logger = logging.getLogger(__name__)

_RESAMPLE_EXPORT_CHECKED = False


def ensure_qwen3_asr_audio_quantization() -> bool:
    """Let the loader check Qwen3-ASR audio scales and per-layer settings."""
    try:
        from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel
    except ImportError:
        return False

    original = Qwen3ASRModel.model_quant_predicate
    if getattr(original, "_omlx_audio_quantization", False):
        return True

    @wraps(original)
    def model_quant_predicate(self, path, module):
        # The shared loader still checks scales or an explicit layer override.
        if path.startswith("audio_tower."):
            return True
        return original(self, path, module)

    model_quant_predicate._omlx_audio_quantization = True
    Qwen3ASRModel.model_quant_predicate = model_quant_predicate
    return True


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
