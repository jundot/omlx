# SPDX-License-Identifier: Apache-2.0
"""Sanitize compatibility for merged gemma4 MTP checkpoints on the mlx-lm path.

Merged gemma4 checkpoints carry the assistant MTP head under
``language_model.mtp.*`` (see ``mlx_vlm_mtp.gemma4_vlm_runtime``). The
Lightning MTP head only exists on the mlx-vlm classes, so any load that
goes through mlx-lm's text-only ``models/gemma4`` adapter — DFlash targets
and the VLM→LLM fallback — has no binding site for those tensors and
strict ``load_weights`` fails with "Received N parameters not in model".

Stock ``mlx_lm.models.gemma4.Model.sanitize`` already discards the other
multimodal-only trees (vision_tower, audio_tower, ...); this patch makes
it discard the MTP head the same way.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MTP_KEY_PREFIXES = ("mtp.", "model.mtp.", "language_model.mtp.")


def apply() -> bool:
    """Wrap ``mlx_lm.models.gemma4.Model.sanitize`` to strip mtp.* keys.

    Idempotent via a class marker. Returns False when mlx-lm's gemma4
    module is not importable.
    """
    try:
        from mlx_lm.models import gemma4 as lm_gemma4
    except ImportError:
        logger.debug("mlx_lm.models.gemma4 not importable; sanitize patch skipped")
        return False

    cls = lm_gemma4.Model
    if getattr(cls, "_omlx_gemma4_mtp_sanitize_patched", False):
        return True

    original_sanitize = cls.sanitize

    def sanitize(self, weights):
        weights = {
            k: v
            for k, v in weights.items()
            if not k.removeprefix("model.").startswith(_MTP_KEY_PREFIXES)
        }
        return original_sanitize(self, weights)

    cls.sanitize = sanitize
    cls._omlx_gemma4_mtp_sanitize_patched = True
    logger.debug("mlx-lm gemma4 sanitize patched to strip merged mtp.* keys")
    return True
