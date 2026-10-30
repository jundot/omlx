# SPDX-License-Identifier: Apache-2.0
"""oMLX safety policy for the pinned mlx-vlm image models."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_mlx_vlm_image_compat_patch() -> bool:
    """Apply oMLX's image model loading policy to mlx-vlm."""
    global _APPLIED
    if _APPLIED:
        return False

    _patch_z_image_tokenizer_security()

    _APPLIED = True
    logger.info("mlx-vlm image compatibility patch applied")
    return True


def is_applied() -> bool:
    """Return whether the compatibility layer has been registered."""
    return _APPLIED


def _patch_z_image_tokenizer_security() -> None:
    """Keep native Z-Image tokenizer loading local and remote-code-free."""
    pipeline = importlib.import_module("mlx_vlm.models.z_image.pipeline")
    tokenizer = pipeline.AutoTokenizer
    if getattr(tokenizer, "_omlx_safe_z_image_tokenizer", False):
        return

    class SafeAutoTokenizer:
        _omlx_safe_z_image_tokenizer = True

        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            kwargs["local_files_only"] = True
            kwargs["trust_remote_code"] = False
            return tokenizer.from_pretrained(*args, **kwargs)

    pipeline.AutoTokenizer = SafeAutoTokenizer


__all__ = ["apply_mlx_vlm_image_compat_patch", "is_applied"]
