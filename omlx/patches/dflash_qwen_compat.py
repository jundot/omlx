# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for dflash-mlx Qwen target ops."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def install_dflash_qwen_compat_patch() -> bool:
    """Keep dflash-mlx Qwen ops compatible with mlx-lm's outer Model.model.

    mlx-lm bdb77da added ``Model.model`` to the Qwen3.5/3.6 outer wrapper for
    pipeline support. dflash-mlx 0.1.9 detects a text wrapper by checking
    ``hasattr(target_model, "model")`` first, so it starts treating the outer
    model as the text wrapper. For untied Qwen checkpoints this makes
    ``logits_from_hidden`` project through ``embed_tokens.as_linear`` instead
    of ``TextModel.lm_head``, producing invalid verifier logits without raising.
    """
    try:
        from dflash_mlx.engine import target_qwen_gdn
    except ImportError:
        logger.debug("dflash_mlx.engine.target_qwen_gdn not importable")
        return False

    cls = getattr(target_qwen_gdn, "QwenGdnTargetOps", None)
    if cls is None:
        return False
    if getattr(cls, "_omlx_qwen_text_wrapper_compat", False):
        return True

    original = cls.text_wrapper

    def text_wrapper(self: Any, target_model: Any) -> Any:
        language_model = getattr(target_model, "language_model", None)
        if language_model is not None and hasattr(language_model, "model"):
            return language_model
        return original(self, target_model)

    cls.text_wrapper = text_wrapper
    cls._omlx_qwen_text_wrapper_compat = True
    logger.debug("dflash Qwen text wrapper compatibility patch installed")
    return True
