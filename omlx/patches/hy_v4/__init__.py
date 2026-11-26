# SPDX-License-Identifier: Apache-2.0
"""Tencent Hy4 and Hy4-flash support for oMLX's mlx-vlm runtime."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_MODEL_TYPES = ("hy_v4_vl", "hy_v4_flash")
_MODEL_ALIASES = tuple(f"mlx_vlm.models.{t}" for t in _MODEL_TYPES)
_PARSER_ALIASES = ("mlx_vlm.tools.parsers.hy_v4", "mlx_lm.tool_parsers.hy_v4")

_PROCESSOR_INSTALLED = False


def _register(aliases, module) -> None:
    for alias in aliases:
        existing = sys.modules.get(alias)
        if existing is not None and existing is not module:
            logger.warning(
                "%s (%s) replaced by the oMLX Hy4 module",
                alias,
                getattr(existing, "__file__", "?"),
            )
        sys.modules[alias] = module


def _register_processor() -> None:
    """Route AutoProcessor and the chat message format to the Hy4 processor."""
    global _PROCESSOR_INSTALLED
    if _PROCESSOR_INSTALLED:
        return
    from mlx_vlm.models.base import install_auto_processor_patch
    from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat

    from .processor import HyV4Processor

    install_auto_processor_patch(list(_MODEL_TYPES), HyV4Processor)
    # Without an entry apply_chat_template drops the image parts of messages.
    for model_type in _MODEL_TYPES:
        MODEL_CONFIG.setdefault(model_type, MessageFormat.LIST_WITH_IMAGE_FIRST)
    _PROCESSOR_INSTALLED = True


def apply_patch() -> bool:
    """Register the Hy4 model, processor and tool parser modules. Idempotent."""
    from ..hy_v3 import apply_hy_v3_patch
    from . import model, tool_parser

    _register(_MODEL_ALIASES, model)
    _register(_PARSER_ALIASES, tool_parser)
    _register_processor()
    # Hy4 shares Hy3's <think:opensource> pair; that patch teaches mlx-lm it.
    apply_hy_v3_patch()
    return True
