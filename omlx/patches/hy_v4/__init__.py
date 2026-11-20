# SPDX-License-Identifier: Apache-2.0
"""Tencent Hy4 and Hy4-flash support for oMLX's mlx-vlm runtime."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_MODEL_ALIASES = ("mlx_vlm.models.hy_v4_vl", "mlx_vlm.models.hy_v4_flash")
_PARSER_ALIASES = ("mlx_vlm.tools.parsers.hy_v4", "mlx_lm.tool_parsers.hy_v4")


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


def apply_patch() -> bool:
    """Register the Hy4 model and tool parser modules. Idempotent."""
    from ..hy_v3 import apply_hy_v3_patch
    from . import model, tool_parser

    _register(_MODEL_ALIASES, model)
    _register(_PARSER_ALIASES, tool_parser)
    # Hy4 shares Hy3's <think:opensource> pair; that patch teaches mlx-lm it.
    apply_hy_v3_patch()
    return True
