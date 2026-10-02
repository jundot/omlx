# SPDX-License-Identifier: Apache-2.0
"""Laguna runtime extensions not yet covered by the pinned mlx-lm.

Retains compressed checkpoint loading, attention variants and fused kernels.
Retains JSON tool calls and schema-aware string arguments.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

PR_HEAD_SHA = "0857ee1cf1f4ba7c43e73b836309d9c884529ca8"
PR_URL = "https://github.com/ml-explore/mlx-lm/pull/1223"

_APPLIED = False


def apply_laguna_patch() -> bool:
    """Keep Laguna checkpoint formats and kernels not yet provided upstream."""
    global _APPLIED
    if _APPLIED:
        return False

    # Upstream lacks compressed FP8/INT4/NVFP4 loading and oMLX's fused paths.
    from . import laguna_model, tool_parser
    import mlx_lm.models
    import mlx_lm.tool_parsers

    sys.modules["mlx_lm.models.laguna"] = laguna_model
    mlx_lm.models.laguna = laguna_model
    sys.modules["mlx_lm.tool_parsers.laguna"] = tool_parser
    mlx_lm.tool_parsers.laguna = tool_parser
    _APPLIED = True
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_laguna_patch", "is_applied", "PR_HEAD_SHA", "PR_URL"]
