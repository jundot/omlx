# SPDX-License-Identifier: Apache-2.0
"""Attach ``rollback_state`` slot to ``mlx_lm.models.cache.ArraysCache``.

PR 990 adds a class-level ``rollback_state: Optional[tuple] = None`` slot so
GatedDeltaNet can snapshot ``(conv_state, ssm_state)`` after the confirmed
prefix of an MTP draft+verify forward, then restore that snapshot when the
draft token is rejected.

The patch is a 4-line class attribute add. Both PR 990 (Qwen3.5/3.6) and
PR 15 (DeepSeek-V4) rely on it; the slot is inert when no MTP path runs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply() -> bool:
    """Attach ``rollback_state = None`` to ``ArraysCache`` (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        logger.debug("mlx_lm.models.cache not importable; skipping rollback_state")
        return False

    if hasattr(ArraysCache, "rollback_state") and not hasattr(
        ArraysCache, "_omlx_rollback_attached"
    ):
        # Upstream may have added it natively (e.g. once PR 990 lands).
        _PATCHED = True
        ArraysCache._omlx_rollback_attached = "upstream"
        return True

    ArraysCache.rollback_state = None
    ArraysCache._omlx_rollback_attached = "patch"
    _PATCHED = True
    return True
