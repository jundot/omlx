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

def apply() -> bool:
    """Attach ``rollback_state = None`` to ``ArraysCache`` (idempotent).

    Idempotency is checked against the live class attribute, not a
    module-level flag — keeps the patch consistent with the rest of
    mlx_lm_mtp after the #1388 self-healing refactor.
    """
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        logger.debug("mlx_lm.models.cache not importable; skipping rollback_state")
        return False

    if hasattr(ArraysCache, "_omlx_rollback_attached"):
        return True

    if hasattr(ArraysCache, "rollback_state"):
        # Upstream may have added it natively (e.g. once PR 990 lands).
        ArraysCache._omlx_rollback_attached = "upstream"
        return True

    ArraysCache.rollback_state = None
    ArraysCache._omlx_rollback_attached = "patch"
    return True
