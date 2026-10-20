# SPDX-License-Identifier: Apache-2.0
"""Let cluster ranks build oMLX's DeepSeek V4.1 instead of mlx-lm's.

mlx-lm ships its own ``deepseek_v41``, which refuses Engram layers and cannot
read oMLX-converted checkpoints. This replaces it with an adapter over oMLX's
implementation, loaded one pipeline stage or tensor-parallel slice at a time.
"""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_QUALNAME = "mlx_lm.models.deepseek_v41"


def apply_deepseek_v41_mlx_lm_patch() -> bool:
    """Make ``deepseek_v41`` resolve to oMLX's model. Safe to call repeatedly."""

    try:
        from . import model
    except Exception as exc:  # noqa: BLE001 - a missing dependency must not abort
        logger.warning("Could not register %s: %s", _QUALNAME, exc)
        return False
    # Replaces mlx-lm's module even when a capability probe imported it first.
    sys.modules[_QUALNAME] = model
    importlib.import_module("mlx_lm.models").deepseek_v41 = model
    return True


__all__ = ["apply_deepseek_v41_mlx_lm_patch"]
