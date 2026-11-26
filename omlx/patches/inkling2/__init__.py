# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (``inkling2``) model registration for mlx-lm."""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_QUALNAME = "mlx_lm.models.inkling2"
# Looped checkpoints have their own model_type and the same model.
INKLING2_MODEL_TYPES = frozenset({"inkling2", "inkling2_loop"})


def apply_inkling2_patch() -> bool:
    """Make ``mlx_lm.models.inkling2`` resolve to oMLX's model."""
    try:
        from . import model
    except Exception as exc:  # noqa: BLE001 - a missing dependency must not abort
        logger.warning("Could not register %s: %s", _QUALNAME, exc)
        return False
    sys.modules[_QUALNAME] = model
    importlib.import_module("mlx_lm.models").inkling2 = model
    remapping = importlib.import_module("mlx_lm.utils").MODEL_REMAPPING
    remapping["inkling2_loop"] = "inkling2"
    return True


__all__ = ["INKLING2_MODEL_TYPES", "apply_inkling2_patch"]
