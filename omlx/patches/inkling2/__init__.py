# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (``inkling2``) model registration for mlx-lm."""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_QUALNAME = "mlx_lm.models.inkling2"


def apply_inkling2_patch() -> bool:
    """Make ``mlx_lm.models.inkling2`` resolve to oMLX's model."""
    try:
        from . import model
    except Exception as exc:  # noqa: BLE001 - a missing dependency must not abort
        logger.warning("Could not register %s: %s", _QUALNAME, exc)
        return False
    sys.modules[_QUALNAME] = model
    importlib.import_module("mlx_lm.models").inkling2 = model
    return True


__all__ = ["apply_inkling2_patch"]
