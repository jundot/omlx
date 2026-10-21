# SPDX-License-Identifier: Apache-2.0
"""Register the Muse Spark text model until mlx-lm provides it."""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_TARGET = "mlx_lm.models.muse_spark"

_APPLIED = False


def apply_muse_spark_patch() -> bool:
    """Expose ``muse_spark_model`` as ``mlx_lm.models.muse_spark``."""
    global _APPLIED
    if _APPLIED:
        return False

    from . import muse_spark_model

    # oMLX keeps its own module even if mlx-lm gains one, so the LLM and VLM
    # paths run the same text model.
    sys.modules[_TARGET] = muse_spark_model
    package, name = _TARGET.rsplit(".", 1)
    setattr(importlib.import_module(package), name, muse_spark_model)
    _APPLIED = True
    logger.info("Registered %s", _TARGET)
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_muse_spark_patch", "is_applied"]
