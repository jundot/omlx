# SPDX-License-Identifier: Apache-2.0
"""Register oMLX's text-only Qwen4-Exp model for cluster ranks.

The single-node path serves Qwen4-Exp through mlx-vlm, complete with vision and
its embedded Lightning MTP head. A cluster rank loads through mlx-lm and holds
one pipeline stage, so this adapter registers a text-only root at
``mlx_lm.models.qwen4_exp`` before the worker's capability probe runs.
"""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_QUALNAME = "mlx_lm.models.qwen4_exp"


def apply_qwen4_exp_mlx_lm_patch() -> bool:
    """Make ``qwen4_exp`` resolve to the text-only rank model. Idempotent."""

    try:
        from . import model
    except Exception as exc:  # noqa: BLE001 - a missing dependency must not abort
        logger.warning("Could not register %s: %s", _QUALNAME, exc)
        return False
    sys.modules[_QUALNAME] = model
    importlib.import_module("mlx_lm.models").qwen4_exp = model
    return True


__all__ = ["apply_qwen4_exp_mlx_lm_patch"]
