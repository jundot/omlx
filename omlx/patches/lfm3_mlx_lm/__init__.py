# SPDX-License-Identifier: Apache-2.0
"""Let mlx-lm load LiquidAI LFM3 (``lfm3``).

Pinned mlx-lm ships ``lfm2``, ``lfm2_moe`` and ``lfm2-vl`` but no ``lfm3``.
A checkpoint whose ``config.json`` says ``"model_type": "lfm3"`` therefore
fails in ``mlx_lm.utils._get_classes`` with ``Model type lfm3 not supported``
before any weight is read. This registers the vendored architecture as
``mlx_lm.models.lfm3`` so the existing loader, quantizer and cache paths apply
unchanged.

The vendored module is Apple's implementation from mlx-lm PR #2016 (MIT, see
``LICENSE.mlx-lm``). Text checkpoints only: LFM3-VL carries its own vision tree
and is served by mlx-vlm.
"""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

PR_URL = "https://github.com/ml-explore/mlx-lm/pull/2016"
PR_HEAD_SHA = "9027ac5b062175fba890c612c3905d5d8d5b5294"

_QUALNAME = "mlx_lm.models.lfm3"
_APPLIED = False


def is_lfm3(config: dict) -> bool:
    """Whether ``config`` declares the LFM3 text architecture."""
    return isinstance(config, dict) and config.get("model_type") == "lfm3"


def apply_lfm3_mlx_lm_patch() -> bool:
    """Make ``lfm3`` loadable by mlx-lm. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        from . import model
    except Exception as exc:  # noqa: BLE001 - a broken vendor tree must not abort
        logger.warning("Could not register %s: %s", _QUALNAME, exc)
        return False

    sys.modules[_QUALNAME] = model
    # ``from mlx_lm.models import lfm3`` reads the package attribute.
    importlib.import_module("mlx_lm.models").lfm3 = model
    _APPLIED = True
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "apply_lfm3_mlx_lm_patch",
    "is_applied",
    "is_lfm3",
    "PR_HEAD_SHA",
    "PR_URL",
]
