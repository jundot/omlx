# SPDX-License-Identifier: Apache-2.0
"""Let mlx-lm load GLM-5.3 (``glm5_next``), so the cluster can serve it.

A cluster rank is an ``mlx_lm.server``. Pinned mlx-lm has no ``glm5_next``,
and oMLX serves the model on one Mac through its vendored mlx-vlm tree. This
registers ``mlx_lm.models.glm5_next`` as a text-only adapter over that same
tree. A rank holds one pipeline stage, or the tensor-parallel slice that
``omlx.cluster.tensor_strategies`` cuts.
"""

from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

_QUALNAME = "mlx_lm.models.glm5_next"


def apply_glm5_next_mlx_lm_patch() -> bool:
    """Make ``glm5_next`` loadable by mlx-lm. Safe to call repeatedly."""

    try:
        from . import model
    except Exception as exc:  # noqa: BLE001 - a missing vendor tree must not abort
        logger.warning("Could not register %s: %s", _QUALNAME, exc)
        return False
    sys.modules[_QUALNAME] = model
    # ``from mlx_lm.models import glm5_next`` reads the package attribute.
    importlib.import_module("mlx_lm.models").glm5_next = model
    return True


__all__ = ["apply_glm5_next_mlx_lm_patch"]
