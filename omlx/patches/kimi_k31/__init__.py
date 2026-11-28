# SPDX-License-Identifier: Apache-2.0
"""Let mlx-lm load Kimi K3.1 Pro (``kimi_k31``).

Pinned mlx-lm has Kimi K3 and the DeepSeek-V3.2 indexer but no ``kimi_k31``.
This registers ``mlx_lm.models.kimi_k31`` as K3 with sparse attention on its
full-attention layers.
"""

from __future__ import annotations

import importlib
import sys

_QUALNAME = "mlx_lm.models.kimi_k31"


def apply_kimi_k31_patch() -> bool:
    """Make ``kimi_k31`` loadable by mlx-lm. Safe to call repeatedly."""

    from . import model

    sys.modules[_QUALNAME] = model
    # ``from mlx_lm.models import kimi_k31`` reads the package attribute.
    importlib.import_module("mlx_lm.models").kimi_k31 = model
    return True


__all__ = ["apply_kimi_k31_patch"]
