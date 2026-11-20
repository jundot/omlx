# SPDX-License-Identifier: Apache-2.0
"""Register the vendored Ripple architecture for mlx-lm class lookup."""

import sys


def apply_ripple_patch() -> None:
    import mlx_lm.models

    from . import model

    sys.modules["mlx_lm.models.ricursive"] = model
    mlx_lm.models.ricursive = model
