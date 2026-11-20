# SPDX-License-Identifier: Apache-2.0
"""Register the separately supplied SenseNova Loop text model with mlx-lm."""

from __future__ import annotations

import importlib
import sys

_APPLIED = False


def apply_sensenova_loop_patch() -> bool:
    global _APPLIED
    if _APPLIED:
        return False

    try:
        registered_model = importlib.import_module("sensenova_mlx.registered_model")
    except ModuleNotFoundError as exc:
        if exc.name in {"sensenova_mlx", "sensenova_mlx.registered_model"}:
            raise RuntimeError(
                "SenseNova requires the separately supplied sensenova_mlx support package; "
                "see docs/sensenova.md"
            ) from exc
        raise

    import mlx_lm.models

    sys.modules["mlx_lm.models.sensenova_loop"] = registered_model
    mlx_lm.models.sensenova_loop = registered_model
    _APPLIED = True
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_sensenova_loop_patch", "is_applied"]
