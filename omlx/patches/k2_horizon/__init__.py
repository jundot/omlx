# SPDX-License-Identifier: Apache-2.0
"""K2-Horizon (model_type: k2_horizon) support for oMLX.

Registers the community k2_horizon MLX adapter so oMLX can load
IFM/K2-Horizon-MoVA-36B-A4B and its quantizations.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_APPLIED = False

_VENDORED_MODULES = (
    ("mlx_lm.models.k2_horizon", "k2_horizon_model.py", "mlx_lm.models"),
)


def _register_module(qualname: str, filename: str, package: str) -> None:
    if qualname in sys.modules:
        return

    file_path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[qualname] = module
    spec.loader.exec_module(module)

    parent = importlib.import_module(package)
    setattr(parent, qualname.rsplit(".", 1)[1], module)
    logger.info("Registered %s from %s", qualname, filename)


def apply_k2_horizon_patch() -> bool:
    """Register K2-Horizon support."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.debug("mlx_lm not importable - k2_horizon patch skipped")
        return False

    try:
        importlib.import_module("mlx_lm.models.k2_horizon")
        upstream = True
    except ImportError:
        upstream = False

    if not upstream:
        for qualname, filename, package in _VENDORED_MODULES:
            _register_module(qualname, filename, package)
        _APPLIED = True
        logger.info("K2-Horizon mlx-lm patch applied")
        return True

    _APPLIED = True
    logger.debug("mlx_lm.models.k2_horizon already available upstream")
    return False
