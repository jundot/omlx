# SPDX-License-Identifier: Apache-2.0
"""Ling 3.0 Flash support for the pinned mlx-lm dependency.

Vendors ``mlx_lm.models.bailing_hybrid`` from scaryrawr/mlx-lm's
``ling-3.0-flash`` branch without changing oMLX's official mlx-lm pin. The
branch is based on that exact pinned revision and adds the mixed MLA/KDA model
used by Ling 3.0 Flash.

MLX-LM resolves architectures by importing modules under its own namespace.
Register the vendored implementation there only while upstream lacks it.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from .swiglu_clamp import ensure_swiglu_clamp

logger = logging.getLogger(__name__)

BRANCH_HEAD_SHA = "d719464ff754e65d9dec496ef3fea27bddefd79c"
SOURCE_URL = (
    "https://github.com/scaryrawr/mlx-lm/blob/ling-3.0-flash/"
    "mlx_lm/models/bailing_hybrid.py"
)

_MODULE_NAME = "mlx_lm.models.bailing_hybrid"
_APPLIED = False


def _register_module() -> None:
    if _MODULE_NAME in sys.modules:
        return

    file_path = Path(__file__).parent / "bailing_hybrid_model.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_MODULE_NAME} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.bailing_hybrid = module
    except BaseException:
        if sys.modules.get(_MODULE_NAME) is module:
            sys.modules.pop(_MODULE_NAME)
        raise

    logger.info("Registered %s from %s", _MODULE_NAME, file_path.name)


def apply_bailing_hybrid_patch() -> bool:
    """Register Ling's ``bailing_hybrid`` model when upstream lacks it."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        module = importlib.import_module(_MODULE_NAME)
    except ModuleNotFoundError as error:
        if error.name == "mlx_lm":
            logger.debug("mlx_lm not importable - bailing_hybrid patch skipped")
            return False
        if error.name != _MODULE_NAME:
            raise
        _register_module()
        applied = True
    else:
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.bailing_hybrid = module
        applied = False

    # Whichever build ended up live, make sure Ling's trained SwiGLU clamp is
    # in force. The vendored copy implements it in-source; an mlx-lm build
    # that already ships bailing_hybrid does not.
    module = importlib.import_module(_MODULE_NAME)
    if ensure_swiglu_clamp(module):
        logger.info("Ling SwiGLU clamp installed on %s", _MODULE_NAME)

    _APPLIED = True

    if applied:
        logger.info(
            "Ling 3.0 Flash mlx-lm patch applied (branch head %s)",
            BRANCH_HEAD_SHA[:8],
        )
        return True

    logger.debug("mlx_lm.models.bailing_hybrid already available upstream")
    return False


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "BRANCH_HEAD_SHA",
    "SOURCE_URL",
    "apply_bailing_hybrid_patch",
    "is_applied",
]
