# SPDX-License-Identifier: Apache-2.0
"""MiMo V2.5 text-model support for the pinned mlx-lm dependency.

This ports ml-explore/mlx-lm#1219 into oMLX without modifying the installed
mlx-lm package. The upstream PR adds ``mlx_lm.models.mimo_v2`` for the text
backbone shared by MiMo V2.5 and MiMo V2.5 Pro. The model intentionally drops
vision, audio, speech, and MTP weights during sanitize.

MLX-LM resolves model architectures with a dynamic import under its own
namespace. Until upstream ships the module, register the vendored file under
that name so the normal loader path remains unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PR_HEAD_SHA = "96dd6fc2e0d2a1c43fcbd5b3c4be7dee640ad304"
PR_URL = "https://github.com/ml-explore/mlx-lm/pull/1219"

_MODULE_NAME = "mlx_lm.models.mimo_v2"
_APPLIED = False


def _register_module() -> None:
    if _MODULE_NAME in sys.modules:
        return

    file_path = Path(__file__).parent / "mimo_v2_model.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_MODULE_NAME} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.mimo_v2 = module
    except BaseException:
        if sys.modules.get(_MODULE_NAME) is module:
            sys.modules.pop(_MODULE_NAME)
        raise

    logger.info("Registered %s from %s", _MODULE_NAME, file_path.name)


def apply_mimo_v2_patch() -> bool:
    """Register ``mlx_lm.models.mimo_v2`` when upstream lacks it."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        module = importlib.import_module(_MODULE_NAME)
    except ModuleNotFoundError as error:
        if error.name == "mlx_lm":
            logger.debug("mlx_lm not importable - MiMo V2.5 patch skipped")
            return False
        if error.name != _MODULE_NAME:
            raise
        _register_module()
        applied = True
    else:
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.mimo_v2 = module
        applied = False

    _APPLIED = True
    if applied:
        logger.info(
            "MiMo V2.5 mlx-lm patch applied (PR 1219 head %s)",
            PR_HEAD_SHA[:8],
        )
        return True

    logger.debug("mlx_lm.models.mimo_v2 already available upstream")
    return False


def is_applied() -> bool:
    return _APPLIED


__all__ = ["PR_HEAD_SHA", "PR_URL", "apply_mimo_v2_patch", "is_applied"]
