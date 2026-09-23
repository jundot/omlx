# SPDX-License-Identifier: Apache-2.0
"""MiMo V2.5 text-model support for the pinned mlx-lm dependency.

This ports ml-explore/mlx-lm#1219 into oMLX without modifying the installed
mlx-lm package, then extends that model with oMLX's MiMo multimodal and native
MTP support.

MLX-LM resolves model architectures with a dynamic import under its own
namespace. Register the vendored file under that name so the normal loader path
remains unchanged. The vendored module deliberately replaces an installed
upstream module because released mlx-lm builds do not contain these extensions.
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
    file_path = Path(__file__).parent / "mimo_v2_model.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_MODULE_NAME} from {file_path}")

    previous = sys.modules.get(_MODULE_NAME)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.mimo_v2 = module
    except BaseException:
        if sys.modules.get(_MODULE_NAME) is module:
            if previous is None:
                sys.modules.pop(_MODULE_NAME)
            else:
                sys.modules[_MODULE_NAME] = previous
        raise

    logger.info("Registered %s from %s", _MODULE_NAME, file_path.name)


def apply_mimo_v2_patch() -> bool:
    """Register oMLX's extended ``mlx_lm.models.mimo_v2`` implementation."""
    global _APPLIED
    file_path = (Path(__file__).parent / "mimo_v2_model.py").resolve()
    current = sys.modules.get(_MODULE_NAME)
    current_path = getattr(current, "__file__", None)
    if _APPLIED and current_path and Path(current_path).resolve() == file_path:
        return False

    try:
        importlib.import_module("mlx_lm.models")
    except ModuleNotFoundError as error:
        if error.name == "mlx_lm":
            logger.debug("mlx_lm not importable - MiMo V2.5 patch skipped")
            return False
        raise

    _register_module()
    _APPLIED = True
    logger.info(
        "MiMo V2.5 mlx-lm patch applied (PR 1219 head %s)",
        PR_HEAD_SHA[:8],
    )
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = ["PR_HEAD_SHA", "PR_URL", "apply_mimo_v2_patch", "is_applied"]
