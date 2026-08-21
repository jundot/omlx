# SPDX-License-Identifier: Apache-2.0
"""Intern-S2-Mobius text-model support for the pinned mlx-lm dependency.

Registers ``mlx_lm.models.interns2_mobius`` from a vendored copy so the normal
mlx-lm loader path resolves the architecture without modifying the installed
package. Intern-S2-Mobius is a hybrid Gated Delta Net / full-attention MoE model
(a Qwen3.5 sibling); the vendored module drops the vision and MTP weights during
sanitize, so this patch brings up text generation only.

MLX-LM resolves model architectures with a dynamic import under its own
namespace, keyed on the checkpoint's top-level ``model_type`` ("interns2_mobius").
Until upstream ships the module, register the vendored file under that name so
the normal loader path remains unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_MODULE_NAME = "mlx_lm.models.interns2_mobius"
_APPLIED = False


def _register_module() -> None:
    if _MODULE_NAME in sys.modules:
        return

    file_path = Path(__file__).parent / "interns2_mobius_model.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_MODULE_NAME} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    # Insert before executing so intra-module imports and repeated patch
    # attempts see the same module object.
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.interns2_mobius = module
    except BaseException:
        # Do not poison future imports with a partially initialized module.
        if sys.modules.get(_MODULE_NAME) is module:
            sys.modules.pop(_MODULE_NAME)
        raise

    logger.info("Registered %s from %s", _MODULE_NAME, file_path.name)


def apply_interns2_mobius_patch() -> bool:
    """Register ``mlx_lm.models.interns2_mobius`` when upstream lacks it."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        module = importlib.import_module(_MODULE_NAME)
    except ModuleNotFoundError as error:
        if error.name == "mlx_lm":
            logger.debug("mlx_lm not importable - Intern-S2-Mobius patch skipped")
            return False
        # Only a missing Intern-S2-Mobius module justifies the vendored
        # fallback; surface errors caused by an installed upstream module.
        if error.name != _MODULE_NAME:
            raise
        _register_module()
        applied = True
    else:
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.interns2_mobius = module
        applied = False

    _APPLIED = True
    if applied:
        logger.info("Intern-S2-Mobius mlx-lm patch applied")
        return True

    logger.debug("mlx_lm.models.interns2_mobius already available upstream")
    return False


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_interns2_mobius_patch", "is_applied"]
