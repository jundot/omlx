# SPDX-License-Identifier: Apache-2.0
"""Raon-1 mini support without ``trust_remote_code``.

KRAFTON publishes Raon-1 mini's MLX architecture as a custom module next to the
weights (``config.json`` → ``"model_file": "raon_mlx_model.py"``), which mlx-lm
only imports under ``trust_remote_code``. oMLX vendors a reviewed copy of that
module as ``mlx_lm.models.raon1_mini`` and, for the matching file name, drops
``model_file`` from the loaded config so mlx-lm resolves the vendored class by
``model_type`` instead of executing code from the model directory.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_REPO = "KRAFTON/Raon-1-mini-preview-mlx"
SOURCE_REVISION = "c558ec0f8050925e44a80e7ee357f41fbcf8cfe5"
MODEL_TYPE = "raon1_mini"
# The remote module this vendored copy stands in for.
REMOTE_MODEL_FILE = "raon_mlx_model.py"

_MODULE_NAME = f"mlx_lm.models.{MODEL_TYPE}"
_APPLIED = False


def normalize_raon1_mini_model_file(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve Raon through the vendored module instead of the remote file."""
    if (
        config.get("model_type") == MODEL_TYPE
        and config.get("model_file") == REMOTE_MODEL_FILE
    ):
        config.pop("model_file")
    return config


def _register_module() -> None:
    file_path = Path(__file__).parent / "raon1_mini_model.py"
    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None and getattr(existing, "__file__", None) == str(file_path):
        return

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_MODULE_NAME} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        models_pkg = importlib.import_module("mlx_lm.models")
        setattr(models_pkg, MODEL_TYPE, module)
    except BaseException:
        if sys.modules.get(_MODULE_NAME) is module:
            if existing is None:
                sys.modules.pop(_MODULE_NAME)
            else:
                sys.modules[_MODULE_NAME] = existing
        raise

    logger.info("Registered %s from %s", _MODULE_NAME, file_path.name)


def apply_raon1_mini_patch() -> bool:
    """Register the vendored Raon-1 mini model under mlx-lm."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        importlib.import_module("mlx_lm.models")
    except ModuleNotFoundError as error:
        if error.name == "mlx_lm":
            logger.debug("mlx_lm not importable - raon1_mini patch skipped")
            return False
        raise
    _register_module()
    _APPLIED = True
    logger.info(
        "Raon-1 mini mlx-lm patch applied (%s@%s)", SOURCE_REPO, SOURCE_REVISION[:8]
    )
    return True


__all__ = [
    "MODEL_TYPE",
    "REMOTE_MODEL_FILE",
    "SOURCE_REPO",
    "SOURCE_REVISION",
    "apply_raon1_mini_patch",
    "normalize_raon1_mini_model_file",
]
