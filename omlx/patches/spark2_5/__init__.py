"""Spark-X2.5 support for mlx-lm builds that do not provide spark2_5."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_MODULE_NAME = "mlx_lm.models.spark2_5"
_APPLIED = False


def _register_module() -> None:
    if _MODULE_NAME in sys.modules:
        return

    file_path = Path(__file__).parent / "spark2_5_model.py"

    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        str(file_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {_MODULE_NAME} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_lm.models"

    sys.modules[_MODULE_NAME] = module

    try:
        spec.loader.exec_module(module)

        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.spark2_5 = module
    except BaseException:
        if sys.modules.get(_MODULE_NAME) is module:
            sys.modules.pop(_MODULE_NAME)
        raise

    logger.info("Registered %s from %s", _MODULE_NAME, file_path.name)


def apply_spark2_5_patch() -> bool:
    """Register Spark-X2.5 when upstream mlx-lm lacks it."""
    global _APPLIED

    if _APPLIED:
        return False

    try:
        module = importlib.import_module(_MODULE_NAME)
    except ModuleNotFoundError as error:
        if error.name == "mlx_lm":
            logger.debug("mlx_lm not importable - spark2_5 patch skipped")
            return False

        if error.name != _MODULE_NAME:
            raise

        _register_module()
        applied = True
    else:
        models_pkg = importlib.import_module("mlx_lm.models")
        models_pkg.spark2_5 = module
        applied = False

    _APPLIED = True

    if applied:
        logger.info("Spark-X2.5 mlx-lm patch applied")
        return True

    logger.debug("mlx_lm.models.spark2_5 already available upstream")
    return False


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "apply_spark2_5_patch",
    "is_applied",
]
