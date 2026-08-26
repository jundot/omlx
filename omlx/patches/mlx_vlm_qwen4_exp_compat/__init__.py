# SPDX-License-Identifier: Apache-2.0
"""Register the vendored Qwen4-Exp implementation with mlx-vlm."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_APPLIED = False


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_string = str(path)
    if path_string not in package_path:
        package_path.append(path_string)


def apply_mlx_vlm_qwen4_exp_compat_patch() -> bool:
    """Expose ``mlx_vlm.models.qwen4_exp`` from oMLX's vendor tree."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_vlm
        import mlx_vlm.models

        _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
        _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")
        importlib.import_module("mlx_vlm.models.qwen4_exp")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Qwen4-Exp mlx-vlm registration failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Qwen4-Exp mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def configure_qwen4_exp_runtime(
    model_path: str | Path,
    mode: str | None = None,
) -> str:
    """Select resident or SSD-backed PLE before model construction."""
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import configure_ple_runtime

    resolved = configure_ple_runtime(model_path, mode=mode)
    logger.info("Qwen4-Exp PLE mode for %s: %s", model_path, resolved)
    return resolved
