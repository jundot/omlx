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
        _patch_prompt_utils()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Qwen4-Exp mlx-vlm registration failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Qwen4-Exp mlx-vlm compatibility patch applied")
    return True


def _patch_prompt_utils() -> None:
    """Teach the pinned formatter Qwen4's Qwen3.5-compatible media layout."""
    import mlx_vlm.prompt_utils as prompt_utils

    current = prompt_utils.get_message_json
    if getattr(current, "_omlx_qwen4_exp", False):
        return

    def get_message_json(model_type, *args, **kwargs):
        if model_type == "qwen4_exp":
            model_type = "qwen3_5_moe"
        return current(model_type, *args, **kwargs)

    get_message_json._omlx_qwen4_exp = True
    prompt_utils.get_message_json = get_message_json


def is_applied() -> bool:
    return _APPLIED


def configure_qwen4_exp_runtime(
    model_path: str | Path,
    mode: str | None = None,
    *,
    mtp_enabled: bool = False,
) -> str:
    """Select PLE storage and optional Lightning MTP before construction."""
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        configure_mtp_runtime,
        configure_ple_runtime,
    )

    resolved = configure_ple_runtime(model_path, mode=mode)
    mtp_runtime = configure_mtp_runtime(model_path, enabled=mtp_enabled)
    logger.info("Qwen4-Exp PLE mode for %s: %s", model_path, resolved)
    if mtp_enabled and not mtp_runtime.enabled:
        logger.warning(
            "Qwen4-Exp Lightning MTP was requested for %s, but no embedded "
            "MTP tensors were found",
            model_path,
        )
    elif mtp_runtime.enabled:
        logger.info(
            "Qwen4-Exp Lightning MTP enabled for %s (checkpoint layout: %s)",
            model_path,
            mtp_runtime.checkpoint_prefix,
        )
    return resolved
