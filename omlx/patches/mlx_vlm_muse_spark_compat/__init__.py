# SPDX-License-Identifier: Apache-2.0
"""Register the oMLX Muse Spark VLM ahead of upstream model discovery.

The package reuses the vendored Muse Glimmer vision tower and processor, and
the oMLX Muse Spark text model (``omlx.patches.muse_spark``).
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"

_MODEL_TYPE = "muse_spark"

_APPLIED = False


def apply_mlx_vlm_muse_spark_compat_patch() -> bool:
    """Install the vendored muse_spark module and mlx-vlm discovery hooks."""
    global _APPLIED
    if _APPLIED:
        return False

    from ..mlx_vlm_muse_glimmer_compat import (
        apply_mlx_vlm_muse_glimmer_compat_patch,
    )
    from ..mlx_vlm_muse_glimmer_compat import is_applied as glimmer_applied

    apply_mlx_vlm_muse_glimmer_compat_patch()
    if not glimmer_applied():
        raise RuntimeError(
            "Muse Spark needs the Muse Glimmer mlx-vlm compatibility patch."
        )

    import mlx_vlm
    import mlx_vlm.models
    import mlx_vlm.prompt_utils as prompt_utils

    _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
    _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")
    # Importing the package registers MuseSparkProcessor with AutoProcessor.
    importlib.import_module(f"mlx_vlm.models.{_MODEL_TYPE}")
    _register_prompt_format(prompt_utils)

    _APPLIED = True
    logger.info("Muse Spark mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_str = str(path)
    if path_str not in package_path:
        package_path.insert(0, path_str)


def _register_prompt_format(prompt_utils: Any) -> None:
    # Without a MODEL_CONFIG entry apply_chat_template drops the image parts.
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if isinstance(model_config, dict) and message_format is not None:
        model_config.setdefault(_MODEL_TYPE, message_format.LIST_WITH_IMAGE_FIRST)


__all__ = ["apply_mlx_vlm_muse_spark_compat_patch", "is_applied"]
