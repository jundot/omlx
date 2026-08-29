# SPDX-License-Identifier: Apache-2.0
"""Register the vendored GLM-5.3-Flash implementation with mlx-vlm."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PR_URL = "https://github.com/Blaizzy/mlx-vlm/pull/2030"
PR_MERGE_SHA = "fa27a9a692770c39fdf57b9a985fad084a90aec2"
_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_APPLIED = False


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_string = str(path)
    if path_string not in package_path:
        package_path.append(path_string)


def apply_mlx_vlm_glm5_next_compat_patch() -> bool:
    """Expose ``mlx_vlm.models.glm5_next`` from oMLX's vendor tree."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_vlm
        import mlx_vlm.models

        from omlx.patches.deepseek_v4 import apply_pooling_cache_support

        apply_pooling_cache_support()
        _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
        _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")
        importlib.import_module("mlx_vlm.models.glm5_next")

        # mlx-vlm has no glm5_next entry in MODEL_CONFIG, so get_message_json()
        # raises "Unsupported model: glm5_next" on every turn that carries no
        # image.  That aborts _format_messages_for_vlm_template() as a whole,
        # so the engine falls back to mlx-vlm's generic formatter, which emits
        # no image placeholders -- while oMLX still extracts the images.  Any
        # conversation mixing an image with a plain turn then fails with
        # "More images were provided than image tokens."  GLM-5.3 takes the
        # same list-with-image-first shape as glm4v.
        from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat

        MODEL_CONFIG.setdefault("glm5_next", MessageFormat.LIST_WITH_IMAGE_FIRST)
    except Exception as exc:  # noqa: BLE001
        logger.debug("GLM-5.3 mlx-vlm registration failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("GLM-5.3 mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "PR_MERGE_SHA",
    "PR_URL",
    "apply_mlx_vlm_glm5_next_compat_patch",
    "is_applied",
]
