# SPDX-License-Identifier: Apache-2.0
"""Let mlx-vlm load LiquidAI LFM3-VL (``lfm3_vl``).

mlx-vlm merged native LFM3 on 2026-10-27 (PR #2607, refined by #2608), but the
release oMLX pins, 0.7.10, predates the merge. Its loader therefore fails in
``mlx_vlm.utils.get_model_and_args`` with ``Model type lfm3_vl not supported``
before a weight is read, so a 2.2 GB checkpoint that fits on one Mac is
unservable.

The vendored tree is upstream's, unchanged (Apache-2.0, ``LICENSE`` and
``README.md`` alongside it): ``lfm3`` is the text tower and ``lfm3_vl`` adds
the hybrid conv/attention vision tower, the projector and the processor. They
are registered as ``mlx_vlm.models.lfm3`` / ``mlx_vlm.models.lfm3_vl`` by
extending the package path, so the existing loader, quantizer and cache paths
apply unchanged.

Scope is LFM3-VL only. ``lfm3_audio``, ``lfm3_omni`` and the ``d1_lfm3_*``
decision models are separate trees with their own serving paths.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PR_URL = "https://github.com/Blaizzy/mlx-vlm/pull/2607"
FOLLOW_UP_PR_URL = "https://github.com/Blaizzy/mlx-vlm/pull/2608"
PR_HEAD_SHA = "bfd7fb194ff1"

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"

_MODEL_TYPES = ("lfm3_vl",)
# ``lfm3`` is the text tower LFM3-VL embeds; it must be importable before the
# VL module, which imports it relatively.
_MODULE_NAMES = ("lfm3", "lfm3_vl")

_APPLIED = False


def is_lfm3_vl(config: dict) -> bool:
    """Whether ``config`` declares the LFM3-VL architecture."""
    return isinstance(config, dict) and config.get("model_type") == "lfm3_vl"


def apply_mlx_vlm_lfm3_vl_compat_patch() -> bool:
    """Install the vendored LFM3-VL module and its mlx-vlm hooks.

    Safe to call repeatedly.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        _install_vendor_namespace()
        _import_vendor_modules()

        import mlx_vlm.prompt_utils as prompt_utils

        _patch_prompt_utils(prompt_utils)
        _register_auto_processor()
    except Exception as exc:  # noqa: BLE001 - a broken vendor tree must not abort
        logger.warning("Could not register LFM3-VL with mlx-vlm: %s", exc)
        return False

    _APPLIED = True
    logger.info("LFM3-VL mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _install_vendor_namespace() -> None:
    import mlx_vlm
    import mlx_vlm.models

    _prepend_package_path(mlx_vlm, _VENDOR_MLX_VLM)
    _prepend_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")


def _prepend_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_str = str(path)
    if path_str not in package_path:
        package_path.insert(0, path_str)


def _import_vendor_modules() -> None:
    for name in _MODULE_NAMES:
        importlib.import_module(f"mlx_vlm.models.{name}")


def _patch_prompt_utils(prompt_utils: Any) -> None:
    # ``apply_chat_template`` raises "Unsupported model" for any model_type
    # missing from MODEL_CONFIG, which would drop image parts and fail.
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if not isinstance(model_config, dict) or message_format is None:
        return
    placeholder = getattr(message_format, "LIST_WITH_IMAGE_FIRST", None)
    if placeholder is None:
        return
    for model_type in _MODEL_TYPES:
        model_config.setdefault(model_type, placeholder)


def _register_auto_processor() -> None:
    from mlx_vlm.models.base import install_auto_processor_patch

    # Import through the installed namespace: the vendor tree has no package
    # __init__ chain of its own.
    processing = importlib.import_module("mlx_vlm.models.lfm3_vl.processing_lfm3_vl")
    install_auto_processor_patch(list(_MODEL_TYPES), processing.Lfm3VlProcessor)


__all__ = [
    "apply_mlx_vlm_lfm3_vl_compat_patch",
    "is_applied",
    "is_lfm3_vl",
    "FOLLOW_UP_PR_URL",
    "PR_HEAD_SHA",
    "PR_URL",
]
