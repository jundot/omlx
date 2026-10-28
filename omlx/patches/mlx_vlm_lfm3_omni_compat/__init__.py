# SPDX-License-Identifier: Apache-2.0
"""Let mlx-vlm load LiquidAI LFM3-Omni (``lfm3_omni``).

mlx-vlm merged native LFM3-Omni on 2026-10-27 (PR #2610), but the release oMLX
pins, 0.7.10, predates the merge. Its loader therefore fails in
``mlx_vlm.utils.get_model_and_args`` with::

    Model type lfm3_omni not supported. Error: No module named
    'mlx_vlm.speculative.drafters.lfm3_omni'

before a weight is read, so the 2.7 GB checkpoint is unservable.

LFM3-Omni is an any-to-any model: text, images, video and speech in; text and
speech out, the speech half being the MaskGIT depth decoder and the Liquid
codec. mlx-vlm's own loader, generation loop and ``generate_audio`` serve it
directly, so nothing here reimplements the model — the vendored ``lfm3_omni``
tree is upstream's, unchanged (Apache-2.0, ``README.md`` alongside it), and is
registered as ``mlx_vlm.models.lfm3_omni`` by extending the package path.

The tree is a shell around three trees it imports at module import time:

* ``lfm3`` (text tower) and ``lfm3_vl`` (vision tower, projector, processor),
  vendored by ``omlx/patches/mlx_vlm_lfm3_vl_compat``;
* ``lfm3_audio`` (speech encoder, depth decoder, codec loader), vendored by
  ``omlx/patches/mlx_vlm_lfm3_audio_compat``.

Their vendor roots are prepended here too, in dependency order, so this patch
works whether or not those two were applied first. ``lfm2.language`` is only
imported, not vendored: it is identical in 0.7.10 and upstream.

Revert by deleting this directory plus the ``lfm3_omni`` dispatch block in
``omlx/utils/model_loading.py`` and the model type in
``omlx/model_discovery.py``.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PR_URL = "https://github.com/Blaizzy/mlx-vlm/pull/2610"
PR_HEAD_SHA = "7771deda56db"

_PATCH_ROOT = Path(__file__).resolve().parent
_VENDOR_MLX_VLM = _PATCH_ROOT / "vendor" / "mlx_vlm"
# Sibling patches, resolved relative to this file rather than through the
# package, so a missing install shows up as a clear message.
_AUDIO_VENDOR_MLX_VLM = (
    _PATCH_ROOT.parent / "mlx_vlm_lfm3_audio_compat" / "vendor" / "mlx_vlm"
)
_VL_VENDOR_MLX_VLM = (
    _PATCH_ROOT.parent / "mlx_vlm_lfm3_vl_compat" / "vendor" / "mlx_vlm"
)

_MODEL_TYPES = ("lfm3_omni",)
# Dependencies first: every module imports the tree above it relatively, and
# ``lfm3_omni`` imports all three.
_MODULE_NAMES = ("lfm3", "lfm3_vl", "lfm3_audio", "lfm3_omni")

_APPLIED = False


def is_lfm3_omni(config: dict) -> bool:
    """Whether ``config`` declares the LFM3-Omni architecture."""
    return isinstance(config, dict) and config.get("model_type") == "lfm3_omni"


def apply_mlx_vlm_lfm3_omni_compat_patch() -> bool:
    """Install the vendored LFM3-Omni module and its mlx-vlm hooks.

    Safe to call repeatedly. A missing sibling vendor tree logs which one is
    missing and returns False instead of aborting the load.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        if not _install_vendor_namespace():
            return False
        _import_vendor_modules()

        import mlx_vlm.prompt_utils as prompt_utils

        _patch_prompt_utils(prompt_utils)
        _register_auto_processor()
    except Exception as exc:  # noqa: BLE001 - a broken vendor tree must not abort
        logger.warning("Could not register LFM3-Omni with mlx-vlm: %s", exc)
        return False

    _APPLIED = True
    logger.info("LFM3-Omni mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _install_vendor_namespace() -> bool:
    """Prepend every vendor root, dependencies first. False if one is missing."""
    missing = [
        (path, name)
        for path, name in (
            (_VL_VENDOR_MLX_VLM, "mlx_vlm_lfm3_vl_compat (lfm3, lfm3_vl)"),
            (_AUDIO_VENDOR_MLX_VLM, "mlx_vlm_lfm3_audio_compat (lfm3_audio)"),
        )
        if not (path / "models").is_dir()
    ]
    if missing:
        for path, name in missing:
            logger.error(
                "LFM3-Omni needs the %s vendor tree; %s is missing", name, path
            )
        return False

    import mlx_vlm
    import mlx_vlm.models

    # The audio and VL roots only hold their own packages, so prepending them
    # before this one leaves every name resolvable.
    for vendor in (_VL_VENDOR_MLX_VLM, _AUDIO_VENDOR_MLX_VLM, _VENDOR_MLX_VLM):
        _prepend_package_path(mlx_vlm, vendor)
        _prepend_package_path(mlx_vlm.models, vendor / "models")
    return True


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
    # missing from MODEL_CONFIG, which would drop image and audio parts.
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
    processing = importlib.import_module(
        "mlx_vlm.models.lfm3_omni.processing_lfm3_omni"
    )
    install_auto_processor_patch(list(_MODEL_TYPES), processing.Lfm3OmniProcessor)


__all__ = [
    "apply_mlx_vlm_lfm3_omni_compat_patch",
    "is_applied",
    "is_lfm3_omni",
    "PR_HEAD_SHA",
    "PR_URL",
]
