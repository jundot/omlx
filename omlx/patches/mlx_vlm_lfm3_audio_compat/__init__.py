# SPDX-License-Identifier: Apache-2.0
"""Let mlx-vlm load LiquidAI LFM3-Audio (``lfm3_audio``).

mlx-vlm merged native LFM3-Audio on 2026-10-27 (PR #2609), but the release
oMLX pins, 0.7.10, predates the merge. Its loader therefore fails in
``mlx_vlm.utils.get_model_and_args`` with ``Model type lfm3_audio not
supported. Error: No module named
'mlx_vlm.speculative.drafters.lfm3_audio'`` before a weight is read, so the
2.33 GB 8-bit checkpoint is unservable.

LFM3-Audio is an omni model served by mlx-vlm, not mlx-audio: it takes 16 kHz
speech in and emits text, 24 kHz speech, or both, through one backbone. The
vendored tree is upstream's, unchanged (Apache-2.0, ``README.md`` alongside
it), and adds the audio encoder, the projector, the depth decoder that turns
MaskGIT frames into 12 codebook codes, and the decode-only Liquid codec. The
``lfm3`` text tower it imports relatively comes from
``../mlx_vlm_lfm3_vl_compat/vendor``, which must stay installed; both paths are
prepended here so a bare ``import mlx_vlm.models.lfm3_audio`` resolves.

Revert by deleting this directory together with the ``lfm3_audio`` block in
``omlx/utils/model_loading.py``; the pinned wheel is untouched.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PR_URL = "https://github.com/Blaizzy/mlx-vlm/pull/2609"
PR_HEAD_SHA = "6e49f5de1ff0"

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
# The shared text tower lives in the LFM3-VL patch; only one copy is vendored.
_TEXT_TOWER_VENDOR = (
    Path(__file__).resolve().parent.parent
    / "mlx_vlm_lfm3_vl_compat"
    / "vendor"
    / "mlx_vlm"
)

_MODEL_TYPES = ("lfm3_audio",)
# ``lfm3`` is the text tower the audio model imports relatively and must be
# importable first; ``lfm2`` (its encoder trunk) ships with the pinned wheel.
_MODULE_NAMES = ("lfm3", "lfm3_audio")

_APPLIED = False


def is_lfm3_audio(config: dict) -> bool:
    """Whether ``config`` declares the LFM3-Audio architecture."""
    return isinstance(config, dict) and config.get("model_type") == "lfm3_audio"


def apply_mlx_vlm_lfm3_audio_compat_patch() -> bool:
    """Install the vendored LFM3-Audio module and its mlx-vlm hooks.

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
        logger.warning("Could not register LFM3-Audio with mlx-vlm: %s", exc)
        return False

    _APPLIED = True
    logger.info("LFM3-Audio mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _install_vendor_namespace() -> None:
    import mlx_vlm
    import mlx_vlm.models

    if not (_TEXT_TOWER_VENDOR / "models" / "lfm3").is_dir():
        raise FileNotFoundError(
            "LFM3-Audio needs the shared lfm3 text tower from "
            f"{_TEXT_TOWER_VENDOR}; install the mlx_vlm_lfm3_vl_compat patch too"
        )

    for vendor in (_VENDOR_MLX_VLM, _TEXT_TOWER_VENDOR):
        _prepend_package_path(mlx_vlm, vendor)
        _prepend_package_path(mlx_vlm.models, vendor / "models")


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
    # missing from MODEL_CONFIG, which would drop the audio placeholders.
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
    # __init__ chain of its own. The module registers itself on import, and the
    # explicit call keeps the hook in place if that changes upstream.
    processing = importlib.import_module(
        "mlx_vlm.models.lfm3_audio.processing_lfm3_audio"
    )
    install_auto_processor_patch(list(_MODEL_TYPES), processing.Lfm3AudioProcessor)


__all__ = [
    "apply_mlx_vlm_lfm3_audio_compat_patch",
    "is_applied",
    "is_lfm3_audio",
    "PR_HEAD_SHA",
    "PR_URL",
]
