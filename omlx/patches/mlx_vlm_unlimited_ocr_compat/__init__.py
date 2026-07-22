# SPDX-License-Identifier: Apache-2.0
"""Baidu Unlimited-OCR compatibility layer for the pinned mlx-vlm.

The `unlimited_ocr` model package landed in mlx-vlm PR #1427 (commit
`9909cee`, 2026-07-01), which is newer than oMLX's current mlx-vlm pin
(`78b96eb`, 2026-06-28). Rather than bumping the whole pin (8 upstream
commits, KEEP-patch re-verification), this vendors the model package and
wires up only the discovery surface oMLX needs:

- installs the vendored `mlx_vlm.models.unlimited_ocr` package onto the real
  `mlx_vlm.models` namespace so `get_model_and_args` can import it (its own
  relative imports `..deepseekocr`/`..base`/`..cache` resolve against the real
  mlx-vlm package, which already ships `deepseekocr` at the pin),
- adds the `unlimited-ocr -> unlimited_ocr` entry to `mlx_vlm.utils`'
  `MODEL_REMAPPING` (without it, `get_model_and_args` tries to import the
  literal dashed module name and falls through to the
  `mlx_vlm.speculative.drafters.unlimited-ocr` lookup seen in issue #2314),
- reproduces the PR's single-`<image>` multi-page prompt semantics
  (`MessageFormat.SINGLE_IMAGE_TOKEN`) by wrapping `prompt_utils`'
  `get_message_json` for the `unlimited-ocr` model type, without mutating the
  frozen `MessageFormat` enum.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"

# Config model_type carried by baidu/Unlimited-OCR checkpoints. The mlx-vlm
# module directory is the underscore form (`unlimited_ocr`).
_MODEL_TYPE = "unlimited-ocr"
_MODULE_NAME = "unlimited_ocr"

_APPLIED = False


def apply_mlx_vlm_unlimited_ocr_compat_patch() -> bool:
    """Install the vendored Unlimited-OCR module and mlx-vlm discovery hooks."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        _install_vendor_namespace()
        _import_vendor_modules()

        import mlx_vlm.prompt_utils as prompt_utils
        import mlx_vlm.utils as vlm_utils

        _patch_model_remapping(vlm_utils)
        _patch_prompt_utils(prompt_utils)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unlimited-OCR mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Unlimited-OCR mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _install_vendor_namespace() -> None:
    import mlx_vlm
    import mlx_vlm.models

    _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
    _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_str = str(path)
    if path_str not in package_path:
        package_path.append(path_str)


def _import_vendor_modules() -> None:
    # Importing the package runs its __init__, which imports
    # processing_unlimitedocr (install_auto_processor_patch) and unlimitedocr
    # (AutoProcessor.register) so the processor is discoverable.
    importlib.import_module(f"mlx_vlm.models.{_MODULE_NAME}")


def _patch_model_remapping(vlm_utils: Any) -> None:
    remapping = getattr(vlm_utils, "MODEL_REMAPPING", None)
    if isinstance(remapping, dict):
        remapping.setdefault(_MODEL_TYPE, _MODULE_NAME)


def _patch_prompt_utils(prompt_utils: Any) -> None:
    # apply_chat_template short-circuits to text-only formatting for any
    # model_type missing from MODEL_CONFIG, which would skip get_message_json
    # (and the <image> token) entirely. Register the key so the image branch is
    # taken; the mapped MessageFormat is never consulted because the
    # get_message_json wrapper below intercepts unlimited-ocr first.
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if isinstance(model_config, dict) and message_format is not None:
        placeholder = getattr(message_format, "IMAGE_TOKEN", None)
        if placeholder is not None:
            model_config.setdefault(_MODEL_TYPE, placeholder)

    original = getattr(prompt_utils, "get_message_json", None)
    if original is None or getattr(original, "_omlx_unlimited_ocr_compat", False):
        return

    def patched_get_message_json(
        model_name: str,
        prompt: str,
        role: str = "user",
        skip_image_token: bool = False,
        skip_audio_token: bool = False,
        num_images: int = 0,
        num_audios: int = 0,
        **kwargs,
    ):
        if _is_unlimited_ocr(model_name):
            content = "" if prompt is None else str(prompt)
            # Unlimited-OCR uses a single literal <image> token for one or many
            # pages, matching upstream infer_multi (MessageFormat.SINGLE_IMAGE_TOKEN).
            if role == "user" and not skip_image_token and num_images > 0:
                content = f"<image>{content}"
            return {"role": role, "content": content}
        return original(
            model_name,
            prompt,
            role=role,
            skip_image_token=skip_image_token,
            skip_audio_token=skip_audio_token,
            num_images=num_images,
            num_audios=num_audios,
            **kwargs,
        )

    patched_get_message_json._omlx_unlimited_ocr_compat = True
    patched_get_message_json._omlx_original = original
    prompt_utils.get_message_json = patched_get_message_json


def _is_unlimited_ocr(model_name: Any) -> bool:
    return (
        isinstance(model_name, str)
        and model_name.lower().replace("_", "-") == _MODEL_TYPE
    )


__all__ = ["apply_mlx_vlm_unlimited_ocr_compat_patch", "is_applied"]
