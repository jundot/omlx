# SPDX-License-Identifier: Apache-2.0
"""Keep a single <image> token in Unlimited-OCR prompts that already carry one."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Config model_type carried by baidu/Unlimited-OCR checkpoints.
_MODEL_TYPE = "unlimited-ocr"

_APPLIED = False


def apply_mlx_vlm_unlimited_ocr_compat_patch() -> bool:
    """Wrap mlx-vlm's get_message_json for Unlimited-OCR prompts."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_vlm.prompt_utils as prompt_utils

        _patch_prompt_utils(prompt_utils)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unlimited-OCR mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Unlimited-OCR mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _patch_prompt_utils(prompt_utils: Any) -> None:
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
            content = prompt_utils.extract_text_from_content(prompt)
            # Unlimited-OCR uses a single literal <image> token for one or many
            # pages, matching upstream infer_multi (MessageFormat.SINGLE_IMAGE_TOKEN).
            # The model card's prompt already starts with <image>; adding a second
            # one fails the processor's token count check.
            if (
                role == "user"
                and not skip_image_token
                and num_images > 0
                and "<image>" not in content
            ):
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
