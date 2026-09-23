# SPDX-License-Identifier: Apache-2.0
"""Register oMLX Inkling, its processor, and checkpoint mapping."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"

# Checkpoints carry "inkling_mm_model"; the module directory is "inkling".
_MODEL_TYPES = ("inkling", "inkling_mm_model")
_MODULE_NAME = "inkling"

_APPLIED = False


def apply_mlx_vlm_inkling_compat_patch() -> bool:
    """Install the vendored inkling module and mlx-vlm discovery hooks."""
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
        _patch_load_config(vlm_utils)
        _register_auto_processor()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Inkling mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Inkling mlx-vlm compatibility patch applied")
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
        package_path.insert(0, path_str)


def _import_vendor_modules() -> None:
    importlib.import_module("mlx_vlm.models.activations")
    importlib.import_module("mlx_vlm.models.mlp")
    importlib.import_module(f"mlx_vlm.models.{_MODULE_NAME}")


def _patch_model_remapping(vlm_utils: Any) -> None:
    remapping = getattr(vlm_utils, "MODEL_REMAPPING", None)
    if isinstance(remapping, dict):
        remapping.setdefault("inkling_mm_model", _MODULE_NAME)


def _is_inkling(model_name: Any) -> bool:
    return isinstance(model_name, str) and model_name.lower() in _MODEL_TYPES


def _patch_prompt_utils(prompt_utils: Any) -> None:
    # apply_chat_template short-circuits to text-only formatting for any
    # model_type missing from MODEL_CONFIG, dropping image parts entirely.
    # Register both keys; the mapped format is never consulted because the
    # get_message_json wrapper intercepts inkling first.
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if isinstance(model_config, dict) and message_format is not None:
        placeholder = getattr(message_format, "LIST_WITH_IMAGE_FIRST", None) or getattr(
            message_format, "LIST_WITH_IMAGE", None
        )
        if placeholder is not None:
            for model_type in _MODEL_TYPES:
                model_config.setdefault(model_type, placeholder)

    original = getattr(prompt_utils, "get_message_json", None)
    if original is None or getattr(original, "_omlx_inkling_compat", False):
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
        if _is_inkling(model_name):
            text = "" if prompt is None else str(prompt)
            content: list[dict[str, Any]] | str
            if role == "user" and not skip_image_token and num_images > 0:
                content = [{"type": "image"} for _ in range(num_images)]
                content.append({"type": "text", "text": text})
            else:
                content = text
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

    patched_get_message_json._omlx_inkling_compat = True
    patched_get_message_json._omlx_original = original
    prompt_utils.get_message_json = patched_get_message_json


def _patch_load_config(vlm_utils: Any) -> None:
    original = getattr(vlm_utils, "load_config", None)
    if original is None or getattr(original, "_omlx_inkling_compat", False):
        return

    def patched_load_config(model_path, **kwargs):
        config = original(model_path, **kwargs)
        try:
            if (
                not isinstance(config, dict)
                or config.get("model_type") not in _MODEL_TYPES
            ):
                return config
            if not config.get("quantization") and not config.get(
                "quantization_config"
            ):
                hf_quant_path = Path(model_path) / "hf_quant_config.json"
                if hf_quant_path.exists():
                    with open(hf_quant_path, encoding="utf-8") as f:
                        hf_quant = json.load(f).get("quantization", {})
                    if hf_quant.get("quant_algo") == "NVFP4":
                        # nvfp4 implies group_size=16 (PR #1756). The
                        # sanitize in the vendored model passes the packed
                        # expert codes/scales through byte-identically.
                        config["quantization"] = config["quantization_config"] = {
                            "group_size": 16,
                            "bits": 4,
                            "mode": "nvfp4",
                        }

            from mlx_vlm.models.inkling.config import build_qkvr_fusion_policy

            text_config = config.setdefault("text_config", {})
            quantization = config.get("quantization")
            if not isinstance(quantization, dict):
                quantization = config.get("quantization_config")
            mtp_config = config.get("mtp_config") or {}
            mtp_layers = int(
                text_config.get("mtp_num_hidden_layers")
                or mtp_config.get("num_nextn_predict_layers")
                or mtp_config.get("num_mtp_layers")
                or 0
            )
            trunk_policy = build_qkvr_fusion_policy(
                quantization,
                int(text_config.get("num_hidden_layers") or 0),
                "language_model.model.layers.{layer_idx}.self_attn.",
            )
            mtp_policy = build_qkvr_fusion_policy(
                quantization,
                mtp_layers,
                "language_model.mtp.blocks.{layer_idx}.transformer_block.self_attn.",
            )
            text_config["qkvr_fused_layers"] = trunk_policy
            text_config["mtp_qkvr_fused_layers"] = mtp_policy
        except Exception as exc:  # noqa: BLE001
            logger.debug("Inkling hf_quant_config translation failed: %s", exc)
        return config

    patched_load_config._omlx_inkling_compat = True
    patched_load_config._omlx_original = original
    vlm_utils.load_config = patched_load_config


def _register_auto_processor() -> None:
    from mlx_vlm.models.base import install_auto_processor_patch

    # Import through the installed namespace (the vendor tree has no
    # package __init__ chain of its own).
    processing = importlib.import_module(
        f"mlx_vlm.models.{_MODULE_NAME}.processing_inkling"
    )
    install_auto_processor_patch(list(_MODEL_TYPES), processing.InklingProcessor)


__all__ = ["apply_mlx_vlm_inkling_compat_patch", "is_applied"]
