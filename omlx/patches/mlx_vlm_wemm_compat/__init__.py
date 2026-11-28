# SPDX-License-Identifier: Apache-2.0
"""Tencent WeMM-Video (wemm_qwen3_5_moe & wemm_qwen3_5) compatibility layer for mlx-vlm.

Supports Tencent's omni-modal WeMM-Video family (2B, 4B, 9B dense and 35B-A3B MoE),
combining Qwen3.5 language/vision with a 32-layer Whisper audio encoder and 2-layer
BERT Q-Former:

- Installs ``mlx_vlm.models.wemm_qwen3_5_moe`` and ``mlx_vlm.models.wemm_qwen3_5``
  onto the ``mlx_vlm.models`` namespace.
- Registers ``wemm_qwen3_5_moe`` and ``wemm_qwen3_5`` in ``prompt_utils.MODEL_CONFIG``
  with ``LIST_WITH_IMAGE_FIRST``.
- Registers the torch-free ``WeMMProcessor`` with ``AutoProcessor``.
- Wraps ``mlx_vlm.utils.load_config`` so dense WeMM-Video checkpoints (which declare
  ``architectures: ["WeMMVideoForConditionalGeneration"]`` and ``model_type: "qwen3_5"``
  with ``audio_config.qformer_config``) route to ``wemm_qwen3_5`` instead of losing
  their audio tower on stock ``qwen3_5``.
- Preserves video file paths across ``mlx_vlm.utils.prepare_inputs`` so ``WeMMProcessor``
  can decode bundled audio tracks for omni-modal ``video_url`` requests.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
WEMM_MODEL_TYPES = ("wemm_qwen3_5_moe", "wemm_qwen3_5")
_APPLIED = False


def is_wemm_config(config: dict | None) -> bool:
    """Return True when *config* belongs to a Tencent WeMM-Video checkpoint."""
    if not isinstance(config, dict):
        return False
    if config.get("model_type") in WEMM_MODEL_TYPES:
        return True
    archs = config.get("architectures") or []
    if "WeMMVideoForConditionalGeneration" in archs:
        return True
    audio_cfg = config.get("audio_config")
    if isinstance(audio_cfg, dict) and isinstance(audio_cfg.get("qformer_config"), dict):
        return config.get("model_type") in ("qwen3_5", "qwen3_5_moe")
    return False


def normalize_wemm_model_type(config: dict | None) -> str | None:
    """Return the effective WeMM model_type for a checkpoint config."""
    if not isinstance(config, dict):
        return None
    mt = config.get("model_type")
    if mt in WEMM_MODEL_TYPES:
        return mt
    if is_wemm_config(config):
        tc = config.get("text_config") or {}
        if int(tc.get("num_experts", 0) or 0) > 0 or mt == "qwen3_5_moe":
            return "wemm_qwen3_5_moe"
        return "wemm_qwen3_5"
    return mt


def apply_mlx_vlm_wemm_compat_patch() -> bool:
    """Install the vendored WeMM-Video modules and mlx-vlm discovery hooks."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        _install_vendor_namespace()
        _import_vendor_modules()

        import mlx_vlm.prompt_utils as prompt_utils
        import mlx_vlm.utils as vlm_utils

        _register_prompt_format(prompt_utils)
        _patch_load_config(vlm_utils)
        _patch_prepare_inputs(vlm_utils)
        _register_auto_processor()
    except Exception as exc:  # noqa: BLE001
        logger.debug("WeMM-Video mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("WeMM-Video mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def _install_vendor_namespace() -> None:
    import mlx_vlm
    import mlx_vlm.models

    for mt in WEMM_MODEL_TYPES:
        existing = sys.modules.get(f"mlx_vlm.models.{mt}")
        if existing is not None and not hasattr(existing, "__path__"):
            sys.modules.pop(f"mlx_vlm.models.{mt}", None)
            if hasattr(mlx_vlm.models, mt):
                delattr(mlx_vlm.models, mt)
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
    importlib.import_module("mlx_vlm.models.wemm_qwen3_5_moe")
    importlib.import_module("mlx_vlm.models.wemm_qwen3_5")


def _register_prompt_format(prompt_utils: Any) -> None:
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if isinstance(model_config, dict) and message_format is not None:
        placeholder = getattr(message_format, "LIST_WITH_IMAGE_FIRST", None)
        if placeholder is not None:
            for mt in WEMM_MODEL_TYPES:
                model_config.setdefault(mt, placeholder)


def _patch_load_config(vlm_utils: Any) -> None:
    original = getattr(vlm_utils, "load_config", None)
    if original is None or getattr(original, "_omlx_wemm_compat", False):
        return

    def patched_load_config(model_path, **kwargs):
        config = original(model_path, **kwargs)
        if isinstance(config, dict) and is_wemm_config(config):
            effective = normalize_wemm_model_type(config)
            if effective:
                config["model_type"] = effective
        return config

    patched_load_config._omlx_wemm_compat = True
    patched_load_config._omlx_original = original
    vlm_utils.load_config = patched_load_config


def _patch_prepare_inputs(vlm_utils: Any) -> None:
    original = getattr(vlm_utils, "prepare_inputs", None)
    if original is None or getattr(original, "_omlx_wemm_compat", False):
        return

    from mlx_vlm.models.wemm_qwen3_5_moe.processing_wemm import WeMMProcessor

    def patched_prepare_inputs(processor, *args, **kwargs):
        if isinstance(processor, WeMMProcessor):
            vids = kwargs.get("videos")
            if vids is not None:
                vlist = vids if isinstance(vids, list) else [vids]
                processor._recent_video_paths = [
                    str(v)
                    for v in vlist
                    if isinstance(v, (str, bytes)) or hasattr(v, "__fspath__")
                ]
        return original(processor, *args, **kwargs)

    patched_prepare_inputs._omlx_wemm_compat = True
    patched_prepare_inputs._omlx_original = original
    vlm_utils.prepare_inputs = patched_prepare_inputs


def _register_auto_processor() -> None:
    from mlx_vlm.models.base import install_auto_processor_patch
    from mlx_vlm.models.wemm_qwen3_5_moe.processing_wemm import WeMMProcessor

    install_auto_processor_patch(list(WEMM_MODEL_TYPES), WeMMProcessor)


__all__ = [
    "WEMM_MODEL_TYPES",
    "apply_mlx_vlm_wemm_compat_patch",
    "is_applied",
    "is_wemm_config",
    "normalize_wemm_model_type",
]
