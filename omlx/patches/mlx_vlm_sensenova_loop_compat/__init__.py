# SPDX-License-Identifier: Apache-2.0
"""Register oMLX SenseNova-Loop ahead of upstream model discovery.

SenseNova-Loop-9B is published as a vLLM-style multimodal config whose text
backbone is byte-identical to mlx-lm's qwen3_5 text_config.  The shared-weight
looped decoder with manifold-constrained routing (MHC) needs a dedicated
mlx-vlm model implementation, which is vendored here so oMLX's VLM engine can
load the official checkpoint without waiting for an upstream mlx-vlm PR.

This compat module mirrors ``mlx_vlm_muse_glimmer_compat``: it installs the
vendored package into mlx-vlm's namespace, imports the model package (which
registers the processor with AutoProcessor as a side effect), and registers the
image-first chat prompt format."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"

_MODEL_TYPE = "sensenova_loop"

_APPLIED = False


def apply_mlx_vlm_sensenova_loop_compat_patch() -> bool:
    """Install the vendored sensenova_loop module and mlx-vlm discovery hooks."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        _install_vendor_namespace()
        _import_vendor_modules()

        import mlx_vlm.prompt_utils as prompt_utils

        _register_prompt_format(prompt_utils)
    except Exception as exc:  # noqa: BLE001
        logger.debug("SenseNova-Loop mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("SenseNova-Loop mlx-vlm compatibility patch applied")
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
    # Importing the package registers sensenova_loop's processor with
    # AutoProcessor and pulls in the qwen3_5/qwen3_vl base modules the
    # model builds on.  The shared-weight looped decoder is self-contained
    # (config.py / language.py / sensenova_loop.py), so no activations shim
    # is required.
    importlib.import_module(f"mlx_vlm.models.{_MODEL_TYPE}")


def _register_prompt_format(prompt_utils: Any) -> None:
    # apply_chat_template short-circuits to text-only formatting for any
    # model_type missing from MODEL_CONFIG, dropping image parts entirely.
    # The pinned get_message_json handles LIST_WITH_IMAGE_FIRST generically,
    # so the MODEL_CONFIG entry is the only registration needed.
    model_config = getattr(prompt_utils, "MODEL_CONFIG", None)
    message_format = getattr(prompt_utils, "MessageFormat", None)
    if isinstance(model_config, dict) and message_format is not None:
        placeholder = getattr(message_format, "LIST_WITH_IMAGE_FIRST", None)
        if placeholder is not None:
            model_config.setdefault(_MODEL_TYPE, placeholder)


__all__ = ["apply_mlx_vlm_sensenova_loop_compat_patch", "is_applied"]
