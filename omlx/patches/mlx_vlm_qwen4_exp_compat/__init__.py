# SPDX-License-Identifier: Apache-2.0
"""Register the vendored Qwen4-Exp implementation with mlx-vlm."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_APPLIED = False


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_string = str(path)
    if path_string not in package_path:
        package_path.append(path_string)


def apply_mlx_vlm_qwen4_exp_compat_patch() -> bool:
    """Expose ``mlx_vlm.models.qwen4_exp`` from oMLX's vendor tree."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_vlm
        import mlx_vlm.models

        _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
        _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")
        importlib.import_module("mlx_vlm.models.qwen4_exp")
        _patch_vlm_model_adapter()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Qwen4-Exp mlx-vlm registration failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Qwen4-Exp mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


def configure_qwen4_exp_runtime(
    model_path: str | Path,
    mode: str | None = None,
    *,
    mtp_enabled: bool = False,
) -> str:
    """Select PLE storage and optional native MTP before construction."""
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        configure_mtp_runtime,
        configure_ple_runtime,
    )

    resolved = configure_ple_runtime(model_path, mode=mode)
    mtp_runtime = configure_mtp_runtime(model_path, enabled=mtp_enabled)
    logger.info("Qwen4-Exp PLE mode for %s: %s", model_path, resolved)
    if mtp_enabled and not mtp_runtime.enabled:
        logger.warning(
            "Qwen4-Exp MTP requested for %s but no embedded MTP tensors were found",
            model_path,
        )
    elif mtp_runtime.enabled:
        logger.info(
            "Qwen4-Exp native MTP enabled for %s (checkpoint layout: %s)",
            model_path,
            mtp_runtime.checkpoint_prefix,
        )
    return resolved


def _patch_vlm_model_adapter() -> None:
    """Expose model-family MTP hooks through oMLX's language adapter."""
    from omlx.models.vlm import VLMModelAdapter

    if getattr(VLMModelAdapter, "_omlx_mtp_adapter_patched", False):
        return

    @property
    def mtp(self):
        language_model = self._language_model
        getter = getattr(language_model, "get_mtp_module", None)
        if callable(getter):
            return getter()
        return getattr(language_model, "mtp", None)

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        return self._language_model.mtp_forward(
            hidden_states,
            next_token_ids,
            mtp_cache,
            return_hidden=return_hidden,
            logits_keep=logits_keep,
        )

    def make_mtp_cache(self):
        return self._language_model.make_mtp_cache()

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        return self._language_model.rollback_speculative_cache(
            caches,
            gdn_states,
            accepted,
            block_size,
        )

    VLMModelAdapter.mtp = mtp
    VLMModelAdapter.mtp_forward = mtp_forward
    VLMModelAdapter.make_mtp_cache = make_mtp_cache
    VLMModelAdapter.rollback_speculative_cache = rollback_speculative_cache
    VLMModelAdapter._omlx_mtp_adapter_patched = True
