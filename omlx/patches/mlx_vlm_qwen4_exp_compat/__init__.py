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
        _patch_prompt_utils()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Qwen4-Exp mlx-vlm registration failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Qwen4-Exp mlx-vlm compatibility patch applied")
    return True


def _patch_prompt_utils() -> None:
    """Teach the pinned formatter Qwen4's Qwen3.5-compatible media layout."""
    import mlx_vlm.prompt_utils as prompt_utils

    current = prompt_utils.get_message_json
    if getattr(current, "_omlx_qwen4_exp", False):
        return

    def get_message_json(model_type, *args, **kwargs):
        if model_type == "qwen4_exp":
            model_type = "qwen3_5_moe"
        return current(model_type, *args, **kwargs)

    get_message_json._omlx_qwen4_exp = True
    prompt_utils.get_message_json = get_message_json


def is_applied() -> bool:
    return _APPLIED


_PROFILER_ARMED = False


def _arm_layer_profiler() -> None:
    """OMLX_PROFILE_LAYERS=1: time linear_attn / self_attn / mlp per call.

    Diagnostic only (plan v4, F4.2 E2). A sync barrier per submodule call
    makes decode much slower; only the RELATIVE shares are meaningful —
    the same distortion as the ZMLX repro capsule this compares against.
    State updates not depended on by the module output settle at the next
    barrier, slightly shifting cost to the following module.
    """
    global _PROFILER_ARMED
    import os
    import time

    if os.environ.get("OMLX_PROFILE_LAYERS") != "1" or _PROFILER_ARMED:
        return
    _PROFILER_ARMED = True

    import mlx.core as mx
    from mlx_vlm.models.qwen4_exp import language as _lang

    acc: dict[str, list[float]] = {}

    def _leaves(out):
        if isinstance(out, mx.array):
            return [out]
        if isinstance(out, (tuple, list)):
            return [a for x in out for a in _leaves(x)]
        return []

    def _wrap(cls, name):
        orig = cls.__call__

        def timed(self, *a, **k):
            mx.synchronize()
            t0 = time.perf_counter()
            out = orig(self, *a, **k)
            arrs = _leaves(out)
            if arrs:
                mx.eval(*arrs)
            slot = acc.setdefault(name, [0.0, 0])
            slot[0] += time.perf_counter() - t0
            slot[1] += 1
            if slot[1] % 480 == 0:
                logger.info(
                    "layer-profile: %s",
                    " · ".join(
                        f"{n} {v[0] * 1000 / v[1]:.3f} ms/call x{v[1]}"
                        for n, v in sorted(acc.items())
                    ),
                )
            return out

        cls.__call__ = timed

    _wrap(_lang.Qwen4ExpGatedDeltaNet, "linear_attn")
    _wrap(_lang.Qwen4ExpAttention, "self_attn")
    _wrap(_lang.Qwen3_5MoeSparseMoeBlock, "mlp")
    logger.info("layer profiler armed (OMLX_PROFILE_LAYERS=1) — diagnostic only")


def configure_qwen4_exp_runtime(
    model_path: str | Path,
    mode: str | None = None,
    *,
    mtp_enabled: bool = False,
) -> str:
    """Select PLE storage and optional Lightning MTP before construction."""
    apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        configure_mtp_runtime,
        configure_ple_runtime,
    )

    resolved = configure_ple_runtime(model_path, mode=mode)
    mtp_runtime = configure_mtp_runtime(model_path, enabled=mtp_enabled)
    logger.info("Qwen4-Exp PLE mode for %s: %s", model_path, resolved)
    _arm_layer_profiler()
    if mtp_enabled and not mtp_runtime.enabled:
        logger.warning(
            "Qwen4-Exp Lightning MTP was requested for %s, but no embedded "
            "MTP tensors were found",
            model_path,
        )
    elif mtp_runtime.enabled:
        logger.info(
            "Qwen4-Exp Lightning MTP enabled for %s (checkpoint layout: %s)",
            model_path,
            mtp_runtime.checkpoint_prefix,
        )
    return resolved
