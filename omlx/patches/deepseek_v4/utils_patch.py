# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4's contributions to ``mlx_lm.utils.load_model``.

Three changes from mlx-lm PR 1192 are applied:

1. ``_load_safetensors`` replaces ``mx.load`` on ``mlx_lm.utils`` so
   safetensors files declaring the F8_E8M0 dtype (used by DeepSeek V4 fp8
   block-scale tensors) can be reinterpreted as U8 in place.
2. A config transform decides whether the native ratio-128 attention path is
   viable for this checkpoint's bit width.
3. A ``fp8`` quantization handler builds the per-layer quantization spec via
   ``deepseek_v4.make_quantization_config``.

The three reach ``load_model`` through the registration seams in
``omlx.patches.mlx_lm_load_model_inputs``, which owns the single copy of the
construction pipeline. Registering rather than carrying a second copy of the
function is what lets a checkpoint arriving in memory get the same fp8 and
ratio-128 treatment as one read off disk.

When mlx-lm merges PR 1192 upstream this module should be removed, which also
removes its registrations.
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as _utils

logger = logging.getLogger(__name__)

SAFETENSORS_DTYPE_FALLBACKS = {"F8_E8M0": "U8"}

_PATCHED = False


def _native_ratio128_attention_enabled(config: dict[str, Any]) -> bool:
    """Keep the native ratio-128 attention path off for sub-4-bit V4."""
    if not str(config.get("model_type", "")).startswith("deepseek_v4"):
        return True

    quantizations = [config.get("quantization"), config.get("quantization_config")]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        quantizations.append(text_config.get("quantization_config"))

    for quantization in quantizations:
        bits = quantization.get("bits") if isinstance(quantization, dict) else None
        if (
            isinstance(bits, (int, float))
            and not isinstance(bits, bool)
            and float(bits) < 4
        ):
            return False
    return True


def _apply_ratio128_policy(config: dict[str, Any]) -> None:
    """Settle ``use_native_ratio128_attention`` before the model args are built."""

    if not str(config.get("model_type", "")).startswith("deepseek_v4"):
        return
    config["use_native_ratio128_attention"] = bool(
        config.get("use_native_ratio128_attention", True)
    ) and _native_ratio128_attention_enabled(config)


def _fp8_quantization(model: nn.Module, config: dict[str, Any]) -> dict | None:
    """Build the per-layer quantization spec for a V4 fp8 checkpoint."""

    if not str(config.get("model_type", "")).startswith("deepseek_v4"):
        return None
    from mlx_lm.models.deepseek_v4 import make_quantization_config

    return make_quantization_config(model)


def _load_safetensors(path: str) -> dict:
    """Load a safetensors file with a dtype fallback for F8_E8M0.

    DeepSeek V4 fp8 checkpoints declare ``F8_E8M0`` for the per-block
    exponent scale tensors. ``mx.load`` rejects unknown dtypes; the
    fallback rewrites the safetensors header in place to advertise the
    bytes as ``U8`` (raw uint8), loads, and restores the original header.
    """
    try:
        return mx.load(path)
    except RuntimeError as e:
        if not any(dtype in str(e) for dtype in SAFETENSORS_DTYPE_FALLBACKS):
            raise
        load_error = e

    with open(path, "r+b") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        original_header = f.read(header_len)
        header = json.loads(original_header)
        changed = False

        for tensor_info in header.values():
            if not isinstance(tensor_info, dict):
                continue
            dtype = tensor_info.get("dtype")
            if dtype in SAFETENSORS_DTYPE_FALLBACKS:
                tensor_info["dtype"] = SAFETENSORS_DTYPE_FALLBACKS[dtype]
                changed = True

        if not changed:
            raise load_error

        patched_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        if len(patched_header) > header_len:
            raise RuntimeError(
                f"Cannot reinterpret unsupported safetensors dtype in {path}: "
                "patched header is larger than the original header."
            )

        try:
            f.seek(8)
            f.write(patched_header)
            f.write(b" " * (header_len - len(patched_header)))
            f.flush()
            return mx.load(path)
        finally:
            f.seek(8)
            f.write(original_header)
            f.flush()


def apply_utils_patch() -> bool:
    """Register the DeepSeek V4 load path with ``load_model``. Idempotent."""

    global _PATCHED
    if _PATCHED:
        return False

    from omlx.patches.mlx_lm_load_model_inputs import (
        install_load_model_inputs,
        register_config_transform,
        register_quant_method,
    )

    install_load_model_inputs()

    _utils.SAFETENSORS_DTYPE_FALLBACKS = SAFETENSORS_DTYPE_FALLBACKS
    _utils._load_safetensors = _load_safetensors
    register_config_transform("deepseek_v4_ratio128", _apply_ratio128_policy)
    register_quant_method("fp8", _fp8_quantization)

    _PATCHED = True
    logger.info(
        "deepseek_v4 registered with load_model (fp8 quantization, F8_E8M0 "
        "fallback, ratio-128 attention policy)"
    )
    return True
