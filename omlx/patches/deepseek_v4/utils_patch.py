# SPDX-License-Identifier: Apache-2.0
"""Patch ``mlx_lm.utils.load_model`` for DeepSeek V4 support.

Two surgical changes from PR 1192 are applied:

1. Weight loading goes through ``_load_safetensors`` instead of ``mx.load`` so
   safetensors files declaring the F8_E8M0 dtype (used by DeepSeek V4 fp8
   block-scale tensors) can be reinterpreted as U8 in-place.
2. The ``elif quant_method == "fp8" and model_type == "deepseek_v4"`` branch
   in the quantization config dispatch builds the per-layer quantization
   spec via ``deepseek_v4.make_quantization_config``.

The rest of ``load_model``'s body is identical to the v0.31.3 (``ed1fca4``)
upstream — copied verbatim from PR 1192 head ``5c10538``. mlx-lm is pinned
to a commit, so the body is stable.

When mlx-lm merges PR 1192 upstream this patch should be removed.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import logging
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as _utils
from mlx.utils import tree_map

logger = logging.getLogger(__name__)

SAFETENSORS_DTYPE_FALLBACKS = {"F8_E8M0": "U8"}

_PATCHED = False


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


def _build_patched_load_model() -> Callable:
    """Build the replacement ``load_model`` closure.

    Captures ``_get_classes`` from the live ``mlx_lm.utils`` module so the
    default behaves the same as upstream. Internal helpers (``load_config``,
    ``_transform_awq_weights``) are looked up dynamically at call time so
    they pick up any other patches applied to ``mlx_lm.utils``.
    """
    default_get_classes = _utils._get_classes

    def patched_load_model(
        model_path: Path,
        lazy: bool = False,
        strict: bool = True,
        model_config: dict[str, Any] | None = None,
        get_model_classes: Callable = default_get_classes,
    ) -> tuple[nn.Module, dict]:
        config = _utils.load_config(model_path)
        if model_config is not None:
            config.update(model_config)

        weight_files = glob.glob(str(model_path / "model*.safetensors"))

        if not weight_files and strict:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        weights = {}
        for wf in weight_files:
            weights.update(_load_safetensors(wf))  # PR 1192 change

        if (model_file := config.get("model_file")) is not None:
            spec = importlib.util.spec_from_file_location(
                "custom_model",
                model_path / model_file,
            )
            arch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(arch)
            model_class, model_args_class = arch.Model, arch.ModelArgs
        else:
            model_class, model_args_class = get_model_classes(config=config)

        if "quantization_config" not in config:
            text_config = config.get("text_config", {})
            if "quantization_config" in text_config:
                config["quantization_config"] = text_config["quantization_config"]

        model_args = model_args_class.from_dict(config)
        model = model_class(model_args)

        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        def _quantize(quantization):
            def class_predicate(p, m):
                if p in config["quantization"]:
                    return config["quantization"][p]
                if not hasattr(m, "to_quantized"):
                    return False
                return f"{p}.scales" in weights

            nn.quantize(
                model,
                group_size=quantization["group_size"],
                bits=quantization["bits"],
                mode=quantization.get("mode", "affine"),
                class_predicate=class_predicate,
            )

        if (quantization := config.get("quantization", None)) is not None:
            _quantize(quantization)
        elif quantization_config := config.get("quantization_config", False):
            quant_method = quantization_config["quant_method"]
            if quant_method == "bitnet":
                from mlx_lm.models.bitlinear_layers import bitnet_quantize

                model = bitnet_quantize(model, quantization_config)
            elif quant_method == "mxfp4":
                quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
                config["quantization"] = quantization
                config["quantization_config"] = quantization
                _quantize(quantization)
            elif quant_method == "compressed-tensors":
                quantization = {"group_size": 32, "bits": 4, "mode": "affine"}
                config["quantization"] = quantization
                config["quantization_config"] = quantization
                _quantize(quantization)
            elif quant_method in ("awq", "gptq"):
                weights, quantization = _utils._transform_awq_weights(
                    weights, quantization_config
                )
                config["quantization"] = quantization
                config["quantization_config"] = quantization
                _quantize(quantization)
            elif (
                quant_method == "fp8"
                and config.get("model_type", None) == "deepseek_v4"
            ):  # PR 1192 new branch
                from mlx_lm.models.deepseek_v4 import make_quantization_config

                quantization = make_quantization_config(model)
                config["quantization"] = quantization
                config["quantization_config"] = quantization
                _quantize(quantization)

        if config.get("quantize_activations", False):

            def _maybe_qq(m):
                if isinstance(m, nn.QuantizedLinear):
                    if m.mode not in ("nvfp4", "mxfp8"):
                        raise ValueError(
                            f"Mode ({m.mode}) does not support activation quantization"
                        )
                    if m.get("bias", False):
                        raise ValueError(
                            "Linear layer with bias does not support activation quantization"
                        )
                    out_dims, in_dims = m.weight.shape
                    in_dims *= 32 // m.bits
                    return nn.QQLinear(in_dims, out_dims, m.group_size, m.bits, m.mode)
                return m

            leaves = tree_map(
                _maybe_qq, model.leaf_modules(), is_leaf=nn.Module.is_module
            )
            model.update_modules(leaves)

        model.eval()
        model.load_weights(list(weights.items()), strict=strict)

        if not lazy:
            mx.eval(model.parameters())

        return model, config

    return patched_load_model


def apply_utils_patch() -> bool:
    """Replace ``mlx_lm.utils.load_model`` and inject ``_load_safetensors``.

    Idempotent. Also updates other ``mlx_lm.*`` modules that imported
    ``load_model`` directly via ``from .utils import load_model``.
    """
    global _PATCHED
    if _PATCHED:
        return False

    patched = _build_patched_load_model()

    _utils.SAFETENSORS_DTYPE_FALLBACKS = SAFETENSORS_DTYPE_FALLBACKS
    _utils._load_safetensors = _load_safetensors
    _utils.load_model = patched

    # Update any module that has a stale binding to the original load_model.
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("mlx_lm"):
            continue
        if mod_name == "mlx_lm.utils":
            continue
        existing = getattr(mod, "load_model", None)
        if existing is not None and existing is not patched:
            try:
                mod.load_model = patched
            except Exception:
                pass

    _PATCHED = True
    logger.info("mlx_lm.utils.load_model replaced (deepseek_v4 fp8 + F8_E8M0 fallback)")
    return True
