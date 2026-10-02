# SPDX-License-Identifier: Apache-2.0
"""Exact-weight loader for Unsloth's mixed ModelOpt Qwen3.8 VLM.

``unsloth/Qwen3.8-27B-NVFP4`` is a compressed-tensors checkpoint with two
different language-layer formats:

* per-channel E4M3 FP8 weights for attention, GDN, the LM head, and late MLPs;
* packed E2M1 NVFP4 weights with E4M3 group scales and an FP32 tensor divisor
  for the remaining MLPs.

mlx-vlm's generic compressed-tensors path cannot represent the extra channel
or tensor scale without either expanding FP8 weights to dense arrays or
folding the FP32 NVFP4 divisor into E4M3 (which re-rounds the group scales).
This module instead uses MLX's native MXFP8/NVFP4 carriers and applies the
original scale after the matmul.  Packed weight codes and stored group/channel
scales therefore remain unchanged.

The dispatch predicate is intentionally strict and currently recognizes the
published dense 64-layer Qwen3.8 VLM geometry and exact config-group targets.
Unknown compressed-tensors combinations continue through the normal loader.
Activations remain in the runtime dtype (A16); the checkpoint's activation
quantization metadata is not enabled by this compatibility path.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)

_FP8_TARGETS = (
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*lm_head",
    r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
)
_NVFP4_TARGETS = (r"re:.*mlp\.(gate|up|down)_proj$",)


@dataclass(frozen=True)
class _QuantRule:
    pattern: re.Pattern[str]
    kind: str


_ACTIVE_RULES: tuple[_QuantRule, ...] | None = None
_PATCHED_MODEL_CLASSES: set[type] = set()


class ScaledQuantizedLinear(nn.Module):
    """MLX quantized carrier followed by an exact source-owned scale."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        *,
        group_size: int,
        bits: int,
        mode: str,
        scale_shape: tuple[int, ...],
        bias: bool = False,
    ):
        super().__init__()
        if input_dims % group_size or (input_dims * bits) % 32:
            raise ValueError(
                f"Invalid quantized geometry input={input_dims}, "
                f"group={group_size}, bits={bits}"
            )
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.weight = mx.zeros((output_dims, input_dims * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // group_size), dtype=mx.uint8)
        self.global_scale = mx.ones(scale_shape, dtype=mx.float32)
        if bias:
            self.bias = mx.zeros((output_dims,))
        self.freeze()

    @classmethod
    def from_linear(cls, linear: nn.Module, kind: str):
        output_dims, input_dims = linear.weight.shape
        has_bias = linear.get("bias") is not None
        if kind == "scaled_mxfp8_channel":
            settings = {
                "group_size": 32,
                "bits": 8,
                "mode": "mxfp8",
                "scale_shape": (output_dims,),
            }
        elif kind == "scaled_nvfp4":
            settings = {
                "group_size": 16,
                "bits": 4,
                "mode": "nvfp4",
                "scale_shape": (),
            }
        else:  # pragma: no cover - protected by the config parser
            raise ValueError(f"Unsupported quantization kind: {kind}")
        return cls(input_dims, output_dims, bias=has_bias, **settings)

    def __call__(self, x):
        y = mx.quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        y = y * self["global_scale"].astype(y.dtype)
        if "bias" in self:
            y = y + self["bias"]
        return y


def _group_matches(
    group: Any,
    *,
    fmt: str,
    targets: tuple[str, ...],
    bits: int,
    strategy: str,
    group_size: int | None,
) -> bool:
    if not isinstance(group, dict) or group.get("format") != fmt:
        return False
    if tuple(group.get("targets") or ()) != targets:
        return False
    weights = group.get("weights")
    if not isinstance(weights, dict):
        return False
    return (
        weights.get("type") == "float"
        and weights.get("num_bits") == bits
        and weights.get("strategy") == strategy
        and weights.get("group_size") == group_size
        and weights.get("dynamic") is False
        and weights.get("symmetric") is True
    )


def is_supported_config(config: dict[str, Any]) -> bool:
    """Return whether *config* matches the validated Qwen3.8 checkpoint."""

    if config.get("model_type") != "qwen3_5":
        return False
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        return False
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        return False
    if (
        text.get("num_hidden_layers") != 64
        or text.get("hidden_size") != 5120
        or text.get("num_experts") is not None
    ):
        return False
    if (
        vision.get("model_type") != "qwen3_5_vision"
        or vision.get("hidden_size") != 1152
        or vision.get("out_hidden_size") != 5120
    ):
        return False

    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        return False
    if (
        quant.get("quant_method") != "compressed-tensors"
        or quant.get("format") != "mixed-precision"
    ):
        return False
    groups = quant.get("config_groups")
    if not isinstance(groups, dict) or set(groups) != {"group_0", "group_1"}:
        return False
    return _group_matches(
        groups["group_0"],
        fmt="float-quantized",
        targets=_FP8_TARGETS,
        bits=8,
        strategy="channel",
        group_size=None,
    ) and _group_matches(
        groups["group_1"],
        fmt="nvfp4-pack-quantized",
        targets=_NVFP4_TARGETS,
        bits=4,
        strategy="tensor_group",
        group_size=16,
    )


def _rules_from_config(config: dict[str, Any]) -> tuple[_QuantRule, ...]:
    if not is_supported_config(config):
        raise ValueError("Unsupported ModelOpt mixed Qwen3.8 configuration")
    groups = config["quantization_config"]["config_groups"]
    rules: list[_QuantRule] = []
    for group_name, kind in (
        ("group_0", "scaled_mxfp8_channel"),
        ("group_1", "scaled_nvfp4"),
    ):
        for target in groups[group_name]["targets"]:
            rules.append(_QuantRule(re.compile(target.removeprefix("re:")), kind))
    return tuple(rules)


def quantization_kind_for_path(path: str, rules: tuple[_QuantRule, ...]) -> str | None:
    """Resolve the first compressed-tensors config-group matching *path*."""

    # The published config excludes ``re:^mtp.*``. The mlx-vlm runtime nests
    # that subtree under ``language_model.mtp``, so apply the exclusion after
    # path sanitization as well as in the source namespace.
    if path.startswith("mtp.") or ".mtp." in path:
        return None
    for rule in rules:
        if rule.pattern.search(path):
            return rule.kind
    return None


def _replace_modelopt_modules(model: nn.Module, rules: tuple[_QuantRule, ...]) -> int:
    leaves = dict(
        tree_flatten(
            model.leaf_modules(),
            is_leaf=lambda module: isinstance(module, nn.Module),
        )
    )
    replaced = 0
    for path, module in list(leaves.items()):
        kind = quantization_kind_for_path(path, rules)
        if kind is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"ModelOpt target {path} is {type(module).__name__}, expected Linear"
            )
        leaves[path] = ScaledQuantizedLinear.from_linear(module, kind)
        replaced += 1
    if replaced == 0:
        raise ValueError("ModelOpt config matched no Qwen3.8 Linear modules")
    model.update_modules(tree_unflatten(list(leaves.items())))
    return replaced


def _patch_qwen35_model_class() -> None:
    from mlx_vlm.models.qwen3_5 import qwen3_5

    model_class = qwen3_5.Model
    if model_class in _PATCHED_MODEL_CLASSES:
        return
    original_init = model_class.__init__

    def patched_init(self, config):
        original_init(self, config)
        if _ACTIVE_RULES is not None:
            count = _replace_modelopt_modules(self, _ACTIVE_RULES)
            logger.info("Bound %d exact-weight ModelOpt modules", count)

    model_class.__init__ = patched_init
    model_class._omlx_qwen38_modelopt_mixed = True
    _PATCHED_MODEL_CLASSES.add(model_class)


def _pack_u8x4(value: mx.array) -> mx.array:
    if value.dtype != mx.uint8 or value.shape[-1] % 4:
        raise TypeError(f"Cannot byte-pack {value.dtype} shape={value.shape}")
    return value.view(mx.uint32)


def transform_weights_exact(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map source tensors to exact MLX carriers without re-quantizing scales."""

    source_keys = set(weights)
    output: dict[str, mx.array] = {}
    consumed: set[str] = set()

    for key, value in weights.items():
        if key in consumed:
            continue

        if key.endswith(".weight_packed"):
            prefix = key[: -len(".weight_packed")]
            scale_key = f"{prefix}.weight_scale"
            divisor_key = f"{prefix}.weight_global_scale"
            input_scale_key = f"{prefix}.input_global_scale"
            required = {scale_key, divisor_key, input_scale_key}
            missing = required - source_keys
            if missing:
                raise KeyError(f"Incomplete NVFP4 module {prefix}: {sorted(missing)}")

            scales = weights[scale_key]
            if value.dtype != mx.uint8 or scales.dtype != mx.uint8:
                raise TypeError(
                    f"Unexpected NVFP4 dtypes for {prefix}: "
                    f"{value.dtype}, {scales.dtype}"
                )
            if value.shape[-1] != scales.shape[-1] * 8:
                raise ValueError(f"NVFP4 group geometry mismatch: {prefix}")

            output[f"{prefix}.weight"] = _pack_u8x4(value)
            output[f"{prefix}.scales"] = scales
            output[f"{prefix}.global_scale"] = mx.reciprocal(
                weights[divisor_key].astype(mx.float32)
            ).squeeze()
            # Safetensors shards do not guarantee that the packed tensor is
            # visited before its sidecars. Remove any sidecar key emitted by
            # an earlier iteration so the transform is order-independent.
            output.pop(scale_key, None)
            output.pop(divisor_key, None)
            output.pop(input_scale_key, None)
            consumed.update(required | {key})
            continue

        if key.endswith(".weight_scale"):
            prefix = key[: -len(".weight_scale")]
            weight_key = f"{prefix}.weight"
            if weight_key in source_keys:
                fp8 = weights[weight_key]
                if fp8.dtype != mx.uint8 or not mx.issubdtype(value.dtype, mx.floating):
                    raise TypeError(
                        f"Unexpected channel-FP8 dtypes for {prefix}: "
                        f"{fp8.dtype}, {value.dtype}"
                    )
                if fp8.shape[-1] % 32 or value.size != fp8.shape[0]:
                    raise ValueError(f"Channel-FP8 geometry mismatch: {prefix}")

                output[weight_key] = _pack_u8x4(fp8)
                output[f"{prefix}.scales"] = mx.full(
                    (fp8.shape[0], fp8.shape[1] // 32),
                    127,
                    dtype=mx.uint8,
                )
                output[f"{prefix}.global_scale"] = value.reshape(-1)
                consumed.update({key, weight_key})
                continue

        if key.endswith((".input_global_scale", ".weight_global_scale")):
            prefix = key.rsplit(".", 1)[0]
            if f"{prefix}.weight_packed" in source_keys:
                consumed.add(key)
                continue
            raise ValueError(f"Unconsumed compressed-tensors metadata: {key}")

        if key.endswith((".k_scale", ".v_scale")):
            consumed.add(key)
            continue

        if key == "model.visual.patch_embed.proj.weight":
            if value.ndim != 5:
                raise ValueError(
                    f"Unexpected Qwen3.8 vision patch shape: {value.shape}"
                )
            value = value.transpose(0, 2, 3, 4, 1)

        output[key] = value
        consumed.add(key)

    if consumed != source_keys:
        raise RuntimeError(
            f"Unprocessed ModelOpt tensors: {sorted(source_keys - consumed)[:20]}"
        )
    return output


def load(model_name: str | Path):
    """Load the validated source checkpoint through mlx-vlm's public API."""

    global _ACTIVE_RULES

    model_path = Path(model_name)
    config = json.loads((model_path / "config.json").read_text())
    rules = _rules_from_config(config)
    _patch_qwen35_model_class()

    import mlx_vlm.utils as vlm_utils

    original_load_config = vlm_utils.load_config
    original_transform = vlm_utils._transform_compressed_tensors_weights

    def load_config_without_generic_quantization(path, *args, **kwargs):
        loaded = original_load_config(path, *args, **kwargs)
        loaded = copy.deepcopy(loaded)
        loaded.pop("quantization", None)
        loaded.pop("quantization_config", None)
        text_config = loaded.get("text_config")
        if isinstance(text_config, dict):
            text_config.pop("quantization", None)
            text_config.pop("quantization_config", None)
        return loaded

    def transform_without_scale_folding(loaded_weights, _quantization_config):
        return transform_weights_exact(loaded_weights), None

    _ACTIVE_RULES = rules
    vlm_utils.load_config = load_config_without_generic_quantization
    vlm_utils._transform_compressed_tensors_weights = transform_without_scale_folding
    try:
        logger.info("Using exact-weight mixed ModelOpt loader for %s", model_path)
        return vlm_utils.load(model_path, trust_remote_code=False, strict=True)
    finally:
        vlm_utils._transform_compressed_tensors_weights = original_transform
        vlm_utils.load_config = original_load_config
        _ACTIVE_RULES = None
