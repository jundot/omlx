# SPDX-License-Identifier: Apache-2.0
"""Patch ``mlx_lm.utils.load_model`` for DeepSeek V4 support.

Two surgical changes from PR 1192 are applied:

1. Weight loading goes through ``_load_safetensors`` instead of ``mx.load`` so
   safetensors files declaring the F8_E8M0 dtype (used by DeepSeek V4 fp8
   block-scale tensors) can be reinterpreted as U8 in-place.
2. The ``elif quant_method == "fp8" and model_type.startswith("deepseek_v4")``
   branch
   in the quantization config dispatch builds the per-layer quantization
   spec via ``deepseek_v4.make_quantization_config``.

The rest of ``load_model``'s body is identical to the v0.31.3 (``ed1fca4``)
upstream — copied verbatim from PR 1192 head ``5c10538``. mlx-lm is pinned
to a commit, so the body is stable.

When mlx-lm merges PR 1192 upstream this patch should be removed.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import logging
import re
import struct
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as _utils
from mlx.utils import tree_map

logger = logging.getLogger(__name__)

SAFETENSORS_DTYPE_FALLBACKS = {"F8_E8M0": "U8"}

_PATCHED = False


def _contains_sub4_bits(value: Any) -> bool:
    """Return whether a quantization tree contains a numeric sub-4-bit mode."""
    if isinstance(value, Mapping):
        bits = value.get("bits")
        if (
            isinstance(bits, (int, float))
            and not isinstance(bits, bool)
            and float(bits) < 4
        ):
            return True
        return any(_contains_sub4_bits(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_sub4_bits(child) for child in value)
    return False


def _native_ratio128_attention_enabled(config: dict[str, Any]) -> bool:
    """Keep the native ratio-128 attention path off for sub-4-bit V4."""
    if not str(config.get("model_type", "")).startswith("deepseek_v4"):
        return True

    quantizations = [config.get("quantization"), config.get("quantization_config")]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        quantizations.append(text_config.get("quantization_config"))

    return not any(_contains_sub4_bits(quantization) for quantization in quantizations)


_TARGET_ONLY_SCHEMA = "mlx-serve.dsv4-target-only-gold-view-provenance.v2"
_TARGET_ONLY_CONFIG_SIZE = 37_869
_TARGET_ONLY_CONFIG_SHA256 = (
    "ab61e3230f196c6eba04bfa81158dd527a7f356b6d926cc4794907a19f35b75d"
)
_QUANTIZATION_GLOBAL_KEYS = frozenset({"bits", "group_size", "mode"})
_ROUTED_EXPERT_MODULE_RE = re.compile(
    r"^model\.layers\.(\d+)\.ffn\.switch_mlp\.(gate_proj|down_proj|up_proj)$"
)
_SHARED_EXPERT_MODULE_RE = re.compile(
    r"^model\.layers\.(\d+)\.ffn\.shared_experts\.(gate_proj|down_proj|up_proj)$"
)
_EXPERT_PROJECTION_TO_CHECKPOINT = {
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
}


def _is_target_only_checkpoint(config: Mapping[str, Any]) -> bool:
    """Return whether config opts into the strict mlx-serve target-only ABI."""
    provenance = config.get("mlx_serve_converter")
    return (
        config.get("model_type") == "deepseek_v4"
        and isinstance(provenance, Mapping)
        and provenance.get("schema") == _TARGET_ONLY_SCHEMA
    )


def _load_target_only_config_identity(config_path: Path) -> dict[str, Any]:
    """Bind the strict path to its one published config and return raw JSON."""
    raw_config = config_path.read_bytes()
    digest = hashlib.sha256(raw_config).hexdigest()
    if len(raw_config) != _TARGET_ONLY_CONFIG_SIZE or digest != _TARGET_ONLY_CONFIG_SHA256:
        raise ValueError(
            "Target-only checkpoint config.json does not match the published "
            "artifact identity."
        )
    parsed_config = json.loads(raw_config)
    if not isinstance(parsed_config, dict):
        raise ValueError("Target-only checkpoint config.json must be an object.")
    return parsed_config


def _canonical_target_only_module_path(module_path: str) -> str:
    """Convert one target-only converter module name to the model-tree name.

    The published target-only artifact retains the converter's compact names:
    ``embed``, ``head``, and fused ``ffn.experts.w{1,2,3}``.  The injected
    model instead exposes ``model.embed_tokens``, ``lm_head``, and
    ``ffn.switch_mlp.{gate,down,up}_proj``.  Both spellings are accepted as
    input, but callers must detect aliases that collide on this one canonical
    name rather than allow one to overwrite the other.
    """
    top_level = {
        "embed": "model.embed_tokens",
        "head": "lm_head",
        "norm": "model.norm",
    }
    canonical = top_level.get(module_path, module_path)
    if canonical.startswith("layers."):
        canonical = "model." + canonical

    if _ROUTED_EXPERT_MODULE_RE.fullmatch(canonical):
        # Already canonical: nothing to do.
        return canonical
    if _SHARED_EXPERT_MODULE_RE.fullmatch(canonical):
        # Already canonical: nothing to do.
        return canonical

    for source, destination in (
        (".ffn.experts.w1", ".ffn.switch_mlp.gate_proj"),
        (".ffn.experts.w2", ".ffn.switch_mlp.down_proj"),
        (".ffn.experts.w3", ".ffn.switch_mlp.up_proj"),
        (".ffn.shared_experts.w1", ".ffn.shared_experts.gate_proj"),
        (".ffn.shared_experts.w2", ".ffn.shared_experts.down_proj"),
        (".ffn.shared_experts.w3", ".ffn.shared_experts.up_proj"),
    ):
        if canonical.endswith(source):
            return canonical[: -len(source)] + destination
    return canonical


def _canonical_target_only_weight_key(weight_key: str) -> str:
    """Convert one target-only parameter key to the model-tree spelling."""
    if weight_key in {"hc_head_fn", "hc_head_base", "hc_head_scale"}:
        return "model.hc_head." + weight_key.removeprefix("hc_head_")

    if weight_key.startswith("layers."):
        canonical = "model." + weight_key
    else:
        canonical = weight_key

    for source, destination in (
        (".hc_attn.fn", ".attn_hc.fn"),
        (".hc_attn.base", ".attn_hc.base"),
        (".hc_attn.scale", ".attn_hc.scale"),
        (".hc_ffn.fn", ".ffn_hc.fn"),
        (".hc_ffn.base", ".ffn_hc.base"),
        (".hc_ffn.scale", ".ffn_hc.scale"),
        (".hc_attn_fn", ".attn_hc.fn"),
        (".hc_attn_base", ".attn_hc.base"),
        (".hc_attn_scale", ".attn_hc.scale"),
        (".hc_ffn_fn", ".ffn_hc.fn"),
        (".hc_ffn_base", ".ffn_hc.base"),
        (".hc_ffn_scale", ".ffn_hc.scale"),
    ):
        if canonical.endswith(source):
            return canonical[: -len(source)] + destination

    if "." not in canonical:
        return _canonical_target_only_module_path(canonical)
    module_path, suffix = canonical.rsplit(".", 1)
    return f"{_canonical_target_only_module_path(module_path)}.{suffix}"


def _canonicalize_target_only_weights(weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
    """Canonicalize target-only checkpoint keys, rejecting alias collisions."""
    canonical_weights = {}
    sources = {}
    for source, value in weights.items():
        canonical = _canonical_target_only_weight_key(source)
        if canonical in canonical_weights:
            raise ValueError(
                "Target-only checkpoint aliases collide on "
                f"{canonical!r}: {sources[canonical]!r} and {source!r}."
            )
        canonical_weights[canonical] = value
        sources[canonical] = source
    return canonical_weights


def _target_only_quantized_modules(weights: Mapping[str, mx.array]) -> set[str]:
    """Inventory affine modules from every quantized tensor spelling."""
    modules = set()
    for key, value in weights.items():
        if key.endswith((".scales", ".biases")):
            modules.add(key.rsplit(".", 1)[0])
        elif key.endswith(".weight") and getattr(value, "dtype", None) == mx.uint32:
            modules.add(key.removesuffix(".weight"))
    return modules


def _validate_target_only_policy(module_path: str, policy: Any) -> dict[str, Any]:
    """Validate one affine target-only quantization policy without defaults."""
    if not isinstance(policy, dict) or not policy:
        raise ValueError(
            "Target-only quantization policy for "
            f"{module_path!r} must be a non-empty object."
        )
    if set(policy) != _QUANTIZATION_GLOBAL_KEYS:
        raise ValueError(
            "Target-only quantization policy for "
            f"{module_path!r} must contain exactly bits, group_size, and mode."
        )

    bits = policy["bits"]
    group_size = policy["group_size"]
    mode = policy["mode"]
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (2, 3, 4, 8):
        raise ValueError(
            f"Target-only quantization policy for {module_path!r} has invalid bits."
        )
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size <= 0
    ):
        raise ValueError(
            "Target-only quantization policy for "
            f"{module_path!r} has invalid group_size."
        )
    if mode != "affine":
        raise ValueError(
            "Target-only quantization policy for "
            f"{module_path!r} must use affine mode."
        )
    return policy


def _target_only_quantization_policies(
    config: Mapping[str, Any], weights: Mapping[str, mx.array]
) -> dict[str, dict[str, Any]]:
    """Build fail-closed policies for every scaled target-only module.

    There is deliberately no global-default fallback here.  The target-only
    converter stores per-module affine tensors and its manifest pins the
    layout, so a missing, malformed, or aliased policy would load a different
    model than the artifact declares.
    """
    quantization = config.get("quantization")
    if not isinstance(quantization, Mapping):
        raise ValueError("Target-only checkpoint requires a quantization object.")
    _validate_target_only_policy(
        "default",
        {key: quantization.get(key) for key in _QUANTIZATION_GLOBAL_KEYS},
    )

    policies = {}
    sources = {}
    for source, policy in quantization.items():
        if source in _QUANTIZATION_GLOBAL_KEYS:
            continue
        if not isinstance(source, str):
            raise ValueError("Target-only quantization policy names must be strings.")
        canonical = _canonical_target_only_module_path(source)
        if canonical in policies:
            raise ValueError(
                "Target-only quantization aliases collide on "
                f"{canonical!r}: {sources[canonical]!r} and {source!r}."
            )
        policies[canonical] = _validate_target_only_policy(canonical, policy)
        sources[canonical] = source

    quantized_modules = _target_only_quantized_modules(weights)
    for module_path in sorted(quantized_modules):
        if module_path not in policies:
            raise ValueError(
                "Target-only checkpoint has scaled weights for "
                f"{module_path!r} without an exact quantization policy."
            )
    if unexpected := set(policies) - quantized_modules:
        raise ValueError(
            "Target-only checkpoint has quantization policies without quantized "
            f"weights: {', '.join(sorted(unexpected))}."
        )
    return policies


def _validate_target_only_affine_triplets(
    weights: Mapping[str, mx.array], policies: Mapping[str, Mapping[str, Any]]
) -> None:
    """Validate every target-only affine tensor triplet before construction.

    The checkpoint tensors are the source of truth: every declared scaled
    module must carry ``weight``, ``scales``, and ``biases`` in the affine
    packed layout.  No model object is built until those raw checkpoint
    dtypes agree with the published affine layout.  The policy itself is
    trusted only because the raw config bytes are pinned above; we do not
    derive bitwidth or group size from tensor shape.
    """
    for module_path in policies:
        keys = {
            suffix: f"{module_path}.{suffix}"
            for suffix in ("weight", "scales", "biases")
        }
        missing = [suffix for suffix, key in keys.items() if key not in weights]
        if missing:
            raise ValueError(
                "Target-only affine tensor triplet for "
                f"{module_path!r} is missing {', '.join(missing)}."
            )

        weight = weights[keys["weight"]]
        scales = weights[keys["scales"]]
        biases = weights[keys["biases"]]
        if weight.dtype != mx.uint32:
            raise ValueError(
                "Target-only affine weight for "
                f"{module_path!r} must have uint32 packed dtype."
            )
        if scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
            raise ValueError(
                "Target-only affine scales and biases for "
                f"{module_path!r} must have bfloat16 dtype."
            )


def _deepseek_v4_quantization_policy(
    module_path: str, quantization: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Resolve non-target-only DeepSeek V4 checkpoint policy aliases.

    Strict target-only artifacts use ``_target_only_quantization_policies``
    instead.  This compatibility helper preserves the prior behavior for
    existing DeepSeek V4 checkpoints that do not declare the target-only ABI.

    Return the exact per-module policy when the checkpoint declares one, or
    ``None`` so the caller can use the checkpoint's global default.
    """
    candidates = [module_path]
    if module_path.startswith("model."):
        candidates.append(module_path.removeprefix("model."))

    if module_path == "model.embed_tokens":
        candidates.append("embed")
    elif module_path == "lm_head":
        candidates.append("head")

    if match := _ROUTED_EXPERT_MODULE_RE.fullmatch(module_path):
        layer_idx, projection = match.groups()
        candidates.append(
            f"layers.{layer_idx}.ffn.experts."
            f"{_EXPERT_PROJECTION_TO_CHECKPOINT[projection]}"
        )
    elif match := _SHARED_EXPERT_MODULE_RE.fullmatch(module_path):
        layer_idx, projection = match.groups()
        candidates.append(
            f"layers.{layer_idx}.ffn.shared_experts."
            f"{_EXPERT_PROJECTION_TO_CHECKPOINT[projection]}"
        )

    for candidate in candidates:
        policy = quantization.get(candidate)
        if isinstance(policy, dict):
            return policy
    return None


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
        trust_remote_code: bool = False,
    ) -> tuple[nn.Module, dict]:
        config_path = model_path / "config.json"
        config = _utils.load_config(model_path)
        target_only = _is_target_only_checkpoint(config)
        target_only_raw_config = None
        if target_only:
            # mlx-lm's production config wrapper expands quantization aliases
            # (e.g. language_model.*) after reading this file.  Runtime model
            # args use that expanded config, but target-only policy truth is
            # deliberately derived only from these identity-bound raw bytes.
            target_only_raw_config = _load_target_only_config_identity(config_path)
            if model_config is not None:
                raise ValueError(
                    "Target-only checkpoint does not accept model config overrides."
                )
        if model_config is not None:
            config.update(model_config)

        if (
            (model_file := config.get("model_file")) is not None
            and not trust_remote_code
        ):
            raise ValueError(
                f"The model at {model_path} requires executing custom model "
                f"code ({model_file!r}). Pass trust_remote_code=True if you "
                "trust this model."
            )

        weight_files = glob.glob(str(model_path / "model*.safetensors"))

        if not weight_files and strict:
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        weights = {}
        for wf in weight_files:
            weights.update(_load_safetensors(wf))  # PR 1192 change

        target_only_policies = None
        if target_only:
            # The target-only converter's names are an ABI, not a best-effort
            # compatibility hint.  Canonicalize once before sanitize() and
            # reject bad metadata before nn.quantize can allocate a model with
            # a fallback/default precision.  This applies even with
            # ``strict=False`` because strictness only controls final weight
            # matching, never artifact provenance.
            weights = _canonicalize_target_only_weights(weights)
            target_only_policies = _target_only_quantization_policies(
                target_only_raw_config, weights
            )
            _validate_target_only_affine_triplets(weights, target_only_policies)

        if model_file is not None:
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

        if str(config.get("model_type", "")).startswith("deepseek_v4"):
            config["use_native_ratio128_attention"] = bool(
                config.get("use_native_ratio128_attention", True)
            ) and _native_ratio128_attention_enabled(config)

        model_args = model_args_class.from_dict(config)
        model = model_class(model_args)

        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        def _quantize(quantization):
            def class_predicate(p, m):
                if target_only_policies is not None:
                    if p in target_only_policies:
                        return target_only_policies[p]
                elif str(config.get("model_type", "")).startswith("deepseek_v4"):
                    if policy := _deepseek_v4_quantization_policy(p, quantization):
                        return policy
                elif p in config["quantization"]:
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
            elif quant_method == "fp8" and str(config.get("model_type", "")).startswith(
                "deepseek_v4"
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
        model.load_weights(list(weights.items()), strict=True if target_only else strict)

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
