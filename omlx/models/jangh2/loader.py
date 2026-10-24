"""Strict JANGH2 loading through oMLX's ordinary VLM engine.

Packed experts are replaced before affine quantization. The existing Qwen4
architecture owns PLE storage, cache classes, fused projections and teardown;
this loader does not patch scheduler, stream, chat or upstream loader globals.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from .checkpoint import Checkpoint, inspect_checkpoint
from .metadata import (
    canonical_map,
    choose_ple_prefix,
    is_mtp,
    is_ple_storage,
    ple_layer_indices,
    reconcile_ple_module_parameters,
    safe_shard_path,
)
from .performance import PerformanceOptions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadOptions:
    """Model-local execution choices; environment variables are not consulted."""

    backend: Literal["metal", "reference-fast"] = "metal"
    performance: PerformanceOptions = field(default_factory=PerformanceOptions)

    def __post_init__(self):
        if self.backend not in ("metal", "reference-fast"):
            raise ValueError("Native JANGH2 backend must be metal or reference-fast")
        if not isinstance(self.performance, PerformanceOptions):
            raise TypeError("performance must be PerformanceOptions")
        if self.performance.prefill != "legacy" or self.performance.ple_device_mib:
            raise ValueError(
                "Native JANGH2 grouped prefill/device caching requires separate qualification"
            )


def compare_parameters(
    expected: Mapping[str, Any], supplied: Mapping[str, Any]
) -> None:
    """Require exact parameter names and shapes before model transformations."""
    missing, extra = (
        sorted(set(expected) - set(supplied)),
        sorted(set(supplied) - set(expected)),
    )
    bad = sorted(
        (key, tuple(expected[key].shape), tuple(supplied[key].shape))
        for key in set(expected) & set(supplied)
        if tuple(expected[key].shape) != tuple(supplied[key].shape)
    )
    if missing or extra or bad:
        raise ValueError(
            "Strict JANGH2 parameter audit failed: "
            f"missing={missing[:12]}, unexpected={extra[:12]}, shape_mismatches={bad[:8]}"
        )


def recenter_residual_norms(model, weights, norm_class, xp) -> int:
    """Apply the declared direct-gamma convention without a statistical guess."""
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, norm_class):
            continue
        key = name + ".weight"
        if key not in weights:
            raise ValueError(f"Missing JANGH2 residual RMSNorm tensor: {key}")
        value = weights[key]
        if not xp.issubdtype(value.dtype, xp.floating):
            raise ValueError(f"Non-floating JANGH2 residual RMSNorm tensor: {key}")
        residual = value.astype(xp.float32) - 1.0
        narrow = residual.astype(value.dtype)
        exact = xp.array_equal(narrow.astype(xp.float32), residual)
        weights[key] = (
            narrow
            if bool(exact.item() if hasattr(exact, "item") else exact)
            else residual
        )
        count += 1
    return count


def _model_config(checkpoint: Checkpoint) -> dict[str, Any]:
    config = copy.deepcopy(checkpoint.config)
    declaration = config.get("jang_config")
    if (
        not isinstance(declaration, Mapping)
        or declaration.get("norm_convention") != "runtime_plus1_applied"
    ):
        raise ValueError(
            "JANGH2 requires explicit jang_config.norm_convention=runtime_plus1_applied"
        )
    text = config["text_config"]
    if text.get("tie_word_embeddings", False):
        raise ValueError("JANGH2 tied embeddings are not supported")
    if text.get("ple_storage"):
        raise ValueError(
            "JANGH2 indexed PLE cannot be combined with an external PLE manifest"
        )
    indices = ple_layer_indices(
        text.get("ple_layer_ids", []), text.get("num_hidden_layers")
    )
    text["ple_storage_mode"] = "mmap"
    text["ple_model_path"] = str(checkpoint.root)
    text["ple_tensor_prefixes"] = {
        str(index): choose_ple_prefix(
            f"model.language_model.layers.{index}.ple.ple_embedding.ngram_embedding",
            checkpoint.weight_map,
        )
        for index in indices
    }
    config["mtp_enabled"] = False
    return config


def _runtime_config(checkpoint, utils):
    """Merge ordinary generation sidecars without replacing audited structure."""
    config = _model_config(checkpoint)
    defaults = utils.load_config(checkpoint.root)
    for key in (*utils.GENERATION_CONFIG_DEFAULT_KEYS, "generation_config"):
        if key in defaults:
            config[key] = copy.deepcopy(defaults[key])
    return config


def _resident_weights(checkpoint, mx):
    by_file: dict[str, list[str]] = {}
    for key, filename in checkpoint.weight_map.items():
        if not is_ple_storage(key) and not is_mtp(key):
            by_file.setdefault(filename, []).append(key)
    weights = {}
    for filename, keys in sorted(by_file.items()):
        # Loading a mixed shard is lazy. Only declared resident tensors are
        # retained; the SSD PLE payload never enters model.parameters().
        arrays = mx.load(str(safe_shard_path(checkpoint.root, filename)))
        for key in keys:
            if key not in arrays:
                raise ValueError(f"Indexed JANGH2 tensor is missing: {filename}:{key}")
            weights[key] = arrays[key]
        del arrays
    return canonical_map(weights)


def _sanitize(model, weights, language, config, mx, tree_flatten):
    # Repair only the checkpoint's verified PLE aliases and singleton Conv1d
    # channel before the current vendor sanitizer inspects their live shapes.
    expected = dict(tree_flatten(model.parameters()))
    weights, _ = reconcile_ple_module_parameters(weights, expected)
    recentered = recenter_residual_norms(model, weights, language.Qwen4ExpRMSNorm, mx)
    model._omlx_explicit_rmsnorm_residual = True
    before = set(weights)
    weights = model.sanitize(weights)
    vision = getattr(model, "vision_tower", None)
    if vision is not None and hasattr(vision, "sanitize"):
        weights = vision.sanitize(weights)
    weights = canonical_map(weights)
    lost = before - set(weights)
    allowed = {key for key in lost if "position_ids" in key or is_mtp(key)}
    if lost - allowed:
        raise ValueError(
            f"JANGH2 sanitizer dropped checkpoint tensors: {sorted(lost - allowed)[:12]}"
        )
    logger.info("JANGH2: explicitly recentered %d residual RMSNorm tensors", recentered)
    return weights


def _load_model(checkpoint, options, *, lazy):
    import mlx.core as mx
    import mlx_vlm.utils as utils
    from mlx.utils import tree_flatten

    from ...patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch
    from .experts import quantize_affine, replace_experts, validate_loaded_experts
    from .ple import PLEState

    config = _runtime_config(checkpoint, utils)
    apply_mlx_vlm_qwen4_exp_compat_patch()
    architecture = import_module("mlx_vlm.models.qwen4_exp")
    language = import_module("mlx_vlm.models.qwen4_exp.language")
    model_config = architecture.ModelConfig.from_dict(config)
    model_config = utils.update_module_configs(
        model_config,
        architecture,
        config,
        ["text", "vision", "perceiver", "projector", "audio"],
    )
    model_config = utils.apply_generation_config_defaults(model_config, config)
    model = None
    try:
        model = architecture.Model(model_config)
        replacements = replace_experts(
            model,
            config,
            backend=options.backend,
            chunk_size=8,
            options=options.performance,
        )
        weights = _resident_weights(checkpoint, mx)
        weights = _sanitize(model, weights, language, config, mx, tree_flatten)
        validate_loaded_experts(model, weights)
        quantize_affine(model, config, weights)
        compare_parameters(dict(tree_flatten(model.parameters())), weights)
        # Preserve the current architecture's exact HC fusion and compiled
        # paths. Its load_weights override runs after the strict raw audit.
        model.load_weights(list(weights.items()), strict=True)
        del weights
        if not lazy:
            mx.eval(model.parameters())
        model.model_path = checkpoint.root
        model._jangh2_backend = options.backend
        model._jangh2_bridge_model = True
        model._jangh2_ple_state = PLEState("native-mmap")
        model.eval()
        logger.info(
            "JANGH2: strict native load complete; %d %s expert blocks",
            len(replacements),
            options.backend,
        )
        return model
    except BaseException as exc:
        _close_on_failure(model, exc)
        raise


def _close_on_failure(model, primary):
    if model is not None:
        try:
            model.close()
        except BaseException as cleanup:
            primary.add_note(f"JANGH2 model cleanup failed: {type(cleanup).__name__}")


def load_model(
    model_path: str | Path, *, options: LoadOptions | None = None, lazy: bool = True
):
    """Load only the model; the normal VLM engine materializes its lazy state."""
    from ...patches.mlx_vlm_qwen4_exp_compat.ple_load_resources import (
        ple_load_resources,
    )

    options = options or LoadOptions()
    checkpoint = inspect_checkpoint(model_path)
    with ple_load_resources():
        return _load_model(checkpoint, options, lazy=lazy)


def load(
    model_path: str | Path,
    *,
    options: LoadOptions | None = None,
    trust_remote_code: bool = False,
    lazy: bool = True,
):
    """Return the model and processor without replacing mlx-vlm.load_model."""
    import mlx_vlm.utils as utils

    from ...patches.mlx_vlm_qwen4_exp_compat.ple_load_resources import (
        ple_load_resources,
    )

    options = options or LoadOptions()
    checkpoint = inspect_checkpoint(model_path)
    model = None
    with ple_load_resources():
        try:
            model = _load_model(checkpoint, options, lazy=lazy)
            processor = utils.load_processor(
                checkpoint.root,
                True,
                eos_token_ids=getattr(model.config, "eos_token_id", None),
                trust_remote_code=trust_remote_code,
            )
            image_processor = utils.load_image_processor(
                checkpoint.root, trust_remote_code=trust_remote_code
            )
            if image_processor is not None:
                processor.image_processor = image_processor
            return model, processor
        except BaseException as exc:
            _close_on_failure(model, exc)
            raise
