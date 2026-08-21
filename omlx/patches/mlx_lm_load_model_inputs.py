# SPDX-License-Identifier: Apache-2.0
"""``load_model`` that accepts a pre-loaded config and weight dict.

mlx-lm's ``load_model`` reads ``config.json`` and globs ``model*.safetensors``
off the filesystem, so a caller already holding weights in memory has no way
in. A rank receiving a peer's weights over the cluster fabric holds exactly
that: the bits are resident, and spilling them to disk so the loader can read
them back again is the one thing it must not do.

This is mlx-lm issue #969 and its PR #1002, both closed unreviewed during the
2026-08-21 tracker cull, implemented against the pinned mlx-lm. Two deliberate
departures from PR #1002:

- ``config`` and ``weights`` are keyword-only. PR #1002 inserted them as
  positional parameters ahead of ``trust_remote_code``, which silently
  reassigns any caller passing ``get_model_classes`` or ``trust_remote_code``
  positionally.
- ``model_path`` accepts ``None``. Once a caller supplies both a config and a
  weight dict there is nothing left to read, and a receiving rank should not
  have to invent a directory to name. The paths that genuinely need it — the
  disk read and custom ``model_file`` resolution — check for it and say so.

``mlx_lm.utils.load_model`` is replaced rather than wrapped, because the disk
read sits in the middle of the construction pipeline with no seam around it.
The body mirrors pinned mlx-lm (``ab1806e``, v0.31.3). Architecture-specific
behaviour that would otherwise need a second copy of the whole function
reaches this one through ``register_config_transform`` and
``register_quant_method``, so the in-memory path and the disk path get
identical treatment.

KEEP: re-verify the body against ``mlx_lm.utils.load_model`` on every mlx-lm
pin bump.
"""

from __future__ import annotations

import contextlib
import glob
import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as _utils
from mlx.utils import tree_map

logger = logging.getLogger(__name__)

# Runs on the assembled config after ``model_config`` is merged and before the
# model args are built, so a transform can still steer construction.
ConfigTransform = Callable[[dict[str, Any]], None]

# Consulted for a ``quantization_config`` method mlx-lm has no branch for.
# Returns the quantization spec to apply, or None to decline.
QuantMethod = Callable[[nn.Module, dict[str, Any]], dict[str, Any] | None]

_CONFIG_TRANSFORMS: dict[str, ConfigTransform] = {}
_QUANT_METHODS: dict[str, QuantMethod] = {}

_INSTALLED = False


def register_config_transform(name: str, transform: ConfigTransform) -> None:
    """Register a config transform under ``name``, replacing any prior one.

    Transforms run in registration order regardless of where the config came
    from, which is the point: a config handed in by a caller gets the same
    normalisation as one read from ``config.json``.
    """

    _CONFIG_TRANSFORMS[name] = transform


def register_quant_method(method: str, handler: QuantMethod) -> None:
    """Register a handler for a ``quant_method`` mlx-lm does not know.

    mlx-lm's own branches stay authoritative; a handler is only consulted for
    a method none of them claim. Returning ``None`` declines, which leaves the
    model unquantized exactly as upstream does for an unrecognised method.
    """

    _QUANT_METHODS[method] = handler


def _read_weight_files(model_path: Path, strict: bool) -> dict[str, mx.array]:
    """Glob and load the checkpoint's safetensors shards.

    ``_load_safetensors`` is read off ``mlx_lm.utils`` at call time rather than
    bound at import: the DeepSeek V4 patch injects one there to reinterpret the
    F8_E8M0 block-scale dtype, and it may be installed after this module.
    """

    read = getattr(_utils, "_load_safetensors", None) or mx.load
    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files and strict:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(read(wf))
    return weights


def load_model(
    model_path: Path | None,
    lazy: bool = False,
    strict: bool = True,
    model_config: dict[str, Any] | None = None,
    get_model_classes: Callable[..., tuple[type, type]] | None = None,
    trust_remote_code: bool = False,
    *,
    config: dict[str, Any] | None = None,
    weights: dict[str, mx.array] | None = None,
) -> tuple[nn.Module, dict]:
    """Load and initialize the model from a path, or from memory.

    Args:
        model_path: The path to load the model from. May be ``None`` only when
            both ``config`` and ``weights`` are supplied and the config does
            not name a custom ``model_file``.
        lazy: If False eval the model parameters to make sure they are loaded
            in memory before returning, otherwise they will be loaded when
            needed. Default: ``False``
        strict: Whether or not to raise an exception if weights don't match.
            Default: ``True``
        model_config: Optional configuration parameters for the model, merged
            over the resolved config.
        get_model_classes: A function that returns the model class and model
            args class given a config. Defaults to ``mlx_lm.utils._get_classes``,
            resolved at call time.
        trust_remote_code: If ``True``, allow executing a custom model
            architecture file specified by the config's ``model_file`` key.
            Default: ``False``.
        config: A config to build from instead of reading ``config.json``.
            Modified in place during construction — quantization keys are
            added — so pass a copy to keep the caller's dict pristine.
        weights: A weight dict to build from instead of reading safetensors
            off disk. An empty dict is a valid answer and does not fall back
            to disk; it means the caller has no weights to bind, which
            ``strict=False`` then permits.

    Returns:
        The loaded and initialized model, and its config.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be
            instantiated, if the config requests a custom ``model_file`` and
            ``trust_remote_code`` is not enabled, or if ``model_path`` is
            ``None`` and something still needs to be read from disk.
    """
    if config is None:
        if model_path is None:
            raise ValueError(
                "load_model needs a model_path to read config.json from, or a "
                "config to build from."
            )
        config = _utils.load_config(Path(model_path))
    if model_config is not None:
        config.update(model_config)

    # Refuse a custom architecture before any weights are read. A checkpoint
    # that is about to be rejected should not first cost a multi-gigabyte read,
    # and an untrusted rank should not get that far.
    model_file = config.get("model_file")
    custom_arch_file: Path | None = None
    if model_file is not None:
        if not trust_remote_code:
            raise ValueError(
                f"The model at {model_path} requires importing and running a "
                f"custom module ({model_file!r}) to build its architecture. This "
                "is disabled by default. Pass trust_remote_code=True if you "
                "trust this model."
            )
        if model_path is None:
            raise ValueError(
                f"The config names a custom module ({model_file!r}) but no "
                "model_path was given to resolve it against."
            )
        custom_arch_file = Path(model_path) / model_file

    if weights is None:
        if model_path is None:
            raise ValueError(
                "load_model needs a model_path to read safetensors from, or a "
                "weights dict to build from."
            )
        weights = _read_weight_files(Path(model_path), strict)

    if custom_arch_file is not None:
        spec = importlib.util.spec_from_file_location(
            "custom_model",
            custom_arch_file,
        )
        arch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arch)
        model_class, model_args_class = arch.Model, arch.ModelArgs
    else:
        resolve_classes = get_model_classes or _utils._get_classes
        model_class, model_args_class = resolve_classes(config=config)

    if "quantization_config" not in config:
        text_config = config.get("text_config", {})
        if "quantization_config" in text_config:
            config["quantization_config"] = text_config["quantization_config"]

    for transform in tuple(_CONFIG_TRANSFORMS.values()):
        transform(config)

    model_args = model_args_class.from_dict(config)

    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    def _quantize(quantization):
        def class_predicate(p, m):
            # Handle custom per layer quantizations
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
        # Handle legacy quantization config
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
            # Transform AutoAWQ/GPTQ packed weights to MLX format
            weights, quantization = _utils._transform_awq_weights(
                weights, quantization_config
            )
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            _quantize(quantization)
        elif (handler := _QUANT_METHODS.get(quant_method)) is not None:
            quantization = handler(model, config)
            if quantization is not None:
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
                        "Linear layer with bias does not support activation "
                        "quantization"
                    )
                out_dims, in_dims = m.weight.shape
                in_dims *= 32 // m.bits
                return nn.QQLinear(in_dims, out_dims, m.group_size, m.bits, m.mode)
            return m

        leaves = tree_map(_maybe_qq, model.leaf_modules(), is_leaf=nn.Module.is_module)

        model.update_modules(leaves)

    model.eval()
    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        mx.eval(model.parameters())

    return model, config


load_model._omlx_load_model_inputs = True  # type: ignore[attr-defined]


def install_load_model_inputs() -> bool:
    """Replace ``mlx_lm.utils.load_model``. Idempotent.

    Also rebinds any ``mlx_lm`` module that captured the original through
    ``from .utils import load_model``, which an import-time binding would
    otherwise pin to the version this one replaced.
    """

    global _INSTALLED
    if _INSTALLED:
        return False

    _utils.load_model = load_model

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("mlx_lm"):
            continue
        if mod_name == "mlx_lm.utils":
            continue
        existing = getattr(mod, "load_model", None)
        if existing is not None and existing is not load_model:
            # A module can refuse the assignment (__slots__, a lazy-import
            # proxy); it just keeps the binding it already had.
            with contextlib.suppress(Exception):
                mod.load_model = load_model

    _INSTALLED = True
    logger.info(
        "mlx_lm.utils.load_model replaced (pre-loaded config/weights, mlx-lm #969)"
    )
    return True


def is_installed() -> bool:
    return _INSTALLED


__all__ = [
    "install_load_model_inputs",
    "is_installed",
    "load_model",
    "register_config_transform",
    "register_quant_method",
]
