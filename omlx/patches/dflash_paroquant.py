# SPDX-License-Identifier: Apache-2.0
"""Load ParoQuant targets without replacing their rotation-aware projections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def is_paroquant_config(config: dict) -> bool:
    quant = config.get("quantization_config") or {}
    return (
        isinstance(quant, dict)
        and str(quant.get("quant_method", "")).lower() == "paroquant"
    )


def paroquant_dflash_compatibility(config: dict) -> tuple[bool, str]:
    """Keep initial support restricted to the dense Qwen3.8-27B layout."""
    text = config.get("text_config") or config
    quant = config.get("quantization_config") or {}
    if (
        config.get("model_type") != "qwen3_5"
        or not isinstance(text, dict)
        or text.get("num_hidden_layers") != 64
        or text.get("hidden_size") != 5120
        or text.get("vocab_size") != 248320
        or text.get("num_experts", 0) not in (0, None)
        or quant.get("bits") != 4
        or quant.get("group_size") != 128
        or quant.get("krot") != 8
    ):
        return (
            False,
            "ParoQuant DFlash currently supports only the dense Qwen3.8-27B 4-bit, group-128, krot-8 layout",
        )
    return True, ""


def load_target_bundle(model_ref: str | Path, **kwargs: Any) -> Any:
    """Dispatch locally registered ParoQuant checkpoints to their own loader.

    Ordinary targets retain dflash-mlx's loader and verification optimizations.
    ParoQuant's rotation modules must execute their original forward methods;
    the first supported path deliberately skips target verify-linear rewrites.
    """
    from dflash_mlx.runtime.loading import load_target_bundle as standard_load

    config_path = Path(model_ref) / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if not is_paroquant_config(config):
        return standard_load(model_ref, **kwargs)

    compatible, reason = paroquant_dflash_compatibility(config)
    if not compatible:
        raise ValueError(reason)
    try:
        from paroquant.inference.backends.mlx.load import load
    except ImportError as exc:
        raise ImportError(
            'ParoQuant DFlash requires ParoQuant; install "omlx[paroquant]"'
        ) from exc

    from dflash_mlx.engine.target_ops import resolve_target_ops
    from dflash_mlx.runtime.loading import LoadedTargetBundle

    model, tokenizer, is_vlm = load(
        str(model_ref), lazy=kwargs.get("lazy", True), force_text=True
    )
    if is_vlm:
        raise ValueError("ParoQuant DFlash requires a text-only target")
    ops = resolve_target_ops(model)
    ops.install_speculative_hooks(model)
    return LoadedTargetBundle(
        model=model,
        tokenizer=tokenizer,
        target_ops=ops,
        meta={
            "resolved_model_ref": str(model_ref),
            "config": config,
            "quantize_kv_cache": bool(kwargs.get("quantize_kv_cache", False)),
            "target_family": ops.family(model),
            "quant_method": "paroquant",
            "verify_linear_enabled": False,
            "verify_linear_swapped": 0,
            "verify_mode": "off",
        },
    )


def validate_paroquant_draft(target_meta: dict, draft_meta: dict) -> None:
    """Reject incompatible drafts before their first projection or cache update."""
    if target_meta.get("quant_method") != "paroquant":
        return
    target = target_meta["config"]
    text = target.get("text_config") or target
    draft = draft_meta.get("config") or {}
    spec = draft.get("dflash_config") or {}
    layers = spec.get("target_layer_ids") or []
    if (
        draft.get("hidden_size") != text["hidden_size"]
        or draft.get("vocab_size") != text["vocab_size"]
        or draft.get("num_target_layers", spec.get("num_target_layers"))
        != text["num_hidden_layers"]
        or not layers
        or any(
            type(i) is not int or not 0 <= i < text["num_hidden_layers"] for i in layers
        )
    ):
        raise ValueError(
            "DFlash draft dimensions or capture layers do not match the ParoQuant target"
        )
