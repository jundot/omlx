# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 Flash-Next (model type ``qwen4_exp``): planning without a GPU library.

The n-gram embedding (PLE) table stays on the host, which sends each chunk's rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ... import checkpoint

CONTRACT = 1
# The embedding and the PLE layer stay on the first stage: only that stage
# takes the host-fed PLE rows and keeps the token history.
FIRST_STAGE_MIN = 2

_NOT_LOADED = (".mtp.", ".ple_embedding.", ".lm_head.", ".hyper_connection_mixer.")


def resident(tensor: str) -> bool:
    """Whether a helper loads this tensor: language layers and the embedding."""
    name = tensor.removeprefix("model.")
    return name.startswith("language_model.") and not any(
        part in name for part in _NOT_LOADED
    )


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") or config


def layer_kinds(text: dict[str, Any]) -> list[str]:
    """``"linear"`` or ``"qsa"`` for each layer."""
    kinds = text.get("layer_types")
    if kinds is None:
        interval = text.get("full_attention_interval", 4)
        return [
            "linear" if (i + 1) % interval else "qsa"
            for i in range(text["num_hidden_layers"])
        ]
    return ["linear" if kind == "linear_attention" else "qsa" for kind in kinds]


def ple_layers(text: dict[str, Any]) -> list[int]:
    """Zero-based layers with a PLE block (the config counts from one)."""
    return sorted({int(i) - 1 for i in text.get("ple_layer_ids") or []})


def check_first_stage(text: dict[str, Any]) -> None:
    """Every PLE layer must sit below the first stage minimum."""
    late = [i for i in ple_layers(text) if i >= FIRST_STAGE_MIN]
    if late:
        raise ValueError(
            f"PLE layers {late} do not fit in a first stage of "
            f"{FIRST_STAGE_MIN} layers"
        )


def plan(model_dir: str | Path) -> tuple[int, list[int]]:
    """(bytes outside the layers, bytes per layer) a helper holds."""
    config = checkpoint.load_config(model_dir)
    check_first_stage(text_config(config))
    shared, layers = checkpoint.memory_plan(model_dir, resident)
    # A stage unpacks the linears of a layer (not its expert banks) to dense
    # 16-bit weights, which takes more than their packed tensors.
    index = checkpoint.tensors(model_dir)
    for name, weight in index.items():
        base = name.removesuffix(".weight")
        layer = checkpoint.layer_of(name)
        packed = [index.get(f"{base}.{part}") for part in ("scales", "biases")]
        if (
            base == name
            or layer is None
            or len(weight.shape) != 2
            or None in packed
            or not resident(name)
        ):
            continue
        bits = checkpoint.quantization(config, base)["bits"]
        dense = weight.shape[0] * (weight.shape[1] * 32 // bits) * 2
        layers[layer] += dense - weight.nbytes - sum(t.nbytes for t in packed)
    return shared, layers
