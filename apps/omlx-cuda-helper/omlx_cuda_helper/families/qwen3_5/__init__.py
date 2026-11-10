# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 / 3.6 / 3.8 dense (``qwen3_5``): planning without a GPU library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ... import checkpoint

CONTRACT = 1
FIRST_STAGE_MIN = 1


def resident(tensor: str) -> bool:
    """Whether a helper loads this tensor: the embedding and the layers."""
    name = tensor.removeprefix("model.")
    return (
        name.startswith("language_model.")
        and ".mtp." not in name
        and (".layers." in name or ".embed_tokens." in name)
    )


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") or config


def linear_layers(text: dict[str, Any]) -> list[bool]:
    """Whether each layer is a gated delta net; the others are attention."""
    kinds = text.get("layer_types")
    if kinds:
        return [kind == "linear_attention" for kind in kinds]
    interval = text.get("full_attention_interval", 4)
    return [(i + 1) % interval != 0 for i in range(text["num_hidden_layers"])]


def check_layout(tensors: dict[str, Any]) -> None:
    """Refuse a checkpoint that is not in the MLX layout.

    A Hugging Face checkpoint keeps its conv taps on another axis and its
    norm gains minus one; the host converts them while it loads.
    """
    if not any(name.startswith("language_model.model.layers.") for name in tensors):
        raise ValueError("Only checkpoints converted to the MLX layout are supported")


def plan(model_dir: str | Path) -> tuple[int, list[int]]:
    """(bytes outside the layers, bytes per layer) a helper holds."""
    config = checkpoint.load_config(model_dir)
    check_layout(checkpoint.tensors(model_dir))
    shared, layers = checkpoint.memory_plan(model_dir, resident)
    # A stage unpacks every linear of a layer to a dense 16-bit weight, which
    # takes more than its packed tensors. The embedding stays packed.
    index = checkpoint.tensors(model_dir)
    for name, weight in index.items():
        base = name.removesuffix(".weight")
        layer = checkpoint.layer_of(name)
        packed = [index.get(f"{base}.{part}") for part in ("scales", "biases")]
        if base == name or layer is None or None in packed or not resident(name):
            continue
        bits = checkpoint.quantization(config, base)["bits"]
        dense = weight.shape[0] * (weight.shape[1] * 32 // bits) * 2
        layers[layer] += dense - weight.nbytes - sum(t.nbytes for t in packed)
    return shared, layers
