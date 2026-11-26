# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 (inkling2): the GPU-free part.

A cut sends the partial block sum and the inputs of the block boundary layers
before it, and every layer keeps its own rows, so a stage can start at any
layer. Stages read the published checkpoint (``model.llm.*``) and the runtime
layout oQ writes (``model.*``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ... import checkpoint

CONTRACT = 1
FIRST_STAGE_MIN = 1

SOURCE_ROOT = "model.llm."
_ROOTS = (SOURCE_ROOT, "model.")
_SHARED = ("embed.", "embed_norm.")
_LAYER = re.compile(r"^layers\.(\d+)\.")
# Source attention-residual queries: attn_queries[i] belongs to layer i + 1.
_QUERY = re.compile(r"^attn_res\.(attn|mlp)_queries\.(\d+)$")


def local_name(tensor: str) -> str:
    """A tensor name without its model root."""
    for root in _ROOTS:
        if tensor.startswith(root):
            return tensor[len(root) :]
    return tensor


def owner(tensor: str) -> int | None | bool:
    """Layer of a tensor a stage loads, None for the embedding, else False."""
    name = local_name(tensor)
    if name.startswith(_SHARED):
        return None
    if match := _LAYER.match(name):
        return int(match.group(1))
    if match := _QUERY.match(name):
        return int(match.group(2)) + (match.group(1) == "attn")
    return False


def resident(tensor: str) -> bool:
    """Whether a stage loads this tensor: the layers and the embedding."""
    return owner(tensor) is not False


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") or config


def is_source(index: dict[str, Any]) -> bool:
    """Whether the checkpoint keeps the published names (``model.llm.*``)."""
    return any(name.startswith(SOURCE_ROOT) for name in index)


def _dense_extra(config: dict[str, Any], index: dict[str, Any], name: str) -> int:
    """Bytes a quantized dense linear gains when a stage unpacks it to 16 bits.

    Routed expert banks and the embedding stay packed.
    """
    base = name.removesuffix(".weight")
    if base == name or ".experts." in name or local_name(name).startswith("embed."):
        return 0
    packed = [index.get(f"{base}.{part}") for part in ("scales", "biases")]
    if packed[0] is None:
        return 0
    weight = index[name]
    bits = checkpoint.quantization(config, base)["bits"]
    dense = weight.shape[0] * (weight.shape[1] * 32 // bits) * 2
    return dense - weight.nbytes - sum(t.nbytes for t in packed if t is not None)


def plan(model_dir: str | Path) -> tuple[int, list[int]]:
    """(bytes outside the layers, bytes per layer) a helper holds."""
    config = checkpoint.load_config(model_dir)
    index = checkpoint.tensors(model_dir)
    count = text_config(config)["num_hidden_layers"]
    shared, layers = 0, [0] * count
    for name, tensor in index.items():
        layer = owner(name)
        if layer is False:
            continue
        size = tensor.nbytes + _dense_extra(config, index, name)
        if layer is None:
            shared += size
        elif layer < count:
            layers[layer] += size
    return shared, layers
