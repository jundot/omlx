# SPDX-License-Identifier: Apache-2.0
"""Planning surface of the Qwen3.5 blocks.

The checkpoint layout is the same for the dense and MoE variants: the same
``language_model.`` tree holds the embedding and the layers, and the layer
kinds come from the same ``text_config``. A family's ``__init__`` re-exports
this module, so it must stay importable without a GPU library — it reads only
the safetensors headers and ``config.json``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .. import checkpoint

# This module must not import ``oqh``: that module imports mlx.core, and the
# daemon runs ``plan()`` in-process, so reaching it from here takes session
# start down on a box whose MLX is missing or broken.
#
# That is a standing constraint on future work, not a note about one commit.
# The planning surface can no longer use ``oqh`` at all, so when a later oQh
# change needs a fact here, the fact goes to ``checkpoint`` — which reads the
# headers without a GPU library — and the GPU-side code stays in ``oqh``.
# ``is_oqh`` and ``OQH_SIGNS`` moved there for exactly this reason. If someone
# re-adds the import instead, tests/test_gpu_free_planning.py refuses it, and
# that refusal is the intended behaviour rather than an obstacle to route
# around.

CONTRACT = 1
FIRST_STAGE_MIN = 1

# The expert projections a MoE layer banks. The loader joins the first two into
# one gathered bank and takes the third on its own; the plan refuses oQh signs
# on either, so both read this one list.
BANK_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def is_banked(name: str) -> bool:
    """Whether the loader banks this tensor: a MoE layer's expert projections."""
    return ".switch_mlp." in name and name.rpartition(".")[2] in BANK_PROJECTIONS


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
        if base == name or layer is None or not resident(name):
            continue
        # An expert bank is 3-D [experts, out, in]; unpacking one to dense
        # 16-bit would multiply the memory estimate by the expert count.
        if len(weight.shape) != 2:
            # The loader refuses an oQh bank and only ever banks the expert
            # projections, so the estimate asks its question the same way: a
            # signed tensor the loader never banks is skipped like any other.
            if is_banked(base) and checkpoint.is_oqh(index, base):
                raise ValueError(f"oQh expert banks are not supported: {name}")
            continue
        signs = index.get(f"{base}.{checkpoint.OQH_SIGNS}")
        if signs is not None:
            # oQh: dense weight [N, K]; only the signs stay next to it.
            parts = ("rscale", "oqh_scales", "oqh_biases")
            packed = [t for p in parts if (t := index.get(f"{base}.{p}")) is not None]
            dense = weight.shape[0] * math.prod(signs.shape) * 2
            layers[layer] += dense - weight.nbytes - sum(t.nbytes for t in packed)
            continue
        packed = [index.get(f"{base}.{part}") for part in ("scales", "biases")]
        if None in packed:
            continue
        bits = checkpoint.quantization(config, base)["bits"]
        dense = weight.shape[0] * (weight.shape[1] * 32 // bits) * 2
        layers[layer] += dense - weight.nbytes - sum(t.nbytes for t in packed)
    return shared, layers
