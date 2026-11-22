# SPDX-License-Identifier: Apache-2.0
"""Read an MLX checkpoint folder without a GPU library.

Only the safetensors headers and ``config.json`` are read, so the daemon can
plan which layers fit on which GPU before any worker starts.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class Tensor:
    file: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


def weight_files(model_dir: str | Path) -> list[str]:
    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.exists():
        return sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return ["model.safetensors"]


def tensors(model_dir: str | Path) -> dict[str, Tensor]:
    """Every tensor of the checkpoint, from the headers of the files present."""
    root = Path(model_dir)
    found: dict[str, Tensor] = {}
    for name in weight_files(root):
        path = root / name
        if not path.exists():
            # The host copies only the files a helper's layers need.
            continue
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        for key, spec in header.items():
            if key != "__metadata__":
                start, end = spec["data_offsets"]
                found[key] = Tensor(
                    name, spec["dtype"], tuple(spec["shape"]), end - start
                )
    return found


def layer_of(tensor: str) -> int | None:
    """Index of the transformer layer a tensor belongs to, if any."""
    match = _LAYER.search(tensor)
    return int(match.group(1)) if match else None


def memory_plan(
    model_dir: str | Path, resident: Callable[[str], bool]
) -> tuple[int, list[int]]:
    """(bytes outside the layers, bytes per layer) of the tensors a helper loads."""
    shared, per_layer = 0, {}
    for name, tensor in tensors(model_dir).items():
        if not resident(name):
            continue
        layer = layer_of(name)
        if layer is None:
            shared += tensor.nbytes
        else:
            per_layer[layer] = per_layer.get(layer, 0) + tensor.nbytes
    count = max(per_layer) + 1 if per_layer else 0
    return shared, [per_layer.get(i, 0) for i in range(count)]


def load_config(model_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / "config.json").read_text())


def quantization(config: dict[str, Any], module: str) -> dict[str, Any] | None:
    """Quantization settings of one module path, or None for a dense checkpoint.

    A per-module entry replaces the checkpoint default. Whether the module is
    quantized at all is decided by its tensors (a ``.scales`` tensor exists).
    """
    quant = config.get("quantization") or config.get("quantization_config")
    if not quant:
        return None
    override = quant.get(module)
    if isinstance(override, dict):
        return override
    return {k: v for k, v in quant.items() if not isinstance(v, dict)}


def packable(quant: dict[str, Any] | None) -> bool:
    """Whether a stage may keep a quantized linear packed: affine codes only.
    The floating point modes have no 16-bit scales to compute in."""
    return quant is not None and quant.get("mode", "affine") == "affine"


# oQh tensors (see ``oqh``): the input rotation signs name an oQh linear.
OQH_SIGNS = "oqh_signs"
OQH_VERSION = 1


def oqh_version(config: dict[str, Any]) -> int | None:
    """The oQh format version of a checkpoint, None when it has no oQh tensors."""
    block = config.get("omlx_oqh")
    if block is None:
        return None
    return int(block.get("version", 0)) if isinstance(block, dict) else 0
