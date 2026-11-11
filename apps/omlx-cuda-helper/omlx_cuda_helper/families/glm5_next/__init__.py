# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 (glm5_next): KDA and sparse MLA layers on hyper-connection streams.

Only the streams ``h`` [B, L, hc_mult, D] cross a cut, and every layer keeps
its own state, so a stage continues from ``h`` alone at any layer.
"""

from __future__ import annotations

from pathlib import Path

from ... import checkpoint

CONTRACT = 1
FIRST_STAGE_MIN = 1

_ROOTS = ("language_model.model.", "model.language_model.")
# Tensors that stay packed: the expert banks and the embedding table.
_PACKED = (".mlp.switch_mlp.", ".mlp.experts.", "embed_tokens.")
_FLOAT_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


def root(tensor: str) -> str | None:
    """Checkpoint prefix of a language-model tensor: MLX or Hugging Face."""
    return next((r for r in _ROOTS if tensor.startswith(r)), None)


def resident(tensor: str) -> bool:
    """Whether a stage loads this tensor: the layers and the embedding."""
    prefix = root(tensor)
    if prefix is None or ".mtp." in tensor:
        return False
    return tensor[len(prefix) :].startswith(("layers.", "embed_tokens."))


def plan(model_dir: str | Path) -> tuple[int, list[int]]:
    """(bytes outside the layers, bytes per layer) a helper holds on the GPU."""
    found = checkpoint.tensors(model_dir)
    config = checkpoint.load_config(model_dir)

    def unpacked_bytes(name: str) -> int | None:
        """Size after unpacking of a packed linear's tensor, else None."""
        stem, _, leaf = name.rpartition(".")
        scales = found.get(f"{stem}.scales")
        if scales is None or any(mark in name for mark in _PACKED):
            return None
        quant = checkpoint.quantization(config, stem) or {}
        if leaf != "weight":
            return 0
        weights = found[name].nbytes * 8 // quant.get("bits", 4)
        return weights * _FLOAT_BYTES.get(scales.dtype, 4)

    def loaded(name: str) -> bool:
        return resident(name) and unpacked_bytes(name) != 0

    shared, per_layer = checkpoint.memory_plan(model_dir, loaded)
    for name in found:
        size = unpacked_bytes(name) if resident(name) else None
        if size:
            per_layer[checkpoint.layer_of(name)] += size - found[name].nbytes
    return shared, per_layer
