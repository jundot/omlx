# SPDX-License-Identifier: Apache-2.0
"""Header-only eligibility checks for the experimental expert offload setting."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_SUPPORTED_TYPES = frozenset({"deepseek_v41", "qwen4_exp", "gemma4", "olmoe"})


def moe_offload_compatibility(model_path):
    """Return eligibility and a reason without loading any model tensors."""
    try:
        path = Path(model_path).expanduser().resolve()
        config = path / "config.json"
        raw = json.loads(config.read_text())
        if raw.get("model_type") not in _SUPPORTED_TYPES:
            return False, "MoE expert offload is not supported for this model type."
        files = [config, *path.glob("*.safetensors")]
        index = path / "model.safetensors.index.json"
        if index.exists():
            files.append(index)
        signature = tuple(
            (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(files)
        )
        return _inspect(str(path), signature)
    except (OSError, TypeError, ValueError, KeyError):
        return False, "Could not verify the expert checkpoint layout."


@lru_cache(maxsize=128)
def _inspect(path, signature):
    raw = json.loads((Path(path) / "config.json").read_text())
    kind = raw["model_type"]
    if kind == "deepseek_v41":
        from .deepseek_v41.moe_offload import estimate_expert_savings

        if estimate_expert_savings(path, 0.125) > 0:
            return True, ""
        return False, "The checkpoint has no offloadable routed experts."

    from .moe_expert_offload import CheckpointExpertStore

    text = raw.get("text_config", raw)
    count = int(text.get("num_experts") or 0)
    layers = int(text.get("num_hidden_layers") or 0)
    hidden = int(text.get("hidden_size") or 0)
    intermediate = int(
        text.get("intermediate_size" if kind == "olmoe" else "moe_intermediate_size")
        or 0
    )
    if min(count, layers, hidden, intermediate) <= 0 or (
        kind == "gemma4" and not text.get("enable_moe_block")
    ):
        return False, "The model does not have the supported MoE geometry."
    quant = raw.get("quantization", text.get("quantization"))
    if not isinstance(quant, dict):
        return False, "Expert offload requires an MLX quantized checkpoint."
    store = CheckpointExpertStore(path)
    for layer in range(layers):
        if kind == "olmoe":
            parent = f"model.layers.{layer}.mlp"
            prefix = parent + ".switch_mlp"
        elif kind == "qwen4_exp":
            parent = f"language_model.model.layers.{layer}.mlp"
            prefix = parent + ".switch_mlp"
        else:
            parent = f"language_model.model.layers.{layer}.experts"
            prefix = parent + ".switch_glu"
        per_expert = not store.has(prefix + ".gate_proj.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = prefix + "." + proj
            spec = quant.get(key, quant)
            if not isinstance(spec, dict):
                return False, f"Unsupported expert quantization: {key}"
            bits = spec.get("bits", 4)
            group = spec.get("group_size", 64)
            mode = spec.get("mode", "affine")
            if mode not in ("affine", "mxfp4", "mxfp8") or bits not in (
                2,
                3,
                4,
                5,
                6,
                8,
            ):
                return False, f"Unsupported expert quantization: {key}"
            output, width = (
                (hidden, intermediate)
                if proj == "down_proj"
                else (intermediate, hidden)
            )
            if (
                not isinstance(group, int)
                or group <= 0
                or width % group
                or width * bits % 32
            ):
                return False, f"Unsupported expert packing: {key}"
            fields = (
                ("weight", "scales", "biases")
                if mode == "affine"
                else ("weight", "scales")
            )
            for expert in range(count) if per_expert else (None,):
                base = f"{parent}.experts.{expert}.{proj}" if per_expert else key
                if store.has(base + ".bias"):
                    return False, "Per-expert linear bias is not supported."
                for field in fields:
                    name = base + "." + field
                    shape = (
                        output,
                        width * bits // 32 if field == "weight" else width // group,
                    )
                    if not per_expert:
                        shape = (count, *shape)
                    dtypes = (
                        {"U32"}
                        if field == "weight"
                        else ({"F16", "BF16", "F32"} if mode == "affine" else {"U8"})
                    )
                    if not store.has(name):
                        return False, f"Checkpoint is missing expert tensor: {name}"
                    actual_shape, dtype = store.spec(name)
                    if actual_shape != shape or dtype not in dtypes:
                        return (
                            False,
                            f"Unsupported expert tensor shape or dtype: {name}",
                        )
    return True, ""
