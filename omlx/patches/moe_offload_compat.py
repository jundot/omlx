# SPDX-License-Identifier: Apache-2.0
"""Header-only eligibility checks for the experimental expert offload setting."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_SUPPORTED_TYPES = frozenset(
    {
        "deepseek_v41",
        "deepseek_v4",
        "qwen4_exp",
        "qwen3_5_moe",
        "gemma4",
        "olmoe",
        "glm_moe_dsa",
        "glm5_next",
    }
)


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

    from .moe_expert_offload import CheckpointExpertStore, _qwen35_checkpoint_prefix

    text = raw.get("text_config", raw)
    if kind in ("glm_moe_dsa", "deepseek_v4", "glm5_next"):
        count = int(text.get("n_routed_experts") or 0)
        first_moe = int(text.get("first_k_dense_replace") or 0)
        moe_freq = int(text.get("moe_layer_freq") or 1)
    else:
        count = int(text.get("num_experts") or 0)
        first_moe, moe_freq = 0, 1
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
    # glm5_next picks sparse layers per layer, not by frequency.
    sparse_layers = None
    if kind == "glm5_next":
        types = text.get("mlp_layer_types")
        if not isinstance(types, list) or len(types) != layers:
            return False, "The model does not have the supported MoE geometry."
        sparse_layers = {i for i, t in enumerate(types) if t == "sparse"}
        if not sparse_layers:
            return False, "The model does not have the supported MoE geometry."
    quant = raw.get("quantization", text.get("quantization"))
    if not isinstance(quant, dict):
        return False, "Expert offload requires an MLX quantized checkpoint."
    store = CheckpointExpertStore(path)
    if min(layers - first_moe, moe_freq) <= 0:
        return False, "The model does not have the supported MoE geometry."
    for layer in range(layers):
        if layer < first_moe or layer % moe_freq:
            continue  # dense layer (GLM's first_k_dense_replace)
        if sparse_layers is not None and layer not in sparse_layers:
            continue  # dense layer (glm5_next's mlp_layer_types)
        if kind in ("olmoe", "glm_moe_dsa"):
            parent = f"model.layers.{layer}.mlp"
            prefix = parent + ".switch_mlp"
        elif kind in ("qwen4_exp", "qwen3_5_moe"):
            parent = f"language_model.model.layers.{layer}.mlp"
            prefix = parent + ".switch_mlp"
        elif kind == "deepseek_v4":
            parent = f"model.layers.{layer}.ffn"
            prefix = parent + ".switch_mlp"
        elif kind == "glm5_next":
            parent = f"language_model.model.layers.{layer}.mlp"
            prefix = parent + ".switch_mlp"
        else:
            parent = f"language_model.model.layers.{layer}.experts"
            prefix = parent + ".switch_glu"
        if kind == "qwen3_5_moe":
            prefix = _qwen35_checkpoint_prefix(store, prefix)
        if kind in ("deepseek_v4", "glm5_next", "qwen3_5_moe") and not store.has(
            prefix + ".gate_proj.weight"
        ):
            # These adapters read the stacked slabs positionally; a
            # per-expert layout wraps nothing.
            return (
                False,
                f"Checkpoint is missing expert tensor: {prefix}.gate_proj.weight",
            )
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
