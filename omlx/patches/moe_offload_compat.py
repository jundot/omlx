# SPDX-License-Identifier: Apache-2.0
"""Header-only eligibility checks for the experimental expert offload setting."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# Expert layouts the common adapter can read, as the layer-path template and the
# projection prefix beneath it. Which one applies is decided by the checkpoint's
# own tensors, so a family does not have to be listed anywhere to be supported:
# the per-tensor validation below is the gate, and it is stricter than a list of
# model types could be.
_LAYER_LAYOUTS = (
    ("model.layers.{layer}.mlp", ".switch_mlp"),
    ("language_model.model.layers.{layer}.mlp", ".switch_mlp"),
    ("language_model.model.layers.{layer}.experts", ".switch_glu"),
)


def moe_offload_compatibility(model_path):
    """Return eligibility and a reason without loading any model tensors."""
    try:
        path = Path(model_path).expanduser().resolve()
        config = path / "config.json"
        raw = json.loads(config.read_text())
        if not isinstance(raw, dict) or not raw.get("model_type"):
            return False, "The checkpoint declares no model type."
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
    if raw.get("model_type") == "deepseek_v41":
        from .deepseek_v41.moe_offload import estimate_expert_savings

        if estimate_expert_savings(path, 0.125) > 0:
            return True, ""
        return False, "The checkpoint has no offloadable routed experts."

    from .moe_expert_offload import CheckpointExpertStore

    text = raw.get("text_config", raw)
    count = int(text.get("num_experts") or 0)
    layers = int(text.get("num_hidden_layers") or 0)
    hidden = int(text.get("hidden_size") or 0)
    # Declared-off is falsy (False, 0, null); absent means the tensors decide.
    if min(count, layers, hidden) <= 0 or not text.get("enable_moe_block", True):
        return False, "The model does not have the supported MoE geometry."
    quant = raw.get("quantization", text.get("quantization"))
    if not isinstance(quant, dict):
        return False, "Expert offload requires an MLX quantized checkpoint."
    store = CheckpointExpertStore(path)
    reasons = []
    for key in ("moe_intermediate_size", "intermediate_size"):
        intermediate = int(text.get(key) or 0)
        if intermediate <= 0:
            continue
        for parent_template, projection in _LAYER_LAYOUTS:
            reason = _layout_reason(
                store,
                quant,
                parent_template,
                projection,
                (count, layers, hidden, intermediate),
            )
            if reason is None:
                return True, ""
            reasons.append(reason)
    # "Missing tensor" only means the layout is not this one; a real defect in
    # the layout that is present is the more useful thing to report.
    defects = [r for r in reasons if not r.startswith("Checkpoint is missing")]
    if defects:
        return False, defects[0]
    if reasons:
        return False, reasons[0]
    return False, "The model does not have the supported MoE geometry."


def _layout_reason(store, quant, parent_template, projection, geometry):
    """None when the checkpoint matches this layout, otherwise why it does not."""
    count, layers, hidden, intermediate = geometry
    for layer in range(layers):
        parent = parent_template.format(layer=layer)
        prefix = parent + projection
        per_expert = not store.has(prefix + ".gate_proj.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = prefix + "." + proj
            spec = quant.get(key, quant)
            if not isinstance(spec, dict):
                return f"Unsupported expert quantization: {key}"
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
                return f"Unsupported expert quantization: {key}"
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
                return f"Unsupported expert packing: {key}"
            fields = (
                ("weight", "scales", "biases")
                if mode == "affine"
                else ("weight", "scales")
            )
            for expert in range(count) if per_expert else (None,):
                base = f"{parent}.experts.{expert}.{proj}" if per_expert else key
                if store.has(base + ".bias"):
                    return "Per-expert linear bias is not supported."
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
                        return f"Checkpoint is missing expert tensor: {name}"
                    actual_shape, dtype = store.spec(name)
                    if actual_shape != shape or dtype not in dtypes:
                        return f"Unsupported expert tensor shape or dtype: {name}"
    return None
