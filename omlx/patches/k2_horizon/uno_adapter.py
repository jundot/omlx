# SPDX-License-Identifier: Apache-2.0
"""Strict loading of the released K2 Uno conditional LoRA adapters."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

TARGETS = {
    "self_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


class ConditionalLoRALinear(nn.Module):
    """Keep AR weights unchanged and apply low-rank updates to selected rows."""

    def __init__(self, base, a, b, scale):
        super().__init__()
        self.linear = base
        self.lora_a = a.T
        self.lora_b = b.T
        self.scale = scale

    def __call__(self, x):
        return self.linear(x)

    def conditional_forward(self, x, row_mask):
        if row_mask is None:
            return self.linear(x)
        if row_mask.shape != x.shape[:-1]:
            raise ValueError(
                f"Uno row mask {row_mask.shape} does not match {x.shape[:-1]}"
            )
        hidden = (x @ self.lora_a) * row_mask[..., None].astype(x.dtype)
        delta = (hidden @ self.lora_b) * self.scale
        return self.linear(x) + delta.astype(x.dtype)


def load_uno_adapter(model, path: str | Path, *, base_model_id: str) -> dict:
    path = Path(path)
    config = json.loads((path / "adapter_config.json").read_text())
    if base_model_id not in ("IFM/K2-Horizon-0.9B", "IFM/K2-Horizon-7B"):
        raise ValueError(f"No released K2 Uno adapter for {base_model_id}")
    if config.get("base_model_name_or_path") != base_model_id:
        raise ValueError(
            "Uno adapter base_model_name_or_path does not match the selected base"
        )
    if model.args.num_experts or model.args.attention_gate_func is not None:
        raise ValueError("Released Uno adapters require a dense, ungated K2 base")
    expected_settings = {
        "peft_type": "LORA",
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": None,
        "layers_pattern": None,
        "layers_to_transform": None,
    }
    for key, value in expected_settings.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported Uno adapter {key}={config.get(key)!r}")
    for key in (
        "use_dora",
        "use_rslora",
        "rank_pattern",
        "alpha_pattern",
        "loftq_config",
    ):
        if config.get(key):
            raise ValueError(f"Unsupported Uno adapter {key}={config[key]!r}")
    target_names = {name for names in TARGETS.values() for name in names}
    if set(config.get("target_modules") or []) != target_names:
        raise ValueError("Uno adapter must target all seven released projections")
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("Uno adapter rank must be a positive integer")
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Uno adapter alpha must be finite and positive")
    weights_path = path / "adapter_model.safetensors"
    tensors = mx.load(str(weights_path))
    replacements = []
    consumed = set()
    for i, layer in enumerate(model.layers):
        for scope, names in TARGETS.items():
            parent = getattr(layer, scope)
            for name in names:
                prefix = f"model.layers.{i}.{scope}.{name}"
                keys = [f"{prefix}.lora_{part}.weight" for part in ("A", "B")]
                if any(key not in tensors for key in keys):
                    raise ValueError(f"Missing Uno LoRA pair: {prefix}")
                base = getattr(parent, name)
                if not isinstance(base, (nn.Linear, nn.QuantizedLinear)):
                    raise ValueError(f"Uno requires a Linear projection: {prefix}")
                a, b = (tensors[key] for key in keys)
                out_dims, in_dims = base.weight.shape
                if isinstance(base, nn.QuantizedLinear):
                    in_dims = in_dims * 32 // base.bits
                if a.shape != (rank, in_dims) or b.shape != (out_dims, rank):
                    raise ValueError(
                        f"Uno LoRA shape mismatch: {prefix}: A={a.shape}, B={b.shape}, W={base.weight.shape}"
                    )
                if any(
                    t.dtype not in (mx.bfloat16, mx.float16, mx.float32) for t in (a, b)
                ):
                    raise ValueError(
                        f"Uno adapter requires floating-point weights: {prefix}"
                    )
                # The reference casts LoRA weights to the base activation dtype.
                dtype = (
                    base.scales.dtype
                    if isinstance(base, nn.QuantizedLinear)
                    else base.weight.dtype
                )
                a, b = a.astype(dtype), b.astype(dtype)
                consumed.update(keys)
                replacements.append((parent, name, base, a, b))
    if consumed != set(tensors):
        raise ValueError(
            f"Unexpected Uno adapter tensors: {sorted(set(tensors) - consumed)[:3]}"
        )
    finite = mx.stack(
        [mx.all(mx.isfinite(t)) for _, _, _, a, b in replacements for t in (a, b)]
    )
    if not mx.all(finite).item():
        raise ValueError("Uno adapter contains non-finite weights")
    for parent, name, base, a, b in replacements:
        setattr(parent, name, ConditionalLoRALinear(base, a, b, alpha / rank))
    model.eval()
    model._uno_adapter_loaded = True
    return {
        "base_model_id": base_model_id,
        "adapter_path": str(path),
        "rank": rank,
        "alpha": alpha,
        "scale": alpha / rank,
        "tensors": len(tensors),
        "pairs": len(replacements),
    }
