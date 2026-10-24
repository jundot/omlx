# SPDX-License-Identifier: Apache-2.0
"""Streaming NVFP4 routed-expert support for MiniMax M3.1 checkpoints.

M3.1 stores each routed expert projection in compressed-tensors NVFP4 form:
``weight_packed`` (two E2M1 codes per byte), ``weight_scale`` (E4M3 per 16
values) and a scalar ``weight_global_scale``. The real weight is
``fp4 * weight_scale / weight_global_scale``.

oQ needs two views of these tensors. Re-quantizing to a lower level must start
from the real weight, so the logical ``<proj>.weight`` materializes the exact
dequantized value. Keeping the source NVFP4 bytes needs the packed codes and
block scales unchanged, so the same key also offers them as a packed form. The
global scale cannot be folded into the E4M3 block scales without re-rounding
them, so the quantizer writes it separately as a per-expert vector; see
:func:`routed_expert_scale_tensors`.
"""

from __future__ import annotations

import mlx.core as mx

NVFP4_QUANT_INFO = {"kind": "nvfp4", "bits": 4, "group_size": 16, "mode": "nvfp4"}

# MLX dequantizes nvfp4 with global_scale / (6 * 448).
MLX_NVFP4_GLOBAL_SCALE = 6.0 * 448.0

_PROJ_NAMES = (("w1", "gate"), ("w2", "down"), ("w3", "up"))
_LAYER_PREFIXES = ("language_model.model.layers", "model.layers")


def has_nvfp4_routed_experts(config: dict) -> bool:
    """Whether a MiniMax config declares NVFP4 routed experts."""
    for quant_config in (
        config.get("quantization_config"),
        (config.get("text_config") or {}).get("quantization_config"),
    ):
        if (
            isinstance(quant_config, dict)
            and str(quant_config.get("moe_quant_algo") or "").upper() == "NVFP4"
        ):
            return True
    return False


def _text_config(config: dict) -> dict:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _nvfp4_moe_prefixes(index, config) -> list[str]:
    """``...block_sparse_moe`` prefixes whose experts are stored as NVFP4."""
    text_config = _text_config(config)
    num_layers = int(text_config.get("num_hidden_layers") or 0)
    prefixes = []
    for layer_idx in range(num_layers):
        for root in _LAYER_PREFIXES:
            prefix = f"{root}.{layer_idx}.block_sparse_moe"
            if index.source_shape(f"{prefix}.experts.0.w1.weight_packed") is not None:
                prefixes.append(prefix)
                break
    return prefixes


def _inverse_global_scale(index, key: str) -> mx.array:
    return mx.reciprocal(index.load_source(key).astype(mx.float32).reshape(()))


def register(index, config) -> int:
    """Expose NVFP4 expert projections as logical ``.weight`` tensors."""
    if not has_nvfp4_routed_experts(config):
        return 0

    num_experts = int(_text_config(config).get("num_local_experts") or 0)
    count = 0
    for prefix in _nvfp4_moe_prefixes(index, config):
        for expert in range(num_experts):
            for proj, _ in _PROJ_NAMES:
                count += _register_projection(
                    index, f"{prefix}.experts.{expert}.{proj}"
                )
    return count


def _register_projection(index, base: str) -> int:
    packed_key = f"{base}.weight_packed"
    scale_key = f"{base}.weight_scale"
    global_key = f"{base}.weight_global_scale"
    input_key = f"{base}.input_global_scale"
    packed_shape = index.source_shape(packed_key)
    if packed_shape is None:
        raise ValueError(f"MiniMax NVFP4 expert is missing {packed_key}")
    rows, packed_cols = packed_shape
    if (
        index.source_dtype(packed_key) != "U8"
        or index.source_shape(scale_key) != (rows, packed_cols * 2 // 16)
        or index.source_shape(global_key) is None
    ):
        raise ValueError(f"Unrecognized MiniMax NVFP4 expert layout: {base}")

    def load_packed():
        weight = index.load_source(packed_key).view(mx.uint32)
        scales = index.load_source(scale_key)
        mx.eval(weight, scales)
        return weight, scales

    def materialize():
        weight, scales = load_packed()
        real = mx.dequantize(
            weight, scales, group_size=16, bits=4, mode="nvfp4", dtype=mx.float32
        ) * _inverse_global_scale(index, global_key)
        return real.astype(mx.bfloat16)

    hides = [packed_key, scale_key, global_key]
    if index.source_shape(input_key) is not None:
        hides.append(input_key)
    index.register_virtual(
        f"{base}.weight",
        (rows, packed_cols * 2),
        "BF16",
        materialize,
        hides=tuple(hides),
        quant_info=NVFP4_QUANT_INFO,
        load_packed=load_packed,
    )
    return 1


def routed_expert_scale_tensors(index, config, per_layer_config: dict) -> dict:
    """mlx-lm ``global_scale`` vectors of the projections passed through as NVFP4.

    Each holds ``2688 / weight_global_scale`` per expert, the value mlx-lm
    writes. A projection re-quantized from the dequantized weight already holds
    real values and gets no vector.
    """
    num_experts = int(_text_config(config).get("num_local_experts") or 0)
    tensors = {}
    for prefix in _nvfp4_moe_prefixes(index, config):
        for proj, name in _PROJ_NAMES:
            switch = f"{prefix}.switch_mlp.{name}_proj"
            if (per_layer_config.get(switch) or {}).get("mode") != "nvfp4":
                continue
            tensors[f"{switch}.global_scale"] = mx.stack(
                [
                    MLX_NVFP4_GLOBAL_SCALE
                    / index.load_source(
                        f"{prefix}.experts.{e}.{proj}.weight_global_scale"
                    )
                    .astype(mx.float32)
                    .reshape(())
                    for e in range(num_experts)
                ]
            )
    return tensors
