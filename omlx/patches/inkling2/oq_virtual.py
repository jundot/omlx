# SPDX-License-Identifier: Apache-2.0
"""Routed-expert source tensors of Inkling 2 as oQ's logical tensors.

The runtime needs the codes viewed as ``uint32`` and the amax in MLX's
convention, which oQ's streaming plan cannot replay, so both happen here as the
source is read. Each projection becomes ``<proj>.weight`` (passed through or
requantized) plus ``<proj>.global_scale`` for NVFP4; the dequantized value
leaves the tensor scale to the caller.
"""

from __future__ import annotations

import mlx.core as mx

from .oq_layout import NVFP4_GLOBAL_SCALE, SOURCE_PREFIX, is_inkling2, text_config

NVFP4_QUANT_INFO = {"kind": "nvfp4", "bits": 4, "group_size": 16, "mode": "nvfp4"}
MXFP8_QUANT_INFO = {"kind": "mxfp8", "bits": 8, "group_size": 32, "mode": "mxfp8"}
_PROJECTIONS = ("w13", "w2")
# Experts dequantized per step; a whole layer is up to 16 GB in BF16.
_EXPERT_CHUNK = 32


def register(index, config) -> int:
    """Expose the routed expert projections as logical ``.weight`` tensors."""
    if not is_inkling2(config):
        return 0
    count = 0
    for layer in range(int(text_config(config)["num_hidden_layers"])):
        for proj in _PROJECTIONS:
            count += _register_projection(
                index, f"{SOURCE_PREFIX}layers.{layer}.mlp.experts.{proj}"
            )
    return count


def _register_projection(index, base: str) -> int:
    weight_key = f"{base}_weight"
    scale_key = f"{base}_weight_scale"
    amax_key = f"{base}_weight_amax"
    denom_key = f"{base}_weight_outer_scale_denominator"
    shape = index.source_shape(weight_key)
    if shape is None:
        # An oQ output or a calibration proxy already holds the runtime layout.
        return 0
    scale_shape = index.source_shape(scale_key)
    nvfp4 = index.source_shape(amax_key) is not None
    info = NVFP4_QUANT_INFO if nvfp4 else MXFP8_QUANT_INFO
    cols = shape[-1] * 2 if nvfp4 else shape[-1]
    logical = (*shape[:-1], cols)
    if (
        len(shape) != 3
        or index.source_dtype(weight_key) != ("U8" if nvfp4 else "F8_E4M3")
        or index.source_dtype(scale_key) != ("F8_E4M3" if nvfp4 else "U8")
        or scale_shape != (*shape[:-1], cols // info["group_size"])
        or cols % info["group_size"]
        or (nvfp4 and index.source_shape(denom_key) is None)
    ):
        raise ValueError(f"Unrecognized Inkling 2 expert layout: {base}")

    def load_packed():
        weight = index.load_source(weight_key).view(mx.uint32)
        scales = index.load_source(scale_key)
        mx.eval(weight, scales)
        return weight, scales

    def materialize():
        weight, scales = load_packed()
        parts = []
        for start in range(0, shape[0], _EXPERT_CHUNK):
            part = mx.dequantize(
                weight[start : start + _EXPERT_CHUNK],
                scales[start : start + _EXPERT_CHUNK],
                None,
                group_size=info["group_size"],
                bits=info["bits"],
                mode=info["mode"],
                dtype=mx.bfloat16,
            )
            mx.eval(part)
            parts.append(part)
        del weight, scales
        result = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        mx.eval(result)
        return result

    hides = [weight_key, scale_key]
    if nvfp4:
        hides += [amax_key, denom_key]
    index.register_virtual(
        f"{base}.weight",
        logical,
        "BF16",
        materialize,
        hides=tuple(hides),
        quant_info=info,
        load_packed=load_packed,
    )
    if not nvfp4:
        return 1

    def global_scale():
        # The amax in MLX's convention, as the runtime's loader derives it.
        amax = index.load_source(amax_key).astype(mx.float32)
        denom = index.load_source(denom_key).astype(mx.float32)
        return amax * (NVFP4_GLOBAL_SCALE / denom)

    index.register_virtual(f"{base}.global_scale", (shape[0],), "F32", global_scale)
    return 2
