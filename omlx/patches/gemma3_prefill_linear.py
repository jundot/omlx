# SPDX-License-Identifier: Apache-2.0
"""Transient BF16 dense projections for Gemma 3 12B prefill on M1 Max.

Dequantizing each matrix once per long prefill call avoids repeated unpacking
inside the quantized matmul tiles. The dependency on the activation delays
materialization until the preceding layer has produced its input. Each module
continues to store its original packed weights, scales, and biases.
"""

import mlx.core as mx
import mlx.nn as nn


# Measured long-prefill range on the M1 Max.
_MIN_ROWS = 512
_MAX_ROWS = 2048
_PROJECTIONS = (
    ("self_attn", "q_proj", 3840, 4096),
    ("self_attn", "k_proj", 3840, 2048),
    ("self_attn", "v_proj", 3840, 2048),
    ("self_attn", "o_proj", 4096, 3840),
    ("mlp", "gate_proj", 3840, 15360),
    ("mlp", "up_proj", 3840, 15360),
    ("mlp", "down_proj", 15360, 3840),
)
_SHAPES = frozenset(
    (input_dim, output_dim) for _, _, input_dim, output_dim in _PROJECTIONS
)
_SCRATCH_GROUPS = (
    ("self_attn", ("q_proj", "k_proj", "v_proj")),
    ("self_attn", ("o_proj",)),
    ("mlp", ("gate_proj", "up_proj")),
    ("mlp", ("down_proj",)),
)


def _supported_linear(linear, input_dim, output_dim):
    return (
        (input_dim, output_dim) in _SHAPES
        and getattr(linear, "mode", None) == "affine"
        and getattr(linear, "bits", None) == 4
        and getattr(linear, "group_size", None) == 64
        and "bias" not in linear
        and isinstance(linear.get("weight"), mx.array)
        and linear.weight.dtype == mx.uint32
        and linear.weight.shape == (output_dim, input_dim // 8)
        and isinstance(linear.get("scales"), mx.array)
        and linear.scales.dtype == mx.bfloat16
        and linear.scales.shape == (output_dim, input_dim // 64)
        and isinstance(linear.get("biases"), mx.array)
        and linear.biases.dtype == mx.bfloat16
        and linear.biases.shape == linear.scales.shape
    )


class Gemma3PrefillLinear(nn.QuantizedLinear):
    """A loaded quantized projection with a bounded long-prefill path."""

    def __call__(self, x):
        if (
            x.ndim != 3
            or x.shape[0] != 1
            or not _MIN_ROWS <= x.shape[1] <= _MAX_ROWS
            or x.dtype != mx.bfloat16
            or not _supported_linear(self, x.shape[-1], self.weight.shape[0])
        ):
            return super().__call__(x)
        weight = mx.depends(self.weight, x)
        dense = mx.dequantize(
            weight, self.scales, self.biases, group_size=64, bits=4, mode="affine"
        )
        return x @ dense.T


def _prefill_scratch_bytes(model) -> int:
    """Bound simultaneous dense weights once when installing this route.

    Q/K/V and gate/up each share an activation and may materialize together.
    The dependency in ``Gemma3PrefillLinear`` separates successive groups and
    layers. The complete 12B route reserves 225 MiB for the gate/up pair,
    exceeding its 60 MiB Q/K/V group and 112.5 MiB down projection.
    """
    layers = getattr(model, "layers", ())
    if not isinstance(layers, (list, tuple)):
        return 0
    peak = 0
    for layer in layers:
        for parent_name, names in _SCRATCH_GROUPS:
            parent = getattr(layer, parent_name, None)
            group_bytes = 0
            for name in names:
                linear = getattr(parent, name, None)
                if type(linear) is Gemma3PrefillLinear:
                    output_dim, packed_dim = linear.weight.shape
                    input_dim = packed_dim * 8
                    if _supported_linear(linear, input_dim, output_dim):
                        group_bytes += output_dim * input_dim * 2
            peak = max(peak, group_bytes)
    return peak


def apply_gemma3_prefill_linear_patch(model) -> int:
    """Rebind eligible projections on this loaded Gemma 3 12B instance."""
    if not mx.metal.is_available():
        return 0
    if mx.device_info().get("device_name") != "Apple M1 Max":
        return 0

    from mlx_vlm.models.gemma3 import language

    language_model = getattr(model, "language_model", None)
    if type(language_model) is not language.LanguageModel:
        return 0
    text_model = language_model.model
    if type(text_model) is not language.Gemma3Model:
        return 0
    config = text_model.config
    geometry = (
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    if geometry != (3840, 15360, 48, 16, 8, 256) or len(text_model.layers) != 48:
        return 0

    count = 0
    for layer in text_model.layers:
        if (
            type(layer) is not language.TransformerBlock
            or type(layer.self_attn) is not language.Attention
            or type(layer.mlp) is not language.MLP
        ):
            continue
        for parent_name, name, input_dim, output_dim in _PROJECTIONS:
            linear = getattr(getattr(layer, parent_name), name)
            if type(linear) is nn.QuantizedLinear and _supported_linear(
                linear, input_dim, output_dim
            ):
                # QuantizedLinear's storage and parameter paths stay identical.
                linear.__class__ = Gemma3PrefillLinear
                count += 1
    language_model._omlx_prefill_scratch_bytes = _prefill_scratch_bytes(language_model)
    language_model._omlx_prefill_scratch_min_tokens = _MIN_ROWS
    return count
