# SPDX-License-Identifier: Apache-2.0
"""Transient BF16 dense projections for Gemma 3 12B prefill on M1 Max.

Dequantizing each matrix once per long prefill call avoids repeated unpacking
inside the quantized matmul tiles. The dependency on the activation delays
materialization until the preceding layer has produced its input. Each module
continues to store its original packed weights, scales, and biases.
"""

from importlib import import_module

import mlx.core as mx
import mlx.nn as nn

from ..utils.hardware import get_mlx_device_name

_MIN_ROWS = 512
_MAX_ROWS = 2048
# Projections sharing an input may materialize their BF16 weights together.
_PROJECTION_GROUPS = (
    (
        "self_attn",
        (("q_proj", 3840, 4096), ("k_proj", 3840, 2048), ("v_proj", 3840, 2048)),
    ),
    ("self_attn", (("o_proj", 4096, 3840),)),
    ("mlp", (("gate_proj", 3840, 15360), ("up_proj", 3840, 15360))),
    ("mlp", (("down_proj", 15360, 3840),)),
)


def _supported_linear(linear, input_dim, output_dim):
    biases = linear.get("biases")
    return (
        linear.mode == "affine"
        and linear.bits == 4
        and linear.group_size == 64
        and "bias" not in linear
        and linear.weight.dtype == mx.uint32
        and linear.weight.shape == (output_dim, input_dim // 8)
        and linear.scales.dtype == mx.bfloat16
        and linear.scales.shape == (output_dim, input_dim // 64)
        and biases is not None
        and biases.dtype == mx.bfloat16
        and biases.shape == linear.scales.shape
    )


class Gemma3PrefillLinear(nn.QuantizedLinear):
    """A loaded quantized projection with a bounded long-prefill path."""

    def __call__(self, x):
        if (
            x.ndim != 3
            or x.shape[0] != 1
            or not _MIN_ROWS <= x.shape[1] <= _MAX_ROWS
            or x.dtype != mx.bfloat16
        ):
            return super().__call__(x)
        weight = mx.depends(self.weight, x)
        dense = mx.dequantize(
            weight,
            self.scales,
            self.biases,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        return x @ dense.T


def apply_gemma3_prefill_linear_patch(model) -> int:
    """Rebind eligible projections on this loaded Gemma 3 12B instance."""
    if get_mlx_device_name() != "Apple M1 Max":
        return 0

    language = import_module("mlx_vlm.models.gemma3.language")

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
    peak_bytes = 0
    for layer in text_model.layers:
        if (
            type(layer) is not language.TransformerBlock
            or type(layer.self_attn) is not language.Attention
            or type(layer.mlp) is not language.MLP
        ):
            continue
        for parent_name, projections in _PROJECTION_GROUPS:
            parent = getattr(layer, parent_name)
            group_bytes = 0
            for name, input_dim, output_dim in projections:
                linear = getattr(parent, name)
                if type(linear) not in (nn.QuantizedLinear, Gemma3PrefillLinear):
                    continue
                if not _supported_linear(linear, input_dim, output_dim):
                    continue
                if type(linear) is nn.QuantizedLinear:
                    linear.__class__ = Gemma3PrefillLinear
                    count += 1
                group_bytes += input_dim * output_dim * mx.bfloat16.size
            peak_bytes = max(peak_bytes, group_bytes)
    language_model._omlx_prefill_scratch_bytes = peak_bytes
    language_model._omlx_prefill_scratch_min_tokens = _MIN_ROWS
    return count
