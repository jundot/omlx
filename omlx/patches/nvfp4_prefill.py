# SPDX-License-Identifier: Apache-2.0
"""Opt-in transient BF16 MLP projections for dense Qwen 27B prefill."""

import math
import os

import mlx.core as mx
import mlx.nn as nn


class NVFP4PrefillLinear(nn.QuantizedLinear):
    """Keep packed weights; materialize one projection at a time for prefill."""

    def __call__(self, x):
        if x.dtype != mx.bfloat16 or math.prod(x.shape[:-1]) < 1024:
            return super().__call__(x)
        weight = mx.dequantize(
            self.weight,
            scales=self.scales,
            biases=self.get("biases"),
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            dtype=x.dtype,
        )
        y = x @ weight.T
        if "bias" in self:
            y = y + self.bias
        # Bound temporary weight lifetime across successive projections.
        mx.eval(y)
        return y


def apply_nvfp4_prefill(model):
    """Install once per supported model, without changing global MLX methods."""
    if os.environ.get("OMLX_NVFP4_PREFILL") != "1":
        return 0
    text = getattr(model, "language_model", model)
    args = getattr(text, "args", None)
    if (
        type(text).__module__ != "mlx_lm.models.qwen3_5"
        or getattr(args, "hidden_size", None) != 5120
        or getattr(args, "intermediate_size", None) != 17408
    ):
        return 0
    layers = list(getattr(text, "layers", ()))
    if len(layers) != 64:
        return 0
    targets = [
        getattr(getattr(layer, "mlp", None), name, None)
        for layer in layers[:-1]
        for name in ("gate_proj", "up_proj", "down_proj")
    ]
    if any(
        type(layer) not in (nn.QuantizedLinear, NVFP4PrefillLinear)
        or layer.mode != "nvfp4"
        for layer in targets
    ):
        return 0
    changed = 0
    for layer in targets:
        if type(layer) is nn.QuantizedLinear:
            # Preserve parameter names, tensor identities and frozen state.
            layer.__class__ = NVFP4PrefillLinear
            changed += 1
    # The final MLP can be omitted by cache-only lazy prefill.
    return changed
