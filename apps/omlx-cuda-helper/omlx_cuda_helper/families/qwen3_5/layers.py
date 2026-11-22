# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 dense blocks: the shared Qwen3.5 blocks with the dense gated MLP.

Attention, GatedDeltaNet, the K/V cache and the rotary rounding are shared
with the MoE family in ``families/_qwen35_layers.py``; this module supplies
only the dense MLP and the layer that binds it.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ... import oqh
from .._qwen35_layers import (
    Attention,
    GatedDeltaNet,
    KVCache,
    Weights,
    causal_conv,
    rotary_frequencies,
    rotate,
)
from .._qwen35_layers import (
    Layer as _Layer,
)

__all__ = [
    "Attention",
    "GatedDeltaNet",
    "KVCache",
    "Layer",
    "MLP",
    "Weights",
    "causal_conv",
    "rotate",
    "rotary_frequencies",
]


class MLP:
    """The dense gated MLP of one layer, the block a sibling family replaces."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.gate = w.linear("gate_proj")
        self.up = w.linear("up_proj")
        self.down = w.linear("down_proj")
        oqh.check_shared(self.gate, self.up)
        # The linear that reads the layer's post-norm rows: an oQh one rotates
        # them, and the shared layer asks the block for it.
        self.reader = self.gate

    def parameters(self) -> list[mx.array]:
        linears = (self.gate, self.up, self.down)
        return [a for layer in linears for a in layer.parameters()]

    def enable_kernels(self) -> None:
        """The dense MLP has no CUDA kernel; its only path is the reference."""

    def __call__(self, x: mx.array, rot: mx.array | None = None) -> mx.array:
        if self.down.rotation is None:
            return self.down(nn.silu(self.gate(x)) * self.up(x))
        # oQh: the activation is taken in float32 and rotated for down.
        rot = self.gate.rotated(x) if rot is None else rot
        act = oqh.act_rotate(self.gate(x, rot), self.up(x, rot), *self.down.rotation)
        return self.down(act, act)


class Layer(_Layer):
    """A qwen3_5 layer: the shared blocks with the dense gated MLP."""

    def __init__(self, w: Weights, text: dict[str, Any], linear: bool):
        super().__init__(w, text, linear, mlp_block=MLP)
