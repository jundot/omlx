"""Exact MLX implementations of DFly2's checkpointed fusion/correction heads.

Formulas are transcribed from the pinned oMLX DFly implementation
(`farm_source/dfly.py`, DFlyFusion and DFlyCorrection). These functions consume
raw tensors keyed exactly as the official checkpoint index names them, so the
same functions can be run on a parent-owned isolated checkpoint staging area.
"""
from __future__ import annotations

from collections.abc import Mapping

import mlx.core as mx


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps) * weight


def _linear(x: mx.array, weight: mx.array) -> mx.array:
    """HF/torch linear: weights are `[out_features, in_features]`."""
    return x @ weight.T


def mtp_fusion(anchor: mx.array, target_hidden: mx.array, weights: Mapping[str, mx.array], *, eps: float) -> mx.array:
    """DFly2 slot-0 fusion: `E + gate * eh_proj([enorm(E), hnorm(H)])`."""
    base = "model.mtp_fusion."
    e = _rms_norm(anchor, weights[base + "enorm.weight"], eps)
    h = _rms_norm(target_hidden, weights[base + "hnorm.weight"], eps)
    fused = _linear(mx.concatenate([e, h], axis=-1), weights[base + "eh_proj.weight"])
    return anchor + weights[base + "gate"] * fused


def hidden_correction(hidden: mx.array, previous_embedding: mx.array, weights: Mapping[str, mx.array], *, eps: float) -> mx.array:
    """DFly2 slot correction: `h + down(silu(gate(z)) * up(z))`."""
    base = "model.hidden_correction."
    h = _rms_norm(hidden, weights[base + "hidden_norm.weight"], eps)
    e = _rms_norm(previous_embedding, weights[base + "embed_norm.weight"], eps)
    z = mx.concatenate([h, e], axis=-1)
    gate_input = _linear(z, weights[base + "gate_proj.weight"])
    gated = gate_input * mx.sigmoid(gate_input)
    up = _linear(z, weights[base + "up_proj.weight"])
    return hidden + _linear(gated * up, weights[base + "down_proj.weight"])


def final_norm(hidden: mx.array, weights: Mapping[str, mx.array], *, eps: float) -> mx.array:
    return _rms_norm(hidden, weights["model.final_norm.weight"], eps)
