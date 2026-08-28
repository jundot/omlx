# SPDX-License-Identifier: Apache-2.0
"""Resident-aware approximate routing for Soft-REAP expert banks."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_5_moe.language import (
    _target_verify_linear,
    _target_verify_switch_glu,
)


def resident_preferred_topk(
    gates: mx.array,
    resident_mask: mx.array,
    *,
    top_k: int,
    threshold_percent: float,
) -> tuple[mx.array, mx.array]:
    """Route near-ties to resident experts, retaining original gate weights.

    A resident expert may displace a cold expert when its unmodified router
    probability is within ``threshold_percent`` of the cold probability.
    """

    threshold = min(100.0, max(0.0, float(threshold_percent))) / 100.0
    if threshold >= 1.0:
        adjusted = mx.where(resident_mask, gates, -mx.inf)
    elif threshold > 0.0:
        adjusted = mx.where(resident_mask, gates / (1.0 - threshold), gates)
    else:
        adjusted = gates
    indices = mx.argpartition(adjusted, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, indices, axis=-1)
    return indices, scores / scores.sum(axis=-1, keepdims=True)


class ResidentPreferredMoeBlock(nn.Module):
    """Qwen3.5/3.8 MoE block with optional resident-near-tie routing."""

    def __init__(self, original: Any, *, threshold_percent: float):
        super().__init__()
        self.num_experts = int(original.num_experts)
        self.top_k = int(original.top_k)
        self.gate = original.gate
        self.switch_mlp = original.switch_mlp
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.threshold_percent = float(threshold_percent)

    def __call__(self, x: mx.array, target_verify: bool = False) -> mx.array:
        gates = _target_verify_linear(self.gate, x, target_verify)
        gates = mx.softmax(gates, axis=-1, precise=True)
        indices, scores = resident_preferred_topk(
            gates,
            self.switch_mlp.resident_mask,
            top_k=self.top_k,
            threshold_percent=self.threshold_percent,
        )
        # The streaming pool owns cache-sized expert-major chunking. Calling
        # mlx-vlm's target-verify helper directly would bypass that whole-GLU
        # path and stream each projection separately for a wide verify batch.
        if getattr(self.switch_mlp, "_omlx_expert_streaming", False):
            routed = self.switch_mlp(x, indices)
        else:
            routed = _target_verify_switch_glu(
                self.switch_mlp, x, indices, target_verify
            )
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = self.shared_expert(x, target_verify)
        shared = (
            mx.sigmoid(_target_verify_linear(self.shared_expert_gate, x, target_verify))
            * shared
        )
        return routed + shared
