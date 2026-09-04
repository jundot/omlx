# SPDX-License-Identifier: Apache-2.0
"""Load-time repair for miscalibrated MTP-head RMSNorm weights.

Qwen3-Next RMSNorm gammas are zero-centered in the raw checkpoint; the MLX
convention stores them shifted by +1. Older oQ conversions decided the shift
per-key from the tensor mean (``mean < 0.5 -> shift``), which misclassifies
head norms whose raw mean already sits at or above 0.5 — observed:
``q_norm``/``k_norm`` (raw ~0.75), ``post_attention_layernorm`` (raw ~0.87 on
Qwen3.6-35B-A3B), and ``mtp.norm`` (raw ~1.3-1.9) — leaving them one below
the correct value. Measured cost: tens of points of draft acceptance.

Detection is reference-based, not absolute: for each vulnerable head norm,
the mean of the *backbone's* counterpart tensors in the same payload is the
anchor (backbone norms are always converted correctly — their raw means sit
well below the old heuristic's cutoff). A correctly stored head norm sits at
or slightly above its backbone anchor; a broken one sits ~1 below. Repair
applies +1 when ``anchor_mean - head_mean > 0.4``. Idempotent: after the
shift the value lands above the anchor, so a second pass is a no-op.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

# Head-norm suffix -> backbone counterpart suffix. Only norms whose raw mean
# can plausibly exceed the old heuristic's 0.5 cutoff are listed; the rest
# (input_layernorm, pre_fc_norm_*) have strongly negative raw means and were
# always converted correctly.
_REPAIR_GROUPS = {
    ".self_attn.q_norm.weight": ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight": ".self_attn.k_norm.weight",
    ".post_attention_layernorm.weight": ".post_attention_layernorm.weight",
    "mtp.norm.weight": "model.norm.weight",
}

# Minimum gap below the backbone anchor that marks a missing +1. Correct
# head norms measure 0.19-0.56 ABOVE their anchor on Qwen3.6-27B/35B-A3B;
# broken ones measure 0.44-0.74 below.
_REPAIR_MARGIN = 0.4


def _mean(v) -> float:
    import mlx.core as mx

    return float(v.astype(mx.float32).mean().item())


def repair_legacy_head_norms(
    weights: Union[dict, List[Tuple[str, Any]]],
) -> Tuple[Union[dict, List[Tuple[str, Any]]], int]:
    """Re-apply the missing +1 shift to raw-stored MTP-head norms.

    Accepts and returns either a dict or a list of (key, tensor) pairs (the
    two shapes ``nn.Module.load_weights`` payloads take). Returns the
    (possibly-modified) payload and the number of tensors repaired.
    """
    is_list = not isinstance(weights, dict)
    items = list(weights.items()) if isinstance(weights, dict) else list(weights)

    head_indices: List[Tuple[int, str]] = []  # (item index, head suffix)
    for i, (k, v) in enumerate(items):
        if "mtp." not in k or getattr(v, "ndim", 0) != 1:
            continue
        for head_sfx in _REPAIR_GROUPS:
            if k.endswith(head_sfx):
                head_indices.append((i, head_sfx))
                break
    if not head_indices:
        return weights, 0

    # Backbone anchors: mean-of-means over counterpart tensors.
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    needed = {_REPAIR_GROUPS[sfx] for _, sfx in head_indices}
    for k, v in items:
        if "mtp." in k or getattr(v, "ndim", 0) != 1:
            continue
        for ref_sfx in needed:
            if k.endswith(ref_sfx):
                try:
                    sums[ref_sfx] = sums.get(ref_sfx, 0.0) + _mean(v)
                    counts[ref_sfx] = counts.get(ref_sfx, 0) + 1
                except Exception:
                    pass
                break

    fixed = 0
    for i, head_sfx in head_indices:
        ref_sfx = _REPAIR_GROUPS[head_sfx]
        if not counts.get(ref_sfx):
            continue
        anchor = sums[ref_sfx] / counts[ref_sfx]
        k, v = items[i]
        try:
            gap = anchor - _mean(v)
        except Exception:
            continue
        if gap > _REPAIR_MARGIN:
            items[i] = (k, v + 1.0)
            fixed += 1
    if fixed:
        logger.info(
            "Repaired %d MTP-head norms (+1 zero-centered gamma shift)",
            fixed,
        )
    return (items if is_list else dict(items)), fixed
