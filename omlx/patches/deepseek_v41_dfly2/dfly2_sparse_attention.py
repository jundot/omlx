"""Numerically testable sparse attention over DFly2-selected target cache rows.

This is the portable post-cache portion of DFly2's DeepSeek-V4.1 attention path.
It intentionally takes already-projected target K/V rows: reconstructing them from
oMLX's packed MLA cache is target-runtime-specific and is guarded at integration.
"""
from __future__ import annotations

import mlx.core as mx


def selected_target_attention(
    query: mx.array,
    selected_keys: mx.array,
    selected_values: mx.array,
    valid_mask: mx.array,
    *,
    scale: float | None = None,
) -> mx.array:
    """Attend to a padded selected-target-cache set.

    Args:
      query: `[B, T, H, D]` projected draft queries.
      selected_keys: `[B, T, R, H, D]` target-cache K rows at the DFly2 picks.
      selected_values: `[B, T, R, H, V]` matching target-cache V rows.
      valid_mask: `[B, T, R]`, false for the `-1` padding emitted by context gather.
      scale: optional attention scale; default `D**-0.5`.

    The all-invalid case deliberately returns zeros rather than NaNs, making it safe
    for early prefill or a target indexer with no historical selections.
    """
    if query.ndim != 4 or selected_keys.ndim != 5 or selected_values.ndim != 5:
        raise ValueError("expected query [B,T,H,D] and selected K/V [B,T,R,H,*]")
    b, t, h, d = query.shape
    if selected_keys.shape[:2] != (b, t) or selected_keys.shape[3:] != (h, d):
        raise ValueError("query and selected_keys dimensions do not agree")
    if selected_values.shape[:4] != selected_keys.shape[:4]:
        raise ValueError("selected_keys and selected_values selection dimensions do not agree")
    if valid_mask.shape != selected_keys.shape[:3]:
        raise ValueError("valid_mask must be [B,T,R]")

    scores = mx.sum(query[:, :, None, :, :] * selected_keys, axis=-1)
    scores = scores * (d ** -0.5 if scale is None else scale)
    mask = valid_mask[:, :, :, None]
    # Stable masked softmax that remains defined if an entire row is invalid.
    neg_inf = mx.array(-float("inf"), scores.dtype)
    masked = mx.where(mask, scores, neg_inf)
    row_max = mx.max(masked, axis=2, keepdims=True)
    finite_max = mx.where(mx.isfinite(row_max), row_max, mx.zeros_like(row_max))
    exp_scores = mx.where(mask, mx.exp(scores - finite_max), mx.zeros_like(scores))
    denom = mx.sum(exp_scores, axis=2, keepdims=True)
    probs = exp_scores / mx.maximum(denom, mx.array(1.0, denom.dtype))
    return mx.sum(probs[..., None] * selected_values, axis=2)
