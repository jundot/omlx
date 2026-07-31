"""Nucleus (top-p) / top-k truncation for lossless speculative *sampling*.

Adapted in part from ARahim3/mlx-dspark (MIT); see THIRD_PARTY_NOTICES.md.

To keep speculative sampling lossless while honoring OpenAI's ``top_p`` / ``top_k``, the
truncation must be applied to the **target** distribution the accept test samples from — and,
so acceptance doesn't collapse, to the **draft** proposal too. This module provides one
primitive, :func:`truncate_probs`, used on both sides (draft q and target p) with the same
parameters. The output is then an exact sample from ``top-p/top-k(softmax(target_logits / T))``.

Identity fast-path: ``top_p >= 1.0`` and ``top_k <= 0`` returns the input untouched, so the
default temperature-sampling path (and greedy) is byte-for-byte unchanged.
"""

from __future__ import annotations

import mlx.core as mx


def truncate_probs(probs: mx.array, top_p: float = 1.0, top_k: int = 0) -> mx.array:
    """Top-k then top-p (nucleus) truncate a probability distribution over the last axis,
    renormalized. ``probs`` sums to 1 per row and stays that way. Ties at the nucleus
    boundary are kept (standard nucleus behavior)."""
    top_p = 1.0 if top_p is None else float(top_p)
    top_k = 0 if not top_k else int(top_k)
    if top_p >= 1.0 and top_k <= 0:
        return probs

    p = probs
    v = p.shape[-1]
    if 0 < top_k < v:
        # kth-largest value per row (ascending sort -> index v-top_k), keep >= it
        kth = mx.sort(p, axis=-1)[..., v - top_k][..., None]
        p = mx.where(p >= kth, p, 0.0)

    if top_p < 1.0:
        sorted_desc = -mx.sort(-p, axis=-1)  # descending values per row
        csum = mx.cumsum(sorted_desc, axis=-1)
        # a token is in the nucleus if the cumulative mass *before* it is still < top_p
        # (this always keeps at least the top-1 token)
        in_nucleus = (csum - sorted_desc) < top_p
        inf = mx.array(float("inf"), dtype=p.dtype)
        cutoff = mx.min(mx.where(in_nucleus, sorted_desc, inf), axis=-1, keepdims=True)
        p = mx.where(p >= cutoff, p, 0.0)

    return p / mx.maximum(p.sum(axis=-1, keepdims=True), 1e-12)


def sample_probs(probs: mx.array) -> mx.array:
    """Categorical sample from a (possibly truncated) probability row/batch of rows."""
    return mx.random.categorical(mx.log(probs + 1e-20))
