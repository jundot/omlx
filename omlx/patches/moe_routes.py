# SPDX-License-Identifier: Apache-2.0
"""Sorted MoE routes without the replicated token rows."""

from __future__ import annotations

import mlx.core as mx


def sort_routes(x, indices):
    """mlx-lm's ``_gather_sort`` without the gather.

    Returns ``(x_tok, row_map, idx, inv_order)``: the token rows
    ``x.flatten(0, -3)``, the sorted row -> token row map ``order // k``,
    the sorted expert indices and the inverse order, preserving upstream's
    ordering, so ``x_tok[row_map]`` is exactly ``_gather_sort``'s sorted ``x``.
    Callers keep that indexing lazy: it only runs when a consumer needs the
    replicated ``[T * k, 1, K]`` rows (``m5_gather_qmm.fused_gate_up_activation``
    with ``token_rows`` reads them in place instead).
    """
    *_, top_k = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    # ``order`` is a permutation: every output slot is written exactly once.
    # Inverting it by scatter avoids sorting those unique integers again.
    # Retain the small-route path to avoid scatter setup on short batches.
    if order.size >= 4096:
        inv_order = mx.put_along_axis(
            mx.zeros_like(order),
            order,
            mx.arange(order.size, dtype=order.dtype),
            axis=0,
        )
    else:
        inv_order = mx.argsort(order)
    return x.flatten(0, -3), order // top_k, indices[order], inv_order
