# SPDX-License-Identifier: Apache-2.0
"""Sorted MoE routes without the replicated token rows."""

from __future__ import annotations

import mlx.core as mx


def sort_routes(x, indices):
    """mlx-lm's ``_gather_sort`` without the gather.

    Returns ``(x_tok, row_map, idx, inv_order)``: the token rows
    ``x.flatten(0, -3)``, the sorted row -> token row map ``order // k``,
    the sorted expert indices and the inverse order, computed by the same
    ops, so ``x_tok[row_map]`` is exactly ``_gather_sort``'s sorted ``x``.
    Callers keep that indexing lazy: it only runs when a consumer needs the
    replicated ``[T * k, 1, K]`` rows (``m5_gather_qmm.fused_gate_up_activation``
    with ``token_rows`` reads them in place instead).
    """
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3), order // M, indices[order], inv_order
