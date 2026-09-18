# SPDX-License-Identifier: Apache-2.0
"""Shared gather-sort/scatter-unsort permutation for MoE switch paths.

Canonical copy of the ``switch_layers`` routing permutation (the sorted
-indices contract that ``gather_mm``/``gather_qmm`` consume), extended
with the ``inverse_scatter`` mode the glm_moe_dsa family needs. Leaf
module — no omlx imports — so every switch patch shares one definition
instead of carrying a vendored copy per family.
"""

from __future__ import annotations

import mlx.core as mx


def inverse_permutation(order, inverse_scatter: bool = False):
    """Inverse of a sort permutation.

    Default is a second argsort; ``inverse_scatter`` builds it as a
    scatter (put_along_axis) for kernels whose unsort expects that form.
    """
    if inverse_scatter:
        return mx.put_along_axis(
            mx.zeros_like(order),
            order,
            mx.arange(order.size, dtype=order.dtype),
            axis=0,
        )
    return mx.argsort(order)


def gather_sort(x, indices, inverse_scatter: bool = False):
    """Sort routed rows by expert id.

    Returns ``(x_gathered, sorted_indices, inv_order)``: rows of *x*
    gathered in sorted-index order, the flattened sorted indices, and the
    inverse permutation ``scatter_unsort`` consumes to restore order.
    """
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = inverse_permutation(order, inverse_scatter)
    lhs_indices = order // M
    x = x.flatten(0, -3)
    return x[lhs_indices], indices[order], inv_order


def scatter_unsort(x, inv_order, shape=None):
    """Undo ``gather_sort``: permute rows back, then restore *shape*."""
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x
