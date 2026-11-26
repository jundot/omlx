# SPDX-License-Identifier: Apache-2.0
"""Expert offload for Inkling 2's routed experts.

``Inkling2Experts`` keeps its own forward (interleaved ``w13``, softcapped
SwiGLU, NVFP4 post-scales), so like the DeepSeek V4 adapter this one swaps
slot tensors into its ``w13``/``w2`` projections and translates expert ids to
slot ids through the shared ``moe_slot_cache.SlotCache``. Over-capacity prefill
runs expert-major chunks through ``routes`` with the kernel of the whole call,
and takes the w2 post-scale from the full ``global_scale`` table.
"""

from __future__ import annotations

import logging
import weakref
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...scheduler import _sync_and_clear_cache
from ..moe_expert_offload import (
    CheckpointExpertStore,
    _GLUStoreView,
    _minimum_experts,
)
from ..moe_slot_cache import SlotCache
from .model import _NVFP4_DENOMINATOR, Inkling2Layer

logger = logging.getLogger(__name__)


class OffloadedInkling2Experts(nn.Module):
    """``Inkling2Experts`` whose experts live in a shared ``SlotCache``."""

    def __init__(self, experts, capacity: int, view: _GLUStoreView):
        super().__init__()
        self.cache = SlotCache(experts, capacity, view, projs=("w13", "w2"))

    def _weighted(self, z: mx.array, inds: mx.array, weights: mx.array):
        c = self.cache
        scale = c.global_scales.get("w2")
        if scale is not None and not c.glu.w2.scales_in_kernel(z.dtype):
            weights = weights * (scale[inds] * (1.0 / _NVFP4_DENOMINATOR))
        return z, weights

    def unreduced(self, x: mx.array, inds: mx.array, weights: mx.array):
        """Expert outputs [N, K, H] and weights [N, K], as
        ``Inkling2Experts.unreduced``."""
        c = self.cache
        n_tok, k = inds.shape
        call = (inds.size, n_tok, c.n_experts)
        ids = None
        fits = n_tok * k <= c.capacity or n_tok == 1
        if not fits:
            ids = inds.reshape(-1).tolist()
            fits = len(set(ids)) <= c.capacity
        if fits:
            c.ensure(inds)
            z = c.glu.routes(x, mx.take(c.map, inds), call)
            return self._weighted(z, inds, weights)

        ids_np = np.asarray(ids, dtype=np.int64)
        resident = np.zeros(c.n_experts, dtype=bool)
        resident[list(c.slot_of)] = True
        order = np.argsort(ids_np + c.n_experts * ~resident[ids_np], kind="stable")
        sorted_ids = ids_np[order]
        run_starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_ids)) + 1))
        cuts = run_starts[:: c.capacity].tolist() + [len(ids)]
        outs = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            chunk = sorted_ids[start:end]
            c.ensure_ids(np.unique(chunk).tolist())
            slots = mx.take(c.map, mx.array(chunk, dtype=mx.int32))
            xe = mx.take(x, mx.array(order[start:end] // k, dtype=mx.int32), axis=0)
            o = c.glu.routes(xe, slots[:, None], call)[:, 0]
            mx.eval(o)
            outs.append(o)
        inverse = mx.array(np.argsort(order, kind="stable"), dtype=mx.int32)
        z = mx.take(mx.concatenate(outs), inverse, axis=0).reshape(n_tok, k, -1)
        return self._weighted(z, inds, weights)


def _post_with_eager_moe(layer):
    proxy = weakref.proxy(layer)
    attn = mx.compile(lambda *a: Inkling2Layer._post_attn(proxy, *a))
    mlp = mx.compile(lambda *a: Inkling2Layer._post_mlp(proxy, *a))

    def post(out, gate, h, mlp_blocks, s2, s3, messages=None):
        residual, x, x32, s2 = attn(out, gate, h, mlp_blocks, s2, messages)
        z, zw = proxy.mlp.unreduced(x, x32)
        h_out, s3 = mlp(residual, z, zw, s3, h)
        return h_out, s2, s3

    return post


def apply_inkling2_moe_expert_offload(
    model, model_dir: Path, resident_fraction: float = 0.25
) -> int:
    """Wrap every Inkling 2 layer's routed experts; returns the number wrapped."""
    layers = [m for m in model.modules() if isinstance(m, Inkling2Layer)]
    if not layers:
        return 0
    store = CheckpointExpertStore(model_dir)
    minimum = _minimum_experts(model_dir)
    total_bytes = resident_bytes = 0
    for layer in layers:
        experts = layer.mlp.experts
        n_experts = experts.w13.weight.shape[0]
        capacity = min(n_experts, max(minimum, round(n_experts * resident_fraction)))
        layer_bytes = sum(
            a.nbytes for _, a in nn.utils.tree_flatten(experts.parameters())
        )
        total_bytes += layer_bytes
        resident_bytes += layer_bytes * capacity // n_experts
        path = f"model.layers.{layer.layer_idx}.mlp.experts"
        layer.mlp.experts = OffloadedInkling2Experts(
            experts, capacity, _GLUStoreView(store, path)
        )
        layer._post_compiled = _post_with_eager_moe(layer)
        layer._post_exact = _post_with_eager_moe(layer)
        _sync_and_clear_cache()
    logger.info(
        "inkling2 moe expert offload: wrapped %d layers at %.1f%% residency "
        "(expert tables: %.2f GB total, %.2f GB resident)",
        len(layers),
        100 * resident_fraction,
        total_bytes / 1e9,
        resident_bytes / 1e9,
    )
    return len(layers)


__all__ = ["OffloadedInkling2Experts", "apply_inkling2_moe_expert_offload"]
