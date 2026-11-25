# SPDX-License-Identifier: Apache-2.0
"""Expert offload for Inkling 2's routed experts.

``Inkling2Experts`` keeps its own forward (interleaved ``w13``, softcapped
SwiGLU, NVFP4 post-scales), so like the DeepSeek V4 adapter this one swaps
slot tensors into its ``w13``/``w2`` projections and translates expert ids to
slot ids through the shared ``moe_slot_cache.SlotCache``. Over-capacity prefill runs
expert-major chunks through ``routes`` and reassembles them for
``unreduced``, with the w2 post-scale taken from the full ``global_scale``
table. Decode steps whose reads are slow run the resident routes
while the missing experts are read (``OffloadSwitchGLU._forward_overlap``).
Calls that fit the cache match the unwrapped module bit for bit; chunked
calls differ at rounding scale (the gather kernel tiles the smaller chunks
differently).
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
    _SORT_MIN_ROUTES,
    CheckpointExpertStore,
    _GLUStoreView,
    _io_pool,
    _minimum_experts,
    _resolve_model_dir,
)
from ..moe_slot_cache import SlotCache
from .model import _NVFP4_DENOMINATOR, Inkling2Layer, moe_sum

logger = logging.getLogger(__name__)

_PROJS = ("w13", "w2")


class OffloadedInkling2Experts(nn.Module):
    """``Inkling2Experts`` whose experts live in a shared ``SlotCache``."""

    def __init__(self, experts, capacity: int, view: _GLUStoreView):
        super().__init__()
        self.cache = SlotCache(experts, capacity, view, projs=_PROJS)
        self._overlap = True  # decode fetch/compute overlap (off: tests)

    def _weighted(self, z: mx.array, inds: mx.array, weights: mx.array):
        """``Inkling2Experts.unreduced``'s output for routes ``z`` gathered by
        slot: the w2 post-scale goes on the weights by expert id.

        Float16 already applied the slot's scale inside gather_qmm.
        """
        c = self.cache
        scale = c.global_scales.get("w2")
        if scale is not None and not c.glu.w2.scales_in_kernel(z.dtype):
            weights = weights * (scale[inds] * (1.0 / _NVFP4_DENOMINATOR))
        return z, weights

    def _routes(self, x: mx.array, ids: list, pos: list, k: int) -> mx.array:
        """Unweighted outputs of the flat route positions ``pos`` only, one row
        per route: the per-row gather of the full decode call."""
        c = self.cache
        rows = mx.take(x, mx.array([p // k for p in pos], dtype=mx.int32), axis=0)
        slots = mx.array([c.slot_of[ids[p]] for p in pos], dtype=mx.int32)
        return c.glu.routes(rows, slots[:, None])[:, 0]

    def _forward_overlap(self, x: mx.array, inds: mx.array, weights: mx.array):
        """Decode step that keeps the GPU busy while slow reads are pending.

        Same contract as ``OffloadSwitchGLU._forward_overlap``: fast reads keep
        the serial step; slow ones dispatch the resident routes first (evaluated
        before the first slot write), keep the GPU clocked through the wait,
        then gather the missing routes. Every route runs once, per row.
        """
        c = self.cache
        n_tok, k = inds.shape
        ids = inds.reshape(-1).tolist()
        hit = [e in c.slot_of for e in ids]  # residency before any install
        hit_pos = [p for p, h in enumerate(hit) if h]
        y_hit = None

        def on_slow_read():
            nonlocal y_hit
            if hit_pos:
                y_hit = self._routes(x, ids, hit_pos, k)
                mx.async_eval(y_hit)
            return y_hit

        slow = c.ensure_ids(ids, on_slow_read=on_slow_read)
        if y_hit is None:  # fast reads, or no resident route
            return c.glu.unreduced(x, mx.take(c.map, inds), weights)
        miss_pos = [p for p, h in enumerate(hit) if not h]
        y = mx.concatenate([y_hit, self._routes(x, ids, miss_pos, k)])
        back = [0] * len(ids)  # route p's row in y
        for row, p in enumerate(hit_pos + miss_pos):
            back[p] = row
        z = mx.take(y, mx.array(back, dtype=mx.int32), axis=0).reshape(n_tok, k, -1)
        out = self._weighted(z, inds, weights)
        if slow:
            mx.async_eval(*out)
        return out

    def __call__(self, x: mx.array, inds: mx.array, weights: mx.array) -> mx.array:
        """FP32 sum, as ``Inkling2Experts.__call__``."""
        return moe_sum(*self.unreduced(x, inds, weights))

    def unreduced(self, x: mx.array, inds: mx.array, weights: mx.array):
        """Expert outputs [N, K, H] and weights [N, K], as
        ``Inkling2Experts.unreduced`` (the layer calls this one)."""
        c = self.cache
        n_tok, k = inds.shape
        if k > c.capacity:
            raise ValueError("Expert cache capacity is smaller than routing top-k")
        ids = None
        fits = n_tok * k <= c.capacity or n_tok == 1
        if not fits:
            ids = inds.reshape(-1).tolist()
            fits = len(set(ids)) <= c.capacity
        if fits:
            if (
                self._overlap
                and not c.warm
                and inds.size < _SORT_MIN_ROUTES
                and _io_pool() is not None
            ):
                return self._forward_overlap(x, inds, weights)
            c.ensure(inds)
            return c.glu.unreduced(x, mx.take(c.map, inds), weights)

        # Expert-major chunks of at most ``capacity`` distinct experts, already
        # resident experts first: they need no read, and installing a missing
        # expert can then never evict one this call still has to run.
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
            o = c.glu.routes(xe, slots[:, None])[:, 0]
            mx.eval(o)
            outs.append(o)
        inverse = mx.array(np.argsort(order, kind="stable"), dtype=mx.int32)
        z = mx.take(mx.concatenate(outs), inverse, axis=0).reshape(n_tok, k, -1)
        return self._weighted(z, inds, weights)


def _split_post(layer):
    """``Inkling2Layer._post`` as compiled pre-MoE and post-MoE stages around
    the eager MoE. A weak proxy, as in ``Inkling2Layer``: no reference cycle."""
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
    model,
    model_path: str | Path,
    resident_fraction: float = 0.25,
    *,
    mtp_resident: bool = False,
) -> int:
    """Wrap every Inkling 2 layer's routed experts; returns the number wrapped.

    The DSpark draft has no routed experts, so ``mtp_resident`` changes nothing.
    """
    layers = [m for m in model.modules() if isinstance(m, Inkling2Layer)]
    if not layers:
        return 0
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return 0
    store = CheckpointExpertStore(model_dir)
    if not store:
        logger.warning(
            "inkling2 moe expert offload: no safetensors under %s", model_dir
        )
        return 0
    minimum = _minimum_experts(model_dir)
    wrapped = total_bytes = resident_bytes = 0
    for layer in layers:
        experts = layer.mlp.experts
        if isinstance(experts, OffloadedInkling2Experts):
            continue
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
        # The slot cache syncs routes to the host, which cannot run inside the
        # compiled post-attention stages (decode and row-exact verify): each
        # gets its own compiled halves around an eager MoE.
        layer._post_compiled = _split_post(layer)
        layer._post_exact = _split_post(layer)
        wrapped += 1
        _sync_and_clear_cache()
    if wrapped:
        logger.info(
            "inkling2 moe expert offload: wrapped %d layers at %.1f%% residency "
            "(expert tables: %.2f GB total, %.2f GB resident)",
            wrapped,
            100 * resident_fraction,
            total_bytes / 1e9,
            resident_bytes / 1e9,
        )
    return wrapped


__all__ = ["OffloadedInkling2Experts", "apply_inkling2_moe_expert_offload"]
