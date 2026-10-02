# SPDX-License-Identifier: Apache-2.0
"""Expert offload for the GLM DSA MoE block (``glm_moe_dsa``).

The GLM-5.x flagship checkpoints (``glm_moe_dsa``: 78 layers, 256 routed
experts, top-8) run their routed experts through this package's own
:class:`~omlx.patches.glm_moe_dsa.switch_layers.SwitchGLU`, which differs
from the stock mlx-lm module the common adapter wraps in two ways: the gate
and up projections are fused into one ``gate_up_proj`` tensor at load, and a
sorted call can return its routes already weighted and summed through the
native ``glm_moe_weighted_sum`` kernel. Re-implementing that forward would
fork it, so this adapter keeps the module and swaps what it computes on.

Each projection's parameters are replaced, before lazy weights materialize,
by slot tensors of ``capacity`` experts; expert ids are translated to slot
ids and the module's own ``__call__`` runs unchanged on them. Every kernel
choice inside it (sort threshold, weighted sum, inverse scatter) is a
function of the routes and the slot tensors' shapes, and every use of an
index is a gather, so a route computes the same numbers against its slot
as it would against its expert. A miss reads the expert's gate, up and down
slabs through the common checkpoint store (positional reads on its shared
pool) from either layout a checkpoint may ship, stacked ``[E, ...]`` tensors
or one tensor per expert (which the loader stacks), and writes the two
halves of the fused row in place.

Over-capacity prefill, where one call routes to more distinct experts than
the cache holds, is chunked on expert boundaries exactly as the common
adapter and the DeepSeek V4.1 adapter do: each expert is installed at most
once per call, chunks run under the module's own forward with one route per
row, and the weighted sum, when the caller asked for it, is applied to the
reassembled routes the way the model does when the kernel is unavailable.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import wait
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...scheduler import _sync_and_clear_cache
from ..moe_expert_offload import (
    _DTYPES,
    CheckpointExpertStore,
    _GLUStoreView,
    _io_batch,
    _io_pool,
    _minimum_experts,
    _resolve_model_dir,
)
from .kernels import fast as glm_fast

logger = logging.getLogger(__name__)

# checkpoint projections, in the order the fused row concatenates them
_SOURCES = ("gate_proj", "up_proj", "down_proj")

# Misses are positional reads through the common store on its shared pool
# (``OMLX_MOE_OFFLOAD_IO_WORKERS`` / ``OMLX_MOE_OFFLOAD_IO_BATCH``), the
# DeepSeek V4.1 adapter's shape: the store's memmap path faults a 20 MiB
# expert in 16 KiB pages on the compute thread, which measured 0.4 GB/s on
# the GLM-5.2 checkpoint against the 8 GB/s and more a positional read of the
# whole slab gets from the same drive. GLM experts are large, so on top of the
# pool's expert-count window at most INFLIGHT_BYTES of payload sit ahead of
# the slot writes.
INFLIGHT_BYTES = 512 * 1024 * 1024


def is_glm_switch_glu(obj) -> bool:
    """The GLM package's SwitchGLU, fused or not (never a wrapped one)."""
    return (
        type(obj).__name__ == "SwitchGLU"
        and type(obj).__module__ == "omlx.patches.glm_moe_dsa.switch_layers"
    )


def _layout(glu) -> list[tuple[str, tuple[str, ...]]]:
    """``(module projection, checkpoint projections it is built from)``."""
    if "gate_up_proj" in glu:
        return [
            ("gate_up_proj", ("gate_proj", "up_proj")),
            ("down_proj", ("down_proj",)),
        ]
    return [(p, (p,)) for p in _SOURCES]


def _fields(lin) -> tuple[str, ...]:
    return ("weight", "scales") + (("biases",) if lin.get("biases") is not None else ())


def _is_quantized(lin) -> bool:
    return type(lin).__name__ == "QuantizedSwitchLinear" and all(
        hasattr(lin, a) for a in ("group_size", "bits", "mode")
    )


def resolve_view(glu, store: CheckpointExpertStore, path: str):
    """Validate the checkpoint against the module; ``(view, None)`` or ``(None, reason)``.

    The checkpoint must hold the split ``gate_proj``/``up_proj``/``down_proj``
    with shapes the module's (possibly fused) projections are built from, in
    a storage dtype the store can read, in either layout: stacked under the
    module's tree path, or one tensor per expert under its parent
    (``<parent>.experts.<e>.<proj>.<field>``, which the loader stacks). The
    per-expert layout is verified expert by expert, as the common adapter
    does, so a checkpoint missing one expert is skipped rather than read.
    """
    layout = _layout(glu)
    for lin_name, _ in layout:
        lin = glu.get(lin_name)
        if lin is None or not _is_quantized(lin):
            return None, f"{lin_name} is not a QuantizedSwitchLinear"
        if "bias" in lin:
            return None, f"{lin_name} has per-expert bias (unsupported)"
    n_experts = glu[layout[0][0]]["weight"].shape[0]
    stacked = _GLUStoreView(store, path)
    parent = path.rsplit(".", 1)[0] if "." in path else ""
    view = (
        stacked
        if stacked.has("gate_proj", "weight")
        else _GLUStoreView(store, parent, per_expert=True)
    )
    for lin_name, sources in layout:
        lin = glu[lin_name]
        for field in _fields(lin):
            module_shape = tuple(lin[field].shape)
            if module_shape[1] % len(sources):
                return None, f"{lin_name}.{field} rows do not split into {sources}"
            want = (
                module_shape[0],
                module_shape[1] // len(sources),
                *module_shape[2:],
            )
            for src in sources:
                if view is stacked:
                    checks = [(view._name(src, field, 0), want)]
                else:
                    checks = [
                        (view._name(src, field, e), want[1:]) for e in range(n_experts)
                    ]
                for name, want_shape in checks:
                    if not store.has(name):
                        return None, f"checkpoint has no tensor {name!r}"
                    shape, dtype = store.spec(name)
                    if shape != want_shape:
                        return None, f"{name!r} shape {shape} != expected {want_shape}"
                    if dtype not in _DTYPES:
                        return None, f"{name!r} has unsupported dtype {dtype!r}"
    return view, None


class _SlotCache:
    """LRU slots living inside the module's own projection tensors."""

    moe_offload_cache = True

    def __init__(self, glu, capacity: int, view: _GLUStoreView):
        self.glu = glu
        self.view = view
        self.layout = _layout(glu)
        self.n_experts = glu[self.layout[0][0]]["weight"].shape[0]
        self.capacity = min(capacity, self.n_experts)
        self.resident: dict[str, list] = {}
        for lin_name, _ in self.layout:
            lin = glu[lin_name]
            arrays = []
            for field in _fields(lin):
                src = lin[field]
                slots = mx.zeros((self.capacity, *src.shape[1:]), dtype=src.dtype)
                setattr(lin, field, slots)  # drops the lazy full-size array
                arrays.append(slots)
            self.resident[lin_name] = arrays
        self.slot_of: dict[int, int] = {}  # expert id -> slot, LRU ordered
        self.free = list(range(self.capacity))
        self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self.hits = self.misses = 0
        self.fetched_bytes = 0
        self.warm = False
        # (module projection, field, row offset, row count, checkpoint slab)
        # for one expert, in slot-write order; the byte total sizes the window.
        self._writes: list[tuple[str, str, int, int, str]] = []
        for lin_name, sources in self.layout:
            lin = glu[lin_name]
            rows = lin["weight"].shape[1] // len(sources)
            for field in _fields(lin):
                for j, src in enumerate(sources):
                    self._writes.append((lin_name, field, j * rows, rows, src))
        self.expert_bytes = sum(
            view.plan(src, field, 0).nbytes for _, field, _, _, src in self._writes
        )

    def ensure(self, idx: mx.array) -> None:
        if self.warm:
            return
        self.ensure_ids(idx.reshape(-1).tolist())

    def ensure_ids(self, ids) -> None:
        """Make every expert in ``ids`` resident.

        Two passes, as in the V4.1 adapter: hits are touched first so the
        whole working set is protected from eviction, then the misses' reads
        start on the shared pool, at most the pool's window and
        ``INFLIGHT_BYTES`` ahead of the serial installs, which write slots in
        the order the misses were seen so victims and counters match a serial
        fetch. Without a pool (``OMLX_MOE_OFFLOAD_IO_WORKERS`` <= 1) each miss
        is read inline. A failed read leaves completed installs intact, and
        every read this call started is drained before it raises.
        """
        needed = list(dict.fromkeys(int(e) for e in ids))
        misses = []
        for e in needed:
            if e in self.slot_of:
                self.slot_of[e] = self.slot_of.pop(e)  # re-insert: LRU order
                self.hits += 1
            else:
                misses.append(e)
        if not misses:
            return
        protected = set(needed)
        pool = _io_pool()
        window = 0
        if pool is not None:
            window = max(1, min(_io_batch(), INFLIGHT_BYTES // max(1, self.expert_bytes)))
        pending: dict[int, list] = {}
        submitted = 0

        def plans(e):
            return [
                (write, self.view.plan(write[4], write[1], e)) for write in self._writes
            ]

        def submit(limit):
            nonlocal submitted
            if pool is None:
                return
            while submitted < min(limit, len(misses)):
                e = misses[submitted]
                submitted += 1
                pending[e] = [
                    (write, plan, pool.submit(CheckpointExpertStore.read, plan))
                    for write, plan in plans(e)
                ]

        try:
            submit(window)
            for done, e in enumerate(misses):
                # Refill before this expert's writes so at most ``window``
                # experts' bytes exist at once, counting the one written here.
                submit(done + window)
                if e in pending:
                    raws = [(write, plan, f.result()) for write, plan, f in pending[e]]
                else:
                    raws = [
                        (write, plan, CheckpointExpertStore.read(plan))
                        for write, plan in plans(e)
                    ]
                if self.free:
                    slot = self.free.pop()
                else:
                    victim = next(v for v in self.slot_of if v not in protected)
                    slot = self.slot_of.pop(victim)
                    self.map[victim] = -1
                for (lin_name, field, row0, rows, _), plan, raw in raws:
                    target = self.glu[lin_name][field]
                    array = CheckpointExpertStore.to_mx(plan, raw)
                    if row0 == 0 and rows == target.shape[1]:
                        target[slot] = array
                    else:
                        target[slot, row0 : row0 + rows] = array
                self.slot_of[e] = slot
                self.map[e] = slot
                self.misses += 1
                self.fetched_bytes += sum(plan.nbytes for _, plan, _ in raws)
                pending.pop(e, None)
                del raws
        finally:
            futures = [f for group in pending.values() for _, _, f in group]
            for future in futures:
                future.cancel()
            if futures:
                wait(futures)
        self.warm = len(self.slot_of) == self.n_experts


class OffloadedSwitchGLU(nn.Module):
    """GLM SwitchGLU whose experts live in a :class:`_SlotCache`."""

    def __init__(self, glu, capacity: int, view: _GLUStoreView):
        super().__init__()
        # A plain attribute, like the common adapter: the module with the slot
        # tensors stays out of the tree so parameter walks see the wrapper.
        self.cache = _SlotCache(glu, capacity, view)

    def _forward_expert_major(self, flat_x: mx.array, ids: list[int], k: int):
        """Routes sorted by expert, cut into chunks of ``capacity`` experts."""
        c = self.cache
        d_model = flat_x.shape[-1]
        ids_np = np.asarray(ids, dtype=np.int64)
        order = np.argsort(ids_np, kind="stable")
        sorted_ids = ids_np[order]
        run_starts = np.flatnonzero(np.diff(sorted_ids)) + 1
        run_starts = np.concatenate(([0], run_starts))
        cuts = run_starts[:: c.capacity].tolist() + [len(ids)]
        outs = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            chunk_ids = sorted_ids[start:end]
            c.ensure_ids(np.unique(chunk_ids).tolist())
            slots = mx.take(c.map, mx.array(chunk_ids, dtype=mx.int32))
            t_idx = mx.array(order[start:end] // k, dtype=mx.int32)
            xe = mx.take(flat_x, t_idx, axis=0)
            o = c.glu(xe, slots.reshape(-1, 1))[:, 0, :]
            mx.eval(o)
            outs.append(o)
        out = mx.concatenate(outs, axis=0)
        inverse = mx.array(np.argsort(order, kind="stable"), dtype=mx.int32)
        return mx.take(out, inverse, axis=0).reshape(-1, k, d_model)

    def __call__(self, x: mx.array, indices: mx.array, scores=None, weighted_sum=False):
        c = self.cache
        flat_i = indices.reshape(-1, indices.shape[-1])
        n_tok, k = flat_i.shape
        if k > c.capacity:
            raise ValueError("Expert cache capacity is smaller than routing top-k")
        fits = n_tok * k <= c.capacity or n_tok == 1
        ids = None
        if not fits:
            ids = flat_i.reshape(-1).tolist()
            fits = len(set(ids)) <= c.capacity
        if fits:
            c.ensure(indices)
            slots = mx.take(c.map, indices)
            return c.glu(x, slots, scores=scores, weighted_sum=weighted_sum)
        y = self._forward_expert_major(x.reshape(-1, x.shape[-1]), ids, k)
        y = y.reshape(indices.shape + (x.shape[-1],))
        # Sum exactly when the module's own forward would have: it returns the
        # routes unsummed unless the call is sorted and the native kernel is
        # present, and the caller applies the scores itself in that case.
        if (
            weighted_sum
            and scores is not None
            and indices.size >= 64
            and hasattr(glm_fast, "glm_moe_weighted_sum")
        ):
            y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        return y


def _iter_glm_switch_glus(model):
    seen = set()

    def walk(parent, key, obj, path):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if is_glm_switch_glu(obj):
            yield (parent, key, obj, path)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from walk(obj, k, v, f"{path}.{k}" if path else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from walk(obj, i, v, f"{path}.{i}")

    yield from walk(None, None, model, "")


def apply_glm_moe_expert_offload(
    model, model_path: str | Path, resident_fraction: float = 0.25
) -> int:
    """Wrap every covered GLM SwitchGLU; returns the number wrapped.

    Same contract as ``apply_moe_expert_offload``: runs before lazy weights
    materialize, honors the kill switch, and skips (with a logged reason)
    any module the checkpoint does not cover.
    """
    if os.environ.get("OMLX_MOE_EXPERT_OFFLOAD", "1") == "0":
        return 0
    targets = list(_iter_glm_switch_glus(model))
    if not targets:
        return 0
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return 0
    minimum = _minimum_experts(model_dir)
    store = CheckpointExpertStore(model_dir)
    if not store:
        logger.warning("glm moe expert offload: no safetensors under %s", model_dir)
        return 0
    wrapped = 0
    total_bytes = resident_bytes = 0
    for parent, key, glu, path in targets:
        view, reason = resolve_view(glu, store, path)
        if view is None:
            logger.info("glm moe expert offload: skipping %s (%s)", path, reason)
            continue
        n_experts = glu[_layout(glu)[0][0]]["weight"].shape[0]
        capacity = min(n_experts, max(minimum, round(n_experts * resident_fraction)))
        layer_bytes = sum(
            int(np.prod(lin[f].shape)) * lin[f].dtype.size
            for lin_name, _ in _layout(glu)
            for lin in (glu[lin_name],)
            for f in _fields(lin)
        )
        total_bytes += layer_bytes
        resident_bytes += layer_bytes * capacity // n_experts
        new = OffloadedSwitchGLU(glu, capacity, view)
        if isinstance(parent, nn.Module):
            setattr(parent, key, new)
        else:
            parent[key] = new
        wrapped += 1
        _sync_and_clear_cache()
    if wrapped:
        logger.info(
            "glm moe expert offload: wrapped %d layers at %.1f%% residency "
            "(expert tables: %.2f GB total, %.2f GB resident)",
            wrapped,
            100 * resident_fraction,
            total_bytes / 1e9,
            resident_bytes / 1e9,
        )
    return wrapped


__all__ = [
    "OffloadedSwitchGLU",
    "apply_glm_moe_expert_offload",
    "is_glm_switch_glu",
    "resolve_view",
]
