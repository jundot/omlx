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
slabs from the checkpoint's own safetensors (the split layout the
checkpoints ship) and writes the two halves of the fused row in place.

Over-capacity prefill, where one call routes to more distinct experts than
the cache holds, is chunked on expert boundaries exactly as the common
adapter and the DeepSeek V4.1 adapter do: each expert is installed at most
once per call, chunks run under the module's own forward with one route per
row, and the weighted sum, when the caller asked for it, is applied to the
reassembled routes the way the model does when the kernel is unavailable.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ...scheduler import _sync_and_clear_cache
from ..moe_expert_offload import (
    _DTYPES,
    CheckpointExpertStore,
    _GLUStoreView,
    _minimum_experts,
    _resolve_model_dir,
)
from .kernels import fast as glm_fast

logger = logging.getLogger(__name__)

# checkpoint projections, in the order the fused row concatenates them
_SOURCES = ("gate_proj", "up_proj", "down_proj")

# Misses are read with positional reads on a small pool, the DeepSeek V4.1
# adapter's shape: the common store's memmap path faults a 20 MiB expert in
# 16 KiB pages on the compute thread, which measured 0.4 GB/s on the GLM-5.2
# checkpoint against the 8 GB/s and more a positional read of the whole slab
# gets from the same drive. At most INFLIGHT_BYTES of payload sit ahead of
# the slot writes.
INFLIGHT_BYTES = 512 * 1024 * 1024
EXPERT_IO_WORKERS = 24
_EXPERT_IO_POOL = ThreadPoolExecutor(
    max_workers=EXPERT_IO_WORKERS, thread_name_prefix="glm-expert-io"
)

Slab = namedtuple("Slab", "name fd offset nbytes np_dtype mx_view shape")


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
    stacked under the module's tree path with shapes the module's (possibly
    fused) projections are built from, in a storage dtype the store can read.
    """
    layout = _layout(glu)
    for lin_name, _ in layout:
        lin = glu.get(lin_name)
        if lin is None or not _is_quantized(lin):
            return None, f"{lin_name} is not a QuantizedSwitchLinear"
        if "bias" in lin:
            return None, f"{lin_name} has per-expert bias (unsupported)"
    view = _GLUStoreView(store, path)
    for lin_name, sources in layout:
        lin = glu[lin_name]
        for field in _fields(lin):
            module_shape = tuple(lin[field].shape)
            want = (
                module_shape[0],
                module_shape[1] // len(sources),
                *module_shape[2:],
            )
            if module_shape[1] % len(sources):
                return None, f"{lin_name}.{field} rows do not split into {sources}"
            for src in sources:
                name = view._name(src, field, 0)
                if not store.has(name):
                    return None, f"checkpoint has no tensor {name!r}"
                shape, dtype = store.spec(name)
                if shape != want:
                    return None, f"{name!r} shape {shape} != expected {want}"
                if dtype not in _DTYPES:
                    return None, f"{name!r} has unsupported dtype {dtype!r}"
    return view, None


class _SlabReader:
    """Positional reads of one expert's slabs from the stacked checkpoint."""

    def __init__(self, store: CheckpointExpertStore, prefix: str):
        self._store = store
        self._prefix = prefix
        self._fds: dict[Path, int] = {}
        self._lock = Lock()

    def _fd(self, shard: Path) -> int:
        with self._lock:
            fd = self._fds.get(shard)
            if fd is None:
                fd = self._fds[shard] = os.open(shard, os.O_RDONLY)
            return fd

    def plan(self, proj: str, field: str, expert: int) -> Slab:
        name = f"{self._prefix}.{proj}.{field}"
        shard, dtype, shape, offset = self._store._specs[name]
        np_dtype, mx_view = _DTYPES[dtype]
        nbytes = int(np.prod(shape[1:])) * np.dtype(np_dtype).itemsize
        return Slab(
            name,
            self._fd(shard),
            offset + expert * nbytes,
            nbytes,
            np_dtype,
            mx_view,
            tuple(shape[1:]),
        )

    @staticmethod
    def read(slab: Slab) -> bytearray:
        """The slab's bytes; positional reads only, so any thread may call it."""
        buffer = bytearray(slab.nbytes)
        view = memoryview(buffer)
        done = 0
        while done < slab.nbytes:
            count = os.preadv(slab.fd, [view[done:]], slab.offset + done)
            if count <= 0:
                raise ValueError(f"Truncated tensor data: {slab.name}")
            done += count
        return buffer

    @staticmethod
    def to_array(slab: Slab, raw: bytearray) -> mx.array:
        out = mx.array(np.frombuffer(raw, dtype=slab.np_dtype).reshape(slab.shape))
        return out.view(slab.mx_view) if slab.mx_view is not None else out

    def close(self) -> None:
        with self._lock:
            fds, self._fds = self._fds, {}
        for fd in fds.values():
            with contextlib.suppress(OSError):
                os.close(fd)


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
        self.reader = _SlabReader(view._store, view._prefix)
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
            self.reader.plan(src, field, 0).nbytes
            for _, field, _, _, src in self._writes
        )

    def ensure(self, idx: mx.array) -> None:
        if self.warm:
            return
        self.ensure_ids(idx.reshape(-1).tolist())

    def ensure_ids(self, ids) -> None:
        """Make every expert in ``ids`` resident.

        Two passes, as in the V4.1 adapter: hits are touched first so the
        whole working set is protected from eviction, then the misses' reads
        start on the pool, at most ``INFLIGHT_BYTES`` ahead of the serial
        installs, which write slots in the order the misses were seen so
        victims and counters match a serial fetch. A failed read leaves
        completed installs intact, and every read this call started is
        drained before it raises.
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
        pending: dict[int, list] = {}
        window = max(1, INFLIGHT_BYTES // max(1, self.expert_bytes))
        submitted = 0

        def submit(limit):
            nonlocal submitted
            while submitted < min(limit, len(misses)):
                e = misses[submitted]
                submitted += 1
                pending[e] = [
                    (write, slab, _EXPERT_IO_POOL.submit(_SlabReader.read, slab))
                    for write in self._writes
                    for slab in (self.reader.plan(write[4], write[1], e),)
                ]

        try:
            submit(window)
            for done, e in enumerate(misses):
                # Refill before this expert's writes so at most ``window``
                # experts' bytes exist at once, counting the one written here.
                submit(done + window)
                raws = [(write, slab, f.result()) for write, slab, f in pending[e]]
                if self.free:
                    slot = self.free.pop()
                else:
                    victim = next(v for v in self.slot_of if v not in protected)
                    slot = self.slot_of.pop(victim)
                    self.map[victim] = -1
                for (lin_name, field, row0, rows, _), slab, raw in raws:
                    target = self.glu[lin_name][field]
                    array = _SlabReader.to_array(slab, raw)
                    if row0 == 0 and rows == target.shape[1]:
                        target[slot] = array
                    else:
                        target[slot, row0 : row0 + rows] = array
                self.slot_of[e] = slot
                self.map[e] = slot
                self.misses += 1
                self.fetched_bytes += sum(slab.nbytes for _, slab, _ in raws)
                del pending[e], raws
        finally:
            for group in pending.values():
                for _, _, f in group:
                    if not f.cancel():
                        f.exception()
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
