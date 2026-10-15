# SPDX-License-Identifier: Apache-2.0
"""MoE expert offloading: stream non-resident experts from the checkpoint.

For Mixture-of-Experts models whose expert tables do not fit in memory, keep
only ``resident_fraction`` of each layer's experts in a contiguous slot tensor
and fetch the rest on demand from the model's own safetensors shards
(positional slab reads — no converted copy of the checkpoint, no write
path). Routing is computed exactly as shipped; a cache miss changes *when*
an expert's weights are read, never *which* expert runs. Accuracy is
therefore preserved by construction, at a latency cost (measured on a 26B/128-expert model: accuracy
flat down to 12% residency, throughput falling roughly as memory^0.5).

Applied once post-load, before lazy weights materialize: each stock
``SwitchGLU`` whose projections are quantized and fully covered by the
checkpoint is replaced with an ``OffloadSwitchGLU``. The original module —
and with it the lazy references to the full expert tensors — is dropped, so
the non-resident experts are never materialized at all. Instances that are
unsupported (non-quantized, fused ``gate_up_proj``, per-expert ``bias``, or
tensor names the checkpoint does not contain) are left untouched.

Numerical contract, measured against the pinned mlx-lm: decode and unsorted
prefill are bit-identical to the stock path at any residency; the sorted
prefill kernel (``indices.size >= 64``) is presentation-invariant at real
model dimensions, so full-residency prefill is bit-identical too. Partial
residency can legitimately chunk a prefill below the sort threshold, where
the sorted and unsorted gather_qmm kernels differ by ~4e-3 absolute on
~5-magnitude outputs — rounding, not routing (see the test suite's
assertion policy).
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
import threading
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import (
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)

from ..scheduler import _sync_and_clear_cache

logger = logging.getLogger(__name__)

_PROJS = ("gate_proj", "up_proj", "down_proj")

_PER_EXPERT_PROJ_RE = re.compile(
    r"^(?P<parent>.+)\.experts\.(?P<idx>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<field>weight|scales|biases)$"
)

# A pending slab read: everything needed to turn a byte range of a shard
# into an mx.array, and nothing that touches MLX or cache state — so the
# ``os.pread`` half can run on any thread.
_ReadPlan = namedtuple("_ReadPlan", "fd offset nbytes np_dtype mx_view shape")

# safetensors dtype tag -> (numpy transport dtype, mlx dtype to view as).
# bf16 has no numpy equivalent, so it travels as raw uint16 and is
# reinterpreted on the mlx side; everything else converts directly.
_DTYPES = {
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, None),
    "F32": (np.float32, None),
    "U32": (np.uint32, None),
    "I32": (np.int32, None),
    "U8": (np.uint8, None),
}


def _minimum_experts(model_dir):
    path = Path(model_dir) / "config.json"
    if not path.exists():
        return 8
    config = json.loads(path.read_text())
    text = config.get("text_config", config)
    return max(
        8,
        *(
            int(text.get(k) or 0)
            for k in (
                "num_experts_per_tok",
                "num_experts_per_token",
                "n_activated_experts",
            )
        ),
    )


class CheckpointExpertStore:
    """Per-expert slab reads from a model directory's safetensors shards.

    Expert tables are stored stacked with the expert axis leading
    (``[num_experts, ...]``), so one expert is a contiguous byte range in the
    shard. Each shard is opened once read-only and read with ``os.pread``,
    which takes the offset as an argument instead of carrying one on the
    descriptor — so reads of different experts are safe to run concurrently,
    off the calling thread. The read splits in two: :meth:`read` produces
    bytes and touches neither MLX nor cache state (any thread), :meth:`to_mx`
    turns those bytes into an array (the calling thread's stream).
    """

    def __init__(self, model_path: str | Path):
        self._specs: dict[str, tuple[Path, str, tuple[int, ...], int]] = {}
        self._fds: dict[Path, int] = {}
        model_path = Path(model_path)
        for shard in sorted(model_path.glob("*.safetensors")):
            with open(shard, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            data_base = 8 + header_len
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                self._specs[name] = (
                    shard,
                    spec["dtype"],
                    tuple(spec["shape"]),
                    data_base + spec["data_offsets"][0],
                )
            # Opened once here, read-only and never mutated afterwards: the
            # store is fully populated before any fetch, which is what makes
            # concurrent reads against it safe without a lock.
            self._fds[shard] = os.open(shard, os.O_RDONLY)

    def __del__(self):
        for fd in self._fds.values():
            try:
                os.close(fd)
            except Exception:
                pass

    def __bool__(self) -> bool:
        return bool(self._specs)

    def has(self, name: str) -> bool:
        return name in self._specs

    def spec(self, name: str) -> tuple[tuple[int, ...], str]:
        _, dtype, shape, _ = self._specs[name]
        return shape, dtype

    def _plan(
        self, name: str, start_elem: int, n_elems: int, out_shape: tuple[int, ...]
    ) -> _ReadPlan:
        shard, dtype, _, offset = self._specs[name]
        np_dtype, mx_view = _DTYPES[dtype]
        itemsize = np.dtype(np_dtype).itemsize
        return _ReadPlan(
            self._fds[shard],
            offset + start_elem * itemsize,
            n_elems * itemsize,
            np_dtype,
            mx_view,
            out_shape,
        )

    def plan_expert(self, name: str, expert: int) -> _ReadPlan:
        """Plan one expert's slab of a stacked ``[num_experts, ...]`` tensor."""
        _, _, shape, _ = self._specs[name]
        slab = int(np.prod(shape[1:]))
        return self._plan(name, expert * slab, slab, shape[1:])

    def plan_tensor(self, name: str) -> _ReadPlan:
        """Plan a whole tensor (per-expert checkpoint layouts)."""
        _, _, shape, _ = self._specs[name]
        return self._plan(name, 0, int(np.prod(shape)), shape)

    @staticmethod
    def read(plan: _ReadPlan) -> bytes:
        """The plan's raw bytes. Thread-safe: positional reads only."""
        chunks = []
        got = 0
        while got < plan.nbytes:
            chunk = os.pread(plan.fd, plan.nbytes - got, plan.offset + got)
            if not chunk:
                raise OSError(f"short read of {plan.nbytes} bytes at {plan.offset}")
            chunks.append(chunk)
            got += len(chunk)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    @staticmethod
    def to_mx(plan: _ReadPlan, raw: bytes) -> mx.array:
        """Reinterpret a plan's bytes as its array (host-side, one copy)."""
        out = mx.array(np.frombuffer(raw, dtype=plan.np_dtype).reshape(plan.shape))
        return out.view(plan.mx_view) if plan.mx_view is not None else out

    def fetch_expert(self, name: str, expert: int) -> mx.array:
        """One expert's slab of a stacked ``[num_experts, ...]`` tensor."""
        plan = self.plan_expert(name, expert)
        return self.to_mx(plan, self.read(plan))

    def fetch_tensor(self, name: str) -> mx.array:
        """A whole tensor (per-expert checkpoint layouts)."""
        plan = self.plan_tensor(name)
        return self.to_mx(plan, self.read(plan))


class _GLUStoreView:
    """Adapt the flat store to one SwitchGLU's checkpoint naming scheme.

    Two layouts exist in the wild. Newer conversions store experts stacked
    under the module-tree name (``<glu>.gate_proj.weight`` with shape
    ``[E, ...]``). Older ones store one tensor per expert under the GLU's
    parent (``<parent>.experts.<e>.gate_proj.weight``), which mlx-lm's
    ``sanitize()`` stacks at load — so the stacked names never exist in the
    file. The view hides the difference from :class:`ExpertCache`.
    """

    def __init__(self, store: CheckpointExpertStore, prefix: str,
                 per_expert: bool = False):
        self._store = store
        self._prefix = prefix  # stacked: the GLU path; per-expert: its parent
        self._per_expert = per_expert

    def _name(self, proj: str, field: str, expert: int) -> str:
        if self._per_expert:
            return f"{self._prefix}.experts.{expert}.{proj}.{field}"
        return f"{self._prefix}.{proj}.{field}"

    def has(self, proj: str, field: str) -> bool:
        return self._store.has(self._name(proj, field, 0))

    def plan(self, proj: str, field: str, expert: int) -> _ReadPlan:
        if self._per_expert:
            return self._store.plan_tensor(self._name(proj, field, expert))
        return self._store.plan_expert(self._name(proj, field, 0), expert)

    def fetch(self, proj: str, field: str, expert: int) -> mx.array:
        if self._per_expert:
            return self._store.fetch_tensor(self._name(proj, field, expert))
        return self._store.fetch_expert(self._name(proj, field, 0), expert)


# One reader pool for the whole process. A miss is IO, not compute: the
# useful width is the storage queue depth, so it is wider than the core
# count. ``_IO_WORKERS`` <= 1 keeps the serial path and creates no threads.
# At most 4 x workers experts' payloads are in flight, which is what bounds
# the extra host memory the pipeline holds.
_IO_WORKERS = 12
_IO_LOCK = threading.Lock()
_IO_POOL: ThreadPoolExecutor | None = None
_IO_BATCH = 0
_IO_CONFIGURED = False


def _io_pool() -> ThreadPoolExecutor | None:
    """The shared reader pool, or ``None`` when reads must stay serial."""
    global _IO_POOL, _IO_BATCH, _IO_CONFIGURED
    with _IO_LOCK:
        if not _IO_CONFIGURED:
            _IO_CONFIGURED = True
            workers = _IO_WORKERS
            if workers > 1:
                _IO_BATCH = 4 * workers
                _IO_POOL = ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="omlx-moe-io"
                )
        return _IO_POOL


def _io_batch() -> int:
    """Experts whose reads may be in flight at once."""
    _io_pool()
    return _IO_BATCH


def _shutdown_io_pool() -> None:
    """Drop the pool; the next fetch rebuilds it from ``_IO_WORKERS`` (tests)."""
    global _IO_POOL, _IO_BATCH, _IO_CONFIGURED
    with _IO_LOCK:
        pool, _IO_POOL, _IO_BATCH, _IO_CONFIGURED = _IO_POOL, None, 0, False
    if pool is not None:
        pool.shutdown(wait=True)


# Eviction score decay (the best setting of a replayed-trace grid).
_SCORE_DECAY = 0.7
_SCORE_DECAY_EVERY = 4

# Poll interval while the overlap path waits on reads (see _GpuKeepalive).
_KEEPALIVE_PERIOD_S = 50e-6
_KEEPALIVE_LOCAL = threading.local()
# A decode step overlaps only when its first missing expert is still being read
# this long after the reads were issued. Faster reads (page cache, fast internal
# storage) keep the serial order: an idle gap this short barely lowers the GPU
# clock, and splitting the gather in two would cost more than it hides.
_OVERLAP_GRACE_S = 0.5e-3


class _GpuKeepalive:
    """Keep the GPU clocked while the calling thread waits on expert reads.

    Apple GPUs lower their clock after a couple of milliseconds idle, and the
    next burst of work then runs slower until the clock recovers. A decode
    step's reads leave the GPU idle for that long, so the next layer's GPU
    work pays for it; one short gather at the start of the wait does not
    prevent it. While reads are in flight this submits a trivial kernel every
    ``_KEEPALIVE_PERIOD_S``, at most two in flight, so the GPU never goes
    fully idle. The kernels touch nothing the model reads. One per thread:
    engines on different threads use different streams.
    """

    def __init__(self):
        self._x = mx.zeros((8,), dtype=mx.float32)
        self._inflight: list = []

    def wait(self, futures) -> None:
        """Block until ``futures`` finish, keeping the GPU busy meanwhile."""
        while True:
            _, pending = wait(futures, timeout=_KEEPALIVE_PERIOD_S)
            if not pending:
                return
            y = self._x + 1
            mx.async_eval(y)
            self._inflight.append(y)
            if len(self._inflight) > 2:
                mx.eval(self._inflight.pop(0))


def _gpu_keepalive() -> _GpuKeepalive:
    ka = getattr(_KEEPALIVE_LOCAL, "ka", None)
    if ka is None:
        ka = _KEEPALIVE_LOCAL.ka = _GpuKeepalive()
    return ka


def _drain(pending: dict) -> None:
    """Cancel or finish unused reads, so no read outlives the shard descriptors."""
    futures = [f for group in pending.values() for _, _, _, f in group]
    for future in futures:
        future.cancel()
    if futures:
        wait(futures)


class ExpertCache:
    """Contiguous resident slots over one layer's experts.

    A miss evicts the expert with the lowest decayed routing count (+1 per
    route, x``_SCORE_DECAY`` every ``_SCORE_DECAY_EVERY`` calls).

    Holds no reference to the wrapped module's expert tensors — only the
    resident slots and the store view. That is the difference between saving
    memory and adding it: keeping the source tensors referenced alongside the
    slots costs the full expert set *plus* the cache.
    """

    moe_offload_cache = True  # walked by materialize_offload_state / stats

    def __init__(
        self, glu: SwitchGLU, capacity: int, disk: _GLUStoreView, floor: int = 1
    ):
        self.n_experts = glu.gate_proj["weight"].shape[0]
        self.full_capacity = min(capacity, self.n_experts)
        # The capacity release_slots() shrinks to: the routing top-k floor.
        self.floor = min(max(floor, 1), self.full_capacity)
        self.projs = _PROJS
        self.disk = disk
        self._slot_specs = {
            name: [
                None if lin.get(f) is None else (lin[f].shape[1:], lin[f].dtype)
                for f in ("weight", "scales", "biases")
            ]
            for name in self.projs
            for lin in (getattr(glu, name),)
        }
        # Per-projection quantization metadata: mixed-bit checkpoints (e.g.
        # oQ profiles with 8-bit down_proj over 4-bit gate/up) are valid and
        # must not inherit gate_proj's parameters.
        self.qparams = {
            name: (
                getattr(glu, name).group_size,
                getattr(glu, name).bits,
                getattr(glu, name).mode,
            )
            for name in self.projs
        }
        self.score = np.zeros(self.n_experts, dtype=np.float32)
        self._calls = 0
        self.hits = self.misses = 0
        self.fetched_bytes = 0
        self._allocate(self.full_capacity)
        self.expert_bytes = sum(plan.nbytes for _, _, plan in self._plans(0))

    def _allocate(self, capacity: int) -> None:
        """Replace the slots with ``capacity`` empty ones, dropping every expert."""
        self.capacity = capacity
        self.resident: dict[str, list] = {
            name: [
                None if spec is None else mx.zeros((capacity,) + spec[0], dtype=spec[1])
                for spec in specs
            ]
            for name, specs in self._slot_specs.items()
        }
        self.slot_of: dict[int, int] = {}  # Expert id -> slot
        self.slot_expert = np.full(capacity, -1, dtype=np.int64)
        self.free = list(range(capacity))
        self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self.warm = False

    def release_slots(self) -> int:
        """Shrink to ``floor`` empty slots; return the slot bytes given up."""
        if self.capacity <= self.floor:
            return 0
        freed = (self.capacity - self.floor) * self.expert_bytes
        self._allocate(self.floor)
        return freed

    def restore_slots(self) -> None:
        """Return to the full capacity after :meth:`release_slots`."""
        if self.capacity < self.full_capacity:
            self._allocate(self.full_capacity)

    def _plans(self, e: int) -> list:
        """Read plans for expert ``e``'s tensors, in slot-write order."""
        out = []
        for name in self.projs:
            rb = self.resident[name][2]
            out.append((name, 0, self.disk.plan(name, "weight", e)))
            out.append((name, 1, self.disk.plan(name, "scales", e)))
            if rb is not None and self.disk.has(name, "biases"):
                out.append((name, 2, self.disk.plan(name, "biases", e)))
        return out

    def _submit(self, pool, e: int) -> list:
        """Start reading expert ``e``; a task per tensor keeps its reads parallel."""
        return [
            (name, field, plan, pool.submit(CheckpointExpertStore.read, plan))
            for name, field, plan in self._plans(e)
        ]

    def _reserve(self, protected=frozenset()) -> int:
        """Claim a slot, evicting the lowest-count expert outside ``protected``."""
        if self.free:
            return self.free.pop()
        scores = self.score[self.slot_expert]
        for e in protected:
            slot = self.slot_of.get(e)
            if slot is not None:
                scores[slot] = np.inf
        slot = int(np.argmin(scores))
        old_e = int(self.slot_expert[slot])
        del self.slot_of[old_e]
        self.slot_expert[slot] = -1
        self.map[old_e] = -1
        return slot

    def _write(self, slot: int, payload: list) -> None:
        """Copy one expert's fetched bytes into ``slot``."""
        for name, field, plan, raw in payload:
            self.resident[name][field][slot] = CheckpointExpertStore.to_mx(plan, raw)

    def _install(
        self, e: int, payload: list | None = None, protected=frozenset()
    ) -> int:
        if payload is None:
            payload = [
                (n, f, pl, CheckpointExpertStore.read(pl))
                for n, f, pl in self._plans(e)
            ]
        slot = self._reserve(protected)
        try:
            self._write(slot, payload)
        except BaseException:
            # A partial write invalidates the evicted expert too.
            self.free.append(slot)
            raise
        self.slot_of[e] = slot
        self.slot_expert[slot] = e
        self.map[e] = slot
        self.fetched_bytes += self.expert_bytes
        self.warm = len(self.slot_of) == self.n_experts
        return slot

    def ensure(self, idx: mx.array) -> None:
        """Make every expert in ``idx`` resident; misses never evict the call's experts.

        Reads run on the IO pool; slot writes and eviction stay on this thread.
        """
        if self.warm:  # nothing can miss; skip it
            return
        self._ensure_ids(idx.reshape(-1).tolist())

    def _read_ahead(self, ids, limit: int) -> dict:
        """Start the first ``limit`` misses' reads without touching cache state.

        Pass the result to :meth:`_ensure_ids` as ``pending``."""
        pool = _io_pool()
        out: dict[int, list] = {}
        if pool is None:
            return out
        for e in dict.fromkeys(int(e) for e in ids):
            if len(out) >= limit:
                break
            if e not in self.slot_of:
                out[e] = self._submit(pool, e)
        return out

    def _ensure_ids(self, ids, on_slow_read=None, pending=None) -> bool:
        """:meth:`ensure` for routes already read back from the device.

        With ``on_slow_read`` (the decode overlap path), the call checks once,
        at its first miss and before any slot write, whether that expert's
        reads finish within ``_OVERLAP_GRACE_S`` of being issued. If they do
        not, ``on_slow_read()`` runs and may dispatch GPU work; what it returns
        is evaluated right before the first slot write, and the rest of the
        call's reads are waited on with the GPU kept busy (see
        :meth:`OffloadSwitchGLU._forward_overlap`). Returns whether that
        happened. ``pending`` holds reads from :meth:`_read_ahead`.
        """
        pending = dict(pending or {})
        needed = list(dict.fromkeys(int(e) for e in ids))
        if len(needed) > self.capacity:
            _drain(pending)
            raise ValueError("Expert cache capacity is smaller than the call's routes")
        np.add.at(self.score, np.asarray(ids, dtype=np.int64), 1.0)
        self._calls += 1
        if self._calls % _SCORE_DECAY_EVERY == 0:
            self.score *= _SCORE_DECAY
        misses = [e for e in needed if e not in self.slot_of]
        self.hits += len(needed) - len(misses)
        if not misses:
            _drain(pending)
            return False
        protected = frozenset(needed)
        pool = _io_pool()
        window = _io_batch()
        sent = 0
        keepalive = barrier = None
        slow = False

        def prefetch(upto: int) -> None:
            nonlocal sent
            if pool is None:
                return
            while sent < min(upto, len(misses)):
                e = misses[sent]
                sent += 1
                if e not in pending:
                    pending[e] = self._submit(pool, e)

        try:
            prefetch(window)
            deadline = time.perf_counter() + _OVERLAP_GRACE_S
            for done, e in enumerate(misses):
                self.misses += 1
                # Count the current payload in the window until its writes finish.
                prefetch(done + window)
                group = pending.get(e)
                if on_slow_read is not None:  # the first miss: nothing written yet
                    futures = [f for _, _, _, f in group or ()]
                    timeout = max(0.0, deadline - time.perf_counter())
                    if futures and wait(futures, timeout=timeout).not_done:
                        slow = True
                        barrier = on_slow_read()
                        keepalive = _gpu_keepalive()
                    on_slow_read = None
                if group is None:
                    self._install(e, protected=protected)
                    continue
                if keepalive is not None:
                    keepalive.wait([f for _, _, _, f in group])
                if barrier is not None:
                    mx.eval(barrier)  # right before the first slot write
                    barrier = None
                self._install(
                    e,
                    [(name, field, plan, f.result()) for name, field, plan, f in group],
                    protected,
                )
                del pending[e], group
        finally:
            _drain(pending)
        # No mx.eval here: installs are already-materialized host arrays, and
        # evaluating every resident tensor on every miss measured 22% slower
        # at identical peak memory. Prefill's transient is bounded by the
        # per-chunk eval in __call__, which is a different mechanism.
        return slow

    def qmm(
        self, name: str, x: mx.array, slots: mx.array, sorted_indices: bool = False
    ) -> mx.array:
        # sorted_indices selects a different kernel; the wrapper mirrors the
        # stock SwitchGLU's sort decision so the kernel choice — and with it
        # the numerics — matches the path the resident model would take.
        rw, rs, rb = self.resident[name]
        group_size, bits, mode = self.qparams[name]
        return mx.gather_qmm(
            x,
            rw,
            rs,
            rb,
            rhs_indices=slots,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
            sorted_indices=sorted_indices,
        )


# The stock SwitchGLU sorts routes from this many on (prefill); decode is below.
_SORT_MIN_ROUTES = 64


def _pad_floor(capacity: int) -> int:
    # GatherQMM uses sorted QMM only when B >= 16 and B / E >= 4
    # (E = resident slots); below that, padding would change kernels.
    return max(16, 4 * capacity)


def _expert_route_chunks(
    run_starts: np.ndarray, n_routes: int, capacity: int
) -> list[tuple[int, int]]:
    """Chunks of at most ``capacity`` experts, power-of-two sized if that pads less."""
    cuts = run_starts[::capacity].tolist() + [n_routes]
    whole = list(zip(cuts[:-1], cuts[1:]))
    floor = _pad_floor(capacity)
    if any(end - start < floor for start, end in whole):
        return whole
    packed = []
    start = 0
    while start < n_routes:
        first = int(np.searchsorted(run_starts, start, side="right")) - 1
        last = first + capacity
        limit = int(run_starts[last]) if last < len(run_starts) else n_routes
        size = 1 << ((limit - start).bit_length() - 1)
        tail = n_routes - start - size
        if 0 < tail < floor:
            size = size // 2 if size // 2 >= floor else limit - start
        if size < floor:
            return whole
        packed.append((start, start + size))
        start += size

    def padded_rows(chunks):
        return sum(1 << (end - start - 1).bit_length() for start, end in chunks)

    return packed if padded_rows(packed) < padded_rows(whole) else whole


class OffloadSwitchGLU(nn.Module):
    """SwitchGLU whose experts live in an :class:`ExpertCache`."""

    def __init__(
        self, glu: SwitchGLU, capacity: int, disk: _GLUStoreView, floor: int = 1
    ):
        super().__init__()
        self.cache = ExpertCache(glu, capacity, disk, floor)
        self.activation = glu.activation
        # Fetch/compute overlap in decode: the GPU stays busy while slow
        # reads of missing experts are pending.
        self._overlap = True

    def _glu(self, x: mx.array, slots: mx.array) -> mx.array:
        c = self.cache
        up = c.qmm("up_proj", x, slots)
        gate = c.qmm("gate_proj", x, slots)
        return c.qmm("down_proj", self.activation(up, gate), slots)

    def _glu_routes(self, flat_x: mx.array, flat: list, pos: list, k: int) -> mx.array:
        """The experts' output for the flat route positions ``pos`` only.

        Each route gets its token's row and its expert's slot, the layout of
        the expert-major prefill chunks: the same unsorted kernel as the full
        decode gather, so every row is bit-identical to it.
        """
        rows = mx.take(flat_x, mx.array([p // k for p in pos], dtype=mx.int32), axis=0)
        slots = mx.array([self.cache.slot_of[flat[p]] for p in pos], dtype=mx.int32)
        return self._glu(mx.expand_dims(rows, (-2, -3)), slots.reshape(-1, 1))

    def _forward_overlap(self, x: mx.array, indices: mx.array) -> mx.array:
        """Decode step that keeps the GPU busy while slow reads are pending.

        The serial step reads the route set back, reads the misses, then
        computes, and the GPU idles through the read. When the first missing
        expert arrives within ``_OVERLAP_GRACE_S`` of its reads being issued
        (page cache, fast storage) that is the right order, and this step
        takes it unchanged. When it does not, the GPU would idle for
        milliseconds, and Apple GPUs lower their clock after a couple of
        milliseconds idle, so the next layer's work runs slower too. Then the
        resident routes' ``gather_qmm`` is dispatched while the reads
        continue, a trivial kernel keeps the GPU clocked for the rest of the
        wait (:class:`_GpuKeepalive`), the missing routes are gathered once
        they are installed, and the layer's output is dispatched as soon as
        it is built. Every route is computed once, by the same kernel as the
        serial gather (``gather_qmm`` is per-row), so the output is
        bit-identical. The resident gather is evaluated right before the
        first slot write: while it still references the resident arrays a
        write cannot happen in place and copies the whole array (2.5 ms
        instead of 0.5 ms per expert at 2.8 MB experts).
        """
        c = self.cache
        flat = indices.reshape(-1).tolist()  # the sync
        hit = [e in c.slot_of for e in flat]  # residency before any install
        hit_pos = [p for p, h in enumerate(hit) if h]
        flat_x = x.reshape(-1, x.shape[-1])
        k = indices.shape[-1]
        y_hit = None

        def on_slow_read():
            nonlocal y_hit
            if hit_pos:
                y_hit = self._glu_routes(flat_x, flat, hit_pos, k)
                mx.async_eval(y_hit)
            return y_hit

        slow = c._ensure_ids(flat, on_slow_read)
        if y_hit is None:  # fast reads, or no resident route: one gather
            out = self._glu(mx.expand_dims(x, (-2, -3)), mx.take(c.map, indices))
            out = out.squeeze(-2)
        else:
            miss_pos = [p for p, h in enumerate(hit) if not h]
            y = mx.concatenate([y_hit, self._glu_routes(flat_x, flat, miss_pos, k)])
            back = [0] * len(flat)  # route p's row in y
            for row, p in enumerate(hit_pos + miss_pos):
                back[p] = row
            out = mx.take(y, mx.array(back, dtype=mx.int32), axis=0)
            out = out.reshape(indices.shape + (y.shape[-1],))
        if slow:
            mx.async_eval(out)
        return out

    def _forward(self, x: mx.array, indices: mx.array) -> mx.array:
        c = self.cache
        if (
            self._overlap
            and not c.warm
            and indices.size < _SORT_MIN_ROUTES
            and _io_pool() is not None
        ):
            return self._forward_overlap(x, indices)
        c.ensure(indices)
        slots = mx.take(c.map, indices)
        x = mx.expand_dims(x, (-2, -3))
        # Mirror the stock SwitchGLU's sort rule exactly (threshold and all):
        # decode calls are far below it, and forcing the sort there measured
        # slower than it saved.
        do_sort = indices.size >= _SORT_MIN_ROUTES
        inv = None
        if do_sort:
            x, slots, inv = _gather_sort(x, slots)
        up = c.qmm("up_proj", x, slots, do_sort)
        gate = c.qmm("gate_proj", x, slots, do_sort)
        out = c.qmm("down_proj", self.activation(up, gate), slots, do_sort)
        if do_sort:
            out = _scatter_unsort(out, inv, indices.shape)
        return out.squeeze(-2)

    def _forward_expert_major(
        self, flat_x: mx.array, ids: list[int], k: int, do_sort: bool
    ) -> mx.array:
        """Over-capacity prefill: chunk the routes on expert boundaries.

        ``ids[t * k + j]`` is the expert of token ``t``'s ``j``-th route. The
        routes are sorted by expert, resident experts first, and cut into
        chunks of up to ``capacity`` distinct experts (see
        :func:`_expert_route_chunks`), the same shape as the DeepSeek V4.1
        adapter's sorted prefill: each expert is installed at most once per call
        (the token-chunked path re-fetched an expert in every chunk that
        touched it, evicting on the way). Routes within a chunk are
        independent — the cross-expert weighted sum happens in the caller —
        so the chunk runs with one expert index per route, under the kernel
        the resident model would choose for the whole call (sorted at or
        above the stock threshold, else unsorted), and the outputs are put
        back in route order once at the end. Each chunk is evaluated before
        the next is built, which bounds the prefill transient. The next
        chunk's first reads start before that eval; their slot writes wait.
        """
        c = self.cache
        d_model = flat_x.shape[-1]
        ids_np = np.asarray(ids, dtype=np.int64)
        resident = np.zeros(c.n_experts, dtype=np.bool_)
        resident[c.slot_expert[c.slot_expert >= 0]] = True
        # Resident experts first: a miss evicts only experts already used.
        rank = ids_np + (~resident[ids_np]) * c.n_experts
        order = np.argsort(rank, kind="stable")  # routes grouped by expert
        sorted_ids = ids_np[order]
        # every position where a new expert's run begins, chunked by capacity
        run_starts = np.flatnonzero(np.diff(sorted_ids)) + 1
        run_starts = np.concatenate(([0], run_starts))
        chunks = _expert_route_chunks(run_starts, len(ids), c.capacity)
        outs = []
        ahead: dict = {}
        try:
            for n, (start, end) in enumerate(chunks):
                chunk_ids = sorted_ids[start:end]
                c._ensure_ids(np.unique(chunk_ids).tolist(), pending=ahead)
                ahead = {}
                n_routes = end - start
                padded_routes = n_routes
                if n_routes >= _pad_floor(c.capacity):
                    # Power-of-two sizes repeat across layers, so the Metal pool
                    # reuses those buffers instead of keeping one per size.
                    padded_routes = 1 << (n_routes - 1).bit_length()
                token_ids = order[start:end] // k
                if padded_routes != n_routes:
                    chunk_ids = np.pad(
                        chunk_ids, (0, padded_routes - n_routes), mode="edge"
                    )
                    token_ids = np.pad(
                        token_ids, (0, padded_routes - n_routes), mode="edge"
                    )
                slots = mx.take(c.map, mx.array(chunk_ids, dtype=mx.int32))
                slots = slots.reshape(-1, 1)
                t_idx = mx.array(token_ids, dtype=mx.int32)
                xe = mx.expand_dims(mx.take(flat_x, t_idx, axis=0), (-2, -3))
                inv = None
                if do_sort:
                    xe, slots, inv = _gather_sort(xe, slots)
                up = c.qmm("up_proj", xe, slots, do_sort)
                gate = c.qmm("gate_proj", xe, slots, do_sort)
                o = c.qmm("down_proj", self.activation(up, gate), slots, do_sort)
                if do_sort:
                    o = _scatter_unsort(o, inv, (padded_routes, 1))
                o = o.squeeze(-2)[:n_routes, 0, :]
                if n + 1 < len(chunks):
                    s1, e1 = chunks[n + 1]
                    ahead = c._read_ahead(
                        np.unique(sorted_ids[s1:e1]).tolist(), _io_batch()
                    )
                mx.eval(o)
                outs.append(o)
        finally:
            _drain(ahead)
        out = mx.concatenate(outs, axis=0)
        inverse = mx.array(np.argsort(order, kind="stable"), dtype=mx.int32)
        return mx.take(out, inverse, axis=0).reshape(-1, k, d_model)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        # A single _forward must have every expert it routes to resident AT
        # ONCE: a long prefill can route to more distinct experts than the
        # cache holds, in which case earlier installs would be evicted before
        # the gather runs and their slots would read garbage. Decode (working
        # set = batch x top_k) takes the no-sync fast path; larger calls pay
        # one readback to decide, and go expert-major only when the distinct
        # working set genuinely exceeds capacity.
        c = self.cache
        flat_i = indices.reshape(-1, indices.shape[-1])
        n_tok, k = flat_i.shape
        if k > c.capacity:
            raise ValueError("Expert cache capacity is smaller than routing top-k")
        if n_tok * k <= c.capacity or n_tok == 1:
            return self._forward(x, indices)
        ids = flat_i.reshape(-1).tolist()
        if len(set(ids)) <= c.capacity:
            return self._forward(x, indices)
        flat_x = x.reshape(-1, x.shape[-1])
        # Mirror the stock SwitchGLU's sort rule for the call as a whole, so
        # every chunk runs the kernel the resident model would have used.
        out = self._forward_expert_major(flat_x, ids, k, indices.size >= 64)
        return out.reshape(indices.shape + (x.shape[-1],))


def _resolve_model_dir(model_path: str | Path) -> Path | None:
    """Resolve a model name to its local checkpoint directory.

    Local directories pass through; hub repo ids resolve against the local
    HF cache only (the model was just loaded from it, so it is present) —
    this never triggers a download.
    """
    p = Path(model_path)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download

        # Restrict to the shards (all the store reads) so an mlx-lm-style
        # partial cache — model files only, no README etc. — resolves. A
        # patternless local_files_only lookup would demand the repo's full
        # file list and fail on exactly such caches.
        return Path(
            snapshot_download(
                str(model_path),
                allow_patterns=["*.safetensors"],
                local_files_only=True,
            )
        )
    except Exception:
        logger.warning(
            "moe expert offload: cannot resolve %r to a local " "checkpoint directory",
            str(model_path),
        )
        return None


def _is_mtp_path(path: str) -> bool:
    """Module path or checkpoint tensor name under an embedded MTP draft head.

    Matches ``mtp.*`` and ``*.mtp.*`` by path segment, so the wrappers and
    admission agree on which experts stay resident.
    """
    return "mtp" in path.split(".")


def _is_stock_switch_glu(obj) -> bool:
    # mlx-lm and mlx-vlm each define their own SwitchGLU class; match by
    # name + shape of the contract, not identity, so the VLM-served path
    # (the default for Gemma 4 checkpoints) is covered. OffloadSwitchGLU
    # has a different name, so re-wrapping is naturally excluded.
    # The GLM DSA package's SwitchGLU (fused gate/up, native weighted sum)
    # has its own adapter; see omlx.patches.glm_moe_dsa.moe_offload. The
    # DeepSeek V4 package's SwitchGLU (native block/pair kernels), shared by
    # glm5_next, has its own too; see omlx.patches.deepseek_v4.moe_offload.
    return (
        type(obj).__name__ == "SwitchGLU"
        and hasattr(obj, "activation")
        and type(obj).__module__ != "omlx.patches.glm_moe_dsa.switch_layers"
        and type(obj).__module__ != "omlx.patches.deepseek_v4.switch_layers"
    )


def _is_quantized_switch_linear(lin) -> bool:
    return type(lin).__name__ == "QuantizedSwitchLinear" and all(
        hasattr(lin, a) for a in ("group_size", "bits", "mode")
    )


def _iter_switch_glus(model):
    """Yield ``(parent, key, module, tree_path)`` for every stock SwitchGLU.

    mlx ``nn.Module`` subclasses ``dict`` — children are dict items, not
    attributes — so this walks ``.items()`` and list entries, building the
    same dotted paths ``tree_flatten`` produces (which is what checkpoint
    tensor names are matched against at load time).
    """
    seen = set()

    def walk(parent, key, obj, path):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if _is_stock_switch_glu(obj):
            yield (parent, key, obj, path)
            return
        if isinstance(obj, dict):  # includes nn.Module
            for k, v in obj.items():
                yield from walk(obj, k, v, f"{path}.{k}" if path else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                yield from walk(obj, i, v, f"{path}.{i}")

    yield from walk(None, None, model, "")


def _qwen35_checkpoint_prefix(store, path):
    # Qwen's loader adds language_model. to text-only checkpoint keys.
    if path.startswith("language_model.model.layers.") and not store.has(
        path + ".gate_proj.weight"
    ):
        flat = path.removeprefix("language_model.")
        if store.has(flat + ".gate_proj.weight"):
            return flat
    return path


def _resolve_store_view(
    glu: SwitchGLU, store: CheckpointExpertStore, path: str
) -> tuple[_GLUStoreView | None, str | None]:
    """Validate coverage and return a view in whichever naming scheme the
    checkpoint uses, or ``(None, reason)``.

    Stacked scheme: tensors live under the GLU's own tree path with shape
    ``[E, ...]``. Per-expert scheme: one tensor per expert under the GLU's
    parent (``<parent>.experts.<e>.<proj>.<field>`` — the layout mlx-lm's
    ``sanitize()`` stacks at load, e.g. OLMoE / Qwen2-MoE conversions);
    every expert's tensor is verified. Anything else — including layouts
    that also rename the projections, like Mixtral's ``w1/w2/w3`` — is
    reported for a graceful skip. Unknown storage dtypes are rejected here
    so the failure mode stays "runs resident" instead of a fetch-time
    KeyError mid-generation.
    """
    if type(glu).__module__.startswith("omlx.patches.deepseek_v4"):
        return (
            None,
            "custom weighted expert kernels require a dedicated offload adapter",
        )
    n = None
    fields_of: dict[str, list[str]] = {}
    for proj in _PROJS:
        lin = getattr(glu, proj, None)
        if lin is None or not _is_quantized_switch_linear(lin):
            return None, f"{proj} is not a QuantizedSwitchLinear"
        if "bias" in lin:
            return None, f"{proj} has per-expert bias (unsupported)"
        n = lin["weight"].shape[0] if n is None else n
        fields_of[proj] = ["weight", "scales"] + (
            ["biases"] if lin.get("biases") is not None else []
        )

    stacked = _GLUStoreView(store, path)
    parent = path.rsplit(".", 1)[0] if "." in path else ""
    view = (
        stacked
        if stacked.has("gate_proj", "weight")
        else _GLUStoreView(store, parent, per_expert=True)
    )

    for proj in _PROJS:
        lin = getattr(glu, proj)
        for field in fields_of[proj]:
            module_shape = tuple(lin[field].shape)
            if view is stacked:
                checks = [(view._name(proj, field, 0), module_shape)]
            else:
                checks = [
                    (view._name(proj, field, e), module_shape[1:]) for e in range(n)
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


def apply_moe_expert_offload(
    model,
    model_path: str | Path,
    resident_fraction: float = 0.25,
    *,
    mtp_resident: bool = False,
) -> int:
    """Replace covered SwitchGLU instances with offloaded ones.

    Returns the number of layers wrapped (0 when the model has no stock
    SwitchGLU or the checkpoint does not cover them). Must run before lazy
    weights are materialized for the memory saving to exist.

    ``mtp_resident`` keeps the embedded MTP draft head's experts resident
    (glm5_next Lightning MTP + offload; see
    ``omlx.patches.deepseek_v4.moe_offload``; qwen4_exp's native head under
    ``mtp.*`` is skipped by the generic traversal below). Other families
    reject the combination at validation.
    """
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return 0
    minimum = _minimum_experts(model_dir)
    config_path = Path(model_dir) / "config.json"
    kind = (
        json.loads(config_path.read_text()).get("model_type")
        if config_path.exists()
        else None
    )
    store = CheckpointExpertStore(model_dir)
    if not store:
        logger.warning("moe expert offload: no safetensors under %s", model_dir)
        return 0

    # GLM DSA blocks have their own adapter (fused gate/up, native weighted
    # sum); it shares this store format and the same wrap-before-materialize
    # contract, so the engine sees one count. DeepSeek V4 / glm5_next blocks
    # (native block kernels, split projections) follow the same pattern; see
    # omlx.patches.deepseek_v4.moe_offload.
    from .glm_moe_dsa.moe_offload import apply_glm_moe_expert_offload

    wrapped = apply_glm_moe_expert_offload(model, model_dir, resident_fraction)
    from .deepseek_v4.moe_offload import apply_deepseek_v4_moe_expert_offload

    wrapped += apply_deepseek_v4_moe_expert_offload(
        model, model_dir, resident_fraction, mtp_resident=mtp_resident
    )
    total_bytes = resident_bytes = 0
    for parent, key, glu, path in list(_iter_switch_glus(model)):
        if mtp_resident and _is_mtp_path(path):
            # Lightning MTP drafts from this head every step; streaming its
            # experts would put SSD reads on the draft path. Admission counts
            # it as resident (estimate_offload_admission_bytes mtp_resident).
            logger.info("moe expert offload: keeping MTP head resident: %s", path)
            continue
        checkpoint_path = (
            _qwen35_checkpoint_prefix(store, path) if kind == "qwen3_5_moe" else path
        )
        view, reason = _resolve_store_view(glu, store, checkpoint_path)
        if view is None:
            logger.info("moe expert offload: skipping %s (%s)", path, reason)
            continue
        n_experts = glu.gate_proj["weight"].shape[0]
        capacity = min(n_experts, max(minimum, round(n_experts * resident_fraction)))
        layer_bytes = sum(
            int(np.prod(lin[f].shape)) * lin[f].dtype.size
            for p in _PROJS
            for lin in (getattr(glu, p),)
            for f in (
                ["weight", "scales"]
                + (["biases"] if lin.get("biases") is not None else [])
            )
        )
        total_bytes += layer_bytes
        resident_bytes += layer_bytes * capacity // n_experts
        new = OffloadSwitchGLU(glu, capacity, view, minimum)
        if isinstance(parent, nn.Module):
            setattr(parent, key, new)  # registers via Module.__setattr__
        else:
            parent[key] = new  # plain list / plain dict
        wrapped += 1
        # Dropped source buffers land in the MLX pool, which the server pins
        # to total RAM, so drain per layer to bound the load transient
        # (same reasoning as the gate/up fusion patch, #2304).
        _sync_and_clear_cache()

    if total_bytes:
        logger.info(
            "moe expert offload: wrapped %d layers at %.1f%% residency "
            "(expert tables: %.2f GB total, %.2f GB resident)",
            wrapped,
            100 * resident_fraction,
            total_bytes / 1e9,
            resident_bytes / 1e9,
        )
    return wrapped


def _checkpoint_signature(model_dir: Path) -> tuple:
    """Stat identity of the config and shards, so cached scans follow edits."""
    files = [model_dir / "config.json", *model_dir.glob("*.safetensors")]
    out = []
    for path in sorted(files):
        if path.exists():
            st = path.stat()
            out.append((path.name, st.st_size, st.st_mtime_ns))
    return tuple(out)


@lru_cache(maxsize=64)
def _offload_layout(
    model_dir: str, signature: tuple, mtp_resident: bool = False
) -> tuple[tuple[int, int], ...]:
    """``(expert count, expert bytes)`` per container the wrapper would wrap.

    Derived from the same structural rules ``apply_moe_expert_offload``
    enforces, so an estimate built on it cannot promise savings the wrapper
    will not deliver: a container counts only when all three
    ``{gate,up,down}_proj`` projections are present *with quantization
    scales* (unquantized checkpoints wrap nothing) in a supported layout —
    stacked 3-D tensors or per-expert ``.experts.<n>.<proj>.<field>`` names.
    Renamed layouts (Mixtral-style ``w1/w2/w3``) match neither and are not
    listed. Families served by their own adapters are not listed either.
    Header-only: the shard bodies are never read. ``signature`` is the
    cache key's edit guard (see ``_checkpoint_signature``). With
    ``mtp_resident`` the draft head's tensors are left out, since the
    adapter keeps them resident.
    """
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        kind = json.loads(config_path.read_text()).get("model_type", "")
        if kind == "deepseek_v41":
            # V4.1 admission is owned by its own adapter's estimator.
            return ()
        if kind == "qwen3_5_moe":
            from .moe_offload_compat import moe_offload_compatibility

            if not moe_offload_compatibility(model_dir)[0]:
                return ()
    # stacked: container -> {"bytes", "fields": {(proj, field)}, "e": set}
    # per-expert: container -> {"bytes", "per_e": {idx: {(proj, field)}}}
    # Field completeness is tracked PER EXPERT, not container-wide: the
    # wrapper verifies every expert's tensors, so one complete expert
    # must not vouch for 31 incomplete ones (reported: 1 complete + 31
    # gate-only experts estimated 972,736 from 1,000,000 while zero
    # modules wrapped).
    stacked: dict[str, dict] = {}
    per_expert: dict[str, dict] = {}

    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if mtp_resident and _is_mtp_path(name):
                # The draft head stays resident (glm5_next / qwen4_exp
                # Lightning MTP + offload): its slab must not be discounted
                # here, or admission overcommits by exactly the bytes the
                # adapter refuses to offload.
                continue
            b0, b1 = spec["data_offsets"]
            m = _PER_EXPERT_PROJ_RE.match(name)
            if m:
                b = per_expert.setdefault(m.group("parent"), {"bytes": 0, "per_e": {}})
                b["bytes"] += b1 - b0
                b["per_e"].setdefault(int(m.group("idx")), set()).add(
                    (m.group("proj"), m.group("field"))
                )
                continue
            shape = spec.get("shape", ())
            if len(shape) == 3:
                parts = name.rsplit(".", 2)
                if (
                    len(parts) == 3
                    and parts[1] in _PROJS
                    and parts[2] in ("weight", "scales", "biases")
                ):
                    b = stacked.setdefault(
                        parts[0], {"bytes": 0, "fields": set(), "e": set()}
                    )
                    b["bytes"] += b1 - b0
                    b["fields"].add((parts[1], parts[2]))
                    b["e"].add(int(shape[0]))

    required = {(p, f) for p in _PROJS for f in ("weight", "scales")}
    layout: list[tuple[int, int]] = []
    for b in stacked.values():
        if not required <= b["fields"]:
            continue  # unquantized or partial: wraps nothing
        if len(b["e"]) != 1:  # projections disagree on E
            continue
        n = next(iter(b["e"]))
        if n > 0:
            layout.append((n, b["bytes"]))
    for b in per_expert.values():
        per_e = b["per_e"]
        if not per_e or any(not required <= s for s in per_e.values()):
            continue  # any incomplete expert: the wrapper rejects the layer
        layout.append((len(per_e), b["bytes"]))
    return tuple(layout)


def _layout_for(model_path: str | Path, mtp_resident: bool = False):
    """``(model_dir, layout)`` for a resolvable checkpoint, else ``None``."""
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return None
    model_dir = Path(model_dir)
    return model_dir, _offload_layout(
        str(model_dir), _checkpoint_signature(model_dir), bool(mtp_resident)
    )


def _admission_from_layout(
    layout: tuple[tuple[int, int], ...],
    full_size: int,
    resident_fraction: float,
    minimum: int,
) -> int:
    """``full_size`` less the bytes the runtime would leave on disk.

    Each container's savings honor the runtime's routing-aware capacity
    floor: ``capacity = min(E, max(minimum, round(E * fraction)))``, so tiny
    fractions do not under-report the resident share.
    """
    saved = 0.0
    for n, nbytes in layout:
        capacity = min(n, max(minimum, round(n * resident_fraction)))
        saved += nbytes * (1.0 - capacity / n)
    if saved <= 0:
        return full_size
    return full_size - int(saved)


def estimate_offload_admission_bytes(
    model_path: str | Path,
    full_size: int,
    resident_fraction: float = 0.25,
    *,
    mtp_resident: bool = False,
) -> int:
    """Admission-time size estimate with offload active.

    See ``_offload_layout`` for what counts as offloadable and
    ``_admission_from_layout`` for the capacity floor. Falls back to
    ``full_size`` on any failure — admission must never get more permissive
    by accident.
    """
    try:
        found = _layout_for(model_path, mtp_resident)
        if found is None:
            return full_size
        model_dir, layout = found
        return _admission_from_layout(
            layout, full_size, resident_fraction, _minimum_experts(model_dir)
        )
    except Exception:
        logger.debug("offload admission estimate failed", exc_info=True)
        return full_size


def offload_capacity_fractions(
    model_path: str | Path, *, mtp_resident: bool = False
) -> tuple[float, ...]:
    """Every resident fraction that maps to a distinct whole-expert capacity.

    The runtime keeps ``min(E, max(floor, round(E * fraction)))`` experts per
    container, so only ``capacity / E`` values change what is resident, and a
    setting chosen from this list round-trips through the runtime exactly.
    Ascending. Empty when nothing would be wrapped (dense or unquantized
    checkpoints, families with their own adapters).
    """
    try:
        found = _layout_for(model_path, mtp_resident)
    except Exception:
        logger.debug("offload layout scan failed", exc_info=True)
        return ()
    if found is None:
        return ()
    model_dir, layout = found
    minimum = _minimum_experts(model_dir)
    fractions: set[float] = set()
    for n, _ in layout:
        for capacity in range(min(n, minimum), n + 1):
            fractions.add(capacity / n)
    return tuple(sorted(fractions))


def fit_resident_fraction(
    model_path: str | Path,
    full_size: int,
    budget_bytes: int,
    *,
    mtp_resident: bool = False,
) -> float | None:
    """Largest resident fraction whose admission estimate fits ``budget_bytes``.

    Built on :func:`estimate_offload_admission_bytes` (same layout, same
    floor), so the returned fraction's estimate fits and the next larger
    capacity's does not. ``1.0`` when the fully resident model fits, ``None``
    when even the routing floor does not. The result is a whole number of
    experts per container expressed as a fraction, so passing it back as the
    setting reproduces the same capacity.
    """
    candidates = offload_capacity_fractions(model_path, mtp_resident=mtp_resident)
    if not candidates:
        return 1.0 if full_size <= budget_bytes else None
    # The estimate is non-decreasing in the fraction (every capacity is),
    # so the fitting candidates form a prefix; bisect for its end.
    lo, hi = 0, len(candidates)
    while lo < hi:
        mid = (lo + hi) // 2
        size = estimate_offload_admission_bytes(
            model_path, full_size, candidates[mid], mtp_resident=mtp_resident
        )
        if size <= budget_bytes:
            lo = mid + 1
        else:
            hi = mid
    return candidates[lo - 1] if lo else None


def materialize_offload_state(model) -> int:
    """Evaluate every offload cache's arrays on the loading thread's stream.

    ``ExpertCache`` keeps its slot map and resident slots on plain object
    attributes, so the engine's ``materialize_lazy_state`` walk never reaches
    them. Left lazy, they stay bound to the loader thread's stream and the
    first request from another thread dies with ``RuntimeError: There is no
    Stream(gpu, N) in current thread``. Reproduced live on a 24GB M5 Pro the
    moment the VLM path ran with offload enabled. Call this right after
    ``apply_moe_expert_offload``; returns the number of layers materialized.
    """
    caches = moe_offload_caches(model)
    arrays = []
    for cache in caches:
        arrays.append(cache.map)
        for triple in cache.resident.values():
            arrays.extend(a for a in triple if a is not None)
    if arrays:
        mx.eval(*arrays)
    return len(caches)


def moe_offload_caches(model) -> list:
    """Every offload cache under ``model``, in no particular order."""
    caches = []
    stack = [model]
    seen = set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        cache = getattr(obj, "cache", None)
        if getattr(cache, "moe_offload_cache", False):
            caches.append(cache)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
    return caches


def release_moe_offload_slots(caches) -> int:
    """Shrink every cache to its floor for a prefill; return the bytes given up.

    Expert-major prefill streams almost every expert anyway; decode restores them."""
    return sum(c.release_slots() for c in caches if hasattr(c, "release_slots"))


def restore_moe_offload_slots(caches) -> None:
    """Return every cache shrunk by :func:`release_moe_offload_slots` to full size."""
    for c in caches:
        if hasattr(c, "restore_slots"):
            c.restore_slots()


def moe_offload_stats(model=None, caches=None) -> dict:
    """Aggregate hit/miss/byte counters over all offloaded layers."""
    if caches is None:
        caches = moe_offload_caches(model)
    hits = sum(c.hits for c in caches)
    misses = sum(c.misses for c in caches)
    total = hits + misses
    return {
        "layers": len(caches),
        "hits": hits,
        "misses": misses,
        "hit_rate": (hits / total) if total else None,
        "fetched_bytes": sum(getattr(c, "fetched_bytes", 0) for c in caches),
    }


__all__ = [
    "CheckpointExpertStore",
    "OffloadSwitchGLU",
    "apply_moe_expert_offload",
    "materialize_offload_state",
    "moe_offload_caches",
    "moe_offload_stats",
    "release_moe_offload_slots",
    "restore_moe_offload_slots",
]
