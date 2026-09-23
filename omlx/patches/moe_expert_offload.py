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
import weakref
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
        self._slabs: dict[str, int] = {}
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
        slab = self._slabs.get(name)
        if slab is None:
            _, _, shape, _ = self._specs[name]
            slab = int(np.prod(shape[1:]))
            self._slabs[name] = slab
        _, _, shape, _ = self._specs[name]
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
# useful width is the storage queue depth, so the default is wider than the
# core count. ``OMLX_MOE_OFFLOAD_IO_WORKERS`` <= 1 (or unparseable) keeps the
# serial path and creates no threads at all;
# ``OMLX_MOE_OFFLOAD_IO_BATCH`` caps how many experts' payloads may be in
# flight, which is what bounds the extra host memory the pipeline holds.
_IO_WORKERS = 12
_IO_LOCK = threading.Lock()
_IO_POOL: ThreadPoolExecutor | None = None
_IO_BATCH = 0
_IO_CONFIGURED = False


def _env_int(name: str, default: int, invalid: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return invalid


def _io_pool() -> ThreadPoolExecutor | None:
    """The shared reader pool, or ``None`` when reads must stay serial."""
    global _IO_POOL, _IO_BATCH, _IO_CONFIGURED
    with _IO_LOCK:
        if not _IO_CONFIGURED:
            _IO_CONFIGURED = True
            workers = _env_int("OMLX_MOE_OFFLOAD_IO_WORKERS", _IO_WORKERS, 0)
            if workers > 1:
                _IO_BATCH = max(
                    1,
                    _env_int("OMLX_MOE_OFFLOAD_IO_BATCH", 4 * workers, 4 * workers),
                )
                _IO_POOL = ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="omlx-moe-io"
                )
        return _IO_POOL


def _io_batch() -> int:
    """Experts whose reads may be in flight at once."""
    _io_pool()
    return _IO_BATCH


def _pin_frac() -> float:
    """Resident-slot fraction protected by hot-set pinning (0 disables).

    Swept 0.75/0.875/1.0 on MiMo at fraction 0.125: 0.875 (28 pinned, 4
    dynamic slots) beat 0.75 (24/8) slightly and 1.0 (31/1) clearly — the
    unpredictable tail needs a few churn slots."""
    try:
        return min(
            max(float(os.environ.get("OMLX_MOE_OFFLOAD_PIN", 0.875)), 0.0), 1.0
        )
    except ValueError:
        return 0.875


def _shutdown_io_pool() -> None:
    """Drop the pool; the next fetch re-reads the environment (tests)."""
    global _IO_POOL, _IO_BATCH, _IO_CONFIGURED
    with _IO_LOCK:
        pool, _IO_POOL, _IO_BATCH, _IO_CONFIGURED = _IO_POOL, None, 0, False
    if pool is not None:
        pool.shutdown(wait=True)


# -- Residency governor ----------------------------------------------------
#
# A fixed resident fraction is chosen once at load and never revisited, but
# free memory keeps moving. The governor re-evaluates dynamic cache
# capacity from the same two signals the old expert_streaming governor
# used — pressure (system free memory) and hunger (new cache misses since
# the last look):
#
#   free < LOW   -> every cache to its floor, ignoring cooldown (desperate)
#   free < TGT   -> shrink by just enough slots to restore TGT free
#   spare > 0 and misses accrued -> grow by the affordable share of spare
#   otherwise    -> stable (hysteresis; no churn)
#
# Watermarks are fractions of physical RAM (8% desperate, 15% target) so
# they port across machines; growth spends ``free - TGT`` so the machine
# never dips into the target band just to grow. Growth is still bounded by
# each cache's ``max_capacity`` — for auto mode that is the load-time
# budget fit, so the governor can only ever regrow to what the memory
# guard proved safe. Cooldowns are asymmetric (15s shrink, 30s grow) and
# desperation bypasses them, so shrink->regrow cannot flap.
#
# ``_governor_tick`` runs at ``OffloadSwitchGLU.__call__`` entry — a call
# boundary, so a resize can never invalidate the per-call
# ``unique <= capacity`` contract nor a staged chunk's in-flight reads.
# Each call resizes only its own cache, which staggers the per-layer
# reallocation across the layer loop instead of one multi-GiB burst.
#
# ``cache.dynamic`` decides participation: set for auto-fraction caches
# and for explicit fractions only when ``OMLX_MOE_OFFLOAD_DYNAMIC=1`` —
# an explicit percentage is a user pin and stays fixed by default.

_GOV_EVAL_S = 1.0
_GOV_SHRINK_COOLDOWN_S = 15.0
_GOV_GROW_COOLDOWN_S = 30.0
_GOV_LOW_FRAC = 0.08
_GOV_TGT_FRAC = 0.15
_GOV_CACHES: "weakref.WeakSet[ExpertCache]" = weakref.WeakSet()
_GOV_STATE = {
    "band_until": 0.0,
    "band": None,
    "last_action": 0.0,
    "misses": 0,
    "ram": 0,
}


def _dynamic_default(auto: bool) -> bool:
    raw = os.environ.get("OMLX_MOE_OFFLOAD_DYNAMIC")
    if raw is not None:
        return raw != "0"
    return auto


def _free_and_total() -> tuple[int, int]:
    try:
        from ..utils.psutil_compat import get_total_memory, virtual_memory

        return int(virtual_memory().available), int(get_total_memory())
    except Exception:
        return 0, 0


def _spendable_free_bytes() -> int:
    """Free memory beyond the governor's target band, in bytes."""
    avail, ram = _free_and_total()
    if avail <= 0 or ram <= 0:
        return 0  # unknown: never let a guess inflate residency
    return max(0, avail - int(_GOV_TGT_FRAC * ram))


def _eval_band(now: float) -> tuple | None:
    """Pick this evaluation window's action for every dynamic cache."""
    state = _GOV_STATE
    caches = [c for c in list(_GOV_CACHES) if c.dynamic]
    if not caches:
        return None
    avail, ram = _free_and_total()
    if avail <= 0 or ram <= 0:
        return None
    if not state["ram"]:
        state["ram"] = ram
    misses = sum(c.misses for c in caches)
    hungry = misses > state["misses"]
    state["misses"] = misses
    if avail < _GOV_LOW_FRAC * ram:
        state["last_action"] = now
        return ("floor", 0)
    slot = max(c.slot_bytes for c in caches) or 1
    n = len(caches)
    if avail < _GOV_TGT_FRAC * ram:
        if now - state["last_action"] < _GOV_SHRINK_COOLDOWN_S:
            return None
        # Free exactly enough slots across all caches to reach the target.
        needed = _GOV_TGT_FRAC * ram - avail
        slots = max(1, int(needed / (n * slot)) + 1)
        state["last_action"] = now
        return ("shrink", slots)
    if not hungry or now - state["last_action"] < _GOV_GROW_COOLDOWN_S:
        return None
    spare = avail - _GOV_TGT_FRAC * ram
    slots = int(spare / (n * slot))
    if slots < 2:
        return None  # too small to pay a reallocation
    state["last_action"] = now
    return ("grow", slots)


def _governor_tick(cache: "ExpertCache") -> None:
    """Apply the current evaluation's band to ``cache``, if dynamic."""
    state = _GOV_STATE
    now = time.monotonic()
    if now >= state["band_until"]:
        state["band_until"] = now + _GOV_EVAL_S
        state["band"] = _eval_band(now)
    band = state["band"]
    if band is None or not cache.dynamic:
        return
    mode, slots = band
    if mode == "floor":
        changed = cache.resize(cache.min_capacity)
    elif mode == "shrink":
        changed = cache.resize(cache.capacity - slots)
    else:
        changed = cache.resize(cache.capacity + slots)
    if changed:
        logger.info(
            "moe expert offload: governor %s a layer to %d slots",
            mode,
            cache.capacity,
        )


class ExpertCache:
    """Contiguous resident slots over one layer's experts, LRU eviction.

    Holds no reference to the wrapped module's expert tensors — only the
    resident slots and the store view. That is the difference between saving
    memory and adding it: keeping the source tensors referenced alongside the
    slots costs the full expert set *plus* the cache.
    """

    moe_offload_cache = True  # walked by materialize_offload_state / stats

    def __init__(
        self,
        glu: SwitchGLU,
        capacity: int,
        disk: _GLUStoreView,
        *,
        min_capacity: int = 8,
        max_capacity: int | None = None,
        dynamic: bool = False,
    ):
        self.n_experts = glu.gate_proj["weight"].shape[0]
        self.capacity = min(capacity, self.n_experts)
        # Governor bounds: it may shrink to ``min_capacity`` under memory
        # pressure (floor stays >= top_k so a call's working set always
        # fits) and grow back to ``max_capacity`` — ``n_experts`` in auto
        # mode, the requested capacity for explicit fractions.
        self.min_capacity = max(1, min(int(min_capacity), self.capacity))
        self.max_capacity = min(
            self.n_experts, max(self.capacity, int(max_capacity or capacity))
        )
        self.dynamic = bool(dynamic)
        _GOV_CACHES.add(self)
        self.projs = _PROJS
        self.disk = disk
        self.resident: dict[str, list] = {}
        for name in self.projs:
            lin = getattr(glu, name)
            has_b = lin.get("biases") is not None
            w, s = lin["weight"], lin["scales"]
            b = lin["biases"] if has_b else None
            self.resident[name] = [
                mx.zeros((self.capacity,) + w.shape[1:], dtype=w.dtype),
                mx.zeros((self.capacity,) + s.shape[1:], dtype=s.dtype),
                (
                    None
                    if b is None
                    else mx.zeros((self.capacity,) + b.shape[1:], dtype=b.dtype)
                ),
            ]
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
        self.slot_of: dict[int, int] = {}  # expert id -> slot, LRU ordered
        self.free = list(range(self.capacity))
        self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)
        self.hits = self.misses = 0
        self.warm = False
        # Hot-set pinning: ``ensure`` calls (the device-side path used by
        # decode) update a per-expert route counter; the top ``pin_n`` are
        # then refused as eviction victims. Decode routing is heavily
        # skewed (measured: top-32 experts cover ~78% of routes vs 12.5%
        # uniform) while prefill's profile does not transfer to decode —
        # so ``ensure_ids`` (the prefill chunker's host path) never
        # teaches. ``freq``/``pins`` stay unused when the pin fraction is 0.
        self.pin_n = max(
            0, min(int(self.capacity * _pin_frac()), self.capacity - 1)
        )
        self.freq = (
            np.zeros(self.n_experts, dtype=np.int64) if self.pin_n else None
        )
        self.pins = frozenset()
        # Bytes one slot holds across all projections — the governor's
        # shrink/grow granularity when converting memory bands to slots.
        self.slot_bytes = sum(int(plan.nbytes) for _, _, plan in self._plans(0))

    def resize(self, new_capacity: int) -> bool:
        """Re-target the resident slot count, keeping the hottest experts.

        Pinned experts survive first, then the most recently used; the LRU
        tail is dropped when shrinking. Slot tensors are re-allocated at
        the new size so a shrink really returns memory to the pool. Called
        only from the governor, at a call boundary — never while
        ``ensure_ids`` holds in-flight reads or protected experts.
        """
        new_capacity = max(
            self.min_capacity, min(int(new_capacity), self.max_capacity)
        )
        if new_capacity == self.capacity:
            return False
        order = list(self.slot_of)  # LRU order: oldest first
        pinned = [e for e in order if e in self.pins][-new_capacity:]
        keep = set(pinned)
        room = new_capacity - len(keep)
        if room > 0:
            for e in reversed(order):  # newest first
                if e not in keep:
                    keep.add(e)
                    room -= 1
                    if room <= 0:
                        break
        survivors = [e for e in order if e in keep]
        idx = [self.slot_of[e] for e in survivors]
        src = mx.array(idx, dtype=mx.int32)
        for name in self.projs:
            tensors = self.resident[name]
            for field, t in enumerate(tensors):
                if t is None:
                    continue
                nt = mx.zeros((new_capacity,) + t.shape[1:], dtype=t.dtype)
                if idx:
                    nt[: len(idx)] = mx.take(t, src, axis=0)
                tensors[field] = nt
        self.slot_of = {e: i for i, e in enumerate(survivors)}
        self.free = list(range(len(survivors), new_capacity))
        self.capacity = new_capacity
        self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)
        if survivors:
            self.map[mx.array(survivors, dtype=mx.int32)] = mx.array(
                [self.slot_of[e] for e in survivors], dtype=mx.int32
            )
        self.pin_n = max(
            0, min(int(new_capacity * _pin_frac()), new_capacity - 1)
        )
        if self.pin_n:
            if self.freq is None:
                self.freq = np.zeros(self.n_experts, dtype=np.int64)
            top = np.argpartition(self.freq, -self.pin_n)[-self.pin_n :]
            self.pins = frozenset(int(t) for t in top if self.freq[t] > 0)
        else:
            self.freq = None
            self.pins = frozenset()
        self.warm = len(self.slot_of) == self.n_experts
        _sync_and_clear_cache()
        return True

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

    def _reserve(self, protected: frozenset | None = None) -> int:
        """Claim a slot, evicting the LRU expert if none is free.

        ``protected`` holds the current call's needed experts: evicting one
        of those would refetch it later in the same call, so pick the LRU
        resident outside the set. Cannot deadlock — callers keep
        ``len(needed) <= capacity``, so a non-protected resident always
        exists when a miss needs a slot.
        """
        if self.free:
            return self.free.pop()
        prot = protected or ()
        pins = self.pins
        fallback = None
        for cand in self.slot_of:
            if cand in prot:
                continue
            if fallback is None:
                fallback = cand
            if cand not in pins:
                slot = self.slot_of.pop(cand)
                self.map[cand] = -1
                return slot
        # Every non-protected resident is pinned — degrade to plain LRU.
        old_e = fallback if fallback is not None else next(iter(self.slot_of))
        slot = self.slot_of.pop(old_e)
        self.map[old_e] = -1
        return slot

    def _write(self, slot: int, payload: list) -> None:
        """Copy one expert's fetched bytes into ``slot``."""
        for name, field, plan, raw in payload:
            self.resident[name][field][slot] = CheckpointExpertStore.to_mx(plan, raw)

    def _install(
        self, e: int, payload: list | None = None, protected: frozenset | None = None
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
        self.map[e] = slot
        self.warm = len(self.slot_of) == self.n_experts
        return slot

    def ensure(self, idx: mx.array) -> None:
        """Make every expert in ``idx`` resident.

        Two passes. The first classifies the call's misses without touching
        any cache state and starts their reads on the IO pool, at most
        ``OMLX_MOE_OFFLOAD_IO_BATCH`` experts in flight; the second is the
        serial install loop, which takes bytes from the pipeline instead of
        reading them itself (an expert the first pass did not queue, because
        eviction unseated it in the meantime, falls back to a serial read).
        Every mutation — ``slot_of``, ``free``, ``map``, the counters, the
        slots — happens on the calling thread in the serial order, so LRU
        victims, hit/miss counts and resident bytes are identical to the
        serial path. Concurrent ``ensure`` calls on one cache stay
        unsupported, exactly as before.

        The ``.tolist()`` is a device->host readback and therefore a sync per
        MoE layer per step. Removing it needs prefetch (resolve layer L+1's
        residency during layer L's compute) — deliberately not in v1.
        """
        if self.warm:  # nothing can miss; skip it
            return
        self.ensure_ids(
            (int(e) for e in idx.reshape(-1).tolist()), learn=True
        )

    def stage_ids(self, ids) -> dict:
        """Speculatively read non-resident experts' bytes on the IO pool.

        Returns ``{e: [(name, field, plan, future)]}`` for a later
        :meth:`ensure_ids` call to consume. Bytes only — no cache state is
        touched, so a stale or unused stash is always safe to drop (the
        caller waits the futures out in ``ensure_ids``'s finally).
        ``slot_of`` is unchanged between the stage call and the consuming
        ensure, so ``e not in slot_of`` is an exact miss prediction.
        """
        staged: dict[int, list] = {}
        pool = _io_pool()
        if pool is None or self.warm:
            return staged
        for e in set(int(i) for i in ids):
            if e in self.slot_of:
                continue
            staged[e] = [
                (name, field, plan, pool.submit(CheckpointExpertStore.read, plan))
                for name, field, plan in self._plans(e)
            ]
        return staged

    def ensure_ids(
        self, ids, staged: dict | None = None, learn: bool = False
    ) -> None:
        """Host-side variant of :meth:`ensure`: takes an iterable of expert
        ids and skips the device->host readback entirely (the prefill
        chunker already holds them as numpy). ``staged`` may carry futures
        from :meth:`stage_ids`; they are consumed — serially, in the same
        order the demand path would install them — or waited out.
        ``learn`` marks a token-level call whose routes may update the
        pin table — the prefill chunker never passes it."""
        if self.warm:
            return
        # Ascending expert id = ascending file offset per shard, so misses
        # are read in on-disk order (set iteration order scrambles it).
        id_list = [int(e) for e in ids]
        needed = sorted(set(id_list))
        if self.pin_n and learn:
            # Token-level call: count routes and refresh the pinned set.
            # Host-path (prefill chunk) calls never teach — their routing
            # profile was measured not to transfer to decode.
            np.add.at(self.freq, id_list, 1)
            top = np.argpartition(self.freq, -self.pin_n)[-self.pin_n :]
            self.pins = frozenset(int(t) for t in top if self.freq[t] > 0)
        protected = frozenset(needed)
        pool = _io_pool()
        queue = [e for e in needed if e not in self.slot_of] if pool is not None else []
        window = _io_batch()
        pending: dict[int, list] = {}
        sent = 0

        def prefetch(upto: int) -> None:
            nonlocal sent
            while sent < min(upto, len(queue)):
                e = queue[sent]
                sent += 1
                if staged is not None and e in staged:
                    continue  # bytes already in flight via stage_ids
                pending[e] = []
                for name, field, plan in self._plans(e):
                    pending[e].append(
                        (
                            name,
                            field,
                            plan,
                            pool.submit(CheckpointExpertStore.read, plan),
                        )
                    )

        try:
            prefetch(window)
            done = 0
            for e in needed:
                if e in self.slot_of:
                    slot = self.slot_of.pop(e)  # re-insert: LRU order
                    self.slot_of[e] = slot
                    self.hits += 1
                    continue
                self.misses += 1
                # Refill an exhausted window before falling back to a serial read.
                if sent < len(queue) and queue[sent] == e:
                    prefetch(done + window)
                group = pending.pop(e, None)
                if group is None and staged:
                    group = staged.pop(e, None)
                if group is None:
                    self._install(e, protected=protected)
                    continue
                # Count the current payload in the window until its writes finish.
                prefetch(done + window)
                self._install(
                    e,
                    [(name, field, plan, f.result()) for name, field, plan, f in group],
                    protected=protected,
                )
                done += 1
        finally:
            # Finish reads before the store's shard descriptors can be released.
            futures = [f for group in pending.values() for _, _, _, f in group]
            if staged:
                futures += [f for group in staged.values() for _, _, _, f in group]
                staged.clear()
            for future in futures:
                future.cancel()
            if futures:
                wait(futures)

        # No mx.eval here: installs are already-materialized host arrays, and
        # evaluating every resident tensor on every miss measured 22% slower
        # at identical peak memory. Prefill's transient is bounded by the
        # per-chunk eval in __call__, which is a different mechanism.

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


class OffloadSwitchGLU(nn.Module):
    """SwitchGLU whose experts live in an :class:`ExpertCache`."""

    def __init__(
        self,
        glu: SwitchGLU,
        capacity: int,
        disk: _GLUStoreView,
        **cache_kwargs,
    ):
        super().__init__()
        self.cache = ExpertCache(glu, capacity, disk, **cache_kwargs)
        self.activation = glu.activation

    def _forward(self, x: mx.array, indices: mx.array) -> mx.array:
        c = self.cache
        c.ensure(indices)
        slots = mx.take(c.map, indices)
        x = mx.expand_dims(x, (-2, -3))
        # Mirror the stock SwitchGLU's sort rule exactly (threshold and all):
        # decode calls are far below it, and forcing the sort there measured
        # slower than it saved.
        do_sort = indices.size >= 64
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
        routes are sorted by expert and cut into chunks holding every route
        of up to ``capacity`` distinct experts, the same shape as the
        DeepSeek V4.1 adapter's sorted prefill: an expert's routes all land
        in one chunk, so each expert is installed at most once per call
        (the token-chunked path re-fetched an expert in every chunk that
        touched it, evicting on the way). Routes within a chunk are
        independent — the cross-expert weighted sum happens in the caller —
        so the chunk runs with one expert index per route, under the kernel
        the resident model would choose for the whole call (sorted at or
        above the stock threshold, else unsorted), and the outputs are put
        back in route order once at the end. Each chunk is evaluated before
        the next is built, which bounds the prefill transient.
        """
        c = self.cache
        d_model = flat_x.shape[-1]
        ids_np = np.asarray(ids, dtype=np.int64)
        order = np.argsort(ids_np, kind="stable")  # routes grouped by expert
        sorted_ids = ids_np[order]
        # every position where a new expert's run begins, chunked by capacity
        run_starts = np.flatnonzero(np.diff(sorted_ids)) + 1
        run_starts = np.concatenate(([0], run_starts))
        cuts = run_starts[:: c.capacity].tolist() + [len(ids)]
        spans = list(zip(cuts[:-1], cuts[1:]))
        outs = []
        staged = None
        for i, (start, end) in enumerate(spans):
            chunk_ids = sorted_ids[start:end]
            c.ensure_ids(np.unique(chunk_ids), staged=staged)
            staged = None
            # Stage the next chunk's missing experts now: the reads overlap
            # this chunk's qmm + mx.eval instead of starting after them.
            # Bytes only — slot installs still happen inside ensure_ids,
            # after the eval, so an in-flight gather never sees torn slots.
            if i + 1 < len(spans):
                ns, ne = spans[i + 1]
                staged = c.stage_ids(np.unique(sorted_ids[ns:ne]))
            n_routes = end - start
            padded_routes = n_routes
            # GatherQMM uses sorted QMM only when B >= 16 and B / E >= 4
            # (E = resident slots); below that, padding would change kernels.
            if n_routes >= max(16, 4 * c.capacity):
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
            mx.eval(o)
            outs.append(o)
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
        # Governor first: a resize must land before this call's capacity
        # read so the chunked path always honors ``unique <= capacity``.
        _governor_tick(c)
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
    resident_fraction: float | None = None,
    *,
    mtp_resident: bool = False,
) -> int:
    """Replace covered SwitchGLU instances with offloaded ones.

    Returns the number of layers wrapped (0 when disabled via
    ``OMLX_MOE_EXPERT_OFFLOAD=0``, the model has no stock SwitchGLU, or the
    checkpoint does not cover them). Must run before lazy weights are
    materialized for the memory saving to exist.

    ``resident_fraction=None`` (or 0) is automatic mode: the fraction is
    resolved against the memory guard's budget, and the caches stay
    dynamic — the governor may grow them toward full residency when free
    memory allows and misses accrue, and shrink them under pressure. An
    explicit fraction pins the capacity (and ceiling), unless
    ``OMLX_MOE_OFFLOAD_DYNAMIC=1`` re-enables shrinking as a safety valve.

    ``mtp_resident`` keeps the embedded MTP draft head's experts resident
    (glm5_next Lightning MTP + offload; see
    ``omlx.patches.deepseek_v4.moe_offload``). Other families reject the
    combination at validation, so only this adapter's path consumes it.
    """
    if os.environ.get("OMLX_MOE_EXPERT_OFFLOAD", "1") == "0":
        return 0
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

    # The check runs on the raw value first: automatic (None/0) delegates
    # to the v41 fitter for that family, whose admission accounting is
    # more accurate than the generic resident estimate.
    refusal = moe_offload_memory_check(
        model_dir, resident_fraction, mtp_resident=mtp_resident
    )
    if refusal is not None:
        raise ValueError(refusal)

    auto = not resident_fraction
    resolved = resolve_moe_offload_fraction(
        model_dir, resident_fraction, mtp_resident=mtp_resident
    )
    if resolved is None:
        resolved = 0.25  # uninspectable checkpoint: keep the old default
    resident_fraction = resolved
    # The governor's ceiling is the static-budget fit — never higher, so a
    # regrow can never exceed what the memory guard proved safe. The start
    # capacity may be smaller when free memory is tight at load.
    ceiling_fraction = (
        resolve_moe_offload_fraction(
            model_dir,
            None,
            mtp_resident=mtp_resident,
            budget_bytes=_moe_offload_memory_limit(),
        )
        if auto
        else resident_fraction
    )
    if ceiling_fraction is None:
        ceiling_fraction = resident_fraction

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
        ceiling = min(
            n_experts, max(minimum, round(n_experts * ceiling_fraction))
        )
        new = OffloadSwitchGLU(
            glu,
            capacity,
            view,
            min_capacity=minimum,
            max_capacity=ceiling,
            dynamic=_dynamic_default(auto),
        )
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


def _safetensors_signature(model_dir: Path) -> tuple:
    files = sorted(Path(model_dir).glob("*.safetensors"))
    return tuple(
        (p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files
    )


@lru_cache(maxsize=64)
def _expert_containers(
    path: str, signature: tuple, mtp_resident: bool
) -> tuple[dict, dict]:
    """Scan shard headers once: expert containers and their byte sizes.

    stacked: container -> {"bytes", "fields": {(proj, field)}, "e": {n}}
    per_expert: container -> {"bytes", "per_e": {idx: {(proj, field)}}}
    """
    stacked: dict[str, dict] = {}
    per_expert: dict[str, dict] = {}
    for shard in sorted(Path(path).glob("*.safetensors")):
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if mtp_resident and (name.startswith("mtp.") or ".mtp." in name):
                # The draft head stays resident (glm5_next Lightning MTP +
                # offload): its slab must not be discounted here, or
                # admission overcommits by exactly the bytes the adapter
                # refuses to offload.
                continue
            b0, b1 = spec["data_offsets"]
            m = _PER_EXPERT_PROJ_RE.match(name)
            if m:
                b = per_expert.setdefault(
                    m.group("parent"), {"bytes": 0, "per_e": {}}
                )
                b["bytes"] += b1 - b0
                b["per_e"].setdefault(int(m.group("idx")), set()).add(
                    (m.group("proj"), m.group("field"))
                )
                continue
            shape = spec.get("shape", ())
            if len(shape) == 3:
                parts = name.rsplit(".", 2)
                if len(parts) == 3 and parts[1] in _PROJS and parts[2] in (
                    "weight",
                    "scales",
                    "biases",
                ):
                    b = stacked.setdefault(
                        parts[0], {"bytes": 0, "fields": set(), "e": set()}
                    )
                    b["bytes"] += b1 - b0
                    b["fields"].add((parts[1], parts[2]))
                    b["e"].add(int(shape[0]))
    return stacked, per_expert


def _offloadable_expert_bytes(model_dir: Path, mtp_resident: bool) -> int:
    """Bytes the wrapper could offload: every complete expert container."""
    stacked, per_expert = _expert_containers(
        str(model_dir), _safetensors_signature(model_dir), mtp_resident
    )
    required = {(p, f) for p in _PROJS for f in ("weight", "scales")}
    total = 0
    for b in stacked.values():
        if required <= b["fields"] and len(b["e"]) == 1:
            total += b["bytes"]
    for b in per_expert.values():
        if b["per_e"] and all(required <= s for s in b["per_e"].values()):
            total += b["bytes"]
    return total


def resolve_moe_offload_fraction(
    model_path: str | Path,
    resident_fraction: float | None,
    *,
    mtp_resident: bool = False,
    engram_ssd_offload: bool = False,
    budget_bytes: int | None = None,
) -> float | None:
    """Resolve ``None``/0 (automatic) to a concrete resident fraction.

    Automatic picks the largest fraction whose resident set fits the
    memory budget — the guard's ceiling minus the never-offloaded bytes,
    divided by offloadable expert bytes. ``budget_bytes`` caps the budget
    below the guard's limit (used for the static ceiling); by default the
    budget is also bounded by the memory free *right now* (the governor
    regrows toward the static ceiling later, so an under-pressure start
    only costs misses, never an OOM). Explicit fractions pass through.
    ``None`` means uninspectable — callers should keep their default.
    """
    if resident_fraction:
        return resident_fraction
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return None
    limit = _moe_offload_memory_limit()
    if limit <= 0:
        return 1.0  # guard disabled: "as much as fits" means all experts
    if budget_bytes is not None:
        budget = min(limit, budget_bytes)
    else:
        budget = limit
    try:
        full = sum(
            f.stat().st_size for f in Path(model_dir).glob("*.safetensors")
        )
        if not full:
            return None
        config_path = Path(model_dir) / "config.json"
        kind = (
            json.loads(config_path.read_text()).get("model_type")
            if config_path.exists()
            else None
        )
        if kind == "deepseek_v41":
            # Its adapter owns capacity math (Engram residency, slot
            # floors) — reuse its fitter instead of re-deriving here.
            from .deepseek_v41.moe_offload import fit_resident_fraction

            fitted = fit_resident_fraction(
                model_dir,
                budget,
                engram_ssd_offload=engram_ssd_offload,
                mtp_resident=mtp_resident,
            )
            # None: even the routing floor overruns the budget — return
            # the floor so the memory check produces the refusal message.
            return fitted if fitted is not None else 0.0
        experts = _offloadable_expert_bytes(model_dir, mtp_resident)
        base = full - experts
        if budget_bytes is None:
            # Auto start also respects free memory *now*: the resident set
            # is allocated before the first request, when whatever else is
            # running still holds its memory.
            budget = min(budget, base + _spendable_free_bytes())
        if experts <= 0 or base >= budget:
            # Below the floor or past the budget already; the memory check
            # still makes the final call on whether even the floor fits.
            return 0.0
        fraction = min(1.0, (budget - base) / experts)
        # Per-layer ``round(n * f)`` and the capacity floor quantize the
        # resident set into plateaus — the linear target can land half a
        # slot per layer over budget. Binary-search the real estimator
        # (header scan is cached; this is pure arithmetic) for the largest
        # fraction the check will actually admit.
        estimate = estimate_offload_admission_bytes(
            model_dir, full, fraction, mtp_resident=mtp_resident
        )
        if estimate > limit:
            lo, hi, best = 0.0, fraction, None
            for _ in range(40):
                mid = (lo + hi) / 2
                if (
                    estimate_offload_admission_bytes(
                        model_dir, full, mid, mtp_resident=mtp_resident
                    )
                    <= limit
                ):
                    best, lo = mid, mid
                else:
                    hi = mid
            # ``None`` means even the per-layer floor exceeds the budget —
            # return the floor so the memory check explains the refusal.
            fraction = best if best is not None else 0.0
        return fraction
    except (OSError, TypeError, ValueError):
        return None


def estimate_offload_admission_bytes(
    model_path: str | Path,
    full_size: int,
    resident_fraction: float = 0.25,
    *,
    mtp_resident: bool = False,
) -> int:
    """Admission-time size estimate with offload active.

    Derived from the same structural rules ``apply_moe_expert_offload``
    enforces, so the estimate cannot promise savings the wrapper will not
    deliver: a container counts only when all three ``{gate,up,down}_proj``
    projections are present *with quantization scales* (unquantized
    checkpoints wrap nothing) in a supported layout — stacked 3-D tensors
    or per-expert ``.experts.<n>.<proj>.<field>`` names. Renamed layouts
    (Mixtral-style ``w1/w2/w3``) match neither and discount nothing. Each
    layer's savings honor the runtime's routing-aware capacity floor:
    ``capacity = min(E, max(8, top_k, round(E * fraction)))``, so tiny fractions
    do not under-report the resident share. Falls back to ``full_size`` on
    any failure — admission must never get more permissive by accident.
    """
    if os.environ.get("OMLX_MOE_EXPERT_OFFLOAD", "1") == "0":
        return full_size
    try:
        model_dir = _resolve_model_dir(model_path)
        if model_dir is None:
            return full_size
        minimum = _minimum_experts(model_dir)
        config_path = Path(model_dir) / "config.json"
        if config_path.exists():
            kind = json.loads(config_path.read_text()).get("model_type", "")
            if kind == "deepseek_v41":
                # V4.1 admission is owned by its own adapter's estimator.
                return full_size
            if kind == "qwen3_5_moe":
                from .moe_offload_compat import moe_offload_compatibility

                if not moe_offload_compatibility(model_dir)[0]:
                    return full_size
        # Field completeness is tracked PER EXPERT, not container-wide: the
        # wrapper verifies every expert's tensors, so one complete expert
        # must not vouch for 31 incomplete ones (reported: 1 complete + 31
        # gate-only experts estimated 972,736 from 1,000,000 while zero
        # modules wrapped).
        stacked, per_expert = _expert_containers(
            str(model_dir), _safetensors_signature(model_dir), mtp_resident
        )

        required = {(p, f) for p in _PROJS for f in ("weight", "scales")}
        saved = 0.0
        for b in stacked.values():
            if not required <= b["fields"]:
                continue  # unquantized or partial: wraps nothing
            if len(b["e"]) != 1:  # projections disagree on E
                continue
            n = next(iter(b["e"]))
            if n <= 0:
                continue
            capacity = min(n, max(minimum, round(n * resident_fraction)))
            saved += b["bytes"] * (1.0 - capacity / n)
        for b in per_expert.values():
            per_e = b["per_e"]
            if not per_e or any(not required <= s for s in per_e.values()):
                continue  # any incomplete expert: the wrapper rejects the layer
            n = len(per_e)
            capacity = min(n, max(minimum, round(n * resident_fraction)))
            saved += b["bytes"] * (1.0 - capacity / n)
        if saved <= 0:
            return full_size
        return full_size - int(saved)
    except Exception:
        logger.debug("offload admission estimate failed", exc_info=True)
        return full_size


def moe_offload_resident_estimate(
    model_path: str | Path,
    resident_fraction: float,
    *,
    mtp_resident: bool = False,
) -> int | None:
    """Bytes the checkpoint would hold resident at ``resident_fraction``.

    Non-expert tensors plus the per-layer expert slots — the same arithmetic
    ``apply_moe_expert_offload`` delivers. ``None`` when the checkpoint
    cannot be inspected; callers should treat that as "unknown", not "fits".
    """
    try:
        model_dir = _resolve_model_dir(model_path)
        if model_dir is None:
            return None
        config_path = Path(model_dir) / "config.json"
        kind = (
            json.loads(config_path.read_text()).get("model_type")
            if config_path.exists()
            else None
        )
        full = sum(
            f.stat().st_size for f in Path(model_dir).glob("*.safetensors")
        )
        if not full:
            return None
        if kind == "deepseek_v41":
            # Its adapter owns the savings estimate (same split the engine
            # pool uses at admission).
            from .deepseek_v41.moe_offload import estimate_expert_savings

            return max(
                0,
                full
                - estimate_expert_savings(
                    model_dir, resident_fraction, mtp_resident=mtp_resident
                ),
            )
        return estimate_offload_admission_bytes(
            model_dir, full, resident_fraction, mtp_resident=mtp_resident
        )
    except (OSError, TypeError, ValueError):
        return None


def _moe_offload_memory_limit() -> int:
    """Byte budget for the offload resident set on this machine.

    Defaults to 75% of Metal's recommended working set: the estimate covers
    weights only, so the remainder is left for activations, KV cache, vision
    towers and in-flight expert reads. Measured overheads over the estimate:
    ~2 GiB on MiMo at fraction 0.125, >7 GiB on a glm5_next VLM — the margin
    has to cover the latter, not the former. ``OMLX_MOE_OFFLOAD_MEM_LIMIT_GB``
    overrides the limit; ``0`` disables the guard for setups the static
    ceiling misjudges.
    """
    raw = os.environ.get("OMLX_MOE_OFFLOAD_MEM_LIMIT_GB")
    if raw is not None:
        try:
            return int(float(raw) * 2**30)
        except ValueError:
            pass
    return int(
        (mx.device_info().get("max_recommended_working_set_size") or 0) * 0.75
    )


def moe_offload_memory_check(
    model_path: str | Path,
    resident_fraction: float | None,
    *,
    mtp_resident: bool = False,
    engram_ssd_offload: bool = False,
) -> str | None:
    """Refusal reason when the resident set would not fit, else ``None``.

    Lets the settings surface reject a fraction that could only ever end in
    a Metal out-of-memory error at first request — the resident set is fixed
    at load, so there is no way a load that starts over-budget recovers.
    A ``None``/0 fraction is automatic mode: it resolves against the same
    budget first, so only a model whose non-expert base alone overruns the
    limit is refused here.
    """
    limit = _moe_offload_memory_limit()
    if limit <= 0:
        return None
    if not resident_fraction:
        # Automatic mode on deepseek_v41: its adapter's fitter decides —
        # the adapter's admission accounting (Engram residency modes) is
        # more accurate than the generic estimate, which counts every
        # non-expert byte as resident. A found fit already proves the
        # budget; None means even the routing floor overruns it.
        model_dir = _resolve_model_dir(model_path)
        kind = None
        if model_dir is not None:
            config_path = Path(model_dir) / "config.json"
            if config_path.exists():
                try:
                    kind = json.loads(config_path.read_text()).get("model_type")
                except (OSError, ValueError):
                    kind = None
        if kind == "deepseek_v41":
            from .deepseek_v41.moe_offload import fit_resident_fraction

            if (
                fit_resident_fraction(
                    model_dir,
                    limit,
                    engram_ssd_offload=engram_ssd_offload,
                    mtp_resident=mtp_resident,
                )
                is not None
            ):
                return None
            floor = moe_offload_resident_estimate(
                model_path, 0.0, mtp_resident=mtp_resident
            )
            if floor is None:
                return (
                    "MoE expert offload cannot fit this model within the "
                    f"~{limit / 2**30:.1f} GiB resident-weight budget on "
                    "this machine."
                )
            return (
                f"MoE expert offload would keep ~{floor / 2**30:.1f} GiB "
                f"resident even at its minimum capacity, but only "
                f"~{limit / 2**30:.1f} GiB of GPU memory is budgeted for "
                "resident weights on this machine."
            )
    resolved = resolve_moe_offload_fraction(
        model_path,
        resident_fraction,
        mtp_resident=mtp_resident,
        engram_ssd_offload=engram_ssd_offload,
        # The check is "could this fit at all" — the static guard budget,
        # not the avail-aware start capacity the governor grows into.
        budget_bytes=limit,
    )
    if resolved is None:
        return None  # uninspectable: the load-time apply re-checks anyway
    estimate = moe_offload_resident_estimate(
        model_path, resolved, mtp_resident=mtp_resident
    )
    if estimate is None or estimate <= limit:
        return None
    return (
        f"MoE expert offload would keep ~{estimate / 2**30:.1f} GiB resident "
        f"at {resolved:g} residency, but only "
        f"~{limit / 2**30:.1f} GiB of GPU memory is budgeted for resident "
        "weights on this machine. Lower "
        "moe_expert_offload_resident_fraction or free memory first."
    )


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
    arrays = []
    layers = 0
    stack = [model]
    seen = set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        cache = getattr(obj, "cache", None)
        if getattr(cache, "moe_offload_cache", False):
            layers += 1
            arrays.append(cache.map)
            for triple in cache.resident.values():
                arrays.extend(a for a in triple if a is not None)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
    if arrays:
        mx.eval(*arrays)
    return layers


def moe_offload_stats(model) -> dict:
    """Aggregate hit/miss counters and residency over all offloaded layers."""
    hits = misses = layers = slots = experts = 0
    dynamic = False
    stack = [model]
    seen = set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        cache = getattr(obj, "cache", None)
        if getattr(cache, "moe_offload_cache", False):
            hits += cache.hits
            misses += cache.misses
            # Family adapters share the hits/misses contract; residency
            # introspection degrades gracefully on caches without it.
            slots += getattr(cache, "capacity", 0)
            experts += getattr(cache, "n_experts", 0)
            dynamic = dynamic or getattr(cache, "dynamic", False)
            layers += 1
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            stack.extend(obj)
    total = hits + misses
    return {
        "layers": layers,
        "hits": hits,
        "misses": misses,
        "hit_rate": (hits / total) if total else None,
        # Current residency as a fraction of experts — moves when the
        # governor resizes caches; ``dynamic`` marks governor-managed.
        "resident_fraction": (slots / experts) if experts else None,
        "dynamic": dynamic or None,
    }


__all__ = [
    "CheckpointExpertStore",
    "OffloadSwitchGLU",
    "apply_moe_expert_offload",
    "materialize_offload_state",
    "moe_offload_stats",
    "resolve_moe_offload_fraction",
]
