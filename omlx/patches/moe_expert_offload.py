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
import re
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import (
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)

from omlx.utils.metal_sync import _sync_and_clear_cache
from omlx.utils.safetensors import read_safetensors_header

from .expert_streaming._env import env_bool, env_int
from .expert_streaming.residency_adapter import PerLayerResidencyAdapter
from .expert_streaming.shard_bank import ExpertBackingStore, np_to_mx
from .expert_streaming.slot_cache import SlotArena, SlotBookkeeping

logger = logging.getLogger(__name__)

_PROJS = ("gate_proj", "up_proj", "down_proj")

_PER_EXPERT_PROJ_RE = re.compile(
    r"^(?P<parent>.+)\.experts\.(?P<idx>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<field>weight|scales|biases)$"
)

# A pending slab read: everything needed to turn an expert-row range of a
# shard into an mx.array — the shared backing store, the tensor key, the
# row range (``hi=None`` plans a whole tensor), the output shape and the
# safetensors dtype tag — and nothing that touches MLX or cache state, so
# the store read can run on any thread.
_ReadPlan = namedtuple("_ReadPlan", "store key lo hi shape dtype_str")

# Dtype tags this adapter serves bit-exactly (bf16 travels as raw uint16
# and is reinterpreted on the mlx side via np_to_mx; the rest convert
# directly). The shared SAFETENSORS_NUMPY_DTYPES table is a superset —
# tags outside this set are declined at coverage time, so the failure
# mode stays "runs resident" instead of a fetch-time surprise.
_SUPPORTED_DTYPES = frozenset({"BF16", "F16", "F32", "U32", "I32", "U8"})


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
    shard. Reads delegate to the shared
    :class:`~omlx.patches.expert_streaming.shard_bank.ExpertBackingStore`
    (mmap + positional ``preadv`` — no fd carried on the descriptor), so
    reads of different experts are safe to run concurrently, off the
    calling thread. The read splits in two: :meth:`read` produces the
    ndarray rows and touches neither MLX nor cache state (any thread),
    :meth:`to_mx` turns them into an array (the calling thread's stream).

    The ``_specs`` index (name -> dtype/shape) answers the coverage probes
    (``has``/``spec``) without opening a reader.
    """

    def __init__(self, model_path: str | Path):
        model_path = Path(model_path)
        self._specs: dict[str, tuple[str, tuple[int, ...]]] = {}
        for shard in sorted(model_path.glob("*.safetensors")):
            for name, spec in read_safetensors_header(shard).items():
                if name == "__metadata__":
                    continue
                self._specs[name] = (
                    str(spec["dtype"]),
                    tuple(spec["shape"]),
                )
        # Raises ValueError on repacked co-activation checkpoints
        # (expert_order.json) — serving those would silently return
        # permuted experts. Callers turn that into a declined wrap.
        self._backing = ExpertBackingStore(model_path)
        # B8 downgrade marker: stamped by _apply_legacy_adapter when the
        # unified streaming backend owned this model type but produced
        # nothing, so the fallback is visible instead of silent.
        self.streaming_fallback_reason: str | None = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __bool__(self) -> bool:
        return bool(self._specs)

    def has(self, name: str) -> bool:
        return name in self._specs

    def spec(self, name: str) -> tuple[tuple[int, ...], str]:
        dtype, shape = self._specs[name]
        return shape, dtype

    def plan_expert(self, name: str, expert: int) -> _ReadPlan:
        """Plan one expert's slab of a stacked ``[num_experts, ...]`` tensor."""
        dtype, _shape = self._specs[name]
        return _ReadPlan(
            self._backing, name, int(expert), int(expert) + 1, _shape[1:], dtype
        )

    def plan_tensor(self, name: str) -> _ReadPlan:
        """Plan a whole tensor (per-expert checkpoint layouts)."""
        dtype, shape = self._specs[name]
        return _ReadPlan(self._backing, name, 0, None, shape, dtype)

    @staticmethod
    def read(plan: _ReadPlan) -> np.ndarray:
        """The plan's rows as an ndarray, in the transport dtype.

        Thread-safe: the backing store's reads are positional (preadv on
        a shared read-only mapping). ``hi=None`` plans whole tensors.
        """
        arr = (
            plan.store.load_tensor(plan.key)
            if plan.hi is None
            else plan.store.read_span(plan.key, plan.lo, plan.hi)
        )
        # read_span returns (hi-lo, *per_shape); expert plans want the
        # row itself, whole-tensor plans already carry the stored shape.
        return arr if arr.shape == plan.shape else arr.reshape(plan.shape)

    @staticmethod
    def to_mx(plan: _ReadPlan, raw: np.ndarray) -> mx.array:
        """Promote a plan's rows to its mx.array (host-side, one copy)."""
        return np_to_mx(raw, plan.dtype_str)

    def fetch_expert(self, name: str, expert: int) -> mx.array:
        """One expert's slab of a stacked ``[num_experts, ...]`` tensor."""
        plan = self.plan_expert(name, expert)
        return self.to_mx(plan, self.read(plan))

    def fetch_tensor(self, name: str) -> mx.array:
        """A whole tensor (per-expert checkpoint layouts)."""
        plan = self.plan_tensor(name)
        return self.to_mx(plan, self.read(plan))

    def close(self) -> None:
        """Close the shard readers (also called via the engine's
        ``_expert_streaming_backing`` shutdown hook). Idempotent."""
        backing = getattr(self, "_backing", None)
        if backing is not None:
            backing.close()


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


def _checkpoint_prefixes(runtime_prefix: str) -> list[str]:
    """Candidate checkpoint spellings for a runtime module-tree prefix.

    mlx-vlm nests the language model under an extra ``.model`` segment, so
    the runtime tree says ``language_model.model.layers.N`` while the
    checkpoint (and its quantization policy) spells
    ``language_model.layers.N`` — a mapping sanitize() resolves at load but
    raw-name store matching never sees. Prefer the exact spelling, then the
    de-nested one. Every candidate still faces full shape+dtype validation,
    so a wrong guess cannot be accepted.
    """
    if ".model." in runtime_prefix:
        return [runtime_prefix, runtime_prefix.replace(".model.", ".")]
    return [runtime_prefix]


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


def _io_pool() -> ThreadPoolExecutor | None:
    """The shared reader pool, or ``None`` when reads must stay serial.

    Not ``streaming_switch.io_pool_for``: that pool is never None
    (depth <= 1 maps to the shared default executor) while the legacy
    contract is workers <= 1 -> serial reads with no pool threads at all.
    """
    global _IO_POOL, _IO_BATCH, _IO_CONFIGURED
    with _IO_LOCK:
        if not _IO_CONFIGURED:
            _IO_CONFIGURED = True
            # malformed -> 0 -> serial (the env_int invalid sentinel).
            workers = env_int(
                "OMLX_MOE_OFFLOAD_IO_WORKERS", _IO_WORKERS, invalid=0
            )
            if workers > 1:
                _IO_BATCH = max(
                    1,
                    env_int("OMLX_MOE_OFFLOAD_IO_BATCH", 4 * workers),
                )
                _IO_POOL = ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="omlx-moe-io"
                )
        return _IO_POOL


def _io_batch() -> int:
    """Experts whose reads may be in flight at once."""
    _io_pool()
    return _IO_BATCH


def _shutdown_io_pool() -> None:
    """Drop the pool; the next fetch re-reads the environment (tests)."""
    global _IO_POOL, _IO_BATCH, _IO_CONFIGURED
    with _IO_LOCK:
        pool, _IO_POOL, _IO_BATCH, _IO_CONFIGURED = _IO_POOL, None, 0, False
    if pool is not None:
        pool.shutdown(wait=True)


class _CeilingBookkeeping(SlotBookkeeping):
    """SlotBookkeeping whose ``cap`` is a residency COUNT ceiling.

    The shared ``acquire`` pops a free row unconditionally — under the
    unified cache rooms == cap, so a free row is always inside the
    budget. This cache's cap is a governor ceiling that can sit BELOW
    rooms after a shrink: free rows can exist while the resident count
    must not grow past cap, so a full count takes the LRU victim's row
    instead of a free one (the legacy ``_reserve`` rule). In ensure_set
    terms the victim choice is unchanged either way: every needed
    resident was already touched to the MRU end, so the coldest row is
    never a needed one when a miss is pending.
    """

    def acquire(self, needed, verify: bool = False, scratch: int = 0):
        if self.free and len(self.slot_of) >= self.cap:
            evicted = self.evict_oldest_outside(needed)
            if evicted is not None:
                return evicted[1], evicted[0], False
            # Every resident is needed or pinned — the caller grows.
            return self.rooms, None, True
        return super().acquire(needed, verify=verify, scratch=scratch)


class _CheckpointArena(SlotArena):
    """SlotArena over checkpoint ``(plan, raw)`` payloads.

    ``produce`` hands back ``{proj: {field: (plan, raw ndarray)}}`` —
    raw rows straight off ``CheckpointExpertStore.read``. The mx decode
    (``to_mx``) runs here, inside the row-write path, which is what
    keeps the two fetch-failure classes distinct: a read failure inside
    ``produce`` rolls the whole batch back (victims restored,
    byte-identical bookkeeping) while a decode failure lands
    mid-commit — the interrupted row returns to ``free`` and its victim
    stays evicted. Same split the legacy ``_install`` kept.
    """

    def __init__(
        self, capacity, projections, arrays, bind, eval_params=None
    ):
        super().__init__(capacity, projections, arrays, bind, eval_params)
        # Same rooms/cap and free rows — only acquire() differs.
        self.book = _CeilingBookkeeping(self.book.rooms)

    def write_row(self, slot: int, payload: dict) -> None:
        decoded = {
            proj: {
                fname: CheckpointExpertStore.to_mx(plan, raw)
                for fname, (plan, raw) in fields.items()
            }
            for proj, fields in payload.items()
        }
        super().write_row(slot, decoded)


class ExpertCache:
    """Contiguous resident slots over one layer's experts, LRU eviction.

    Holds no reference to the wrapped module's expert tensors — only the
    resident slots and the store view. That is the difference between saving
    memory and adding it: keeping the source tensors referenced alongside the
    slots costs the full expert set *plus* the cache.

    Residency runs on the shared :class:`SlotArena` engine — slot map,
    grow and the acquire/produce/commit/rollback protocol are the same
    machinery the DSv4.1 ``_ExpertSlots`` uses; kept here are only the
    checkpoint-specific parts: read plans, the IO-pool pipeline, the
    ``to_mx`` decode and the expert->slot ``map`` mirror.
    """

    def __init__(self, glu: SwitchGLU, capacity: int, disk: _GLUStoreView):
        self.n_experts = glu.gate_proj["weight"].shape[0]
        # ``capacity`` is the governor-driven ceiling; ``rooms`` the
        # physical row count of the slot tensors (> cap after shrink).
        capacity = min(int(capacity), self.n_experts)
        self.projs = _PROJS
        self.disk = disk
        self.resident: dict[str, list] = {}
        for name in self.projs:
            lin = getattr(glu, name)
            w, s = lin["weight"], lin["scales"]
            b = lin["biases"] if lin.get("biases") is not None else None
            self.resident[name] = [
                mx.zeros((capacity,) + w.shape[1:], dtype=w.dtype),
                mx.zeros((capacity,) + s.shape[1:], dtype=s.dtype),
                (
                    None
                    if b is None
                    else mx.zeros((capacity,) + b.shape[1:], dtype=b.dtype)
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
        self.arena = _CheckpointArena(
            capacity, _PROJS, self._arena_arrays, self._arena_bind
        )
        self.book = self.arena.book
        self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)
        # Per-expert bytes for the governor's budget<->slots math.
        self.per_expert_bytes = sum(
            int(np.prod(a.shape[1:])) * a.dtype.size
            for triple in self.resident.values()
            for a in triple
            if a is not None
        )
        # Residency mutation is serialized per layer by ``_lock``; the
        # arena's internal lock only ever nests inside it (grow/set_cap),
        # and every entry point takes ``_lock`` first, so the order never
        # inverts. Request-level serialization comes from the engine's
        # streaming backing marker.
        self._lock = threading.RLock()

    # SlotBookkeeping aliases — the bookkeeping fields live on ``book``;
    # these keep the pre-existing attribute surface for callers/tests.
    @property
    def slot_of(self):
        return self.book.slot_of

    @property
    def free(self):
        return self.book.free

    @property
    def rooms(self):
        return self.book.rooms

    @property
    def capacity(self):
        return self.book.cap

    @capacity.setter
    def capacity(self, v):
        self.book.cap = int(v)

    @property
    def hits(self):
        return self.book.hits

    @hits.setter
    def hits(self, v):
        self.book.hits = int(v)

    @property
    def misses(self):
        return self.book.misses

    @misses.setter
    def misses(self, v):
        self.book.misses = int(v)

    @property
    def evictions(self):
        return self.book.evictions

    @evictions.setter
    def evictions(self, v):
        self.book.evictions = int(v)

    @property
    def warm(self) -> bool:
        """Full residency: no ``ensure`` can miss, so it skips the sync."""
        return len(self.book.slot_of) == self.n_experts

    # -- SlotArena host bindings --------------------------------------------
    def _arena_arrays(self, proj: str) -> dict:
        """Bound rows of one projection as ``{field: array}``."""
        w, s, b = self.resident[proj]
        out = {"weight": w, "scales": s}
        if b is not None:
            out["biases"] = b
        return out

    def _arena_bind(self, proj: str, fields: dict) -> None:
        """Rebind grown storage into the positional ``[w, s, b]`` list."""
        self.resident[proj] = [
            fields["weight"],
            fields["scales"],
            fields.get("biases"),
        ]

    def _plans(self, e: int) -> list:
        """Read plans for expert ``e``'s tensors, in slot-write order."""
        out = []
        for name in self.projs:
            out.append((name, "weight", self.disk.plan(name, "weight", e)))
            out.append((name, "scales", self.disk.plan(name, "scales", e)))
            if self.resident[name][2] is not None and self.disk.has(
                name, "biases"
            ):
                out.append((name, "biases", self.disk.plan(name, "biases", e)))
        return out

    def _read_one(self, e: int) -> dict:
        """Serial payload: one expert's raw rows, in slot-write order."""
        payload: dict[str, dict] = {}
        for name, field, plan in self._plans(e):
            payload.setdefault(name, {})[field] = (
                plan,
                CheckpointExpertStore.read(plan),
            )
        return payload

    def _produce(self, fetch_list: list) -> list:
        """Pass 2 for the arena: fetch raw rows for the missed experts.

        Payloads are ``{proj: {field: (plan, raw)}}`` aligned with
        ``fetch_list`` — the mx decode stays out of here so a read
        failure rolls the whole batch back (victims restored), while a
        decode failure inside ``write_row`` lands mid-commit.

        With the IO pool up, each expert's field reads submit together
        and at most ``_io_batch()`` experts are in flight — the bound on
        the host bytes the pipeline holds. Joins run in fetch_list
        order; a failed join cancels + waits every outstanding future
        itself (the arena cannot see caller futures). No pool: serial
        inline reads, same order.
        """
        if not fetch_list:
            return []
        pool = _io_pool()
        if pool is None:
            return [self._read_one(item[0]) for item in fetch_list]
        window = _io_batch()
        pending: dict[int, list] = {}
        sent = 0

        def prefetch(upto: int) -> None:
            nonlocal sent
            while sent < min(upto, len(fetch_list)):
                e = fetch_list[sent][0]
                sent += 1
                pending[e] = [
                    (
                        name,
                        field,
                        plan,
                        pool.submit(CheckpointExpertStore.read, plan),
                    )
                    for name, field, plan in self._plans(e)
                ]

        try:
            prefetch(window)
            payloads = []
            for done, (e, _slot, _victim) in enumerate(fetch_list):
                # Keep the in-flight window full as experts are consumed.
                prefetch(done + window)
                payload: dict[str, dict] = {}
                for name, field, plan, fut in pending.pop(e):
                    payload.setdefault(name, {})[field] = (plan, fut.result())
                payloads.append(payload)
            return payloads
        finally:
            # Finish reads before the store's shard descriptors can be
            # released; a failed join leaves the rest to cancel here.
            futures = [t[3] for group in pending.values() for t in group]
            for future in futures:
                future.cancel()
            if futures:
                wait(futures)

    def _sync_map(self, before: dict) -> None:
        """Reconcile ``map`` with ``slot_of`` after a residency mutation.

        The book's internal evictions (``acquire``/``trim_to_cap``)
        carry no per-evict hook, so the expert->slot mirror is
        maintained by diffing: new or moved entries write their row,
        disappeared entries write -1. An unchanged book skips the writes
        entirely — ``ensure`` runs per layer per token.
        """
        after = self.book.slot_of
        for e, slot in after.items():
            if before.get(e) != slot:
                self.map[e] = slot
        for e in before:
            if e not in after:
                self.map[e] = -1

    def _install(self, e: int, payload: list | None = None) -> int:
        """Install one expert through the shared arena path.

        Kept for callers/tests that drive single-expert installs
        directly; ``payload`` accepts the legacy
        ``(proj, field, plan, raw)`` tuples — the field as a name or a
        positional index into ``[w, s, b]`` — to install pre-fetched
        bytes; ``None`` fetches through the produce pipeline. Returns
        the slot the expert landed in.
        """
        e = int(e)
        with self._lock:
            before = dict(self.book.slot_of)
            try:
                if payload is None:
                    self.arena.ensure_set(
                        {e}, frozen=False, produce=self._produce
                    )
                else:
                    grouped: dict[str, dict] = {}
                    for name, field, plan, raw in payload:
                        fname = (
                            field
                            if isinstance(field, str)
                            else ("weight", "scales", "biases")[field]
                        )
                        grouped.setdefault(name, {})[fname] = (plan, raw)
                    self.arena.ensure_set(
                        {e},
                        frozen=False,
                        produce=lambda fl: [grouped for _ in fl],
                    )
            finally:
                self._sync_map(before)
        return self.book.slot_of[e]

    def ensure(self, idx: mx.array) -> None:
        """Make every expert in ``idx`` resident.

        The miss set's reads pipeline on the IO pool (at most
        ``OMLX_MOE_OFFLOAD_IO_BATCH`` experts in flight) while every
        mutation — ``slot_of``, ``free``, ``map``, the counters, the
        slots — happens on the calling thread inside the arena's
        acquire/produce/commit protocol, so LRU victims, hit/miss
        counts and resident bytes are identical to the serial path.
        Concurrent ``ensure`` calls on one cache stay unsupported,
        exactly as before.

        ``ensure_set`` rejects a working set wider than the cap: a set
        that fits is kept in ONE call (callers gather every needed
        expert at once, so they must co-reside); a wider set — only
        reachable by driving ``ensure`` directly — can never co-reside
        anyway and runs as <=cap chunks in set order, later chunks
        evicting earlier installs via LRU, the same victims the serial
        loop produced. Chunk width also floors at the IO window so a
        pooled run never holds more than a batch of payloads in host
        memory at once (fetch payloads are all live until their
        commit-time decode).

        The ``.tolist()`` is a device->host readback and therefore a sync per
        MoE layer per step. Removing it needs prefetch (resolve layer L+1's
        residency during layer L's compute).
        """
        if self.warm:  # nothing can miss; skip it
            return
        needed = set(int(e) for e in idx.reshape(-1).tolist())
        with self._lock:
            before = dict(self.book.slot_of)
            try:
                cap = max(1, int(self.book.cap))
                if len(needed) <= cap:
                    self.arena.ensure_set(
                        needed, frozen=False, produce=self._produce
                    )
                else:
                    width = cap
                    if _io_pool() is not None:
                        width = max(1, min(cap, _io_batch()))
                    ordered = list(needed)
                    for i in range(0, len(ordered), width):
                        self.arena.ensure_set(
                            ordered[i : i + width],
                            frozen=False,
                            produce=self._produce,
                        )
            finally:
                self._sync_map(before)
        # No mx.eval here: installs are already-materialized host arrays, and
        # evaluating every resident tensor on every miss measured 22% slower
        # at identical peak memory. Prefill's transient is bounded by the
        # per-chunk eval in __call__, which is a different mechanism.

    def resize(self, cap: int) -> None:
        """Retarget the residency ceiling (governor path).

        Shrink evicts LRU-tail rows until the resident set fits the new
        cap; grow past the physical row count reallocs the slot tensors
        through the arena's two-phase grow — the new arrays are built
        and evaluated BEFORE the rebind, so a failure keeps the old
        storage serving (same contract as the V4.1 ``_ExpertSlots``
        grow).
        """
        with self._lock:
            cap = max(1, min(int(cap), self.n_experts))
            before = dict(self.book.slot_of)
            if cap > self.rooms:
                self.arena.grow(cap)
            self.capacity = cap
            self.book.trim_to_cap(())
            self._sync_map(before)

    def clear(self) -> None:
        """Drop all residency (governor pressure path)."""
        with self._lock:
            self.book.reset()
            self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)

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

    def __init__(self, glu: SwitchGLU, capacity: int, disk: _GLUStoreView):
        super().__init__()
        self.cache = ExpertCache(glu, capacity, disk)
        self.activation = glu.activation
        # Governor wiring (set post-wrap by _apply_legacy_adapter): the
        # shared LegacyOffloadState and this layer's index inside it.
        self._state = None
        self._layer = -1

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
        (token-chunked would re-fetch an expert in every chunk that touched
        it, evicting on the way). Routes within a chunk are
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
        outs = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            chunk_ids = sorted_ids[start:end]
            c.ensure(mx.array(np.unique(chunk_ids), dtype=mx.int32))
            slots = mx.take(c.map, mx.array(chunk_ids, dtype=mx.int32))
            slots = slots.reshape(-1, 1)
            t_idx = mx.array(order[start:end] // k, dtype=mx.int32)
            xe = mx.expand_dims(mx.take(flat_x, t_idx, axis=0), (-2, -3))
            inv = None
            if do_sort:
                xe, slots, inv = _gather_sort(xe, slots)
            up = c.qmm("up_proj", xe, slots, do_sort)
            gate = c.qmm("gate_proj", xe, slots, do_sort)
            o = c.qmm("down_proj", self.activation(up, gate), slots, do_sort)
            if do_sort:
                o = _scatter_unsort(o, inv, (end - start, 1))
            o = o.squeeze(-2)[:, 0, :]
            mx.eval(o)
            outs.append(o)
        out = mx.concatenate(outs, axis=0)
        inverse = mx.array(np.argsort(order, kind="stable"), dtype=mx.int32)
        return mx.take(out, inverse, axis=0).reshape(-1, k, d_model)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        # A single _forward must have every expert it routes to resident AT
        # ONCE: a long prefill can route to more distinct experts than the
        # cache holds, in which case earlier installs would be evicted
        # before the gather runs and their slots would read garbage. Chunk
        # the routes on expert boundaries so each expert is installed at
        # most once per call. Decode (working set = batch x top_k) takes
        # the no-sync fast path.
        c = self.cache
        flat_i = indices.reshape(-1, indices.shape[-1])
        n_tok, k = flat_i.shape
        if k > c.capacity:
            raise ValueError("Expert cache capacity is smaller than routing top-k")
        if n_tok * k <= c.capacity:
            before = c.misses
            out = self._forward(x, indices)
            state = self._state
            if n_tok == 1 and state is not None:
                # Decode-shaped visit: one routed row per call, the unit
                # the governor's hunger window counts (same contract as
                # the V4.1 OffloadedExpert note_visit).
                state.note_visit(self._layer, c.misses > before)
            return out
        # One host sync for the whole routing matrix; the distinct-set runs
        # on it — re-tolist()ing every mx slice would be a device sync each.
        ids = flat_i.reshape(-1).tolist()
        if len(set(ids)) <= c.capacity:
            return self._forward(x, indices)
        flat_x = x.reshape(-1, x.shape[-1])
        # Mirror the stock SwitchGLU's sort rule for the call as a whole, so
        # every chunk runs the kernel the resident model would have used.
        out = self._forward_expert_major(flat_x, ids, k, indices.size >= 64)
        return out.reshape(indices.shape + (x.shape[-1],))


class LegacyOffloadState(PerLayerResidencyAdapter):
    """Governor-facing cache over the legacy per-layer ``ExpertCache``s.

    Same duck-type ``V41StreamingBacking`` presents to
    ``ExpertResidencyGovernor``, now via the shared
    ``PerLayerResidencyAdapter`` base — ``capacity`` / ``resize`` /
    ``clear`` / ``set_layer_caps`` / ``layer_cap_overrides`` / ``stats``
    plus the ``note_visit`` decode counter. Units are per-layer like
    V4.1: ``per_slot`` is one resident row in every wrapped layer, the
    governor sees ``num_layers=1``, and ``capacity`` reports the uniform
    per-layer base. Attached to the model as ``_moe_offload_legacy_state``;
    the scheduler still finds the ``CheckpointExpertStore`` marker for
    request serialization (the guard's mini-bank transient does not apply
    to these persistent slot buffers).
    """

    _GOVERNOR_LABEL = "moe expert offload"

    def __init__(
        self,
        caches: list,
        *,
        dynamic: bool = True,
        max_budget_bytes: int | None = None,
        min_budget_bytes: int | None = None,
        stall_target: float | None = None,
        min_cap: int | None = None,
        streaming_fallback_reason: str | None = None,
    ) -> None:
        # ``caches`` lands before super().__init__: the base ctor runs the
        # hooks below (_iter_units / _per_slot_bytes / _initial_base_cap)
        # while wiring num_layers/per_slot/base_cap and arming the governor.
        self.caches = list(caches)
        # B8 downgrade marker: set when the unified streaming backend
        # owned this model type but converted nothing — surfaced through
        # summary() so the fallback is visible, not silent.
        self.streaming_fallback_reason = streaming_fallback_reason
        super().__init__(
            self.caches,
            dynamic=dynamic,
            max_budget_bytes=max_budget_bytes,
            min_budget_bytes=min_budget_bytes,
            stall_target=stall_target,
            min_cap=min_cap,
        )

    # -- PerLayerResidencyAdapter hooks ------------------------------------
    def _iter_units(self):
        """The per-layer ``ExpertCache``s in wrap order."""
        return iter(self.caches)

    def _per_slot_bytes(self) -> int:
        return sum(
            max(1, int(getattr(c, "per_expert_bytes", 0) or 0))
            for c in self.caches
        )

    def _initial_base_cap(self) -> int:
        return (
            min(int(c.capacity) for c in self.caches) if self.caches else 0
        )

    def _floor_cap(self) -> int:
        # Never shrink a layer below one token's decode working set:
        # the wrap-time capacity already floors at the model's routing
        # top-k, and going below it turns every step into a fetch storm.
        # Covers the no-explicit-min_cap case (8 like the legacy wrap
        # default); an explicit min_cap wins inside arm_dynamic.
        return min(8, self.base_cap)

    def _apply_layer_caps(self) -> None:
        for idx, cache in enumerate(self.caches):
            cache.resize(self._cap_for(idx))

    def _clear_units(self) -> None:
        for cache in self.caches:
            cache.clear()

    # No ``streaming_guard_info``: the scheduler's prefill-bank transient
    # exists for the generic path's lazy mini-banks; the legacy slot
    # buffers are persistent and pre-allocated, so that term does not
    # apply. The scheduler reads it via getattr-with-default.

    def summary(self) -> dict:
        # The governor snapshot runs OUTSIDE the state lock on purpose:
        # governor.summary() takes gov._lock while observe()/tick() hold
        # gov._lock across cache.resize() -> state._lock. Taking the
        # state lock first here would invert the mandated
        # governor->cache order and AB-BA deadlock against a tick; the
        # governor numbers don't need the state lock anyway.
        gov = self.governor.summary() if self.governor else {}
        with self._lock:
            hits = sum(c.hits for c in self.caches)
            misses = sum(c.misses for c in self.caches)
            resident = sum(len(c.slot_of) for c in self.caches)
        total = hits + misses
        out = {
            "hits": hits,
            "misses": misses,
            "hit_rate": (hits / total) if total else 0.0,
            "evictions": self.evictions,
            "resident": resident,
            "capacity_per_layer": self.base_cap,
            "layers": self.num_layers,
            "governor": gov,
        }
        if self.streaming_fallback_reason:
            # Folds into expert_streaming_summary's backing slot.
            out["streaming_fallback_reason"] = self.streaming_fallback_reason
        return out


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
    return type(obj).__name__ == "SwitchGLU" and hasattr(obj, "activation")


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

    parent = path.rsplit(".", 1)[0] if "." in path else ""
    # Stacked scheme first (exact spelling, then de-nested checkpoint
    # spelling), then per-expert. The first probe hit wins; the loop below
    # validates it strictly. Falling through to an exact-spelling
    # per-expert view keeps the decline reason pointing at a concrete path
    # when the checkpoint has neither scheme.
    view: _GLUStoreView | None = None
    for cand in _checkpoint_prefixes(path):
        stacked = _GLUStoreView(store, cand)
        if stacked.has("gate_proj", "weight"):
            view = stacked
            break
    if view is None:
        for cand in _checkpoint_prefixes(parent):
            per_expert = _GLUStoreView(store, cand, per_expert=True)
            if per_expert.has("gate_proj", "weight"):
                view = per_expert
                break
    if view is None:
        view = _GLUStoreView(store, parent, per_expert=True)

    for proj in _PROJS:
        lin = getattr(glu, proj)
        for field in fields_of[proj]:
            module_shape = tuple(lin[field].shape)
            if not view._per_expert:
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
                if dtype not in _SUPPORTED_DTYPES:
                    return None, f"{name!r} has unsupported dtype {dtype!r}"
    return view, None


def apply_moe_expert_offload(
    model,
    model_path: str | Path,
    resident_fraction: float = 0.25,
    model_settings=None,
) -> int:
    """Replace covered SwitchGLU instances with offloaded ones.

    Returns the number of layers wrapped (0 when disabled via
    ``OMLX_MOE_EXPERT_OFFLOAD=0``, the model has no stock SwitchGLU, or the
    checkpoint does not cover them). Must run before lazy weights are
    materialized for the memory saving to exist.

    Alias contract (unified backend): ``moe_expert_offload_*`` remain
    supported load-time keys. On model types covered by expert_streaming
    they are served by that stack (fraction = initial budget, dynamic
    governor on); elsewhere the legacy adapter runs unchanged.
    """
    if not env_bool("OMLX_MOE_EXPERT_OFFLOAD", True):
        return 0
    # Unified backend: model types covered by expert_streaming route
    # to our stack — the resident fraction becomes the INITIAL budget and
    # the dynamic governor adapts from there. The legacy fetch-on-miss
    # adapter below stays for gemma4/olmoe (and any layout it alone
    # supports); DeepSeek V4.1 never reaches here (own loader + adapter in
    # patches/deepseek_v41). Falls through to legacy when streaming
    # converts nothing, so behavior never regresses by accident.
    if _streaming_owns_model(model_path):
        routed = _apply_via_streaming(
            model, model_path, resident_fraction, model_settings
        )
        if routed:
            return routed
        logger.info(
            "moe expert offload: streaming backend converted nothing for "
            "%s; falling back to the legacy fetch-on-miss adapter",
            model_path,
        )
        return _apply_legacy_adapter(
            model,
            model_path,
            resident_fraction,
            model_settings,
            streaming_fallback_reason=(
                "expert_streaming owns this model type but converted "
                "nothing"
            ),
        )
    return _apply_legacy_adapter(
        model, model_path, resident_fraction, model_settings
    )


def _apply_legacy_adapter(
    model,
    model_path: str | Path,
    resident_fraction: float = 0.25,
    model_settings=None,
    *,
    streaming_fallback_reason: str | None = None,
) -> int:
    """Legacy fetch-on-miss adapter (legacy-owned types + unit tests).

    Unchanged behavior: wraps covered stock SwitchGLUs in OffloadSwitchGLU.
    Must run before lazy weights are materialized for the memory saving
    to exist.

    ``streaming_fallback_reason`` marks a downgrade: the unified backend
    owned this model type but produced nothing, so the wrapped state is
    stamped with the reason (visible via ``expert_streaming_summary``)
    and logged at error level instead of passing silently.
    """
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return 0
    minimum = _minimum_experts(model_dir)
    try:
        store = CheckpointExpertStore(model_dir)
    except Exception:
        # An unreadable (or repacked co-activation) checkpoint declines
        # offload — the model stays resident, the load must not crash.
        logger.warning(
            "moe expert offload: cannot open checkpoint store under %s",
            model_dir,
            exc_info=True,
        )
        return 0
    if not store:
        logger.warning("moe expert offload: no safetensors under %s", model_dir)
        return 0

    wrapped = 0
    wrapped_mods: list = []
    total_bytes = resident_bytes = 0
    for parent, key, glu, path in list(_iter_switch_glus(model)):
        view, reason = _resolve_store_view(glu, store, path)
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
        new = OffloadSwitchGLU(glu, capacity, view)
        new._layer = wrapped  # wrap order == LegacyOffloadState index
        if isinstance(parent, nn.Module):
            setattr(parent, key, new)  # registers via Module.__setattr__
        else:
            parent[key] = new  # plain list / plain dict
        wrapped_mods.append(new)
        wrapped += 1
        # Dropped source buffers land in the MLX pool, which the server pins
        # to total RAM, so drain per layer to bound the load transient.
        _sync_and_clear_cache()

    if wrapped:
        try:
            # The scheduler serializes requests on this marker — concurrent
            # requests would thrash an LRU sized for a single stream. The
            # store doubles as the shutdown
            # handle — shutdown_expert_streaming() calls close() on it.
            store.streaming_fallback_reason = streaming_fallback_reason
            model._expert_streaming_backing = store  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            # Governor-facing aggregate over the per-layer caches (same
            # duck-type as the V4.1 backing): pressure shrinks/clears the
            # ceilings, proven decode hunger grows them. Precedence is the
            # shared _dynamic_armed rule — explicit setting > env > auto.
            from .expert_streaming import _dynamic_armed, _io_overrides

            io_ov = _io_overrides(model_settings)
            _mg = io_ov.get("expert_streaming_dynamic_max_gib")
            _ng = io_ov.get("expert_streaming_dynamic_min_gib")
            state = LegacyOffloadState(
                [m.cache for m in wrapped_mods],
                dynamic=_dynamic_armed(
                    io_ov.get("expert_streaming_dynamic"), model_settings
                ),
                max_budget_bytes=(
                    int(_mg * 1024**3) if _mg is not None else None
                ),
                min_budget_bytes=(
                    int(_ng * 1024**3) if _ng is not None else None
                ),
                stall_target=io_ov.get(
                    "expert_streaming_dynamic_stall_target"
                ),
                min_cap=minimum,
                streaming_fallback_reason=streaming_fallback_reason,
            )
            for m in wrapped_mods:
                m._state = state
            model._moe_offload_legacy_state = state  # type: ignore[attr-defined]
        except Exception:
            logger.debug("legacy offload state wiring failed", exc_info=True)
        if streaming_fallback_reason:
            # The downgrade must be loud: a streaming-owned model running
            # the legacy adapter is a degraded mode, not a silent choice.
            logger.error(
                "moe expert offload: DOWNGRADED to the legacy "
                "fetch-on-miss adapter: %s (%d layers wrapped)",
                streaming_fallback_reason,
                wrapped,
            )
        logger.info(
            "moe expert offload: wrapped %d layers at %.1f%% residency "
            "(expert tables: %.2f GB total, %.2f GB resident)",
            wrapped,
            100 * resident_fraction,
            total_bytes / 1e9,
            resident_bytes / 1e9,
        )
    return wrapped


def _read_config_model_type(model_path: str | Path) -> str | None:
    """Effective config model_type (top level wins, text_config fallback).

    Thin delegate kept for ``model_loading``'s settings validation: the
    canonical reader is ``expert_streaming.residency.load_config_model_type``
    (normalized); this wrapper only adds HF-repo-id resolution via
    ``_resolve_model_dir``.
    """
    try:
        model_dir = _resolve_model_dir(model_path)
        if model_dir is None:
            return None
        from .expert_streaming.residency import load_config_model_type

        return load_config_model_type(model_dir) or None
    except Exception:
        return None


def _streaming_owns_model(model_path: str | Path) -> bool:
    """True when expert_streaming covers this model type (unified backend)."""
    try:
        from .expert_streaming.residency import streaming_owns_model
    except Exception:
        return False
    # streaming_owns_model reads config.json from a local dir; resolve
    # hub repo ids first so both spellings work here.
    model_dir = _resolve_model_dir(model_path)
    return model_dir is not None and streaming_owns_model(model_dir)


def _apply_via_streaming(
    model,
    model_path: str | Path,
    resident_fraction: float,
    model_settings=None,
) -> int:
    """Convert via the expert_streaming stack; 0 when it converts nothing.

    Downgrade contract: a failure while support is unproven (imports, the
    structural estimate itself) degrades to the legacy adapter — the
    caller stamps that fallback with a downgrade marker. Once
    ``est.supported`` holds the generic converter owns this model family,
    so a crash there is re-raised: a broken conversion must not hide
    behind a silent legacy fallback.
    """
    import dataclasses

    try:
        from ..model_settings import ModelSettings
        from .expert_streaming import (
            _budget_is_pinned,
            convert_model_to_streaming,
        )
        from .expert_streaming.residency import expert_streaming_estimate

        est = expert_streaming_estimate(str(model_path))
    except Exception:
        # Raised while support is unproven: degrade to the legacy adapter
        # (downgrade-stamped by the caller) instead of crashing a load it
        # could still serve.
        logger.error(
            "moe expert offload: streaming backend unavailable for %s; "
            "falling back to the legacy fetch-on-miss adapter",
            model_path,
            exc_info=True,
        )
        return 0
    if not est.supported:
        return 0
    if getattr(model, "_expert_streaming_backing", None) is not None:
        # Already converted by the engine's direct streaming path (both
        # engines stamp the backing on the model): do NOT reconvert —
        # just report the layer count so fusion-skip/materialize logic
        # downstream keeps working. Canonical keys win on double opt-in.
        return int(est.num_moe_layers or 0)
    table_gib = float(est.expert_bytes or 0) / 1024**3
    if table_gib > 0:
        # Honor the user's chosen residency as the STARTING budget; the
        # governor adapts from there (dynamic=True forces it over a pin).
        budget_gib: float | None = max(
            0.5, min(64.0, float(resident_fraction) * table_gib)
        )
    else:
        budget_gib = None  # unknown tables: RAM-scaled auto budget
    if isinstance(model_settings, ModelSettings):
        # The alias contract must not discard the user's streaming
        # tunables (io_depth, pins, cache policy, governor knobs): clone
        # the original settings and force only the fields the alias owns
        # — enabled, a translated budget when none is pinned, and the
        # dynamic governor's default-on.
        overrides: dict = {"expert_streaming_enabled": True}
        if getattr(model_settings, "expert_streaming_dynamic", None) is None:
            overrides["expert_streaming_dynamic"] = True
        if budget_gib is not None and not _budget_is_pinned(model_settings):
            overrides["expert_streaming_budget_gib"] = budget_gib
        settings = dataclasses.replace(model_settings, **overrides)
    else:
        settings = ModelSettings(
            expert_streaming_enabled=True,
            expert_streaming_budget_gib=budget_gib,
            expert_streaming_dynamic=True,
        )
    try:
        _, backing = convert_model_to_streaming(model, model_path, settings)
    except Exception:
        # Fail hard: the structural estimate says streaming owns this
        # family, so a converter crash is a real defect — re-raise rather
        # than hide it behind a legacy fallback that was never asked for.
        logger.error(
            "moe expert offload: streaming backend crashed on a "
            "streaming-owned model (%s); refusing silent legacy fallback",
            model_path,
            exc_info=True,
        )
        raise
    if backing is None:
        return 0
    try:
        # Keep mmap readers alive for the model lifetime (cf. batched.py).
        model._expert_streaming_backing = backing  # type: ignore[attr-defined]
    except Exception:
        pass
    wrapped = int(est.num_moe_layers or 0)
    logger.info(
        "moe expert offload: unified backend (expert_streaming) converted "
        "%d layers (initial budget %s from %.0f%% residency, dynamic "
        "governor on)",
        wrapped,
        f"{budget_gib:.2f} GiB" if budget_gib else "auto",
        100 * float(resident_fraction),
    )
    return wrapped


def estimate_offload_admission_bytes(
    model_path: str | Path, full_size: int, resident_fraction: float = 0.25
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
    if not env_bool("OMLX_MOE_EXPERT_OFFLOAD", True):
        return full_size
    try:
        model_dir = _resolve_model_dir(model_path)
        if model_dir is None:
            return full_size
        minimum = _minimum_experts(model_dir)
        config_path = Path(model_dir) / "config.json"
        if config_path.exists():
            kind = json.loads(config_path.read_text()).get("model_type", "")
            # Unified-streaming types: the converter serves any stacked
            # expert bank (quantized or dense), so savings come from the
            # structural estimate — the regex mirror below only describes
            # the legacy adapter's quantized-only contract.
            try:
                from .expert_streaming.residency import (
                    SUPPORTED_TYPES as _STREAMING_TYPES,
                )
                from .expert_streaming.residency import (
                    expert_streaming_estimate,
                    normalize_model_type,
                )

                if normalize_model_type(kind) in _STREAMING_TYPES:
                    est = expert_streaming_estimate(model_dir)
                    if not est.supported or est.experts_per_layer <= 0:
                        return full_size
                    n = int(est.experts_per_layer)
                    capacity = min(
                        n, max(minimum, round(n * resident_fraction))
                    )
                    saved = int(
                        est.expert_bytes * (1.0 - capacity / n)
                    )
                    return full_size - saved if saved > 0 else full_size
            except Exception:
                logger.debug(
                    "streaming-estimate admission fallback failed",
                    exc_info=True,
                )
            if kind == "deepseek_v41":
                return full_size
        # stacked: container -> {"bytes", "fields": {(proj, field)}, "e": set}
        # per-expert: container -> {"bytes", "per_e": {idx: {(proj, field)}}}
        # Field completeness is tracked PER EXPERT, not container-wide: the
        # wrapper verifies every expert's tensors, so one complete expert
        # must not vouch for 31 incomplete ones.
        stacked: dict[str, dict] = {}
        per_expert: dict[str, dict] = {}

        for shard in sorted(Path(model_dir).glob("*.safetensors")):
            header = read_safetensors_header(shard)
            for name, spec in header.items():
                if name == "__metadata__":
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
                        "weight", "scales", "biases"
                    ):
                        b = stacked.setdefault(
                            parts[0], {"bytes": 0, "fields": set(), "e": set()}
                        )
                        b["bytes"] += b1 - b0
                        b["fields"].add((parts[1], parts[2]))
                        b["e"].add(int(shape[0]))

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


def materialize_offload_state(model) -> int:
    """Evaluate every offload cache's arrays on the loading thread's stream.

    ``ExpertCache`` keeps its slot map and resident slots on plain object
    attributes, so the engine's ``materialize_lazy_state`` walk never reaches
    them. Left lazy, they stay bound to the loader thread's stream and the
    first request from another thread dies with ``RuntimeError: There is no
    Stream(gpu, N) in current thread``. Call this right after
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
        if isinstance(obj, OffloadSwitchGLU):
            layers += 1
            cache = obj.cache
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
    """Aggregate hit/miss counters over all offloaded layers."""
    hits = misses = layers = 0
    stack = [model]
    seen = set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, OffloadSwitchGLU):
            hits += obj.cache.hits
            misses += obj.cache.misses
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
    }


__all__ = [
    "CheckpointExpertStore",
    "OffloadSwitchGLU",
    "apply_moe_expert_offload",
    "materialize_offload_state",
    "moe_offload_stats",
]
