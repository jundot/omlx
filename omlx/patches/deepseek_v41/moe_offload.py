# SPDX-License-Identifier: Apache-2.0
"""Bounded expert residency for V4.1, preserving its projection arithmetic."""

import contextvars
import json
import math
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .convert import repack_weight
from .quantization import QuantizedProjection
from .storage import TensorFile, decode_array
from ..expert_streaming.slot_cache import (
    SlotArena,
    working_set_step,
)


def _fetch_threads() -> int:
    """Parallel miss-fetch width (0 = serial path).

    Opt-in via OMLX_V41_FETCH_THREADS: workers run plan.fetch (file IO +
    CPU decode, no mx assignment/eval) while the main thread assigns the
    joined results in deterministic order. TensorFile serializes per-file
    reads on its own lock; threads parallelize across the 82 shard files
    and overlap decode with waiting IO. LRU/frozen order and numerics are
    identical to the serial path by construction (ordered join).
    """
    try:
        return max(0, int(os.environ.get("OMLX_V41_FETCH_THREADS", "0")))
    except (TypeError, ValueError):
        return 0


def _stage_enabled() -> bool:
    """Prev-token speculative staging (parity with the generic path).

    After a decode ensure, the next MoE layer's likely expert set (its
    routing on the previous token) is fetched on a dedicated worker while
    this layer's MoE compute runs. Payloads join through the same commit
    path as demand fetches, so ordering and numerics are identical.
    OMLX_V41_STAGE=0 disables.
    """
    return os.environ.get("OMLX_V41_STAGE", "1") != "0"


def _stage_max(plan) -> int:
    try:
        cap = int(os.environ.get("OMLX_V41_STAGE_MAX", "0"))
    except (TypeError, ValueError):
        cap = 0
    if cap <= 0:
        cap = max(16, 2 * int(getattr(plan, "n_activated", 8) or 8))
    return cap


def _stage_one(plan, prefix, expert):
    return {proj: plan.fetch(prefix, proj, expert) for proj in _PROJECTIONS}


def _stage_span(plan, prefix, experts):
    """Coalesced speculative read: one merged span, per-expert payloads.

    ``experts`` is an ascending run (see ``_span_groups``); the shared
    ``fetch_span`` covers [first, last] once per projection and each
    expert decodes only its own row — gap overfetch costs I/O, not
    decode CPU (same contract as the demand path).
    """
    lo, hi = experts[0], experts[-1] + 1
    raw = {proj: plan.fetch_span(prefix, proj, lo, hi) for proj in _PROJECTIONS}
    return {
        expert: {
            proj: {
                field: decode_array(block[expert - lo], dtype)
                for field, (block, dtype) in raw[proj].items()
            }
            for proj in _PROJECTIONS
        }
        for expert in experts
    }


class _StagedSpan:
    """Per-expert handle on one coalesced staged span read.

    The staging worker submits ONE task per merged run; every expert in
    the run maps to a handle whose ``result()`` slices out its own
    payload, so the demand join keeps its per-expert contract.
    """

    __slots__ = ("_fut", "_expert")

    def __init__(self, fut, expert):
        self._fut, self._expert = fut, int(expert)

    def result(self, timeout=None):
        got = self._fut.result(timeout)
        return None if got is None else got.get(self._expert)

    def cancel(self):
        return self._fut.cancel()

    def done(self):
        return self._fut.done()


class _WorkingSetOverCap(ValueError):
    """The chunk's routed set exceeds the LIVE cap after a shrink.

    ``_ensure_locked`` raises this ahead of the arena's own overflow
    check so ``OffloadedExpert.__call__`` can tell a mid-call governor
    shrink (retry with a finer split) from a genuinely unservable
    working set. Subclasses ValueError and keeps the arena's message so
    direct ``ensure`` callers keep the historical capacity contract.
    """


def _span_reads() -> bool:
    """Explicit bounded demand reads, not the mmap gather.

    Logical ids are physical rows — every expert is read through the
    shared ExpertBackingStore (stateless preadv into caller buffers)
    instead of the mmap page-fault gather.
    OMLX_V41_SPAN_READS=0 keeps the per-expert gather as an
    escape hatch.
    """
    return os.environ.get("OMLX_V41_SPAN_READS", "1") != "0"


def _span_gap() -> int:
    """Max physical-row gap merged into one span read (overfetch rows).

    Default 1 — merges strictly adjacent rows only (diff <= 1 means
    contiguous bytes: zero dead bytes, one fewer command, same file
    lock). gap >= 2 bridges holes and reads unrouted experts (dead
    bytes), a net loss on NVMe. Env override stays for
    per-machine tuning on slower-seek disks."""
    try:
        return max(0, int(os.environ.get("OMLX_V41_SPAN_GAP", "1")))
    except (TypeError, ValueError):
        return 1


def _span_max_rows() -> int:
    """Cap on merged span length in rows (bounds one read's overfetch)."""
    try:
        return max(1, int(os.environ.get("OMLX_V41_SPAN_MAX", "64")))
    except (TypeError, ValueError):
        return 64


def _ra_enabled() -> bool:
    """Exact-demand kernel readahead (megaplan F1) — F_RDADVISE only.

    ``OffloadedExpert.__call__`` already holds the whole call's routing
    in ``flat_i`` before any fetch runs, so prefill and verify demand is
    COMPUTED, not predicted: advise it into the page cache ahead of the
    ensure loop and the demand reads hit warm pages. Hints only — zero
    userspace bytes materialize, ordering and numerics are untouched.
    OMLX_V41_RA=0 disables.
    """
    return os.environ.get("OMLX_V41_RA", "1") != "0"


def _ra_tokens() -> int:
    """Lookahead window in route rows (tokens) advised per chunk."""
    try:
        return max(1, int(os.environ.get("OMLX_V41_RA_TOKENS", "64")))
    except (TypeError, ValueError):
        return 64


def _ra_gap() -> int:
    """Id gap bridged by one advisory run (dead bytes cost no decode)."""
    try:
        return max(0, int(os.environ.get("OMLX_V41_RA_GAP", "8")))
    except (TypeError, ValueError):
        return 8


def _ra_max_rows() -> int:
    """Cap on one advisory run's span in rows."""
    try:
        return max(1, int(os.environ.get("OMLX_V41_RA_MAX", "256")))
    except (TypeError, ValueError):
        return 256


def _ra_next() -> bool:
    """Whole-bank advise of the NEXT MoE layer when demand saturates.

    At saturated coverage (long prompts) the next layer's demand is
    almost surely its whole bank, so advising it at this layer's entry
    buys a full layer of readahead lead time. Gated by
    ``_ra_next_cov`` so short prompts only advise the exact demand.
    OMLX_V41_RA_NEXT=0 disables.
    """
    return os.environ.get("OMLX_V41_RA_NEXT", "1") != "0"


def _ra_next_cov() -> float:
    """Coverage threshold (unique demanded / bank size) for next-layer."""
    try:
        return min(1.0, max(0.0, float(os.environ.get(
            "OMLX_V41_RA_NEXT_COV", "0.5"
        ))))
    except (TypeError, ValueError):
        return 0.5


def _prefill_cap() -> int:
    """Transient per-layer residency ceiling during prefill (megaplan F2).

    ``working_set_step`` = cap // top_k: at cap=10/top_k=6 a prefill
    chunk is ONE token, serializing thousands of ensures per layer.
    Raising ``book.cap`` for the call's duration widens the chunk; rooms
    grow on demand inside the call and ``compact(rooms=entry)`` hands
    them back in the finally — decode-hot residents sit at the MRU end
    (frozen commits never touch them), so the restore compact keeps
    exactly them. Default 64 rows/layer (megaplan A/B winner: the
    transient footprint decays into a warm resident set for decode);
    0 disables. OMLX_V41_PREFILL_CAP, rows per layer.
    """
    try:
        return max(0, int(os.environ.get("OMLX_V41_PREFILL_CAP", "64")))
    except (TypeError, ValueError):
        return 0


def _chunk_lookahead() -> bool:
    """Materialized next-chunk staging for multi-row calls (F2).

    Each iteration submits the NEXT chunk's exact demand to the staging
    worker, so its fetch overlaps this chunk's ensure+compute and the
    ensure joins on staged payloads instead of reading cold. Exact
    demand — every staged row is consumed by the next ensure; nothing
    mispredicts. OMLX_V41_CHUNK_STAGE=0 disables.
    """
    return os.environ.get("OMLX_V41_CHUNK_STAGE", "1") != "0"


def _verify_scratch() -> int:
    """Extra rows verify may hold per layer on top of cap (megaplan F4).

    A DSpark verify block routes ~depth*top_k unique experts per layer —
    at cap=10 the union (~30) cannot fit, so today every verify call
    re-splits into one-token chunks and verify-installed rows churn each
    other while the FIRST chunk also eats decode residents. With
    scratch > 0 the admission bound becomes cap+scratch: the union lands
    in ONE ensure (one coalesced fetch round), acquire() evicts only
    verify-installed rows, and rooms grow up to the bound instead of
    consuming decode-hot residents. Scratch rows persist between rounds
    as cold-end residents — free cross-iteration hits — and the decode
    trim/governor compact reclaims them under pressure. Default 8
    rows/layer (megaplan A/B winner: verify_misses −22%, resident
    evictions 443→121; 32 OOM'd the 48 GiB bench box — size against
    headroom). 0 keeps the legacy chunked-verify behavior.
    OMLX_V41_VERIFY_SCRATCH, rows/layer.
    """
    try:
        return max(0, int(os.environ.get("OMLX_V41_VERIFY_SCRATCH", "8")))
    except (TypeError, ValueError):
        return 0


def _draft_offload() -> int:
    """Resident expert rows per DSpark draft stage (megaplan F5).

    The preserved DSpark head is three stages x 128 routed experts —
    ~7-10 GiB on the production layout OUTSIDE the trunk offload plan.
    A positive value wraps ``mtp.*.ffn.experts`` in the same
    OffloadedExpert machinery with this many resident rows per stage,
    freeing most of that memory at the cost of draft misses on the
    speculative path. Draft slots deliberately stay out of the trunk
    backing: the governor's budget math keeps covering only the 40
    backbone layers. Default 8 rows/stage (megaplan A/B winner: the
    freed ~6.4 GiB raised trunk cap 11→17 and was a net decode win on
    the 48 GiB bench box). 0 keeps draft stages fully resident.
    OMLX_V41_DRAFT_OFFLOAD.
    """
    try:
        return max(0, int(os.environ.get("OMLX_V41_DRAFT_OFFLOAD", "8")))
    except (TypeError, ValueError):
        return 0


def _span_groups(rows, gap, cap_rows):
    """Ascending expert rows -> merged span lists.

    Neighbors merge while ``row - prev <= gap`` and the run stays within
    ``cap_rows`` rows. ``gap`` bounds the DIFFERENCE between adjacent ids:
    0 disables merging entirely (adjacent ids differ by 1 > 0); only
    gap >= 1 joins strictly contiguous rows. This is NOT the generic
    path's ``merge_gap`` (missing rows bridged) — V4.1 gap=1 corresponds
    to generic merge_gap=0.
    """
    groups, cur = [], []
    for row in rows:
        if cur and (row - cur[-1] > gap or row - cur[0] + 1 > cap_rows):
            groups.append(cur)
            cur = []
        cur.append(row)
    if cur:
        groups.append(cur)
    return groups

# Layer 3 seam: the DSpark verify driver enters
# verify_scope() around the draft-block forward. While set, _ExpertSlots
# serves verify traffic WITHOUT disturbing decode-hot LRU order — hits
# skip the move-to-end reorder and misses land on the oldest slot. The
# shared block kernels (layer 1) and plan builders (layer 2) are never
# touched; only this adapter's recency accounting changes.
_verify_frozen: contextvars.ContextVar = contextvars.ContextVar(
    "v41_verify_frozen", default=False
)


@contextmanager
def verify_scope():
    """Freeze expert-slot recency for one DSpark verify block."""
    token = _verify_frozen.set(True)
    try:
        yield
    finally:
        _verify_frozen.reset(token)

_PROJECTIONS = ("w1", "w3", "w2")
_FLOAT_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


class ExpertOffloadPlan:
    """Validate every routed tensor before promising any memory savings."""

    def __init__(self, path, raw, mapping, config, fraction, draft_rows=0):
        if not 0 < fraction <= 1:
            raise ValueError("MoE resident fraction must be in (0, 1]")
        self.path = Path(path)
        self.mapping = mapping
        self.converted = raw.get("omlx_deepseek_v41")
        self.count = config.n_routed_experts
        self.n_activated = int(config.n_activated_experts)
        self.capacity = min(
            self.count, max(config.n_activated_experts, round(self.count * fraction))
        )
        # Per-prefix counts/capacities: draft stages (F5) carry their own
        # 128-expert bank and resident-row budget, so shape validation and
        # slot sizing cannot rely on the trunk's count/capacity globals.
        self._count_of = {}
        self._capacity_of = {}
        self.layers = {}
        self.excluded_keys = set()
        self.resident_bytes = 0
        self.full_bytes = 0
        self._headers = {}
        self._readers = {}
        self._readers_lock = threading.Lock()
        self._store = None
        self._closed = False
        self._stage_pool = None
        self.draft_bytes = 0
        # A dir carrying expert_order.json is a repacked co-activation
        # checkpoint — expert ordering is unsupported; serving it as
        # identity would silently return permuted experts. Refuse loudly.
        if (self.path / "expert_order.json").is_file():
            raise ValueError(
                "repacked co-activation checkpoint (expert_order.json): "
                "ordering support was removed — use the original checkpoint"
            )
        # Draft weights count as savings only when they are actually
        # excluded. A preserved DSpark head loads its language_model.mtp.*
        # tensors like any other — excluding them here would fail the
        # completeness check on preserve_mtp + offload + converted
        # checkpoints, and counting them in draft_bytes would overstate
        # estimate_expert_savings by the resident head size.
        if self.converted is not None and not getattr(
            config, "preserve_mtp", False
        ):
            for key in mapping:
                if key.startswith("language_model.mtp."):
                    entry = self._entry(key)
                    self.draft_bytes += (
                        entry["data_offsets"][1] - entry["data_offsets"][0]
                    )
        for layer in range(config.n_layers):
            prefix = f"language_model.layers.{layer}.ffn.experts"
            specs = {}
            for proj in _PROJECTIONS:
                shape = (
                    (config.dim, config.moe_inter_dim)
                    if proj == "w2"
                    else (config.moe_inter_dim, config.dim)
                )
                spec, size = self._projection(prefix, proj, shape, self.count)
                specs[proj] = spec
                self.full_bytes += size
                self.resident_bytes += size * self.capacity // self.count
            self.layers[prefix] = specs
            self._count_of[prefix] = self.count
            self._capacity_of[prefix] = self.capacity
        # F5 opt-in DSpark draft offload: same stacked/per-expert layout
        # as trunk experts (converted names stack under language_model.
        # mtp.{stage}.ffn.experts), smaller bank and its own row budget.
        # Bytes stay in draft_* counters — full_bytes/resident_bytes keep
        # describing the 40 trunk layers the governor sizes against.
        self.draft_rows = int(draft_rows or 0)
        self.draft_full_bytes = 0
        self.draft_resident_bytes = 0
        if self.draft_rows:
            dcount = int(
                getattr(config, "dspark_n_routed_experts", 0)
                or config.n_routed_experts
            )
            dactive = int(
                getattr(config, "dspark_n_activated_experts", 0)
                or config.n_activated_experts
            )
            if not getattr(config, "preserve_mtp", False) or not getattr(
                config, "n_mtp_layers", 0
            ):
                raise ValueError(
                    "DSpark draft offload requires a preserved draft head "
                    "(preserve_mtp with n_mtp_layers > 0)"
                )
            # A request wider than the bank is a resident stage, not an
            # error — the default must stay safe on small draft banks.
            self.draft_rows = min(self.draft_rows, dcount)
            if self.draft_rows < dactive:
                raise ValueError(
                    "OMLX_V41_DRAFT_OFFLOAD must hold at least the "
                    f"activated set ({dactive}); got {self.draft_rows}"
                )
            for stage in range(config.n_mtp_layers):
                prefix = f"language_model.mtp.{stage}.ffn.experts"
                specs = {}
                for proj in _PROJECTIONS:
                    shape = (
                        (config.dim, config.moe_inter_dim)
                        if proj == "w2"
                        else (config.moe_inter_dim, config.dim)
                    )
                    spec, size = self._projection(prefix, proj, shape, dcount)
                    specs[proj] = spec
                    self.draft_full_bytes += size
                    self.draft_resident_bytes += (
                        size * self.draft_rows // dcount
                    )
                self.layers[prefix] = specs
                self._count_of[prefix] = dcount
                self._capacity_of[prefix] = self.draft_rows

    def count_of(self, prefix):
        return self._count_of.get(prefix, self.count)

    def capacity_of(self, prefix):
        return self._capacity_of.get(prefix, self.capacity)

    def _entry(self, key):
        filename = self.mapping[key]
        if filename not in self._headers:
            path = self.path / filename
            with path.open("rb") as file:
                length = struct.unpack("<Q", file.read(8))[0]
                header = json.loads(file.read(length))
            self._headers[filename] = header
        entry = self._headers[filename][key]
        self.excluded_keys.add(key)
        return entry

    def _projection(self, prefix, proj, logical, count=None):
        count = self.count if count is None else int(count)
        if self.converted is not None:
            name = f"{prefix}.{proj}"
            spec = self.converted.get("quantized_modules", {}).get(name)
            fields = ["weight"]
            if spec:
                fields += ["scales"] + (["biases"] if spec["mode"] == "affine" else [])
            entries = {f: self._entry(f"{name}.{f}") for f in fields}
            for field, entry in entries.items():
                shape = (count, *logical)
                dtype = entry["dtype"]
                if spec:
                    bits, group = spec["bits"], spec.get("group_size", 32)
                    if logical[-1] % group:
                        raise ValueError(f"Invalid expert group size: {name}")
                    shape = (
                        count,
                        logical[0],
                        (
                            logical[1] * bits // 32
                            if field == "weight"
                            else logical[1] // group
                        ),
                    )
                    allowed = (
                        {"U32"}
                        if field == "weight"
                        else (set(_FLOAT_BYTES) if spec["mode"] == "affine" else {"U8"})
                    )
                    if dtype not in allowed:
                        raise ValueError(f"Invalid expert dtype: {name}.{field}")
                elif dtype not in _FLOAT_BYTES:
                    raise ValueError(f"Unsupported expert dtype: {name}")
                if tuple(entry["shape"]) != shape:
                    raise ValueError(f"Invalid expert shape: {name}.{field}")
            size = sum(
                e["data_offsets"][1] - e["data_offsets"][0] for e in entries.values()
            )
        else:
            base = prefix.removeprefix("language_model.")
            signature = None
            size = 0
            for expert in range(count):
                name = f"{base}.{expert}.{proj}"
                entry = self._entry(name + ".weight")
                dtype = entry["dtype"]
                if dtype.startswith("F8_E4M3") or dtype in ("I8", "U8"):
                    bits = 8 if dtype.startswith("F8_E4M3") else 4
                    scale = self._entry(name + ".scale")
                    expected = (logical[0], logical[1] * bits // 8)
                    scale_shape = (
                        math.ceil(logical[0] / 32) if bits == 8 else logical[0],
                        logical[1] // 32,
                    )
                    if (
                        not scale["dtype"].startswith("F8_E8M0")
                        or tuple(scale["shape"]) != scale_shape
                    ):
                        raise ValueError(f"Invalid expert scales: {name}")
                    current = {"bits": bits, "mode": f"mxfp{bits}"}
                    size += math.prod(expected) + logical[0] * logical[1] // 32
                elif dtype in _FLOAT_BYTES:
                    expected, current = logical, None
                    if name + ".scale" in self.mapping:
                        raise ValueError(f"Unexpected expert scales: {name}")
                    size += math.prod(logical) * _FLOAT_BYTES[dtype]
                else:
                    raise ValueError(f"Unsupported expert dtype: {name}")
                if tuple(entry["shape"]) != expected:
                    raise ValueError(f"Invalid expert shape: {name}")
                item = (dtype, current)
                if expert and item != signature:
                    raise ValueError(f"Mixed expert formats within {prefix}.{proj}")
                signature, spec = item, current
        return spec, size

    def _backing(self):
        """Shared ExpertBackingStore for converted stacked expert rows.

        Lazy: only the converted path resolves stacked keys through it;
        whole-tensor resident fills and unconverted checkpoints keep the
        TensorFile path below.

        ``_closed`` is re-checked inside ``_readers_lock`` for the same
        reason ``_read``/``_stage_executor`` do: close() swaps
        ``_store`` out under that lock, so a caller that passed the
        unlocked check just before close must not create a store no
        close() will ever reap.
        """
        store = self._store
        if store is None:
            with self._readers_lock:
                if self._closed:
                    raise RuntimeError("MoE expert store is closed")
                store = self._store
                if store is None:
                    from ..expert_streaming.shard_bank import (
                        ExpertBackingStore,
                    )

                    store = self._store = ExpertBackingStore(self.path)
        return store

    def _read(self, key, expert=None):
        if self._closed:
            raise RuntimeError("MoE expert store is closed")
        if expert is not None:
            # Stacked expert rows ride the shared backing store — preadv
            # into caller buffers, run coalescing, read telemetry.
            store = self._backing()
            reader = store._reader_for_key(key, int(expert))
            return (
                store.load_expert_slice(key, int(expert)),
                reader.header[key]["dtype"],
            )
        filename = self.mapping[key]
        reader = self._readers.get(filename)
        if reader is None:
            # Get-or-create under a lock: parallel miss-fetch workers may
            # race here; TensorFile.read itself is already thread-safe.
            # Re-check _closed inside the lock so close() cannot clear the
            # registry and then leak a reader created just after it.
            with self._readers_lock:
                if self._closed:
                    raise RuntimeError("MoE expert store is closed")
                reader = self._readers.get(filename)
                if reader is None:
                    reader = self._readers[filename] = TensorFile(
                        self.path / filename
                    )
        return reader.read(key, rows=expert)

    def _read_span(self, key, lo, hi):
        if self._closed:
            raise RuntimeError("MoE expert store is closed")
        store = self._backing()
        reader = store._reader_for_key(key, lo)
        rp = reader._rp_for(key)
        block = np.empty((hi - lo, rp.expert_bytes), dtype=np.uint8)
        if not store.read_expert_into(
            [(key, list(range(lo, hi)))], [block]
        ):
            raise ValueError(f"Span read failed: {key}[{lo}:{hi}]")
        return (
            block.view(rp.np_dtype).reshape(hi - lo, *rp.per_shape),
            reader.header[key]["dtype"],
        )

    def fetch(self, prefix, proj, expert):
        if self.converted is not None:
            spec = self.layers[prefix][proj]
            fields = ["weight"]
            if spec:
                fields += ["scales"] + (["biases"] if spec["mode"] == "affine" else [])
            row = int(expert)
            return {
                field: decode_array(*self._read(f"{prefix}.{proj}.{field}", row))
                for field in fields
            }
        name = f"{prefix.removeprefix('language_model.')}.{expert}.{proj}"
        raw, dtype = self._read(name + ".weight")
        scale, scale_dtype = (
            self._read(name + ".scale")
            if name + ".scale" in self.mapping
            else (None, None)
        )
        return repack_weight(raw, dtype, scale, scale_dtype)[0]

    def _field_keys(self, prefix, proj):
        """Stacked store keys for one projection (weight + quant fields)."""
        spec = self.layers[prefix][proj]
        fields = ["weight"]
        if spec:
            fields += ["scales"] + (
                ["biases"] if spec["mode"] == "affine" else []
            )
        return [f"{prefix}.{proj}.{f}" for f in fields]

    def fetch_span(self, prefix, proj, lo, hi):
        """Read physical rows ``[lo, hi)`` of one projection contiguously.

        Returns raw ``{field: (np_block, dtype)}`` — the caller decodes
        only the requested rows, so gap-merged overfetch costs I/O but
        no decode CPU. Converted checkpoints only (stacked dim-0 rows).
        """
        return {
            key.rsplit(".", 1)[-1]: self._read_span(key, lo, hi)
            for key in self._field_keys(prefix, proj)
        }

    def _fetch_executor(self):
        """Miss-fetch pool: the generic per-depth executor registry.

        io_pool_for keeps one shared executor per worker depth
        process-wide (device depth is global, not
        per-model) so V4.1 rides the same pools as every other streamed
        family and engine reloads don't respawn threads. Only reached
        when OMLX_V41_FETCH_THREADS > 0 — callers guard on it.
        """
        from ..expert_streaming.streaming_switch import io_pool_for

        return io_pool_for(_fetch_threads())

    def _stage_executor(self):
        """Dedicated single-worker staging pool.

        Speculative next-layer fetches queue behind each other here
        instead of competing with demand workers on the fetch pool —
        and stay available when ``OMLX_V41_FETCH_THREADS`` is 0.

        ``_closed`` and ``_stage_pool`` move together under
        ``_readers_lock`` (close() takes the same lock), so a
        stage_predicted racing engine stop gets None and no-ops instead
        of resurrecting a pool only a second close() would reap.
        """
        with self._readers_lock:
            if self._closed:
                return None
            if self._stage_pool is None:
                self._stage_pool = ThreadPoolExecutor(max_workers=1)
            return self._stage_pool

    def close(self):
        with self._readers_lock:
            self._closed = True
            # Only the stage pool is plan-owned; the fetch pool is the
            # shared io_pool_for registry — shutting it would break other
            # users. _read()/_backing()/_stage_executor() re-check
            # _closed inside this lock, so none can leak a reader, store
            # or pool past it.
            pool = self._stage_pool
            self._stage_pool = None
            readers = list(self._readers.values())
            self._readers.clear()
            # ``_store`` moves with ``_closed`` under the same lock —
            # _backing() re-checks _closed inside it, so a store cannot
            # be created after this swap and leak past close().
            store = self._store
            self._store = None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        # _read() can still be opening readers on fetch workers; the
        # registry swap above ran under the lock so close can't race a
        # late insert.
        for reader in readers:
            reader.close()
        if store is not None:
            try:
                store.close()
            except Exception:
                pass


class _ExpertSlots:
    def __init__(self, expert, plan, prefix):
        self.expert, self.plan, self.prefix = expert, plan, prefix
        # Working ceiling (governor-driven) vs physical rooms (array rows).
        # Without a backing both stay pinned at the plan capacity, matching
        # the static-fraction adapter.
        # Residency lives in the shared SlotArena: slot map,
        # lock, grow/compact and the acquire/commit/rollback protocol are
        # the same primitive the unified streaming linears use.
        self.specs = {}
        capacity = plan.capacity_of(prefix)
        self.arena = SlotArena(
            capacity,
            _PROJECTIONS,
            self._arrays,
            lambda proj, values: self._bind(proj, values, self.specs[proj]),
            lambda: mx.eval(expert.parameters()),
        )
        self.book = self.arena.book
        self._slots_lock = self.arena.lock
        self.backing = None
        self.layer = None
        # Span-read telemetry: merged contiguous fetches per ensure.
        self.span_reads = 0
        self.span_demand_rows = 0
        self.span_phys_rows = 0
        self.span_fallbacks = 0  # merged reads that degraded to per-expert
        self.staged_failures = 0  # staged futures that yielded no payload
        # F1 exact-demand readahead telemetry (F_RDADVISE hints only).
        self.ra_calls = 0
        self.ra_rows = 0
        self.ra_bytes = 0
        for proj in _PROJECTIONS:
            sample = plan.fetch(prefix, proj, 0)
            values = {
                field: mx.zeros((capacity, *array.shape), dtype=array.dtype)
                for field, array in sample.items()
            }
            spec = plan.layers[prefix][proj]
            self.specs[proj] = spec
            self._bind(proj, values, spec)
        mx.eval(expert.parameters())
        expert.eval()

    # SlotBookkeeping aliases — the bookkeeping fields moved to ``book``;
    # these keep a READ-ONLY attribute surface for callers and tests.
    # Residency mutations go through ``book``/``arena`` explicitly (or
    # ``clear_residency`` below) so they stay visible at the call site.
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
    def cap(self):
        return self.book.cap

    @property
    def hits(self):
        return self.book.hits

    @property
    def misses(self):
        return self.book.misses

    @property
    def evictions(self):
        return self.book.evictions

    # Staging aliases — the side dict and counters live on the arena.
    @property
    def _staged(self):
        return self.arena.staged

    @property
    def staged_hits(self):
        return self.arena.staged_hits

    @property
    def staged_drops(self):
        return self.arena.staged_drops

    @property
    def staged_submits(self):
        return self.arena.staged_submits

    def clear_residency(self):
        """Drop all residency and cancel pending staged fetches."""
        with self._slots_lock:
            self.book.reset()
            staged = list(self._staged.values())
            self._staged.clear()
        for fut in staged:
            try:
                fut.cancel()
            except Exception:
                pass

    def _bind(self, proj, values, spec):
        if spec:
            setattr(self.expert, proj, QuantizedProjection(**values, **spec))
        else:
            getattr(self.expert, proj).weight = values["weight"]

    def _arrays(self, proj):
        lin = getattr(self.expert, proj)
        if isinstance(lin, dict):
            return lin
        names = ["weight"]
        if getattr(lin, "scales", None) is not None:
            names.append("scales")
        if getattr(lin, "biases", None) is not None:
            names.append("biases")
        return {name: getattr(lin, name) for name in names}

    def compact(self, keep, rooms=None):
        """Keep the most-recent ``keep`` entries, remap rows 0..k-1.

        ``rooms`` (>= kept) also shrinks physical storage back below the
        high-water mark — the F2 prefill-cap restore hands grown rows
        back without needing residents to drop first.
        """
        return self.arena.compact(keep, rooms=rooms)

    def ensure(self, indices, phase=None):
        verify = bool(_verify_frozen.get())
        if verify:
            phase = "verify"
        elif phase is None:
            # Direct-call fallback only; OffloadedExpert.__call__ decides
            # once from the pre-chunk route count (a 1-row prefill tail
            # chunk is not decode). A flat 1-D list is one token's route
            # set — the same single-row case a (1, top_k) array encodes.
            phase = (
                "decode"
                if indices.ndim <= 1 or indices.shape[0] == 1
                else "prefill"
            )
        # Per-layer lock: fetch IO runs inside, but only serializes THIS
        # layer — the governor's resize and other layers proceed. Stats
        # are reported to the backing after the lock is released so the
        # lock order stays slots->(nothing) and resize's
        # backing._lock->slots._slots_lock cannot deadlock against it.
        with self._slots_lock:
            rows, missed = self._ensure_locked(
                indices, frozen=verify or phase == "prefill", verify=verify
            )
        if phase == "decode" and self.backing is not None:
            self.backing.note_visit(self.layer, missed)
            # Record this token's routing as the predictor for the
            # same layer next token, then stage the NEXT MoE layer's
            # predicted set while this layer's compute runs. Frozen
            # (verify) ensures skip both — verify traffic would pollute
            # the decode predictor and draft cycles race the schedule.
            self.backing.note_routing(
                self.layer, {int(e) for e in indices.reshape(-1).tolist()}
            )
            self.backing.stage_next(self.layer)
        elif phase == "verify" and self.backing is not None:
            # Telemetry only (megaplan F0): verify stalls are real wall
            # time but deliberately stay OUT of the governor's hunger
            # signal until the verify path can hold its working set
            # (F4 scratch rows).
            self.backing.note_verify_visit(self.layer, missed)
            # F3: feed the verify predictor (per-round routed union) and
            # advise the next layer's predicted demand — under DSpark
            # virtually all trunk traffic is verify, so this predictor
            # is the only one that sees real routing.
            self.backing.note_verify_routing(
                self.layer, {int(e) for e in indices.reshape(-1).tolist()}
            )
            self.backing.advise_verify_next(self.layer)
        return rows

    def stage_predicted(self, experts, limit=None):
        """Submit speculative fetches for predicted demand.

        Runs under this layer's slots lock so residency checks and the
        staged dict stay coherent with a concurrent ensure; payloads are
        pure data — slot assignment, eviction and commit order still
        happen only inside ``_ensure_locked``. Converted checkpoints
        coalesce the predicted set into merged span reads under the same
        ``_span_groups`` bounds as demand — one ``fetch_span`` per run on
        the single staging worker instead of three per-expert fetches.

        ``limit`` overrides the staged-dict bound: the F2 chunk-lookahead
        submits EXACT next-chunk demand (bounded by chunk width), not a
        prediction that may go unconsumed.
        """
        if not _stage_enabled():
            return
        with self._slots_lock:
            room = (
                _stage_max(self.plan) if limit is None else int(limit)
            ) - len(self._staged)
            todo = sorted(
                {
                    int(e)
                    for e in experts
                    if int(e) not in self.book and int(e) not in self._staged
                }
            )[: max(0, room)]
            if not todo:
                return
            pool = self.plan._stage_executor()
            if pool is None:
                # Plan closed between the residency check and here —
                # staging is a hint, never on the demand path; drop it.
                return
            staged_before = len(self._staged)
            if self.plan.converted is not None:
                for group in _span_groups(todo, _span_gap(), _span_max_rows()):
                    try:
                        fut = pool.submit(
                            _stage_span, self.plan, self.prefix, group
                        )
                    except Exception:
                        break
                    for expert in group:
                        self._staged[expert] = _StagedSpan(fut, expert)
            else:
                for expert in todo:
                    try:
                        self._staged[expert] = pool.submit(
                            _stage_one, self.plan, self.prefix, expert
                        )
                    except Exception:
                        break
            self.arena.staged_submits += len(self._staged) - staged_before

    def _advise_keys(self, experts):
        """F_RDADVISE the given expert rows across every projection field.

        Converted checkpoints store stacked rows, so a sorted id run is
        one contiguous byte range per field — ``advise_expert_run``
        collapses each into a single radvisory (and splits at tier
        boundaries internally). Returns (rows, bytes) actually advised;
        0/0 on any failure — readahead is a hint, never the demand path.
        """
        if self.plan.converted is None or not experts:
            return 0, 0
        try:
            store = self.plan._backing()
        except Exception:
            return 0, 0
        total = 0
        for proj in _PROJECTIONS:
            for key in self.plan._field_keys(self.prefix, proj):
                for run in _span_groups(experts, _ra_gap(), _ra_max_rows()):
                    try:
                        ok, nbytes, _seg = store.advise_expert_run(
                            key, run[0], run[-1] - run[0] + 1
                        )
                    except Exception:
                        ok, nbytes = False, 0
                    if ok:
                        total += nbytes
        return len(experts), total

    def advise_demand(self, experts):
        """Readahead the demanded-but-missing experts of an upcoming chunk.

        Exact-demand prefetch (megaplan F1): the caller already knows the
        routing, so the advisory covers only what the ensure will truly
        fetch — residents and in-flight staged rows are filtered under
        the slots lock, then the fcntls run lock-free.
        """
        if not _ra_enabled() or self.plan.converted is None:
            return 0
        with self._slots_lock:
            missing = sorted(
                {
                    int(e)
                    for e in experts
                    if int(e) not in self.book.slot_of
                    and int(e) not in self._staged
                }
            )
        if not missing:
            return 0
        rows, nbytes = self._advise_keys(missing)
        self.ra_calls += 1
        self.ra_rows += rows
        self.ra_bytes += nbytes
        return rows

    def advise_bank(self):
        """Whole-bank readahead for this layer (saturated-coverage case).

        Used for the NEXT layer during a long prefill: when this layer's
        union covers most of the bank, the next layer's almost surely
        does too — advising all rows at once buys a full layer of lead
        time while this layer's compute+IO runs.
        """
        if (
            not _ra_enabled()
            or not _ra_next()
            or self.plan.converted is None
        ):
            return 0
        try:
            store = self.plan._backing()
        except Exception:
            return 0
        total = 0
        for proj in _PROJECTIONS:
            for key in self.plan._field_keys(self.prefix, proj):
                try:
                    ok, nbytes, _seg = store.advise_expert_run(
                        key, 0, self.plan.count_of(self.prefix)
                    )
                except Exception:
                    ok, nbytes = False, 0
                if ok:
                    total += nbytes
        if total:
            self.ra_calls += 1
            self.ra_rows += self.plan.count_of(self.prefix)
            self.ra_bytes += total
        return total

    def _ensure_locked(self, indices, frozen, verify=False):
        needed = set(indices.reshape(-1).tolist())
        scratch = _verify_scratch() if verify else 0
        if len(needed) > self.book.cap + scratch:
            # A governor shrink can land between the caller's chunk-size
            # read and this ensure: surface the overflow as a marker so
            # __call__ re-splits the tail and retries instead of dying
            # on the arena's hard check mid-prefill. Runs under
            # _slots_lock, so book.cap is the live post-shrink value.
            raise _WorkingSetOverCap(
                "Expert working set exceeds resident capacity"
            )
        # ``frozen`` commits land at the eviction-candidate end and hits
        # skip LRU promotion. Verify blocks use it so draft-only experts
        # leave first; prefill chunks use it as scan resistance — a long
        # prompt then churns its own cold-end rows instead of sweeping
        # the decode-hot set (the generic path's _prefill_cap pair plays
        # the same role on its shared pool). Under verify, ``scratch``
        # admits a working set wider than cap (F4): the verify union
        # lands in one ensure and victims come from the verify-installed
        # region only.
        missed = self.arena.ensure_set(
            needed, frozen, self._produce, verify=verify, scratch=scratch
        )
        return (
            self.arena.rows_for(indices.reshape(-1).tolist()).reshape(
                indices.shape
            ),
            missed,
        )

    def _produce(self, fetch_list):
        """Pass 2: fetch (IO + CPU decode, no mx assignment/eval). Serial by
        default; worker threads when OMLX_V41_FETCH_THREADS>0. Workers
        only touch plan.fetch; the ordered join keeps commit order (and
        hence LRU/frozen order and numerics) identical to serial.
        """

        def _one(item):
            expert, _slot, _victim = item
            fut = self._staged.pop(expert, None)
            if fut is not None:
                payload = None
                try:
                    payload = fut.result()
                except Exception:
                    pass
                if payload:
                    self.arena.staged_hits += 1
                    return payload
                self.staged_failures += 1
            return {
                proj: self.plan.fetch(self.prefix, proj, expert)
                for proj in _PROJECTIONS
            }

        threads = _fetch_threads()
        if (
            _span_reads()
            and fetch_list
            and self.plan.converted is not None
        ):
            return self._fetch_spans(fetch_list)
        if threads > 0 and len(fetch_list) > 1:
            pool = self.plan._fetch_executor()
            return list(pool.map(_one, fetch_list))
        return [_one(item) for item in fetch_list]

    def _fetch_spans(self, fetch_list):
        """Contiguous-row fetch for missed experts.

        Logical ids are physical rows.
        Sorts the misses and merges runs under ``OMLX_V41_SPAN_GAP``
        (bounded by SPAN_MAX — a row-DIFFERENCE bound, so gap=0 never
        merges and only gap>=1 joins strictly adjacent ids; the generic
        ``merge_gap`` counts bridged holes instead, so V4.1 gap=1 ==
        generic merge_gap=0), reads each span once per projection
        through ``fetch_span`` (bounded readinto), and returns payloads
        aligned with ``fetch_list`` — commit order and per-expert
        numerics are identical to the per-expert path by construction.
        A span that fails degrades its rows to the per-expert path
        (``span_fallbacks``) instead of failing the whole ensure.
        """
        gap = _span_gap()
        cap_rows = _span_max_rows()
        # Staged futures keep their contract: consume them per expert and
        # span-fetch only true demand misses.
        demand = []
        resolved = {}
        for item in fetch_list:
            expert = item[0]
            fut = self._staged.pop(expert, None)
            if fut is None:
                demand.append(item)
                continue
            payload = None
            try:
                payload = fut.result()
            except Exception:
                pass
            if payload:
                self.arena.staged_hits += 1
                resolved[expert] = payload
            else:
                self.staged_failures += 1
                demand.append(item)
        demand.sort(key=lambda item: int(item[0]))
        spans = _span_groups([int(item[0]) for item in demand], gap, cap_rows)

        def _one_span(span):
            lo, hi = span[0], span[-1] + 1
            payloads = {}
            for proj in _PROJECTIONS:
                raw = self.plan.fetch_span(self.prefix, proj, lo, hi)
                for expert in span:
                    payloads.setdefault(expert, {})[proj] = {
                        field: decode_array(block[expert - lo], dtype)
                        for field, (block, dtype) in raw.items()
                    }
            return payloads

        failed = []
        if _fetch_threads() > 0 and len(spans) > 1:
            pool = self.plan._fetch_executor()
            futures = [pool.submit(_one_span, span) for span in spans]
            for span, fut in zip(spans, futures):
                try:
                    resolved.update(fut.result())
                except Exception:
                    failed.append(span)
        else:
            for span in spans:
                try:
                    resolved.update(_one_span(span))
                except Exception:
                    failed.append(span)
        for span in failed:
            self.span_fallbacks += 1
            for expert in span:
                resolved[expert] = _stage_one(self.plan, self.prefix, expert)
        self.span_reads += len(spans)
        self.span_demand_rows += len(demand)
        self.span_phys_rows += sum(span[-1] - span[0] + 1 for span in spans)
        return [resolved[expert] for expert, _s, _v in fetch_list]


class OffloadedExpert(nn.Module):
    """Use V4.1's existing Expert forward with only resident projection slots."""

    def __init__(self, expert, plan, prefix):
        super().__init__()
        self.slots = _ExpertSlots(expert, plan, prefix)

    @property
    def quantizes_input(self):
        return self.slots.expert.quantizes_input

    def __call__(
        self, x, indices, weights=None, sorted_indices=False, *, input_quantized=False
    ):
        if self.slots.plan._closed:
            raise RuntimeError("MoE expert store is closed")
        if indices.size == 0:
            return mx.zeros((*indices.shape, 1, x.shape[-1]), dtype=x.dtype)
        if sorted_indices:
            flat_i = indices.reshape(-1, 1)
            flat_x = x.reshape(-1, 1, x.shape[-1])
        else:
            flat_i = indices.reshape(-1, indices.shape[-1])
            flat_x = x.reshape(-1, 1, 1, x.shape[-1])
        flat_w = None if weights is None else weights.reshape(flat_i.shape)
        # Phase is decided once per call from the pre-chunk route count:
        # a single-row prefill TAIL chunk must not score as decode —
        # that would cause a spurious governor visit and overwrite the
        # prev_uniq predictor with a one-expert set.
        phase = "decode" if flat_i.shape[0] == 1 else "prefill"
        outputs = []
        start = 0
        # F1 exact-demand readahead: this call's whole routing is known
        # up front, so each iteration advises the next RA_TOKENS rows'
        # missing experts into the page cache (F_RDADVISE — kernel hints,
        # zero userspace bytes). Covers prefill AND multi-row verify
        # blocks alike; single-row decode calls have nothing to advise.
        ra_advised = set()
        ra_upto = 0
        ra_window = _ra_tokens()
        multi = flat_i.shape[0] > 1
        is_verify = bool(_verify_frozen.get())
        # F2 transient prefill capacity: raise the working cap for the
        # call's duration so one ensure covers cap//top_k tokens instead
        # of one — ~10x fewer serialized fetch rounds at the benchmark's
        # cap=10. Grows rooms on demand; the finally restores the cap
        # (unless the governor retargeted mid-call) and compacts the
        # extra rooms back out, keeping the MRU-end decode residents.
        entry_cap = entry_rooms = raised_cap = None
        if multi and not is_verify:
            target = min(
                _prefill_cap(), self.slots.plan.count_of(self.slots.prefix)
            )
            with self.slots._slots_lock:
                entry_cap = self.slots.book.cap
                entry_rooms = self.slots.book.rooms
                if target > entry_cap:
                    self.slots.book.cap = target
                    raised_cap = target
        if (
            multi
            and _ra_enabled()
            and not is_verify
            and self.slots.backing is not None
        ):
            # Next-layer whole-bank advise (F1b): this call's union
            # coverage is the proxy for the next layer's — at saturated
            # coverage its demand is almost surely the whole bank, and
            # advising now buys a full layer of readahead lead time.
            union = {int(e) for e in flat_i.reshape(-1).tolist()}
            if len(union) >= _ra_next_cov() * max(
                1, self.slots.plan.count_of(self.slots.prefix)
            ):
                self.slots.backing.advise_next_layer(self.slots.layer)
        try:
            while start < flat_i.shape[0]:
                # Bound each chunk by route count without an O(prompt^2)
                # search. Governor-driven ceiling, not the plan's initial
                # capacity — read under the slots lock so it cannot tear
                # against a mid-call resize, and re-read per chunk so a
                # governor shrink re-splits the remaining rows instead of
                # leaving a chunk's routed set wider than the live cap
                # (needed > book.cap aborts ensure mid-prefill).
                with self.slots._slots_lock:
                    # Under verify the scratch rows (F4) widen the
                    # working set: one ensure covers the whole block's
                    # union instead of cap//top_k tokens.
                    step = working_set_step(
                        self.slots.cap
                        + (_verify_scratch() if is_verify else 0),
                        flat_i.shape[-1],
                    )
                if multi:
                    if _ra_enabled():
                        horizon = min(
                            flat_i.shape[0], start + step + ra_window
                        )
                        if horizon > ra_upto:
                            window = flat_i[ra_upto:horizon]
                            todo = {
                                int(e) for e in window.reshape(-1).tolist()
                            } - ra_advised
                            if todo:
                                ra_advised |= todo
                                self.slots.advise_demand(todo)
                            ra_upto = horizon
                    if _chunk_lookahead():
                        # F2 staged lookahead: the NEXT chunk's exact
                        # demand fetches on the staging worker while this
                        # chunk ensures+computes; the next ensure joins
                        # on the staged payloads. Exact — every staged
                        # row is consumed, so the bound is chunk width.
                        nxt = flat_i[start + step : start + 2 * step]
                        if nxt.size:
                            self.slots.stage_predicted(
                                {
                                    int(e)
                                    for e in nxt.reshape(-1).tolist()
                                },
                                limit=step * flat_i.shape[-1],
                            )
                idx = flat_i[start : start + step]
                try:
                    slots = self.slots.ensure(idx, phase=phase)
                except _WorkingSetOverCap:
                    # A shrink landed between the step read and the
                    # ensure: re-read the live cap and re-split the tail
                    # rather than aborting the request. The finer step
                    # always makes progress (needed <= step*top_k <= old
                    # cap); when it cannot shrink further the working
                    # set is unservable.
                    with self.slots._slots_lock:
                        shrunk = working_set_step(
                            self.slots.cap
                            + (_verify_scratch() if is_verify else 0),
                            flat_i.shape[-1],
                        )
                    if shrunk >= step:
                        raise
                    continue
                value = flat_x[start : start + step]
                scores = (
                    None if flat_w is None else flat_w[start : start + step]
                )
                if sorted_indices:
                    slots = slots.reshape(-1)
                    order = mx.argsort(slots)
                    inverse = mx.argsort(order)
                    value, slots = value[order], slots[order]
                    scores = (
                        None
                        if scores is None
                        else scores.reshape(-1)[order]
                    )
                out = self.slots.expert(
                    value,
                    slots,
                    scores,
                    sorted_indices=sorted_indices,
                    input_quantized=input_quantized,
                )
                if sorted_indices:
                    out = out[inverse]
                mx.eval(out)
                outputs.append(out)
                start += step
        finally:
            if raised_cap is not None:
                # Restore the working ceiling — unless the governor
                # retargeted it mid-call (a smaller live cap wins) — and
                # hand the transiently grown rows back: compact keeps the
                # MRU tail (decode residents survive; prefill commits
                # landed at the cold end) and releases empty storage.
                with self.slots._slots_lock:
                    if self.slots.book.cap == raised_cap:
                        self.slots.book.cap = entry_cap
                    keep = min(
                        len(self.slots.book.slot_of), self.slots.book.cap
                    )
                self.slots.compact(keep, rooms=entry_rooms)
        return mx.concatenate(outputs, axis=0).reshape(*indices.shape, 1, x.shape[-1])


def estimate_expert_savings(path, fraction):
    path = Path(path)
    files = [path / "config.json", path / "model.safetensors.index.json"]
    files.extend(path.glob("*.safetensors"))
    signature = tuple(
        (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(files)
    )
    return _estimate_expert_savings(str(path), fraction, signature)


@lru_cache(maxsize=32)
def _estimate_expert_savings(path, fraction, signature):
    from .config import ModelConfig

    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    mapping = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    plan = ExpertOffloadPlan(path, raw, mapping, ModelConfig.from_dict(raw), fraction)
    # Keep the existing residency estimator's 5% nonexpert safety allowance.
    return plan.full_bytes - plan.resident_bytes + plan.draft_bytes
