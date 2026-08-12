# SPDX-License-Identifier: Apache-2.0
"""MoE expert offloading: stream non-resident experts from the checkpoint.

For Mixture-of-Experts models whose expert tables do not fit in memory, keep
only ``resident_fraction`` of each layer's experts in a contiguous slot tensor
and fetch the rest on demand from the model's own safetensors shards (mmap
slab reads — no converted copy of the checkpoint, no write path). Routing is
computed exactly as shipped; a cache miss changes *when* an expert's weights
are read, never *which* expert runs. Accuracy is therefore preserved by
construction, at a latency cost (measured on a 26B/128-expert model: accuracy
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
import concurrent.futures
import logging
import os
import re
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import (
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)

# oMLX's native-kernel models (DeepSeek V4, GLM-5.2) carry their own
# SwitchGLU / QuantizedSwitchLinear classes under omlx.patches.* — same
# structure, same checkpoint naming (stacked [E, ...] + weight/scales/
# biases), different class identity. Include them (guarded) so the
# type-based walk covers the native-kernel families too; the store's
# per-layer verification still decides what actually wraps.
try:  # pragma: no cover - import is environment-dependent
    from omlx.patches.deepseek_v4.switch_layers import (  # type: ignore
        QuantizedSwitchLinear as Ds4QuantizedSwitchLinear,
        SwitchGLU as Ds4SwitchGLU,
    )
except Exception:  # pragma: no cover
    Ds4QuantizedSwitchLinear = ()
    Ds4SwitchGLU = ()

from ..scheduler import _sync_and_clear_cache

logger = logging.getLogger(__name__)

_PROJS = ("gate_proj", "up_proj", "down_proj")

_PER_EXPERT_PROJ_RE = re.compile(
    r"^(?P<parent>.+)\.experts\.(?P<idx>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<field>weight|scales|biases)$"
)

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


class CheckpointExpertStore:
    """Per-expert slab reads from a model directory's safetensors shards.

    Expert tables are stored stacked with the expert axis leading
    (``[num_experts, ...]``), so one expert is a contiguous byte range in the
    shard. Shards are memory-mapped read-only; a fetch copies out exactly one
    expert's slab. No MLX/Metal calls happen here except the final host-side
    ``mx.array`` construction, so fetches are safe to move off the Metal
    thread later (prefetch).
    """

    def __init__(self, model_path: str | Path):
        self._specs: dict[str, tuple[Path, str, tuple[int, ...], int]] = {}
        self._mm: dict[Path, np.memmap] = {}
        # background slab prefetch (v3): reads overlap GPU compute; the
        # main thread consumes from _prefetch_queue at fetch time.
        self._prefetch_off = os.environ.get("OMLX_MOE_OFFLOAD_DISABLE_READ_PREFETCH") == "1"
        self._prefetch_queue: dict = {}
        self._prefetch_hits = 0
        self._pool = None
        if not self._prefetch_off:
            workers = int(os.environ.get("OMLX_MOE_OFFLOAD_PREFETCH_WORKERS", "2"))
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="omlx-expert-prefetch"
            )
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

    def __bool__(self) -> bool:
        return bool(self._specs)

    def has(self, name: str) -> bool:
        return name in self._specs

    def spec(self, name: str) -> tuple[tuple[int, ...], str]:
        _, dtype, shape, _ = self._specs[name]
        return shape, dtype

    def _read(self, name: str, start_elem: int, n_elems: int,
              out_shape: tuple[int, ...]) -> mx.array:
        shard, dtype, _, offset = self._specs[name]
        np_dtype, mx_view = _DTYPES[dtype]
        itemsize = np.dtype(np_dtype).itemsize
        mm = self._mm.get(shard)
        if mm is None:
            mm = self._mm[shard] = np.memmap(shard, dtype=np.uint8, mode="r")
        start = offset + start_elem * itemsize
        raw = np.array(mm[start : start + n_elems * itemsize])  # one copy
        out = mx.array(raw.view(np_dtype).reshape(out_shape))
        return out.view(mx_view) if mx_view is not None else out

    def _read_cached(self, name: str, expert: int) -> mx.array | None:
        """Consume a background-prefetched slab, if one is pending."""
        if not self._prefetch_queue:
            return None
        key = (name, expert)
        arr = self._prefetch_queue.pop(key, None)
        if arr is not None:
            self._prefetch_hits += 1
        return arr

    def prefetch(self, name: str, expert: int) -> None:
        """Read a slab on a background thread (overlaps GPU compute).

        The mmap read + numpy copy + mx.array host-side creation runs
        off the main thread; the main thread's later ``fetch`` consumes
        the result from the queue (the host->device transfer still
        happens in the main thread, bandwidth-bound).
        """
        if not self._pool or self._prefetch_off:
            return
        key = (name, expert)
        if key in self._prefetch_queue:
            return
        slab = int(np.prod(self._specs[name][2][1:]))
        self._prefetch_queue[key] = None  # placeholder: no duplicate submits
        self._pool.submit(self._prefetch_work, name, expert, slab)

    def _prefetch_work(self, name: str, expert: int, slab: int) -> None:
        try:
            arr = self._read(name, expert * slab, slab, self._specs[name][2][1:])
            self._prefetch_queue[(name, expert)] = arr
        except Exception:
            self._prefetch_queue.pop((name, expert), None)

    def fetch_range(self, name: str, lo: int, hi: int) -> mx.array:
        """Contiguous span ``[lo, hi)`` of a stacked tensor, one read."""
        _, _, shape, _ = self._specs[name]
        slab = int(np.prod(shape[1:]))
        return self._read(name, lo * slab, (hi - lo) * slab, (hi - lo,) + shape[1:])

    def fetch_expert(self, name: str, expert: int) -> mx.array:
        """One expert's slab of a stacked ``[num_experts, ...]`` tensor."""
        cached = self._read_cached(name, expert)
        if cached is not None:
            return cached
        _, _, shape, _ = self._specs[name]
        slab = int(np.prod(shape[1:]))
        return self._read(name, expert * slab, slab, shape[1:])

    def fetch_tensor(self, name: str) -> mx.array:
        """A whole tensor (per-expert checkpoint layouts)."""
        _, _, shape, _ = self._specs[name]
        return self._read(name, 0, int(np.prod(shape)), shape)


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

    def prefetch(self, proj: str, field: str, expert: int) -> None:
        """Background-read one slab (overlaps GPU compute)."""
        self._store.prefetch(self._name(proj, field, 0), expert)

    def fetch_cached(self, proj: str, field: str, expert: int) -> mx.array | None:
        """Consume a background-prefetched slab, if present."""
        return self._store._read_cached(self._name(proj, field, 0), expert)

    def fetch_range(self, proj: str, field: str, lo: int, hi: int) -> mx.array:
        """Contiguous span ``[lo, hi)`` of one field's stacked tensor."""
        if self._per_expert:
            raise NotImplementedError("per-expert layouts have no contiguous span")
        return self._store.fetch_range(self._name(proj, field, 0), lo, hi)

    def fetch(self, proj: str, field: str, expert: int) -> mx.array:
        if self._per_expert:
            return self._store.fetch_tensor(self._name(proj, field, expert))
        return self._store.fetch_expert(self._name(proj, field, 0), expert)


class ExpertCache:
    """Contiguous resident slots over one layer's experts, LRU eviction.

    Holds no reference to the wrapped module's expert tensors — only the
    resident slots and the store view. That is the difference between saving
    memory and adding it: keeping the source tensors referenced alongside the
    slots costs the full expert set *plus* the cache.
    """

    def __init__(self, glu: SwitchGLU, capacity: int, disk: _GLUStoreView):
        self.n_experts = glu.gate_proj["weight"].shape[0]
        self.capacity = min(capacity, self.n_experts)
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
        self.usage = np.zeros(self.n_experts, dtype=np.int64)  # routing heat
        self.pinned: set = set()  # learned hot set, never evicted
        self._prefetching: set = set()  # dedupe for background reads

    def _install(self, e: int) -> int:
        if self.free:
            slot = self.free.pop()
        else:
            # LRU victim, skipping the learned pinned set (pinning is
            # advisory: if every resident expert is pinned, evict LRU).
            victim = next((e for e in self.slot_of if e not in self.pinned), None)
            old_e = next(iter(self.slot_of)) if victim is None else victim
            slot = self.slot_of.pop(old_e)
            self.map[old_e] = -1
        for name in self.projs:
            rw, rs, rb = self.resident[name]
            rw[slot] = self.disk.fetch(name, "weight", e)
            rs[slot] = self.disk.fetch(name, "scales", e)
            if rb is not None and self.disk.has(name, "biases"):
                rb[slot] = self.disk.fetch(name, "biases", e)
        self.slot_of[e] = slot
        self.map[e] = slot
        # once every expert has a slot no eviction can occur, so residency is
        # permanently satisfied and the per-token check is pure overhead
        self.warm = len(self.slot_of) == self.n_experts
        return slot

    def _install_many(self, missing: list, protected: set = frozenset()) -> None:
        """Batch-install missing experts: clustered range reads + one
        indexed write per projection (stacked layouts).

        The transfer volume is unchanged (the same slabs move), but the
        per-op Python/launch overhead collapses: a decode step installs
        ~100 experts (~300 small reads+writes) which measured ~900 ms on
        M5 Pro 48 GB — vs ~73 ms/step with zero installs. Reads are
        clustered on consecutive experts (gap <= 8) so a sparse miss set
        doesn't pull the whole [min, max] span.

        ``protected``: experts needed by the current call that are already
        resident must never be evicted here — ``missing`` is computed once
        up front, so an evicted-but-needed expert would read a stale slot
        (map -> -1). The chunked path guarantees distinct <= capacity, so
        there is always enough evictable headroom.
        """
        if isinstance(protected, frozenset):
            protected = set(protected)
        missing = sorted(set(missing))
        slots = []
        for e in missing:
            if self.free:
                slot = self.free.pop()
            else:
                victim = next(
                    (v for v in self.slot_of if v not in self.pinned and v not in protected),
                    None,
                )
                old_e = next(iter(self.slot_of)) if victim is None else victim
                slot = self.slot_of.pop(old_e)
                self.map[old_e] = -1
            slots.append(slot)
        clusters = []
        cur = [missing[0]]
        for a, b in zip(missing, missing[1:]):
            if b - a <= 8:
                cur.append(b)
            else:
                clusters.append(cur)
                cur = [b]
        clusters.append(cur)
        slot_arr = mx.array(slots)
        use_b = self.resident[self.projs[0]][2] is not None
        # consume background-prefetched slabs first (order = missing order);
        # the rest go through clustered range reads. Parts are per-projection
        # so each resident tensor gets one indexed write.
        parts: dict[str, dict[str, list]] = {
            name: {"w": [], "s": [], "b": []} for name in self.projs
        }
        uncached = []
        for e in missing:
            got = {}
            ready = True
            for name in self.projs:
                w = self.disk.fetch_cached(name, "weight", e)
                s = self.disk.fetch_cached(name, "scales", e)
                b = self.disk.fetch_cached(name, "biases", e) if use_b else None
                if w is None or s is None or (use_b and b is None):
                    ready = False
                    break
                got[name] = (w, s, b)
            if ready:
                for name in self.projs:
                    w, s, b = got[name]
                    parts[name]["w"].append(w)
                    parts[name]["s"].append(s)
                    parts[name]["b"].append(b)
            else:
                uncached.append(e)
        if uncached:
            clusters = []
            cur = [uncached[0]]
            for a, bb in zip(uncached, uncached[1:]):
                if bb - a <= 8:
                    cur.append(bb)
                else:
                    clusters.append(cur)
                    cur = [bb]
            clusters.append(cur)
            for name in self.projs:
                for cl in clusters:
                    parts[name]["w"].append(
                        mx.take(
                            self.disk.fetch_range(name, "weight", cl[0], cl[-1] + 1),
                            mx.array([e - cl[0] for e in cl]), axis=0,
                        )
                    )
                    parts[name]["s"].append(
                        mx.take(
                            self.disk.fetch_range(name, "scales", cl[0], cl[-1] + 1),
                            mx.array([e - cl[0] for e in cl]), axis=0,
                        )
                    )
                    if use_b:
                        parts[name]["b"].append(
                            mx.take(
                                self.disk.fetch_range(name, "biases", cl[0], cl[-1] + 1),
                                mx.array([e - cl[0] for e in cl]), axis=0,
                            )
                        )
        for name in self.projs:
            rw, rs, rb = self.resident[name]
            rw[slot_arr] = mx.concatenate(parts[name]["w"], axis=0)
            rs[slot_arr] = mx.concatenate(parts[name]["s"], axis=0)
            if use_b:
                rb[slot_arr] = mx.concatenate(parts[name]["b"], axis=0)
        for e, s in zip(missing, slots):
            self.slot_of[e] = s
            self.map[e] = s
        self.warm = len(self.slot_of) == self.n_experts

    def ensure(self, idx: mx.array) -> set:
        """Make every expert in ``idx`` resident; returns the host id set.

        The ``.tolist()`` is a device->host readback and therefore a sync per
        MoE layer per step. Removing it needs prefetch (resolve layer L+1's
        residency during layer L's compute) — the prev-step prefetch in
        ``OffloadSwitchGLU.__call__`` is the v1 approximation (host-side
        indices, no readback, installs queued a step early).
        """
        if self.warm:  # nothing can miss; skip it
            return set(int(e) for e in idx.reshape(-1).tolist())
        needed = set(int(e) for e in idx.reshape(-1).tolist())
        for e in needed:
            self.usage[e] += 1
        missing = [e for e in needed if e not in self.slot_of]
        self.hits += len(needed) - len(missing)
        self.misses += len(missing)
        if missing:
            if self.disk._per_expert:
                for e in missing:
                    self._install(e)
            else:
                self._install_many(missing, needed)
        for e in needed:
            if e in self.slot_of:
                slot = self.slot_of.pop(e)  # re-insert: LRU order
                self.slot_of[e] = slot
        # No mx.eval here: installs are already-materialized host arrays, and
        # evaluating every resident tensor on every miss measured 22% slower
        # at identical peak memory. Prefill's transient is bounded by the
        # per-chunk eval in __call__, which is a different mechanism.
        return needed

    def ensure_host(self, needed: set) -> None:
        """Install from a host id set — no device readback.

        The async-prefetch path: prev-step routing ids are already on the
        host, so their installs queue into the graph a step early (evaluated
        with the current step) instead of paying a per-layer sync. Same
        LRU/hits/misses bookkeeping as :meth:`ensure`.
        """
        for e in needed:
            if e in self.slot_of:
                slot = self.slot_of.pop(e)  # re-insert: LRU order
                self.slot_of[e] = slot
                self.hits += 1
            else:
                self.misses += 1
        missing = [e for e in needed if e not in self.slot_of]
        if missing:
            if self.disk._per_expert:
                for e in missing:
                    self._install(e)
            else:
                self._install_many(missing)

    def prefetch_next(self, experts: set) -> None:
        """Submit background reads for next-step candidates (prev-step
        routing, temporal locality). Non-resident experts only; deduped
        with a bounded seen-set."""
        pool = getattr(self.disk._store, "_pool", None)
        if pool is None:
            return
        for e in experts:
            if e in self.slot_of:
                continue
            key = ("next", e)
            if key in self._prefetching:
                continue
            if len(self._prefetching) > 2048:
                self._prefetching.clear()
            self._prefetching.add(key)
            for name in self.projs:
                self.disk.prefetch(name, "weight", e)
                self.disk.prefetch(name, "scales", e)
                if self.resident[name][2] is not None:
                    self.disk.prefetch(name, "biases", e)

    def pin_hot(self, n: int) -> None:
        """Pin the ``n`` most-routed experts (learned hot set).

        Routing heat accumulates in :meth:`ensure`; pinning the top-n makes
        them immune to LRU eviction. On agent workloads the hot set is
        small and stable, so a pinned working set turns most decode fetches
        into hits (Colibri's learned-pin pattern, oMLX-side).
        """
        if n <= 0:
            return
        # argsort (not argpartition): deterministic under usage ties (e.g.
        # experts that were never routed — all-zero usage), so the pinned
        # set is stable and reproducible; 256 experts/layer is small.
        hot = np.argsort(-self.usage)[:n]
        self.pinned.update(int(e) for e in hot)

    def shrink(self, keep_fraction: float = 0.5) -> int:
        """Release resident slots after a prefill burst; returns freed count.

        Prefill touches a wide expert working set. Keeping it all wired on
        a memory-constrained machine evicts the OS file-cache pages the
        store's mmap reads rely on, and decode misses then degrade to raw
        disk reads instead of page-cache hits (measured on M5 Pro 48 GB:
        +3 GiB of wired expert cache -> decode 6.5 -> 2.2 t/s). Rebuilding
        the resident buffers at the hot-subset size frees the wired memory
        back to the OS, so the page cache serves the next misses at RAM
        speed (the ds4-ssd post-prefill bank-shrink pattern).
        """
        if self.warm or self.capacity <= 8:
            return 0
        keep = max(8, min(self.capacity, int(self.capacity * keep_fraction)))
        victims = [e for e in self.slot_of if e not in self.pinned][
            : max(0, len(self.slot_of) - keep)
        ]
        if not victims:
            return 0
        for e in victims:
            del self.slot_of[e]
            self.map[e] = -1
        new_cap = len(self.slot_of)
        for name in self.projs:
            rw, rs, rb = self.resident[name]
            nw = mx.zeros((new_cap,) + rw.shape[1:], dtype=rw.dtype)
            ns = mx.zeros((new_cap,) + rs.shape[1:], dtype=rs.dtype)
            nb = (
                None
                if rb is None
                else mx.zeros((new_cap,) + rb.shape[1:], dtype=rb.dtype)
            )
            for i, e in enumerate(self.slot_of):
                old = self.slot_of[e]
                nw[i] = rw[old]
                ns[i] = rs[old]
                if nb is not None and rb is not None:
                    nb[i] = rb[old]
            self.resident[name] = [nw, ns, nb]
        self.slot_of = {e: i for i, e in enumerate(self.slot_of)}
        self.map = mx.full((self.n_experts,), -1, dtype=mx.int32)
        for e, s in self.slot_of.items():
            self.map[e] = s
        self.capacity = new_cap
        self.free = []
        try:
            mx.clear_cache()
        except AttributeError:
            pass  # older MLX
        return len(victims)

    @property
    def native_kind(self) -> str | None:
        """'affine'/'mxfp4' when the resident cache tensors qualify for the
        native block kernels (mirrors the stock QuantizedSwitchLinear
        conditions: sorted path, group/bits/mode, uint32 packed weights,
        matching scales dtype, custom-kernel symbols present). None ->
        plain gather_qmm fallback, exactly like the stock non-native path.
        """
        try:
            from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
        except Exception:
            return None
        rw, rs, rb = self.resident["gate_proj"]
        # All projections must share one qualifying (group, bits, mode):
        # the block-list kernels dispatch a single format for the whole GLU.
        # Mixed-bit GLUs (e.g. 3-bit gate/up with an 8-bit down) fall back
        # to gather_qmm — same contract as the stock all() on native_kinds.
        if len(set(self.qparams.values())) != 1:
            return None
        g, b, m = next(iter(self.qparams.values()))
        if (
            # The C++ binding admits (128, 2) alongside the stock
            # (64, 2)/(64, 3) — fused_moe.cpp supported_deepseek_affine,
            # verified bit-exact vs gather_qmm on the real 2.4bit-mixed.
            (g, b) in ((64, 2), (64, 3), (128, 2))
            and m == "affine"
            and rb is not None
            and rw.dtype == mx.uint32
            and rs.dtype in (mx.float16, mx.bfloat16)
            and glm_fast.has_symbol("deepseek_affine_gather_qmm_blocks")
        ):
            return "affine"
        if (
            g == 32
            and b == 4
            and m == "mxfp4"
            and rw.dtype == mx.uint32
            and rs.dtype == mx.uint8
            and glm_fast.has_symbol("deepseek_mxfp4_gather_qmm_blocks")
        ):
            return "mxfp4"
        return None

    def qmm_native(
        self,
        name: str,
        x: mx.array,
        block_meta,
        block_count,
        block_variant,
        kind: str,
    ) -> mx.array:
        """Native block-kernel gather over the resident slot tensors.

        ``block_meta`` is built from *slot* ids (see ``_forward``), so the
        kernel gathers from the cache's pre-stacked tensors exactly like the
        stock path gathers from the full expert tensor.
        """
        from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

        rw, rs, rb = self.resident[name]
        if kind == "mxfp4":
            return glm_fast.deepseek_mxfp4_gather_qmm_blocks(
                x, rw, rs, block_meta, block_count, block_variant
            )
        return glm_fast.deepseek_affine_gather_qmm_blocks(
            x, rw, rs, rb, block_meta, block_count, self.group, self.bits,
            block_variant,
        )

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
        self, glu: SwitchGLU, capacity: int, disk: _GLUStoreView,
        shrink_after_prefill: float = 0.0,
    ):
        super().__init__()
        self.cache = ExpertCache(glu, capacity, disk)
        self.activation = glu.activation
        # 0 = keep the full prefill working set wired; (0, 1] = after a
        # chunked (prefill) call, shrink the resident set to that fraction
        # of capacity so the OS page cache can serve decode misses (see
        # ExpertCache.shrink). Default off: behavioral change, tune per
        # machine class.
        self.shrink_after_prefill = shrink_after_prefill
        # v1 async prefetch: prev-step routing ids installed host-side a step
        # early (see __call__/_forward). A/B knob OMLX_MOE_OFFLOAD_DISABLE_PREFETCH.
        self._prefetch = (
            os.environ.get("OMLX_MOE_OFFLOAD_DISABLE_PREFETCH") != "1"
        )
        self._prev: set | None = None

    def _forward(self, x: mx.array, indices: mx.array) -> mx.array:
        c = self.cache
        needed = c.ensure(indices)
        if needed and indices.size <= 64:
            # decode-sized: record for the next step's host-side prefetch
            self._prev = needed
        slots = mx.take(c.map, indices)
        x = mx.expand_dims(x, (-2, -3))
        # Mirror the stock SwitchGLU's sort rule exactly (threshold and all):
        # decode calls are far below it, and forcing the sort there measured
        # slower than it saved.
        do_sort = indices.size >= 64
        inv = None
        if do_sort:
            x, slots, inv = _gather_sort(x, slots)
        # Native block kernels (prefill/sorted path): when the cache's
        # quantized format qualifies, build block_meta from *slot* ids and
        # run the same kernels the resident model would — the fast oQ
        # dequant path instead of the generic gather_qmm fallback.
        kind = c.native_kind if do_sort else None
        if kind is not None and os.environ.get("OMLX_MOE_OFFLOAD_DISABLE_NATIVE") == "1":
            kind = None  # A/B knob: force the gather_qmm fallback
        block_plan = None
        if kind is not None and x.dtype in (mx.float16, mx.bfloat16):
            try:
                from omlx.patches.deepseek_v4.switch_layers import (
                    _block_config,
                    _build_mxfp4_blocks,
                )

                block_bm, block_variant = _block_config(slots.size, kind)
                block_meta, block_count = _build_mxfp4_blocks(
                    slots, c.capacity, block_bm
                )
                block_plan = (block_meta, block_count, block_variant)
            except Exception:
                block_plan = None  # fall back to gather_qmm
        if block_plan is not None:
            block_meta, block_count, block_variant = block_plan
            up = c.qmm_native("up_proj", x, block_meta, block_count, block_variant, kind)
            gate = c.qmm_native(
                "gate_proj", x, block_meta, block_count, block_variant, kind
            )
            out = c.qmm_native(
                "down_proj", self.activation(up, gate),
                block_meta, block_count, block_variant, kind,
            )
        else:
            up = c.qmm("up_proj", x, slots, do_sort)
            gate = c.qmm("gate_proj", x, slots, do_sort)
            out = c.qmm("down_proj", self.activation(up, gate), slots, do_sort)
        if do_sort:
            out = _scatter_unsort(out, inv, indices.shape)
        return out.squeeze(-2)

    def __call__(self, x: mx.array, indices: mx.array, scores=None) -> mx.array:
        # ``scores`` is accepted (and ignored) for native-kernel families
        # whose MoE (e.g. DeepseekV4MoE) performs the weighted expert sum
        # outside the switch module — same contract as their stock
        # SwitchGLU. A single call must have every expert it routes to
        # resident AT ONCE:
        # a long prefill can route to more distinct experts than the cache
        # holds, in which case earlier installs would be evicted before the
        # gather runs and their slots would read garbage. Chunk the token axis
        # until each chunk's working set fits. Decode (working set =
        # batch x top_k) takes the no-sync fast path.
        c = self.cache
        flat_i = indices.reshape(-1, indices.shape[-1])
        n_tok, k = flat_i.shape
        # v1 async prefetch: prev-step routing ids are host-side (recorded in
        # _forward), so install them with no readback — the writes join this
        # step's graph and are resident for the next step. Decode-only (a
        # prefill's wide working set makes prev-step data meaningless).
        if n_tok == 1 and self._prefetch and self._prev is not None:
            c.ensure_host(self._prev)
            c.prefetch_next(self._prev)  # background reads for next step
        if n_tok * k <= c.capacity or n_tok == 1:
            return self._forward(x, indices)
        distinct = len(set(int(e) for e in flat_i.reshape(-1).tolist()))
        if distinct <= c.capacity:
            return self._forward(x, indices)

        flat_x = x.reshape(-1, x.shape[-1])
        # halve until each chunk fits; capacity >= top_k guarantees termination
        size = n_tok
        while size > 1:
            size = max(1, size // 2)
            ok = True
            for s in range(0, n_tok, size):
                if (
                    len(set(int(e) for e in flat_i[s : s + size].reshape(-1).tolist()))
                    > c.capacity
                ):
                    ok = False
                    break
            if ok:
                break
        # Evaluate each chunk before building the next: left lazy, every
        # chunk's intermediates coexist and the prefill transient exceeds the
        # memory the offload exists to save.
        outs = []
        for s in range(0, n_tok, size):
            o = self._forward(flat_x[s : s + size], flat_i[s : s + size])
            mx.eval(o)
            outs.append(o)
        out = mx.concatenate(outs, axis=0)
        if self.shrink_after_prefill > 0:
            # prefill burst is over: release the wide working set back to
            # the OS so decode misses hit the page cache (see shrink())
            self.cache.shrink(self.shrink_after_prefill)
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
    if not stacked.has("gate_proj", "weight"):
        # oMLX's native-kernel families (DeepSeek V4, GLM-5.2) are not
        # wrapped in an outer container, so their module-tree path lacks the
        # checkpoint's leading "model." segment (checkpoints use
        # ``model.layers.N.ffn.switch_mlp.*``). Retry with the prefix.
        prefixed = _GLUStoreView(store, f"model.{path}")
        if prefixed.has("gate_proj", "weight"):
            stacked = prefixed
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
    model, model_path: str | Path, resident_fraction: float = 0.25
) -> int:
    """Replace covered SwitchGLU instances with offloaded ones.

    Returns the number of layers wrapped (0 when disabled via
    ``OMLX_MOE_EXPERT_OFFLOAD=0``, the model has no stock SwitchGLU, or the
    checkpoint does not cover them). Must run before lazy weights are
    materialized for the memory saving to exist.
    """
    if os.environ.get("OMLX_MOE_EXPERT_OFFLOAD", "1") == "0":
        return 0
    model_dir = _resolve_model_dir(model_path)
    if model_dir is None:
        return 0
    store = CheckpointExpertStore(model_dir)
    if not store:
        logger.warning("moe expert offload: no safetensors under %s", model_dir)
        return 0

    wrapped = 0
    total_bytes = resident_bytes = 0
    for parent, key, glu, path in list(_iter_switch_glus(model)):
        view, reason = _resolve_store_view(glu, store, path)
        if view is None:
            logger.info("moe expert offload: skipping %s (%s)", path, reason)
            continue
        n_experts = glu.gate_proj["weight"].shape[0]
        capacity = max(8, min(n_experts, round(n_experts * resident_fraction)))
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
        new = OffloadSwitchGLU(
            glu, capacity, view,
            shrink_after_prefill=float(
                os.environ.get("OMLX_MOE_OFFLOAD_SHRINK_AFTER_PREFILL", "0")
            ),
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

    if wrapped:
        logger.info(
            "moe expert offload: wrapped %d layers at %.0f%% residency "
            "(expert tables: %.2f GB total, %.2f GB resident)",
            wrapped,
            100 * resident_fraction,
            total_bytes / 1e9,
            resident_bytes / 1e9,
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
    layer's savings honor the runtime's minimum-eight capacity floor:
    ``capacity = max(8, min(E, round(E * fraction)))``, so tiny fractions
    do not under-report the resident share. Falls back to ``full_size`` on
    any failure — admission must never get more permissive by accident.
    """
    try:
        model_dir = _resolve_model_dir(model_path)
        if model_dir is None:
            return full_size
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
            capacity = max(8, min(n, round(n * resident_fraction)))
            saved += b["bytes"] * (1.0 - capacity / n)
        for b in per_expert.values():
            per_e = b["per_e"]
            if not per_e or any(not required <= s for s in per_e.values()):
                continue  # any incomplete expert: the wrapper rejects the layer
            n = len(per_e)
            capacity = max(8, min(n, round(n * resident_fraction)))
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
