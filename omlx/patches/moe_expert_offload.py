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

import contextlib
import json
import logging
import os
import re
import struct
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
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
            with contextlib.suppress(Exception):  # interpreter teardown
                os.close(fd)

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


def _shutdown_io_pool() -> None:
    """Drop the pool; the next fetch re-reads the environment (tests)."""
    global _IO_POOL, _IO_BATCH, _IO_CONFIGURED
    with _IO_LOCK:
        pool, _IO_POOL, _IO_BATCH, _IO_CONFIGURED = _IO_POOL, None, 0, False
    if pool is not None:
        pool.shutdown(wait=True)


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

    def _reserve(self, e: int) -> int:
        """Claim a slot for ``e``, evicting the LRU expert if none is free."""
        if self.free:
            slot = self.free.pop()
        else:
            old_e = next(iter(self.slot_of))  # LRU victim
            slot = self.slot_of.pop(old_e)
            self.map[old_e] = -1
        self.slot_of[e] = slot
        self.map[e] = slot
        # once every expert has a slot no eviction can occur, so residency is
        # permanently satisfied and the per-token check is pure overhead
        self.warm = len(self.slot_of) == self.n_experts
        return slot

    def _write(self, slot: int, payload: list) -> None:
        """Copy one expert's fetched bytes into ``slot``."""
        for name, field, plan, raw in payload:
            self.resident[name][field][slot] = CheckpointExpertStore.to_mx(plan, raw)

    def _install(self, e: int) -> int:
        plans = self._plans(e)
        slot = self._reserve(e)
        self._write(
            slot, [(n, f, pl, CheckpointExpertStore.read(pl)) for n, f, pl in plans]
        )
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
        needed = set(int(e) for e in idx.reshape(-1).tolist())
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
                pending[e] = [
                    (name, field, plan, pool.submit(CheckpointExpertStore.read, plan))
                    for name, field, plan in self._plans(e)
                ]

        prefetch(window)
        done = 0
        for e in needed:
            if e in self.slot_of:
                slot = self.slot_of.pop(e)  # re-insert: LRU order
                self.slot_of[e] = slot
                self.hits += 1
                continue
            self.misses += 1
            group = pending.pop(e, None)
            if group is None:
                self._install(e)
                continue
            # Refill before the writes, never after, and only up to the
            # experts already drained: at most ``window`` experts' payloads
            # exist at once, counting the one being written here.
            prefetch(done + window)
            self._write(
                self._reserve(e),
                [(name, field, plan, f.result()) for name, field, plan, f in group],
            )
            done += 1
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

    def __init__(self, glu: SwitchGLU, capacity: int, disk: _GLUStoreView):
        super().__init__()
        self.cache = ExpertCache(glu, capacity, disk)
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

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        # A single call must have every expert it routes to resident AT ONCE:
        # a long prefill can route to more distinct experts than the cache
        # holds, in which case earlier installs would be evicted before the
        # gather runs and their slots would read garbage. Chunk the token axis
        # until each chunk's working set fits. Decode (working set =
        # batch x top_k) takes the no-sync fast path.
        c = self.cache
        flat_i = indices.reshape(-1, indices.shape[-1])
        n_tok, k = flat_i.shape
        if k > c.capacity:
            raise ValueError("Expert cache capacity is smaller than routing top-k")
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
    minimum = _minimum_experts(model_dir)
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
            "moe expert offload: wrapped %d layers at %.1f%% residency "
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
            if kind.startswith("deepseek_v4") or kind in ("glm5_next", "glm_moe_dsa"):
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
