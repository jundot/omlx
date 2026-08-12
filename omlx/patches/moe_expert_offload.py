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

from ..scheduler import _sync_and_clear_cache

logger = logging.getLogger(__name__)

_PROJS = ("gate_proj", "up_proj", "down_proj")

_PER_EXPERT_RE = re.compile(r"\.experts\.\d+\.")

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

    def fetch_expert(self, name: str, expert: int) -> mx.array:
        """One expert's slab of a stacked ``[num_experts, ...]`` tensor."""
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

    def _install(self, e: int) -> int:
        if self.free:
            slot = self.free.pop()
        else:
            old_e = next(iter(self.slot_of))  # LRU victim
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

    def ensure(self, idx: mx.array) -> None:
        """Make every expert in ``idx`` resident.

        The ``.tolist()`` is a device->host readback and therefore a sync per
        MoE layer per step. Removing it needs prefetch (resolve layer L+1's
        residency during layer L's compute) — deliberately not in v1.
        """
        if self.warm:  # nothing can miss; skip it
            return
        needed = set(int(e) for e in idx.reshape(-1).tolist())
        for e in needed:
            if e in self.slot_of:
                slot = self.slot_of.pop(e)  # re-insert: LRU order
                self.slot_of[e] = slot
                self.hits += 1
            else:
                self.misses += 1
                self._install(e)
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

    Scans safetensors headers (no load) for expert tensors in either
    supported layout — stacked 3-D ``{gate,up,down}_proj`` tensors under any
    container name (``switch_glu``, ``switch_mlp``, ...) or per-expert
    ``.experts.<n>.`` names — and subtracts the non-resident share. Falls
    back to ``full_size`` on any failure — admission must never get more
    permissive by accident than the plain estimate would be conservative.
    """
    try:
        model_dir = _resolve_model_dir(model_path)
        if model_dir is None:
            return full_size
        expert_bytes = 0
        for shard in sorted(Path(model_dir).glob("*.safetensors")):
            with open(shard, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                stacked = len(spec.get("shape", ())) == 3 and any(
                    f".{p}." in name for p in _PROJS
                )
                if stacked or _PER_EXPERT_RE.search(name):
                    b0, b1 = spec["data_offsets"]
                    expert_bytes += b1 - b0
        if expert_bytes <= 0:
            return full_size
        saved = int(expert_bytes * (1.0 - resident_fraction))
        return max(full_size - saved, full_size - expert_bytes)
    except Exception:
        logger.debug("offload admission estimate failed", exc_info=True)
        return full_size


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
    "moe_offload_stats",
]
