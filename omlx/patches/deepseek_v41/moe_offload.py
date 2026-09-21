# SPDX-License-Identifier: Apache-2.0
"""Bounded expert residency for V4.1, preserving its projection arithmetic.

Non-resident experts are read from the checkpoint's own safetensors shards
with positional ``os.preadv`` calls on a small reader pool of their own. An
expert slab is megabytes of contiguous bytes, the opposite access pattern
from the sparse Engram row gathers that ``storage.TensorFile`` serves through
an ``MADV_RANDOM`` mapping: a faulting gather reads one page per fault, while
a positional read lets the kernel issue large requests. Routing is computed
exactly as shipped; a miss changes when an expert's weights are read, never
which expert runs.
"""

import contextlib
import json
import math
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Lock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .convert import repack_weight, source_quantization_spec
from .quantization import QuantizedProjection
from .residency import (
    checkpoint_signature,
    deepseek_v41_residency_estimate,
    header_with_offset,
)
from .storage import SAFETENSORS_NUMPY_DTYPES, decode_array

_PROJECTIONS = ("w1", "w3", "w2")
_FLOAT_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
# Host bytes one residency update may hold in flight before the serial slot
# writes drain them (about 32 oQ3e experts).
INFLIGHT_BYTES = 512 * 1024 * 1024
# Expert reads get their own pool: the Engram page pool runs long sequential
# page touches that would otherwise queue ahead of a layer's misses.
EXPERT_IO_WORKERS = 24
_EXPERT_IO_POOL = ThreadPoolExecutor(
    max_workers=EXPERT_IO_WORKERS, thread_name_prefix="v41-expert-io"
)

# One positional read: a whole tensor, or one leading-axis row of it.
Slab = namedtuple("Slab", "key expert fd offset nbytes dtype shape")


def _fields(spec):
    """Tensor fields of a projection: packed weights plus quantization metadata."""
    if not spec:
        return ["weight"]
    return ["weight", "scales"] + (["biases"] if spec["mode"] == "affine" else [])


def _to_numpy(slab, raw):
    return np.frombuffer(raw, dtype=SAFETENSORS_NUMPY_DTYPES[slab.dtype]).reshape(
        slab.shape
    )


def _to_array(slab, raw):
    array = _to_numpy(slab, raw)
    if slab.dtype == "BF16":
        return mx.array(array).view(mx.bfloat16)
    if slab.dtype.startswith("F8_"):
        return decode_array(array, slab.dtype)
    return mx.array(array)


class ExpertOffloadPlan:
    """Validate every routed tensor before promising any memory savings."""

    def __init__(self, path, raw, mapping, config, fraction, mtp_resident=False):
        if not 0 < fraction <= 1:
            raise ValueError("MoE resident fraction must be in (0, 1]")
        self.path = Path(path)
        self.mapping = mapping
        self.converted = raw.get("omlx_deepseek_v41")
        self.config = raw
        self.count = config.n_routed_experts
        self.floor = config.n_activated_experts
        self.capacity = min(self.count, max(self.floor, round(self.count * fraction)))
        self.mtp_resident = bool(mtp_resident)
        self.layers = {}
        self.layer_bytes = {}
        self.excluded_keys = set()
        self.full_bytes = 0
        self._headers = {}
        self._data_start = {}
        self._fds = {}
        self._lock = Lock()
        self._closed = False
        self.draft_bytes = 0
        if self.converted is not None:
            for key in mapping:
                if key.startswith("language_model.mtp."):
                    # ``_entry`` registers the key for exclusion as a side
                    # effect. When the draft head is resident we still want the
                    # byte count for reporting, so read the header directly and
                    # leave the key out of ``excluded_keys``.
                    if self.mtp_resident:
                        entry = self._header(self.mapping[key])[key]
                    else:
                        entry = self._entry(key)
                    self.draft_bytes += (
                        entry["data_offsets"][1] - entry["data_offsets"][0]
                    )
        for layer in range(config.n_layers):
            prefix = f"language_model.layers.{layer}.ffn.experts"
            specs = {}
            self.layer_bytes[prefix] = 0
            for proj in _PROJECTIONS:
                shape = (
                    (config.dim, config.moe_inter_dim)
                    if proj == "w2"
                    else (config.moe_inter_dim, config.dim)
                )
                specs[proj] = self._projection(prefix, proj, shape)
            self.layers[prefix] = specs
        self.expert_bytes = (
            self.full_bytes // (self.count * len(self.layers)) if self.layers else 0
        )

    def _header(self, filename):
        if filename not in self._headers:
            full = self.path / filename
            stat = full.stat()
            header, start = header_with_offset(
                str(full), stat.st_size, stat.st_mtime_ns
            )
            self._headers[filename] = header
            self._data_start[filename] = start
        return self._headers[filename]

    def _entry(self, key):
        entry = self._header(self.mapping[key])[key]
        self.excluded_keys.add(key)
        return entry

    def _projection(self, prefix, proj, logical):
        if self.converted is not None:
            name = f"{prefix}.{proj}"
            spec = self.converted.get("quantized_modules", {}).get(name)
            entries = {f: self._entry(f"{name}.{f}") for f in _fields(spec)}
            for field, entry in entries.items():
                shape = (self.count, *logical)
                dtype = entry["dtype"]
                if spec:
                    bits, group = spec["bits"], spec.get("group_size", 32)
                    if logical[-1] % group:
                        raise ValueError(f"Invalid expert group size: {name}")
                    shape = (
                        self.count,
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
            for expert in range(self.count):
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
                elif dtype == "U32":
                    # mlx_lm affine packing: the declared format fixes the
                    # logical width, and the shapes below must agree with it.
                    current = source_quantization_spec(self.config, name)
                    if current is None:
                        raise ValueError(
                            f"Expert is declared dense but stored packed: {name}"
                        )
                    bits, group = current["bits"], current["group_size"]
                    expected = (logical[0], logical[1] * bits // 32)
                    entries = {
                        "weight": entry,
                        "scales": self._entry(name + ".scales"),
                        "biases": self._entry(name + ".biases"),
                    }
                    for field, item in entries.items():
                        want = (
                            expected
                            if field == "weight"
                            else (logical[0], logical[1] // group)
                        )
                        allowed = {"U32"} if field == "weight" else set(_FLOAT_BYTES)
                        if item["dtype"] not in allowed or tuple(item["shape"]) != want:
                            raise ValueError(f"Invalid expert tensor: {name}.{field}")
                    if entries["scales"]["dtype"] != entries["biases"]["dtype"]:
                        raise ValueError(f"Affine metadata dtypes differ: {name}")
                    # Count every field: expert_bytes sizes the INFLIGHT window.
                    size += sum(
                        item["data_offsets"][1] - item["data_offsets"][0]
                        for item in entries.values()
                    )
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
        self.full_bytes += size
        self.layer_bytes[prefix] += size
        return spec

    def resident_bytes_at(self, capacity):
        """Expert bytes resident with ``capacity`` slots per layer."""
        return sum(size * capacity // self.count for size in self.layer_bytes.values())

    @property
    def resident_bytes(self):
        return self.resident_bytes_at(self.capacity)

    def _fd(self, filename):
        with self._lock:
            if self._closed:
                raise RuntimeError("MoE expert store is closed")
            fd = self._fds.get(filename)
            if fd is None:
                fd = self._fds[filename] = os.open(self.path / filename, os.O_RDONLY)
            return fd

    def slab(self, key, expert=None):
        """Read plan for one tensor, or for one leading-axis row of it."""
        filename = self.mapping[key]
        entry = self._header(filename)[key]
        dtype = entry["dtype"]
        if dtype not in SAFETENSORS_NUMPY_DTYPES:
            raise ValueError(f"Unsupported source tensor dtype: {dtype}")
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        nbytes = end - start
        itemsize = np.dtype(SAFETENSORS_NUMPY_DTYPES[dtype]).itemsize
        if nbytes != math.prod(shape) * itemsize:
            raise ValueError(f"Invalid tensor byte length: {key}")
        if expert is not None:
            if not shape or not 0 <= expert < shape[0]:
                raise IndexError(f"Expert {expert} outside {key}")
            nbytes //= shape[0]
            start += expert * nbytes
            shape = shape[1:]
        return Slab(
            key,
            expert,
            self._fd(filename),
            self._data_start[filename] + start,
            nbytes,
            dtype,
            shape,
        )

    @staticmethod
    def read(slab):
        """The slab's bytes. Positional reads only, so any thread may call it."""
        buffer = bytearray(slab.nbytes)
        view = memoryview(buffer)
        done = 0
        while done < slab.nbytes:
            count = os.preadv(slab.fd, [view[done:]], slab.offset + done)
            if count <= 0:
                raise ValueError(f"Truncated tensor data: {slab.key}")
            done += count
        return buffer

    def slabs(self, prefix, proj, expert):
        """Read plans for one expert's projection, in decode order."""
        if self.converted is not None:
            return [
                self.slab(f"{prefix}.{proj}.{field}", expert)
                for field in _fields(self.layers[prefix][proj])
            ]
        name = f"{prefix.removeprefix('language_model.')}.{expert}.{proj}"
        spec = self.layers[prefix][proj]
        if spec and spec["mode"] == "affine":
            # Affine source: mlx_lm stores plural metadata beside the weight.
            return [self.slab(name + "." + field) for field in _fields(spec)]
        slabs = [self.slab(name + ".weight")]
        if name + ".scale" in self.mapping:
            slabs.append(self.slab(name + ".scale"))
        return slabs

    def decode(self, slabs, raws):
        """Projection arrays from the slabs' bytes, keyed by field."""
        if self.converted is not None or slabs[0].dtype == "U32":
            # The packed weight stays uint32: QuantizedProjection drives the
            # quantized matmul, so dequantizing here would defeat the point.
            return {
                slab.key.rsplit(".", 1)[1]: _to_array(slab, raw)
                for slab, raw in zip(slabs, raws)
            }
        scale = _to_numpy(slabs[1], raws[1]) if len(slabs) > 1 else None
        return repack_weight(
            _to_numpy(slabs[0], raws[0]),
            slabs[0].dtype,
            scale,
            slabs[1].dtype if scale is not None else None,
        )[0]

    def fetch(self, prefix, proj, expert):
        slabs = self.slabs(prefix, proj, expert)
        return self.decode(slabs, [self.read(slab) for slab in slabs])

    def close(self):
        """Release the shard descriptors.

        Reads in flight must have drained: the engine closes a model after its
        executor is idle, and ``ensure_ids`` waits for every read it started
        before it returns or raises.
        """
        with self._lock:
            self._closed = True
            fds, self._fds = self._fds, {}
        for fd in fds.values():
            with contextlib.suppress(OSError):
                os.close(fd)


class _ExpertSlots:
    def __init__(self, expert, plan, prefix):
        self.expert, self.plan, self.prefix = expert, plan, prefix
        self.slot_of = {}
        self.free = list(range(plan.capacity))
        self.hits = self.misses = 0
        self.fetched_bytes = 0
        for proj in _PROJECTIONS:
            sample = plan.fetch(prefix, proj, 0)
            values = {
                field: mx.zeros((plan.capacity, *array.shape), dtype=array.dtype)
                for field, array in sample.items()
            }
            spec = plan.layers[prefix][proj]
            if spec:
                setattr(expert, proj, QuantizedProjection(**values, **spec))
            else:
                getattr(expert, proj).weight = values["weight"]
        mx.eval(expert.parameters())
        expert.eval()

    def ensure(self, indices):
        slots = self.ensure_ids(indices.reshape(-1).tolist())
        return mx.array(slots, dtype=mx.int32).reshape(indices.shape)

    def ensure_ids(self, ids):
        """Make every expert in ``ids`` resident; return their slots in order.

        Two passes. The first starts the misses' reads on the reader pool,
        at most ``INFLIGHT_BYTES`` of payload ahead of the installs; the
        second installs them serially in the order the misses were seen, so
        eviction victims, counters and resident bytes match a serial fetch
        exactly. A failed read or decode leaves completed installs intact, and
        every read this call started is drained before it raises.
        """
        needed = list(dict.fromkeys(ids))
        if len(needed) > self.plan.capacity:
            raise ValueError("Expert working set exceeds resident capacity")
        # Protect the entire working set, including hits encountered after misses.
        misses = []
        for expert in needed:
            if expert in self.slot_of:
                self.hits += 1
                self.slot_of[expert] = self.slot_of.pop(expert)
            else:
                misses.append(expert)
        protected = set(needed)
        pending = {}
        window = max(1, INFLIGHT_BYTES // max(1, self.plan.expert_bytes))
        submitted = 0

        def submit(limit):
            nonlocal submitted
            while submitted < min(limit, len(misses)):
                expert = misses[submitted]
                submitted += 1
                pending[expert] = []
                for proj in _PROJECTIONS:
                    slabs = self.plan.slabs(self.prefix, proj, expert)
                    futures = []
                    pending[expert].append((proj, slabs, futures))
                    for slab in slabs:
                        futures.append(_EXPERT_IO_POOL.submit(self.plan.read, slab))

        try:
            submit(window)
            for done, expert in enumerate(misses):
                # Refill before this expert's writes so at most ``window``
                # experts' bytes exist at once, counting the one written here.
                submit(done + window)
                arrays, nbytes = {}, 0
                for proj, slabs, futures in pending[expert]:
                    raws = [future.result() for future in futures]
                    arrays[proj] = self.plan.decode(slabs, raws)
                    nbytes += sum(slab.nbytes for slab in slabs)
                    del raws, futures
                slot = (
                    self.free.pop()
                    if self.free
                    else self.slot_of.pop(
                        next(e for e in self.slot_of if e not in protected)
                    )
                )
                try:
                    for proj, fields in arrays.items():
                        lin = getattr(self.expert, proj)
                        for field, array in fields.items():
                            lin[field][slot] = array
                except BaseException:
                    self.free.append(slot)  # Unmapped, so any partial write is inert.
                    raise
                self.slot_of[expert] = slot
                self.misses += 1
                self.fetched_bytes += nbytes
                # Completed futures own their payloads; release them before refill.
                del pending[expert]
        finally:
            for projections in pending.values():
                for _, _, futures in projections:
                    for future in futures:
                        if not future.cancel():
                            future.exception()
        return [self.slot_of[e] for e in ids]


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
            outputs = self._sorted(x, indices, weights, input_quantized)
        else:
            outputs = self._routed(x, indices, weights, input_quantized)
        return mx.concatenate(outputs, axis=0).reshape(*indices.shape, 1, x.shape[-1])

    def _routed(self, x, indices, weights, input_quantized):
        """Per-token top-k routes: bound each chunk by its route count."""
        flat_i = indices.reshape(-1, indices.shape[-1])
        flat_x = x.reshape(-1, 1, 1, x.shape[-1])
        flat_w = None if weights is None else weights.reshape(flat_i.shape)
        step = max(1, self.slots.plan.capacity // flat_i.shape[-1])
        outputs = []
        for start in range(0, flat_i.shape[0], step):
            idx = flat_i[start : start + step]
            slots = self.slots.ensure(idx)
            out = self.slots.expert(
                flat_x[start : start + step],
                slots,
                None if flat_w is None else flat_w[start : start + step],
                sorted_indices=False,
                input_quantized=input_quantized,
            )
            mx.eval(out)
            outputs.append(out)
        return outputs

    def _sorted(self, x, indices, weights, input_quantized):
        """Routes sorted by expert: chunk on expert boundaries.

        A chunk holds every route of up to ``capacity`` distinct experts, so a
        prefill reads each expert once per layer and runs one kernel per
        chunk instead of one per ``capacity`` routes.
        """
        ids = indices.reshape(-1).tolist()
        rows = x.reshape(-1, 1, x.shape[-1])
        scores = None if weights is None else weights.reshape(-1)
        capacity = self.slots.plan.capacity
        outputs = []
        start = 0
        while start < len(ids):
            end, distinct = start, 0
            while end < len(ids) and distinct < capacity:
                run = end + 1
                while run < len(ids) and ids[run] == ids[end]:
                    run += 1
                end, distinct = run, distinct + 1
            slots = mx.array(self.slots.ensure_ids(ids[start:end]), dtype=mx.int32)
            order = mx.argsort(slots)
            inverse = mx.argsort(order)
            out = self.slots.expert(
                rows[start:end][order],
                slots[order],
                None if scores is None else scores[start:end][order],
                sorted_indices=True,
                input_quantized=input_quantized,
            )[inverse]
            mx.eval(out)
            outputs.append(out)
            start = end
        return outputs


def _plan(path, fraction, mtp_resident=False):
    from .config import ModelConfig

    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    mapping = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    return ExpertOffloadPlan(
        path,
        raw,
        mapping,
        ModelConfig.from_dict(raw),
        fraction,
        mtp_resident=mtp_resident,
    )


def estimate_expert_savings(path, fraction, *, mtp_resident=False):
    return _estimate_expert_savings(
        str(path), fraction, checkpoint_signature(path), bool(mtp_resident)
    )


@lru_cache(maxsize=32)
def _estimate_expert_savings(path, fraction, signature, mtp_resident):
    plan = _plan(path, fraction, mtp_resident)
    # Keep the existing residency estimator's 5% nonexpert safety allowance.
    # The draft head only counts as savings when offload strips it.
    draft = 0 if plan.mtp_resident else plan.draft_bytes
    return plan.full_bytes - plan.resident_bytes + draft


def _admission(plan, capacity, estimate, engram_ssd_offload, file_bytes):
    """``EnginePool._entry_runtime_resident_size`` for one expert capacity.

    With Engram tables: the header residency estimate for the selected
    Engram mode, less 1.05 times the expert savings. Without them the pool
    has no residency estimate and discounts the savings from the discovery
    size (shard file sizes with a 5% allowance) instead.
    """
    draft = 0 if plan.mtp_resident else plan.draft_bytes
    saved = plan.full_bytes - plan.resident_bytes_at(capacity) + draft
    if estimate.supported:
        base = estimate.mmap_bytes if engram_ssd_offload else estimate.resident_bytes
        return max(0, base - int(saved * 1.05))
    return max(0, file_bytes() - saved)


def _file_bytes(path):
    from ...model_discovery import estimate_model_size

    return lambda: estimate_model_size(Path(path))


def admission_bytes(path, fraction, *, engram_ssd_offload=True, mtp_resident=False):
    """The engine pool's admission estimate for expert offload at ``fraction``."""
    return _admission_bytes(
        str(path),
        float(fraction),
        bool(engram_ssd_offload),
        checkpoint_signature(path),
        bool(mtp_resident),
    )


@lru_cache(maxsize=32)
def _admission_bytes(path, fraction, engram_ssd_offload, signature, mtp_resident):
    plan = _plan(path, fraction, mtp_resident)
    estimate = deepseek_v41_residency_estimate(path)
    return _admission(
        plan, plan.capacity, estimate, engram_ssd_offload, _file_bytes(path)
    )


def fit_resident_fraction(
    path, budget_bytes, *, engram_ssd_offload=True, mtp_resident=False
):
    """Largest resident fraction whose admission estimate fits ``budget_bytes``.

    Returns ``None`` when even the routing floor does not fit. The result is
    a whole number of experts per layer expressed as a fraction, so passing
    it back as the setting reproduces the same capacity.
    """
    return _fit_resident_fraction(
        str(path),
        int(budget_bytes),
        bool(engram_ssd_offload),
        checkpoint_signature(path),
        bool(mtp_resident),
    )


@lru_cache(maxsize=32)
def _fit_resident_fraction(
    path, budget_bytes, engram_ssd_offload, signature, mtp_resident
):
    plan = _plan(path, 1.0, mtp_resident)
    estimate = deepseek_v41_residency_estimate(path)
    file_bytes = _file_bytes(path)
    for capacity in range(plan.count, plan.floor - 1, -1):
        if (
            _admission(plan, capacity, estimate, engram_ssd_offload, file_bytes)
            <= budget_bytes
        ):
            return capacity / plan.count
    return None
