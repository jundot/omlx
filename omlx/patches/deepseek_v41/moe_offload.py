# SPDX-License-Identifier: Apache-2.0
"""Bounded expert residency for V4.1, preserving its projection arithmetic."""

import json
import math
import struct
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .convert import repack_weight
from .quantization import QuantizedProjection
from .storage import TensorFile, decode_array

_PROJECTIONS = ("w1", "w3", "w2")
_FLOAT_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


class ExpertOffloadPlan:
    """Validate every routed tensor before promising any memory savings."""

    def __init__(self, path, raw, mapping, config, fraction):
        if not 0 < fraction <= 1:
            raise ValueError("MoE resident fraction must be in (0, 1]")
        self.path = Path(path)
        self.mapping = mapping
        self.converted = raw.get("omlx_deepseek_v41")
        self.count = config.n_routed_experts
        self.capacity = min(
            self.count, max(config.n_activated_experts, round(self.count * fraction))
        )
        self.layers = {}
        self.excluded_keys = set()
        self.resident_bytes = 0
        self.full_bytes = 0
        self._headers = {}
        self._readers = {}
        self._closed = False
        self.draft_bytes = 0
        if self.converted is not None:
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
                specs[proj] = self._projection(prefix, proj, shape)
            self.layers[prefix] = specs

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

    def _projection(self, prefix, proj, logical):
        if self.converted is not None:
            name = f"{prefix}.{proj}"
            spec = self.converted.get("quantized_modules", {}).get(name)
            fields = ["weight"]
            if spec:
                fields += ["scales"] + (["biases"] if spec["mode"] == "affine" else [])
            entries = {f: self._entry(f"{name}.{f}") for f in fields}
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
        self.resident_bytes += size * self.capacity // self.count
        return spec

    def _read(self, key, expert=None):
        if self._closed:
            raise RuntimeError("MoE expert store is closed")
        filename = self.mapping[key]
        if filename not in self._readers:
            self._readers[filename] = TensorFile(self.path / filename)
        return self._readers[filename].read(key, rows=expert)

    def fetch(self, prefix, proj, expert):
        if self.converted is not None:
            spec = self.layers[prefix][proj]
            fields = ["weight"]
            if spec:
                fields += ["scales"] + (["biases"] if spec["mode"] == "affine" else [])
            return {
                field: decode_array(*self._read(f"{prefix}.{proj}.{field}", expert))
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

    def close(self):
        self._closed = True
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()


class _ExpertSlots:
    def __init__(self, expert, plan, prefix):
        self.expert, self.plan, self.prefix = expert, plan, prefix
        self.slot_of = {}
        self.free = list(range(plan.capacity))
        self.hits = self.misses = 0
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
        needed = set(indices.reshape(-1).tolist())
        if len(needed) > self.plan.capacity:
            raise ValueError("Expert working set exceeds resident capacity")
        # Protect the entire working set, including hits encountered after misses.
        for expert in needed:
            if expert in self.slot_of:
                self.hits += 1
                self.slot_of[expert] = self.slot_of.pop(expert)
        for expert in needed:
            if expert in self.slot_of:
                continue
            slot = (
                self.free.pop()
                if self.free
                else self.slot_of.pop(next(e for e in self.slot_of if e not in needed))
            )
            for proj in _PROJECTIONS:
                for field, array in self.plan.fetch(self.prefix, proj, expert).items():
                    lin = getattr(self.expert, proj)
                    lin[field][slot] = array
            self.slot_of[expert] = slot
            self.misses += 1
        return mx.array(
            [self.slot_of[e] for e in indices.reshape(-1).tolist()], dtype=mx.int32
        ).reshape(indices.shape)


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
        # Bound each chunk by route count without an O(prompt^2) search.
        step = max(1, self.slots.plan.capacity // flat_i.shape[-1])
        outputs = []
        for start in range(0, flat_i.shape[0], step):
            idx = flat_i[start : start + step]
            slots = self.slots.ensure(idx)
            value = flat_x[start : start + step]
            scores = None if flat_w is None else flat_w[start : start + step]
            if sorted_indices:
                slots = slots.reshape(-1)
                order = mx.argsort(slots)
                inverse = mx.argsort(order)
                value, slots = value[order], slots[order]
                scores = None if scores is None else scores.reshape(-1)[order]
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
