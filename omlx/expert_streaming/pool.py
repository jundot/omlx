# SPDX-License-Identifier: Apache-2.0
"""Fixed resident/cached expert banks for compile-stable MLX execution."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .safetensors import PROJECTIONS, ExpertReader, TensorLocation

_MLX_DTYPES = {
    "U32": mx.uint32,
    "U8": mx.uint8,
    "I8": mx.int8,
    "F32": mx.float32,
    "F16": mx.float16,
    "BF16": mx.bfloat16,
}


@dataclass
class ExpertPoolStats:
    route_lookups: int = 0
    pinned_hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    loads: int = 0
    pinned_loads: int = 0
    cold_loads: int = 0
    expert_major_calls: int = 0
    qmm_calls: int = 0
    speculative_routes: int = 0
    speculative_misses: int = 0
    hotness_decays: int = 0
    warm_start_loads: int = 0
    bank_bind_seconds: float = 0.0
    bank_materialize_seconds: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        total = self.pinned_hits + self.cache_hits + self.cache_misses
        return {
            "route_lookups": self.route_lookups,
            "pinned_hits": self.pinned_hits,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "evictions": self.evictions,
            "loads": self.loads,
            "pinned_loads": self.pinned_loads,
            "cold_loads": self.cold_loads,
            "expert_major_calls": self.expert_major_calls,
            "qmm_calls": self.qmm_calls,
            "speculative_routes": self.speculative_routes,
            "speculative_misses": self.speculative_misses,
            "hotness_decays": self.hotness_decays,
            "warm_start_loads": self.warm_start_loads,
            "bank_bind_seconds": self.bank_bind_seconds,
            "bank_materialize_seconds": self.bank_materialize_seconds,
            "hit_rate": (self.pinned_hits + self.cache_hits) / total if total else 1.0,
        }


class StreamingQuantizedSwitchLinear:
    """Projection-compatible facade used by Qwen target-verify helpers."""

    def __init__(self, owner: StreamingSwitchGLU, projection: str):
        self._owner = owner
        self._projection_name = projection
        metadata = owner.projection_metadata[projection]
        self.group_size = metadata["group_size"]
        self.bits = metadata["bits"]
        self.mode = metadata["mode"]

    @property
    def weight(self) -> mx.array:
        return self._owner._array(self._projection_name, "weight")

    @property
    def scales(self) -> mx.array:
        return self._owner._array(self._projection_name, "scales")

    @property
    def biases(self) -> mx.array | None:
        return self._owner._array(self._projection_name, "biases")

    @property
    def input_dims(self) -> int:
        return self.scales.shape[2] * self.group_size

    @property
    def output_dims(self) -> int:
        return self.weight.shape[1]

    @property
    def num_experts(self) -> int:
        return self._owner.num_experts

    def __call__(self, x, indices, sorted_indices=False):
        del sorted_indices
        return self._owner.project_indices(self._projection_name, x, indices)


class StreamingSwitchGLU(nn.Module):
    """SwitchGLU whose stable bank contains pinned rows plus a hot tier."""

    _omlx_expert_streaming = True

    def __init__(
        self,
        *,
        layer: int,
        num_experts: int,
        top_k: int,
        pinned_experts: tuple[int, ...],
        cache_slots: int,
        locations: dict[tuple[str, str], TensorLocation],
        projection_metadata: dict[str, dict[str, Any]],
        activation: Any,
        reader: ExpertReader,
        cache_policy: str = "route_frequency",
    ):
        super().__init__()
        self.layer = int(layer)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.pinned_experts = tuple(int(value) for value in pinned_experts)
        self._pinned_set = frozenset(self.pinned_experts)
        self.pinned_count = len(self.pinned_experts)
        cold_count = self.num_experts - self.pinned_count
        minimum_working_set = min(self.top_k, cold_count)
        self.cache_slots = min(cold_count, max(int(cache_slots), minimum_working_set))
        self.pool_size = self.pinned_count + self.cache_slots
        self.locations = locations
        self.projection_metadata = projection_metadata
        for projection in PROJECTIONS:
            self.projection_metadata[projection].setdefault(
                "parts",
                tuple(part for candidate, part in locations if candidate == projection),
            )
        self.activation = activation
        self._reader = reader
        if cache_policy not in {"lru", "route_frequency"}:
            raise ValueError(f"Unknown expert cache policy: {cache_policy}")
        self.cache_policy = cache_policy
        self._lock = threading.RLock()
        self._expert_to_slot: dict[int, int] = {
            expert: slot for slot, expert in enumerate(self.pinned_experts)
        }
        mask = np.zeros(self.num_experts, dtype=np.bool_)
        mask[list(self.pinned_experts)] = True
        slot_map = np.zeros(self.num_experts, dtype=np.int32)
        for expert, slot in self._expert_to_slot.items():
            slot_map[expert] = slot
        self._slot_map_np = slot_map
        self._resident_mask_np = mask
        self._slot_map = mx.array(slot_map)
        self._resident_mask = mx.array(mask)
        self._dynamic_lru: OrderedDict[int, int] = OrderedDict()
        self._route_hotness = np.zeros(self.num_experts, dtype=np.uint64)
        self._route_counts = np.zeros(self.num_experts, dtype=np.uint64)
        self._last_used = np.zeros(self.num_experts, dtype=np.uint64)
        self._route_tokens = 0
        self._hotness_decay_interval = 16
        self._next_hotness_decay = self._hotness_decay_interval
        self._access_clock = 0
        self._free_slots = list(range(self.pinned_count, self.pool_size))
        self._last_indices: mx.array | None = None
        self._last_slots: mx.array | None = None
        self._execution_mode = "checked"
        self._speculative_routes: list[tuple[mx.array, mx.array]] = []
        self.stats = ExpertPoolStats()

        for projection in PROJECTIONS:
            for part in self.projection_metadata[projection]["parts"]:
                location = locations[(projection, part)]
                setattr(
                    self,
                    self._array_name(projection, part),
                    mx.zeros(
                        (self.pool_size, *location.shape[1:]),
                        dtype=_MLX_DTYPES[location.dtype],
                    ),
                )

        self.gate_proj = StreamingQuantizedSwitchLinear(self, "gate_proj")
        self.up_proj = StreamingQuantizedSwitchLinear(self, "up_proj")
        self.down_proj = StreamingQuantizedSwitchLinear(self, "down_proj")
        self._load_into_slots(
            self.pinned_experts,
            list(range(self.pinned_count)),
            load_kind="pinned",
        )

    @staticmethod
    def _array_name(projection: str, part: str) -> str:
        return f"_bank_{projection}_{part}"

    def _array(self, projection: str, part: str) -> mx.array | None:
        return getattr(self, self._array_name(projection, part), None)

    @property
    def resident_mask(self) -> mx.array:
        return self._resident_mask

    @property
    def execution_mode(self) -> str:
        return self._execution_mode

    @property
    def cache_full(self) -> bool:
        """Whether every evictable expert slot has been populated."""
        return len(self._dynamic_lru) >= self.cache_slots

    def set_execution_mode(self, mode: str) -> None:
        if mode not in {"checked", "speculative"}:
            raise ValueError(f"Unknown expert streaming execution mode: {mode}")
        with self._lock:
            self._execution_mode = mode
            self._last_indices = None
            self._last_slots = None
            if mode == "speculative":
                self._speculative_routes.clear()

    def take_speculative_routes(self) -> list[tuple[mx.array, mx.array]]:
        with self._lock:
            routes = self._speculative_routes
            self._speculative_routes = []
            return routes

    def _load_into_slots(
        self,
        experts: tuple[int, ...] | list[int],
        slots: list[int],
        *,
        load_kind: str = "cold",
    ) -> None:
        if not experts:
            return
        preload = load_kind == "pinned"
        # Pinned preload stays component-at-a-time to cap transient memory.
        # Dynamic misses are small and benefit from one flat I/O queue.
        components = (
            None
            if preload
            else self._reader.read_many(
                self.locations,
                experts,
                use_file_cache=True,
            )
        )
        for projection in PROJECTIONS:
            for part in self.projection_metadata[projection]["parts"]:
                rows = (
                    self._reader.read_rows(
                        self.locations[(projection, part)],
                        experts,
                        use_file_cache=False,
                    )
                    if components is None
                    else components[(projection, part)]
                )
                target = self._array(projection, part)
                bind_started = time.perf_counter()
                target[slots] = rows
                self.stats.bank_bind_seconds += time.perf_counter() - bind_started
                if preload:
                    # Pinned construction can stage hundreds of rows, so commit
                    # one component at a time and release it immediately.
                    materialize_started = time.perf_counter()
                    mx.eval(target)
                    self.stats.bank_materialize_seconds += (
                        time.perf_counter() - materialize_started
                    )
                # Dynamic updates deliberately remain lazy. The layer's QMM
                # consumes the updated bank immediately, and the next layer's
                # router evaluation provides the required synchronization.
        self.stats.loads += len(experts)
        if load_kind == "pinned":
            self.stats.pinned_loads += len(experts)
        elif load_kind == "warm_start":
            self.stats.warm_start_loads += len(experts)
        else:
            self.stats.cold_loads += len(experts)

    @staticmethod
    def _flatten_indices(indices: mx.array) -> tuple[int, ...]:
        return tuple(int(value) for value in np.asarray(indices.tolist()).reshape(-1))

    def _note_route(self, values: tuple[int, ...]) -> None:
        route_tokens = max(1, (len(values) + self.top_k - 1) // self.top_k)
        self._route_tokens += route_tokens
        while self._route_tokens >= self._next_hotness_decay:
            self._route_hotness >>= 1
            self._next_hotness_decay += self._hotness_decay_interval
            self.stats.hotness_decays += 1
        for expert in values:
            if self._route_hotness[expert] < np.iinfo(np.uint64).max:
                self._route_hotness[expert] += 1
            if self._route_counts[expert] < np.iinfo(np.uint64).max:
                self._route_counts[expert] += 1
            self._access_clock += 1
            self._last_used[expert] = self._access_clock

    def hotlist(self) -> list[tuple[int, int]]:
        """Return non-pinned experts ranked by lifetime router selections."""
        with self._lock:
            experts = [
                expert
                for expert in range(self.num_experts)
                if expert not in self._pinned_set and self._route_counts[expert] > 0
            ]
            experts.sort(
                key=lambda expert: (
                    int(self._route_counts[expert]),
                    int(self._last_used[expert]),
                ),
                reverse=True,
            )
            return [(expert, int(self._route_counts[expert])) for expert in experts]

    def preload_hotlist(self, entries: list[tuple[int, int]]) -> int:
        """Fill evictable slots from a prior run's learned route profile."""
        with self._lock:
            valid: list[tuple[int, int]] = []
            seen: set[int] = set()
            for expert, count in entries:
                expert = int(expert)
                count = int(count)
                if (
                    expert in seen
                    or expert in self._pinned_set
                    or not 0 <= expert < self.num_experts
                    or count <= 0
                ):
                    continue
                seen.add(expert)
                valid.append((expert, count))
            valid.sort(key=lambda entry: entry[1], reverse=True)
            for expert, count in valid:
                self._route_counts[expert] = min(count, int(np.iinfo(np.uint64).max))
            selected = valid[: self.cache_slots]
            missing = [
                expert for expert, _ in selected if expert not in self._expert_to_slot
            ]
            if not missing:
                return 0
            slots = self._allocate_misses(missing)
            self._load_into_slots(missing, slots, load_kind="warm_start")
            counts = dict(selected)
            for expert in missing:
                score = min(counts[expert], int(np.iinfo(np.uint64).max))
                self._route_hotness[expert] = score
                self._access_clock += 1
                self._last_used[expert] = self._access_clock
            arrays = [
                self._array(projection, part)
                for projection in PROJECTIONS
                for part in self.projection_metadata[projection]["parts"]
            ]
            materialize_started = time.perf_counter()
            mx.eval(*arrays, self._slot_map, self._resident_mask)
            self.stats.bank_materialize_seconds += (
                time.perf_counter() - materialize_started
            )
            return len(missing)

    def _eviction_victim(self, protected: set[int]) -> tuple[int, int]:
        candidates = [
            (expert, slot)
            for expert, slot in self._dynamic_lru.items()
            if expert not in protected
        ]
        if not candidates:
            raise RuntimeError(
                f"Layer {self.layer} cannot evict a cache slot without replacing "
                "an expert used by the current route"
            )
        if self.cache_policy == "lru":
            return candidates[0]
        return min(
            candidates,
            key=lambda item: (
                int(self._route_hotness[item[0]]),
                int(self._last_used[item[0]]),
            ),
        )

    def _allocate_misses(
        self, missing: list[int], *, protected: set[int] | None = None
    ) -> list[int]:
        protected = protected or set()
        slots: list[int] = []
        for expert in missing:
            if self._free_slots:
                slot = self._free_slots.pop(0)
            else:
                evicted, slot = self._eviction_victim(protected)
                del self._dynamic_lru[evicted]
                del self._expert_to_slot[evicted]
                self._resident_mask_np[evicted] = False
                self._slot_map_np[evicted] = 0
                self.stats.evictions += 1
            self._expert_to_slot[expert] = slot
            self._resident_mask_np[expert] = True
            self._slot_map_np[expert] = slot
            self._dynamic_lru[expert] = slot
            slots.append(slot)
        self._resident_mask = mx.array(self._resident_mask_np)
        self._slot_map = mx.array(self._slot_map_np)
        return slots

    def _ensure_values(self, values: tuple[int, ...]) -> list[int]:
        unique = list(dict.fromkeys(values))
        dynamic_working_set = [
            expert for expert in unique if expert not in self._pinned_set
        ]
        if len(dynamic_working_set) > self.cache_slots:
            raise RuntimeError(
                f"Layer {self.layer} needs {len(dynamic_working_set)} unpinned "
                f"experts concurrently but has only {self.cache_slots} "
                "streaming slots"
            )
        self._note_route(values)
        missing = [expert for expert in unique if expert not in self._expert_to_slot]
        self.stats.route_lookups += len(values)
        self.stats.pinned_hits += sum(
            1
            for expert in values
            if self._expert_to_slot.get(expert, self.pinned_count) < self.pinned_count
        )
        self.stats.cache_hits += sum(
            1 for expert in values if expert in self._dynamic_lru
        )
        self.stats.cache_misses += sum(1 for expert in values if expert in missing)

        for expert in unique:
            if expert in self._dynamic_lru and expert not in missing:
                self._dynamic_lru.move_to_end(expert)
        if missing:
            slots = self._allocate_misses(missing, protected=set(unique))
            self._load_into_slots(missing, slots)
        return [self._expert_to_slot[expert] for expert in values]

    def ensure(self, indices: mx.array) -> mx.array:
        with self._lock:
            if indices is self._last_indices and self._last_slots is not None:
                return self._last_slots
            if self._execution_mode == "speculative":
                mapped = self._slot_map[indices]
                missing = mx.logical_not(self._resident_mask[indices])
                self._speculative_routes.append((indices, missing))
                self._last_indices = indices
                self._last_slots = mapped
                return mapped
            values = self._flatten_indices(indices)
            slots = self._ensure_values(values)
            mapped = mx.array(slots, dtype=mx.int32).reshape(indices.shape)
            self._last_indices = indices
            self._last_slots = mapped
            return mapped

    def record_speculative_route(self, values: tuple[int, ...]) -> None:
        """Account for a graph-native route after its arrays are evaluated."""
        with self._lock:
            self._note_route(values)
            self.stats.speculative_routes += 1
            self.stats.route_lookups += len(values)
            self.stats.pinned_hits += sum(
                1
                for expert in values
                if self._expert_to_slot.get(expert, self.pinned_count)
                < self.pinned_count
            )
            self.stats.cache_hits += sum(
                1 for expert in values if expert in self._dynamic_lru
            )
            missing = {
                expert for expert in values if expert not in self._expert_to_slot
            }
            misses = sum(1 for expert in values if expert in missing)
            self.stats.cache_misses += misses
            self.stats.speculative_misses += misses
            for expert in dict.fromkeys(values):
                if expert in self._dynamic_lru:
                    self._dynamic_lru.move_to_end(expert)

    def promote(self, values: tuple[int, ...]) -> None:
        """Load cold experts while deferring bank materialization to the retry."""
        with self._lock:
            unique = list(dict.fromkeys(values))
            dynamic_working_set = [
                expert for expert in unique if expert not in self._pinned_set
            ]
            if len(dynamic_working_set) > self.cache_slots:
                raise RuntimeError(
                    f"Layer {self.layer} needs {len(dynamic_working_set)} unpinned "
                    f"experts concurrently but has only {self.cache_slots} "
                    "streaming slots"
                )
            missing = [
                expert for expert in unique if expert not in self._expert_to_slot
            ]
            if missing:
                slots = self._allocate_misses(missing, protected=set(unique))
                self._load_into_slots(missing, slots)
                # The retried QMM immediately consumes the updated banks, so
                # forcing every projection here only materializes the same
                # writes twice. Commit the tiny routing maps now and let the
                # retry evaluate each bank update with its first consumer.
                mx.eval(self._slot_map, self._resident_mask)
            self._last_indices = None
            self._last_slots = None

    def _map_known_values(self, indices: mx.array, values: tuple[int, ...]) -> mx.array:
        slots = self._ensure_values(values)
        mapped = mx.array(slots, dtype=mx.int32).reshape(indices.shape)
        self._last_indices = indices
        self._last_slots = mapped
        return mapped

    def project(self, projection: str, x: mx.array, slot_indices: mx.array) -> mx.array:
        metadata = self.projection_metadata[projection]
        self.stats.qmm_calls += 1
        output = mx.gather_qmm(
            x,
            self._array(projection, "weight"),
            self._array(projection, "scales"),
            self._array(projection, "biases"),
            rhs_indices=slot_indices,
            transpose=True,
            group_size=metadata["group_size"],
            bits=metadata["bits"],
            mode=metadata["mode"],
            sorted_indices=False,
        )
        bias = self._array(projection, "bias")
        if bias is not None:
            output = output + mx.expand_dims(bias[slot_indices], -2)
        return output

    def _forward_projection_expert_major(
        self,
        projection: str,
        x: mx.array,
        indices: mx.array,
        values: tuple[int, ...],
    ) -> mx.array:
        """Materialize one projection in cache-sized expert groups.

        This is a safety path for helpers which call ``gate_proj``, ``up_proj``
        and ``down_proj`` directly instead of invoking the owning SwitchGLU.
        Whole-GLU expert-major execution is preferred because it only streams
        each group once.
        """
        self.stats.expert_major_calls += 1
        unique = list(dict.fromkeys(values))
        pinned = [expert for expert in unique if expert in self._pinned_set]
        dynamic = [expert for expert in unique if expert not in self._pinned_set]
        groups: list[list[int]] = []
        if pinned:
            groups.append(pinned)
        groups.extend(
            dynamic[offset : offset + self.cache_slots]
            for offset in range(0, len(dynamic), self.cache_slots)
        )

        k = indices.shape[-1]
        input_dims = x.shape[-1]
        vectors = x.reshape(-1, input_dims)
        flat_indices = np.asarray(values, dtype=np.int32)
        metadata = self.projection_metadata[projection]
        output_dims = self._array(projection, "weight").shape[1]
        flat_output = mx.zeros((len(values), output_dims), dtype=x.dtype)
        route_specific_input = vectors.shape[0] == len(values)

        for group in groups:
            mask = np.isin(flat_indices, np.asarray(group, dtype=np.int32))
            positions_np = np.nonzero(mask)[0].astype(np.int32)
            if positions_np.size == 0:
                continue
            logical_np = flat_indices[positions_np]
            logical = mx.array(logical_np, dtype=mx.int32)
            slots = self._map_known_values(
                logical, tuple(int(value) for value in logical_np)
            )
            vector_positions = (
                positions_np if route_specific_input else positions_np // k
            )
            selected = vectors[mx.array(vector_positions, dtype=mx.int32)][:, None, :]
            output = mx.gather_qmm(
                selected,
                self._array(projection, "weight"),
                self._array(projection, "scales"),
                self._array(projection, "biases"),
                rhs_indices=slots,
                transpose=True,
                group_size=metadata["group_size"],
                bits=metadata["bits"],
                mode=metadata["mode"],
                sorted_indices=False,
            ).squeeze(-2)
            bias = self._array(projection, "bias")
            if bias is not None:
                output = output + bias[slots]
            # The next group can overwrite the same bank slots.
            mx.eval(output)
            flat_output[mx.array(positions_np, dtype=mx.int32)] = output

        return flat_output.reshape((*indices.shape, 1, output_dims))

    def project_indices(
        self, projection: str, x: mx.array, indices: mx.array
    ) -> mx.array:
        """Project logical expert indices, chunking oversized routed sets."""
        with self._lock:
            if self._execution_mode == "speculative":
                slots = self.ensure(indices)
                return self.project(projection, x, slots)
            values = self._flatten_indices(indices)
            dynamic_working_set = {
                expert for expert in values if expert not in self._pinned_set
            }
            if len(dynamic_working_set) > self.cache_slots:
                return self._forward_projection_expert_major(
                    projection, x, indices, values
                )
            slots = self.ensure(indices)
            # Router-order sorting does not imply slot-order sorting after
            # remapping, so streamed banks always use unsorted slot indices.
            return self.project(projection, x, slots)

    def _forward_resident_set(
        self, x: mx.array, indices: mx.array, values: tuple[int, ...]
    ) -> mx.array:
        slots = self._map_known_values(indices, values)
        expanded = mx.expand_dims(x, (-2, -3))
        up = self.project("up_proj", expanded, slots)
        gate = self.project("gate_proj", expanded, slots)
        output = self.project("down_proj", self.activation(up, gate), slots)
        return output.squeeze(-2)

    def _forward_expert_major(
        self,
        x: mx.array,
        indices: mx.array,
        values: tuple[int, ...],
    ) -> mx.array:
        self.stats.expert_major_calls += 1
        unique = list(dict.fromkeys(values))
        pinned = [
            expert
            for expert in unique
            if expert in self._expert_to_slot and expert not in self._dynamic_lru
        ]
        cold = [expert for expert in unique if expert not in pinned]
        groups: list[list[int]] = []
        if pinned:
            groups.append(pinned)
        groups.extend(
            cold[offset : offset + self.cache_slots]
            for offset in range(0, len(cold), self.cache_slots)
        )

        k = indices.shape[-1]
        hidden = x.shape[-1]
        flat_x = x.reshape(-1, hidden)
        flat_indices = np.asarray(values, dtype=np.int32)
        flat_output = mx.zeros((len(values), hidden), dtype=x.dtype)
        for group in groups:
            mask = np.isin(flat_indices, np.asarray(group, dtype=np.int32))
            positions_np = np.nonzero(mask)[0].astype(np.int32)
            if positions_np.size == 0:
                continue
            logical_np = flat_indices[positions_np]
            logical = mx.array(logical_np, dtype=mx.int32)
            slots = self._map_known_values(
                logical, tuple(int(value) for value in logical_np)
            )
            token_positions = mx.array(positions_np // k, dtype=mx.int32)
            selected = flat_x[token_positions][:, None, :]
            up = self.project("up_proj", selected, slots)
            gate = self.project("gate_proj", selected, slots)
            output = self.project(
                "down_proj", self.activation(up, gate), slots
            ).squeeze(-2)
            # Evaluate before the next group mutates dynamic bank slots.
            mx.eval(output)
            flat_output[mx.array(positions_np, dtype=mx.int32)] = output
        return flat_output.reshape((*indices.shape, hidden))

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        scores: mx.array | None = None,
        weighted_sum: bool = False,
        **_kwargs,
    ) -> mx.array:
        with self._lock:
            if self._execution_mode == "speculative":
                slots = self.ensure(indices)
                expanded = mx.expand_dims(x, (-2, -3))
                up = self.project("up_proj", expanded, slots)
                gate = self.project("gate_proj", expanded, slots)
                output = self.project("down_proj", self.activation(up, gate), slots)
                output = output.squeeze(-2)
                if weighted_sum and scores is not None:
                    output = (output * scores[..., None].astype(output.dtype)).sum(-2)
                return output
            values = self._flatten_indices(indices)
            dynamic_working_set = {
                expert for expert in values if expert not in self._pinned_set
            }
            if len(dynamic_working_set) > self.cache_slots:
                output = self._forward_expert_major(x, indices, values)
            else:
                output = self._forward_resident_set(x, indices, values)
            if weighted_sum and scores is not None:
                output = (output * scores[..., None].astype(output.dtype)).sum(-2)
            return output

    def snapshot(self) -> dict[str, Any]:
        result = self.stats.as_dict()
        result.update(
            {
                "layer": self.layer,
                "pinned_experts": self.pinned_count,
                "cache_slots": self.cache_slots,
                "cache_policy": self.cache_policy,
                "resident_slots": self.pool_size,
                "resident_experts": len(self._expert_to_slot),
                "expert_bytes": sum(
                    location.row_bytes for location in self.locations.values()
                ),
            }
        )
        return result
