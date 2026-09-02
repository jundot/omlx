# SPDX-License-Identifier: Apache-2.0
"""Fixed resident/cached expert banks for compile-stable MLX execution."""

from __future__ import annotations

import os
import threading
import time
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from .fast_resource import FastExpertLoad
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
    scratch_loads: int = 0
    expert_major_calls: int = 0
    qmm_calls: int = 0
    hotness_decays: int = 0
    warm_start_loads: int = 0
    bank_bind_seconds: float = 0.0
    bank_materialize_seconds: float = 0.0
    scratch_prefetch_wait_seconds: float = 0.0
    scratch_prefetch_requests: int = 0
    scratch_mlx_materialize_seconds: float = 0.0
    sorted_prefill_groups: int = 0
    sorted_prefill_routes: int = 0
    sorted_qmm_calls: int = 0
    route_materialize_calls: int = 0
    route_materialize_first_seconds: float = 0.0
    route_materialize_seconds: float = 0.0
    just_in_time_loads: int = 0
    native_demand_calls: int = 0
    native_demand_callbacks: int = 0
    native_demand_positions: int = 0
    native_demand_destination_fence_skips: int = 0
    native_demand_submit_seconds: float = 0.0
    native_demand_callback_seconds: float = 0.0
    elastic_decode_cache_activations: int = 0
    elastic_decode_cache_demotions: int = 0
    scratch_call_seconds: float = 0.0
    scratch_groups: int = 0
    scratch_load_seconds: float = 0.0
    scratch_gather_seconds: float = 0.0
    scratch_compute_seconds: float = 0.0
    scratch_scatter_seconds: float = 0.0
    scratch_submit_seconds: float = 0.0

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
            "scratch_loads": self.scratch_loads,
            "expert_major_calls": self.expert_major_calls,
            "qmm_calls": self.qmm_calls,
            "hotness_decays": self.hotness_decays,
            "warm_start_loads": self.warm_start_loads,
            "bank_bind_seconds": self.bank_bind_seconds,
            "bank_materialize_seconds": self.bank_materialize_seconds,
            "scratch_prefetch_wait_seconds": self.scratch_prefetch_wait_seconds,
            "scratch_prefetch_requests": self.scratch_prefetch_requests,
            "scratch_mlx_materialize_seconds": self.scratch_mlx_materialize_seconds,
            "sorted_prefill_groups": self.sorted_prefill_groups,
            "sorted_prefill_routes": self.sorted_prefill_routes,
            "sorted_qmm_calls": self.sorted_qmm_calls,
            "route_materialize_calls": self.route_materialize_calls,
            "route_materialize_first_seconds": self.route_materialize_first_seconds,
            "route_materialize_seconds": self.route_materialize_seconds,
            "just_in_time_loads": self.just_in_time_loads,
            "native_demand_calls": self.native_demand_calls,
            "scratch_call_seconds": self.scratch_call_seconds,
            "scratch_groups": self.scratch_groups,
            "scratch_load_seconds": self.scratch_load_seconds,
            "scratch_gather_seconds": self.scratch_gather_seconds,
            "scratch_compute_seconds": self.scratch_compute_seconds,
            "scratch_scatter_seconds": self.scratch_scatter_seconds,
            "scratch_submit_seconds": self.scratch_submit_seconds,
            "native_demand_callbacks": self.native_demand_callbacks,
            "native_demand_positions": self.native_demand_positions,
            "native_demand_destination_fence_skips": (
                self.native_demand_destination_fence_skips
            ),
            "native_demand_submit_seconds": self.native_demand_submit_seconds,
            "native_demand_callback_seconds": self.native_demand_callback_seconds,
            "elastic_decode_cache_activations": (
                self.elastic_decode_cache_activations
            ),
            "elastic_decode_cache_demotions": self.elastic_decode_cache_demotions,
            "hit_rate": (self.pinned_hits + self.cache_hits) / total if total else 1.0,
        }


def _scratch_prefetch_depth() -> int:
    """Number of scratch groups whose I/O may be in flight ahead of compute."""

    raw = os.environ.get("OMLX_SCRATCH_PREFETCH_DEPTH", "3")
    try:
        return int(raw)
    except ValueError:
        return 3


class _NativeDemandCallback:
    """Weak callback retained by the native lazy route resolver."""

    def __init__(self, pool: StreamingSwitchGLU):
        self._pool = weakref.ref(pool)

    def __call__(self, values: list[int]) -> list[int]:
        pool = self._pool()
        if pool is None:
            raise RuntimeError("Expert pool closed before native demand resolved")
        return pool._resolve_native_demand(tuple(int(value) for value in values))


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
        fuse_gate_up: bool = True,
        scratch_slots: int = 0,
        sort_prefill: bool = True,
        native_demand: bool = False,
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
        self.scratch_slots = min(
            max(0, cold_count - self.cache_slots), max(0, int(scratch_slots))
        )
        self.bank_size = self.pool_size + self.scratch_slots
        self._prefill_cache_slots = self.cache_slots
        self._prefill_pool_size = self.pool_size
        self._prefill_scratch_slots = self.scratch_slots
        self._elastic_decode_cache_enabled = False
        self._elastic_decode_cache_active = False
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
        self._sort_prefill = bool(sort_prefill)
        self._native_demand = bool(native_demand)
        self._native_demand_active = self._native_demand
        self._native_demand_callback = (
            _NativeDemandCallback(self) if self._native_demand else None
        )
        if self._native_demand:
            from omlx.custom_kernels.fast_resource_loading import resolve_route_async

            if resolve_route_async is None:
                raise RuntimeError(
                    "Native expert demand requested but the custom kernel is unavailable"
                )
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
        self._last_values: tuple[int, ...] | None = None
        self._execution_mode = "checked"
        self.stats = ExpertPoolStats()

        gate_metadata = self.projection_metadata["gate_proj"]
        up_metadata = self.projection_metadata["up_proj"]
        gate_parts = tuple(gate_metadata["parts"])
        up_parts = tuple(up_metadata["parts"])
        self._fused_gate_up = bool(fuse_gate_up) and (
            gate_parts == up_parts
            and (gate_metadata["group_size"], gate_metadata["bits"], gate_metadata["mode"])
            == (up_metadata["group_size"], up_metadata["bits"], up_metadata["mode"])
            and all(
                locations[("gate_proj", part)].dtype
                == locations[("up_proj", part)].dtype
                and locations[("gate_proj", part)].shape[1:]
                == locations[("up_proj", part)].shape[1:]
                for part in gate_parts
            )
        )
        if self._fused_gate_up:
            self.projection_metadata["gate_up_proj"] = {
                **gate_metadata,
                "parts": gate_parts,
            }

        allocated_projections = (
            ("gate_up_proj", "down_proj")
            if self._fused_gate_up
            else PROJECTIONS
        )
        for projection in allocated_projections:
            for part in self.projection_metadata[projection]["parts"]:
                source_projection = (
                    "gate_proj" if projection == "gate_up_proj" else projection
                )
                location = locations[(source_projection, part)]
                shape = location.shape[1:]
                if projection == "gate_up_proj":
                    shape = (shape[0] * 2, *shape[1:])
                setattr(
                    self,
                    self._array_name(projection, part),
                    mx.zeros(
                        (self.bank_size, *shape),
                        dtype=_MLX_DTYPES[location.dtype],
                    ),
                )

        self.gate_proj = StreamingQuantizedSwitchLinear(self, "gate_proj")
        self.up_proj = StreamingQuantizedSwitchLinear(self, "up_proj")
        if self._fused_gate_up:
            self.gate_up_proj = StreamingQuantizedSwitchLinear(
                self, "gate_up_proj"
            )
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
        if self._fused_gate_up and projection in {"gate_proj", "up_proj"}:
            fused = getattr(self, self._array_name("gate_up_proj", part), None)
            if fused is None:
                return None
            midpoint = fused.shape[1] // 2
            return (
                fused[:, :midpoint]
                if projection == "gate_proj"
                else fused[:, midpoint:]
            )
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

    @property
    def all_experts_resident(self) -> bool:
        return len(self._expert_to_slot) == self.num_experts

    def _publish_route_maps(self) -> None:
        """Push the CPU route maps to the arrays the remap kernel binds.

        With native demand the maps are overwritten in place so a decode
        graph built one step ahead (mlx-lm pipelines with ``async_eval``)
        still reads current residency when its remap kernel executes.
        Re-allocating them would hand that graph a stale snapshot, and a
        stale "resident" verdict can name a slot since refilled with another
        expert. Without the native bridge a fresh array is equivalent, since
        every route is then resolved synchronously on the CPU.
        """

        if self._native_demand:
            from omlx.custom_kernels.fast_resource_loading import publish_route_maps

            if publish_route_maps is not None:
                publish_route_maps(
                    self._slot_map,
                    self._resident_mask,
                    self._slot_map_np.tobytes(),
                    self._resident_mask_np.tobytes(),
                )
                return
        self._resident_mask = mx.array(self._resident_mask_np)
        self._slot_map = mx.array(self._slot_map_np)

    def set_execution_mode(self, mode: str) -> None:
        if mode not in {"checked", "resident"}:
            raise ValueError(f"Unknown expert streaming execution mode: {mode}")
        with self._lock:
            self._execution_mode = mode
            self._last_indices = None
            self._last_slots = None
            self._last_values = None

    def set_native_demand_active(self, active: bool) -> None:
        with self._lock:
            self._native_demand_active = self._native_demand and bool(active)

    def enable_elastic_decode_cache(self, enabled: bool = True) -> None:
        with self._lock:
            if self._elastic_decode_cache_active and not enabled:
                self._set_elastic_decode_cache_active(False)
            self._elastic_decode_cache_enabled = bool(enabled)

    def set_elastic_decode_cache_active(self, active: bool) -> bool:
        with self._lock:
            return self._set_elastic_decode_cache_active(active)

    def _set_elastic_decode_cache_active(self, active: bool) -> bool:
        active = bool(active) and self._elastic_decode_cache_enabled
        if active == self._elastic_decode_cache_active:
            return True
        if self._prefill_scratch_slots == 0:
            return True
        elastic_slots = set(range(self._prefill_pool_size, self.bank_size))
        if active:
            self._free_slots.extend(elastic_slots)
            self._free_slots = sorted(set(self._free_slots))
            self.cache_slots = self._prefill_cache_slots + self._prefill_scratch_slots
            self.pool_size = self.bank_size
            self.scratch_slots = 0
            self._elastic_decode_cache_active = True
            self.stats.elastic_decode_cache_activations += 1
        else:
            demoted = [
                expert
                for expert, slot in self._dynamic_lru.items()
                if slot in elastic_slots
            ]
            for expert in demoted:
                del self._dynamic_lru[expert]
                del self._expert_to_slot[expert]
                self._resident_mask_np[expert] = False
                self._slot_map_np[expert] = 0
            self._free_slots = [
                slot for slot in self._free_slots if slot not in elastic_slots
            ]
            self.cache_slots = self._prefill_cache_slots
            self.pool_size = self._prefill_pool_size
            self.scratch_slots = self._prefill_scratch_slots
            self._elastic_decode_cache_active = False
            self.stats.elastic_decode_cache_demotions += 1
            self._publish_route_maps()
        self._last_indices = None
        self._last_slots = None
        self._last_values = None
        return True

    def fast_resource_targets(
        self,
    ) -> dict[tuple[str, str], tuple[mx.array, int]]:
        """Return stable MLX bank destinations for Fast Resource Loading."""

        targets: dict[tuple[str, str], tuple[mx.array, int]] = {}
        for projection in PROJECTIONS:
            for part in self.projection_metadata[projection]["parts"]:
                target_projection = (
                    "gate_up_proj"
                    if self._fused_gate_up
                    and projection in {"gate_proj", "up_proj"}
                    else projection
                )
                target = self._array(target_projection, part)
                inner_offset = 0
                if target_projection == "gate_up_proj" and projection == "up_proj":
                    target_row_bytes = int(target.nbytes) // int(target.shape[0])
                    inner_offset = target_row_bytes // 2
                targets[(projection, part)] = (target, inner_offset)
        return targets

    def _load_into_slots(
        self,
        experts: tuple[int, ...] | list[int],
        slots: list[int],
        *,
        load_kind: str = "cold",
        preloaded_components: dict[tuple[str, str], mx.array]
        | FastExpertLoad
        | None = None,
        destinations_idle: bool = False,
    ) -> None:
        if not experts:
            return
        preload = load_kind == "pinned"
        # Pinned preload stays component-at-a-time to cap transient memory.
        # Dynamic misses are small and benefit from one flat I/O queue.
        components = preloaded_components
        use_fast_resource_loading = isinstance(components, FastExpertLoad) or (
            components is None
            and self._reader.fast_resource_loading
            and self._reader.fast_resource_compatible(self.locations)
            and not preload
            and (
                load_kind == "scratch"
                or self._reader.fast_resource_loading_scope == "all"
            )
        )
        if use_fast_resource_loading:
            targets = self.fast_resource_targets()
            fast_load = (
                components
                if isinstance(components, FastExpertLoad)
                else (
                    self._reader.begin_fast_direct_many(
                        self.locations, experts, slots, targets
                    )
                    if (
                        load_kind == "cold"
                        and destinations_idle
                        and self._reader.fast_direct_exact
                        and self._reader.fast_direct_compatible(self.locations)
                    )
                    else self._reader.begin_fast_many(self.locations, experts)
                )
            )
            fast_stats = self._reader.finish_fast_many(
                fast_load,
                slots,
                targets,
                destinations_idle=destinations_idle,
            )
            self.stats.native_demand_destination_fence_skips += int(
                destinations_idle
            )
            if load_kind == "scratch":
                self.stats.scratch_prefetch_wait_seconds += float(
                    fast_stats["io_wait_seconds"]
                )
            self.stats.bank_bind_seconds += float(fast_stats["copy_seconds"])
            self.stats.loads += len(experts)
            if load_kind == "cold":
                self.stats.just_in_time_loads += len(experts)
            if load_kind == "warm_start":
                self.stats.warm_start_loads += len(experts)
            else:
                self.stats.cold_loads += len(experts)
                if load_kind == "scratch":
                    self.stats.scratch_loads += len(experts)
            return
        if components is None and not preload:
            components = self._reader.read_many(
                self.locations,
                experts,
                use_file_cache=True,
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
                target_projection = (
                    "gate_up_proj"
                    if self._fused_gate_up
                    and projection in {"gate_proj", "up_proj"}
                    else projection
                )
                target = self._array(target_projection, part)
                bind_started = time.perf_counter()
                if target_projection == "gate_up_proj":
                    midpoint = target.shape[1] // 2
                    output_slice = (
                        slice(0, midpoint)
                        if projection == "gate_proj"
                        else slice(midpoint, None)
                    )
                    target[slots, output_slice] = rows
                else:
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
            if load_kind == "cold":
                self.stats.just_in_time_loads += len(experts)
            if load_kind == "scratch":
                self.stats.scratch_loads += len(experts)

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

    def preload_hotlist(self, entries: list[tuple[int, int]]) -> tuple[int, int]:
        """Populate every configured hot slot, prioritizing learned experts."""
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
            selected_experts = {expert for expert, _ in selected}
            if len(selected) < self.cache_slots:
                selected.extend(
                    (expert, 0)
                    for expert in range(self.num_experts)
                    if expert not in self._pinned_set and expert not in selected_experts
                )
                selected = selected[: self.cache_slots]
            missing = [
                expert for expert, _ in selected if expert not in self._expert_to_slot
            ]
            if not missing:
                return 0, 0
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
            learned = sum(1 for expert in missing if counts[expert] > 0)
            return learned, len(missing) - learned

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
        self._publish_route_maps()
        return slots

    def _ensure_values(
        self,
        values: tuple[int, ...],
        *,
        destinations_idle: bool = False,
    ) -> list[int]:
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
            self._load_into_slots(
                missing,
                slots,
                destinations_idle=destinations_idle,
            )
        return [self._expert_to_slot[expert] for expert in values]

    def _resolve_native_demand(self, values: tuple[int, ...]) -> list[int]:
        """Resolve one GPU-produced exact route after prior consumers finish."""

        started = time.perf_counter()
        with self._lock:
            slots = self._ensure_values(values, destinations_idle=True)
            self.stats.native_demand_callbacks += 1
            self.stats.native_demand_positions += len(values)
            self.stats.native_demand_callback_seconds += time.perf_counter() - started
            return slots

    def _ensure_native(self, indices: mx.array) -> mx.array:
        from omlx.custom_kernels.fast_resource_loading import (
            check_async_route_error,
            resolve_route_async,
        )

        if resolve_route_async is None or self._native_demand_callback is None:
            raise RuntimeError("Native expert demand resolver is unavailable")
        # A failed callback fills the previous route with -1 slots and parks
        # the error in the extension. Raise it here, on the inference thread,
        # before another route is submitted, instead of dispatching gather_qmm
        # with out-of-range expert indices and returning garbage tokens.
        if check_async_route_error is not None:
            check_async_route_error()
        entry, mapped = resolve_route_async(
            indices,
            self._slot_map,
            self._resident_mask,
            self._native_demand_callback,
        )
        submit_started = time.perf_counter()
        mx.async_eval(entry)
        self.stats.native_demand_submit_seconds += time.perf_counter() - submit_started
        self.stats.native_demand_calls += 1
        self._last_indices = indices
        self._last_slots = mapped
        self._last_values = None
        return mapped

    def ensure(self, indices: mx.array) -> mx.array:
        with self._lock:
            if indices is self._last_indices and self._last_slots is not None:
                return self._last_slots
            if self._execution_mode == "resident":
                mapped = self._slot_map[indices]
                self._last_indices = indices
                self._last_slots = mapped
                self._last_values = None
                return mapped
            if self._native_demand_active:
                return self._ensure_native(indices)
            materialize_started = time.perf_counter()
            values = self._flatten_indices(indices)
            materialize_seconds = time.perf_counter() - materialize_started
            self.stats.route_materialize_calls += 1
            self.stats.route_materialize_seconds += materialize_seconds
            if self.stats.route_materialize_calls == 1:
                self.stats.route_materialize_first_seconds = materialize_seconds
            slots = self._ensure_values(values)
            mapped = mx.array(slots, dtype=mx.int32).reshape(indices.shape)
            self._last_indices = indices
            self._last_slots = mapped
            self._last_values = values
            return mapped

    def _map_known_values(self, indices: mx.array, values: tuple[int, ...]) -> mx.array:
        slots = self._ensure_values(values)
        mapped = mx.array(slots, dtype=mx.int32).reshape(indices.shape)
        self._last_indices = indices
        self._last_slots = mapped
        self._last_values = values
        return mapped

    def project(
        self,
        projection: str,
        x: mx.array,
        slot_indices: mx.array,
        *,
        sorted_indices: bool = False,
    ) -> mx.array:
        metadata = self.projection_metadata[projection]
        self.stats.qmm_calls += 1
        self.stats.sorted_qmm_calls += int(sorted_indices)
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
            sorted_indices=sorted_indices,
        )
        bias = self._array(projection, "bias")
        if bias is not None:
            output = output + mx.expand_dims(bias[slot_indices], -2)
        return output

    def _project_gate_up(
        self,
        x: mx.array,
        slot_indices: mx.array,
        *,
        sorted_indices: bool = False,
    ) -> tuple[mx.array, mx.array]:
        if not self._fused_gate_up:
            return (
                self.project(
                    "gate_proj", x, slot_indices, sorted_indices=sorted_indices
                ),
                self.project(
                    "up_proj", x, slot_indices, sorted_indices=sorted_indices
                ),
            )
        gate_up = self.project(
            "gate_up_proj", x, slot_indices, sorted_indices=sorted_indices
        )
        return tuple(mx.split(gate_up, 2, axis=-1))

    def _uses_f16_affine_moe(self, x: mx.array) -> bool:
        """Match the native fast path for BF16 state with FP16 affine metadata."""

        if x.dtype != mx.bfloat16:
            return False
        for projection in PROJECTIONS:
            metadata = self.projection_metadata[projection]
            if metadata["mode"] != "affine":
                return False
            weight = self._array(projection, "weight")
            scales = self._array(projection, "scales")
            biases = self._array(projection, "biases")
            bias = self._array(projection, "bias")
            if (
                weight is None
                or weight.dtype != mx.uint32
                or scales is None
                or scales.dtype != mx.float16
                or biases is None
                or biases.dtype != mx.float16
                or bias is not None
            ):
                return False
        return True

    def _sort_group_for_qmm(
        self, selected: mx.array, slots: mx.array
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        """Sort route rows by physical bank slot for large gather QMMs."""
        if not self._sort_prefill or slots.size < 64:
            return selected, slots, None
        order = mx.argsort(slots)
        self.stats.sorted_prefill_groups += 1
        self.stats.sorted_prefill_routes += int(slots.size)
        return selected[order], slots[order], mx.argsort(order)

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
            selected, slots, inv_order = self._sort_group_for_qmm(selected, slots)
            output = self.project(
                projection,
                selected,
                slots,
                sorted_indices=inv_order is not None,
            ).squeeze(-2)
            if inv_order is not None:
                output = output[inv_order]
            # The next group can overwrite the same bank slots.
            mx.eval(output)
            flat_output[mx.array(positions_np, dtype=mx.int32)] = output

        return flat_output.reshape((*indices.shape, 1, output_dims))

    def project_indices(
        self, projection: str, x: mx.array, indices: mx.array
    ) -> mx.array:
        """Project logical expert indices, chunking oversized routed sets."""
        with self._lock:
            if self._execution_mode == "resident":
                slots = self.ensure(indices)
                return self.project(projection, x, slots)
            if indices is self._last_indices and self._last_slots is not None:
                return self.project(projection, x, self._last_slots)
            if self._native_demand_active:
                slots = self._ensure_native(indices)
                return self.project(projection, x, slots)
            values = self._flatten_indices(indices)
            dynamic_working_set = {
                expert for expert in values if expert not in self._pinned_set
            }
            if len(dynamic_working_set) > self.cache_slots:
                return self._forward_projection_expert_major(
                    projection, x, indices, values
                )
            slots = self._map_known_values(indices, values)
            # Router-order sorting does not imply slot-order sorting after
            # remapping, so streamed banks always use unsorted slot indices.
            return self.project(projection, x, slots)

    def _forward_resident_set(
        self, x: mx.array, indices: mx.array, values: tuple[int, ...]
    ) -> mx.array:
        slots = self._map_known_values(indices, values)
        expanded = mx.expand_dims(x, (-2, -3))
        inv_order = None
        if self._sort_prefill and slots.size >= 64:
            expanded, slots, inv_order = _gather_sort(expanded, slots)
            self.stats.sorted_prefill_groups += 1
            self.stats.sorted_prefill_routes += int(slots.size)
        sorted_indices = inv_order is not None
        gate, up = self._project_gate_up(
            expanded, slots, sorted_indices=sorted_indices
        )
        output = self.project(
            "down_proj",
            self.activation(up, gate),
            slots,
            sorted_indices=sorted_indices,
        )
        if inv_order is not None:
            output = _scatter_unsort(output, inv_order, indices.shape)
        return output.squeeze(-2)

    def _forward_expert_major(
        self,
        x: mx.array,
        indices: mx.array,
        values: tuple[int, ...],
    ) -> mx.array:
        if self.scratch_slots:
            return self._forward_expert_major_scratch(x, indices, values)
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
            selected, slots, inv_order = self._sort_group_for_qmm(selected, slots)
            sorted_indices = inv_order is not None
            gate, up = self._project_gate_up(
                selected, slots, sorted_indices=sorted_indices
            )
            output = self.project(
                "down_proj",
                self.activation(up, gate),
                slots,
                sorted_indices=sorted_indices,
            ).squeeze(-2)
            if inv_order is not None:
                output = output[inv_order]
            # Evaluate before the next group mutates dynamic bank slots.
            mx.eval(output)
            flat_output[mx.array(positions_np, dtype=mx.int32)] = output
        return flat_output.reshape((*indices.shape, hidden))

    def _forward_expert_major_scratch(
        self,
        x: mx.array,
        indices: mx.array,
        values: tuple[int, ...],
    ) -> mx.array:
        """Execute one-shot cold experts without replacing hot-cache rows."""

        self.stats.expert_major_calls += 1
        call_started = time.perf_counter()
        unique = list(dict.fromkeys(values))
        resident = [expert for expert in unique if expert in self._expert_to_slot]
        cold = [expert for expert in unique if expert not in self._expert_to_slot]

        self._note_route(values)
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
        cold_set = set(cold)
        self.stats.cache_misses += sum(1 for expert in values if expert in cold_set)
        for expert in dict.fromkeys(values):
            if expert in self._dynamic_lru:
                self._dynamic_lru.move_to_end(expert)

        groups: list[tuple[list[int], bool]] = []
        if resident:
            groups.append((resident, False))
        groups.extend(
            (cold[offset : offset + self.scratch_slots], True)
            for offset in range(0, len(cold), self.scratch_slots)
        )

        k = indices.shape[-1]
        hidden = x.shape[-1]
        flat_x = x.reshape(-1, hidden)
        flat_indices = np.asarray(values, dtype=np.int32)
        flat_output = mx.zeros((len(values), hidden), dtype=x.dtype)
        scratch_base = self.pool_size
        cold_groups = [group for group, is_scratch in groups if is_scratch]
        # Keep several scratch groups of I/O in flight.  Reads land in the
        # loader's staging buffers, not the scratch slots, so the depth is
        # bounded by staging memory rather than by the scratch bank, and a
        # depth of one left the SSD idle during every group's compute.
        prefetch_depth = max(1, _scratch_prefetch_depth())
        prefetch_queue: deque = deque()
        prefetched_index = 0
        submit_started = time.perf_counter()
        while prefetched_index < len(cold_groups) and len(prefetch_queue) < prefetch_depth:
            prefetch_queue.append(
                self._reader.read_many_async(
                    self.locations,
                    cold_groups[prefetched_index],
                    use_file_cache=True,
                )
            )
            prefetched_index += 1
            self.stats.scratch_prefetch_requests += 1
        self.stats.scratch_submit_seconds += time.perf_counter() - submit_started

        for group, is_scratch in groups:
            mask = np.isin(flat_indices, np.asarray(group, dtype=np.int32))
            positions_np = np.nonzero(mask)[0].astype(np.int32)
            if positions_np.size == 0:
                continue
            logical_np = flat_indices[positions_np]
            if is_scratch:
                wait_started = time.perf_counter()
                cpu_components = prefetch_queue.popleft().result()
                self.stats.scratch_prefetch_wait_seconds += (
                    time.perf_counter() - wait_started
                )
                if isinstance(cpu_components, FastExpertLoad):
                    components = cpu_components
                else:
                    materialize_started = time.perf_counter()
                    components = self._reader.materialize_many(
                        self.locations, cpu_components
                    )
                    self.stats.scratch_mlx_materialize_seconds += (
                        time.perf_counter() - materialize_started
                    )
                if prefetched_index < len(cold_groups):
                    submit_started = time.perf_counter()
                    prefetch_queue.append(
                        self._reader.read_many_async(
                            self.locations,
                            cold_groups[prefetched_index],
                            use_file_cache=True,
                        )
                    )
                    prefetched_index += 1
                    self.stats.scratch_prefetch_requests += 1
                    self.stats.scratch_submit_seconds += (
                        time.perf_counter() - submit_started
                    )
                group_slots = list(range(scratch_base, scratch_base + len(group)))
                load_started = time.perf_counter()
                self._load_into_slots(
                    group,
                    group_slots,
                    load_kind="scratch",
                    preloaded_components=components,
                )
                self.stats.scratch_load_seconds += time.perf_counter() - load_started
                scratch_map = dict(zip(group, group_slots, strict=True))
                slot_values = [scratch_map[int(expert)] for expert in logical_np]
            else:
                slot_values = [self._expert_to_slot[int(expert)] for expert in logical_np]
            gather_started = time.perf_counter()
            slots = mx.array(slot_values, dtype=mx.int32)
            token_positions = mx.array(positions_np // k, dtype=mx.int32)
            selected = flat_x[token_positions][:, None, :]
            selected, slots, inv_order = self._sort_group_for_qmm(selected, slots)
            sorted_indices = inv_order is not None
            gate, up = self._project_gate_up(
                selected, slots, sorted_indices=sorted_indices
            )
            output = self.project(
                "down_proj",
                self.activation(up, gate),
                slots,
                sorted_indices=sorted_indices,
            ).squeeze(-2)
            if inv_order is not None:
                output = output[inv_order]
            compute_started = time.perf_counter()
            self.stats.scratch_gather_seconds += compute_started - gather_started
            mx.eval(output)
            scatter_started = time.perf_counter()
            self.stats.scratch_compute_seconds += scatter_started - compute_started
            flat_output[mx.array(positions_np, dtype=mx.int32)] = output
            self.stats.scratch_scatter_seconds += time.perf_counter() - scatter_started
            self.stats.scratch_groups += 1

        self.stats.scratch_call_seconds += time.perf_counter() - call_started
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
            original_dtype = x.dtype
            use_f16_moe = self._uses_f16_affine_moe(x)
            if use_f16_moe:
                x = x.astype(mx.float16)

            def finish(output: mx.array) -> mx.array:
                if use_f16_moe:
                    output = output.astype(original_dtype)
                if weighted_sum and scores is not None:
                    output = (
                        output * scores[..., None].astype(output.dtype)
                    ).sum(-2)
                return output

            if self._execution_mode == "resident":
                slots = self.ensure(indices)
                expanded = mx.expand_dims(x, (-2, -3))
                gate, up = self._project_gate_up(expanded, slots)
                output = self.project("down_proj", self.activation(up, gate), slots)
                return finish(output.squeeze(-2))
            if self._native_demand_active:
                slots = self._ensure_native(indices)
                expanded = mx.expand_dims(x, (-2, -3))
                gate, up = self._project_gate_up(expanded, slots)
                output = self.project("down_proj", self.activation(up, gate), slots)
                return finish(output.squeeze(-2))
            values = self._flatten_indices(indices)
            dynamic_working_set = {
                expert for expert in values if expert not in self._pinned_set
            }
            if len(dynamic_working_set) > self.cache_slots:
                output = self._forward_expert_major(x, indices, values)
            else:
                output = self._forward_resident_set(x, indices, values)
            return finish(output)

    def snapshot(self) -> dict[str, Any]:
        result = self.stats.as_dict()
        result.update(
            {
                "layer": self.layer,
                "pinned_experts": self.pinned_count,
                "cache_slots": self.cache_slots,
                "cache_policy": self.cache_policy,
                "resident_slots": self.pool_size,
                "scratch_slots": self.scratch_slots,
                "bank_slots": self.bank_size,
                "resident_experts": len(self._expert_to_slot),
                "fused_gate_up": self._fused_gate_up,
                "sorted_prefill": self._sort_prefill,
                "native_demand": self._native_demand,
                "native_demand_active": self._native_demand_active,
                "elastic_decode_cache_enabled": self._elastic_decode_cache_enabled,
                "elastic_decode_cache_active": self._elastic_decode_cache_active,
                "f16_affine_moe": True,
                "expert_bytes": sum(
                    location.row_bytes for location in self.locations.values()
                ),
            }
        )
        return result
