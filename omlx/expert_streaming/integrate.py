# SPDX-License-Identifier: Apache-2.0
"""Install deterministic SSD-streamed expert banks into lazy MLX models."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import weakref
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from .adapters import discover_moe_layers, projection_schema
from .execution import ExpertStreamingExecution
from .manifest import SoftReapManifest, load_soft_reap_manifest
from .pool import StreamingSwitchGLU
from .safetensors import ExpertReader, SafetensorExpertIndex

logger = logging.getLogger(__name__)


class _NativeDemandSynchronizePatch:
    """Swap ``mlx.core.synchronize`` for a GIL-releasing variant.

    ``mx.eval`` releases the GIL while it waits for Metal, but
    ``mx.synchronize`` does not. A native-demand miss is resolved inside a
    Metal completion handler that must take the GIL to publish the missing
    expert rows, and ``waitUntilCompleted`` only returns after completion
    handlers have run. Any ``mx.synchronize()`` issued while such a command
    buffer is in flight (mlx-vlm's ``wired_limit`` exit, the engine pool's
    unload path, the scheduler's stream barriers) therefore deadlocks the
    process. While at least one native-demand runtime is installed, route
    every ``mx.synchronize`` call through the extension so the wait happens
    without the GIL; restore the original when the last runtime closes.
    """

    _lock = threading.Lock()
    _depth = 0
    _original: Any = None

    @classmethod
    def acquire(cls) -> None:
        from omlx.custom_kernels.fast_resource_loading import (
            synchronize_with_gil_released,
        )

        if synchronize_with_gil_released is None:
            raise RuntimeError(
                "Native expert demand requires the Fast Resource Loading extension"
            )
        with cls._lock:
            if cls._depth == 0:
                cls._original = mx.synchronize

                def synchronize(stream: Any = None) -> None:
                    synchronize_with_gil_released(stream)

                synchronize.__doc__ = getattr(cls._original, "__doc__", None)
                synchronize.__wrapped__ = cls._original  # type: ignore[attr-defined]
                mx.synchronize = synchronize
            cls._depth += 1

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            if cls._depth == 0:
                return
            cls._depth -= 1
            if cls._depth == 0 and cls._original is not None:
                mx.synchronize = cls._original
                cls._original = None

    @classmethod
    def active(cls) -> bool:
        with cls._lock:
            return cls._depth > 0


@dataclass
class ExpertStreamingRuntime:
    model_path: Path
    manifest: SoftReapManifest
    cache_budget_bytes: int
    cache_slots_per_layer: int
    scratch_budget_bytes: int
    scratch_slots_per_layer: int
    cache_policy: str
    streaming_mode: str
    native_demand: bool
    native_demand_decode_only: bool
    decode_scratch_as_cache: bool
    fast_resource_loading: bool
    fast_resource_loading_scope: str
    fast_resource_max_gap_bytes: int
    direct_io: bool
    reader: ExpertReader
    pools: list[StreamingSwitchGLU]
    adapters: list[Any] = field(default_factory=list)
    hotlist_profile_path: Path | None = None
    hotlist_fingerprint: str | None = None
    hotlist_preloaded: int = 0
    optimistic_preloaded: int = 0
    hotlist_profile_error: str | None = None
    execution: ExpertStreamingExecution | None = None
    _attached_models: list[weakref.ReferenceType[Any]] = field(default_factory=list)
    _synchronize_patched: bool = False

    def attach_model(self, model: Any) -> None:
        model._omlx_expert_streaming_runtime = self
        self._attached_models.append(weakref.ref(model))
        if self.execution is None:
            self.execution = ExpertStreamingExecution(self)
        self.execution.attach(model)

    def set_decode_cache_active(self, active: bool) -> bool:
        """Lend prompt-processing scratch rows to deterministic decode caching."""

        if not self.decode_scratch_as_cache:
            return True
        return all(pool.set_elastic_decode_cache_active(active) for pool in self.pools)

    def stats(self) -> dict[str, Any]:
        if self.native_demand:
            from omlx.custom_kernels.fast_resource_loading import (
                check_async_route_error,
            )

            if check_async_route_error is not None:
                check_async_route_error()
        layers = [pool.snapshot() for pool in self.pools]
        integer_fields = (
            "route_lookups",
            "pinned_hits",
            "cache_hits",
            "cache_misses",
            "evictions",
            "loads",
            "pinned_loads",
            "cold_loads",
            "scratch_loads",
            "scratch_prefetch_requests",
            "expert_major_calls",
            "qmm_calls",
            "sorted_prefill_groups",
            "sorted_prefill_routes",
            "sorted_qmm_calls",
            "hotness_decays",
            "warm_start_loads",
            "route_materialize_calls",
            "just_in_time_loads",
            "native_demand_calls",
            "native_demand_callbacks",
            "native_demand_positions",
            "native_demand_destination_fence_skips",
            "elastic_decode_cache_activations",
            "elastic_decode_cache_demotions",
            "scratch_groups",
        )
        timing_fields = (
            "bank_bind_seconds",
            "bank_materialize_seconds",
            "route_materialize_first_seconds",
            "route_materialize_seconds",
            "scratch_prefetch_wait_seconds",
            "scratch_mlx_materialize_seconds",
            "native_demand_submit_seconds",
            "native_demand_callback_seconds",
            "scratch_call_seconds",
            "scratch_load_seconds",
            "scratch_gather_seconds",
            "scratch_compute_seconds",
            "scratch_scatter_seconds",
            "scratch_submit_seconds",
        )
        totals = {
            key: sum(int(layer.get(key, 0)) for layer in layers)
            for key in integer_fields
        }
        totals.update(
            {
                key: sum(float(layer.get(key, 0.0)) for layer in layers)
                for key in timing_fields
            }
        )
        attempts = totals["pinned_hits"] + totals["cache_hits"] + totals["cache_misses"]
        totals["hit_rate"] = (
            (totals["pinned_hits"] + totals["cache_hits"]) / attempts
            if attempts
            else 1.0
        )
        return {
            "enabled": True,
            "manifest": str(self.manifest.source) if self.manifest.source else None,
            "streaming_mode": self.streaming_mode,
            "cache_policy": self.cache_policy,
            "cache_budget_bytes": self.cache_budget_bytes,
            "cache_slots_per_layer": self.cache_slots_per_layer,
            "scratch_budget_bytes": self.scratch_budget_bytes,
            "scratch_slots_per_layer": self.scratch_slots_per_layer,
            "layer_count": len(self.pools),
            "resident_experts": sum(int(layer["resident_experts"]) for layer in layers),
            "resident_capacity": sum(pool.pool_size for pool in self.pools),
            "execution_bank_slots": max((pool.bank_size for pool in self.pools), default=0),
            "execution_banks_per_layer": 1 if self.pools else 0,
            "fused_gate_up": bool(self.pools)
            and all(bool(layer["fused_gate_up"]) for layer in layers),
            "sorted_prefill": bool(self.pools)
            and all(bool(layer["sorted_prefill"]) for layer in layers),
            "native_demand": self.native_demand,
            "native_demand_decode_only": self.native_demand_decode_only,
            "decode_scratch_as_cache": self.decode_scratch_as_cache,
            "fast_resource_loading": self.fast_resource_loading,
            "fast_resource_loading_scope": self.fast_resource_loading_scope,
            "fast_resource_max_gap_bytes": self.fast_resource_max_gap_bytes,
            "direct_io": self.direct_io,
            "execution": self.execution.stats.as_dict() if self.execution else {},
            "ssd_bytes_read": self.reader.bytes_read,
            "ssd_read_operations": self.reader.read_operations,
            "ssd_preload_bytes_read": self.reader.direct_bytes_read,
            "ssd_preload_read_operations": self.reader.direct_read_operations,
            "ssd_cold_bytes_read": self.reader.file_cache_bytes_read,
            "ssd_cold_read_operations": self.reader.file_cache_read_operations,
            "ssd_io_seconds": self.reader.io_seconds,
            "ssd_decode_seconds": self.reader.decode_seconds,
            "ssd_readahead_descriptors": self.reader.readahead_descriptors,
            "ssd_no_cache_descriptors": self.reader.no_cache_descriptors,
            "frl_loads": self.reader.fast_loads,
            "frl_bytes_read": self.reader.fast_bytes_read,
            "frl_read_operations": self.reader.fast_read_operations,
            "frl_io_wait_seconds": self.reader.fast_io_wait_seconds,
            "frl_copy_seconds": self.reader.fast_copy_seconds,
            "frl_queue_acquire_seconds": self.reader.fast_queue_acquire_seconds,
            "frl_submission_to_completion_seconds": (
                self.reader.fast_submission_to_completion_seconds
            ),
            "frl_overlapped_submissions": self.reader.fast_overlapped_submissions,
            "frl_max_inflight": self.reader.fast_max_inflight,
            "frl_direct_loads": self.reader.fast_direct_loads,
            "frl_direct_bytes": self.reader.fast_direct_bytes,
            "hotlist_profile": (
                str(self.hotlist_profile_path) if self.hotlist_profile_path else None
            ),
            "hotlist_preloaded": self.hotlist_preloaded,
            "optimistic_preloaded": self.optimistic_preloaded,
            "hotlist_profile_error": self.hotlist_profile_error,
            **totals,
            "layers": layers,
        }

    def _save_hotlist(self) -> None:
        if self.hotlist_profile_path is None or self.hotlist_fingerprint is None:
            return
        payload = {
            "version": 1,
            "fingerprint": self.hotlist_fingerprint,
            "num_experts": self.pools[0].num_experts if self.pools else 0,
            "layers": {
                str(pool.layer): [list(entry) for entry in pool.hotlist()]
                for pool in self.pools
            },
        }
        try:
            self.hotlist_profile_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=f".{self.hotlist_profile_path.name}.",
                dir=self.hotlist_profile_path.parent,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, separators=(",", ":"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.hotlist_profile_path)
            except Exception:
                with suppress(OSError):
                    os.unlink(temporary)
                raise
        except (OSError, TypeError, ValueError) as exc:
            self.hotlist_profile_error = f"save: {exc}"
            logger.warning("Could not save expert hotlist profile: %s", exc)

    def close(self) -> None:
        self._save_hotlist()
        if self.execution is not None:
            self.execution.close()
        if self._synchronize_patched:
            self._synchronize_patched = False
            _NativeDemandSynchronizePatch.release()
        for reference in self._attached_models:
            model = reference()
            if (
                model is not None
                and getattr(model, "_omlx_expert_streaming_runtime", None) is self
            ):
                delattr(model, "_omlx_expert_streaming_runtime")
        self._attached_models.clear()
        for adapter in reversed(self.adapters):
            adapter.replace_switch(adapter.switch_mlp)
        self.reader.close()


def _hotlist_identity(
    model_path: Path, profile_dir: str | Path | None
) -> tuple[Path | None, str | None]:
    if profile_dir is None:
        return None, None
    index_path = model_path / "model.safetensors.index.json"
    stat = index_path.stat()
    identity = f"{model_path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    fingerprint = hashlib.sha256(identity).hexdigest()
    return Path(profile_dir).expanduser() / f"{fingerprint[:24]}.json", fingerprint


def _load_hotlist(
    profile_path: Path | None,
    fingerprint: str | None,
    pools: list[StreamingSwitchGLU],
) -> tuple[int, int, str | None]:
    def fill(entries: dict[int, list[tuple[int, int]]] | None = None):
        loaded = [
            pool.preload_hotlist(entries.get(pool.layer, []) if entries else [])
            for pool in pools
        ]
        return sum(value[0] for value in loaded), sum(value[1] for value in loaded)

    if profile_path is None or fingerprint is None or not profile_path.is_file():
        learned, optimistic = fill()
        return learned, optimistic, None
    try:
        payload = json.loads(profile_path.read_text())
        if payload.get("version") != 1 or payload.get("fingerprint") != fingerprint:
            learned, optimistic = fill()
            return learned, optimistic, None
        expected_experts = pools[0].num_experts if pools else 0
        if int(payload.get("num_experts", -1)) != expected_experts:
            learned, optimistic = fill()
            return learned, optimistic, None
        layers = payload.get("layers")
        if not isinstance(layers, dict):
            raise ValueError("layers must be an object")
        parsed: dict[int, list[tuple[int, int]]] = {}
        for pool in pools:
            raw_entries = layers.get(str(pool.layer), [])
            if not isinstance(raw_entries, list):
                raise ValueError(f"layer {pool.layer} must be a list")
            entries: list[tuple[int, int]] = []
            for entry in raw_entries:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError(f"invalid layer {pool.layer} hotlist entry")
                entries.append((int(entry[0]), int(entry[1])))
            parsed[pool.layer] = entries
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid expert hotlist profile %s: %s", profile_path, exc)
        learned, optimistic = fill()
        return learned, optimistic, f"load: {exc}"
    learned, optimistic = fill(parsed)
    return learned, optimistic, None


def install_expert_streaming(
    model: Any,
    model_path: str | Path,
    manifest_path: str | Path | None,
    *,
    cache_experts: int = 32,
    scratch_experts: int = 32,
    cache_policy: str = "route_frequency",
    streaming_mode: str = "soft_reap",
    hotlist_profile_dir: str | Path | None = None,
    fast_resource_loading: bool | str = False,
    fast_resource_max_gap_bytes: int = 64 * 1024,
    direct_io: bool = False,
    native_demand: bool = False,
    native_demand_decode_only: bool = True,
    decode_scratch_as_cache: bool = False,
) -> ExpertStreamingRuntime:
    """Replace routed MoE layers before the lazy checkpoint is evaluated."""

    streaming_mode = str(streaming_mode).strip().lower()
    if streaming_mode not in {"soft_reap", "cache_only"}:
        raise ValueError("Expert streaming mode must be soft_reap or cache_only")
    cache_policy = str(cache_policy).strip().lower()
    if cache_policy not in {"lru", "route_frequency"}:
        raise ValueError("Expert cache policy must be lru or route_frequency")
    fast_resource_max_gap_bytes = int(fast_resource_max_gap_bytes)
    if fast_resource_max_gap_bytes < 0:
        raise ValueError("Fast Resource Loading gap must be non-negative")
    if not 0 <= int(cache_experts) <= 512:
        raise ValueError("Expert cache size must be between 0 and 512")
    if not 0 <= int(scratch_experts) <= 512:
        raise ValueError("Expert scratch size must be between 0 and 512")

    targets = discover_moe_layers(model)
    if not targets:
        raise ValueError("Model contains no supported routed MoE layers")
    geometries = {(target.num_experts, target.top_k) for target in targets}
    if len(geometries) != 1:
        raise ValueError(
            "Expert streaming requires consistent expert count and top-k across layers"
        )
    layer_ids = [target.layer_id for target in targets]
    num_experts = targets[0].num_experts
    top_k = targets[0].top_k
    if streaming_mode == "soft_reap":
        if not manifest_path:
            raise ValueError("Soft-REAP mode requires an expert pin manifest")
        manifest = load_soft_reap_manifest(
            manifest_path,
            layer_ids=layer_ids,
            num_experts=num_experts,
        )
    else:
        manifest = SoftReapManifest.empty(layer_ids=layer_ids)

    index = SafetensorExpertIndex(model_path)
    resolved_model_path = Path(model_path).expanduser().resolve()
    hotlist_profile_path, hotlist_fingerprint = _hotlist_identity(
        resolved_model_path, hotlist_profile_dir
    )
    resolved: dict[int, tuple[dict[str, dict[str, Any]], dict]] = {}
    expert_bytes_all_layers = 0
    for target in targets:
        schema = projection_schema(target.switch_mlp)
        locations = index.layer(
            target.layer_id,
            container_name=target.container_name,
            schema=schema,
            num_experts=target.num_experts,
        )
        runtime_schema = {
            projection: {
                key: value for key, value in metadata.items() if key != "arrays"
            }
            for projection, metadata in schema.items()
        }
        resolved[target.layer_id] = (runtime_schema, locations)
        expert_bytes_all_layers += sum(location.row_bytes for location in locations.values())

    minimum_cache_slots = max(
        min(top_k, num_experts - len(manifest.experts_for_layer(layer)))
        for layer in layer_ids
    )
    cache_slots = max(0, int(cache_experts), minimum_cache_slots)
    scratch_slots = max(0, int(scratch_experts))
    reader = ExpertReader(
        index,
        cold_max_gap_bytes=fast_resource_max_gap_bytes,
        fast_resource_loading=fast_resource_loading,
        direct_io=direct_io,
    )
    if (native_demand or direct_io) and not reader.fast_resource_loading:
        reader.close()
        raise ValueError("Native expert demand and direct I/O require Fast Resource Loading")

    pools: list[StreamingSwitchGLU] = []
    replaced_targets = []
    try:
        for ordinal, target in enumerate(targets):
            schema, locations = resolved[target.layer_id]
            logger.info(
                "Expert streaming loading layer %d/%d (%d pinned, %d cache, %d scratch)",
                ordinal + 1,
                len(targets),
                len(manifest.experts_for_layer(target.layer_id)),
                cache_slots,
                scratch_slots,
            )
            pool = StreamingSwitchGLU(
                layer=target.layer_id,
                num_experts=num_experts,
                top_k=top_k,
                pinned_experts=manifest.experts_for_layer(target.layer_id),
                cache_slots=cache_slots,
                scratch_slots=scratch_slots,
                locations=locations,
                projection_metadata=schema,
                activation=target.switch_mlp.activation,
                reader=reader,
                cache_policy=cache_policy,
                native_demand=native_demand,
            )
            pool.enable_elastic_decode_cache(decode_scratch_as_cache)
            target.replace_switch(pool)
            replaced_targets.append(target)
            pools.append(pool)
            mx.clear_cache()
    except Exception:
        for target in reversed(replaced_targets):
            target.replace_switch(target.switch_mlp)
        reader.close()
        raise

    try:
        hotlist_preloaded, optimistic_preloaded, hotlist_profile_error = _load_hotlist(
            hotlist_profile_path, hotlist_fingerprint, pools
        )
    except Exception:
        for target in reversed(replaced_targets):
            target.replace_switch(target.switch_mlp)
        reader.close()
        raise

    cache_budget_bytes = cache_slots * expert_bytes_all_layers
    scratch_budget_bytes = sum(
        pool.scratch_slots
        * sum(location.row_bytes for location in pool.locations.values())
        for pool in pools
    )
    runtime = ExpertStreamingRuntime(
        model_path=resolved_model_path,
        manifest=manifest,
        cache_budget_bytes=cache_budget_bytes,
        cache_slots_per_layer=cache_slots,
        scratch_budget_bytes=scratch_budget_bytes,
        scratch_slots_per_layer=max((pool.scratch_slots for pool in pools), default=0),
        cache_policy=cache_policy,
        streaming_mode=streaming_mode,
        native_demand=bool(native_demand),
        native_demand_decode_only=bool(native_demand_decode_only),
        decode_scratch_as_cache=bool(decode_scratch_as_cache),
        fast_resource_loading=reader.fast_resource_loading,
        fast_resource_loading_scope=reader.fast_resource_loading_scope,
        fast_resource_max_gap_bytes=fast_resource_max_gap_bytes,
        direct_io=bool(reader.fast_direct_exact),
        reader=reader,
        pools=pools,
        adapters=replaced_targets,
        hotlist_profile_path=hotlist_profile_path,
        hotlist_fingerprint=hotlist_fingerprint,
        hotlist_preloaded=hotlist_preloaded,
        optimistic_preloaded=optimistic_preloaded,
        hotlist_profile_error=hotlist_profile_error,
    )
    try:
        if native_demand:
            _NativeDemandSynchronizePatch.acquire()
            runtime._synchronize_patched = True
        runtime.attach_model(model)
        language_model = getattr(model, "language_model", None)
        if language_model is not None and language_model is not model:
            runtime.attach_model(language_model)
    except Exception:
        runtime.close()
        raise

    logger.info(
        "Expert streaming ready: mode=%s layers=%d pinned=%s cache=%d scratch=%d "
        "frl=%s direct=%s native_demand=%s bank=%.2f GiB",
        streaming_mode,
        len(targets),
        manifest.pinned_count_range,
        pools[0].cache_slots if pools else 0,
        pools[0].scratch_slots if pools else 0,
        reader.fast_resource_loading_scope,
        reader.fast_direct_exact,
        native_demand,
        sum(
            pool.bank_size
            * sum(location.row_bytes for location in pool.locations.values())
            for pool in pools
        )
        / 1024**3,
    )
    return runtime
