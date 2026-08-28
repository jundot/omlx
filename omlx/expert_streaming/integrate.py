# SPDX-License-Identifier: Apache-2.0
"""Install SSD-streamed expert banks into already-lazy MLX MoE models."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from .adapters import discover_moe_layers, projection_schema
from .execution import SpeculativeExecution
from .manifest import SoftReapManifest, load_soft_reap_manifest
from .pool import StreamingSwitchGLU
from .routing import ResidentPreferredMoeBlock
from .safetensors import ExpertReader, SafetensorExpertIndex

logger = logging.getLogger(__name__)


@dataclass
class ExpertStreamingRuntime:
    model_path: Path
    manifest: SoftReapManifest
    cache_budget_bytes: int
    cache_slots_per_layer: int
    scratch_budget_bytes: int
    scratch_slots_per_layer: int
    substitution_threshold_percent: float
    streaming_mode: str
    execution_policy: str
    reader: ExpertReader
    pools: list[StreamingSwitchGLU]
    hotlist_profile_path: Path | None = None
    hotlist_fingerprint: str | None = None
    hotlist_preloaded: int = 0
    optimistic_preloaded: int = 0
    hotlist_profile_error: str | None = None
    execution: SpeculativeExecution | None = None

    def attach_model(self, model: Any) -> None:
        if self.execution is None:
            self.execution = SpeculativeExecution(
                self,
                policy=self.execution_policy,
            )
        self.execution.attach(model)

    def stats(self) -> dict[str, Any]:
        layers = [pool.snapshot() for pool in self.pools]
        totals = {
            key: sum(int(layer[key]) for layer in layers)
            for key in (
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
                "speculative_routes",
                "speculative_misses",
                "hotness_decays",
                "warm_start_loads",
            )
        }
        timing_totals = {
            key: sum(float(layer[key]) for layer in layers)
            for key in (
                "bank_bind_seconds",
                "bank_materialize_seconds",
                "scratch_prefetch_wait_seconds",
                "scratch_mlx_materialize_seconds",
            )
        }
        attempts = totals["pinned_hits"] + totals["cache_hits"] + totals["cache_misses"]
        totals["hit_rate"] = (
            (totals["pinned_hits"] + totals["cache_hits"]) / attempts
            if attempts
            else 1.0
        )
        return {
            "enabled": True,
            "manifest": str(self.manifest.source) if self.manifest.source else None,
            "cache_budget_bytes": self.cache_budget_bytes,
            "cache_slots_per_layer": self.cache_slots_per_layer,
            "scratch_budget_bytes": self.scratch_budget_bytes,
            "scratch_slots_per_layer": self.scratch_slots_per_layer,
            "layer_count": len(self.pools),
            "resident_experts": sum(
                int(layer["resident_experts"]) for layer in layers
            ),
            "resident_capacity": sum(pool.pool_size for pool in self.pools),
            "execution_bank_slots": (
                max((pool.bank_size for pool in self.pools), default=0)
            ),
            "execution_banks_per_layer": 1 if self.pools else 0,
            "fused_gate_up": bool(self.pools)
            and all(bool(layer["fused_gate_up"]) for layer in layers),
            "sorted_prefill": bool(self.pools)
            and all(bool(layer["sorted_prefill"]) for layer in layers),
            "substitution_threshold_percent": self.substitution_threshold_percent,
            "streaming_mode": self.streaming_mode,
            "cache_policy": "route_frequency",
            "execution_policy": self.execution_policy,
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
            "hotlist_profile": (
                str(self.hotlist_profile_path) if self.hotlist_profile_path else None
            ),
            "hotlist_preloaded": self.hotlist_preloaded,
            "optimistic_preloaded": self.optimistic_preloaded,
            "hotlist_profile_error": self.hotlist_profile_error,
            **totals,
            **timing_totals,
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
    profile_path = Path(profile_dir).expanduser() / f"{fingerprint[:24]}.json"
    return profile_path, fingerprint


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
        logger.warning(
            "Ignoring invalid expert hotlist profile %s: %s", profile_path, exc
        )
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
    substitution_threshold_percent: float = 0.0,
    execution_policy: str = "checked",
    streaming_mode: str = "soft_reap",
    hotlist_profile_dir: str | Path | None = None,
) -> ExpertStreamingRuntime:
    """Replace main-layer SwitchGLUs before the lazy checkpoint is evaluated."""

    substitution_threshold_percent = float(substitution_threshold_percent)
    if not 0.0 <= substitution_threshold_percent <= 100.0:
        raise ValueError("Expert substitution threshold must be between 0% and 100%")
    execution_policy = str(execution_policy).strip().lower()
    if execution_policy not in {"checked", "speculative"}:
        raise ValueError(
            "Expert streaming execution policy must be checked or speculative"
        )
    streaming_mode = str(streaming_mode).strip().lower()
    if streaming_mode not in {"soft_reap", "cache_only"}:
        raise ValueError("Expert streaming mode must be soft_reap or cache_only")
    targets = discover_moe_layers(model)
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
        # ``arrays`` are only shape/dtype descriptors for checkpoint indexing.
        # Never retain them in the runtime schema: they are the original full
        # expert banks that streaming is meant to replace.
        runtime_schema = {
            projection: {
                key: value
                for key, value in metadata.items()
                if key != "arrays"
            }
            for projection, metadata in schema.items()
        }
        resolved[target.layer_id] = (runtime_schema, locations)
        expert_bytes_all_layers += sum(
            location.row_bytes for location in locations.values()
        )
    minimum_cache_slots = max(
        min(top_k, num_experts - len(manifest.experts_for_layer(layer)))
        for layer in layer_ids
    )
    cache_slots = max(0, int(cache_experts), minimum_cache_slots)
    cache_budget_bytes = cache_slots * expert_bytes_all_layers
    scratch_slots = max(0, int(scratch_experts))
    reader = ExpertReader(index)
    pools: list[StreamingSwitchGLU] = []
    try:
        for ordinal, target in enumerate(targets):
            layer_idx = target.layer_id
            original = target.switch_mlp
            schema, locations = resolved[layer_idx]
            logger.info(
                "Expert streaming loading layer %d/%d (%d pinned experts, "
                "%d hot slots, %d scratch slots)",
                ordinal + 1,
                len(targets),
                len(manifest.experts_for_layer(layer_idx)),
                cache_slots,
                scratch_slots,
            )
            pool = StreamingSwitchGLU(
                layer=layer_idx,
                num_experts=num_experts,
                top_k=top_k,
                pinned_experts=manifest.experts_for_layer(layer_idx),
                cache_slots=cache_slots,
                scratch_slots=scratch_slots,
                locations=locations,
                projection_metadata=schema,
                activation=original.activation,
                reader=reader,
                cache_policy="route_frequency",
            )
            target.replace_switch(pool)
            if substitution_threshold_percent > 0:
                module_name = type(target.moe).__module__.lower()
                if "qwen" not in module_name:
                    raise ValueError(
                        "Resident substitution is currently available only for "
                        "Qwen-family MoE routers"
                    )
                setattr(target.layer, target.container_name, ResidentPreferredMoeBlock(
                    target.moe,
                    threshold_percent=substitution_threshold_percent,
                ))
            pools.append(pool)
            mx.clear_cache()
    except Exception:
        reader.close()
        raise

    try:
        hotlist_preloaded, optimistic_preloaded, hotlist_profile_error = _load_hotlist(
            hotlist_profile_path, hotlist_fingerprint, pools
        )
    except Exception:
        reader.close()
        raise

    scratch_budget_bytes = sum(
        pool.scratch_slots
        * sum(location.row_bytes for location in pool.locations.values())
        for pool in pools
    )
    actual_scratch_slots = max((pool.scratch_slots for pool in pools), default=0)
    runtime = ExpertStreamingRuntime(
        model_path=resolved_model_path,
        manifest=manifest,
        cache_budget_bytes=cache_budget_bytes,
        cache_slots_per_layer=int(cache_slots),
        scratch_budget_bytes=scratch_budget_bytes,
        scratch_slots_per_layer=actual_scratch_slots,
        substitution_threshold_percent=float(substitution_threshold_percent),
        streaming_mode=streaming_mode,
        execution_policy=execution_policy,
        reader=reader,
        pools=pools,
        hotlist_profile_path=hotlist_profile_path,
        hotlist_fingerprint=hotlist_fingerprint,
        hotlist_preloaded=hotlist_preloaded,
        optimistic_preloaded=optimistic_preloaded,
        hotlist_profile_error=hotlist_profile_error,
    )
    model._omlx_expert_streaming_runtime = runtime
    runtime.attach_model(model)
    language_model = getattr(model, "language_model", None)
    if language_model is not None and language_model is not model:
        language_model._omlx_expert_streaming_runtime = runtime
        runtime.attach_model(language_model)
    logger.info(
        "Expert streaming ready: mode=%s, %d layers, pinned range %s, "
        "%d hot slots/layer, %d scratch slots/layer, learned preload=%d, "
        "optimistic preload=%d, "
        "%.2f GiB bank",
        streaming_mode,
        len(targets),
        manifest.pinned_count_range,
        pools[0].cache_slots if pools else 0,
        pools[0].scratch_slots if pools else 0,
        hotlist_preloaded,
        optimistic_preloaded,
        sum(
            pool.bank_size
            * sum(location.row_bytes for location in pool.locations.values())
            for pool in pools
        )
        / 1024**3,
    )
    return runtime
