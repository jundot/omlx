# SPDX-License-Identifier: Apache-2.0
"""Install SSD-streamed expert banks into already-lazy MLX MoE models."""

from __future__ import annotations

import logging
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
    substitution_threshold_percent: float
    streaming_mode: str
    execution_policy: str
    reader: ExpertReader
    pools: list[StreamingSwitchGLU]
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
                "expert_major_calls",
                "speculative_routes",
                "speculative_misses",
                "hotness_decays",
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
            **totals,
            "layers": layers,
        }

    def close(self) -> None:
        if self.execution is not None:
            self.execution.close()
        self.reader.close()


def install_expert_streaming(
    model: Any,
    model_path: str | Path,
    manifest_path: str | Path | None,
    *,
    cache_experts: int = 32,
    substitution_threshold_percent: float = 0.0,
    execution_policy: str = "checked",
    streaming_mode: str = "soft_reap",
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
    reader = ExpertReader(index)
    pools: list[StreamingSwitchGLU] = []
    try:
        for ordinal, target in enumerate(targets):
            layer_idx = target.layer_id
            original = target.switch_mlp
            schema, locations = resolved[layer_idx]
            logger.info(
                "Expert streaming loading layer %d/%d (%d pinned experts, %d hot slots)",
                ordinal + 1,
                len(targets),
                len(manifest.experts_for_layer(layer_idx)),
                cache_slots,
            )
            pool = StreamingSwitchGLU(
                layer=layer_idx,
                num_experts=num_experts,
                top_k=top_k,
                pinned_experts=manifest.experts_for_layer(layer_idx),
                cache_slots=cache_slots,
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

    runtime = ExpertStreamingRuntime(
        model_path=Path(model_path).expanduser().resolve(),
        manifest=manifest,
        cache_budget_bytes=cache_budget_bytes,
        cache_slots_per_layer=int(cache_slots),
        substitution_threshold_percent=float(substitution_threshold_percent),
        streaming_mode=streaming_mode,
        execution_policy=execution_policy,
        reader=reader,
        pools=pools,
    )
    model._omlx_expert_streaming_runtime = runtime
    runtime.attach_model(model)
    language_model = getattr(model, "language_model", None)
    if language_model is not None and language_model is not model:
        language_model._omlx_expert_streaming_runtime = runtime
        runtime.attach_model(language_model)
    logger.info(
        "Expert streaming ready: mode=%s, %d layers, pinned range %s, "
        "%d hot slots/layer, %.2f GiB bank",
        streaming_mode,
        len(targets),
        manifest.pinned_count_range,
        pools[0].cache_slots if pools else 0,
        sum(
            pool.pool_size
            * sum(location.row_bytes for location in pool.locations.values())
            for pool in pools
        )
        / 1024**3,
    )
    return runtime
