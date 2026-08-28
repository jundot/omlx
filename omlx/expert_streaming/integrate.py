# SPDX-License-Identifier: Apache-2.0
"""Install SSD-streamed expert banks into already-lazy MLX MoE models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from .execution import SpeculativeExecution
from .manifest import SoftReapManifest, load_soft_reap_manifest
from .pool import StreamingSwitchGLU
from .routing import ResidentPreferredMoeBlock
from .safetensors import PROJECTIONS, ExpertReader, SafetensorExpertIndex

logger = logging.getLogger(__name__)


def _main_layers(model: Any) -> list[Any]:
    candidates = (
        getattr(
            getattr(getattr(model, "language_model", None), "model", None),
            "layers",
            None,
        ),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    )
    for layers in candidates:
        if isinstance(layers, (list, tuple)) and layers:
            return list(layers)
    raise ValueError("Could not locate the model's routed MoE layers")


def _projection_metadata(switch_mlp: Any) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for projection in PROJECTIONS:
        module = getattr(switch_mlp, projection, None)
        if module is None or not all(
            hasattr(module, name) for name in ("weight", "scales", "biases")
        ):
            raise ValueError(
                "Expert streaming currently requires stacked affine-quantized SwitchGLU weights"
            )
        metadata[projection] = {
            "group_size": int(getattr(module, "group_size", 64)),
            "bits": int(getattr(module, "bits", 4)),
            "mode": str(getattr(module, "mode", "affine")),
        }
    return metadata


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
    layers = _main_layers(model)
    first_mlp = getattr(layers[0], "mlp", None)
    num_experts = int(getattr(first_mlp, "num_experts", 0) or 0)
    top_k = int(getattr(first_mlp, "top_k", 0) or 0)
    if not num_experts or not top_k:
        raise ValueError(
            "The selected model does not expose compatible routed MoE geometry"
        )
    if streaming_mode == "soft_reap":
        if not manifest_path:
            raise ValueError("Soft-REAP mode requires an expert pin manifest")
        manifest = load_soft_reap_manifest(
            manifest_path,
            num_layers=len(layers),
            num_experts=num_experts,
        )
    else:
        manifest = SoftReapManifest.empty(len(layers))
    index = SafetensorExpertIndex(model_path)
    expert_bytes_all_layers = sum(
        index.expert_bytes(layer) for layer in range(len(layers))
    )
    minimum_cache_slots = max(
        min(top_k, num_experts - len(manifest.experts_for_layer(layer)))
        for layer in range(len(layers))
    )
    cache_slots = max(0, int(cache_experts), minimum_cache_slots)
    cache_budget_bytes = cache_slots * expert_bytes_all_layers
    reader = ExpertReader(index)
    pools: list[StreamingSwitchGLU] = []
    try:
        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            original = getattr(mlp, "switch_mlp", None)
            if original is None:
                raise ValueError(f"Layer {layer_idx} has no switch_mlp")
            logger.info(
                "Expert streaming loading layer %d/%d (%d pinned experts, %d hot slots)",
                layer_idx + 1,
                len(layers),
                len(manifest.experts_for_layer(layer_idx)),
                cache_slots,
            )
            pool = StreamingSwitchGLU(
                layer=layer_idx,
                num_experts=num_experts,
                top_k=top_k,
                pinned_experts=manifest.experts_for_layer(layer_idx),
                cache_slots=cache_slots,
                locations=index.layer(layer_idx),
                projection_metadata=_projection_metadata(original),
                activation=original.activation,
                reader=reader,
                cache_policy="route_frequency",
            )
            mlp.switch_mlp = pool
            if substitution_threshold_percent > 0:
                layer.mlp = ResidentPreferredMoeBlock(
                    mlp,
                    threshold_percent=substitution_threshold_percent,
                )
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
        len(layers),
        manifest.pinned_count_range,
        pools[0].cache_slots if pools else 0,
        sum(pool.pool_size * index.expert_bytes(pool.layer) for pool in pools)
        / 1024**3,
    )
    return runtime
