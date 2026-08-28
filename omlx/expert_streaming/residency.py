# SPDX-License-Identifier: Apache-2.0
"""Header-only resident-memory estimates for Soft-REAP admission control."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .manifest import SoftReapManifest, load_soft_reap_manifest
from .safetensors import SafetensorExpertIndex

_OVERHEAD_FACTOR = 1.05


@dataclass(frozen=True)
class ExpertStreamingEstimate:
    checkpoint_bytes: int
    streamed_tensor_bytes: int
    fixed_bytes: int
    pinned_bytes: int
    cache_bytes: int
    resident_bytes: int
    cache_slots_per_layer: int


def estimate_for_model_settings(
    model_path: str | Path,
    settings: object | None,
) -> ExpertStreamingEstimate | None:
    """Resolve a configured estimate, returning ``None`` when disabled."""

    if settings is None or not getattr(settings, "expert_streaming_enabled", False):
        return None
    streaming_mode = str(
        getattr(settings, "expert_streaming_mode", "soft_reap")
    ).lower()
    if streaming_mode not in {"soft_reap", "cache_only"}:
        raise ValueError("Expert streaming mode must be soft_reap or cache_only")
    manifest = getattr(settings, "expert_streaming_manifest", None)
    if streaming_mode == "soft_reap" and not manifest:
        raise ValueError("Expert streaming is enabled without a Soft-REAP manifest")
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") or config
    num_layers = int(text.get("num_hidden_layers", 0) or 0)
    num_experts = int(text.get("num_experts", text.get("n_routed_experts", 0)) or 0)
    top_k = int(
        text.get("num_experts_per_tok", text.get("num_experts_per_token", 0)) or 0
    )
    if not num_layers or not num_experts or not top_k:
        raise ValueError("Model config does not declare compatible MoE geometry")
    return estimate_expert_streaming_residency(
        path,
        manifest,
        cache_experts=int(getattr(settings, "expert_streaming_cache_experts", 32)),
        num_layers=num_layers,
        num_experts=num_experts,
        top_k=top_k,
        streaming_mode=streaming_mode,
    )


def estimate_expert_streaming_residency(
    model_path: str | Path,
    manifest_path: str | Path | None,
    *,
    cache_experts: int,
    num_layers: int,
    num_experts: int,
    top_k: int,
    streaming_mode: str = "soft_reap",
) -> ExpertStreamingEstimate:
    path = Path(model_path).expanduser().resolve()
    if streaming_mode == "soft_reap":
        if not manifest_path:
            raise ValueError("Soft-REAP mode requires an expert pin manifest")
        manifest = load_soft_reap_manifest(
            manifest_path,
            num_layers=num_layers,
            num_experts=num_experts,
        )
    elif streaming_mode == "cache_only":
        manifest = SoftReapManifest.empty(num_layers)
    else:
        raise ValueError("Expert streaming mode must be soft_reap or cache_only")
    index = SafetensorExpertIndex(path)
    layers = list(range(num_layers))
    checkpoint_bytes = sum(file.stat().st_size for file in path.glob("*.safetensors"))
    streamed_tensor_bytes = index.tensor_bytes(layers)
    fixed_bytes = max(0, checkpoint_bytes - streamed_tensor_bytes)
    per_layer_expert_bytes = [index.expert_bytes(layer) for layer in layers]
    pinned_bytes = sum(
        len(manifest.experts_for_layer(layer)) * per_layer_expert_bytes[layer]
        for layer in layers
    )
    one_slot_all_layers = sum(per_layer_expert_bytes)
    requested_slots = max(0, int(cache_experts))
    minimum_slots = max(
        min(top_k, num_experts - len(manifest.experts_for_layer(layer)))
        for layer in layers
    )
    cache_slots = max(requested_slots, minimum_slots)
    cache_bytes = cache_slots * one_slot_all_layers
    resident_bytes = int((fixed_bytes + pinned_bytes + cache_bytes) * _OVERHEAD_FACTOR)
    return ExpertStreamingEstimate(
        checkpoint_bytes=checkpoint_bytes,
        streamed_tensor_bytes=streamed_tensor_bytes,
        fixed_bytes=fixed_bytes,
        pinned_bytes=pinned_bytes,
        cache_bytes=cache_bytes,
        resident_bytes=resident_bytes,
        cache_slots_per_layer=cache_slots,
    )
