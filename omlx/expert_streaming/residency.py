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
    # Non-expert tensors that stay on their mmap/SSD path while streaming is
    # enabled (currently the Qwen4-Exp PLE), excluded from ``fixed_bytes``.
    mmap_bytes: int
    pinned_bytes: int
    cache_bytes: int
    scratch_bytes: int
    resident_bytes: int
    cache_slots_per_layer: int


def _forced_mmap_bytes(model_path: Path) -> int:
    """Bytes the loader keeps on an mmap path whenever streaming is enabled.

    ``maybe_apply_pre_load_patches`` forces the Qwen4-Exp PLE onto its mmap/SSD
    path for every expert-streaming load, so those tensors never become
    resident and must not be charged against the admission ceiling.
    """

    try:
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    model_type = str(config.get("model_type", "")).replace("-", "_").lower()
    if model_type != "qwen4_exp":
        return 0
    from ..patches.mlx_vlm_qwen4_exp_compat.residency import (
        qwen4_exp_residency_estimate,
    )

    estimate = qwen4_exp_residency_estimate(model_path)
    return int(estimate.ple_bytes) if estimate.supported else 0


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
        scratch_experts=int(
            getattr(settings, "expert_streaming_scratch_experts", 32)
        ),
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
    scratch_experts: int = 32,
    num_layers: int,
    num_experts: int,
    top_k: int,
    streaming_mode: str = "soft_reap",
) -> ExpertStreamingEstimate:
    path = Path(model_path).expanduser().resolve()
    index = SafetensorExpertIndex(path)
    layers = index.expert_layer_ids()
    if not layers:
        raise ValueError("Checkpoint contains no indexed routed-expert tensors")
    if streaming_mode == "soft_reap":
        if not manifest_path:
            raise ValueError("Soft-REAP mode requires an expert pin manifest")
        manifest = load_soft_reap_manifest(
            manifest_path,
            layer_ids=layers,
            num_experts=num_experts,
        )
    elif streaming_mode == "cache_only":
        manifest = SoftReapManifest.empty(layer_ids=layers)
    else:
        raise ValueError("Expert streaming mode must be soft_reap or cache_only")
    checkpoint_bytes = sum(file.stat().st_size for file in path.glob("*.safetensors"))
    streamed_tensor_bytes = sum(index.streamed_storage_bytes(layer) for layer in layers)
    mmap_bytes = _forced_mmap_bytes(path)
    fixed_bytes = max(0, checkpoint_bytes - streamed_tensor_bytes - mmap_bytes)
    per_layer_expert_bytes = {
        layer: index.expert_storage_bytes(layer) for layer in layers
    }
    pinned_bytes = sum(
        len(manifest.experts_for_layer(layer)) * per_layer_expert_bytes[layer]
        for layer in layers
    )
    one_slot_all_layers = sum(per_layer_expert_bytes.values())
    requested_slots = max(0, int(cache_experts))
    minimum_slots = max(
        min(top_k, num_experts - len(manifest.experts_for_layer(layer)))
        for layer in layers
    )
    cache_slots = max(requested_slots, minimum_slots)
    cache_bytes = cache_slots * one_slot_all_layers
    available_scratch = max(
        0,
        min(
            num_experts
            - len(manifest.experts_for_layer(layer))
            - cache_slots
            for layer in layers
        ),
    )
    scratch_slots = min(max(0, int(scratch_experts)), available_scratch)
    scratch_bytes = scratch_slots * one_slot_all_layers
    resident_bytes = int(
        (fixed_bytes + pinned_bytes + cache_bytes + scratch_bytes)
        * _OVERHEAD_FACTOR
    )
    return ExpertStreamingEstimate(
        checkpoint_bytes=checkpoint_bytes,
        streamed_tensor_bytes=streamed_tensor_bytes,
        fixed_bytes=fixed_bytes,
        mmap_bytes=mmap_bytes,
        pinned_bytes=pinned_bytes,
        cache_bytes=cache_bytes,
        scratch_bytes=scratch_bytes,
        resident_bytes=resident_bytes,
        cache_slots_per_layer=cache_slots,
    )
