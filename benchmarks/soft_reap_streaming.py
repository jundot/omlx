#!/usr/bin/env python3
"""Reproducible Soft-REAP memory, exactness, and end-to-end benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import psutil

from omlx.expert_streaming import (
    estimate_expert_streaming_residency,
    install_expert_streaming,
)
from omlx.expert_streaming.manifest import load_soft_reap_manifest
from omlx.expert_streaming.safetensors import ExpertReader, SafetensorExpertIndex


def _geometry(model_path: Path) -> tuple[int, int, int]:
    config = json.loads((model_path / "config.json").read_text())
    text = config.get("text_config") or config
    return (
        int(text["num_hidden_layers"]),
        int(text.get("num_experts", text.get("n_routed_experts"))),
        int(text.get("num_experts_per_tok", text.get("num_experts_per_token"))),
    )


def _gib(value: int) -> float:
    return round(value / 1024**3, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--streaming-mode",
        choices=("soft_reap", "cache_only"),
        default="soft_reap",
    )
    parser.add_argument("--cache-experts", type=int, default=32)
    parser.add_argument("--scratch-experts", type=int, default=32)
    parser.add_argument("--cache-policy", choices=("lru", "route_frequency"), default="route_frequency")
    parser.add_argument("--no-fast-resource-loading", action="store_true")
    parser.add_argument("--no-direct-io", action="store_true")
    parser.add_argument("--no-native-demand", action="store_true")
    parser.add_argument("--no-decode-scratch-cache", action="store_true")
    parser.add_argument("--io-coalescing-kib", type=int, default=64)
    parser.add_argument("--full-load", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--direct-replay",
        action="store_true",
        help="Replay an identical language-model forward to isolate a hot no-miss pass.",
    )
    args = parser.parse_args()
    if not 0 <= args.cache_experts <= 512:
        parser.error("--cache-experts must be between 0 and 512")
    if not 0 <= args.scratch_experts <= 512:
        parser.error("--scratch-experts must be between 0 and 512")
    if not 0 <= args.io_coalescing_kib <= 4096:
        parser.error("--io-coalescing-kib must be between 0 and 4096")
    if args.runs <= 0 or args.max_tokens <= 0:
        parser.error("--runs and --max-tokens must be positive")
    if args.no_fast_resource_loading and (
        not args.no_direct_io or not args.no_native_demand
    ):
        parser.error(
            "--no-fast-resource-loading requires --no-direct-io and "
            "--no-native-demand"
        )
    if args.streaming_mode == "soft_reap" and args.manifest is None:
        parser.error("--manifest is required in soft_reap mode")
    layers, experts, top_k = _geometry(args.model)
    index = SafetensorExpertIndex(args.model)
    layer_ids = index.expert_layer_ids() or list(range(layers))
    manifest = (
        load_soft_reap_manifest(
            args.manifest, layer_ids=layer_ids, num_experts=experts
        )
        if args.manifest is not None
        else None
    )
    estimate = estimate_expert_streaming_residency(
        args.model,
        args.manifest,
        cache_experts=args.cache_experts,
        scratch_experts=args.scratch_experts,
        num_layers=layers,
        num_experts=experts,
        top_k=top_k,
        streaming_mode=args.streaming_mode,
    )
    report = {
        "model": str(args.model.resolve()),
        "layers": layers,
        "experts": experts,
        "top_k": top_k,
        "streaming_mode": args.streaming_mode,
        "pinned_range": manifest.pinned_count_range if manifest else (0, 0),
        "checkpoint_gib": _gib(estimate.checkpoint_bytes),
        "fixed_ple_trunk_gib": _gib(estimate.fixed_bytes),
        "pinned_gib": _gib(estimate.pinned_bytes),
        "cache_gib": _gib(estimate.cache_bytes),
        "projected_resident_gib": _gib(estimate.resident_bytes),
        "cache_slots_per_layer": estimate.cache_slots_per_layer,
        "scratch_slots_per_layer": args.scratch_experts,
        "cache_policy": args.cache_policy,
    }

    # Always verify direct SSD row reads against MLX's checkpoint mapping.
    reader = ExpertReader(index)
    sample_layer = layer_ids[0]
    ffn_marker = f"layers.{sample_layer}.ffn."
    container = "ffn" if any(ffn_marker in key for key in index.weight_map) else "mlp"
    location = index.layer(sample_layer, container_name=container)[
        ("gate_proj", "weight")
    ]
    sample_ids = (
        list(manifest.experts_for_layer(sample_layer)[:2]) + [experts - 1]
        if manifest
        else [0, experts - 1]
    )
    started = time.perf_counter()
    rows = reader.read_rows(location, sample_ids)
    mx.eval(rows)
    elapsed = time.perf_counter() - started
    reference = mx.load(str(location.path))[location.name][sample_ids]
    mx.eval(reference)
    report["row_read_exact"] = bool(mx.all(rows == reference).item())
    report["sample_read_ms"] = round(elapsed * 1000, 3)
    report["sample_read_mib"] = round(reader.bytes_read / 1024**2, 3)
    reader.close()

    if args.full_load:
        from mlx_vlm import generate
        from mlx_vlm.utils import load as vlm_load

        from omlx.model_settings import ModelSettings
        from omlx.utils.model_loading import (
            materialize_lazy_state,
            maybe_apply_pre_load_patches,
        )

        settings = ModelSettings(
            expert_streaming_enabled=True,
            expert_streaming_mode=args.streaming_mode,
            expert_streaming_manifest=(
                str(args.manifest.resolve()) if args.manifest else None
            ),
            expert_streaming_cache_experts=args.cache_experts,
            expert_streaming_scratch_experts=args.scratch_experts,
            expert_streaming_cache_policy=args.cache_policy,
            expert_streaming_fast_resource_loading=not args.no_fast_resource_loading,
            expert_streaming_direct_io=not args.no_direct_io,
            expert_streaming_native_demand=not args.no_native_demand,
            expert_streaming_decode_scratch_as_cache=not args.no_decode_scratch_cache,
            expert_streaming_io_coalescing_kib=args.io_coalescing_kib,
            qwen4_ple_ssd_offload=False,
            mtp_enabled=True,
        )
        maybe_apply_pre_load_patches(args.model, model_settings=settings, for_vlm=True)
        started = time.perf_counter()
        model, processor = vlm_load(str(args.model), lazy=True)
        runtime = install_expert_streaming(
            model,
            args.model,
            args.manifest,
            cache_experts=args.cache_experts,
            scratch_experts=args.scratch_experts,
            cache_policy=args.cache_policy,
            fast_resource_loading=not args.no_fast_resource_loading,
            direct_io=not args.no_direct_io,
            native_demand=not args.no_native_demand,
            decode_scratch_as_cache=not args.no_decode_scratch_cache,
            fast_resource_max_gap_bytes=args.io_coalescing_kib * 1024,
            streaming_mode=args.streaming_mode,
        )
        materialize_lazy_state(model)
        report["load_seconds"] = round(time.perf_counter() - started, 3)
        report["mlx_active_gib"] = _gib(mx.get_active_memory())
        report["mlx_peak_gib"] = _gib(mx.get_peak_memory())
        report["process_rss_gib"] = _gib(psutil.Process().memory_info().rss)
        print(
            json.dumps(
                {
                    "stage": "loaded",
                    "load_seconds": report["load_seconds"],
                    "mlx_active_gib": report["mlx_active_gib"],
                    "mlx_peak_gib": report["mlx_peak_gib"],
                    "process_rss_gib": report["process_rss_gib"],
                }
            ),
            flush=True,
        )

        report["runs"] = []
        if args.direct_replay:
            tokenizer = getattr(processor, "tokenizer", processor)
            input_ids = mx.array([tokenizer.encode(args.prompt)])
            language_model = getattr(model, "language_model", model)
            outputs = []
            for run in range(2):
                started = time.perf_counter()
                result = language_model(input_ids, cache=None)
                logits = getattr(result, "logits", result)
                mx.eval(logits)
                outputs.append(logits)
                report["runs"].append(
                    {
                        "run": run + 1,
                        "forward_seconds": round(time.perf_counter() - started, 3),
                    }
                )
            report["direct_replay_exact"] = bool(
                mx.all(outputs[0] == outputs[1]).item()
            )
            report["direct_replay_max_abs"] = float(
                mx.max(mx.abs(outputs[0] - outputs[1])).item()
            )
        else:
            for run in range(args.runs):
                started = time.perf_counter()
                result = generate(
                    model,
                    processor,
                    args.prompt,
                    max_tokens=args.max_tokens,
                    temp=0.0,
                    verbose=False,
                )
                generation_seconds = time.perf_counter() - started
                report["runs"].append(
                    {
                        "run": run + 1,
                        "generation_seconds": round(generation_seconds, 3),
                        "generation_tokens": int(result.generation_tokens),
                        "generation_tokens_per_second": round(
                            result.generation_tokens / generation_seconds, 3
                        ),
                        "output": result.text,
                    }
                )
                print(
                    json.dumps({"stage": "run", **report["runs"][-1]}),
                    flush=True,
                )
        report["streaming"] = runtime.stats()
        runtime.close()

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
