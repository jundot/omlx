#!/usr/bin/env python3
"""Benchmark Qwen3.5-family GPU, single-ANE, and dual-ANE prefill paths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("OMLX_QWEN35_Q4_MLP_ALLOW_GS128", "1")

import mlx.core as mx


def inject_extension(path: Path):
    name = "omlx.custom_kernels.qwen35_prefill._ext"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load native extension at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def hidden_tensor(output: Any) -> mx.array:
    hidden = output.hidden_states if hasattr(output, "hidden_states") else output
    return hidden[-1] if isinstance(hidden, (list, tuple)) else hidden


def cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    value = mx.sum(af * bf) / (mx.sqrt(mx.sum(af * af)) * mx.sqrt(mx.sum(bf * bf)))
    mx.eval(value)
    return float(value.item())


def accuracy(model: Any, reference: mx.array, candidate: mx.array) -> dict[str, Any]:
    lm = model.language_model
    if hasattr(lm, "lm_head"):
        reference_logits = lm.lm_head(reference[:, -1:, :])
        candidate_logits = lm.lm_head(candidate[:, -1:, :])
    else:
        reference_logits = lm.model.embed_tokens.as_linear(reference[:, -1:, :])
        candidate_logits = lm.model.embed_tokens.as_linear(candidate[:, -1:, :])
    difference = candidate.astype(mx.float32) - reference.astype(mx.float32)
    mx.eval(reference_logits, candidate_logits, difference)
    return {
        "hidden_cosine": cosine(reference, candidate),
        "hidden_rmse": float(mx.sqrt(mx.mean(mx.square(difference))).item()),
        "hidden_max_abs": float(mx.max(mx.abs(difference)).item()),
        "logit_cosine": cosine(reference_logits, candidate_logits),
        "gpu_top_token": int(mx.argmax(reference_logits, axis=-1).item()),
        "candidate_top_token": int(mx.argmax(candidate_logits, axis=-1).item()),
        "top_token_match": bool(
            int(mx.argmax(reference_logits, axis=-1).item())
            == int(mx.argmax(candidate_logits, axis=-1).item())
        ),
    }


def run_body(model: Any, tokens: mx.array) -> mx.array:
    if getattr(model, "_omlx_benchmark_force_lm", False):
        return hidden_tensor(model.language_model.model(tokens))
    return hidden_tensor(
        model.language_model(tokens, skip_logits=True, return_hidden=True)
    )


def benchmark_mode(
    model: Any,
    tokens: mx.array,
    repeats: int,
) -> tuple[dict[str, Any], mx.array]:
    output = run_body(model, tokens)
    mx.eval(output)
    mx.synchronize()

    profile = os.environ.get("OMLX_ANE_PROFILE") == "1" and bool(
        getattr(model, "_omlx_ane_resident_program_count", 0)
    )
    if profile:
        from omlx.custom_kernels.qwen35_prefill import fast

        fast.qwen35_ane_profile_reset()

    samples = []
    graph_build_samples = []
    execution_samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = run_body(model, tokens)
        graph_built = time.perf_counter()
        mx.eval(output)
        mx.synchronize()
        finished = time.perf_counter()
        samples.append(finished - started)
        graph_build_samples.append(graph_built - started)
        execution_samples.append(finished - graph_built)
    median = statistics.median(samples)
    profile_result: dict[str, Any] = {}
    if profile:
        raw = fast.qwen35_ane_profile_snapshot()
        elapsed_ns = sum(samples) * 1e9
        for category, metrics in raw.items():
            operations = metrics["operations"]
            profile_result[category] = {
                "operations": int(operations),
                "input_ready_ms_per_op": metrics["pack_ns"] / operations / 1e6
                if operations
                else 0.0,
                "parallel_region_ms_per_op": metrics["ane_region_ns"]
                / operations
                / 1e6
                if operations
                else 0.0,
                "ane0_eval_ms_per_op": metrics["ane0_eval_ns"]
                / operations
                / 1e6
                if operations
                else 0.0,
                "ane1_eval_ms_per_op": metrics["ane1_eval_ns"]
                / operations
                / 1e6
                if operations
                else 0.0,
                "ane0_launch_us_per_op": metrics["ane0_launch_ns"]
                / operations
                / 1e3
                if operations
                else 0.0,
                "ane1_launch_us_per_op": metrics["ane1_launch_ns"]
                / operations
                / 1e3
                if operations
                else 0.0,
                "gpu_qmm_ms_per_op": metrics["gpu_qmm_ns"]
                / operations
                / 1e6
                if operations
                else 0.0,
                "gpu_completion_ms_per_op": metrics["gpu_completion_ns"]
                / operations
                / 1e6
                if operations
                else 0.0,
                "cpu_matmul_ms_per_op": metrics["cpu_matmul_ns"]
                / operations
                / 1e6
                if operations
                else 0.0,
                "cpu_completion_ms_per_op": metrics["cpu_completion_ns"]
                / operations
                / 1e6
                if operations
                else 0.0,
                "gap_before_ms_per_op": metrics["gap_before_ns"]
                / operations
                / 1e6
                if operations
                else 0.0,
                "ane_last": int(metrics["ane_last"]),
                "gpu_last": int(metrics["gpu_last"]),
                "ane0_duty_cycle": metrics["ane0_eval_ns"] / elapsed_ns,
                "ane1_duty_cycle": metrics["ane1_eval_ns"] / elapsed_ns,
            }
        profile_result["total"] = {
            "ane0_duty_cycle": sum(
                metrics["ane0_eval_ns"] for metrics in raw.values()
            )
            / elapsed_ns,
            "ane1_duty_cycle": sum(
                metrics["ane1_eval_ns"] for metrics in raw.values()
            )
            / elapsed_ns,
        }
    return (
        {
            "median_seconds": median,
            "samples_seconds": samples,
            "median_graph_build_ms": statistics.median(graph_build_samples) * 1e3,
            "graph_build_samples_ms": [value * 1e3 for value in graph_build_samples],
            "median_execution_ms": statistics.median(execution_samples) * 1e3,
            "prompt_tokens_per_second": int(tokens.size) / median,
            **({"ane_profile": profile_result} if profile_result else {}),
        },
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument(
        "--force-lm",
        action="store_true",
        help="load through oMLX's text-model path, matching the app benchmark",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help="worker count for the optional fp16 CPU share (default: 8; 0=auto)",
    )
    parser.add_argument(
        "--disable-cpu-shared-resource",
        action="store_true",
        help="disable performance-aware shared-resource CPU scheduling",
    )
    parser.add_argument(
        "--cpu-threads-grid",
        nargs="+",
        type=int,
        help="Benchmark several CPU worker counts after one ANE compilation",
    )
    parser.add_argument(
        "--cpu-gdn-fraction-grid",
        nargs="+",
        type=float,
        help="Benchmark several CPU GDN shares after one ANE compilation",
    )
    parser.add_argument(
        "--cpu-down-fraction-grid",
        nargs="+",
        type=float,
        help="Benchmark several CPU down shares after one ANE compilation",
    )
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument(
        "--ane-sequence-length",
        type=int,
        help="Fixed ANE program rows (defaults to --tokens; use 2048 to test wide tiling)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("gpu", "single", "dual"),
        default=("gpu", "dual"),
    )
    parser.add_argument("--single-mlp-fraction", type=float, default=0.40)
    parser.add_argument("--single-gdn-fraction", type=float, default=0.40)
    parser.add_argument("--dual-mlp-fraction", type=float, default=0.53)
    parser.add_argument("--dual-gdn-fraction", type=float, default=0.50)
    parser.add_argument(
        "--cpu-fraction",
        type=float,
        default=0.0,
        help="Optional fp16 CPU share of each MLP gate/up projection",
    )
    parser.add_argument(
        "--cpu-down-fraction",
        type=float,
        default=0.0,
        help="Optional fp16 CPU share of each MLP down projection",
    )
    parser.add_argument(
        "--ane-down-fraction",
        type=float,
        default=0.0,
        help=(
            "Experimental output-row share, or per-ANE hidden share with "
            "--ane-fused-down"
        ),
    )
    parser.add_argument(
        "--ane-fused-down",
        action="store_true",
        help="Fuse each ANE gate/up slice through its partial down projection",
    )
    parser.add_argument(
        "--cpu-gdn-fraction",
        type=float,
        default=0.0,
        help="Optional fp16 CPU share of the residual GDN qkv projection",
    )
    parser.add_argument(
        "--disable-gdn",
        action="store_true",
        help="benchmark MLP offload without compiling or dispatching GDN",
    )
    args = parser.parse_args()
    ane_sequence_length = args.ane_sequence_length or args.tokens
    if "single" in args.modes and "dual" in args.modes:
        parser.error(
            "benchmark single and dual ANE in separate processes so resident "
            "programs from the first mode do not consume the second mode's budget"
        )
    if args.cpu_threads_grid and any(
        value < 0 or value > 64 for value in args.cpu_threads_grid
    ):
        parser.error("CPU worker counts must be between 0 and 64")
    if args.cpu_gdn_fraction_grid and any(
        value < 0 or value > 0.50 for value in args.cpu_gdn_fraction_grid
    ):
        parser.error("CPU GDN fractions must be between 0 and 0.50")
    if args.cpu_down_fraction_grid and any(
        value < 0 or value > 0.50 for value in args.cpu_down_fraction_grid
    ):
        parser.error("CPU down fractions must be between 0 and 0.50")

    native_ext = inject_extension(args.extension) if args.extension else None
    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.patches.qwen35_ane_prefill import enable_qwen35_ane_prefill
    from omlx.patches.qwen35_q4_mlp import (
        apply_qwen35_q4_lm_prefill_linear_patch,
        apply_qwen35_q4_mlp_patch,
    )

    native_ext = native_ext or fast._ext
    if native_ext is None:
        raise RuntimeError("The Qwen3.5 native extension is unavailable")

    print(f"Loading {args.model}", flush=True)
    if args.force_lm:
        from omlx.utils.model_loading import load_text_model

        model, _ = load_text_model(str(args.model))
        model._omlx_benchmark_force_lm = True
    else:
        from mlx_vlm.utils import load_model

        model = load_model(args.model, lazy=False, strict=False)
    apply_qwen35_q4_mlp_patch()
    if args.force_lm:
        # The app installs this after loading so it wraps the final class
        # implementation (including the optional MTP compatibility patch).
        apply_qwen35_q4_lm_prefill_linear_patch()
    mx.random.seed(0)
    tokens = mx.random.randint(0, 1000, shape=(1, args.tokens), dtype=mx.int32)
    mx.eval(tokens)

    results: dict[str, Any] = {
        "model": str(args.model),
        "prompt_tokens": args.tokens,
        "repeats": args.repeats,
    }
    reference = None
    for mode in args.modes:
        if mode == "single":
            started = time.perf_counter()
            mlp_layers = enable_qwen35_ane_prefill(
                model,
                sequence_length=ane_sequence_length,
                fraction=args.single_mlp_fraction,
                gdn=not args.disable_gdn,
                gdn_fraction=args.single_gdn_fraction,
                dual_ane=False,
                cpu_fraction=args.cpu_fraction,
                cpu_down_fraction=args.cpu_down_fraction,
                ane_down_fraction=args.ane_down_fraction,
                fused_down=args.ane_fused_down,
                cpu_gdn_fraction=args.cpu_gdn_fraction,
                cpu_threads=args.cpu_threads,
                cpu_shared_resource=not args.disable_cpu_shared_resource,
            )
            compile_seconds = time.perf_counter() - started
        elif mode == "dual":
            started = time.perf_counter()
            mlp_layers = enable_qwen35_ane_prefill(
                model,
                sequence_length=ane_sequence_length,
                fraction=args.dual_mlp_fraction,
                gdn=not args.disable_gdn,
                gdn_fraction=args.dual_gdn_fraction,
                dual_ane=True,
                cpu_fraction=args.cpu_fraction,
                cpu_down_fraction=args.cpu_down_fraction,
                ane_down_fraction=args.ane_down_fraction,
                fused_down=args.ane_fused_down,
                cpu_gdn_fraction=args.cpu_gdn_fraction,
                cpu_threads=args.cpu_threads,
                cpu_shared_resource=not args.disable_cpu_shared_resource,
            )
            compile_seconds = time.perf_counter() - started
        else:
            mlp_layers = 0
            compile_seconds = 0.0

        variants: list[tuple[str, int | None, float | None, float | None]] = [
            (mode, None, None, None)
        ]
        if mode in ("single", "dual") and args.cpu_threads_grid:
            variants = [
                (f"{mode}_cpu_threads_{threads}", threads, None, None)
                for threads in args.cpu_threads_grid
            ]
        if mode in ("single", "dual") and args.cpu_gdn_fraction_grid:
            variants = [
                (f"{mode}_cpu_gdn_{fraction:.3f}", None, fraction, None)
                for fraction in args.cpu_gdn_fraction_grid
            ]
        if mode in ("single", "dual") and args.cpu_down_fraction_grid:
            variants = [
                (f"{mode}_cpu_down_{fraction:.3f}", None, None, fraction)
                for fraction in args.cpu_down_fraction_grid
            ]
        for result_key, cpu_threads, cpu_gdn_fraction, cpu_down_fraction in variants:
            if cpu_threads is not None:
                for module in model.modules():
                    config = getattr(module, "_omlx_ane_prefill_config", None)
                    if config is not None:
                        module._omlx_ane_prefill_config = replace(
                            config, cpu_threads=cpu_threads
                        )
                    gdn_config = getattr(module, "_omlx_ane_gdn_config", None)
                    if gdn_config is not None:
                        module._omlx_ane_gdn_config = replace(
                            gdn_config, cpu_threads=cpu_threads
                        )
            if cpu_gdn_fraction is not None:
                from omlx.patches import qwen35_ane_prefill as ane_patch

                for module in model.modules():
                    gdn_config = getattr(module, "_omlx_ane_gdn_config", None)
                    gdn_state = getattr(module, "_omlx_ane_gdn_state", None)
                    if gdn_config is None or gdn_state is None:
                        continue
                    updated_config = replace(
                        gdn_config, cpu_fraction=cpu_gdn_fraction
                    )
                    updated_state = ane_patch._prepare_gdn_runtime_state(
                        module,
                        updated_config,
                        gdn_state.model,
                        gdn_state.model1,
                    )
                    if updated_state is None:
                        raise RuntimeError(
                            f"CPU GDN fraction {cpu_gdn_fraction:.3f} is ineligible"
                        )
                    module._omlx_ane_gdn_config = updated_config
                    module._omlx_ane_gdn_state = updated_state
                mx.clear_cache()
            if cpu_down_fraction is not None:
                from omlx.patches import qwen35_ane_prefill as ane_patch

                for module in model.modules():
                    state = getattr(module, "_omlx_ane_prefill_state", None)
                    if state is None or not hasattr(module, "down_proj"):
                        continue
                    module._omlx_ane_prefill_state = replace(
                        state,
                        down_cpu=ane_patch._prepare_cpu_linear(
                            module.down_proj, cpu_down_fraction
                        ),
                    )
                mx.clear_cache()
            measured, output = benchmark_mode(model, tokens, args.repeats)
            measured.update(
                {
                    "compile_seconds": compile_seconds,
                    "mlp_layers": mlp_layers,
                    "cpu_threads": cpu_threads
                    if cpu_threads is not None
                    else args.cpu_threads,
                    "cpu_shared_resource": not args.disable_cpu_shared_resource,
                    "cpu_down_fraction": (
                        cpu_down_fraction
                        if cpu_down_fraction is not None
                        else args.cpu_down_fraction
                    ),
                    "ane_down_fraction": args.ane_down_fraction,
                    "cpu_gdn_fraction": (
                        cpu_gdn_fraction
                        if cpu_gdn_fraction is not None
                        else args.cpu_gdn_fraction
                    ),
                    "dual_mlp_layers": int(
                        getattr(model, "_omlx_ane_dual_prefill_count", 0)
                    )
                    if mode != "gpu"
                    else 0,
                    "resident_programs": int(
                        getattr(model, "_omlx_ane_resident_program_count", 0)
                    )
                    if mode != "gpu"
                    else 0,
                    "procedures": int(
                        getattr(model, "_omlx_ane_procedure_count", 0)
                    )
                    if mode != "gpu"
                    else 0,
                    "gdn_layers": int(
                        getattr(model, "_omlx_ane_gdn_prefill_count", 0)
                    )
                    if mode != "gpu"
                    else 0,
                    "down_layers": int(
                        getattr(model, "_omlx_ane_down_prefill_count", 0)
                    )
                    if mode != "gpu"
                    else 0,
                }
            )
            if mode == "gpu":
                reference = output
            elif reference is not None:
                measured["accuracy_vs_gpu"] = accuracy(model, reference, output)
                measured["speedup_vs_gpu"] = (
                    results["gpu"]["median_seconds"] / measured["median_seconds"]
                )
            results[result_key] = measured
            print(
                f"{result_key.upper()} {json.dumps(measured, sort_keys=True)}",
                flush=True,
            )

    print("RESULT " + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
