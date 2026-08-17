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
    reference_logits = lm.lm_head(reference[:, -1:, :])
    candidate_logits = lm.lm_head(candidate[:, -1:, :])
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
    for _ in range(repeats):
        started = time.perf_counter()
        output = run_body(model, tokens)
        mx.eval(output)
        mx.synchronize()
        samples.append(time.perf_counter() - started)
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
                "ane_region_ms_per_op": metrics["ane_region_ns"]
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
            "prompt_tokens_per_second": int(tokens.size) / median,
            **({"ane_profile": profile_result} if profile_result else {}),
        },
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--tokens", type=int, default=2048)
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
    args = parser.parse_args()
    if "single" in args.modes and "dual" in args.modes:
        parser.error(
            "benchmark single and dual ANE in separate processes so resident "
            "programs from the first mode do not consume the second mode's budget"
        )

    native_ext = inject_extension(args.extension) if args.extension else None
    from mlx_vlm.utils import load_model

    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.patches.qwen35_ane_prefill import enable_qwen35_ane_prefill
    from omlx.patches.qwen35_q4_mlp import apply_qwen35_q4_mlp_patch

    native_ext = native_ext or fast._ext
    if native_ext is None:
        raise RuntimeError("The Qwen3.5 native extension is unavailable")

    print(f"Loading {args.model}", flush=True)
    model = load_model(args.model, lazy=False, strict=False)
    apply_qwen35_q4_mlp_patch()
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
                sequence_length=args.tokens,
                fraction=args.single_mlp_fraction,
                gdn=True,
                gdn_fraction=args.single_gdn_fraction,
                dual_ane=False,
            )
            compile_seconds = time.perf_counter() - started
        elif mode == "dual":
            started = time.perf_counter()
            mlp_layers = enable_qwen35_ane_prefill(
                model,
                sequence_length=args.tokens,
                fraction=args.dual_mlp_fraction,
                gdn=True,
                gdn_fraction=args.dual_gdn_fraction,
                dual_ane=True,
            )
            compile_seconds = time.perf_counter() - started
        else:
            mlp_layers = 0
            compile_seconds = 0.0

        measured, output = benchmark_mode(model, tokens, args.repeats)
        measured.update(
            {
                "compile_seconds": compile_seconds,
                "mlp_layers": mlp_layers,
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
                "gdn_layers": int(getattr(model, "_omlx_ane_gdn_prefill_count", 0))
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
        results[mode] = measured
        print(f"{mode.upper()} {json.dumps(measured, sort_keys=True)}", flush=True)

    print("RESULT " + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
