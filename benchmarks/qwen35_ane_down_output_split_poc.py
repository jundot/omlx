#!/usr/bin/env python3
"""Probe output-row ANE splitting for a real Qwen3.5-family down projection."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.activations import swiglu


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    value = mx.sum(af * bf) / (
        mx.sqrt(mx.sum(mx.square(af))) * mx.sqrt(mx.sum(mx.square(bf)))
    )
    mx.eval(value)
    return float(value.item())


def _measure(call, repeats: int) -> tuple[float, list[float]]:
    value = call()
    mx.eval(value)
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = call()
        mx.eval(value)
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def _first_mlp(model: Any) -> Any:
    for module in model.modules():
        if all(
            hasattr(module, name)
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            return module
    raise RuntimeError("No dense Qwen MLP was found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=(0.10, 0.20, 0.30, 0.40, 0.50),
    )
    args = parser.parse_args()

    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.patches.qwen35_q4_mlp import _linear_qmm
    from omlx.utils.model_loading import load_text_model

    if not fast.qwen35_ane_available():
        raise RuntimeError("The private ANE runtime is unavailable")
    if not fast.has_symbol("qwen35_ane_compile_linear_bank"):
        raise RuntimeError("The ANE procedure-bank compiler is unavailable")

    print(f"Loading {args.model}", flush=True)
    model, _ = load_text_model(str(args.model))
    mlp = _first_mlp(model)
    down = mlp.down_proj
    bits = int(down.bits)
    group_size = int(down.group_size)
    output_dim = int(down.weight.shape[0])
    input_dim = int(down.weight.shape[1]) * 32 // bits

    mx.random.seed(0)
    model_dim = (
        int(mlp.gate_proj.weight.shape[1])
        * 32
        // int(mlp.gate_proj.bits)
    )
    x = mx.random.normal((1, args.tokens, model_dim)).astype(mx.float16)
    gate = _linear_qmm(mlp.gate_proj, x, 8)
    up = _linear_qmm(mlp.up_proj, x, 8)
    activation = mx.contiguous(swiglu(gate, up))
    mx.eval(activation)

    reference = _linear_qmm(down, activation, 8)
    mx.eval(reference)
    gpu_seconds, gpu_samples = _measure(
        lambda: _linear_qmm(down, activation, 8), args.repeats
    )

    prepared = []
    weights0 = []
    weights1 = []
    for fraction in args.fractions:
        ane_outputs = (int(output_dim * fraction) // 128) * 128
        split = ane_outputs // 2
        gpu_outputs = output_dim - ane_outputs
        if (
            ane_outputs <= 0
            or split % 64
            or gpu_outputs <= 0
            or gpu_outputs % 64
        ):
            print(f"Skipping invalid fraction {fraction:.4f}", flush=True)
            continue
        dense0 = mx.contiguous(
            mx.dequantize(
                down.weight[:split],
                down.scales[:split],
                down.biases[:split],
                group_size=group_size,
                bits=bits,
            ).astype(mx.float32)
        )
        dense1 = mx.contiguous(
            mx.dequantize(
                down.weight[split:ane_outputs],
                down.scales[split:ane_outputs],
                down.biases[split:ane_outputs],
                group_size=group_size,
                bits=bits,
            ).astype(mx.float32)
        )
        gpu_weight = mx.contiguous(down.weight[ane_outputs:])
        gpu_scales = mx.contiguous(down.scales[ane_outputs:])
        gpu_biases = mx.contiguous(down.biases[ane_outputs:])
        mx.eval(
            dense0,
            dense1,
            gpu_weight,
            gpu_scales,
            gpu_biases,
        )
        weights0.append(dense0)
        weights1.append(dense1)
        prepared.append(
            (
                fraction,
                ane_outputs,
                gpu_weight,
                gpu_scales,
                gpu_biases,
            )
        )

    started = time.perf_counter()
    models0 = fast.qwen35_ane_compile_linear_bank(weights0, args.tokens, 1)
    models1 = fast.qwen35_ane_compile_linear_bank(weights1, args.tokens, 2)
    compile_seconds = time.perf_counter() - started
    del weights0, weights1

    results = []
    for index, entry in enumerate(prepared):
        fraction, ane_outputs, gpu_weight, gpu_scales, gpu_biases = entry

        def candidate_call(
            gpu_weight=gpu_weight,
            gpu_scales=gpu_scales,
            gpu_biases=gpu_biases,
            model0=models0[index],
            model1=models1[index],
        ):
            return fast.qwen35_ane_dual_affine_qmm_t(
                activation,
                gpu_weight,
                gpu_scales,
                gpu_biases,
                model0,
                model1,
                bits,
                8,
                group_size,
                0,
            )

        seconds, samples = _measure(candidate_call, args.repeats)
        # Some private-runtime builds produce an invalid result for the very
        # first evaluation after a freshly loaded bank. _measure deliberately
        # performs and discards that warm-up before accuracy is inspected.
        candidate = candidate_call()
        mx.eval(candidate)
        difference = candidate.astype(mx.float32) - reference.astype(mx.float32)
        mx.eval(difference)
        result = {
            "requested_fraction": fraction,
            "realized_fraction": ane_outputs / output_dim,
            "ane_outputs": ane_outputs,
            "median_ms": seconds * 1000,
            "samples_ms": [sample * 1000 for sample in samples],
            "speedup_vs_gpu": gpu_seconds / seconds,
            "cosine": _cosine(reference, candidate),
            "rmse": float(mx.sqrt(mx.mean(mx.square(difference))).item()),
            "max_abs": float(mx.max(mx.abs(difference)).item()),
        }
        results.append(result)
        print("CANDIDATE " + json.dumps(result, sort_keys=True), flush=True)

    print(
        "RESULT "
        + json.dumps(
            {
                "model": str(args.model),
                "tokens": args.tokens,
                "layer": type(mlp).__name__,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "bits": bits,
                "group_size": group_size,
                "compile_seconds": compile_seconds,
                "gpu_median_ms": gpu_seconds * 1000,
                "gpu_samples_ms": [sample * 1000 for sample in gpu_samples],
                "candidates": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
