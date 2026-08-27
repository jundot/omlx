#!/usr/bin/env python3
"""Tune one real Qwen GDN projection across dual ANE, CPU, and GPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx


def _measure(factory, repeats: int) -> tuple[float, list[float], tuple]:
    output = factory()
    if output is None:
        raise RuntimeError("GDN dispatch was ineligible")
    mx.eval(*output)
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = factory()
        if output is None:
            raise RuntimeError("GDN dispatch failed")
        mx.eval(*output)
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples, output


def _cosine_tuple(reference: tuple, candidate: tuple) -> float:
    left = mx.concatenate([value.reshape(-1) for value in reference]).astype(
        mx.float32
    )
    right = mx.concatenate([value.reshape(-1) for value in candidate]).astype(
        mx.float32
    )
    cosine = mx.sum(left * right) / (
        mx.sqrt(mx.sum(mx.square(left))) * mx.sqrt(mx.sum(mx.square(right)))
    )
    mx.eval(cosine)
    return float(cosine.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--fractions", nargs="+", type=float, default=(0.35, 0.40, 0.45, 0.50)
    )
    parser.add_argument(
        "--cpu-fractions", nargs="+", type=float, default=(0.0,)
    )
    args = parser.parse_args()

    from omlx.patches import qwen35_ane_prefill as patch
    from omlx.utils.model_loading import load_text_model

    print(f"Loading {args.model}", flush=True)
    model, _ = load_text_model(str(args.model))
    gdn = next(module for module in model.modules() if patch._eligible_gdn(module))
    linears = patch._gdn_linears(gdn)
    input_dim = int(linears[0].weight.shape[1]) * 32 // int(linears[0].bits)
    mx.random.seed(0)
    x = mx.random.normal((1, args.tokens, input_dim)).astype(
        linears[0].scales.dtype
    )

    def gpu_call():
        return tuple(patch._tail_qmm_or_linear(linear, x, 8) for linear in linears)

    gpu_seconds, gpu_samples, reference = _measure(gpu_call, args.repeats)
    prepared = []
    prepared_outputs = set()
    qkv, z, _, _ = linears
    z_outputs = int(z.weight.shape[0])
    qkv_outputs = int(qkv.weight.shape[0])
    total_outputs = z_outputs + qkv_outputs
    for fraction in args.fractions:
        ane_outputs = patch._recurrent_safe_gdn_ane_outputs(
            z_outputs, qkv_outputs, fraction, 128
        )
        if not ane_outputs or ane_outputs in prepared_outputs:
            continue
        config = patch._AneGDNConfig(args.tokens, fraction, 8, True)
        value = patch._prepare_gdn_for_bank(gdn, config)
        if value is not None:
            state, dense0, dense1 = value
            prepared.append(
                (fraction, ane_outputs / total_outputs, state, dense0, dense1)
            )
            prepared_outputs.add(ane_outputs)
    if not prepared:
        raise RuntimeError("No recurrent-safe GDN ANE width could be prepared")
    mx.eval(
        *[entry[3] for entry in prepared],
        *[entry[4] for entry in prepared],
    )
    banks = patch._compile_dual_banks(
        [entry[3] for entry in prepared],
        [entry[4] for entry in prepared],
        args.tokens,
    )
    if banks is None:
        raise RuntimeError("GDN calibration bank failed to compile")
    models0, models1, programs = banks
    results = []
    for index, (requested_fraction, effective_fraction, _state, _, _) in enumerate(
        prepared
    ):
        for cpu_fraction in args.cpu_fractions:
            config = patch._AneGDNConfig(
                args.tokens,
                requested_fraction,
                8,
                True,
                cpu_fraction=cpu_fraction,
                cpu_threads=args.cpu_threads,
                cpu_shared_resource=True,
            )
            runtime = patch._prepare_gdn_runtime_state(
                gdn, config, models0[index], models1[index]
            )
            if runtime is None:
                continue
            gdn._omlx_ane_gdn_config = config
            gdn._omlx_ane_gdn_state = runtime
            gdn._omlx_ane_gdn_failed = False
            seconds, samples, output = _measure(
                lambda: patch._gdn_backend_exact(gdn, x), args.repeats
            )
            result = {
                "ane_fraction": effective_fraction,
                "requested_ane_fraction": requested_fraction,
                "cpu_fraction": cpu_fraction,
                "median_ms": seconds * 1000,
                "samples_ms": [sample * 1000 for sample in samples],
                "speedup_vs_gpu": gpu_seconds / seconds,
                "cosine": _cosine_tuple(reference, output),
            }
            results.append(result)
            print("CANDIDATE " + json.dumps(result, sort_keys=True), flush=True)
    print(
        "RESULT "
        + json.dumps(
            {
                "gpu_median_ms": gpu_seconds * 1000,
                "gpu_samples_ms": [sample * 1000 for sample in gpu_samples],
                "resident_programs": programs,
                "candidates": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
