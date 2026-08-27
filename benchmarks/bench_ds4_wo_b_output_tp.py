#!/usr/bin/env python3
"""Real-weight DS4 wo_b output-sharding benchmark and parity probe.

This does not initialize distributed MLX.  It isolates the compute term on one
Mac with all 43 checkpoint ``wo_b`` matrices, simulates the two-rank algebra,
and prices the old/new collective schedules from measured link constants.

The runtime prototype is decode-only: M>1 keeps the original full projection.
The forced M=2048 candidate below exists to show why promoting this layout to
prefill requires a separate communication-overlap result.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open


def _qmm(value, weight, scales):
    return mx.quantized_matmul(
        value,
        weight,
        scales=scales,
        biases=None,
        transpose=True,
        group_size=32,
        bits=8,
        mode="mxfp8",
    )


def _load_wo_b(model: Path):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    stack = ExitStack()
    files = {}
    modules = []
    for layer in range(43):
        weight_key = f"layers.{layer}.attn.wo_b.weight"
        scales_key = f"layers.{layer}.attn.wo_b.scales"
        filename = index[weight_key]
        if filename not in files:
            files[filename] = stack.enter_context(
                safe_open(model / filename, framework="numpy")
            )
        source = files[filename]
        weight = mx.array(source.get_tensor(weight_key))
        scales = mx.array(source.get_tensor(scales_key))
        modules.append((weight, scales))
    mx.eval([value for pair in modules for value in pair])
    return stack, modules


def _run_pass(modules, value, *, half: bool):
    mx.synchronize()
    started = time.perf_counter()
    for weight, scales in modules:
        stop = weight.shape[0] // 2 if half else weight.shape[0]
        out = _qmm(value, weight[:stop], scales[:stop])
        mx.eval(out)
    mx.synchronize()
    return time.perf_counter() - started


def _time_abba(modules, full_value, half_value, *, cycles: int):
    # Compile and fault every real layer/shape before timing. The half shard is
    # otherwise always second and inherits the full pass's warm allocator and
    # kernel cache, which can exaggerate M=1 speedup by nearly 3x.
    _run_pass(modules, full_value, half=False)
    _run_pass(modules, half_value, half=True)
    full_samples = []
    half_samples = []
    for _ in range(cycles):
        full_samples.append(_run_pass(modules, full_value, half=False))
        half_samples.append(_run_pass(modules, half_value, half=True))
        half_samples.append(_run_pass(modules, half_value, half=True))
        full_samples.append(_run_pass(modules, full_value, half=False))
    return full_samples, half_samples


def _parity_pass(modules, left, right):
    exact = 0
    elements = 0
    max_abs = 0.0
    max_rel = 0.0
    sum_abs = 0.0
    sum_squared = 0.0
    reference_squared = 0.0
    for weight, scales in modules:
        current = _qmm(left, weight, scales) + _qmm(right, weight, scales)
        latent = left + right
        split = weight.shape[0] // 2
        candidate = mx.concatenate(
            [
                _qmm(latent, weight[:split], scales[:split]),
                _qmm(latent, weight[split:], scales[split:]),
            ],
            axis=-1,
        )
        mx.eval(current, candidate)
        equal = current == candidate
        delta = mx.abs(current.astype(mx.float32) - candidate.astype(mx.float32))
        denominator = mx.maximum(mx.abs(current.astype(mx.float32)), 1e-12)
        exact += int(mx.sum(equal).item())
        elements += int(equal.size)
        max_abs = max(max_abs, float(mx.max(delta).item()))
        max_rel = max(max_rel, float(mx.max(delta / denominator).item()))
        sum_abs += float(mx.sum(delta).item())
        sum_squared += float(mx.sum(delta * delta).item())
        reference_squared += float(
            mx.sum(current.astype(mx.float32) ** 2).item()
        )
    return {
        "exact_elements": exact,
        "elements": elements,
        "exact_fraction": exact / elements if elements else 1.0,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": sum_abs / elements if elements else 0.0,
        "rmse": math.sqrt(sum_squared / elements) if elements else 0.0,
        "relative_l2": (
            math.sqrt(sum_squared / reference_squared)
            if reference_squared > 0
            else 0.0
        ),
    }


def _collective_price(rows: int, *, latency_s: float, bandwidth: float):
    bf16 = 2
    hidden = rows * 4096 * bf16
    latent = rows * 8192 * bf16
    local_hidden = rows * 2048 * bf16
    current = latency_s + hidden / bandwidth
    candidate = 2 * latency_s + (latent + local_hidden) / bandwidth
    return {
        "current_seconds_43_layers": current * 43,
        "candidate_seconds_43_layers": candidate * 43,
        "added_seconds_43_layers": (candidate - current) * 43,
        "current_payload_bytes_per_layer": hidden,
        "candidate_payload_bytes_per_layer": latent + local_hidden,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--decode-repeats", type=int, default=5)
    parser.add_argument("--prefill-repeats", type=int, default=1)
    parser.add_argument("--collective-latency-us", type=float, default=28.5)
    parser.add_argument("--collective-bandwidth-gbps", type=float, default=6.2)
    parser.add_argument("--skip-prefill-parity", action="store_true")
    args = parser.parse_args()

    stack, modules = _load_wo_b(args.model.expanduser())
    try:
        mx.random.seed(7)
        result = {
            "layers": len(modules),
            "full_wo_b_bytes": sum(w.nbytes + s.nbytes for w, s in modules),
            "local_decode_shard_bytes": sum(
                w[: w.shape[0] // 2].nbytes + s[: s.shape[0] // 2].nbytes
                for w, s in modules
            ),
            "rows": {},
        }
        latency = args.collective_latency_us * 1e-6
        bandwidth = args.collective_bandwidth_gbps * 1e9
        for rows, repeats in ((1, args.decode_repeats), (2048, args.prefill_repeats)):
            left = mx.random.normal((rows, 8192)).astype(mx.bfloat16)
            right = mx.random.normal((rows, 8192)).astype(mx.bfloat16)
            full_samples, half_samples = _time_abba(
                modules,
                left,
                left + right,
                cycles=repeats,
            )
            parity = (
                None
                if rows == 2048 and args.skip_prefill_parity
                else _parity_pass(modules, left, right)
            )
            collective = _collective_price(
                rows,
                latency_s=latency,
                bandwidth=bandwidth,
            )
            full_median = statistics.median(full_samples)
            half_median = statistics.median(half_samples)
            result["rows"][str(rows)] = {
                "full_wo_b_seconds_43_layers": full_median,
                "half_wo_b_seconds_43_layers": half_median,
                "compute_speedup": full_median / half_median,
                "parity_current_sum_after_wo_b_vs_latent_sum_before_wo_b": parity,
                "collectives": collective,
                "isolated_compute_plus_collective_current_seconds": (
                    full_median + collective["current_seconds_43_layers"]
                ),
                "isolated_compute_plus_collective_candidate_seconds": (
                    half_median + collective["candidate_seconds_43_layers"]
                ),
            }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        stack.close()


if __name__ == "__main__":
    main()
