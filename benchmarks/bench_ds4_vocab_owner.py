#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare DS4's exact full vocabulary head with its two TP row shards.

This is an isolated real-weight probe.  It does not load the transformer,
start a server, contact a peer, or modify checkpoint/runtime state.  The same
script runs independently on each Mac; compare rank-0's full-head median with
the current critical half-head median on rank 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=40)
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def evaluate(value: mx.array) -> None:
    mx.eval(value)
    mx.synchronize()


def main() -> int:
    args = parse_args()
    index = json.loads(
        (args.model / "model.safetensors.index.json").read_text()
    )["weight_map"]
    key = "head.weight" if "head.weight" in index else "lm_head.weight"
    weight = mx.load(str(args.model / index[key]))[key]
    if tuple(weight.shape) != (129280, 4096) or weight.dtype != mx.bfloat16:
        raise ValueError(
            f"expected DS4 BF16 head [129280,4096], got {weight.shape} {weight.dtype}"
        )
    midpoint = int(weight.shape[0]) // 2
    first = mx.contiguous(weight[:midpoint])
    second = mx.contiguous(weight[midpoint:])
    mx.random.seed(20260823)
    hidden = mx.random.normal((1, 1, weight.shape[1])).astype(mx.bfloat16)
    mx.eval(first, second, hidden)
    mx.synchronize()

    def full() -> mx.array:
        return hidden @ weight.T

    def rank0_half() -> mx.array:
        return hidden @ first.T

    def rank1_half() -> mx.array:
        return hidden @ second.T

    functions = {
        "full": full,
        "rank0_half": rank0_half,
        "rank1_half": rank1_half,
    }
    for _ in range(args.warmup):
        for function in functions.values():
            evaluate(function())

    timings = {name: [] for name in functions}
    orders = (
        ("full", "rank0_half", "rank1_half"),
        ("rank1_half", "rank0_half", "full"),
        ("rank0_half", "full", "rank1_half"),
    )
    for cycle in range(args.cycles):
        for name in orders[cycle % len(orders)]:
            started = time.perf_counter_ns()
            evaluate(functions[name]())
            timings[name].append((time.perf_counter_ns() - started) / 1e6)

    full_logits = full()
    reconstructed = mx.concatenate([rank0_half(), rank1_half()], axis=-1)
    mx.eval(full_logits, reconstructed)
    mx.synchronize()
    exact = bool(mx.array_equal(full_logits, reconstructed).item())
    full_token = int(mx.argmax(full_logits, axis=-1).item())
    reconstructed_token = int(mx.argmax(reconstructed, axis=-1).item())
    digest = hashlib.sha256(
        np.asarray(full_logits.view(mx.uint16)).tobytes()
    ).hexdigest()
    result = {
        "device": mx.device_info().get("device_name"),
        "model": str(args.model),
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "parity": {
            "array_equal": exact,
            "full_token": full_token,
            "reconstructed_token": reconstructed_token,
            "logits_sha256": digest,
        },
        "timings": {
            name: summary(values) for name, values in timings.items()
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if exact and full_token == reconstructed_token else 2


if __name__ == "__main__":
    raise SystemExit(main())
