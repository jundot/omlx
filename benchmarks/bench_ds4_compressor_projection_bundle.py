#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate one BF16 GEMM for DS4 compressor/indexer projections sharing X."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx


def _summary(values):
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=8)
    args = parser.parse_args()

    prefix = f"layers.{args.layer}.attn."
    names = (
        prefix + "compressor.wkv.weight",
        prefix + "compressor.wgate.weight",
        prefix + "indexer.compressor.wkv.weight",
        prefix + "indexer.compressor.wgate.weight",
        prefix + "indexer.weights_proj.weight",
    )
    index = json.loads(
        (args.model / "model.safetensors.index.json").read_text()
    )["weight_map"]
    tensors = {}
    for shard in sorted({index[name] for name in names}):
        tensors.update(mx.load(str(args.model / shard)))
    weights = tuple(tensors[name] for name in names)
    widths = tuple(int(weight.shape[0]) for weight in weights)
    combined = mx.concatenate(weights, axis=0)
    mx.random.seed(15000)
    x = mx.random.normal((1, args.tokens, 4096)).astype(mx.bfloat16)
    mx.eval(x, combined)
    mx.synchronize()

    def baseline():
        return tuple(x @ weight.T for weight in weights)

    def candidate():
        output = x @ combined.T
        boundaries = []
        start = 0
        for width in widths:
            boundaries.append(output[..., start : start + width])
            start += width
        return tuple(boundaries)

    expected, actual = baseline(), candidate()
    mx.eval(*expected, *actual)
    mx.synchronize()
    exact = all(
        bool(mx.array_equal(reference, value).item())
        for reference, value in zip(expected, actual)
    )

    def evaluate(function):
        mx.eval(*function())
        mx.synchronize()

    for _ in range(args.warmup):
        evaluate(candidate)
        evaluate(baseline)
    timings = {"candidate": [], "baseline": []}
    for _ in range(args.cycles):
        for name, function in (
            ("candidate", candidate),
            ("baseline", baseline),
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            started = time.perf_counter_ns()
            evaluate(function)
            timings[name].append((time.perf_counter_ns() - started) / 1e6)

    baseline_stats = _summary(timings["baseline"])
    candidate_stats = _summary(timings["candidate"])
    print(
        json.dumps(
            {
                "model": str(args.model),
                "layer": args.layer,
                "tokens": args.tokens,
                "widths": widths,
                "combined_bytes": combined.nbytes,
                "exact": exact,
                "baseline": baseline_stats,
                "candidate": candidate_stats,
                "speedup": baseline_stats["median_ms"]
                / candidate_stats["median_ms"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
