#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate rank-owned rows of DS4's replicated input projections.

This is an isolated checkpoint probe. It verifies that slicing independent
output rows preserves every BF16 projection boundary, then measures how much
rank-local compute remains under a weighted TP ownership vector. The packed
all-gather is intentionally excluded so its measured JACCL cost can be added
separately before any production seam is considered.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

PROJECTIONS = {
    0: (("wq_a", "mxfp8"), ("wkv", "mxfp8")),
    4: (
        ("wq_a", "mxfp8"),
        ("wkv", "mxfp8"),
        ("compressor.wkv", "dense"),
        ("compressor.wgate", "dense"),
        ("indexer.compressor.wkv", "dense"),
        ("indexer.compressor.wgate", "dense"),
    ),
    128: (
        ("wq_a", "mxfp8"),
        ("wkv", "mxfp8"),
        ("compressor.wkv", "dense"),
        ("compressor.wgate", "dense"),
    ),
}


def _load(model: Path, layer: int, ratio: int) -> list[dict[str, Any]]:
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    specs = []
    keys = []
    for name, kind in PROJECTIONS[ratio]:
        prefix = f"layers.{layer}.attn.{name}"
        spec = {"name": name, "kind": kind, "weight_key": f"{prefix}.weight"}
        keys.append(spec["weight_key"])
        if kind == "mxfp8":
            spec["scales_key"] = f"{prefix}.scales"
            keys.append(spec["scales_key"])
        specs.append(spec)
    tensors = {}
    for shard in sorted({index[key] for key in keys}):
        loaded = mx.load(str(model / shard))
        tensors.update({key: loaded[key] for key in keys if key in loaded})
    for spec in specs:
        spec["weight"] = tensors[spec["weight_key"]]
        if spec["kind"] == "mxfp8":
            spec["scales"] = tensors[spec["scales_key"]]
    mx.eval(
        *[
            value
            for spec in specs
            for value in (spec["weight"], spec.get("scales"))
            if value is not None
        ]
    )
    mx.synchronize()
    return specs


def _range(rows: int, weights: tuple[int, ...], rank: int) -> tuple[int, int]:
    total = sum(weights)
    start = rows * sum(weights[:rank]) // total
    stop = rows * sum(weights[: rank + 1]) // total
    return start, stop


def _project(x, spec, start: int = 0, stop: int | None = None):
    weight = spec["weight"]
    stop = int(weight.shape[0]) if stop is None else stop
    weight = weight[start:stop]
    if spec["kind"] == "mxfp8":
        return mx.quantized_matmul(
            x,
            weight,
            scales=spec["scales"][start:stop],
            transpose=True,
            group_size=32,
            bits=8,
            mode="mxfp8",
        )
    return x @ weight.T


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--ratio", type=int, choices=(0, 4, 128), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--shard-weights", default="3,5")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    weights = tuple(int(item) for item in args.shard_weights.split(","))
    if not 0 <= args.rank < len(weights):
        raise ValueError("rank is outside shard vector")
    specs = _load(args.model, args.layer, args.ratio)
    mx.random.seed(20260822 + args.layer + args.tokens)
    x = mx.random.normal((1, args.tokens, 4096)).astype(mx.bfloat16)
    mx.eval(x)

    ranges = [
        _range(int(spec["weight"].shape[0]), weights, args.rank) for spec in specs
    ]

    def baseline():
        return [_project(x, spec) for spec in specs]

    def candidate():
        return [
            _project(x, spec, start, stop)
            for spec, (start, stop) in zip(specs, ranges)
        ]

    expected = baseline()
    actual = candidate()
    mx.eval(*expected, *actual)
    mx.synchronize()
    exact = [
        bool(mx.array_equal(full[..., start:stop], owned).item())
        for full, owned, (start, stop) in zip(expected, actual, ranges)
    ]

    def evaluate(fn):
        values = fn()
        mx.eval(*values)
        mx.synchronize()

    for _ in range(args.warmup):
        evaluate(candidate)
        evaluate(baseline)
    timings = {"baseline": [], "candidate": []}
    for _ in range(args.cycles):
        for name, fn in (
            ("baseline", baseline),
            ("candidate", candidate),
            ("candidate", candidate),
            ("baseline", baseline),
        ):
            started = time.perf_counter_ns()
            evaluate(fn)
            timings[name].append((time.perf_counter_ns() - started) / 1e6)

    result = {
        "model": str(args.model),
        "layer": args.layer,
        "ratio": args.ratio,
        "rank": args.rank,
        "shard_weights": weights,
        "tokens": args.tokens,
        "projection_rows": {
            spec["name"]: {
                "full": int(spec["weight"].shape[0]),
                "owned": stop - start,
                "start": start,
                "stop": stop,
                "exact": match,
            }
            for spec, (start, stop), match in zip(specs, ranges, exact)
        },
        "all_boundaries_exact": all(exact),
        "baseline": _summary(timings["baseline"]),
        "candidate": _summary(timings["candidate"]),
    }
    result["local_compute_speedup"] = (
        result["baseline"]["median_ms"] / result["candidate"]["median_ms"]
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
