#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate one wider MXFP8 Q-B projection for main + sparse-indexer queries."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx


def _weighted_heads(value, rank: int):
    """Slice each of eight DS4 output groups by the signed 3:5 vector."""

    pieces = []
    for segment in mx.split(value, 8, axis=0):
        start = segment.shape[0] * (0 if rank == 0 else 3) // 8
        stop = segment.shape[0] * (3 if rank == 0 else 8) // 8
        pieces.append(segment[start:stop])
    return mx.contiguous(mx.concatenate(pieces, axis=0))


def _load(model: Path, layer: int, rank: int):
    index = json.loads(
        (model / "model.safetensors.index.json").read_text()
    )["weight_map"]
    names = (
        f"layers.{layer}.attn.wq_b.weight",
        f"layers.{layer}.attn.wq_b.scales",
        f"layers.{layer}.attn.indexer.wq_b.weight",
        f"layers.{layer}.attn.indexer.wq_b.scales",
    )
    tensors = {}
    for shard in sorted({index[name] for name in names}):
        tensors.update(mx.load(str(model / shard)))
    main_weight = _weighted_heads(tensors[names[0]], rank)
    main_scales = _weighted_heads(tensors[names[1]], rank)
    index_weight = tensors[names[2]]
    index_scales = tensors[names[3]]
    mx.eval(main_weight, main_scales, index_weight, index_scales)
    mx.synchronize()
    return main_weight, main_scales, index_weight, index_scales


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
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=8)
    args = parser.parse_args()

    main_w, main_s, index_w, index_s = _load(args.model, args.layer, args.rank)
    main_width = int(main_w.shape[0])
    combined_w = mx.concatenate([main_w, index_w], axis=0)
    combined_s = mx.concatenate([main_s, index_s], axis=0)
    mx.random.seed(12000 + args.rank)
    x = mx.random.normal((1, args.tokens, 1024)).astype(mx.bfloat16)
    mx.eval(x, combined_w, combined_s)
    mx.synchronize()

    kwargs = {
        "transpose": True,
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }

    def baseline():
        return (
            mx.quantized_matmul(x, main_w, scales=main_s, **kwargs),
            mx.quantized_matmul(x, index_w, scales=index_s, **kwargs),
        )

    def candidate():
        output = mx.quantized_matmul(x, combined_w, scales=combined_s, **kwargs)
        return output[..., :main_width], output[..., main_width:]

    expected, actual = baseline(), candidate()
    mx.eval(*expected, *actual)
    mx.synchronize()
    exact = all(
        bool(mx.array_equal(reference, value).item())
        for reference, value in zip(expected, actual)
    )

    def evaluate(function):
        value = function()
        mx.eval(*value)
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

    candidate_stats = _summary(timings["candidate"])
    baseline_stats = _summary(timings["baseline"])
    print(
        json.dumps(
            {
                "model": str(args.model),
                "layer": args.layer,
                "rank": args.rank,
                "tokens": args.tokens,
                "main_width": main_width,
                "indexer_width": int(index_w.shape[0]),
                "exact": exact,
                "combined_bytes": combined_w.nbytes + combined_s.nbytes,
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
