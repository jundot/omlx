#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate one wider stock/NAX gather-QMM against separate DS4 gate/up calls."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from bench_ds4_tp_prefill_moe_asymmetric import evaluate, load_tp_layer


def _summary(values: list[float]) -> dict[str, float]:
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

    layer = load_tp_layer(args.model, args.layer, args.rank)
    width = int(layer["width"])
    mx.random.seed(9100 + args.rank)
    x = mx.random.normal((1, args.tokens, 4096)).astype(mx.bfloat16)
    offsets = (0, 37, 79, 131, 181, 233)
    routes = mx.array(
        [
            [
                [(token * 73 + offset) % 256 for offset in offsets]
                for token in range(args.tokens)
            ]
        ],
        dtype=mx.uint32,
    )
    flat = routes.flatten()
    order = mx.argsort(flat)
    sorted_ids = flat[order]
    sorted_x = mx.expand_dims(x, (-2, -3)).flatten(0, -3)[order // 6]
    sorted_x = sorted_x.astype(mx.float16)

    # Output-axis concatenation changes no dot-product order. A production
    # loader can replace the two original banks with this backing allocation
    # and expose gate/up views, avoiding persistent duplicate checkpoint data.
    combined_weight = mx.concatenate(
        [layer["up_weight"], layer["gate_weight"]], axis=1
    )
    combined_scales = mx.concatenate(
        [layer["up_scales"], layer["gate_scales"]], axis=1
    )
    mx.eval(sorted_x, sorted_ids, combined_weight, combined_scales)
    mx.synchronize()

    kwargs = {
        "transpose": True,
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
        "sorted_indices": True,
    }

    def baseline():
        return (
            mx.gather_qmm(
                sorted_x,
                layer["up_weight"],
                layer["up_scales"],
                None,
                rhs_indices=sorted_ids,
                **kwargs,
            ),
            mx.gather_qmm(
                sorted_x,
                layer["gate_weight"],
                layer["gate_scales"],
                None,
                rhs_indices=sorted_ids,
                **kwargs,
            ),
        )

    def candidate():
        pair = mx.gather_qmm(
            sorted_x,
            combined_weight,
            combined_scales,
            None,
            rhs_indices=sorted_ids,
            **kwargs,
        )
        return pair[..., :width], pair[..., width:]

    expected = baseline()
    actual = candidate()
    evaluate((*expected, *actual))
    pair_exact = all(
        bool(mx.array_equal(reference, value).item())
        for reference, value in zip(expected, actual)
    )

    def activate(pair):
        up, gate = pair
        gate = mx.minimum(gate, 10.0)
        return (gate * mx.sigmoid(gate)) * mx.clip(up, -10.0, 10.0)

    expected_activation, actual_activation = activate(expected), activate(actual)
    mx.eval(expected_activation, actual_activation)
    activation_exact = bool(
        mx.array_equal(expected_activation, actual_activation).item()
    )

    for _ in range(args.warmup):
        evaluate(candidate())
        evaluate(baseline())
    timings = {"candidate": [], "baseline": []}
    for _ in range(args.cycles):
        for name, function in (
            ("candidate", candidate),
            ("baseline", baseline),
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            started = time.perf_counter_ns()
            evaluate(function())
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
                "width": width,
                "pair_exact": pair_exact,
                "activation_exact": activation_exact,
                "combined_bytes": combined_weight.nbytes + combined_scales.nbytes,
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
