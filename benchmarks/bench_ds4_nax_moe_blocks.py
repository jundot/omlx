#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate the isolated M5 NAX expert-blocked DS4 routed-MoE projection.

The promoted runtime remains fixed to signed 3:5 rank-1 geometry. This harness
also accepts an explicit two-rank vector so equal-width candidates can be
measured before changing that production gate. M=1024,
top-k=6, 256 experts and hidden width 4096 remain fixed.
It compares stock BF16 ``mx.gather_qmm`` with an expert-local BM32 NAX
primitive at every projection/activation/restore boundary. No serving model
imports or dispatches the candidate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from benchmarks.bench_ds4_tp_prefill_moe_asymmetric import load_tp_layer
from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v4.switch_layers import _build_mxfp4_blocks

TOKENS = 1024
TOPK = 6
ROUTES = TOKENS * TOPK
EXPERTS = 256
HIDDEN = 4096
LOCAL_INTERMEDIATE = 1280
STOCK_BM = 64
CANDIDATE_BM = 32
MIN_COMPOSED_SPEEDUP = 1.45
MIN_PAIR_SPEEDUP_VS_SEPARATE_NAX = 1.02
SYMBOL = "deepseek_mxfp4_gather_qmm_blocks_nax"
PAIR_SYMBOL = "deepseek_mxfp4_gather_qmm_pair_blocks_nax"


@mx.compile
def _limited_swiglu(gate: mx.array, up: mx.array, limit: float) -> mx.array:
    gate = mx.minimum(gate, limit)
    up = mx.clip(up, -limit, limit)
    return nn.silu(gate) * up


def synthetic_routes() -> tuple[int, ...]:
    offsets = (0, 37, 79, 131, 181, 233)
    return tuple(
        (token * 73 + offset) % EXPERTS
        for token in range(TOKENS)
        for offset in offsets
    )


def route_fixture(name: str) -> tuple[int, ...]:
    """Deterministic route distributions for boundary/skew safety gates."""

    if name == "synthetic":
        return synthetic_routes()
    if name == "ragged":
        counts = [0, 1, 31, 32, 33] + [25] * 23 + [24] * (EXPERTS - 28)
    elif name == "skewed":
        counts = [512] + [23] * 22 + [22] * (EXPERTS - 23)
    elif name == "max-blocks":
        # Near the fixed 448-block ABI ceiling: one very hot expert and one
        # route on every other expert produce 440 BM32 work items.
        counts = [ROUTES - (EXPERTS - 1)] + [1] * (EXPERTS - 1)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown route fixture: {name}")
    assert len(counts) == EXPERTS and sum(counts) == ROUTES
    return tuple(
        expert for expert, count in enumerate(counts) for _ in range(count)
    )


def structural_work_report(
    routes: Sequence[int] | None = None,
    *,
    stock_bm: int = STOCK_BM,
    candidate_bm: int = CANDIDATE_BM,
) -> dict[str, int | float]:
    """Count global-tile expert segments versus expert-local BM32 blocks."""

    route_ids = tuple(synthetic_routes() if routes is None else routes)
    sorted_experts = sorted(route_ids)
    stock_segments = 0
    for start in range(0, len(sorted_experts), stock_bm):
        stock_segments += len(set(sorted_experts[start : start + stock_bm]))
    counts = [0] * EXPERTS
    for expert in route_ids:
        counts[expert] += 1
    candidate_blocks = sum(
        (count + candidate_bm - 1) // candidate_bm for count in counts if count
    )
    stock_rows = stock_segments * stock_bm
    candidate_rows = candidate_blocks * candidate_bm
    return {
        "routes": len(route_ids),
        "active_experts": sum(count > 0 for count in counts),
        "min_routes_per_active_expert": min(count for count in counts if count),
        "max_routes_per_expert": max(counts),
        "stock_bm": stock_bm,
        "stock_global_tiles": (len(route_ids) + stock_bm - 1) // stock_bm,
        "stock_expert_segments": stock_segments,
        "stock_row_equivalents": stock_rows,
        "candidate_bm": candidate_bm,
        "candidate_expert_blocks": candidate_blocks,
        "candidate_row_equivalents": candidate_rows,
        "row_work_ratio": stock_rows / candidate_rows,
        "weight_segment_ratio": stock_segments / candidate_blocks,
    }


def _evaluate(value: Any) -> None:
    values = value if isinstance(value, (tuple, list)) else (value,)
    mx.eval(*values)
    mx.synchronize()


def _summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _abba(
    candidate: Callable[[], Any],
    baseline: Callable[[], Any],
    *,
    warmup: int,
    cycles: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _evaluate(candidate())
        _evaluate(baseline())
    samples: dict[str, list[float]] = {"candidate": [], "baseline": []}
    for _ in range(cycles):
        for label, function in (
            ("candidate", candidate),
            ("baseline", baseline),
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            started = time.perf_counter_ns()
            _evaluate(function())
            samples[label].append((time.perf_counter_ns() - started) / 1e6)
    candidate_stats = _summary(samples["candidate"])
    baseline_stats = _summary(samples["baseline"])
    return {
        "candidate": candidate_stats,
        "baseline": baseline_stats,
        "speedup": baseline_stats["median_ms"] / candidate_stats["median_ms"],
        "candidate_samples_ms": samples["candidate"],
        "baseline_samples_ms": samples["baseline"],
    }


def _parity(candidate: mx.array, baseline: mx.array) -> dict[str, Any]:
    _evaluate((candidate, baseline))
    exact = bool(mx.array_equal(candidate, baseline).item())
    if exact:
        max_abs = 0.0
        mismatches = 0
    else:
        delta = mx.abs(candidate.astype(mx.float32) - baseline.astype(mx.float32))
        max_abs = float(mx.max(delta).item())
        mismatches = int(mx.sum(candidate != baseline).item())
    return {
        "array_equal": exact,
        "mismatches": mismatches,
        "max_abs": max_abs,
        "elements": int(candidate.size),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--shard-weights",
        default="3,5",
        help="two comma-separated positive tensor shard weights",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--min-speedup", type=float, default=MIN_COMPOSED_SPEEDUP)
    parser.add_argument(
        "--route-pattern",
        choices=("synthetic", "ragged", "skewed", "max-blocks"),
        default="synthetic",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _shard_weights(value: str) -> tuple[int, int]:
    try:
        parts = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("--shard-weights must contain integers") from exc
    if len(parts) != 2 or any(item <= 0 for item in parts):
        raise ValueError("--shard-weights requires two positive integers")
    return parts


def main() -> None:
    args = _parse_args()
    shard_weights = _shard_weights(args.shard_weights)
    if not fast.has_symbol(SYMBOL):
        raise RuntimeError(f"native symbol {SYMBOL} is unavailable")
    if not fast.has_symbol(PAIR_SYMBOL):
        raise RuntimeError(f"native symbol {PAIR_SYMBOL} is unavailable")
    if not fast.ds4_projection_nax_kernels_built():
        raise RuntimeError("optional NAX metallib is unavailable")
    if not fast.ds4_projection_nax_device_available():
        raise RuntimeError("physical device does not support the NAX primitive")

    tensors = load_tp_layer(
        args.model,
        args.layer,
        rank=args.rank,
        shard_weights=shard_weights,
    )
    local_intermediate = tensors["width"]

    mx.random.seed(20260822)
    hidden = mx.random.normal((1, TOKENS, HIDDEN)).astype(mx.bfloat16)
    route_values = route_fixture(args.route_pattern)
    routes = mx.array(route_values, dtype=mx.uint32).reshape(1, TOKENS, TOPK)
    scores = mx.softmax(mx.random.normal(routes.shape), axis=-1).astype(mx.float32)
    flat_routes = routes.flatten()
    order = mx.argsort(flat_routes)
    inv_order = mx.argsort(order)
    sorted_experts = flat_routes[order]
    sorted_hidden = mx.expand_dims(hidden, (-2, -3)).flatten(0, -3)[
        order // TOPK
    ]
    block_meta, block_count = _build_mxfp4_blocks(
        sorted_experts, EXPERTS, CANDIDATE_BM
    )
    _evaluate(
        (
            sorted_hidden,
            sorted_experts,
            inv_order,
            scores,
            block_meta,
            block_count,
        )
    )

    stock_kwargs = {
        "transpose": True,
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
        "sorted_indices": True,
    }

    def stock_projection(x: mx.array, weight: mx.array, scales: mx.array):
        return mx.gather_qmm(
            x,
            weight,
            scales,
            None,
            rhs_indices=sorted_experts,
            **stock_kwargs,
        )

    def candidate_projection(x: mx.array, weight: mx.array, scales: mx.array):
        return fast.deepseek_mxfp4_gather_qmm_blocks_nax(
            x, weight, scales, block_meta, block_count
        )

    def stock_pair():
        return (
            stock_projection(
                sorted_hidden, tensors["up_weight"], tensors["up_scales"]
            ),
            stock_projection(
                sorted_hidden, tensors["gate_weight"], tensors["gate_scales"]
            ),
        )

    def candidate_pair():
        packed = fast.deepseek_mxfp4_gather_qmm_pair_blocks_nax(
            sorted_hidden,
            tensors["up_weight"],
            tensors["up_scales"],
            tensors["gate_weight"],
            tensors["gate_scales"],
            block_meta,
            block_count,
        )
        return packed[0], packed[1]

    def separate_nax_pair():
        return (
            candidate_projection(
                sorted_hidden, tensors["up_weight"], tensors["up_scales"]
            ),
            candidate_projection(
                sorted_hidden, tensors["gate_weight"], tensors["gate_scales"]
            ),
        )

    def activate(pair: tuple[mx.array, mx.array]) -> mx.array:
        up, gate = pair
        return _limited_swiglu(gate, up, 10.0)

    def stock_activated():
        return activate(stock_pair())

    def candidate_activated():
        return activate(candidate_pair())

    stock_mid = stock_activated()
    candidate_mid = candidate_activated()
    _evaluate((stock_mid, candidate_mid))

    def stock_down_fixed():
        return stock_projection(
            stock_mid, tensors["down_weight"], tensors["down_scales"]
        )

    def candidate_down_fixed():
        return candidate_projection(
            stock_mid, tensors["down_weight"], tensors["down_scales"]
        )

    def stock_composed():
        return stock_projection(
            stock_activated(), tensors["down_weight"], tensors["down_scales"]
        )

    def candidate_composed():
        return candidate_projection(
            candidate_activated(),
            tensors["down_weight"],
            tensors["down_scales"],
        )

    stock_up, stock_gate = stock_pair()
    candidate_up, candidate_gate = candidate_pair()
    stock_down = stock_down_fixed()
    candidate_down = candidate_down_fixed()
    stock_full = stock_composed()
    candidate_full = candidate_composed()

    def restore(value: mx.array) -> mx.array:
        return value[inv_order].reshape(1, TOKENS, TOPK, 1, HIDDEN).squeeze(-2)

    stock_routes = restore(stock_full)
    candidate_routes = restore(candidate_full)
    stock_reduced = (
        stock_routes * scores[..., None].astype(stock_routes.dtype)
    ).sum(-2)
    candidate_reduced = (
        candidate_routes * scores[..., None].astype(candidate_routes.dtype)
    ).sum(-2)

    parity = {
        "up_bf16": _parity(candidate_up, stock_up),
        "gate_bf16": _parity(candidate_gate, stock_gate),
        "limited_swiglu_bf16": _parity(candidate_mid, stock_mid),
        "down_fixed_input_bf16": _parity(candidate_down, stock_down),
        "composed_sorted_down_bf16": _parity(candidate_full, stock_full),
        "restored_route_rows_bf16": _parity(candidate_routes, stock_routes),
        "score_reduced_local_output_bf16": _parity(
            candidate_reduced, stock_reduced
        ),
    }
    all_exact = all(boundary["array_equal"] for boundary in parity.values())

    timings = {
        "paired_vs_separate_nax": _abba(
            candidate_pair,
            separate_nax_pair,
            warmup=args.warmup,
            cycles=args.cycles,
        ),
        "pair": _abba(
            candidate_pair,
            stock_pair,
            warmup=args.warmup,
            cycles=args.cycles,
        ),
        "pair_plus_limited_swiglu": _abba(
            candidate_activated,
            stock_activated,
            warmup=args.warmup,
            cycles=args.cycles,
        ),
        "down_fixed_input": _abba(
            candidate_down_fixed,
            stock_down_fixed,
            warmup=args.warmup,
            cycles=args.cycles,
        ),
        "composed_routed_projection": _abba(
            candidate_composed,
            stock_composed,
            warmup=args.warmup,
            cycles=args.cycles,
        ),
    }
    composed_speedup = timings["composed_routed_projection"]["speedup"]
    pair_speedup = timings["paired_vs_separate_nax"]["speedup"]
    passed = (
        all_exact
        and composed_speedup >= args.min_speedup
        and pair_speedup >= MIN_PAIR_SPEEDUP_VS_SEPARATE_NAX
    )
    report = {
        "device": mx.device_info(),
        "model": str(args.model),
        "layer": args.layer,
        "rank": args.rank,
        "route_pattern": args.route_pattern,
        "shard_weights": list(shard_weights),
        "shape": {
            "tokens": TOKENS,
            "routes": ROUTES,
            "experts": EXPERTS,
            "hidden": HIDDEN,
            "local_intermediate": local_intermediate,
            "up_gate_weight": list(tensors["up_weight"].shape),
            "down_weight": list(tensors["down_weight"].shape),
            "block_meta": list(block_meta.shape),
        },
        "kernel": {
            "symbol": PAIR_SYMBOL,
            "bm": 32,
            "bn": 64,
            "bk": 64,
            "wm": 1,
            "wn": 2,
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
        },
        "structural_work": structural_work_report(route_values),
        "parity": parity,
        "all_boundaries_array_equal": all_exact,
        "timings": timings,
        "gate": {
            "minimum_composed_speedup": args.min_speedup,
            "composed_speedup": composed_speedup,
            "minimum_pair_speedup_vs_separate_nax": (
                MIN_PAIR_SPEEDUP_VS_SEPARATE_NAX
            ),
            "pair_speedup_vs_separate_nax": pair_speedup,
            "passed": passed,
            "production_dispatch": False,
        },
        "checkpoint_shards": tensors["shards"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
