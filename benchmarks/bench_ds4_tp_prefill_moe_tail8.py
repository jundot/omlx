#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analytical and future GPU gate for the DS4 BM32/BM8 tail-cull probe.

The current live server owns the M3 GPU, so the default mode performs no MLX
or model work. It quantifies the exact padded-MMA opportunity and the measured
phase-A break-even point. A later native agent can expose the frozen symbols
and run ``--model`` only after the public model is unloaded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


PAIR_SYMBOL = "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8"
DOWN_SYMBOL = "deepseek_mxfp4_gather_qmm_blocks_tail8"
MEASURED_M3_SHARED_X_MS = 8.209271
MEASURED_M3_PAIR_CONCAT_MS = 8.017229


@dataclass(frozen=True)
class Shape:
    tokens: int = 1024
    topk: int = 6
    experts: int = 256
    hidden: int = 4096
    local_intermediate: int = 1024
    bm: int = 32
    micro_bm: int = 8

    @property
    def routes(self) -> int:
        return self.tokens * self.topk


def current_mma_rows(rows: int, bm: int = 32) -> int:
    if rows < 0:
        raise ValueError("rows must be non-negative")
    return 0 if rows == 0 else math.ceil(rows / bm) * bm


def tail8_mma_rows(rows: int, bm: int = 32, micro_bm: int = 8) -> int:
    if rows < 0 or bm % micro_bm:
        raise ValueError("invalid rows or microtile")
    total = 0
    while rows:
        block_rows = min(rows, bm)
        total += math.ceil(block_rows / micro_bm) * micro_bm
        rows -= block_rows
    return total


def poisson_padding(mean_routes: float = 24.0, max_rows: int = 160) -> dict[str, float]:
    if mean_routes <= 0:
        raise ValueError("mean_routes must be positive")
    probability = math.exp(-mean_routes)
    current = 0.0
    tail8 = 0.0
    routes = 0.0
    mass = probability
    for rows in range(1, max_rows + 1):
        probability *= mean_routes / rows
        mass += probability
        current += probability * current_mma_rows(rows)
        tail8 += probability * tail8_mma_rows(rows)
        routes += probability * rows
    if mass < 0.999999:
        raise ValueError("Poisson truncation lost material probability")
    return {
        "expected_routes": routes,
        "current_mma_rows": current,
        "tail8_mma_rows": tail8,
        "mma_row_reduction": 1.0 - tail8 / current,
        "mma_only_speedup": current / tail8,
    }


def projected_time_ms(
    measured_ms: float, mma_fraction: float, mma_row_reduction: float
) -> float:
    if measured_ms <= 0 or not 0 <= mma_fraction <= 1:
        raise ValueError("invalid measured time or MMA fraction")
    if not 0 <= mma_row_reduction < 1:
        raise ValueError("invalid row reduction")
    return measured_ms * (1.0 - mma_fraction * mma_row_reduction)


def required_mma_fraction(
    measured_ms: float, target_ms: float, mma_row_reduction: float
) -> float:
    if target_ms <= 0 or target_ms > measured_ms:
        raise ValueError("target must be in (0, measured]")
    return (1.0 - target_ms / measured_ms) / mma_row_reduction


def persistent_fusion_blocker(shape: Shape) -> dict[str, object]:
    max_threadgroup = 32 * 1024
    bk_padded = 32 + 8
    weight_staging = 2 * 32 * bk_padded * 2
    rows = (8, 16, 24, 32)
    candidates = []
    for bm in rows:
        mid = bm * shape.local_intermediate * 2
        x_staging = bm * bk_padded * 2
        total = mid + x_staging + weight_staging
        candidates.append(
            {
                "bm": bm,
                "mid_bytes": mid,
                "minimum_pair_staging_bytes": x_staging + weight_staging,
                "minimum_total_bytes": total,
                "fits_32k": total <= max_threadgroup,
                "weight_read_amplification_at_24_routes": (
                    math.ceil(24 / bm) if total <= max_threadgroup else None
                ),
            }
        )
    return {
        "m3_max_threadgroup_bytes": max_threadgroup,
        "candidates": candidates,
        "conclusion": (
            "BM16+ cannot retain the FP16 mid with mandatory pair staging; "
            "BM8 fits but rereads every expert projection 3x at 24 routes."
        ),
    }


def analysis_report(shape: Shape) -> dict[str, object]:
    uniform_current = shape.experts * current_mma_rows(24)
    uniform_tail8 = shape.experts * tail8_mma_rows(24)
    uniform_reduction = 1.0 - uniform_tail8 / uniform_current
    poisson = poisson_padding(24.0)
    pair_target_ms = MEASURED_M3_PAIR_CONCAT_MS
    promotion_target_ms = MEASURED_M3_PAIR_CONCAT_MS / 1.05
    sensitivity = {}
    for fraction in (0.1, 0.2, 0.3, 0.4, 0.5):
        candidate_ms = projected_time_ms(
            MEASURED_M3_SHARED_X_MS, fraction, uniform_reduction
        )
        sensitivity[str(fraction)] = {
            "projected_ms_uniform_24": candidate_ms,
            "speedup_vs_pair_concat": MEASURED_M3_PAIR_CONCAT_MS / candidate_ms,
        }
    return {
        "shape": asdict(shape),
        "symbols": {"pair": PAIR_SYMBOL, "down": DOWN_SYMBOL},
        "live_gpu_benchmark_safe": False,
        "uniform_24_route_fixture": {
            "current_mma_rows": uniform_current,
            "tail8_mma_rows": uniform_tail8,
            "mma_row_reduction": uniform_reduction,
            "mma_only_speedup": uniform_current / uniform_tail8,
            "block_count_change": 0,
            "weight_read_amplification": 1.0,
        },
        "poisson_mean_24_sensitivity": poisson,
        "measured_phase_a_ms": {
            "shared_x_bm32": MEASURED_M3_SHARED_X_MS,
            "pair_concat": MEASURED_M3_PAIR_CONCAT_MS,
        },
        "required_mma_fraction_uniform": {
            "break_even_vs_pair_concat": required_mma_fraction(
                MEASURED_M3_SHARED_X_MS, pair_target_ms, uniform_reduction
            ),
            "pass_1_05x_gate": required_mma_fraction(
                MEASURED_M3_SHARED_X_MS,
                promotion_target_ms,
                uniform_reduction,
            ),
        },
        "required_mma_fraction_poisson": {
            "break_even_vs_pair_concat": required_mma_fraction(
                MEASURED_M3_SHARED_X_MS,
                pair_target_ms,
                poisson["mma_row_reduction"],
            ),
            "pass_1_05x_gate": required_mma_fraction(
                MEASURED_M3_SHARED_X_MS,
                promotion_target_ms,
                poisson["mma_row_reduction"],
            ),
        },
        "uniform_sensitivity": sensitivity,
        "persistent_fusion_blocker": persistent_fusion_blocker(shape),
        "speed_claimed": False,
    }


def _evaluate(mx, value) -> None:
    mx.eval(value)
    mx.synchronize()


def _abba(
    mx,
    candidate: Callable[[], object],
    baseline: Callable[[], object],
    warmup: int,
    cycles: int,
) -> dict[str, list[float]]:
    for _ in range(warmup):
        _evaluate(mx, candidate())
        _evaluate(mx, baseline())
    timings = {"candidate": [], "baseline": []}
    for _ in range(cycles):
        for name, function in (
            ("candidate", candidate),
            ("baseline", baseline),
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            started = time.perf_counter_ns()
            _evaluate(mx, function())
            timings[name].append((time.perf_counter_ns() - started) / 1e6)
    return timings


def _summary(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def run_gpu_gate(args: argparse.Namespace, shape: Shape) -> dict[str, object]:
    # Intentionally lazy: analysis mode does not initialize Metal while the
    # public TP2 worker owns the M3 GPU.
    import mlx.core as mx

    from bench_ds4_tp_prefill_moe import load_tp_layer
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.deepseek_v4.switch_layers import (
        _block_config,
        _build_mxfp4_blocks,
    )

    pair_candidate = getattr(fast, PAIR_SYMBOL, None)
    down_candidate = getattr(fast, DOWN_SYMBOL, None)
    if pair_candidate is None or down_candidate is None:
        raise RuntimeError("tail8 native symbols are not wired")
    if not fast.has_symbol(PAIR_SYMBOL) or not fast.has_symbol(DOWN_SYMBOL):
        raise RuntimeError("tail8 native symbols are unavailable")

    (
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        shards,
    ) = load_tp_layer(args.model, args.layer, args.rank)
    mx.random.seed(1000 + shape.tokens)
    x = mx.random.normal((1, shape.tokens, shape.hidden)).astype(mx.bfloat16)
    offsets = (0, 37, 79, 131, 181, 233)
    routes = mx.array(
        [
            [
                [(token * 73 + offset) % shape.experts for offset in offsets]
                for token in range(shape.tokens)
            ]
        ],
        dtype=mx.uint32,
    )
    flat_routes = routes.flatten()
    order = mx.argsort(flat_routes)
    sorted_ids = flat_routes[order]
    sorted_x = mx.expand_dims(x, (-2, -3)).flatten(0, -3)[order // shape.topk]
    sorted_x = sorted_x.astype(mx.float16)
    bm, variant = _block_config(shape.routes, "mxfp4")
    block_meta, block_count = _build_mxfp4_blocks(
        sorted_ids, shape.experts, bm
    )

    def activate(up, gate):
        gate = mx.minimum(gate, 10.0)
        return (gate * mx.sigmoid(gate)) * mx.clip(up, -10.0, 10.0)

    def baseline_pair():
        pair = fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
            sorted_x,
            up_weight,
            up_scales,
            gate_weight,
            gate_scales,
            block_meta,
            block_count,
            variant,
        )
        return activate(
            pair[..., : shape.local_intermediate],
            pair[..., shape.local_intermediate :],
        )

    def candidate_pair():
        return pair_candidate(
            sorted_x,
            up_weight,
            up_scales,
            gate_weight,
            gate_scales,
            block_meta,
            block_count,
            10.0,
            variant,
        )

    baseline_value = baseline_pair()
    candidate_value = candidate_pair()
    mx.eval(baseline_value, candidate_value)
    mx.synchronize()
    pair_exact = bool(mx.array_equal(candidate_value, baseline_value).item())
    pair_timings = _abba(
        mx, candidate_pair, baseline_pair, args.warmup, args.cycles
    )
    pair_candidate_stats = _summary(pair_timings["candidate"])
    pair_baseline_stats = _summary(pair_timings["baseline"])
    pair_speedup = (
        pair_baseline_stats["median_ms"] / pair_candidate_stats["median_ms"]
    )

    def baseline_down():
        return fast.deepseek_mxfp4_gather_qmm_blocks(
            baseline_value,
            down_weight,
            down_scales,
            block_meta,
            block_count,
            variant,
        )

    def candidate_down():
        return down_candidate(
            baseline_value,
            down_weight,
            down_scales,
            block_meta,
            block_count,
            variant,
        )

    baseline_down_value = baseline_down()
    candidate_down_value = candidate_down()
    mx.eval(baseline_down_value, candidate_down_value)
    mx.synchronize()
    down_exact = bool(
        mx.array_equal(candidate_down_value, baseline_down_value).item()
    )
    down_timings = _abba(
        mx, candidate_down, baseline_down, args.warmup, args.cycles
    )
    down_candidate_stats = _summary(down_timings["candidate"])
    down_baseline_stats = _summary(down_timings["baseline"])
    down_speedup = (
        down_baseline_stats["median_ms"] / down_candidate_stats["median_ms"]
    )

    def baseline_full():
        return fast.deepseek_mxfp4_gather_qmm_blocks(
            baseline_pair(),
            down_weight,
            down_scales,
            block_meta,
            block_count,
            variant,
        )

    def candidate_full():
        return down_candidate(
            candidate_pair(),
            down_weight,
            down_scales,
            block_meta,
            block_count,
            variant,
        )

    baseline_full_value = baseline_full()
    candidate_full_value = candidate_full()
    mx.eval(baseline_full_value, candidate_full_value)
    mx.synchronize()
    full_exact = bool(
        mx.array_equal(candidate_full_value, baseline_full_value).item()
    )
    full_timings = _abba(
        mx, candidate_full, baseline_full, args.warmup, args.cycles
    )
    full_candidate_stats = _summary(full_timings["candidate"])
    full_baseline_stats = _summary(full_timings["baseline"])
    full_speedup = (
        full_baseline_stats["median_ms"] / full_candidate_stats["median_ms"]
    )
    passed = (
        pair_exact
        and down_exact
        and full_exact
        and full_speedup >= args.min_speedup
    )
    if args.strict and not passed:
        raise SystemExit(2)
    return {
        "device": "m3-ultra",
        "model": str(args.model),
        "layer": args.layer,
        "rank": args.rank,
        "shards": shards,
        "pair": {
            "array_equal": pair_exact,
            "candidate": pair_candidate_stats,
            "baseline": pair_baseline_stats,
            "speedup": pair_speedup,
        },
        "down": {
            "array_equal": down_exact,
            "candidate": down_candidate_stats,
            "baseline": down_baseline_stats,
            "speedup": down_speedup,
        },
        "full_routed_projection": {
            "array_equal": full_exact,
            "candidate": full_candidate_stats,
            "baseline": full_baseline_stats,
            "speedup": full_speedup,
        },
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = Shape()
    report: dict[str, object] = {"analysis": analysis_report(shape)}
    if args.model is not None:
        report["gpu_gate"] = run_gpu_gate(args, shape)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
