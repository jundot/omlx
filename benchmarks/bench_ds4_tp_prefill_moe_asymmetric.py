#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-layer DS4 M=1024 routed-MoE benchmark for exact 3:5 TP slices.

Rank 0 owns 768 intermediate rows (3/8) and rank 1 owns 1280 (5/8).
Gate/up slice output rows; down slices the matching packed input and E8M0
scale groups. The harness compares stock gather-QMM with every existing
BM/BN native block variant and records exactness at pair, down, and composed
projection boundaries.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v4.switch_layers import _build_mxfp4_blocks


VARIANTS = {
    0: (8, 32),
    1: (16, 32),
    2: (32, 32),
    3: (16, 64),
    4: (32, 64),
}
SHARD_WEIGHTS = (3, 5)
FULL_INTERMEDIATE = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=4)
    return parser.parse_args()


def shard_bounds(
    rank: int,
    shard_weights: tuple[int, ...] = SHARD_WEIGHTS,
) -> tuple[int, int]:
    if not shard_weights or not 0 <= rank < len(shard_weights):
        raise ValueError("rank is outside the explicit TP shard vector")
    if any(value <= 0 for value in shard_weights):
        raise ValueError("TP shard weights must be positive")
    total = sum(shard_weights)
    if FULL_INTERMEDIATE % total:
        raise ValueError("TP shard weights do not divide the intermediate width")
    unit = FULL_INTERMEDIATE // total
    start = unit * sum(shard_weights[:rank])
    stop = start + unit * shard_weights[rank]
    return start, stop


def load_tp_layer(
    model_dir: Path,
    layer: int,
    rank: int,
    shard_weights: tuple[int, ...] = SHARD_WEIGHTS,
):
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    prefix = f"layers.{layer}.ffn.experts."
    shards = sorted(
        {
            shard
            for key, shard in index["weight_map"].items()
            if key.startswith(prefix)
        }
    )
    tensors = {}
    for shard in shards:
        tensors.update(mx.load(str(model_dir / shard)))

    start, stop = shard_bounds(rank, shard_weights)
    width = stop - start
    packed_start, packed_stop = start // 8, stop // 8
    scale_start, scale_stop = start // 32, stop // 32

    def stack(projection: str, suffix: str, indexer):
        return mx.stack(
            [
                tensors[f"{prefix}{expert}.{projection}.{suffix}"][indexer]
                for expert in range(256)
            ]
        )

    gate_weight = stack("w1", "weight", (slice(start, stop), slice(None)))
    gate_scales = stack("w1", "scales", (slice(start, stop), slice(None)))
    up_weight = stack("w3", "weight", (slice(start, stop), slice(None)))
    up_scales = stack("w3", "scales", (slice(start, stop), slice(None)))
    down_weight = stack(
        "w2", "weight", (slice(None), slice(packed_start, packed_stop))
    )
    down_scales = stack(
        "w2", "scales", (slice(None), slice(scale_start, scale_stop))
    )
    mx.eval(
        gate_weight,
        gate_scales,
        up_weight,
        up_scales,
        down_weight,
        down_scales,
    )
    mx.synchronize()
    return {
        "up_weight": up_weight,
        "up_scales": up_scales,
        "gate_weight": gate_weight,
        "gate_scales": gate_scales,
        "down_weight": down_weight,
        "down_scales": down_scales,
        "start": start,
        "stop": stop,
        "width": width,
        "shards": shards,
    }


def evaluate(value) -> None:
    mx.eval(*(value if isinstance(value, tuple) else (value,)))
    mx.synchronize()


def abba(candidate, baseline, warmup: int, cycles: int):
    for _ in range(warmup):
        evaluate(candidate())
        evaluate(baseline())
    timings = {"candidate": [], "baseline": []}
    for _ in range(cycles):
        for name, function in (
            ("candidate", candidate),
            ("baseline", baseline),
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            started = time.perf_counter_ns()
            evaluate(function())
            timings[name].append((time.perf_counter_ns() - started) / 1e6)
    return timings


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def compare(candidate, baseline, warmup: int, cycles: int) -> dict[str, object]:
    candidate_value = candidate()
    baseline_value = baseline()
    evaluate(candidate_value)
    evaluate(baseline_value)
    if isinstance(candidate_value, tuple):
        exact = all(
            bool(mx.array_equal(left, right).item())
            for left, right in zip(candidate_value, baseline_value)
        )
    else:
        exact = bool(mx.array_equal(candidate_value, baseline_value).item())
    timings = abba(candidate, baseline, warmup, cycles)
    candidate_stats = summary(timings["candidate"])
    baseline_stats = summary(timings["baseline"])
    return {
        "exact": exact,
        "candidate": candidate_stats,
        "baseline": baseline_stats,
        "speedup": baseline_stats["median_ms"] / candidate_stats["median_ms"],
    }


def main() -> None:
    args = parse_args()
    tensors = load_tp_layer(args.model, args.layer, args.rank)
    width = tensors["width"]
    hidden = 4096
    experts = 256

    mx.random.seed(1000 + args.tokens)
    x = mx.random.normal((1, args.tokens, hidden)).astype(mx.bfloat16)
    offsets = (0, 37, 79, 131, 181, 233)
    routes = mx.array(
        [
            [
                [(token * 73 + offset) % experts for offset in offsets]
                for token in range(args.tokens)
            ]
        ],
        dtype=mx.uint32,
    )
    flat_routes = routes.flatten()
    order = mx.argsort(flat_routes)
    sorted_ids = flat_routes[order]
    sorted_x = mx.expand_dims(x, (-2, -3)).flatten(0, -3)[order // 6]
    sorted_x = sorted_x.astype(mx.float16)
    block_plans = {}
    for bm in sorted({config[0] for config in VARIANTS.values()}):
        block_plans[bm] = _build_mxfp4_blocks(sorted_ids, experts, bm)
    block_arrays = sum((list(plan) for plan in block_plans.values()), [])
    mx.eval(sorted_x, sorted_ids, *block_arrays)
    mx.synchronize()

    stock_kwargs = {
        "transpose": True,
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
        "sorted_indices": True,
    }

    def stock_pair_tuple():
        return (
            mx.gather_qmm(
                sorted_x,
                tensors["up_weight"],
                tensors["up_scales"],
                None,
                rhs_indices=sorted_ids,
                **stock_kwargs,
            ),
            mx.gather_qmm(
                sorted_x,
                tensors["gate_weight"],
                tensors["gate_scales"],
                None,
                rhs_indices=sorted_ids,
                **stock_kwargs,
            ),
        )

    def stock_pair_concat():
        return mx.concatenate(stock_pair_tuple(), axis=-1)

    stock_up, stock_gate = stock_pair_tuple()
    gate_limited = mx.minimum(stock_gate, 10.0)
    activated = (gate_limited * mx.sigmoid(gate_limited)) * mx.clip(
        stock_up, -10.0, 10.0
    )
    mx.eval(activated)
    mx.synchronize()

    def stock_down():
        return mx.gather_qmm(
            activated,
            tensors["down_weight"],
            tensors["down_scales"],
            None,
            rhs_indices=sorted_ids,
            **stock_kwargs,
        )

    def stock_full():
        up, gate = stock_pair_tuple()
        gate = mx.minimum(gate, 10.0)
        mid = (gate * mx.sigmoid(gate)) * mx.clip(up, -10.0, 10.0)
        return mx.gather_qmm(
            mid,
            tensors["down_weight"],
            tensors["down_scales"],
            None,
            rhs_indices=sorted_ids,
            **stock_kwargs,
        )

    results = []
    for variant, (bm, bn) in VARIANTS.items():
        block_meta, block_count = block_plans[bm]

        def pair_concat(v=variant, meta=block_meta, count=block_count):
            return fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
                sorted_x,
                tensors["up_weight"],
                tensors["up_scales"],
                tensors["gate_weight"],
                tensors["gate_scales"],
                meta,
                count,
                v,
            )

        def pair_tuple(v=variant, meta=block_meta, count=block_count):
            pair = fast.deepseek_mxfp4_gather_qmm_pair_blocks(
                sorted_x,
                tensors["up_weight"],
                tensors["up_scales"],
                tensors["gate_weight"],
                tensors["gate_scales"],
                meta,
                count,
                v,
            )
            return pair[0], pair[1]

        def down(v=variant, meta=block_meta, count=block_count):
            return fast.deepseek_mxfp4_gather_qmm_blocks(
                activated,
                tensors["down_weight"],
                tensors["down_scales"],
                meta,
                count,
                v,
            )

        def full(v=variant, meta=block_meta, count=block_count):
            pair = fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
                sorted_x,
                tensors["up_weight"],
                tensors["up_scales"],
                tensors["gate_weight"],
                tensors["gate_scales"],
                meta,
                count,
                v,
            )
            up = pair[..., :width]
            gate = mx.minimum(pair[..., width:], 10.0)
            mid = (gate * mx.sigmoid(gate)) * mx.clip(up, -10.0, 10.0)
            return fast.deepseek_mxfp4_gather_qmm_blocks(
                mid,
                tensors["down_weight"],
                tensors["down_scales"],
                meta,
                count,
                v,
            )

        results.append(
            {
                "variant": variant,
                "bm": bm,
                "bn": bn,
                "pair_concat": compare(
                    pair_concat, stock_pair_concat, args.warmup, args.cycles
                ),
                "pair_separate_output": compare(
                    pair_tuple, stock_pair_tuple, args.warmup, args.cycles
                ),
                "down": compare(down, stock_down, args.warmup, args.cycles),
                "full": compare(full, stock_full, args.warmup, args.cycles),
            }
        )

    best_pair = min(
        results, key=lambda row: row["pair_concat"]["candidate"]["median_ms"]
    )
    best_down = min(
        results, key=lambda row: row["down"]["candidate"]["median_ms"]
    )
    tail8_candidate = None
    if fast.has_symbol(
        "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8"
    ) and fast.has_symbol("deepseek_mxfp4_gather_qmm_blocks_tail8"):
        block_meta, block_count = block_plans[32]

        def best_pair_activated():
            pair = fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
                sorted_x,
                tensors["up_weight"],
                tensors["up_scales"],
                tensors["gate_weight"],
                tensors["gate_scales"],
                block_meta,
                block_count,
                2,
            )
            up = pair[..., :width]
            gate = mx.minimum(pair[..., width:], 10.0)
            return (gate * mx.sigmoid(gate)) * mx.clip(up, -10.0, 10.0)

        def tail8_pair():
            return fast.deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8(
                sorted_x,
                tensors["up_weight"],
                tensors["up_scales"],
                tensors["gate_weight"],
                tensors["gate_scales"],
                block_meta,
                block_count,
                10.0,
                2,
            )

        best_activated = best_pair_activated()
        evaluate(best_activated)

        def best_block_down():
            return fast.deepseek_mxfp4_gather_qmm_blocks(
                best_activated,
                tensors["down_weight"],
                tensors["down_scales"],
                block_meta,
                block_count,
                2,
            )

        def tail8_down():
            return fast.deepseek_mxfp4_gather_qmm_blocks_tail8(
                best_activated,
                tensors["down_weight"],
                tensors["down_scales"],
                block_meta,
                block_count,
                2,
            )

        def best_block_full():
            return fast.deepseek_mxfp4_gather_qmm_blocks(
                best_pair_activated(),
                tensors["down_weight"],
                tensors["down_scales"],
                block_meta,
                block_count,
                2,
            )

        def tail8_full():
            return fast.deepseek_mxfp4_gather_qmm_blocks_tail8(
                tail8_pair(),
                tensors["down_weight"],
                tensors["down_scales"],
                block_meta,
                block_count,
                2,
            )

        tail8_candidate = {
            "baseline": "BM32/BN32 variant 2",
            "pair_activated": compare(
                tail8_pair, best_pair_activated, args.warmup, args.cycles
            ),
            "down": compare(
                tail8_down, best_block_down, args.warmup, args.cycles
            ),
            "full": compare(
                tail8_full, best_block_full, args.warmup, args.cycles
            ),
        }
        tail8_candidate["passes_1_10"] = bool(
            tail8_candidate["pair_activated"]["exact"]
            and tail8_candidate["down"]["exact"]
            and tail8_candidate["full"]["exact"]
            and tail8_candidate["full"]["speedup"] >= 1.10
        )
    print(
        json.dumps(
            {
                "model": str(args.model),
                "layer": args.layer,
                "rank": args.rank,
                "shard_weights": SHARD_WEIGHTS,
                "slice": {
                    "start": tensors["start"],
                    "stop": tensors["stop"],
                    "width": width,
                    "down_packed_width": tensors["down_weight"].shape[-1],
                    "down_scale_width": tensors["down_scales"].shape[-1],
                },
                "tokens": args.tokens,
                "routes": routes.size,
                "shards": tensors["shards"],
                "results": results,
                "best_existing": {
                    "pair_variant": best_pair["variant"],
                    "pair_median_ms": best_pair["pair_concat"]["candidate"][
                        "median_ms"
                    ],
                    "down_variant": best_down["variant"],
                    "down_median_ms": best_down["down"]["candidate"]["median_ms"],
                    "dominant": (
                        "pair"
                        if best_pair["pair_concat"]["candidate"]["median_ms"]
                        >= best_down["down"]["candidate"]["median_ms"]
                        else "down"
                    ),
                },
                "tail8_candidate": tail8_candidate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
