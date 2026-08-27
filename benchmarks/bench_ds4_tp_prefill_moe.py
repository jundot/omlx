#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark real-layer DS4 TP4/4 prefill SwitchGLU kernels on M3 Ultra.

The checkpoint tensors are sliced exactly like equal-width TP rank 0/1:
gate/up are sliced on their output axis and down on its packed input axis.
Both paths consume identical sorted routes and FP16 activations, then use the
same LimitedSwiGLU, inverse permutation, BF16 cast, and score reduction.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v4.switch_layers import (
    _block_config,
    _build_mxfp4_blocks,
)


VARIANTS = {
    0: (8, 32),
    1: (16, 32),
    2: (32, 32),
    3: (16, 64),
    4: (32, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=(512, 1024, 2048))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--no-sweep", action="store_true")
    return parser.parse_args()


def load_tp_layer(model_dir: Path, layer: int, rank: int):
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

    intermediate = 1024
    output_start = rank * intermediate
    output_stop = output_start + intermediate
    packed_start = rank * (intermediate // 8)
    packed_stop = packed_start + intermediate // 8
    scale_start = rank * (intermediate // 32)
    scale_stop = scale_start + intermediate // 32

    def stack(projection: str, suffix: str, indexer):
        return mx.stack(
            [
                tensors[f"{prefix}{expert}.{projection}.{suffix}"][indexer]
                for expert in range(256)
            ]
        )

    gate_weight = stack(
        "w1", "weight", (slice(output_start, output_stop), slice(None))
    )
    gate_scales = stack(
        "w1", "scales", (slice(output_start, output_stop), slice(None))
    )
    up_weight = stack(
        "w3", "weight", (slice(output_start, output_stop), slice(None))
    )
    up_scales = stack(
        "w3", "scales", (slice(output_start, output_stop), slice(None))
    )
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
    return (
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        shards,
    )


def evaluate(value) -> None:
    mx.eval(*(value if isinstance(value, tuple) else (value,)))
    mx.synchronize()


def abba(first, second, warmup: int, cycles: int):
    for _ in range(warmup):
        evaluate(first())
        evaluate(second())
    timings = {"block": [], "stock": []}
    for _ in range(cycles):
        for name, function in (
            ("block", first),
            ("stock", second),
            ("stock", second),
            ("block", first),
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


def comparison(block_values, stock_values) -> dict:
    block = summary(block_values)
    stock = summary(stock_values)
    return {
        "block": block,
        "stock": stock,
        "speedup": stock["median_ms"] / block["median_ms"],
    }


def main() -> None:
    args = parse_args()
    if not (
        fast.is_native_available()
        and fast.has_symbol("deepseek_mxfp4_gather_qmm_blocks")
        and fast.has_symbol("deepseek_mxfp4_gather_qmm_pair_concat_blocks")
    ):
        raise RuntimeError("DeepSeek native block-list kernels are unavailable")

    (
        up_weight,
        up_scales,
        gate_weight,
        gate_scales,
        down_weight,
        down_scales,
        shards,
    ) = load_tp_layer(args.model, args.layer, args.rank)
    experts = 256
    hidden = 4096
    intermediate = 1024
    stock_kwargs = {
        "transpose": True,
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
        "sorted_indices": True,
    }
    results = []

    for tokens in args.tokens:
        mx.random.seed(1000 + tokens)
        x = mx.random.normal((1, tokens, hidden)).astype(mx.bfloat16)
        offsets = (0, 37, 79, 131, 181, 233)
        routes = mx.array(
            [
                [
                    [int((token * 73 + offset) % experts) for offset in offsets]
                    for token in range(tokens)
                ]
            ],
            dtype=mx.uint32,
        )
        scores = mx.softmax(
            mx.random.normal((1, tokens, 6)).astype(mx.float32), axis=-1
        )
        flat_routes = routes.flatten()
        order = mx.argsort(flat_routes)
        inverse = mx.argsort(order)
        sorted_x = mx.expand_dims(x, (-2, -3)).flatten(0, -3)[order // 6]
        sorted_x = sorted_x.astype(mx.float16)
        sorted_ids = flat_routes[order]
        current_bm, current_variant = _block_config(routes.size, "mxfp4")
        block_meta, block_count = _build_mxfp4_blocks(
            sorted_ids, experts, current_bm
        )
        mx.eval(sorted_x, sorted_ids, inverse, block_meta, block_count, scores)
        mx.synchronize()

        def stock_pair():
            return (
                mx.gather_qmm(
                    sorted_x,
                    up_weight,
                    up_scales,
                    None,
                    rhs_indices=sorted_ids,
                    **stock_kwargs,
                ),
                mx.gather_qmm(
                    sorted_x,
                    gate_weight,
                    gate_scales,
                    None,
                    rhs_indices=sorted_ids,
                    **stock_kwargs,
                ),
            )

        def block_pair(meta=block_meta, count=block_count, variant=current_variant):
            pair = fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
                sorted_x,
                up_weight,
                up_scales,
                gate_weight,
                gate_scales,
                meta,
                count,
                variant,
            )
            return pair[..., :intermediate], pair[..., intermediate:]

        block_pair_value = block_pair()
        stock_pair_value = stock_pair()
        evaluate((*block_pair_value, *stock_pair_value))
        pair_exact = all(
            mx.array_equal(block, stock).item()
            for block, stock in zip(block_pair_value, stock_pair_value)
        )
        pair_times = abba(block_pair, stock_pair, args.warmup, args.cycles)

        up, gate = block_pair_value
        gate = mx.minimum(gate, 10.0)
        activated = (gate * mx.sigmoid(gate)) * mx.clip(up, -10.0, 10.0)
        mx.eval(activated)
        mx.synchronize()

        def block_down(meta=block_meta, count=block_count, variant=current_variant):
            return fast.deepseek_mxfp4_gather_qmm_blocks(
                activated,
                down_weight,
                down_scales,
                meta,
                count,
                variant,
            )

        def stock_down():
            return mx.gather_qmm(
                activated,
                down_weight,
                down_scales,
                None,
                rhs_indices=sorted_ids,
                **stock_kwargs,
            )

        block_down_value, stock_down_value = block_down(), stock_down()
        evaluate((block_down_value, stock_down_value))
        down_exact = mx.array_equal(block_down_value, stock_down_value).item()
        down_times = abba(block_down, stock_down, args.warmup, args.cycles)

        def finish(value):
            value = value[inverse].reshape(1, tokens, 6, 1, hidden)
            value = value.squeeze(-2).astype(mx.bfloat16)
            return (value * scores[..., None].astype(value.dtype)).sum(-2)

        def stock_full():
            up_value, gate_value = stock_pair()
            gate_value = mx.minimum(gate_value, 10.0)
            activation = (gate_value * mx.sigmoid(gate_value)) * mx.clip(
                up_value, -10.0, 10.0
            )
            return finish(
                mx.gather_qmm(
                    activation,
                    down_weight,
                    down_scales,
                    None,
                    rhs_indices=sorted_ids,
                    **stock_kwargs,
                )
            )

        def make_block_full(meta, count, variant):
            def block_full():
                pair = fast.deepseek_mxfp4_gather_qmm_pair_concat_blocks(
                    sorted_x,
                    up_weight,
                    up_scales,
                    gate_weight,
                    gate_scales,
                    meta,
                    count,
                    variant,
                )
                up_value = pair[..., :intermediate]
                gate_value = mx.minimum(pair[..., intermediate:], 10.0)
                activation = (gate_value * mx.sigmoid(gate_value)) * mx.clip(
                    up_value, -10.0, 10.0
                )
                return finish(
                    fast.deepseek_mxfp4_gather_qmm_blocks(
                        activation,
                        down_weight,
                        down_scales,
                        meta,
                        count,
                        variant,
                    )
                )

            return block_full

        block_full = make_block_full(block_meta, block_count, current_variant)
        block_full_value, stock_full_value = block_full(), stock_full()
        evaluate((block_full_value, stock_full_value))
        difference = mx.abs(
            block_full_value.astype(mx.float32) - stock_full_value.astype(mx.float32)
        )
        full_exact = mx.array_equal(block_full_value, stock_full_value).item()
        full_times = abba(block_full, stock_full, args.warmup, args.cycles)

        row = {
            "tokens": tokens,
            "routes": tokens * 6,
            "current": {
                "variant": current_variant,
                "bm": current_bm,
                "bn": VARIANTS[current_variant][1],
            },
            "parity": {
                "pair": bool(pair_exact),
                "down": bool(down_exact),
                "full": bool(full_exact),
                "max_abs": float(mx.max(difference).item()),
            },
            "pair": comparison(pair_times["block"], pair_times["stock"]),
            "down": comparison(down_times["block"], down_times["stock"]),
            "full": comparison(full_times["block"], full_times["stock"]),
        }

        if not args.no_sweep:
            sweep = []
            for variant, (bm, bn) in VARIANTS.items():
                meta, count = _build_mxfp4_blocks(sorted_ids, experts, bm)
                mx.eval(meta, count)
                mx.synchronize()
                candidate = make_block_full(meta, count, variant)
                candidate_value = candidate()
                evaluate(candidate_value)
                candidate_times = abba(
                    candidate, stock_full, args.warmup, args.cycles
                )
                candidate_stats = comparison(
                    candidate_times["block"], candidate_times["stock"]
                )
                sweep.append(
                    {
                        "variant": variant,
                        "bm": bm,
                        "bn": bn,
                        "exact": bool(
                            mx.array_equal(candidate_value, stock_full_value).item()
                        ),
                        **candidate_stats,
                    }
                )
            row["sweep"] = sweep
        results.append(row)

    print(
        json.dumps(
            {
                "model": str(args.model),
                "layer": args.layer,
                "tp": f"4/4 rank {args.rank}",
                "shards": shards,
                "weight_bytes": sum(
                    value.nbytes
                    for value in (
                        up_weight,
                        up_scales,
                        gate_weight,
                        gate_scales,
                        down_weight,
                        down_scales,
                    )
                ),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
