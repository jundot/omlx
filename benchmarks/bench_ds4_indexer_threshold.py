#!/usr/bin/env python3
"""Measure the DS4 ratio-4 prefill-indexer slope around pooled N=4096."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast


def _resolve_score_kernel(requested: str) -> str:
    """Resolve the benchmark route to the same score kernel serving uses."""

    qualified_device = mx.device_info().get("device_name") in {
        "Apple M2 Ultra",
        "Apple M3 Ultra",
        "Apple M5 Max",
    }
    mma_available = bool(
        getattr(fast, "_EXT_MMA_SCORE", False) and qualified_device
    )
    if requested == "auto":
        return "mma" if mma_available else "steel"
    if requested == "mma" and not mma_available:
        raise SystemExit("the requested MMA score kernel is unavailable")
    return requested


def _score(
    mode: str,
    q: mx.array,
    keys: mx.array,
    weights: mx.array,
    query_offset: int,
) -> mx.array:
    if mode == "mma":
        return fast.dsa_indexer_scores_mma(
            q,
            keys,
            weights,
            mask_ratio=4,
            mask_q_offset=query_offset,
        )
    return fast.dsa_indexer_scores(
        q,
        keys,
        weights,
        causal=False,
        mask_ratio=4,
        mask_q_offset=query_offset,
        use_nax=False,
    )


def build_taper_attribution(
    measurements: list[dict[str, Any]],
    observed_tps: tuple[float, ...],
    *,
    logical_chunk_tokens: int,
) -> dict[str, Any]:
    """Attribute adjacent observed wall growth to the indexer measurement.

    Each rate is converted to wall milliseconds for one logical prompt chunk.
    The ratio of adjacent indexer growth to adjacent wall growth is deliberately
    allowed above one: that is useful evidence that another component became
    faster or that the isolated replay is not the live critical rank.
    """

    if len(measurements) != len(observed_tps):
        raise ValueError("one observed TPS value is required per pooled length")
    if logical_chunk_tokens < 1 or any(rate <= 0 for rate in observed_tps):
        raise ValueError("logical chunk tokens and observed TPS must be positive")

    points = []
    for measurement, rate in zip(measurements, observed_tps):
        wall_ms = logical_chunk_tokens * 1000.0 / rate
        indexer_ms = float(measurement["parallel_layers_ms"])
        points.append(
            {
                "context_tokens": int(measurement["context_tokens"]),
                "observed_tps": rate,
                "observed_chunk_ms": wall_ms,
                "indexer_ms": indexer_ms,
                "non_indexer_residual_ms": wall_ms - indexer_ms,
                "indexer_wall_fraction": indexer_ms / wall_ms,
            }
        )

    intervals = []
    for before, after in zip(points, points[1:]):
        wall_growth = after["observed_chunk_ms"] - before["observed_chunk_ms"]
        indexer_growth = after["indexer_ms"] - before["indexer_ms"]
        intervals.append(
            {
                "context_start": before["context_tokens"],
                "context_end": after["context_tokens"],
                "observed_wall_growth_ms": wall_growth,
                "indexer_growth_ms": indexer_growth,
                "indexer_share_of_wall_growth": (
                    indexer_growth / wall_growth if wall_growth > 0 else None
                ),
            }
        )
    return {"points": points, "intervals": intervals}


def _measure(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-tokens", type=int, default=512)
    parser.add_argument(
        "--pooled-tokens",
        default="3584,3840,4096,4352,4608,5120,6144,8192",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--layers", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-kernel",
        choices=("auto", "mma", "steel"),
        default="auto",
        help="auto selects the serving-default exact MMA kernel when built",
    )
    parser.add_argument(
        "--no-verify-exact",
        action="store_true",
        help="skip the pre-timing bitwise MMA-vs-Steel score check",
    )
    parser.add_argument(
        "--observed-tps",
        default="",
        help="optional comma-separated live TPS values aligned with pooled lengths",
    )
    parser.add_argument(
        "--logical-chunk-tokens",
        type=int,
        default=1024,
        help="global prompt tokens represented by one TP row-sharded replay",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not fast.has_symbol("dsa_indexer_scores") or not fast.has_symbol(
        "dsa_topk_indices"
    ):
        raise SystemExit("native DS4 indexer kernels are unavailable")
    pooled_lengths = tuple(int(value) for value in args.pooled_tokens.split(","))
    if args.query_tokens < 2 or any(value <= 512 for value in pooled_lengths):
        raise SystemExit("query tokens must be >=2 and pooled lengths >512")
    if args.logical_chunk_tokens < args.query_tokens:
        raise SystemExit("logical chunk tokens cannot be smaller than query rows")
    score_kernel = _resolve_score_kernel(args.score_kernel)

    mx.random.seed(args.seed)
    query_tokens = args.query_tokens
    q = mx.random.uniform(-0.5, 0.5, (1, 64, query_tokens, 128)).astype(mx.bfloat16)
    weights = mx.random.uniform(-0.5, 0.5, (1, query_tokens, 64)).astype(mx.bfloat16)
    mx.eval(q, weights)

    layer_queries = [
        mx.random.uniform(-0.5, 0.5, q.shape).astype(mx.bfloat16)
        for _ in range(args.layers)
    ]
    layer_weights = [
        mx.random.uniform(-0.5, 0.5, weights.shape).astype(mx.bfloat16)
        for _ in range(args.layers)
    ]
    mx.eval(*layer_queries, *layer_weights)

    print(
        f"score_kernel={score_kernel} query_rows={query_tokens} "
        f"layers={args.layers} logical_chunk_tokens={args.logical_chunk_tokens}"
    )
    print(
        "context\tN\tscore_ms\ttopk_ms\tsort_ms\tcombined_ms\t"
        "parallel_layers_ms\tparallel_per_layer_ms\texact"
    )
    measurements: list[dict[str, Any]] = []
    for pooled_tokens in pooled_lengths:
        keys = mx.random.uniform(-0.5, 0.5, (1, 1, pooled_tokens, 128)).astype(
            mx.bfloat16
        )
        mx.eval(keys)
        # ``query_tokens`` is rank-local after row sharding, whereas the cache
        # offset advances by the global chunk.  Keeping those separate makes
        # the same harness valid for TP2 (512 local / 1024 logical) and
        # single-node (1024 local / 1024 logical) replays.
        query_offset = pooled_tokens * 4 - args.logical_chunk_tokens

        def score(keys=keys, query_offset=query_offset):
            return _score(score_kernel, q, keys, weights, query_offset)

        resident_scores = score()
        mx.eval(resident_scores)
        score_exact = True
        if score_kernel == "mma" and not args.no_verify_exact:
            steel_scores = _score("steel", q, keys, weights, query_offset)
            mx.eval(steel_scores)
            score_exact = bool(
                mx.array_equal(
                    resident_scores.view(mx.uint16),
                    steel_scores.view(mx.uint16),
                ).item()
            )
            if not score_exact:
                raise RuntimeError(
                    f"MMA score output is not bit-exact at pooled N={pooled_tokens}"
                )

        def topk(resident_scores=resident_scores):
            return fast.dsa_topk_indices(resident_scores, 512, bucketed=False)

        resident_indices = topk()
        mx.eval(resident_indices)
        score_ms = _measure(score, warmup=args.warmup, repeats=args.repeats)
        topk_ms = _measure(topk, warmup=args.warmup, repeats=args.repeats)
        sort_ms = _measure(
            lambda resident_indices=resident_indices: mx.sort(
                resident_indices, axis=-1
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        combined_ms = _measure(
            lambda: mx.sort(
                fast.dsa_topk_indices(score(), 512, bucketed=False),
                axis=-1,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        layer_keys = [
            mx.random.uniform(-0.5, 0.5, keys.shape).astype(mx.bfloat16)
            for _ in range(args.layers)
        ]
        mx.eval(*layer_keys)

        def parallel_graph(layer_keys=layer_keys, query_offset=query_offset):
            outputs = []
            for layer_q, layer_k, layer_w in zip(
                layer_queries, layer_keys, layer_weights
            ):
                layer_scores = _score(
                    score_kernel,
                    layer_q,
                    layer_k,
                    layer_w,
                    query_offset,
                )
                outputs.append(
                    mx.sort(
                        fast.dsa_topk_indices(layer_scores, 512, bucketed=False),
                        axis=-1,
                    )
                )
            return outputs

        parallel_ms = _measure(
            parallel_graph,
            warmup=max(1, args.warmup // 2),
            repeats=max(3, args.repeats // 2),
        )
        measurement = {
            "context_tokens": pooled_tokens * 4,
            "pooled_tokens": pooled_tokens,
            "score_ms": score_ms,
            "topk_ms": topk_ms,
            "sort_ms": sort_ms,
            "combined_ms": combined_ms,
            "parallel_layers_ms": parallel_ms,
            "parallel_per_layer_ms": parallel_ms / args.layers,
            "score_bit_exact": score_exact,
        }
        measurements.append(measurement)
        print(
            f"{measurement['context_tokens']}\t{pooled_tokens}\t"
            f"{score_ms:.3f}\t{topk_ms:.3f}\t{sort_ms:.3f}\t"
            f"{combined_ms:.3f}\t{parallel_ms:.3f}\t"
            f"{parallel_ms / args.layers:.3f}\t{score_exact}"
        )

    observed = tuple(
        float(value) for value in args.observed_tps.split(",") if value.strip()
    )
    attribution = None
    if observed:
        attribution = build_taper_attribution(
            measurements,
            observed,
            logical_chunk_tokens=args.logical_chunk_tokens,
        )
        print("\ncontext\tobserved_tps\twall_ms\tindexer_ms\tresidual_ms\tindexer_wall")
        for point in attribution["points"]:
            print(
                f"{point['context_tokens']}\t{point['observed_tps']:.3f}\t"
                f"{point['observed_chunk_ms']:.3f}\t{point['indexer_ms']:.3f}\t"
                f"{point['non_indexer_residual_ms']:.3f}\t"
                f"{point['indexer_wall_fraction']:.4f}"
            )
        print("\ninterval\twall_growth_ms\tindexer_growth_ms\tindexer_slope_share")
        for interval in attribution["intervals"]:
            share = interval["indexer_share_of_wall_growth"]
            share_text = "n/a" if share is None else f"{share:.4f}"
            print(
                f"{interval['context_start']}->{interval['context_end']}\t"
                f"{interval['observed_wall_growth_ms']:.3f}\t"
                f"{interval['indexer_growth_ms']:.3f}\t{share_text}"
            )

    report = {
        "schema_version": 1,
        "scope": "isolated_ds4_ratio4_indexer_taper",
        "score_kernel": score_kernel,
        "query_rows": query_tokens,
        "logical_chunk_tokens": args.logical_chunk_tokens,
        "layers": args.layers,
        "measurements": measurements,
        "attribution": attribution,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
