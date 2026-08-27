#!/usr/bin/env python3
"""Physical A/B gate for the default-off DS4 MMA WM4xWN1 score tile.

The candidate changes only simdgroup ownership inside the existing 64x64,
128-thread, 16KB-K-panel MMA kernel.  It is accepted only when complete BF16
score sheets and deterministic top-k results are bit-identical and the
21-layer score+top-k+temporal-sort graph has a positive median gain at every
requested ratio-4 context.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast


def _timed(fn: Callable[[], object]) -> float:
    started = time.perf_counter()
    mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - started) * 1e3


def _measure_interleaved(
    baseline: Callable[[], object],
    candidate: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> tuple[list[float], list[float]]:
    """Alternate first-run ownership so thermal/order drift is symmetric."""

    for index in range(warmup):
        order = (baseline, candidate) if index % 2 == 0 else (candidate, baseline)
        for fn in order:
            mx.eval(fn())
            mx.synchronize()

    samples = {"baseline": [], "candidate": []}
    for index in range(repeats):
        order = (
            (("baseline", baseline), ("candidate", candidate))
            if index % 2 == 0
            else (("candidate", candidate), ("baseline", baseline))
        )
        for name, fn in order:
            samples[name].append(_timed(fn))
    return samples["baseline"], samples["candidate"]


def _bit_equal(left: mx.array, right: mx.array) -> bool:
    mx.eval(left, right)
    return bool(mx.array_equal(left.view(mx.uint16), right.view(mx.uint16)).item())


def _score(
    q: mx.array,
    k: mx.array,
    weights: mx.array,
    query_offset: int,
    *,
    candidate: bool,
) -> mx.array:
    return fast.dsa_indexer_scores_mma(
        q,
        k,
        weights,
        mask_ratio=4,
        mask_q_offset=query_offset,
        use_wm4_wn1=candidate,
    )


def _steel_score(
    q: mx.array,
    k: mx.array,
    weights: mx.array,
    query_offset: int,
) -> mx.array:
    return fast.dsa_indexer_scores(
        q,
        k,
        weights,
        causal=False,
        mask_ratio=4,
        mask_q_offset=query_offset,
        use_nax=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-tokens", type=int, default=512)
    parser.add_argument("--logical-chunk-tokens", type=int, default=1024)
    parser.add_argument("--pooled-tokens", default="7500,25000,62500")
    parser.add_argument("--layers", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-gain",
        type=float,
        default=1.0,
        help="minimum baseline/candidate median ratio required at every point",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not fast._EXT_MMA_SCORE or not fast._EXT_MMA_WM4:
        raise SystemExit(
            "the rebuilt glm_moe_dsa extension with use_wm4_wn1 is required"
        )
    if not fast.has_symbol("dsa_indexer_scores") or not fast.has_symbol(
        "dsa_topk_indices"
    ):
        raise SystemExit("native Steel score/top-k references are unavailable")
    if args.query_tokens < 64:
        raise SystemExit("the WM4xWN1 candidate requires at least 64 query rows")
    if args.logical_chunk_tokens < args.query_tokens:
        raise SystemExit("logical chunk tokens cannot be smaller than query rows")
    pooled_lengths = tuple(int(value) for value in args.pooled_tokens.split(","))
    if any(value <= 512 for value in pooled_lengths):
        raise SystemExit("all pooled lengths must exceed top-k=512")

    mx.random.seed(args.seed)
    query_tokens = args.query_tokens
    q = mx.random.uniform(-0.5, 0.5, (1, 64, query_tokens, 128)).astype(mx.bfloat16)
    weights = mx.random.uniform(-0.5, 0.5, (1, query_tokens, 64)).astype(mx.bfloat16)
    layer_queries = [
        mx.random.uniform(-0.5, 0.5, q.shape).astype(mx.bfloat16)
        for _ in range(args.layers)
    ]
    layer_weights = [
        mx.random.uniform(-0.5, 0.5, weights.shape).astype(mx.bfloat16)
        for _ in range(args.layers)
    ]
    mx.eval(q, weights, *layer_queries, *layer_weights)

    print(
        f"device={mx.device_info().get('device_name')} B=1 H=64 "
        f"M={query_tokens} D=128 "
        f"layers={args.layers} ratio=4"
    )
    if not fast.dsa_indexer_mma_wm4_wn1_eligible():
        raise SystemExit(
            "WM4xWN1 is physically qualified only on applegpu_g15d (M3 Ultra)"
        )
    print(
        "context\tN\tscore_exact\ttopk_exact\tbaseline_score_ms\t"
        "candidate_score_ms\tscore_gain\tbaseline_21l_ms\t"
        "candidate_21l_ms\tgraph_gain\tpass"
    )

    measurements = []
    for pooled_tokens in pooled_lengths:
        query_offset = pooled_tokens * 4 - args.logical_chunk_tokens
        keys = mx.random.uniform(-0.5, 0.5, (1, 1, pooled_tokens, 128)).astype(
            mx.bfloat16
        )
        mx.eval(keys)

        baseline_scores = _score(q, keys, weights, query_offset, candidate=False)
        candidate_scores = _score(q, keys, weights, query_offset, candidate=True)
        steel_scores = _steel_score(q, keys, weights, query_offset)
        mx.eval(baseline_scores, candidate_scores, steel_scores)
        score_exact = _bit_equal(baseline_scores, candidate_scores) and _bit_equal(
            steel_scores, candidate_scores
        )

        baseline_topk = fast.dsa_topk_indices(baseline_scores, 512, bucketed=False)
        candidate_topk = fast.dsa_topk_indices(candidate_scores, 512, bucketed=False)
        mx.eval(baseline_topk, candidate_topk)
        topk_exact = bool(mx.array_equal(baseline_topk, candidate_topk).item())
        if not score_exact or not topk_exact:
            raise SystemExit(
                f"lossless gate failed at pooled N={pooled_tokens}: "
                f"score_exact={score_exact}, topk_exact={topk_exact}"
            )

        baseline_score_samples, candidate_score_samples = _measure_interleaved(
            lambda keys=keys, query_offset=query_offset: _score(
                q, keys, weights, query_offset, candidate=False
            ),
            lambda keys=keys, query_offset=query_offset: _score(
                q, keys, weights, query_offset, candidate=True
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )

        layer_keys = [
            mx.random.uniform(-0.5, 0.5, keys.shape).astype(mx.bfloat16)
            for _ in range(args.layers)
        ]
        mx.eval(*layer_keys)

        def graph(
            candidate: bool,
            layer_keys=layer_keys,
            query_offset=query_offset,
        ):
            outputs = []
            for layer_q, layer_k, layer_w in zip(
                layer_queries, layer_keys, layer_weights
            ):
                scores = _score(
                    layer_q,
                    layer_k,
                    layer_w,
                    query_offset,
                    candidate=candidate,
                )
                indices = fast.dsa_topk_indices(scores, 512, bucketed=False)
                outputs.append(mx.sort(indices, axis=-1))
            return outputs

        baseline_graph_samples, candidate_graph_samples = _measure_interleaved(
            lambda: graph(False),
            lambda: graph(True),
            warmup=max(1, args.warmup // 2),
            repeats=args.repeats,
        )

        baseline_score_ms = statistics.median(baseline_score_samples)
        candidate_score_ms = statistics.median(candidate_score_samples)
        baseline_graph_ms = statistics.median(baseline_graph_samples)
        candidate_graph_ms = statistics.median(candidate_graph_samples)
        score_gain = baseline_score_ms / candidate_score_ms
        graph_gain = baseline_graph_ms / candidate_graph_ms
        passed = score_gain > args.min_gain and graph_gain > args.min_gain
        measurement = {
            "context_tokens": pooled_tokens * 4,
            "pooled_tokens": pooled_tokens,
            "score_bit_exact": score_exact,
            "topk_exact": topk_exact,
            "baseline_score_ms": baseline_score_ms,
            "candidate_score_ms": candidate_score_ms,
            "score_gain": score_gain,
            "baseline_score_samples_ms": baseline_score_samples,
            "candidate_score_samples_ms": candidate_score_samples,
            "baseline_graph_ms": baseline_graph_ms,
            "candidate_graph_ms": candidate_graph_ms,
            "graph_gain": graph_gain,
            "baseline_graph_samples_ms": baseline_graph_samples,
            "candidate_graph_samples_ms": candidate_graph_samples,
            "passed": passed,
        }
        measurements.append(measurement)
        print(
            f"{pooled_tokens * 4}\t{pooled_tokens}\t{score_exact}\t"
            f"{topk_exact}\t"
            f"{baseline_score_ms:.3f}\t{candidate_score_ms:.3f}\t"
            f"{score_gain:.4f}\t{baseline_graph_ms:.3f}\t"
            f"{candidate_graph_ms:.3f}\t{graph_gain:.4f}\t{passed}"
        )

    passed = all(point["passed"] for point in measurements)
    report = {
        "schema_version": 1,
        "scope": "ds4_ratio4_mma_wm4_wn1_gate",
        "device": mx.device_info(),
        "query_rows": query_tokens,
        "logical_chunk_tokens": args.logical_chunk_tokens,
        "layers": args.layers,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "minimum_gain": args.min_gain,
        "measurements": measurements,
        "passed": passed,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise SystemExit("WM4xWN1 rejected: no positive median at every context")


if __name__ == "__main__":
    main()
