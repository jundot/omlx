#!/usr/bin/env python3
"""Isolated Steel-vs-NAX benchmark for DS4-Flash prefill indexer scores.

Run only with the rebuilt ``glm_moe_dsa`` extension on an M5/NAX Mac. The
benchmark refuses to report a speedup unless the complete BF16 score sheet and
the deterministic top-k output match Steel exactly.
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
from omlx.custom_kernels.nax import is_nax_available


def _sync() -> None:
    mx.synchronize()


def _measure(fn, *, warmup: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        mx.eval(fn())
    _sync()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        _sync()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-tokens", type=int, default=512)
    parser.add_argument("--pooled-tokens", type=int, default=4096)
    parser.add_argument("--query-offset", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-inexact",
        action="store_true",
        help="measure a rejected candidate after printing the failed lossless gate",
    )
    args = parser.parse_args()

    if not is_nax_available():
        raise SystemExit("NAX is unavailable on this host/runtime")
    if not glm_fast._EXT_NAX_SCORE:
        raise SystemExit("glm_moe_dsa extension predates the NAX score ABI")
    if not glm_fast.dsa_indexer_nax_kernels_built():
        raise SystemExit("omlx_glm_kernels_nax.metallib is not built")
    if args.query_tokens < 16 or args.pooled_tokens <= 512:
        raise SystemExit("NAX route requires query_tokens>=16 and pooled_tokens>512")

    M = args.query_tokens
    N = args.pooled_tokens
    mx.random.seed(args.seed)
    q = mx.random.uniform(-0.5, 0.5, (1, 64, M, 128)).astype(mx.bfloat16)
    k = mx.random.uniform(-0.5, 0.5, (1, 1, N, 128)).astype(mx.bfloat16)
    w = mx.random.uniform(-0.5, 0.5, (1, M, 64)).astype(mx.bfloat16)
    mx.eval(q, k, w)

    def score(use_nax: bool):
        return glm_fast.dsa_indexer_scores(
            q,
            k,
            w,
            causal=False,
            mask_ratio=4,
            mask_q_offset=args.query_offset,
            use_nax=use_nax,
        )

    steel = score(False)
    nax = score(True)
    mx.eval(steel, nax)
    score_exact = bool(
        mx.array_equal(steel.view(mx.uint16), nax.view(mx.uint16)).item()
    )
    differing_scores = int(
        mx.sum(steel.view(mx.uint16) != nax.view(mx.uint16)).item()
    )
    max_abs = float(
        mx.max(mx.abs(steel.astype(mx.float32) - nax.astype(mx.float32))).item()
    )
    idx_steel = glm_fast.dsa_topk_indices(steel, 512, bucketed=False)
    idx_nax = glm_fast.dsa_topk_indices(nax, 512, bucketed=False)
    mx.eval(idx_steel, idx_nax)
    topk_exact = bool(mx.array_equal(idx_steel, idx_nax).item())
    if (not score_exact or not topk_exact) and not args.allow_inexact:
        raise SystemExit(
            f"lossless gate failed: score_exact={score_exact}, topk_exact={topk_exact}"
        )
    if not glm_fast.dsa_indexer_nax_runtime_active():
        raise SystemExit("NAX pipeline demoted to Steel during validation")

    steel_ms, steel_samples = _measure(
        lambda: score(False), warmup=args.warmup, repeats=args.repeats
    )
    nax_ms, nax_samples = _measure(
        lambda: score(True), warmup=args.warmup, repeats=args.repeats
    )

    print(f"shape: B=1 H=64 M={M} N={N} D=128 BF16 ratio=4")
    print(
        "lossless: "
        f"score_exact={score_exact} topk_exact={topk_exact} "
        f"different_scores={differing_scores}/{steel.size} max_abs={max_abs:.9g}"
    )
    print(f"Steel median: {steel_ms:.3f} ms  samples={steel_samples}")
    print(f"NAX   median: {nax_ms:.3f} ms  samples={nax_samples}")
    print(f"speedup: {steel_ms / nax_ms:.3f}x")


if __name__ == "__main__":
    main()
