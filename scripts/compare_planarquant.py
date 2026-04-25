#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic PlanarQuant3 performance and accuracy comparison."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import mlx.core as mx

from omlx.cache.planarquant.kv_cache import PlanarQuantKVCache

SYNTHETIC_MODEL_LABEL = "synthetic KV-cache/attention microbench (no neural model)"


def _sync() -> None:
    try:
        mx.synchronize()
    except Exception:
        pass


def _time_ms(fn, iters: int) -> float:
    samples: list[float] = []
    for _ in range(max(1, iters)):
        _sync()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _cosine(a: mx.array, b: mx.array) -> float:
    a32 = a.reshape(-1).astype(mx.float32)
    b32 = b.reshape(-1).astype(mx.float32)
    dot = float(mx.sum(a32 * b32).item())
    na = float(mx.sqrt(mx.sum(a32 * a32)).item())
    nb = float(mx.sqrt(mx.sum(b32 * b32)).item())
    return dot / (na * nb + 1e-10)


def _error_metrics(ref: mx.array, out: mx.array) -> dict[str, float]:
    diff = ref.astype(mx.float32) - out.astype(mx.float32)
    return {
        "cosine": _cosine(ref, out),
        "max_abs": float(mx.max(mx.abs(diff)).item()),
        "mean_abs": float(mx.mean(mx.abs(diff)).item()),
        "rmse": float(mx.sqrt(mx.mean(diff * diff)).item()),
    }


def _case(
    *,
    batch: int,
    heads: int,
    length: int,
    dim: int,
    quantize_v: bool,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    mx.random.seed(seed)
    scale = 1.0 / dim**0.5
    keys = (mx.random.normal((batch, heads, length, dim)) * 0.1).astype(mx.float16)
    values = (mx.random.normal((batch, heads, length, dim)) * 0.1).astype(mx.float16)
    queries = (mx.random.normal((batch, heads, 1, dim)) * 0.1).astype(mx.float16)
    mx.eval(keys, values, queries)

    t0 = time.perf_counter()
    cache = PlanarQuantKVCache(quantize_v=quantize_v)
    cache.update_and_fetch(keys, values)
    cache.finalize_prefill()
    _sync()
    pq_prefill_finalize_ms = (time.perf_counter() - t0) * 1000.0

    fp16_bytes = int(keys.nbytes + values.nbytes)
    packed_bytes = int(cache.nbytes)

    def fp16_attention():
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale
        )

    fp16_out = fp16_attention()
    mx.eval(fp16_out)
    fp16_decode_ms = _time_ms(fp16_attention, iters)

    t0 = time.perf_counter()
    pq_out = cache.decode_attention(queries, scale=scale)
    mx.eval(pq_out)
    _sync()
    pq_decode_cold_ms = (time.perf_counter() - t0) * 1000.0
    runtime_bytes = int(cache.nbytes)

    pq_decode_warm_ms = _time_ms(
        lambda: cache.decode_attention(queries, scale=scale), iters
    )

    metrics = _error_metrics(fp16_out, pq_out)
    mode = "k_v" if quantize_v else "k_only"
    return {
        "mode": mode,
        "batch": batch,
        "heads": heads,
        "length": length,
        "dim": dim,
        "fp16_bytes": fp16_bytes,
        "packed_bytes": packed_bytes,
        "runtime_bytes": runtime_bytes,
        "packed_compression": fp16_bytes / packed_bytes,
        "runtime_compression": fp16_bytes / runtime_bytes,
        "fp16_decode_ms": fp16_decode_ms,
        "pq_decode_cold_ms": pq_decode_cold_ms,
        "pq_decode_warm_ms": pq_decode_warm_ms,
        "pq_warm_vs_fp16": pq_decode_warm_ms / max(fp16_decode_ms, 1e-9),
        "pq_prefill_finalize_ms": pq_prefill_finalize_ms,
        **metrics,
    }


def run_comparisons(
    *,
    lengths: list[int],
    batch: int = 1,
    heads: int = 8,
    dim: int = 128,
    iters: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for length in lengths:
        for quantize_v in (True, False):
            results.append(
                _case(
                    batch=batch,
                    heads=heads,
                    length=length,
                    dim=dim,
                    quantize_v=quantize_v,
                    iters=iters,
                    seed=seed + length + (0 if quantize_v else 1),
                )
            )
    return results


def _print_table(results: list[dict[str, Any]]) -> None:
    print(
        f"{'mode':<7} {'B':>2} {'H':>3} {'T':>5} {'cos':>9} "
        f"{'rmse':>10} {'packed':>8} {'runtime':>8} "
        f"{'fp16 ms':>9} {'pq warm':>9} {'pq/fp16':>8}"
    )
    print("-" * 94)
    for r in results:
        print(
            f"{r['mode']:<7} {r['batch']:>2} {r['heads']:>3} {r['length']:>5} "
            f"{r['cosine']:>9.6f} {r['rmse']:>10.6g} "
            f"{r['packed_compression']:>7.2f}x {r['runtime_compression']:>7.2f}x "
            f"{r['fp16_decode_ms']:>9.3f} {r['pq_decode_warm_ms']:>9.3f} "
            f"{r['pq_warm_vs_fp16']:>8.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="1024,4096,8192,16384,32768,65536,131072")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.95)
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    results = run_comparisons(
        lengths=lengths,
        batch=args.batch,
        heads=args.heads,
        dim=args.dim,
        iters=args.iters,
        seed=args.seed,
    )
    min_cos = min(r["cosine"] for r in results)
    if args.json:
        print(
            json.dumps(
                {
                    "model": SYNTHETIC_MODEL_LABEL,
                    "results": results,
                    "min_cosine": min_cos,
                },
                indent=2,
            )
        )
    else:
        print(f"model: {SYNTHETIC_MODEL_LABEL}")
        _print_table(results)
        print(f"\nminimum cosine: {min_cos:.6f}")
    return 0 if min_cos >= args.min_cosine else 1


if __name__ == "__main__":
    raise SystemExit(main())
