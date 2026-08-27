#!/usr/bin/env python3
"""Measure DS4's complete ratio-4 pooled-cache append set by logical offset."""

from __future__ import annotations

import argparse
import os
import time

import mlx.core as mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=21)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--pooled-rows-per-step", type=int, default=256)
    parser.add_argument("--initial-context-tokens", type=int, default=32768)
    args = parser.parse_args()

    os.environ["OMLX_DSV4_POOL_INITIAL_CONTEXT_TOKENS"] = str(
        args.initial_context_tokens
    )
    from omlx.patches.deepseek_v4.cache_extras import PoolingCache

    compressor = [PoolingCache(4) for _ in range(args.layers)]
    indexer = [PoolingCache(4) for _ in range(args.layers)]
    comp_rows = mx.random.normal(
        (1, args.pooled_rows_per_step, 512)
    ).astype(mx.bfloat16)
    index_rows = mx.random.normal(
        (1, args.pooled_rows_per_step, 128)
    ).astype(mx.bfloat16)
    mx.eval(comp_rows, index_rows)

    print("logical_rows\tappend_ms\tcomp_capacity\tindex_capacity")
    for _step in range(args.steps):
        started = time.perf_counter()
        for cache in compressor:
            cache.update_and_fetch(comp_rows)
        for cache in indexer:
            cache.update_and_fetch(index_rows)
        mx.eval(
            *[cache._pool_buf for cache in compressor],
            *[cache._pool_buf for cache in indexer],
        )
        mx.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1e3
        print(
            f"{compressor[0]._pool_len}\t{elapsed_ms:.3f}\t"
            f"{compressor[0]._pool_buf.shape[1]}\t"
            f"{indexer[0]._pool_buf.shape[1]}"
        )


if __name__ == "__main__":
    main()
