#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark oMLX attention dispatch over fixed FP16, TQ4, and affine4 caches.

Run with the server and other GPU workloads stopped:
    python benchmarks/bench_affine4.py --fallbacks > attention.json

Timings include dispatch, evaluation, and synchronization. Cache construction,
reference calculations, and native-path observation are excluded. L>1 queries
occupy the final cached positions and use a causal mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import sys
import time
from contextlib import nullcontext
from functools import partial
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch


def _measure(call, warmup, iterations):
    import mlx.core as mx

    for _ in range(warmup):
        mx.eval(call())
    mx.synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        mx.eval(call())
        mx.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "p90": sorted(samples)[math.ceil(0.9 * len(samples)) - 1],
    }


def _agreement(output, reference):
    import mlx.core as mx

    output, reference = output.astype(mx.float32), reference.astype(mx.float32)
    if not mx.all(mx.isfinite(output) & mx.isfinite(reference)).item():
        return {"finite": False, "max_abs": None, "relative_l2": None}
    difference = output - reference
    return {
        "finite": True,
        "max_abs": mx.max(mx.abs(difference)).item(),
        "relative_l2": (
            mx.sqrt(mx.sum(difference * difference))
            / mx.maximum(mx.sqrt(mx.sum(reference * reference)), 1e-30)
        ).item(),
    }


def _case(dim, query_heads, tokens, query_length, args):
    import mlx.core as mx
    from mlx_lm.models import base
    from mlx_lm.models.cache import KVCache

    from omlx import affine4
    from omlx.turboquant_kv import TurboQuantKVCache

    mx.random.seed(args.seed + dim * 1009 + tokens * 17 + query_length)
    keys = mx.random.normal((1, 4, tokens, dim), dtype=mx.float16)
    values = mx.random.normal(keys.shape, dtype=mx.float16)
    queries = mx.random.normal((1, query_heads, query_length, dim), dtype=mx.float16)
    scale = dim**-0.5
    mask = "causal" if query_length > 1 else None
    mx.eval(queries, keys, values)
    fp16_reference = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, mask=mask
    )
    mx.eval(fp16_reference)
    results = []

    factories = (
        ("fp16", KVCache),
        ("tq4", lambda: TurboQuantKVCache(bits=4, seed=args.seed)),
        ("affine4", lambda: affine4.Affine4KVCache(bits=4, seed=args.seed)),
    )
    order = -1 if args.order == "reverse" else 1
    for scheme, factory in factories[::order]:
        cache = factory()
        key_state, value_state = cache.update_and_fetch(keys, values)
        mx.eval(cache.state)
        if scheme == "fp16":
            dense_keys, dense_values = key_state, value_state
        else:
            dense_keys, dense_values = cache.dequantize(key_state, value_state)
        dequant_reference = mx.fast.scaled_dot_product_attention(
            queries.astype(mx.float32),
            dense_keys.astype(mx.float32),
            dense_values.astype(mx.float32),
            scale=scale,
            mask=mask,
        )
        mx.eval(dequant_reference)

        dispatch = partial(
            base.scaled_dot_product_attention,
            queries,
            key_state,
            value_state,
            cache,
            scale,
            mask,
        )

        def record(
            method, call, observe_native=False, cache=cache, reference=dequant_reference
        ):
            native_used = False
            native_attention = affine4._native_attention

            def observe(*positional, **keywords):
                nonlocal native_used
                result = native_attention(*positional, **keywords)
                native_used = result is not None
                return result

            if observe_native:
                with patch.object(affine4, "_native_attention", observe):
                    output = call()
                    mx.eval(output)
            else:
                output = call()
                mx.eval(output)
            agreement = _agreement(output, reference)
            quantization_error = _agreement(output, fp16_reference)
            results.append(
                {
                    "method": method,
                    "batch": 1,
                    "head_dim": dim,
                    "query_heads": query_heads,
                    "kv_heads": 4,
                    "tokens": tokens,
                    "query_length": query_length,
                    "native_mpp": native_used,
                    "cache_bytes": int(cache.nbytes),
                    "vs_dequant_fp32": agreement,
                    "vs_fp16": quantization_error,
                    "wall_ms": _measure(call, args.warmup, args.iterations),
                }
            )

        calls = [(scheme, dispatch)]
        if scheme == "affine4" and args.fallbacks:
            packed_keys, packed_values = (
                cache._unwrap(key_state),
                cache._unwrap(value_state),
            )

            def rotated(
                dtype, reference=False, cache=cache, ks=packed_keys, vs=packed_values
            ):
                if reference:
                    rk = cache.key_codec._codes(ks) * ks.norms[..., None]
                    rv = cache.value_codec._codes(vs) * vs.norms[..., None]
                else:
                    rk = cache.key_codec.dequantize_rotated(ks)
                    rv = cache.value_codec.dequantize_rotated(vs)
                output = mx.fast.scaled_dot_product_attention(
                    cache.key_codec.prepare_queries(queries).astype(dtype),
                    rk.astype(dtype),
                    rv.astype(dtype),
                    scale=scale,
                    mask=mask,
                )
                return cache.value_codec._rotate_inverse(
                    output.astype(mx.float32)
                ).astype(queries.dtype)

            def dequant_fp16(cache=cache, key_state=key_state, value_state=value_state):
                restored_keys, restored_values = cache.dequantize(
                    key_state, value_state
                )
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    restored_keys.astype(queries.dtype),
                    restored_values.astype(queries.dtype),
                    scale=scale,
                    mask=mask,
                )

            calls.extend(
                [
                    ("affine4_portable_dispatch", dispatch),
                    ("affine4_portable_reference", partial(rotated, mx.float32, True)),
                    ("affine4_rotated_fp16", partial(rotated, mx.float16)),
                    ("affine4_dequant_fp16", dequant_fp16),
                ]
            )
        for method, call in calls[::order]:
            context = (
                patch.object(affine4, "_m5_mpp_available", lambda: False)
                if method == "affine4_portable_dispatch"
                else nullcontext()
            )
            with context:
                record(method, call, observe_native=method == "affine4")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument(
        "--dims", type=int, nargs="+", choices=[128, 256], default=[128, 256]
    )
    parser.add_argument(
        "--query-lengths", type=int, nargs="+", choices=[1, 4], default=[1, 4]
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fallbacks",
        action="store_true",
        help="Compare optimized/reference portable paths, rotated FP16, and dequant+FP16",
    )
    parser.add_argument(
        "--order",
        choices=["forward", "reverse"],
        default="forward",
        help="Order of geometries and attention methods",
    )
    args = parser.parse_args()
    if (
        args.warmup < 1
        or args.iterations < 1
        or min(args.tokens) < max(args.query_lengths)
    ):
        parser.error(
            "warmup/iterations must be positive and tokens must cover every query row"
        )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import mlx.core as mx

    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch

    logging.basicConfig(level=logging.WARNING)
    apply_turboquant_attention_patch()
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "benchmarks/bench_affine4.py",
            "omlx/affine4.py",
            "omlx/patches/turboquant_attention.py",
            "omlx/turboquant_kv.py",
        )
    }
    cases = [
        (dim, tokens, query_length)
        for dim in args.dims
        for tokens in args.tokens
        for query_length in args.query_lengths
    ]
    results = [
        row
        for dim, tokens, query_length in cases[
            :: (-1 if args.order == "reverse" else 1)
        ]
        for row in _case(dim, {128: 16, 256: 24}[dim], tokens, query_length, args)
    ]
    print(
        json.dumps(
            {
                "scope": "attention-only",
                "mlx_version": mx.__version__,
                "dependencies": {name: version(name) for name in ("mlx-lm", "mlx-vlm")},
                "python_version": sys.version.split()[0],
                "source_sha256": source_hashes,
                "device": mx.device_info(),
                "dtype": "float16",
                "timed_region": "attention dispatch + eval + synchronize; cache construction excluded",
                "options": vars(args),
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if all(row["vs_dequant_fp32"]["finite"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
