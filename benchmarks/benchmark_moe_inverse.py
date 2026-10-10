# SPDX-License-Identifier: Apache-2.0
"""Synthetic integer route-inversion benchmark; no model download needed.

Run from the repository: python benchmarks/benchmark_moe_inverse.py
Times inversion alone, not whole-model prefill. Both arms evaluate the exact
same permutation. Includes output equality and interleaved timing.
"""

import argparse
import gc
import json
import statistics
import time

import mlx.core as mx

p = argparse.ArgumentParser()
p.add_argument(
    "--rows", type=int, nargs="+", default=[1024, 4096, 8192, 32768, 131072, 524288]
)
p.add_argument("--repeats", type=int, default=20)
a = p.parse_args()
if a.repeats < 1 or any(n < 1 for n in a.rows):
    p.error("rows and repeats must be positive")
for n in a.rows:
    order = mx.argsort(mx.random.randint(0, 256, (n,), key=mx.random.key(73)))
    mx.eval(order)

    def sort(order=order):
        return mx.argsort(order)

    def scatter(order=order, n=n):
        return mx.put_along_axis(
            mx.zeros_like(order), order, mx.arange(n, dtype=order.dtype), axis=0
        )

    ref, out = sort(), scatter()
    mx.eval(ref, out)
    if not mx.array_equal(ref, out).item():
        raise AssertionError("Inverse permutation mismatch")
    timings = {"sort": [], "scatter": []}
    for names in [("sort", "scatter"), ("scatter", "sort")] * a.repeats:
        for name in names:
            fn = sort if name == "sort" else scatter
            start = time.perf_counter()
            for _ in range(10):
                mx.eval(fn())
            timings[name].append((time.perf_counter() - start) / 10)
    del out
    peak_extra_bytes = {}
    for name, fn in (("sort", sort), ("scatter", scatter)):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        before = mx.get_active_memory()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        peak_extra_bytes[name] = mx.get_peak_memory() - before
        del out
    print(
        json.dumps(
            {
                "rows": n,
                "exact": True,
                "peak_extra_bytes": peak_extra_bytes,
                "median_ms": {
                    k: 1000 * statistics.median(v) for k, v in timings.items()
                },
                "seconds": timings,
            }
        ),
        flush=True,
    )
