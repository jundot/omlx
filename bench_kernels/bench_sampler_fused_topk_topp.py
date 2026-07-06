#!/usr/bin/env python
"""Microbenchmark: old (full-vocab argsort) vs. fused top-k+top-p sampler
fast path, at the real 248,320-token production vocab.

Compares the actual end-to-end swap inside make_sampler for the
top_p+top_k-only case (temp > 0, both filters active, no min_p/xtc):
  - old   : apply_top_p(logprobs, top_p) -> apply_top_k(..., top_k) ->
            categorical_sampling(..., temp)  (make_sampler's composition
            before this change; still reachable via the unfused
            apply_top_p/apply_top_k/categorical_sampling helpers)
  - fused : omlx.utils.sampling._fused_top_k_top_p_sample — argpartition
            down to the top_k candidates, sort/cumsum/mask only those, then
            sample directly and map back through the gathered indices —
            never sorts or scatters at full-vocab scale, and never
            materializes a vocab-sized filtered array at all.

Run with the omlx venv interpreter:
  /Volumes/Untitled/GitHub/omlx/.venv/bin/python bench_kernels/bench_sampler_fused_topk_topp.py

Methodology (matches bench_kernels/bench_head_norm_sampler.py exactly):
  - Decode shapes: logits [B, 248320] for B in {1, 8} (production vocab size,
    from Ornith-1.0-35B-4bit's text_config.vocab_size).
  - Production sampler config: temperature=1.0, top_p=0.95, top_k=20.
  - Warmup 20 iters with mx.eval.
  - Mode A (amortized): K=50 calls chained WITHOUT intermediate eval, input
    varied each iteration so MLX can't collapse identical subgraphs, ONE
    mx.eval on the final outputs at the end; report wall/K ms.
  - time.perf_counter for all wall-clock measurement.
"""

import time

import mlx.core as mx

from omlx.utils.sampling import (
    _fused_top_k_top_p_sample,
    apply_top_k,
    apply_top_p,
    categorical_sampling,
)

VOCAB = 248320
BATCH_SIZES = (1, 8)
WARMUP = 20
K = 50
TOP_P = 0.95
TOP_K = 20
TEMP = 1.0


def old_top_k_top_p_sample(
    logprobs: mx.array, top_k: int, top_p: float, temp: float
) -> mx.array:
    """The pre-fusion pipeline: full-vocab top_p -> full-vocab top_k ->
    categorical_sampling, exactly as make_sampler composed them before."""
    out = apply_top_p(logprobs, top_p)
    out = apply_top_k(out, top_k)
    return categorical_sampling(out, temp)


def time_mode_a(make_input, forward, k=K):
    """Chained, no intermediate eval -- one mx.eval at the end. Returns ms/call."""
    outputs = []
    t0 = time.perf_counter()
    for i in range(k):
        x = make_input(i)
        y = forward(x)
        outputs.append(y)
    mx.eval(outputs)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / k


def warmup(make_input, forward, n=WARMUP):
    for i in range(n):
        mx.eval(forward(make_input(i)))


def bench(make_input, forward):
    warmup(make_input, forward)
    return time_mode_a(make_input, forward)


def make_logits(batch):
    base = mx.random.normal((batch, VOCAB)).astype(mx.bfloat16) * 3.0

    def make_input(i):
        # Vary data each iteration so MLX can't collapse identical subgraphs.
        return base + (i * 1e-6)

    return make_input


def main():
    mx.random.seed(0)
    print(f"vocab={VOCAB} top_p={TOP_P} top_k={TOP_K} temp={TEMP} K={K} warmup={WARMUP}")

    results = []
    for batch in BATCH_SIZES:
        make_input = make_logits(batch)

        ms_old = bench(
            make_input, lambda x: old_top_k_top_p_sample(x, TOP_K, TOP_P, TEMP)
        )
        ms_new = bench(
            make_input, lambda x: _fused_top_k_top_p_sample(x, TOP_K, TOP_P, TEMP)
        )
        speedup = ms_old / ms_new if ms_new else float("inf")

        results.append((batch, ms_old, ms_new, speedup))
        print(
            f"B={batch:2d}  old={ms_old:8.4f} ms  fused={ms_new:8.4f} ms  "
            f"speedup={speedup:5.2f}x"
        )

    print()
    for batch, ms_old, ms_new, speedup in results:
        status = "PASS" if speedup >= 3.0 else "FAIL"
        print(f"[{status}] B={batch}: speedup={speedup:.2f}x (gate: >=3x)")


if __name__ == "__main__":
    main()
