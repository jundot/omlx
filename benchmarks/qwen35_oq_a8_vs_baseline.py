#!/usr/bin/env python3
"""Compare the oQ A8 kernels against the W4/W5A16 paths that already ship.

The A8 path quantizes activations, which changes inference numerics. That cost
is only worth paying if it buys throughput over what oMLX already runs, so
this is the measurement that decides whether to ship, not the raw-ceiling
comparison, which flatters A8 by ignoring the incumbent.

Four paths on identical shapes in one session:

    mlx qmm    stock mx.quantized_matmul (uses NAX on M5)
    omlx nax   oMLX's own affine NAX kernel (qwen35_qmm_nax.metal)
    a8 dec     oQ A8 with the register weight decoder
    a8 i4      oQ A8 Q4 with no decode at all (int8 x int4b_format)

    PYTHONPATH=. python benchmarks/qwen35_oq_a8_vs_baseline.py
"""


import sys
import time

sys.path.insert(0, "/Users/power_spy/Coding/omlx")

import mlx.core as mx
import numpy as np

from omlx.custom_kernels.qwen35_prefill import fast

GS = 64
SHAPES = [
    ("mlp_gate_up", 5120, 17408, 4),
    ("mlp_down_late", 17408, 5120, 4),
    ("mlp_down_early", 17408, 5120, 5),
    ("linear_attn_out", 6144, 5120, 5),
]
M = 2048
DTYPE = mx.bfloat16


def timeit(fn, warmup=3, iters=15):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t) / iters


def tops(m, n, k, s):
    return 2.0 * m * n * k / s / 1e12


print(f"M={M} dtype={DTYPE}  (TOP/s, higher is better)\n")
hdr = f"{'shape':<17}{'bits':>5}{'mlx qmm':>10}{'omlx nax':>10}{'a8 dec':>9}{'a8 i4':>8}"
print(hdr)
print("-" * len(hdr))

rows = []
for name, K, N, bits in SHAPES:
    rng = np.random.default_rng(0)
    w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.05, dtype=DTYPE)
    packed, scales, biases = mx.quantize(w, group_size=GS, bits=bits, mode="affine")
    x = mx.array((rng.standard_normal((M, K)) * 0.5).astype(np.float32), DTYPE)
    mx.eval(packed, scales, biases, x)

    # 1. Stock MLX quantized_matmul (uses NAX on M5).
    t_mlx = timeit(lambda: mx.quantized_matmul(
        x, packed, scales, biases, transpose=True,
        group_size=GS, bits=bits, mode="affine"))

    # 2. oMLX's own NAX affine kernel (qwen35_qmm_nax.metal).
    fn = getattr(fast, f"qwen35_q{bits}_affine_qmm_t")
    try:
        t_omlx = timeit(lambda: fn(x, packed, scales, biases, group_size=GS))
    except Exception as e:
        print(f"  {name}: omlx nax failed: {e}")
        t_omlx = float("inf")

    # 3. oQ A8, decoding path.
    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, 0)
    mx.eval(qa, sa, ra)
    t_dec = timeit(lambda: fast.qwen35_oq_a8_qmm_t(
        qa, sa, ra, packed, scales, biases, bits, 0, 2))

    # 4. oQ A8, decode-free Q4.
    if bits == 4:
        w4, s4, b4 = fast.qwen35_oq_a8_prepare_q4(packed, scales, biases)
        mx.eval(w4, s4, b4)
        t_i4 = timeit(lambda: fast.qwen35_oq_a8_i4_qmm_t(qa, sa, ra, w4, s4, b4, 0, 0))
    else:
        t_i4 = None

    r = dict(
        name=name, bits=bits, K=K, N=N,
        mlx=tops(M, N, K, t_mlx),
        omlx=tops(M, N, K, t_omlx),
        dec=tops(M, N, K, t_dec),
        i4=tops(M, N, K, t_i4) if t_i4 else None,
    )
    rows.append(r)
    print(f"{name:<17}{bits:>5}{r['mlx']:>10.2f}{r['omlx']:>10.2f}"
          f"{r['dec']:>9.2f}{(f'{r[chr(105)+chr(52)]:.2f}' if r['i4'] else '-'):>8}")

print()
best_existing = max(max(r["mlx"], r["omlx"]) for r in rows)
best_a8 = max(max(r["dec"], r["i4"] or 0) for r in rows)
print(f"best existing W4/W5A16 path : {best_existing:.2f} TOP/s")
print(f"best oQ A8 path             : {best_a8:.2f} TOP/s")
if best_a8 > best_existing:
    print(f"A8 is {best_a8 / best_existing:.2f}x the existing path")
else:
    print(f"A8 is {best_a8 / best_existing:.2f}x -- SLOWER than what already ships")
