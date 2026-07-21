"""Bonsai t5 profiler — diagnose 1.585-bit vs 2-bit decode throughput.

Usage:
    python tools/bonsai_bench.py [--reps 50]

Reports per-kernel DRAM bandwidth, time per layer, and estimated tok/s for
all Bonsai-27B projection shapes.  Prints a diagnosis of where time is spent.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Model dimensions for Ternary-Bonsai-27B / Qwen3.5-27B (text part)
# ---------------------------------------------------------------------------
# hidden_size=5120, intermediate_size=17408, group_size=128 → bpg=26
# 64 layers: 48 linear_attention + 16 full_attention (every 4th)

HIDDEN = 5120
INTER  = 17408
GS     = 128  # group_size

# (label, N, K)  — weight is (N, K), input is (..., K) → output (..., N)
PROJECTIONS = [
    # linear attention (48 layers)
    ("lin.in_proj_qkv",  10240, HIDDEN),   # 48 × per layer
    ("lin.in_proj_z",     6144, HIDDEN),
    ("lin.out_proj",     HIDDEN,  6144),
    ("lin.gate_proj",    INTER,  HIDDEN),
    ("lin.up_proj",      INTER,  HIDDEN),
    ("lin.down_proj",   HIDDEN,   INTER),
    # full attention (16 layers)
    ("attn.q_proj",      6144, HIDDEN),    # 16 × per layer
    ("attn.k_proj",      1024, HIDDEN),
    ("attn.v_proj",      1024, HIDDEN),
    ("attn.o_proj",     HIDDEN,  6144),
    ("attn.gate_proj",   INTER,  HIDDEN),
    ("attn.up_proj",     INTER,  HIDDEN),
    ("attn.down_proj",  HIDDEN,   INTER),
]

# Number of each layer type
N_LIN  = 48
N_FULL = 16
LIN_LAYERS  = {"lin.in_proj_qkv", "lin.in_proj_z", "lin.out_proj",
               "lin.gate_proj", "lin.up_proj", "lin.down_proj"}
FULL_LAYERS = {"attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
               "attn.gate_proj", "attn.up_proj", "attn.down_proj"}


# ---------------------------------------------------------------------------
# Weight generation helpers
# ---------------------------------------------------------------------------

def _make_t5_weight(N: int, K: int, gs: int = GS) -> tuple[mx.array, mx.array]:
    """Random uint8 t5 weight + float16 scales."""
    bpg = 26 if gs == 128 else 13  # bytes per group (ceil(gs/5))
    n_groups = K // gs
    w = mx.array(np.random.randint(0, 243, (N, n_groups * bpg), dtype=np.uint8))
    sc = mx.array(np.random.randn(N, n_groups).astype(np.float16))
    return w, sc


def _make_q2_weight(N: int, K: int, gs: int = GS) -> tuple[mx.array, mx.array, mx.array]:
    """Random uint32 2-bit affine weight + float16 scales + biases."""
    n_groups = K // gs
    w = mx.array(np.random.randint(0, 2**32, (N, K // 16), dtype=np.uint32))
    sc = mx.array(np.random.randn(N, n_groups).astype(np.float16))
    bi = mx.array(np.random.randn(N, n_groups).astype(np.float16))
    return w, sc, bi


def _make_input(M: int, K: int) -> mx.array:
    return mx.array(np.random.randn(M, K).astype(np.float16))


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def bench(fn: Callable, reps: int = 40, warmup: int = 5) -> float:
    """Return median wall-clock ms per call after warmup."""
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times))


# ---------------------------------------------------------------------------
# DRAM bytes accounting
# ---------------------------------------------------------------------------

def t5_bytes(N: int, K: int, gs: int = GS) -> int:
    bpg = 26 if gs == 128 else 13
    n_groups = K // gs
    w_bytes  = N * n_groups * bpg       # uint8 weights
    sc_bytes = N * n_groups * 2         # float16 scales
    act_bytes = K * 2                   # float16 activation row (M=1)
    return w_bytes + sc_bytes + act_bytes


def q2_bytes(N: int, K: int, gs: int = GS) -> int:
    n_groups = K // gs
    w_bytes   = N * (K // 16) * 4      # uint32 weights
    sc_bytes  = N * n_groups * 2        # float16 scales
    bi_bytes  = N * n_groups * 2        # float16 biases
    act_bytes = K * 2                   # float16 activation row (M=1)
    return w_bytes + sc_bytes + bi_bytes + act_bytes


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

@dataclass
class Result:
    label: str
    N: int
    K: int
    M: int
    kernel: str
    ms: float
    bw_gbs: float    # effective DRAM bandwidth (GB/s)
    bytes_read: int


def run_benchmark(reps: int = 40) -> list[Result]:
    from omlx.custom_kernels.bonsai.fast import (
        bonsai_t5_qmv, bonsai_t5_qmv_wide,
        bonsai_q2_affine_qmv, bonsai_qmv_wide,
        has_native, is_nax_available, _arch_gen,
    )

    if not has_native():
        print("ERROR: native bonsai extension not loaded — rebuild first.")
        print("  cd /path/to/bonsai/csrc && cmake ... && cmake --build .")
        sys.exit(1)

    gen = _arch_gen()
    nax = is_nax_available()
    print(f"GPU: Apple gen-{gen}  NAX={nax}  native=True")
    print()

    results: list[Result] = []

    for label, N, K in PROJECTIONS:
        # --- t5 M=1 ---
        wt, sct = _make_t5_weight(N, K)
        x1 = _make_input(1, K)
        sc1 = sct.astype(mx.float16)
        ms_t5_1 = bench(lambda: bonsai_t5_qmv(x1, wt, sc1), reps=reps)
        nb = t5_bytes(N, K)
        results.append(Result(label, N, K, 1, "t5_qmv",  ms_t5_1, nb / ms_t5_1 / 1e6, nb))

        # --- q2 M=1 ---
        wq, scq, biq = _make_q2_weight(N, K)
        scq1 = scq.astype(mx.float16)
        biq1 = biq.astype(mx.float16)
        ms_q2_1 = bench(lambda: bonsai_q2_affine_qmv(x1, wq, scq1, biq1), reps=reps)
        nb2 = q2_bytes(N, K)
        results.append(Result(label, N, K, 1, "q2_affine_qmv", ms_q2_1, nb2 / ms_q2_1 / 1e6, nb2))

        # --- t5 M=5 ---
        x5 = _make_input(5, K)
        sc5 = sct.astype(mx.float16)
        ms_t5_5 = bench(lambda: bonsai_t5_qmv_wide(x5, wt, sc5), reps=reps)
        nb5 = t5_bytes(N, K) + (K * 2 * 4)  # 5 activation rows
        results.append(Result(label, N, K, 5, "t5_qmv_wide", ms_t5_5, nb5 / ms_t5_5 / 1e6, nb5))

        # --- q2 M=5 ---
        x5q = _make_input(5, K)
        ms_q2_5 = bench(lambda: bonsai_qmv_wide(x5q, wq, scq1, biq1, bits=2), reps=reps)
        nb2_5 = q2_bytes(N, K) + (K * 2 * 4)
        results.append(Result(label, N, K, 5, "q2_wide", ms_q2_5, nb2_5 / ms_q2_5 / 1e6, nb2_5))

    return results


def summarize(results: list[Result]) -> None:
    print(f"{'Layer':<22} {'N':>6} {'K':>6} {'M':>2}  {'Kernel':<18} {'ms':>7}  {'BW GB/s':>9}")
    print("-" * 80)

    t5_1_rows: list[Result] = []
    q2_1_rows: list[Result] = []

    for r in results:
        if r.M == 1:
            print(f"{r.label:<22} {r.N:>6} {r.K:>6} {r.M:>2}  {r.kernel:<18} {r.ms:>7.3f}  {r.bw_gbs:>9.1f}")
            if r.kernel == "t5_qmv":
                t5_1_rows.append(r)
            elif r.kernel == "q2_affine_qmv":
                q2_1_rows.append(r)

    print()
    print("M=5 (small-batch / MTP verify):")
    print(f"{'Layer':<22} {'N':>6} {'K':>6} {'M':>2}  {'Kernel':<18} {'ms':>7}  {'BW GB/s':>9}")
    print("-" * 80)
    for r in results:
        if r.M == 5:
            print(f"{r.label:<22} {r.N:>6} {r.K:>6} {r.M:>2}  {r.kernel:<18} {r.ms:>7.3f}  {r.bw_gbs:>9.1f}")

    print()
    print("=" * 80)

    # Per-layer ratio
    t5_by_lk  = {(r.label, r.K): r for r in t5_1_rows}
    q2_by_lk  = {(r.label, r.K): r for r in q2_1_rows}
    print("Per-layer t5/q2 ratio (M=1):")
    for k, t5r in t5_by_lk.items():
        q2r = q2_by_lk.get(k)
        if q2r:
            ratio = t5r.ms / q2r.ms
            flag = "FASTER" if ratio < 1 else f"{ratio:.2f}x slower"
            print(f"  {t5r.label:<22} N={t5r.N:>6}  t5={t5r.ms:.3f}ms  q2={q2r.ms:.3f}ms  → {flag}")

    print()

    # Weighted total (account for number of each layer type)
    t5_1_ms = sum(r.ms * (N_LIN if r.label in LIN_LAYERS else N_FULL) for r in t5_1_rows)
    q2_1_ms = sum(r.ms * (N_LIN if r.label in LIN_LAYERS else N_FULL) for r in q2_1_rows)
    ratio_1  = t5_1_ms / q2_1_ms if q2_1_ms else float("inf")

    print("NOTE: These timings include per-kernel GPU sync overhead from mx.eval().")
    print("Real inference pipelines many kernels before a sync — actual ratios are more accurate.")
    print()
    print(f"Kernel-isolated timing ratio  t5/q2  (M=1):  {ratio_1:.2f}x")
    print()

    # Effective BW
    t5_bws = [r.bw_gbs for r in t5_1_rows]
    q2_bws = [r.bw_gbs for r in q2_1_rows]
    print(f"Mean effective BW  t5_qmv:        {float(np.mean(t5_bws)):.1f} GB/s")
    print(f"Mean effective BW  q2_affine_qmv: {float(np.mean(q2_bws)):.1f} GB/s")
    print()
    if ratio_1 < 1.15:
        print("✓ t5 within 15% of 2-bit — bandwidth-limited as expected.")
    elif ratio_1 < 1.5:
        print("~ t5 ~1.2–1.5x slower per kernel — magic divmod-3 ALU overhead visible.")
        print("  In real inference the gap narrows because non-kernel work (attn, KV, norms)")
        print("  is shared. Expect real-world t5 to be within 20–30% of 2-bit decode speed.")
    elif ratio_1 < 2.0:
        print("~ t5 kernel is measurably slower (1.5–2×). Likely ALU-bound on divmod chain.")
        print("  Check kernel threadgroup size / occupancy in Metal Frame Debugger.")
    else:
        print("✗ t5 significantly slower (>2×). Suspect kernel correctness or routing issue.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=40, help="Benchmark repetitions (default: 40)")
    args = ap.parse_args()

    print(f"Bonsai t5 vs 2-bit decode profiler  (reps={args.reps})")
    print(f"GPU: {mx.device_info().get('architecture', 'unknown')}")
    print()

    results = run_benchmark(reps=args.reps)
    summarize(results)


if __name__ == "__main__":
    main()
