#!/usr/bin/env python3
"""Bonsai 1-bit / 2-bit qmv decode microbenchmark.

Measures achieved DRAM bandwidth (GB/s) and latency (µs) for each kernel
variant across Bonsai-27B projection shapes, batch sizes M ∈ {1,2,3,4,5},
bits ∈ {1,2}, and group sizes ∈ {64,128}.

Usage
-----
    python benchmarks/bonsai_decode_bench.py [--M 1,2,3,4,5] [--bits 1,2]
                                              [--gs 64,128] [--iters 100]
                                              [--warmup 10] [--dtype fp16]

Results are printed as a markdown table.  Pass --csv to emit CSV instead.

Bandwidth accounting
--------------------
Bytes streamed per qmv call:
  weights:  N * K * bits / 8
  scales:   N * (K // group_size) * sizeof(T)
  biases:   N * (K // group_size) * sizeof(T)  (0 for sym variants)
  x:        M * K * sizeof(T)
  y:        M * N * sizeof(T)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

# ---------------------------------------------------------------------------
# Bonsai fast import
# ---------------------------------------------------------------------------

try:
    import omlx.custom_kernels.bonsai.fast as bf
    _NATIVE = bf.has_native()
except ImportError:
    bf = None  # type: ignore[assignment]
    _NATIVE = False

# ---------------------------------------------------------------------------
# t5 tensor factory (base-3 ternary, I-D)
# ---------------------------------------------------------------------------

try:
    from tools.repack_ternary_t5 import pack_t5 as _pack_t5
    _HAS_T5_REPACK = True
except ImportError:
    _HAS_T5_REPACK = False

# ---------------------------------------------------------------------------
# Projection shapes for Qwen3.6-27B (Bonsai-27B base)
# ---------------------------------------------------------------------------

SHAPES_27B = [
    # (name,         N,     K)
    ("q_proj",     8192,  7168),
    ("k_proj",     1024,  7168),
    ("v_proj",     1024,  7168),
    ("o_proj",     7168,  8192),
    ("gate_proj", 22016,  7168),
    ("up_proj",   22016,  7168),
    ("down_proj",  7168, 22016),
]

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}
DTYPE_BYTES = {mx.float16: 2, mx.bfloat16: 2, mx.float32: 4}


# ---------------------------------------------------------------------------
# Tensor factories
# ---------------------------------------------------------------------------

def make_1bit_tensors(M: int, N: int, K: int, group_size: int, dtype: mx.Dtype):
    """MLX uint32 1-bit packing: 32 values per uint32."""
    x      = mx.random.normal((M, K)).astype(dtype)
    w      = mx.zeros((N, K // 32), dtype=mx.uint32)
    n_g    = K // group_size
    scales = mx.ones((N, n_g), dtype=dtype)
    biases = -scales * 0.5  # symmetric Bonsai layout
    return x, w, scales, biases


def make_2bit_tensors(M: int, N: int, K: int, group_size: int, dtype: mx.Dtype):
    """MLX uint32 2-bit packing: 16 values per uint32."""
    x      = mx.random.normal((M, K)).astype(dtype)
    w      = mx.zeros((N, K // 16), dtype=mx.uint32)
    n_g    = K // group_size
    scales = mx.ones((N, n_g), dtype=dtype)
    biases = -scales  # symmetric Bonsai ternary layout
    return x, w, scales, biases


def make_t5_tensors(M: int, N: int, K: int, group_size: int, dtype: mx.Dtype):
    """t5 base-3 ternary packing: ceil(group_size/5) uint8 bytes per group."""
    import numpy as np
    x      = mx.random.normal((M, K)).astype(dtype)
    n_g    = K // group_size
    scales = mx.ones((N, n_g), dtype=dtype)
    if _HAS_T5_REPACK:
        rng    = np.random.default_rng(0)
        quants = rng.integers(0, 3, size=(N, K), dtype=np.uint8)
        w_np   = _pack_t5(quants, group_size)
        w      = mx.array(w_np)
    else:
        bpg = (group_size + 4) // 5
        w   = mx.zeros((N, n_g * bpg), dtype=mx.uint8)
    return x, w, scales


# ---------------------------------------------------------------------------
# Bandwidth calculation
# ---------------------------------------------------------------------------

def bytes_streamed(
    M: int, N: int, K: int, group_size: int, bits: int,
    dtype: mx.Dtype, symmetric: bool = False, is_t5: bool = False,
) -> int:
    import math
    T = DTYPE_BYTES[dtype]
    n_g = K // group_size
    if is_t5:
        # t5: ceil(group_size/5) bytes per group, no biases (always symmetric)
        bpg = math.ceil(group_size / 5)
        w_bytes    = N * n_g * bpg
        bias_bytes = 0
    else:
        w_bytes    = N * K * bits // 8
        bias_bytes = 0 if symmetric else N * n_g * T
    scale_bytes = N * n_g * T
    x_bytes     = M * K * T
    y_bytes     = M * N * T
    return w_bytes + scale_bytes + bias_bytes + x_bytes + y_bytes


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_fn(fn: Callable, warmup: int, iters: int) -> float:
    """Return mean wall time in seconds over `iters` iterations."""
    # Warm-up (shader compile + cache fill)
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


# ---------------------------------------------------------------------------
# Kernel variants
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    name: str
    bits: int
    requires_native: bool = True
    symmetric: bool = False
    is_t5: bool = False  # base-3 ternary format (I-D)


def get_variants(bits: int) -> list[Variant]:
    variants = []
    if bits == 1:
        variants += [
            Variant("q1_fast",       1, requires_native=True,  symmetric=False),
            Variant("q1_fast_sym",   1, requires_native=True,  symmetric=True),
            Variant("q1_wide",       1, requires_native=True,  symmetric=False),
            Variant("q1_wide_sym",   1, requires_native=True,  symmetric=True),
            Variant("mlx_fallback",  1, requires_native=False, symmetric=False),
        ]
    else:
        variants += [
            Variant("q2_fast",       2, requires_native=True,  symmetric=False),
            Variant("q2_fast_sym",   2, requires_native=True,  symmetric=True),
            Variant("q2_wide",       2, requires_native=True,  symmetric=False),
            Variant("q2_wide_sym",   2, requires_native=True,  symmetric=True),
            Variant("t5_fast",       2, requires_native=True,  symmetric=True,  is_t5=True),
            Variant("t5_wide",       2, requires_native=True,  symmetric=True,  is_t5=True),
            Variant("mlx_fallback",  2, requires_native=False, symmetric=False),
        ]
    return variants


def call_variant(v: Variant, x, w, scales, biases, M: int) -> mx.array | None:
    if not _NATIVE and v.requires_native:
        return None
    if bf is None:
        return None

    # t5 variants: no biases, different weight format
    if v.is_t5:
        wide = "wide" in v.name and M >= 3 and bf._use_qmv_wide(2, M)
        fn_name = "bonsai_t5_qmv_wide" if wide else "bonsai_t5_qmv"
        if not bf.has_symbol(fn_name):
            return None
        fn = getattr(bf, fn_name)
        try:
            return fn(x, w, scales)
        except Exception:
            return None

    if v.name.startswith("q1_fast"):
        fn = bf.bonsai_q1_affine_qmv_sym if v.symmetric else bf.bonsai_q1_affine_qmv
        if not bf.has_symbol(fn.__name__.split(".")[-1]):
            return None
        return fn(x, w, scales, biases)

    elif v.name.startswith("q1_wide"):
        sym_name = "bonsai_q1_affine_qmv_wide_sym"
        aff_name = "bonsai_q1_affine_qmv_wide"
        if v.symmetric:
            if not bf.has_symbol(sym_name):
                return None
            return bf.bonsai_q1_affine_qmv_wide_sym(x, w, scales, biases)
        else:
            if not bf.has_symbol(aff_name):
                return None
            return bf.bonsai_q1_affine_qmv_wide(x, w, scales, biases)

    elif v.name.startswith("q2_fast"):
        fn = bf.bonsai_q2_affine_qmv_sym if v.symmetric else bf.bonsai_q2_affine_qmv
        sym_name = "bonsai_q2_affine_qmv_sym"
        aff_name = "bonsai_q2_affine_qmv"
        if v.symmetric and not bf.has_symbol(sym_name):
            return None
        if not v.symmetric and not bf.has_symbol(aff_name):
            return None
        return fn(x, w, scales, biases)

    elif v.name.startswith("q2_wide"):
        sym_name = "bonsai_q2_affine_qmv_wide_sym"
        aff_name = "bonsai_q2_affine_qmv_wide"
        if v.symmetric and not bf.has_symbol(sym_name):
            return None
        if not v.symmetric and not bf.has_symbol(aff_name):
            return None
        return (bf.bonsai_q2_affine_qmv_wide_sym if v.symmetric else bf.bonsai_q2_affine_qmv_wide)(
            x, w, scales, biases
        )

    elif v.name == "mlx_fallback":
        gs = w.shape[-1] * (32 // v.bits) // (scales.shape[-1])
        return mx.quantized_matmul(
            x, w, scales=scales, biases=biases,
            transpose=True, group_size=gs, bits=v.bits,
        )

    return None


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------

@dataclass
class Row:
    layer: str
    N: int
    K: int
    M: int
    bits: int
    gs: int
    variant: str
    us: float
    gbps: float
    note: str = ""


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_bench(
    M_values: list[int],
    bits_values: list[int],
    gs_values: list[int],
    dtype: mx.Dtype,
    warmup: int,
    iters: int,
    shapes: list[tuple[str, int, int]],
) -> list[Row]:
    rows: list[Row] = []

    for bits in bits_values:
        make_fn = make_1bit_tensors if bits == 1 else make_2bit_tensors
        for gs in gs_values:
            for M in M_values:
                for name, N, K in shapes:
                    if K % gs != 0 or N % 64 != 0:
                        continue

                    x, w, scales, biases = make_fn(M, N, K, gs, dtype)
                    mx.eval(x, w, scales, biases)

                    # t5 tensors (shared across t5 variants for this shape)
                    t5_tensors = None

                    for v in get_variants(bits):
                        # Skip wide variants for M < 3 (not instantiated for M=1,2
                        # in the wide path; fast is used instead)
                        if "wide" in v.name and M < 2:
                            continue

                        # t5 variants need their own weight tensor
                        if v.is_t5:
                            if not _HAS_T5_REPACK and bf is None:
                                continue
                            if t5_tensors is None:
                                t5x, t5w, t5sc = make_t5_tensors(M, N, K, gs, dtype)
                                mx.eval(t5x, t5w, t5sc)
                                t5_tensors = (t5x, t5w, t5sc)
                            t5x, t5w, t5sc = t5_tensors
                            out = call_variant(v, t5x, t5w, t5sc, None, M)
                        else:
                            out = call_variant(v, x, w, scales, biases, M)
                        if out is None:
                            continue

                        # Check if this variant is available (not just falling back)
                        try:
                            mx.eval(out)
                        except Exception as e:
                            rows.append(Row(name, N, K, M, bits, gs, v.name, 0, 0, f"ERROR: {e}"))
                            continue

                        bw = bytes_streamed(M, N, K, gs, bits, dtype, v.symmetric, v.is_t5)

                        if v.is_t5:
                            _t5x, _t5w, _t5sc = t5_tensors  # type: ignore[misc]
                            def fn(v=v, _x=_t5x, _w=_t5w, _sc=_t5sc, M=M):
                                return call_variant(v, _x, _w, _sc, None, M)
                        else:
                            def fn(v=v, x=x, w=w, scales=scales, biases=biases, M=M):
                                return call_variant(v, x, w, scales, biases, M)

                        try:
                            t = time_fn(fn, warmup, iters)
                        except Exception as e:
                            rows.append(Row(name, N, K, M, bits, gs, v.name, 0, 0, f"ERROR: {e}"))
                            continue

                        rows.append(Row(
                            layer=name, N=N, K=K, M=M, bits=bits, gs=gs,
                            variant=v.name,
                            us=t * 1e6,
                            gbps=bw / t / 1e9,
                        ))

    return rows


# ---------------------------------------------------------------------------
# Dispatch overhead measurement
# ---------------------------------------------------------------------------

def measure_dispatch_overhead(
    dtype: mx.Dtype, warmup: int, iters: int,
) -> None:
    """Measure Python overhead of the patched QuantizedLinear.__call__.

    A Qwen3.6-27B decode step makes ~448 calls (64 blocks × 7 projections).
    This test creates a single representative quantized layer and measures:
    (a) patched call time, (b) raw C++ kernel time, (c) Python no-op overhead.
    """
    import math
    from omlx.patches.bonsai_qmv import _is_symmetric, _is_t5_format

    T = DTYPE_BYTES[dtype]

    # Representative shape: o_proj (7168×8192) with group_size=128, bits=2
    N, K, gs = 7168, 8192, 128
    M = 1

    # Create a QuantizedLinear with our construct patch active
    from omlx.patches.bonsai_qmv import apply_bonsai_construct_patch
    apply_bonsai_construct_patch()

    from mlx.nn import QuantizedLinear
    layer = QuantizedLinear(K, N, bias=False, group_size=gs, bits=2)
    import numpy as np
    import mlx.core as mx
    layer.weight = mx.array(np.random.randint(0, 4, (N, K // 16), dtype=np.uint32))
    layer.scales = mx.array(np.random.randn(N, K // gs).astype(np.float16).__abs__())
    layer.biases = mx.array(-np.array(layer.scales, copy=True))

    x = mx.array(np.random.randn(M, K).astype(np.float16))

    # (a) Full patched call
    def patched_call():
        return layer(x)

    mx.eval(patched_call())  # warmup compile
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(patched_call())
    mx.synchronize()
    t_patched = (time.perf_counter() - t0) / iters

    # (b) Raw C++ kernel (bypassing the patch)
    from omlx.custom_kernels.bonsai.fast import bonsai_q2_affine_qmv_sym
    sym = _is_symmetric(layer, 2)

    w, sc, bi = layer.weight, layer.scales, layer.biases
    if sym:
        def raw_call():
            return bonsai_q2_affine_qmv_sym(x, w, sc, bi)
    else:
        def raw_call():
            return bonsai_q2_affine_qmv(x, w, sc, bi)

    mx.eval(raw_call())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(raw_call())
    mx.synchronize()
    t_raw = (time.perf_counter() - t0) / iters

    # (c) No-op Python overhead: just the branch/getattr logic, no kernel
    sym_cache = getattr(layer, "_bonsai_sym_cache", None)
    bits = layer.bits

    def noop_dispatch():
        nonlocal sym_cache
        m = bits
        if m != 2: return
        s = getattr(layer, "_bonsai_sym_cache", None)
        if s is None:
            s = _is_symmetric(layer, bits)
        _is_t5_format(layer)  # forces the uint8 check
        # No kernel call — just the Python overhead

    t0 = time.perf_counter()
    for _ in range(iters):
        noop_dispatch()
    t_noop = (time.perf_counter() - t0) / iters

    # (d) Estimate per-token overhead for 448 calls
    per_call_overhead = t_patched - t_raw
    per_token_448 = per_call_overhead * 448 * 1e6

    print(f"\n--- Dispatch Overhead (warmup={warmup}, iters={iters}, dtype={dtype}) ---")
    print(f"  (a) Patched __call__ : {t_patched*1e6:8.1f} µs")
    print(f"  (b) Raw C++ kernel   : {t_raw*1e6:8.1f} µs")
    print(f"  (c) No-op dispatch   : {t_noop*1e6:8.1f} µs")
    print(f"  overhead per call    : {per_call_overhead*1e6:8.1f} µs")
    print(f"  overhead × 448 calls : {per_token_448:8.0f} µs = {per_token_448/1000:.1f} ms/tok")
    print()
    if per_token_448 > 2000:
        print("  → CONFIRMED: dispatch overhead is dominant bottleneck.")
        print("    Load-time specialization (#1 fix) would eliminate this per-call cost.")
    else:
        print("  → Dispatch overhead is minor; bandwidth/compute is the bottleneck.")

def print_markdown(rows: list[Row]) -> None:
    print(f"\n{'layer':<12} {'N':>6} {'K':>6} {'M':>2} {'bits':>4} {'gs':>4} "
          f"{'variant':<18} {'µs':>8} {'GB/s':>8}  note")
    print("-" * 90)
    for r in rows:
        note = f"  {r.note}" if r.note else ""
        print(f"{r.layer:<12} {r.N:>6} {r.K:>6} {r.M:>2} {r.bits:>4} {r.gs:>4} "
              f"{r.variant:<18} {r.us:>8.1f} {r.gbps:>8.1f}{note}")


def print_csv(rows: list[Row]) -> None:
    print("layer,N,K,M,bits,gs,variant,us,gbps,note")
    for r in rows:
        print(f"{r.layer},{r.N},{r.K},{r.M},{r.bits},{r.gs},{r.variant},"
              f"{r.us:.2f},{r.gbps:.2f},{r.note}")


def print_summary(rows: list[Row]) -> None:
    """Print a compact M=1..5 comparison for fast vs wide per bits/gs."""
    print("\n=== wide vs fast speedup (M=3..5, bits=1) ===")
    print(f"{'layer':<12} {'gs':>4}  ", end="")
    for M in (3, 4, 5):
        print(f"  M={M}(fast→wide)", end="")
    print()
    print("-" * 70)

    by_key: dict[tuple, dict[str, float]] = {}
    for r in rows:
        key = (r.layer, r.bits, r.gs, r.M)
        by_key.setdefault(key, {})[r.variant] = r.gbps

    seen: set[tuple[str, int, int]] = set()
    for r in rows:
        if r.bits != 1 or r.M not in (3, 4, 5):
            continue
        k = (r.layer, r.bits, r.gs)
        if k in seen:
            continue
        seen.add(k)
        vals = []
        for M in (3, 4, 5):
            fast = by_key.get((r.layer, r.bits, r.gs, M), {}).get("q1_fast", 0)
            wide = by_key.get((r.layer, r.bits, r.gs, M), {}).get("q1_wide", 0)
            if fast > 0 and wide > 0:
                vals.append(f"  {wide/fast:>5.2f}×")
            else:
                vals.append("     n/a")
        print(f"{r.layer:<12} {r.gs:>4}  {''.join(vals)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--M",      default="1,2,3,4,5",
                   help="batch sizes (comma-separated, default 1,2,3,4,5)")
    p.add_argument("--bits",   default="1,2",
                   help="quantization widths (default 1,2)")
    p.add_argument("--gs",     default="64,128",
                   help="group sizes (default 64,128)")
    p.add_argument("--iters",  type=int, default=100,
                   help="timed iterations per kernel (default 100)")
    p.add_argument("--warmup", type=int, default=10,
                   help="warm-up iterations (default 10)")
    p.add_argument("--dtype",  default="fp16", choices=list(DTYPE_MAP),
                   help="activation dtype (default fp16)")
    p.add_argument("--csv",    action="store_true",
                   help="emit CSV instead of markdown table")
    p.add_argument("--summary", action="store_true",
                   help="print wide-vs-fast speedup summary after table")
    p.add_argument("--layer",  default=None,
                   help="restrict to a specific layer name (e.g. gate_proj)")
    p.add_argument("--dispatch-overhead", action="store_true",
                   help="measure Python dispatch overhead per call (confirms #1 bottleneck)")
    return p.parse_args()


def main():
    args = parse_args()
    M_values   = [int(x) for x in args.M.split(",")]
    bits_values = [int(x) for x in args.bits.split(",")]
    gs_values  = [int(x) for x in args.gs.split(",")]
    dtype      = DTYPE_MAP[args.dtype]

    shapes = SHAPES_27B
    if args.layer:
        shapes = [(n, N, K) for n, N, K in SHAPES_27B if n == args.layer]
        if not shapes:
            print(f"unknown layer '{args.layer}'; choices: {[n for n,_,_ in SHAPES_27B]}")
            sys.exit(1)

    print(f"native ext: {_NATIVE}")
    if _NATIVE and bf is not None:
        print(f"NAX available: {bf.is_nax_available()}")
        arch = mx.device_info().get("architecture", "unknown")
        print(f"GPU arch: {arch}")
    print(f"dtype: {args.dtype}  warmup: {args.warmup}  iters: {args.iters}")
    print(f"M: {M_values}  bits: {bits_values}  group_size: {gs_values}")

    if args.dispatch_overhead:
        measure_dispatch_overhead(dtype, args.warmup, args.iters)
        return

    rows = run_bench(M_values, bits_values, gs_values, dtype, args.warmup, args.iters, shapes)

    if args.csv:
        print_csv(rows)
    else:
        print_markdown(rows)
        if args.summary:
            print_summary(rows)


if __name__ == "__main__":
    main()
