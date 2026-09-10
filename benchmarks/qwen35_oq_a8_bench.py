#!/usr/bin/env python3
"""Benchmark the oQ mixed-bit QxA8 prefill kernels on M5.

Covers the Q4 and Q5 shape families that dominate Qwen3.8-27B's linear
compute, across the M regimes, with tile autotuning and four separately
reported costs.

    raw QMM              the GEMM alone, activations already quantized
    stage A              activation quantization alone
    amortized            stage A shared across `--share` projections, plus one QMM
    complete projection  stage A + QMM for a projection that shares nothing

Sharing matters: the MLP feeds one activation to gate and up, and a linear
attention block feeds one to qkv/z/a/b, so charging each projection a full
Stage A overstates its cost.

    python benchmarks/qwen35_oq_a8_bench.py
    python benchmarks/qwen35_oq_a8_bench.py --autotune --m 2048
    python benchmarks/qwen35_oq_a8_bench.py --shape mlp_down --bits 5
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

GROUP_SIZE = 64

# Qwen3.8-27B linear shapes, named by the projection family they come from
#. Each entry is (K, N, bits, share) where `share` is how many
# projections reuse one Stage A.
SHAPES = {
    # Q4 carries ~80.2% of transformer linear GEMM work, most of it here.
    "mlp_gate_up": (5120, 17408, 4, 2),
    "linear_attn_qkv": (5120, 10240, 4, 4),
    "mlp_down_late": (17408, 5120, 4, 1),
    "attn_q": (5120, 12288, 4, 1),
    # These three are ~94.9% of the Q5 work.
    "mlp_down_early": (17408, 5120, 5, 1),
    "linear_attn_out": (6144, 5120, 5, 1),
    "linear_attn_z": (5120, 6144, 5, 4),
}

M_REGIMES = (128, 256, 512, 1024, 2048, 4096, 8192)

# Tile variants for the decoding path, mirroring oq_a8_nax_variant() in
# qwen35_oq_a8.cpp.
VARIANTS = {
    0: "bm128 bn64 wm2 wn2",
    1: "bm128 bn128 wm2 wn2",
    2: "bm64 bn64 wm2 wn2",
    3: "bm64 bn128 wm2 wn2",
    4: "bm256 bn64 wm4 wn2",
    5: "bm128 bn64 wm4 wn1",
    6: "bm64 bn64 wm1 wn2",
}

# Tile variants for the decode-free Q4 path, mirroring oq_a8_i4_variant().
# These are (TM, TN, SG): the whole threadgroup cooperates on one matmul.
I4_VARIANTS = {
    0: "tm64 tn64 sg4",
    1: "tm128 tn64 sg4",
    2: "tm128 tn32 sg4",
    3: "tm64 tn32 sg4",
    4: "tm64 tn32 sg2",
    5: "tm32 tn32 sg1",
    6: "tm64 tn32 sg1",
    7: "tm32 tn64 sg1",
}

# Whole-model linear split, per token.
F_Q4_GFLOPS = 39.0557
F_Q5_GFLOPS = 9.6454

# Stop conditions.
TARGETS = {4: (35.0, 38.0), 5: (33.0, 36.0)}


def make_weights(n_out: int, k: int, bits: int, dtype):
    rng = np.random.default_rng(0)
    w = mx.array(rng.standard_normal((n_out, k)).astype(np.float32) * 0.05, dtype=dtype)
    packed, scales, biases = mx.quantize(
        w, group_size=GROUP_SIZE, bits=bits, mode="affine"
    )
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def timeit(fn, warmup: int = 3, iters: int = 20) -> float:
    """Seconds per call, synchronizing so lazy evaluation is not measured away."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - start) / iters


def tops(m: int, n: int, k: int, seconds: float) -> float:
    if seconds <= 0:
        return float("nan")
    return 2.0 * m * n * k / seconds / 1e12


def verdict(bits: int, value: float) -> str:
    good, strong = TARGETS[bits]
    if value >= strong:
        return "strong"
    if value >= good:
        return "good"
    if value >= good - 5.0:
        return "marginal"
    return "below target"


def bench_one(fast, name, m, k, n, bits, share, act_mode, variant, dtype,
              use_i4=False):
    packed, scales, biases = make_weights(n, k, bits, dtype)
    rng = np.random.default_rng(1)
    x = mx.array((rng.standard_normal((m, k)) * 0.5).astype(np.float32), dtype)
    mx.eval(x)

    qa, sa, ra = fast.qwen35_oq_a8_quantize(x, act_mode)
    mx.eval(qa, sa, ra)

    if use_i4:
        # The flip and bias fold are load-time work, not per-call, so they are
        # prepared outside the timed region.
        w4, s4, b4 = fast.qwen35_oq_a8_prepare_q4(packed, scales, biases)
        mx.eval(w4, s4, b4)
        t_qmm = timeit(
            lambda: fast.qwen35_oq_a8_i4_qmm_t(
                qa, sa, ra, w4, s4, b4, act_mode, variant
            )
        )
    else:
        t_qmm = timeit(
            lambda: fast.qwen35_oq_a8_qmm_t(
                qa, sa, ra, packed, scales, biases, bits, act_mode, variant
            )
        )
    t_stage_a = timeit(lambda: fast.qwen35_oq_a8_quantize(x, act_mode))

    return {
        "name": name,
        "m": m,
        "k": k,
        "n": n,
        "bits": bits,
        "variant": variant,
        "path": "i4" if use_i4 else "dec",
        "qmm_ms": t_qmm * 1e3,
        "qmm_tops": tops(m, n, k, t_qmm),
        "stage_a_ms": t_stage_a * 1e3,
        # Stage A is paid once for `share` projections.
        "amortized_tops": tops(m, n, k, t_qmm + t_stage_a / share),
        "projection_tops": tops(m, n, k, t_qmm + t_stage_a),
    }


HEADER = (
    f"{'shape':<18}{'bits':>5}{'M':>7}{'path':>5}{'var':>4}{'qmm ms':>10}"
    f"{'qmm TOP/s':>11}{'A ms':>8}{'amort':>8}{'proj':>8}  verdict"
)


def print_header():
    print(HEADER)
    print("-" * len(HEADER))


def print_rows(rows):
    for r in rows:
        print(
            f"{r['name']:<18}{r['bits']:>5}{r['m']:>7}{r['path']:>5}"
            f"{r['variant']:>4}"
            f"{r['qmm_ms']:>10.3f}{r['qmm_tops']:>11.2f}{r['stage_a_ms']:>8.3f}"
            f"{r['amortized_tops']:>8.2f}{r['projection_tops']:>8.2f}"
            f"  {verdict(r['bits'], r['qmm_tops'])}"
        )


def summarize_model(rows):
    """Project the linear-only prefill ceiling from the measured rates."""
    by_bits = {}
    for r in rows:
        by_bits.setdefault(r["bits"], []).append(r["qmm_tops"])
    if 4 not in by_bits or 5 not in by_bits:
        return

    p_q4 = max(by_bits[4])
    p_q5 = max(by_bits[5])
    # GFLOP / TOP-per-second lands in milliseconds directly.
    t_ms = F_Q4_GFLOPS / p_q4 + F_Q5_GFLOPS / p_q5
    print()
    print(f"best Q4 {p_q4:.2f} TOP/s, best Q5 {p_q5:.2f} TOP/s")
    print(f"linear GEMM time  {t_ms:.3f} ms/token")
    print(f"linear-only ceiling ~{1e3 / t_ms:.0f} prompt tok/s")
    print("(before attention, DeltaNet, conv, norms, Stage A and launch overhead)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", choices=sorted(SHAPES))
    parser.add_argument("--bits", type=int, choices=(4, 5))
    parser.add_argument("--m", type=int, action="append")
    parser.add_argument("--variant", type=int, default=0, choices=sorted(VARIANTS))
    parser.add_argument(
        "--act-mode",
        type=int,
        default=0,
        choices=(0, 1),
        help="0 = per-row scales, 1 = per-group-64 scales",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="sweep every tile variant and report the best per shape",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16"))
    parser.add_argument(
        "--no-i4",
        action="store_true",
        help="keep Q4 on the decoding kernel instead of the decode-free "
             "int8 x int4 path",
    )
    args = parser.parse_args()

    try:
        from omlx.custom_kernels.qwen35_prefill import fast
    except Exception as exc:
        print(f"native extension unavailable: {exc}")
        return 1
    if not fast.oq_a8_available():
        print(
            "oQ A8 NAX kernels are unavailable: this needs an M5-class GPU with "
            "tensor units and a build whose NAX metallib was compiled."
        )
        return 1

    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    shapes = args.shape or sorted(SHAPES)
    # Prefill optimization targets M >= 512; smaller regimes are reported for
    # context only, and decode (M ~ 1) is out of scope.
    ms = args.m or [m for m in M_REGIMES if m >= 512]

    rows = []
    print_header()
    for name in shapes:
        k, n, bits, share = SHAPES[name]
        if args.bits is not None and bits != args.bits:
            continue
        # Q4 goes decode-free by default: the hardware consumes the packed
        # nibbles directly and it measures roughly twice the decoding path.
        use_i4 = bits == 4 and not args.no_i4
        tiles = I4_VARIANTS if use_i4 else VARIANTS
        shape_variants = sorted(tiles) if args.autotune else [
            args.variant if args.variant in tiles else 0
        ]
        for m in ms:
            best = None
            for variant in shape_variants:
                if n % _tile_n(tiles, variant) != 0:
                    continue
                try:
                    row = bench_one(
                        fast, name, m, k, n, bits, share, args.act_mode, variant,
                        dtype, use_i4,
                    )
                except Exception as exc:
                    print(f"  {name} M={m} variant={variant}: {exc}")
                    continue
                if best is None or row["qmm_tops"] > best["qmm_tops"]:
                    best = row
            if best is not None:
                rows.append(best)
                print_rows([best])

    if args.autotune:
        print()
        print("best tile per shape:")
        for r in rows:
            tiles = I4_VARIANTS if r["path"] == "i4" else VARIANTS
            print(f"  {r['name']:<18} M={r['m']:<6} {r['path']} variant "
                  f"{r['variant']} ({tiles[r['variant']]})")

    summarize_model(rows)
    return 0


def _tile_n(tiles: dict, variant: int) -> int:
    """The N extent of a tile, from its description string."""
    return int(tiles[variant].split()[1][2:])


if __name__ == "__main__":
    raise SystemExit(main())
