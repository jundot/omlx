# SPDX-License-Identifier: Apache-2.0
"""Stock vs oMLX NAX qmm tiles for Qwen hybrid-layer projections on M5.

The Qwen q4/q8 prefill-linear patch routes affine `QuantizedLinear` prefill
matmuls through the bundled `qwen35_q{bits}_affine_qmm_t` op, which on M5
selects a NAX tile chosen by `OMLX_QWEN35_QMM_NAX_VARIANT` (0 = the tile
stock MLX ships). This benchmark times stock `mx.quantized_matmul` against
each bundled NAX tile variant (``fast.NAX_QMM_VARIANTS``) at the projection shapes of a Qwen4-Exp GDN /
attention layer, so the variant default and the q8 min-token threshold
(`OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS`, 16384) can be set from numbers.

    python benchmarks/bench_qwen35_nax_qmm_variants.py --bits 8 --tokens 2048 8192
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from omlx.custom_kernels.qwen35_prefill import fast

# (name, N, K) — Qwen4-Exp hidden 2560, GDN 48v/16k heads x 128, attn 24q/2kv x 256
SHAPES = [
    ("gdn.in_proj_qkv", 10240, 2560),
    ("gdn.in_proj_z", 6144, 2560),
    ("gdn.out_proj", 2560, 6144),
    ("attn.q_proj", 12288, 2560),
    ("attn.o_proj", 2560, 6144),
]


def _time(fn, iters):
    for _ in range(2):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--tokens", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument(
        "--variants", type=int, nargs="+", default=list(fast.NAX_QMM_VARIANTS)
    )
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--shapes", nargs="+", default=[s[0] for s in SHAPES])
    args = ap.parse_args()

    if not fast.is_native_available():
        raise SystemExit(f"native extension unavailable: {fast.import_error()}")
    native = getattr(fast, f"qwen35_q{args.bits}_affine_qmm_t")
    print(
        f"mlx {mx.__version__}  {mx.device_info().get('device_name')}  "
        f"nax={fast.is_nax_available()} native_nax_qmm={fast._qmm_use_nax()}  "
        f"q{args.bits}/gs{args.group_size}"
    )
    print(
        f"{'shape':>16} {'N':>6} {'K':>5} {'T':>6} {'mode':>10} {'ms':>8} {'GFLOP/s':>9} {'maxerr':>8}"
    )
    for name, n, k in SHAPES:
        if name not in args.shapes:
            continue
        w = (mx.random.normal((n, k), key=mx.random.key(n * k)) * 0.02).astype(
            mx.bfloat16
        )
        wq, sc, bi = mx.quantize(w, group_size=args.group_size, bits=args.bits)
        mx.eval(wq, sc, bi)
        for t in args.tokens:
            x = mx.random.normal((1, t, k), key=mx.random.key(t)).astype(mx.bfloat16)
            mx.eval(x)
            flop = 2.0 * t * n * k

            def stock(x=x, wq=wq, sc=sc, bi=bi):
                return mx.quantized_matmul(
                    x,
                    wq,
                    sc,
                    bi,
                    transpose=True,
                    group_size=args.group_size,
                    bits=args.bits,
                )

            ref = stock()
            mx.eval(ref)
            ms = _time(stock, args.iters)
            print(
                f"{name:>16} {n:>6} {k:>5} {t:>6} {'stock':>10} {ms:>8.2f} {flop / ms / 1e6:>9.0f} {'-':>8}"
            )
            for v in args.variants:
                fast.QMM_NAX_VARIANT = v

                def mine(x=x, wq=wq, sc=sc, bi=bi):
                    return native(x, wq, sc, bi, 8, args.group_size)

                try:
                    out = mine()
                    mx.eval(out)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"{name:>16} {n:>6} {k:>5} {t:>6} {'nax v' + str(v):>10} {'error':>8} {exc}"
                    )
                    continue
                err = (
                    mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
                )
                ms = _time(mine, args.iters)
                print(
                    f"{name:>16} {n:>6} {k:>5} {t:>6} {'nax v' + str(v):>10} {ms:>8.2f} {flop / ms / 1e6:>9.0f} {err:>8.3g}"
                )


if __name__ == "__main__":
    main()
