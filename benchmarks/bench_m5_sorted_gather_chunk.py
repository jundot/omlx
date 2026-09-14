# SPDX-License-Identifier: Apache-2.0
"""Per-layer MoE prefill cost vs chunk width on M5 (sorted gather_qmm paths).

Reproduces one routed-expert MoE layer at a configurable geometry (default:
Qwen4-Exp, 512 experts / top-10 / hidden 2560 / inter 640, 4-bit gs64 with
the fused gate+up layout) and times the sorted SwitchGLU path for several
prefill chunk widths under three dispatch modes:

- ``sorted``      raw ``sorted_indices=True`` (NAX rhs kernel; corrupt past
                  32768 rows on mlx <= 0.32.2, timed for reference only)
- ``unsorted``    what the pre-segmenting M5 reroute did past the row cap
- ``segmented``   the oMLX reroute with <=32768-row sorted slices

Prints ms per layer call and derived tokens/s so the chunk-width policy can
be judged on numbers. Run from a checkout with the reroute installed::

    python benchmarks/bench_m5_sorted_gather_chunk.py --chunks 2048 4096 8192
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

import omlx.patches.m5_gather_qmm as reroute


def _build(experts, hidden, inter, bits, group_size, key):
    k1, k2 = mx.random.split(key)
    gate_up = mx.random.normal((experts, 2 * inter, hidden), key=k1) * 0.02
    down = mx.random.normal((experts, hidden, inter), key=k2) * 0.02
    gu = mx.quantize(gate_up.astype(mx.bfloat16), group_size=group_size, bits=bits)
    dn = mx.quantize(down.astype(mx.bfloat16), group_size=group_size, bits=bits)
    mx.eval(*gu, *dn)
    return gu, dn


def _layer(x, inds, gu, dn, bits, group_size, sorted_flag, gather_qmm):
    x5 = mx.expand_dims(x, (-2, -3))
    xs, idx, inv = _gather_sort(x5, inds)
    h = gather_qmm(
        xs,
        *gu,
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        sorted_indices=sorted_flag,
    )
    g, u = mx.split(h, 2, axis=-1)
    a = mx.sigmoid(g) * g * u
    y = gather_qmm(
        a,
        *dn,
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        sorted_indices=sorted_flag,
    )
    return _scatter_unsort(y, inv, inds.shape).squeeze(-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--chunks", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["sorted", "unsorted", "segmented"])
    args = ap.parse_args()

    reroute.apply_m5_gather_qmm_workaround()
    raw = reroute._original_gather_qmm or mx.gather_qmm
    print(
        f"mlx {mx.__version__}  device {mx.device_info().get('device_name')}  "
        f"sorted rhs kernel defective: {reroute._sorted_gather_qmm_defective()}"
    )
    gu, dn = _build(
        args.experts,
        args.hidden,
        args.inter,
        args.bits,
        args.group_size,
        mx.random.key(0),
    )
    print(
        f"geometry: E={args.experts} topk={args.topk} hidden={args.hidden} "
        f"inter={args.inter} q{args.bits}/gs{args.group_size}"
    )
    print(f"{'chunk':>6} {'rows':>7} {'mode':>10} {'ms/layer':>9} {'tok/s':>10}")
    for chunk in args.chunks:
        kx, ki = mx.random.split(mx.random.key(chunk))
        x = mx.random.normal((1, chunk, args.hidden), key=kx).astype(mx.bfloat16)
        inds = mx.random.randint(0, args.experts, (1, chunk, args.topk), key=ki)
        inds = inds.astype(mx.uint32)
        mx.eval(x, inds)
        for mode in args.modes:
            if mode == "sorted":
                sorted_flag, gather = True, raw
            elif mode == "unsorted":
                sorted_flag, gather = False, raw
            elif mode == "segmented":
                sorted_flag, gather = True, reroute._gather_qmm_rerouted
            else:
                raise SystemExit(f"unknown mode {mode}")

            def fn(x=x, inds=inds, sorted_flag=sorted_flag, gather=gather):
                return _layer(
                    x, inds, gu, dn, args.bits, args.group_size, sorted_flag, gather
                )

            for _ in range(2):
                mx.eval(fn())
            mx.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                mx.eval(fn())
            mx.synchronize()
            ms = (time.perf_counter() - t0) / args.iters * 1e3
            print(
                f"{chunk:>6} {chunk * args.topk:>7} {mode:>10} {ms:>9.2f} "
                f"{chunk / ms * 1e3:>10.0f}"
            )


if __name__ == "__main__":
    main()
