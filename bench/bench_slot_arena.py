# SPDX-License-Identifier: Apache-2.0
"""Microbench: per-call assembly cost — stack-per-call vs persistent slot arena.

Decides the V4-2 residency representation:
  arm "stack"  — generic path: mx.stack(U per-expert rows) every call,
                 then gather_qmm with remapped indices.
  arm "arena"  — V4.1 path: persistent (cap, ...) bank; gather_qmm reads
                 rows by slot index. Hits pay no assembly.

Shapes mirror V4.1 oQ4e experts (4-bit, group 64):
  w1/w3: hidden 7168 -> moe 2048 rows; w2: 2048 -> 7168.
We use w1 dims for both arms (the assembly delta is shape-proportional).
"""
import time

import mlx.core as mx
import numpy as np


def run(U=12, CAP=16, ITERS=200, hidden=7168, moe=2048, bits=4, gs=64):
    rng = np.random.default_rng(0)
    packed = hidden * bits // 32  # u32 cols per output row (transposed layout)
    groups = hidden // gs

    # Persistent arena: (CAP, moe, packed) weights + scales/biases.
    W = mx.array(rng.integers(0, 2**32 - 1, size=(CAP, moe, packed), dtype=np.uint32))
    S = mx.array(rng.standard_normal((CAP, moe, groups)).astype(np.float16))
    B = mx.array(rng.standard_normal((CAP, moe, groups)).astype(np.float16))
    mx.eval(W, S, B)

    # Detached per-expert copies = what the LRU bundles hold.
    bundles = [(W[i], S[i], B[i]) for i in range(CAP)]
    mx.eval(*[t for b in bundles for t in b])

    x = mx.array(rng.standard_normal((1, 1, hidden)).astype(np.float16))

    def timed(fn):
        # warmup
        for _ in range(10):
            mx.eval(fn())
        t0 = time.perf_counter()
        for _ in range(ITERS):
            mx.eval(fn())
        return (time.perf_counter() - t0) / ITERS * 1e3

    uniq = rng.choice(CAP, size=U, replace=False)

    # ARM A: generic — stack bundles into a fresh bank every call.
    def arm_stack():
        wb = mx.stack([bundles[e][0] for e in uniq], axis=0)
        sb = mx.stack([bundles[e][1] for e in uniq], axis=0)
        bb = mx.stack([bundles[e][2] for e in uniq], axis=0)
        idx = mx.arange(U, dtype=mx.int32)  # remapped positions
        return mx.gather_qmm(x, wb, sb, bb, rhs_indices=idx,
                             transpose=True, group_size=gs, bits=bits)

    # ARM B: V4.1 arena — persistent bank, indices select slot rows.
    slot_ids = mx.array(uniq, dtype=mx.int32)

    def arm_arena():
        return mx.gather_qmm(x, W, S, B, rhs_indices=slot_ids,
                             transpose=True, group_size=gs, bits=bits)

    # Isolate the pure assembly overhead too.
    def arm_stack_only():
        return (mx.stack([bundles[e][0] for e in uniq], axis=0),
                mx.stack([bundles[e][1] for e in uniq], axis=0),
                mx.stack([bundles[e][2] for e in uniq], axis=0))

    a = timed(arm_stack)
    b = timed(arm_arena)
    s = timed(arm_stack_only)
    print(f"U={U} cap={CAP} iters={ITERS}  ({moe}x{hidden}@{bits}b gs{gs})")
    print(f"  stack-per-call + qmm : {a:8.3f} ms/call")
    print(f"  arena (persistent)   : {b:8.3f} ms/call")
    print(f"  mx.stack alone       : {s:8.3f} ms/call")
    print(f"  delta (a-b)          : {a - b:8.3f} ms/call/proj  "
          f"x3 proj x43 layers = {(a - b) * 3 * 43:8.1f} ms/token")


if __name__ == "__main__":
    import sys

    kw = {}
    if len(sys.argv) > 1:
        kw["U"] = int(sys.argv[1])
    if len(sys.argv) > 2:
        kw["ITERS"] = int(sys.argv[2])
    run(**kw)
