# SPDX-License-Identifier: Apache-2.0
"""Multi-row verify qmv for pre-NAX Apple GPUs (M1 class).

Speculative verify multiplies a skinny row batch (M = 1 + draft depth)
against the quantized weights. On applegpu_g13 (M1 / M1 Pro / Max / Ultra)
stock MLX re-runs its one-row ``qmv`` per row, so time grows linearly with M,
and the split-K / sg8 verify kernels tuned on M3 Ultra land at ~165 GB/s. At
M = 2..4 these GPUs are limited by ALU and by re-reading the input rows, not
by DRAM, so this kernel keeps MLX's ``qmv_fast`` layout (32 lanes stride K,
each simdgroup owns ``RPSG`` adjacent output columns) and changes two things:

- each lane unpacks its weight chunk once per output column and applies it
  to every row (the M rows stream past the unpacked tile in registers), and
- the unpack is generated per bit width with compile-time shifts, so 5-bit
  weights (common in oQ "e" mixes) get the same treatment as 4-bit.

Measured on M1 Max (fp16; summed Qwen3.8-27B projection shapes per call,
gain vs the route that served them before - stock qmv, vk split-K or sg8):

    rows      2      3      4      5
    4-bit   +9%   +16%   +13%    0%
    5-bit  +17%   +26%    -5%  -17%

so 4-bit routes up to 4 rows and 5-bit up to 3; deeper verifies keep sg8.

Numerics: fp32 accumulation, per-group ``scale * sum(x*q) + bias * sum(x)``
like qmv. Rows are not bit-identical to the one-row decode (same class as the
other verify kernels); row-exact verify models never get here.

Gating: applegpu_g13 only (where it was measured).
"""

from __future__ import annotations

import logging
from functools import cache

logger = logging.getLogger(__name__)

MIN_ROWS = 2
# Largest row count each bit width still wins at (M1 Max sweep over the
# Qwen3.8 projection shapes, vs the routes below it): beyond these, sg8's
# simdgroup-matrix tiles are cheaper than streaming more rows per lane.
_MAX_ROWS = {4: 4, 5: 3}
MAX_ROWS = max(_MAX_ROWS.values())
# Values each lane owns per K block: a 4-bit pair of words, or a whole
# 5-word 5-bit run (32 values, so every lane starts word aligned).
_VPT = {4: 16, 5: 32}
_SUPPORTED_BITS = frozenset(_VPT)
# Tiny projections (GDN a/b heads, k/v of the draft) stay on stock qmv: the
# per-launch overhead of a custom kernel outweighs the GPU win there.
_MIN_N = 1024


@cache
def enabled() -> bool:
    from omlx.utils.hardware import get_apple_gpu_generation

    on = get_apple_gpu_generation() == 13
    if on:
        logger.info("M1 verify qmv enabled (4-bit rows 2..4, 5-bit rows 2..3)")
    return on


def eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    bits = int(bits)
    if bits not in _SUPPORTED_BITS or not enabled():
        return False
    vpt = _VPT[bits]
    return (
        MIN_ROWS <= int(M) <= _MAX_ROWS[bits]
        and dtype in (mx.float16, mx.bfloat16)
        and int(group_size) in (32, 64, 128)
        and int(group_size) % vpt == 0
        and int(K) % (vpt * 32) == 0
        and int(N) % 16 == 0
        and int(N) >= _MIN_N
    )


def geometry(M: int, bits: int, N: int) -> tuple[int, int]:
    """(output columns per simdgroup, simdgroups per threadgroup).

    5-bit lanes hold 32 unpacked weights per column, so they own fewer
    columns to stay out of register spills."""
    return (2, 2) if bits == 5 else (4, 2)


def _unpack_source(bits: int, vpt: int) -> str:
    """``unpack_w``: a lane's ``vpt`` weights from its ``WORDS`` packed words
    into half4s, with every shift a compile-time constant (5-bit runs straddle
    word boundaries)."""
    mask = (1 << bits) - 1
    vals = []
    for v in range(vpt):
        bit = v * bits
        word, sh = divmod(bit, 32)
        if sh + bits <= 32:
            vals.append(f"half((w[{word}] >> {sh}u) & {mask}u)")
        else:
            vals.append(
                f"half(((w[{word}] >> {sh}u) | (w[{word + 1}] << {32 - sh}u)) & {mask}u)"
            )
    body = "\n".join(
        f"  o[{i}] = half4({', '.join(vals[4 * i:4 * i + 4])});"
        for i in range(vpt // 4)
    )
    return (
        "METAL_FUNC void unpack_w(const device uint32_t* w, thread half4* o) {\n"
        f"{body}\n}}\n"
    )


def _source(M: int, bits: int, rpsg: int, nsg: int) -> str:
    vpt = _VPT[bits]
    return f"""
    constexpr int M = {M};
    constexpr int RPSG = {rpsg};
    constexpr int NSG = {nsg};
    constexpr int VPT = {vpt};
    constexpr int V4 = VPT / 4;
    constexpr int WORDS = VPT * {bits} / 32;
    constexpr int BLOCK = VPT * 32;

    const int K = x_shape[1];
    const int N = w_shape[0];
    const int Kw = w_shape[1];
    const int Kg = scales_shape[1];
    const int GS = K / Kg;

    const uint simd_gid = thread_position_in_threadgroup.y;
    const uint simd_lid = thread_position_in_threadgroup.x;
    const int out_row = int(threadgroup_position_in_grid.y) * (NSG * RPSG) + int(simd_gid) * RPSG;

    const device uint32_t* ws = w + out_row * Kw + int(simd_lid) * WORDS;
    const device T* sc = scales + out_row * Kg;
    const device T* bi = biases + out_row * Kg;
    const device T* xp = x + int(simd_lid) * VPT;

    float result[M][RPSG];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; m++)
      #pragma clang loop unroll(full)
      for (int r = 0; r < RPSG; r++) result[m][r] = 0.0f;

    for (int k = 0; k < K; k += BLOCK) {{
      const int g = (k + int(simd_lid) * VPT) / GS;
      const int kw = k * WORDS / VPT;
      // Unpack this lane's weights for every owned column once (exact in half).
      half4 wf[RPSG][V4];
      float sr[RPSG], br[RPSG];
      #pragma clang loop unroll(full)
      for (int r = 0; r < RPSG; r++) {{
        unpack_w(ws + r * Kw + kw, wf[r]);
        sr[r] = float(sc[r * Kg + g]);
        br[r] = float(bi[r * Kg + g]);
      }}
      // Stream the M input rows past the unpacked tile.
      #pragma clang loop unroll(full)
      for (int m = 0; m < M; m++) {{
        const device vec<T, 4>* xr = (const device vec<T, 4>*)(xp + m * K + k);
        float acc[RPSG];
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPSG; r++) acc[r] = 0.0f;
        float4 xsum4 = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < V4; i++) {{
          float4 xv = float4(xr[i]);
          xsum4 += xv;
          #pragma clang loop unroll(full)
          for (int r = 0; r < RPSG; r++) acc[r] += dot(xv, float4(wf[r][i]));
        }}
        float xsum = (xsum4.x + xsum4.y) + (xsum4.z + xsum4.w);
        #pragma clang loop unroll(full)
        for (int r = 0; r < RPSG; r++) result[m][r] += sr[r] * acc[r] + br[r] * xsum;
      }}
    }}

    #pragma clang loop unroll(full)
    for (int m = 0; m < M; m++) {{
      #pragma clang loop unroll(full)
      for (int r = 0; r < RPSG; r++) {{
        float v = simd_sum(result[m][r]);
        if (simd_lid == 0) y[m * N + out_row + r] = static_cast<T>(v);
      }}
    }}
    """


@cache
def _kernel(M: int, bits: int, rpsg: int, nsg: int):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name=f"omlx_m1_verify_qmv_m{M}_b{bits}_r{rpsg}_s{nsg}",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        source=_source(M, bits, rpsg, nsg),
        header="#include <metal_simdgroup>\n" + _unpack_source(bits, _VPT[bits]),
    )


def m1_qmm(x2, w_q, scales, biases, *, bits: int, group_size: int):
    """(M, K) x (N, K)^T -> (M, N) for the rows ``eligible`` accepts."""
    del group_size  # derived in-kernel from the scales shape
    M = int(x2.shape[0])
    N = int(w_q.shape[0])
    rpsg, nsg = geometry(M, int(bits), N)
    (y,) = _kernel(M, int(bits), rpsg, nsg)(
        inputs=[x2, w_q, scales, biases],
        template=[("T", x2.dtype)],
        grid=(32, (N // (rpsg * nsg)) * nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x2.dtype],
    )
    return y
