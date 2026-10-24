# SPDX-License-Identifier: Apache-2.0
#
# Adapted from mlx-dspark (src/mlx_dspark/skinny_qmm.py)
#   https://github.com/ARahim3/mlx-dspark
#
#   MIT License
#
#   Copyright (c) 2026 erahim3
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#
# The weights-as-MMA-operand layout with masked fields against pre-scaled
# inputs is mlx-dspark's (which credits TensorFold's simd_qmm and llama.cpp's
# pre-scaled-input dot product). oMLX adds fp16 inputs (mlx-dspark reads bf16
# bit patterns only), the applegpu_g13 gate and the verify routing.
"""Verify-width (5-16 rows) 4-bit quantized matmul on simdgroup matrices, M1.

Speculative verify on M1 multiplies 5-16 rows against every projection; the
multi-row qmv (``m1_verify_qmv``) stops paying from 5 rows, and sg8 / mma land
at ~2x the one-row decode time there. This kernel computes ``C = W X^T`` on
the 8x8 ``simdgroup_matrix`` units with the WEIGHTS as operand A: each lane
loads two packed words of its weight row straight into registers (no
threadgroup staging, no barriers in the K loop) and up to two 8-row tiles of
X share each weight read. Within a 64-input group, MMA step ``s`` takes nibble
field ``s`` of 8 consecutive words, masked in place (``word & (0xF << 4s)``)
against inputs pre-scaled by ``2^-4s`` (exact in fp32), and scale/bias factor
out per group (``acc += scale * P + bias * sum(x)``). Groups are interleaved
over the threadgroup's simdgroups (split-K) and reduced at the end.

Numerics: fp32 accumulation; row-count invariant; not bit-identical to the
one-row decode (same class as the other verify kernels).

Measured on M1 Max (fp16, dependent up/down chain of Qwen3.8-27B shapes, ms
per projection; one-row decode is 0.169):

    rows        5      6      8    9-16
    before  0.315  0.328  0.353   0.660   (sg8 / mma)
    this    0.289  0.288  0.287   0.471

Gating: applegpu_g13 only (where it was measured).
"""

from __future__ import annotations

import logging
import math
from functools import cache

logger = logging.getLogger(__name__)

GROUP = 64
BITS = 4
RT_MAX = 2  # 8-row tiles per threadgroup (16 rows)
# Measured window on M1 Max (4-bit, 5-16 rows).
MIN_ROWS = 5
MAX_ROWS = 16
# Small outputs (GDN a/b heads) stay on the stock path.
_MIN_N = 1024

_HEADER = r"""
// Element e of a packed word holding two 16-bit inputs, as fp32.
template <typename T>
inline float x_at(uint w, int e) {
  if (metal::is_same_v<T, bfloat>) {
    return as_type<float>((e & 1) ? (w & 0xFFFF0000u) : (w << 16));
  }
  return float(as_type<half2>(w)[e & 1]);
}
"""

_SRC = r"""
  constexpr int G = K / 64;                   // quant groups along K
  constexpr int FIELDS = 8;                   // 4-bit fields per word = MMA steps
  constexpr int WPR = K / 8;                  // packed words per weight row
  const int lane = int(thread_index_in_simdgroup);
  const int c = int(simdgroup_index_in_threadgroup);
  const int fm = ((lane >> 4) << 2) + ((lane >> 1) & 3);       // fragment row
  const int fn = (((lane >> 3) & 1) << 2) + ((lane & 1) << 1);  // fragment col pair
  const int R = X_shape[0];
  const int n0 = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int m0 = int(threadgroup_position_in_grid.y) * (8 * RT);

  int nrow[NT];
  for (int t = 0; t < NT; ++t) nrow[t] = min(n0 + 8 * t + fm, N - 1);
  const device T* x0[RT];
  const device T* x1[RT];
  for (int r = 0; r < RT; ++r) {
    x0[r] = X + size_t(min(m0 + 8 * r + fn, R - 1)) * K;
    x1[r] = X + size_t(min(m0 + 8 * r + fn + 1, R - 1)) * K;
  }
  float acc[RT][NT][2];
  for (int r = 0; r < RT; ++r)
    for (int t = 0; t < NT; ++t) { acc[r][t][0] = 0.0f; acc[r][t][1] = 0.0f; }

  for (int g = c; g < G; g += S) {
    simdgroup_matrix<float, 8, 8> P[RT][NT];
    float sum0[RT], sum1[RT];
    for (int r = 0; r < RT; ++r) {
      sum0[r] = 0.0f; sum1[r] = 0.0f;
      for (int t = 0; t < NT; ++t) P[r][t] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    uint2 w[NT];
    for (int t = 0; t < NT; ++t)
      w[t] = *(const device uint2*)((const device uint*)Wq + size_t(nrow[t]) * WPR + 8 * g + fn);
    const int xo = 64 * g + FIELDS * fm;      // this lane's 8-input run
    uint4 xa[RT], xb[RT];
    for (int r = 0; r < RT; ++r) {
      xa[r] = *(const device uint4*)(x0[r] + xo);
      xb[r] = *(const device uint4*)(x1[r] + xo);
      for (int e = 0; e < FIELDS; ++e) {
        sum0[r] += x_at<T>(xa[r][e / 2], e);
        sum1[r] += x_at<T>(xb[r][e / 2], e);
      }
    }
    #pragma clang loop unroll(full)
    for (int s = 0; s < FIELDS; ++s) {
      const uint mask = 0xFu << (4 * s);
      const float pre = as_type<float>(uint(127 - 4 * s) << 23);  // 2^-(4s)
      simdgroup_matrix<float, 8, 8> b[RT];
      for (int r = 0; r < RT; ++r) {
        b[r].thread_elements()[0] = x_at<T>(xa[r][s / 2], s) * pre;
        b[r].thread_elements()[1] = x_at<T>(xb[r][s / 2], s) * pre;
      }
      for (int t = 0; t < NT; ++t) {
        simdgroup_matrix<float, 8, 8> a;
        a.thread_elements()[0] = float(w[t].x & mask);
        a.thread_elements()[1] = float(w[t].y & mask);
        for (int r = 0; r < RT; ++r) simdgroup_multiply_accumulate(P[r][t], a, b[r], P[r][t]);
      }
    }
    for (int r = 0; r < RT; ++r) {            // group input sums over the 8 fm lanes
      for (int d = 2; d <= 16; d <<= 1) {
        if (d == 8) continue;
        sum0[r] += simd_shuffle_xor(sum0[r], ushort(d));
        sum1[r] += simd_shuffle_xor(sum1[r], ushort(d));
      }
    }
    for (int t = 0; t < NT; ++t) {
      const float sc = float(SC[size_t(nrow[t]) * G + g]);
      const float bi = float(BI[size_t(nrow[t]) * G + g]);
      for (int r = 0; r < RT; ++r) {
        acc[r][t][0] = fma(bi, sum0[r], fma(sc, P[r][t].thread_elements()[0], acc[r][t][0]));
        acc[r][t][1] = fma(bi, sum1[r], fma(sc, P[r][t].thread_elements()[1], acc[r][t][1]));
      }
    }
  }

  threadgroup float part[S * RT * NT * 64];   // split-K partials
  for (int r = 0; r < RT; ++r)
    for (int t = 0; t < NT; ++t)
      for (int e = 0; e < 2; ++e) part[((c * RT + r) * NT + t) * 64 + fm * 8 + fn + e] = acc[r][t][e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = c * 32 + lane; i < RT * NT * 64; i += S * 32) {
    float v = 0.0f;
    for (int q = 0; q < S; ++q) v += part[q * (RT * NT * 64) + i];
    const int r = i / (NT * 64), t = (i / 64) % NT, e = i % 64;
    const int row = m0 + 8 * r + (e % 8), n = n0 + 8 * t + (e / 8);
    if (row < R && n < N) OUT[size_t(row) * N + n] = T(v);
  }
"""


@cache
def enabled() -> bool:
    from omlx.utils.hardware import get_apple_gpu_generation

    on = get_apple_gpu_generation() == 13
    if on:
        logger.info("M1 verify MMA enabled (4-bit rows %d..%d)", MIN_ROWS, MAX_ROWS)
    return on


def eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    if int(bits) != BITS or not enabled():
        return False
    return (
        MIN_ROWS <= int(M) <= MAX_ROWS
        and int(group_size) == GROUP
        and dtype in (mx.float16, mx.bfloat16)
        and int(K) % GROUP == 0
        and int(N) % 8 == 0
        and int(N) >= _MIN_N
    )


def geometry(n: int, rt: int) -> tuple[int, int]:
    """(8-column tiles per simdgroup, simdgroups per threadgroup)."""
    # Two row tiles double the register footprint: on M1 the compiled
    # pipeline then admits at most 384 threads; 4 x 4 measured best there.
    if rt > 1:
        return 4, 4
    if n >= 4096:
        return 4, 8
    return 4, 16


@cache
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="omlx_m1_verify_mma",
        input_names=["X", "Wq", "SC", "BI"],
        output_names=["OUT"],
        source=_SRC,
        header=_HEADER,
    )


def m1_mma_qmm(x2, w_q, scales, biases, *, bits: int = 4, group_size: int = 64):
    """(M, K) x (N, K)^T -> (M, N), 4-bit affine groups of 64, M in 5..16."""
    del bits, group_size
    rows, k = int(x2.shape[0]), int(x2.shape[1])
    n = int(w_q.shape[0])
    rt = min(RT_MAX, (rows + 7) // 8)
    nt, s = geometry(n, rt)
    (y,) = _kernel()(
        inputs=[x2, w_q, scales, biases],
        template=[
            ("T", x2.dtype),
            ("K", k),
            ("N", n),
            ("NT", nt),
            ("RT", rt),
            ("S", s),
        ],
        grid=(math.ceil(n / (8 * nt)) * s * 32, math.ceil(rows / (8 * rt)), 1),
        threadgroup=(s * 32, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x2.dtype],
    )
    return y
