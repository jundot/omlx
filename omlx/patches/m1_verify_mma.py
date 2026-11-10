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
# bit patterns only), 5-, 6- and 8-bit weights, the applegpu_g13 gate and the verify
# routing.
"""Verify-width (5-16 rows) 4-, 5-, 6- and 8-bit quantized matmul on simdgroup matrices, M1.

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

5-bit packing puts 8 values in 5 bytes, so the 8-value chunk k = 8j..8j+7 of
a group is bytes 5j..5j+4. A lane's two chunks (j = fn, fn + 1, fn even) are
10 bytes at an even offset, read as three aligned words and split into a
32-bit low word and an 8-bit high byte per chunk. MMA step s takes field s:
fields 0..5 masked in place in the low word against inputs pre-scaled by
2^-5s, field 7 masked in place in the high byte against 2^-3, and field 6,
which straddles the two, shifted out (pre-scale 1). 6-bit chunks are 6 bytes,
so a lane's two are 12 word-aligned bytes: fields 0..4 in place in the low
word, field 5 shifted out, fields 6 and 7 in place in the high half. 8-bit
chunks are two words each, field s in place in byte s mod 4 of word s / 4.
Only the row stride, the weight load and the per-step field extraction differ
(``_UNPACK``).

Numerics: fp32 accumulation; row-count invariant; not bit-identical to the
one-row decode (same class as the other verify kernels).

Measured on M1 Max (fp16, dependent up/down chain of Qwen3.8-27B shapes, ms
per projection; one-row decode is 0.169):

    rows        5      6      8    9-16
    before  0.315  0.328  0.353   0.660   (sg8 / mma)
    this    0.289  0.288  0.287   0.471

5-bit, chain over the Qwen3.8-27B layers whose projection is 5-bit, vs the
sg8 / mma route it replaces:

    rows             5     8     9    12    16
    mlp.down_proj  -29%  -33%  -45%  -44%  -43%
    gdn.in_proj_z   -9%  -20%  -39%  -38%  -36%
    gdn.out_proj   -10%  -22%  -41%  -40%  -38%
    attn.o_proj    -10%  -19%  -38%  -38%  -36%

6- and 8-bit (stock qmm before; Qwen3.6-35B-A3B shapes, fp16, ms):

    rows                         7 (stock -> this)   12 (stock -> this)
    lm_head 8-bit [248320x2048]    4.80 -> 1.93        4.79 -> 3.06
    lm_head 6-bit [248320x2048]    4.96 -> 1.95        4.98 -> 3.02
    8-bit [8192x2048]              0.45 -> 0.30        0.81 -> 0.55
    6-bit [8192x2048]              0.83 -> 0.40        0.85 -> 0.53

End to end, Qwen3.6-35B-A3B with DFlash (block 6, fixed verify), decode
tok/s: packs with 6-bit group-64 projections +18% to +24%; a pack whose only
6/8-bit group-64 weights are the lm_head and four attention layers is flat.

Gating: applegpu_g13 only (where it was measured).
"""

from __future__ import annotations

import logging
import math
from functools import cache

logger = logging.getLogger(__name__)

GROUP = 64
RT_MAX = 2  # 8-row tiles per threadgroup (16 rows)
# Measured window on M1 Max (4/5/6/8-bit, 5-16 rows).
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
@STRIDE@
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
@LOAD@
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
@PRE@
      simdgroup_matrix<float, 8, 8> b[RT];
      for (int r = 0; r < RT; ++r) {
        b[r].thread_elements()[0] = x_at<T>(xa[r][s / 2], s) * pre;
        b[r].thread_elements()[1] = x_at<T>(xb[r][s / 2], s) * pre;
      }
      for (int t = 0; t < NT; ++t) {
        simdgroup_matrix<float, 8, 8> a;
@FIELD@
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

# Per-bit-width pieces of _SRC: row stride, weight load, step pre-scale and
# field extraction.
_UNPACK = {
    4: {
        "@STRIDE@": r"""
  constexpr int FIELDS = 8;                   // 4-bit fields per word = MMA steps
  constexpr int WPR = K / 8;                  // packed words per weight row
""",
        "@LOAD@": r"""
    uint2 w[NT];
    for (int t = 0; t < NT; ++t)
      w[t] = *(const device uint2*)((const device uint*)Wq + size_t(nrow[t]) * WPR + 8 * g + fn);
""",
        "@PRE@": r"""
      const uint mask = 0xFu << (4 * s);
      const float pre = as_type<float>(uint(127 - 4 * s) << 23);  // 2^-(4s)
""",
        "@FIELD@": r"""
        a.thread_elements()[0] = float(w[t].x & mask);
        a.thread_elements()[1] = float(w[t].y & mask);
""",
    },
    5: {
        "@STRIDE@": r"""
  constexpr int FIELDS = 8;                   // 5-bit fields per chunk = MMA steps
  constexpr int UPR = K * 5 / 16;             // packed ushorts per weight row
""",
        "@LOAD@": r"""
    // Chunks fn and fn + 1 of group g (10 bytes at byte 5 * fn of the
    // group's 40) from three aligned words. The group starts 8-byte aligned;
    // for fn = 2 and 6 the run starts 2 bytes into its first word.
    uint lo0[NT], hi0[NT], lo1[NT], hi1[NT];
    const int wfirst = (5 * fn) / 4;            // 0, 2, 5, 7
    const int sh = ((5 * fn) % 4) * 8;          // 0 or 16 bits
    for (int t = 0; t < NT; ++t) {
      const device uint* p = (const device uint*)Wq + size_t(nrow[t]) * (UPR / 2) + 10 * g + wfirst;
      const uint w0 = p[0], w1 = p[1], w2 = p[2];
      // The 10 bytes as b[0..9]: a = bytes 0..3, b = bytes 4..7, c = bytes 8..9.
      const uint a = sh ? ((w0 >> 16) | (w1 << 16)) : w0;
      const uint b = sh ? ((w1 >> 16) | (w2 << 16)) : w1;
      const uint c = sh ? (w2 >> 16) : (w2 & 0xFFFFu);
      lo0[t] = a;
      hi0[t] = b & 0xFFu;
      lo1[t] = (b >> 8) | (c << 24);
      hi1[t] = c >> 8;
    }
""",
        "@PRE@": r"""
      // 2^-(5s) for the in-place fields 0..5, 1 for the shifted field 6,
      // 2^-3 for field 7 (bits 3..7 of the high byte).
      const float pre = s < 6 ? as_type<float>(uint(127 - 5 * s) << 23)
                              : (s == 6 ? 1.0f : 0.125f);
""",
        "@FIELD@": r"""
        if (s < 6) {
          const uint mask = 0x1Fu << (5 * s);
          a.thread_elements()[0] = float(lo0[t] & mask);
          a.thread_elements()[1] = float(lo1[t] & mask);
        } else if (s == 6) {
          a.thread_elements()[0] = float((lo0[t] >> 30) | ((hi0[t] & 7u) << 2));
          a.thread_elements()[1] = float((lo1[t] >> 30) | ((hi1[t] & 7u) << 2));
        } else {
          a.thread_elements()[0] = float(hi0[t] & 0xF8u);
          a.thread_elements()[1] = float(hi1[t] & 0xF8u);
        }
""",
    },
    6: {
        "@STRIDE@": r"""
  constexpr int FIELDS = 8;                   // 6-bit fields per chunk = MMA steps
  constexpr int WPR = K * 6 / 32;             // packed words per weight row
""",
        "@LOAD@": r"""
    // Chunks fn and fn + 1 of group g are 12 bytes at byte 6 * fn of the
    // group's 48, word aligned since fn is even: chunk 0 is word 0 plus the
    // low half of word 1, chunk 1 the high half of word 1 plus word 2.
    uint lo0[NT], hi0[NT], lo1[NT], hi1[NT];
    for (int t = 0; t < NT; ++t) {
      const device uint* p = (const device uint*)Wq + size_t(nrow[t]) * WPR + 12 * g + 3 * fn / 2;
      const uint w0 = p[0], w1 = p[1], w2 = p[2];
      lo0[t] = w0;
      hi0[t] = w1 & 0xFFFFu;
      lo1[t] = (w1 >> 16) | (w2 << 16);
      hi1[t] = w2 >> 16;
    }
""",
        "@PRE@": r"""
      // 2^-(6s) for the in-place fields 0..4, 1 for the shifted field 5,
      // 2^-4 and 2^-10 for fields 6 and 7 (bits 4..9 and 10..15 of the high half).
      const float pre = s < 5 ? as_type<float>(uint(127 - 6 * s) << 23)
                              : (s == 5 ? 1.0f : (s == 6 ? 0.0625f : 0.0009765625f));
""",
        "@FIELD@": r"""
        if (s < 5) {
          const uint mask = 0x3Fu << (6 * s);
          a.thread_elements()[0] = float(lo0[t] & mask);
          a.thread_elements()[1] = float(lo1[t] & mask);
        } else if (s == 5) {
          a.thread_elements()[0] = float((lo0[t] >> 30) | ((hi0[t] & 0xFu) << 2));
          a.thread_elements()[1] = float((lo1[t] >> 30) | ((hi1[t] & 0xFu) << 2));
        } else {
          const uint mask = s == 6 ? 0x3F0u : 0xFC00u;
          a.thread_elements()[0] = float(hi0[t] & mask);
          a.thread_elements()[1] = float(hi1[t] & mask);
        }
""",
    },
    8: {
        "@STRIDE@": r"""
  constexpr int FIELDS = 8;                   // 8-bit fields per chunk = MMA steps
  constexpr int WPR = K / 4;                  // packed words per weight row
""",
        "@LOAD@": r"""
    // Chunks fn and fn + 1 of group g: words 2 * fn .. 2 * fn + 3 of its 16.
    uint4 w[NT];
    for (int t = 0; t < NT; ++t)
      w[t] = *(const device uint4*)((const device uint*)Wq + size_t(nrow[t]) * WPR + 16 * g + 2 * fn);
""",
        "@PRE@": r"""
      const uint mask = 0xFFu << (8 * (s & 3));
      const float pre = as_type<float>(uint(127 - 8 * (s & 3)) << 23);  // 2^-(8 (s mod 4))
""",
        "@FIELD@": r"""
        a.thread_elements()[0] = float(w[t][s >> 2] & mask);
        a.thread_elements()[1] = float(w[t][2 + (s >> 2)] & mask);
""",
    },
}


def _source(bits: int) -> str:
    src = _SRC
    for marker, text in _UNPACK[bits].items():
        src = src.replace(marker, text.strip("\n"))
    return src


@cache
def enabled() -> bool:
    from omlx.utils.hardware import get_apple_gpu_generation

    on = get_apple_gpu_generation() == 13
    if on:
        logger.info(
            "M1 verify MMA enabled (4/5/6/8-bit rows %d..%d)", MIN_ROWS, MAX_ROWS
        )
    return on


def eligible(M: int, K: int, N: int, bits: int, group_size: int, dtype) -> bool:
    import mlx.core as mx

    if int(bits) not in _UNPACK or not enabled():
        return False
    return (
        MIN_ROWS <= int(M) <= MAX_ROWS
        and int(group_size) == GROUP
        and dtype in (mx.float16, mx.bfloat16)
        and int(K) % GROUP == 0
        and int(N) % 8 == 0
        and int(N) >= _MIN_N
    )


def geometry(n: int, rt: int, bits: int = 4) -> tuple[int, int]:
    """(8-column tiles per simdgroup, simdgroups per threadgroup)."""
    # 8-bit holds four weight words per tile; 2 x 4 measured best (or within
    # noise of it) on every shape and row count, and 16 simdgroups exceed the
    # compiled pipeline's thread limit.
    if bits == 8:
        return 2, 4
    # Two row tiles double the register footprint: on M1 the compiled
    # pipeline then admits at most 384 threads; 4 x 4 measured best there.
    if rt > 1:
        return 4, 4
    if n >= 4096:
        return 4, 8
    # Small N: two tiles keep the 4 x 16 simdgroup count in 256 threads; the
    # 5-bit bf16 pipeline admits only 448 threads on M1 and M2.
    return 2, 8


@cache
def _kernel(bits: int):
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="omlx_m1_verify_mma" if bits == 4 else f"omlx_m1_verify_mma{bits}",
        input_names=["X", "Wq", "SC", "BI"],
        output_names=["OUT"],
        source=_source(bits),
        header=_HEADER,
    )


def m1_mma_qmm(x2, w_q, scales, biases, *, bits: int, group_size: int = 64):
    """(M, K) x (N, K)^T -> (M, N), 4/5/6/8-bit affine groups of 64, M in 5..16."""
    del group_size
    rows, k = int(x2.shape[0]), int(x2.shape[1])
    n = int(w_q.shape[0])
    rt = min(RT_MAX, (rows + 7) // 8)
    nt, s = geometry(n, rt, int(bits))
    (y,) = _kernel(int(bits))(
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
