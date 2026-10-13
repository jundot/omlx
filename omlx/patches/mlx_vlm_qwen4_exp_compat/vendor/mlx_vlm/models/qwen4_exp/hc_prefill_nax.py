# SPDX-License-Identifier: Apache-2.0
"""Tensor-unit (NAX) kernels for the Qwen4 hyper-connection prefill.

The MLX-matmul prefill path (``hc_fused._mlx_prefill``) runs each
hyper-connection as six dispatches: the stream norm (with the pending
residual write), MLX's quantized matmul for the 10240 -> 320 down projection,
an eager divide by the stream count, the compiled SiLU, MLX's quantized
matmul for the 320 -> 10240 up projection (writing [rows, 10240]) and a tail
kernel that gates, multiplies and averages the four streams and computes the
inject weights. Here it takes three:

1. ``norm_inject``: the same stream norm; it also computes the inject
   projection's per-simdgroup partial sums with the tail kernel's thread
   layout.
2. ``down_silu``: MLX's NAX ``affine_qmm_t_nax`` tile loop for the down
   projection with the divide and SiLU in its epilogue.
3. ``up_tail``: the same tile loop for the up projection over column tiles
   that hold 16 hidden columns of all four streams, with the sigmoid gate,
   the stream product and the four-stream mean in its epilogue (only
   [rows, hidden] is written), plus the final inject reduction.

Every value is rounded where the six-dispatch path rounds it, so outputs are
bit-identical: the tile loops issue the same ``matmul2d`` (16x32x16) calls in
the same K order on the same bf16 dequantized weights as MLX's kernel (only
which output columns share a tile changes), the epilogues repeat the
eager/compiled operations' bf16 expressions and the tail kernel's fp32 ones.
They are only used where MLX's ``QuantizedMatmul`` itself takes the unsplit
NAX kernel (see ``plain_qmm_nax``), and ``hc_fused`` checks the first call of
each specialization against the MLX path bit for bit.

The NAX fragment/mma helpers and the weight dequantization below are
transcribed from MLX v0.32.2 (``mlx/backend/metal/kernels/steel/gemm/nax.h``
and ``quantized_nax.h``). MLX is Copyright © 2023-2025 Apple Inc. and licensed
under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import mlx.core as mx

from .hc_projection import _HEADER as _QDOT_HEADER

_KERNELS: dict[str, object] = {}

# MLX's affine_qmm_t_nax tile: 64x64 outputs, 2x2 simdgroups of 32x32, K
# steps of 64 (one quantization group).
_BM = 64
_BN = 64
# Hidden columns per stream in one up-projection tile (4 streams x 16 = 64).
UP_COLS = 16
# Each K step loads one quantization group's scale and bias.
GROUP_SIZE = 64

_NAX_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// ---- MLX quantized_nax.h: 8-bit-word packing -------------------------------
template <int bits>
inline constexpr short hcn_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 8 / bits);
}

template <int bits>
inline constexpr short hcn_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? 1 : (bits == 5 ? 5 : 3);
}

// QuantizedBlockLoader<T, 64, 64, BKP, 1, 128, 64, bits>: thread t
// dequantizes NR packs of weight-tile row t / 2 (one scale per row and K
// step since the group size is the K step).
template <int bits>
struct HcnLoader {
  static constant constexpr short PF = hcn_pack_factor<bits>();
  static constant constexpr short BPP = hcn_bytes_per_pack<bits>();
  static constant constexpr short BCP = 64 / PF;
  static constant constexpr short NR = BCP * 64 / 128;
  static_assert(NR * BPP % 4 == 0, "whole 32-bit words per loader thread");
};

inline uint hcn_byte(thread const uint32_t* wv, int k) {
  return (wv[k >> 2] >> ((k & 3) * 8)) & 0xffu;
}

// The loader thread's NR packs, read as 32-bit words and stored as vectors.
// Each value is MLX's dequantize() expression on the same integer (scale * q
// is exact in fp32, so the single rounding of the add is the same).
template <typename T, int bits>
inline void hcn_load_w(
    const device uint8_t* src, T scale, T bias, threadgroup T* dst) {
  using L = HcnLoader<bits>;
  constexpr int NW = L::NR * L::BPP / 4;
  const float s = float(scale);
  const float b = float(bias);
  uint32_t wv[NW];
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < NW; i++) {
    wv[i] = ((const device uint32_t*)src)[i];
  }
  if (bits == 4) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < NW; i++) {
      vec<T, 8> v;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 8; j++) {
        v[j] = static_cast<T>(s * float((wv[i] >> (4 * j)) & 0xfu) + b);
      }
      *(threadgroup vec<T, 8>*)(dst + 8 * i) = v;
    }
  } else if (bits == 8) {
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < NW; i++) {
      vec<T, 4> v;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < 4; j++) {
        v[j] = static_cast<T>(s * float((wv[i] >> (8 * j)) & 0xffu) + b);
      }
      *(threadgroup vec<T, 4>*)(dst + 4 * i) = v;
    }
  } else if (bits == 5) {
    _Pragma("clang loop unroll(full)")
    for (int p = 0; p < L::NR; p++) {
      const uint w0 = hcn_byte(wv, 5 * p);
      const uint w1 = hcn_byte(wv, 5 * p + 1);
      const uint w2 = hcn_byte(wv, 5 * p + 2);
      const uint w3 = hcn_byte(wv, 5 * p + 3);
      const uint w4 = hcn_byte(wv, 5 * p + 4);
      vec<T, 8> v;
      v[0] = static_cast<T>(float(w0 & 0x1f) * s + b);
      v[1] = static_cast<T>(float(((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * s + b);
      v[2] = static_cast<T>(float((w1 & 0x7c) >> 2) * s + b);
      v[3] = static_cast<T>(float(((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * s + b);
      v[4] = static_cast<T>(float(((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * s + b);
      v[5] = static_cast<T>(float((w3 & 0x3e) >> 1) * s + b);
      v[6] = static_cast<T>(float(((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * s + b);
      v[7] = static_cast<T>(float((w4 & 0xf8) >> 3) * s + b);
      *(threadgroup vec<T, 8>*)(dst + 8 * p) = v;
    }
  } else if (bits == 6) {
    _Pragma("clang loop unroll(full)")
    for (int p = 0; p < L::NR; p++) {
      const uint w0 = hcn_byte(wv, 3 * p);
      const uint w1 = hcn_byte(wv, 3 * p + 1);
      const uint w2 = hcn_byte(wv, 3 * p + 2);
      vec<T, 4> v;
      v[0] = static_cast<T>(float(w0 & 0x3f) * s + b);
      v[1] = static_cast<T>(float(((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * s + b);
      v[2] = static_cast<T>(float(((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * s + b);
      v[3] = static_cast<T>(float((w2 >> 2) & 0x3f) * s + b);
      *(threadgroup vec<T, 4>*)(dst + 4 * p) = v;
    }
  }
}

// ---- MLX steel/gemm/nax.h: 16x16 fragments and the 16x32x16 matmul ------
inline short2 hcn_frag_coord(ushort lane) {
  const short qid = short(lane >> 2);
  const short fm = (qid & 4) | ((short(lane) >> 1) & 3);
  const short fn = ((qid & 2) | (short(lane) & 1)) * 4;
  return short2(fn, fm);
}

typedef vec<float, 8> hcn_acc;

// BaseNAXFrag::mma: C[0..1] += A * B[0..1]^T over one 16-deep K slice.
template <typename T>
inline void hcn_mma(
    thread hcn_acc& c0,
    thread hcn_acc& c1,
    thread const vec<T, 8>& a,
    thread const vec<T, 8>& b0,
    thread const vec<T, 8>& b1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16,
      32,
      16,
      false,
      true,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<T, T, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<T, T, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      float>();
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; i++) {
    ct_a[i] = a[i];
  }
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; i++) {
    ct_b[i] = b0[i];
    ct_b[8 + i] = b1[i];
  }
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; i++) {
    ct_c[i] = c0[i];
    ct_c[8 + i] = c1[i];
  }
  gemm_op.run(ct_a, ct_b, ct_c);
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; i++) {
    c0[i] = ct_c[i];
    c1[i] = ct_c[8 + i];
  }
}

// A fragment (rows fm, fm + 8; columns fn .. fn + 3) of a row-major device
// matrix; unless FULL, rows at or past `rows` read as zero (load_safe).
template <typename T, bool FULL>
inline vec<T, 8> hcn_load_a(const device T* src, int ld, short2 fc, int rows) {
  vec<T, 8> v;
  _Pragma("clang loop unroll(full)")
  for (short r = 0; r < 2; r++) {
    const int row = fc.y + r * 8;
    vec<T, 4> q = vec<T, 4>(0);
    if (FULL || row < rows) {
      q = *(const device vec<T, 4>*)(src + row * ld + fc.x);
    }
    v[4 * r] = q[0];
    v[4 * r + 1] = q[1];
    v[4 * r + 2] = q[2];
    v[4 * r + 3] = q[3];
  }
  return v;
}

template <typename T, int LD>
inline vec<T, 8> hcn_load_b(const threadgroup T* src, short2 fc) {
  vec<T, 8> v;
  _Pragma("clang loop unroll(full)")
  for (short r = 0; r < 2; r++) {
    const vec<T, 4> q =
        *(const threadgroup vec<T, 4>*)(src + (fc.y + r * 8) * LD + fc.x);
    v[4 * r] = q[0];
    v[4 * r + 1] = q[1];
    v[4 * r + 2] = q[2];
    v[4 * r + 3] = q[3];
  }
  return v;
}

// qmm_t_nax_tgp_impl's K loop for one 32x32 simdgroup tile: per 64-deep K
// step the weight tile is dequantized between two barriers, then kk1 = 0,
// 32 each run tile_matmad_nax's (mm, kk) sequence of 16x32x16 matmuls.
template <typename T, int bits, int BKP, bool FULL>
inline void hcn_gemm(
    thread hcn_acc (&C)[2][2],
    const device T* xa,
    int K,
    const threadgroup T* wt,
    const device uint8_t* wsrc,
    const device T* ssrc,
    const device T* bsrc,
    threadgroup T* wdst,
    short2 fc,
    int rows) {
  using L = HcnLoader<bits>;
  for (int k = 0; k < K; k += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hcn_load_w<T, bits>(wsrc + k * L::BPP / L::PF, ssrc[k / 64], bsrc[k / 64], wdst);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(disable)")
    for (short kk1 = 0; kk1 < 64; kk1 += 32) {
      vec<T, 8> A[2][2];
      vec<T, 8> B[2][2];
      volatile int compiler_barrier;
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 2; i++) {
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 2; j++) {
          A[i][j] = hcn_load_a<T, FULL>(
              xa + (i * 16) * K + k + kk1 + j * 16, K, fc, rows - i * 16);
        }
      }
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 2; i++) {
        _Pragma("clang loop unroll(full)")
        for (short j = 0; j < 2; j++) {
          B[i][j] = hcn_load_b<T, BKP>(wt + (i * 16) * BKP + kk1 + j * 16, fc);
        }
      }
      _Pragma("clang loop unroll(full)")
      for (short mm = 0; mm < 2; mm++) {
        _Pragma("clang loop unroll(full)")
        for (short kk = 0; kk < 2; kk++) {
          hcn_mma<T>(C[mm][0], C[mm][1], A[mm][kk], B[0][kk], B[1][kk]);
        }
      }
      (void)compiler_barrier;
    }
  }
}

// MLX's Sigmoid functor (unary_ops.h): the compiled nn.silu is x * sigmoid(x)
// on the same bf16 expressions.
template <typename T>
inline T hcn_sigmoid(T x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
"""

# hc_projection's hc_load_vector reading from threadgroup memory (same
# arithmetic on the same bf16 values).
_TG_LOAD = r"""
template <typename T, int N, int bits>
inline float hc_load_vector_tg(const threadgroup T* x, thread float* xt) {
    float sum = 0.0f;
    if (bits == 4) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 16.0f;
            xt[i + 2] = x[i + 2] / 256.0f;
            xt[i + 3] = x[i + 3] / 4096.0f;
        }
    } else if (bits == 5) {
        for (int i = 0; i < N; i += 8) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
                + x[i + 4] + x[i + 5] + x[i + 6] + x[i + 7];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 32.0f;
            xt[i + 2] = x[i + 2] / 4.0f;
            xt[i + 3] = x[i + 3] / 128.0f;
            xt[i + 4] = x[i + 4] / 16.0f;
            xt[i + 5] = x[i + 5] / 2.0f;
            xt[i + 6] = x[i + 6] / 64.0f;
            xt[i + 7] = x[i + 7] / 8.0f;
        }
    } else if (bits == 6) {
        for (int i = 0; i < N; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            xt[i] = x[i];
            xt[i + 1] = x[i + 1] / 64.0f;
            xt[i + 2] = x[i + 2] / 16.0f;
            xt[i + 3] = x[i + 3] / 4.0f;
        }
    } else if (bits == 8) {
        for (int i = 0; i < N; ++i) {
            sum += x[i];
            xt[i] = x[i];
        }
    }
    return sum;
}
"""

# Stream norm, optionally applying the pending residual write first
# (hc_fused._WN_SOURCE / _NP_SOURCE, same expressions), plus the inject
# projection's partial sums. The tail kernel's thread t < 256 dots elements
# [t * K / 256, (t + 1) * K / 256) of the row, so stream s holds its threads
# 64 s .. 64 s + 63 (simdgroups 2 s and 2 s + 1). The up kernel adds the eight
# per-simdgroup sums of each inject row in order.
_NORM_TEMPLATE = r"""
    const uint row = threadgroup_position_in_grid.z;
    const uint s = threadgroup_position_in_grid.y;
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[8];
    threadgroup T xs[H];
    constexpr int PER = (H + 255) / 256;
    const size_t base = (size_t)row * K + (size_t)s * H;
    const device T* wp = w + (size_t)s * H;
    float v[PER];
    float ss = 0.0f;
@LOAD@    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (int i = 0; i < 8; ++i) tot += part[i];
    const float inv = metal::precise::rsqrt(tot / float(H) + eps[0]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        if (k < H) {
            const T o = T(v[i] * inv * (1.0f + float(wp[k])));
            xn[base + k] = o;
            xs[k] = o;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Thread t works for inject row r = t / 64 (HC * 64 == 256 threads) as
    // the tail kernel's thread 64 s + t % 64: the rows are independent sums,
    // each accumulated in the tail kernel's order.
    constexpr int PF = hc_pack_factor<BITS_I>();
    constexpr int BP = hc_bytes_per_pack<BITS_I>();
    constexpr int ROW_BYTES = K * BP / PF;
    constexpr int GROUPS = K / 64;
    constexpr int PERI = K / 256;
    const int r = int(t) / 64;
    const int vt = int(t) % 64;
    float res = 0.0f;
    float xv[PF];
    const int e0 = (64 * int(s) + vt) * PERI;
    const device uint8_t* wr = (const device uint8_t*)inject_w + r * ROW_BYTES;
    for (int e = e0; e < e0 + PERI; e += PF) {
        const float sum = hc_load_vector_tg<T, PF, BITS_I>(xs + (e - int(s) * H), xv);
        const int gi = e / 64;
        res += hc_qdot<PF, BITS_I>(
            wr + e * BP / PF, xv, float(inject_s[r * GROUPS + gi]),
            float(inject_b[r * GROUPS + gi]), sum);
    }
    const float vv = simd_sum(res);
    if (lane == 0) {
        inj_part[((size_t)row * HC + r) * 8 + 2 * s + (vt / 32)] = vv;
    }
"""

_WRITE_LOAD = r"""
    const device T* bp = branch + (size_t)row * H;
    const float g = float(gate[(size_t)row * HC + s]);
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = 0.0f;
        if (k < H) {
            const T y = T(float(x[base + k]) + float(T(float(bp[k]) * g)));
            y_out[base + k] = y;
            v[i] = float(y);
        }
        ss += v[i] * v[i];
    }
"""
_PLAIN_LOAD = r"""
    for (int i = 0; i < PER; ++i) {
        const int k = t + i * 256;
        v[i] = (k < H) ? float(x[base + k]) : 0.0f;
        ss += v[i] * v[i];
    }
"""
_NORM_SOURCES = {
    True: _NORM_TEMPLATE.replace("@LOAD@", _WRITE_LOAD),
    False: _NORM_TEMPLATE.replace("@LOAD@", _PLAIN_LOAD),
}


# Down projection (x [M, K] @ dequant(w)[N, K]^T) with the eager divide by HC
# and the compiled SiLU applied to the bf16 product.
_DOWN_SOURCE = r"""
    constexpr int BKP = 64 + 16 / sizeof(T);
    using L = HcnLoader<BITS>;
    constexpr int KW = K * L::BPP / L::PF;
    constexpr int KG = K / 64;
    threadgroup T Ws[64 * BKP];
    const int M = int(x_shape[0]);
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const ushort lane = ushort(thread_index_in_simdgroup);
    const int y_row = int(threadgroup_position_in_grid.y) * 64;
    const int y_col = int(threadgroup_position_in_grid.x) * 64;
    const int bi = (L::NR * int(t)) / L::BCP;
    const int bj = (L::NR * int(t)) % L::BCP;
    const device uint8_t* wsrc =
        (const device uint8_t*)w + (size_t)(y_col + bi) * KW + bj * L::BPP;
    const device T* ssrc = scales + (size_t)(y_col + bi) * KG;
    const device T* bsrc = biases + (size_t)(y_col + bi) * KG;
    threadgroup T* wdst = Ws + bi * BKP + bj * L::PF;
    const short tm = 32 * short(sg / 2);
    const short tn = 32 * short(sg % 2);
    const short2 fc = hcn_frag_coord(lane);
    const int rows = M - (y_row + tm);
    const device T* xa = x + (size_t)(y_row + tm) * K;
    hcn_acc C[2][2] = {{hcn_acc(0), hcn_acc(0)}, {hcn_acc(0), hcn_acc(0)}};
    if (y_row + 64 <= M) {
        hcn_gemm<T, BITS, BKP, true>(
            C, xa, K, Ws + tn * BKP, wsrc, ssrc, bsrc, wdst, fc, rows);
    } else {
        hcn_gemm<T, BITS, BKP, false>(
            C, xa, K, Ws + tn * BKP, wsrc, ssrc, bsrc, wdst, fc, rows);
    }
    for (short mm = 0; mm < 2; mm++) {
        for (short r = 0; r < 2; r++) {
            const int lr = mm * 16 + fc.y + r * 8;
            if (lr >= rows) continue;
            device T* op = act + (size_t)(y_row + tm + lr) * N + y_col + tn + fc.x;
            for (short nn = 0; nn < 2; nn++) {
                for (short c = 0; c < 4; c++) {
                    const T v = static_cast<T>(C[mm][nn][r * 4 + c]);
                    const T q = v / T(HC);
                    const T sgm = hcn_sigmoid(q);
                    op[nn * 16 + c] = q * sgm;
                }
            }
        }
    }
"""

# Up projection over column tiles {s * H + h0 + j : s < 4, j < 16}: simdgroup
# column half 0 holds streams 0 and 1, half 1 streams 2 and 3, each as one
# 16-column fragment. The epilogue repeats the tail kernel per element:
# g = T(sigmoid(T(up))), p = T(g * xn), acc = p0, T(acc + p1), T(. + p2),
# T(. + p3), mixed = T(acc / HC). Threadgroups of the first column tile also
# finish the inject weights from the norm kernel's partial sums.
_UP_SOURCE = r"""
    constexpr int BKP = 64 + 16 / sizeof(T);
    using L = HcnLoader<BITS>;
    constexpr int KW = R * L::BPP / L::PF;
    constexpr int KG = R / 64;
    constexpr int WIDTH = HC * H;
    threadgroup T Ws[64 * BKP];
    threadgroup T half01[64 * 16];
    const int M = int(act_shape[0]);
    const uint t = thread_index_in_threadgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const ushort lane = ushort(thread_index_in_simdgroup);
    const int y_row = int(threadgroup_position_in_grid.y) * 64;
    const int h0 = int(threadgroup_position_in_grid.x) * 16;
    const int bi = (L::NR * int(t)) / L::BCP;
    const int bj = (L::NR * int(t)) % L::BCP;
    const int wrow = (bi / 16) * H + h0 + (bi % 16);
    const device uint8_t* wsrc =
        (const device uint8_t*)w + (size_t)wrow * KW + bj * L::BPP;
    const device T* ssrc = scales + (size_t)wrow * KG;
    const device T* bsrc = biases + (size_t)wrow * KG;
    threadgroup T* wdst = Ws + bi * BKP + bj * L::PF;
    const short tm = 32 * short(sg / 2);
    const short tn = 32 * short(sg % 2);
    const short2 fc = hcn_frag_coord(lane);
    const int rows = M - (y_row + tm);
    const device T* xa = act + (size_t)(y_row + tm) * R;
    // This simdgroup's epilogue holds streams s0 and s0 + 1; their normed
    // inputs are fetched before the K loop.
    const int s0 = 2 * int(sg % 2);
    vec<T, 4> xq[2][2][2];
    for (short mm = 0; mm < 2; mm++) {
        for (short r = 0; r < 2; r++) {
            const int lr = mm * 16 + fc.y + r * 8;
            const int row = lr < rows ? y_row + tm + lr : 0;
            const device T* xr = xn + (size_t)row * WIDTH + h0 + fc.x;
            for (short nn = 0; nn < 2; nn++) {
                xq[mm][r][nn] = *(const device vec<T, 4>*)(xr + (s0 + nn) * H);
            }
        }
    }
    hcn_acc C[2][2] = {{hcn_acc(0), hcn_acc(0)}, {hcn_acc(0), hcn_acc(0)}};
    if (y_row + 64 <= M) {
        hcn_gemm<T, BITS, BKP, true>(
            C, xa, R, Ws + tn * BKP, wsrc, ssrc, bsrc, wdst, fc, rows);
    } else {
        hcn_gemm<T, BITS, BKP, false>(
            C, xa, R, Ws + tn * BKP, wsrc, ssrc, bsrc, wdst, fc, rows);
    }
    float p[2][2][8];
    for (short mm = 0; mm < 2; mm++) {
        for (short r = 0; r < 2; r++) {
            for (short nn = 0; nn < 2; nn++) {
                for (short c = 0; c < 4; c++) {
                    const T up = static_cast<T>(C[mm][nn][r * 4 + c]);
                    const T g = T(1.0f / (1.0f + metal::exp(-float(up))));
                    p[mm][nn][r * 4 + c] =
                        float(T(float(g) * float(xq[mm][r][nn][c])));
                }
            }
        }
    }
    if (s0 == 0) {
        for (short mm = 0; mm < 2; mm++) {
            for (short r = 0; r < 2; r++) {
                const int lr = mm * 16 + fc.y + r * 8;
                if (lr >= rows) continue;
                for (short c = 0; c < 4; c++) {
                    const int e = r * 4 + c;
                    half01[(tm + lr) * 16 + fc.x + c] = T(p[mm][0][e] + p[mm][1][e]);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s0 != 0) {
        for (short mm = 0; mm < 2; mm++) {
            for (short r = 0; r < 2; r++) {
                const int lr = mm * 16 + fc.y + r * 8;
                if (lr >= rows) continue;
                device T* op = mixed + (size_t)(y_row + tm + lr) * H + h0 + fc.x;
                for (short c = 0; c < 4; c++) {
                    const int e = r * 4 + c;
                    float acc = float(half01[(tm + lr) * 16 + fc.x + c]);
                    acc = float(T(acc + p[mm][0][e]));
                    acc = float(T(acc + p[mm][1][e]));
                    op[c] = T(acc * (1.0f / float(HC)));
                }
            }
        }
    }
    if (threadgroup_position_in_grid.x == 0) {
        for (int i = int(t); i < 64 * HC; i += 128) {
            const int row = y_row + i / HC;
            if (row >= M) continue;
            const int r = i % HC;
            const device float* pp = inj_part + ((size_t)row * HC + r) * 8;
            float v = 0.0f;
            for (int j = 0; j < 8; ++j) v += pp[j];
            const float q = float(T(float(T(v)) / float(HC)));
            const float gate = float(T(1.0f / (1.0f + metal::exp(-q))));
            inj[(size_t)row * HC + r] = T(2.0f * gate);
        }
    }
"""


def _kernel(name, input_names, output_names, source, header):
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            header=header,
            source=source,
            ensure_row_contiguous=True,
        )
        _KERNELS[name] = kernel
    return kernel


def plain_qmm_nax(rows: int, n_out: int) -> bool:
    """Whether MLX's QuantizedMatmul uses the unsplit NAX qmm for this shape.

    MLX 0.32 splits K for few output tiles (``qmm_splitk``: fewer than ~512
    32x32 tiles; M5 source builds also split on the NAX path below 256 64x64
    tiles), which sums in a different order. Stay on MLX's kernel there.
    """
    t64 = -(-n_out // 64) * -(-rows // 64)
    t32 = -(-n_out // 32) * -(-rows // 32)
    return rows > 64 and t64 >= 256 and 512 // t32 <= 1


def norm_inject(module, flat, write, rows, hc, hidden, dtype, eps, inject):
    """Stream norm (+ pending write) and the inject projection's partial sums.

    Returns ``(written or None, normed [rows, width], inj_part [rows, hc, 8])``.
    """
    width = hc * hidden
    inputs = [flat]
    names = ["x"]
    if write is not None:
        inputs += [write[0], write[1]]
        names += ["branch", "gate"]
    inputs += [module.hc_norm.weight, eps, inject.weight, inject.scales, inject.biases]
    names += ["w", "eps", "inject_w", "inject_s", "inject_b"]
    out_names = (["y_out"] if write is not None else []) + ["xn", "inj_part"]
    shapes = ([(rows, width)] if write is not None else []) + [
        (rows, width),
        (rows, hc, 8),
    ]
    dtypes = ([dtype] if write is not None else []) + [dtype, mx.float32]
    name = "omlx_qwen4_hc_prefill_norm_inject" + ("_write" if write is not None else "")
    outs = _kernel(
        name, names, out_names, _NORM_SOURCES[write is not None], _QDOT_HEADER + _TG_LOAD
    )(
        inputs=inputs,
        template=[
            ("T", dtype),
            ("K", width),
            ("H", hidden),
            ("HC", hc),
            ("BITS_I", inject.bits),
        ],
        grid=(256, hc, rows),
        threadgroup=(256, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    if write is not None:
        return outs[0], outs[1], outs[2]
    return None, outs[0], outs[1]


def down_silu(normed, down, rows, hc):
    """silu(down(normed) / hc) with MLX's rounding, [rows, lowrank]."""
    width = normed.shape[-1]
    lowrank = down.weight.shape[0]
    return _kernel(
        "omlx_qwen4_hc_prefill_down_silu",
        ["x", "w", "scales", "biases"],
        ["act"],
        _DOWN_SOURCE,
        _NAX_HEADER,
    )(
        inputs=[normed, down.weight, down.scales, down.biases],
        template=[
            ("T", normed.dtype),
            ("BITS", down.bits),
            ("K", width),
            ("N", lowrank),
            ("HC", hc),
        ],
        grid=(128 * (lowrank // _BN), -(-rows // _BM), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, lowrank)],
        output_dtypes=[normed.dtype],
    )[0]


def up_tail(act, up, normed, inj_part, rows, hc, hidden):
    """Mixed input [rows, hidden] and inject weights [rows, hc] (from ``inj_part``)."""
    lowrank = act.shape[-1]
    dtype = act.dtype
    return _kernel(
        "omlx_qwen4_hc_prefill_up_tail",
        ["act", "w", "scales", "biases", "xn", "inj_part"],
        ["mixed", "inj"],
        _UP_SOURCE,
        _NAX_HEADER,
    )(
        inputs=[act, up.weight, up.scales, up.biases, normed, inj_part],
        template=[
            ("T", dtype),
            ("BITS", up.bits),
            ("R", lowrank),
            ("H", hidden),
            ("HC", hc),
        ],
        grid=(128 * (hidden // UP_COLS), -(-rows // _BM), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, hidden), (rows, hc)],
        output_dtypes=[dtype, dtype],
    )
