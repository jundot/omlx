// SPDX-License-Identifier: Apache-2.0
// oQ Q8 W8A8 GEMM on the M5 NAX tensor units.
//
// Packed Q8 GS64 affine weights and their scales/biases are read as the
// checkpoint stores them: codes as bytes in uint32 words, metadata as
// [N, K/64]. Codes are centered to signed INT8 in registers and multiplied
// against dynamically quantized INT8 activations through the int8 x int8 ->
// int32 datapath. No unpacked weight matrix or metadata copy is written to
// device memory.
//
// MLX affine dequantization is w = s * q + b with q in [0, 255]. Flipping the
// top bit of each byte turns q into q - 128 as a two's-complement INT8, so
//
//   sum_k a_k w_k = s * (acc + 128 * r) + b * r
//
// per GS64 group, where acc = sum a_k * (q_k - 128) is the INT32 tensor-op
// result and r = sum a_k is the group sum Stage A already produces. The
// integers acc and 128 * r are far below 2^24, so the FP32 add is exact.
//
// The tensor op computes C[16 x 32] = A[16 x 16] * B[32 x 16]^T per K step.
// Weights are A (16 rows per fragment) and activations are B (32 tokens), so
// a lane's accumulator elements sit in the four weight rows whose codes it
// loads, and their scales and biases are scalar loads from [N, K/64]. Ra and
// Sa are token-contiguous ([K/64, M]) and are the four-wide loads.
//
// A lane's 16 codes of one group are four words, one per micro-K step, so the
// activation is read in checkpoint K order, not Stage A v8's permuted order.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

// clang-format off
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

#include "oq_a8_decode.h"
// clang-format on

using namespace metal;
using namespace mlx::steel;
using namespace omlx::oq_a8;

namespace {
constant constexpr int kFragM = 16;
constant constexpr int kFragN = 32;
constant constexpr int kFragK = 16;
constant constexpr int kElemsPerFrag = 8;
constant constexpr int kDestElems = 2 * kElemsPerFrag;
constant constexpr int kStepsPerGroup = kGroupSize / kFragK; // 4
constant constexpr uint32_t kCenter = 0x80808080u;
} // namespace

// One simdgroup owns 32 weight rows (two A fragments) x 32 tokens (one B
// fragment). WM simdgroups tile tokens and WN tile weight rows.
template <typename T, int ACT_MODE, int WM, int WN>
[[kernel]] void oq_q8_a8_qmm_t_nax(
    const device int8_t* qa [[buffer(0)]],
    const device float* sa [[buffer(1)]],
    const device short* ra [[buffer(2)]],
    const device uint32_t* w [[buffer(3)]],
    const device T* scales [[buffer(4)]], // [N, K/64]
    const device T* biases [[buffer(5)]], // [N, K/64]
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = kFragN * WM; // tokens per threadgroup
  constexpr int BN = 2 * kFragM * WN; // weight rows per threadgroup
  constexpr int words = oq_group_words(8);

  const int groups = K / kGroupSize;
  const int sg_m = int(simd_gid) % WM;
  const int sg_n = int(simd_gid) / WM;
  const int tok_base = int(tid.y) * BM + sg_m * kFragN;
  const int n_base = int(tid.x) * BN + sg_n * (2 * kFragM);

  const short2 coord = BaseNAXFrag::get_coord();

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      kFragM,
      kFragN,
      kFragK,
      /* transpose_left = */ false,
      /* transpose_right = */ true,
      /* relaxed_precision = */ false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = mpp::tensor_ops::matmul2d_descriptor(
      kFragM,
      kFragN,
      kFragK,
      /* transpose_left = */ false,
      /* transpose_right = */ true,
      /* relaxed_precision = */ false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  mpp::tensor_ops::matmul2d<desc_set, metal::execution_simdgroup> op_set;

  auto ct_a =
      op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      int32_t>();

  float Cf[2][kDestElems];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < 2; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kDestElems; ++e) {
      Cf[i][e] = 0.0f;
    }
  }

  // Destination element e of fragment i: weight row n_base + 16 i + y + 8 r,
  // token tok_base + 16 h + x + c, with h = e >> 3, r = (e & 7) >> 2, c = e & 3.
  const int y = int(coord.y);
  const int x = int(coord.x);
  // Which 16-code run of the affine group this lane owns: 0..3.
  const int cx = x >> 2;

  const int n_lane = n_base + y;
  const device uint32_t* wbase = w + size_t(n_lane) * size_t(groups) * words;
  const int w_row = groups * words;
  const int w_stride8 = 8 * w_row;
  const int w_stride16 = kFragM * w_row;

  // Scale and bias of the lane's four weight rows n_lane + 8 r + 16 i.
  const device T* srow = scales + size_t(n_lane) * size_t(groups);
  const device T* brow = biases + size_t(n_lane) * size_t(groups);
  const size_t meta_stride8 = size_t(8) * size_t(groups);
  const size_t meta_stride16 = size_t(16) * size_t(groups);

  // Ra and Sa are [K/64, M]. Four adjacent tokens per lane are one vector load
  // when M is a multiple of 4; otherwise they fall back to clamped scalars.
  const bool vec_ok = (M & 3) == 0;

  for (int g = 0; g < groups; ++g) {
    uint4 wg[4];
    STEEL_PRAGMA_UNROLL
    for (int q = 0; q < 4; ++q) {
      const device uint32_t* wr = wbase + (q & 1) * w_stride8 +
          (q >> 1) * w_stride16 + size_t(g) * words;
      wg[q] = reinterpret_cast<const device uint4*>(wr)[cx];
    }

    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < kStepsPerGroup; ++t) {
      // Right operand: 32 tokens, rows past M read row M-1 and are never
      // stored.
      STEEL_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        const int base = (q >> 1) * kElemsPerFrag + (q & 1) * 4;
        const int m = min(tok_base + y + (q & 1) * 8 + (q >> 1) * 16, M - 1);
        const char4 quad = as_type<char4>(
            *reinterpret_cast<const device uint32_t*>(
                qa + size_t(m) * size_t(K) + size_t(g) * kGroupSize +
                size_t(cx) * 16 + size_t(t) * 4));
        ct_b[base + 0] = quad.x;
        ct_b[base + 1] = quad.y;
        ct_b[base + 2] = quad.z;
        ct_b[base + 3] = quad.w;
      }
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < 2; ++i) {
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          const char4 quad = as_type<char4>(wg[i * 2 + r][t] ^ kCenter);
          ct_a[r * 4 + 0] = quad.x;
          ct_a[r * 4 + 1] = quad.y;
          ct_a[r * 4 + 2] = quad.z;
          ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) {
          if (i == 0) {
            op_set.run(ct_a, ct_b, acc0);
          } else {
            op_set.run(ct_a, ct_b, acc1);
          }
        } else {
          if (i == 0) {
            op.run(ct_a, ct_b, acc0);
          } else {
            op.run(ct_a, ct_b, acc1);
          }
        }
      }
    }

    // Token-side group sums: tokens tok_base + 16 h + x + c.
    const device short* rrow = ra + size_t(g) * size_t(M);
    float r_g[2][4];
    float r_c[2][4];
    STEEL_PRAGMA_UNROLL
    for (int h = 0; h < 2; ++h) {
      const int t0 = tok_base + h * kFragM + x;
      if (vec_ok) {
        const short4 v =
            *reinterpret_cast<const device short4*>(rrow + min(t0, M - 4));
        STEEL_PRAGMA_UNROLL
        for (int c = 0; c < 4; ++c) {
          r_g[h][c] = float(v[c]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (int c = 0; c < 4; ++c) {
          r_g[h][c] = float(rrow[min(t0 + c, M - 1)]);
        }
      }
      STEEL_PRAGMA_UNROLL
      for (int c = 0; c < 4; ++c) {
        r_c[h][c] = 128.0f * r_g[h][c];
      }
    }

    // Weight-side group metadata of the lane's four rows.
    float swv[2][2];
    float bwv[2][2];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        const size_t off = size_t(r) * meta_stride8 + size_t(i) * meta_stride16;
        swv[i][r] = float(srow[off + g]);
        bwv[i][r] = float(brow[off + g]);
      }
    }

    if (ACT_MODE == 0) {
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kDestElems; ++e) {
        const int h = e >> 3;
        const int r = (e & 7) >> 2;
        const int c = e & 3;
        Cf[0][e] = metal::fma(
            swv[0][r],
            float(acc0[e]) + r_c[h][c],
            metal::fma(bwv[0][r], r_g[h][c], Cf[0][e]));
        Cf[1][e] = metal::fma(
            swv[1][r],
            float(acc1[e]) + r_c[h][c],
            metal::fma(bwv[1][r], r_g[h][c], Cf[1][e]));
      }
    } else {
      const device float* arow = sa + size_t(g) * size_t(M);
      float s_g[2][4];
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        const int t0 = tok_base + h * kFragM + x;
        if (vec_ok) {
          const float4 v =
              *reinterpret_cast<const device float4*>(arow + min(t0, M - 4));
          STEEL_PRAGMA_UNROLL
          for (int c = 0; c < 4; ++c) {
            s_g[h][c] = v[c];
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (int c = 0; c < 4; ++c) {
            s_g[h][c] = arow[min(t0 + c, M - 1)];
          }
        }
      }
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kDestElems; ++e) {
        const int h = e >> 3;
        const int r = (e & 7) >> 2;
        const int c = e & 3;
        Cf[0][e] = metal::fma(
            s_g[h][c],
            metal::fma(
                swv[0][r], float(acc0[e]) + r_c[h][c], bwv[0][r] * r_g[h][c]),
            Cf[0][e]);
        Cf[1][e] = metal::fma(
            s_g[h][c],
            metal::fma(
                swv[1][r], float(acc1[e]) + r_c[h][c], bwv[1][r] * r_g[h][c]),
            Cf[1][e]);
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < 2; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kDestElems; ++e) {
      const int h = e >> 3;
      const int r = (e & 7) >> 2;
      const int c = e & 3;
      const int m = tok_base + h * kFragM + x + c;
      if (m < M) {
        const int n = n_base + i * kFragM + y + r * 8;
        const float v = ACT_MODE == 0 ? sa[m] * Cf[i][e] : Cf[i][e];
        out[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(v);
      }
    }
  }
}

#define instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, wm, wn)                \
  instantiate_kernel(                                                         \
      "oq_q8_a8_qmm_t_nax_am" #act_mode "_" #type "_wm_" #wm "_wn_" #wn,      \
      oq_q8_a8_qmm_t_nax,                                                     \
      type,                                                                   \
      act_mode,                                                               \
      wm,                                                                     \
      wn)

// Two tiles, both BN = 64 weight rows: (1,2) for long prompts and (2,2) for
// up to 1024 rows, where the wider token tile wins. The host picks by M.
#define instantiate_oq_q8_a8_qmm_t_nax_tiles(act_mode, type)                  \
  instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, 1, 2);                       \
  instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, 2, 2)

instantiate_oq_q8_a8_qmm_t_nax_tiles(0, float16_t);
instantiate_oq_q8_a8_qmm_t_nax_tiles(0, bfloat16_t);
instantiate_oq_q8_a8_qmm_t_nax_tiles(1, float16_t);
instantiate_oq_q8_a8_qmm_t_nax_tiles(1, bfloat16_t);

#endif // __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
