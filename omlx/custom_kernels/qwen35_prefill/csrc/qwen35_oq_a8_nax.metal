// oQ mixed-bit QxA8 GEMM on the M5 NAX tensor units.
//
// Packed Q4/Q5 affine weights are decoded straight into INT8 tensor-op
// fragment registers and multiplied against dynamically quantized INT8
// activations through the int8 x int8 -> int32 datapath, with the affine
// correction applied at every GS64 boundary.
//
//        Qa (INT8, device)  ------------------\
//                                              >-- INT8 NAX matmul2d --> INT32
//        packed Q4/Q5 uint32 -> register decode/
//                                                        |
//                                          GS64 affine correction (FP32)
//                                                        |
//                                                     BF16/FP16
//
// Nothing is staged through threadgroup memory and no unpacked INT8 weight
// matrix is ever materialized in device memory, so the kernel keeps
// oQ's bandwidth advantage: the weights are read exactly once, still packed.
//
// Compiled into the separate omlx_qwen35_prefill_kernels_nax metallib with a
// 26.2 deployment floor; the C++ op only loads it when the runtime reports NAX
// support.

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

// One NAX fragment is 16x16 with 8 elements per lane. The matmul primitive
// below is 16(m) x 32(n) x 16(k). A micro-K of 16 is what the correction
// needs: four of these steps cover one GS64 affine group exactly.
constant constexpr int kFragM = 16;
constant constexpr int kFragN = 32;
constant constexpr int kFragK = 16;
constant constexpr int kElemsPerFrag = 8;
constant constexpr int kStepsPerGroup = kGroupSize / kFragK; // 4
// The 32-wide N of one matmul is carried as two adjacent 16-wide fragments.
constant constexpr int kFragNHalf = kFragN / 2;

using frag_i8 = BaseNAXFrag::dtype_frag_t<int8_t>;

// int8 x int8 -> int32 on the tensor units.
//
// relaxed_precision is false: it exists to let the implementation trade
// accuracy for speed on float paths, and the integer dot must stay exact for
// the group accumulator to be valid.
inline void oq_mma_i8(
    thread int32_t (&Cn0)[kElemsPerFrag],
    thread int32_t (&Cn1)[kElemsPerFrag],
    const thread frag_i8& A,
    const thread frag_i8& Bn0,
    const thread frag_i8& Bn1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      kFragM,
      kFragN,
      kFragK,
      /* transpose_left = */ false,
      // The weight is stored [N, K] row-major, so the right operand is
      // supplied transposed and no repacking is needed.
      /* transpose_right = */ true,
      /* relaxed_precision = */ false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  auto ct_a =
      gemm_op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      gemm_op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      int32_t>();

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kElemsPerFrag; i++) {
    ct_a[i] = A[i];
    ct_b[i] = Bn0[i];
    ct_b[kElemsPerFrag + i] = Bn1[i];
    ct_c[i] = Cn0[i];
    ct_c[kElemsPerFrag + i] = Cn1[i];
  }

  gemm_op.run(ct_a, ct_b, ct_c);

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kElemsPerFrag; i++) {
    Cn0[i] = ct_c[i];
    Cn1[i] = ct_c[kElemsPerFrag + i];
  }
}

// Decode one 16(n) x 16(k) weight fragment out of the packed stream.
//
// The fragment layout hands this lane rows {n_off, n_off + 8} and four
// adjacent K positions, which is why oq_decode4() is the right decode unit:
// four consecutive Q5 codes span at most two uint32 words.
template <int BITS>
inline void oq_load_weight_frag(
    thread frag_i8& dst,
    const device uint32_t* w,
    int n0,
    int groups,
    int group,
    int k_in_group,
    short2 coord) {
  constexpr int words = oq_group_words(BITS);
  const int c0 = k_in_group + int(coord.x);

  const device uint32_t* row0 =
      w + (size_t(n0 + int(coord.y)) * size_t(groups) + size_t(group)) * words;
  const device uint32_t* row1 = row0 + size_t(8) * size_t(groups) * words;

  int8_t q0[4];
  int8_t q1[4];
  oq_decode4<BITS>(row0, c0, q0);
  oq_decode4<BITS>(row1, c0, q1);

  STEEL_PRAGMA_UNROLL
  for (short j = 0; j < 4; j++) {
    dst[j] = q0[j];
    dst[4 + j] = q1[j];
  }
}

// ACT_MODE 0: Sa is [M]. ACT_MODE 1: Sa is [M, K/64].
//
// BM x BN output per threadgroup, WM x WN simdgroups, so each simdgroup owns
// TM fragment-rows of 16 and TN fragment-columns of 32.
template <
    typename T,
    int BITS,
    int ACT_MODE,
    int BM,
    int BN,
    int WM,
    int WN>
[[kernel]] void oq_a8_qmm_t_nax(
    const device int8_t* qa [[buffer(0)]],
    const device float* sa [[buffer(1)]],
    const device short* ra [[buffer(2)]],
    const device uint32_t* w [[buffer(3)]],
    const device T* scales [[buffer(4)]],
    const device T* biases [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int TM = BM / (WM * kFragM);
  constexpr int TN = BN / (WN * kFragN);
  static_assert(TM * WM * kFragM == BM, "BM must tile into 16-row fragments");
  static_assert(TN * WN * kFragN == BN, "BN must tile into 32-column fragments");

  const int groups = K / kGroupSize;

  // group_dims is (32, WM, WN), so the linear simdgroup index runs down WM
  // first.
  const int sg_m = int(simd_gid) % WM;
  const int sg_n = int(simd_gid) / WM;

  const int row_base = int(tid.y) * BM + sg_m * (TM * kFragM);
  const int col_base = int(tid.x) * BN + sg_n * (TN * kFragN);

  const short2 coord = BaseNAXFrag::get_coord();

  // Row indices this lane owns, per fragment-row and per half of the
  // fragment. N is a multiple of BN (checked host-side) so the column indices
  // never need a bound check; M is arbitrary, so rows do.
  int m_idx[TM][2];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (int r = 0; r < 2; r++) {
      m_idx[i][r] = row_base + i * kFragM + int(coord.y) + r * 8;
    }
  }

  // Column indices for the affine scale/bias lookups. Element e of an output
  // fragment sits at row coord.y + (e >> 2) * 8 and column coord.x + (e & 3).
  int n_idx[TN][2][4];
  STEEL_PRAGMA_UNROLL
  for (int j = 0; j < TN; j++) {
    STEEL_PRAGMA_UNROLL
    for (int h = 0; h < 2; h++) {
      STEEL_PRAGMA_UNROLL
      for (int c = 0; c < 4; c++) {
        n_idx[j][h][c] = col_base + j * kFragN + h * kFragNHalf + int(coord.x) + c;
      }
    }
  }

  // Per-row activation scale is loop-invariant; the groupwise one is not.
  float sa_row[TM][2];
  if (ACT_MODE == 0) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (int r = 0; r < 2; r++) {
        sa_row[i][r] = m_idx[i][r] < M ? sa[m_idx[i][r]] : 0.0f;
      }
    }
  }

  // Persistent FP32 accumulator: lives across the whole K dimension.
  float Cf[TM][TN][2][kElemsPerFrag];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < TN; j++) {
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; h++) {
        STEEL_PRAGMA_UNROLL
        for (int e = 0; e < kElemsPerFrag; e++) {
          Cf[i][j][h][e] = 0.0f;
        }
      }
    }
  }

  for (int g = 0; g < groups; g++) {
    // Group accumulator: INT32, lifetime exactly K = 64. The weight scale and
    // bias change every 64 K values, so folding the integer dot across the
    // whole K extent and scaling once at the end would be wrong.
    int32_t Cg[TM][TN][2][kElemsPerFrag];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        STEEL_PRAGMA_UNROLL
        for (int h = 0; h < 2; h++) {
          STEEL_PRAGMA_UNROLL
          for (int e = 0; e < kElemsPerFrag; e++) {
            Cg[i][j][h][e] = 0;
          }
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < kStepsPerGroup; t++) {
      const int k0 = g * kGroupSize + t * kFragK;

      // Activation fragments are shared across every column tile.
      frag_i8 Af[TM];
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < TM; i++) {
        BaseNAXFrag::load_rows<int8_t>(
            Af[i],
            qa + size_t(row_base) * size_t(K) + size_t(k0),
            K,
            Int<1>{},
            M - row_base,
            i * kFragM,
            0);
      }

      // Weight fragments are shared across every row tile: decoded once per
      // (column tile, K step) and reused for all TM.
      frag_i8 Bf[TN][2];
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        STEEL_PRAGMA_UNROLL
        for (int h = 0; h < 2; h++) {
          oq_load_weight_frag<BITS>(
              Bf[j][h],
              w,
              col_base + j * kFragN + h * kFragNHalf,
              groups,
              g,
              t * kFragK,
              coord);
        }
      }

      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < TM; i++) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < TN; j++) {
          oq_mma_i8(Cg[i][j][0], Cg[i][j][1], Af[i], Bf[j][0], Bf[j][1]);
        }
      }
    }

    // GS64 affine correction:
    //
    //   contribution = Sa * ( Sw * D + Bw * Ra )
    //
    // with D the INT32 dot just finished and Ra the activation group sum.
    float sw[TN][2][4];
    float bw[TN][2][4];
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < TN; j++) {
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; h++) {
        STEEL_PRAGMA_UNROLL
        for (int c = 0; c < 4; c++) {
          const size_t off = size_t(n_idx[j][h][c]) * size_t(groups) + size_t(g);
          sw[j][h][c] = float(scales[off]);
          bw[j][h][c] = float(biases[off]);
        }
      }
    }

    float ra_g[TM][2];
    float sa_g[TM][2];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (int r = 0; r < 2; r++) {
        const int m = m_idx[i][r];
        const bool live = m < M;
        const size_t off = size_t(m) * size_t(groups) + size_t(g);
        ra_g[i][r] = live ? float(ra[off]) : 0.0f;
        if (ACT_MODE == 0) {
          sa_g[i][r] = sa_row[i][r];
        } else {
          sa_g[i][r] = live ? sa[off] : 0.0f;
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        STEEL_PRAGMA_UNROLL
        for (int h = 0; h < 2; h++) {
          STEEL_PRAGMA_UNROLL
          for (int e = 0; e < kElemsPerFrag; e++) {
            const int r = e >> 2;
            const int c = e & 3;
            const float d = float(Cg[i][j][h][e]);
            Cf[i][j][h][e] +=
                sa_g[i][r] * (sw[j][h][c] * d + bw[j][h][c] * ra_g[i][r]);
          }
        }
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < TN; j++) {
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; h++) {
        STEEL_PRAGMA_UNROLL
        for (int e = 0; e < kElemsPerFrag; e++) {
          const int m = m_idx[i][e >> 2];
          if (m < M) {
            out[size_t(m) * size_t(N) + size_t(n_idx[j][h][e & 3])] =
                static_cast<T>(Cf[i][j][h][e]);
          }
        }
      }
    }
  }
}

#define instantiate_oq_a8_qmm_t_nax(bits, act_mode, type, bm, bn, wm, wn)     \
  instantiate_kernel(                                                         \
      "oq_a8_qmm_t_nax_q" #bits "_am" #act_mode "_" #type "_bm_" #bm          \
      "_bn_" #bn "_wm_" #wm "_wn_" #wn,                                       \
      oq_a8_qmm_t_nax,                                                        \
      type,                                                                   \
      bits,                                                                   \
      act_mode,                                                               \
      bm,                                                                     \
      bn,                                                                     \
      wm,                                                                     \
      wn)

// Tile variants must stay in sync with oq_a8_nax_variant() in
// qwen35_oq_a8.cpp. Variant 0 is the 128x64 tile the raw INT8 sweep measured
// fastest on this device; the rest are the autotuning surface. Q4
// and Q5 are tuned independently because the Q5 decoder costs more registers.
#define instantiate_oq_a8_qmm_t_nax_tiles(bits, act_mode, type)               \
  instantiate_oq_a8_qmm_t_nax(bits, act_mode, type, 128, 64, 2, 2);           \
  instantiate_oq_a8_qmm_t_nax(bits, act_mode, type, 128, 128, 2, 2);          \
  instantiate_oq_a8_qmm_t_nax(bits, act_mode, type, 64, 64, 2, 2);            \
  instantiate_oq_a8_qmm_t_nax(bits, act_mode, type, 64, 128, 2, 2);           \
  instantiate_oq_a8_qmm_t_nax(bits, act_mode, type, 256, 64, 4, 2);           \
  instantiate_oq_a8_qmm_t_nax(bits, act_mode, type, 128, 64, 4, 1);           \
  instantiate_oq_a8_qmm_t_nax(bits, act_mode, type, 64, 64, 1, 2)

#define instantiate_oq_a8_qmm_t_nax_bits(bits)                                \
  instantiate_oq_a8_qmm_t_nax_tiles(bits, 0, float16_t);                      \
  instantiate_oq_a8_qmm_t_nax_tiles(bits, 0, bfloat16_t);                     \
  instantiate_oq_a8_qmm_t_nax_tiles(bits, 1, float16_t);                      \
  instantiate_oq_a8_qmm_t_nax_tiles(bits, 1, bfloat16_t)

instantiate_oq_a8_qmm_t_nax_bits(4);
instantiate_oq_a8_qmm_t_nax_bits(5);

///////////////////////////////////////////////////////////////////////////////
// Persistent-accumulator GEMM
///////////////////////////////////////////////////////////////////////////////
//
// Same math as oq_a8_qmm_t_nax above, restructured around what the tensor-op
// API actually wants.
//
// The first version rebuilt the operand and destination cooperative tensors on
// every 16-wide K step and copied 32 accumulator registers in and out of each
// run(). Over a 5120-deep K that is 320 round trips per output fragment, and it
// costs more than the arithmetic; the register pressure also forced the tile
// down to 32x32 per simdgroup.
//
// Here the destination cooperative tensors are created once and accumulate in
// place across all four steps of an affine group -- run() is
// multiply_accumulate, so nothing is copied out until the group closes and the
// affine correction folds it into FP32. Operands are filled straight from
// device memory and the packed weight stream, with no intermediate fragment
// arrays.
//
// Two row fragments per simdgroup, held as named variables: a cooperative
// tensor owns per-thread storage and cannot be placed in an array.

// One 16x32 destination gives each lane 16 elements: two 16x16 halves of 8.
constant constexpr int kDestElems = 2 * kElemsPerFrag;

template <typename T, int BITS, int ACT_MODE, int WM, int WN>
[[kernel]] void oq_a8_qmm_t_nax_v2(
    const device int8_t* qa [[buffer(0)]],
    const device float* sa [[buffer(1)]],
    const device short* ra [[buffer(2)]],
    const device uint32_t* w [[buffer(3)]],
    const device T* scales [[buffer(4)]],
    const device T* biases [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int TM = 2;
  constexpr int BM = TM * kFragM * WM;
  constexpr int BN = kFragN * WN;
  constexpr int words = oq_group_words(BITS);

  const int groups = K / kGroupSize;
  const int sg_m = int(simd_gid) % WM;
  const int sg_n = int(simd_gid) / WM;
  const int row_base = int(tid.y) * BM + sg_m * (TM * kFragM);
  const int col_base = int(tid.x) * BN + sg_n * kFragN;

  const short2 coord = BaseNAXFrag::get_coord();

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      kFragM,
      kFragN,
      kFragK,
      /* transpose_left = */ false,
      /* transpose_right = */ true,
      /* relaxed_precision = */ false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  // The first K step of each affine group overwrites the accumulator rather
  // than adding to it, so the group boundary costs no register clear.
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

  // Output coordinates for this lane. Element e covers half h = e >> 3, and
  // within a half sits at row coord.y + ((e & 7) >> 2) * 8, column
  // coord.x + (e & 3).
  int n_of[kDestElems];
  int n_col[8];
  int m_of[TM][kDestElems];
  STEEL_PRAGMA_UNROLL
  for (int c = 0; c < 8; ++c) {
    n_col[c] = col_base + (c >> 2) * kFragM + int(coord.x) + (c & 3);
  }
  STEEL_PRAGMA_UNROLL
  for (int e = 0; e < kDestElems; ++e) {
    const int ee = e & 7;
    n_of[e] = col_base + (e >> 3) * kFragM + int(coord.x) + (ee & 3);
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      const int m = row_base + i * kFragM + int(coord.y) + ((ee >> 2) * 8);
      m_of[i][e] = m < M ? m : -1;
    }
  }

  float sa_row[TM][2];
  if (ACT_MODE == 0) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        const int m = row_base + i * kFragM + int(coord.y) + r * 8;
        sa_row[i][r] = m < M ? sa[m] : 0.0f;
      }
    }
  }

  float Cf[TM][kDestElems];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kDestElems; ++e) {
      Cf[i][e] = 0.0f;
    }
  }

  // The four weight rows this lane decodes: two per 16-wide half.
  const device uint32_t* wrow[4];
  STEEL_PRAGMA_UNROLL
  for (int q = 0; q < 4; ++q) {
    const int n = col_base + (q >> 1) * kFragM + int(coord.y) + (q & 1) * 8;
    wrow[q] = w + size_t(n) * size_t(groups) * words;
  }

  const int m0_base = row_base + int(coord.y);

  for (int g = 0; g < groups; ++g) {
    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < kStepsPerGroup; ++t) {
      const int k_in_group = t * kFragK + int(coord.x);

      // Weight operand: shared by both row fragments, so decoded once.
      STEEL_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        int8_t quad[4];
        oq_decode4<BITS>(wrow[q] + size_t(g) * words, k_in_group, quad);
        const int base = (q >> 1) * kElemsPerFrag + (q & 1) * 4;
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          ct_b[base + j] = quad[j];
        }
      }

      const int k0 = g * kGroupSize + t * kFragK + int(coord.x);

      // The four K values a lane needs are contiguous and 4-byte aligned
      // (coord.x is a multiple of 4, k0 a multiple of 16, K a multiple of 64),
      // so they come in as one 32-bit load instead of four byte loads. This is
      // the difference between 2 and 8 load instructions per fragment, and the
      // kernel is bound by load issue rather than bandwidth.
      STEEL_PRAGMA_UNROLL
      for (int hf = 0; hf < 2; ++hf) {
        const int m_lo = m0_base + hf * kFragM;
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          const int m = m_lo + r * 8;
          const uint32_t packed_a = m < M
              ? *reinterpret_cast<const device uint32_t*>(
                    qa + size_t(m) * size_t(K) + size_t(k0))
              : 0u;
          const char4 quad = as_type<char4>(packed_a);
          ct_a[r * 4 + 0] = quad.x;
          ct_a[r * 4 + 1] = quad.y;
          ct_a[r * 4 + 2] = quad.z;
          ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) {
          if (hf == 0) {
            op_set.run(ct_a, ct_b, acc0);
          } else {
            op_set.run(ct_a, ct_b, acc1);
          }
        } else {
          if (hf == 0) {
            op.run(ct_a, ct_b, acc0);
          } else {
            op.run(ct_a, ct_b, acc1);
          }
        }
      }
    }

    // Affine correction at the GS64 boundary.
    //
    // The 16 destination elements a lane owns cover only 8 distinct columns:
    // within a 16-wide half, elements e and e+4 sit in the same column, one
    // 8 rows below the other. Loading scales/biases once per column instead of
    // once per element halves the per-group scale traffic, which is otherwise
    // the largest load term here -- these are strided by `groups` and so cost
    // a separate transaction each.
    float sw[8];
    float bw[8];
    STEEL_PRAGMA_UNROLL
    for (int c = 0; c < 8; ++c) {
      const size_t off = size_t(n_col[c]) * size_t(groups) + size_t(g);
      sw[c] = float(scales[off]);
      bw[c] = float(biases[off]);
    }

    float r_g[TM][2];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        const int m = row_base + i * kFragM + int(coord.y) + r * 8;
        const size_t off = size_t(m) * size_t(groups) + size_t(g);
        r_g[i][r] = m < M ? float(ra[off]) : 0.0f;
      }
    }

    // In per-row mode the activation scale does not depend on g, so it comes
    // out of the K loop entirely and is applied once at the store. That drops
    // a multiply per destination element per group, which at GS64 is the
    // single largest term in the correction.
    if (ACT_MODE == 0) {
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kDestElems; ++e) {
        const int r = ((e & 7) >> 2);
        const int c = ((e >> 3) << 2) | (e & 3);
        Cf[0][e] += sw[c] * float(acc0[e]) + bw[c] * r_g[0][r];
        Cf[1][e] += sw[c] * float(acc1[e]) + bw[c] * r_g[1][r];
      }
    } else {
      float s_g[TM][2];
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          const int m = row_base + i * kFragM + int(coord.y) + r * 8;
          const size_t off = size_t(m) * size_t(groups) + size_t(g);
          s_g[i][r] = m < M ? sa[off] : 0.0f;
        }
      }
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kDestElems; ++e) {
        const int r = ((e & 7) >> 2);
        const int c = ((e >> 3) << 2) | (e & 3);
        Cf[0][e] += s_g[0][r] * (sw[c] * float(acc0[e]) + bw[c] * r_g[0][r]);
        Cf[1][e] += s_g[1][r] * (sw[c] * float(acc1[e]) + bw[c] * r_g[1][r]);
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kDestElems; ++e) {
      const int m = m_of[i][e];
      if (m >= 0) {
        const float v =
            ACT_MODE == 0 ? sa_row[i][(e & 7) >> 2] * Cf[i][e] : Cf[i][e];
        out[size_t(m) * size_t(N) + size_t(n_of[e])] = static_cast<T>(v);
      }
    }
  }
}

#define instantiate_oq_a8_qmm_t_nax_v2(bits, act_mode, type, wm, wn)          \
  instantiate_kernel(                                                         \
      "oq_a8_qmm_t_nax_v2_q" #bits "_am" #act_mode "_" #type "_wm_" #wm       \
      "_wn_" #wn,                                                             \
      oq_a8_qmm_t_nax_v2,                                                     \
      type,                                                                   \
      bits,                                                                   \
      act_mode,                                                               \
      wm,                                                                     \
      wn)

#define instantiate_oq_a8_qmm_t_nax_v2_tiles(bits, act_mode, type)            \
  instantiate_oq_a8_qmm_t_nax_v2(bits, act_mode, type, 2, 2);                 \
  instantiate_oq_a8_qmm_t_nax_v2(bits, act_mode, type, 4, 2);                 \
  instantiate_oq_a8_qmm_t_nax_v2(bits, act_mode, type, 2, 4);                 \
  instantiate_oq_a8_qmm_t_nax_v2(bits, act_mode, type, 4, 4);                 \
  instantiate_oq_a8_qmm_t_nax_v2(bits, act_mode, type, 1, 4);                 \
  instantiate_oq_a8_qmm_t_nax_v2(bits, act_mode, type, 8, 2);                 \
  instantiate_oq_a8_qmm_t_nax_v2(bits, act_mode, type, 1, 2)

#define instantiate_oq_a8_qmm_t_nax_v2_bits(bits)                             \
  instantiate_oq_a8_qmm_t_nax_v2_tiles(bits, 0, float16_t);                   \
  instantiate_oq_a8_qmm_t_nax_v2_tiles(bits, 0, bfloat16_t);                  \
  instantiate_oq_a8_qmm_t_nax_v2_tiles(bits, 1, float16_t);                   \
  instantiate_oq_a8_qmm_t_nax_v2_tiles(bits, 1, bfloat16_t)

instantiate_oq_a8_qmm_t_nax_v2_bits(4);
instantiate_oq_a8_qmm_t_nax_v2_bits(5);

///////////////////////////////////////////////////////////////////////////////
// Step-transposed GEMM
///////////////////////////////////////////////////////////////////////////////
//
// Same math as oq_a8_qmm_t_nax_v2 above, with the operand loads collapsed by
// choosing which K each fragment slot stands for. It reads the checkpoint's
// own packed weight stream, unmodified, and still gets each lane its whole
// affine group in one load.
//
// v2 spends most of its remaining overhead on load issue: 32 loads per lane
// per group across the two operands. They are that many because a lane's K
// positions are scattered -- fragment column `coord.x` and K step t put it at
// 16t + coord.x, so its sixteen codes for a group sit in four runs of four,
// 16 apart: four separate reads per operand row, times four rows.
//
// That scatter is not inherent. What a lane reads is decided by which K each
// *fragment slot* stands for, and that assignment is free: the four steps
// that cover one affine group are summed before the correction
// lands, so any bijection
//
//     (step t, column group c, lane column j)  ->  k in [0, 64)
//
// is a valid schedule, and the natural k = 16t + 4c + j is simply the one
// that scatters the lane. What a step needs is four codes, not four
// *consecutive* codes, so
//
//     k = 16c + 8*(t >> 1) + 2j + (t & 1)
//
// -- the lane's contiguous 16-code run, split into even and odd positions --
// is equally valid. It hands lane c the run k in [16c, 16c + 16) for the
// whole group, which in the untouched Q4 layout is eight bytes: one uint2.
// And it is the assignment under which a step is exactly the even nibbles of
// one of those words, or its odd ones, so a nibble mask of a uint32
// reinterpreted as char4 is the fragment quad outright -- two instructions.
//
// What moves in exchange is the activation, whose K order has to follow. That
// costs nothing to keep: Stage A rebuilds Qa on every prefill anyway, so its
// order is chosen for free and never stored, and the lane still reads its
// group as one contiguous run. Nothing else is touched -- no unpacked weight
// matrix is materialized and the module's own arrays, with MLX's
// decode path over them, are exactly as loaded, so a routed projection never
// holds a second copy of its weights.
//
// Q5 rides the same schedule off the same untouched stream. Its sixteen codes
// are the 80 bits at bit 80c, which starts mid-word for odd c, so the three
// words that cover them are normalized once per group -- one variable shift,
// at group scope -- after which every per-step field offset is a compile-time
// constant.

// One 5-bit code out of a lane's normalized 96-bit Q5 window.
//
// OFF is a compile-time bit offset, so the word it lands in and whether it
// crosses into the next one are both settled at compile time: the common case
// is a single extract_bits, and only the four fields that straddle a word
// boundary pay a shift-or.
template <int OFF>
inline int8_t oq_q5_window_code(uint3 w) {
  constexpr int wi = OFF >> 5;
  constexpr int sh = OFF & 31;
  const uint32_t lo = (wi == 0) ? w.x : ((wi == 1) ? w.y : w.z);
  if (sh <= 27) {
    return static_cast<int8_t>(metal::extract_bits(lo, sh, 5));
  }
  const uint32_t hi = (wi == 0) ? w.y : w.z;
  return static_cast<int8_t>(((lo >> sh) | (hi << (32 - sh))) & 0x1fu);
}

template <typename T, int BITS, int ACT_MODE, int WM, int WN>
[[kernel]] void oq_a8_qmm_t_nax_v8(
    const device int8_t* qa [[buffer(0)]],
    const device float* sa [[buffer(1)]],
    const device short* ra [[buffer(2)]],
    const device uint32_t* w [[buffer(3)]],
    const device T* scales [[buffer(4)]],
    const device T* biases [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int TM = 2;
  constexpr int BM = TM * kFragM * WM;
  constexpr int BN = kFragN * WN;
  constexpr int words = oq_group_words(BITS);

  const int groups = K / kGroupSize;
  const int sg_m = int(simd_gid) % WM;
  const int sg_n = int(simd_gid) / WM;
  const int row_base = int(tid.y) * BM + sg_m * (TM * kFragM);
  const int col_base = int(tid.x) * BN + sg_n * kFragN;

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

  // Element e of a 16x32 destination sits in half e>>3, at row
  // coord.y + ((e & 7) >> 2) * 8 and column coord.x + (e & 3). So the 16
  // elements span two 4-wide runs of columns, one per half.
  //
  // Those coordinates are three adds from `e`, and caching all 48 of them
  // costs more registers than the accumulators themselves -- enough to spill
  // and to slow the K loop measurably. They are recomputed at the store.
  const int n_run0 = col_base + int(coord.x);

  // The four row indices are two adds from `row_base`, and the per-row
  // activation scale is only wanted once, at the store. Caching either across
  // the K loop just holds registers the accumulators need.
  const int m_base = row_base + int(coord.y);

  float Cf[TM][kDestElems];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kDestElems; ++e) {
      Cf[i][e] = 0.0f;
    }
  }

  // One base pointer and two strides rather than four row pointers: on a
  // 64-bit address that is four registers instead of eight, and registers are
  // what this loop is short of.
  const device uint32_t* wbase =
      w + size_t(col_base + int(coord.y)) * size_t(groups) * words;
  const int w_stride8 = 8 * groups * words;
  const int w_stride16 = kFragM * groups * words;

  const int m0_base = row_base + int(coord.y);
  // Fragment column group: 0..3. Under the step-transposed schedule this is
  // also the index of the 16-code run of the affine group this lane owns.
  const int cx = int(coord.x) >> 2;

  for (int g = 0; g < groups; ++g) {
    // The lane's whole group in one read per operand row.
    //
    // Q4: codes 16c..16c+15 are words 2c and 2c+1 of the group -- a uint2 at
    // index c, and the group base is 32-byte aligned so the vector load is
    // aligned too.
    //
    // Q5: the same sixteen codes are the 80 bits at bit 80c, which three
    // words always cover (bit 80c lies at word 5c/2 rounded down, offset 0 or
    // 16). They are normalized here, once per group, so that bit 0 of `wq` is
    // code 16c and every per-step offset below is a compile-time constant.
    // The shift is spelled in two steps because a shift by the full width is
    // undefined and `sh` is 0 for even c.
    // Only one of these is live per instantiation. Both are declared at full
    // size rather than collapsing the unused one: BITS is a template
    // parameter, so the dead branch is eliminated and the dead array with it
    // -- measured identical -- whereas a size-1 array would leave
    // out-of-range subscripts in source the compiler is entitled to reason
    // about before it drops them.
    uint2 wg[4];
    uint3 wq[4];
    const int w5_bit = BITS == 5 ? 80 * cx : 0;
    const int w5_word = w5_bit >> 5;
    const int w5_sh = w5_bit & 31;
    STEEL_PRAGMA_UNROLL
    for (int q = 0; q < 4; ++q) {
      const device uint32_t* wr = wbase + (q & 1) * w_stride8 +
          (q >> 1) * w_stride16 + size_t(g) * words;
      if (BITS == 4) {
        wg[q] = reinterpret_cast<const device uint2*>(wr)[cx];
      } else {
        const uint32_t a0 = wr[w5_word];
        const uint32_t a1 = wr[w5_word + 1];
        const uint32_t a2 = wr[w5_word + 2];
        wq[q].x = (a0 >> w5_sh) | ((a1 << (31 - w5_sh)) << 1);
        wq[q].y = (a1 >> w5_sh) | ((a2 << (31 - w5_sh)) << 1);
        wq[q].z = a2 >> w5_sh;
      }
    }

    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < kStepsPerGroup; ++t) {
      // Step t is codes 16c + 8*(t>>1) + 2j + (t&1) of the untouched group.
      //
      // Q4: that is the even nibbles of word (t>>1) of the lane's pair, or its
      // odd ones -- one mask, or a shift and a mask, and the result read as
      // char4 is the quad.
      //
      // Q5: the same four codes are 5-bit fields 10 bits apart in the
      // normalized window, at compile-time offsets, so each is one extract
      // and only the four that straddle a word pay more.
      STEEL_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        const int base = (q >> 1) * kElemsPerFrag + (q & 1) * 4;
        if (BITS == 4) {
          const uint32_t word = wg[q][t >> 1];
          const char4 quad = as_type<char4>(
              ((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
          ct_b[base + 0] = quad.x;
          ct_b[base + 1] = quad.y;
          ct_b[base + 2] = quad.z;
          ct_b[base + 3] = quad.w;
        } else {
          // Field j sits at bit 40*(t>>1) + 5*(t&1) + 10j of the window. The
          // step is spelled out per t so that every offset reaches
          // oq_q5_window_code() as a template argument.
          const uint3 wv = wq[q];
          if (t == 0) {
            ct_b[base + 0] = oq_q5_window_code<0>(wv);
            ct_b[base + 1] = oq_q5_window_code<10>(wv);
            ct_b[base + 2] = oq_q5_window_code<20>(wv);
            ct_b[base + 3] = oq_q5_window_code<30>(wv);
          } else if (t == 1) {
            ct_b[base + 0] = oq_q5_window_code<5>(wv);
            ct_b[base + 1] = oq_q5_window_code<15>(wv);
            ct_b[base + 2] = oq_q5_window_code<25>(wv);
            ct_b[base + 3] = oq_q5_window_code<35>(wv);
          } else if (t == 2) {
            ct_b[base + 0] = oq_q5_window_code<40>(wv);
            ct_b[base + 1] = oq_q5_window_code<50>(wv);
            ct_b[base + 2] = oq_q5_window_code<60>(wv);
            ct_b[base + 3] = oq_q5_window_code<70>(wv);
          } else {
            ct_b[base + 0] = oq_q5_window_code<45>(wv);
            ct_b[base + 1] = oq_q5_window_code<55>(wv);
            ct_b[base + 2] = oq_q5_window_code<65>(wv);
            ct_b[base + 3] = oq_q5_window_code<75>(wv);
          }
        }
      }

      STEEL_PRAGMA_UNROLL
      for (int hf = 0; hf < 2; ++hf) {
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          // Deliberately not hoisted into a uint4 per row: the group's 16
          // activation bytes are contiguous under this schedule, but holding
          // four of those across the K step loop costs 16 registers and
          // measures slower than the twelve loads it removes. The weight side
          // is hoisted because a uint2 costs half as much.
          //
          // Rows past M read row M-1 rather than selecting a zero. The
          // accumulator element for such a row is never stored, so the value
          // does not matter, and clamping keeps the load unconditional --
          // worth 0.7 ms of the 11.4 here, because the select sat on four
          // loads in every K step.
          const int m = min(m0_base + hf * kFragM + r * 8, M - 1);
          const char4 quad = as_type<char4>(
              *reinterpret_cast<const device uint32_t*>(
                  qa + size_t(m) * size_t(K) + size_t(g) * kGroupSize +
                  size_t(cx) * 16 + size_t(t) * 4));
          ct_a[r * 4 + 0] = quad.x;
          ct_a[r * 4 + 1] = quad.y;
          ct_a[r * 4 + 2] = quad.z;
          ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) {
          if (hf == 0) {
            op_set.run(ct_a, ct_b, acc0);
          } else {
            op_set.run(ct_a, ct_b, acc1);
          }
        } else {
          if (hf == 0) {
            op.run(ct_a, ct_b, acc0);
          } else {
            op.run(ct_a, ct_b, acc1);
          }
        }
      }
    }

    // Group-major metadata: one row of scales, one of biases, one of Ra.
    const device T* srow = scales + size_t(g) * size_t(N);
    const device T* brow = biases + size_t(g) * size_t(N);
    const device short* rrow = ra + size_t(g) * size_t(M);

    // Held in their stored width. Sixteen floats would be sixteen registers;
    // as vec<T, 4> they are four, and every index into them below is a
    // compile-time constant so the widening is free of address arithmetic.
    vec<T, 4> sv[2];
    vec<T, 4> bv[2];
    STEEL_PRAGMA_UNROLL
    for (int h = 0; h < 2; ++h) {
      const int n0 = n_run0 + h * kFragM;
      sv[h] = *reinterpret_cast<const device vec<T, 4>*>(srow + n0);
      bv[h] = *reinterpret_cast<const device vec<T, 4>*>(brow + n0);
    }

    float r_g[TM][2];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        const int m = min(m_base + i * kFragM + r * 8, M - 1);
        r_g[i][r] = float(rrow[m]);
      }
    }

    // Written as nested fma() on purpose. The kernel is built with
    // -fno-fast-math, so `Cf += sw * a + bw * r` is four instructions -- the
    // compiler may not contract or reassociate it. Spelling the contraction
    // out makes it two, and the correction runs on every destination element
    // of every affine group.
    if (ACT_MODE == 0) {
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kDestElems; ++e) {
        const int r = ((e & 7) >> 2);
        const float swc = float(sv[e >> 3][e & 3]);
        const float bwc = float(bv[e >> 3][e & 3]);
        Cf[0][e] = metal::fma(
            swc, float(acc0[e]), metal::fma(bwc, r_g[0][r], Cf[0][e]));
        Cf[1][e] = metal::fma(
            swc, float(acc1[e]), metal::fma(bwc, r_g[1][r], Cf[1][e]));
      }
    } else {
      const device float* arow = sa + size_t(g) * size_t(M);
      float s_g[TM][2];
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          const int m = min(m_base + i * kFragM + r * 8, M - 1);
          s_g[i][r] = arow[m];
        }
      }
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kDestElems; ++e) {
        const int r = ((e & 7) >> 2);
        const float swc = float(sv[e >> 3][e & 3]);
        const float bwc = float(bv[e >> 3][e & 3]);
        Cf[0][e] = metal::fma(
            s_g[0][r],
            metal::fma(swc, float(acc0[e]), bwc * r_g[0][r]),
            Cf[0][e]);
        Cf[1][e] = metal::fma(
            s_g[1][r],
            metal::fma(swc, float(acc1[e]), bwc * r_g[1][r]),
            Cf[1][e]);
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kDestElems; ++e) {
      const int ee = e & 7;
      const int r = ee >> 2;
      const int m = row_base + i * kFragM + int(coord.y) + r * 8;
      if (m < M) {
        const int n = col_base + (e >> 3) * kFragM + int(coord.x) + (ee & 3);
        const float v = ACT_MODE == 0 ? sa[m] * Cf[i][e] : Cf[i][e];
        out[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(v);
      }
    }
  }
}

#define instantiate_oq_a8_qmm_t_nax_v8(bits, act_mode, type, wm, wn)          \
  instantiate_kernel(                                                         \
      "oq_a8_qmm_t_nax_v8_q" #bits "_am" #act_mode "_" #type "_wm_" #wm       \
      "_wn_" #wn,                                                             \
      oq_a8_qmm_t_nax_v8,                                                     \
      type,                                                                   \
      bits,                                                                   \
      act_mode,                                                               \
      wm,                                                                     \
      wn)

// Tile variants must stay in sync with oq_a8_v2_variant() in qwen35_oq_a8.cpp
// and with _VARIANT_TILES in omlx/patches/qwen35_oq_a8.py, which index the
// same table from the 800 base. The field is flat within ~3% at M=2048, and Q4
// and Q5 are tuned independently because the Q5 decoder reads a wider window
// per row.
#define instantiate_oq_a8_qmm_t_nax_v8_tiles(bits, act_mode, type)            \
  instantiate_oq_a8_qmm_t_nax_v8(bits, act_mode, type, 2, 2);                 \
  instantiate_oq_a8_qmm_t_nax_v8(bits, act_mode, type, 4, 2);                 \
  instantiate_oq_a8_qmm_t_nax_v8(bits, act_mode, type, 2, 4);                 \
  instantiate_oq_a8_qmm_t_nax_v8(bits, act_mode, type, 4, 4);                 \
  instantiate_oq_a8_qmm_t_nax_v8(bits, act_mode, type, 1, 4);                 \
  instantiate_oq_a8_qmm_t_nax_v8(bits, act_mode, type, 8, 2);                 \
  instantiate_oq_a8_qmm_t_nax_v8(bits, act_mode, type, 1, 2)

#define instantiate_oq_a8_qmm_t_nax_v8_bits(bits)                             \
  instantiate_oq_a8_qmm_t_nax_v8_tiles(bits, 0, float16_t);                   \
  instantiate_oq_a8_qmm_t_nax_v8_tiles(bits, 0, bfloat16_t);                  \
  instantiate_oq_a8_qmm_t_nax_v8_tiles(bits, 1, float16_t);                   \
  instantiate_oq_a8_qmm_t_nax_v8_tiles(bits, 1, bfloat16_t)

instantiate_oq_a8_qmm_t_nax_v8_bits(4);
instantiate_oq_a8_qmm_t_nax_v8_bits(5);

///////////////////////////////////////////////////////////////////////////////
// Q4 decode-free path
///////////////////////////////////////////////////////////////////////////////
//
// The tensor units take packed 4-bit weights directly: int8_t x int4b_format
// accumulates into int32_t, measured at 35.6 TOP/s on M5 Pro against 44.2 for
// int8 x int8. So the Q4 path needs no decoder at all -- the checkpoint's
// packed uint32 stream is handed to the hardware as-is.
//
// int4b_format reads a nibble as two's-complement signed, while oQ's affine
// codes are unsigned 0..15. Flipping the top bit of each nibble reconciles
// them exactly:
//
//     (q ^ 8) read as signed 4-bit  ==  q - 8      for every q in 0..15
//
// and the offset folds into the affine bias, since
//
//     W = Sw*q + Bw = Sw*(q - 8) + (8*Sw + Bw)
//
// Both transforms are one-time and happen at load: `weight ^ 0x88888888` and
// `biases + 8*scales`. The hot path is untouched, and MLX's nibble order
// already matches what int4b_format expects, so nothing is repacked.
//
// Unlike the decoder path above, this one lets the matmul primitive own the
// inner loop over a k-tile at threadgroup scope, which is where the measured
// rate comes from. The k-tile is the affine group, so the correction still
// lands on the GS64 boundary.
//
// Q5 has no native 5-bit tensor format and stays on oq_a8_qmm_t_nax.

// TM x TN output per threadgroup, SG simdgroups cooperating on one matmul.
template <typename T, int ACT_MODE, int TM, int TN, int SG>
[[kernel]] void oq_a8_qmm_t_i4_nax(
    const device int8_t* qa [[buffer(0)]],
    const device float* sa [[buffer(1)]],
    const device short* ra [[buffer(2)]],
    const device uchar* w [[buffer(3)]],
    const device T* scales [[buffer(4)]],
    const device T* biases [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  const int groups = K / kGroupSize;
  const int row0 = int(tid.y) * TM;
  const int col0 = int(tid.x) * TN;

  // dextents is (fastest, slowest). Qa is [M, K] and the weight is [N, K],
  // so the weight is the transposed right operand and needs no repacking.
  auto tA = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(
      (device int8_t*)qa, dextents<int32_t, 2>(K, M));
  auto tB = tensor<device int4b_format, dextents<int32_t, 2>, tensor_inline>(
      (device uchar*)w, dextents<int32_t, 2>(K, N));
  // k = kGroupSize: one run() per affine group, so the scale and bias stay
  // valid for the whole integer dot they multiply.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      TM,
      TN,
      kGroupSize,
      /* transpose_left = */ false,
      /* transpose_right = */ true,
      /* relaxed_precision = */ false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<SG>> op;

  auto mA = tA.slice(0, row0);
  auto mB = tB.slice(0, col0);

  // Persistent FP32 accumulator, and the INT32 group accumulator that is
  // reset every 64 K values. Both are cooperative tensors of
  // the same destination shape, so element i of one addresses the same output
  // as element i of the other.
  auto ct_f =
      op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();

  // Per-thread element count for a dense TM x TN tile spread over SG
  // simdgroups. get_capacity() is a runtime call, so the arrays below are
  // sized by the static tile arithmetic and the runtime capacity only bounds
  // the loops.
  constexpr int kMaxCap = (TM * TN) / (SG * 32);

  // The output coordinate of each element is fixed for the whole kernel, but
  // get_multidimensional_index() is a builtin: calling it inside the group
  // loop costs more than the matmul it corrects. Resolve it once.
  int elem_m[kMaxCap];
  int elem_n[kMaxCap];
  const uint16_t cap = min(uint16_t(kMaxCap), ct_f.get_capacity());

#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cap; ++i) {
    if (ct_f.is_valid_element(i)) {
      const auto idx = ct_f.get_multidimensional_index(i);
      const int m = row0 + int(idx[1]);
      // A row past the end is marked dead once, rather than re-tested per
      // group.
      elem_m[i] = m < M ? m : -1;
      elem_n[i] = col0 + int(idx[0]);
      ct_f[i] = 0.0f;
    } else {
      elem_m[i] = -1;
      elem_n[i] = 0;
    }
  }

  for (int g = 0; g < groups; ++g) {
    auto ct_c = op.template get_destination_cooperative_tensor<
        decltype(mA),
        decltype(mB),
        int32_t>();

#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cap; ++i) {
      ct_c[i] = 0;
    }

    auto kA = tA.slice(g * kGroupSize, row0);
    auto kB = tB.slice(g * kGroupSize, col0);
    op.run(kA, kB, ct_c);

    // Affine correction. The cooperative layout is implementation defined, so
    // the output coordinate comes from the tensor itself rather than from an
    // assumed fragment mapping.
#pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cap; ++i) {
      const int m = elem_m[i];
      if (m < 0) {
        continue;
      }
      const int n = elem_n[i];

      const size_t woff = size_t(n) * size_t(groups) + size_t(g);
      const size_t aoff = size_t(m) * size_t(groups) + size_t(g);
      const float sw = float(scales[woff]);
      // biases carries 8*Sw + Bw; the -8 code offset is already folded in.
      const float bw = float(biases[woff]);
      const float r = float(ra[aoff]);
      const float s = ACT_MODE == 0 ? sa[m] : sa[aoff];

      ct_f[i] += s * (sw * float(ct_c[i]) + bw * r);
    }
  }

  // Written element-wise rather than through cooperative_tensor::store(): the
  // accumulator is FP32 while the output is FP16/BF16, and the row bound has
  // to be honoured for the final partial tile of M.
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < cap; ++i) {
    const int m = elem_m[i];
    if (m >= 0) {
      out[size_t(m) * size_t(N) + size_t(elem_n[i])] = static_cast<T>(ct_f[i]);
    }
  }
}

#define instantiate_oq_a8_qmm_t_i4_nax(act_mode, type, tm, tn, sg)            \
  instantiate_kernel(                                                         \
      "oq_a8_qmm_t_i4_nax_am" #act_mode "_" #type "_tm_" #tm "_tn_" #tn       \
      "_sg_" #sg,                                                             \
      oq_a8_qmm_t_i4_nax,                                                     \
      type,                                                                   \
      act_mode,                                                               \
      tm,                                                                     \
      tn,                                                                     \
      sg)

// Tiles taken from the raw int8 x int4 sweep on this device, best first.
#define instantiate_oq_a8_qmm_t_i4_nax_tiles(act_mode, type)                  \
  instantiate_oq_a8_qmm_t_i4_nax(act_mode, type, 64, 64, 4);                  \
  instantiate_oq_a8_qmm_t_i4_nax(act_mode, type, 128, 64, 4);                 \
  instantiate_oq_a8_qmm_t_i4_nax(act_mode, type, 128, 32, 4);                 \
  instantiate_oq_a8_qmm_t_i4_nax(act_mode, type, 64, 32, 4);                  \
  instantiate_oq_a8_qmm_t_i4_nax(act_mode, type, 64, 32, 2);                  \
  instantiate_oq_a8_qmm_t_i4_nax(act_mode, type, 32, 32, 1);                  \
  instantiate_oq_a8_qmm_t_i4_nax(act_mode, type, 64, 32, 1);                  \
  instantiate_oq_a8_qmm_t_i4_nax(act_mode, type, 32, 64, 1)

instantiate_oq_a8_qmm_t_i4_nax_tiles(0, float16_t);
instantiate_oq_a8_qmm_t_i4_nax_tiles(0, bfloat16_t);
instantiate_oq_a8_qmm_t_i4_nax_tiles(1, float16_t);
instantiate_oq_a8_qmm_t_i4_nax_tiles(1, bfloat16_t);

#endif // __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
