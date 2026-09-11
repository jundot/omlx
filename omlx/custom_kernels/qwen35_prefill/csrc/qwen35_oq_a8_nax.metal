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
constant constexpr int kDestElems = 2 * kElemsPerFrag;
constant constexpr int kStepsPerGroup = kGroupSize / kFragK; // 4

using frag_i8 = BaseNAXFrag::dtype_frag_t<int8_t>;

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

// Tile variants must stay in sync with oq_a8_nax_variant() in qwen35_oq_a8.cpp
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

#endif // __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
