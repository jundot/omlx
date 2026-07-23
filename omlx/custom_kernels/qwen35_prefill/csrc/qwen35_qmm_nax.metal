// NAX (neural accelerator) variants of the Qwen3.5/3.6 affine qmm_t kernels.
//
// Same buffer ABI as qwen35_qmm.metal, but the inner loop runs on the M5
// tensor units through MLX's qmm_t_nax_tgp_impl (quantized_nax.h). This file
// is compiled into a separate metallib (omlx_qwen35_prefill_kernels_nax) with
// -mmacosx-version-min=26.2 so the classic metallib keeps its deployment
// target; the C++ op only loads it when the runtime reports NAX support.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"
// clang-format on

#define define_qwen35_q_affine_qmm_t_nax(bits)                                \
  template <                                                                   \
      typename T,                                                              \
      const int BM,                                                            \
      const int BK,                                                            \
      const int BN,                                                            \
      const int WM,                                                            \
      const int WN>                                                            \
  [[kernel]] void qwen35_q##bits##_affine_qmm_t_nax(                          \
      const device uint32_t* w [[buffer(0)]],                                  \
      const device T* scales [[buffer(1)]],                                    \
      const device T* biases [[buffer(2)]],                                    \
      const device T* x [[buffer(3)]],                                         \
      device T* y [[buffer(4)]],                                               \
      const constant int& K [[buffer(5)]],                                     \
      const constant int& N [[buffer(6)]],                                     \
      const constant int& M [[buffer(7)]],                                     \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint lid [[thread_index_in_threadgroup]],                                \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]) {                           \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                           \
                                                                               \
    threadgroup T Ws[BN * BK_padded];                                          \
                                                                               \
    qmm_t_nax_tgp_impl<T, 64, bits, true, BM, BK, BN, WM, WN>(                 \
        w,                                                                     \
        scales,                                                                \
        biases,                                                                \
        x,                                                                     \
        y,                                                                     \
        Ws,                                                                    \
        K,                                                                     \
        N,                                                                     \
        M,                                                                     \
        tid,                                                                   \
        lid,                                                                   \
        simd_gid,                                                              \
        simd_lid);                                                             \
  }

#define instantiate_qwen35_q_affine_qmm_t_nax(bits, type, bm, bk, bn, wm, wn) \
  instantiate_kernel(                                                         \
      "qwen35_q" #bits "_affine_qmm_t_nax_" #type "_bm_" #bm "_bk_" #bk      \
      "_bn_" #bn "_wm_" #wm "_wn_" #wn,                                      \
      qwen35_q##bits##_affine_qmm_t_nax,                                      \
      type,                                                                   \
      bm,                                                                     \
      bk,                                                                     \
      bn,                                                                     \
      wm,                                                                     \
      wn)

// Tile variants mirror qwen_q_affine_nax_variant() in qwen35_prefill.cpp.
// Variant 0 matches the tile MLX ships for affine_qmm_t_nax (64/64/64, 2x2);
// the rest are the tuning surface for the M5 sweep. BK is capped at the
// group size (64): QuantizedBlockLoader requires group_size >= columns.
#define instantiate_qwen35_q_affine_nax_variants(bits)                        \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, float16_t, 64, 64, 64, 2, 2);   \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, bfloat16_t, 64, 64, 64, 2, 2);  \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, float16_t, 32, 64, 64, 2, 2);   \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, bfloat16_t, 32, 64, 64, 2, 2);  \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, float16_t, 128, 64, 64, 2, 2);  \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, bfloat16_t, 128, 64, 64, 2, 2); \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, float16_t, 64, 64, 128, 2, 2);  \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, bfloat16_t, 64, 64, 128, 2, 2); \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, float16_t, 64, 32, 64, 2, 2);   \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, bfloat16_t, 64, 32, 64, 2, 2);  \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, float16_t, 64, 64, 64, 4, 1);   \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, bfloat16_t, 64, 64, 64, 4, 1);  \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, float16_t, 64, 64, 64, 1, 4);   \
  instantiate_qwen35_q_affine_qmm_t_nax(bits, bfloat16_t, 64, 64, 64, 1, 4)

define_qwen35_q_affine_qmm_t_nax(4)
define_qwen35_q_affine_qmm_t_nax(5)
define_qwen35_q_affine_qmm_t_nax(6)
define_qwen35_q_affine_qmm_t_nax(8)

instantiate_qwen35_q_affine_nax_variants(4);
instantiate_qwen35_q_affine_nax_variants(5);
instantiate_qwen35_q_affine_nax_variants(6);
instantiate_qwen35_q_affine_nax_variants(8);

#endif // __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
