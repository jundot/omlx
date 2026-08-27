#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "kernels/quantized_moe.h"

#define define_qwen35_q_affine_qmm_t(bits)                                    \
  template <typename T, const int BM, const int BK, const int BN>              \
  [[kernel]] void qwen35_q##bits##_affine_qmm_t(                              \
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
    threadgroup T Xs[BM * BK_padded];                                          \
    threadgroup T Ws[BN * BK_padded];                                          \
                                                                               \
    qmm_t_impl<T, 64, bits, true, BM, BK, BN>(                                 \
        w,                                                                     \
        scales,                                                                \
        biases,                                                                \
        x,                                                                     \
        y,                                                                     \
        Xs,                                                                    \
        Ws,                                                                    \
        K,                                                                     \
        N,                                                                     \
        M,                                                                     \
        K,                                                                     \
        tid,                                                                   \
        lid,                                                                   \
        simd_gid,                                                              \
        simd_lid);                                                             \
  }

#define instantiate_qwen35_q_affine_qmm_t(bits, type, bm, bk, bn)             \
  instantiate_kernel(                                                         \
      "qwen35_q" #bits "_affine_qmm_t_" #type "_bm_" #bm "_bk_" #bk         \
      "_bn_" #bn,                                                            \
      qwen35_q##bits##_affine_qmm_t,                                          \
      type,                                                                   \
      bm,                                                                     \
      bk,                                                                     \
      bn)

#define instantiate_qwen35_q_affine_variants(bits)                            \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 32, 32, 32);             \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 32, 32, 32);            \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 32, 64, 32);             \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 32, 64, 32);            \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 32, 64, 64);             \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 32, 64, 64);            \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 64, 64, 64);             \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 64, 64, 64);            \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 16, 64, 64);             \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 16, 64, 64);            \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 64, 64, 128);            \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 64, 64, 128);           \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 128, 64, 64);            \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 128, 64, 64);           \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 128, 64, 32);            \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 128, 64, 32);           \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 64, 32, 64);             \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 64, 32, 64);            \
  instantiate_qwen35_q_affine_qmm_t(bits, float16_t, 128, 32, 64);            \
  instantiate_qwen35_q_affine_qmm_t(bits, bfloat16_t, 128, 32, 64)

// group_size=128 variants (Bonsai 27B and other large models)
#define define_qwen35_q_affine_qmm128_t(bits)                                 \
  template <typename T, const int BM, const int BK, const int BN>             \
  [[kernel]] void qwen35_q##bits##_affine_qmm128_t(                          \
      const device uint32_t* w [[buffer(0)]],                                 \
      const device T* scales [[buffer(1)]],                                   \
      const device T* biases [[buffer(2)]],                                   \
      const device T* x [[buffer(3)]],                                        \
      device T* y [[buffer(4)]],                                              \
      const constant int& K [[buffer(5)]],                                    \
      const constant int& N [[buffer(6)]],                                    \
      const constant int& M [[buffer(7)]],                                    \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint lid [[thread_index_in_threadgroup]],                               \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                       \
      uint simd_lid [[thread_index_in_simdgroup]]) {                          \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                          \
                                                                              \
    threadgroup T Xs[BM * BK_padded];                                         \
    threadgroup T Ws[BN * BK_padded];                                         \
                                                                              \
    qmm_t_impl<T, 128, bits, true, BM, BK, BN>(                              \
        w,                                                                    \
        scales,                                                                \
        biases,                                                                \
        x,                                                                    \
        y,                                                                    \
        Xs,                                                                   \
        Ws,                                                                   \
        K,                                                                    \
        N,                                                                    \
        M,                                                                    \
        K,                                                                    \
        tid,                                                                  \
        lid,                                                                  \
        simd_gid,                                                             \
        simd_lid);                                                            \
  }

#define instantiate_qwen35_q_affine_qmm128_t(bits, type, bm, bk, bn)         \
  instantiate_kernel(                                                         \
      "qwen35_q" #bits "_affine_qmm128_t_" #type "_bm_" #bm "_bk_" #bk     \
      "_bn_" #bn,                                                             \
      qwen35_q##bits##_affine_qmm128_t,                                       \
      type,                                                                   \
      bm,                                                                     \
      bk,                                                                     \
      bn)

#define instantiate_qwen35_q_affine_variants128(bits)                         \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 32, 32, 32);         \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 32, 32, 32);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 32, 64, 32);         \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 32, 64, 32);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 32, 64, 64);         \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 32, 64, 64);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 64, 64, 64);         \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 64, 64, 64);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 16, 64, 64);         \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 16, 64, 64);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 64, 64, 128);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 64, 64, 128);       \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 128, 64, 64);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 128, 64, 64);       \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 128, 64, 32);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 128, 64, 32);       \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 64, 32, 64);         \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 64, 32, 64);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, float16_t, 128, 32, 64);        \
  instantiate_qwen35_q_affine_qmm128_t(bits, bfloat16_t, 128, 32, 64)

#define instantiate_qwen35_moe_weighted_sum_tiled(type, score_type, topk,      \
                                                  threads)                    \
  instantiate_kernel(                                                          \
      "moe_weighted_sum_tiled_" #type "_score_" #score_type "_topk_" #topk     \
      "_t_" #threads,                                                          \
      moe_weighted_sum_tiled,                                                  \
      type,                                                                    \
      score_type,                                                              \
      topk,                                                                    \
      threads)

define_qwen35_q_affine_qmm_t(2)
define_qwen35_q_affine_qmm_t(4)
define_qwen35_q_affine_qmm_t(5)
define_qwen35_q_affine_qmm_t(6)
define_qwen35_q_affine_qmm_t(8)

instantiate_qwen35_q_affine_variants(2);
instantiate_qwen35_q_affine_variants(4);
instantiate_qwen35_q_affine_variants(5);
instantiate_qwen35_q_affine_variants(6);
instantiate_qwen35_q_affine_variants(8);

define_qwen35_q_affine_qmm128_t(2)
define_qwen35_q_affine_qmm128_t(4)
define_qwen35_q_affine_qmm128_t(5)
define_qwen35_q_affine_qmm128_t(6)
define_qwen35_q_affine_qmm128_t(8)

instantiate_qwen35_q_affine_variants128(2);
instantiate_qwen35_q_affine_variants128(4);
instantiate_qwen35_q_affine_variants128(5);
instantiate_qwen35_q_affine_variants128(6);
instantiate_qwen35_q_affine_variants128(8);

// Grouped affine suffix used by DeepSeek-V4 wo_a.  Inputs and outputs are
// laid out [groups, M, K/N], while weights are the concatenated per-group
// row blocks.  Keeping the group dimension explicit avoids a block-diagonal
// weight (and the corresponding 8x memory/compute expansion for 8 groups).
template <typename T>
[[kernel]] void qwen35_q8_affine_grouped_qmm_t(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& groups [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BM = 64;
  constexpr int BK = 32;
  constexpr int BN = 64;
  constexpr int BK_padded = BK + 16 / sizeof(T);
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const uint group = tid.z;
  if (group >= static_cast<uint>(groups)) {
    return;
  }
  const device uint32_t* group_w = w + group * N * (K / 4);
  const device T* group_scales = scales + group * N * (K / 64);
  const device T* group_biases = biases + group * N * (K / 64);
  const device T* group_x = x + group * M * K;
  device T* group_y = y + group * M * N;
  const uint3 local_tid(tid.x, tid.y, 0);
  qmm_t_impl<T, 64, 8, true, BM, BK, BN>(
      group_w,
      group_scales,
      group_biases,
      group_x,
      group_y,
      Xs,
      Ws,
      K,
      N,
      M,
      K,
      local_tid,
      lid,
      simd_gid,
      simd_lid);
}

instantiate_kernel(
    "qwen35_q8_affine_grouped_qmm_t_float16_t_bm_64_bk_32_bn_64",
    qwen35_q8_affine_grouped_qmm_t,
    float16_t);
instantiate_kernel(
    "qwen35_q8_affine_grouped_qmm_t_bfloat16_t_bm_64_bk_32_bn_64",
    qwen35_q8_affine_grouped_qmm_t,
    bfloat16_t);

instantiate_qwen35_moe_weighted_sum_tiled(float16_t, float, 8, 256);
instantiate_qwen35_moe_weighted_sum_tiled(bfloat16_t, float, 8, 256);
instantiate_qwen35_moe_weighted_sum_tiled(float16_t, float, 6, 256);
instantiate_qwen35_moe_weighted_sum_tiled(bfloat16_t, float, 6, 256);
