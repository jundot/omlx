#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "affine_quantized.h"

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

// The ANE hybrid GPU suffix reads its first split_n output rows from one
// matrix and the rest from a second, so gate and up rows stay in place.
#define define_qwen35_q_affine_qmm_t_split(bits, group_size, suffix)          \
  template <typename T, const int BM, const int BK, const int BN>              \
  [[kernel]] void qwen35_q##bits##_affine_qmm##suffix##_t_split(              \
      const device uint32_t* w [[buffer(0)]],                                  \
      const device T* scales [[buffer(1)]],                                    \
      const device T* biases [[buffer(2)]],                                    \
      const device T* x [[buffer(3)]],                                         \
      device T* y [[buffer(4)]],                                               \
      const constant int& K [[buffer(5)]],                                     \
      const constant int& N [[buffer(6)]],                                     \
      const constant int& M [[buffer(7)]],                                     \
      const device uint32_t* w1 [[buffer(8)]],                                 \
      const device T* scales1 [[buffer(9)]],                                   \
      const device T* biases1 [[buffer(10)]],                                  \
      const constant int& split_n [[buffer(11)]],                              \
      uint3 tid [[threadgroup_position_in_grid]],                              \
      uint lid [[thread_index_in_threadgroup]],                                \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]]) {                           \
    constexpr int BK_padded = (BK + 16 / sizeof(T));                           \
                                                                               \
    threadgroup T Xs[BM * BK_padded];                                          \
    threadgroup T Ws[BN * BK_padded];                                          \
                                                                               \
    const bool second = static_cast<int>(tid.x) * BN >= split_n;               \
    qmm_t_impl<T, group_size, bits, true, BM, BK, BN>(                         \
        second ? w1 : w,                                                       \
        second ? scales1 : scales,                                             \
        second ? biases1 : biases,                                             \
        x,                                                                     \
        second ? y + split_n : y,                                              \
        Xs,                                                                    \
        Ws,                                                                    \
        K,                                                                     \
        N,                                                                     \
        M,                                                                     \
        K,                                                                     \
        second ? uint3(tid.x - split_n / BN, tid.y, tid.z) : tid,              \
        lid,                                                                   \
        simd_gid,                                                              \
        simd_lid);                                                             \
  }

#define instantiate_qwen35_q_affine_qmm_t_split(bits, suffix, type)          \
  instantiate_kernel(                                                         \
      "qwen35_q" #bits "_affine_qmm" #suffix "_t_split_" #type               \
      "_bm_64_bk_32_bn_64",                                                  \
      qwen35_q##bits##_affine_qmm##suffix##_t_split,                          \
      type,                                                                   \
      64,                                                                     \
      32,                                                                     \
      64)

#define instantiate_qwen35_q_affine_split_variants(bits)                      \
  instantiate_qwen35_q_affine_qmm_t_split(bits, , float16_t);                 \
  instantiate_qwen35_q_affine_qmm_t_split(bits, , bfloat16_t);                \
  instantiate_qwen35_q_affine_qmm_t_split(bits, 128, float16_t);              \
  instantiate_qwen35_q_affine_qmm_t_split(bits, 128, bfloat16_t)

define_qwen35_q_affine_qmm_t_split(4, 64, )
define_qwen35_q_affine_qmm_t_split(5, 64, )
define_qwen35_q_affine_qmm_t_split(6, 64, )
define_qwen35_q_affine_qmm_t_split(8, 64, )
define_qwen35_q_affine_qmm_t_split(4, 128, 128)
define_qwen35_q_affine_qmm_t_split(5, 128, 128)
define_qwen35_q_affine_qmm_t_split(6, 128, 128)
define_qwen35_q_affine_qmm_t_split(8, 128, 128)

instantiate_qwen35_q_affine_split_variants(4);
instantiate_qwen35_q_affine_split_variants(5);
instantiate_qwen35_q_affine_split_variants(6);
instantiate_qwen35_q_affine_split_variants(8);
