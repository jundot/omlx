// SPDX-License-Identifier: Apache-2.0
// Optional M5 TensorOps tile sweep for isolated DS4 M=1024 MXFP8 projections.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

// clang-format off: MLX quantized headers require the Steel declarations first.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/fp_quantized_nax.h"
// clang-format on

using namespace metal;

template <typename T, int BM, int BK, int BN, int WM, int WN>
[[kernel]] void ds4_projection_mxfp8_qmm_t_nax(
    const device uint32_t *weight [[buffer(0)]],
    const device uint8_t *scales [[buffer(1)]],
    const device T *input [[buffer(2)]], device T *output [[buffer(3)]],
    const constant int &K [[buffer(4)]], const constant int &N [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(bfloat);
  threadgroup bfloat Ws[BN * BK_padded];
  const size_t batch = tid.z;
  weight += batch * size_t(N) * size_t(K / 4);
  scales += batch * size_t(N) * size_t(K / 32);
  input += batch * size_t(M) * size_t(K);
  output += batch * size_t(M) * size_t(N);
  const uint3 local_tid = uint3(tid.x, tid.y, 0);
  fp_qmm_t_impl<T, 32, 8, true, BM, BK, BN, WM, WN, bfloat>(
      weight, scales, input, output, Ws, K, N, M, local_tid, lid, simd_gid,
      simd_lid);
}

#define instantiate_ds4_projection_mxfp8_nax(type, bm, bk, bn, wm, wn)         \
  instantiate_kernel("ds4_projection_mxfp8_qmm_t_nax_" #type "_bm" #bm         \
                     "_bk" #bk "_bn" #bn "_wm" #wm "_wn" #wn,                  \
                     ds4_projection_mxfp8_qmm_t_nax, type, bm, bk, bn, wm, wn)

// Variant 0 is MLX stock. Every other variant keeps a 32x32 output tile per
// SIMDgroup while sweeping the threadgroup envelope independently on M and N.
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 64, 64, 64, 2, 2);
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 32, 64, 32, 1, 1);
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 32, 64, 64, 1, 2);
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 64, 64, 32, 2, 1);
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 64, 64, 128, 2, 4);
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 128, 64, 64, 4, 2);
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 128, 64, 128, 4, 4);
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 32, 64, 128, 1, 4);
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 128, 64, 32, 4, 1);
instantiate_ds4_projection_mxfp8_nax(bfloat16_t, 64, 32, 64, 2, 2);

#endif
