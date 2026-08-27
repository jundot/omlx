// SPDX-License-Identifier: Apache-2.0
// Isolated DS4 M=1024 MXFP8 projection-tile sweep; no model dispatch.

// clang-format off: MLX quantized headers require the Steel declarations first.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"
// clang-format on

using namespace metal;

// MLX's stock classic MXFP8 qmm_t is BM32/BK32/BN32 with a 2x2 warp tile.
// This wrapper calls the same fp_qmm_t_impl, preserving its dequantization,
// MMA K walk, FP32 accumulation, and BF16 store while exposing only the tile
// dimensions as an isolated tuning surface for DS4's large Q-B/O-B banks.
template <typename T, int BM, int BK, int BN>
[[kernel]] void ds4_projection_mxfp8_qmm_t(
    const device uint32_t *weight [[buffer(0)]],
    const device uint8_t *scales [[buffer(1)]],
    const device T *input [[buffer(2)]], device T *output [[buffer(3)]],
    const constant int &K [[buffer(4)]], const constant int &N [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16 / sizeof(T);
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  const size_t batch = tid.z;
  weight += batch * size_t(N) * size_t(K / 4);
  scales += batch * size_t(N) * size_t(K / 32);
  input += batch * size_t(M) * size_t(K);
  output += batch * size_t(M) * size_t(N);
  const uint3 local_tid = uint3(tid.x, tid.y, 0);
  fp_qmm_t_impl<T, 32, 8, true, BM, BK, BN>(weight, scales, input, output, Xs,
                                            Ws, K, N, M, K, local_tid, lid,
                                            simd_gid, simd_lid);
}

#define instantiate_ds4_projection_mxfp8(type, bm, bk, bn)                     \
  instantiate_kernel("ds4_projection_mxfp8_qmm_t_" #type "_bm" #bm "_bk" #bk   \
                     "_bn" #bn,                                                \
                     ds4_projection_mxfp8_qmm_t, type, bm, bk, bn)

#define instantiate_ds4_projection_mxfp8_variants(type)                        \
  instantiate_ds4_projection_mxfp8(type, 32, 32, 32);                          \
  instantiate_ds4_projection_mxfp8(type, 64, 32, 32);                          \
  instantiate_ds4_projection_mxfp8(type, 128, 32, 32);                         \
  instantiate_ds4_projection_mxfp8(type, 32, 32, 64);                          \
  instantiate_ds4_projection_mxfp8(type, 64, 32, 64);                          \
  instantiate_ds4_projection_mxfp8(type, 128, 32, 64);                         \
  instantiate_ds4_projection_mxfp8(type, 32, 64, 64);                          \
  instantiate_ds4_projection_mxfp8(type, 64, 64, 64);                          \
  instantiate_ds4_projection_mxfp8(type, 128, 64, 64);                         \
  instantiate_ds4_projection_mxfp8(type, 64, 32, 128)

instantiate_ds4_projection_mxfp8_variants(float16_t);
instantiate_ds4_projection_mxfp8_variants(bfloat16_t);
