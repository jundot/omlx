// SPDX-License-Identifier: Apache-2.0
// Exact DS4 M=1024 O-A BF16-boundary layout finalizer for the O-A -> O-B
// chain.  The reduction body below is derived from MLX fp_qmm_t_impl; see
// MLX_LICENSE.txt.  Only the output row stride/group offset are generalized.

// clang-format off: MLX quantized headers require the Steel declarations first.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"
// clang-format on

using namespace metal;

template <typename T, const int group_size, const int bits,
          const bool aligned_N, const int BM, const int BK, const int BN>
METAL_FUNC void ds4_output_fp_qmm_t_impl(
    const device uint32_t *w, const device uint8_t *scales, const device T *x,
    device T *y, threadgroup T *Xs, threadgroup T *Ws, const constant int &K,
    const constant int &N, const constant int &M, const constant int &K_eff,
    const int y_stride, const int y_group_offset,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = BK + 16 / sizeof(T);

  using mma_t = mlx::steel::BlockMMA<T, T, BM, BN, BK, WM, WN, false, true,
                                     BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t =
      QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE,
                           group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t *)w;
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(y_stride) + y_group_offset + y_col;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  if (num_els < BM) {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_safe(short2(BK, num_els));
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  } else {
    if (!aligned_N && num_outs < BN) {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_safe(short2(BK, num_outs));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    } else {
      for (int k = 0; k < K_eff; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_x.load_unsafe();
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mma_op.mma(Xs, Ws);
        loader_x.next();
        loader_w.next();
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(y, y_stride, short2(num_outs, num_els));
  } else {
    mma_op.store_result(y, y_stride);
  }
}

template <typename T, int BM, int BK, int BN>
[[kernel]] void ds4_output_oa_interleaved(
    const device uint32_t *weight [[buffer(0)]],
    const device uint8_t *scales [[buffer(1)]],
    const device T *input [[buffer(2)]], device T *output [[buffer(3)]],
    const constant int &K [[buffer(4)]], const constant int &N [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    const constant int &groups [[buffer(7)]],
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
  const int y_stride = groups * N;
  const int y_group_offset = batch * N;
  const uint3 local_tid = uint3(tid.x, tid.y, 0);
  ds4_output_fp_qmm_t_impl<T, 32, 8, true, BM, BK, BN>(
      weight, scales, input, output, Xs, Ws, K, N, M, K, y_stride,
      y_group_offset, local_tid, lid, simd_gid, simd_lid);
}

#define instantiate_ds4_output_oa(type, bm, bk, bn)                            \
  instantiate_kernel("ds4_output_oa_interleaved_" #type "_bm" #bm "_bk" #bk    \
                     "_bn" #bn,                                                \
                     ds4_output_oa_interleaved, type, bm, bk, bn)

instantiate_ds4_output_oa(bfloat16_t, 32, 32, 32);
instantiate_ds4_output_oa(bfloat16_t, 64, 32, 32);
