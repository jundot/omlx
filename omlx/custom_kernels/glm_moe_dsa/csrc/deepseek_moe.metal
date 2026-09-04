#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"

template <int bits, int wsize = 8>
inline constexpr short deepseek_affine_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short deepseek_affine_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename U, int N, int bits>
inline void deepseek_affine_dequantize(
    const device uint8_t* w,
    U scale,
    U bias,
    threadgroup U* w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 8,
      "DeepSeek affine MoE supports bits in {2, 3, 4, 8}");

  if (bits == 2) {
    U s[4] = {
        scale,
        scale / static_cast<U>(4.0f),
        scale / static_cast<U>(16.0f),
        scale / static_cast<U>(64.0f)};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = s[0] * (w[i] & 0x03) + bias;
      w_local[4 * i + 1] = s[1] * (w[i] & 0x0c) + bias;
      w_local[4 * i + 2] = s[2] * (w[i] & 0x30) + bias;
      w_local[4 * i + 3] = s[3] * (w[i] & 0xc0) + bias;
    }
  } else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      const device uint8_t* wp = w + 3 * i;
      threadgroup U* out = w_local + 8 * i;
      out[0] = (wp[0] & 0x7) * scale + bias;
      out[1] = ((wp[0] & 0x38) >> 3) * scale + bias;
      out[2] = (((wp[0] & 0xc0) >> 6) + ((wp[1] & 0x1) << 2)) * scale + bias;
      out[3] = ((wp[1] & 0xe) >> 1) * scale + bias;
      out[4] = ((wp[1] & 0x70) >> 4) * scale + bias;
      out[5] = (((wp[1] & 0x80) >> 7) + ((wp[2] & 0x3) << 1)) * scale + bias;
      out[6] = ((wp[2] & 0x1c) >> 2) * scale + bias;
      out[7] = ((wp[2] & 0xe0) >> 5) * scale + bias;
    }
  } else if (bits == 4) {
    U s[2] = {scale, scale / static_cast<U>(16.0f)};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = s[0] * (w[i] & 0x0f) + bias;
      w_local[2 * i + 1] = s[1] * (w[i] & 0xf0) + bias;
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = scale * w[i] + bias;
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct DeepseekAffineBlockLoader {
  static_assert(BCOLS <= group_size, "group_size must cover BCOLS");
  static_assert(group_size % BCOLS == 0, "group_size must divide BCOLS");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 8,
      "DeepSeek affine MoE supports bits in {2, 3, 4, 8}");

  static constant constexpr const short pack_factor =
      deepseek_affine_pack_factor<bits, 8>();
  static constant constexpr const short bytes_per_pack =
      deepseek_affine_bytes_per_pack<bits, 8>();
  static constant constexpr const short BCOLS_PACKED = BCOLS / pack_factor;
  static constant constexpr const short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  static constant constexpr const short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  DeepseekAffineBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED * bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS * src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads * thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    const T scale = *scales;
    const T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      deepseek_affine_dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    const T scale = *scales;
    const T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      deepseek_affine_dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void next() {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void deepseek_mxfp4_gather_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device uint8_t* scales [[buffer(2)]],
    const device int32_t* block_meta [[buffer(3)]],
    const device int32_t* block_count [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& max_blocks [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& K [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)M;
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int block_id = int(tid.y);
  const int nblocks = block_count[0];
  if (block_id >= max_blocks || block_id >= nblocks) {
    return;
  }

  const int y_col = int(tid.x) * BN;
  if (y_col >= N) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const short tgp_bm = short(min(BM, rows));
  const short tgp_bn = short(min(BN, N - y_col));
  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;

  const device T* xl = x + size_t(row_start) * K;
  device T* yl = y + size_t(row_start) * N + y_col;
  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device uint8_t* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl, sl, K, Ws, simd_group_id, simd_lane_id);

  if (rows == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(yl, N);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(BN, tgp_bm));
  } else if (rows == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, BM));
  } else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, tgp_bm));
  }
}

#define instantiate_deepseek_mxfp4_blocks(type, bm, bn, bk, wm, wn)            \
  instantiate_kernel(                                                          \
      "deepseek_mxfp4_gather_blocks_rhs_" #type "_bm_" #bm "_bn_" #bn         \
      "_bk_" #bk "_wm_" #wm "_wn_" #wn,                                       \
      deepseek_mxfp4_gather_blocks_rhs,                                        \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn)

instantiate_deepseek_mxfp4_blocks(float16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(float16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(float16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(float16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(float16_t, 32, 64, 32, 1, 2);

instantiate_deepseek_mxfp4_blocks(bfloat16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(bfloat16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(bfloat16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(bfloat16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_blocks(bfloat16_t, 32, 64, 32, 1, 2);

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void deepseek_mxfp4_gather_pair_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w0 [[buffer(1)]],
    const device uint8_t* scales0 [[buffer(2)]],
    const device uint32_t* w1 [[buffer(3)]],
    const device uint8_t* scales1 [[buffer(4)]],
    const device int32_t* block_meta [[buffer(5)]],
    const device int32_t* block_count [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& max_blocks [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    const constant int& K [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int pair_id = int(tid.z);
  const int block_id = int(tid.y);
  const int nblocks = block_count[0];
  if (pair_id >= 2 || block_id >= max_blocks || block_id >= nblocks) {
    return;
  }

  const int y_col = int(tid.x) * BN;
  if (y_col >= N) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const short tgp_bm = short(min(BM, rows));
  const short tgp_bn = short(min(BN, N - y_col));
  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;

  const device uint32_t* w = pair_id == 0 ? w0 : w1;
  const device uint8_t* scales = pair_id == 0 ? scales0 : scales1;

  const device T* xl = x + size_t(row_start) * K;
  device T* yl = y + size_t(pair_id) * size_t(M) * N +
      size_t(row_start) * N + y_col;
  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device uint8_t* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl, sl, K, Ws, simd_group_id, simd_lane_id);

  if (rows == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(yl, N);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(BN, tgp_bm));
  } else if (rows == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, BM));
  } else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, tgp_bm));
  }
}

#define instantiate_deepseek_mxfp4_pair_blocks(type, bm, bn, bk, wm, wn)       \
  instantiate_kernel(                                                          \
      "deepseek_mxfp4_gather_pair_blocks_rhs_" #type "_bm_" #bm "_bn_" #bn    \
      "_bk_" #bk "_wm_" #wm "_wn_" #wn,                                       \
      deepseek_mxfp4_gather_pair_blocks_rhs,                                   \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn)

instantiate_deepseek_mxfp4_pair_blocks(float16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(float16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(float16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(float16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(float16_t, 32, 64, 32, 1, 2);

instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_blocks(bfloat16_t, 32, 64, 32, 1, 2);

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void deepseek_mxfp4_gather_pair_concat_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w0 [[buffer(1)]],
    const device uint8_t* scales0 [[buffer(2)]],
    const device uint32_t* w1 [[buffer(3)]],
    const device uint8_t* scales1 [[buffer(4)]],
    const device int32_t* block_meta [[buffer(5)]],
    const device int32_t* block_count [[buffer(6)]],
    device T* y [[buffer(7)]],
    const constant int& max_blocks [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& N [[buffer(10)]],
    const constant int& K [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)M;
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int block_id = int(tid.y);
  const int nblocks = block_count[0];
  if (block_id >= max_blocks || block_id >= nblocks) {
    return;
  }

  const int y_col_concat = int(tid.x) * BN;
  if (y_col_concat >= 2 * N) {
    return;
  }

  const int pair_id = y_col_concat >= N ? 1 : 0;
  const int y_col = pair_id == 0 ? y_col_concat : y_col_concat - N;
  if (y_col >= N) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const short tgp_bm = short(min(BM, rows));
  const short tgp_bn = short(min(BN, N - y_col));
  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int N_out = 2 * N;

  const device uint32_t* w = pair_id == 0 ? w0 : w1;
  const device uint8_t* scales = pair_id == 0 ? scales0 : scales1;

  const device T* xl = x + size_t(row_start) * K;
  device T* yl = y + size_t(row_start) * N_out +
      size_t(pair_id) * N + y_col;
  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device uint8_t* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl, sl, K, Ws, simd_group_id, simd_lane_id);

  if (rows == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(yl, N_out);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N_out, short2(0, 0), short2(BN, tgp_bm));
  } else if (rows == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N_out, short2(0, 0), short2(tgp_bn, BM));
  } else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N_out, short2(0, 0), short2(tgp_bn, tgp_bm));
  }
}

#define instantiate_deepseek_mxfp4_pair_concat_blocks(type, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                          \
      "deepseek_mxfp4_gather_pair_concat_blocks_rhs_" #type "_bm_" #bm         \
      "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn,                           \
      deepseek_mxfp4_gather_pair_concat_blocks_rhs,                            \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn)

instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(float16_t, 32, 64, 32, 1, 2);

instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_pair_concat_blocks(bfloat16_t, 32, 64, 32, 1, 2);

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int GROUP_SIZE,
    int BITS>
[[kernel]] void deepseek_affine_gather_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device int32_t* block_meta [[buffer(4)]],
    const device int32_t* block_count [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& max_blocks [[buffer(7)]],
    const constant int& M [[buffer(8)]],
    const constant int& N [[buffer(9)]],
    const constant int& K [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)M;
  constexpr int pack_factor = deepseek_affine_pack_factor<BITS, 8>();
  constexpr int bytes_per_pack = deepseek_affine_bytes_per_pack<BITS, 8>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = DeepseekAffineBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      GROUP_SIZE,
      BITS>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int block_id = int(tid.y);
  const int nblocks = block_count[0];
  if (block_id >= max_blocks || block_id >= nblocks) {
    return;
  }

  const int y_col = int(tid.x) * BN;
  if (y_col >= N) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const short tgp_bm = short(min(BM, rows));
  const short tgp_bn = short(min(BN, N - y_col));
  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / GROUP_SIZE;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;

  const device T* xl = x + size_t(row_start) * K;
  device T* yl = y + size_t(row_start) * N + y_col;
  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device T* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;
  const device T* bl =
      biases + size_t(expert) * stride_s + size_t(y_col) * K_g;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl, sl, bl, K, Ws, simd_group_id, simd_lane_id);

  if (rows == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(yl, N);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(BN, tgp_bm));
  } else if (rows == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, BM));
  } else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, tgp_bm));
  }
}

#define instantiate_deepseek_affine_blocks(type, bm, bn, bk, wm, wn, gs, bits) \
  instantiate_kernel(                                                          \
      "deepseek_affine_gather_blocks_rhs_" #type "_gs_" #gs "_b_" #bits       \
      "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn,              \
      deepseek_affine_gather_blocks_rhs,                                       \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn,                                                                      \
      gs,                                                                      \
      bits)

instantiate_deepseek_affine_blocks(float16_t, 16, 32, 32, 1, 2, 64, 2);
instantiate_deepseek_affine_blocks(float16_t, 32, 32, 32, 1, 2, 64, 2);
instantiate_deepseek_affine_blocks(float16_t, 16, 32, 32, 1, 2, 64, 3);
instantiate_deepseek_affine_blocks(float16_t, 32, 32, 32, 1, 2, 64, 3);

instantiate_deepseek_affine_blocks(bfloat16_t, 16, 32, 32, 1, 2, 64, 2);
instantiate_deepseek_affine_blocks(bfloat16_t, 32, 32, 32, 1, 2, 64, 2);
instantiate_deepseek_affine_blocks(bfloat16_t, 16, 32, 32, 1, 2, 64, 3);
instantiate_deepseek_affine_blocks(bfloat16_t, 32, 32, 32, 1, 2, 64, 3);

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int GROUP_SIZE,
    int BITS>
[[kernel]] void deepseek_affine_gather_pair_concat_blocks_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w0 [[buffer(1)]],
    const device T* scales0 [[buffer(2)]],
    const device T* biases0 [[buffer(3)]],
    const device uint32_t* w1 [[buffer(4)]],
    const device T* scales1 [[buffer(5)]],
    const device T* biases1 [[buffer(6)]],
    const device int32_t* block_meta [[buffer(7)]],
    const device int32_t* block_count [[buffer(8)]],
    device T* y [[buffer(9)]],
    const constant int& max_blocks [[buffer(10)]],
    const constant int& M [[buffer(11)]],
    const constant int& N [[buffer(12)]],
    const constant int& K [[buffer(13)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)M;
  constexpr int pack_factor = deepseek_affine_pack_factor<BITS, 8>();
  constexpr int bytes_per_pack = deepseek_affine_bytes_per_pack<BITS, 8>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = DeepseekAffineBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      GROUP_SIZE,
      BITS>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int block_id = int(tid.y);
  const int nblocks = block_count[0];
  if (block_id >= max_blocks || block_id >= nblocks) {
    return;
  }

  const int y_col_concat = int(tid.x) * BN;
  if (y_col_concat >= 2 * N) {
    return;
  }

  const int pair_id = y_col_concat >= N ? 1 : 0;
  const int y_col = pair_id == 0 ? y_col_concat : y_col_concat - N;
  if (y_col >= N) {
    return;
  }

  const int row_start = block_meta[block_id * 3 + 0];
  const int expert = block_meta[block_id * 3 + 1];
  const int rows = block_meta[block_id * 3 + 2];
  if (rows <= 0) {
    return;
  }

  const short tgp_bm = short(min(BM, rows));
  const short tgp_bn = short(min(BN, N - y_col));
  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const short2 tile_x = short2(k_remain, tgp_bm);
  const short2 tile_w = short2(k_remain, tgp_bn);

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / GROUP_SIZE;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int N_out = 2 * N;

  const device uint32_t* w = pair_id == 0 ? w0 : w1;
  const device T* scales = pair_id == 0 ? scales0 : scales1;
  const device T* biases = pair_id == 0 ? biases0 : biases1;

  const device T* xl = x + size_t(row_start) * K;
  device T* yl = y + size_t(row_start) * N_out +
      size_t(pair_id) * N + y_col;
  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device T* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;
  const device T* bl =
      biases + size_t(expert) * stride_s + size_t(y_col) * K_g;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
  thread loader_w_t loader_w(wl, sl, bl, K, Ws, simd_group_id, simd_lane_id);

  if (rows == BM && tgp_bn == BN) {
    gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result(yl, N_out);
  } else if (tgp_bn == BN) {
    gemm_loop_unaligned<false, true, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N_out, short2(0, 0), short2(BN, tgp_bm));
  } else if (rows == BM) {
    gemm_loop_unaligned<true, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N_out, short2(0, 0), short2(tgp_bn, BM));
  } else {
    gemm_loop_unaligned<false, false, true>(
        Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
    if (k_remain != 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
    }
    mma_op.store_result_slice(yl, N_out, short2(0, 0), short2(tgp_bn, tgp_bm));
  }
}

#define instantiate_deepseek_affine_pair_concat_blocks(                        \
    type, bm, bn, bk, wm, wn, gs, bits)                                        \
  instantiate_kernel(                                                          \
      "deepseek_affine_gather_pair_concat_blocks_rhs_" #type "_gs_" #gs       \
      "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm              \
      "_wn_" #wn,                                                             \
      deepseek_affine_gather_pair_concat_blocks_rhs,                           \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn,                                                                      \
      gs,                                                                      \
      bits)

instantiate_deepseek_affine_pair_concat_blocks(float16_t, 16, 32, 32, 1, 2, 64, 2);
instantiate_deepseek_affine_pair_concat_blocks(float16_t, 32, 32, 32, 1, 2, 64, 2);
instantiate_deepseek_affine_pair_concat_blocks(float16_t, 16, 32, 32, 1, 2, 64, 3);
instantiate_deepseek_affine_pair_concat_blocks(float16_t, 32, 32, 32, 1, 2, 64, 3);

instantiate_deepseek_affine_pair_concat_blocks(bfloat16_t, 16, 32, 32, 1, 2, 64, 2);
instantiate_deepseek_affine_pair_concat_blocks(bfloat16_t, 32, 32, 32, 1, 2, 64, 2);
instantiate_deepseek_affine_pair_concat_blocks(bfloat16_t, 16, 32, 32, 1, 2, 64, 3);
instantiate_deepseek_affine_pair_concat_blocks(bfloat16_t, 32, 32, 32, 1, 2, 64, 3);

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN>
[[kernel]] void deepseek_mxfp4_gather_expert_rhs(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device uint8_t* scales [[buffer(2)]],
    const device uint32_t* indices [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& M [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& E [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 32;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using mma_t = mlx::steel::BlockMMA<
      T,
      T,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      true,
      BK_padded,
      BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      true,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  const int y_col = int(tid.x) * BN;
  const int expert = int(tid.y);
  if (y_col >= N || expert >= E) {
    return;
  }

  int lo = 0;
  int hi = M;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (indices[mid] < uint32_t(expert)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int start = lo;

  hi = M;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (indices[mid] <= uint32_t(expert)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int end = lo;
  if (start >= end) {
    return;
  }

  const int K_it = K / BK;
  const int k_remain = K - K_it * BK;
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const short tgp_bn = short(min(BN, N - y_col));
  const short2 tile_w = short2(k_remain, tgp_bn);

  const device uint8_t* wl =
      ((const device uint8_t*)w) + size_t(expert) * stride_w +
      size_t(y_col) * K_w;
  const device uint8_t* sl =
      scales + size_t(expert) * stride_s + size_t(y_col) * K_g;

  for (int row = start; row < end; row += BM) {
    const int rows = min(BM, end - row);
    const short tgp_bm = short(rows);
    const short2 tile_x = short2(k_remain, tgp_bm);

    const device T* xl = x + size_t(row) * K;
    device T* yl = y + size_t(row) * N + y_col;

    thread mma_t mma_op(simd_group_id, simd_lane_id);
    thread loader_x_t loader_x(xl, K, Xs, simd_group_id, simd_lane_id);
    thread loader_w_t loader_w(wl, sl, K, Ws, simd_group_id, simd_lane_id);

    if (rows == BM && tgp_bn == BN) {
      gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
      if (k_remain != 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result(yl, N);
    } else if (tgp_bn == BN) {
      gemm_loop_unaligned<false, true, true>(
          Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      if (k_remain != 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result_slice(yl, N, short2(0, 0), short2(BN, tgp_bm));
    } else if (rows == BM) {
      gemm_loop_unaligned<true, false, true>(
          Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      if (k_remain != 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, BM));
    } else {
      gemm_loop_unaligned<false, false, true>(
          Xs, Ws, mma_op, loader_x, loader_w, K_it, tgp_bm, tgp_bn, BK);
      if (k_remain != 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemm_loop_finalize(Xs, Ws, mma_op, loader_x, loader_w, tile_x, tile_w);
      }
      mma_op.store_result_slice(yl, N, short2(0, 0), short2(tgp_bn, tgp_bm));
    }
  }
}

#define instantiate_deepseek_mxfp4_expert(type, bm, bn, bk, wm, wn)            \
  instantiate_kernel(                                                          \
      "deepseek_mxfp4_gather_expert_rhs_" #type "_bm_" #bm "_bn_" #bn         \
      "_bk_" #bk "_wm_" #wm "_wn_" #wn,                                       \
      deepseek_mxfp4_gather_expert_rhs,                                        \
      type,                                                                    \
      bm,                                                                      \
      bn,                                                                      \
      bk,                                                                      \
      wm,                                                                      \
      wn)

instantiate_deepseek_mxfp4_expert(float16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(float16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(float16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(float16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(float16_t, 32, 64, 32, 1, 2);

instantiate_deepseek_mxfp4_expert(bfloat16_t, 8, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(bfloat16_t, 16, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(bfloat16_t, 32, 32, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(bfloat16_t, 16, 64, 32, 1, 2);
instantiate_deepseek_mxfp4_expert(bfloat16_t, 32, 64, 32, 1, 2);
