#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h"

using namespace mlx::steel;

#define instantiate_dspark_gemm(                                         \
    tname, dtype, bm, bn, bk, wm, wn, trans_b)                           \
  instantiate_kernel(                                                    \
      "omlx_dspark_gemm_" #tname "_nt" #trans_b,                       \
      gemm,                                                              \
      dtype,                                                             \
      bm,                                                                \
      bn,                                                                \
      bk,                                                                \
      wm,                                                                \
      wn,                                                                \
      false,                                                             \
      trans_b)

instantiate_dspark_gemm(float16, float16_t, 64, 64, 16, 1, 2, false);
instantiate_dspark_gemm(float16, float16_t, 64, 32, 32, 2, 2, true);
instantiate_dspark_gemm(bfloat16, bfloat16_t, 64, 64, 16, 1, 2, false);
instantiate_dspark_gemm(bfloat16, bfloat16_t, 64, 32, 32, 2, 2, true);
instantiate_dspark_gemm(float32, float, 32, 64, 16, 1, 2, true);

// Load a logical KV tile through a physical-ring row map while retaining the
// exact thread-to-element layout of Steel's BlockLoader. The MMA tile and its
// accumulation order therefore stay identical to an ordinary M=1 GEMM.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short tgp_size,
    bool transpose_b,
    short n_reads = (BCOLS * BROWS) / tgp_size,
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct DSparkRingBlockLoader {
  STEEL_CONST short vec_size = n_reads;

  const device T* source;
  const device uint* indices;
  const int source_ld;
  int logical_row_base;
  int source_col_base;

  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;

  struct alignas(sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  METAL_FUNC DSparkRingBlockLoader(
      const device T* source_,
      const device uint* indices_,
      const int source_ld_,
      const int logical_row_base_,
      const int source_col_base_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]])
      : source(source_),
        indices(indices_),
        source_ld(source_ld_),
        logical_row_base(logical_row_base_),
        source_col_base(source_col_base_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj) {}

  METAL_FUNC void load_unsafe() const {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      const uint physical_row = indices[logical_row_base + bi + i];
      const device T* src = source + size_t(physical_row) * source_ld +
          source_col_base + bj;
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)src);
    }
  }

  METAL_FUNC void next() {
    if (transpose_b) {
      source_col_base += BCOLS;
    } else {
      logical_row_base += BROWS;
    }
  }
};

template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void dspark_ring_scores(
    const device T* A [[buffer(0)]],
    const device T* source [[buffer(1)]],
    const device uint* indices [[buffer(2)]],
    device float* C [[buffer(3)]],
    const constant GEMMSpiltKParams* params [[buffer(4)]],
    const constant int64_t& batch_stride_a [[buffer(5)]],
    const constant int& source_ld [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 16;
  constexpr int WM = 2;
  constexpr int WN = 2;
  using gemm_kernel =
      GEMMKernel<T, float, BM, BN, BK, WM, WN, false, true, true, true>;
  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = DSparkRingBlockLoader<
      T,
      BN,
      BK,
      BK + gemm_kernel::tgp_padding_b,
      WM * WN * 32,
      true>;
  using mma_t = typename gemm_kernel::mma_t;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  const int partition = int(tid.z) % params->split_k_partitions;
  const int row = int(tid.z) / params->split_k_partitions;
  const int c_row = int(tid.y) * BM;
  const int c_col = int(tid.x) * BN;
  const int k_start = params->split_k_partition_size * partition;

  A += size_t(row) * batch_stride_a + size_t(c_row) * params->lda + k_start;
  indices += size_t(row) * params->N;
  C += size_t(row) * params->split_k_partitions *
          params->split_k_partition_stride +
      size_t(partition) * params->split_k_partition_stride +
      size_t(c_row) * params->ldc + c_col;

  thread loader_a_t loader_a(
      A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(
      source,
      indices,
      source_ld,
      c_col,
      k_start,
      Bs,
      simd_group_id,
      simd_lane_id);
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  for (int k = 0; k < params->gemm_k_iterations_aligned; ++k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_unsafe();
    loader_b.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.store_result(C, params->ldc);
}

template <typename T>
[[kernel, max_total_threads_per_threadgroup(64)]] void dspark_ring_values(
    const device T* A [[buffer(0)]],
    const device T* source [[buffer(1)]],
    const device uint* indices [[buffer(2)]],
    device T* D [[buffer(3)]],
    const constant GEMMParams* params [[buffer(4)]],
    const constant int64_t& batch_stride_a [[buffer(5)]],
    const constant int& source_ld [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;
  constexpr int WM = 1;
  constexpr int WN = 2;
  using gemm_kernel =
      GEMMKernel<T, T, BM, BN, BK, WM, WN, false, false, true, true>;
  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = DSparkRingBlockLoader<
      T,
      BK,
      BN,
      BN + gemm_kernel::tgp_padding_b,
      WM * WN * 32,
      false>;
  using mma_t = typename gemm_kernel::mma_t;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
  threadgroup_barrier(mem_flags::mem_none);

  const int row = int(tid.z);
  const int c_row = int(tid.y) * BM;
  const int c_col = int(tid.x) * BN;
  A += size_t(row) * batch_stride_a + size_t(c_row) * params->lda;
  indices += size_t(row) * params->K;
  D += size_t(row) * params->batch_stride_d + size_t(c_row) * params->ldd +
      c_col;

  thread loader_a_t loader_a(
      A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(
      source,
      indices,
      source_ld,
      0,
      c_col,
      Bs,
      simd_group_id,
      simd_lane_id);
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  for (int k = 0; k < params->gemm_k_iterations_aligned; ++k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_unsafe();
    loader_b.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }

  threadgroup_barrier(mem_flags::mem_none);
  mma_op.store_result(D, params->ldd);
}

// MLX uses split-K for the M=64 fallback attention GEMMs when K is at least
// max(M, N). A regular batched GEMM changes the reduction order at long KV
// lengths, so preserve the single-token path while extending grid.z with an
// independent verify-row dimension.
template <
    typename T,
    typename U,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_b,
    bool MN_aligned,
    bool K_aligned>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
dspark_gemm_splitk(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device U* C [[buffer(2)]],
    const constant GEMMSpiltKParams* params [[buffer(3)]],
    const constant int64_t& batch_stride_a [[buffer(4)]],
    const constant int64_t& batch_stride_b [[buffer(5)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using gemm_kernel = GEMMKernel<
      T,
      U,
      BM,
      BN,
      BK,
      WM,
      WN,
      false,
      transpose_b,
      MN_aligned,
      K_aligned>;
  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  const int partition = int(tid.z) % params->split_k_partitions;
  const int row = int(tid.z) / params->split_k_partitions;
  const int c_row = int(tid.y) * BM;
  const int c_col = int(tid.x) * BN;
  const int k_start = params->split_k_partition_size * partition;

  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const size_t k_start_long = size_t(k_start);
  A += size_t(row) * batch_stride_a + k_start_long +
      c_row_long * params->lda;
  B += size_t(row) * batch_stride_b +
      (transpose_b ? k_start_long + c_col_long * params->ldb
                   : c_col_long + k_start_long * params->ldb);
  C += size_t(row) * params->split_k_partitions *
          params->split_k_partition_stride +
      size_t(partition) * params->split_k_partition_stride +
      c_row_long * params->ldc + c_col_long;

  thread loader_a_t loader_a(
      A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(
      B, params->ldb, Bs, simd_group_id, simd_lane_id);
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  const short tgp_bm = min(BM, params->M - c_row);
  const short tgp_bn = min(BN, params->N - c_col);
  const short leftover_bk = params->K % BK;
  const int gemm_k_iterations = params->gemm_k_iterations_aligned;

  if (MN_aligned || (tgp_bm == BM && tgp_bn == BN)) {
    gemm_kernel::gemm_loop(
        As,
        Bs,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk,
        LoopAlignment<true, true, true>{});
  } else if (tgp_bn == BN) {
    gemm_kernel::gemm_loop(
        As,
        Bs,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk,
        LoopAlignment<false, true, true>{});
  } else if (tgp_bm == BM) {
    gemm_kernel::gemm_loop(
        As,
        Bs,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk,
        LoopAlignment<true, false, true>{});
  } else {
    gemm_kernel::gemm_loop(
        As,
        Bs,
        gemm_k_iterations,
        loader_a,
        loader_b,
        mma_op,
        tgp_bm,
        tgp_bn,
        leftover_bk,
        LoopAlignment<false, false, true>{});
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (partition + 1 == params->split_k_partitions) {
    const int remaining =
        (params->K - (k_start + params->split_k_partition_size)) / BK;
    if (!K_aligned || remaining > 0) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          remaining,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, false, K_aligned>{});
    }
  }

  if (MN_aligned || (tgp_bm == BM && tgp_bn == BN)) {
    mma_op.store_result(C, params->ldc);
  } else {
    mma_op.store_result_safe(C, params->ldc, short2(tgp_bn, tgp_bm));
  }
}

template <typename AccT, typename OutT>
[[kernel]] void dspark_gemm_splitk_accum(
    const device AccT* C_split [[buffer(0)]],
    device OutT* D [[buffer(1)]],
    const constant int& partitions [[buffer(2)]],
    const constant int& partition_stride [[buffer(3)]],
    const constant int& ldd [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  const size_t matrix_offset = size_t(gid.z) * partition_stride;
  D += matrix_offset + gid.x + size_t(gid.y) * ldd;
  C_split += size_t(gid.z) * partitions * partition_stride + gid.x +
      size_t(gid.y) * ldd;

  AccT value = 0;
  for (int partition = 0; partition < partitions; ++partition) {
    value += C_split[size_t(partition) * partition_stride];
  }
  D[0] = TransformNone<OutT, AccT>::apply(value);
}

#define instantiate_dspark_splitk(                                      \
    tname, dtype, trans_b, aname, mn_aligned, kname, k_aligned)         \
  instantiate_kernel(                                                    \
      "omlx_dspark_gemm_splitk_" #tname "_nt" #trans_b "_MN_" #aname \
      "_K_" #kname,                                                     \
      dspark_gemm_splitk,                                                \
      dtype,                                                             \
      float,                                                             \
      32,                                                                \
      32,                                                                \
      16,                                                                \
      2,                                                                 \
      2,                                                                 \
      trans_b,                                                           \
      mn_aligned,                                                        \
      k_aligned)

#define instantiate_dspark_splitk_alignments(tname, dtype, trans_b)      \
  instantiate_dspark_splitk(                                             \
      tname, dtype, trans_b, taligned, true, taligned, true);            \
  instantiate_dspark_splitk(                                             \
      tname, dtype, trans_b, taligned, true, naligned, false);           \
  instantiate_dspark_splitk(                                             \
      tname, dtype, trans_b, naligned, false, taligned, true);           \
  instantiate_dspark_splitk(                                             \
      tname, dtype, trans_b, naligned, false, naligned, false)

instantiate_dspark_splitk_alignments(float16, float16_t, false);
instantiate_dspark_splitk_alignments(float16, float16_t, true);
instantiate_dspark_splitk_alignments(bfloat16, bfloat16_t, false);
instantiate_dspark_splitk_alignments(bfloat16, bfloat16_t, true);

instantiate_kernel(
    "omlx_dspark_ring_scores_float16", dspark_ring_scores, float16_t);
instantiate_kernel(
    "omlx_dspark_ring_scores_bfloat16", dspark_ring_scores, bfloat16_t);
instantiate_kernel(
    "omlx_dspark_ring_values_float16", dspark_ring_values, float16_t);
instantiate_kernel(
    "omlx_dspark_ring_values_bfloat16", dspark_ring_values, bfloat16_t);

instantiate_kernel(
    "omlx_dspark_gemm_splitk_accum_float16_float32",
    dspark_gemm_splitk_accum,
    float,
    float16_t);
instantiate_kernel(
    "omlx_dspark_gemm_splitk_accum_bfloat16_float32",
    dspark_gemm_splitk_accum,
    float,
    bfloat16_t);
