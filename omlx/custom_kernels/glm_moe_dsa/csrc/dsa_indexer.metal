#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

namespace mlx {
namespace steel {

struct OMLXDSATopKParams {
  int rows;
  int L;
  int K;
  int topk;
  bool causal_valid_prefix;
};

} // namespace steel
} // namespace mlx

#define DSATopKParams OMLXDSATopKParams
#include "kernels/steel_dsa_indexer_score.h"
#undef DSATopKParams

#define instantiate_dsa_indexer_score(iname, itype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                   \
      "steel_dsa_indexer_score_" #iname                                 \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,              \
      dsa_indexer_score, itype, bm, bn, bk, wm, wn)

#define instantiate_dsa_topk_indices(iname, itype, topk, threads)       \
  instantiate_kernel(                                                   \
      "steel_dsa_topk_indices_" #iname "_topk" #topk "_t" #threads,    \
      dsa_topk_indices_16bit,                                           \
      itype,                                                            \
      uint,                                                             \
      topk,                                                             \
      threads)

// bn=64,wm=2,wn=2 (128 threads): the historical config, used for all
// shapes. A bn=128/wn=4 variant measured slower on M3 Ultra and was
// removed; see the tile-config note in dsa_indexer.cpp. bm/bn/wm/wn never
// enter the per-element K-reduction order (bk=16 and the MMA fragment
// K-layout are unchanged), so tile configs are bit-identical.
// NOTE: a K-panel hoist (resident B tile across heads) was tried and
// REVERTED: the 27KB threadgroup footprint collapses occupancy to 1
// threadgroup/core (32KB budget) and costs ~18% despite the traffic saving.
instantiate_dsa_indexer_score(float16, half, 64, 64, 16, 2, 2);
instantiate_dsa_indexer_score(bfloat16, bfloat16_t, 64, 64, 16, 2, 2);

instantiate_dsa_topk_indices(float16, half, 2048, 1024);
instantiate_dsa_topk_indices(bfloat16, bfloat16_t, 2048, 1024);
instantiate_dsa_topk_indices(float16, half, 512, 1024);
instantiate_dsa_topk_indices(bfloat16, bfloat16_t, 512, 1024);

instantiate_kernel(
    "dspark_fp32_topk_indices_topk512_t256",
    dspark_fp32_topk_indices,
    512,
    256);

// ── DC-1: fused decode indexer scan ──────────────────────────────────────────
// One thread per key position: score(key) = sum_h w_h * relu(q_h · k_key), fp32
// accumulation throughout (strictly tighter than the bf16 op-chain it replaces).
// The query heads (HEADS*DIM, 8KB at 32 heads / 16KB at 64) + scaled weights
// live in threadgroup memory; K is streamed exactly once; the [B,H,1,S] / relu
// / weighted / summed intermediates of the unfused chain never exist. The
// weight pointer has its own type WT so fp32 weights can feed the fp32-score
// configuration without quantizing the head weights (DeepSeek V4's reference
// decode path keeps them in fp32).
struct OMLXDSADecodeParams {
  int S;
  int64_t k_batch_stride; // elements
  int64_t k_row_stride;   // elements (last dim must be contiguous)
};

template <typename T, typename OutT, typename WT, int HEADS, int DIM, int THREADS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void dsa_decode_scores_kernel(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device WT* W [[buffer(2)]],
    device OutT* OUT [[buffer(3)]],
    const constant OMLXDSADecodeParams& params [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
  (void)simd_lane;
  (void)simd_id;
  const uint lid = lid3.x;
  const int b = int(tid.z);
  const int S = params.S;
  const device T* q_base = Q + size_t(b) * HEADS * DIM;
  const device T* k_base = K + size_t(b) * size_t(params.k_batch_stride);
  const device WT* w_base = W + size_t(b) * HEADS;
  device OutT* out_base = OUT + size_t(b) * size_t(S);

  threadgroup T qs[HEADS * DIM];
  threadgroup float ws[HEADS];
  for (int i = int(lid); i < HEADS * DIM; i += THREADS) {
    qs[i] = q_base[i];
  }
  for (int i = int(lid); i < HEADS; i += THREADS) {
    ws[i] = float(w_base[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int key = int(tid.x) * THREADS + int(lid);
  if (key >= S) {
    return;
  }
  // K is addressed by strides so capacity-backed cache SLICES need no copy
  // (rows stay 16B-aligned: row stride is a multiple of 8 elements, enforced
  // by the host). One coalesced vec<T,4> stream per key row.
  const device vec<T, 4>* krow4 = reinterpret_cast<const device vec<T, 4>*>(
      k_base + size_t(key) * size_t(params.k_row_stride));
  const threadgroup vec<T, 4>* qs4 =
      reinterpret_cast<const threadgroup vec<T, 4>*>(qs);

  float acc[HEADS];
  STEEL_PRAGMA_UNROLL
  for (short h = 0; h < HEADS; ++h) {
    acc[h] = 0.0f;
  }

  constexpr short kChunks = DIM / 4;
  for (short c = 0; c < kChunks; ++c) {
    const float4 kf = float4(krow4[c]);
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < HEADS; ++h) {
      acc[h] += metal::dot(float4(qs4[h * kChunks + c]), kf);
    }
  }

  float total = 0.0f;
  STEEL_PRAGMA_UNROLL
  for (short h = 0; h < HEADS; ++h) {
    total += metal::max(acc[h], 0.0f) * ws[h];
  }
  out_base[key] = OutT(total);
}

#define instantiate_dsa_decode_scores(iname, itype, oname, otype, wname, wtype, heads, dim, threads) \
  instantiate_kernel(                                                     \
      "dsa_decode_scores_" #iname "_o" #oname "_w" #wname "_h" #heads     \
      "_d" #dim "_t" #threads,                                            \
      dsa_decode_scores_kernel,                                           \
      itype,                                                              \
      otype,                                                              \
      wtype,                                                              \
      heads,                                                              \
      dim,                                                                \
      threads)

// same-dtype scores (default: feeds the native 16-bit radix top-k) and fp32
// scores (opt-in: selection then matches fp32 ground truth exactly).
// 32 heads: GLM-MoE-DSA; weights always in the query dtype (wsame).
instantiate_dsa_decode_scores(float16, half, same, half, same, half, 32, 128, 256);
instantiate_dsa_decode_scores(bfloat16, bfloat16_t, same, bfloat16_t, same, bfloat16_t, 32, 128, 256);
instantiate_dsa_decode_scores(float16, half, f32, float, same, half, 32, 128, 256);
instantiate_dsa_decode_scores(bfloat16, bfloat16_t, f32, float, same, bfloat16_t, 32, 128, 256);
// 64 heads: DeepSeek V4. The wf32 variants take fp32 head weights so the fused
// scan reproduces the reference fp32 decode/verify path's weight precision
// bit-for-bit (only the dot-product accumulation order differs); they pair
// with fp32 score output.
instantiate_dsa_decode_scores(float16, half, same, half, same, half, 64, 128, 256);
instantiate_dsa_decode_scores(bfloat16, bfloat16_t, same, bfloat16_t, same, bfloat16_t, 64, 128, 256);
instantiate_dsa_decode_scores(float16, half, f32, float, same, half, 64, 128, 256);
instantiate_dsa_decode_scores(bfloat16, bfloat16_t, f32, float, same, bfloat16_t, 64, 128, 256);
instantiate_dsa_decode_scores(float16, half, f32, float, f32, float, 64, 128, 256);
instantiate_dsa_decode_scores(bfloat16, bfloat16_t, f32, float, f32, float, 64, 128, 256);
