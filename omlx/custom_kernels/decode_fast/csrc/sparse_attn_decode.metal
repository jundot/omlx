// SPDX-License-Identifier: Apache-2.0
// Fused sparse decode attention for DeepSeek-V4-style MQA layers.
//
// One decode step attends a sliding-window KV segment plus a selected pooled
// segment through a single shared K=V head (D=512). The stock omlx path runs
// this as four rowwise GEMMs with logsumexp/exp glue under mx.compile (~45us
// per layer-call at S~640, ~245us for a DSpark verify batch); this kernel does
// the whole thing in one dispatch.
//
// All 64 query heads share the same KV rows, so each threadgroup processes
// HPG=4 heads: every KV row is read from global memory once per threadgroup
// instead of once per head (4x less L2 traffic, the dominant cost).
//
//   pass 1: scores[hh][j] = dot(q_hh, kv_j)   (row-per-thread, 4 heads at once)
//   pass 2: per-head logaddexp(logsumexp, sink) normalizer; weights in place
//   pass 3: out[hh][d] = sum_j weights[hh][j] * kv_j[d]  (dim-per-thread)
//
// Every (batch, head) pair follows a fixed, deterministic reduction order
// independent of batch size, so one-token decode and DSpark verify rows
// (batch-carried, B<=8) get bitwise-identical per-row results — the invariant
// omlx's decode_consistency machinery requires.
//
// Intermediate roundings mimic the composed bf16 reference: scores are rounded
// to T after the dot, per-segment logsumexp outputs and their logaddexp are
// T-rounded, the sink logaddexp stays fp32 (sink promotion), weights are fp32,
// and the value pass keeps the reference's separate local/pooled fp32 partial
// sums before a single final rounding to T.

#include <metal_math>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Bound on W + P (window + selected pooled rows). V4 decode: 128 + 512.
#define OMLX_SPARSE_MAX_S 1024

template <typename T, int D, int HPG>
[[kernel]] void omlx_sparse_attn_decode(
    const device T* q [[buffer(0)]],          // [B, H, 1, D]
    const device T* local_kv [[buffer(1)]],   // [B, 1, W, D]
    const device T* pooled [[buffer(2)]],     // [B, 1, P, D]
    const device float* sinks [[buffer(3)]],  // [H]
    device T* out [[buffer(4)]],              // [B, H, 1, D]
    const constant int& W [[buffer(5)]],
    const constant int& P [[buffer(6)]],
    const constant int& H [[buffer(7)]],
    const constant size_t& q_b_stride [[buffer(8)]],
    const constant size_t& q_h_stride [[buffer(9)]],
    const constant size_t& kv_b_stride [[buffer(10)]],
    const constant size_t& kv_row_stride [[buffer(11)]],
    const constant size_t& pooled_b_stride [[buffer(12)]],
    const constant size_t& pooled_row_stride [[buffer(13)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr int NTHREADS = 256;

  const int h0 = tg_pos.x * HPG;
  const int b = tg_pos.y;
  const int S = W + P;

  threadgroup float scores[HPG][OMLX_SPARSE_MAX_S];
  threadgroup float qbuf[HPG][D];
  threadgroup float partials[2 * NTHREADS / 32];
  threadgroup float nrm_shared = 0.0f;

  const device T* qbase = q + (size_t)b * q_b_stride + (size_t)h0 * q_h_stride;

  // Stage the HPG query rows in fp32.
  for (int i = tid; i < HPG * D; i += NTHREADS) {
    const int hh = i / D;
    qbuf[hh][i % D] = static_cast<float>(qbase[hh * q_h_stride + (i % D)]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pass 1: row-per-thread dots against all HPG heads. Thread t owns rows
  // t, t+256, ...; each row's 1KB is consumed by one thread in cache-line
  // order, and each row read serves HPG heads.
  for (int j = tid; j < S; j += NTHREADS) {
    const device T* row = (j < W)
        ? local_kv + (size_t)b * kv_b_stride + (size_t)j * kv_row_stride
        : pooled + (size_t)b * pooled_b_stride + (size_t)(j - W) * pooled_row_stride;
    float acc[HPG];
#pragma unroll
    for (int hh = 0; hh < HPG; hh++) {
      acc[hh] = 0.f;
    }
#pragma unroll 4
    for (int c = 0; c < D / 8; c++) {
      const vec<T, 8> rv =
          *reinterpret_cast<const device vec<T, 8>*>(row + c * 8);
#pragma unroll
      for (int hh = 0; hh < HPG; hh++) {
        const threadgroup float4* qa_h =
            reinterpret_cast<const threadgroup float4*>(&qbuf[hh][c * 8]);
        const threadgroup float4* qb_h =
            reinterpret_cast<const threadgroup float4*>(&qbuf[hh][c * 8 + 4]);
        acc[hh] += qa_h->x * static_cast<float>(rv[0]) +
            qa_h->y * static_cast<float>(rv[1]) +
            qa_h->z * static_cast<float>(rv[2]) +
            qa_h->w * static_cast<float>(rv[3]) +
            qb_h->x * static_cast<float>(rv[4]) +
            qb_h->y * static_cast<float>(rv[5]) +
            qb_h->z * static_cast<float>(rv[6]) +
            qb_h->w * static_cast<float>(rv[7]);
      }
    }
#pragma unroll
    for (int hh = 0; hh < HPG; hh++) {
      // Round to the storage dtype, matching the reference GEMM's bf16/fp16
      // output before the softmax glue.
      scores[hh][j] = static_cast<float>(static_cast<T>(acc[hh]));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pass 2: per-head normalizer, mirroring the reference rounding chain:
  //   lse_local  = T(logsumexp(scores[0:W]))     (T-rounded, per segment)
  //   lse_pooled = T(logsumexp(scores[W:S]))
  //   nrm_bf     = T(logaddexp(lse_local, lse_pooled))
  //   nrm        = logaddexp(fp32(nrm_bf), sink)  (fp32 via sink promotion)
  //   weight[j]  = exp(fp32(score[j]) - nrm)      (fp32, unrounded)
  for (int hh = 0; hh < HPG; hh++) {
    threadgroup float* sc = scores[hh];
    float m_loc = -INFINITY;
    float m_pool = -INFINITY;
    for (int j = tid; j < S; j += NTHREADS) {
      if (j < W) {
        m_loc = max(m_loc, sc[j]);
      } else {
        m_pool = max(m_pool, sc[j]);
      }
    }
    m_loc = simd_max(m_loc);
    m_pool = simd_max(m_pool);
    const uint sg = tid / 32;
    const uint sl = tid % 32;
    if (sl == 0) {
      partials[sg] = m_loc;
      partials[NTHREADS / 32 + sg] = m_pool;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint g = 1; g < NTHREADS / 32; g++) {
      m_loc = max(m_loc, partials[g]);
      m_pool = max(m_pool, partials[NTHREADS / 32 + g]);
    }
    m_loc = max(m_loc, partials[0]);
    m_pool = max(m_pool, partials[NTHREADS / 32]);
    // All threads must finish reading the max partials before the sum
    // partials overwrite the buffer.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s_loc = 0.f;
    float s_pool = 0.f;
    for (int j = tid; j < S; j += NTHREADS) {
      if (j < W) {
        s_loc += precise::exp(sc[j] - m_loc);
      } else {
        s_pool += precise::exp(sc[j] - m_pool);
      }
    }
    s_loc = simd_sum(s_loc);
    s_pool = simd_sum(s_pool);
    if (sl == 0) {
      partials[sg] = s_loc;
      partials[NTHREADS / 32 + sg] = s_pool;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float sum_loc = 0.f;
      float sum_pool = 0.f;
#pragma unroll
      for (uint g = 0; g < NTHREADS / 32; g++) {
        sum_loc += partials[g];
        sum_pool += partials[NTHREADS / 32 + g];
      }
      const float sink = sinks[h0 + hh];
      // W == 0 / P == 0 segments contribute -inf, like an empty logsumexp.
      float lse_loc = (W > 0) ? (m_loc + precise::log(sum_loc)) : -INFINITY;
      float lse_pool = (P > 0) ? (m_pool + precise::log(sum_pool)) : -INFINITY;
      // Round each segment's logsumexp to T, then logaddexp in T.
      lse_loc = static_cast<float>(static_cast<T>(lse_loc));
      lse_pool = static_cast<float>(static_cast<T>(lse_pool));
      float m2 = max(lse_loc, lse_pool);
      float nrm_bf = static_cast<float>(static_cast<T>(
          m2 + precise::log(precise::exp(lse_loc - m2) +
                            precise::exp(lse_pool - m2))));
      // logaddexp with the fp32 sink stays in fp32.
      float m3 = max(nrm_bf, sink);
      nrm_shared = m3 +
          precise::log(precise::exp(nrm_bf - m3) + precise::exp(sink - m3));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float nrm = nrm_shared;

    for (int j = tid; j < S; j += NTHREADS) {
      sc[j] = precise::exp(sc[j] - nrm);
    }
    if (hh + 1 < HPG) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pass 3: weighted values, with the reference's separate local/pooled
  // fp32 partial sums. Thread t owns output dims {2t, 2t+1} and walks all
  // rows once, applying all HPG heads' weights per row read.
  static_assert(D == 2 * NTHREADS, "pass 3 assumes 2 output dims per thread");
  const int d0 = tid * 2;
  float acc_loc[HPG][2];
  float acc_pool[HPG][2];
#pragma unroll
  for (int hh = 0; hh < HPG; hh++) {
    acc_loc[hh][0] = 0.f;
    acc_loc[hh][1] = 0.f;
    acc_pool[hh][0] = 0.f;
    acc_pool[hh][1] = 0.f;
  }
  {
    const device T* rows = local_kv + (size_t)b * kv_b_stride + d0;
    int j = 0;
    for (; j + 4 <= W; j += 4) {
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const vec<T, 2> rv = *reinterpret_cast<const device vec<T, 2>*>(
            rows + (j + u) * kv_row_stride);
        const float v0 = static_cast<float>(rv[0]);
        const float v1 = static_cast<float>(rv[1]);
#pragma unroll
        for (int hh = 0; hh < HPG; hh++) {
          const float w = scores[hh][j + u];
          acc_loc[hh][0] += w * v0;
          acc_loc[hh][1] += w * v1;
        }
      }
    }
    for (; j < W; j++) {
      const vec<T, 2> rv =
          *reinterpret_cast<const device vec<T, 2>*>(rows + j * kv_row_stride);
      const float v0 = static_cast<float>(rv[0]);
      const float v1 = static_cast<float>(rv[1]);
#pragma unroll
      for (int hh = 0; hh < HPG; hh++) {
        const float w = scores[hh][j];
        acc_loc[hh][0] += w * v0;
        acc_loc[hh][1] += w * v1;
      }
    }
  }
  {
    const device T* rows = pooled + (size_t)b * pooled_b_stride + d0;
    int j = 0;
    for (; j + 4 <= P; j += 4) {
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const vec<T, 2> rv = *reinterpret_cast<const device vec<T, 2>*>(
            rows + (j + u) * pooled_row_stride);
        const float v0 = static_cast<float>(rv[0]);
        const float v1 = static_cast<float>(rv[1]);
#pragma unroll
        for (int hh = 0; hh < HPG; hh++) {
          const float w = scores[hh][W + j + u];
          acc_pool[hh][0] += w * v0;
          acc_pool[hh][1] += w * v1;
        }
      }
    }
    for (; j < P; j++) {
      const vec<T, 2> rv = *reinterpret_cast<const device vec<T, 2>*>(
          rows + j * pooled_row_stride);
      const float v0 = static_cast<float>(rv[0]);
      const float v1 = static_cast<float>(rv[1]);
#pragma unroll
      for (int hh = 0; hh < HPG; hh++) {
        const float w = scores[hh][W + j];
        acc_pool[hh][0] += w * v0;
        acc_pool[hh][1] += w * v1;
      }
    }
  }
  device T* orow = out + (size_t)(b * H + h0) * D;
#pragma unroll
  for (int hh = 0; hh < HPG; hh++) {
    orow[hh * D + d0] = static_cast<T>(acc_loc[hh][0] + acc_pool[hh][0]);
    orow[hh * D + d0 + 1] = static_cast<T>(acc_loc[hh][1] + acc_pool[hh][1]);
  }
}

#define INSTANTIATE_SPARSE_ATTN_DECODE(T, D, HPG)                          \
  template [[host_name("omlx_sparse_attn_decode_" #T "_" #D "_" #HPG)]]    \
      [[kernel]] void                                                      \
      omlx_sparse_attn_decode<T, D, HPG>(                                  \
          const device T* q [[buffer(0)]],                                 \
          const device T* local_kv [[buffer(1)]],                          \
          const device T* pooled [[buffer(2)]],                            \
          const device float* sinks [[buffer(3)]],                         \
          device T* out [[buffer(4)]],                                     \
          const constant int& W [[buffer(5)]],                             \
          const constant int& P [[buffer(6)]],                             \
          const constant int& H [[buffer(7)]],                             \
          const constant size_t& q_b_stride [[buffer(8)]],                 \
          const constant size_t& q_h_stride [[buffer(9)]],                 \
          const constant size_t& kv_b_stride [[buffer(10)]],               \
          const constant size_t& kv_row_stride [[buffer(11)]],             \
          const constant size_t& pooled_b_stride [[buffer(12)]],           \
          const constant size_t& pooled_row_stride [[buffer(13)]],         \
          uint2 tg_pos [[threadgroup_position_in_grid]],                   \
          uint tid [[thread_index_in_threadgroup]]);

INSTANTIATE_SPARSE_ATTN_DECODE(bfloat16_t, 512, 4)
INSTANTIATE_SPARSE_ATTN_DECODE(float16_t, 512, 4)
INSTANTIATE_SPARSE_ATTN_DECODE(float, 512, 4)
