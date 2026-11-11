// Adapted from TensorFold zig/kernels/cuda/gdn_prefill.cu, Prefetch
// (Apache-2.0, Copyright 2026 TensorFold contributors). Changes: bf16 queries,
// keys and update rates only; renamed with a gdn_ prefix.
#include <cuda_bf16.h>

constexpr int GDN_DK = 128;
constexpr int GDN_DV = 128;

__device__ __forceinline__ float4 gdn_widen(uint2 w) {
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&w.x);
  const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w.y);
  return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
}

// The inputs of STEPS steps from t0 on: keys and queries of one key head,
// values of ROWS value rows from col0, decay and update rate of the head.
template <int THREADS, int ROWS, int STEPS> struct GdnPrefetch {
  static constexpr int KQ = STEPS * (GDN_DK / 4) / THREADS, VV = STEPS * ROWS / 8 / THREADS;
  static_assert(KQ * THREADS == STEPS * (GDN_DK / 4) && VV * THREADS * 8 == STEPS * ROWS,
                "whole slots a thread");
  uint2 kr[KQ], qr[KQ];
  uint4 vr[VV];
  float gr, br;

  __device__ __forceinline__ void fetch(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                        const __nv_bfloat16* v, const float* g,
                                        const __nv_bfloat16* beta, int t0, int W, int hk, int hv,
                                        int head, int key_head, int col0) {
#pragma unroll
    for (int it = 0; it < KQ; ++it) {
      const int i = threadIdx.x + it * THREADS, st = i / (GDN_DK / 4), c = i % (GDN_DK / 4);
      const size_t at = (size_t(min(t0 + st, W - 1)) * hk + key_head) * GDN_DK + 4 * c;
      kr[it] = *reinterpret_cast<const uint2*>(k + at);
      qr[it] = *reinterpret_cast<const uint2*>(q + at);
    }
#pragma unroll
    for (int it = 0; it < VV; ++it) {
      const int i = threadIdx.x + it * THREADS, st = i / (ROWS / 8), c = i % (ROWS / 8);
      vr[it] = *reinterpret_cast<const uint4*>(
          v + (size_t(min(t0 + st, W - 1)) * hv + head) * GDN_DV + col0 + 8 * c);
    }
    if (threadIdx.x < STEPS) {
      const size_t at = size_t(min(t0 + int(threadIdx.x), W - 1)) * hv + head;
      gr = g[at];
      br = __bfloat162float(beta[at]);
    }
  }

  __device__ __forceinline__ void store(float4 (*ks)[GDN_DK / 4], float4 (*qs)[GDN_DK / 4],
                                        float* gs, float* bs,
                                        __nv_bfloat16 (*vs)[ROWS]) const {
#pragma unroll
    for (int it = 0; it < KQ; ++it) {
      const int i = threadIdx.x + it * THREADS;
      ks[i / (GDN_DK / 4)][i % (GDN_DK / 4)] = gdn_widen(kr[it]);
      qs[i / (GDN_DK / 4)][i % (GDN_DK / 4)] = gdn_widen(qr[it]);
    }
#pragma unroll
    for (int it = 0; it < VV; ++it) {
      const int i = threadIdx.x + it * THREADS;
      *reinterpret_cast<uint4*>(&vs[i / (ROWS / 8)][8 * (i % (ROWS / 8))]) = vr[it];
    }
    if (threadIdx.x < STEPS) {
      gs[threadIdx.x] = gr;
      bs[threadIdx.x] = br;
    }
  }
};
