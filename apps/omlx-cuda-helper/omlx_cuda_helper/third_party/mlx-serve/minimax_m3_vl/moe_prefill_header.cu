// From mlx-serve src/kernels/moe/moe_prefill_cuda_header.cu (MIT, Copyright
// (c) 2026 David Dalcu), unchanged below this comment.
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
__device__ __forceinline__ uint32_t msv_smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void msv_cp16(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void msv_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void msv_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void msv_ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void msv_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float2 msv_e2m1x2(uint32_t v8) {
  __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(v8 & 0xff), __NV_E2M1);
  return __half22float2(*reinterpret_cast<__half2*>(&h));
}
__device__ __forceinline__ float2 msv_e4m3x2(uint32_t v16) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v16 & 0xffff), __NV_E4M3);
  return __half22float2(*reinterpret_cast<__half2*>(&h));
}
__device__ __forceinline__ float msv_e4m3(uint32_t s) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(s & 0xff), __NV_E4M3);
  return __half2float(*reinterpret_cast<__half*>(&h));
}
__device__ __forceinline__ float msv_ue8m0(uint32_t s) { return s == 0 ? __int_as_float(0x00400000) : __int_as_float(s << 23); }
// Rows [begin, end) of sorted `inds` equal to e: warp 0 runs a 32-ary search, a few rounds of loads.
__device__ __forceinline__ void msv_expert_range(const uint32_t* inds, int R, uint32_t e, int* range) {
  const int lane = threadIdx.x & 31;
  if ((threadIdx.x >> 5) == 0) {
    for (int which = 0; which < 2; ++which) {
      const uint32_t v = e + which;
      int lo = 0, hi = R;
      while (hi - lo > 32) {
        const int step = (hi - lo + 31) / 32;
        const int p = lo + lane * step;
        const int cnt = __popc(__ballot_sync(0xffffffff, p < hi && inds[p] < v));
        const int nlo = cnt == 0 ? lo : lo + (cnt - 1) * step + 1;
        hi = lo + cnt * step < hi ? lo + cnt * step : hi;
        lo = nlo;
      }
      const int res = lo + __popc(__ballot_sync(0xffffffff, lo + lane < hi && inds[lo + lane] < v));
      if (lane == 0) range[which] = res;
    }
  }
  __syncthreads();
}
// One BM x 128 output tile over the whole K: A rows from `arow(row)` (bf16), the 128 B rows from
// `wrow(r)` / `srow(r)` (codes, scales). Weights stream through registers BD 128-wide K stages ahead,
// each warp load reading whole 64+ byte row pieces; the CTA turns a stage into one bf16 tile that
// ldmatrix feeds to mma.m16n8k16, K in order, as qmm_sm80 does. Leaves smem free for the caller.
template <int FMT, int K, int BD, int ST, typename T, typename ARow, typename WRow, typename SRow>
__device__ __forceinline__ void msv_expert_tile(uint8_t* smem, const uint32_t* lut, int mv, ARow arow, WRow wrow, SRow srow, float (&acc)[2][4][4]) {
  constexpr int BM = 64, BN = 128, KS = 128;
  constexpr int BITS = FMT == 1 ? 8 : 4;
  constexpr int GROUP = FMT == 3 ? 16 : 32;
  constexpr int RB = KS * BITS / 8;
  constexpr int KT = K / KS;
  constexpr int A_ST = BM * KS * 2;
  constexpr int CPR = RB / 16;
  constexpr int CV = 128 / BITS;
  constexpr int NL = BN * CPR / 256;
  uint8_t* sa = smem;
  uint8_t* sw = smem + ST * A_ST;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int cc = tid % CPR;
  auto load_a = [&](int t) {
    const int st = t % ST;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int q = tid + i * 256, row = q >> 4, c = q & 15;
      const bool ok = row < mv;
      msv_cp16(msv_smem(sa + st * A_ST + row * 256 + ((c ^ (row & 7)) << 4)), arow(ok ? row : 0) + t * KS + c * 8, ok);
    }
  };
  // Load i: B row (i * 256 + tid) / CPR, chunk cc; the thread dequantizes exactly what it loads.
  uint4 braw[BD][NL];
  uint32_t bsc[BD][NL];
  auto load_b = [&](int t, uint4 (&dst)[NL], uint32_t (&sc)[NL]) {
    #pragma unroll
    for (int i = 0; i < NL; ++i) {
      const int r = (i * 256 + tid) / CPR;
      dst[i] = __ldcg(reinterpret_cast<const uint4*>(wrow(r) + (long)t * RB + cc * 16));
      const uint8_t* sp = srow(r) + (t * KS + cc * CV) / GROUP;
      if constexpr (FMT == 3) sc[i] = __ldcg(reinterpret_cast<const unsigned short*>(sp));
      else sc[i] = __ldcg(sp);
    }
  };
  #pragma unroll
  for (int a = 0; a < 2; ++a)
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      #pragma unroll
      for (int v = 0; v < 4; ++v) acc[a][j][v] = 0.f;
  #pragma unroll
  for (int i = 0; i < ST - 1; ++i) { if (i < KT) load_a(i); msv_commit(); }
  #pragma unroll
  for (int i = 0; i < BD; ++i) if (i < KT) load_b(i, braw[i], bsc[i]);
  for (int t0 = 0; t0 < KT; t0 += BD) {
    #pragma unroll
    for (int j = 0; j < BD; ++j) {
      const int t = t0 + j;
      if (t < KT) {
        msv_wait<ST - 2 < 0 ? 0 : ST - 2>();
        __syncthreads();
        if (t + ST - 1 < KT) load_a(t + ST - 1);
        msv_commit();
        // The product of two bf16 values is exact here, so the bf16x2 multiply rounds it as qmm_sm80's does.
        #pragma unroll
        for (int i = 0; i < NL; ++i) {
          const int r = (i * 256 + tid) / CPR;
          const uint4 v = braw[j][i];
          const uint32_t vw[4] = {v.x, v.y, v.z, v.w};
          uint32_t out[CV / 2];
          if constexpr (FMT == 3) {
            const __nv_bfloat162 s0 = __float2bfloat162_rn(msv_e4m3(bsc[j][i])), s1 = __float2bfloat162_rn(msv_e4m3(bsc[j][i] >> 8));
            #pragma unroll
            for (int w4 = 0; w4 < 4; ++w4)
              #pragma unroll
              for (int q = 0; q < 4; ++q) {
                uint32_t c = lut[(vw[w4] >> (8 * q)) & 0xff];
                __nv_bfloat162 p = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&c), w4 < 2 ? s0 : s1);
                out[4 * w4 + q] = *reinterpret_cast<uint32_t*>(&p);
              }
          } else {
            const __nv_bfloat162 s0 = __float2bfloat162_rn(msv_ue8m0(bsc[j][i]));
            #pragma unroll
            for (int w4 = 0; w4 < 4; ++w4)
              #pragma unroll
              for (int q = 0; q < 2; ++q) {
                uint32_t c = lut[(vw[w4] >> (16 * q)) & 0xff] | (lut[(vw[w4] >> (16 * q + 8)) & 0xff] << 16);
                __nv_bfloat162 p = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&c), s0);
                out[2 * w4 + q] = *reinterpret_cast<uint32_t*>(&p);
              }
          }
          #pragma unroll
          for (int q = 0; q < CV / 8; ++q) {
            const int c = cc * (CV / 8) + q;
            *reinterpret_cast<uint4*>(sw + r * 256 + ((c ^ (r & 7)) << 4)) = make_uint4(out[4 * q], out[4 * q + 1], out[4 * q + 2], out[4 * q + 3]);
          }
        }
        if (t + BD < KT) load_b(t + BD, braw[j], bsc[j]);
        __syncthreads();
        if (wm * 32 < mv) {
          const uint8_t* at = sa + (t % ST) * A_ST;
          #pragma unroll
          for (int kb = 0; kb < 8; ++kb) {
            uint32_t af[2][4];
            #pragma unroll
            for (int a = 0; a < 2; ++a) {
              const int idx = lane >> 3;
              const int row = wm * 32 + a * 16 + (idx & 1) * 8 + (lane & 7);
              const int c = kb * 2 + (idx >> 1);
              msv_ldsm4(msv_smem(at + row * 256 + ((c ^ (row & 7)) << 4)), af[a][0], af[a][1], af[a][2], af[a][3]);
            }
            uint32_t bf[4][2];
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
              const int idx = lane >> 3;
              const int n = wn * 32 + p * 16 + (idx >> 1) * 8 + (lane & 7);
              const int c = kb * 2 + (idx & 1);
              msv_ldsm4(msv_smem(sw + n * 256 + ((c ^ (n & 7)) << 4)), bf[2 * p][0], bf[2 * p][1], bf[2 * p + 1][0], bf[2 * p + 1][1]);
            }
            #pragma unroll
            for (int a = 0; a < 2; ++a) {
              if (wm * 32 + a * 16 < mv) {
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) msv_mma(acc[a][jj], af[a], bf[jj][0], bf[jj][1]);
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();
}
// Code -> bf16 table after the tiles: nvfp4 maps a byte to two bf16 (low nibble first), mxfp8 to one.
template <int FMT, int ST>
__device__ __forceinline__ uint32_t* msv_lut(uint8_t* smem) {
  uint32_t* lut = reinterpret_cast<uint32_t*>(smem + ST * 64 * 128 * 2 + 128 * 128 * 2);
  const int i = threadIdx.x;
  if constexpr (FMT == 3) {
    const float2 p = msv_e2m1x2(i);
    __nv_bfloat162 b = __floats2bfloat162_rn(p.x, p.y);
    lut[i] = *reinterpret_cast<uint32_t*>(&b);
  } else {
    const float2 p = msv_e4m3x2(i);
    __nv_bfloat162 b = __floats2bfloat162_rn(p.x, p.x);
    lut[i] = *reinterpret_cast<uint32_t*>(&b) & 0xffff;
  }
  return lut;
}
