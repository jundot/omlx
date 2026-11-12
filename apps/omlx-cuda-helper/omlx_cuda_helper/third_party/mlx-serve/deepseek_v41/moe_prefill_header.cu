// Adapted from mlx-serve src/kernels/moe/moe_prefill_cuda_header.cu (MIT,
// Copyright (c) 2026 David Dalcu). Changed: the expert weights are affine
// codes (3, 4 or 8 bits, 64 weights a group, bf16 scales and biases), where a
// weight is code * scale + bias in float rounded once, or with MX MLX mxfp4
// codes (E2M1, one E8M0 scale per 32 weights), whose weights bf16 holds exactly.
// Each tensor-core step starts from zero and its 16 products are added to
// the running sum with a correctly rounded float add: a tensor core does not
// round its own accumulator to nearest, which over a 4096-wide row moves
// several times more 16-bit results than an in-order float32 sum does.
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include <cuda_bf16.h>
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
  const float zero = 0.f;
  float p[4];
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(zero));
  c[0] = __fadd_rn(c[0], p[0]);
  c[1] = __fadd_rn(c[1], p[1]);
  c[2] = __fadd_rn(c[2], p[2]);
  c[3] = __fadd_rn(c[3], p[3]);
}
// Bits of code * scale + bias, computed in float and rounded once to bf16.
__device__ __forceinline__ uint32_t msv_affine(uint32_t code, float scale, float bias) {
  const __nv_bfloat16 v = __float2bfloat16_rn(fmaf(static_cast<float>(code), scale, bias));
  return *reinterpret_cast<const unsigned short*>(&v);
}
// The float value of E2M1 code q (sign bit 3) times the E8M0 scale
// 2**(e - 127), built from its bits. The product is exact in float and in
// bf16, so the order of the decode does not matter.
__device__ __forceinline__ float msv_mx4(uint32_t q, float scale) {
  const uint32_t m = q & 7u;
  const uint32_t mag = m < 2u ? m * 0x3F000000u : (((126u + (m >> 1)) << 23) | ((m & 1u) << 22));
  return __uint_as_float(mag | ((q & 8u) << 28)) * scale;
}
// Code i of 64 3-bit codes packed as one little-endian bit stream in w.
__device__ __forceinline__ uint32_t msv_code3(const unsigned long long* w, int i) {
  const int bit = 3 * i, word = bit >> 6, shift = bit & 63;
  unsigned long long v = w[word] >> shift;
  if (shift > 61) v |= w[word + 1] << (64 - shift);
  return static_cast<uint32_t>(v & 7ull);
}
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
// `wrow(r)` (codes), `srow(r)` and `brow(r)` (scales, biases). Weights stream through registers BD
// 128-wide K stages ahead, each warp load reading whole 64+ byte row pieces; the CTA turns a stage
// into one bf16 tile that ldmatrix feeds to mma.m16n8k16, K in order. Leaves smem free for the caller.
template <int BITS, int K, int BD, int ST, bool MX, typename ARow, typename WRow, typename SRow, typename BRow>
__device__ __forceinline__ void msv_expert_tile(uint8_t* smem, int mv, ARow arow, WRow wrow, SRow srow, BRow brow, float (&acc)[2][4][4]) {
  constexpr int BM = 64, BN = 128, KS = 128, GROUP = MX ? 32 : 64;
  constexpr int RB = KS * BITS / 8;
  constexpr int KT = K / KS;
  constexpr int A_ST = BM * KS * 2;
  // 3-bit codes: thread tid holds row tid / 2, the 64 codes (24 bytes, one
  // group) at k (tid % 2) * 64 of a stage, in two uint4 registers.
  constexpr int CPR = BITS == 3 ? 1 : RB / 16;
  constexpr int CV = BITS == 3 ? 64 : 128 / BITS;
  constexpr int NL = BITS == 3 ? 1 : BN * CPR / 256;
  constexpr int NB = BITS == 3 ? 2 : NL;
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
  uint4 braw[BD][NB];
  uint32_t bsc[BD][NL];
  auto load_b = [&](int t, uint4 (&dst)[NB], uint32_t (&sc)[NL]) {
    if constexpr (BITS == 3) {
      const int r = tid >> 1, part = tid & 1;
      const uint2* p = reinterpret_cast<const uint2*>(wrow(r) + (long)t * RB + part * 24);
      const uint2 a = __ldcg(p), b = __ldcg(p + 1), c = __ldcg(p + 2);
      dst[0] = make_uint4(a.x, a.y, b.x, b.y);
      dst[1] = make_uint4(c.x, c.y, 0u, 0u);
      const int g = (t * KS + part * 64) / GROUP;
      sc[0] = static_cast<uint32_t>(__ldcg(reinterpret_cast<const unsigned short*>(srow(r) + g))) |
          (static_cast<uint32_t>(__ldcg(reinterpret_cast<const unsigned short*>(brow(r) + g))) << 16);
      return;
    }
    #pragma unroll
    for (int i = 0; i < NL; ++i) {
      const int r = (i * 256 + tid) / CPR;
      dst[i] = __ldcg(reinterpret_cast<const uint4*>(wrow(r) + (long)t * RB + cc * 16));
      const int g = (t * KS + cc * CV) / GROUP;
      if constexpr (MX) {
        sc[i] = static_cast<uint32_t>(__ldcg(reinterpret_cast<const unsigned char*>(srow(r) + g)));
      } else {
        sc[i] = static_cast<uint32_t>(__ldcg(reinterpret_cast<const unsigned short*>(srow(r) + g))) |
            (static_cast<uint32_t>(__ldcg(reinterpret_cast<const unsigned short*>(brow(r) + g))) << 16);
      }
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
        if constexpr (BITS == 3) {
          const int r = tid >> 1, part = tid & 1;
          const uint4 lo = braw[j][0], hi = braw[j][1];
          const unsigned long long w[3] = {
              lo.x | (static_cast<unsigned long long>(lo.y) << 32),
              lo.z | (static_cast<unsigned long long>(lo.w) << 32),
              hi.x | (static_cast<unsigned long long>(hi.y) << 32)};
          const unsigned short sbits = bsc[j][0] & 0xffff, bbits = bsc[j][0] >> 16;
          const float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&sbits));
          const float bias = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bbits));
          uint32_t out[32];
          #pragma unroll
          for (int i = 0; i < 64; i += 2)
            out[i / 2] = msv_affine(msv_code3(w, i), scale, bias) | (msv_affine(msv_code3(w, i + 1), scale, bias) << 16);
          #pragma unroll
          for (int q = 0; q < 8; ++q) {
            const int c = part * 8 + q;
            *reinterpret_cast<uint4*>(sw + r * 256 + ((c ^ (r & 7)) << 4)) = make_uint4(out[4 * q], out[4 * q + 1], out[4 * q + 2], out[4 * q + 3]);
          }
        }
        #pragma unroll
        for (int i = 0; i < (BITS == 3 ? 0 : NL); ++i) {
          const int r = (i * 256 + tid) / CPR;
          const uint4 v = braw[j][i];
          const uint32_t vw[4] = {v.x, v.y, v.z, v.w};
          const unsigned short sbits = bsc[j][i] & 0xffff, bbits = bsc[j][i] >> 16;
          const float scale = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&sbits));
          const float bias = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bbits));
          uint32_t out[CV / 2];
          if constexpr (BITS == 4) {
            if constexpr (MX) {
              const uint32_t e8 = bsc[j][i] & 0xffu;
              // 2**(e8 - 127); e8 = 0 is the subnormal 2**-127.
              const float mx_scale = __uint_as_float(e8 ? e8 << 23 : 0x00400000u);
              #pragma unroll
              for (int w4 = 0; w4 < 4; ++w4)
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                  const uint32_t pair = (vw[w4] >> (8 * q)) & 0xff;
                  const __nv_bfloat162 v = __floats2bfloat162_rn(msv_mx4(pair & 15, mx_scale), msv_mx4(pair >> 4, mx_scale));
                  out[4 * w4 + q] = *reinterpret_cast<const uint32_t*>(&v);
                }
            } else {
              // The 16 values a code of this group can take.
              uint32_t tab[16];
              #pragma unroll
              for (int q = 0; q < 16; ++q) tab[q] = msv_affine(q, scale, bias);
              #pragma unroll
              for (int w4 = 0; w4 < 4; ++w4)
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                  const uint32_t pair = (vw[w4] >> (8 * q)) & 0xff;
                  out[4 * w4 + q] = tab[pair & 15] | (tab[pair >> 4] << 16);
                }
            }
          } else {
            #pragma unroll
            for (int w4 = 0; w4 < 4; ++w4)
              #pragma unroll
              for (int q = 0; q < 2; ++q)
                out[2 * w4 + q] = msv_affine((vw[w4] >> (16 * q)) & 0xff, scale, bias) |
                    (msv_affine((vw[w4] >> (16 * q + 8)) & 0xff, scale, bias) << 16);
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
// MX (sm_120a / sm_121a): the same tile on the block-scaled mma. A rows are
// E4M3 codes with one E8M0 scale per 32, the activation's FP8 round trip kept
// as codes; B rows are the mxfp4 codes, each widened to one byte with the
// E2M1 code in bits 5:2 as the mma reads it. The products are exact as in the
// bf16 tile, and each k32 step starts from zero and is added in order.
__device__ __forceinline__ void msv_cp4(uint32_t dst, const void* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(dst), "l"(src), "r"(valid ? 4 : 0));
}
__device__ __forceinline__ void msv_mma_mx(float* c, const uint32_t* a, uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
// CUDA 13 names an sm_12xa target by __CUDA_ARCH_SPECIFIC__, CUDA 12 by the FEAT macros.
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) || \
    (defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ / 100 == 12) || \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ / 100 == 12)
  const float zero = 0.f;
  const uint16_t z = 0;
  float p[4];
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, {%11}, {%12,%13}, {%14}, {%15,%16};\n"
      : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(zero),
        "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
  c[0] = __fadd_rn(c[0], p[0]);
  c[1] = __fadd_rn(c[1], p[1]);
  c[2] = __fadd_rn(c[2], p[2]);
  c[3] = __fadd_rn(c[3], p[3]);
#else
  __trap();
#endif
}
// The 8 E2M1 codes of w (code i in bits 4i..4i+3) as 8 bytes, code i in bits 5:2 of byte i.
__device__ __forceinline__ uint2 msv_fp4_bytes(uint32_t w) {
  const uint32_t h = w >> 16;
  const uint32_t lo = (w & 0xFu) | ((w & 0xF0u) << 4) | ((w & 0xF00u) << 8) | ((w & 0xF000u) << 12);
  const uint32_t hi = (h & 0xFu) | ((h & 0xF0u) << 4) | ((h & 0xF00u) << 8) | ((h & 0xF000u) << 12);
  return make_uint2(lo << 2, hi << 2);
}
// msv_expert_tile with MX: A codes from `acode(row)` and scales from
// `ascale(row)`, B mxfp4 codes from `wrow(r)` and scales from `srow(r)`. Smem
// holds ST stages of A codes and scales, then the B bytes and scales of one stage.
// One pass covers MT tiles of 64 rows, so each B stage is read once for all of them.
template <int K, int BD, int ST, int MT, typename ACode, typename AScale, typename WRow, typename SRow>
__device__ __forceinline__ void msv_expert_tile_mx(uint8_t* smem, int mv, ACode acode, AScale ascale, WRow wrow, SRow srow, float (&acc)[MT][2][4][4]) {
  constexpr int BM = 64, BN = 128, KS = 128, G = KS / 32;
  constexpr int KT = K / KS;
  constexpr int A_ST = MT * BM * KS, AS_ST = MT * BM * G;
  static_assert(MT * BM <= 256, "the A scales of a pass load one per thread");
  constexpr int RB = KS / 2, CPR = RB / 16, NL = BN * CPR / 256;
  uint8_t* sa = smem;
  uint8_t* sas = sa + ST * A_ST;
  uint8_t* sw = sas + ST * AS_ST;
  uint8_t* sws = sw + BN * KS;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp >> 2, wn = warp & 3;
  const int cc = tid % CPR;
  auto load_a = [&](int t) {
    const int st = t % ST;
    #pragma unroll
    for (int i = 0; i < 2 * MT; ++i) {
      const int q = tid + i * 256, row = q >> 3, c = q & 7;
      const bool ok = row < mv;
      msv_cp16(msv_smem(sa + st * A_ST + row * KS + ((c ^ (row & 7)) << 4)), acode(ok ? row : 0) + t * KS + c * 16, ok);
    }
    if (tid < MT * BM) {
      const bool ok = tid < mv;
      msv_cp4(msv_smem(sas + st * AS_ST + tid * G), ascale(ok ? tid : 0) + t * G, ok);
    }
  };
  // Load i: B row (i * 256 + tid) / CPR, 32 codes (one scale group) at chunk cc.
  uint4 braw[BD][NL];
  uint32_t bsc[BD][NL];
  auto load_b = [&](int t, uint4 (&dst)[NL], uint32_t (&sc)[NL]) {
    #pragma unroll
    for (int i = 0; i < NL; ++i) {
      const int r = (i * 256 + tid) / CPR;
      dst[i] = __ldcg(reinterpret_cast<const uint4*>(wrow(r) + (long)t * RB + cc * 16));
      sc[i] = static_cast<uint32_t>(__ldcg(reinterpret_cast<const unsigned char*>(srow(r) + t * G + cc)));
    }
  };
  #pragma unroll
  for (int m = 0; m < MT; ++m)
    #pragma unroll
    for (int a = 0; a < 2; ++a)
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int v = 0; v < 4; ++v) acc[m][a][j][v] = 0.f;
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
        #pragma unroll
        for (int i = 0; i < NL; ++i) {
          const int r = (i * 256 + tid) / CPR;
          const uint4 v = braw[j][i];
          const uint2 b0 = msv_fp4_bytes(v.x), b1 = msv_fp4_bytes(v.y);
          const uint2 b2 = msv_fp4_bytes(v.z), b3 = msv_fp4_bytes(v.w);
          const int c = 2 * cc;
          *reinterpret_cast<uint4*>(sw + r * KS + ((c ^ (r & 7)) << 4)) = make_uint4(b0.x, b0.y, b1.x, b1.y);
          *reinterpret_cast<uint4*>(sw + r * KS + (((c + 1) ^ (r & 7)) << 4)) = make_uint4(b2.x, b2.y, b3.x, b3.y);
          sws[r * G + cc] = static_cast<uint8_t>(bsc[j][i]);
        }
        if (t + BD < KT) load_b(t + BD, braw[j], bsc[j]);
        __syncthreads();
        if (wm * 32 < mv) {
          const uint8_t* at = sa + (t % ST) * A_ST;
          const uint8_t* ast = sas + (t % ST) * AS_ST;
          #pragma unroll
          for (int kb = 0; kb < G; ++kb) {
            uint32_t bf[4][2], sfb[4];
            #pragma unroll
            for (int p = 0; p < 2; ++p) {
              const int idx = lane >> 3;
              const int n = wn * 32 + p * 16 + (idx >> 1) * 8 + (lane & 7);
              const int c = kb * 2 + (idx & 1);
              msv_ldsm4(msv_smem(sw + n * KS + ((c ^ (n & 7)) << 4)), bf[2 * p][0], bf[2 * p][1], bf[2 * p + 1][0], bf[2 * p + 1][1]);
            }
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) sfb[jj] = sws[(wn * 32 + jj * 8 + (lane >> 2)) * G + kb];
            #pragma unroll
            for (int m = 0; m < MT; ++m) {
              const int r0 = m * BM + wm * 32;
              if (r0 < mv) {
                uint32_t af[2][4], sfa[2];
                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                  const int idx = lane >> 3;
                  const int row = r0 + a * 16 + (idx & 1) * 8 + (lane & 7);
                  const int c = kb * 2 + (idx >> 1);
                  msv_ldsm4(msv_smem(at + row * KS + ((c ^ (row & 7)) << 4)), af[a][0], af[a][1], af[a][2], af[a][3]);
                  // Lane l supplies the scale of row 8 * (l % 2) + l / 4 of the 16.
                  sfa[a] = ast[(r0 + a * 16 + 8 * (lane & 1) + (lane >> 2)) * G + kb];
                }
                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                  if (r0 + a * 16 < mv) {
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) msv_mma_mx(acc[m][a][jj], af[a], bf[jj][0], bf[jj][1], sfa[a], sfb[jj]);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();
}
