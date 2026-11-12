# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for the hot paths of DeepSeek V4.1 prefill.

A kernel runs only on CUDA after it reproduced its reference in ops.py or
quant.py (MLX's gather_qmm for the expert tiles) on a small input. Kernels
keep the host's Metal order of fp32 ops (fused multiply-adds, short sums) where
they can, so they may differ from an op chain in the last fp32 bit.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import ops, quant

logger = logging.getLogger(__name__)

_HEADER = r"""
// Values in the activation dtype T (float or __nv_bfloat16) as fp32.
inline float v41_float(float x) { return x; }
inline float v41_float(__nv_bfloat16 x) { return __bfloat162float(x); }
// fp32 rounded to the activation dtype.
inline float v41_round(float x, float) { return x; }
inline float v41_round(float x, __nv_bfloat16) {
  return __bfloat162float(__float2bfloat16(x));
}
inline void v41_store(float v, float* out) { *out = v; }
inline void v41_store(float v, __nv_bfloat16* out) { *out = __float2bfloat16(v); }
// Activation round trip of the 32 values of one group, scale 2**ceil(log2(amax / limit)).
// BITS 8: E4M3 values, ties to even. BITS 4: E2M1 values, ties to the even code.
template <int BITS, unsigned FLOOR, unsigned INV_LIMIT>
inline void v41_quantize(float* v) {
  const float limit = BITS == 8 ? 448.0f : 6.0f;
  float amax = __uint_as_float(FLOOR);
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const unsigned pb = __float_as_uint(__fmul_rn(amax, __uint_as_float(INV_LIMIT)));
  const int e = int((pb >> 23) & 0xFFu) - 127 + int((pb & 0x7FFFFFu) != 0u);
  const float scale = __uint_as_float(unsigned(e + 127) << 23);
  const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int i = 0; i < 32; ++i) {
    const float scaled = fminf(fmaxf(v[i] / scale, -limit), limit);
    const float a = fabsf(scaled);
    float q;
    if (BITS == 8) {
      // The grid step comes from the exponent bits; below 2**-6 it is 2**-9.
      int se = int((__float_as_uint(a) >> 23) & 0xFFu) - 127 - 3;
      if (se < -9) se = -9;
      const float step = __uint_as_float(unsigned(se + 127) << 23);
      q = fminf(rintf(a / step) * step, 448.0f);
    } else {
      q = levels[int(a > 0.25f) + int(a >= 0.75f) + int(a > 1.25f) +
                 int(a >= 1.75f) + int(a > 2.5f) + int(a >= 3.5f) + int(a > 5.0f)];
    }
    v[i] = (scaled < 0.0f ? -q : q) * scale;
  }
}
// v41_quantize<8>'s round trip of 32 values kept as E4M3 codes (the MX tile
// kernels' activations); returns the E8M0 byte of the group's scale.
template <unsigned FLOOR, unsigned INV_LIMIT>
inline unsigned v41_quantize_codes(const float* v, unsigned char* codes) {
  float amax = __uint_as_float(FLOOR);
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  const unsigned pb = __float_as_uint(__fmul_rn(amax, __uint_as_float(INV_LIMIT)));
  const int e = int((pb >> 23) & 0xFFu) - 127 + int((pb & 0x7FFFFFu) != 0u);
  const float scale = __uint_as_float(unsigned(e + 127) << 23);
  for (int i = 0; i < 32; ++i) {
    const float scaled = fminf(fmaxf(v[i] / scale, -448.0f), 448.0f);
    const float a = fabsf(scaled);
    int se = int((__float_as_uint(a) >> 23) & 0xFFu) - 127 - 3;
    if (se < -9) se = -9;
    const float step = __uint_as_float(unsigned(se + 127) << 23);
    const float q = fminf(rintf(a / step) * step, 448.0f);
    // E4M3: exponent bias 7 and three mantissa bits; below 2**-6, steps of 2**-9.
    const unsigned qb = __float_as_uint(q), qe = (qb >> 23) & 0xFFu;
    const unsigned mag = q == 0.0f ? 0u
        : qe < 121u ? unsigned(q * 512.0f) : ((qe - 120u) << 3) | ((qb >> 20) & 7u);
    codes[i] = (unsigned char)(mag | (scaled < 0.0f ? 0x80u : 0u));
  }
  return unsigned(e + 127);
}
// Eight consecutive values as fp32; p is 8-byte aligned.
inline void v41_load8(const float* p, float* v) {
  for (int i = 0; i < 8; ++i) v[i] = p[i];
}
inline void v41_load8(const __nv_bfloat16* p, float* v) {
  const unsigned long long* w = reinterpret_cast<const unsigned long long*>(p);
  for (int i = 0; i < 8; ++i)
    v[i] = __uint_as_float(unsigned((w[i / 4] >> (16 * (i % 4))) & 0xFFFFu) << 16);
}
// 32 consecutive values as fp32 and back, in 16-byte accesses; p is 16-byte
// aligned (a multiple of 32 values from the start of an array).
inline void v41_load32(const float* p, float* v) {
  const float4* q = reinterpret_cast<const float4*>(p);
  for (int i = 0; i < 8; ++i) {
    const float4 f = q[i];
    v[4 * i] = f.x;
    v[4 * i + 1] = f.y;
    v[4 * i + 2] = f.z;
    v[4 * i + 3] = f.w;
  }
}
inline void v41_load32(const __nv_bfloat16* p, float* v) {
  const uint4* q = reinterpret_cast<const uint4*>(p);
  for (int i = 0; i < 4; ++i) {
    const uint4 u = q[i];
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
    for (int k = 0; k < 4; ++k) {
      v[8 * i + 2 * k] = __uint_as_float(w[k] << 16);
      v[8 * i + 2 * k + 1] = __uint_as_float(w[k] & 0xFFFF0000u);
    }
  }
}
inline void v41_store32(const float* v, float* p) {
  float4* q = reinterpret_cast<float4*>(p);
  for (int i = 0; i < 8; ++i) q[i] = make_float4(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
}
inline void v41_store32(const float* v, __nv_bfloat16* p) {
  uint4* q = reinterpret_cast<uint4*>(p);
  for (int i = 0; i < 4; ++i) {
    unsigned w[4];
    for (int k = 0; k < 4; ++k) {
      const __nv_bfloat162 pair = __floats2bfloat162_rn(v[8 * i + 2 * k], v[8 * i + 2 * k + 1]);
      w[k] = *reinterpret_cast<const unsigned*>(&pair);
    }
    q[i] = make_uint4(w[0], w[1], w[2], w[3]);
  }
}
// Order-preserving key: larger scores give larger keys, and -0 equals +0.
inline unsigned v41_score_key(float x) {
  const unsigned u = __float_as_uint(x == 0.0f ? 0.0f : x);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
__device__ __forceinline__ unsigned v41_smem(const void* p) {
  return unsigned(__cvta_generic_to_shared(p));
}
// 16 bytes from global to shared memory; zeros when !valid.
__device__ __forceinline__ void v41_cp16(unsigned dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void v41_ldsm4(unsigned addr, unsigned* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void v41_ldsm4t(unsigned addr, unsigned* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// c += a b on bf16 tensor cores, m16n8k16, float32 sums.
__device__ __forceinline__ void v41_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned v41_pack(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const unsigned*>(&v);
}
"""

# Index scores of dots [N, B, H] (one key per row): the weighted ReLU of each
# head and the head sum are rounded to T; hidden keys score -inf.
_INDEX_SCORES = r"""
  const int N = meta[0], B = meta[1], H = meta[2], start = meta[3], ratio = meta[4];
  const size_t elem = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (elem >= size_t(N) * B) return;
  const int n = int(elem / B), q = int(elem % B);
  float result = __uint_as_float(0xff800000u);
  if (n < (start + q + 1) / ratio) {
    const T* d = dots + elem * H;
    const float* w = weights + size_t(q) * H;
    float total = 0.0f;
    for (int h = 0; h < H; ++h) {
      const float s = fmaxf(v41_float(d[h]), 0.0f);
      total += v41_round(s * w[h], T());
    }
    result = v41_round(total, T());
  }
  scores[size_t(q) * N + n] = result;
"""

# ops.index_scores on bf16 tensor cores. A block of 8 warps stages 64 keys at
# a time in shared memory; warp w scores query 8 b + w, its H heads as A
# fragments in registers. Each head's dot product rounds to T, its ReLU times
# the head weight rounds to T, and the head sum (in another order than the
# reference) rounds to T. Keys past the last one a query sees score -inf; a
# tile no query of the block sees is not multiplied.
_INDEX_TC = r"""
  constexpr int KT = 64, CH = D / 8, QB = 8, HM = H / 16;
  __shared__ __align__(128) unsigned char ks[KT * D * 2];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int B = meta[0], N = meta[1], start = meta[2], ratio = meta[3];
  const int q = blockIdx.x * QB + w;
  const bool live = q < B;
  const int qq = live ? q : B - 1;
  const int seen = (start + qq + 1) / ratio;
  const int last = blockIdx.x * QB + QB < B ? blockIdx.x * QB + QB : B;
  const int block_seen = (start + last) / ratio;
  const float hidden = __uint_as_float(0xff800000u);
  unsigned qa[HM][D / 16][4];
  const __nv_bfloat16* qrow = queries + size_t(qq) * H * D;
  #pragma unroll
  for (int m = 0; m < HM; ++m)
    #pragma unroll
    for (int s = 0; s < D / 16; ++s) {
      const int r0 = 16 * m + g, k = 16 * s + 2 * t;
      qa[m][s][0] = *reinterpret_cast<const unsigned*>(qrow + r0 * D + k);
      qa[m][s][1] = *reinterpret_cast<const unsigned*>(qrow + (r0 + 8) * D + k);
      qa[m][s][2] = *reinterpret_cast<const unsigned*>(qrow + r0 * D + k + 8);
      qa[m][s][3] = *reinterpret_cast<const unsigned*>(qrow + (r0 + 8) * D + k + 8);
    }
  float hw[HM][2];
  #pragma unroll
  for (int m = 0; m < HM; ++m) {
    hw[m][0] = weights[size_t(qq) * H + 16 * m + g];
    hw[m][1] = weights[size_t(qq) * H + 16 * m + g + 8];
  }
  float* out = scores + size_t(qq) * N;
  for (int k0 = 0; k0 < N; k0 += KT) {
    if (k0 >= block_seen) {
      for (int i = lane; i < KT; i += 32)
        if (live && k0 + i < N) out[k0 + i] = hidden;
      continue;
    }
    __syncthreads();
    for (int c = tid; c < KT * CH; c += 256) {
      const int r = c / CH, k = c % CH, n = k0 + r;
      v41_cp16(v41_smem(ks + r * D * 2 + ((k ^ (r & 7)) << 4)),
               keys + size_t(n < N ? n : 0) * D + k * 8, n < N);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < KT / 8; ++j) {
      float c[HM][4];
      #pragma unroll
      for (int m = 0; m < HM; ++m) c[m][0] = c[m][1] = c[m][2] = c[m][3] = 0.0f;
      #pragma unroll
      for (int s = 0; s < D / 16; ++s) {
        unsigned kb[4];
        const int kr = 8 * j + (lane & 7), kk = 2 * s + ((lane >> 3) & 1);
        v41_ldsm4(v41_smem(ks + kr * D * 2 + ((kk ^ (kr & 7)) << 4)), kb);
        #pragma unroll
        for (int m = 0; m < HM; ++m) v41_mma(c[m], qa[m][s], kb[0], kb[1]);
      }
      float s0 = 0.0f, s1 = 0.0f;
      #pragma unroll
      for (int m = 0; m < HM; ++m) {
        s0 += v41_round(fmaxf(v41_round(c[m][0], T()), 0.0f) * hw[m][0], T());
        s0 += v41_round(fmaxf(v41_round(c[m][2], T()), 0.0f) * hw[m][1], T());
        s1 += v41_round(fmaxf(v41_round(c[m][1], T()), 0.0f) * hw[m][0], T());
        s1 += v41_round(fmaxf(v41_round(c[m][3], T()), 0.0f) * hw[m][1], T());
      }
      #pragma unroll
      for (int off = 4; off < 32; off <<= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, off);
        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
      }
      const int n = k0 + 8 * j + 2 * t;
      if (live && g == 0) {
        if (n < N) out[n] = n < seen ? v41_round(s0, T()) : hidden;
        if (n + 1 < N) out[n + 1] = n + 1 < seen ? v41_round(s1, T()) : hidden;
      }
    }
  }
"""

# Exact top K of each score row as ops.select_topk returns it, one block per
# row. Histogram passes over the key bits find the K-th largest key.
_INDEX_SELECT = r"""
  constexpr unsigned BINS = 4096, TPB = 256;
  const unsigned tid = threadIdx.x, row = blockIdx.x;
  const unsigned n = meta[0], K = meta[1];
  const float* s = scores + size_t(row) * n;
  int* out = ids + size_t(row) * K;
  __shared__ unsigned hist[BINS];
  __shared__ unsigned part[TPB], part_equal[TPB];
  __shared__ unsigned found_bin, found_above;
  unsigned prefix = 0, above = 0;
  for (unsigned p = 0; p < 3; ++p) {
    const unsigned shift = p == 0 ? 20u : p == 1 ? 8u : 0u;
    const unsigned bits = p == 2 ? 8u : 12u, nb = 1u << bits, per = nb / TPB;
    for (unsigned b = tid; b < nb; b += TPB) hist[b] = 0u;
    __syncthreads();
    for (unsigned i = tid; i < n; i += TPB) {
      const unsigned key = v41_score_key(s[i]);
      if (p == 0 || (key >> (shift + bits)) == prefix)
        atomicAdd(&hist[(key >> shift) & (nb - 1u)], 1u);
    }
    __syncthreads();
    // Thread t owns the bins nb - 1 - t * per - j, counted down from the top.
    unsigned local = 0;
    for (unsigned j = 0; j < per; ++j) local += hist[nb - 1u - (tid * per + j)];
    part[tid] = local;
    __syncthreads();
    if (tid == 0) {
      const unsigned want = K - above;
      unsigned acc = 0, t = 0;
      for (; t + 1 < TPB; ++t) {
        if (acc + part[t] >= want) break;
        acc += part[t];
      }
      for (unsigned j = 0; j < per; ++j) {
        const unsigned bin = nb - 1u - (t * per + j);
        const unsigned c = hist[bin];
        if (acc + c >= want || j + 1 == per) {
          found_bin = bin;
          found_above = acc;
          break;
        }
        acc += c;
      }
    }
    __syncthreads();
    prefix = (prefix << bits) | found_bin;
    above += found_above;
    __syncthreads();
  }
  // prefix is the K-th largest key; take K - above ties at it by position.
  const unsigned kinf = v41_score_key(__uint_as_float(0xff800000u));
  const unsigned ties = prefix == kinf ? 0u : K - above;
  const unsigned lead = prefix == kinf ? K - above : 0u;
  const unsigned first = unsigned((size_t(n) * tid) / TPB);
  const unsigned last = unsigned((size_t(n) * (tid + 1)) / TPB);
  unsigned greater = 0, equal = 0;
  for (unsigned i = first; i < last; ++i) {
    const unsigned key = v41_score_key(s[i]);
    greater += key > prefix;
    equal += key == prefix;
  }
  part[tid] = greater;
  part_equal[tid] = equal;
  __syncthreads();
  if (tid == 0) {
    unsigned g = 0, e = 0;
    for (unsigned t = 0; t < TPB; ++t) {
      const unsigned a = part[t], b = part_equal[t];
      part[t] = g;
      part_equal[t] = e;
      g += a;
      e += b;
    }
  }
  __syncthreads();
  for (unsigned i = tid; i < lead; i += TPB) out[i] = -1;
  unsigned rank = part_equal[tid];
  unsigned o = lead + part[tid] + (rank < ties ? rank : ties);
  for (unsigned i = first; i < last; ++i) {
    const unsigned key = v41_score_key(s[i]);
    const bool take = key > prefix || (key == prefix && rank < ties);
    rank += key == prefix;
    if (take) out[o++] = int(i);
  }
"""

# One thread per 32 values: RoPE on the last RD values, then the activation
# round trip. One product of each rotation is fused, as in the host's kernel.
_TRANSFORM = r"""
  const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (group >= meta[0]) return;
  const size_t base = group * 32;
  const int d0 = int(base % D);
  float v[32];
  v41_load32(x + base, v);
  if (RD > 0 && d0 + 32 > D - RD) {
    const size_t position = (base / D) / meta[1];
    const float* c = cosv + position * (RD / 2);
    const float* s = sinv + position * (RD / 2);
    for (int i = 0; i < 32; i += 2) {
      const int d = d0 + i - (D - RD);
      if (d < 0) continue;
      const float cs = c[d / 2], sn = INVERSE ? -s[d / 2] : s[d / 2];
      const float a = v[i], b = v[i + 1];
      v[i] = v41_round(fmaf(a, cs, -__fmul_rn(b, sn)), T());
      v[i + 1] = v41_round(fmaf(a, sn, __fmul_rn(b, cs)), T());
    }
  }
  if (BITS) v41_quantize<BITS ? BITS : 8, FLOOR, INV_LIMIT>(v);
  v41_store32(v, y + base);
"""

# The FP8 round trip of x [L, D] as E4M3 codes [L, D] and E8M0 scales [L, D / 32].
_FP8_CODES = r"""
  const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (group >= size_t(x_shape[0]) * x_shape[1] / 32) return;
  float v[32];
  v41_load32(x + group * 32, v);
  scales[group] = (unsigned char)v41_quantize_codes<FLOOR, INV_LIMIT>(v, codes + group * 32);
"""

# One thread per group of 32 values: silu(gate) * up with the reference
# clamps, times the route weight of the row, rounded to T, then FP8.
_SWIGLU = r"""
  const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (group >= meta[0]) return;
  const size_t base = group * 32;
  const float limit = __uint_as_float(LIMIT);
  const float weight = WEIGHTED ? weights[group / meta[1]] : 1.0f;
  float v[32];
  for (int i = 0; i < 32; ++i) {
    float g = v41_float(gate[base + i]), u = v41_float(up[base + i]);
    if (LIMIT) {
      g = fminf(g, limit);
      u = fminf(fmaxf(u, -limit), limit);
    }
    const float e = 1.0f / (1.0f + expf(fabsf(g)));
    float value = __fmul_rn(__fmul_rn(g, g < 0.0f ? e : 1.0f - e), u);
    if (WEIGHTED) value = __fmul_rn(value, weight);
    v[i] = v41_round(value, T());
  }
  if (BITS) v41_quantize<8, FLOOR, INV_LIMIT>(v);
  for (int i = 0; i < 32; ++i) v41_store(v[i], y + base + i);
"""

# One thread per (position, d). Streams accumulate by fused multiply-add from
# zero and the post product is added unfused, as in the host's Metal kernel.
_HC_POST = r"""
  const size_t z = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (z >= size_t(meta[0]) * D) return;
  const size_t row = z / D, d = z % D;
  float r[N];
  for (int i = 0; i < N; ++i) r[i] = v41_float(residual[(row * N + i) * D + d]);
  const float value = v41_float(x[z]);
  for (int j = 0; j < N; ++j) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) sum = fmaf(comb[(row * N + i) * N + j], r[i], sum);
    sum = __fadd_rn(__fmul_rn(post[row * N + j], value), sum);
    v41_store(sum, y + (row * N + j) * D + d);
  }
"""

# One block of 256 threads per position: the N streams mixed by pre (each
# product, then the sum, unfused), rounded to T, then the RMS norm, as the
# host's Metal kernel orders them.
_HC_PRE_NORM = r"""
  constexpr int PER = (D + 255) / 256;
  __shared__ float sums[8];
  const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* p = pre + size_t(row) * N;
  float values[PER];
  float total = 0.0f;
  #pragma unroll
  for (int t = 0; t < PER; ++t) {
    const int d = tid + 256 * t;
    float value = 0.0f;
    if (d < D) {
      #pragma unroll
      for (int i = 0; i < N; ++i)
        value = __fadd_rn(__fmul_rn(v41_float(x[(size_t(row) * N + i) * D + d]), p[i]), value);
      value = v41_round(value, T());
    }
    values[t] = value;
    total = __fadd_rn(total, __fmul_rn(value, value));
  }
  for (int off = 16; off; off >>= 1) total += __shfl_xor_sync(0xffffffffu, total, off);
  if (lane == 0) sums[warp] = total;
  __syncthreads();
  if (warp == 0) {
    float sum = lane < 8 ? sums[lane] : 0.0f;
    for (int off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (lane == 0) sums[0] = rsqrtf(sum / float(D) + eps[0]);
  }
  __syncthreads();
  const float r = sums[0];
  #pragma unroll
  for (int t = 0; t < PER; ++t) {
    const int d = tid + 256 * t;
    if (d < D) v41_store(__fmul_rn(__fmul_rn(values[t], r), v41_float(weight[d])), y + size_t(row) * D + d);
  }
"""

# Mix projections, one position per thread and slice of the width: its M dot
# products with fn_t [K, M] and its sum of squares over the slice, float32
# sums in k order. A block stages 64 positions x 64 k of the streams and fn_t.
_HC_PROJECTION = r"""
  constexpr int R = 64, KT = 64;
  __shared__ float xs[R][KT + 1];
  __shared__ __align__(16) float fs[KT][M];
  const int tid = threadIdx.x, row0 = blockIdx.x * R, rows = meta[0];
  const int part = blockIdx.y, k_begin = part * (K / SPLIT), k_end = k_begin + K / SPLIT;
  float acc[M], square = 0.0f;
  #pragma unroll
  for (int j = 0; j < M; ++j) acc[j] = 0.0f;
  for (int k0 = k_begin; k0 < k_end; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < R * KT; i += R) {
      const int r = i / KT, k = i % KT;
      xs[r][k] = row0 + r < rows ? v41_float(x[size_t(row0 + r) * K + k0 + k]) : 0.0f;
    }
    for (int i = tid; i < KT * M; i += R) fs[i / M][i % M] = fn[size_t(k0) * M + i];
    __syncthreads();
    for (int k = 0; k < KT; ++k) {
      const float v = xs[tid][k];
      square = fmaf(v, v, square);
      #pragma unroll
      for (int j = 0; j < M; ++j) acc[j] = fmaf(v, fs[k][j], acc[j]);
    }
  }
  if (row0 + tid < rows) {
    float* out = parts + (size_t(part) * rows + row0 + tid) * (M + 1);
    #pragma unroll
    for (int j = 0; j < M; ++j) out[j] = acc[j];
    out[M] = square;
  }
"""

# Hyper-connection weights of one position per thread, in float32: its mixes
# from the SPLIT partial sums (added in order) scaled by the RMS, then pre,
# post and comb, softmax rows then Sinkhorn sweeps (columns first, then rows
# and columns in turn), as ops.hc_mixes.
_HC_WEIGHTS = r"""
  constexpr int M = N * (N + 2);
  const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x, rows = meta[0];
  if (row >= rows) return;
  float m[M + 1];
  for (int j = 0; j <= M; ++j) m[j] = parts[row * (M + 1) + j];
  for (int p = 1; p < SPLIT; ++p)
    for (int j = 0; j <= M; ++j) m[j] += parts[(p * rows + row) * (M + 1) + j];
  const float rms = rsqrtf(m[M] / float(K) + eps[1]);
  for (int j = 0; j < M; ++j) m[j] *= rms;
  const float e = eps[0];
  for (int i = 0; i < N; ++i) {
    pre[row * N + i] = 1.0f / (1.0f + expf(-(m[i] * scale[0] + base[i]))) + e;
    post[row * N + i] = 2.0f / (1.0f + expf(-(m[N + i] * scale[1] + base[N + i])));
  }
  float c[N * N];
  for (int r = 0; r < N; ++r) {
    float top = __uint_as_float(0xff800000u), sum = 0.0f;
    for (int k = 0; k < N; ++k) {
      c[r * N + k] = m[2 * N + r * N + k] * scale[2] + base[2 * N + r * N + k];
      top = fmaxf(top, c[r * N + k]);
    }
    for (int k = 0; k < N; ++k) {
      c[r * N + k] = expf(c[r * N + k] - top);
      sum += c[r * N + k];
    }
    for (int k = 0; k < N; ++k) c[r * N + k] = c[r * N + k] / sum + e;
  }
  for (int it = 0; it < ITERS; ++it) {
    if (it) {
      for (int r = 0; r < N; ++r) {
        float total = 0.0f;
        for (int k = 0; k < N; ++k) total += c[r * N + k];
        total += e;
        for (int k = 0; k < N; ++k) c[r * N + k] /= total;
      }
    }
    for (int k = 0; k < N; ++k) {
      float total = 0.0f;
      for (int r = 0; r < N; ++r) total += c[r * N + k];
      total += e;
      for (int r = 0; r < N; ++r) c[r * N + k] /= total;
    }
  }
  for (int i = 0; i < N * N; ++i) comb[row * N * N + i] = c[i];
"""

# One thread per (token, d): the K routed rows of a token, found through the
# inverse of the expert sort, summed in route order, plus the shared expert.
_COMBINE = r"""
  const size_t z = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (z >= size_t(meta[0]) * D) return;
  const size_t row = z / D, d = z % D;
  float sum = 0.0f;
  for (int j = 0; j < K; ++j)
    sum += v41_float(routed[size_t(inverse[row * K + j]) * D + d]);
  v41_store(sum + v41_float(shared[z]), y + z);
"""

# Rows of a value table as fp32 keys [B, S, D], one thread per 8 values; a
# negative row id gives zeros. bfloat16 widens by a shift of its bits.
_ATTENTION_KEYS = r"""
  constexpr int G = D / 8;
  const size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t slot = t / G;
  if (slot >= size_t(meta[0]) * S) return;
  const int row = ids[slot];
  float v[8];
  if (row < 0) {
    for (int i = 0; i < 8; ++i) v[i] = 0.0f;
  } else {
    v41_load8(table + size_t(row) * D + (t % G) * 8, v);
  }
  float4* out = reinterpret_cast<float4*>(keys + t * 8);
  out[0] = make_float4(v[0], v[1], v[2], v[3]);
  out[1] = make_float4(v[4], v[5], v[6], v[7]);
"""

# Weights of one (query, head) per thread, in the tiles of ops._attend; the
# denominator sums the unrounded weights and the sink.
_ATTENTION_WEIGHTS = r"""
  const size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= size_t(meta[0]) * H) return;
  const int* row = ids + (t / H) * S;
  const float* raw = dots + t * S;
  float* out = weights + t * S;
  constexpr int TILES = (S + 63) / 64;
  const float hidden = __uint_as_float(0xff800000u);
  float maxima[TILES];
  float running = -1e30f;
  for (int tile = 0; tile < TILES; ++tile) {
    const int end = tile * 64 + 64 < S ? tile * 64 + 64 : S;
    for (int s = tile * 64; s < end; ++s) {
      const float score = row[s] >= 0 ? raw[s] * scale[0] : hidden;
      out[s] = score;
      running = fmaxf(running, score);
    }
    maxima[tile] = running;
  }
  float total = 0.0f;
  for (int tile = 0; tile < TILES; ++tile) {
    const int end = tile * 64 + 64 < S ? tile * 64 + 64 : S;
    const float carry = expf(maxima[tile] - running);
    float sum = 0.0f;
    for (int s = tile * 64; s < end; ++s) {
      const float p = expf(out[s] - maxima[tile]);
      sum += p;
      out[s] = v41_round(p, T()) * carry;
    }
    total += sum * carry;
  }
  denominator[t] = total + expf(sink[t % H] - running);
"""

# ops.sparse_attention on bf16 tensor cores, one block of 8 warps per (query,
# 64 heads), so a key tile is fetched once per query. Rows of bf16 sit in
# shared memory with their 16-byte chunks XOR-swizzled by the row. Slots go in
# tiles of 64 as in ops._attend: a tile is scored whole, its weights use the
# running maximum through it and round to bf16 for the value product, and the
# denominator sums the unrounded weights. Warp w scores keys 16 (w % 4) on of a
# tile for heads 32 (w / 4) on and sums their values of dims D / 4 (w % 4) on.
# The queries stream in 64-dim groups through three buffers, each group staged
# with the same dims of the tile's keys.
_ATTENTION_TC = r"""
  constexpr int HB = 64, KT = 64, WD = D / 4, QD = 64, NQ = D / QD;
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* ks = smem;
  unsigned char* qs = ks + KT * D * 2;
  unsigned char* ps = qs + 3 * HB * QD * 2;
  float* part = reinterpret_cast<float*>(ps + HB * KT * 2);
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  const int kq = w & 3, hp = w >> 2;
  const int S = meta[0], groups = H / HB;
  const int row = blockIdx.x / groups, h0 = (blockIdx.x % groups) * HB;
  const int* slot = ids + size_t(row) * S;
  const float hidden = __uint_as_float(0xff800000u), sc_scale = scale[0];
  const __nv_bfloat16* qrow = q + (size_t(row) * H + h0) * D;
  // Thread tid stages rows tid / 8 and tid / 8 + 32 of each chunk, 16 bytes tid % 8.
  const int lr = tid >> 3, lk = tid & 7;
  float acc[2][WD / 8][4];
  #pragma unroll
  for (int m = 0; m < 2; ++m)
    #pragma unroll
    for (int n = 0; n < WD / 8; ++n)
      acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.0f;
  // Running maxima and this thread's parts of the denominators of heads
  // 16 (2 hp + m) + g and 16 (2 hp + m) + g + 8.
  float rmax[2][2] = {{-1e30f, -1e30f}, {-1e30f, -1e30f}};
  float ls[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
  for (int b = 0; b * KT < S; ++b) {
    const int sa = b * KT + lr, sb = sa + 32;
    const int ida = sa < S ? slot[sa] : -1, idb = sb < S ? slot[sb] : -1;
    // Group c: dims 64 c on of the 64 heads into Q buffer c % 3 and of the 64 keys.
    auto stage = [&](int c) {
      unsigned char* qb = qs + (c % 3) * HB * QD * 2;
      const int k = 8 * c + lk;
      v41_cp16(v41_smem(qb + lr * QD * 2 + ((lk ^ (lr & 7)) << 4)), qrow + lr * D + k * 8, true);
      v41_cp16(v41_smem(qb + (lr + 32) * QD * 2 + ((lk ^ (lr & 7)) << 4)),
               qrow + (lr + 32) * D + k * 8, true);
      v41_cp16(v41_smem(ks + lr * D * 2 + ((k ^ (lr & 7)) << 4)),
               table + size_t(ida < 0 ? 0 : ida) * D + k * 8, ida >= 0);
      v41_cp16(v41_smem(ks + (lr + 32) * D * 2 + ((k ^ (lr & 7)) << 4)),
               table + size_t(idb < 0 ? 0 : idb) * D + k * 8, idb >= 0);
      asm volatile("cp.async.commit_group;\n" ::);
    };
    stage(0);
    if (NQ > 1) stage(1);
    float sc[2][2][4];
    #pragma unroll
    for (int m = 0; m < 2; ++m)
      #pragma unroll
      for (int j = 0; j < 2; ++j) sc[m][j][0] = sc[m][j][1] = sc[m][j][2] = sc[m][j][3] = 0.0f;
    for (int c = 0; c < NQ; ++c) {
      if (c + 1 < NQ) {
        asm volatile("cp.async.wait_group 1;\n" ::);
      } else {
        asm volatile("cp.async.wait_group 0;\n" ::);
      }
      // Group c is visible, and every warp is done with Q buffer (c + 2) % 3.
      __syncthreads();
      if (c + 2 < NQ) stage(c + 2);
      const unsigned char* qb = qs + (c % 3) * HB * QD * 2;
      #pragma unroll
      for (int u = 0; u < QD / 16; ++u) {
        const int s = c * (QD / 16) + u;
        unsigned a[2][4], kb[4];
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int qr = 32 * hp + 16 * m + (lane & 15), qk = 2 * u + (lane >> 4);
          v41_ldsm4(v41_smem(qb + qr * QD * 2 + ((qk ^ (qr & 7)) << 4)), a[m]);
        }
        const int j = lane >> 3, kr = 16 * kq + (j >> 1) * 8 + (lane & 7), kk = 2 * s + (j & 1);
        v41_ldsm4(v41_smem(ks + kr * D * 2 + ((kk ^ (kr & 7)) << 4)), kb);
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
          v41_mma(sc[m][0], a[m], kb[0], kb[1]);
          v41_mma(sc[m][1], a[m], kb[2], kb[3]);
        }
      }
    }
    float top[2][2] = {{hidden, hidden}, {hidden, hidden}};
    #pragma unroll
    for (int j = 0; j < 2; ++j)
      #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int s = b * KT + 16 * kq + 8 * j + 2 * t + e;
        const bool ok = s < S && slot[s] >= 0;
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
          sc[m][j][e] = ok ? sc[m][j][e] * sc_scale : hidden;
          sc[m][j][2 + e] = ok ? sc[m][j][2 + e] * sc_scale : hidden;
          top[m][0] = fmaxf(top[m][0], sc[m][j][e]);
          top[m][1] = fmaxf(top[m][1], sc[m][j][2 + e]);
        }
      }
    #pragma unroll
    for (int m = 0; m < 2; ++m)
      #pragma unroll
      for (int off = 1; off < 4; off <<= 1) {
        top[m][0] = fmaxf(top[m][0], __shfl_xor_sync(0xffffffffu, top[m][0], off));
        top[m][1] = fmaxf(top[m][1], __shfl_xor_sync(0xffffffffu, top[m][1], off));
      }
    if (t == 0) {
      #pragma unroll
      for (int m = 0; m < 2; ++m) {
        part[kq * HB + 32 * hp + 16 * m + g] = top[m][0];
        part[kq * HB + 32 * hp + 16 * m + g + 8] = top[m][1];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int h = 32 * hp + 16 * m + g;
      const float n0 = fmaxf(rmax[m][0], fmaxf(fmaxf(part[h], part[HB + h]),
                                             fmaxf(part[2 * HB + h], part[3 * HB + h])));
      const float n1 = fmaxf(rmax[m][1], fmaxf(fmaxf(part[h + 8], part[HB + h + 8]),
                                             fmaxf(part[2 * HB + h + 8], part[3 * HB + h + 8])));
      const float f0 = expf(rmax[m][0] - n0), f1 = expf(rmax[m][1] - n1);
      rmax[m][0] = n0;
      rmax[m][1] = n1;
      float s0 = 0.0f, s1 = 0.0f;
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          sc[m][j][e] = expf(sc[m][j][e] - n0);
          sc[m][j][2 + e] = expf(sc[m][j][2 + e] - n1);
          s0 += sc[m][j][e];
          s1 += sc[m][j][2 + e];
        }
        const int chunk = 2 * kq + j;
        *reinterpret_cast<unsigned*>(ps + h * KT * 2 + ((chunk ^ (g & 7)) << 4) + 4 * t) =
            v41_pack(sc[m][j][0], sc[m][j][1]);
        *reinterpret_cast<unsigned*>(ps + (h + 8) * KT * 2 + ((chunk ^ (g & 7)) << 4) + 4 * t) =
            v41_pack(sc[m][j][2], sc[m][j][3]);
      }
      ls[m][0] = ls[m][0] * f0 + s0;
      ls[m][1] = ls[m][1] * f1 + s1;
      #pragma unroll
      for (int n = 0; n < WD / 8; ++n) {
        acc[m][n][0] *= f0;
        acc[m][n][1] *= f0;
        acc[m][n][2] *= f1;
        acc[m][n][3] *= f1;
      }
    }
    __syncthreads();
    #pragma unroll
    for (int s = 0; s < KT / 16; ++s) {
      unsigned a[2][4];
      #pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int pr = 32 * hp + 16 * m + (lane & 15), pk = 2 * s + (lane >> 4);
        v41_ldsm4(v41_smem(ps + pr * KT * 2 + ((pk ^ (pr & 7)) << 4)), a[m]);
      }
      #pragma unroll
      for (int p = 0; p < WD / 16; ++p) {
        unsigned vb[4];
        const int j = lane >> 3, vr = 16 * s + (j & 1) * 8 + (lane & 7);
        const int vk = WD / 8 * kq + 2 * p + (j >> 1);
        v41_ldsm4t(v41_smem(ks + vr * D * 2 + ((vk ^ (vr & 7)) << 4)), vb);
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
          v41_mma(acc[m][2 * p], a[m], vb[0], vb[1]);
          v41_mma(acc[m][2 * p + 1], a[m], vb[2], vb[3]);
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int m = 0; m < 2; ++m)
    #pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
      ls[m][0] += __shfl_xor_sync(0xffffffffu, ls[m][0], off);
      ls[m][1] += __shfl_xor_sync(0xffffffffu, ls[m][1], off);
    }
  if (t == 0) {
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
      part[kq * HB + 32 * hp + 16 * m + g] = ls[m][0];
      part[kq * HB + 32 * hp + 16 * m + g + 8] = ls[m][1];
    }
  }
  __syncthreads();
  #pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int h = 32 * hp + 16 * m + g;
    const float d0 = part[h] + part[HB + h] + part[2 * HB + h] + part[3 * HB + h] +
                     expf(sink[h0 + h] - rmax[m][0]);
    const float d1 = part[h + 8] + part[HB + h + 8] + part[2 * HB + h + 8] +
                     part[3 * HB + h + 8] + expf(sink[h0 + h + 8] - rmax[m][1]);
    __nv_bfloat16* out0 = o + (size_t(row) * H + h0 + h) * D + WD * kq + 2 * t;
    __nv_bfloat16* out1 = out0 + 8 * D;
    #pragma unroll
    for (int n = 0; n < WD / 8; ++n) {
      *reinterpret_cast<unsigned*>(out0 + 8 * n) = v41_pack(acc[m][n][0] / d0, acc[m][n][1] / d0);
      *reinterpret_cast<unsigned*>(out1 + 8 * n) = v41_pack(acc[m][n][2] / d1, acc[m][n][3] / d1);
    }
  }
"""

# Scores held at once by the index scan, in elements of dots.
_INDEX_DOTS = 1 << 28

# Expert tiles: 8 warps a block, two staged 64 x 128 row tiles and one
# 128 x 128 weight tile in bf16; at most _TILE_BUDGET blocks at once.
_TILE_THREADS = 256
_TILE_SHARED = 2 * 64 * 128 * 2 + 128 * 128 * 2
# The MX tiles stage bytes: two stages of _TILE_MX_PASS 64 x 128 code tiles,
# one 128 x 128 weight tile and their scales. A pass multiplies all its row
# tiles by each weight stage, so a weight slab is read once per pass; on GB10
# the slab reads bound the kernel. The gate/up epilogue reuses the memory for
# its bf16 rows x 128 tile.
_TILE_MX_PASS = 2
_TILE_STAGES, _TILE_PREFETCH = 2, 4
_TILE_MX_SHARED = _TILE_STAGES * _TILE_MX_PASS * 64 * (128 + 4) + 128 * (128 + 4)
_TILE_BUDGET = 16384
_BORROWED = Path(__file__).resolve().parents[2] / "third_party/mlx-serve/deepseek_v41"


def available() -> bool:
    cuda = getattr(mx, "cuda", None)
    return cuda is not None and cuda.is_available()


@cache
def _kernel(name: str, source: str, inputs: tuple[str, ...], outputs: tuple[str, ...]):
    return mx.fast.cuda_kernel(
        name="deepseek_v41_" + name,
        input_names=list(inputs),
        output_names=list(outputs),
        source=source,
        header=_HEADER,
    )


@cache
def _tile_kernel(
    name: str,
    body: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    shared: int = _TILE_SHARED,
):
    return mx.fast.cuda_kernel(
        name="deepseek_v41_" + name,
        input_names=list(inputs),
        output_names=list(outputs),
        source=(_BORROWED / body).read_text(),
        header=(_BORROWED / "moe_prefill_header.cu").read_text() + _HEADER,
        shared_memory=shared,
    )


def _tile_format(bank) -> str | None:
    """The bank's format when the expert tile kernels read it: "affine" (3- or
    4-bit codes, bf16 scales and biases, 64 weights a group) or "mxfp4" (E2M1
    codes, E8M0 scales, 32 weights a group); None otherwise."""
    spec = bank.spec or {}
    mode = spec.get("mode")
    if mode == "affine":
        fits = (
            spec.get("group_size") == 64
            and bank.bits in (3, 4)
            and bank.biases is not None
            and bank.scales.dtype == mx.bfloat16
        )
    elif mode == "mxfp4":
        fits = (
            spec.get("group_size", 32) == 32
            and bank.bits == 4
            and bank.biases is None
            and bank.scales.dtype == mx.uint8
        )
    else:
        return None
    rows, width = bank.weight.shape[1], bank.weight.shape[2] * 32 // bank.bits
    return mode if fits and not rows % 128 and not width % 128 else None


def experts_fit(banks) -> bool:
    """Whether the expert tile kernels multiply by these banks, in one format
    and code width, with 128-row tiles."""
    first = _tile_format(banks[0])
    return first is not None and all(
        _tile_format(b) == first and b.bits == banks[0].bits for b in banks
    )


def _tile_layout(bank) -> list[tuple[str, int | bool]]:
    """Template values of a bank's format: code bits, mxfp4 and group size."""
    mxfp4 = bank.spec["mode"] == "mxfp4"
    return [("BITS", bank.bits), ("MX", mxfp4), ("GROUP", 32 if mxfp4 else 64)]


def _tile_grid(tiles: int, experts: int, routes: int) -> tuple[int, int, int]:
    # Each launch covers every 64-row tile of an expert in `splits` blocks.
    splits = min(max(_TILE_BUDGET // (tiles * experts), 1), 8, -(-routes // 64))
    return (_TILE_THREADS * tiles, experts, splits)


def expert_gate_up(x, token, experts, gate, up, weights, limit, quantize):
    """layers._swiglu of each route's gate and up projection, in one pass.

    x [L, in] bf16 rows, ``token`` [R] the row of each route and ``experts``
    [R] its expert, ascending; ``weights`` [R] fp32 route weights. Returns
    [R, inner] bf16, FP8 round-tripped when ``quantize`` is set.
    """
    count, inner = gate.weight.shape[:2]
    routes = experts.size
    return _tile_kernel(
        "expert_gate_up",
        "moe_prefill_gate_up.cu",
        ("x", "rows", "inds", "gw", "gs", "gb", "uw", "us", "ub", "rw"),
        ("act",),
    )(
        inputs=[
            x,
            token,
            experts,
            gate.weight,
            gate.scales,
            # An mxfp4 bank has no biases; its scales fill the unread slot.
            gate.scales if gate.biases is None else gate.biases,
            up.weight,
            up.scales,
            up.scales if up.biases is None else up.biases,
            weights,
        ],
        template=[
            ("T", mx.bfloat16),
            ("STG", _TILE_STAGES),
            ("BDEP", _TILE_PREFETCH),
            *_tile_layout(gate),
            ("K", x.shape[-1]),
            ("I", inner),
            ("LIMIT", _bits(limit) if limit else 0),
            *(("Q" + n if n == "BITS" else n, v) for n, v in _quantizer(quantize * 8)),
        ],
        grid=_tile_grid(inner // 64, count, routes),
        threadgroup=(_TILE_THREADS, 1, 1),
        output_shapes=[(routes, inner)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]


def expert_down(act, experts, down):
    """Routes ``act`` [R, inner] through their expert's down projection."""
    count, out = down.weight.shape[:2]
    return _tile_kernel(
        "expert_down", "moe_prefill_down.cu", ("x", "w", "s", "b", "inds"), ("y",)
    )(
        inputs=[
            act,
            down.weight,
            down.scales,
            down.scales if down.biases is None else down.biases,
            experts,
        ],
        template=[
            ("T", mx.bfloat16),
            ("STG", _TILE_STAGES),
            ("BDEP", _TILE_PREFETCH),
            *_tile_layout(down),
            ("K", act.shape[1]),
            ("N", out),
        ],
        grid=_tile_grid(out // 128, count, act.shape[0]),
        threadgroup=(_TILE_THREADS, 1, 1),
        output_shapes=[(act.shape[0], out)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]


def mx_experts_fit(banks) -> bool:
    """Whether the MX tile kernels take these banks: mxfp4 on sm_120 / sm_121,
    whose block-scaled mma multiplies E4M3 activations by E2M1 weights."""
    info = mx.device_info()
    return (
        available()
        and info.get("compute_capability_major") == 12
        and experts_fit(banks)
        and _tile_format(banks[0]) == "mxfp4"
    )


def fp8_codes(x: mx.array) -> tuple[mx.array, mx.array]:
    """quant.quantize(x, 8) of x [L, D] as E4M3 codes and E8M0 scales [L, D / 32]."""
    groups = x.size // 32
    return _kernel("fp8_codes", _FP8_CODES, ("x",), ("codes", "scales"))(
        inputs=[x],
        template=[(n, v) for n, v in _quantizer(8) if n != "BITS"],
        grid=(groups, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (x.shape[0], x.shape[1] // 32)],
        output_dtypes=[mx.uint8, mx.uint8],
    )


def expert_gate_up_mx(codes, scales, token, experts, gate, up, weights, limit):
    """expert_gate_up of quantized rows on the MX tiles: ``codes`` and
    ``scales`` from fp8_codes. Returns the FP8 round trip of the SwiGLU output
    [R, inner] as codes and scales, which expert_down_mx takes."""
    count, inner = gate.weight.shape[:2]
    routes = experts.size
    return _tile_kernel(
        "expert_gate_up_mx",
        "moe_prefill_gate_up_mx.cu",
        ("xc", "xs", "rows", "inds", "gw", "gs", "uw", "us", "rw"),
        ("acodes", "ascales"),
        _TILE_MX_SHARED,
    )(
        inputs=[codes, scales, token, experts, gate.weight, gate.scales]
        + [up.weight, up.scales, weights],
        template=[
            ("STG", _TILE_STAGES),
            ("BDEP", _TILE_PREFETCH),
            ("MT", _TILE_MX_PASS),
            ("K", codes.shape[-1]),
            ("I", inner),
            ("LIMIT", _bits(limit) if limit else 0),
            *((n, v) for n, v in _quantizer(8) if n != "BITS"),
        ],
        grid=_tile_grid(inner // 64, count, routes),
        threadgroup=(_TILE_THREADS, 1, 1),
        output_shapes=[(routes, inner), (routes, inner // 32)],
        output_dtypes=[mx.uint8, mx.uint8],
    )


def expert_down_mx(codes, scales, experts, down):
    """expert_down of expert_gate_up_mx's codes and scales on the MX tiles."""
    count, out = down.weight.shape[:2]
    return _tile_kernel(
        "expert_down_mx",
        "moe_prefill_down_mx.cu",
        ("xc", "xs", "w", "s", "inds"),
        ("y",),
        _TILE_MX_SHARED,
    )(
        inputs=[codes, scales, down.weight, down.scales, experts],
        template=[
            ("STG", _TILE_STAGES),
            ("BDEP", _TILE_PREFETCH),
            ("MT", _TILE_MX_PASS),
            ("K", codes.shape[1]),
            ("N", out),
        ],
        grid=_tile_grid(out // 128, count, codes.shape[0]),
        threadgroup=(_TILE_THREADS, 1, 1),
        output_shapes=[(codes.shape[0], out)],
        output_dtypes=[mx.bfloat16],
    )[
        0
    ]


def index_tc_fits(q: mx.array) -> bool:
    heads, width = q.shape[-2:]
    return q.dtype == mx.bfloat16 and heads % 16 == 0 and heads <= 32 and width == 128


def _index_scores_tc(q, keys, weights, start, ratio):
    """ops.index_scores of queries q [B, H, D] on tensor cores: [B, N] fp32."""
    rows, heads, width = q.shape
    n = keys.shape[0]
    return _kernel(
        "index_scores_tc",
        _INDEX_TC,
        ("queries", "keys", "weights", "meta"),
        ("scores",),
    )(
        inputs=[q, keys, weights, mx.array([rows, n, start, ratio], mx.int32)],
        template=[("T", q.dtype), ("H", heads), ("D", width)],
        grid=(256 * -(-rows // 8), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.float32],
    )[
        0
    ]


def index_topk(q, keys, weights, start, ratio, count, tensor_cores=False):
    """ops.index_topk with the scores and the selection on CUDA kernels.

    ``tensor_cores`` forms the scores in one kernel (see index_tc_fits).
    """
    length, heads = q.shape[1], q.shape[2]
    n = keys.shape[0]
    count = min(count, n)
    if not count:
        return mx.zeros((1, length, 0), mx.int32)
    step = max(1, (_INDEX_DOTS // 4 if tensor_cores else _INDEX_DOTS // heads) // n)
    scores_kernel = _kernel(
        "index_scores", _INDEX_SCORES, ("dots", "weights", "meta"), ("scores",)
    )
    select_kernel = _kernel("index_select", _INDEX_SELECT, ("scores", "meta"), ("ids",))
    out = []
    for begin in range(0, length, step):
        rows = min(step, length - begin)
        if tensor_cores:
            scores = _index_scores_tc(
                q[0, begin : begin + rows],
                keys,
                weights[0, begin : begin + rows],
                start + begin,
                ratio,
            )
        else:
            # One key per row keeps the heads of a query contiguous.
            queries = q[0, begin : begin + rows].reshape(rows * heads, -1)
            dots = ops.matmul(keys, mx.contiguous(queries.T))
            scores = scores_kernel(
                inputs=[
                    dots,
                    weights[0, begin : begin + rows],
                    mx.array([n, rows, heads, start + begin, ratio], mx.int32),
                ],
                template=[("T", q.dtype)],
                grid=(n * rows, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(rows, n)],
                output_dtypes=[mx.float32],
            )[0]
        part = select_kernel(
            inputs=[scores, mx.array([n, count], mx.uint32)],
            grid=(rows * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(rows, count)],
            output_dtypes=[mx.int32],
        )[0]
        mx.eval(part)
        out.append(part)
    return mx.concatenate(out, 0)[None]


def _bits(value: float) -> int:
    """Bit pattern of an fp32 value, for a kernel template argument."""
    return int(np.float32(value).view(np.uint32))


def _quantizer(bits: int) -> list[tuple[str, int]]:
    """Template arguments of v41_quantize: floor and 1 / limit as fp32 bits."""
    if not bits:
        return [("BITS", 0), ("FLOOR", 0), ("INV_LIMIT", 0)]
    limit = 448.0 if bits == 8 else 6.0
    floor = 1e-4 if bits == 8 else limit * 2.0**-126
    return [("BITS", bits), ("FLOOR", _bits(floor)), ("INV_LIMIT", _bits(1.0 / limit))]


def transform(x, bits=0, rope=None, inverse=False):
    """RoPE with the tables ``rope`` (cos, sin), then quant.quantize(bits).

    x is [1, L, ..., D] with D a multiple of 32; either step may be left out.
    """
    if not x.size:
        return x
    width = x.shape[-1]
    cos, sin = rope if rope is not None else (mx.zeros((1,)), mx.zeros((1,)))
    groups = x.size // 32
    per = x.size // (x.shape[1] * width)
    return _kernel("transform", _TRANSFORM, ("x", "cosv", "sinv", "meta"), ("y",))(
        inputs=[x, cos, sin, mx.array([groups, per], mx.uint32)],
        template=[
            ("T", x.dtype),
            ("D", width),
            ("RD", 0 if rope is None else 2 * cos.shape[-1]),
            ("INVERSE", bool(inverse)),
            *_quantizer(bits),
        ],
        grid=(groups, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def swiglu(gate, up, weights, limit, quantize):
    """layers.swiglu of gate/up [R, 1, N] in their dtype, then FP8 if asked."""
    groups = gate.size // 32
    return _kernel("swiglu", _SWIGLU, ("gate", "up", "weights", "meta"), ("y",))(
        inputs=[
            gate,
            up,
            weights if weights is not None else mx.zeros((1,)),
            mx.array([groups, gate.shape[-1] // 32], mx.uint32),
        ],
        template=[
            ("T", gate.dtype),
            ("WEIGHTED", weights is not None),
            ("LIMIT", _bits(limit) if limit else 0),
            *_quantizer(8 if quantize else 0),
        ],
        grid=(groups, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )[0]


def hc_post(x, residual, post, comb):
    """ops.hc_post in one pass over the streams."""
    streams, width = residual.shape[-2:]
    rows = x.size // width
    return _kernel(
        "hc_post", _HC_POST, ("x", "residual", "post", "comb", "meta"), ("y",)
    )(
        inputs=[x, residual, post, comb, mx.array([rows], mx.uint32)],
        template=[("T", x.dtype), ("D", width), ("N", streams)],
        grid=(rows * width, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[residual.shape],
        output_dtypes=[x.dtype],
    )[
        0
    ]


def hc_pre_norm(x, pre, weight, eps):
    """ops.hc_pre_norm in one pass over the streams x [1, L, N, D]."""
    streams, width = x.shape[-2:]
    rows = x.size // (streams * width)
    return _kernel("hc_pre_norm", _HC_PRE_NORM, ("x", "pre", "weight", "eps"), ("y",))(
        inputs=[x, pre, weight, mx.array([eps], mx.float32)],
        template=[("T", x.dtype), ("D", width), ("N", streams)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*x.shape[:-2], width)],
        output_dtypes=[x.dtype],
    )[0]


# Slices of the width the mix projection sums apart, for enough blocks.
_HC_SPLIT = 8


def hc_mixes(x, fn_t, scale, base, config):
    """ops.hc_mixes of streams x [1, L, n, D] in two kernels."""
    width, count = fn_t.shape
    rows = x.size // width
    n = config.hc_mult
    meta = mx.array([rows], mx.int32)
    parts = _kernel("hc_projection", _HC_PROJECTION, ("x", "fn", "meta"), ("parts",))(
        inputs=[x, fn_t, meta],
        template=[("T", x.dtype), ("K", width), ("M", count), ("SPLIT", _HC_SPLIT)],
        grid=(64 * -(-rows // 64), _HC_SPLIT, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(_HC_SPLIT, rows, count + 1)],
        output_dtypes=[mx.float32],
    )[0]
    lead = x.shape[:-2]
    pre, post, comb = _kernel(
        "hc_weights",
        _HC_WEIGHTS,
        ("parts", "scale", "base", "eps", "meta"),
        ("pre", "post", "comb"),
    )(
        inputs=[
            parts,
            scale.astype(mx.float32),
            base.astype(mx.float32),
            mx.array([config.hc_eps, config.norm_eps], mx.float32),
            meta,
        ],
        template=[
            ("N", n),
            ("K", width),
            ("SPLIT", _HC_SPLIT),
            ("ITERS", config.hc_sinkhorn_iters),
        ],
        grid=(rows, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(*lead, n), (*lead, n), (*lead, n, n)],
        output_dtypes=[mx.float32] * 3,
    )
    return pre, post, comb


def combine(routed, inverse, shared):
    """Routed rows [R, 1, D] in sorted order back to tokens, summed over the
    routes of a token, plus the shared expert [L, D]."""
    rows, width = shared.shape
    return _kernel(
        "combine", _COMBINE, ("routed", "inverse", "shared", "meta"), ("y",)
    )(
        inputs=[routed, inverse.astype(mx.uint32), shared, mx.array([rows], mx.uint32)],
        template=[
            ("T", shared.dtype),
            ("D", width),
            ("K", inverse.size // rows),
        ],
        grid=(rows * width, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shared.shape],
        output_dtypes=[shared.dtype],
    )[
        0
    ]


def _slots(window, pooled, wi, ci):
    """(row ids [L, S] of one table, -1 for masked slots; the table)."""
    rows = window.shape[0]
    ids = mx.where((wi >= 0) & (wi < rows), wi, -1)
    table = window
    if ci is not None and ci.shape[-1]:
        seen = (ci >= 0) & (ci < pooled.shape[0])
        ids = mx.concatenate([ids, mx.where(seen, ci + rows, -1)], 1)
        table = mx.concatenate([window, pooled], 0)
    return ids.astype(mx.int32), table


def _attention_tc_shared(width: int) -> int:
    """Shared bytes: 64 key rows, three 64-dim query groups, weights, parts."""
    return 64 * width * 2 + 3 * 64 * 64 * 2 + 64 * 64 * 2 + 4 * 64 * 4


@cache
def _attention_tc_kernel(width: int):
    return mx.fast.cuda_kernel(
        name="deepseek_v41_attention_tc",
        input_names=["q", "table", "ids", "sink", "scale", "meta"],
        output_names=["o"],
        source=_ATTENTION_TC,
        header=_HEADER,
        shared_memory=_attention_tc_shared(width),
    )


def attention_tc_fits(q: mx.array) -> bool:
    heads, width = q.shape[-2:]
    return (
        q.dtype == mx.bfloat16
        and heads % 64 == 0
        and width % 64 == 0
        and _attention_tc_shared(width) <= 99 * 1024
    )


def sparse_attention_tc(q, window, pooled, wi, ci, sink, scale):
    """ops.sparse_attention with bf16 products on tensor cores."""
    _, length, heads, width = q.shape
    ids, table = _slots(window, pooled, wi, ci)
    # q goes in whole; with q[0] the launch copies all of q first.
    return _attention_tc_kernel(width)(
        inputs=[
            q,
            table,
            ids,
            sink,
            mx.array([scale], mx.float32),
            mx.array([ids.shape[-1]], mx.int32),
        ],
        template=[("D", width), ("H", heads)],
        grid=(256 * length * (heads // 64), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(length, heads, width)],
        output_dtypes=[mx.bfloat16],
    )[0][None]


def sparse_attention(q, window, pooled, wi, ci, sink, scale):
    """ops.sparse_attention with the key gather and the weights on kernels."""
    _, length, heads, width = q.shape
    ids, table = _slots(window, pooled, wi, ci)
    slots = ids.shape[-1]
    step = max(1, ops.ATTENTION_KEYS // slots)
    template = [("T", q.dtype), ("D", width), ("S", slots), ("H", heads)]
    keys_kernel = _kernel(
        "attention_keys", _ATTENTION_KEYS, ("table", "ids", "meta"), ("keys",)
    )
    weights_kernel = _kernel(
        "attention_weights",
        _ATTENTION_WEIGHTS,
        ("dots", "ids", "sink", "scale", "meta"),
        ("weights", "denominator"),
    )
    scale = mx.array([scale], mx.float32)
    out = []
    for begin in range(0, length, step):
        count = min(step, length - begin)
        rows_ids = ids[begin : begin + count]
        meta = mx.array([count], mx.uint32)
        keys = keys_kernel(
            inputs=[table, rows_ids, meta],
            template=template,
            grid=(count * slots * (width // 8), 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(count, slots, width)],
            output_dtypes=[mx.float32],
        )[0]
        dots = q[0, begin : begin + count].astype(mx.float32) @ keys.swapaxes(1, 2)
        weights, denominator = weights_kernel(
            inputs=[dots, rows_ids, sink, scale, meta],
            template=template,
            grid=(count * heads, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(count, heads, slots), (count, heads)],
            output_dtypes=[mx.float32, mx.float32],
        )
        part = ((weights @ keys) / denominator[..., None]).astype(q.dtype)
        mx.eval(part)
        out.append(part)
    return mx.concatenate(out, 0)[None]


def relative_error(got: mx.array, want: mx.array) -> float:
    """Relative L2 distance of ``got`` from ``want``."""
    got, want = got.astype(mx.float32), want.astype(mx.float32)
    return (mx.linalg.norm(got - want) / mx.maximum(mx.linalg.norm(want), 1e-30)).item()


def _differ(got: mx.array, want: mx.array) -> float:
    """Share of values that differ by more than fp32 summation noise."""
    got, want = got.astype(mx.float32), want.astype(mx.float32)
    return mx.mean(mx.abs(got - want) > 1e-5 * mx.abs(want) + 1e-7).item()


def _attention_check(dtype: mx.Dtype) -> bool:
    keys = mx.random.split(mx.random.key(11), 6)
    q = mx.random.normal((1, 40, 8, 64), key=keys[0]).astype(dtype)
    window = mx.random.normal((48, 64), key=keys[1]).astype(dtype)
    pooled = mx.random.normal((30, 64), key=keys[2]).astype(dtype)
    sink = mx.random.normal((8,), key=keys[3])
    wi = mx.random.randint(-3, 48, (40, 70), key=keys[4])
    ci = mx.random.randint(-2, 30, (40, 75), key=keys[5])
    for top, rows in ((None, None), (ci, pooled)):
        want = ops.sparse_attention(q, window, rows, wi, top, sink, 0.125)
        got = sparse_attention(q, window, rows, wi, top, sink, 0.125)
        # A sum that associates differently may round a few outputs apart.
        if _differ(got, want) > 2e-3:
            return False
    return True


@cache
def use_attention(dtype: mx.Dtype) -> bool:
    return available() and _checked("attention", lambda: _attention_check(dtype))


# Relative L2 error of the tensor-core attention against ops.sparse_attention:
# the float32 sums differ in order, and a weight near a bf16 rounding boundary
# may round the other way.
_ATTENTION_TC_ERROR = 1e-3


def _attention_tc_check(dtype: mx.Dtype, heads: int, width: int) -> bool:
    keys = mx.random.split(mx.random.key(12), 6)
    q = mx.random.normal((1, 24, heads, width), key=keys[0]).astype(dtype)
    window = mx.random.normal((160, width), key=keys[1]).astype(dtype)
    pooled = mx.random.normal((300, width), key=keys[2]).astype(dtype)
    sink = mx.random.normal((heads,), key=keys[3])
    wi = mx.random.randint(-3, 160, (24, 128), key=keys[4])
    ci = mx.random.randint(-2, 300, (24, 200), key=keys[5])
    scale = width**-0.5
    for top, rows in ((None, None), (ci, pooled)):
        want = ops.sparse_attention(q, window, rows, wi, top, sink, scale)
        got = sparse_attention_tc(q, window, rows, wi, top, sink, scale)
        error = relative_error(got, want)
        if not error <= _ATTENTION_TC_ERROR:
            logger.warning("DeepSeek V4.1 tensor-core attention differs by %.2e", error)
            return False
    return True


@cache
def use_attention_tc(dtype: mx.Dtype, heads: int, width: int) -> bool:
    """Whether sparse_attention_tc may run for queries of this layout."""
    return available() and _checked(
        "tensor-core attention", lambda: _attention_tc_check(dtype, heads, width)
    )


def _pointwise_check(dtype: mx.Dtype) -> bool:
    """Whether the pointwise kernels reproduce their op chains.

    The quantizers and SwiGLU must match exactly; fused RoPE and stream writes
    may round a few values apart.
    """
    keys = mx.random.split(mx.random.key(7), 8)
    x = (mx.random.normal((1, 9, 3, 64), key=keys[0]) * 3).astype(dtype)
    angles = mx.random.uniform(0, 6.0, (9, 8), key=keys[1])
    rope = (mx.cos(angles), mx.sin(angles))
    for bits in (0, 4, 8):
        if bits and not mx.array_equal(transform(x, bits), quant.quantize(x, bits)):
            return False
        for inverse in (False, True):
            want = ops.rope_apply(x, *rope, inverse)
            if _differ(transform(x, 0, rope, inverse), want) > 2e-3:
                return False
            # The round trip of the kernel's own rotation must be exact.
            got = transform(x, bits, rope, inverse)
            rotated = transform(x, 0, rope, inverse)
            if bits and not mx.array_equal(got, quant.quantize(rotated, bits)):
                return False
    gate = (mx.random.normal((50, 1, 64), key=keys[2]) * 6).astype(dtype)
    up = (mx.random.normal((50, 1, 64), key=keys[3]) * 6).astype(dtype)
    weights = mx.random.uniform(0.1, 1.0, (50,), key=keys[4])
    for limit, scale, quantize in ((10.0, weights, True), (0.0, None, False)):
        want = ops.swiglu(
            gate, up, None if scale is None else scale[:, None, None], dtype, limit
        )
        if quantize:
            want = quant.quantize(want)
        if not mx.array_equal(swiglu(gate, up, scale, limit, quantize), want):
            return False
    streams = (mx.random.normal((1, 9, 4, 64), key=keys[5])).astype(dtype)
    post = mx.random.uniform(0.0, 2.0, (1, 9, 4), key=keys[6])
    comb = mx.random.uniform(0.0, 1.0, (1, 9, 4, 4), key=keys[7])
    want = ops.hc_post(x[:, :, 0], streams, post, comb)
    return _differ(hc_post(x[:, :, 0], streams, post, comb), want) < 2e-3


@cache
def use_pointwise(dtype: mx.Dtype) -> bool:
    """Whether transform, swiglu and hc_post may run for this dtype."""
    return available() and _checked("pointwise", lambda: _pointwise_check(dtype))


def _combine_check(dtype: mx.Dtype) -> bool:
    keys = mx.random.split(mx.random.key(9), 3)
    routed = mx.random.normal((60, 1, 64), key=keys[0]).astype(dtype)
    shared = mx.random.normal((10, 64), key=keys[1]).astype(dtype)
    inverse = mx.argsort(mx.random.uniform(shape=(60,), key=keys[2]))
    want = ops.combine(routed, inverse, shared)
    got = combine(routed, inverse, shared)
    # A sum that associates differently may round a few outputs apart.
    return _differ(got, want) < 1e-3


def _hc_mixes_check(config, width: int) -> bool:
    n = config.hc_mult
    keys = mx.random.split(mx.random.key(14), 4)
    x = mx.random.normal((1, 70, n, width // n), key=keys[0]).astype(mx.bfloat16)
    fn_t = mx.random.normal((width, n * (n + 2)), key=keys[1]) * 0.01
    scale = mx.random.uniform(0.5, 2.0, (3,), key=keys[2])
    base = mx.random.normal((n * (n + 2),), key=keys[3])
    want = ops.hc_mixes(x, fn_t, scale, base, config)
    got = hc_mixes(x, fn_t, scale, base, config)
    # Float32 sums over the width in another order.
    return all(relative_error(a, b) < 1e-5 for a, b in zip(got, want))


_HC_MIXES_CHECKED: dict[tuple, bool] = {}


def use_hc_mixes(config, fn_t: mx.array) -> bool:
    """Whether hc_mixes may run for this layout (the config is not hashable)."""
    width, count = fn_t.shape
    n = config.hc_mult
    key = (n, config.hc_eps, config.norm_eps, config.hc_sinkhorn_iters, width)
    if key not in _HC_MIXES_CHECKED:
        fits = width % (64 * _HC_SPLIT) == 0 and count == n * (n + 2) <= 32
        _HC_MIXES_CHECKED[key] = (
            available()
            and fits
            and _checked("stream mixes", lambda: _hc_mixes_check(config, width))
        )
    return _HC_MIXES_CHECKED[key]


def _hc_pre_norm_check(dtype: mx.Dtype, width: int) -> bool:
    keys = mx.random.split(mx.random.key(13), 3)
    x = (mx.random.normal((1, 40, 4, width), key=keys[0]) * 3).astype(dtype)
    pre = mx.random.uniform(0.0, 1.0, (1, 40, 4), key=keys[1])
    weight = mx.random.normal((width,), key=keys[2]).astype(dtype)
    want = ops.hc_pre_norm(x, pre, weight, 1e-6)
    # Sums of squares that associate differently may round a few outputs apart.
    return _differ(hc_pre_norm(x, pre, weight, 1e-6), want) < 2e-3


@cache
def use_hc_pre_norm(dtype: mx.Dtype, width: int) -> bool:
    return available() and _checked(
        "stream mix", lambda: _hc_pre_norm_check(dtype, width)
    )


@cache
def use_combine(dtype: mx.Dtype) -> bool:
    return available() and _checked("expert combine", lambda: _combine_check(dtype))


def _index_check(dtype: mx.Dtype) -> bool:
    """Whether the index kernels reproduce ops.index_topk on a small input."""
    key = mx.random.key(41)
    k1, k2, k3 = mx.random.split(key, 3)
    q = mx.round(mx.random.normal((1, 70, 8, 32), key=k1) * 2).astype(dtype)
    keys = mx.round(mx.random.normal((300, 32), key=k2) * 2).astype(dtype)
    weights = mx.random.normal((1, 70, 8), key=k3) * 0.05
    for start, count in ((0, 16), (530, 64)):
        want = ops.index_topk(q, keys, weights, start, 2, count)
        got = index_topk(q, keys, weights, start, 2, count)
        if not mx.array_equal(want, got):
            return False
    return True


def _index_tc_check(heads: int, width: int) -> bool:
    """Whether the tensor-core scores select as ops.index_topk on integer
    inputs, whose sums are exact in any order."""
    k1, k2, k3 = mx.random.split(mx.random.key(42), 3)
    q = mx.round(mx.random.normal((1, 70, heads, width), key=k1) * 2)
    keys = mx.round(mx.random.normal((300, width), key=k2) * 2)
    weights = mx.random.normal((1, 70, heads), key=k3) * 0.05
    q, keys = q.astype(mx.bfloat16), keys.astype(mx.bfloat16)
    for start, count in ((0, 16), (530, 64)):
        want = ops.index_topk(q, keys, weights, start, 2, count)
        got = index_topk(q, keys, weights, start, 2, count, tensor_cores=True)
        if not mx.array_equal(want, got):
            return False
    return True


@cache
def use_index_tc(heads: int, width: int) -> bool:
    return available() and _checked(
        "tensor-core index scores", lambda: _index_tc_check(heads, width)
    )


@cache
def use_index(dtype: mx.Dtype) -> bool:
    """Whether index_topk may run for this activation dtype."""
    return available() and _checked("index top-k", lambda: _index_check(dtype))


def _checked(name: str, check) -> bool:
    try:
        passed = bool(check())
    except Exception as error:  # a kernel that does not compile stays off
        logger.warning("DeepSeek V4.1 %s kernel is off: %r", name, error)
        return False
    if not passed:
        logger.warning(
            "DeepSeek V4.1 %s kernel differs from its reference and is off", name
        )
    return passed
