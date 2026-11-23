# SPDX-License-Identifier: Apache-2.0
"""CUDA sources of the GLM-5.3 kernels: bodies for ``mx.fast.cuda_kernel``."""

from __future__ import annotations

from pathlib import Path


def borrowed(project: str, name: str) -> str:
    """Text of a kernel source borrowed from ``project``."""
    root = Path(__file__).resolve().parents[2] / "third_party"
    return (root / project / "glm5_next" / name).read_text()


# Routed rows back in token order, weighted and summed in float32 like the
# host's weighted sum: a block per token, a thread per 8 output columns.
MOE_SUM = """
  const int t = blockIdx.x;
  const int d = threadIdx.x * 8;
  float acc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  for (int j = 0; j < TOPK; ++j) {
    const uint32_t r = inverse[t * TOPK + j];
    const float w = weights[t * TOPK + j];
    const uint4 v = *reinterpret_cast<const uint4*>(down + (long)r * N + d);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = fmaf(__bfloat162float(vb[i]), w, acc[i]);
  }
  uint4 o;
  __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o);
  #pragma unroll
  for (int i = 0; i < 8; ++i) ob[i] = __float2bfloat16_rn(acc[i]);
  *reinterpret_cast<uint4*>(y + (long)t * N + d) = o;
"""

# The COUNT highest scores of a row, those above the cut first and then the ones
# equal to it, each in ascending window order, like ``model.top_windows``.
TOP_WINDOWS = """
  constexpr int NT = 256;
  const int tid = threadIdx.x;
  const long row = blockIdx.x, np = scores_shape[1];
  const unsigned short* s = reinterpret_cast<const unsigned short*>(scores) + row * np;
  __shared__ unsigned int hist[256];
  __shared__ unsigned int cut[3];
  __shared__ unsigned int above[NT], equal[NT];
  auto key = [](unsigned short b) -> unsigned int {
    return (b & 0x8000) ? (unsigned int)((~b) & 0xffff) : (unsigned int)(b | 0x8000);
  };
  hist[tid] = 0;
  __syncthreads();
  for (long i = tid; i < np; i += NT) atomicAdd(&hist[key(s[i]) >> 8], 1u);
  __syncthreads();
  if (tid == 0) {
    unsigned int more = 0, high = 0;
    for (int h = 255; h >= 0; --h) {
      if (more + hist[h] >= COUNT) { high = h; break; }
      more += hist[h];
    }
    cut[0] = high;
    cut[2] = more;
  }
  __syncthreads();
  const unsigned int high = cut[0];
  hist[tid] = 0;
  __syncthreads();
  for (long i = tid; i < np; i += NT) {
    const unsigned int k = key(s[i]);
    if ((k >> 8) == high) atomicAdd(&hist[k & 0xff], 1u);
  }
  __syncthreads();
  if (tid == 0) {
    unsigned int more = cut[2], low = 0;
    for (int l = 255; l >= 0; --l) {
      if (more + hist[l] >= COUNT) { low = l; break; }
      more += hist[l];
    }
    cut[1] = (high << 8) | low;
    cut[2] = more;
  }
  __syncthreads();
  const unsigned int edge = cut[1];
  // A thread owns a contiguous run of windows, so slots follow their order.
  const long seg = (np + NT - 1) / NT, a = tid * seg, b = a + seg < np ? a + seg : np;
  unsigned int na = 0, ne = 0;
  for (long i = a; i < b; ++i) {
    const unsigned int k = key(s[i]);
    na += k > edge;
    ne += k == edge;
  }
  above[tid] = na;
  equal[tid] = ne;
  __syncthreads();
  unsigned int pa = 0, pe = cut[2];
  for (int t = 0; t < tid; ++t) {
    pa += above[t];
    pe += equal[t];
  }
  int* o = out + row * COUNT;
  for (long i = a; i < b; ++i) {
    const unsigned int k = key(s[i]);
    if (k > edge) o[pa++] = (int)i;
    else if (k == edge && pe < COUNT) o[pe++] = (int)i;
  }
"""

FRAGMENTS = r"""
#include <cuda_bf16.h>
__device__ __forceinline__ uint32_t glm_smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void glm_cp16(const void* dst, const void* src, bool read) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(glm_smem(dst)), "l"(src), "r"(read ? 16 : 0));
}
__device__ __forceinline__ void glm_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void glm_wait() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void glm_ldm4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(glm_smem(p)));
}
__device__ __forceinline__ void glm_ldm4t(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(glm_smem(p)));
}
__device__ __forceinline__ void glm_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// c += a @ b for 16 products, for sums that feed an exponent. The tensor
// core does not round its own accumulator to nearest, so it sums the 16
// products from zero and the sum is added here.
__device__ __forceinline__ void glm_mma_rn(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  const float zero = 0.f;
  float p[4];
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
      : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(zero));
  c[0] = __fadd_rn(c[0], p[0]); c[1] = __fadd_rn(c[1], p[1]); c[2] = __fadd_rn(c[2], p[2]); c[3] = __fadd_rn(c[3], p[3]);
}
// Two float32 weights as a bf16 pair and the bf16 pair of what rounding lost.
__device__ __forceinline__ void glm_split(float x, float y, uint32_t& high, uint32_t& low) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  const float2 back = __bfloat1622float2(h);
  const __nv_bfloat162 l = __floats2bfloat162_rn(x - back.x, y - back.y);
  high = *reinterpret_cast<const uint32_t*>(&h);
  low = *reinterpret_cast<const uint32_t*>(&l);
}
"""

# Latent attention of one query row over its selected keys. The weights enter
# as a bf16 pair plus its rounding error, so the float32 sums round once.
SPARSE_ATTENTION = """
  constexpr int D = 512, T = 32, NT = 256, TILES = (W + T - 1) / T;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* keys = reinterpret_cast<__nv_bfloat16*>(smem);
  float* part = reinterpret_cast<float*>(smem + T * D * 2);
  int* stok = reinterpret_cast<int*>(smem + T * D * 2 + NT * 16 * 4);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int hi = warp >> 1, dh = warp & 1, g = lane >> 2, q2 = (lane & 3) * 2;
  const long R = q_shape[1], r = blockIdx.x;
  for (int i = tid; i < TILES * T; i += NT) stok[i] = i < W ? tok[r * W + i] : -1;
  const __nv_bfloat16* q0 = q + ((long)(hi * 16 + g) * R + r) * D + dh * 256 + q2;
  const __nv_bfloat16* q1 = q0 + 8 * R * D;
  const float sc = scale[0];
  const float none = -INFINITY;
  float o[32][4];
  #pragma unroll
  for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float top[2] = {none, none}, sum[2] = {0.f, 0.f};
  __syncthreads();
  for (int t = 0; t < TILES; ++t) {
    #pragma unroll
    for (int i = 0; i < T * 64 / NT; ++i) {
      const int idx = tid + i * NT, row = idx >> 6, c = idx & 63;
      const int at = stok[t * T + row];
      glm_cp16(keys + row * D + ((c ^ (row & 7)) << 3), kv + (long)(at < 0 ? 0 : at) * D + c * 8, at >= 0);
    }
    glm_commit();
    glm_wait();
    __syncthreads();
    // Products of this warp's heads with the keys over its half of the dims.
    float s[4][4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
    #pragma unroll
    for (int kq = 0; kq < 16; ++kq) {
      const uint32_t a[4] = {
          *reinterpret_cast<const uint32_t*>(q0 + kq * 16), *reinterpret_cast<const uint32_t*>(q1 + kq * 16),
          *reinterpret_cast<const uint32_t*>(q0 + kq * 16 + 8), *reinterpret_cast<const uint32_t*>(q1 + kq * 16 + 8)};
      #pragma unroll
      for (int side = 0; side < 2; ++side) {
        uint32_t b[4];
        const int row = side * 16 + (lane >> 4) * 8 + (lane & 7);
        const int c = dh * 32 + kq * 2 + ((lane >> 3) & 1);
        glm_ldm4(b, keys + row * D + ((c ^ (row & 7)) << 3));
        glm_mma_rn(s[2 * side], a, b[0], b[1]);
        glm_mma_rn(s[2 * side + 1], a, b[2], b[3]);
      }
    }
    float* mine = part + tid * 16;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      #pragma unroll
      for (int e = 0; e < 4; ++e) mine[j * 4 + e] = s[j][e];
    __syncthreads();
    // The two halves of a head tile.
    const float* half = part + (hi * 2 * 32 + lane) * 16;
    float tile_top[2] = {none, none};
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = j * 4 + e;
        const float v = half[i] + half[512 + i];
        const bool live = stok[t * T + j * 8 + q2 + (e & 1)] >= 0;
        s[j][e] = live ? v * sc : none;
        tile_top[e >> 1] = fmaxf(tile_top[e >> 1], s[j][e]);
      }
    float factor[2], gain[2] = {0.f, 0.f};
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      tile_top[i] = fmaxf(tile_top[i], __shfl_xor_sync(0xffffffffu, tile_top[i], 1));
      tile_top[i] = fmaxf(tile_top[i], __shfl_xor_sync(0xffffffffu, tile_top[i], 2));
      const float next = fmaxf(top[i], tile_top[i]);
      factor[i] = next == none ? 1.f : __expf(top[i] - next);
      top[i] = next;
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        s[j][e] = s[j][e] == none ? 0.f : __expf(s[j][e] - top[e >> 1]);
        gain[e >> 1] += s[j][e];
      }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      gain[i] += __shfl_xor_sync(0xffffffffu, gain[i], 1);
      gain[i] += __shfl_xor_sync(0xffffffffu, gain[i], 2);
      sum[i] = sum[i] * factor[i] + gain[i];
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      o[i][0] *= factor[0];
      o[i][1] *= factor[0];
      o[i][2] *= factor[1];
      o[i][3] *= factor[1];
    }
    #pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      uint32_t high[4], low[4];
      glm_split(s[2 * kk][0], s[2 * kk][1], high[0], low[0]);
      glm_split(s[2 * kk][2], s[2 * kk][3], high[1], low[1]);
      glm_split(s[2 * kk + 1][0], s[2 * kk + 1][1], high[2], low[2]);
      glm_split(s[2 * kk + 1][2], s[2 * kk + 1][3], high[3], low[3]);
      #pragma unroll
      for (int f = 0; f < 16; ++f) {
        uint32_t b[4];
        const int row = kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
        const int c = dh * 32 + f * 2 + (lane >> 4);
        glm_ldm4t(b, keys + row * D + ((c ^ (row & 7)) << 3));
        glm_mma(o[2 * f], high, b[0], b[1]);
        glm_mma(o[2 * f], low, b[0], b[1]);
        glm_mma(o[2 * f + 1], high, b[2], b[3]);
        glm_mma(o[2 * f + 1], low, b[2], b[3]);
      }
    }
    __syncthreads();
  }
  const float i0 = sum[0] > 0.f ? 1.f / sum[0] : 0.f, i1 = sum[1] > 0.f ? 1.f / sum[1] : 0.f;
  __nv_bfloat16* y0 = out + ((long)(hi * 16 + g) * R + r) * D + dh * 256 + q2;
  __nv_bfloat16* y1 = y0 + 8 * R * D;
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    *reinterpret_cast<__nv_bfloat162*>(y0 + i * 8) = __floats2bfloat162_rn(o[i][0] * i0, o[i][1] * i0);
    *reinterpret_cast<__nv_bfloat162*>(y1 + i * 8) = __floats2bfloat162_rn(o[i][2] * i1, o[i][3] * i1);
  }
"""

# Hyper-connection write: post * branch + comb^T @ residual in float32, one
# rounding. A thread owns 8 dims of one row for every stream.
HC_EXPAND = """
  constexpr int D8 = D / 8;
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long l = i / D8;
  const int d = (int)(i % D8) * 8;
  if (l >= branch_shape[0]) return;
  const uint4 raw = *reinterpret_cast<const uint4*>(branch + l * D + d);
  const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&raw);
  float br[8], res[HC][8];
  #pragma unroll
  for (int e = 0; e < 8; ++e) br[e] = __bfloat162float(rb[e]);
  #pragma unroll
  for (int t = 0; t < HC; ++t) {
    const uint4 rr = *reinterpret_cast<const uint4*>(residual + (l * HC + t) * D + d);
    const __nv_bfloat16* r = reinterpret_cast<const __nv_bfloat16*>(&rr);
    #pragma unroll
    for (int e = 0; e < 8; ++e) res[t][e] = __bfloat162float(r[e]);
  }
  #pragma unroll
  for (int s = 0; s < HC; ++s) {
    const float p = post[l * HC + s];
    uint4 o;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o);
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      float y = __fmul_rn(p, br[e]);
      #pragma unroll
      for (int t = 0; t < HC; ++t) y = __fadd_rn(y, __fmul_rn(comb[(l * HC + t) * HC + s], res[t][e]));
      ob[e] = __float2bfloat16_rn(y);
    }
    *reinterpret_cast<uint4*>(out + (l * HC + s) * D + d) = o;
  }
"""

# Hyper-connection read of one row from its products with the mix weight: the
# sinkhorn plan (post, comb) and the collapsed streams, in float32.
HC_PRE = """
  constexpr int K = HC * D, MIX = (2 + HC) * HC, NT = 256;
  const int tid = threadIdx.x;
  const long l = blockIdx.x;
  __shared__ float squares[NT];
  __shared__ float pre[HC];
  const __nv_bfloat16* xr = x + l * K;
  float sq = 0.f;
  for (int i = tid * 8; i < K; i += NT * 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(xr + i);
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float v = __bfloat162float(xb[e]);
      sq = fmaf(v, v, sq);
    }
  }
  squares[tid] = sq;
  __syncthreads();
  if (tid == 0) {
    float total = 0.f;
    for (int t = 0; t < NT; ++t) total += squares[t];
    const float inv = rsqrtf(total / K + eps[0]);
    const float tiny = eps[1];
    float mix[MIX];
    for (int m = 0; m < MIX; ++m) mix[m] = dots[l * MIX + m] * inv;
    for (int s = 0; s < HC; ++s) {
      pre[s] = 1.f / (1.f + expf(-(mix[s] * scale[0] + base[s]))) + tiny;
      post[l * HC + s] = 2.f / (1.f + expf(-(mix[HC + s] * scale[1] + base[HC + s])));
    }
    float c[HC][HC];
    for (int a = 0; a < HC; ++a) {
      float top = -INFINITY, sum = 0.f;
      for (int b = 0; b < HC; ++b) {
        c[a][b] = mix[2 * HC + a * HC + b] * scale[2] + base[2 * HC + a * HC + b];
        top = fmaxf(top, c[a][b]);
      }
      for (int b = 0; b < HC; ++b) {
        c[a][b] = expf(c[a][b] - top);
        sum += c[a][b];
      }
      for (int b = 0; b < HC; ++b) c[a][b] = c[a][b] / sum + tiny;
    }
    for (int it = 0; it < ITERS; ++it) {
      if (it > 0)
        for (int a = 0; a < HC; ++a) {
          float sum = 0.f;
          for (int b = 0; b < HC; ++b) sum += c[a][b];
          for (int b = 0; b < HC; ++b) c[a][b] /= sum + tiny;
        }
      for (int b = 0; b < HC; ++b) {
        float sum = 0.f;
        for (int a = 0; a < HC; ++a) sum += c[a][b];
        for (int a = 0; a < HC; ++a) c[a][b] /= sum + tiny;
      }
    }
    for (int a = 0; a < HC; ++a)
      for (int b = 0; b < HC; ++b) comb[(l * HC + a) * HC + b] = c[a][b];
  }
  __syncthreads();
  for (int d = tid * 8; d < D; d += NT * 8) {
    float y[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) y[e] = 0.f;
    #pragma unroll
    for (int s = 0; s < HC; ++s) {
      const uint4 raw = *reinterpret_cast<const uint4*>(xr + s * D + d);
      const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&raw);
      #pragma unroll
      for (int e = 0; e < 8; ++e) y[e] = __fadd_rn(y[e], __fmul_rn(pre[s], __bfloat162float(xb[e])));
    }
    uint4 o;
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o);
    #pragma unroll
    for (int e = 0; e < 8; ++e) ob[e] = __float2bfloat16_rn(y[e]);
    *reinterpret_cast<uint4*>(collapsed + l * D + d) = o;
  }
"""

FUNCTORS = """
#include "mlx/backend/cuda/device/binary_ops.cuh"
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include <cuda_bf16.h>
"""

# KDA prework: short convolution, SiLU with every step rounded to bf16, and the
# float32 L2 normalization of q and k.
KDA_PREWORK = """
  constexpr int DIM = 128, C = 3 * H * DIM, PAST = TAPS - 1;
  const int lane = threadIdx.x & 31;
  const long unit = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const long row = unit / (3 * H);
  const int sh = (int)(unit % (3 * H)), section = sh / H, head = sh % H;
  if (row >= proj_shape[0]) return;
  const int ch = sh * DIM + lane * 4;
  const long stride = proj_shape[1];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int tap = 0; tap < TAPS; ++tap) {
    const long src = row + tap - PAST;
    const __nv_bfloat16* xp = src < 0 ? state + (src + PAST) * C + ch : proj + src * stride + ch;
    #pragma unroll
    for (int e = 0; e < 4; ++e)
      acc[e] = fmaf(__bfloat162float(xp[e]), __bfloat162float(w[(ch + e) * TAPS + tap]), acc[e]);
  }
  float f[4], ss = 0.f;
  __nv_bfloat16 act[4];
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    // MLX's own functors: its bf16 sigmoid rounds at every step.
    const __nv_bfloat16 c = __float2bfloat16_rn(acc[e]);
    act[e] = mlx::core::cu::Multiply{}(c, mlx::core::cu::Sigmoid{}(c));
    f[e] = __bfloat162float(act[e]);
    ss = fmaf(f[e], f[e], ss);
  }
  __nv_bfloat16* dst = (section == 0 ? q : section == 1 ? k : v) + (row * H + head) * DIM + lane * 4;
  if (section < 2) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(ss + 1e-6f);
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float n = __fmul_rn(f[e], inv);
      act[e] = __float2bfloat16_rn(section == 0 ? __fmul_rn(n, qscale[0]) : n);
    }
  }
  #pragma unroll
  for (int e = 0; e < 4; ++e) dst[e] = act[e];
"""

# KDA forget factor exp(bound * sigmoid(rate * (a + bias))) in float32.
KDA_DECAY = """
  constexpr int DIM = 128;
  const long i = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (i >= (long)a_shape[0] * H * DIM) return;
  const int hd = (int)(i % (H * DIM));
  const float r = rate[hd / DIM], b = bound[0];
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float x = __fmul_rn(r, __bfloat162float(a[i + e]) + bias[hd + e]);
    const float sy = 1.f / (1.f + expf(fabsf(x)));
    g[i + e] = expf(__fmul_rn(b, x < 0.f ? sy : 1.f - sy));
  }
"""

# KDA output: RMS norm over each head, its gain and the sigmoid gate in
# float32, one rounding. A warp per (row, head).
KDA_NORM_GATE = """
  constexpr int DIM = 128;
  const int lane = threadIdx.x & 31;
  const long unit = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (unit >= (long)y_shape[0] * H) return;
  const long at = unit * DIM + lane * 4;
  float f[4], ss = 0.f;
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    f[e] = __bfloat162float(y[at + e]);
    ss = fmaf(f[e], f[e], ss);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss / DIM + eps[0]);
  #pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float t = __fmul_rn(__bfloat162float(w[lane * 4 + e]), __fmul_rn(f[e], inv));
    const float z = __bfloat162float(gate[at + e]);
    const float sy = 1.f / (1.f + expf(fabsf(z)));
    out[at + e] = __float2bfloat16_rn(__fmul_rn(t, z < 0.f ? sy : 1.f - sy));
  }
"""
