# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for Inkling 2 prefill.

The attention, depth-mix and router kernels port the host's Metal kernels in
``omlx/patches/inkling2/kernels.py``: the same staging precision, softmax
steps and selection rules, on the tensor cores. Routed experts run on the
expert GEMMs the MiniMax family already ships. A kernel is used only after
``enable`` matched it with the plain-ops reference.
"""

from __future__ import annotations

import functools
import logging

import mlx.core as mx

from ..deepseek_v41 import kernels as deepseek
from ..minimax_m3_vl import kernels as minimax
from . import layers

logger = logging.getLogger(__name__)

# Most depth sources the mix kernel takes: eight completed blocks and the
# partial sum (48 layers in blocks of 6).
MIX_SOURCES = 9
_FORMATS = ("nvfp4", "mxfp8")
# Affine code widths the expert tile takes (64 weights a group): what oQ
# writes for Inkling 2 experts below 4 bits, and requantized MXFP8 layers.
_AFFINE_BITS = (2, 3, 4, 8)
_on: set[str] = set()


def available() -> bool:
    return mx.cuda.is_available()


def enabled(name: str) -> bool:
    return name in _on


# One block runs 64 queries of one head in four warps (16 rows each) over
# 64-key tiles staged in half, as the host stages BF16 q/k/v in half. Scores
# are float32 in the log2 domain; the online softmax steps over 16 keys at a
# time, as the host's kernel does, and starts from the row's sink logit.
_ATTENTION_HEADER = """
#include <cuda_bf16.h>
#include <cuda_fp16.h>
__device__ __forceinline__ uint32_t ink_half2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void ink_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ink_ldsm_t2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
"""
_ATTENTION = """
constexpr int BQ = 64, BK = 64, LD = D + 8;
constexpr float LOG2E = 1.4426950408889634f;
__shared__ __align__(16) __half ks[BK * LD];
__shared__ __align__(16) __half vs[BK * LD];
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
const int h = blockIdx.y, hk = h / (HQ / HKV);
const int L = dims[0], S = dims[1];
const int q0 = (blockIdx.x) * BQ;
const int r0 = q0 + warp * 16 + (lane >> 2), r1 = r0 + 8;
const int a0 = S - L + r0, a1 = S - L + r1;
const float qscale = scale[0] * LOG2E;

uint32_t qf[D / 16][4];
#pragma unroll
for (int kb = 0; kb < D / 16; ++kb) {
  const int c = kb * 16 + (lane & 3) * 2;
  float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (r0 < L) {
    const __nv_bfloat16* p = q + ((size_t)r0 * HQ + h) * D;
    v[0] = __bfloat162float(p[c]); v[1] = __bfloat162float(p[c + 1]);
    v[4] = __bfloat162float(p[c + 8]); v[5] = __bfloat162float(p[c + 9]);
  }
  if (r1 < L) {
    const __nv_bfloat16* p = q + ((size_t)r1 * HQ + h) * D;
    v[2] = __bfloat162float(p[c]); v[3] = __bfloat162float(p[c + 1]);
    v[6] = __bfloat162float(p[c + 8]); v[7] = __bfloat162float(p[c + 9]);
  }
  qf[kb][0] = ink_half2(v[0], v[1]);
  qf[kb][1] = ink_half2(v[2], v[3]);
  qf[kb][2] = ink_half2(v[4], v[5]);
  qf[kb][3] = ink_half2(v[6], v[7]);
}
float o[D / 8][4];
#pragma unroll
for (int n = 0; n < D / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
float m0 = r0 < L ? sink[(size_t)r0 * HQ + h] * LOG2E : 0.f;
float m1 = r1 < L ? sink[(size_t)r1 * HQ + h] * LOG2E : 0.f;
float l0 = 1.f, l1 = 1.f;
const __nv_bfloat16* rel0 = rel + ((size_t)(r0 < L ? r0 : 0) * HQ + h) * R;
const __nv_bfloat16* rel1 = rel + ((size_t)(r1 < L ? r1 : 0) * HQ + h) * R;

const int a_first = S - L + q0;
const int a_last = S - L + min(q0 + BQ, L) - 1;
int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
const int k_begin = (lo / 16) * 16;
// The warp's own rows bound which 16-key groups can be visible to it.
const int w_first = S - L + q0 + warp * 16;
const int w_last = min(S - L + q0 + warp * 16 + 15, S - 1);
const bool warp_on = q0 + warp * 16 < L;
const __nv_bfloat16* kbase = k + (size_t)hk * S * D;
const __nv_bfloat16* vbase = v + (size_t)hk * S * D;

for (int k0 = k_begin; k0 <= a_last; k0 += BK) {
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < (BK * D / 8) / 128; ++i) {
    const int c = tid + i * 128;
    const int row = c / (D / 8), col = (c % (D / 8)) * 8;
    const int key = k0 + row;
    uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
    if (key < S) {
      kw = *reinterpret_cast<const uint4*>(kbase + (size_t)key * D + col);
      vw = *reinterpret_cast<const uint4*>(vbase + (size_t)key * D + col);
    }
    const __nv_bfloat16* kb16 = reinterpret_cast<const __nv_bfloat16*>(&kw);
    const __nv_bfloat16* vb16 = reinterpret_cast<const __nv_bfloat16*>(&vw);
    uint4 kh, vh;
    uint32_t* khp = reinterpret_cast<uint32_t*>(&kh);
    uint32_t* vhp = reinterpret_cast<uint32_t*>(&vh);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      khp[j] = ink_half2(__bfloat162float(kb16[2 * j]), __bfloat162float(kb16[2 * j + 1]));
      vhp[j] = ink_half2(__bfloat162float(vb16[2 * j]), __bfloat162float(vb16[2 * j + 1]));
    }
    *reinterpret_cast<uint4*>(ks + row * LD + col) = kh;
    *reinterpret_cast<uint4*>(vs + row * LD + col) = vh;
  }
  __syncthreads();
  if (!warp_on) continue;
  #pragma unroll
  for (int g = 0; g < BK / 16; ++g) {
    const int g0 = k0 + g * 16;
    // A group past every row of the warp, or before every row's window,
    // leaves the softmax as it is.
    if (g0 > w_last) break;
    if (WINDOW > 0 && g0 + 15 < w_first - WINDOW + 1) continue;
    float s[2][4];
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
      s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
      const __half* kr = ks + (g * 16 + t * 8 + (lane >> 2)) * LD + (lane & 3) * 2;
      #pragma unroll
      for (int kb = 0; kb < D / 16; ++kb) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr + kb * 16);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr + kb * 16 + 8);
        ink_mma(s[t], qf[kb], b0, b1);
      }
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
    #pragma unroll
    for (int t = 0; t < 2; ++t)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = g0 + t * 8 + (lane & 3) * 2 + (e & 1);
        const int row = e < 2 ? r0 : r1;
        const int d = (e < 2 ? a0 : a1) - key;
        const bool vis = row < L && key < S && d >= 0 && (WINDOW == 0 || d < WINDOW);
        float x = s[t][e] * qscale;
        if (vis) {
          if (d < R) x += __bfloat162float((e < 2 ? rel0 : rel1)[d]) * LOG2E;
        } else {
          x = -INFINITY;
        }
        s[t][e] = x;
        if (e < 2) mx0 = fmaxf(mx0, x); else mx1 = fmaxf(mx1, x);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float f0 = exp2f(m0 - n0), f1 = exp2f(m1 - n1);
    float sum0 = 0.f, sum1 = 0.f;
    #pragma unroll
    for (int t = 0; t < 2; ++t)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float p = exp2f(s[t][e] - (e < 2 ? n0 : n1));
        s[t][e] = p;
        if (e < 2) sum0 += p; else sum1 += p;
      }
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
    l0 = l0 * f0 + sum0;
    l1 = l1 * f1 + sum1;
    m0 = n0;
    m1 = n1;
    uint32_t pa[4];
    pa[0] = ink_half2(s[0][0], s[0][1]);
    pa[1] = ink_half2(s[0][2], s[0][3]);
    pa[2] = ink_half2(s[1][0], s[1][1]);
    pa[3] = ink_half2(s[1][2], s[1][3]);
    const int vrow = g * 16 + (lane & 15);
    #pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      o[n][0] *= f0; o[n][1] *= f0; o[n][2] *= f1; o[n][3] *= f1;
      uint32_t b0, b1;
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(vs + vrow * LD + n * 8));
      ink_ldsm_t2(addr, b0, b1);
      ink_mma(o[n], pa, b0, b1);
    }
  }
}
const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
for (int n = 0; n < D / 8; ++n) {
  const int c = n * 8 + (lane & 3) * 2;
  if (r0 < L) {
    __nv_bfloat16* p = out + ((size_t)r0 * HQ + h) * D + c;
    p[0] = __float2bfloat16_rn(o[n][0] * i0);
    p[1] = __float2bfloat16_rn(o[n][1] * i0);
  }
  if (r1 < L) {
    __nv_bfloat16* p = out + ((size_t)r1 * HQ + h) * D + c;
    p[0] = __float2bfloat16_rn(o[n][2] * i1);
    p[1] = __float2bfloat16_rn(o[n][3] * i1);
  }
}
"""


@functools.cache
def _attention_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_attention",
        input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
        output_names=["out"],
        source=_ATTENTION,
        header=_ATTENTION_HEADER,
    )


def covers_attention(q: mx.array, extent: int) -> bool:
    """Whether the attention kernel takes these queries [1, L, H, 128]."""
    return q.dtype == mx.bfloat16 and q.shape[0] == 1 and q.shape[-1] == 128


def attention(q, keys, values, rel, sink, scale: float, window: int) -> mx.array:
    """Banded attention with per-query sinks: q [1, L, Hq, D], keys/values
    [1, Hkv, S, D] (the queries are the last L keys), rel [1, L, Hq, R],
    sink [1, L, Hq]. Returns [1, L, Hq, D]."""
    _, length, heads, dim = q.shape
    kv_heads, total = keys.shape[1], keys.shape[2]
    out = _attention_kernel()(
        inputs=[
            mx.contiguous(q[0]),
            mx.contiguous(keys[0]),
            mx.contiguous(values[0]),
            mx.contiguous(rel[0]),
            mx.contiguous(sink[0].astype(mx.float32)),
            mx.array([length, total], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("D", dim),
            ("HQ", heads),
            ("HKV", kv_heads),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
        ],
        grid=(-(-length // 64) * 128, heads, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(length, heads, dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out[None]


# One 256-thread block per token row, as the host's Metal kernel: dot products
# of every source with every query, softmax over the sources, weighted sum.
_MIX = """
constexpr int TG = 256;
const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
const size_t base = (size_t)row * H;
float acc[NS * Q];
#pragma unroll
for (int i = 0; i < NS * Q; ++i) acc[i] = 0.f;
for (int c = tid; c < H; c += TG) {
  float s[NS];
  LOADS
  #pragma unroll
  for (int qi = 0; qi < Q; ++qi) {
    const float qv = query[qi * H + c];
    #pragma unroll
    for (int n = 0; n < NS; ++n) acc[n * Q + qi] = fmaf(s[n], qv, acc[n * Q + qi]);
  }
}
__shared__ float red[NS * Q][TG / 32];
#pragma unroll
for (int i = 0; i < NS * Q; ++i) {
  float v = acc[i];
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
  if (lane == 0) red[i][warp] = v;
}
__syncthreads();
float w[NS * Q];
#pragma unroll
for (int i = 0; i < NS * Q; ++i) {
  float t = 0.f;
  #pragma unroll
  for (int j = 0; j < TG / 32; ++j) t += red[i][j];
  w[i] = t * scale[0];
}
#pragma unroll
for (int qi = 0; qi < Q; ++qi) {
  float m = -INFINITY;
  #pragma unroll
  for (int n = 0; n < NS; ++n) m = fmaxf(m, w[n * Q + qi]);
  float z = 0.f;
  #pragma unroll
  for (int n = 0; n < NS; ++n) {
    w[n * Q + qi] = expf(w[n * Q + qi] - m);
    z += w[n * Q + qi];
  }
  #pragma unroll
  for (int n = 0; n < NS; ++n) w[n * Q + qi] /= z;
}
for (int c = tid; c < H; c += TG) {
  float s[NS];
  LOADS
  #pragma unroll
  for (int qi = 0; qi < Q; ++qi) {
    float y = w[qi] * s[0];
    #pragma unroll
    for (int n = 1; n < NS; ++n) y = fmaf(w[n * Q + qi], s[n], y);
    out[((size_t)row * Q + qi) * H + c] = y;
  }
}
"""


@functools.cache
def _mix_kernel(sources: int, queries: int):
    names = [f"b{i}" for i in range(sources - 1)] + ["p"]
    loads = " ".join(f"s[{i}] = {n}[base + c];" for i, n in enumerate(names))
    return mx.fast.cuda_kernel(
        name=f"omlx_inkling2_mix_{sources}_{queries}",
        input_names=[*names, "query", "scale"],
        output_names=["out"],
        source=_MIX.replace("LOADS", loads),
    )


def mix(blocks: list, partial: mx.array, query: mx.array, scale: float) -> mx.array:
    """Depth mixture of unit-RMS float32 sources [1, L, H] with ``query``
    [Q, H]: [1, L, Q, H] float32."""
    _, length, hidden = partial.shape
    queries = query.shape[0]
    sources = len(blocks) + 1
    return _mix_kernel(sources, queries)(
        inputs=[*blocks, partial, query, mx.array([scale], dtype=mx.float32)],
        template=[("NS", sources), ("Q", queries), ("H", hidden)],
        grid=(length * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, length, queries, hidden)],
        output_dtypes=[mx.float32],
    )[0]


# One warp per token row: sigmoid scores, top-k by score + bias (the lowest
# expert wins a tie), weights renormalized from the unbiased scores.
_ROUTE = """
const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
const int lane = threadIdx.x & 31;
if (row >= rows[0]) return;
constexpr int PER = E / 32;
float sc[PER], key[PER];
#pragma unroll
for (int j = 0; j < PER; ++j) {
  const int e = lane + 32 * j;
  const float s = 1.f / (1.f + expf(-logits[(size_t)row * E + e]));
  sc[j] = s;
  key[j] = s + bias[e];
}
float total = 0.f;
for (int t = 0; t < K; ++t) {
  float best = -INFINITY;
  int bj = 0;
  #pragma unroll
  for (int j = 0; j < PER; ++j) {
    if (key[j] > best) {
      best = key[j];
      bj = j;
    }
  }
  float top = best;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, off));
  uint32_t mine = best == top ? (uint32_t)(lane + 32 * bj) : 0xFFFFFFFFu;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) mine = min(mine, __shfl_xor_sync(0xffffffff, mine, off));
  float s = 0.f;
  #pragma unroll
  for (int j = 0; j < PER; ++j) {
    if (lane == (int)(mine % 32) && j == (int)(mine / 32)) {
      s = sc[j];
      key[j] = -INFINITY;
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
  total += s;
  if (lane == 0) {
    inds[(size_t)row * K + t] = mine;
    weights[(size_t)row * K + t] = s;
  }
}
if (lane == 0) {
  const float norm = route_scale[0] / (total + 1e-20f);
  for (int t = 0; t < K; ++t) weights[(size_t)row * K + t] *= norm;
}
"""


@functools.cache
def _route_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_route",
        input_names=["logits", "bias", "route_scale", "rows"],
        output_names=["inds", "weights"],
        source=_ROUTE,
    )


def route_topk(logits: mx.array, bias: mx.array, top_k: int, route_scale: float):
    """([N, K] uint32 experts, [N, K] float32 weights) of float32 logits."""
    rows, experts = logits.shape
    warps = 8
    return _route_kernel()(
        inputs=[
            logits,
            bias.astype(mx.float32),
            mx.array([route_scale], dtype=mx.float32),
            mx.array([rows], dtype=mx.int32),
        ],
        template=[("E", experts), ("K", top_k)],
        grid=(-(-rows // warps) * warps * 32, 1, 1),
        threadgroup=(warps * 32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )


# The MiniMax router kernel's sums (each logit adds its K products in float32
# FMAs, in ``runs`` parts summed in order) for ROWS token rows per block, so
# every router column loads once for all of them.
_ROUTER = """
extern __shared__ __align__(16) unsigned char smem[];
__nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
const int K = x_shape[1], outputs = w_shape[1], rows = x_shape[0];
const int row0 = blockIdx.x * ROWS, e = threadIdx.x;
for (int i = threadIdx.x; i < ROWS * K; i += blockDim.x) {
  const int r = row0 + i / K;
  xs[i] = r < rows ? x[(size_t)r * K + i % K] : __float2bfloat16(0.f);
}
__syncthreads();
if (e >= outputs) return;
const int parts = runs[0], span = (K / 16 / parts) * 16;
float total[ROWS];
for (int p = 0; p < parts; ++p) {
  const int end = p + 1 == parts ? K : (p + 1) * span;
  float acc[ROWS];
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
  for (int k = p * span; k < end; ++k) {
    const float wk = w[(size_t)k * outputs + e];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = fmaf(__bfloat162float(xs[r * K + k]), wk, acc[r]);
  }
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) total[r] = p == 0 ? acc[r] : total[r] + acc[r];
}
#pragma unroll
for (int r = 0; r < ROWS; ++r)
  if (row0 + r < rows) out[(size_t)(row0 + r) * outputs + e] = total[r];
"""
_ROUTER_ROWS = 4


@functools.cache
def _router_kernel(shared: int):
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_router",
        input_names=["x", "w", "runs"],
        output_names=["out"],
        source=_ROUTER,
        shared_memory=shared,
    )


def router_logits(x: mx.array, columns: mx.array) -> mx.array:
    """Float32 router logits [rows, experts] of BF16 ``x`` [rows, K] and the
    float32 router weight as ``columns`` [K, experts], in the host's order."""
    rows, inputs = x.shape
    outputs = columns.shape[1]
    threads = -(-outputs // 32) * 32
    runs = mx.array([minimax.host_runs(rows, outputs, inputs)], dtype=mx.int32)
    return _router_kernel(_ROUTER_ROWS * inputs * 2)(
        inputs=[x, columns, runs],
        template=[("ROWS", _ROUTER_ROWS)],
        grid=(threads * -(-rows // _ROUTER_ROWS), 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, outputs)],
        output_dtypes=[mx.float32],
    )[0]


# MLX's own sigmoid and tanh, so the fused ops round as the plain ones do.
_MLX_OPS = """
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include <cuda_bf16.h>
"""
# The NVFP4 post-scale (the product rounds to BF16) and the softcapped SwiGLU
# of the interleaved gate (even) and up (odd) columns, one output each.
_ACTIVATION = """
using namespace mlx::core::cu;
const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
const size_t inter = y_shape[1] / 2;
if (id >= (size_t)y_shape[0] * inter) return;
const size_t r = id / inter, i = id % inter;
__nv_bfloat16 g = y[r * 2 * inter + 2 * i], u = y[r * 2 * inter + 2 * i + 1];
if (POST) {
  const __nv_bfloat16 s = __float2bfloat16_rn(post[experts[r]]);
  g = __float2bfloat16_rn(__bfloat162float(g) * __bfloat162float(s));
  u = __float2bfloat16_rn(__bfloat162float(u) * __bfloat162float(s));
}
const float gf = __bfloat162float(g), uf = __bfloat162float(u);
const float gcap = caps[0], ucap = caps[1];
float out = Tanh{}(gf / gcap) * gcap;
out = out * Sigmoid{}(gf);
out = out * (Tanh{}(uf / ucap) * ucap);
act[id] = __float2bfloat16_rn(out);
"""


@functools.cache
def _activation_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_activation",
        input_names=["y", "experts", "post", "caps"],
        output_names=["act"],
        source=_ACTIVATION,
        header=_MLX_OPS,
    )


def activation(y: mx.array, experts: mx.array, post, caps) -> mx.array:
    """``layers.Experts.activation`` of BF16 ``y`` [R, 2I] in one launch."""
    rows, width = y.shape
    count = rows * (width // 2)
    gcap, ucap = caps
    return _activation_kernel()(
        inputs=[
            y,
            experts.astype(mx.uint32),
            post if post is not None else mx.ones((1,), dtype=mx.float32),
            mx.concatenate([gcap.reshape(1), ucap.reshape(1)]),
        ],
        template=[("POST", int(post is not None))],
        grid=(-(-count // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width // 2)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ``layers.static_conv`` in one launch: the taps multiply and add separately
# in float32, round to the weight dtype, add the input and scale.
_CONV_HEADER = """
#include <cuda_bf16.h>
__device__ inline float ink_f(float x) { return x; }
__device__ inline float ink_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ inline void ink_put(float x, float* out) { *out = x; }
__device__ inline void ink_put(float x, __nv_bfloat16* out) { *out = __float2bfloat16_rn(x); }
__device__ inline float ink_round(float x, float) { return x; }
__device__ inline float ink_round(float x, __nv_bfloat16) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
"""
_CONV = """
const int L = x_shape[1], C = x_shape[2];
const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
if (id >= (size_t)(L + TAPS - 1) * C) return;
const int t = id / C, c = id % C;
auto xp = [&](int j) -> float {
  return j < TAPS - 1 ? state[(size_t)j * C + c] : ink_f(x[(size_t)(j - TAPS + 1) * C + c]);
};
if (t >= L) {
  // The last TAPS - 1 rows of [state, x] are the next state.
  window[(size_t)(t - L) * C + c] = xp(t);
  return;
}
float acc = __fmul_rn(ink_f(w[(size_t)c * TAPS]), xp(t));
#pragma unroll
for (int j = 1; j < TAPS; ++j) acc = __fadd_rn(acc, __fmul_rn(ink_f(w[(size_t)c * TAPS + j]), xp(t + j)));
const float rounded = ink_round(acc, w[0]);
ink_put(__fmul_rn(__fadd_rn(rounded, xp(t + TAPS - 1)), scale[0]), y + id);
"""
_CONV_KERNELS: dict = {}


def static_conv(x, w, state, scale: float):
    """(``layers.static_conv`` output, the next state [1, K - 1, C])."""
    key = (x.dtype, w.dtype)
    kernel = _CONV_KERNELS.get(key)
    if kernel is None:
        # A kernel object compiles for the dtypes of its first call only.
        names = "_".join(str(d).rsplit(".", 1)[-1] for d in key)
        kernel = _CONV_KERNELS[key] = mx.fast.cuda_kernel(
            name=f"omlx_inkling2_conv_{names}",
            input_names=["x", "w", "state", "scale"],
            output_names=["y", "window"],
            source=_CONV,
            header=_CONV_HEADER,
        )
    _, length, channels = x.shape
    taps = w.shape[-1]
    count = (length + taps - 1) * channels
    y, window = kernel(
        inputs=[x, w, state, mx.array([scale], dtype=mx.float32)],
        template=[("TAPS", taps)],
        grid=(-(-count // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (1, taps - 1, channels)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return y, window


def _format(bank) -> str | None:
    """The bank's carrier as the expert GEMMs name it, or None."""
    quant = bank.quant
    if quant["mode"] in _FORMATS:
        return quant["mode"] if bank.scales.dtype == mx.uint8 else None
    affine = (
        quant["mode"] == "affine"
        and quant["group_size"] == 64
        and quant["bits"] in _AFFINE_BITS
        and bank.biases is not None
        and bank.scales.dtype == mx.bfloat16
    )
    return f"affine{quant['bits']}" if affine else None


def covers_bank(bank, dtype: mx.Dtype) -> bool:
    """Whether an expert GEMM takes this bank for BF16 rows: a checked carrier
    in 128 tiles."""
    inputs = bank.weight.shape[2] * 32 // bank.quant["bits"]
    if dtype != mx.bfloat16 or inputs % 128 or bank.weight.shape[1] % 128:
        return False
    return f"experts:{_format(bank)}" in _on


def gather(x: mx.array, bank, experts: mx.array) -> mx.array:
    """``x`` [R, K] rows, sorted by expert, each through its expert of ``bank``."""
    if bank.quant["mode"] in _FORMATS:
        return minimax.gather(x, bank, experts)
    return deepseek.expert_down(x, experts, bank)


def weighted_sum(down: mx.array, inverse: mx.array, weights: mx.array) -> mx.array:
    return minimax.weighted_sum(down, inverse, weights)


def _relative(got: mx.array, want: mx.array) -> float:
    a, b = got.astype(mx.float32), want.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-30)).item()


def _check(name: str, ok: bool, detail: str) -> None:
    if ok:
        _on.add(name)
    else:
        _on.discard(name)
        logger.warning(
            "Inkling 2 %s kernel differs (%s); using plain ops", name, detail
        )


def enable(cfg, dtype: mx.Dtype) -> None:
    """Self-check each kernel against the plain ops of ``layers`` on small
    inputs of this model's shapes and keep the ones that agree."""
    if not available() or dtype != mx.bfloat16:
        return
    _on.clear()
    mx.random.seed(0)
    # Attention: a sliding and a full layer, queries past the first key tile.
    heads, kv, dim = cfg.swa_heads, cfg.swa_kv_heads, cfg.swa_head_dim
    worst = 0.0
    for window, extent, length, total in (
        (cfg.window, cfg.window, 150, 150 + cfg.window),
        (0, cfg.rel_extent, 150, 400),
    ):
        q = (mx.random.normal((1, length, heads, dim)) * 0.15).astype(dtype)
        k = (mx.random.normal((1, kv, total, dim)) * 0.15).astype(dtype)
        v = mx.random.normal((1, kv, total, dim)).astype(dtype)
        rel = (mx.random.normal((1, length, heads, extent)) * 0.5).astype(dtype)
        sink = mx.random.normal((1, length, heads)).astype(dtype)
        got = attention(q, k, v, rel, sink, dim**-0.5, window)
        want = layers.masked_attention(
            q.transpose(0, 2, 1, 3), k, v, rel, sink, dim**-0.5, window
        ).transpose(0, 2, 1, 3)
        worst = max(worst, _relative(got, want))
    _check("attention", worst <= 1e-2, f"relative error {worst:.2e}")
    # Depth mix: a full layer's three queries over every block.
    blocks = [
        layers.unit_rms(mx.random.normal((1, 5, cfg.hidden)))
        for _ in range(MIX_SOURCES - 1)
    ]
    partial = layers.unit_rms(mx.random.normal((1, 5, cfg.hidden)))
    query = mx.random.normal((3, cfg.hidden)) * 0.05
    scale = cfg.hidden**-0.5
    src = mx.stack([*blocks, partial], axis=-2)
    weights = mx.softmax((src @ query.T) * scale, axis=-2, precise=True)
    error = _relative(
        mix(blocks, partial, query, scale), weights.swapaxes(-1, -2) @ src
    )
    _check("mix", error <= 1e-5, f"relative error {error:.2e}")
    # Router: the same experts in the same order and the same weights.
    logits = mx.random.normal((64, cfg.experts)) * 2
    bias = mx.random.normal((cfg.experts,)) * 0.05
    inds, w = route_topk(logits, bias, cfg.top_k, cfg.route_scale)
    scores = mx.sigmoid(logits)
    order = mx.argsort(-(scores + bias), axis=-1)[:, : cfg.top_k]
    want_w = mx.take_along_axis(scores, order, axis=-1)
    want_w = want_w / (mx.sum(want_w, axis=-1, keepdims=True) + 1e-20) * cfg.route_scale
    same = mx.array_equal(inds, order.astype(mx.uint32)).item()
    error = _relative(w, want_w)
    _check("route", same and error <= 1e-6, f"same={same}, error {error:.2e}")
    # Router logits: the MiniMax kernel's sums, a few rows per block.
    x = mx.random.normal((70, cfg.hidden)).astype(dtype)
    columns = mx.random.normal((cfg.hidden, cfg.experts)) * 0.02
    same = mx.array_equal(router_logits(x, columns), minimax.router(x, columns))
    _check("router_logits", same.item(), "not equal to the MiniMax kernel")
    # Convs and the expert activation: equal to the plain ops.
    same = True
    for channels in (cfg.hidden, cfg.kv_heads * cfg.head_dim):
        x = mx.random.normal((1, 37, channels)).astype(dtype)
        w = (mx.random.normal((channels, cfg.conv_k)) * 0.3).astype(dtype)
        state = mx.random.normal((1, cfg.conv_k - 1, channels))
        got = static_conv(x, w, state, 0.4472)
        want = layers.plain_static_conv(x, w, state, 0.4472)
        same = same and all(mx.array_equal(a, b).item() for a, b in zip(got, want))
    _check("conv", same, "not equal to the plain ops")
    experts = layers.Experts.__new__(layers.Experts)
    experts.caps = (
        mx.array(cfg.gate_cap, dtype=mx.float32),
        mx.array(cfg.up_cap, dtype=mx.float32),
    )
    experts.w13 = type("Bank", (), {})()
    experts.w13.post = mx.random.uniform(0.001, 0.02, (cfg.experts,))
    y = (mx.random.normal((70, 2 * cfg.inter)) * 3).astype(dtype)
    picked = mx.random.randint(0, cfg.experts, (70,)).astype(mx.uint32)
    got = activation(y, picked, experts.w13.post, experts.caps)
    want = experts.plain_activation(y, picked)
    _check("activation", mx.array_equal(got, want).item(), "not equal to plain ops")
    # Expert GEMMs: each carrier against MLX's gather_qmm, which multiplies
    # with the weights the host's does (mx.dequantize of 3-bit codes differs).
    rows = mx.array([0] * 70 + [1] * 60, dtype=mx.uint32)
    x = mx.random.normal((130, 256)).astype(dtype)
    carriers = [{"mode": m} for m in _FORMATS]
    carriers += [{"mode": "affine", "bits": b, "group_size": 64} for b in _AFFINE_BITS]
    for spec in carriers:
        # BF16 weights, so affine scales and biases are BF16 as checkpoints hold them.
        w = (mx.random.normal((2, 128, 256)) * 0.05).astype(dtype)
        weight, scales, *biases = mx.quantize(w, **spec)
        named = {"b.weight": weight, "b.scales": scales}
        if biases:
            named["b.biases"] = biases[0]
        quant = layers.expert_spec(spec if spec["mode"] == "affine" else spec["mode"])
        bank = layers.Experts.Bank(named, "b", quant)
        want = bank(x[:, None], rows)[:, 0]
        error = _relative(gather(x, bank, rows), want)
        _check(f"experts:{_format(bank)}", error <= 1e-3, f"relative error {error:.2e}")
    # The weighted sum of each token's expert rows, a thread per 8 columns.
    if cfg.hidden <= 8192 and cfg.hidden % 8 == 0:
        down = mx.random.normal((70 * cfg.top_k, cfg.hidden)).astype(dtype)
        inverse = mx.random.permutation(70 * cfg.top_k).astype(mx.uint32)
        weights = mx.random.uniform(0.1, 2.0, (70, cfg.top_k))
        want = down[inverse].reshape(70, cfg.top_k, -1).astype(mx.float32)
        want = mx.sum(want * weights[..., None], axis=-2).astype(dtype)
        error = _relative(weighted_sum(down, inverse, weights), want)
        _check("weighted_sum", error <= 1e-3, f"relative error {error:.2e}")
