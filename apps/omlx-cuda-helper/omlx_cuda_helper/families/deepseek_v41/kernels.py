# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for the hot paths of DeepSeek V4.1 prefill.

A kernel runs only on CUDA after it reproduced its reference in ops.py or
quant.py on a small input. Kernels keep the host's Metal order of fp32 ops
(fused multiply-adds, short sums), so they may differ from an op chain in the
last fp32 bit.
"""

from __future__ import annotations

import logging
from functools import cache

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
// Eight consecutive values as fp32; p is 8-byte aligned.
inline void v41_load8(const float* p, float* v) {
  for (int i = 0; i < 8; ++i) v[i] = p[i];
}
inline void v41_load8(const __nv_bfloat16* p, float* v) {
  const unsigned long long* w = reinterpret_cast<const unsigned long long*>(p);
  for (int i = 0; i < 8; ++i)
    v[i] = __uint_as_float(unsigned((w[i / 4] >> (16 * (i % 4))) & 0xFFFFu) << 16);
}
// Order-preserving key: larger scores give larger keys, and -0 equals +0.
inline unsigned v41_score_key(float x) {
  const unsigned u = __float_as_uint(x == 0.0f ? 0.0f : x);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
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
  for (int i = 0; i < 32; ++i) v[i] = v41_float(x[base + i]);
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
  for (int i = 0; i < 32; ++i) v41_store(v[i], y + base + i);
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

# Scores held at once by the index scan, in elements of dots.
_INDEX_DOTS = 1 << 28


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


def index_topk(q, keys, weights, start, ratio, count):
    """ops.index_topk with the scores and the selection on CUDA kernels."""
    length, heads = q.shape[1], q.shape[2]
    n = keys.shape[0]
    count = min(count, n)
    if not count:
        return mx.zeros((1, length, 0), mx.int32)
    step = max(1, _INDEX_DOTS // (heads * n))
    scores_kernel = _kernel(
        "index_scores", _INDEX_SCORES, ("dots", "weights", "meta"), ("scores",)
    )
    select_kernel = _kernel("index_select", _INDEX_SELECT, ("scores", "meta"), ("ids",))
    out = []
    for begin in range(0, length, step):
        rows = min(step, length - begin)
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


def sparse_attention(q, window, pooled, wi, ci, sink, scale):
    """ops.sparse_attention with the key gather and the weights on kernels."""
    _, length, heads, width = q.shape
    rows = window.shape[0]
    ids = mx.where((wi >= 0) & (wi < rows), wi, -1)
    table = window
    if ci is not None and ci.shape[-1]:
        seen = (ci >= 0) & (ci < pooled.shape[0])
        ids = mx.concatenate([ids, mx.where(seen, ci + rows, -1)], 1)
        table = mx.concatenate([window, pooled], 0)
    ids = ids.astype(mx.int32)
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
