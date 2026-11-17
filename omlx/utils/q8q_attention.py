# SPDX-License-Identifier: Apache-2.0
"""Prefill attention over the affine8 KV cache with INT8 queries: Splash's q8q design as an MLX JIT kernel.

Design credit: Splash (https://github.com/incoai/splash, Apache-2.0), runtime/metal/kernels/common/paged_attention_tile.h
(splash_paged_attention_tile, splash_attention_page_softmax) and kernels/prefill/paged_attention.metal (prefill_attention_q8q_split).
This is an independent re-implementation of that design for oMLX's layout; no Splash code is compiled or linked.

One threadgroup (256 threads, 8 simdgroups) owns one KV head and ROWS=8 query rows: M = 8 rows x G=6 query heads of the
GQA group = 48 fused rows, one contiguous [M, D] block (queries are re-laid out KV-head-major [Hk][qL][G][D]).
Per 32-key page: S = Qi8 . Ki8 (Metal 4 MPP matmul2d, int8 x int8 -> int32, exact), stored to threadgroup memory;
four lanes per fused row apply key scale x query scale x softmax scale, run the online softmax (rescale of the
running output only when some row max grew), fold the value scale into P and store P as bf16; then
O += P(bf16) . Vi8 (MPP, fp32 accumulator, relaxed precision) with V read in place from the cache.
One history split (2048-row chunks), output normalized in the kernel (fp32 [Hk][qL][G][D]).

oMLX affine8 cache: Hadamard-rotated, signed int8 codes (uint32 words viewed as int8), one scale per (token, KV head),
token-major K and V [Hk][T][D]. PVT=True takes a dim-major V copy [Hk][D][T] (Splash's layout; an engine layout change).
"""

from functools import cache

import mlx.core as mx

HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

namespace q8q {

template <int G, int ROWS, int D, int N, bool QK8, bool PVT, bool SKIP, int LPR>
METAL_FUNC void tile(
    const device int8_t* qc, const device bfloat* qb, const device float* qs,
    const device int8_t* kc, const device float* ks,
    const device int8_t* vc, const device float* vs,
    device float* out, int qL, int T, int Tpad, float sm_scale,
    threadgroup float* scores_raw, threadgroup bfloat* probs,
    threadgroup float* row_max, threadgroup float* row_sum, threadgroup float* prev,
    threadgroup atomic_uint* rescale, uint kvh, uint tidx, uint tid,
    int page_lo, int page_hi, device float* stats_m, device float* stats_l) {
  constexpr int M = ROWS * G;
  constexpr int TPL = N / LPR;   // tokens per lane
  static_assert(N == LPR * TPL && TPL % 4 == 0, "lanes span a page");
  static_assert(LPR * M <= 256, "softmax lanes fit the threadgroup");
  using Score = metal::conditional_t<QK8, int, float>;
  using QE = metal::conditional_t<QK8, int8_t, bfloat>;
  threadgroup Score* scores = reinterpret_cast<threadgroup Score*>(scores_raw);

  const int committed = T - qL;                 // the chunk is the last qL keys
  const int first_pos = committed + int(tidx) * ROWS;
  const int visible = first_pos + ROWS;
  const int pages = (visible + N - 1) / N;
  const ulong qoff = (ulong(kvh) * ulong(qL) + ulong(tidx) * ROWS) * G;

  // Softmax lane geometry and its row's score scale (query scale folded in).
  const uint fr = tid / LPR;
  const uint column = (tid % LPR) * TPL;
  const bool lane = tid < uint(LPR * M);
  float score_scale = sm_scale;
  int limit = 0;
  if (lane) {
    if (QK8) score_scale *= qs[qoff + fr];
    limit = first_pos + int(fr / G) + 1;        // causal end of this row
  }
  if (tid < uint(M)) { row_max[tid] = -INFINITY; row_sum[tid] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device QE* qptr;
  if constexpr (QK8) qptr = (device QE*)(qc + qoff * D);
  else qptr = (device QE*)(qb + qoff * D);
  using QT = tensor<device QE, dextents<int, 2>, tensor_inline>;
  using KT = tensor<device int8_t, dextents<int, 2>, tensor_inline>;
  using ST = tensor<threadgroup Score, dextents<int, 2>, tensor_inline>;
  using PT = tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>;
  using OT = tensor<device float, dextents<int, 2>, tensor_inline>;
  QT qt(qptr, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  KT kt((device int8_t*)(kc + ulong(kvh) * ulong(Tpad) * D), dextents<int, 2>{D, Tpad}, array<int, 2>{1, D});
  KT vt((device int8_t*)(vc + ulong(kvh) * ulong(Tpad) * D), dextents<int, 2>{PVT ? Tpad : D, PVT ? D : Tpad},
        array<int, 2>{1, PVT ? Tpad : D});
  ST st(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  PT pt(probs, dextents<int, 2>{N, M}, array<int, 2>{1, N});

  constexpr auto qk_desc = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_desc = matmul2d_descriptor(M, D, N, false, PVT, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<8>> qk;
  matmul2d<pv_desc, execution_simdgroups<8>> pv;
  constexpr int VE0 = PVT ? N : D;
  constexpr int VE1 = PVT ? D : N;
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = kt.template slice<D, N>(0, 0);
  auto v0 = vt.template slice<VE0, VE1>(0, 0);
  auto p0 = pt.template slice<N, M>(0, 0);
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
  const bool full = uint(running.get_capacity()) * 256u == uint(M * D);
  for (ushort i = 0; i < running.get_capacity(); ++i)
    if (full || running.is_valid_element(i)) running[i] = 0.0f;

  const device float* ksp = ks + ulong(kvh) * ulong(Tpad);
  const device float* vsp = vs + ulong(kvh) * ulong(Tpad);
  const int page_end = min(pages, page_hi);
  for (int page = page_lo; page < page_end; ++page) {
    const int t0 = page * N;
    auto kp = kt.template slice<D, N>(0, t0);
    auto sc = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), Score>();
    qk.run(q0, kp, sc);
    sc.store(st.template slice<N, M>(0, 0));
    if (tid == 0) atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane) {
      const int token = t0 + int(column);
      threadgroup const vec<Score, 4>* s4 = reinterpret_cast<threadgroup const vec<Score, 4>*>(scores + fr * N + column);
      const device float4* k4 = reinterpret_cast<const device float4*>(ksp + token);
      float s[TPL];
      for (int q = 0; q < TPL / 4; ++q) {
        float4 x = float4(s4[q]) * k4[q] * score_scale;
        s[4 * q] = x.x; s[4 * q + 1] = x.y; s[4 * q + 2] = x.z; s[4 * q + 3] = x.w;
      }
      float lmax = -INFINITY;
      for (int j = 0; j < TPL; ++j) { s[j] = token + j < limit ? s[j] : -INFINITY; lmax = max(lmax, s[j]); }
      for (int o = 1; o < LPR; o <<= 1) lmax = max(lmax, simd_shuffle_xor(lmax, ushort(o)));
      const float pmax = row_max[fr];
      const float nmax = max(pmax, lmax);
      float p[TPL]; float lsum = 0.0f;
      for (int j = 0; j < TPL; ++j) { p[j] = token + j < limit ? fast::exp(s[j] - nmax) : 0.0f; lsum += p[j]; }
      for (int o = 1; o < LPR; o <<= 1) lsum += simd_shuffle_xor(lsum, ushort(o));
      if (column == 0) {
        const float scale = (nmax == -INFINITY || nmax == pmax) ? 1.0f : fast::exp(pmax - nmax);
        prev[fr] = scale;
        row_sum[fr] = row_sum[fr] * scale + lsum;
        row_max[fr] = nmax;
        if (scale != 1.0f) atomic_store_explicit(rescale, 1u, memory_order_relaxed);
      }
      const device float4* v4 = reinterpret_cast<const device float4*>(vsp + token);
      threadgroup bfloat4* p4 = reinterpret_cast<threadgroup bfloat4*>(probs + fr * N + column);
      for (int q = 0; q < TPL / 4; ++q) {
        const float4 vv = v4[q];
        const int b = 4 * q;
        p4[q] = bfloat4(float4(token + b < limit ? p[b] * vv.x : 0.0f, token + b + 1 < limit ? p[b + 1] * vv.y : 0.0f,
                               token + b + 2 < limit ? p[b + 2] * vv.z : 0.0f, token + b + 3 < limit ? p[b + 3] * vv.w : 0.0f));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!SKIP || atomic_load_explicit(rescale, memory_order_relaxed)) {
      for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!full && !running.is_valid_element(i)) continue;
        auto c = running.get_multidimensional_index(i);
        running[i] *= prev[c[1]];
      }
    }
    auto vp = vt.template slice<VE0, VE1>(PVT ? t0 : 0, PVT ? 0 : t0);
    pv.run(p0, vp, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (ushort i = 0; i < running.get_capacity(); ++i) {
    if (!full && !running.is_valid_element(i)) continue;
    auto c = running.get_multidimensional_index(i);
    running[i] /= row_sum[c[1]];
  }
  OT ot(out + qoff * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(ot.template slice<D, M>(0, 0));
  if (stats_m != nullptr && tid < uint(M)) { stats_m[qoff + tid] = row_max[tid]; stats_l[qoff + tid] = row_sum[tid]; }
}
}  // namespace q8q
"""

SOURCE = r"""
  constexpr int M = {ROWS} * {G};
  alignas(16) threadgroup float scores[M * 32];
  alignas(16) threadgroup bfloat probs[M * 32];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float prev[M];
  threadgroup atomic_uint rescale;
  const uint kvh_i = {ORDER} ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.x;
  const uint tile_i = {ORDER} ? threadgroup_position_in_grid.x : threadgroup_position_in_grid.y;
  const int split = int(threadgroup_position_in_grid.z);
  const int hist_pages = (prm[1] - prm[0]) / 32;          // pages wholly before the chunk
  const int cut = {SPLIT} > 1 ? (hist_pages / {SPLIT}) * split : 0;
  const int cut_hi = ({SPLIT} > 1 && split + 1 < {SPLIT}) ? (hist_pages / {SPLIT}) * (split + 1) : (1 << 30);
  const ulong part = ulong(prm[3]);                           // elements per split of the output
  const ulong spart = ulong(prm[4]);                          // elements per split of the stats
  q8q::tile<{G}, {ROWS}, 256, 32, {QK8}, {PVT}, {SKIP}, {LPR}>(
      qc, reinterpret_cast<const device bfloat*>(qb), qs, kc, ks, vc, vs, out + part * split,
      prm[0], prm[1], prm[2], scl[0], scores, probs, row_max, row_sum, prev, &rescale,
      kvh_i, tile_i, thread_index_in_threadgroup, cut, cut_hi,
      {SPLIT} > 1 ? sm + spart * split : nullptr, {SPLIT} > 1 ? sl + spart * split : nullptr);
"""

INPUTS = ["qc", "qb", "qs", "kc", "ks", "vc", "vs", "prm", "scl"]
OUTPUTS = ["out", "sm", "sl"]


def _b(x):
    return "true" if x else "false"


@cache
def kernel(g=6, rows=8, qk8=True, pvt=False, skip=True, split=1):
    """The JIT kernel for ``g`` query heads per KV head and ``rows`` rows per tile.

    ``qk8``: INT8 queries (else bf16 queries x int8 keys); ``pvt``: dim-major V;
    ``skip``: rescale the running output only when a row max grew; ``split``:
    threadgroups per tile, each over its share of the history pages. The
    engine uses the defaults apart from ``split``.
    """
    src = SOURCE
    params = dict(
        G=g,
        ROWS=rows,
        QK8=_b(qk8),
        PVT=_b(pvt),
        SKIP=_b(skip),
        ORDER=0,
        LPR=4,
        SPLIT=int(split),
    )
    for k, v in params.items():
        src = src.replace("{" + k + "}", str(v))
    name = (
        f"q8q_v2_g{g}_r{rows}_qk8{int(qk8)}_pvt{int(pvt)}_skip{int(skip)}_s{int(split)}"
    )
    return mx.fast.metal_kernel(
        name=name,
        input_names=INPUTS,
        output_names=["out", "sm", "sl"],
        header=HEADER,
        source=src,
        ensure_row_contiguous=True,
    )


_PLACEHOLDERS = {}


def _placeholder(dtype):
    if dtype not in _PLACEHOLDERS:
        _PLACEHOLDERS[dtype] = mx.zeros((64,), dtype=dtype)
    return _PLACEHOLDERS[dtype]


def prepare_kv(codes_u32, scales):
    """Affine8 indices [1, Hk, T, D/4] uint32 and scales [1, Hk, T] -> int8 codes
    [Hk, Tpad, D], fp32 scales [Hk, Tpad] and T (Tpad = T rounded up to a page)."""
    _, heads, tokens, words = codes_u32.shape
    dim = words * 4
    codes = codes_u32.view(mx.int8).reshape(heads, tokens, dim)
    scale = scales.astype(mx.float32).reshape(heads, tokens)
    padded = (tokens + 31) // 32 * 32
    if padded != tokens:
        codes = mx.pad(codes, [(0, 0), (0, padded - tokens), (0, 0)])
        scale = mx.pad(scale, [(0, 0), (0, padded - tokens)])
    return codes, scale, tokens


def quant_queries(rq, kv_heads):
    """Rotated queries [1, H, rows, D] -> KV-head-major int8 codes [Hk, rows, G, D]
    and fp32 scales [Hk, rows, G] (absmax / 127 per row)."""
    _, heads, rows, dim = rq.shape
    group = heads // kv_heads
    x = rq.reshape(kv_heads, group, rows, dim).transpose(0, 2, 1, 3)
    x = x.astype(mx.float32)
    s = mx.maximum(mx.max(mx.abs(x), axis=-1) / 127.0, 1e-30)
    c = mx.clip(mx.round(x / s[..., None]), -127, 127).astype(mx.int8)
    return c, s


def run(rq, k, v, scale, *, rows=8, split=1):
    """Causal attention of rotated bf16 queries [1, H, n, 256] (the last n keys)
    over ``prepare_kv`` operands; fp32 output [1, H, n, 256] in the rotated space.

    ``split`` > 1 runs that many threadgroups per tile over equal shares of the
    history and combines their normalized partials by their softmax weights.
    """
    _, heads, n, dim = rq.shape
    kc, ks, tokens = k
    vc, vs, _ = v
    kv_heads = kc.shape[0]
    padded = ks.shape[1]
    group = heads // kv_heads
    assert dim == 256 and n % rows == 0 and n <= tokens
    qc, qs = quant_queries(rq, kv_heads)
    part = kv_heads * n * group * dim
    spart = kv_heads * n * group
    prm = mx.array([n, tokens, padded, part, spart, 0, 0, 0], dtype=mx.int32)
    scl = mx.array([scale] * 8, dtype=mx.float32)
    out, row_max, row_sum = kernel(group, rows, split=split)(
        inputs=[qc, _placeholder(mx.bfloat16), qs, kc, ks, vc, vs, prm, scl],
        grid=(256 * kv_heads, n // rows, split),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (split, kv_heads, n, group, dim),
            (split, kv_heads, n, group),
            (split, kv_heads, n, group),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    if split > 1:
        top = mx.max(row_max, axis=0, keepdims=True)
        weight = mx.exp(row_max - top) * row_sum
        out = (
            mx.sum(out * weight[..., None], axis=0) / mx.sum(weight, axis=0)[..., None]
        )
    else:
        out = out[0]
    return out.transpose(0, 2, 1, 3).reshape(1, heads, n, dim)
