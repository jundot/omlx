# SPDX-License-Identifier: Apache-2.0
"""Causal attention of the last rows of a long history over the affine8 KV cache, on the Metal 4 tensor ops.

Design credit: Splash (https://github.com/incoai/splash, Apache-2.0), runtime/metal/kernels/common/paged_attention_tile.h
(splash_paged_attention_tile, splash_attention_page_softmax) and kernels/prefill/paged_attention.metal (prefill_attention_q8q_split).
This is an independent re-implementation of that design for oMLX's layout; no Splash code is compiled or linked.

One tile: a threadgroup (256 threads, 8 simdgroups) owns one KV head and a block of rows, fused with the GQA group of that
KV head into one [M, D] block of query rows (M = rows x G, padded to a multiple of 8). Per page of keys: S = Q . K (MPP
matmul2d; INT8 queries x INT8 keys with int32 scores, or bf16 queries x INT8 keys with fp32 scores), stored to threadgroup
memory; a few lanes per fused row apply the key scale (and the query scale), run the online softmax (the running output is
rescaled only when some row max grew), fold the value scale into P and store P as bf16; then O += P . V (MPP, fp32
accumulator) with V read in place. A tile may cover only a share of the history pages (a split); the splits of a tile
write their partials with their row max and sum, and a combine kernel merges them.

Two entry points share the tile:
- ``prefill``: a prefill chunk (a multiple of 8 rows), INT8 queries, 32-key pages, over K and V copied by ``prepare_kv``.
- ``verify``: 2-16 rows (speculative verify, short tails), bf16 queries, 64-key pages, K and V read in place through the
  strided views ``update_and_fetch`` returns, the history split across many threadgroups so the few rows fill the GPU.

oMLX affine8 cache: Hadamard-rotated, signed int8 codes (uint32 words viewed as int8), one scale per (token, KV head),
token-major K and V. Queries come in rotated; the output is fp32 in the rotated space.
"""

from functools import cache

import mlx.core as mx

D = 256
MAX_FUSED_ROWS = (
    48  # rows x G per tile; more no longer keeps the accumulator in registers
)

HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

namespace affine8_tile {

// q: the tile's fused query rows (M x D), qs: their INT8 scales (QK8 only).
// kb / vb: the KV head's codes, token stride kts / vts bytes, T tokens. ksp / vsp: the KV head's scales, token stride sst.
// Rows of the tile sit at positions first_pos .. first_pos + valid - 1 of a causal history; fused row f is row f / G.
// The tile reads pages [page_lo, page_hi) of the pages it can see and writes its output to out (M x D): normalized,
// or, when stats_m is set (a split), unnormalized with its row max and sum.
template <int G, int ROWS, int M, int D, int N, bool QK8, int LPR, typename ST, bool SVEC, bool NORM>
METAL_FUNC void tile(
    const device int8_t* qc, const device bfloat* qb, const device float* qs,
    const device int8_t* kb, int kts, const device int8_t* vb, int vts, int T,
    const device ST* ksp, const device ST* vsp, int sst,
    device float* out, device float* stats_m, device float* stats_l,
    int first_pos, int valid, int page_lo, int page_hi, float sm_scale,
    threadgroup float* scores_raw, threadgroup bfloat* probs,
    threadgroup float* row_max, threadgroup float* row_sum, threadgroup float* prev,
    threadgroup atomic_uint* rescale, uint tid) {
  constexpr int TPL = N / LPR;   // tokens per lane
  static_assert(N == LPR * TPL && TPL % 4 == 0, "lanes span a page");
  static_assert(LPR * M <= 256, "softmax lanes fit the threadgroup");
  static_assert(M % 8 == 0 && ROWS * G <= M, "padded tile");
  using Score = metal::conditional_t<QK8, int, float>;
  using QE = metal::conditional_t<QK8, int8_t, bfloat>;
  threadgroup Score* scores = reinterpret_cast<threadgroup Score*>(scores_raw);

  const int pages = (first_pos + valid + N - 1) / N;

  // Softmax lane geometry and its row's score scale (query scale folded in).
  const uint fr = tid / LPR;
  const uint column = (tid % LPR) * TPL;
  const bool lane = tid < uint(LPR * M);
  float score_scale = sm_scale;
  int limit = 0;                                 // causal end of this fused row; 0 for padding
  if (lane) {
    const int r = int(fr) / G;
    if (r < valid) {
      limit = first_pos + r + 1;
      if (QK8) score_scale *= qs[fr];
    }
  }
  if (tid < uint(M)) { row_max[tid] = -INFINITY; row_sum[tid] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device QE* qptr;
  if constexpr (QK8) qptr = (device QE*)qc;
  else qptr = (device QE*)qb;
  using QT = tensor<device QE, dextents<int, 2>, tensor_inline>;
  using KT = tensor<device int8_t, dextents<int, 2>, tensor_inline>;
  using SCT = tensor<threadgroup Score, dextents<int, 2>, tensor_inline>;
  using PT = tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>;
  using OT = tensor<device float, dextents<int, 2>, tensor_inline>;
  QT qt(qptr, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  KT kt((device int8_t*)kb, dextents<int, 2>{D, T}, array<int, 2>{1, kts});
  KT vt((device int8_t*)vb, dextents<int, 2>{D, T}, array<int, 2>{1, vts});
  SCT st(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  PT pt(probs, dextents<int, 2>{N, M}, array<int, 2>{1, N});

  constexpr auto qk_desc = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto pv_desc = matmul2d_descriptor(M, D, N, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<8>> qk;
  matmul2d<pv_desc, execution_simdgroups<8>> pv;
  auto q0 = qt.template slice<D, M>(0, 0);
  auto k0 = kt.template slice<D, N>(0, 0);
  auto v0 = vt.template slice<D, N>(0, 0);
  auto p0 = pt.template slice<N, M>(0, 0);
  auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(v0), float>();
  const bool full = uint(running.get_capacity()) * 256u == uint(M * D);
  for (ushort i = 0; i < running.get_capacity(); ++i)
    if (full || running.is_valid_element(i)) running[i] = 0.0f;

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
      float s[TPL];
      if constexpr (SVEC) {
        // Contiguous fp32 scales padded to a page: vector loads.
        const device float4* k4 = reinterpret_cast<const device float4*>(ksp + token);
        for (int q = 0; q < TPL / 4; ++q) {
          float4 x = float4(s4[q]) * k4[q] * score_scale;
          s[4 * q] = x.x; s[4 * q + 1] = x.y; s[4 * q + 2] = x.z; s[4 * q + 3] = x.w;
        }
        for (int j = 0; j < TPL; ++j) s[j] = token + j < limit ? s[j] : -INFINITY;
      } else {
        for (int q = 0; q < TPL / 4; ++q) {
          float4 x = float4(s4[q]) * score_scale;
          s[4 * q] = x.x; s[4 * q + 1] = x.y; s[4 * q + 2] = x.z; s[4 * q + 3] = x.w;
        }
        for (int j = 0; j < TPL; ++j) s[j] = token + j < limit ? s[j] * float(ksp[(token + j) * sst]) : -INFINITY;
      }
      float lmax = -INFINITY;
      for (int j = 0; j < TPL; ++j) lmax = max(lmax, s[j]);
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
      threadgroup vec<bfloat, 4>* p4 = reinterpret_cast<threadgroup vec<bfloat, 4>*>(probs + fr * N + column);
      if constexpr (SVEC) {
        const device float4* v4 = reinterpret_cast<const device float4*>(vsp + token);
        for (int q = 0; q < TPL / 4; ++q) {
          const float4 vv = v4[q];
          const int b = 4 * q;
          p4[q] = vec<bfloat, 4>(float4(token + b < limit ? p[b] * vv.x : 0.0f, token + b + 1 < limit ? p[b + 1] * vv.y : 0.0f,
                                        token + b + 2 < limit ? p[b + 2] * vv.z : 0.0f, token + b + 3 < limit ? p[b + 3] * vv.w : 0.0f));
        }
      } else {
        for (int q = 0; q < TPL / 4; ++q) {
          const int b = 4 * q;
          float4 pp;
          for (int e = 0; e < 4; ++e)
            pp[e] = token + b + e < limit ? p[b + e] * float(vsp[(token + b + e) * sst]) : 0.0f;
          p4[q] = vec<bfloat, 4>(pp);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
      for (ushort i = 0; i < running.get_capacity(); ++i) {
        if (!full && !running.is_valid_element(i)) continue;
        auto c = running.get_multidimensional_index(i);
        running[i] *= prev[c[1]];
      }
    }
    auto vp = vt.template slice<D, N>(0, t0);
    pv.run(p0, vp, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // A whole history normalizes here; a split leaves that to the combine, which also weighs it.
  if constexpr (NORM) {
    for (ushort i = 0; i < running.get_capacity(); ++i) {
      if (!full && !running.is_valid_element(i)) continue;
      auto c = running.get_multidimensional_index(i);
      running[i] /= row_sum[c[1]];   // padding rows (sum 0) are dropped by the caller
    }
  }
  OT ot(out, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  running.store(ot.template slice<D, M>(0, 0));
  if (!NORM && tid < uint(M)) { stats_m[tid] = row_max[tid]; stats_l[tid] = row_sum[tid]; }
}

// Pages [lo, hi) of split s of S: the history pages (wholly before the rows) in equal shares; the last split also
// takes the pages of the rows themselves.
inline int2 split_pages(int history_pages, int s, int S) {
  const int lo = (history_pages * s) / S;
  const int hi = s + 1 < S ? (history_pages * (s + 1)) / S : (1 << 30);
  return int2(lo, hi);
}

}  // namespace affine8_tile
"""

# Grid: x = 256 threads per KV head, y = tile, z = split. prm: n (rows), T (keys), extent of K / V (readable tokens),
# split count. K, V and their scales are read through their strides; AX is the KV-head axis (0 for prepared copies
# [Hk, T, ...], 1 for the cache views [1, Hk, T, ...]).
_ENTRY = r"""
  constexpr int M = {M};
  alignas(16) threadgroup float scores[M * {N}];
  alignas(16) threadgroup bfloat probs[M * {N}];
  threadgroup float row_max[M];
  threadgroup float row_sum[M];
  threadgroup float prev[M];
  threadgroup atomic_uint rescale;
  // SX: splits on grid x (neighbouring threadgroups read neighbouring pages of one KV head), else KV heads on x.
  const uint kvh = {SX} ? threadgroup_position_in_grid.z : threadgroup_position_in_grid.x;
  const uint ti = threadgroup_position_in_grid.y;
  const int s = int({SX} ? threadgroup_position_in_grid.x : threadgroup_position_in_grid.z);
  const uint kv_heads = {SX} ? threadgroups_per_grid.z : threadgroups_per_grid.x;
  const int n = prm[0], T = prm[1], ext = prm[2], S = prm[3];
  const uint tiles = threadgroups_per_grid.y;
  const int committed = T - n;                   // the rows are the last n keys
  const int first_pos = committed + int(ti) * {ROWS};
  const int valid = min({ROWS}, n - int(ti) * {ROWS});
  const int2 pr = affine8_tile::split_pages(committed / {N}, s, S);
  const ulong fused = (ulong(kvh) * tiles + ti) * M;   // the tile's first fused row
  const ulong rows_all = ulong(kv_heads) * tiles * M;
  const int kts = int(kc_strides[{AX} + 1]) * int(sizeof(kc[0]));
  const int vts = int(vc_strides[{AX} + 1]) * int(sizeof(vc[0]));
  const device int8_t* kb = reinterpret_cast<const device int8_t*>(kc) + ulong(kvh) * ulong(kc_strides[{AX}]) * sizeof(kc[0]);
  const device int8_t* vb = reinterpret_cast<const device int8_t*>(vc) + ulong(kvh) * ulong(vc_strides[{AX}]) * sizeof(vc[0]);
  affine8_tile::tile<{G}, {ROWS}, M, 256, {N}, {QK8}, {LPR}, {ST}, {SVEC}, {NORM}>(
      qc + fused * 256, reinterpret_cast<const device bfloat*>(qb) + fused * 256, qs + fused,
      kb, kts, vb, vts, ext,
      ks + ulong(kvh) * ulong(ks_strides[{AX}]), vs + ulong(kvh) * ulong(vs_strides[{AX}]), int(ks_strides[{AX} + 1]),
      out + (ulong(s) * rows_all + fused) * 256,
      sm + ulong(s) * rows_all + fused, sl + ulong(s) * rows_all + fused,
      first_pos, valid, pr.x, pr.y, scl[0],
      scores, probs, row_max, row_sum, prev, &rescale, thread_index_in_threadgroup);
"""

# Merge the splits' partials (unnormalized, with their row max and sum) into [1, H, n, D]: thread (d, fused row, tile of a
# KV head). prm: S, n, tiles.
_COMBINE = r"""
  const uint d = thread_position_in_grid.x;
  const uint f = thread_position_in_grid.y;
  const uint ht = thread_position_in_grid.z;
  const int S = prm[0], n = prm[1], tiles = prm[2];
  const int kvh = int(ht) / tiles, ti = int(ht) % tiles;
  const int r = int(f) / {G}, g = int(f) % {G};
  const int row = ti * {ROWS} + r;
  if (r >= {ROWS} || row >= n) return;
  const ulong R = ulong(threads_per_grid.z) * {M};
  const ulong fr = ulong(ht) * {M} + f;
  float top = -INFINITY;
  for (int s = 0; s < S; ++s) top = max(top, sm[ulong(s) * R + fr]);
  float acc = 0.0f, den = 0.0f;
  for (int s = 0; s < S; ++s) {
    const ulong b = ulong(s) * R + fr;
    const float w = sl[b] > 0.0f ? fast::exp(sm[b] - top) : 0.0f;
    den += w * sl[b];
    acc += w * part[b * 256 + d];
  }
  y[(ulong(kvh * {G} + g) * n + row) * 256 + d] = den > 0.0f ? acc / den : 0.0f;
"""

# bf16 queries [1, H, n, D] -> KV-head-major fused rows [Hk][tiles][M][D], zero padded: one simdgroup per fused row.
_LAYOUT = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint f = threadgroup_position_in_grid.y;
  const uint ht = threadgroup_position_in_grid.z;
  const int n = prm[0], tiles = prm[1];
  const int kvh = int(ht) / tiles, ti = int(ht) % tiles;
  const int r = int(f) / {G}, g = int(f) % {G};
  const int row = ti * {ROWS} + r;
  device bfloat* dst = fq + (ulong(ht) * {M} + f) * 256 + lane * 8;
  if (r >= {ROWS} || row >= n) {
    for (int j = 0; j < 8; ++j) dst[j] = bfloat(0.0f);
    return;
  }
  const device bfloat* src = q + (ulong(kvh * {G} + g) * n + row) * 256 + lane * 8;
  for (int j = 0; j < 8; ++j) dst[j] = src[j];
"""


def _fill(src, **kw):
    for k, v in kw.items():
        src = src.replace("{" + k + "}", str(v))
    return src


@cache
def _kernel(g, rows, m, page, qk8, scale_type, axis, split_x, scale_vec, norm):
    lpr = 4 if 4 * m <= 256 else 2
    name = f"omlx_affine8_tile_g{g}_r{rows}_m{m}_n{page}_{'q8' if qk8 else 'bf'}_{scale_type}_a{axis}_x{int(split_x)}_v{int(scale_vec)}_n{int(norm)}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=["qc", "qb", "qs", "kc", "ks", "vc", "vs", "prm", "scl"],
        output_names=["out", "sm", "sl"],
        header=HEADER,
        source=_fill(
            _ENTRY,
            G=g,
            ROWS=rows,
            M=m,
            N=page,
            QK8="true" if qk8 else "false",
            LPR=lpr,
            ST=scale_type,
            AX=axis,
            SX="true" if split_x else "false",
            SVEC="true" if scale_vec else "false",
            NORM="true" if norm else "false",
        ),
        ensure_row_contiguous=False,
    )


@cache
def _combine_kernel(g, rows, m):
    return mx.fast.metal_kernel(
        name=f"omlx_affine8_tile_combine_g{g}_r{rows}_m{m}",
        input_names=["part", "sm", "sl", "prm"],
        output_names=["y"],
        source=_fill(_COMBINE, G=g, ROWS=rows, M=m),
    )


@cache
def _layout_kernel(g, rows, m):
    return mx.fast.metal_kernel(
        name=f"omlx_affine8_tile_layout_g{g}_r{rows}_m{m}",
        input_names=["q", "prm"],
        output_names=["fq"],
        source=_fill(_LAYOUT, G=g, ROWS=rows, M=m),
    )


_PLACEHOLDERS = {}


def _placeholder(dtype):
    if dtype not in _PLACEHOLDERS:
        _PLACEHOLDERS[dtype] = mx.zeros((64,), dtype=dtype)
    return _PLACEHOLDERS[dtype]


def tile_rows(n: int, group: int) -> int:
    """Rows per tile for ``n`` rows and a GQA group of ``group`` query heads."""
    return max(1, min(n, MAX_FUSED_ROWS // group))


def _fused_layout(x, kv_heads, rows, m):
    """[1, H, n, ...] (or [H, n] for scales) -> KV-head-major fused rows [Hk, tiles, M, ...], zero padded."""
    heads, n = x.shape[1], x.shape[2]
    group = heads // kv_heads
    tail = x.shape[3:]
    x = x.reshape(kv_heads, group, n, *tail)
    x = x.transpose(0, 2, 1, *range(3, 3 + len(tail)))
    tiles = -(-n // rows)
    pad_rows = tiles * rows - n
    if pad_rows:
        x = mx.pad(x, [(0, 0), (0, pad_rows)] + [(0, 0)] * (1 + len(tail)))
    x = x.reshape(kv_heads, tiles, rows * group, *tail)
    if m > rows * group:
        x = mx.pad(x, [(0, 0), (0, 0), (0, m - rows * group)] + [(0, 0)] * len(tail))
    return mx.contiguous(x), tiles


def _launch(
    q,
    qscales,
    keys,
    values,
    scale,
    *,
    kv_heads,
    n,
    tokens,
    extent,
    axis,
    rows,
    page,
    qk8,
    split,
    split_x,
    scale_vec,
):
    """One launch of the tile over rows ``q`` [1, H, n, D] (int8 codes with ``qscales`` [1, H, n] if ``qk8``, else
    bf16). keys / values: (codes, scales) with the KV head on ``axis``."""
    heads = q.shape[1]
    group = heads // kv_heads
    m = (rows * group + 7) // 8 * 8
    tiles = -(-n // rows)
    if qk8:
        fq = _fused_layout(q, kv_heads, rows, m)[0]
        fqs = _fused_layout(qscales, kv_heads, rows, m)[0]
    else:
        (fq,) = _layout_kernel(group, rows, m)(
            inputs=[mx.contiguous(q), mx.array([n, tiles], dtype=mx.int32)],
            grid=(32, m, kv_heads * tiles),
            threadgroup=(32, 1, 1),
            output_shapes=[(kv_heads * tiles * m * D,)],
            output_dtypes=[mx.bfloat16],
        )
        fqs = _placeholder(mx.float32)
    kcodes, kscales = keys
    vcodes, vscales = values
    prm = mx.array([n, tokens, extent, split], dtype=mx.int32)
    scl = mx.array([scale], dtype=mx.float32)
    scale_type = "half" if kscales.dtype == mx.float16 else "float"
    rows_all = kv_heads * tiles * m
    out, row_max, row_sum = _kernel(
        group, rows, m, page, qk8, scale_type, axis, split_x, scale_vec, split == 1
    )(
        inputs=[
            fq if qk8 else _placeholder(mx.int8),
            _placeholder(mx.bfloat16) if qk8 else fq,
            fqs,
            kcodes,
            kscales,
            vcodes,
            vscales,
            prm,
            scl,
        ],
        grid=(
            (256 * split, tiles, kv_heads)
            if split_x
            else (256 * kv_heads, tiles, split)
        ),
        threadgroup=(256, 1, 1),
        output_shapes=[(split, rows_all, D), (split, rows_all), (split, rows_all)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    if split > 1:
        (y,) = _combine_kernel(group, rows, m)(
            inputs=[out, row_max, row_sum, mx.array([split, n, tiles], dtype=mx.int32)],
            grid=(D, m, kv_heads * tiles),
            threadgroup=(256, 1, 1),
            output_shapes=[(1, heads, n, D)],
            output_dtypes=[mx.float32],
        )
        return y
    out = out.reshape(kv_heads, tiles, m, D)[:, :, : rows * group]
    out = out.reshape(kv_heads, tiles * rows, group, D)[:, :n]
    return out.transpose(0, 2, 1, 3).reshape(1, heads, n, D)


# ---- prefill: INT8 queries, 32-key pages, K and V copied to contiguous codes --------------------------------------

PREFILL_ROWS = 8
PREFILL_PAGE = 32


def prepare_kv(codes_u32, scales):
    """Affine8 indices [1, Hk, T, D/4] uint32 and scales [1, Hk, T] -> int8 codes [Hk, Tpad, D], fp32 scales
    [Hk, Tpad] and T (Tpad = T rounded up to a page)."""
    _, heads, tokens, words = codes_u32.shape
    dim = words * 4
    codes = codes_u32.view(mx.int8).reshape(heads, tokens, dim)
    scale = scales.astype(mx.float32).reshape(heads, tokens)
    padded = (tokens + PREFILL_PAGE - 1) // PREFILL_PAGE * PREFILL_PAGE
    if padded != tokens:
        codes = mx.pad(codes, [(0, 0), (0, padded - tokens), (0, 0)])
        scale = mx.pad(scale, [(0, 0), (0, padded - tokens)])
    return mx.contiguous(codes), mx.contiguous(scale), tokens


def quant_queries(rq):
    """Rotated queries [1, H, rows, D] -> int8 codes and fp32 scales [1, H, rows] (absmax / 127 per row)."""
    x = rq.astype(mx.float32)
    s = mx.maximum(mx.max(mx.abs(x), axis=-1) / 127.0, 1e-30)
    c = mx.clip(mx.round(x / s[..., None]), -127, 127).astype(mx.int8)
    return c, s


def prefill(rq, k, v, scale, *, split=1):
    """Causal attention of rotated bf16 queries [1, H, n, 256] (the last n keys, n a multiple of 8) over
    ``prepare_kv`` operands; fp32 output [1, H, n, 256] in the rotated space. ``split`` > 1 runs that many
    threadgroups per tile over equal shares of the history and merges their partials."""
    _, heads, n, dim = rq.shape
    kc, ks, tokens = k
    vc, vs, _ = v
    kv_heads, padded = int(kc.shape[0]), int(kc.shape[1])
    assert dim == D and n % PREFILL_ROWS == 0 and n <= tokens
    qc, qs = quant_queries(rq)
    return _launch(
        qc,
        qs,
        (kc, ks),
        (vc, vs),
        scale,
        kv_heads=kv_heads,
        n=n,
        tokens=tokens,
        extent=padded,
        axis=0,
        rows=PREFILL_ROWS,
        page=PREFILL_PAGE,
        qk8=True,
        split=split,
        split_x=False,
        scale_vec=True,
    )


# ---- verify: 2-16 rows, bf16 queries, 64-key pages, K and V read in place -----------------------------------------

VERIFY_PAGE = 64
MAX_SPLITS = 64
# A split reads at least this many pages, and the splits of all tiles aim at this many threadgroups per GPU core.
MIN_SPLIT_PAGES = 16
THREADGROUPS_PER_CORE = 8


def verify_splits(tokens: int, groups: int, cores: int | None) -> int:
    """Splits per tile for ``tokens`` keys and ``groups`` tiles (KV heads x row tiles) on ``cores`` GPU cores."""
    pages = -(-tokens // VERIFY_PAGE)
    by_pages = max(1, pages // MIN_SPLIT_PAGES)
    by_cores = -(-THREADGROUPS_PER_CORE * cores // groups) if cores else MAX_SPLITS
    return max(1, min(by_pages, by_cores, MAX_SPLITS))


def verify(rq, keys, values, scale, *, split):
    """Causal attention of rotated bf16 queries [1, H, n, 256] (the last n keys) over affine8 states (indices
    [1, Hk, T, 64] uint32, norms [1, Hk, T], the strided views ``update_and_fetch`` returns are fine); fp32 output
    [1, H, n, 256] in the rotated space."""
    _, heads, n, dim = rq.shape
    kv_heads = int(keys.indices.shape[1])
    tokens = int(keys.indices.shape[2])
    assert dim == D and n <= tokens
    return _launch(
        rq.astype(mx.bfloat16),
        None,
        (keys.indices, keys.norms),
        (values.indices, values.norms),
        scale,
        kv_heads=kv_heads,
        n=n,
        tokens=tokens,
        extent=tokens,
        axis=1,
        rows=tile_rows(n, heads // kv_heads),
        page=VERIFY_PAGE,
        qk8=False,
        split=split,
        split_x=True,
        scale_vec=False,
    )
