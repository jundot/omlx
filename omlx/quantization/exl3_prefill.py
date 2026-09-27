"""Sushi MIT cooperative matrix prefill adapted to MLX native expert gathers.
Copyright (c) 2026 Theinruj Toranavikrai and David Dalcu.
See docs/licenses/sushi-MIT.txt and docs/exl3-backend.md.
"""

import mlx.core as mx

FRAGMENTS = r"""
#define SMAT_UNROLL _Pragma("clang loop unroll(full)")
static inline uint smat_funnel(const device uint *words, uint end, uint nwords) {
  const uint last = (end - 1u) >> 5u;
  const uint prev = last == 0u ? nwords - 1u : last - 1u;
  return uint((((ulong)words[prev] << 32u) | (ulong)words[last]) >> ((0u - end) & 31u));
}
static inline half2 smat_pair(uint f, uint s0, uint s1) {
  return exl3_pairh(uint2((f >> s0) & 0xffffu, (f >> s1) & 0xffffu));
}
template<uint N>
static inline void smat_group(const device uint *words, uint g, thread half2 *p) {
  if (N == 64u) {
    const ulong m = ((ulong)words[(g + 31u) & 31u] << 32u) | (ulong)words[g];
    SMAT_UNROLL for (uint j = 0u; j < 4u; j++) p[j] = smat_pair(uint(m >> (24u - 8u * j)), 4u, 0u);
  } else if (N == 48u && EXL3_FUNNEL48) {
    const uint lo = smat_funnel(words, 24u * g + 18u, 24u);
    const uint hi = smat_funnel(words, 24u * g + 24u, 24u);
    p[0] = smat_pair(lo, 15u, 12u);
    p[1] = smat_pair(lo, 9u, 6u);
    p[2] = smat_pair(lo, 3u, 0u);
    p[3] = smat_pair(hi, 3u, 0u);
  } else if (N == 40u) {
    const uint lo = smat_funnel(words, 20u * g + 15u, 20u);
    const uint hi = smat_funnel(words, 20u * g + 20u, 20u);
    p[0] = smat_pair(lo, 13u, 10u);
    p[1] = smat_pair(lo, 8u, 5u);
    p[2] = smat_pair(lo, 3u, 0u);
    p[3] = smat_pair(hi, 3u, 0u);
  } else if (N == 36u) {
    const uint f = smat_funnel(words, 18u * g + 18u, 18u);
    p[0] = smat_pair(f, 16u, 14u);
    p[1] = smat_pair(f, 12u, 9u);
    p[2] = smat_pair(f, 7u, 5u);
    p[3] = smat_pair(f, 3u, 0u);
  } else {
    SMAT_UNROLL for (uint j = 0u; j < 4u; j++) {
      const exl3_win w = exl3_pair_window(8u * g + 2u * j, N);
      p[j] = smat_pair(uint((((ulong)words[w.i0] << 32u) | (ulong)words[w.i1]) >> w.sh), w.fresh, 0u);
    }
  }
}
"""
SOURCE = r"""
uint win = uint(threadgroup_position_in_grid.y);
uint sg = uint(simdgroup_index_in_threadgroup);
ushort lane = ushort(thread_index_in_simdgroup);
constexpr uint TILE = 16u;
constexpr uint N = uint(NHW);
constexpr uint PACKED_W = N / 2u;
constexpr uint IT = uint(IDIM) / TILE;
constexpr uint OT = uint(ODIM) / TILE;
const uint start = wstarts[win];
const uint n = wnlive[win];
if (n == 0u || n > uint(WIN)) return;
const uint col0 = uint(threadgroup_position_in_grid.x) * 128u + sg * 32u;
const ushort qid = lane >> 2;
const ushort fm = (qid & 4) + ((lane >> 1) & 3);
const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;
const uint g = uint(fm) * 4u + uint(fn >> 1);
constexpr uint MB = (uint(WIN) + 7u) / 8u;
uint row = start;
const uint end = start + n;
while (row < end) {
const uint run0 = row;
const uint eid = uint(eids[row]);
uint run_end = row + 1u;
while (run_end < end && uint(eids[run_end]) == eid) run_end++;
const uint nlive = run_end - row;
size_t xo[4][2];
SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
  xo[mb][0] = (size_t)(run0 + min(mb * 8u + uint(fn), nlive - 1u)) * (size_t)(IDIM);
  xo[mb][1] = (size_t)(run0 + min(mb * 8u + uint(fn) + 1u, nlive - 1u)) * (size_t)(IDIM);
}
simdgroup_matrix<float, 8, 8> acc[4][4];
SMAT_UNROLL for (uint nb = 0u; nb < 4u; nb++) {
  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) acc[nb][mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
const device uint *tiles = (const device uint *)(trellis + (size_t)eid * (size_t)IT * (size_t)OT * (size_t)N) + (col0 / TILE) * PACKED_W;
for (uint tk = 0u; tk < IT; tk++) {
  const uint kc = tk * TILE + uint(fm);
  simdgroup_matrix<half, 8, 8> b[2][4];
  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    SMAT_UNROLL for (uint kb = 0u; kb < 2u; kb++) {
      b[kb][mb].thread_elements()[0] = x[xo[mb][0] + kc + kb * 8u];
      b[kb][mb].thread_elements()[1] = x[xo[mb][1] + kc + kb * 8u];
    }
  }
  const device uint *words = tiles + (size_t)tk * (size_t)OT * (size_t)PACKED_W;
  simdgroup_matrix<half, 8, 8> a[4][2];
  SMAT_UNROLL for (uint t = 0u; t < 2u; t++) {
    half2 p[4];
    smat_group<N>(words + t * PACKED_W, g, p);
    SMAT_UNROLL for (uint j = 0u; j < 4u; j++) {
      a[2u * t + (j >> 1u)][j & 1u].thread_elements()[0] = p[j].x;
      a[2u * t + (j >> 1u)][j & 1u].thread_elements()[1] = p[j].y;
    }
  }
  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    SMAT_UNROLL for (uint kb = 0u; kb < 2u; kb++) {
      SMAT_UNROLL for (uint nb = 0u; nb < 4u; nb++) simdgroup_multiply_accumulate(acc[nb][mb], a[nb][kb], b[kb][mb], acc[nb][mb]);
    }
  }
}
SMAT_UNROLL for (uint nb = 0u; nb < 4u; nb++) {
  const size_t oc = (size_t)(col0 + nb * 8u + uint(fm));
  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    const uint m = mb * 8u + uint(fn);
    if (m < nlive) y[(size_t)(run0 + m) * (size_t)(ODIM) + oc] = half(acc[nb][mb].thread_elements()[0]);
    if (m + 1u < nlive) y[(size_t)(run0 + m + 1u) * (size_t)(ODIM) + oc] = half(acc[nb][mb].thread_elements()[1]);
  }
}
row = run_end;
}
"""

_KERNELS = {}


def prefill_inner(x, ids, trellis, spec, n, header, sorted_indices):
    rows, k = x.shape
    order = None
    if not sorted_indices:
        order = mx.argsort(ids)
        ids, x = ids[order], x[order]
    kernel = _KERNELS.get(spec.window)
    if kernel is None:
        hdr = "#include <metal_simdgroup_matrix>\n#define EXL3_FUNNEL48 0\n" + header
        hdr += f"\nstatic inline half2 exl3_pairh(uint2 cw) {{ return half2(exl3_decode2(cw,{(1 << spec.window) - 1}u)); }}\n"
        kernel = mx.fast.metal_kernel(
            name=f"omlx_exl3_prefill_w{spec.window}",
            input_names=["x", "trellis", "eids", "wstarts", "wnlive"],
            output_names=["y"],
            source=SOURCE,
            header=hdr + FRAGMENTS,
        )
        _KERNELS[spec.window] = kernel
    starts = mx.arange(0, rows, 32, dtype=mx.uint32)
    lives = mx.minimum(mx.array(rows, dtype=mx.uint32) - starts, 32)
    y = kernel(
        inputs=[x, trellis, ids, starts, lives],
        template=[("IDIM", k), ("ODIM", n), ("WIN", 32), ("NHW", spec.halfwords)],
        grid=(n, len(starts), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.float16],
    )[0]
    return y[mx.argsort(order)] if order is not None else y
