# SPDX-License-Identifier: Apache-2.0
"""Expert-grouped 3-bit routed-expert matmul for float16 prefill chunks.

A threadgroup owns one expert's tile of up to 16 pairs and 64 outputs and
accumulates FP32 simdgroup products in K order, as MLX's ``gather_qmm_rhs``
does. Outputs land at the unsorted pair positions, so the sort and unsort
around the sorted ``gather_qmm`` go away.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from ...custom_kernels.nax import is_nax_available

MAX_PAIRS = 6144
_BM, _BN, _BK = 16, 64, 64


def _extract_lines() -> str:
    out = []
    for j in range(32):
        bit = 3 * j
        if bit + 3 <= 32:
            q = f"((a >> {bit}u) & 7u)"
        elif bit < 32:
            q = f"(((a >> {bit}u) | (b << {32 - bit}u)) & 7u)"
        elif bit + 3 <= 64:
            q = f"((b >> {bit - 32}u) & 7u)"
        elif bit < 64:
            q = f"(((b >> {bit - 32}u) | (d << {64 - bit}u)) & 7u)"
        else:
            q = f"((d >> {bit - 64}u) & 7u)"
        out.append(f"wt[{j}] = dequant<T>({q}, s, bb);")
    return "\n          ".join(out)


_HEADER = r"""
using namespace metal;

// (col, row) of this lane's two elements in an 8x8 simdgroup fragment
// (MLX steel BaseMMAFrag layout).
inline short2 frag_coord(ushort lane) {
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  return short2{fn, fm};
}

template <typename T>
inline T dequant(uint q, T s, T b) {
  return static_cast<T>(q) * s + b;
}
"""

_SOURCE = r"""
    constexpr int WN = 4, LD = BK + 8;
    constexpr int TN = BN / WN, FM = BM / 8, FN = TN / 8;
    constexpr int NTH = 32 * WN;
    constexpr int CPR = BK / 32;
    constexpr int NT = (BN * CPR + NTH - 1) / NTH;
    constexpr int XV = (BM * BK / 8 + NTH - 1) / NTH;
    constexpr int WPR = K_SIZE * 3 / 32;
    constexpr int GPR = K_SIZE / 64;
    threadgroup T Xs[BM * LD];
    threadgroup T Ws[BN * LD];
    threadgroup uint rows_x[BM];
    threadgroup uint rows_p[BM];

    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = sg * 32 + lane;
    const int z = int(threadgroup_position_in_grid.z);
    if (z >= int(item_start[NE])) {
      return;
    }
    // The expert of item z: the last e with item_start[e] <= z.
    int lo = 0, hi = NE;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (int(item_start[mid]) <= z) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const int e = lo;
    const int first = int(seg_start[e]) + (z - int(item_start[e])) * BM;
    const int count = min(BM, int(seg_start[e + 1]) - first);
    const int n0 = int(threadgroup_position_in_grid.y) * BN;
    if (tid < BM) {
      const uint p = order[first + min(int(tid), count - 1)];
      rows_p[tid] = p;
      rows_x[tid] = p / XDIV;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short2 fc = frag_coord(lane);
    const int sn = int(sg);
    simdgroup_float8x8 acc[FM][FN];
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < FM; i++) {
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < FN; j++) {
        acc[i][j] = simdgroup_float8x8(0);
      }
    }

    // An expert stack can pass 2 GiB, so its offsets need 64 bits.
    const size_t wbase = size_t(e) * N_SIZE + n0;
    vec<T, 4> xr[XV][2];
    uint ra[NT], rb[NT], rd[NT];
    T rs[NT], rbias[NT];
    auto fetch = [&](int k0) {
      _Pragma("clang loop unroll(full)")
      for (int u = 0; u < XV; u++) {
        const int v = int(tid) + u * NTH;
        if (v < BM * BK / 8) {
          const int r = v / (BK / 8), c = (v % (BK / 8)) * 8;
          const device vec<T, 4>* src =
              (const device vec<T, 4>*)(x + size_t(rows_x[r]) * K_SIZE + k0 + c);
          xr[u][0] = src[0];
          xr[u][1] = src[1];
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int u = 0; u < NT; u++) {
        const int t = int(tid) + u * NTH;
        if (t < BN * CPR) {
          const int row = t / CPR, ch = t % CPR;
          const size_t wrow = wbase + row;
          const device uint* p = w + wrow * WPR + (k0 / 32 + ch) * 3;
          ra[u] = p[0];
          rb[u] = p[1];
          rd[u] = p[2];
          const int g = (k0 + ch * 32) / 64;
          rs[u] = scales[wrow * GPR + g];
          rbias[u] = biases[wrow * GPR + g];
        }
      }
    };
    fetch(0);
    for (int k0 = 0; k0 < K_SIZE; k0 += BK) {
      _Pragma("clang loop unroll(full)")
      for (int u = 0; u < XV; u++) {
        const int v = int(tid) + u * NTH;
        if (v < BM * BK / 8) {
          const int r = v / (BK / 8), c = (v % (BK / 8)) * 8;
          threadgroup vec<T, 4>* dst = (threadgroup vec<T, 4>*)(Xs + r * LD + c);
          dst[0] = xr[u][0];
          dst[1] = xr[u][1];
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int u = 0; u < NT; u++) {
        const int t = int(tid) + u * NTH;
        if (t < BN * CPR) {
          const int row = t / CPR, ch = t % CPR;
          const uint a = ra[u], b = rb[u], d = rd[u];
          const T s = rs[u], bb = rbias[u];
          T wt[32];
          __EXTRACT__
          threadgroup vec<T, 4>* dst = (threadgroup vec<T, 4>*)(Ws + row * LD + ch * 32);
          _Pragma("clang loop unroll(full)")
          for (int q = 0; q < 8; q++) {
            dst[q] = vec<T, 4>(wt[4 * q], wt[4 * q + 1], wt[4 * q + 2], wt[4 * q + 3]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (k0 + BK < K_SIZE) {
        fetch(k0 + BK);
      }
      _Pragma("clang loop unroll(full)")
      for (int kk = 0; kk < BK; kk += 8) {
        simdgroup_float8x8 A[FM], B[FN];
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < FM; i++) {
          const threadgroup T* p = Xs + (i * 8 + fc.y) * LD + kk + fc.x;
          A[i].thread_elements()[0] = float(p[0]);
          A[i].thread_elements()[1] = float(p[1]);
        }
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < FN; j++) {
          // B(k, n) = W[n][k]
          const threadgroup T* p = Ws + (sn * TN + j * 8 + fc.x) * LD + kk + fc.y;
          B[j].thread_elements()[0] = float(p[0]);
          B[j].thread_elements()[1] = float(p[LD]);
        }
        _Pragma("clang loop unroll(full)")
        for (int i = 0; i < FM; i++) {
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < FN; j++) {
            simdgroup_multiply_accumulate(acc[i][j], A[i], B[j], acc[i][j]);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    _Pragma("clang loop unroll(full)")
    for (int i = 0; i < FM; i++) {
      const int r = i * 8 + fc.y;
      if (r < count) {
        device T* yr = y + size_t(rows_p[r]) * N_SIZE + n0 + sn * TN + fc.x;
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < FN; j++) {
          yr[j * 8] = static_cast<T>(acc[i][j].thread_elements()[0]);
          yr[j * 8 + 1] = static_cast<T>(acc[i][j].thread_elements()[1]);
        }
      }
    }
""".replace("__EXTRACT__", _extract_lines())


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_inkling2_expert_mma_q3",
        input_names=["x", "w", "scales", "biases", "order", "seg_start", "item_start"],
        output_names=["y"],
        header=_HEADER,
        source=_SOURCE,
    )


def supported(proj, dtype) -> bool:
    """True when ``proj`` (an ``ExpertProj``) is a 3-bit group-64 affine
    stack this kernel runs for ``dtype`` activations."""
    if (
        dtype != mx.float16
        or mx.default_device() != mx.gpu
        or is_nax_available()
        or proj.mode != "affine"
        or proj.bits != 3
        or proj.group_size != 64
    ):
        return False
    biases = proj.get("biases")
    return (
        biases is not None
        and proj.scales.dtype == dtype
        and biases.dtype == dtype
        and proj.weight.shape[-2] % _BN == 0
    )


def plan(rhs: mx.array, experts: int):
    """Pairs grouped by expert, on the GPU: (order, seg_start, item_start).

    ``order`` lists pair positions by expert; expert ``e``'s pairs are
    ``order[seg_start[e]:seg_start[e + 1]]``, split into tiles numbered from
    ``item_start[e]``."""
    order = mx.argsort(rhs)
    counts = mx.zeros((experts,), mx.int32).at[rhs.astype(mx.int32)].add(1)
    zero = mx.zeros((1,), mx.int32)
    seg = mx.concatenate([zero, mx.cumsum(counts)])
    item = mx.concatenate([zero, mx.cumsum((counts + _BM - 1) // _BM)])
    return order, seg.astype(mx.uint32), item.astype(mx.uint32)


def gather(proj, x: mx.array, pairs_per_row: int, routing) -> mx.array:
    """``gather_qmm`` over P (row, expert) pairs grouped by ``plan``.

    Pair ``p`` reads row ``p // pairs_per_row`` of ``x`` ([R, K]). Returns
    ``[P, N]`` in pair order."""
    order, seg, item = routing
    experts = int(proj.weight.shape[0])
    n = int(proj.weight.shape[-2])
    pairs = int(order.shape[0])
    return _kernel()(
        inputs=[x, proj.weight, proj.scales, proj.biases, order, seg, item],
        template=[
            ("T", x.dtype),
            ("K_SIZE", int(x.shape[-1])),
            ("N_SIZE", n),
            ("BM", _BM),
            ("BN", _BN),
            ("BK", _BK),
            ("XDIV", int(pairs_per_row)),
            ("NE", experts),
        ],
        grid=(32 * 4, n // _BN, (pairs + _BM - 1) // _BM + experts),
        threadgroup=(32 * 4, 1, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[x.dtype],
    )[0]
