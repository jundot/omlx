# SPDX-License-Identifier: Apache-2.0
"""Fused MoE router top-k for Qwen3.5/3.6 (and qwen3-next family) decode.

The routing chain after the gate linear — full softmax over the expert
count, ``argpartition`` top-k, ``take_along_axis``, top-k renormalize —
is five tiny-tensor ops per MoE layer per token. On the 256-expert
Qwen3.6-35B-A3B that chain measures ~51 us per layer at decode against a
~5 us fused launch: x40 layers it is ~2 ms of a ~9 ms decode step.

The gate linear and the softmax stay composed (the softmax output IS the
selection key the composed ``argpartition`` reads, so reusing it makes
the selected SET match bit-for-bit); one launch per row then selects
top-k over the rounded probabilities — ties resolve to the HIGHEST
index, mlx argpartition's empirically pinned behavior at this shape —
and renormalizes over the selected k in fp32 with one final rounding.
Scores can differ from the composed sum/divide chain by reduction-order
ulp; the routed expert set never differs.

Only short rows route here (decode and spec-verify widths); prefill keeps
the composed chain, whose cost amortizes over the chunk.

One-row decode also folds the combine ``(y * scores).sum(-2) +
sigmoid(shared_gate) * shared`` (five launches) into one bit-identical launch;
OMLX_QWEN35_MOE_COMBINE_FUSED=0 keeps the composed ops.

``softmax_topk_row`` runs the precise softmax and the top-k of one row in
one launch for the fused routed decode: one simdgroup reproduces MLX's
single-row block softmax (its per-simdgroup partial maxima and sums and
their order) and then selects exactly as ``fused_router_topk`` does, so
indices and scores are bit-identical to the two launches. The bf16
selection packs each probability with its expert index into one integer
key and keeps each lane's three largest keys at hand, so a selection is one
simd_max; ``TOPK_HEADER`` exports it for launches that fold the routing in.
``softmax_topk_rows`` runs that source verbatim for each row of a verify
window in one launch. OMLX_QWEN35_MOE_ROUTER_SOFTMAX_FOLD=0 keeps the two
launches.

``router_gemv`` runs the bias-free bf16 gate linear of one row with MLX's
one-row gemv arithmetic (per-lane column order, shuffle-down tree) but one
simdgroup per expert instead of MLX's four experts per simdgroup on 32
threadgroups, so the logits are bit-identical and the 2.6 MB weight read
spreads over the whole GPU. A verify window's rows share one launch with
one simdgroup per (expert, row), each row keeping the one-row arithmetic;
the simdgroups of one expert are adjacent, so its weight row is fetched
once. OMLX_QWEN35_MOE_ROUTER_GEMV=0 keeps ``nn.Linear``.
"""

from __future__ import annotations

import logging
import os
from functools import wraps

import mlx.core as mx

logger = logging.getLogger(__name__)

_KERNEL = None
_ENGAGED_LOGGED = False
_MAX_ROWS = 8
_COMBINE_KERNEL = None
_COMBINE_DISABLED = os.environ.get(
    "OMLX_QWEN35_MOE_COMBINE_FUSED", "1"
).strip().lower() in {"0", "false", "no", "off"}
# Top-k widths whose k-sum order was checked against mlx's reduction.
_COMBINE_TOP_K = (8, 10)
_SOFTMAX_FOLD_DISABLED = os.environ.get("OMLX_QWEN35_MOE_ROUTER_SOFTMAX_FOLD", "1") == "0"
_SOFTMAX_TOPK_KERNEL = None

# MLX 0.32.2 block softmax (precise: float accumulation, 4 reads per thread)
# of one row of NE logits. MLX runs it on NE / 4 threads, NE / 128 simdgroups;
# lane l of this one simdgroup holds what thread l of each of those
# simdgroups s holds (elements (s * 32 + l) * 4 + i), so every simd_max /
# simd_sum sees the same lane values, then the per-simdgroup partials meet
# in lanes 0..S-1 as in MLX's threadgroup step (other lanes -inf / 0).
# vals[s * 4 + i] is the probability rounded to T, as a float.
_SOFTMAX_ROW_HEADER = """
template <typename T, int NE>
METAL_FUNC void omlx_router_softmax_row(
    const device T* logits, uint lane, thread float* vals) {
  constexpr int S = NE / 128;
  float ld[S][4];
  for (int s = 0; s < S; s++) {
    for (int i = 0; i < 4; i++) {
      ld[s][i] = float(logits[(s * 32 + int(lane)) * 4 + i]);
    }
  }
  float lane_max = -metal::numeric_limits<float>::infinity();
  for (int s = 0; s < S; s++) {
    float m = -metal::numeric_limits<float>::max();
    for (int i = 0; i < 4; i++) {
      m = (m < ld[s][i]) ? ld[s][i] : m;
    }
    m = simd_max(m);
    lane_max = int(lane) == s ? m : lane_max;
  }
  const float maxval = simd_max(lane_max);
  float lane_sum = 0;
  for (int s = 0; s < S; s++) {
    float n = 0;
    for (int i = 0; i < 4; i++) {
      const float e = metal::fast::exp(ld[s][i] - maxval);
      ld[s][i] = e;
      n += e;
    }
    n = simd_sum(n);
    lane_sum = int(lane) == s ? n : lane_sum;
  }
  const float normalizer = 1 / simd_sum(lane_sum);
  for (int s = 0; s < S; s++) {
    for (int i = 0; i < 4; i++) {
      vals[s * 4 + i] = float(static_cast<T>(ld[s][i] * normalizer));
    }
  }
}
"""

# fused_router_topk's selection and renormalization over those values, for
# lane ``lane`` of one simdgroup. Selection j is the j-th largest
# (probability, expert) pair: equal probabilities resolve to the highest
# index, as in the per-row launch. ``nth(logits, lane, n)`` is the expert of
# selection n alone (every lane returns it), ``write<K>`` stores the top K
# and their renormalized scores.
#
# The generic scan (the FP32 test instantiations) is the per-row launch's: a
# lane's values ascend in expert index, so ">=" keeps the highest index of
# equal values in the lane and simd_max(index) the highest across lanes. For
# bf16 the values are bf16 probabilities in [+0, 1] or NaN (never selected),
# so each packs into a unique key ((bf16 bits + 1) << 16) | expert, 0 for
# NaN, whose order is that selection order. Each lane keeps its three
# largest remaining keys, so a selection is one simd_max over the lanes'
# largest; the selected lane moves on and rescans its keys once its three
# are used. Key 0 stands for the per-row launch's -inf / NE when nothing is
# left.
_TOPK_SELECT_HEADER = _SOFTMAX_ROW_HEADER + """
template <typename T, int NE>
struct omlx_router_topk {
  enum : uint { PER = uint(NE) / 32 };

  static METAL_FUNC void select(
      thread const float* vals, thread bool* taken, uint lane, thread float& p, thread uint& i) {
    float best = -INFINITY;
    uint best_i = uint(NE);
    uint best_k = 0;
    for (uint k = 0; k < PER; ++k) {
      if (!taken[k] && vals[k] >= best) {
        best = vals[k];
        best_i = ((k / 4) * 32 + lane) * 4 + (k % 4);
        best_k = k;
      }
    }
    float gbest = simd_max(best);
    uint cand = (best == gbest) ? best_i : 0u;
    uint gbest_i = simd_max(cand);
    if (best == gbest && best_i == gbest_i) {
      taken[best_k] = true;
    }
    p = gbest;
    i = gbest_i;
  }

  static METAL_FUNC uint nth(const device T* logits, uint lane, int n) {
    float vals[PER];
    omlx_router_softmax_row<T, NE>(logits, lane, vals);
    bool taken[PER];
    for (uint k = 0; k < PER; ++k) {
      taken[k] = false;
    }
    float p;
    uint i = uint(NE);
    for (int j = 0; j <= n; ++j) {
      select(vals, taken, lane, p, i);
    }
    return i;
  }

  template <int K>
  static METAL_FUNC void write(
      const device T* logits, uint lane, device uint32_t* indices, device T* scores) {
    float vals[PER];
    omlx_router_softmax_row<T, NE>(logits, lane, vals);
    bool taken[PER];
    for (uint k = 0; k < PER; ++k) {
      taken[k] = false;
    }
    float sel_p[K];
    uint sel_i[K];
    for (int j = 0; j < K; ++j) {
      select(vals, taken, lane, sel_p[j], sel_i[j]);
    }
    omlx_router_topk_store<T, K>(sel_p, sel_i, lane, indices, scores);
  }
};

template <int NE>
struct omlx_router_topk<bfloat16_t, NE> {
  enum : uint { PER = uint(NE) / 32 };

  static METAL_FUNC void keys(const device bfloat16_t* logits, uint lane, thread uint* key) {
    float vals[PER];
    omlx_router_softmax_row<bfloat16_t, NE>(logits, lane, vals);
    for (uint k = 0; k < PER; ++k) {
      const uint expert = ((k / 4) * 32 + lane) * 4 + (k % 4);
      key[k] = metal::isnan(vals[k])
          ? 0u
          : ((((as_type<uint>(vals[k]) >> 16) + 1u) << 16) | expert);
    }
  }

  // The lane's three largest keys below ``below`` in descending order (0:
  // none left).
  static METAL_FUNC void top3(thread const uint* key, uint below, thread uint* head) {
    uint a = 0u, b = 0u, c = 0u;
    for (uint k = 0; k < PER; ++k) {
      const uint v = key[k] < below ? key[k] : 0u;
      const bool ga = v > a, gb = v > b, gc = v > c;
      c = gb ? b : (gc ? v : c);
      b = ga ? a : (gb ? v : b);
      a = ga ? v : a;
    }
    head[0] = a;
    head[1] = b;
    head[2] = c;
  }

  // The next selection's key: the largest lane head. Its lane moves to its
  // next key, refilling its heads below the selection once all three are
  // taken.
  static METAL_FUNC uint step(thread const uint* key, thread uint* head, thread int& left) {
    const uint sel = simd_max(head[0]);
    if (sel != 0u && head[0] == sel) {
      head[0] = head[1];
      head[1] = head[2];
      head[2] = 0u;
      if (--left == 0) {
        top3(key, sel, head);
        left = 3;
      }
    }
    return sel;
  }

  static METAL_FUNC uint nth(const device bfloat16_t* logits, uint lane, int n) {
    uint key[PER];
    keys(logits, lane, key);
    uint head[3];
    top3(key, 0xffffffffu, head);
    int left = 3;
    uint sel = 0u;
    for (int j = 0; j <= n; ++j) {
      sel = step(key, head, left);
    }
    return sel == 0u ? uint(NE) : (sel & 0xffffu);
  }

  template <int K>
  static METAL_FUNC void write(
      const device bfloat16_t* logits, uint lane, device uint32_t* indices,
      device bfloat16_t* scores) {
    uint key[PER];
    keys(logits, lane, key);
    uint head[3];
    top3(key, 0xffffffffu, head);
    int left = 3;
    float sel_p[K];
    uint sel_i[K];
    for (int j = 0; j < K; ++j) {
      const uint sel = step(key, head, left);
      sel_p[j] = sel == 0u ? -INFINITY : as_type<float>(((sel >> 16) - 1u) << 16);
      sel_i[j] = sel == 0u ? uint(NE) : (sel & 0xffffu);
    }
    omlx_router_topk_store<bfloat16_t, K>(sel_p, sel_i, lane, indices, scores);
  }
};
"""

# Lane 0 renormalizes the selection over its own sum and stores it.
_TOPK_STORE_HEADER = """
template <typename T, int K>
METAL_FUNC void omlx_router_topk_store(
    thread const float* sel_p, thread const uint* sel_i, uint lane,
    device uint32_t* indices, device T* scores) {
  if (lane == 0) {
    float total = 0.0f;
    for (uint j = 0; j < K; ++j) {
      total += sel_p[j];
    }
    float inv = 1.0f / total;
    for (uint j = 0; j < K; ++j) {
      indices[j] = sel_i[j];
      scores[j] = T(sel_p[j] * inv);
    }
  }
}
"""

# Headers for kernels selecting experts (qwen35_moe_routed_decode reuses it).
TOPK_HEADER = _TOPK_STORE_HEADER + _TOPK_SELECT_HEADER

_SOFTMAX_TOPK_SOURCE = """
    const uint lane = thread_position_in_threadgroup.x;
    omlx_router_topk<T, NE>::template write<K>(logits, lane, indices, scores);
"""

_SOURCE = """
    // One threadgroup (a single simdgroup) per row; each lane owns
    // NE/32 experts. Input is the composed chain's OWN softmax output, so
    // the selection keys are bit-identical to what argpartition reads —
    // the selected set matches exactly: ties on equal rounded
    // probabilities resolve to the HIGHEST index, the empirically pinned
    // mlx argpartition behavior at this shape (see the parity test).
    // Renormalization
    // runs in fp32 with one rounding at the end (score values may differ
    // from the composed sum/divide by reduction-order ulp).
    constexpr uint PER = uint(NE) / 32;
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    const device T* g = probs + row * uint(NE);

    float vals[PER];
    for (uint i = 0; i < PER; ++i) {
        vals[i] = float(g[lane * PER + i]);
    }

    bool taken[PER];
    for (uint i = 0; i < PER; ++i) {
        taken[i] = false;
    }

    float sel_p[K];
    uint sel_i[K];
    for (uint j = 0; j < K; ++j) {
        float best = -INFINITY;
        uint best_i = uint(NE);
        for (uint i = 0; i < PER; ++i) {
            if (!taken[i] && vals[i] >= best) {
                best = vals[i];
                best_i = lane * PER + i;
            }
        }
        float gbest = simd_max(best);
        uint cand = (best == gbest) ? best_i : 0u;
        uint gbest_i = simd_max(cand);
        if (best == gbest && best_i == gbest_i) {
            taken[best_i - lane * PER] = true;
        }
        sel_p[j] = gbest;
        sel_i[j] = gbest_i;
    }

    if (lane == 0) {
        float total = 0.0f;
        for (uint j = 0; j < K; ++j) {
            total += sel_p[j];
        }
        float inv = 1.0f / total;
        for (uint j = 0; j < K; ++j) {
            indices[row * uint(K) + j] = sel_i[j];
            scores[row * uint(K) + j] = T(sel_p[j] * inv);
        }
    }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_moe_router_topk",
            input_names=["probs"],
            output_names=["indices", "scores"],
            source=_SOURCE,
        )
    return _KERNEL


def fused_router_topk(probs, top_k: int):
    """top-k + renormalize over softmaxed gate probabilities.

    ``probs`` is [..., NE] (the composed chain's own softmax output);
    returns (indices [..., k] uint32, scores [..., k] in the input
    dtype), selection descending."""
    gates = probs
    shape = gates.shape
    rows = 1
    for d in shape[:-1]:
        rows *= d
    ne = shape[-1]
    inds, scores = _kernel()(
        inputs=[gates],
        template=[
            ("T", gates.dtype),
            ("NE", ne),
            ("K", top_k),
        ],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*shape[:-1], top_k), (*shape[:-1], top_k)],
        output_dtypes=[mx.uint32, gates.dtype],
    )
    return inds, scores


_GEMV_DISABLED = os.environ.get("OMLX_QWEN35_MOE_ROUTER_GEMV", "1") == "0"
_GEMV_KERNEL = None
_GEMV_SIMDGROUPS = 4

# MLX 0.32.2 gemv for ``x @ W.T`` with W [N, K] row-major (the kernel MLX
# picks for one row when 64 < K < 16 * N: BN 1, SM 1, SN 32, TN 4). Lane l
# accumulates columns l * 4 + 128 * i for i ascending, each product of the
# T weight and the float input added in tn order, then MLX's
# simd_shuffle_down tree leaves the row sum in lane 0. Rows are independent
# (MLX's TM rows per simdgroup only share the input loads), so any number
# of rows per simdgroup gives the same bits.
_GEMV_ROWS_HEADER = """
template <typename T, int K, int R>
METAL_FUNC void omlx_router_gemv_rows(
    const device T* mat, const device T* x, uint lane, thread float* result) {
  for (int r = 0; r < R; r++) {
    result[r] = 0;
  }
  for (int i = 0; i < K / 128; i++) {
    const int bn = int(lane) * 4 + i * 128;
    float v_coeff[4];
    for (int tn = 0; tn < 4; tn++) {
      v_coeff[tn] = static_cast<float>(x[bn + tn]);
    }
    for (int r = 0; r < R; r++) {
      T inter[4];
      for (int tn = 0; tn < 4; tn++) {
        inter[tn] = mat[r * K + bn + tn];
      }
      for (int tn = 0; tn < 4; tn++) {
        result[r] += inter[tn] * v_coeff[tn];
      }
    }
  }
  for (int r = 0; r < R; r++) {
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
      result[r] += simd_shuffle_down(result[r], sn);
    }
  }
}
"""

# One simdgroup per (expert, row): simdgroup s computes expert s / M of row
# s % M with the one-row arithmetic, so an M-row window (a verify window)
# has M times the simdgroups of one row and every row keeps its bits. The
# M simdgroups of one expert are adjacent and share its weight row through
# the cache.
_GEMV_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const int s = int(threadgroup_position_in_grid.y) * NSG +
                  int(simdgroup_index_in_threadgroup);
    const int expert = s / M;
    const int m = s % M;
    float result[1];
    omlx_router_gemv_rows<T, K, 1>(w + size_t(expert) * K, x + m * K, lane, result);
    if (lane == 0) {
      y[m * N + expert] = static_cast<T>(result[0]);
    }
"""


def router_gemv(weight):
    """A launcher for ``x @ weight.T`` (a bias-free bf16 ``nn.Linear``) on
    bf16 rows of width K, or None outside the layout it reproduces.

    It runs MLX's one-row gemv arithmetic with one simdgroup per row (MLX
    runs four rows per simdgroup on N / 16 threadgroups). The layout is
    that gemv's: 64 < K < 16 * N and K % 128 == 0 (no guarded tail).
    Several input rows (a verify window, at most ``_MAX_ROWS``) run in one
    launch, each row with that one-row arithmetic.
    OMLX_QWEN35_MOE_ROUTER_GEMV=0 returns None.
    """
    global _GEMV_KERNEL
    if _GEMV_DISABLED or weight.ndim != 2 or weight.dtype != mx.bfloat16:
        return None
    n, k = weight.shape
    if not 64 < k < 16 * n or k % 128 or n % _GEMV_SIMDGROUPS:
        return None
    if _GEMV_KERNEL is None:
        _GEMV_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_moe_router_gemv",
            input_names=["x", "w"],
            output_names=["y"],
            header=_GEMV_ROWS_HEADER,
            source=_GEMV_SOURCE,
        )
    kernel = _GEMV_KERNEL

    def launch(x):
        rows = x.size // k
        return kernel(
            inputs=[x, weight],
            template=[
                ("T", mx.bfloat16),
                ("K", k),
                ("N", n),
                ("M", rows),
                ("NSG", _GEMV_SIMDGROUPS),
            ],
            grid=(32, n * rows, 1),
            threadgroup=(32, _GEMV_SIMDGROUPS, 1),
            output_shapes=[(*x.shape[:-1], n)],
            output_dtypes=[mx.bfloat16],
        )[0]

    return launch


def router_logits_row(x, weight):
    """``x @ weight.T`` through ``router_gemv`` for one bf16 row, or None."""
    if x.size != x.shape[-1] or x.shape[-1] != weight.shape[-1] or x.dtype != mx.bfloat16:
        return None
    launch = router_gemv(weight)
    return None if launch is None else launch(x)


def softmax_topk_eligible(logits, max_rows: int = _MAX_ROWS) -> bool:
    """Whether ``softmax_topk_rows`` reproduces the routing of ``logits``
    ([..., NE], 1..``max_rows`` rows), so a launch may run its source:
    bf16 rows of MLX's single-row block softmax with whole simdgroups (NE %
    128 == 0, NE <= 4096), unless OMLX_QWEN35_MOE_ROUTER_SOFTMAX_FOLD=0."""
    ne = logits.shape[-1]
    return (
        not _SOFTMAX_FOLD_DISABLED
        and logits.size // ne <= max_rows
        and ne % 128 == 0
        and ne <= 4096
        and logits.dtype == mx.bfloat16
    )


def softmax_topk_row(logits, top_k: int):
    """``fused_router_topk(mx.softmax(logits, axis=-1, precise=True), top_k)``
    for one bf16 row of gate logits, in one launch.

    Returns None outside the layout this reproduces
    (``softmax_topk_eligible``) or for more than one row.
    """
    global _SOFTMAX_TOPK_KERNEL
    ne = logits.shape[-1]
    if logits.size != ne or not softmax_topk_eligible(logits):
        return None
    if _SOFTMAX_TOPK_KERNEL is None:
        _SOFTMAX_TOPK_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_moe_router_softmax_topk_row",
            input_names=["logits"],
            output_names=["indices", "scores"],
            header=TOPK_HEADER,
            source=_SOFTMAX_TOPK_SOURCE,
        )
    lead = logits.shape[:-1]
    return _SOFTMAX_TOPK_KERNEL(
        inputs=[logits],
        template=[("T", mx.bfloat16), ("NE", ne), ("K", top_k)],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*lead, top_k), (*lead, top_k)],
        output_dtypes=[mx.uint32, mx.bfloat16],
    )


# softmax_topk_row's body, run by threadgroup y on row y of a window: the
# kernel's pointers are rebound to that row in an inner scope, so each row
# runs the one-row source verbatim.
_SOFTMAX_TOPK_ROWS_SOURCE = (
    """
    const uint window_row = threadgroup_position_in_grid.y;
    const auto logits_row = logits + window_row * uint(NE);
    const auto indices_row = indices + window_row * uint(K);
    const auto scores_row = scores + window_row * uint(K);
    {
    const auto logits = logits_row;
    const auto indices = indices_row;
    const auto scores = scores_row;
"""
    + _SOFTMAX_TOPK_SOURCE
    + """
    }
"""
)
_SOFTMAX_TOPK_ROWS_KERNEL = None


def softmax_topk_rows(logits, top_k: int, max_rows: int = _MAX_ROWS):
    """``softmax_topk_row`` for each of 1..``max_rows`` rows of bf16 gate
    logits ``[..., NE]``, in one launch; None where ``softmax_topk_row``
    declines."""
    global _SOFTMAX_TOPK_ROWS_KERNEL
    ne = logits.shape[-1]
    rows = logits.size // ne
    if rows == 1:
        return softmax_topk_row(logits, top_k)
    if not softmax_topk_eligible(logits, max_rows):
        return None
    if _SOFTMAX_TOPK_ROWS_KERNEL is None:
        _SOFTMAX_TOPK_ROWS_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_moe_router_softmax_topk_rows",
            input_names=["logits"],
            output_names=["indices", "scores"],
            header=TOPK_HEADER,
            source=_SOFTMAX_TOPK_ROWS_SOURCE,
        )
    lead = logits.shape[:-1]
    return _SOFTMAX_TOPK_ROWS_KERNEL(
        inputs=[logits],
        template=[("T", mx.bfloat16), ("NE", ne), ("K", top_k)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*lead, top_k), (*lead, top_k)],
        output_dtypes=[mx.uint32, mx.bfloat16],
    )


# Matches the composed ops bit for bit: each product, sum step, sigmoid step
# and the final multiply/add round to T like mlx's kernels. The k-sum follows
# mlx col_reduce_small for one row: lane l (of 8) folds rows l, l + 8, ...
# onto +0, then lanes 1..7 are added onto lane 0 in order. mx.sigmoid is
# 1 / (1 + exp(|x|)) (1 - that for x >= 0) with mlx's non-fast-math exp.
_COMBINE_SOURCE = """
    const uint h = thread_position_in_grid.x;
    if (h >= uint(H)) return;
    const device T* rp = routed + h;
    T lane[8];
    for (int l = 0; l < 8; ++l) {
        lane[l] = T(0.0f);
    }
    for (int j = 0; j < K; ++j) {
        const T p = T(float(rp[j * H]) * float(scores[j]));
        lane[j % 8] = T(float(p) + float(lane[j % 8]));
    }
    T acc = lane[0];
    for (int l = 1; l < 8; ++l) {
        acc = T(float(lane[l]) + float(acc));
    }
    const float g = float(gate[0]);
    const T e = T(1.0f + float(T(metal::precise::exp(metal::abs(g)))));
    const T y = T(metal::precise::divide(1.0f, float(e)));
    const T s = g < 0.0f ? y : T(1.0f - float(y));
    const T sh = T(float(s) * float(shared[h]));
    out[h] = T(float(acc) + float(sh));
"""


def fused_moe_combine(routed, scores, shared, gate):
    """``(routed * scores[..., None]).sum(-2) + mx.sigmoid(gate) * shared`` in one launch.

    One row only: ``routed`` [..., k, H], ``scores`` [..., k], ``shared``
    [..., H], ``gate`` [..., 1], all bf16, k in _COMBINE_TOP_K. Each row is
    reduced in mlx's one-row col_reduce_small order. Returns None when the
    operands are outside that layout or the kill switch is set.
    """
    global _COMBINE_KERNEL
    if _COMBINE_DISABLED or routed.ndim < 2:
        return None
    shape = routed.shape
    lead = shape[:-2]
    top_k, hidden = shape[-2:]
    if not (
        top_k in _COMBINE_TOP_K
        and lead.count(1) == len(lead)
        and scores.shape == (*lead, top_k)
        and shared.shape == (*lead, hidden)
        and gate.shape == (*lead, 1)
        and routed.dtype == scores.dtype == shared.dtype == gate.dtype == mx.bfloat16
    ):
        return None
    if _COMBINE_KERNEL is None:
        _COMBINE_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_moe_combine_row",
            input_names=["routed", "scores", "shared", "gate"],
            output_names=["out"],
            source=_COMBINE_SOURCE,
        )
    return _COMBINE_KERNEL(
        inputs=[routed, scores, shared, gate],
        template=[("T", mx.bfloat16), ("K", top_k), ("H", hidden)],
        grid=(hidden, 1, 1),
        threadgroup=(min(256, hidden), 1, 1),
        output_shapes=[(*lead, hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]


def router_eligible(x, num_experts: int, max_rows: int = _MAX_ROWS) -> bool:
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    return (
        rows <= max_rows
        and num_experts % 32 == 0
        and x.dtype in (mx.bfloat16, mx.float16)
    )


def _ensure_vlm_verify_patch() -> None:
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    original = Qwen3_5BatchInvariantForward._feed_forward
    if getattr(original, "_omlx_router_fused", False):
        return

    @wraps(original)
    def verify(self, feed_forward, x):
        if not (
            isinstance(feed_forward, Qwen3_5MoeSparseMoeBlock)
            and x.ndim == 3
            and router_eligible(x, feed_forward.num_experts)
        ):
            return original(self, feed_forward, x)
        gates = mx.softmax(self._linear(feed_forward.gate, x), axis=-1, precise=True)
        indices, scores = fused_router_topk(gates, feed_forward.top_k)
        shared = self._feed_forward(feed_forward.shared_expert, x)
        shared = mx.sigmoid(self._linear(feed_forward.shared_expert_gate, x)) * shared
        routed = self._switch_glu(feed_forward.switch_mlp, x, indices)
        return (routed * scores[..., None]).sum(axis=-2) + shared

    verify._omlx_router_fused = True
    Qwen3_5BatchInvariantForward._feed_forward = verify


def apply_qwen35_moe_router_patch() -> bool:
    """Route short-row MoE gating through the fused top-k launch.

    Wraps ``Qwen3NextSparseMoeBlock.__call__`` (shared by qwen3_5_moe and
    qwen3-next in mlx-lm): the fast arm reuses the module's own gate,
    experts, and shared expert; prefill rows and sharded runs keep the
    original body untouched."""
    global _ENGAGED_LOGGED
    if not mx.metal.is_available():
        return False
    _ensure_vlm_verify_patch()
    try:
        from mlx_lm.models import qwen3_next as q3n
    except ImportError:
        return False
    cls = getattr(q3n, "Qwen3NextSparseMoeBlock", None)
    if cls is None or getattr(cls, "_omlx_router_fused", False):
        return cls is not None

    orig_call = cls.__call__

    def patched_call(self, x):
        if (
            self.sharding_group is not None
            or not self.norm_topk_prob
            or not router_eligible(x, self.num_experts)
        ):
            return orig_call(self, x)
        try:
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            inds, scores = fused_router_topk(gates, self.top_k)

            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)

            shared_y = self.shared_expert(x)
            shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
            return y + shared_y
        except Exception:
            logger.warning(
                "fused MoE router failed; composed fallback", exc_info=True
            )
            return orig_call(self, x)

    cls.__call__ = patched_call
    cls._omlx_router_fused = True

    try:
        from mlx_vlm.models.qwen3_5_moe import language as vlm_moe
    except ImportError:
        vlm_moe = None
    vcls = getattr(vlm_moe, "Qwen3_5MoeSparseMoeBlock", None) if vlm_moe else None
    if vcls is not None and not getattr(vcls, "_omlx_router_fused", False):
        vlm_orig = vcls.__call__

        def vlm_patched_call(self, x):
            if not router_eligible(x, self.num_experts):
                return vlm_orig(self, x)
            # Children by item: this runs for every MoE layer of every decode step.
            gates = mx.softmax(self["gate"](x), axis=-1, precise=True)
            inds, scores = fused_router_topk(gates, self.top_k)
            y = self["switch_mlp"](x, inds)
            shared_y = self["shared_expert"](x)
            shared_gate = self["shared_expert_gate"](x)
            combined = fused_moe_combine(y, scores, shared_y, shared_gate)
            if combined is not None:
                return combined
            y = (y * scores[..., None]).sum(axis=-2)
            shared_y = mx.sigmoid(shared_gate) * shared_y
            return y + shared_y

        vcls.__call__ = vlm_patched_call
        vcls._omlx_router_fused = True

    if not _ENGAGED_LOGGED:
        _ENGAGED_LOGGED = True
        logger.info("Qwen MoE fused router top-k patch applied")
    return True
