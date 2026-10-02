# SPDX-License-Identifier: Apache-2.0
"""Fused routed and shared experts for one-token Qwen3.5-MoE-family decode.

After the fused router, a one-token MoE block runs its routed experts as the
gate+up ``gather_qmm``, the compiled SwiGLU and the down ``gather_qmm`` (each
``gather_qmm`` behind an ``arange`` for its row indices), the shared expert
as three ``quantized_matmul`` launches and a SwiGLU, the shared-expert gate
as one more ``quantized_matmul``, and one combine launch for the
score-weighted sum plus the gated shared expert. At batch-one decode these
launches are short and mostly depend on each other, so this patch runs the
same arithmetic in two launches after the router:

1. gate+up with a SwiGLU epilogue, for the ten selected experts and the
   shared expert, plus the shared-expert gate row. Each simdgroup computes
   the gate rows and the matching up rows of one expert with MLX's
   ``qmv_fast`` lane partition and add order, rounds both to the activation
   dtype, then applies MLX's ``Sigmoid`` and the two multiplies of the
   compiled ``swiglu`` in its order. The gate row follows MLX's ``qmv`` for
   a one-row output; its threadgroup comes first in the grid, so its long
   serial K walk overlaps the expert rows.
2. down with the combine. Simdgroup ``j`` of a threadgroup runs the stock
   ``qmv`` work (including its guarded K tail) of selected expert ``j`` for
   the threadgroup's rows, one more simdgroup the shared expert's. Each row
   is rounded, then the combine of ``qwen35_moe_router.fused_moe_combine``
   follows: score products summed in MLX's ``col_reduce_small`` order, plus
   ``sigmoid(shared_gate) * shared``.

The router ahead of them (gate linear, precise softmax, fused top-k) runs
the gate linear as MLX's one-row gemv spread over one simdgroup per expert
(``qwen35_moe_router.router_gemv``; MLX's own launch has 32 threadgroups).
Its softmax and top-k run inside the gate+up launch where the shapes allow
(``qwen35_moe_router.softmax_topk_eligible``): the launch's first block
stores the row's selection and scores for the down launch, and every
simdgroup of an expert block recomputes the selection it serves from the
router logits with ``softmax_topk_row``'s arithmetic, so the block is three
dependent launches instead of four. ``OMLX_QWEN35_MOE_TOPK_FOLD=0`` runs
the softmax and top-k as their own launch
(``qwen35_moe_router.softmax_topk_row``).

The result is bit-identical to the composed path. The quantized dot products
reuse the MLX 0.32.2 transcription in ``moe_verify_gather`` (4, 5, 6 and
8 bits; group size 32, 64 or 128), one instantiation per weight format. MLX
picks ``qmv_fast`` when N % 8 == 0 and K is a multiple of its kernel block
(512 for 4/5-bit, 256 for 6/8-bit weights) and ``qmv`` otherwise.
Routed experts are taken where gate+up takes ``qmv_fast`` and down takes
``qmv``: one bf16 token, top-k 10, affine experts with bf16 scales,
hidden on the ``qmv_fast`` block and intermediate off it (Qwen3.8-Flash-Next:
2560 and 640, 4-bit or oQ5e's 5-bit). A shared expert or gate outside that format
(unquantized, packed, ...) runs as composed launches and only its outputs
enter the combine. Prefill, multi-row calls and every other shape keep the
original body. If the first launch fails, the patch disables itself and the
block keeps its composed body. ``OMLX_QWEN35_MOE_ROUTED_DECODE=0`` keeps the
composed body; ``OMLX_QWEN35_MOE_SHARED_FOLD=0`` keeps the shared expert and
its gate as composed launches.

Row-exact MTP verify windows (1..8 rows whose logits must equal serial
decode) run the same arithmetic for the whole window in the same launches:
the router gemv over all rows (``qwen35_moe_router.router_gemv``), then
window variants of the launches above in which each threadgroup serves one
row with the one-token source verbatim (threadgroups of one weight block for
consecutive rows are adjacent, so shared experts are fetched once through
the cache). Up to three rows select inside the gate+up launch; from four
rows on the recomputed selections cost more than the routing launch
(``softmax_topk_rows``) they save, so it runs on its own. Every row is
bit-identical to the fused one-token call on that row. This replaces the
verifier's composed MoE (about fifteen launches, two of them binding the
whole stacked experts) only while row-exact verify is armed and the block
folds its shared expert; one-row windows run the one-token launches. If the
first window launch fails, the verifier keeps its composed MoE.
``OMLX_QWEN35_MOE_VERIFY_WINDOW=0`` keeps it too.

MLX commits a command buffer once the inputs bound to it exceed its size cap
(50 MB by default), counting each input array whole. The stacked expert
weights are hundreds of MB, so every launch that binds them ends a command
buffer (~10-20 us of host CPU and a GPU gap per commit, and the host blocks
once too many are in flight). The kernels therefore bind a one-expert view
that shares the stacked array's buffer at offset 0 and index the other
experts from it. The weights are resident model parameters, so the cap has
nothing to bound here. Only scheduling changes;
``OMLX_QWEN35_MOE_ROUTED_DECODE_VIEWS=0`` binds the whole arrays.
"""

from __future__ import annotations

import logging
import os
from functools import cache, wraps
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .module_cache import cached_per_module
from .moe_verify_gather import _BITS, _GROUP_SIZES, qmv_fast_layout
from .moe_verify_gather import _HEADER as _QMV_HEADER
from .qwen35_moe_router import (
    TOPK_HEADER,
    fused_router_topk,
    router_eligible,
    router_gemv,
    softmax_topk_eligible,
    softmax_topk_row,
    softmax_topk_rows,
)
from .qwen35_verify_qmm import is_row_exact_armed

logger = logging.getLogger(__name__)

TOP_K = 10
_GATE_UP_ROWS = 2  # gate rows (and as many up rows) per simdgroup
_GATE_UP_SIMDGROUPS = 2
_DOWN_ROWS = 4
_ENABLED = os.environ.get("OMLX_QWEN35_MOE_ROUTED_DECODE", "1") != "0"
_SHARED_FOLD = os.environ.get("OMLX_QWEN35_MOE_SHARED_FOLD", "1") != "0"
_VIEWS_ENABLED = os.environ.get("OMLX_QWEN35_MOE_ROUTED_DECODE_VIEWS", "1") != "0"
_DISABLED = False
_PROVEN = False
_VERIFY_WINDOW = os.environ.get("OMLX_QWEN35_MOE_VERIFY_WINDOW", "1") != "0"
_TOPK_FOLD = os.environ.get("OMLX_QWEN35_MOE_TOPK_FOLD", "1") != "0"
# Every gate+up simdgroup recomputes its row's selection, so the folded
# routing's ALU grows with the rows while the launch it saves does not: from
# four rows on the gate+up launch runs out of ALU headroom and the separate
# routing launch is as fast or faster (M5 Ultra, oQ5e shapes).
_TOPK_FOLD_MAX_ROWS = 3
# Row-exact verify windows up to the hyper-connection and GDN verify
# kernels' 16-row ceiling; the kernels index rows from the grid.
WINDOW_MAX_ROWS = 16
_WINDOW_DISABLED = False
_WINDOW_PROVEN = False


class _Format(NamedTuple):
    """One weight format: MLX's ``qmv_fast`` (fast) or ``qmv`` traversal."""

    bits: int
    group_size: int
    fast: bool


# Rows of one mat-vec, per format namespace: MLX's qmv_fast (FAST) or qmv
# traversal (full K blocks, then the guarded tail) of one input vector. Rows
# [0, NA) are rows row_a.. of (wa, sa, ba), rows [NA, NA + NB) rows row_b..
# of (wb, sb, bb); result[row] ends as the row's simd_sum.
_ROWS = r"""
template <typename T, int K, int NA, int NB>
METAL_FUNC void qmv_rows(
    const device uint8_t* wa,
    const device T* sa,
    const device T* ba,
    size_t row_a,
    const device uint8_t* wb,
    const device T* sb,
    const device T* bb,
    size_t row_b,
    const device T* x,
    uint simd_lid,
    thread float* result) {
  constexpr int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
  constexpr int in_vec_size_g = K / GS;
  const int lane_w = int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
  const int lane_g = int(simd_lid) / SCALE_STEP_PER_THREAD;
  wa += row_a * in_vec_size_w + lane_w;
  sa += row_a * in_vec_size_g + lane_g;
  ba += row_a * in_vec_size_g + lane_g;
  wb += row_b * in_vec_size_w + lane_w;
  sb += row_b * in_vec_size_g + lane_g;
  bb += row_b * in_vec_size_g + lane_g;
  x += int(simd_lid) * VALUES_PER_THREAD;

  float x_thread[VALUES_PER_THREAD];
  for (int row = 0; row < NA + NB; row++) {
    result[row] = 0;
  }
  int k = 0;
  for (; k < (FAST ? K : K - BLOCK_SIZE); k += BLOCK_SIZE) {
    float sum = load_vector<T>(x, x_thread);
    for (int row = 0; row < NA + NB; row++) {
      const bool a = row < NA;
      const int r = a ? row : row - NA;
      const device uint8_t* wl = (a ? wa : wb) + r * in_vec_size_w;
      float s = (a ? sa : sb)[r * in_vec_size_g];
      float b = (a ? ba : bb)[r * in_vec_size_g];
      result[row] += qdot_n(wl, x_thread, s, b, sum, VALUES_PER_THREAD);
    }
    wa += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    wb += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
    sa += BLOCK_SIZE / GS;
    ba += BLOCK_SIZE / GS;
    sb += BLOCK_SIZE / GS;
    bb += BLOCK_SIZE / GS;
    x += BLOCK_SIZE;
  }
  if (!FAST) {
    const int remaining = clamp(
        int(K - k - int(simd_lid) * VALUES_PER_THREAD), 0, VALUES_PER_THREAD);
    if (remaining > 0) {
      float sum = load_vector_safe<T>(x, x_thread, remaining);
      for (int row = 0; row < NA + NB; row++) {
        const bool a = row < NA;
        const int r = a ? row : row - NA;
        const device uint8_t* wl = (a ? wa : wb) + r * in_vec_size_w;
        float s = (a ? sa : sb)[r * in_vec_size_g];
        float b = (a ? ba : bb)[r * in_vec_size_g];
        result[row] += qdot_n(wl, x_thread, s, b, sum, remaining);
      }
    }
  }
  for (int row = 0; row < NA + NB; row++) {
    result[row] = simd_sum(result[row]);
  }
}
"""

_COMMON = r"""
using namespace metal;

// MLX 0.32.2 Sigmoid, evaluated in T as the compiled swiglu does.
template <typename U>
inline U omlx_mlx_sigmoid(U x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

// swiglu(gate, up) of the compiled mlx-vlm activation: both rounded to T,
// silu(gate) = gate * sigmoid(gate), then times up, each op in T.
template <typename T, int RPS>
METAL_FUNC void swiglu_store(thread const float* result, device T* yp, uint simd_lid) {
  if (simd_lid == 0) {
    for (int row = 0; row < RPS; row++) {
      T g = static_cast<T>(result[row]);
      T u = static_cast<T>(result[row + RPS]);
      T t = g * omlx_mlx_sigmoid<T>(g);
      yp[row] = t * u;
    }
  }
}
"""

# One threadgroup of NSG simdgroups (each RPS gate + RPS up rows) per row
# block b = threadgroup y, blocks in order: [the shared-expert gate row, the
# NS / (NSG * RPS) shared-expert blocks,] then NI / (NSG * RPS) blocks per
# selected expert. Output y holds silu(gate) * up of the TOP_K experts ([10,
# NI]), then [those NS rows of the shared expert, then the gate row].
_GATE_UP_HEAD = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int ROWS = NSG * RPS;
    int b = int(threadgroup_position_in_grid.y);
    float result[2 * RPS];
"""

_GATE_UP_SHARED = r"""
    if (b == 0) {
      // shared_expert_gate: MLX's qmv for its one output row. First in the
      // grid, so its serial K walk overlaps the expert blocks.
      if (simd_gid == 0) {
        gt::qmv_rows<T, K, 1, 0>(
            (const device uint8_t*)g_w, g_s, g_b, 0,
            (const device uint8_t*)g_w, g_s, g_b, 0,
            x, simd_lid, result);
        if (simd_lid == 0) {
          y[10 * NI + NS] = static_cast<T>(result[0]);
        }
      }
      return;
    }
    if (b <= NS / ROWS) {
      const int out_row = (b - 1) * ROWS + int(simd_gid) * RPS;
      st::qmv_rows<T, K, RPS, RPS>(
          (const device uint8_t*)sg_w, sg_s, sg_b, out_row,
          (const device uint8_t*)su_w, su_s, su_b, out_row,
          x, simd_lid, result);
      swiglu_store<T, RPS>(result, y + 10 * NI + out_row, simd_lid);
      return;
    }
    b -= 1 + NS / ROWS;
"""

_GATE_UP_ROUTED = r"""
    const int slot = b / (NI / ROWS);
    const int out_row = (b % (NI / ROWS)) * ROWS + int(simd_gid) * RPS;
    const size_t expert = size_t(rhs[slot]);
    rt::qmv_rows<T, K, RPS, RPS>(
        (const device uint8_t*)w, scales, biases, expert * (2 * NI) + out_row,
        (const device uint8_t*)w, scales, biases, expert * (2 * NI) + NI + out_row,
        x, simd_lid, result);
    swiglu_store<T, RPS>(result, y + size_t(slot) * NI + out_row, simd_lid);
"""

# Threadgroup (32, NPART): simdgroup j < TOP_K computes RPS rows of selected
# expert j (simdgroup TOP_K the shared expert's), then simdgroup 0 combines.
# Output is [N].
_DOWN_HEAD = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int out_row = int(tid.y) * RPS;
    const int slot = int(simd_gid);
    threadgroup T part[NPART * RPS];
    float result[RPS];
"""

_DOWN_SHARED_ROWS = r"""
    if (slot == 10) {
      sd::qmv_rows<T, KS, RPS, 0>(
          (const device uint8_t*)sd_w, sd_s, sd_b, out_row,
          (const device uint8_t*)sd_w, sd_s, sd_b, out_row,
          x + 10 * K, simd_lid, result);
    } else
"""

_DOWN_TAIL = r"""
    {
      const size_t expert = size_t(rhs[slot]);
      rd::qmv_rows<T, K, RPS, 0>(
          (const device uint8_t*)w, scales, biases, expert * N + out_row,
          (const device uint8_t*)w, scales, biases, expert * N + out_row,
          x + slot * K, simd_lid, result);
    }
    if (simd_lid == 0) {
      for (int row = 0; row < RPS; row++) {
        part[slot * RPS + row] = static_cast<T>(result[row]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_gid == 0 && int(simd_lid) < RPS) {
      // fused_moe_combine's arithmetic: each product and each add rounds to
      // T; the k-sum is MLX's one-row col_reduce_small (lane j % 8 folds
      // rows j, j + 8 onto +0, then lanes 1..7 add onto lane 0 in order).
      const int h = out_row + int(simd_lid);
      T lane[8];
      for (int l = 0; l < 8; ++l) {
        lane[l] = T(0.0f);
      }
      for (int j = 0; j < 10; ++j) {
        const T p = T(float(part[j * RPS + int(simd_lid)]) * float(scores[j]));
        lane[j % 8] = T(float(p) + float(lane[j % 8]));
      }
      T acc = lane[0];
      for (int l = 1; l < 8; ++l) {
        acc = T(float(lane[l]) + float(acc));
      }
      const float g = float(GATE_VALUE);
      const T e = T(1.0f + float(T(metal::precise::exp(metal::abs(g)))));
      const T sy = T(metal::precise::divide(1.0f, float(e)));
      const T sg = g < 0.0f ? sy : T(1.0f - float(sy));
      const T sh = T(float(sg) * float(SHARED_VALUE));
      y[h] = T(float(acc) + float(sh));
    }
"""

# Verify-window variants: row ``window_row`` of an M-row window runs the
# one-token source verbatim on its own input, routing and output (the kernel
# pointers are rebound in an inner scope). Threadgroup y is
# ``block * M + window_row``, so the rows of one weight block run back to
# back and share its fetch through the cache.
_GATE_UP_WINDOW_HEAD = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int ROWS = NSG * RPS;
    const int window_row = int(threadgroup_position_in_grid.y) % M;
    int b = int(threadgroup_position_in_grid.y) / M;
    float result[2 * RPS];
    const auto x_row = x + window_row * K;
    const auto rhs_row = rhs + window_row * 10;
    const auto y_row = y + window_row * (10 * NI + NS + 1);
    {
    const auto x = x_row;
    const auto rhs = rhs_row;
    const auto y = y_row;
"""

_DOWN_WINDOW_HEAD = r"""
    const uint3 tid = threadgroup_position_in_grid;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int window_row = int(tid.y) % M;
    const int out_row = int(tid.y) / M * RPS;
    const int slot = int(simd_gid);
    threadgroup T part[NPART * RPS];
    float result[RPS];
    const auto x_row = x + window_row * (10 * K + KS + 1);
    const auto rhs_row = rhs + window_row * 10;
    const auto scores_row = scores + window_row * 10;
    const auto y_row = y + window_row * N;
    {
    const auto x = x_row;
    const auto rhs = rhs_row;
    const auto scores = scores_row;
    const auto y = y_row;
"""

# Gate+up with the softmax + top-k folded in, for one-token decode (M = 1)
# and verify windows alike: threadgroup y = block * M + window_row serves row
# window_row with the blocks above, whose first block also selects that
# row's experts: its simdgroup 1 stores the TOP_K experts and their scores
# from the row's router logits (softmax_topk_row's outputs, which the down
# launch reads). Every simdgroup of an expert block recomputes selection
# ``slot`` from the logits with the same arithmetic before streaming its
# rows, so no launch sits between the router gemv and the experts.
_GATE_UP_TOPK_HEAD = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    constexpr int ROWS = NSG * RPS;
    const int window_row = int(threadgroup_position_in_grid.y) % M;
    int b = int(threadgroup_position_in_grid.y) / M;
    float result[2 * RPS];
    const auto x_row = x + window_row * K;
    const auto logits_row = logits + window_row * NE;
    const auto indices_row = indices + window_row * 10;
    const auto scores_row = scores + window_row * 10;
    const auto y_row = y + window_row * YW;
    {
    const auto x = x_row;
    const auto logits = logits_row;
    const auto indices = indices_row;
    const auto scores = scores_row;
    const auto y = y_row;
    if (b == 0 && simd_gid == 1) {
      omlx_router_topk<T, NE>::template write<10>(logits, simd_lid, indices, scores);
      return;
    }
"""

# Without a folded shared expert the first block only selects.
_GATE_UP_TOPK_UNSHARED = r"""
    if (b == 0) {
      return;
    }
    b -= 1;
"""

_GATE_UP_TOPK_ROUTED = _GATE_UP_ROUTED.replace(
    "size_t(rhs[slot])", "size_t(omlx_router_topk<T, NE>::nth(logits, simd_lid, slot))"
)


def _format_header(namespace: str, fmt: _Format) -> str:
    header = (
        _QMV_HEADER.replace("__BITS__", str(fmt.bits))
        .replace("__GS__", str(fmt.group_size))
        .replace("__FAST__", "1" if fmt.fast else "0")
    )
    return f"namespace {namespace} {{\n{header}\n{_ROWS}\n}}  // namespace {namespace}\n"


def _name(fmt: _Format) -> str:
    return f"b{fmt.bits}g{fmt.group_size}{'f' if fmt.fast else 's'}"


@cache
def _gate_up_kernel(routed: _Format, shared: _Format | None, gate: _Format | None):
    """Gate+up/SwiGLU launch; with ``shared`` (and ``gate``) it also runs the
    shared expert's gate+up and the shared-expert gate row."""
    header = _COMMON + _format_header("rt", routed)
    inputs = ["x", "w", "scales", "biases", "rhs"]
    source = _GATE_UP_HEAD
    name = f"omlx_qwen35_moe_gate_up_decode_{_name(routed)}"
    if shared is not None:
        header += _format_header("st", shared) + _format_header("gt", gate)
        inputs += ["sg_w", "sg_s", "sg_b", "su_w", "su_s", "su_b", "g_w", "g_s", "g_b"]
        source += _GATE_UP_SHARED
        name += f"_shared_{_name(shared)}_gate_{_name(gate)}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=["y"],
        header=header,
        source=source + _GATE_UP_ROUTED,
    )


@cache
def _down_kernel(routed: _Format, shared: _Format | None):
    """Down/combine launch; with ``shared`` it also runs the shared expert's
    down rows, otherwise the shared expert's output is an input."""
    header = _COMMON + _format_header("rd", routed)
    name = f"omlx_qwen35_moe_down_combine_decode_{_name(routed)}"
    if shared is None:
        inputs = ["shared", "gate", "x", "w", "scales", "biases", "rhs", "scores"]
        source = _DOWN_HEAD + _DOWN_TAIL.replace("SHARED_VALUE", "shared[h]").replace(
            "GATE_VALUE", "gate[0]"
        )
    else:
        header += _format_header("sd", shared)
        # x is the gate+up output: expert rows, shared rows, gate row.
        inputs = ["x", "w", "scales", "biases", "sd_w", "sd_s", "sd_b", "rhs", "scores"]
        source = (
            _DOWN_HEAD
            + _DOWN_SHARED_ROWS
            + _DOWN_TAIL.replace("SHARED_VALUE", "part[10 * RPS + int(simd_lid)]").replace(
                "GATE_VALUE", "x[10 * K + KS]"
            )
        )
        name += f"_shared_{_name(shared)}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=["y"],
        header=header,
        source=source,
    )


@cache
def _gate_up_window_kernel(routed: _Format, shared: _Format, gate: _Format):
    """The folded gate+up/SwiGLU launch for every row of a verify window."""
    header = (
        _COMMON
        + _format_header("rt", routed)
        + _format_header("st", shared)
        + _format_header("gt", gate)
    )
    return mx.fast.metal_kernel(
        name=(
            f"omlx_qwen35_moe_gate_up_window_{_name(routed)}"
            f"_shared_{_name(shared)}_gate_{_name(gate)}"
        ),
        input_names=[
            "x", "w", "scales", "biases", "rhs",
            "sg_w", "sg_s", "sg_b", "su_w", "su_s", "su_b", "g_w", "g_s", "g_b",
        ],
        output_names=["y"],
        header=header,
        source=_GATE_UP_WINDOW_HEAD + _GATE_UP_SHARED + _GATE_UP_ROUTED + "    }\n",
    )


@cache
def _down_window_kernel(routed: _Format, shared: _Format):
    """The folded down/combine launch for every row of a verify window."""
    header = _COMMON + _format_header("rd", routed) + _format_header("sd", shared)
    tail = _DOWN_TAIL.replace("SHARED_VALUE", "part[10 * RPS + int(simd_lid)]").replace(
        "GATE_VALUE", "x[10 * K + KS]"
    )
    return mx.fast.metal_kernel(
        name=f"omlx_qwen35_moe_down_combine_window_{_name(routed)}_shared_{_name(shared)}",
        input_names=["x", "w", "scales", "biases", "sd_w", "sd_s", "sd_b", "rhs", "scores"],
        output_names=["y"],
        header=header,
        source=_DOWN_WINDOW_HEAD + _DOWN_SHARED_ROWS + tail + "    }\n",
    )


@cache
def _gate_up_topk_kernel(routed: _Format, shared: _Format | None, gate: _Format | None):
    """Gate+up/SwiGLU with the router softmax + top-k folded in, for M rows;
    with ``shared`` (and ``gate``) it also runs the shared expert's gate+up
    and the shared-expert gate row."""
    header = _COMMON + TOPK_HEADER + _format_header("rt", routed)
    inputs = ["x", "w", "scales", "biases", "logits"]
    source = _GATE_UP_TOPK_HEAD
    name = f"omlx_qwen35_moe_gate_up_topk_{_name(routed)}"
    if shared is None:
        source += _GATE_UP_TOPK_UNSHARED
    else:
        header += _format_header("st", shared) + _format_header("gt", gate)
        inputs += ["sg_w", "sg_s", "sg_b", "su_w", "su_s", "su_b", "g_w", "g_s", "g_b"]
        source += _GATE_UP_SHARED
        name += f"_shared_{_name(shared)}_gate_{_name(gate)}"
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=["y", "indices", "scores"],
        header=header,
        source=source + _GATE_UP_TOPK_ROUTED + "    }\n",
    )


def _quantized_ok(layer, cls) -> bool:
    # Exactly the class whose call is a bare quantized_matmul / gather_qmm;
    # subclasses and repacked layers may compute differently.
    return (
        type(layer) is cls
        and layer.bits in _BITS
        and layer.group_size in _GROUP_SIZES
        and layer.mode == "affine"
        and "biases" in layer
        and "bias" not in layer
        and layer["scales"].dtype == mx.bfloat16
        and layer["biases"].dtype == mx.bfloat16
    )


def _mlx_format(layer, k: int, n: int) -> _Format | None:
    """The format MLX's one-row mat-vec runs for ``layer`` ([n, k] weights),
    or None when the packed shape does not match."""
    if layer["weight"].shape[-2:] != (n, k * layer.bits // 32) or k % layer.group_size:
        return None
    return _Format(layer.bits, layer.group_size, qmv_fast_layout(k, n, layer.bits))


def _address(a: mx.array) -> int:
    return np.frombuffer(memoryview(a), dtype=np.uint8).ctypes.data


def _expert_view(a: mx.array) -> mx.array:
    """``a[:1]`` when it shares ``a``'s buffer at offset 0, else ``a``.

    The kernels index every expert from the bound pointer, so a view that
    were copied (or offset) would read the wrong bytes; keep the whole array
    unless the first-expert slice provably aliases it."""
    view = a[:1]
    mx.eval(view)
    whole, first = memoryview(a), memoryview(view)
    if whole.c_contiguous and first.c_contiguous and _address(view) == _address(a):
        return view
    return a


class _Plan(NamedTuple):
    hidden: int
    fold: bool  # shared expert and its gate run inside the two launches
    gate_up_kernel: object
    gate_up_operands: tuple
    gate_up_template: list
    gate_up_grid: tuple
    gate_up_output: tuple
    down_kernel: object
    down_operands: tuple
    down_template: list
    down_grid: tuple
    down_threadgroup: tuple
    shared_gate_up_operands: tuple = ()  # gate_proj, up_proj, gate (fold)
    shared_down_operands: tuple = ()  # down_proj (fold)
    router_logits: object = None  # qwen35_moe_router.router_gemv launcher
    window_gate_up_kernel: object = None  # verify-window launches (fold)
    window_down_kernel: object = None
    topk_kernel: object = None  # gate+up with the router top-k folded in
    topk_blocks: int = 0  # its blocks per row
    topk_width: int = 0  # its gate+up output per row


def _shared_formats(block, hidden: int):
    """(gate+up, down, gate) formats and operands of a foldable shared
    expert and gate, or None."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    shared, gate = block.get("shared_expert"), block.get("shared_expert_gate")
    if type(shared) is not Qwen3_5MLP:
        return None
    layers = (shared.get("gate_proj"), shared.get("up_proj"), shared.get("down_proj"), gate)
    if not all(_quantized_ok(layer, nn.QuantizedLinear) for layer in layers):
        return None
    gate_proj, up_proj, down_proj, _ = layers
    width = gate_proj["weight"].shape[0]
    gu_fmt = _mlx_format(gate_proj, hidden, width)
    d_fmt = _mlx_format(down_proj, width, hidden)
    g_fmt = _mlx_format(gate, hidden, 1)
    if (
        gu_fmt is None
        or d_fmt is None
        or g_fmt is None
        or _mlx_format(up_proj, hidden, width) != gu_fmt
        or width % (_GATE_UP_ROWS * _GATE_UP_SIMDGROUPS)
    ):
        return None
    operands = lambda layer: tuple(layer[k] for k in ("weight", "scales", "biases"))
    return (
        width,
        gu_fmt,
        d_fmt,
        g_fmt,
        operands(gate_proj) + operands(up_proj) + operands(gate),
        operands(down_proj),
    )


def _build_plan(block) -> _Plan | None:
    """Kernels and operands for one MoE block, or None outside the layout."""
    from mlx_vlm.models.switch_layers import QuantizedSwitchLinear, SwiGLU

    switch_mlp = block.get("switch_mlp")
    if block.training or switch_mlp is None:
        return None
    if type(switch_mlp.get("activation")) is not SwiGLU:
        return None
    gate_up = switch_mlp.get("gate_up_proj")
    down = switch_mlp.get("down_proj")
    if not (
        _quantized_ok(gate_up, QuantizedSwitchLinear)
        and _quantized_ok(down, QuantizedSwitchLinear)
    ):
        return None
    hidden = down["weight"].shape[1]
    inter = down["weight"].shape[-1] * 32 // down.bits
    gu_fmt = _mlx_format(gate_up, hidden, 2 * inter)
    d_fmt = _mlx_format(down, inter, hidden)
    if (
        gu_fmt is None
        or d_fmt is None
        or not gu_fmt.fast
        or d_fmt.fast
        or gate_up["weight"].shape[0] != down["weight"].shape[0]
    ):
        return None
    view = _expert_view if _VIEWS_ENABLED else (lambda a: a)
    gate_up_operands = tuple(view(gate_up[k]) for k in ("weight", "scales", "biases"))
    down_operands = tuple(view(down[k]) for k in ("weight", "scales", "biases"))
    shared = _shared_formats(block, hidden) if _SHARED_FOLD else None
    gate = block.get("gate")
    router_logits = None
    if type(gate) is nn.Linear and "bias" not in gate and gate["weight"].shape[-1] == hidden:
        router_logits = router_gemv(gate["weight"])
    rows = _GATE_UP_ROWS * _GATE_UP_SIMDGROUPS
    gate_up_template = [
        ("T", mx.bfloat16),
        ("K", hidden),
        ("NI", inter),
        ("RPS", _GATE_UP_ROWS),
        ("NSG", _GATE_UP_SIMDGROUPS),
    ]
    down_template = [
        ("T", mx.bfloat16),
        ("K", inter),
        ("N", hidden),
        ("RPS", _DOWN_ROWS),
    ]
    if shared is None:
        return _Plan(
            hidden=hidden,
            router_logits=router_logits,
            fold=False,
            gate_up_kernel=_gate_up_kernel(gu_fmt, None, None),
            gate_up_operands=gate_up_operands,
            gate_up_template=gate_up_template,
            gate_up_grid=(32, _GATE_UP_SIMDGROUPS * TOP_K * inter // rows, 1),
            gate_up_output=(TOP_K, inter),
            down_kernel=_down_kernel(d_fmt, None),
            down_operands=down_operands,
            down_template=down_template + [("NPART", TOP_K)],
            down_grid=(32, TOP_K * hidden // _DOWN_ROWS, 1),
            down_threadgroup=(32, TOP_K, 1),
            topk_kernel=_gate_up_topk_kernel(gu_fmt, None, None) if _TOPK_FOLD else None,
            topk_blocks=1 + TOP_K * inter // rows,
            topk_width=TOP_K * inter,
        )
    width, sgu_fmt, sd_fmt, g_fmt, shared_gate_up, shared_down = shared
    return _Plan(
        hidden=hidden,
        router_logits=router_logits,
        fold=True,
        gate_up_kernel=_gate_up_kernel(gu_fmt, sgu_fmt, g_fmt),
        gate_up_operands=gate_up_operands,
        gate_up_template=gate_up_template + [("NS", width)],
        gate_up_grid=(
            32,
            _GATE_UP_SIMDGROUPS * (1 + width // rows + TOP_K * inter // rows),
            1,
        ),
        gate_up_output=(TOP_K * inter + width + 1,),
        down_kernel=_down_kernel(d_fmt, sd_fmt),
        down_operands=down_operands,
        down_template=down_template + [("KS", width), ("NPART", TOP_K + 1)],
        down_grid=(32, (TOP_K + 1) * hidden // _DOWN_ROWS, 1),
        down_threadgroup=(32, TOP_K + 1, 1),
        shared_gate_up_operands=shared_gate_up,
        shared_down_operands=shared_down,
        window_gate_up_kernel=_gate_up_window_kernel(gu_fmt, sgu_fmt, g_fmt),
        window_down_kernel=_down_window_kernel(d_fmt, sd_fmt),
        topk_kernel=_gate_up_topk_kernel(gu_fmt, sgu_fmt, g_fmt) if _TOPK_FOLD else None,
        topk_blocks=1 + width // rows + TOP_K * inter // rows,
        topk_width=TOP_K * inter + width + 1,
    )


def routed_decode_plan(block, x) -> _Plan | None:
    """The block's cached plan when ``x`` is one bf16 token it can route."""
    if _DISABLED or x.dtype != mx.bfloat16 or block.top_k != TOP_K:
        return None
    hidden = x.shape[-1]
    if x.size != hidden:
        return None
    # Depth 2 keys the plan on the expert and shared-expert weight arrays.
    plan = cached_per_module(block, "_omlx_routed_decode_plan", _build_plan, depth=2)
    if plan is None or plan.hidden != hidden:
        return None
    return plan


def _down(plan: _Plan, h, indices, scores, shape, shared=None, gate=None):
    """The one-token down/combine launch over gate+up output ``h``."""
    if plan.fold:
        down_inputs = [h, *plan.down_operands, *plan.shared_down_operands, indices, scores]
    else:
        # The shared expert comes first in the inputs, so its launches are
        # encoded (and can run) before the router's.
        down_inputs = [shared, gate, h, *plan.down_operands, indices, scores]
    return plan.down_kernel(
        inputs=down_inputs,
        template=plan.down_template,
        grid=plan.down_grid,
        threadgroup=plan.down_threadgroup,
        output_shapes=[shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def routed_decode(plan: _Plan, x, indices, scores, shared=None, gate=None):
    """``(switch_mlp(x, indices) * scores[..., None]).sum(axis=-2)
    + mx.sigmoid(shared_expert_gate(x)) * shared_expert(x)`` for one token
    in two launches. ``shared`` and ``gate`` are the composed shared expert
    and gate outputs, used only when the plan does not fold them."""
    h = plan.gate_up_kernel(
        inputs=[x, *plan.gate_up_operands, indices, *plan.shared_gate_up_operands],
        template=plan.gate_up_template,
        grid=plan.gate_up_grid,
        threadgroup=(32, _GATE_UP_SIMDGROUPS, 1),
        output_shapes=[plan.gate_up_output],
        output_dtypes=[mx.bfloat16],
    )[0]
    return _down(plan, h, indices, scores, x.shape, shared, gate)


def _topk_folds(plan: _Plan, logits) -> bool:
    return (
        plan.topk_kernel is not None
        and logits.size // logits.shape[-1] <= _TOPK_FOLD_MAX_ROWS
        and softmax_topk_eligible(logits)
    )


def _gate_up_topk(plan: _Plan, x, logits):
    """Gate+up output, indices and scores of every row of ``x`` from its
    router ``logits`` (the rows' softmax + top-k run inside the launch)."""
    rows = x.size // plan.hidden
    return plan.topk_kernel(
        inputs=[x, *plan.gate_up_operands, logits, *plan.shared_gate_up_operands],
        template=plan.gate_up_template
        + [("NE", logits.shape[-1]), ("M", rows), ("YW", plan.topk_width)],
        grid=(32, _GATE_UP_SIMDGROUPS * plan.topk_blocks * rows, 1),
        threadgroup=(32, _GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(rows, plan.topk_width), (rows, TOP_K), (rows, TOP_K)],
        output_dtypes=[mx.bfloat16, mx.uint32, mx.bfloat16],
    )


def routed_decode_logits(plan: _Plan, x, logits, shared=None, gate=None):
    """``routed_decode`` on the routing of one token's router ``logits``
    (``softmax_topk_row``'s), selected inside the gate+up launch."""
    h, indices, scores = _gate_up_topk(plan, x, logits)
    return _down(plan, h, indices, scores, x.shape, shared, gate)


def _window_down(plan: _Plan, h, indices, scores):
    rows = indices.shape[0]
    return plan.window_down_kernel(
        inputs=[h, *plan.down_operands, *plan.shared_down_operands, indices, scores],
        template=plan.down_template + [("M", rows)],
        grid=(32, plan.down_grid[1] * rows, 1),
        threadgroup=plan.down_threadgroup,
        output_shapes=[(rows, plan.hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]


def routed_window(plan: _Plan, x, indices, scores):
    """``routed_decode`` for each row of ``x`` ([M, hidden], 2..M rows,
    ``indices`` / ``scores`` [M, TOP_K]) in the same two launches: row r
    runs the one-token arithmetic on its own routing. Needs a folded plan."""
    rows = x.shape[0]
    h = plan.window_gate_up_kernel(
        inputs=[x, *plan.gate_up_operands, indices, *plan.shared_gate_up_operands],
        template=plan.gate_up_template + [("M", rows)],
        grid=(32, plan.gate_up_grid[1] * rows, 1),
        threadgroup=(32, _GATE_UP_SIMDGROUPS, 1),
        output_shapes=[(rows, *plan.gate_up_output)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return _window_down(plan, h, indices, scores)


def routed_window_logits(plan: _Plan, x, logits):
    """``routed_window`` on the routing of the rows' router ``logits``,
    selected inside the gate+up launch."""
    h, indices, scores = _gate_up_topk(plan, x, logits)
    return _window_down(plan, h, indices, scores)


def routed_verify_window(block, x):
    """The block's output for a row-exact verify window ``x`` ([B, L, hidden],
    1..``WINDOW_MAX_ROWS`` rows), each row bit-identical to the fused
    one-token call on that row, or None outside the fused layout.

    Router gemv, gate+up (with the softmax + top-k folded in) and
    down/combine each run once for the whole window."""
    global _WINDOW_DISABLED, _WINDOW_PROVEN
    if (
        not _VERIFY_WINDOW
        or _WINDOW_DISABLED
        or _DISABLED
        or x.ndim != 3
        or x.dtype != mx.bfloat16
        or block.top_k != TOP_K
    ):
        return None
    hidden = x.shape[-1]
    rows = x.size // hidden
    if not 1 <= rows <= WINDOW_MAX_ROWS or not router_eligible(
        x, block.num_experts, WINDOW_MAX_ROWS
    ):
        return None
    plan = cached_per_module(block, "_omlx_routed_decode_plan", _build_plan, depth=2)
    if plan is None or plan.hidden != hidden or not plan.fold or plan.router_logits is None:
        return None
    x = x.reshape(rows, hidden)
    logits = plan.router_logits(x)
    folded = _topk_folds(plan, logits)
    if not folded:
        routing = softmax_topk_rows(logits, TOP_K, WINDOW_MAX_ROWS)
        if routing is None:
            routing = fused_router_topk(mx.softmax(logits, axis=-1, precise=True), TOP_K)
        inds, scores = routing
    try:
        if folded:
            y = (routed_decode_logits if rows == 1 else routed_window_logits)(plan, x, logits)
        elif rows == 1:
            y = routed_decode(plan, x, inds, scores)
        else:
            y = routed_window(plan, x, inds, scores)
        if not _WINDOW_PROVEN:
            mx.eval(y)
            _WINDOW_PROVEN = True
            logger.info("Qwen MoE fused verify-window experts engaged")
    except Exception:
        _WINDOW_DISABLED = True
        logger.warning("fused verify-window experts failed; verifier fallback", exc_info=True)
        return None
    return y


def _ensure_verify_window_patch(cls) -> None:
    """Serve row-exact verify MoE blocks through ``routed_verify_window``.

    Wraps the mlx-vlm verifier's ``_feed_forward`` (the router patch's verify
    entry) for ``cls`` blocks while row-exact verify is armed; other verify
    modes, blocks and shapes keep the wrapped body."""
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    original = Qwen3_5BatchInvariantForward._feed_forward
    if getattr(original, "_omlx_routed_verify_window", False):
        return

    @wraps(original)
    def verify_window(self, feed_forward, x):
        if is_row_exact_armed() and isinstance(feed_forward, cls):
            y = routed_verify_window(feed_forward, x)
            if y is not None:
                return y.reshape(x.shape)
        return original(self, feed_forward, x)

    verify_window._omlx_routed_verify_window = True
    Qwen3_5BatchInvariantForward._feed_forward = verify_window


def apply_qwen35_moe_routed_decode_patch() -> bool:
    """Wrap the router-fused mlx-vlm ``Qwen3_5MoeSparseMoeBlock`` call.

    Needs ``qwen35_moe_router`` applied first: the fast arm reuses its fused
    routing launch, so it selects the same experts with the same scores as
    the body it replaces."""
    if not _ENABLED or not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.qwen3_5_moe import language as vlm_moe
    except ImportError:
        return False
    cls = getattr(vlm_moe, "Qwen3_5MoeSparseMoeBlock", None)
    if cls is None or not getattr(cls, "_omlx_router_fused", False):
        return False
    if getattr(cls, "_omlx_routed_decode", False):
        return True
    orig_call = cls.__call__

    def patched_call(self, x):
        global _DISABLED, _PROVEN
        plan = routed_decode_plan(self, x)
        if plan is None or not router_eligible(x, self.num_experts):
            return orig_call(self, x)
        shared = shared_gate = None
        if not plan.fold:
            # Children by item: this runs for every MoE layer of every decode step.
            shared = self["shared_expert"](x)
            shared_gate = self["shared_expert_gate"](x)
            if shared.dtype != x.dtype or shared_gate.dtype != x.dtype:
                return orig_call(self, x)
        if plan.router_logits is not None:
            logits = plan.router_logits(x)
        else:
            logits = self["gate"](x)
        folded = _topk_folds(plan, logits)
        if not folded:
            routing = softmax_topk_row(logits, self.top_k)
            if routing is None:
                routing = fused_router_topk(mx.softmax(logits, axis=-1, precise=True), self.top_k)
            inds, scores = routing
            if scores.dtype != x.dtype:
                return orig_call(self, x)
        try:
            if folded:
                y = routed_decode_logits(plan, x, logits, shared, shared_gate)
            else:
                y = routed_decode(plan, x, inds, scores, shared, shared_gate)
            if not _PROVEN:
                # Surface a kernel build failure while the call can still
                # fall back, once per process.
                mx.eval(y)
                _PROVEN = True
                logger.info("Qwen MoE fused routed-expert decode engaged")
        except Exception:
            _DISABLED = True
            logger.warning(
                "fused routed-expert decode failed; composed fallback",
                exc_info=True,
            )
            return orig_call(self, x)
        return y

    patched_call._omlx_routed_decode_original = orig_call
    cls.__call__ = patched_call
    cls._omlx_routed_decode = True
    if _VERIFY_WINDOW:
        _ensure_verify_window_patch(cls)
    logger.info("Qwen MoE fused routed-expert decode patch applied")
    return True
