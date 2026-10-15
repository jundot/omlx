# SPDX-License-Identifier: Apache-2.0
"""Tensor-unit (NAX) sorted ``gather_qmm`` for M5 hosts, compiled at runtime.

The MoE prefill path (``SwitchGLU`` with ``sorted_indices=True``) runs every
routed expert GEMM through ``mx.gather_qmm``. On M5 GPUs mlx sends it to
the ``*_gather_qmm_rhs_nax`` kernels. Through mlx 0.32.2 that was a
row-block kernel: a threadgroup per 64-row block of the sorted rows,
re-running the whole K loop for every expert present in the block with
only that expert's rows active, so at real MoE prefill sizes (tens of rows
per expert, a third of the blocks spanning two experts) a large share of
the tensor-unit work was masked. mlx 0.32.3 schedules single-expert tiles
itself (ml-explore/mlx#4572) but still reads past the expert's K extent
when ``K % 64 != 0``, which ``m5_gather_qmm`` works around by dropping to
the slow steel path.

This module runs the same product on the tensor units with its own
segmented tile scheduling (faster than mlx 0.32.3's on an M5 Max for MoE
prefill), as ``mx.fast.metal_kernel`` kernels on top of the NAX
tile primitives of the installed mlx (``steel/gemm/nax.h``, read from the
package's ``include`` directory):

- a one-threadgroup pre-pass cuts every expert's run of sorted rows into
  (row_start, expert, rows) tiles of at most BM rows (64, 96 or 128), so
  partial tiles only occur at the end of a run;
- the matmul computes one single-expert BM x 64 output tile per
  threadgroup (BM / 32 x 2 simdgroups, each owning a 32 x 32 block). Two
  schedules share the tile list: ``seg`` (mlx's segmented kernel: the
  weight tile of a K step, 64 or 128 deep, dequantized into threadgroup
  memory between two barriers) and ``db`` (double-buffered 64-deep weight
  tiles, one barrier per K step). Both skip the 16-row activation
  fragments of a partial tile that hold no rows.
- threadgroups are either laid out (column, tile) as mlx does, or with
  the tile index on the grid's x axis in groups of 32 tiles and
  (group, column) on y, so every threadgroup of a row tile shares one x
  coordinate and a tile's columns run 32 threadgroups apart. On M5 Ultra
  the gain tracked how few x coordinates a row tile's threadgroups span
  (activation reuse across its columns): 3-28% from 36 rows per expert.
  Below that weight streaming dominates and the plain layout (each
  expert's column slabs in order) stays faster.

``_plan`` picks the schedule, tile height, K step and layout from the
mean rows per expert and K (measured on M5 Ultra at the Qwen3.8, GLM-5.3
and MiMo-V2.6 expert shapes).

Every configuration dequantizes exactly like mlx (fp32 ``scale * q +
bias`` rounded once to the activation dtype for affine; ``bfloat(e8m0) *
e2m1`` for MXFP4) and issues the same 16x32x16 tensor ops in the same K
order, so every output element is bit-identical to mlx's sorted kernel
wherever that kernel is correct. A K tail (a multiple of 32) runs only its
valid 32-deep sub-steps and never reads weight bytes or scales past K (the
stock kernel reads stale activations there; mlx's fixed kernel still reads
the weight bytes and scales past the row, which can be NaN at the end of
the last expert) and row offsets are 32-bit.

``sorted_gather_qmm_swiglu`` is the MoE gate/up projection with its
activation in the epilogue. The routed experts' gate and up weights are
concatenated along the output axis (``[E, 2 * n, K]``) and today's path
writes the ``[M, 2 * n]`` product, splits it and runs a compiled
elementwise kernel for ``silu(gate) * up`` (GLM-5.3: gate and up clipped
first). The epilogue variant loads each 64-row weight tile as 16-row blocks
alternating the gate rows and the up rows of the same 16 output columns
(same weight layout, same K loop), so every lane holds the fp32 gate and up
accumulators of the same output element; it rounds both to the activation
dtype exactly as the plain store does, applies MLX's own elementwise
functors in that dtype (the op sequence of the compiled kernel, compiled
like it at runtime) and writes only ``[M, n]``: bit-identical, with one
elementwise pass and the ``[M, 2 * n]`` write and read removed.

With ``row_map`` it also reads its activation rows in place: the MoE sort
copies every token's row once per selected expert (``x[order // k]``,
``[T * k, 1, K]``, 0.4-0.5 GB per layer at 8192-token chunks) only to feed
this matmul. The row-mapped variant takes the token rows ``[T, 1, K]`` and
the sorted row -> token row map instead; each lane addresses its four
activation rows through the map (offsets computed once per tile), so every
fragment holds the values it would read from the copy and the tensor ops
are unchanged: bit-identical, and the copy is never computed (callers keep
it lazy, see ``moe_routes.sort_routes``).

Supported: ``transpose=True``, rhs-indices only, ``x`` of shape
``[M, 1, K]`` with a flat sorted ``uint32`` index of length ``M``, bf16/fp16
activations, N a multiple of 32, affine 4/8-bit with group 32/64/128
(scales and biases in the activation dtype) and MXFP4 (group 32); the
epilogue additionally needs ``2 * n % 64 == 0``, the row map a uint32
``[M]`` map and fewer than 2**32 token-row elements. Anything else returns
None and the caller keeps the stock path.

The index must hold each expert's rows as one contiguous run: the tile
pre-pass relies on it (as mlx 0.32.3's own sorted kernel does), and an
expert split over two runs leaves rows unwritten. Every caller sorts the
routes globally first.
"""

from __future__ import annotations

import logging
import math
import threading
from functools import partial
from pathlib import Path
from typing import NamedTuple, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Output tile width and column simdgroups (fixed; the Metal source assumes
# them). Tile heights are multiples of 32 rows (one row simdgroup each).
_BN = 64
_WN = 2
_TILE_ROWS = (64, 96, 128)

# Largest expert count the one-threadgroup pre-pass handles (its run
# bounds live in threadgroup memory).
_MAX_EXPERTS = 2048

# Row tiles per grid-x group in the tile-on-x layout.
_GX = 32

_MLX_UTILS_HEADERS = (
    "mlx/backend/metal/kernels/utils.h",
    "mlx/backend/metal/kernels/bf16.h",
    "mlx/backend/metal/kernels/bf16_math.h",
    "mlx/backend/metal/kernels/complex.h",
    "mlx/backend/metal/kernels/defines.h",
    "mlx/backend/metal/kernels/logging.h",
)


# mlx headers the matmul kernels build on: the NAX tile primitives and the
# fp4/fp8 element types.
_MLX_MM_HEADERS = (
    "mlx/backend/metal/kernels/steel/gemm/nax.h",
    "mlx/backend/metal/kernels/fp4.h",
    "mlx/backend/metal/kernels/fp8.h",
)


def _read_mlx_headers(paths: tuple[str, ...]) -> Optional[str]:
    """Flatten mlx kernel headers from the installed package.

    ``mx.fast.metal_kernel`` already prepends mlx's ``utils.h`` preamble, so
    it (and what it includes) is skipped; quoted mlx includes are inlined
    once and ``#pragma once`` dropped, system includes are kept.
    """
    root = Path(mx.__file__).parent / "include"
    if not root.is_dir():
        return None
    seen = {root / p for p in _MLX_UTILS_HEADERS}

    def expand(rel: str) -> str:
        path = root / rel
        if path in seen:
            return ""
        seen.add(path)
        lines = []
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith('#include "mlx/') and stripped.endswith('"'):
                lines.append(expand(stripped[len('#include "') : -1]))
            elif stripped != "#pragma once":
                lines.append(line)
        return "\n".join(lines)

    try:
        return "\n".join(expand(p) for p in paths)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Tile pre-pass
# ---------------------------------------------------------------------------

_SCAN_HEADER = """
using namespace metal;

// Cuts the sorted rows into (row_start, expert, rows, 0) tiles of at most
// BM rows of one expert, expert-major (the tile order of mlx's segmented
// gather_qmm). One threadgroup: the run bounds of every expert are found in
// parallel over the rows, then a threadgroup scan of the per-expert tile
// counts gives each expert's first tile. At most max_tiles tiles are
// written (a guard for unsorted input, which the contract excludes).
template <int BM>
METAL_FUNC void omlx_gqmm_tile_scan(
    const device uint32_t* idx,
    const constant int* params,
    device uint32_t* tiles,
    device uint32_t* tile_count,
    threadgroup uint32_t* run_start,
    threadgroup uint32_t* run_end,
    threadgroup uint32_t* simd_tot,
    const uint lid,
    const uint tg_size,
    const uint sg,
    const uint lane) {
  const int M = params[0];
  const int E = params[1];
  const uint32_t max_tiles = uint32_t(params[2]);
  for (int e = int(lid); e < E; e += int(tg_size)) {
    run_start[e] = 0;
    run_end[e] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Each thread walks 4 consecutive rows per step, with their neighbours
  // (0xffffffff past either end, never a valid expert).
  for (int g0 = 4 * int(lid); g0 < M; g0 += 4 * int(tg_size)) {
    const int cnt = min(4, M - g0);
    uint32_t v[6];
    v[0] = g0 > 0 ? idx[g0 - 1] : 0xffffffffu;
    for (int j = 0; j < 4; j++) {
      v[j + 1] = j < cnt ? idx[g0 + j] : 0xffffffffu;
    }
    v[5] = g0 + 4 < M ? idx[g0 + 4] : 0xffffffffu;
    for (int j = 0; j < cnt; j++) {
      const uint32_t e = v[j + 1];
      if (e < uint32_t(E)) {
        if (v[j] != e) {
          run_start[e] = uint32_t(g0 + j);
        }
        if (v[j + 2] != e) {
          run_end[e] = uint32_t(g0 + j + 1);
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_simd = (tg_size + 31) / 32;
  uint32_t running = 0;
  for (int base = 0; base < E; base += int(tg_size)) {
    const int e = base + int(lid);
    uint32_t start = 0;
    uint32_t cnt = 0;
    if (e < E) {
      start = run_start[e];
      const uint32_t end = run_end[e];
      cnt = end > start ? end - start : 0;
    }
    const uint32_t nt = (cnt + BM - 1) / BM;
    const uint32_t local = simd_prefix_exclusive_sum(nt);
    const uint32_t stot = simd_sum(nt);
    if (lane == 0) {
      simd_tot[sg] = stot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint32_t prefix = 0;
    uint32_t total = 0;
    for (uint s = 0; s < n_simd; s++) {
      const uint32_t v = simd_tot[s];
      prefix += (s < sg) ? v : 0;
      total += v;
    }
    const uint32_t off = running + prefix + local;
    for (uint32_t j = 0; j < nt && off + j < max_tiles; j++) {
      const uint32_t r = start + j * BM;
      *((device uint4*)tiles + off + j) =
          uint4(r, uint32_t(e), min(uint32_t(BM), start + cnt - r), 0);
    }
    running += total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    tile_count[0] = min(running, max_tiles);
  }
}
"""

_SCAN_SOURCE = """
    threadgroup uint32_t run_start[MAXE];
    threadgroup uint32_t run_end[MAXE];
    threadgroup uint32_t simd_tot[32];
    omlx_gqmm_tile_scan<BM>(
        idx, params, tiles, tile_count, run_start, run_end, simd_tot,
        thread_index_in_threadgroup, threads_per_threadgroup.x,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""

# ---------------------------------------------------------------------------
# Matmul
# ---------------------------------------------------------------------------

_MM_HEADER = """
using namespace metal;
using namespace mlx::steel;

namespace omlx_gqmm {

STEEL_CONST int kBN = 64;
STEEL_CONST int kWN = 2;
STEEL_CONST short kSM = 32;
STEEL_CONST short kSN = kBN / kWN;
STEEL_CONST short kSK = 32;
STEEL_CONST short kTM = kSM / 16;
STEEL_CONST short kTN = kSN / 16;
STEEL_CONST short kTK = kSK / 16;

// Tile geometry: BM rows in BM / 32 row simdgroups times kWN column
// simdgroups, K steps BK deep. kLT loader threads dequantize the kBN x BK
// weight tile, each kVPT consecutive values of one weight row: every
// thread when they split the tile evenly, else the largest power of two
// below the thread count (96-row tiles: 128 of 192).
template <int BM, int BK>
struct Geo {
  STEEL_CONST int kBM = BM;
  STEEL_CONST int kBK = BK;
  STEEL_CONST int kWM = BM / kSM;
  STEEL_CONST int kThreads = kWM * kWN * 32;
  STEEL_CONST int kLT = (kThreads & (kThreads - 1)) == 0
      ? kThreads
      : (kThreads > 256 ? 256 : (kThreads > 128 ? 128 : 64));
  STEEL_CONST int kVPT = kBN * BK / kLT;
  STEEL_CONST int kTPR = BK / kVPT;
  static_assert(BM % kSM == 0 && BK % kSK == 0, "tile geometry");
  static_assert(kTPR >= 1 && kTPR * kVPT == BK, "loader split");
};

// Affine: w = scale * q + bias computed in fp32 and rounded once to T, as
// mlx's dequantize() does (scale * q is exact in fp32).
template <typename T, int GS, int BITS>
struct AffineQ {
  using WT = T;
  STEEL_CONST int kBits = BITS;
  STEEL_CONST int kGroup = GS;
  const device T* scales;
  const device T* biases;

  struct P {
    float s;
    float b;
  };

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
    biases += n;
  }
  METAL_FUNC P params(const int g) const thread {
    return P{float(scales[g]), float(biases[g])};
  }
  METAL_FUNC static WT dq(thread const P& p, const uint32_t q) {
    return static_cast<WT>(p.s * float(q) + p.b);
  }
};

// MXFP4: e2m1 values times the e8m0 group scale, dequantized to bfloat like
// mlx's fp QuantizedBlockLoader (Wtype = bfloat).
template <int GS>
struct Mxfp4Q {
  using WT = bfloat;
  STEEL_CONST int kBits = 4;
  STEEL_CONST int kGroup = GS;
  const device uint8_t* scales;

  struct P {
    float s;
  };

  METAL_FUNC void advance(const size_t n) thread {
    scales += n;
  }
  METAL_FUNC P params(const int g) const thread {
    uint8_t sb = scales[g];
    return P{float(static_cast<bfloat>(*(thread fp8_e8m0*)(&sb)))};
  }
  METAL_FUNC static WT dq(thread const P& p, const uint32_t q) {
    uint8_t qb = uint8_t(q);
    return static_cast<WT>(p.s * float(*(thread fp4_e2m1*)(&qb)));
  }
};

// Gate/up pairing (activation epilogue): weight rows [gate; up] of one
// expert, half_n rows each. Row r of a paired kBN x BK weight tile loads
// weight row pair_row(r) past the tile's first gate row: 16-row blocks
// alternate the gate and the up rows of the same 16 output columns, so the
// two 16-column fragments of every simdgroup's 32-column block accumulate
// gate and up of the same output columns in the same lanes.
METAL_FUNC int pair_row(const int r, const int half_n) {
  return ((r >> 4) & 1) * half_n + ((r >> 5) << 4) + (r & 15);
}

// Activation epilogue of a paired simdgroup block (defined with the
// activation kernels only; see _ACT_HEADER).
template <typename T, int EPI, typename DTile>
METAL_FUNC void store_act(
    thread const DTile& D,
    device T* y,
    const int ld,
    const int rows,
    const T limit);

// Weight-tile loader: loader thread lid owns row lid / kTPR of the
// kBN x BK tile and the kVPT values from column (lid % kTPR) * kVPT, in
// kNG chunks that each lie in one quantization group. fetch() reads the
// packed words and group parameters of one K step, store() dequantizes
// them into threadgroup memory (row stride BKP). The *_tail variants
// cover a K tail of k_valid (a multiple of 32) columns and never touch a
// word or group at or past it. PAIR maps tile rows through pair_row().
template <typename Q, typename G, bool PAIR = false>
struct TileLoader {
  using WT = typename Q::WT;
  using P = typename Q::P;
  STEEL_CONST int kBits = Q::kBits;
  STEEL_CONST int kVPT = G::kVPT;
  STEEL_CONST int kWords = kVPT * kBits / 32;
  STEEL_CONST int kPer = 32 / kBits;
  STEEL_CONST uint32_t kMask = (1u << kBits) - 1u;
  STEEL_CONST int kGV = kVPT < Q::kGroup ? kVPT : Q::kGroup;
  STEEL_CONST int kNG = kVPT / kGV;
  STEEL_CONST int kWPG = kGV * kBits / 32;
  STEEL_CONST int kBKP = G::kBK + 16 / sizeof(WT);
  static_assert(kWords * 32 == kVPT * kBits, "whole words per thread");
  static_assert(kWPG >= 1 && kNG * kWPG == kWords, "group split");

  const device uint32_t* src;
  Q q;
  const short row;
  const short col;
  uint32_t raw[kWords];
  P p[kNG];

  METAL_FUNC TileLoader(
      const device uint8_t* w_tile,
      const int K,
      thread const Q& q_,
      const uint lid,
      const int half_n = 0) thread
      : q(q_),
        row(short(lid / G::kTPR)),
        col(short((lid % G::kTPR) * kVPT)) {
    const size_t w_off = PAIR ? size_t(pair_row(row, half_n)) : size_t(row);
    src = (const device uint32_t*)(w_tile + w_off * (K * kBits / 8) +
                                   col * kBits / 8);
    q.advance(w_off * (K / Q::kGroup));
  }

  METAL_FUNC void fetch(const int kb) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      raw[i] = ptr[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
    }
  }

  METAL_FUNC void fetch_tail(const int kb, const int k_valid) thread {
    const device uint32_t* ptr = src + kb * (G::kBK * kBits / 32);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      if (col + i * kPer < k_valid) {
        raw[i] = ptr[i];
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short g = 0; g < kNG; g++) {
      if (col + g * kGV < k_valid) {
        p[g] = q.params((kb * G::kBK + col + g * kGV) / Q::kGroup);
      }
    }
  }

  METAL_FUNC void store_words(threadgroup WT* Ws, const int k_valid) const
      thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kWords; i++) {
      if (col + i * kPer < k_valid) {
        vec<WT, kPer> v;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kPer; j++) {
          v[j] = Q::dq(p[i / kWPG], (raw[i] >> (kBits * j)) & kMask);
        }
        *(threadgroup vec<WT, kPer>*)(dst + i * kPer) = v;
      }
    }
  }

  METAL_FUNC void store(threadgroup WT* Ws) const thread {
    store_words(Ws, G::kBK);
  }

  METAL_FUNC void zero(threadgroup WT* Ws) const thread {
    threadgroup WT* dst = Ws + row * kBKP + col;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kVPT; i++) {
      dst[i] = WT(0);
    }
  }
};

// One 32-deep sub-step of a simdgroup's 32 x 32 block: full row blocks run
// tile_matmad_nax; partial ones skip the 16-row fragments without rows
// (the tensor ops of the others are the ones tile_matmad_nax issues).
template <typename T, typename WT, int BKP, bool FULL>
METAL_FUNC void sub_step(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xn,
    const threadgroup WT* ws,
    const int K,
    const short sgp_sm) {
  NAXTile<WT, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;

    volatile int compiler_barrier;

    Atile.load(xn, K);
    Btile.template load<WT, BKP, 1>(ws);

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<true>{});

    (void)compiler_barrier;
  } else {
    Btile.template load<WT, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<T, 1, kTK> Arow;
        Arow.load_safe(xn + mm * 16 * K, K, short2(kSK, sgp_sm - mm * 16));
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < kTN; nn += 2) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            BaseNAXFrag::mma(
                Dtile.frag_at(mm, nn),
                Dtile.frag_at(mm, nn + 1),
                Arow.frag_at(0, kk),
                metal::bool_constant<false>{},
                Btile.frag_at(nn, kk),
                Btile.frag_at(nn + 1, kk),
                metal::bool_constant<true>{});
          }
        }
      }
    }
  }
}

// Row-mapped activations (MAP): sorted row r of the product is token row
// rmap[r] of x, read in place instead of from a replicated copy. a_off[i][h]
// is this lane's element offset of activation row i * 16 + h * 8 + sc.y of
// its simdgroup block (rmap[row] * K + sc.x; rows past the block point at
// its last row), so lane values are exactly those NAXTile::load reads from
// the copy.
template <typename T, short R>
METAL_FUNC void load_a_map(
    thread NAXTile<T, R, kTK>& A,
    const device T* xk,
    const thread uint (&a_off)[kTM][2],
    const short i0) {
  STEEL_PRAGMA_UNROLL
  for (short r = 0; r < R; r++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const device T* xp = xk + a_off[i0 + r][h];
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < kTK; kk++) {
        const vec<T, 4> v = *(const device vec<T, 4>*)(xp + kk * 16);
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c++) {
          A.frag_at(r, kk)[h * 4 + c] = v[c];
        }
      }
    }
  }
}

// sub_step with row-mapped activations (xk: the token rows advanced to this
// sub-step's K offset): the same fragment values (rows past sgp_sm of a
// partial block zero as load_safe makes them) and the same tensor ops in the
// same order.
template <typename T, typename WT, int BKP, bool FULL>
METAL_FUNC void sub_step_map(
    thread NAXTile<float, kTM, kTN>& Dtile,
    const device T* xk,
    const thread uint (&a_off)[kTM][2],
    const threadgroup WT* ws,
    const short sgp_sm) {
  NAXTile<WT, kTN, kTK> Btile;
  if constexpr (FULL) {
    NAXTile<T, kTM, kTK> Atile;

    volatile int compiler_barrier;

    load_a_map<T, kTM>(Atile, xk, a_off, 0);
    Btile.template load<WT, BKP, 1>(ws);

    tile_matmad_nax(
        Dtile,
        Atile,
        metal::bool_constant<false>{},
        Btile,
        metal::bool_constant<true>{});

    (void)compiler_barrier;
  } else {
    const short2 sc = BaseNAXFrag::get_coord();
    Btile.template load<WT, BKP, 1>(ws);
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < kTM; mm++) {
      if (mm * 16 < sgp_sm) {
        NAXTile<T, 1, kTK> Arow;
        load_a_map<T, 1>(Arow, xk, a_off, mm);
        STEEL_PRAGMA_UNROLL
        for (short h = 0; h < 2; h++) {
          if (mm * 16 + h * 8 + sc.y >= sgp_sm) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < kTK; kk++) {
              STEEL_PRAGMA_UNROLL
              for (short c = 0; c < 4; c++) {
                Arow.frag_at(0, kk)[h * 4 + c] = T(0);
              }
            }
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short nn = 0; nn < kTN; nn += 2) {
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < kTK; kk++) {
            BaseNAXFrag::mma(
                Dtile.frag_at(mm, nn),
                Dtile.frag_at(mm, nn + 1),
                Arow.frag_at(0, kk),
                metal::bool_constant<false>{},
                Btile.frag_at(nn, kk),
                Btile.frag_at(nn + 1, kk),
                metal::bool_constant<true>{});
          }
        }
      }
    }
  }
}

// Element offsets of this lane's activation rows (see load_a_map) for the
// simdgroup block starting at tile row m0 of a tile of tile_rows rows at
// sorted row row_start.
METAL_FUNC void map_rows(
    thread uint (&a_off)[kTM][2],
    const device uint32_t* rmap,
    const int row_start,
    const int tile_rows,
    const int m0,
    const int K) {
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(m0 + i * 16 + h * 8 + int(sc.y), tile_rows - 1);
      a_off[i][h] = rmap[row_start + r] * uint(K) + uint(sc.x);
    }
  }
}

// seg: mlx's segmented sorted gather kernel (affine_gather_qmm_rhs_seg_nax /
// fp_gather_qmm_rhs_seg_nax): one single-expert BM x kBN tile per
// threadgroup, the weight tile of each BK-deep K step dequantized into
// threadgroup memory between two barriers. K tail (K % BK, a multiple of
// 32): only its sub-steps run. N tail: weight rows past N are zero, stores
// are bounded.
//
// EPI > 0 (activation epilogue; N = 2 * half_n [gate; up] rows, aligned):
// the tile of fused columns [y_col, y_col + kBN) computes gate and up of
// output columns [y_col / 2, y_col / 2 + kBN / 2) through the paired row
// map and writes act(gate, up) to the [M, half_n] output.
template <
    typename T,
    typename Q,
    typename G,
    bool ALIGN_N,
    bool ALIGN_K,
    int EPI = 0,
    bool MAP = false>
METAL_FUNC void gather_seg(
    const device T* x,
    const device uint32_t* rmap,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane,
    const T limit = T(0)) {
  using WT = typename Q::WT;
  constexpr bool kPair = EPI != 0;
  static_assert(!kPair || ALIGN_N, "paired gate/up tiles are full");
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const short tgp_bn = ALIGN_N ? short(kBN) : short(min(kBN, N - y_col));
  const int k_remain = K - K_it * G::kBK;
  // First weight row of the tile past the expert's rows and the output
  // row stride (paired: the tile's first output column, half_n columns).
  const int half_n = N / 2;
  const int w_col = kPair ? y_col / 2 : y_col;
  const int ldy = kPair ? half_n : N;

  const size_t w_row = size_t(expert) * N + w_col;
  q.advance(w_row * K_g);
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, K, q, sgid * 32 + lane, half_n);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);
  const bool row_live = ALIGN_N || loader.row < tgp_bn;

  if constexpr (!MAP) {
    x += size_t(row_start) * K;
  }
  y += size_t(row_start) * ldy + w_col;

  const short tm = kSM * short(sgid / kWN);
  const short tn = kSN * short(sgid % kWN);
  const short sgp_sm = short(min(int(kSM), max(0, rows - int(tm))));
  const short sgp_sn =
      ALIGN_N ? kSN : short(min(int(kSN), max(0, N - (y_col + tn))));
  const bool sg_active = sgp_sm > 0;
  uint a_off[kTM][2];
  if constexpr (MAP) {
    map_rows(a_off, rmap, row_start, rows, int(tm), K);
  }

  NAXTile<float, kTM, kTN> Dtile;
  Dtile.clear();
  // MAP: xn walks K over the token rows; a_off selects each lane's rows.
  const device T* xn = MAP ? x : x + tm * K;
  const threadgroup WT* ws = Ws + tn * BKP;

  dispatch_bool(sgp_sm == kSM, [&](auto kAlignedM) {
    for (int k = 0; k < K_it; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch(k);
          loader.store(Ws);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
        if (sg_active) {
          if constexpr (MAP) {
            sub_step_map<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, a_off, ws + kk1, sgp_sm);
          } else {
            sub_step<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
        }
      }
      xn += G::kBK;
    }

    if (!ALIGN_K) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (loads) {
        if (row_live) {
          loader.fetch_tail(K_it, k_remain);
          loader.store_words(Ws, k_remain);
        } else {
          loader.zero(Ws);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      STEEL_PRAGMA_NO_UNROLL
      for (int kk1 = 0; kk1 < k_remain; kk1 += kSK) {
        if (sg_active) {
          if constexpr (MAP) {
            sub_step_map<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, a_off, ws + kk1, sgp_sm);
          } else {
            sub_step<T, WT, BKP, kAlignedM.value>(
                Dtile, xn + kk1, ws + kk1, K, sgp_sm);
          }
        }
      }
    }

    if constexpr (kPair) {
      if (sg_active) {
        store_act<T, EPI>(
            Dtile, y + tm * ldy + tn / 2, ldy, int(sgp_sm), limit);
      }
    } else if (kAlignedM.value && sgp_sn == kSN) {
      Dtile.store(y + tm * N + tn, N);
    } else if (sg_active) {
      Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
    }
  });
}

// db: the same tiles and arithmetic with double-buffered 64-deep weight
// tiles: the packed words of step k + 1 are fetched before the tensor ops
// of step k and dequantized into the other buffer after them, so each K
// step has a single barrier. Activation fragments are read straight from
// device memory (rows past the tile are clamped to its last row and never
// stored) and 16-row fragments without rows of the tile are skipped.
// Requires K % 64 == 0 and N % 64 == 0. EPI > 0: the activation epilogue
// of gather_seg.
template <typename T, typename Q, typename G, int EPI = 0, bool MAP = false>
METAL_FUNC void gather_db(
    const device T* x,
    const device uint32_t* rmap,
    const device uint8_t* w,
    thread Q& q,
    const uint4 desc,
    const int y_col,
    device T* y,
    const int N,
    const int K,
    threadgroup typename Q::WT* Ws,
    const uint sgid,
    const uint lane,
    const T limit = T(0)) {
  using WT = typename Q::WT;
  static_assert(G::kBK == 64, "db runs 64-deep K steps");
  constexpr bool kPair = EPI != 0;
  constexpr int BKP = G::kBK + 16 / sizeof(WT);
  constexpr int kTile = kBN * BKP;
  const int row_start = int(desc.x);
  const uint32_t expert = desc.y;
  const int tile_rows = int(desc.z);

  const int K_w = K * Q::kBits / 8;
  const int K_g = K / Q::kGroup;
  const int K_it = K / G::kBK;
  const int half_n = N / 2;
  const int w_col = kPair ? y_col / 2 : y_col;

  const size_t w_row = size_t(expert) * N + w_col;
  q.advance(w_row * K_g);
  TileLoader<Q, G, kPair> loader(
      w + w_row * K_w, K, q, sgid * 32 + lane, half_n);
  const bool loads = G::kLT == G::kThreads || sgid * 32 + lane < uint(G::kLT);

  const int m0 = kSM * int(sgid / kWN);
  const int rows = min(int(kSM), tile_rows - m0);
  // MAP: x holds the token rows; offsets address them through rmap.
  const device T* xs = MAP
      ? x
      : x + size_t(row_start + max(0, min(m0, tile_rows - 1))) * K;

  const short2 sc = BaseNAXFrag::get_coord();
  metal::conditional_t<MAP, uint, int> x_off[kTM][2];
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kTM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < 2; h++) {
      const int r = min(int(i * 16 + sc.y + h * 8), max(rows, 1) - 1);
      if constexpr (MAP) {
        x_off[i][h] = rmap[row_start + min(m0 + r, tile_rows - 1)] * uint(K) +
            uint(sc.x);
      } else {
        x_off[i][h] = r * K + sc.x;
      }
    }
  }
  const short m_frags = rows > 0 ? short((rows + 15) / 16) : short(0);
  const threadgroup WT* wsg = Ws + (sgid % kWN) * kSN * BKP;

  NAXTile<float, kTM, kTN> D;
  D.clear();

  if (loads) {
    loader.fetch(0);
    loader.store(Ws);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int kb = 0; kb < K_it; kb++) {
    const bool more = kb + 1 < K_it;
    if (more && loads) {
      loader.fetch(kb + 1);
    }
    const threadgroup WT* wb = wsg + (kb & 1) * kTile;
    STEEL_PRAGMA_UNROLL
    for (short kk1 = 0; kk1 < G::kBK; kk1 += kSK) {
      NAXTile<WT, kTN, 2> Btile;
      Btile.template load<WT, BKP, 1>(wb + kk1);
      const int k = kb * G::kBK + kk1;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kTM; i++) {
        if (i < m_frags) {
          NAXTile<T, 1, 2> Atile;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const device T* xp = xs + x_off[i][h] + k;
            const vec<T, 4> a0 = *(const device vec<T, 4>*)(xp);
            const vec<T, 4> a1 = *(const device vec<T, 4>*)(xp + 16);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              Atile.frag_at(0, 0)[h * 4 + c] = a0[c];
              Atile.frag_at(0, 1)[h * 4 + c] = a1[c];
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short kk = 0; kk < 2; kk++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < kTN; j += 2) {
              BaseNAXFrag::mma(
                  D.frag_at(i, j),
                  D.frag_at(i, j + 1),
                  Atile.frag_at(0, kk),
                  metal::bool_constant<false>{},
                  Btile.frag_at(j, kk),
                  Btile.frag_at(j + 1, kk),
                  metal::bool_constant<true>{});
            }
          }
        }
      }
    }
    if (more && loads) {
      loader.store(Ws + ((kb + 1) & 1) * kTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if constexpr (kPair) {
    if (rows > 0) {
      store_act<T, EPI>(
          D,
          y + size_t(row_start + m0) * half_n + w_col + (kSN / 2) * (sgid % kWN),
          half_n,
          rows,
          limit);
    }
  } else {
    device T* yb = y + size_t(row_start + m0) * N + y_col + kSN * (sgid % kWN);
    if (rows >= kSM) {
      D.store(yb, N);
    } else if (rows > 0) {
      D.store_safe(yb, N, short2(kSN, short(rows)));
    }
  }
}

// The row tile and output column of this threadgroup. GX == 0: grid
// (columns, tiles) as mlx lays it out. GX > 0: tile t on grid x t % GX and
// (t / GX, column) on y, so all threadgroups of a row tile share one x
// coordinate and a tile's columns run GX threadgroups apart.
template <int GX>
METAL_FUNC bool tile_of(
    const device uint32_t* tiles,
    const uint tile_count,
    const uint3 tid,
    const int N,
    thread uint4& desc,
    thread int& y_col) {
  uint t;
  uint c;
  if constexpr (GX > 0) {
    const uint n_cols = uint((N + kBN - 1) / kBN);
    t = (tid.y / n_cols) * GX + tid.x;
    c = tid.y % n_cols;
  } else {
    t = tid.y;
    c = tid.x;
  }
  if (t >= tile_count) {
    return false;
  }
  desc = *((const device uint4*)tiles + t);
  y_col = int(c) * kBN;
  return true;
}

} // namespace omlx_gqmm
"""

_MM_SOURCE_TMPL = """
    {q_type}
    using G = omlx_gqmm::Geo<BM, BK>;
    using WT = typename Q::WT;
    constexpr int BKP = BK + 16 / sizeof(WT);
    threadgroup WT Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP +
                      PAD / sizeof(WT)];
    uint4 desc;
    int y_col;
    if (!omlx_gqmm::tile_of<GX>(
            tiles, tile_count[0], threadgroup_position_in_grid, params[0],
            desc, y_col)) {{
        return;
    }}
    {q_init}
    if constexpr (SCHED == 1) {{
        omlx_gqmm::gather_db<T, Q, G>(
            x, tiles, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    }} else {{
        omlx_gqmm::gather_seg<T, Q, G, ALIGN_N, ALIGN_K>(
            x, tiles, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
    }}
"""

_AFFINE_SOURCE = _MM_SOURCE_TMPL.format(
    q_type="using Q = omlx_gqmm::AffineQ<T, GS, BITS>;",
    q_init="Q q{scales, biases};",
)

_FP_SOURCE = _MM_SOURCE_TMPL.format(
    q_type="using Q = omlx_gqmm::Mxfp4Q<GS>;",
    q_init="Q q{scales};",
)

# ---------------------------------------------------------------------------
# Gate/up activation epilogue
# ---------------------------------------------------------------------------

# MLX's elementwise functors (Sigmoid, Multiply, Minimum, Maximum): the ones
# its compiled-graph kernels call, from the installed package.
_MLX_OPS_HEADERS = (
    "mlx/backend/metal/kernels/unary_ops.h",
    "mlx/backend/metal/kernels/binary_ops.h",
)

_ACT_HEADER = """
namespace omlx_gqmm {

// The unfused path's activation on the two rounded projections, op for op
// and every intermediate in T like MLX's compiled kernel of
// nn.silu(gate) * up (Sigmoid, Multiply, Multiply); EPI == 2 first clips
// like GLM-5.3's clamped SwiGLU: gate = minimum(gate, limit),
// up = minimum(maximum(up, -limit), limit).
template <typename T, int EPI>
METAL_FUNC T act(T g, T u, const T limit) {
  if constexpr (EPI == 2) {
    g = Minimum()(g, limit);
    u = Minimum()(Maximum()(u, T(-limit)), limit);
  }
  return Multiply()(Multiply()(g, Sigmoid()(g)), u);
}

// D.frag_at(i, 0) holds gate and D.frag_at(i, 1) up of the same 16 output
// columns (pair_row), so each lane holds both projections of its (row,
// column) elements. Each is rounded to T as the plain store rounds it,
// then act() writes the [rows, 16] block of the [M, ld] output (y points
// at its first element).
template <typename T, int EPI, typename DTile>
METAL_FUNC void store_act(
    thread const DTile& D,
    device T* y,
    const int ld,
    const int rows,
    const T limit) {
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < DTile::kTileRows; i++) {
    STEEL_PRAGMA_UNROLL
    for (short h = 0; h < BaseNAXFrag::kElemRows; h++) {
      const int r = i * BaseNAXFrag::kFragRows +
          h * BaseNAXFrag::kElemRowsJump + sc.y;
      if (r < rows) {
        vec<T, BaseNAXFrag::kElemCols> v;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < BaseNAXFrag::kElemCols; j++) {
          const short e = h * BaseNAXFrag::kElemCols + j;
          v[j] = act<T, EPI>(
              static_cast<T>(D.frag_at(i, 0)[e]),
              static_cast<T>(D.frag_at(i, 1)[e]),
              limit);
        }
        *(device vec<T, BaseNAXFrag::kElemCols>*)(y + size_t(r) * ld + sc.x) =
            v;
      }
    }
  }
}

} // namespace omlx_gqmm
"""

_ACT_SOURCE_TMPL = """
    {q_type}
    using G = omlx_gqmm::Geo<BM, BK>;
    using WT = typename Q::WT;
    constexpr int BKP = BK + 16 / sizeof(WT);
    threadgroup WT Ws[(SCHED == 1 ? 2 : 1) * omlx_gqmm::kBN * BKP +
                      PAD / sizeof(WT)];
    uint4 desc;
    int y_col;
    if (!omlx_gqmm::tile_of<GX>(
            tiles, tile_count[0], threadgroup_position_in_grid, params[0],
            desc, y_col)) {{
        return;
    }}
    {q_init}
    const T limit = lim[0];
    if constexpr (SCHED == 1) {{
        omlx_gqmm::gather_db<T, Q, G, EPI, {mapped}>(
            x, {rmap}, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup, limit);
    }} else {{
        omlx_gqmm::gather_seg<T, Q, G, true, ALIGN_K, EPI, {mapped}>(
            x, {rmap}, (const device uint8_t*)w, q, desc, y_col, y, params[0],
            params[1], Ws, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup, limit);
    }}
"""


def _act_source(q_type: str, q_init: str, mapped: bool) -> str:
    # Mapped kernels read the sorted rows through the ``rmap`` input; the
    # others pass ``tiles`` as the (unread) row map.
    return _ACT_SOURCE_TMPL.format(
        q_type=q_type,
        q_init=q_init,
        mapped="true" if mapped else "false",
        rmap="rmap" if mapped else "tiles",
    )


_AFFINE_Q = (
    "using Q = omlx_gqmm::AffineQ<T, GS, BITS>;",
    "Q q{scales, biases};",
)
_FP_Q = ("using Q = omlx_gqmm::Mxfp4Q<GS>;", "Q q{scales};")
_AFFINE_ACT_SOURCE = _act_source(*_AFFINE_Q, mapped=False)
_FP_ACT_SOURCE = _act_source(*_FP_Q, mapped=False)
_AFFINE_ACT_MAP_SOURCE = _act_source(*_AFFINE_Q, mapped=True)
_FP_ACT_MAP_SOURCE = _act_source(*_FP_Q, mapped=True)

# Epilogue kinds (the kernel's EPI): silu(gate) * up, and the clamped form.
_EPI_SWIGLU = 1
_EPI_CLAMPED = 2

_SCHED_SEG = 0
_SCHED_DB = 1
_SCHED_NAMES = {_SCHED_SEG: "seg", _SCHED_DB: "db"}


class Plan(NamedTuple):
    """One kernel configuration: schedule, tile rows, K step, layout, pad.

    ``gx`` is the row tiles per grid-x group (0: mlx's (column, tile)
    grid); ``pad`` is extra threadgroup memory in bytes (fewer resident
    threadgroups).
    """

    sched: int
    bm: int
    bk: int
    gx: int
    pad: int

    def describe(self) -> str:
        s = f"{_SCHED_NAMES[self.sched]} {self.bm}x{_BN} bk{self.bk}"
        if self.gx:
            s += f" gx{self.gx}"
        if self.pad:
            s += f" pad{self.pad}"
        return s


_lock = threading.RLock()
_kernels: dict[str, object] = {}
_header_failed = False
_act_header_failed = False
# Self-test verdict per kernel instantiation:
# (dtype, mode, bits, group_size, plan, align_n, align_k) -> bool, and for
# the activation epilogue the same key + (epi, limit).
_verified: dict[tuple, bool] = {}


def _get_act_kernel(kind: str):
    """Build (once) the ``affine_act`` or ``fp_act`` kernel object, or its
    row-mapped variant (``*_act_map``: sorted rows read through ``rmap``)."""
    global _act_header_failed
    kernel = _kernels.get(kind)
    if kernel is not None or _act_header_failed:
        return kernel
    with _lock:
        kernel = _kernels.get(kind)
        if kernel is not None:
            return kernel
        mlx_src = _read_mlx_headers(_MLX_MM_HEADERS + _MLX_OPS_HEADERS)
        if mlx_src is None:
            _act_header_failed = True
            logger.warning(
                "mlx kernel headers not found under %s; NAX gate/up "
                "activation epilogue disabled",
                Path(mx.__file__).parent / "include",
            )
            return None
        header = mlx_src + _MM_HEADER + _ACT_HEADER
        mapped = kind.endswith("_map")
        affine = kind.startswith("affine")
        inputs = ["x", "w", "scales"] + (["biases"] if affine else [])
        inputs += ["tiles", "tile_count", "params", "lim"]
        if mapped:
            inputs.append("rmap")
        source = {
            (True, False): _AFFINE_ACT_SOURCE,
            (False, False): _FP_ACT_SOURCE,
            (True, True): _AFFINE_ACT_MAP_SOURCE,
            (False, True): _FP_ACT_MAP_SOURCE,
        }[(affine, mapped)]
        kernel = mx.fast.metal_kernel(
            name=("omlx_gqmm_affine_swiglu" if affine else "omlx_gqmm_mxfp4_swiglu")
            + ("_map" if mapped else ""),
            input_names=inputs,
            output_names=["y"],
            header=header,
            source=source,
        )
        _kernels[kind] = kernel
        return kernel


def _get_kernel(kind: str):
    """Build (once) the ``scan``, ``affine`` or ``fp`` kernel object (and
    the ``*_act`` epilogue variants)."""
    global _header_failed
    if kind.endswith("_act") or kind.endswith("_act_map"):
        return _get_act_kernel(kind)
    kernel = _kernels.get(kind)
    if kernel is not None or _header_failed:
        return kernel
    with _lock:
        kernel = _kernels.get(kind)
        if kernel is not None:
            return kernel
        if kind == "scan":
            kernel = mx.fast.metal_kernel(
                name="omlx_gqmm_tile_scan",
                input_names=["idx", "params"],
                output_names=["tiles", "tile_count"],
                header=_SCAN_HEADER,
                source=_SCAN_SOURCE,
            )
        else:
            mlx_src = _read_mlx_headers(_MLX_MM_HEADERS)
            if mlx_src is None:
                _header_failed = True
                logger.warning(
                    "mlx kernel headers not found under %s; NAX sorted "
                    "gather_qmm disabled",
                    Path(mx.__file__).parent / "include",
                )
                return None
            if kind == "affine":
                kernel = mx.fast.metal_kernel(
                    name="omlx_gqmm_affine_v2",
                    input_names=[
                        "x",
                        "w",
                        "scales",
                        "biases",
                        "tiles",
                        "tile_count",
                        "params",
                    ],
                    output_names=["y"],
                    header=mlx_src + _MM_HEADER,
                    source=_AFFINE_SOURCE,
                )
            else:
                kernel = mx.fast.metal_kernel(
                    name="omlx_gqmm_mxfp4_v2",
                    input_names=["x", "w", "scales", "tiles", "tile_count", "params"],
                    output_names=["y"],
                    header=mlx_src + _MM_HEADER,
                    source=_FP_SOURCE,
                )
        _kernels[kind] = kernel
        return kernel


def _plan(rows: int, experts: int, K: int, N: int) -> Plan:
    """Kernel configuration for a call (mean rows per expert and K).

    Measured on M5 Ultra (real routing profiles, Qwen3.8 / GLM-5.3 /
    MiMo-V2.6 expert shapes at 1024-8192-token chunks):

    - fewer than 36 rows per expert, or K < 1024 (a short down projection)
      below 120 rows: weight streaming dominates; 64-row db tiles in mlx's
      (column, tile) layout;
    - 36-47 rows: 64-row db tiles, tile-on-x layout (+10%);
    - 48-95 rows: 96-row db tiles, tile-on-x layout (+3-26%);
    - 96+ rows (K >= 1024): 128-row seg tiles with 128-deep K steps and 8 KB
      of extra threadgroup memory (fewer resident threadgroups), tile-on-x
      layout (+5-29%);
    - K < 1024 from 120 rows: 96-row seg tiles, 128-deep K steps (+5%).

    Ragged K or N keeps 64-row seg tiles in the plain layout.
    """
    if K % 64 or N % 64:
        return Plan(_SCHED_SEG, 64, 64, 0, 0)
    per_expert = rows / max(1, experts)
    if per_expert < 36 or (K < 1024 and per_expert < 120):
        return Plan(_SCHED_DB, 64, 64, 0, 0)
    if K < 1024:
        return Plan(_SCHED_SEG, 96, 128, _GX, 0)
    if per_expert < 48:
        return Plan(_SCHED_DB, 64, 64, _GX, 0)
    if per_expert < 96:
        return Plan(_SCHED_DB, 96, 64, _GX, 0)
    return Plan(_SCHED_SEG, 128, 128, _GX, 8192)


def supports(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    indices: mx.array,
    group_size: int,
    bits: int,
    mode: str,
    row_map: Optional[mx.array] = None,
) -> bool:
    """True when ``sorted_gather_qmm`` handles this call (layout/dtypes).

    With ``row_map`` (uint32 ``[M]``, sorted row -> row of ``x``) the rows
    are read through the map and ``x`` may have any row count (32-bit
    element offsets: fewer than 2**32 elements)."""
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 3 or x.shape[1] != 1 or indices.ndim != 1:
        return False
    M, K = int(indices.shape[0]), int(x.shape[2])
    if row_map is None:
        if int(x.shape[0]) != M:
            return False
    elif (
        row_map.ndim != 1
        or int(row_map.shape[0]) != M
        or row_map.dtype != mx.uint32
        or int(x.shape[0]) * K >= 2**32
    ):
        return False
    # Fewer than 8 indices would be bound as a constant buffer.
    if M < 8 or indices.dtype != mx.uint32:
        return False
    if w.ndim != 3 or w.dtype != mx.uint32:
        return False
    E, N = int(w.shape[0]), int(w.shape[1])
    # Ragged N is canaried at N % 64 == 32 only.
    if E == 0 or E > _MAX_EXPERTS or N == 0 or N % 32 or K % 32:
        return False
    if mode == "affine":
        if bits not in (4, 8) or group_size not in (32, 64, 128):
            return False
        if biases is None or K % group_size:
            return False
        if scales.dtype != x.dtype or biases.dtype != x.dtype:
            return False
        if biases.shape != scales.shape:
            return False
    elif mode == "mxfp4":
        if bits != 4 or group_size != 32 or biases is not None:
            return False
        if scales.dtype != mx.uint8:
            return False
    else:
        return False
    if w.shape[2] * 32 != K * bits:
        return False
    return scales.shape == (E, N, K // group_size)


def _launch(
    x,
    w,
    scales,
    biases,
    indices,
    group_size,
    bits,
    mode,
    plan,
    stream,
    epi=0,
    limit=None,
    row_map=None,
):
    """Tile pre-pass + matmul. ``epi`` > 0 runs the activation epilogue on
    the ``[gate; up]`` rows of ``w`` (``N % 64 == 0``) and returns
    ``[M, 1, N / 2]``; with it, ``row_map`` reads the sorted rows as
    ``x[row_map]`` in place."""
    scan = _get_kernel("scan")
    kind = "affine" if mode == "affine" else "fp"
    if row_map is not None and not epi:
        return None
    mm = _get_kernel(
        (f"{kind}_act" + ("_map" if row_map is not None else "")) if epi else kind
    )
    if scan is None or mm is None:
        return None
    M, K = int(indices.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    bm = plan.bm
    max_tiles = (M + bm - 1) // bm + min(E, M)
    kw = {} if stream is None else {"stream": stream}
    tiles, tile_count = scan(
        inputs=[indices, mx.array([M, E, max_tiles], dtype=mx.int32)],
        template=[("BM", bm), ("MAXE", _MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
        **kw,
    )
    inputs = [x, w, scales]
    template = [("T", x.dtype), ("GS", group_size)]
    if mode == "affine":
        inputs.append(biases)
        template.append(("BITS", bits))
    inputs += [tiles, tile_count, mx.array([N, K], dtype=mx.int32)]
    template.append(("SCHED", int(plan.sched)))
    if epi:
        # The clip bound converts like the unfused path's Python-float
        # operand of mx.clip (float, then the activation dtype).
        bound = 0.0 if limit is None else limit
        inputs.append(mx.array(bound, dtype=x.dtype).reshape(1))
        template.append(("EPI", int(epi)))
        if row_map is not None:
            inputs.append(row_map)
    else:
        template.append(("ALIGN_N", N % _BN == 0))
    template += [
        ("ALIGN_K", K % plan.bk == 0),
        ("BM", bm),
        ("BK", plan.bk),
        ("GX", plan.gx),
        ("PAD", plan.pad),
    ]
    n_cols = (N + _BN - 1) // _BN
    if plan.gx:
        tg_grid = (plan.gx, ((max_tiles + plan.gx - 1) // plan.gx) * n_cols)
    else:
        tg_grid = (n_cols, max_tiles)
    return mm(
        inputs=inputs,
        template=template,
        grid=(tg_grid[0] * 32, tg_grid[1] * _WN, bm // 32),
        threadgroup=(32, _WN, bm // 32),
        output_shapes=[(M, 1, N // 2 if epi else N)],
        output_dtypes=[x.dtype],
        **kw,
    )[0]


def _stock_gather_qmm():
    """The raw mlx op, also when ``m5_gather_qmm`` has wrapped it."""
    fn = mx.gather_qmm
    if getattr(fn, "_omlx_m5_reroute", False):
        from omlx.patches import m5_gather_qmm

        fn = m5_gather_qmm._original_gather_qmm or fn
    return fn


# Canary routing: an empty expert, runs spanning several tiles of every
# height, and partial tiles of every size class.
_CANARY_COUNTS = (70, 0, 5, 33, 64, 17, 140, 11)


def _canary_k(plan: Plan, align_k: bool) -> int:
    # Aligned: K % BK == 0. Unaligned: a 64-deep tail (bk 128, still
    # K % 64 == 0) or a 32-deep ragged one (bk 64).
    return 256 if align_k else (320 if plan.bk == 128 else 160)


def _canary_problem(dtype, mode, bits, group_size, N, K, x_scale=0.5):
    """Quantized [E, N, K] experts, their dequantized weights, and sorted
    canary rows ``x`` [M, 1, K] (``x_scale``: a scalar or per-row scale)."""
    E = len(_CANARY_COUNTS)
    k_w, k_x = mx.random.split(mx.random.key(0x2267), 2)
    wf = (mx.random.normal((E, N, K), key=k_w) * 0.05).astype(dtype)
    if mode == "affine":
        wq, scales, biases = mx.quantize(wf, group_size=group_size, bits=bits)
        wd = mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits)
    else:
        wq, scales = mx.quantize(wf, group_size=group_size, bits=bits, mode=mode)
        biases = None
        wd = mx.dequantize(wq, scales, group_size=group_size, bits=bits, mode=mode)
    idx = mx.array(
        [e for e, n in enumerate(_CANARY_COUNTS) for _ in range(n)],
        dtype=mx.uint32,
    )
    M = int(idx.shape[0])
    x = (mx.random.normal((M, 1, K), key=k_x) * x_scale).astype(dtype)
    return wq, scales, biases, wd, x, idx


def _self_test(key: tuple) -> Optional[bool]:
    """Run one kernel instantiation on a small canary.

    K % 64 == 0 must be bit-identical to mlx's sorted kernel (correct
    there); ragged K must match an fp32 dequantized reference to bf16
    rounding. Returns None when the canary could not be evaluated here
    (e.g. while a function transformation is being traced); the caller then
    retries.
    """
    dtype, mode, bits, group_size, plan, align_n, align_k = key
    N = 128 if align_n else 96
    K = _canary_k(plan, align_k)
    try:
        wq, scales, biases, wd, x, idx = _canary_problem(
            dtype, mode, bits, group_size, N, K
        )
        out = _launch(x, wq, scales, biases, idx, group_size, bits, mode, plan, None)
        if out is None:
            return False
        if K % 64 == 0:
            ref = _stock_gather_qmm()(
                x,
                wq,
                scales,
                biases,
                rhs_indices=idx,
                transpose=True,
                group_size=group_size,
                bits=bits,
                mode=mode,
                sorted_indices=True,
            )
            ok = bool(mx.array_equal(out, ref).item())
            detail = "not bit-identical to mlx's sorted kernel"
        else:
            ref = (
                x.astype(mx.float32)
                @ wd[idx].swapaxes(-1, -2).astype(mx.float32)
            )
            err = mx.abs(out.astype(mx.float32) - ref).max().item()
            scale = mx.abs(ref).max().item()
            ok = err <= scale / 64
            detail = f"max err {err:.3g} vs fp32 reference (max {scale:.3g})"
    except Exception as e:  # noqa: BLE001
        if "transformation" in str(e):
            return None
        logger.warning(
            "NAX sorted gather_qmm self-test raised for %s: %s", _describe(key), e
        )
        return False
    if ok:
        logger.debug("NAX sorted gather_qmm armed for %s", _describe(key))
    else:
        logger.warning(
            "NAX sorted gather_qmm disabled for %s: canary %s",
            _describe(key),
            detail,
        )
    return ok


def _describe(key: tuple) -> str:
    dtype, mode, bits, group_size, plan, align_n, align_k = key[:7]
    epi = ""
    if len(key) > 7:
        limit = key[8]
        epi = " + silu(gate) * up" if limit is None else f" + clamped SwiGLU {limit:g}"
        if len(key) > 9:
            epi += ", row map"
    return (
        f"{str(dtype).rsplit('.', 1)[-1]} {mode} {bits}-bit gs{group_size} "
        f"({plan.describe()}{'' if align_n else ', ragged N'}"
        f"{'' if align_k else ', K tail'}){epi}"
    )


@partial(mx.compile, shapeless=True)
def _ref_swiglu(x_gate: mx.array, x_up: mx.array) -> mx.array:
    # mlx-lm's / mlx-vlm's swiglu (SwiGLU of their SwitchGLU and of oMLX's
    # GLM DSA / DeepSeek V4 SwitchGLU).
    return nn.silu(x_gate) * x_up


@partial(mx.compile, shapeless=True)
def _ref_clamped_swiglu(x_up: mx.array, x_gate: mx.array, limit: float) -> mx.array:
    # GLM-5.3's Glm5NextClampedSwiGLU (glm5_next _clamped_swiglu).
    x_gate = mx.clip(x_gate, a_min=None, a_max=limit)
    x_up = mx.clip(x_up, a_min=-limit, a_max=limit)
    return nn.silu(x_gate) * x_up


def reference_activation(
    x_up: mx.array, x_gate: mx.array, limit: Optional[float] = None
) -> mx.array:
    """The unfused path's activation: ``silu(gate) * up`` as MLX's compiled
    kernel computes it, clamped first like GLM-5.3 when ``limit`` is set."""
    if limit is None:
        return _ref_swiglu(x_gate, x_up)
    return _ref_clamped_swiglu(x_up, x_gate, float(limit))


def _bits_equal(a: mx.array, b: mx.array) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return bool(mx.array_equal(a.view(view), b.view(view)).item())


def _self_test_act(key: tuple) -> Optional[bool]:
    """Run one activation-epilogue instantiation on a small canary against
    the unfused path: the plain kernel with the same configuration, split
    into gate and up, then ``reference_activation`` (bitwise, including the
    signs of zeros). The canary rows span 0.1x to 16x the plain canary's
    scale so the projections cover sigmoid's saturated tails and the clip
    bounds. None: could not be evaluated here (see ``_self_test``)."""
    dtype, mode, bits, group_size, plan, _, align_k, epi, limit = key
    K = _canary_k(plan, align_k)
    try:
        M = sum(_CANARY_COUNTS)
        row_scale = mx.power(10.0, mx.linspace(-1.0, 1.2, M)).reshape(M, 1, 1)
        wq, scales, biases, _, x, idx = _canary_problem(
            dtype, mode, bits, group_size, 2 * _BN, K, x_scale=row_scale
        )
        out = _launch(
            x, wq, scales, biases, idx, group_size, bits, mode, plan, None,
            epi=epi, limit=limit,
        )
        gate_up = _launch(
            x, wq, scales, biases, idx, group_size, bits, mode, plan, None
        )
        if out is None or gate_up is None:
            return False
        x_gate, x_up = mx.split(gate_up, 2, axis=-1)
        ok = _bits_equal(out, reference_activation(x_up, x_gate, limit))
    except Exception as e:  # noqa: BLE001
        if "transformation" in str(e):
            return None
        logger.warning(
            "NAX gate/up activation self-test raised for %s: %s", _describe(key), e
        )
        return False
    if ok:
        logger.debug("NAX gate/up activation epilogue armed for %s", _describe(key))
    else:
        logger.warning(
            "NAX gate/up activation epilogue disabled for %s: canary not "
            "bit-identical to the unfused path",
            _describe(key),
        )
    return ok


def _self_test_act_map(key: tuple) -> Optional[bool]:
    """Run one row-mapped activation-epilogue instantiation on a canary: a
    scrambled, repeating row map over a smaller token array must reproduce
    the unmapped epilogue on the materialised rows ``x[row_map]`` bitwise
    (the same configuration's unmapped instantiation is itself checked
    against the unfused path). None: could not be evaluated here."""
    dtype, mode, bits, group_size, plan, _, align_k, epi, limit, _ = key
    K = _canary_k(plan, align_k)
    try:
        M = sum(_CANARY_COUNTS)
        T = M // 3
        tok_scale = mx.power(10.0, mx.linspace(-1.0, 1.2, T)).reshape(T, 1, 1)
        wq, scales, biases, _, _, idx = _canary_problem(
            dtype, mode, bits, group_size, 2 * _BN, K
        )
        x_tok = (
            mx.random.normal((T, 1, K), key=mx.random.key(0x70C)) * tok_scale
        ).astype(dtype)
        row_map = ((mx.arange(M, dtype=mx.uint32) * 7 + 3) % T).astype(mx.uint32)
        out = _launch(
            x_tok, wq, scales, biases, idx, group_size, bits, mode, plan, None,
            epi=epi, limit=limit, row_map=row_map,
        )
        ref = _launch(
            x_tok[row_map], wq, scales, biases, idx, group_size, bits, mode,
            plan, None, epi=epi, limit=limit,
        )
        if out is None or ref is None:
            return False
        ok = _bits_equal(out, ref)
    except Exception as e:  # noqa: BLE001
        if "transformation" in str(e):
            return None
        logger.warning(
            "NAX gate/up row-map self-test raised for %s: %s", _describe(key), e
        )
        return False
    if ok:
        logger.debug("NAX gate/up activation epilogue armed for %s", _describe(key))
    else:
        logger.warning(
            "NAX gate/up row map disabled for %s: canary not bit-identical to "
            "the materialised rows",
            _describe(key),
        )
    return ok


def _checked(key: tuple, test) -> bool:
    """The cached self-test verdict for ``key`` (running ``test`` once)."""
    ok = _verified.get(key)
    if ok is None:
        with _lock:
            ok = _verified.get(key)
            if ok is None:
                ok = test(key)
                if ok is not None:
                    _verified[key] = ok
    return bool(ok)


def sorted_gather_qmm(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
    stream=None,
    plan: Optional[Plan] = None,
    verify: bool = True,
) -> Optional[mx.array]:
    """``x @ w[indices].T`` for sorted rows on the tensor units.

    ``indices`` must group each expert's rows in one contiguous run (see
    the module docstring). ``plan`` pins a configuration (testing); by
    default ``_plan`` picks one. Returns None when the call is not
    supported (see ``supports``), the kernels cannot be
    built or the instantiation failed its one-time self-test; the caller
    then keeps the stock path.
    """
    if not supports(x, w, scales, biases, indices, group_size, bits, mode):
        return None
    M, K = int(x.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    if plan is None:
        plan = _plan(M, E, K, N)
    if plan.sched == _SCHED_DB and (K % 64 or N % 64 or plan.bk != 64):
        # db runs aligned 64-deep K steps only.
        plan = plan._replace(sched=_SCHED_SEG)
    if verify:
        key = (x.dtype, mode, bits, group_size, plan, N % _BN == 0, K % plan.bk == 0)
        if not _checked(key, _self_test):
            return None
    return _launch(x, w, scales, biases, indices, group_size, bits, mode, plan, stream)


def sorted_gather_qmm_swiglu(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: Optional[mx.array],
    indices: mx.array,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
    limit: Optional[float] = None,
    stream=None,
    plan: Optional[Plan] = None,
    verify: bool = True,
    row_map: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """The SwiGLU of a fused gate/up projection for sorted rows, in one kernel.

    ``w`` (with ``scales``/``biases``) holds each expert's gate rows followed
    by its up rows, ``[E, 2 * n, K]``. Returns ``[M, 1, n]``: for
    ``gate, up = split(sorted_gather_qmm(x, w, ...), 2, axis=-1)`` the
    activation ``silu(gate) * up``, or with ``limit`` GLM-5.3's clamped
    ``silu(minimum(gate, limit)) * clip(up, -limit, limit)``. The matmul
    computes gate and up of the same columns in each output tile (weight
    rows paired in the tile loader) and applies the activation to the
    rounded projections in its epilogue instead of writing ``[M, 2 * n]``
    for a separate elementwise pass. Bit-identical to the unfused path:
    each instantiation must match the plain kernel + split +
    ``reference_activation`` on a canary (and the plain kernel its own
    self-test) before it is used.

    With ``row_map`` (uint32 ``[M]``) the sorted rows are ``x[row_map]``
    (``x`` holds the token rows, ``[T, 1, K]``), read in place instead of
    from a replicated ``[M, 1, K]`` copy: the same tiles, values and tensor
    ops, so the output is bit-identical to passing ``x[row_map]`` (checked
    per instantiation on a scrambled canary map).

    Returns None when unsupported (``supports``, or ``2 * n % 64 != 0``) or
    not verified; the caller then
    keeps the unfused path (materialising ``x[row_map]``).
    """
    if not supports(x, w, scales, biases, indices, group_size, bits, mode, row_map):
        return None
    M, K = int(indices.shape[0]), int(x.shape[2])
    E, N = int(w.shape[0]), int(w.shape[1])
    if N % _BN:
        # Every 64-column tile pairs 32 gate with 32 up columns.
        return None
    if limit is not None:
        limit = float(limit)
        if not math.isfinite(limit):
            return None
    if plan is None:
        plan = _plan(M, E, K, N)
    if plan.sched == _SCHED_DB and (K % 64 or plan.bk != 64):
        plan = plan._replace(sched=_SCHED_SEG)
    epi = _EPI_SWIGLU if limit is None else _EPI_CLAMPED
    if verify:
        key = (x.dtype, mode, bits, group_size, plan, True, K % plan.bk == 0)
        # The plain instantiation (the unfused path's kernel) must hold too.
        if not _checked(key, _self_test) or not _checked(
            key + (epi, limit), _self_test_act
        ):
            return None
        if row_map is not None and not _checked(
            key + (epi, limit, "map"), _self_test_act_map
        ):
            return None
    return _launch(
        x, w, scales, biases, indices, group_size, bits, mode, plan, stream,
        epi=epi, limit=limit, row_map=row_map,
    )
