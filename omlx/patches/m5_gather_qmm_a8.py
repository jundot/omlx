# SPDX-License-Identifier: Apache-2.0
"""INT8-activation (A8) sorted ``gather_qmm`` for routed MoE experts on M5.

The routed-expert prefill path (``m5_gather_qmm_nax``) dequantizes the packed
Q4 weight to BF16 in threadgroup memory and multiplies it with BF16 tensor
ops. This module adds the operand path the dense oQ kernels already use
(``qwen35_oq_a8``): the packed codes are decoded straight into INT8 fragment
registers and multiplied through the INT8 x INT8 -> INT32 tensor op, with the
GS64 affine correction applied per group. Only the Gate+Up projection is
covered; the Down projection keeps the A16 path.

    checkpoint Q4 affine weight [E, N, K / 8] (uint32), GS64
    checkpoint scale / bias     [E, N, G], G = K / 64     (read in place)
            |
    Stage A: INT8 token rows, once per token row (before routing)
            |
    tile scan: the one-threadgroup pre-pass of ``m5_gather_qmm_nax``
            |
    stage this tile's 64 weight rows of scale / bias in threadgroup memory
            |
    Q4 codes -> INT8 fragments;  INT8 x INT8 -> INT32
            |
    GS64 affine correction;  SwiGLU on the accumulators
            |
    [M, N / 2] -> the existing A16 sorted down projection

Scheduling is deliberately *not* reimplemented: the tile list comes from the
same pre-pass, a threadgroup still computes one single-expert ``BM x BN``
output tile, and the activation rows are read in place through ``row_map``.
Only the operand specialization changes.

Contract (identical to the dense oQ A8 kernels; see
``tests/test_qwen35_oq_a8.py`` for the reference):

    Stage A      amax = max|x| per row; scale = amax / 127;
                 Qa = rint(x / scale) clipped to [-127, 127] as INT8;
                 Ra = the code sum of each GS64 group.
    GEMM         acc_g = sum over the group of Qa * Qw, exact in INT32.
    correction   out = Sa[m] * sum_g (Sw_g * acc_g + Bw_g * Ra_g)

Stage A runs once per token row and the kernel reads those rows through
``row_map``, so top-k replication does not repeat the quantization.

The kernel reads the checkpoint's ``[E, N, G]`` scale/bias in place. Each
threadgroup stages the metadata of its 64 weight rows as ``[group][row]`` in
threadgroup memory (2 x 64 x G x 2 bytes: 10 KiB for Flash-Next, 20 KiB at
G = 80). A larger ``G``, or one that is not a multiple of 4, uses scalar loads
of the same layout with identical bits.

DeepSeek V4.1 variants (``sorted_gather_qmm_a8`` keyword arguments) add Q2/Q3
codes, separate gate and up tensors, a clamped and route-weighted float
SwiGLU, and GS64 activation scales:

    correction   out = sum_g Sa[m, g] * (Sw_g * acc_g + Bw_g * Ra_g)

The MX variants (``sorted_gather_qmm_a8_mx``) read weights with one E8M0
scale per 32 codes and no bias. MXFP4 codes become twice their E2M1 value,
which is exact in INT8. MXFP8 weights go through ``mxfp8_to_int8`` first, which
rescales each 32-code group to its largest exponent:

    correction   out = sum_h Sa * 2^(Ew_h - 127) * acc_h

over the 32-code groups h, with Sa per row or per GS64 group (Sa[m, h / 2]);
MXFP4 outputs are then halved.
"""

from __future__ import annotations

import logging
import threading

import mlx.core as mx
import mlx.nn as nn

from . import m5_gather_qmm_nax as _nax
from .m5_gather_qmm import _swiglu_limit
from .mlx_lm_mtp.batch_generator import _mtp_language_model, _mtp_module
from .moe_expert_offload import OffloadSwitchGLU

logger = logging.getLogger(__name__)


_BN = 64
_WN = 2
_GROUP = 64
_BITS = (2, 3, 4)
# Stage A activation scaling: one scale per row, or one per GS64 group.
_ACT_ROW = 0
_ACT_G64 = 1
# Weight operand formats: affine GS64 codes, MXFP4, or INT8 from mxfp8_to_int8.
_FMT_AFFINE = 0
_FMT_MXFP4 = 1
_FMT_INT8 = 2
_MX_GROUP = 32
# 64 rows x 384 E8M0 bytes = 24 KiB of staged scales (K = 12288).
_STAGE_MAX_MX_GROUPS = 384
# Largest G (= K / 64) whose tile metadata is staged in threadgroup memory:
# 2 x 64 rows x 80 groups x 2 bytes = 20 KiB. Staging G = 80 (K = 5120) beat
# the scalar loads by about 5% on M5 Max.
_STAGE_MAX_GROUPS = 80

# Tile height. On M5 Max with Flash-Next routes (10K-328K rows) BM32 was the
# fastest or tied; the A16 planner's 64-128 rows lose up to 1.5x.
_BM = 32
# The DeepSeek V4.1 gate/up gather (K = 5120, about 128 rows per expert in an
# 8K-token chunk) reuses each decoded weight tile over more rows: BM64 was 24%
# faster than BM32 on M5 Max. Its down projection keeps BM32.
_BM_SPLIT_SWIGLU = 64

_lock = threading.RLock()
_kernel = None
_kernel_failed = False

_A8_HEADER_TEMPLATE = r"""
using namespace metal;
using namespace mlx::steel;

constant constexpr int kA8FragM = 16;
constant constexpr int kA8FragN = 32;
constant constexpr int kA8FragK = 16;
constant constexpr int kA8Elems = BaseNAXFrag::kElemsPerFrag;
constant constexpr int kA8Dest = 2 * kA8Elems;
constant constexpr int kA8Group = 64;
constant constexpr int kA8Steps = kA8Group / kA8FragK;

// Twice the E2M1 value of both nibbles of a byte as two INT8, low nibble in
// the low byte.
constant constexpr ushort kE2M1x2[256] = {__E2M1_LUT__};

// One single-expert BM x BN output tile of a sorted routed projection, with
// the activation operand quantized to INT8.
//
// Grid: x = column tiles * 32 lanes, y = tile index * WN, z = BM / 32.
// tid.y indexes the tile list produced by omlx_gqmm_tile_scan. The activation
// rows of a tile are sorted rows row_start .. row_start + rows - 1; the kernel
// reads the token row rmap[r] of Qa/Sa/Ra in place, so no [M, K] copy of the
// activation exists.
//
// STAGED selects how the checkpoint's [E, N, G] scale/bias are read:
//   1  the threadgroup's 64 weight rows are staged once in threadgroup memory
//      as [group][local row] (G = SG, a multiple of 4, at most 64)
//   0  scalar loads straight from the checkpoint layout
//
// BITS 4 reads Qa in the Stage A v8 slot order (nibble pairs); BITS 2 and 3
// read it in checkpoint K order, decoding each lane's 16 consecutive codes.
// ACT 1: Sa holds one scale per GS64 group ([G, n_tok], like Ra) instead of
// one per row. SPLIT 1: gate and up are separate [E, N / 2, K] tensors (w and
// wu). EPI 2: the paired epilogue clamps at `limit` and runs the SwiGLU in
// float; RW 1 scales each sorted row by its route weight rw[m].
//
// FMT 1 (MXFP4) and 2 (INT8) read E8M0 scales [E, N, K / 32] (S = uint8_t,
// staged as [group][local row]) and no bias. Step t covers codes
// 32 (t >> 1) + 8 cx + 4 (t & 1) + j in checkpoint K order, so each step pair
// is one scale group with its own accumulator.
// EPI 3: the MiniMax SwiGLU-OAI in float -- gate and up times the expert's
// gsc / usc, gate clamped above and up to +-limit, then
// gate * sigmoid(alpha * gate) * (up + beta).
template <
    typename T,
    typename S,
    int FMT,
    int BITS,
    int WM,
    int WN,
    int EPI,
    int STAGED,
    int SG,
    int ACT,
    int SPLIT,
    int RW>
METAL_FUNC void omlx_a8_gather_nax(
    const device int8_t* qa,
    const device float* sa,
    const device short* ra,
    const device uint32_t* w,
    const device S* scales_in,
    const device S* biases_in,
    const device uint32_t* wu,
    const device S* su_in,
    const device S* bu_in,
    const device float* rw,
    const device float* gsc,
    const device float* usc,
    const device uint32_t* rmap,
    const device uint32_t* tiles,
    const uint32_t tile_count,
    const int N,
    const int K,
    const int n_tok,
    const float limit,
    const float alpha,
    const float beta,
    device T* out,
    threadgroup T* meta_s,
    threadgroup T* meta_b,
    threadgroup uchar* meta_e,
    uint3 tid,
    uint simd_gid,
    uint tl) {
  const device T* scales = reinterpret_cast<const device T*>(scales_in);
  const device T* biases = reinterpret_cast<const device T*>(biases_in);
  const device T* su = reinterpret_cast<const device T*>(su_in);
  const device T* bu = reinterpret_cast<const device T*>(bu_in);
  const device uchar* escales = reinterpret_cast<const device uchar*>(scales_in);
  const device uchar* esu = reinterpret_cast<const device uchar*>(su_in);
  // MXFP4 codes are twice the E2M1 value.
  constexpr float kWScale = FMT == 1 ? 0.5f : 1.0f;
  constexpr int TM = 2;
  constexpr int BM = TM * kA8FragM * WM;
  constexpr int BN = kA8FragN * WN;
  constexpr int words = (kA8Group * BITS) / 32;
  // EPI != 0: N holds each expert's gate rows followed by its up rows. A
  // 64-row weight tile pairs 32 gate with 32 up rows (the A16 epilogue's
  // pair_row): a simdgroup's two 16-column fragments are gate and up of the
  // same 16 output columns, so every lane holds both projections of its
  // elements and the SwiGLU runs on the accumulators, writing [M, N / 2].
  constexpr bool kPair = EPI != 0;
  constexpr bool kSplit = kPair && SPLIT != 0;
  const int half_n = N / 2;
  // Weight rows per expert in each tensor.
  const int rows_e = kSplit ? half_n : N;

  const int groups = K / kA8Group;

  const int tile_idx = int(tid.y);
  if (tile_idx >= int(tile_count)) {
    return;
  }
  const uint4 tile = *((const device uint4*)tiles + tile_idx);
  const int row_start = int(tile.x);
  const int expert = int(tile.y);
  const int tile_rows = int(tile.z);

  // Threadgroup layout is (32 lanes, WN column simdgroups, BM / 32 row
  // simdgroups), so the linear simdgroup index is row * WN + column -- the
  // convention gather_seg uses (tm = sgid / kWN, tn = sgid % kWN).
  const int sg_m = int(simd_gid) / WN;
  const int sg_n = int(simd_gid) % WN;
  const int row_base = sg_m * (TM * kA8FragM);
  // Plain: the simdgroup's 32 weight/output columns. Paired: its 16 output
  // columns (the weight rows are col_base + {0, half_n}).
  const int col_base = kPair ? int(tid.x) * kA8FragN + sg_n * kA8FragM
                             : int(tid.x) * BN + sg_n * kA8FragN;
  // Weight-row distance between the two fragments.
  const int frag_n = kPair ? half_n : kA8FragM;

  const short2 coord = BaseNAXFrag::get_coord();
  const int cx = int(coord.x) >> 2;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      kA8FragM,
      kA8FragN,
      kA8FragK,
      /* transpose_left = */ false,
      /* transpose_right = */ true,
      /* relaxed_precision = */ false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = mpp::tensor_ops::matmul2d_descriptor(
      kA8FragM,
      kA8FragN,
      kA8FragK,
      false,
      true,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  mpp::tensor_ops::matmul2d<desc_set, metal::execution_simdgroup> op_set;

  auto ct_a =
      op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>,
      int32_t>();

  float Cf[TM][kA8Dest];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kA8Dest; ++e) {
      Cf[i][e] = 0.0f;
    }
  }

  const int w_row = groups * words;
  const int n_lane = col_base + int(coord.y);
  const size_t w_exp = size_t(expert) * size_t(rows_e) * size_t(w_row);
  const device uint32_t* wbase = w + w_exp + size_t(n_lane) * size_t(w_row);
  const int w_stride8 = 8 * w_row;
  // The second fragment's rows: the up tensor's same rows, or frag_n rows on.
  const device uint32_t* wbase1 = kSplit
      ? wu + w_exp + size_t(n_lane) * size_t(w_row)
      : wbase + size_t(frag_n) * size_t(w_row);

  // The checkpoint layout: one expert's scale/bias are rows_e rows of G values.
  const size_t m_exp = size_t(expert) * size_t(groups) * size_t(rows_e);
  const device T* s_exp = scales + m_exp;
  const device T* b_exp = biases + m_exp;
  const device T* su_exp = kSplit ? su + m_exp : s_exp;
  const device T* bu_exp = kSplit ? bu + m_exp : b_exp;

  const int n_run0 = col_base + int(coord.x);

  if (STAGED && FMT != 0) {
    // The tile's 64 rows of E8M0 scales, four groups per load.
    constexpr int kThreads = WM * WN * 32;
    const int q4 = (2 * groups) / 4;
    const int total = 64 * q4;
    const size_t e_exp = size_t(expert) * size_t(2 * groups) * size_t(rows_e);
    const device uint32_t* s4 =
        reinterpret_cast<const device uint32_t*>(escales + e_exp);
    const device uint32_t* su4 =
        reinterpret_cast<const device uint32_t*>((kSplit ? esu : escales) + e_exp);
    for (int e = int(tl); e < total; e += kThreads) {
      const int lr = e / q4;
      const int c = e - lr * q4;
      const bool up_row = kPair && lr >= 32;
      const int wr = kPair
          ? (up_row ? (kSplit ? 0 : half_n) + int(tid.x) * 32 + (lr - 32)
                    : int(tid.x) * 32 + lr)
          : int(tid.x) * BN + lr;
      const size_t at = size_t(wr) * size_t(q4) + size_t(c);
      const uchar4 b = as_type<uchar4>((kSplit && up_row) ? su4[at] : s4[at]);
      STEEL_PRAGMA_UNROLL
      for (int k = 0; k < 4; ++k) {
        meta_e[(c * 4 + k) * 64 + lr] = b[k];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (STAGED && FMT == 0) {
    // Stage the 64 weight rows of this tile (rows are contiguous [G] runs in
    // the checkpoint layout) as [group][local row]. Local rows: paired = 32
    // gate rows then 32 up rows; plain = 64 consecutive rows.
    constexpr int kThreads = WM * WN * 32;
    constexpr int kU = 4;
    const int q4 = groups / 4;
    const int total = 64 * q4;
    const device uint2* s4 = reinterpret_cast<const device uint2*>(s_exp);
    const device uint2* b4 = reinterpret_cast<const device uint2*>(b_exp);
    const device uint2* su4 = reinterpret_cast<const device uint2*>(su_exp);
    const device uint2* bu4 = reinterpret_cast<const device uint2*>(bu_exp);
    for (int e0 = int(tl); e0 < total; e0 += kThreads * kU) {
      uint2 vs[kU];
      uint2 vb[kU];
      STEEL_PRAGMA_UNROLL
      for (int u = 0; u < kU; ++u) {
        const int e = e0 + u * kThreads;
        if (e < total) {
          const int lr = e / q4;
          const int c = e - lr * q4;
          const bool up_row = kPair && lr >= 32;
          const int wr = kPair
              ? (up_row ? (kSplit ? 0 : half_n) + int(tid.x) * 32 + (lr - 32)
                        : int(tid.x) * 32 + lr)
              : int(tid.x) * BN + lr;
          const size_t at = size_t(wr) * size_t(q4) + size_t(c);
          vs[u] = (kSplit && up_row) ? su4[at] : s4[at];
          vb[u] = (kSplit && up_row) ? bu4[at] : b4[at];
        }
      }
      STEEL_PRAGMA_UNROLL
      for (int u = 0; u < kU; ++u) {
        const int e = e0 + u * kThreads;
        if (e < total) {
          const int lr = e / q4;
          const int c = e - lr * q4;
          const vec<T, 4> a = as_type<vec<T, 4>>(vs[u]);
          const vec<T, 4> b = as_type<vec<T, 4>>(vb[u]);
          STEEL_PRAGMA_UNROLL
          for (int k = 0; k < 4; ++k) {
            meta_s[(c * 4 + k) * 64 + lr] = a[k];
            meta_b[(c * 4 + k) * 64 + lr] = b[k];
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // A row simdgroup past the tile's rows has nothing to store; no barrier
  // follows the staging.
  if (row_base >= tile_rows) {
    return;
  }
  // Local-row base of this lane's columns in the staged tile.
  const int lr0 = (kPair ? sg_n * kA8FragM : sg_n * kA8FragN) + int(coord.x);
  const int lr_h = kPair ? 32 : kA8FragM;

  // Token row of each (row fragment, row half) this lane touches: constant
  // over the K loop, so the row map is read once per tile, not per group.
  int mt_row[TM][2];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int r = 0; r < 2; ++r) {
      const int ms = min(
          row_base + i * kA8FragM + r * 8 + int(coord.y), tile_rows - 1);
      mt_row[i][r] = int(rmap[row_start + ms]);
    }
  }

  if (FMT != 0) {
    for (int g = 0; g < groups; ++g) {
      // The lane's codes of scale groups 2g and 2g + 1: 8 codes each, one
      // word of nibbles (MXFP4) or two words of bytes (INT8).
      uint2 wv[4][2];
      STEEL_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        const device uint32_t* wr = ((q >> 1) ? wbase1 : wbase) +
            (q & 1) * w_stride8 + size_t(g) * words;
        STEEL_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
          wv[q][h] = FMT == 1
              ? uint2(wr[4 * h + cx], 0u)
              : reinterpret_cast<const device uint2*>(wr)[4 * h + cx];
        }
      }

      STEEL_PRAGMA_UNROLL
      for (int t = 0; t < kA8Steps; ++t) {
        const int h = t >> 1;
        const int o = t & 1;
        STEEL_PRAGMA_UNROLL
        for (int q = 0; q < 4; ++q) {
          const int base = (q >> 1) * kA8Elems + (q & 1) * 4;
          uint32_t word;
          if (FMT == 1) {
            const uint32_t n16 = (wv[q][h].x >> (16 * o)) & 0xffffu;
            word = uint32_t(kE2M1x2[n16 & 0xffu]) |
                (uint32_t(kE2M1x2[n16 >> 8]) << 16);
          } else {
            word = o ? wv[q][h].y : wv[q][h].x;
          }
          const char4 quad = as_type<char4>(word);
          ct_b[base + 0] = quad.x;
          ct_b[base + 1] = quad.y;
          ct_b[base + 2] = quad.z;
          ct_b[base + 3] = quad.w;
        }

        STEEL_PRAGMA_UNROLL
        for (int hf = 0; hf < 2; ++hf) {
          STEEL_PRAGMA_UNROLL
          for (int r = 0; r < 2; ++r) {
            const int mt = mt_row[hf][r];
            const char4 quad = as_type<char4>(
                *reinterpret_cast<const device uint32_t*>(
                    qa + size_t(mt) * size_t(K) + size_t(g) * kA8Group +
                    size_t(h) * 32 + size_t(cx) * 8 + size_t(o) * 4));
            ct_a[r * 4 + 0] = quad.x;
            ct_a[r * 4 + 1] = quad.y;
            ct_a[r * 4 + 2] = quad.z;
            ct_a[r * 4 + 3] = quad.w;
          }
          if (o == 0) {
            if (hf == 0) {
              op_set.run(ct_a, ct_b, acc0);
            } else {
              op_set.run(ct_a, ct_b, acc1);
            }
          } else {
            if (hf == 0) {
              op.run(ct_a, ct_b, acc0);
            } else {
              op.run(ct_a, ct_b, acc1);
            }
          }
        }

        if (o == 1) {
          const int sg32 = 2 * g + h;
          uchar4 ev[2];
          STEEL_PRAGMA_UNROLL
          for (int h2 = 0; h2 < 2; ++h2) {
            ev[h2] = *reinterpret_cast<const threadgroup uchar4*>(
                meta_e + sg32 * 64 + lr0 + h2 * lr_h);
          }
          float s_g[TM][2];
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < TM; ++i) {
            STEEL_PRAGMA_UNROLL
            for (int r = 0; r < 2; ++r) {
              s_g[i][r] = ACT
                  ? sa[size_t(g) * size_t(n_tok) + size_t(mt_row[i][r])]
                  : 1.0f;
            }
          }
          STEEL_PRAGMA_UNROLL
          for (int e = 0; e < kA8Dest; ++e) {
            const int r = ((e & 7) >> 2);
            const float swc = as_type<float>(uint(ev[e >> 3][e & 3]) << 23);
            Cf[0][e] = metal::fma(swc * s_g[0][r], float(acc0[e]), Cf[0][e]);
            Cf[1][e] = metal::fma(swc * s_g[1][r], float(acc1[e]), Cf[1][e]);
          }
        }
      }
    }
  }

  for (int g = 0; FMT == 0 && g < groups; ++g) {
    // BITS 4: two words of nibbles in v8 slot order. BITS 2 / 3: the lane's 16
    // consecutive codes k = 16 cx .. 16 cx + 15, right-aligned in 64 bits.
    uint2 wg[4];
    ulong wl[4];
    STEEL_PRAGMA_UNROLL
    for (int q = 0; q < 4; ++q) {
      const device uint32_t* wr =
          ((q >> 1) ? wbase1 : wbase) + (q & 1) * w_stride8 + size_t(g) * words;
      if (BITS == 4) {
        wg[q] = reinterpret_cast<const device uint2*>(wr)[cx];
      } else if (BITS == 2) {
        wl[q] = ulong(wr[cx]);
      } else {
        const int bit = 48 * cx;
        const ulong pair =
            ulong(wr[bit >> 5]) | (ulong(wr[(bit >> 5) + 1]) << 32);
        wl[q] = pair >> (bit & 31);
      }
    }

    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < kA8Steps; ++t) {
      STEEL_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        const int base = (q >> 1) * kA8Elems + (q & 1) * 4;
        if (BITS == 4) {
          const uint32_t word = wg[q][t >> 1];
          const char4 quad = as_type<char4>(
              ((t & 1) ? (word >> 4) : word) & 0x0f0f0f0fu);
          ct_b[base + 0] = quad.x;
          ct_b[base + 1] = quad.y;
          ct_b[base + 2] = quad.z;
          ct_b[base + 3] = quad.w;
        } else {
          constexpr ulong mask = (1ul << BITS) - 1ul;
          STEEL_PRAGMA_UNROLL
          for (int j = 0; j < 4; ++j) {
            ct_b[base + j] =
                int8_t((wl[q] >> (BITS * (4 * t + j))) & mask);
          }
        }
      }

      STEEL_PRAGMA_UNROLL
      for (int hf = 0; hf < 2; ++hf) {
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          const int mt = mt_row[hf][r];
          const char4 quad = as_type<char4>(
              *reinterpret_cast<const device uint32_t*>(
                  qa + size_t(mt) * size_t(K) + size_t(g) * kA8Group +
                  size_t(cx) * 16 + size_t(t) * 4));
          ct_a[r * 4 + 0] = quad.x;
          ct_a[r * 4 + 1] = quad.y;
          ct_a[r * 4 + 2] = quad.z;
          ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) {
          if (hf == 0) {
            op_set.run(ct_a, ct_b, acc0);
          } else {
            op_set.run(ct_a, ct_b, acc1);
          }
        } else {
          if (hf == 0) {
            op.run(ct_a, ct_b, acc0);
          } else {
            op.run(ct_a, ct_b, acc1);
          }
        }
      }
    }

    // GS64 scale / bias of this group for the lane's four columns, per
    // fragment half.
    vec<T, 4> sv[2];
    vec<T, 4> bv[2];
    if (STAGED) {
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        sv[h] = *reinterpret_cast<const threadgroup vec<T, 4>*>(
            meta_s + g * 64 + lr0 + h * lr_h);
        bv[h] = *reinterpret_cast<const threadgroup vec<T, 4>*>(
            meta_b + g * 64 + lr0 + h * lr_h);
      }
    } else {
      STEEL_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        const bool up_t = kSplit && h != 0;
        const size_t n0 = size_t(n_run0 + (up_t ? 0 : h * frag_n));
        const device T* sp = up_t ? su_exp : s_exp;
        const device T* bp = up_t ? bu_exp : b_exp;
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          sv[h][j] = sp[(n0 + j) * size_t(groups) + size_t(g)];
          bv[h][j] = bp[(n0 + j) * size_t(groups) + size_t(g)];
        }
      }
    }

    // Per (row fragment, row half): the activation group scale (1 for row
    // scaling, applied in the epilogue instead) and its product with Ra.
    float s_g[TM][2];
    float r_g[TM][2];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        const size_t at = size_t(g) * size_t(n_tok) + size_t(mt_row[i][r]);
        s_g[i][r] = ACT ? sa[at] : 1.0f;
        r_g[i][r] = float(ra[at]) * s_g[i][r];
      }
    }

    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kA8Dest; ++e) {
      const int r = ((e & 7) >> 2);
      const float swc = float(sv[e >> 3][e & 3]);
      const float bwc = float(bv[e >> 3][e & 3]);
      Cf[0][e] = metal::fma(
          swc * s_g[0][r], float(acc0[e]), metal::fma(bwc, r_g[0][r], Cf[0][e]));
      Cf[1][e] = metal::fma(
          swc * s_g[1][r], float(acc1[e]), metal::fma(bwc, r_g[1][r], Cf[1][e]));
    }
  }

  if (kPair) {
    // Gate is element e of fragment 0 and up element e of fragment 1 (e + 8).
    // EPI 1: each is rounded to T exactly as the plain store rounds it, then
    // the unfused path's activation runs op for op in T (MLX's Sigmoid and
    // Multiply, as in the A16 epilogue): silu(gate) * up.
    // EPI 2: the DeepSeek V4.1 activation in float -- gate clamped above and
    // up clamped to +-limit (limit 0: none), silu(gate) * up, times rw[m].
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kA8Elems; ++e) {
        const int r = e >> 2;
        const int ms = row_base + i * kA8FragM + int(coord.y) + r * 8;
        if (ms < tile_rows) {
          const int n = col_base + int(coord.x) + (e & 3);
          const int m = row_start + ms;
          const float sc = (ACT ? 1.0f : sa[mt_row[i][r]]) * kWScale;
          if (EPI == 3) {
            float g = sc * Cf[i][e] * gsc[expert];
            float u = sc * Cf[i][e + kA8Elems] * usc[expert];
            if (limit != 0.0f) {
              g = metal::min(g, limit);
              u = metal::clamp(u, -limit, limit);
            }
            const float z = alpha * g;
            const float neg = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
            float v = g * (z < 0.0f ? neg : 1.0f - neg) * (u + beta);
            if (RW) {
              v *= rw[m];
            }
            out[size_t(m) * size_t(half_n) + size_t(n)] = static_cast<T>(v);
          } else if (EPI == 2) {
            float g = sc * Cf[i][e];
            float u = sc * Cf[i][e + kA8Elems];
            if (limit != 0.0f) {
              g = metal::min(g, limit);
              u = metal::clamp(u, -limit, limit);
            }
            const float neg = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
            float v = (g * (g < 0.0f ? neg : 1.0f - neg)) * u;
            if (RW) {
              v *= rw[m];
            }
            out[size_t(m) * size_t(half_n) + size_t(n)] = static_cast<T>(v);
          } else {
            const T g = static_cast<T>(sc * Cf[i][e]);
            const T u = static_cast<T>(sc * Cf[i][e + kA8Elems]);
            out[size_t(m) * size_t(half_n) + size_t(n)] =
                Multiply()(Multiply()(g, Sigmoid()(g)), u);
          }
        }
      }
    }
  } else {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kA8Dest; ++e) {
        const int ee = e & 7;
        const int r = ee >> 2;
        const int ms = row_base + i * kA8FragM + int(coord.y) + r * 8;
        if (ms < tile_rows) {
          const int n =
              col_base + (e >> 3) * kA8FragM + int(coord.x) + (ee & 3);
          const int m = row_start + ms;
          const float sc = (ACT ? 1.0f : sa[mt_row[i][r]]) * kWScale;
          out[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(sc * Cf[i][e]);
        }
      }
    }
  }
}
"""


def _e2m1_pair_lut() -> str:
    """Twice the E2M1 value of each nibble of a byte, as two INT8 bytes."""
    twice = [0, 1, 2, 3, 4, 6, 8, 12]

    def code(n):
        v = twice[n & 7]
        return (-v if n & 8 else v) & 0xFF

    return ", ".join(str(code(b & 15) | (code(b >> 4) << 8)) for b in range(256))


_A8_HEADER = _A8_HEADER_TEMPLATE.replace("__E2M1_LUT__", _e2m1_pair_lut())

_A8_SOURCE = """
    threadgroup T meta_s[(STAGED && FMT == 0) ? SG * 64 : 1];
    threadgroup T meta_b[(STAGED && FMT == 0) ? SG * 64 : 1];
    // uint storage keeps the uchar4 scale reads aligned.
    threadgroup uint meta_e[(STAGED && FMT != 0) ? SG * 16 : 1];
    omlx_a8_gather_nax<T, S, FMT, BITS, WM, WN, EPI, STAGED, SG, ACT, SPLIT, RW>(
        qa, sa, ra, w, scales, biases, wu, su, bu, rw, gsc, usc, rmap, tiles,
        tile_count[0], params[0], params[1], params[2], limit[0], limit[1],
        limit[2], out, meta_s, meta_b, reinterpret_cast<threadgroup uchar*>(meta_e),
        threadgroup_position_in_grid,
        simdgroup_index_in_threadgroup, thread_index_in_threadgroup);
"""


def _get_kernel():
    """Build (once) the A8 gather kernel object, or None."""
    global _kernel, _kernel_failed
    if _kernel is not None or _kernel_failed:
        return _kernel
    with _lock:
        if _kernel is not None or _kernel_failed:
            return _kernel
        mlx_src = _nax._read_mlx_headers(
            _nax._MLX_MM_HEADERS + _nax._MLX_OPS_HEADERS
        )
        if mlx_src is None:
            _kernel_failed = True
            logger.warning("mlx kernel headers not found; A8 gather disabled")
            return None
        try:
            _kernel = mx.fast.metal_kernel(
                name="omlx_a8_gather",
                input_names=[
                    "qa",
                    "sa",
                    "ra",
                    "w",
                    "scales",
                    "biases",
                    "wu",
                    "su",
                    "bu",
                    "rw",
                    "gsc",
                    "usc",
                    "rmap",
                    "tiles",
                    "tile_count",
                    "params",
                    "limit",
                ],
                output_names=["out"],
                header=mlx_src + _A8_HEADER,
                source=_A8_SOURCE,
            )
        except Exception:  # noqa: BLE001
            _kernel_failed = True
            logger.warning("A8 gather kernel failed to build", exc_info=True)
            return None
        return _kernel


def supports(
    x,
    w,
    scales,
    biases,
    indices,
    group_size,
    bits,
    mode,
    row_map=None,
    tokens=None,
) -> bool:
    """Layout gate for the A8 gather.

    Accepts affine Q2/Q3/Q4 with GS64 and metadata of the activation dtype, a
    fused or plain expert weight ``[E, N, K * bits / 32]`` with ``N % 64 == 0``
    and ``K % 64 == 0``, scale/bias in the checkpoint layout ``[E, N, K / 64]``,
    and a uint32 row map of the sorted rows onto the token rows.
    """
    if mode != "affine" or bits not in _BITS or group_size != _GROUP:
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 3 or x.shape[1] != 1 or indices.ndim != 1:
        return False
    M, K = int(indices.shape[0]), int(x.shape[2])
    if M < 8 or indices.dtype != mx.uint32 or K % _GROUP:
        return False
    if row_map is None or row_map.ndim != 1 or int(row_map.shape[0]) != M:
        return False
    if row_map.dtype != mx.uint32:
        return False
    if tokens is not None and int(x.shape[0]) != int(tokens):
        return False
    if w.ndim != 3 or w.dtype != mx.uint32:
        return False
    E, N = int(w.shape[0]), int(w.shape[1])
    if E == 0 or N == 0 or N % _BN:
        return False
    if w.shape[2] * 32 != K * bits:
        return False
    if biases is None or scales.dtype != x.dtype or biases.dtype != x.dtype:
        return False
    return scales.shape == biases.shape and scales.shape == (E, N, K // _GROUP)


def _stage_a(x, bits, act):
    """Q4 reads Qa in the v8 slot order; Q2/Q3 read it in K order. Ra (and
    group-wise Sa) are group-major [G, T] either way."""
    from omlx.custom_kernels.quant_matmul import fast

    stage = fast.oq_a8_stage_a_v8 if bits == 4 else fast.oq_a8_stage_a_natural
    return stage(x, act)


def _staged(groups: int) -> bool:
    """Whether the tile metadata fits the staging buffers and the staging
    loads (four groups at a time) apply."""
    return groups % 4 == 0 and groups <= _STAGE_MAX_GROUPS


def _launch(
    x,
    w,
    scales,
    biases,
    indices,
    row_map,
    *,
    bm,
    swiglu,
    init_value=None,
    bits=4,
    up=None,
    limit=None,
    route_weights=None,
    act=_ACT_ROW,
    fmt=_FMT_AFFINE,
    oai=None,
    expert_scales=None,
):
    """Stage A, the tile scan and the gather kernel; None when unavailable.

    MX formats (``fmt`` != 0) pass ``biases=None`` and E8M0 ``scales``.
    ``oai`` = ``(alpha, beta)`` selects the SwiGLU-OAI epilogue, with
    ``expert_scales`` = ``(gate, up)`` per-expert float multipliers.
    """
    from omlx.custom_kernels.quant_matmul import fast

    if not fast.oq_a8_available():
        return None
    kernel = _get_kernel()
    scan = _nax._get_kernel("scan")
    if kernel is None or scan is None:
        return None

    T, K = int(x.shape[0]), int(x.shape[2])
    M = int(indices.shape[0])
    # With a separate up tensor N counts both halves, as in the fused layout.
    E, N = int(w.shape[0]), int(w.shape[1]) * (1 if up is None else 2)
    wm, wn = bm // 32, _WN
    # The DeepSeek V4.1 epilogue for split, clamped or weighted SwiGLU.
    epi = 0
    if swiglu and oai is not None:
        epi = 3
    elif swiglu:
        plain = up is None and limit is None and route_weights is None
        epi = 1 if plain else 2
    wu, su, bu = up if up is not None else (w, scales, biases)
    if fmt != _FMT_AFFINE:
        # Unread; the kernel signature still binds a bias of the scale type.
        biases, bu = scales, su

    # Stage A is left lazy (no mx.eval): the kernel depends on it through the
    # graph, like the dense oQ A8 path. MX formats read Qa in K order.
    qa, sa, ra = _stage_a(x, 8 if fmt != _FMT_AFFINE else bits, act)
    # Without route weights rw is unread; Sa stands in because a one-element
    # array would be bound in the constant address space.
    rw = route_weights.astype(mx.float32) if route_weights is not None else sa
    if epi != 3:
        gsc = usc = sa
    else:
        if expert_scales is None:
            expert_scales = (mx.ones((E,), dtype=mx.float32),) * 2
        gsc, usc = (s.astype(mx.float32) for s in expert_scales)
        if E < 8:
            # Arrays of fewer than 8 elements bind in the constant address space.
            pad = mx.zeros((8 - E,), dtype=mx.float32)
            gsc, usc = mx.concatenate([gsc, pad]), mx.concatenate([usc, pad])
    alpha, beta = oai if oai is not None else (0.0, 0.0)

    # The grid is sized by the *upper bound* on tiles and the kernel returns
    # early for tile indices past tile_count, so nothing is read back to the
    # host and the launch never synchronises.
    max_tiles = (M + bm - 1) // bm + min(E, M)
    tiles, tile_count = scan(
        inputs=[indices, mx.array([M, E, max_tiles], dtype=mx.int32)],
        template=[("BM", bm), ("MAXE", _nax._MAX_EXPERTS)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(max_tiles * 4,), (1,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )

    groups = K // _GROUP
    staged = _staged(groups)
    stage_groups = groups if staged else 0
    if fmt != _FMT_AFFINE:
        staged, stage_groups = True, K // _MX_GROUP
        bits = 4 if fmt == _FMT_MXFP4 else 8
    kw = {} if init_value is None else {"init_value": init_value}
    # A 64-row weight tile is 64 output columns, or 32 once gate and up are
    # paired.
    return kernel(
        inputs=[
            qa,
            sa,
            ra,
            w,
            scales,
            biases,
            wu,
            su,
            bu,
            rw,
            gsc,
            usc,
            row_map,
            tiles,
            tile_count,
            mx.array([N, K, T], dtype=mx.int32),
            mx.array([float(limit or 0.0), alpha, beta], dtype=mx.float32),
        ],
        template=[
            ("T", x.dtype),
            ("S", scales.dtype if fmt != _FMT_AFFINE else x.dtype),
            ("FMT", fmt),
            ("BITS", bits),
            ("WM", wm),
            ("WN", wn),
            ("EPI", epi),
            ("STAGED", 1 if staged else 0),
            ("SG", stage_groups),
            ("ACT", act),
            ("SPLIT", 1 if up is not None else 0),
            ("RW", 1 if route_weights is not None else 0),
        ],
        grid=((N // _BN) * 32, max_tiles * wn, bm // 32),
        threadgroup=(32, wn, bm // 32),
        output_shapes=[(M, 1, N // 2 if swiglu else N)],
        output_dtypes=[x.dtype],
        **kw,
    )[0]


def sorted_gather_qmm_a8(
    x,
    w,
    scales,
    biases,
    indices,
    row_map,
    *,
    group_size,
    bits,
    mode="affine",
    swiglu=False,
    init_value=None,
    up=None,
    swiglu_limit=None,
    route_weights=None,
    group_scales=False,
    swiglu_alpha=None,
    swiglu_beta=0.0,
    expert_scales=None,
):
    """``activation(x[row_map]) @ w[indices].T`` with an INT8 activation.

    ``x`` holds the *token* rows ``[T, 1, K]``; Stage A runs on them once and
    the kernel reads ``Qa``/``Sa``/``Ra`` through ``row_map``, so the top-k
    replication never re-quantizes an activation. ``scales`` / ``biases`` are
    the checkpoint tensors ``[E, N, K / 64]``, read in place. Returns
    ``[M, 1, N]``, or None when unsupported (the caller keeps the A16 path).

    ``swiglu=True``: ``w`` holds each expert's gate rows followed by its up
    rows (``[E, N, K]``, N = 2 * n) and the kernel returns the SwiGLU
    ``silu(gate) * up`` as ``[M, 1, n]`` (see the kernel header for the exact
    rounding contract). ``up`` instead passes the up projection as its own
    ``(weight, scales, biases)`` with ``w`` holding the gate rows only.
    ``swiglu_limit`` clamps gate above and up to +-limit, and ``route_weights``
    (``[M]``, sorted rows) scales each output row; both, and ``up``, use the
    float DeepSeek V4.1 epilogue.

    ``group_scales=True`` scales activations per GS64 group instead of per
    row, so an outlier sets the scale of its own group only.

    ``swiglu_alpha`` selects the MiniMax SwiGLU-OAI instead:
    ``g * sigmoid(alpha * g) * (u + swiglu_beta)`` in float, where gate and up
    are first multiplied by ``expert_scales`` (``(gate, up)``, ``[E]`` float)
    and clamped by ``swiglu_limit``.

    ``init_value`` pre-fills the output, so tests can detect unwritten
    elements.
    """
    if not supports(x, w, scales, biases, indices, group_size, bits, mode, row_map):
        return None
    if up is not None and not _supports_up(w, scales, biases, up, swiglu):
        return None
    if not _supports_epilogue(
        w, indices, swiglu, route_weights, swiglu_alpha, expert_scales
    ):
        return None
    act = _ACT_G64 if group_scales else _ACT_ROW
    groups = int(x.shape[2]) // _GROUP
    oai = swiglu_alpha is not None
    extended = (
        bits != 4
        or act != _ACT_ROW
        or up is not None
        or swiglu_limit is not None
        or route_weights is not None
        or oai
    )
    # One canary verdict per kernel instantiation.
    if extended:
        key = (
            "a8x",
            x.dtype,
            groups,
            bits,
            bool(swiglu),
            up is not None,
            swiglu_limit is not None,
            route_weights is not None,
            act,
            oai,
        )
        if not _nax._checked(key, _self_test_extended):
            return None
    elif not _nax._checked(("a8", x.dtype, groups), _self_test):
        return None
    return _launch(
        x,
        w,
        scales,
        biases,
        indices,
        row_map,
        bm=_BM_SPLIT_SWIGLU if up is not None else _BM,
        swiglu=swiglu,
        init_value=init_value,
        bits=bits,
        up=up,
        limit=swiglu_limit,
        route_weights=route_weights,
        act=act,
        oai=(float(swiglu_alpha), float(swiglu_beta)) if oai else None,
        expert_scales=expert_scales,
    )


def _supports_epilogue(w, indices, swiglu, route_weights, alpha, expert_scales):
    """Route weights per sorted row, and SwiGLU-OAI scales per expert."""
    if route_weights is not None and route_weights.shape != indices.shape:
        return False
    if alpha is None:
        return expert_scales is None
    if not swiglu:
        return False
    experts = (int(w.shape[0]),)
    return expert_scales is None or (
        len(expert_scales) == 2 and all(s.shape == experts for s in expert_scales)
    )


def _mx_k_supported(k: int) -> bool:
    """Whole GS64 groups, and E8M0 scales that stage four groups per load."""
    groups = k // _MX_GROUP
    return k % _GROUP == 0 and groups % 4 == 0 and groups <= _STAGE_MAX_MX_GROUPS


def mx_shape_supported(n: int, k: int) -> bool:
    """Whether an MX projection with ``n`` output rows and ``k`` inputs fits
    the gather in whole 64-row tiles."""
    return n % _BN == 0 and _mx_k_supported(k)


def _supports_mx(x, w, scales, indices, row_map, fmt, up, swiglu) -> bool:
    """Layout gate for the MX formats: ``w`` ``[E, N, K * bits / 32]`` uint32
    (4-bit MXFP4 or 8-bit INT8 codes), E8M0 ``scales`` ``[E, N, K / 32]`` and
    K a multiple of 64 whose scale tile fits the staging buffer."""
    if fmt not in (_FMT_MXFP4, _FMT_INT8) or x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 3 or x.shape[1] != 1 or indices.ndim != 1:
        return False
    rows, k = int(indices.shape[0]), int(x.shape[2])
    if rows < 8 or indices.dtype != mx.uint32 or not _mx_k_supported(k):
        return False
    if row_map is None or row_map.shape != (rows,) or row_map.dtype != mx.uint32:
        return False
    if w.ndim != 3 or w.dtype != mx.uint32 or scales.dtype != mx.uint8:
        return False
    experts, n = int(w.shape[0]), int(w.shape[1])
    bits = 4 if fmt == _FMT_MXFP4 else 8
    if experts == 0 or n % (_BN // 2 if up is not None else _BN):
        return False
    if w.shape[2] * 32 != k * bits or scales.shape != (experts, n, k // _MX_GROUP):
        return False
    if up is None:
        return True
    return (
        swiglu
        and len(up) == 2
        and up[0].dtype == mx.uint32
        and up[1].dtype == mx.uint8
        and up[0].shape == w.shape
        and up[1].shape == scales.shape
    )


def sorted_gather_qmm_a8_mx(
    x,
    w,
    scales,
    indices,
    row_map,
    *,
    fmt,
    swiglu=False,
    up=None,
    swiglu_limit=None,
    route_weights=None,
    group_scales=False,
    init_value=None,
    bm=None,
    swiglu_alpha=None,
    swiglu_beta=0.0,
    expert_scales=None,
):
    """:func:`sorted_gather_qmm_a8` for MXFP4 codes (``fmt=_FMT_MXFP4``) or
    the INT8 codes of :func:`mxfp8_to_int8` (``fmt=_FMT_INT8``), with E8M0
    ``scales`` per 32 codes. ``up`` is ``(weight, scales)``. Returns None when
    unsupported."""
    if not _supports_mx(x, w, scales, indices, row_map, fmt, up, swiglu):
        return None
    if not swiglu and (route_weights is not None or swiglu_limit is not None):
        return None
    if not _supports_epilogue(
        w, indices, swiglu, route_weights, swiglu_alpha, expert_scales
    ):
        return None
    act = _ACT_G64 if group_scales else _ACT_ROW
    bm = bm or (_BM_SPLIT_SWIGLU if up is not None else _BM)
    oai = swiglu_alpha is not None
    key = (
        "a8mx",
        x.dtype,
        int(x.shape[2]) // _GROUP,
        fmt,
        bool(swiglu),
        up is not None,
        swiglu_limit is not None,
        route_weights is not None,
        act,
        bm,
        oai,
    )
    if not _nax._checked(key, _self_test_extended):
        return None
    return _launch(
        x,
        w,
        scales,
        None,
        indices,
        row_map,
        bm=bm,
        swiglu=swiglu,
        init_value=init_value,
        up=None if up is None else (up[0], up[1], None),
        limit=swiglu_limit,
        route_weights=route_weights,
        act=act,
        fmt=fmt,
        oai=(float(swiglu_alpha), float(swiglu_beta)) if oai else None,
        expert_scales=expert_scales,
    )


# One thread per 32-code group: E4M3 codes rescaled to the group's largest
# exponent Emax. A code M * 2^(e - 10) (M = 8 + mantissa, or the mantissa when
# e = 0) becomes M * 2^(max(e, 1) - Emax + 3), at most 120, rounded to nearest
# even below Emax - 3. The new E8M0 scale is the old one times 2^(Emax - 13).
_MXFP8_TO_INT8_SOURCE = """
    const uint gid = thread_position_in_grid.x;
    if (gid >= uint(params[0])) {
      return;
    }
    const device uint32_t* src = w + size_t(gid) * 8;
    uint32_t v[8];
    int emax = 1;
    for (int i = 0; i < 8; ++i) {
      v[i] = src[i];
      for (int j = 0; j < 4; ++j) {
        emax = max(emax, int((v[i] >> (8 * j + 3)) & 15u));
      }
    }
    device uint32_t* dst = w8 + size_t(gid) * 8;
    for (int i = 0; i < 8; ++i) {
      uint32_t packed = 0u;
      for (int j = 0; j < 4; ++j) {
        const uint32_t b = (v[i] >> (8 * j)) & 0xffu;
        const int e = int((b >> 3) & 15u);
        const int m = int(b & 7u) | (e ? 8 : 0);
        const int s = max(e, 1) - emax + 3;
        int q;
        if (s >= 0) {
          q = m << s;
        } else {
          const int n = -s;
          const int rem = m & ((1 << n) - 1);
          const int mid = 1 << (n - 1);
          q = m >> n;
          q += (rem > mid || (rem == mid && (q & 1))) ? 1 : 0;
        }
        q = (b & 0x80u) ? -q : q;
        packed |= (uint32_t(q) & 0xffu) << (8 * j);
      }
      dst[i] = packed;
    }
    s8[gid] = uchar(clamp(int(scales[gid]) + emax - 13, 0, 254));
"""

_mx8_kernel = None


def mxfp8_to_int8(w, scales):
    """INT8 codes and E8M0 scales (``w8``, ``s8``) of an MXFP8 weight, same
    shapes as ``w`` / ``scales``. The codes are exact down to three binades
    below each group's largest exponent."""
    global _mx8_kernel
    if _mx8_kernel is None:
        with _lock:
            if _mx8_kernel is None:
                _mx8_kernel = mx.fast.metal_kernel(
                    name="omlx_mxfp8_to_int8",
                    input_names=["w", "scales", "params"],
                    output_names=["w8", "s8"],
                    source=_MXFP8_TO_INT8_SOURCE,
                )
    total = int(scales.size)
    return _mx8_kernel(
        inputs=[w, scales, mx.array([total], dtype=mx.int32)],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[w.shape, scales.shape],
        output_dtypes=[mx.uint32, mx.uint8],
    )


def _supports_up(w, scales, biases, up, swiglu) -> bool:
    """A separate up projection matching the gate tensor, for a paired SwiGLU
    whose 64-row weight tiles take 32 rows from each."""
    if not swiglu or len(up) != 3:
        return False
    wu, su, bu = up
    if bu is None or wu.dtype != w.dtype or su.dtype != scales.dtype:
        return False
    if bu.dtype != biases.dtype or int(w.shape[1]) % (_BN // 2):
        return False
    return wu.shape == w.shape and su.shape == scales.shape and bu.shape == biases.shape


# Canary: 2 experts, N = 128 (64 gate + 64 up rows), 64 sorted rows over 16
# token rows. It runs the *production* instantiation (dtype, staging, K), so a
# kernel that cannot compile or runs wrongly on this toolchain declines to the
# A16 path instead of failing a request.
_CANARY_COUNTS = (40, 24)
_CANARY_TOKENS = 16
_CANARY_TOLERANCE = 0.1  # of the reference's max, well above the A8 error (~2%)


def _self_test(key: tuple):
    """True when the fused kernel is within the A8 error of the A16 reference
    on a canary; False when it raises or misbehaves; None when it cannot be
    evaluated here (a function transformation is being traced)."""
    _, dtype, groups = key
    K = groups * _GROUP
    try:
        N = 2 * _BN
        k_w, k_x = mx.random.split(mx.random.key(0xA8), 2)
        wf = (mx.random.normal((len(_CANARY_COUNTS), N, K), key=k_w) * 0.05).astype(
            dtype
        )
        wq, scales, biases = mx.quantize(wf, group_size=_GROUP, bits=4)
        idx = mx.array(
            [e for e, n in enumerate(_CANARY_COUNTS) for _ in range(n)],
            dtype=mx.uint32,
        )
        rows = int(idx.shape[0])
        x = (mx.random.normal((_CANARY_TOKENS, 1, K), key=k_x) * 0.6).astype(dtype)
        row_map = ((mx.arange(rows, dtype=mx.uint32) * 7 + 3) % _CANARY_TOKENS).astype(
            mx.uint32
        )
        out = _launch(x, wq, scales, biases, idx, row_map, bm=_BM, swiglu=True)
        if out is None:
            return False
        gate_up = mx.gather_qmm(
            x[row_map],
            wq,
            scales,
            biases,
            rhs_indices=idx,
            transpose=True,
            group_size=_GROUP,
            bits=4,
            sorted_indices=True,
        )
        gate, up = mx.split(gate_up, 2, axis=-1)
        ref = _nax.reference_activation(up, gate).astype(mx.float32)
        err = mx.abs(out.astype(mx.float32) - ref).max().item()
        scale = mx.abs(ref).max().item()
        ok = err <= _CANARY_TOLERANCE * scale
    except Exception as e:  # noqa: BLE001
        if "transformation" in str(e):
            return None
        logger.warning("routed A8 self-test raised for K=%d: %s", K, e)
        return False
    if not ok:
        logger.warning(
            "routed A8 disabled for K=%d: canary max error %.3g (reference max %.3g)",
            K,
            err,
            scale,
        )
    return ok


_CANARY_LIMIT = 1.0  # small enough that the canary's gate and up hit the clamp
# SwiGLU-OAI constants of MiniMax M3; the kernel takes them at run time.
_CANARY_OAI = (1.702, 1.0)


def _self_test_extended(key: tuple):
    """The canary of :func:`_self_test` for the Q2/Q3, group-scaled, split,
    clamped, weighted and SwiGLU-OAI instantiations, and for the MX formats
    (``"a8mx"`` keys), against a float reference on the dequantized weights."""
    if key[0] == "a8mx":
        _, dtype, groups, fmt, swiglu, split, clamp, weighted, act, bm, oai = key
        bits = 4 if fmt == _FMT_MXFP4 else 8
    else:
        _, dtype, groups, bits, swiglu, split, clamp, weighted, act, oai = key
        fmt, bm = _FMT_AFFINE, _BM_SPLIT_SWIGLU if split else _BM
    k_dim = groups * _GROUP
    try:
        n = _BN
        experts = len(_CANARY_COUNTS)
        keys = mx.random.split(mx.random.key(0xA8 + bits + 16 * fmt), 4)

        def quant(shape, k):
            """(weight, scales, biases) as the kernel reads them, and the
            float weights they encode."""
            wf = (mx.random.normal(shape, key=k) * 0.05).astype(dtype)
            if fmt == _FMT_AFFINE:
                packed = mx.quantize(wf, group_size=_GROUP, bits=bits)
                ref = mx.dequantize(*packed, group_size=_GROUP, bits=bits)
                return packed, ref.astype(mx.float32)
            mode = "mxfp4" if fmt == _FMT_MXFP4 else "mxfp8"
            wq, sc = mx.quantize(wf, group_size=_MX_GROUP, bits=bits, mode=mode)
            ref = mx.dequantize(wq, sc, group_size=_MX_GROUP, bits=bits, mode=mode)
            if fmt == _FMT_INT8:
                wq, sc = mxfp8_to_int8(wq, sc)
            return (wq, sc, None), ref.astype(mx.float32)

        idx = mx.array(
            [e for e, c in enumerate(_CANARY_COUNTS) for _ in range(c)],
            dtype=mx.uint32,
        )
        rows = int(idx.shape[0])
        x = (mx.random.normal((_CANARY_TOKENS, 1, k_dim), key=keys[0]) * 0.6).astype(
            dtype
        )
        row_map = ((mx.arange(rows, dtype=mx.uint32) * 7 + 3) % _CANARY_TOKENS).astype(
            mx.uint32
        )
        weights = (
            mx.random.uniform(0.2, 1.5, (rows,), key=keys[3]) if weighted else None
        )
        limit = _CANARY_LIMIT if clamp else None
        expert_scales = (
            tuple(mx.random.uniform(0.5, 2.0, (2, experts), key=mx.random.key(0xA81)))
            if oai
            else None
        )
        if swiglu and split:
            (w, w_ref), (up, up_ref) = quant((experts, n, k_dim), keys[1]), quant(
                (experts, n, k_dim), keys[2]
            )
        else:
            (w, w_ref), up = quant((experts, 2 * n, k_dim), keys[1]), None
        out = _launch(
            x,
            *w,
            idx,
            row_map,
            bm=bm,
            swiglu=swiglu,
            bits=bits,
            up=up,
            limit=limit,
            route_weights=weights,
            act=act,
            fmt=fmt,
            oai=_CANARY_OAI if oai else None,
            expert_scales=expert_scales,
        )
        if out is None:
            return False

        def gather(ref):
            xs = x[row_map].astype(mx.float32)
            return mx.gather_mm(xs, ref.swapaxes(-1, -2), rhs_indices=idx)

        if not swiglu:
            ref = gather(w_ref)
        else:
            if split:
                gate, upv = gather(w_ref), gather(up_ref)
            else:
                gate, upv = mx.split(gather(w_ref), 2, axis=-1)
            if oai:
                gate = gate * expert_scales[0][idx][:, None, None]
                upv = upv * expert_scales[1][idx][:, None, None]
            if limit is not None:
                gate = mx.minimum(gate, limit)
                upv = mx.clip(upv, -limit, limit)
            if oai:
                alpha, beta = _CANARY_OAI
                ref = gate * mx.sigmoid(alpha * gate) * (upv + beta)
            else:
                ref = nn.silu(gate) * upv
            if weights is not None:
                ref = ref * weights[:, None, None]
        err = mx.abs(out.astype(mx.float32) - ref).max().item()
        scale = mx.abs(ref).max().item()
        ok = err <= _CANARY_TOLERANCE * scale
    except Exception as e:  # noqa: BLE001
        if "transformation" in str(e):
            return None
        logger.warning("routed A8 self-test raised for %s: %s", key[1:], e)
        return False
    if not ok:
        logger.warning(
            "routed A8 disabled for %s: canary max error %.3g (reference max %.3g)",
            key[1:],
            err,
            scale,
        )
    return ok


# -- routed MoE integration --------------------------------------------------
#
# ``apply_qwen35_oq_a8_patch`` (the ``a8_prefill_enabled`` setting) tags the
# backbone SwitchGLU modules of one loaded model. Untagged modules, including
# the MTP draft layer, keep the A16 path.

_TAG = "_omlx_routed_a8_min_tokens"

# Model families measured for speed and NLL on this path (matched on the
# model class module path, as ``qwen35_moe_gate_up`` does).
_MEASURED_FAMILIES = ("qwen4_exp",)

# Decode and verify windows stay A16. ``a8_prefill_min_tokens`` can raise
# this floor but not lower it.
_MIN_TOKENS_FLOOR = 128


def _mtp_module_ids(model) -> set[int]:
    """Ids of every module inside the model's MTP draft head."""
    mtp = _mtp_module(_mtp_language_model(model))
    if mtp is None or not hasattr(mtp, "modules"):
        return set()
    return {id(m) for m in mtp.modules()}


def _is_measured_family(model) -> bool:
    module = type(model).__module__ or ""
    return any(token in module for token in _MEASURED_FAMILIES)


def _is_fused_routed_glu(module) -> bool:
    gate_up = getattr(module, "get", None) and module.get("gate_up_proj")
    return gate_up is not None and module.get("down_proj") is not None


def tag_routed_a8_modules(model, min_tokens: int) -> int:
    """Opt the backbone routed-expert modules of ``model`` in. Returns how many.

    The MTP draft head is excluded by identity, so the exclusion does not
    depend on how its modules are named. Never raises: on an error nothing
    stays tagged and the model keeps the A16 path.
    """
    if not _is_measured_family(model):
        return 0
    tagged = []
    try:
        skip = _mtp_module_ids(model)
        floor = max(int(min_tokens), _MIN_TOKENS_FLOOR)
        for _, module in model.named_modules():
            if id(module) in skip or not _is_fused_routed_glu(module):
                continue
            setattr(module, _TAG, floor)
            tagged.append(module)
    except Exception:  # noqa: BLE001
        logger.warning("routed A8 modules not tagged", exc_info=True)
        for module in tagged:
            setattr(module, _TAG, None)
        return 0
    if tagged:
        logger.info(
            "routed A8 gate/up enabled on %d MoE layers (min_tokens=%d)",
            len(tagged),
            floor,
        )
    return len(tagged)


def _eligible(switch_mlp, activation) -> bool:
    """Plain affine Q4 / GS64 fused Gate+Up with an unclamped SwiGLU, on a
    module that is not an expert-offload wrapper."""
    if not _is_fused_routed_glu(switch_mlp) or isinstance(switch_mlp, OffloadSwitchGLU):
        return False
    gate_up, down = switch_mlp.get("gate_up_proj"), switch_mlp.get("down_proj")
    if "bias" in gate_up or "bias" in down:
        return False
    # ``None in (...)`` would call ``mx.array.__eq__(None)`` and raise.
    if any(gate_up.get(k) is None for k in ("weight", "scales", "biases")):
        return False
    if (gate_up.bits, gate_up.group_size, gate_up.mode) != (4, _GROUP, "affine"):
        return False
    # Unknown or clamped activations are not fused; only silu(gate) * up is.
    return _swiglu_limit(activation) is None


_warned: set[str] = set()


def try_routed_a8(switch_mlp, token_rows, idx, seq_len=None):
    """Sorted rows after the A8 Gate+Up and the A16 Down (``[M, 1, K]``), or
    None to keep the A16 path.

    ``token_rows`` and ``idx`` come from ``moe_routes.sort_routes``.
    ``seq_len`` keeps a batched decode step (many sequences of one token) on
    A16. Exceptions fall back to A16 with one warning per exception type.
    """
    min_tokens = getattr(switch_mlp, _TAG, None)
    if min_tokens is None:
        return None
    try:
        x_tok, row_map = token_rows
        tokens = int(x_tok.shape[0])
        if min(tokens, tokens if seq_len is None else int(seq_len)) < min_tokens:
            return None
        if not _eligible(switch_mlp, switch_mlp.activation):
            return None
        gate_up = switch_mlp.get("gate_up_proj")
        h = sorted_gather_qmm_a8(
            x_tok,
            gate_up["weight"],
            gate_up["scales"],
            gate_up["biases"],
            idx,
            row_map,
            group_size=_GROUP,
            bits=4,
            swiglu=True,
        )
        if h is None:
            return None
        return switch_mlp.down_proj(h, idx, sorted_indices=True)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if name not in _warned:
            _warned.add(name)
            logger.warning(
                "routed A8 gate/up raised %s and fell back to A16", name, exc_info=True
            )
        return None


__all__ = [
    "mx_shape_supported",
    "mxfp8_to_int8",
    "sorted_gather_qmm_a8",
    "sorted_gather_qmm_a8_mx",
    "supports",
    "tag_routed_a8_modules",
    "try_routed_a8",
]
