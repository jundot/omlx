# SPDX-License-Identifier: Apache-2.0
"""Tensor-unit (NAX) QSA main attention, one query per threadgroup.

Qwen4-Exp QSA lets every query attend its own top-512 four-token key blocks
plus the zero-to-three token causal tail. Each threadgroup here is one
(query, KV head): its 12 grouped heads are the rows of one 16-row tensor-unit
tile (4 rows idle) and two simdgroups split the head dim (MLX
attention_nax_dsplit organization: each owns 128 of the 256 dims of Q K^T and
of P V, and they add their partial scores through threadgroup memory). The
threadgroup walks the query's ascending selection list 8 blocks (32 keys) per
step, then its tail block, so every step is exactly the query's own keys: no
block unions, no per-row masks (only tail tokens past the query and the slots
past the list are -inf).

Throughput: the key loop is latency bound, so the kernel keeps its register
and threadgroup-memory footprint small for occupancy: Q is re-read from
device memory (L1) each step instead of held in registers, S and O live in
persistent tensor-unit cooperative tensors (no per-MMA operand copies),
tensor ops are 16x32x32 (Q K^T over 32 head dims; P V with the fp16 hi and lo
pieces of P packed along K), and O is only rescaled when a row max changed.

Numerics: the operand type is AT (see the alias in _ATTN_HEADER). The DENSE
build is bf16 x bf16 accumulated in fp32 - not a free choice: Q/K/V arrive as
bf16 device buffers, and bf16 -> fp16 is NOT a lossless cast (fp16 has 10
mantissa bits but a far narrower exponent), so widening dense operands would
destroy data rather than refine it, while the products themselves would be
identical. The PACKED build stages fp16, because those operands are computed
in-kernel as codebook[fp32 index] x fp16 per-token norm: rounding them to bf16
discards two mantissa bits that fp16 keeps for free (storage stays packed, only
the staged tile widens), and the fp16 range limit is already imposed by the
cache's own fp16 norm field, so no new overflow boundary is introduced.
The online softmax runs in fp32 either way. The tensor unit truncates a float
operand to tf32, so ``P @ V`` takes the fp32 probabilities as two fp16 pieces,
hi = fp16(P) and lo = fp16(P - hi), whose sum is P within 2^-24 absolute (the
size of P's own fp32 rounding); the softmax denominator uses the fp32 P. Only
the fp32 summation grouping differs from the compiled kernel.
"""

from __future__ import annotations

import functools
import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

# How P (fp32 probabilities) enters the P @ V tensor-unit MMA (see module doc):
#   "half2" (default): fp16 hi + fp16 lo pieces, |error| <= 2^-24 per probability.
#   "bf16x3": three bf16 pieces (8+8+8 mantissa bits), fp32-exact for normal P;
#             slower (three 16x32x16 P V ops per fragment instead of one 16x32x32).
PV_MODE = os.environ.get("OMLX_QWEN4_QSA_NAX_PV", "half2")
_PV_MODES = {
    "half2": (mx.float16, 2),
    "bf16x3": (mx.bfloat16, 3),
}
if PV_MODE not in _PV_MODES:
    logger.warning("Unknown OMLX_QWEN4_QSA_NAX_PV=%r; using half2", PV_MODE)
    PV_MODE = "half2"
GQA = 12
HEAD_DIM = 256
COMPRESS = 4
TOPK = 512


def enabled() -> bool:
    """OMLX_QWEN4_QSA_NAX=0 keeps the native direct kernel."""
    return os.environ.get("OMLX_QWEN4_QSA_NAX", "1") != "0"


@functools.lru_cache(maxsize=None)
def nax_available() -> bool:
    """The kernel is built for the tensor units; other GPUs keep the native kernel."""
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:
        return False


_ATTN_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <simd/simd.h>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")

// Attention operand type. The packed build stages FP16: reconstructed values
// come from an exact 2^BITS-entry codebook times an FP16 per-token norm, so
// widening the MMA operand costs no resident memory (storage stays packed) and
// cuts the operand-rounding floor ~8x (10-bit mantissa vs 8). The dense build
// keeps BF16 - its cache already dropped those bits, and FP16 there would be a
// reinterpret of the same 8-bit mantissa, not a gain. Output stays BF16 to
// match the buffer dtype.
#ifdef TQ_BITS
using AT = half;
// Output returns FP32, matching the compiled kernel's contract: the arm
// inverse-rotates the attention result before the final cast to the model
// dtype, so rounding it to BF16 first inserts a lossy step ahead of an exact
// FP32 rotation and costs more than the operand widening saves.
using OT = float;
#else
using AT = bfloat;
using OT = bfloat;
#endif

#ifdef TQ_BITS
// TurboQuant packed-slot unpack, same LSB-first layout as the compiled simdgroup
// kernel (glm_moe_dsa/csrc/kernels/steel_qwen4_qsa_sparse_gqa_tq.h:10-51). This
// kernel only ever asks for 8 contiguous dims at an 8-aligned offset. Reachable bit
// offsets are multiples of 8*BITS, so the intra-word shift is (8*BITS*t) mod 32:
// when 8*BITS divides 32 (BITS 1/2/4) the whole 8-code span always lands in ONE
// word; when it does not (BITS 3/6) the span straddles two words, but the highest
// straddling word index is strictly below the row's last, so the pair stays in
// bounds even at the final token. 8 bits is read as plain bytes. Taking the
// one-word path where it is legal drops a redundant load and every 64-bit shift,
// and removes the past-the-end read the two-word window makes at the last fragment.
template <int BITS>
METAL_FUNC vec<ushort, 8> tq_unpack(const device uchar* row_bytes, int d0) {
  vec<ushort, 8> out;
  constexpr uint32_t M = (1u << BITS) - 1u;
  if constexpr (BITS == 8) {
    const device uchar* p = row_bytes + d0;
    UNROLL for (int j = 0; j < 8; ++j) {
      out[j] = ushort(p[j]);
    }
  } else if constexpr (BITS == 1 || BITS == 2 || BITS == 4) {
    const int bit0 = d0 * BITS;
    const uint word = reinterpret_cast<const device uint*>(row_bytes)[bit0 >> 5];
    const int sh = bit0 & 31;
    UNROLL for (int j = 0; j < 8; ++j) {
      out[j] = ushort((word >> (sh + BITS * j)) & M);
    }
  } else {
    // Straddling widths (3, 6): a code crosses the word boundary, so the 8-code
    // span lives in the (w0, w1) pair. Extract each field with its own shift. This
    // looks wasteful next to a rolling 32-bit walk, but measured it is the faster
    // form by ~15%: the eight `win >> (sh + BITS*j)` shifts are mutually
    // independent and pipeline across the SIMD lanes, whereas a rolling window makes
    // each field depend on the last, serialising the fragment into one dependency
    // chain. The two-word pair here is always in bounds: the largest straddling
    // word index (bit0 + 8*BITS reaching word 22 of 23 at 3 bits, 46 of 47 at 6)
    // stays below the row's own last word, so unlike the sub-word widths this span
    // never reads past the buffer even at the final token.
    const int bit0 = d0 * BITS;
    const device uint* w = reinterpret_cast<const device uint*>(row_bytes) + (bit0 >> 5);
    const int sh = bit0 & 31;
    const uint64_t win = (uint64_t(w[1]) << 32) | uint64_t(w[0]);
    UNROLL for (int j = 0; j < 8; ++j) {
      out[j] = ushort((win >> (sh + BITS * j)) & uint64_t((1u << BITS) - 1u));
    }
  }
  return out;
}
#endif
"""

_ATTN_SOURCE = r"""
    // Grid: (query, KV head); 2 simdgroups: sg = which 128-dim half of D.
    // Tensor-unit fragments (MLX NAX layout): each lane holds rows (fm, fm + 8)
    // x columns (fn .. fn + 3) of a 16x16 fragment. Rows are the 12 heads of
    // the query (rows 12..15 idle); columns of S are keys.
    constexpr int D = 256;
    constexpr int TDH = 8;      // 16-wide head-dim fragments per half
    constexpr int SB = 8;       // blocks (4 tokens each) per step: 32 keys
    constexpr int PV_K = PV_TERMS == 2 ? 32 : 16;
    threadgroup float xchg[2][8 * 32];

    const int tq = int(threadgroup_position_in_grid.x);
    const int kvh = int(threadgroup_position_in_grid.y);
    const ushort dh = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;

    const int q_offset = params[0];
    const int kL = params[1];
    const float scale2 = scale[0] * 1.44269504089f;
    const int p = q_offset + tq;
    const int complete = (p + 1) >> 2;
    const int nsel = min(TOPK, complete);        // valid selected blocks
    const int ntail = p + 1 - (complete << 2);   // 0..3 tail tokens
    const int U = nsel + (ntail > 0 ? 1 : 0);    // blocks: selection, then tail
    const int nsteps = (U + SB - 1) / SB;
    const bool r1_ok = fm + 8 < GQA;             // row fm + 8 is a real head
    const int h0 = kvh * GQA + fm;
    const int h1 = kvh * GQA + (r1_ok ? fm + 8 : fm);

    // Strides (elements): q [1, H, Lq, D], k/v [1, KVH, kL, D]; the last dim is
    // contiguous and rows are 16-byte aligned. Head dims are permuted per lane:
    // fragment pair (2j, 2j + 1) covers the 8 contiguous dims 32 j + 2 fn .. + 7
    // (the first four in fragment 2j) of Q/K and of V/O, so one 16-byte load
    // feeds both fragments. Q and K share the permutation (only the Q K^T
    // summation order changes); O undoes V's at the store.
    const int64_t sqh = q_strides[1], sql = q_strides[2];
#ifndef TQ_BITS
    const uint skl = uint(k_strides[2]), svl = uint(v_strides[2]);
#endif
    const short fcol = 2 * fn;
#ifndef TQ_BITS
    const device AT* kb = (const device AT*)k + kvh * k_strides[1] + dh * 128 + fcol;
    const device AT* vb = (const device AT*)v + kvh * v_strides[1] + dh * 128 + fcol;
#endif
    const device AT* qp0 = (const device AT*)q + h0 * sqh + tq * sql + dh * 128 + fcol;
    const device AT* qp1 = (const device AT*)q + h1 * sqh + tq * sql + dh * 128 + fcol;
    const device int* blk = sel + size_t(tq) * TOPK;

#ifdef TQ_BITS
    // Packed rows are BITS * 32 bytes per (token, KV head): 256 dims at
    // BITS bits. Norms are one fp16 per (token, KV head).
    const device uchar* kp = (const device uchar*)k_pack + size_t(kvh) * kL * (TQ_KBITS * 32);
    const device uchar* vp = (const device uchar*)v_pack + size_t(kvh) * kL * (TQ_VBITS * 32);
    const device half* kn = (const device half*)k_norm + size_t(kvh) * kL;
    const device half* vn = (const device half*)v_norm + size_t(kvh) * kL;
    threadgroup float cbk[1 << TQ_KBITS];
    threadgroup float cbv[1 << TQ_VBITS];
    const short tid = short(dh) * 32 + short(lane);
    UNROLL for (int i = tid; i < (1 << TQ_KBITS); i += 64) {
      cbk[i] = cb_k[i];
    }
    UNROLL for (int i = tid; i < (1 << TQ_VBITS); i += 64) {
      cbv[i] = cb_v[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto kbytes = [&](uint tok) -> const device uchar* { return kp + size_t(tok) * (TQ_KBITS * 32); };
    auto vbytes = [&](uint tok) -> const device uchar* { return vp + size_t(tok) * (TQ_VBITS * 32); };
#endif

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, PV_K, false, false, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
    // Cooperative tensors: element 8 c + i is element i of the c-th 16x16
    // fragment along K (left) / along (K, N) (right) / along N (destination).
    auto qa_ct = qk_op.template get_left_input_cooperative_tensor<AT, AT, float>();
    auto kb_ct = qk_op.template get_right_input_cooperative_tensor<AT, AT, float>();
    using qa_t = metal::remove_addrspace_t<decltype(qa_ct)>;
    using kb_t = metal::remove_addrspace_t<decltype(kb_ct)>;
    auto pa_ct = pv_op.template get_left_input_cooperative_tensor<PT, AT, float>();
    auto vb_ct = pv_op.template get_right_input_cooperative_tensor<PT, AT, float>();
    using pa_t = metal::remove_addrspace_t<decltype(pa_ct)>;
    using vb_t = metal::remove_addrspace_t<decltype(vb_ct)>;
    // O over this half of D: of<j> holds dim fragments (2 j, 2 j + 1).
    auto of0 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of1 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of2 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of3 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    UNROLL for (short i = 0; i < 16; ++i) {
      of0[i] = 0.0f;
      of1[i] = 0.0f;
      of2[i] = 0.0f;
      of3[i] = 0.0f;
    }
    float max_s[2] = {-FLT_MAX, -FLT_MAX};
    float sum_s[2] = {0.0f, 0.0f};

    // Step metadata, prefetched one step ahead: the block of each key row this
    // lane loads (keys 16 (r / 2) + fm + 8 (r % 2) -> slot key / 4) and the
    // number of visible tokens of this lane's S column block (slot 4 f + fn / 4).
    auto blk_at = [&](int u) -> int { return u < nsel ? blk[u] : complete; };
    int nrow_b[4];
    int ncol_n[2];
    auto fetch = [&](int u0) {
      UNROLL for (short r = 0; r < 4; ++r) {
        const int u = u0 + ((16 * (r >> 1) + fm + 8 * (r & 1)) >> 2);
        nrow_b[r] = u < U ? blk_at(u) : 0;
      }
      UNROLL for (short f = 0; f < 2; ++f) {
        const int u = u0 + 4 * f + (fn >> 2);
        ncol_n[f] = u < nsel ? 4 : (u < U ? ntail : 0);
      }
    };
    fetch(0);

    for (int step = 0; step < nsteps; ++step) {
      // The tail block can extend up to three rows past kL: those keys are
      // masked, but P = 0 must never meet unwritten V (0 * NaN = NaN), so
      // they re-read the last valid row instead.
      uint krow[4];
      UNROLL for (short r = 0; r < 4; ++r) {
        krow[r] = uint(min(nrow_b[r] * 4 + (fm & 3), kL - 1));
      }
#ifdef TQ_BITS
      // Key norms are applied to the FP32 scores, not to the bf16 MMA operand:
      // rounding cb[code] * norm to bf16 before the tensor unit costs one extra
      // rounding per key element instead of one per score, and measurably widens
      // drift over long sessions (relL2 0.0045 folded-at-unpack vs 0.0037 for
      // the compiled kernel, which scales in the epilogue). This lane's score
      // columns are exactly the four tokens of one block, so the token identity
      // is cheap to recover here.
      const int u0 = step * SB;
      float cn[2][4];
      UNROLL for (short f = 0; f < 2; ++f) {
        const int blk = blk_at(u0 + 4 * f + (fn >> 2));
        UNROLL for (short j = 0; j < 4; ++j) {
          cn[f][j] = float(kn[min(blk * 4 + j, kL - 1)]);
        }
      }
#endif
      int nvis[2];
      UNROLL for (short f = 0; f < 2; ++f) {
        nvis[f] = ncol_n[f];
      }
      if (step + 1 < nsteps) {
        fetch((step + 1) * SB);
      }

      // S = Q K^T over this half of D: one 16x32x32 op per 32 head dims.
      auto s_ct = qk_op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
      UNROLL for (short i = 0; i < 16; ++i) {
        s_ct[i] = 0.0f;
      }
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        const vec<AT, 8> qa = *(const device vec<AT, 8>*)(qp0 + 32 * jj);
        const vec<AT, 8> qb = r1_ok ? *(const device vec<AT, 8>*)(qp1 + 32 * jj) : vec<AT, 8>(0);
#ifdef TQ_BITS
        vec<AT, 8> a0;
        vec<AT, 8> a1;
        vec<AT, 8> a2;
        vec<AT, 8> a3;
        {
          const vec<ushort, 8> u0 = tq_unpack<TQ_KBITS>(kbytes(krow[0]), dh * 128 + fcol + 32 * jj);
          const vec<ushort, 8> u1 = tq_unpack<TQ_KBITS>(kbytes(krow[1]), dh * 128 + fcol + 32 * jj);
          const vec<ushort, 8> u2 = tq_unpack<TQ_KBITS>(kbytes(krow[2]), dh * 128 + fcol + 32 * jj);
          const vec<ushort, 8> u3 = tq_unpack<TQ_KBITS>(kbytes(krow[3]), dh * 128 + fcol + 32 * jj);
          UNROLL for (short j = 0; j < 8; ++j) {
            a0[j] = AT(cbk[u0[j]]);
            a1[j] = AT(cbk[u1[j]]);
            a2[j] = AT(cbk[u2[j]]);
            a3[j] = AT(cbk[u3[j]]);
          }
        }
#else
        const vec<AT, 8> a0 = *(const device vec<AT, 8>*)(kb + (krow[0] * skl + 32 * jj));
        const vec<AT, 8> a1 = *(const device vec<AT, 8>*)(kb + (krow[1] * skl + 32 * jj));
        const vec<AT, 8> a2 = *(const device vec<AT, 8>*)(kb + (krow[2] * skl + 32 * jj));
        const vec<AT, 8> a3 = *(const device vec<AT, 8>*)(kb + (krow[3] * skl + 32 * jj));
#endif
        UNROLL for (short j = 0; j < 4; ++j) {
          qa_ct[j] = qa[j];
          qa_ct[4 + j] = qb[j];
          qa_ct[8 + j] = qa[4 + j];
          qa_ct[12 + j] = qb[4 + j];
          kb_ct[j] = a0[j];
          kb_ct[4 + j] = a1[j];
          kb_ct[8 + j] = a2[j];
          kb_ct[12 + j] = a3[j];
          kb_ct[16 + j] = a0[4 + j];
          kb_ct[20 + j] = a1[4 + j];
          kb_ct[24 + j] = a2[4 + j];
          kb_ct[28 + j] = a3[4 + j];
        }
        qk_op.run(qa_ct, kb_ct, s_ct);
      }
      vec<float, 8> s[2];
      UNROLL for (short i = 0; i < 8; ++i) {
        s[0][i] = s_ct[i];
        s[1][i] = s_ct[8 + i];
      }

      // Add the other half's partial scores, one 16-key fragment at a time.
      UNROLL for (short f = 0; f < 2; ++f) {
        threadgroup float* mine = xchg[dh];
        const threadgroup float* peer = xchg[1 - dh];
        UNROLL for (short i = 0; i < 8; ++i) {
          mine[lane * 8 + i] = s[f][i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        UNROLL for (short i = 0; i < 8; ++i) {
          s[f][i] += peer[lane * 8 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      // Scale and mask: this lane's columns fn..fn+3 of fragment f are the four
      // tokens of one block; the first nvis of them are visible.
      UNROLL for (short f = 0; f < 2; ++f) {
        UNROLL for (short j = 0; j < 4; ++j) {
          const bool ok = j < nvis[f];
#ifdef TQ_BITS
          s[f][j] = ok ? s[f][j] * scale2 * cn[f][j] : -INFINITY;
          s[f][4 + j] = ok ? s[f][4 + j] * scale2 * cn[f][j] : -INFINITY;
#else
          s[f][j] = ok ? s[f][j] * scale2 : -INFINITY;
          s[f][4 + j] = ok ? s[f][4 + j] * scale2 : -INFINITY;
#endif
        }
      }
      // Online softmax per row (rows fm and fm + 8 of the fragments).
      float factor[2];
      UNROLL for (short i = 0; i < 2; ++i) {
        float m = -INFINITY;
        UNROLL for (short f = 0; f < 2; ++f) {
          m = max(m, max(max(s[f][4 * i], s[f][4 * i + 1]), max(s[f][4 * i + 2], s[f][4 * i + 3])));
        }
        m = max(m, simd_shuffle_xor(m, ushort(1)));
        m = max(m, simd_shuffle_xor(m, ushort(8)));
        const float new_max = max(max_s[i], m);
        float rs = 0.0f;
        UNROLL for (short f = 0; f < 2; ++f) {
          UNROLL for (short j = 0; j < 4; ++j) {
            s[f][4 * i + j] = fast::exp2(s[f][4 * i + j] - new_max);
            rs += s[f][4 * i + j];
          }
        }
        rs += simd_shuffle_xor(rs, ushort(1));
        rs += simd_shuffle_xor(rs, ushort(8));
        factor[i] = fast::exp2(max_s[i] - new_max);
        max_s[i] = new_max;
        sum_s[i] = sum_s[i] * factor[i] + rs;
      }
      // O *= factor (exact no-op when every factor of the simdgroup is 1).
      if (simd_any(factor[0] != 1.0f || factor[1] != 1.0f)) {
        UNROLL for (short h = 0; h < 2; ++h) {
          UNROLL for (short j = 0; j < 4; ++j) {
            of0[8 * h + j] *= factor[0];
            of0[8 * h + 4 + j] *= factor[1];
            of1[8 * h + j] *= factor[0];
            of1[8 * h + 4 + j] *= factor[1];
            of2[8 * h + j] *= factor[0];
            of2[8 * h + 4 + j] *= factor[1];
            of3[8 * h + j] *= factor[0];
            of3[8 * h + 4 + j] *= factor[1];
          }
        }
      }
      // O += P V over this half of D (V fragments are [16 keys x 16 dims]).
      // The tensor unit takes P as PV_TERMS pieces of type PT whose sum is P;
      // two pieces are packed along K into one 16x32x32 op.
      UNROLL for (short f = 0; f < 2; ++f) {
        vec<PT, 8> pc[PV_TERMS];
        {
          vec<float, 8> rest = s[f];
          UNROLL for (short tt = 0; tt < PV_TERMS; ++tt) {
            UNROLL for (short i = 0; i < 8; ++i) {
              const PT piece = PT(rest[i]);
              pc[tt][i] = piece;
              rest[i] -= float(piece);
            }
          }
        }
        UNROLL for (short id = 0; id < TDH; id += 2) {
#ifdef TQ_BITS
          vec<AT, 8> ra;
          vec<AT, 8> rb;
          {
            const int dv = dh * 128 + fcol + 16 * id;
            const vec<ushort, 8> uva = tq_unpack<TQ_VBITS>(vbytes(krow[2 * f]), dv);
            const vec<ushort, 8> uvb = tq_unpack<TQ_VBITS>(vbytes(krow[2 * f + 1]), dv);
            const float na = float(vn[krow[2 * f]]);
            const float nb = float(vn[krow[2 * f + 1]]);
            UNROLL for (short j = 0; j < 8; ++j) {
              ra[j] = AT(cbv[uva[j]] * na);
              rb[j] = AT(cbv[uvb[j]] * nb);
            }
          }
#else
          const vec<AT, 8> ra = *(const device vec<AT, 8>*)(vb + (krow[2 * f] * svl + 16 * id));
          const vec<AT, 8> rb = *(const device vec<AT, 8>*)(vb + (krow[2 * f + 1] * svl + 16 * id));
#endif
          UNROLL for (short j = 0; j < 4; ++j) {
            vb_ct[j] = ra[j];
            vb_ct[4 + j] = rb[j];
            vb_ct[8 + j] = ra[4 + j];
            vb_ct[12 + j] = rb[4 + j];
            if (PV_K == 32) {
              vb_ct[16 + j] = ra[j];
              vb_ct[20 + j] = rb[j];
              vb_ct[24 + j] = ra[4 + j];
              vb_ct[28 + j] = rb[4 + j];
            }
          }
          UNROLL for (short tt = 0; tt < PV_TERMS; tt += PV_K / 16) {
            UNROLL for (short i = 0; i < 8; ++i) {
              pa_ct[i] = pc[tt][i];
              if (PV_K == 32) {
                pa_ct[8 + i] = pc[tt + 1][i];
              }
            }
            if (id == 0) {
              pv_op.run(pa_ct, vb_ct, of0);
            } else if (id == 2) {
              pv_op.run(pa_ct, vb_ct, of1);
            } else if (id == 4) {
              pv_op.run(pa_ct, vb_ct, of2);
            } else {
              pv_op.run(pa_ct, vb_ct, of3);
            }
          }
        }
      }
    }

    UNROLL for (short i = 0; i < 2; ++i) {
      if (i == 1 && !r1_ok) {
        continue;
      }
      const float rr = 1.0f / sum_s[i];
      device OT* o = (device OT*)out + (size_t(tq) * (2 * GQA) + (i == 0 ? h0 : h1)) * D + dh * 128 + fcol;
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        vec<OT, 8> w;
        UNROLL for (short j = 0; j < 4; ++j) {
          float e0, e1;
          if (jj == 0) {
            e0 = of0[4 * i + j];
            e1 = of0[8 + 4 * i + j];
          } else if (jj == 1) {
            e0 = of1[4 * i + j];
            e1 = of1[8 + 4 * i + j];
          } else if (jj == 2) {
            e0 = of2[4 * i + j];
            e1 = of2[8 + 4 * i + j];
          } else {
            e0 = of3[4 * i + j];
            e1 = of3[8 + 4 * i + j];
          }
          w[j] = OT(e0 * rr);
          w[4 + j] = OT(e1 * rr);
        }
        *(device vec<OT, 8>*)(o + 32 * jj) = w;
      }
    }
"""


@functools.lru_cache(maxsize=None)
def _attn_kernel():
    return mx.fast.metal_kernel(
        name="omlx_qwen4_qsa_nax_query_attention",
        input_names=["q", "k", "v", "sel", "params", "scale"],
        output_names=["out"],
        source=_ATTN_SOURCE,
        header=_ATTN_HEADER,
        ensure_row_contiguous=False,
    )


def sparse_gqa_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    selected_blocks: mx.array,
    *,
    q_offset: int,
    scale: float | None = None,
) -> mx.array:
    """QSA main attention for ``queries`` [1, 24, Lq, 256] at absolute rows
    ``q_offset ..``, K/V [1, 2, kL, 256], chronological ``selected_blocks``
    [1, Lq, 512]. Returns [1, Lq, 24, 256].

    Q/K/V may be strided views (cache buffers with spare capacity, sequence
    slices) but their head dim must be contiguous and 16-byte aligned, as for
    the native kernel.
    """

    lq = queries.shape[2]
    kl = keys.shape[2]
    if scale is None:
        scale = HEAD_DIM**-0.5
    sel = selected_blocks.reshape(lq, TOPK)
    if sel.dtype != mx.int32:
        sel = sel.astype(mx.int32)
    sel = mx.contiguous(sel)
    pt, terms = _PV_MODES[PV_MODE]
    params = mx.array([q_offset, kl], dtype=mx.int32)
    (out,) = _attn_kernel()(
        inputs=[
            queries,
            keys,
            values,
            sel,
            params,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("GQA", GQA),
            ("TOPK", TOPK),
            ("PT", pt),
            ("PV_TERMS", terms),
        ],
        grid=(lq * 64, keys.shape[1], 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, lq, 2 * GQA, HEAD_DIM)],
        output_dtypes=[queries.dtype],
    )
    return out


# Integer widths this packed unpack is validated for, matching the native compiled
# kernel's instantiated set (_TQ_NATIVE_BIT_PAIRS, qwen4_qsa_sparse_gqa_tq.metal):
# there is no 5- or 7-bit instantiation anywhere in the stack, so none is offered
# here, and the arm's contract of falling back to a kernel that exists still holds.
#
# Membership is decided by the unpack, not by the codebook. A 256-dim row packs to
# exactly 8*BITS uint32 words, so every integer width addresses cleanly; what has to
# hold is that each 8-dim fragment the tensor-unit arm loads lies inside words the
# row owns. Reachable fragment offsets are 8-aligned, so a span begins at a multiple
# of 8*BITS and the largest intra-word shift it can carry is 32-gcd(8*BITS,32);
# that plus one 8-code span is 32+BITS, within the 64-bit window for every width
# here. At the row's last fragment the two-word path's higher word index stays below
# the row's own last word for the straddling widths (3, 6), and the widths whose
# span fits one word (2, 4) never read the second word at all.
#
# Fractional cache widths quantize at their floor/ceil codec pair (2.5 -> (2, 3),
# 3.5 -> (3, 4)) and ride these asymmetric entries: the norm fold, codebook size and
# row stride are all per-side compile-time constants, so K and V need not agree.
_TQ_BIT_PAIRS = frozenset(
    {(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (6, 6), (8, 8)}
)


@functools.lru_cache(maxsize=None)
def _attn_kernel_tq(k_bits: int, v_bits: int):
    return mx.fast.metal_kernel(
        name=f"omlx_qwen4_qsa_nax_query_attention_tq_k{k_bits}v{v_bits}",
        input_names=[
            "q",
            "sel",
            "params",
            "scale",
            "k_pack",
            "v_pack",
            "k_norm",
            "v_norm",
            "cb_k",
            "cb_v",
        ],
        output_names=["out"],
        source=_ATTN_SOURCE,
        header=_tq_defines(k_bits, v_bits) + _ATTN_HEADER,
        ensure_row_contiguous=False,
    )


def tq_enabled() -> bool:
    """OMLX_QWEN4_QSA_NAX_TQ=0 keeps the packed arm on the compiled kernel.

    Separate from OMLX_QWEN4_QSA_NAX so the dense tensor-unit route can stay
    on while only the packed fusion is retired for an A/B.
    """

    return os.environ.get("OMLX_QWEN4_QSA_NAX_TQ", "1") != "0"


def tq_supported(k_bits: int, v_bits: int) -> bool:
    return (
        enabled()
        and tq_enabled()
        and nax_available()
        and (int(k_bits), int(v_bits)) in _TQ_BIT_PAIRS
    )


def sparse_gqa_attention_tq(
    queries: mx.array,
    k_norms: mx.array,
    k_indices: mx.array,
    v_norms: mx.array,
    v_indices: mx.array,
    k_codebook: mx.array,
    v_codebook: mx.array,
    selected_blocks: mx.array,
    *,
    key_tokens: int,
    q_offset: int,
    k_bits: int,
    v_bits: int,
    scale: float | None = None,
) -> mx.array | None:
    """TurboQuant-packed counterpart of :func:`sparse_gqa_attention`.

    Reads K/V straight from the packed bitstream into the tensor-unit
    cooperative tensors: no dequantized device copy, and the tensor units the
    dense arm gets today. ``selected_blocks`` [1, Lq, 512] must be ascending,
    exactly as the dense arm produces. Returns ``None`` on any unsupported
    shape, dtype or bit width so the caller falls back to the compiled simdgroup
    kernel and then to the portable gather.
    """

    if not tq_supported(k_bits, v_bits):
        return None
    lq = queries.shape[2]
    if queries.ndim != 4 or queries.shape[0] != 1 or queries.shape[3] != HEAD_DIM:
        return None
    if lq < 2:
        return None
    kvh = k_indices.shape[1]
    words = -(-HEAD_DIM * int(k_bits) // 32)
    if tuple(k_indices.shape) != (1, kvh, key_tokens, words):
        return None
    if tuple(v_indices.shape) != (1, kvh, key_tokens, -(-HEAD_DIM * int(v_bits) // 32)):
        return None
    if k_codebook.shape[0] != (1 << int(k_bits)) or v_codebook.shape[0] != (1 << int(v_bits)):
        return None
    if queries.dtype != mx.bfloat16 or k_indices.dtype != mx.uint32:
        return None

    sel = mx.contiguous(selected_blocks.reshape(lq, TOPK).astype(mx.int32))
    if scale is None:
        scale = HEAD_DIM**-0.5
    (out,) = _attn_kernel_tq(int(k_bits), int(v_bits))(
        inputs=[
            queries.astype(mx.float16),  # cast, not reinterpret: packed kernel stages FP16
            sel,
            mx.array([q_offset, key_tokens], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
            mx.contiguous(k_indices),
            mx.contiguous(v_indices),
            mx.contiguous(k_norms.astype(mx.float16)),
            mx.contiguous(v_norms.astype(mx.float16)),
            mx.contiguous(k_codebook.astype(mx.float32)),
            mx.contiguous(v_codebook.astype(mx.float32)),
        ],
        template=[
            ("GQA", GQA),
            ("TOPK", TOPK),
            ("PT", _PV_MODES[PV_MODE][0]),
            ("PV_TERMS", _PV_MODES[PV_MODE][1]),
        ],
        grid=(lq * 64, kvh, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, lq, 2 * GQA, HEAD_DIM)],
        output_dtypes=[mx.float32],
    )
    return out


def _tq_defines(k_bits: int, v_bits: int) -> str:
    """MLX ``template`` entries are textual substitutions, not macros.

    The packed guards in ``_ATTN_SOURCE`` are preprocessor ``#ifdef``/``#ifndef``
    so the dense build stays byte-identical, which means ``TQ_BITS`` has to be a
    real ``#define`` prepended ahead of the guarded helper block in the header.
    """

    return (
        f"#define TQ_BITS 1\n"
        f"#define TQ_KBITS {int(k_bits)}\n"
        f"#define TQ_VBITS {int(v_bits)}\n"
    )
