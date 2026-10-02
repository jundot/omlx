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

Numerics: scores are bf16 x bf16 products accumulated in fp32 and
the online softmax runs in fp32. The tensor unit truncates a float operand to
tf32, so ``P @ V`` takes the fp32 probabilities as two fp16 pieces, hi =
fp16(P) and lo = fp16(P - hi), whose sum is P within 2^-24 absolute (the size
of P's own fp32 rounding); the softmax denominator uses the fp32 P. Only the
fp32 summation grouping differs from the native kernel.
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
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")
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
    const uint skl = uint(k_strides[2]), svl = uint(v_strides[2]);
    const short fcol = 2 * fn;
    const device bfloat* kb = (const device bfloat*)k + kvh * k_strides[1] + dh * 128 + fcol;
    const device bfloat* vb = (const device bfloat*)v + kvh * v_strides[1] + dh * 128 + fcol;
    const device bfloat* qp0 = (const device bfloat*)q + h0 * sqh + tq * sql + dh * 128 + fcol;
    const device bfloat* qp1 = (const device bfloat*)q + h1 * sqh + tq * sql + dh * 128 + fcol;
    const device int* blk = sel + size_t(tq) * TOPK;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, PV_K, false, false, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
    // Cooperative tensors: element 8 c + i is element i of the c-th 16x16
    // fragment along K (left) / along (K, N) (right) / along N (destination).
    auto qa_ct = qk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto kb_ct = qk_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    using qa_t = metal::remove_addrspace_t<decltype(qa_ct)>;
    using kb_t = metal::remove_addrspace_t<decltype(kb_ct)>;
    auto pa_ct = pv_op.template get_left_input_cooperative_tensor<PT, bfloat, float>();
    auto vb_ct = pv_op.template get_right_input_cooperative_tensor<PT, bfloat, float>();
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
        const vec<bfloat, 8> qa = *(const device vec<bfloat, 8>*)(qp0 + 32 * jj);
        const vec<bfloat, 8> qb = r1_ok ? *(const device vec<bfloat, 8>*)(qp1 + 32 * jj) : vec<bfloat, 8>(0);
        const vec<bfloat, 8> a0 = *(const device vec<bfloat, 8>*)(kb + (krow[0] * skl + 32 * jj));
        const vec<bfloat, 8> a1 = *(const device vec<bfloat, 8>*)(kb + (krow[1] * skl + 32 * jj));
        const vec<bfloat, 8> a2 = *(const device vec<bfloat, 8>*)(kb + (krow[2] * skl + 32 * jj));
        const vec<bfloat, 8> a3 = *(const device vec<bfloat, 8>*)(kb + (krow[3] * skl + 32 * jj));
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
          s[f][j] = ok ? s[f][j] * scale2 : -INFINITY;
          s[f][4 + j] = ok ? s[f][4 + j] * scale2 : -INFINITY;
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
          const vec<bfloat, 8> ra = *(const device vec<bfloat, 8>*)(vb + (krow[2 * f] * svl + 16 * id));
          const vec<bfloat, 8> rb = *(const device vec<bfloat, 8>*)(vb + (krow[2 * f + 1] * svl + 16 * id));
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
      device bfloat* o = (device bfloat*)out + (size_t(tq) * (2 * GQA) + (i == 0 ? h0 : h1)) * D + dh * 128 + fcol;
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        vec<bfloat, 8> w;
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
          w[j] = bfloat(e0 * rr);
          w[4 + j] = bfloat(e1 * rr);
        }
        *(device vec<bfloat, 8>*)(o + 32 * jj) = w;
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
