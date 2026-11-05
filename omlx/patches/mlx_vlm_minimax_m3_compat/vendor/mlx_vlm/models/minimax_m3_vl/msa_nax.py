# SPDX-License-Identifier: Apache-2.0
"""Tensor-unit (NAX) MSA prefill kernels for MiniMax M3.

Block scores: one threadgroup per (128-key block, 64 queries), eight
simdgroups of 32 queries x 32 keys. Each index head's Q K^T runs on
16x32x32 tensor ops in fp32, and the masked row max over the block is reduced
in the epilogue, so the [H, L, total_k] token scores never exist. Blocks past
every query of the threadgroup skip the matmuls. The max commutes with the
positive scale, so only the fp32 summation order differs from the native
kernel.

Sparse attention: one threadgroup per (query, KV head). The 16 grouped heads
are the 16 rows of a tensor-unit tile and two simdgroups split the head dim
(as in the Qwen4-Exp QSA kernel). The threadgroup walks the query's ascending
block list 32 keys per step with an fp32 online softmax and writes the output
once, so there are no per-split partials, no CSR metadata and no host sync.
P enters P V as fp16 hi and lo pieces (the tensor unit truncates fp32
operands), and the softmax denominator uses the fp32 P.

Inputs may be strided views (cache buffers, head transposes), but their
128-wide head dim must be contiguous and 16-byte aligned, as for the cache
and projection outputs of the model.
"""

from __future__ import annotations

import functools
from typing import Optional

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available

BLOCK = 128
HEAD_DIM = 128
GQA = 16
TOPK = 16

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
#define UNROLL _Pragma("clang loop unroll(full)")
"""

# Lane layout of a 16x16 tensor-unit fragment (MLX NAX): rows fm and fm + 8,
# columns fn .. fn + 3. Fragment pair (2 j, 2 j + 1) along a 32-wide K slice
# covers the 8 contiguous dims 32 j + 2 fn .. + 7, so one 16-byte load feeds
# both; Q and K share the permutation, which only reorders the dot product.
_SCORES_SOURCE = r"""
    constexpr int D = 128;
    constexpr int BS = 128;
    threadgroup float red[64 * 4];

    const int b = int(threadgroup_position_in_grid.x);
    const int q0 = int(threadgroup_position_in_grid.y) * 64;
    const uint tid = thread_index_in_threadgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short wm = short(sg >> 2);
    const short wn = short(sg & 3);
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    const short fcol = 2 * fn;

    const int L = params[0];
    const int S = params[1];
    const int q_start = params[2];
    const int H = params[3];
    const int NB = params[4];
    const float scale = sc[0];

    const int q_last = min(q0 + 63, L - 1);
    if (b * BS > q_start + q_last) {
        // Every key of the block is in the future of every query here.
        for (uint i = tid; i < uint(64 * H); i += 256) {
            const int q = q0 + int(i % 64);
            if (q < L) {
                out[(size_t(i / 64) * L + q) * NB + b] = -INFINITY;
            }
        }
        return;
    }

    const int64_t sqh = q_strides[1], sql = q_strides[2];
    const int64_t skl = k_strides[2];
    const int sm = q0 + wm * 32;
    const int kb0 = b * BS + wn * 32;

    int qrow[2][2];
    UNROLL for (short tm = 0; tm < 2; ++tm) {
        UNROLL for (short i = 0; i < 2; ++i) {
            qrow[tm][i] = min(sm + tm * 16 + fm + 8 * i, L - 1);
        }
    }
    const device bfloat* krow[4];
    UNROLL for (short r = 0; r < 4; ++r) {
        const int key = min(kb0 + 16 * (r >> 1) + fm + 8 * (r & 1), S - 1);
        krow[r] = (const device bfloat*)k + key * skl + fcol;
    }

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto qa_ct = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto kb_ct = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    using qa_t = metal::remove_addrspace_t<decltype(qa_ct)>;
    using kb_t = metal::remove_addrspace_t<decltype(kb_ct)>;

    for (int h = 0; h < H; ++h) {
        auto s0 = op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
        auto s1 = op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
        UNROLL for (short i = 0; i < 16; ++i) {
            s0[i] = 0.0f;
            s1[i] = 0.0f;
        }
        const device bfloat* qh = (const device bfloat*)q + h * sqh + fcol;
        UNROLL for (short jj = 0; jj < D / 32; ++jj) {
            vec<bfloat, 8> kr[4];
            UNROLL for (short r = 0; r < 4; ++r) {
                kr[r] = *(const device vec<bfloat, 8>*)(krow[r] + 32 * jj);
            }
            UNROLL for (short j = 0; j < 4; ++j) {
                kb_ct[j] = kr[0][j];
                kb_ct[4 + j] = kr[1][j];
                kb_ct[8 + j] = kr[2][j];
                kb_ct[12 + j] = kr[3][j];
                kb_ct[16 + j] = kr[0][4 + j];
                kb_ct[20 + j] = kr[1][4 + j];
                kb_ct[24 + j] = kr[2][4 + j];
                kb_ct[28 + j] = kr[3][4 + j];
            }
            UNROLL for (short tm = 0; tm < 2; ++tm) {
                const vec<bfloat, 8> qa = *(const device vec<bfloat, 8>*)(qh + qrow[tm][0] * sql + 32 * jj);
                const vec<bfloat, 8> qb = *(const device vec<bfloat, 8>*)(qh + qrow[tm][1] * sql + 32 * jj);
                UNROLL for (short j = 0; j < 4; ++j) {
                    qa_ct[j] = qa[j];
                    qa_ct[4 + j] = qb[j];
                    qa_ct[8 + j] = qa[4 + j];
                    qa_ct[12 + j] = qb[4 + j];
                }
                if (tm == 0) {
                    op.run(qa_ct, kb_ct, s0);
                } else {
                    op.run(qa_ct, kb_ct, s1);
                }
            }
        }

        // Masked row max of this simdgroup's 32 keys, then across the four
        // key quarters of the block.
        UNROLL for (short tm = 0; tm < 2; ++tm) {
            UNROLL for (short i = 0; i < 2; ++i) {
                const int qabs = q_start + sm + tm * 16 + fm + 8 * i;
                float m = -INFINITY;
                UNROLL for (short f = 0; f < 2; ++f) {
                    UNROLL for (short j = 0; j < 4; ++j) {
                        const int key = kb0 + 16 * f + fn + j;
                        const float v = tm == 0 ? s0[8 * f + 4 * i + j] : s1[8 * f + 4 * i + j];
                        if (key <= qabs && key < S && v == v) {
                            m = max(m, v);
                        }
                    }
                }
                m = max(m, simd_shuffle_xor(m, ushort(1)));
                m = max(m, simd_shuffle_xor(m, ushort(8)));
                if ((lane & 9) == 0) {
                    red[(wm * 32 + tm * 16 + fm + 8 * i) * 4 + wn] = m;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64) {
            const int q = q0 + int(tid);
            if (q < L) {
                const threadgroup float* rr = red + tid * 4;
                const float m = max(max(rr[0], rr[1]), max(rr[2], rr[3]));
                out[(size_t(h) * L + q) * NB + b] = m * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_ATTN_SOURCE = r"""
    // Grid: (query, KV head); 2 simdgroups: dh = which 64-dim half of D.
    constexpr int D = 128;
    constexpr int TDH = 4;      // 16-wide head-dim fragments per half
    constexpr int BS = 128;
    threadgroup float xchg[2][8 * 32];

    const int tq = int(threadgroup_position_in_grid.x);
    const int kvh = int(threadgroup_position_in_grid.y);
    const ushort dh = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;

    const int q_start = params[0];
    const int Lq = params[1];
    const int kL = params[2];
    const float scale2 = scale[0] * 1.44269504089f;
    const int p = q_start + tq;
    // The last visible key: the query's own, or the last key row there is.
    const int last = min(p, kL - 1);
    const int h0 = kvh * GQA + fm;
    const int h1 = h0 + 8;

    // q [1, HQ, Lq, D], k/v [1, KVH, kL, D] (strided views; the head dim is
    // contiguous and rows are 16-byte aligned).
    const int64_t sqh = q_strides[1], sql = q_strides[2];
    const uint skl = uint(k_strides[2]), svl = uint(v_strides[2]);
    const short fcol = 2 * fn;
    const device bfloat* kb = (const device bfloat*)k + kvh * k_strides[1] + dh * 64 + fcol;
    const device bfloat* vb = (const device bfloat*)v + kvh * v_strides[1] + dh * 64 + fcol;
    const device bfloat* qp0 = (const device bfloat*)q + h0 * sqh + tq * sql + dh * 64 + fcol;
    const device bfloat* qp1 = (const device bfloat*)q + h1 * sqh + tq * sql + dh * 64 + fcol;
    const device int* blk = sel + (size_t(kvh) * Lq + tq) * TOPK;

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> qk_op;
    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, false, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
    auto qa_ct = qk_op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto kb_ct = qk_op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    using qa_t = metal::remove_addrspace_t<decltype(qa_ct)>;
    using kb_t = metal::remove_addrspace_t<decltype(kb_ct)>;
    auto pa_ct = pv_op.template get_left_input_cooperative_tensor<half, bfloat, float>();
    auto vb_ct = pv_op.template get_right_input_cooperative_tensor<half, bfloat, float>();
    using pa_t = metal::remove_addrspace_t<decltype(pa_ct)>;
    using vb_t = metal::remove_addrspace_t<decltype(vb_ct)>;
    // O over this half of D: of<j> holds dim fragments (2 j, 2 j + 1).
    auto of0 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    auto of1 = pv_op.template get_destination_cooperative_tensor<pa_t, vb_t, float>();
    UNROLL for (short i = 0; i < 16; ++i) {
      of0[i] = 0.0f;
      of1[i] = 0.0f;
    }
    float max_s[2] = {-FLT_MAX, -FLT_MAX};
    float sum_s[2] = {0.0f, 0.0f};

    // Step u covers keys 32 (u % 4) .. + 31 of selected block u / 4. Blocks
    // are ascending and the query's own block is last, so the first step
    // that starts past the query ends the walk.
    for (int u = 0; u < TOPK * 4; ++u) {
      const int bid = blk[u >> 2];
      if (bid < 0) {
        break;
      }
      const int base = bid * BS + 32 * (u & 3);
      if (base > last) {
        break;
      }
      // Keys past the last visible one are masked; their rows re-read it so
      // P = 0 never meets an unwritten row (0 * NaN = NaN).
      uint krow[4];
      UNROLL for (short r = 0; r < 4; ++r) {
        krow[r] = uint(min(base + 16 * (r >> 1) + fm + 8 * (r & 1), last));
      }

      // S = Q K^T over this half of D: one 16x32x32 op per 32 head dims.
      auto s_ct = qk_op.template get_destination_cooperative_tensor<qa_t, kb_t, float>();
      UNROLL for (short i = 0; i < 16; ++i) {
        s_ct[i] = 0.0f;
      }
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        const vec<bfloat, 8> qa = *(const device vec<bfloat, 8>*)(qp0 + 32 * jj);
        const vec<bfloat, 8> qb = *(const device vec<bfloat, 8>*)(qp1 + 32 * jj);
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

      // Scale and mask: this lane's columns of fragment f are keys
      // base + 16 f + fn .. + 3.
      UNROLL for (short f = 0; f < 2; ++f) {
        UNROLL for (short j = 0; j < 4; ++j) {
          const bool ok = base + 16 * f + fn + j <= last;
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
          }
        }
      }
      // O += P V over this half of D (V fragments are [16 keys x 16 dims]),
      // P as its fp16 hi and lo pieces packed along K.
      UNROLL for (short f = 0; f < 2; ++f) {
        vec<half, 8> hi, lo;
        UNROLL for (short i = 0; i < 8; ++i) {
          hi[i] = half(s[f][i]);
          lo[i] = half(s[f][i] - float(hi[i]));
        }
        UNROLL for (short id = 0; id < TDH; id += 2) {
          const vec<bfloat, 8> ra = *(const device vec<bfloat, 8>*)(vb + (krow[2 * f] * svl + 16 * id));
          const vec<bfloat, 8> rb = *(const device vec<bfloat, 8>*)(vb + (krow[2 * f + 1] * svl + 16 * id));
          UNROLL for (short j = 0; j < 4; ++j) {
            vb_ct[j] = ra[j];
            vb_ct[4 + j] = rb[j];
            vb_ct[8 + j] = ra[4 + j];
            vb_ct[12 + j] = rb[4 + j];
            vb_ct[16 + j] = ra[j];
            vb_ct[20 + j] = rb[j];
            vb_ct[24 + j] = ra[4 + j];
            vb_ct[28 + j] = rb[4 + j];
          }
          UNROLL for (short i = 0; i < 8; ++i) {
            pa_ct[i] = hi[i];
            pa_ct[8 + i] = lo[i];
          }
          if (id == 0) {
            pv_op.run(pa_ct, vb_ct, of0);
          } else {
            pv_op.run(pa_ct, vb_ct, of1);
          }
        }
      }
    }

    UNROLL for (short i = 0; i < 2; ++i) {
      const float rr = sum_s[i] > 0.0f ? 1.0f / sum_s[i] : 0.0f;
      device bfloat* o = (device bfloat*)out + (size_t(tq) * HQ + (i == 0 ? h0 : h1)) * D + dh * 64 + fcol;
      UNROLL for (short jj = 0; jj < TDH / 2; ++jj) {
        vec<bfloat, 8> w;
        UNROLL for (short j = 0; j < 4; ++j) {
          float e0, e1;
          if (jj == 0) {
            e0 = of0[4 * i + j];
            e1 = of0[8 + 4 * i + j];
          } else {
            e0 = of1[4 * i + j];
            e1 = of1[8 + 4 * i + j];
          }
          w[j] = bfloat(e0 * rr);
          w[4 + j] = bfloat(e1 * rr);
        }
        *(device vec<bfloat, 8>*)(o + 32 * jj) = w;
      }
    }
"""


@functools.lru_cache(maxsize=None)
def nax_available() -> bool:
    """The kernels are built for the tensor units; other GPUs keep the native ones."""
    return bool(is_nax_available())


@functools.lru_cache(maxsize=None)
def _scores_kernel():
    return mx.fast.metal_kernel(
        name="omlx_minimax_msa_block_scores_nax",
        input_names=["q", "k", "params", "sc"],
        output_names=["out"],
        source=_SCORES_SOURCE,
        header=_HEADER,
        ensure_row_contiguous=False,
    )


@functools.lru_cache(maxsize=None)
def _attn_kernel():
    return mx.fast.metal_kernel(
        name="omlx_minimax_msa_attention_nax",
        input_names=["q", "k", "v", "sel", "params", "scale"],
        output_names=["out"],
        source=_ATTN_SOURCE,
        header=_HEADER,
        ensure_row_contiguous=False,
    )


def _rows_aligned(x: mx.array) -> bool:
    """bf16 with a 128-wide head dim (see the module doc for its layout)."""
    return x.dtype == mx.bfloat16 and x.shape[-1] == HEAD_DIM


def block_scores(
    idx_queries: mx.array, idx_keys: mx.array, q_start: int, scale: float
) -> Optional[mx.array]:
    """Masked block max scores ``[1, H, L, num_blocks]`` (fp32, -inf where no
    key of the block is visible) for ``idx_queries`` [1, H, L, 128] at
    absolute rows ``q_start ..`` and ``idx_keys`` [1, 1, S, 128]; None when
    unsupported."""
    if not nax_available():
        return None
    if idx_queries.ndim != 4 or idx_keys.ndim != 4 or idx_queries.shape[0] != 1:
        return None
    if idx_keys.shape[:2] != (1, 1) or not (
        _rows_aligned(idx_queries) and _rows_aligned(idx_keys)
    ):
        return None
    _, heads, length, _ = idx_queries.shape
    total = idx_keys.shape[2]
    num_blocks = (total + BLOCK - 1) // BLOCK
    params = mx.array([length, total, q_start, heads, num_blocks], dtype=mx.int32)
    (out,) = _scores_kernel()(
        inputs=[idx_queries, idx_keys, params, mx.array([scale], dtype=mx.float32)],
        grid=(num_blocks * 256, (length + 63) // 64, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, heads, length, num_blocks)],
        output_dtypes=[mx.float32],
    )
    return out


def sparse_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    q2k: mx.array,
    *,
    q_start: int,
    scale: float,
) -> Optional[mx.array]:
    """MSA prefill attention for ``queries`` [1, 64, L, 128] at absolute rows
    ``q_start ..`` over ``keys``/``values`` [1, 4, S, 128] and the ascending,
    -1 padded block lists ``q2k`` [4, L, 16]. Keys at or past S are never
    attended. Returns [1, L, 64, 128], or None when unsupported."""
    if not nax_available():
        return None
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None
    _, heads, length, _ = queries.shape
    kv_heads = keys.shape[1]
    if (
        queries.shape[0] != 1
        or heads != kv_heads * GQA
        or values.shape != keys.shape
        or q2k.shape != (kv_heads, length, TOPK)
        or not all(_rows_aligned(x) for x in (queries, keys, values))
    ):
        return None
    sel = mx.contiguous(q2k.astype(mx.int32))
    (out,) = _attn_kernel()(
        inputs=[
            queries,
            keys,
            values,
            sel,
            mx.array([q_start, length, keys.shape[2]], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("GQA", GQA),
            ("TOPK", TOPK),
            ("HQ", heads),
        ],
        grid=(length * 64, kv_heads, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, length, heads, HEAD_DIM)],
        output_dtypes=[queries.dtype],
    )
    return out
