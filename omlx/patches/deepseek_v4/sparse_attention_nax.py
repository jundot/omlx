# SPDX-License-Identifier: Apache-2.0
"""Tensor-unit (NAX) window + top-k pooled attention for DeepSeek V4 prefill.

Same math as the ``dsv4_wsdpa_topk`` kernel (``wsdpa_attention.py``): query
``p`` attends its sliding-window rows ``(p - window, p]`` of the local KV,
then its indexer-selected pooled rows ``i < (p + 1) // ratio``, with fp32
scores, an fp32 online softmax and a per-head sink in the denominator.

The layout is the GLM-5.3 sparse MLA tensor-unit kernel's
(``patches/glm_moe_dsa/sparse_mla_nax.py``): one threadgroup per (query,
32-head half), 8 simdgroups, 128-slot tiles; Q K^T and P V run on the tensor
units with fp32 accumulation, and the fp32 probabilities enter P V as two
fp16 pieces (hi + lo). Added for V4: the slot list is the window rows
(computed from the offset, no index array) followed by the top-k rows, each
slot reads from its own KV source, and the sink joins the final denominator.

Versus ``dsv4_wsdpa_topk`` only the fp32 summation order differs (and the
running maximum advances per 128-slot tile instead of per row), so outputs
agree to bf16 rounding. A top-k row counts when ``idx < (p + 1) // ratio``;
with the indexer's temporally sorted rows that is the native kernel's
break-at-first-invisible rule.

DeepSeek V4.1 reuses the kernel on its decoded KV rows with two template
switches: ``QT`` reads the query token-major ``[L, H, 512]`` instead of
``[H, L, 512]``, and ``PLO = 0`` drops the low fp16 piece of P, since V4.1's
reference already rounds P to bf16 before P V.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_D = 512
_HEADS_PER_GROUP = 32
# Short verify blocks stay on the native kernel.
MIN_ROWS = 16

# Per (QT, PLO) instantiation: compiled and run once, or failed.
_BROKEN: set = set()
_READY: set = set()

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

# Fragment layout (MLX BaseNAXFrag, 16x16): lane -> rows fm, fm + 8 and
# columns fn .. fn + 3. Lanes sharing a row differ in lane bits 0 and 3.
_SOURCE = """
    constexpr int D = 512;
    constexpr int BK = 128;
    const int offset = params[0];
    const int W = params[1];
    const int ratio = params[2];
    const int NP = params[3];
    const int NS = params[4];
    const int L = params[5];
    const int K = params[6];
    const int H = params[7];
    const int SLOTS = W + K;
    const float scale = scalep[0];
    const int qi = int(threadgroup_position_in_grid.x);
    const int hh = int(threadgroup_position_in_grid.y);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tid = sg * 32 + lane;
    const int hg = int(sg) / 4;
    const int j4 = int(sg) % 4;   // QK + softmax: slot quarter; PV: dim quarter
    const int head0 = hh * 32 + hg * 16;

    // Visible rows of this query: local rows j0 .. j0 + n_local - 1 of the
    // KV buffer (row 0 is absolute position base), pooled rows < pvis.
    const int p_abs = offset + qi;
    const int base = offset + L - NS;
    const int j0 = max(base, p_abs - W + 1) - base;
    const int n_local = p_abs - base - j0 + 1;
    const int pvis = min((p_abs + 1) / ratio, NP);

    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);
    // Lanes with fn == 0: one per row (fm) of a fragment after the
    // xor-1 / xor-8 row reductions.
    const bool row_writer = (lane & 9u) == 0u;

    threadgroup int sel[2][BK];
    threadgroup int live[2][BK / 32];
    threadgroup float red_max[2][4][16];
    threadgroup float red_sum[2][4][16];
    // P pieces per (head group, 16-slot step, lane): the 8 values of the
    // lane's fragment slots (rows fm, fm + 8 x slots fn .. fn + 3).
    threadgroup half p_hi[2][BK / 16][32 * 8];
    threadgroup half p_lo[2][BK / 16][32 * 8];

    constexpr auto qk_desc = matmul2d_descriptor(
        16, 32, 16, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroup> qk_op;
    constexpr auto pv_desc = matmul2d_descriptor(
        16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_desc, execution_simdgroup> pv_op;

    auto pa = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    auto pm = pv_op.template get_left_input_cooperative_tensor<half, T, float>();
    auto pb = pv_op.template get_right_input_cooperative_tensor<half, T, float>();
    auto o0 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o1 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o2 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    auto o3 = pv_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(pa)>, metal::remove_addrspace_t<decltype(pb)>, float>();
    for (short e = 0; e < 16; ++e) {
        o0[e] = 0.0f;
        o1[e] = 0.0f;
        o2[e] = 0.0f;
        o3[e] = 0.0f;
    }
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.0f, 0.0f};

    const device T* qr0 = q + (QT ? ulong(qi) * H + head0 + fm
                                  : ulong(head0 + fm) * L + qi) * D + fn;
    const device T* qr1 = q + (QT ? ulong(qi) * H + head0 + fm + 8
                                  : ulong(head0 + fm + 8) * L + qi) * D + fn;

    // Slot s of the query: window row s for s < W, then top-k entry s - W.
    auto stage = [&](int t, int b) {
        if (tid < uint(BK)) {
            const int slot = t * BK + int(tid);
            int row = -1;
            if (slot < W) {
                row = slot < n_local ? j0 + slot : -1;
            } else if (slot < SLOTS) {
                const uint idx = topk[ulong(qi) * K + slot - W];
                row = idx < uint(pvis) ? int(idx) : -1;
            }
            sel[b][tid] = row;
            const bool any_live = simd_any(row >= 0);
            if (lane == 0) {
                live[b][sg] = any_live ? 1 : 0;
            }
        }
    };

    const int n_tiles = (SLOTS + BK - 1) / BK;
    stage(0, 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < n_tiles; ++t) {
        const int buf = t & 1;
        const int tile_slots = min(BK, SLOTS - t * BK);
        // A tile without a visible slot leaves the running state unchanged
        // (uniform across the threadgroup).
        if ((live[buf][0] | live[buf][1] | live[buf][2] | live[buf][3]) == 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t + 1 < n_tiles) {
                stage(t + 1, buf ^ 1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // ---- S = Q K^T for slot quarter j4 (16 heads x 32 slots).
        auto qa = qk_op.template get_left_input_cooperative_tensor<T, T, float>();
        auto kb = qk_op.template get_right_input_cooperative_tensor<T, T, float>();
        auto sc = qk_op.template get_destination_cooperative_tensor<
            metal::remove_addrspace_t<decltype(qa)>, metal::remove_addrspace_t<decltype(kb)>, float>();
        for (short e = 0; e < 16; ++e) {
            sc[e] = 0.0f;
        }
        if (j4 * 32 < tile_slots) {
            const device T* kr[2][2];
            for (short tn = 0; tn < 2; ++tn) {
                for (short i = 0; i < 2; ++i) {
                    const int k = j4 * 32 + tn * 16 + fm + i * 8;
                    const int row = sel[buf][k];
                    const device T* src = t * BK + k < W ? kv : pooled;
                    kr[tn][i] = src + ulong(max(row, 0)) * D + fn;
                }
            }
            _Pragma("clang loop unroll_count(4)")
            for (short kk = 0; kk < D; kk += 16) {
                for (short j = 0; j < 4; ++j) {
                    qa[j] = qr0[kk + j];
                    qa[4 + j] = qr1[kk + j];
                }
                for (short tn = 0; tn < 2; ++tn) {
                    for (short i = 0; i < 2; ++i) {
                        for (short j = 0; j < 4; ++j) {
                            kb[tn * 8 + i * 4 + j] = kr[tn][i][kk + j];
                        }
                    }
                }
                qk_op.run(qa, kb, sc);
            }
        }
        bool ok[2][4];
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                ok[tn][j] = sel[buf][j4 * 32 + tn * 16 + fn + j] >= 0;
            }
        }
        // Partial row max of the raw scores over the quarter (scale > 0 and
        // rounding is monotonic: max(s) * c == max(s * c) bitwise).
        float pmx[2];
        for (short i = 0; i < 2; ++i) {
            float m = -INFINITY;
            for (short tn = 0; tn < 2; ++tn) {
                for (short j = 0; j < 4; ++j) {
                    m = ok[tn][j] ? max(m, float(sc[tn * 8 + i * 4 + j])) : m;
                }
            }
            m = max(m, simd_shuffle_xor(m, ushort(1)));
            m = max(m, simd_shuffle_xor(m, ushort(8)));
            pmx[i] = m;
        }
        if (row_writer) {
            red_max[hg][j4][fm] = pmx[0];
            red_max[hg][j4][fm + 8] = pmx[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t + 1 < n_tiles) {
            stage(t + 1, buf ^ 1);
        }

        // ---- Tile row max (the same value in the 4 simdgroups of the group).
        // The tile has a visible slot, so it is finite.
        float factor[2];
        float mnew[2];
        for (short i = 0; i < 2; ++i) {
            const int r = fm + i * 8;
            const float m = max(max(red_max[hg][0][r], red_max[hg][1][r]),
                                max(red_max[hg][2][r], red_max[hg][3][r]));
            mnew[i] = max(m_run[i], m * scale);
            factor[i] = m_run[i] == -INFINITY ? 0.0f : exp(m_run[i] - mnew[i]);
            m_run[i] = mnew[i];
        }
        // ---- P of this quarter, once: exp, fp16 hi + lo pieces, row sums.
        float rs[2] = {0.0f, 0.0f};
        for (short tn = 0; tn < 2; ++tn) {
            vec<half, 8> h, lo;
            for (short i = 0; i < 2; ++i) {
                for (short j = 0; j < 4; ++j) {
                    const float e = ok[tn][j]
                        ? exp(float(sc[tn * 8 + i * 4 + j]) * scale - mnew[i])
                        : 0.0f;
                    const half hi = half(e);
                    h[i * 4 + j] = hi;
                    lo[i * 4 + j] = PLO ? half(e - float(hi)) : half(0.0f);
                    rs[i] += e;
                }
            }
            const int ks = j4 * 2 + tn;
            *(threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]) = h;
            if (PLO) {
                *(threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]) = lo;
            }
        }
        for (short i = 0; i < 2; ++i) {
            rs[i] += simd_shuffle_xor(rs[i], ushort(1));
            rs[i] += simd_shuffle_xor(rs[i], ushort(8));
        }
        if (row_writer) {
            red_sum[hg][j4][fm] = rs[0];
            red_sum[hg][j4][fm + 8] = rs[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- O = O * factor + P V for dim quarter j4 over the whole tile.
        // (factor == 1 exactly for every row of the simdgroup: O * 1 == O.)
        if (!simd_all(factor[0] == 1.0f && factor[1] == 1.0f)) {
            for (short e = 0; e < 16; ++e) {
                const float f = factor[(e >> 2) & 1];
                o0[e] *= f;
                o1[e] *= f;
                o2[e] *= f;
                o3[e] *= f;
            }
        }
        for (short i = 0; i < 2; ++i) {
            const int r = fm + i * 8;
            const float tsum = ((red_sum[hg][0][r] + red_sum[hg][1][r]) + red_sum[hg][2][r]) + red_sum[hg][3][r];
            l_run[i] = l_run[i] * factor[i] + tsum;
        }
        const int n_ks = (tile_slots + 15) / 16;
        for (short ks = 0; ks < n_ks; ++ks) {
            const vec<half, 8> h = *(const threadgroup vec<half, 8>*)(&p_hi[hg][ks][lane * 8]);
            for (short e = 0; e < 8; ++e) {
                pa[e] = h[e];
            }
            if (PLO) {
                const vec<half, 8> lo = *(const threadgroup vec<half, 8>*)(&p_lo[hg][ks][lane * 8]);
                for (short e = 0; e < 8; ++e) {
                    pm[e] = lo[e];
                }
            }
            const int k0 = ks * 16 + fm;
            const int k1 = ks * 16 + fm + 8;
            const device T* v0 = (t * BK + k0 < W ? kv : pooled)
                + ulong(max(sel[buf][k0], 0)) * D + j4 * 128 + fn;
            const device T* v1 = (t * BK + k1 < W ? kv : pooled)
                + ulong(max(sel[buf][k1], 0)) * D + j4 * 128 + fn;
            for (short np = 0; np < 4; ++np) {
                for (short tn = 0; tn < 2; ++tn) {
                    for (short j = 0; j < 4; ++j) {
                        pb[tn * 8 + j] = v0[np * 32 + tn * 16 + j];
                        pb[tn * 8 + 4 + j] = v1[np * 32 + tn * 16 + j];
                    }
                }
                if (np == 0) {
                    pv_op.run(pa, pb, o0);
                    if (PLO) {
                        pv_op.run(pm, pb, o0);
                    }
                } else if (np == 1) {
                    pv_op.run(pa, pb, o1);
                    if (PLO) {
                        pv_op.run(pm, pb, o1);
                    }
                } else if (np == 2) {
                    pv_op.run(pa, pb, o2);
                    if (PLO) {
                        pv_op.run(pm, pb, o2);
                    }
                } else {
                    pv_op.run(pa, pb, o3);
                    if (PLO) {
                        pv_op.run(pm, pb, o3);
                    }
                }
            }
        }
    }

    for (short i = 0; i < 2; ++i) {
        const int head = head0 + fm + i * 8;
        // With no visible slot m_run is -inf: the sink term is +inf and the
        // row is zero, as in the native kernel.
        const float denom = l_run[i] + exp(float(sinks[head]) - m_run[i]);
        const float inv = denom == 0.0f ? 0.0f : 1.0f / denom;
        device T* orow = out + (ulong(head) * L + qi) * D + j4 * 128 + fn;
        for (short tn = 0; tn < 2; ++tn) {
            for (short j = 0; j < 4; ++j) {
                const short e = tn * 8 + i * 4 + j;
                orow[tn * 16 + j] = T(o0[e] * inv);
                orow[32 + tn * 16 + j] = T(o1[e] * inv);
                orow[64 + tn * 16 + j] = T(o2[e] * inv);
                orow[96 + tn * 16 + j] = T(o3[e] * inv);
            }
        }
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_dsv4_sparse_attention_nax",
            input_names=["q", "kv", "pooled", "topk", "sinks", "params", "scalep"],
            output_names=["out"],
            header=_HEADER,
            source=_SOURCE,
        )
    return _KERNEL


@lru_cache(maxsize=1)
def nax_sparse_attention_available() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


def sparse_topk_attention_nax(
    q: mx.array,
    kv: mx.array,
    pooled: mx.array,
    topk: mx.array,
    sinks: mx.array,
    scale: float,
    offset: int,
    window: int,
    ratio: int,
    *,
    q_token_major: bool = False,
    p_lo: bool = True,
) -> Optional[mx.array]:
    """``wsdpa_topk_prefill`` on the tensor units, or None.

    q: [1, H, L, 512] bf16 with H a multiple of 32 ([1, L, H, 512] with
    ``q_token_major``); kv: [1, 1, S, 512] bf16 local rows (row 0 is absolute
    position ``offset + L - S``); pooled: [1, P, 512] bf16; topk: [1, L, K]
    uint32, temporally sorted; sinks: [H]. ``p_lo=False`` feeds P V a single
    fp16 piece of P. Returns [1, H, L, 512] bf16.
    """
    key = (bool(q_token_major), bool(p_lo))
    if key in _BROKEN or not nax_sparse_attention_available() or q.ndim != 4:
        return None
    heads, length = (q.shape[2], q.shape[1]) if q_token_major else q.shape[1:3]
    if (
        q.shape[0] != 1
        or heads % _HEADS_PER_GROUP
        or q.shape[3] != _D
        or q.dtype != mx.bfloat16
        or length < MIN_ROWS
        or kv.ndim != 4
        or kv.shape[:2] != (1, 1)
        or kv.shape[3] != _D
        or kv.dtype != mx.bfloat16
        or kv.shape[2] == 0
        or pooled is None
        or pooled.ndim != 3
        or pooled.shape[0] != 1
        or pooled.shape[1] == 0
        or pooled.shape[2] != _D
        or pooled.dtype != mx.bfloat16
        or topk.dtype != mx.uint32
        or topk.ndim != 3
        or topk.shape[:2] != (1, length)
        or topk.shape[2] == 0
        or sinks.shape != (heads,)
        or window < 1
        or ratio < 1
        or offset < 0
        or not scale > 0
    ):
        return None
    try:
        params = mx.array(
            [
                offset,
                window,
                ratio,
                pooled.shape[1],
                kv.shape[2],
                length,
                topk.shape[2],
                heads,
            ],
            dtype=mx.int32,
        )
        out = _kernel()(
            inputs=[
                mx.contiguous(q[0]),
                mx.contiguous(kv[0, 0]),
                mx.contiguous(pooled[0]),
                mx.contiguous(topk[0]),
                sinks.astype(mx.float32),
                params,
                mx.array([scale], dtype=mx.float32),
            ],
            template=[("T", mx.bfloat16), ("QT", key[0]), ("PLO", key[1])],
            grid=(length * 256, heads // _HEADS_PER_GROUP, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(heads, length, _D)],
            output_dtypes=[mx.bfloat16],
        )[0]
        if key not in _READY:
            # Compile and run once here so a kernel failure falls back.
            mx.eval(out)
            _READY.add(key)
    except Exception:  # noqa: BLE001
        _BROKEN.add(key)
        logger.warning(
            "DSV4 tensor-unit sparse attention kernel failed; using wsdpa topk",
            exc_info=True,
        )
        return None
    return out[None]
