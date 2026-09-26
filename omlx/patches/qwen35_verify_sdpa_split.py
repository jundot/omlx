# SPDX-License-Identifier: Apache-2.0
#
# The tensor-op verify attention (``_gqa_causal_sdpa``) adapts the tile design
# of Splash's runtime/metal/kernels/common/paged_attention_tile.h
# (Apache-2.0, https://github.com/incoai/splash).
"""Collapse Qwen3.5/3.6 verify-width attention into causal vector-kernel calls.

mlx-vlm's ``Qwen3_5Attention`` serves a target-verify forward (q_len =
1 + draft depth) on a batch-1 dense cache with a per-row fallback: L
single-row SDPA calls over per-row K/V slices, concatenated. That costs L
dispatches plus 2L slice ops and a concat per attention layer per verify
cycle, and it is the reason deeper MTP chains stop paying on this family.

MLX's SDPA vector kernel serves causal blocks up to ``q_len * gqa <= 32``
(rows <= 5 on the dense 24/4 layout, <= 4 on the 16/2 MoE one), with
bottom-right alignment — exactly the per-row causal windows the loop
reproduces by hand. So a verify block of L rows needs only ceil(L / limit)
dispatches: rows are chunked at the vector-kernel row limit, each chunk c
covering rows [c0, c1) against ``keys[: kv_len - (L - c1)]`` with
``mask="causal"``. (Same construction as mlx-serve's ``splitCausalSdpa``,
measured +4..9% decode there with speculation on.)

The seam is ``_qwen3_5_left_padded_attention``: it runs FIRST in the
target-verify branch and its non-None result skips the row loop, while a
None keeps every existing path unchanged. This patch wraps it to claim the
batch-1 / dense-cache / head_dim-256 shape and delegate everything else
(left-padded batches, quantized caches) to the original.

Blocks of up to eight rows use ``_gqa_causal_sdpa`` where Metal 4 tensor
ops are available: one threadgroup per KV head and key split holds the rows
of all its query heads, so each key block is read once per group. Otherwise
blocks of four to eight rows use ``_wide_causal_sdpa``: one threadgroup per
query head and key split holds all rows with simdgroup matrices. Both merge
the splits in a second launch, and probabilities enter the value product in
bf16.

MTP draft chains attend through ``ChainKVCache``: chain rows stay in a short
tail next to the head's cache, and ``prefix_tail_attention`` reads both.
"""

from __future__ import annotations

import logging
import struct

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_ENGAGED_LOGGED: set[int] = set()

# MLX fast::ScaledDotProductAttention routes to the vector kernel only while
# q_len * gqa_factor <= 32; wider blocks fall to the composed unfused path.
_VECTOR_ROW_BUDGET = 32


def _log_engaged(q_len: int) -> None:
    # One log per width; a single one-shot log would only witness the first
    # width and hide whether deeper chains route.
    if q_len not in _ENGAGED_LOGGED:
        _ENGAGED_LOGGED.add(q_len)
        logger.info(
            "[verify-split] hd-256 causal vector attention engaged (q_len=%d)",
            q_len,
        )


def _chunked_causal_sdpa(queries, keys, values, scale, limit: int):
    q_len = queries.shape[-2]
    kv_len = keys.shape[-2]
    outs = []
    c0 = 0
    while c0 < q_len:
        c1 = min(c0 + limit, q_len)
        kv_end = kv_len - (q_len - c1)
        outs.append(
            mx.fast.scaled_dot_product_attention(
                queries[..., c0:c1, :],
                keys[..., :kv_end, :],
                values[..., :kv_end, :],
                scale=scale,
                mask="causal",
            )
        )
        c0 = c1
    if len(outs) == 1:
        return outs[0]
    return mx.concatenate(outs, axis=-2)


# Verify blocks wider than the vector kernel's row budget: one threadgroup per
# query head and key split holds all eight rows. Its four simdgroups each own
# 64 head dims, so partial scores meet in threadgroup memory once per 32-key
# block; a second kernel merges the splits.
_WIDE_PARTIAL = """
    constexpr int D = 256;
    constexpr int BK = 32;
    constexpr int SS = 36;   // padded partial-S row stride (floats)
    constexpr int PS = 40;   // padded P row stride (bf16)
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    // Heads vary fastest so the query heads of one KV head read a split together.
    int qh = int(threadgroup_position_in_grid.x);
    int split = int(threadgroup_position_in_grid.y);
    int h = qh / G;
    int T = int(params[0]);
    int L = int(params[1]);
    int chunk = int(params[2]);
    float scale = as_type<float>(params[4]);
    int t_begin = split * chunk;
    int t_end = min(t_begin + chunk, T);

    threadgroup float Sp[4 * 8 * SS];
    threadgroup T_ P[8 * PS];
    threadgroup float st_m[8];
    threadgroup float st_l[8];
    threadgroup float st_a[8];

    // This simdgroup's 64-dim slice of the eight query rows, kept in registers.
    const device T_* qp = q + qh * 8 * D + int(sg) * 64;
    simdgroup_matrix<T_, 8, 8> qa[8];
    for (int dk = 0; dk < 8; ++dk)
        simdgroup_load(qa[dk], qp + dk * 8, D);
    const device T_* kh = k + h * k_strides[1] + int(sg) * 64;
    const device T_* vh = v + h * v_strides[1] + int(sg) * 64;
    int kst = int(k_strides[2]);
    int vst = int(v_strides[2]);

    simdgroup_matrix<float, 8, 8> o[8];
    for (int j = 0; j < 8; ++j) o[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    if (tid < 8) {
        st_m[tid] = -INFINITY;
        st_l[tid] = 0.0f;
    }
    short qid = short(lane / 4);
    short fm = (qid & 4) + short((lane / 2) % 4);

    for (int t0 = t_begin; t0 < t_end; t0 += BK) {
        // Partial scores over this simdgroup's dims for four 8-key tiles.
        for (int c = 0; c < 4; ++c) {
            int ts = min(t0 + c * 8, T - 8);
            simdgroup_matrix<float, 8, 8> s = simdgroup_matrix<float, 8, 8>(0.0f);
            for (int dk = 0; dk < 8; ++dk) {
                simdgroup_matrix<T_, 8, 8> kb;
                simdgroup_load(kb, kh + ts * kst + dk * 8, kst, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qa[dk], kb, s);
            }
            simdgroup_store(s, Sp + int(sg) * 8 * SS + c * 8, SS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: 16 threads per row, two columns each.
        {
            int r = int(tid) / 16;
            int cp = (int(tid) % 16) * 2;
            float sv[2];
            float rmax = -INFINITY;
            for (int e = 0; e < 2; ++e) {
                int col = cp + e;
                int c = col / 8;
                int base = t0 + c * 8;
                int key = min(base, T - 8) + (col - c * 8);
                bool ok = r < L && key >= base && key < t_end && key <= T - L + r;
                float val = Sp[r * SS + col] + Sp[8 * SS + r * SS + col]
                    + Sp[16 * SS + r * SS + col] + Sp[24 * SS + r * SS + col];
                sv[e] = ok ? val * scale : -INFINITY;
                rmax = max(rmax, sv[e]);
            }
            for (int off = 1; off < 16; off <<= 1)
                rmax = max(rmax, simd_shuffle_xor(rmax, ushort(off)));
            float m_old = st_m[r];
            float m_new = max(m_old, rmax);
            float rsum = 0.0f;
            for (int e = 0; e < 2; ++e) {
                float p = m_new == -INFINITY ? 0.0f : exp(sv[e] - m_new);
                rsum += p;
                P[r * PS + cp + e] = T_(p);
            }
            for (int off = 1; off < 16; off <<= 1)
                rsum += simd_shuffle_xor(rsum, ushort(off));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if ((tid % 16) == 0) {
                float alpha = m_new == -INFINITY ? 1.0f : exp(m_old - m_new);
                st_a[r] = alpha;
                st_l[r] = st_l[r] * alpha + rsum;
                st_m[r] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float alpha = st_a[fm];
        for (int j = 0; j < 8; ++j) {
            o[j].thread_elements()[0] *= alpha;
            o[j].thread_elements()[1] *= alpha;
        }
        for (int kt = 0; kt < 4; ++kt) {
            simdgroup_matrix<T_, 8, 8> pa;
            simdgroup_load(pa, P + kt * 8, PS);
            int ts = min(t0 + kt * 8, T - 8);
            // Issue the tile's value loads before the products that use them.
            simdgroup_matrix<T_, 8, 8> vb[8];
            for (int j = 0; j < 8; ++j)
                simdgroup_load(vb[j], vh + ts * vst + j * 8, vst);
            for (int j = 0; j < 8; ++j)
                simdgroup_multiply_accumulate(o[j], pa, vb[j], o[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* ob = o_part + ((split * H + qh) * 8) * D + int(sg) * 64;
    for (int j = 0; j < 8; ++j)
        simdgroup_store(o[j], ob + j * 8, D);
    if (tid < 8) {
        ml_part[((split * H + qh) * 8 + int(tid)) * 2] = st_m[tid];
        ml_part[((split * H + qh) * 8 + int(tid)) * 2 + 1] = st_l[tid];
    }
"""

_WIDE_COMBINE = """
    constexpr int D = 256;
    uint d = thread_position_in_threadgroup.x;
    int r = int(threadgroup_position_in_grid.x);
    int qh = int(threadgroup_position_in_grid.y);
    int L = int(params[1]);
    int n_splits = int(params[3]);
    float M = -INFINITY;
    for (int s = 0; s < n_splits; ++s)
        M = max(M, ml_part[((s * H + qh) * 8 + r) * 2]);
    float total = 0.0f, acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        int idx = (s * H + qh) * 8 + r;
        float m = ml_part[idx * 2];
        if (m == -INFINITY) continue;
        float w = exp(m - M);
        total += w * ml_part[idx * 2 + 1];
        acc += w * o_part[idx * D + d];
    }
    out[(qh * L + r) * D + d] = T_(acc / total);
"""

_WIDE_KERNELS: dict = {}
_WIDE_MAX_ROWS = 8
_WIDE_MIN_ROWS = 4


def _wide_kernels():
    if not _WIDE_KERNELS:
        _WIDE_KERNELS["partial"] = mx.fast.metal_kernel(
            name="omlx_verify_attn_wide_partial",
            input_names=["q", "k", "v", "params"],
            output_names=["o_part", "ml_part"],
            source=_WIDE_PARTIAL,
            # Keys and values are strided cache views; the kernel reads their
            # strides instead of copying them.
            ensure_row_contiguous=False,
        )
        _WIDE_KERNELS["combine"] = mx.fast.metal_kernel(
            name="omlx_verify_attn_wide_combine",
            input_names=["o_part", "ml_part", "params"],
            output_names=["out"],
            source=_WIDE_COMBINE,
        )
    return _WIDE_KERNELS


def _wide_causal_sdpa(queries, keys, values, scale):
    """Causal verify attention for up to eight rows at head_dim 256."""
    _, heads, q_len, dim = queries.shape
    kv_heads, kv_len = keys.shape[1], keys.shape[2]
    rows = queries[0]
    if q_len < _WIDE_MAX_ROWS:
        pad = mx.zeros((heads, _WIDE_MAX_ROWS - q_len, dim), dtype=queries.dtype)
        rows = mx.concatenate([rows, pad], axis=1)
    rows = mx.contiguous(rows)
    # About sixteen key splits per head, measured best on M3 Ultra up to 32k
    # keys. Longer caches miss the system cache, where short splits keep the
    # query heads of a KV head on the same keys (-20% at 64k).
    chunk = min(2048, max(512, (-(-kv_len // 16) + 31) // 32 * 32))
    if kv_len > 32768:
        chunk = 512
    n_splits = -(-kv_len // chunk)
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    params = mx.array([kv_len, q_len, chunk, n_splits, scale_bits], dtype=mx.uint32)
    kernels = _wide_kernels()
    template = [("T_", queries.dtype), ("G", heads // kv_heads), ("H", heads)]
    o_part, ml_part = kernels["partial"](
        inputs=[rows, keys, values, params],
        template=template,
        grid=(heads * 128, n_splits, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[
            (n_splits, heads, _WIDE_MAX_ROWS, dim),
            (n_splits, heads, _WIDE_MAX_ROWS, 2),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    (out,) = kernels["combine"](
        inputs=[o_part, ml_part, params],
        template=template,
        grid=(q_len * 256, heads, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, heads, q_len, dim)],
        output_dtypes=[queries.dtype],
    )
    return out


# Verify blocks through Metal 4 tensor ops: one threadgroup per KV head and
# key split holds the rows of every query head in its group (M = 8 x G), so
# each 64-key block is read once for the whole group.
_GQA_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

_GQA_PARTIAL = """
    constexpr int D = 256;
    constexpr int N = 64;
    constexpr int KPL = N / 4;
    constexpr int M = 8 * G;
    uint tid = thread_position_in_threadgroup.x;
    int h = int(threadgroup_position_in_grid.x);
    int split = int(threadgroup_position_in_grid.y);
    int T = int(params[0]);
    int L = int(params[1]);
    int chunk = int(params[2]);
    float scale = as_type<float>(params[4]);
    int t_begin = split * chunk;
    int t_end = min(t_begin + chunk, T);

    threadgroup float scores[M * N];
    threadgroup T_ probs[M * N];
    threadgroup float row_max[M];
    threadgroup float row_sum[M];
    threadgroup float row_scale[M];
    if (tid < uint(M)) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto qt = tensor(
        const_cast<device T_*>(q) + h * M * D,
        dextents<int, 2>{D, M},
        array<int, 2>{1, D});
    auto st = tensor((threadgroup float*)scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto pt = tensor((threadgroup T_*)probs, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    auto q0 = qt.template slice<D, M>(0, 0);
    auto p0 = pt.template slice<N, M>(0, 0);
    int kst = int(k_strides[2]);
    int vst = int(v_strides[2]);
    device T_* kh = const_cast<device T_*>(k) + h * k_strides[1];
    device T_* vh = const_cast<device T_*>(v) + h * v_strides[1];
    constexpr auto qk_desc = matmul2d_descriptor(
        M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
    constexpr auto pv_desc = matmul2d_descriptor(
        M, D, N, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<qk_desc, execution_simdgroups<8>> qk;
    matmul2d<pv_desc, execution_simdgroups<8>> pv;
    auto v_first = tensor(vh, dextents<int, 2>{D, N}, array<int, 2>{1, vst}).template slice<D, N>(0, 0);
    auto running = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(v_first), float>();
    for (ushort i = 0; i < running.get_capacity(); ++i)
        if (running.is_valid_element(i))
            running[i] = 0.0f;

    // Fused row f is query row f / G of query head h * G + f % G; four
    // threads own a row, KPL keys each.
    int f = int(tid) / 4;
    int col = (int(tid) % 4) * KPL;
    for (int t0 = t_begin; t0 < t_end; t0 += N) {
        int ts = min(t0, T - N);
        auto ks = tensor(kh + ts * kst, dextents<int, 2>{D, N}, array<int, 2>{1, kst})
            .template slice<D, N>(0, 0);
        auto vs = tensor(vh + ts * vst, dextents<int, 2>{D, N}, array<int, 2>{1, vst})
            .template slice<D, N>(0, 0);
        auto sc = qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(ks), float>();
        qk.run(q0, ks, sc);
        sc.store(st.template slice<N, M>(0, 0));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (f < M) {
            int r = f / G;
            float sv[KPL];
            float lmax = -INFINITY;
            for (int j = 0; j < KPL; ++j) {
                int key = ts + col + j;
                bool ok = r < L && key >= t0 && key < t_end && key <= T - L + r;
                sv[j] = ok ? scores[f * N + col + j] * scale : -INFINITY;
                lmax = max(lmax, sv[j]);
            }
            lmax = max(lmax, simd_shuffle_xor(lmax, ushort(1)));
            lmax = max(lmax, simd_shuffle_xor(lmax, ushort(2)));
            float pmax = row_max[f];
            float nmax = max(pmax, lmax);
            float lsum = 0.0f;
            for (int j = 0; j < KPL; ++j) {
                float p = nmax == -INFINITY ? 0.0f : exp(sv[j] - nmax);
                lsum += p;
                probs[f * N + col + j] = T_(p);
            }
            lsum += simd_shuffle_xor(lsum, ushort(1));
            lsum += simd_shuffle_xor(lsum, ushort(2));
            if (col == 0) {
                float a = (nmax == -INFINITY || nmax == pmax) ? 1.0f : exp(pmax - nmax);
                row_scale[f] = a;
                row_sum[f] = row_sum[f] * a + lsum;
                row_max[f] = nmax;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            if (!running.is_valid_element(i))
                continue;
            auto c = running.get_multidimensional_index(i);
            running[i] *= row_scale[c[1]];
        }
        pv.run(p0, vs, running);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto target = tensor(
        o_part + (split * KVH + h) * M * D,
        dextents<int, 2>{D, M},
        array<int, 2>{1, D});
    running.store(target.template slice<D, M>(0, 0));
    if (tid < uint(M)) {
        ml_part[((split * KVH + h) * M + int(tid)) * 2] = row_max[tid];
        ml_part[((split * KVH + h) * M + int(tid)) * 2 + 1] = row_sum[tid];
    }
"""

_GQA_COMBINE = """
    constexpr int D = 256;
    constexpr int M = 8 * G;
    uint d = thread_position_in_threadgroup.x;
    int f = int(threadgroup_position_in_grid.y);
    int h = int(threadgroup_position_in_grid.z);
    int L = int(params[1]);
    int n = int(params[3]);
    int r = f / G;
    if (r >= L)
        return;
    float Mx = -INFINITY;
    for (int s = 0; s < n; ++s)
        Mx = max(Mx, ml_part[((s * KVH + h) * M + f) * 2]);
    float total = 0.0f, acc = 0.0f;
    for (int s = 0; s < n; ++s) {
        int idx = (s * KVH + h) * M + f;
        float m = ml_part[idx * 2];
        if (m == -INFINITY)
            continue;
        float w = exp(m - Mx);
        total += w * ml_part[idx * 2 + 1];
        acc += w * o_part[idx * D + d];
    }
    out[((h * G + f % G) * L + r) * D + d] = T_(acc / total);
"""

_GQA_KERNELS: dict = {}
_GQA_MAX_GROUP = 8
_GQA_MIN_KEYS = 64


def _gqa_causal_sdpa(queries, keys, values, scale):
    """Causal verify attention for up to eight rows, one pass per KV head."""
    _, heads, q_len, dim = queries.shape
    kv_heads, kv_len = keys.shape[1], keys.shape[2]
    group = heads // kv_heads
    rows = queries[0]
    if q_len < _WIDE_MAX_ROWS:
        pad = mx.zeros((heads, _WIDE_MAX_ROWS - q_len, dim), dtype=queries.dtype)
        rows = mx.concatenate([rows, pad], axis=1)
    # Rows of one KV head's query heads, [KV head][row][head in group][dim].
    fused = mx.contiguous(
        rows.reshape(kv_heads, group, _WIDE_MAX_ROWS, dim).transpose(0, 2, 1, 3)
    )
    # Short splits beyond 16k keys keep a KV head's blocks in cache between
    # splits; measured best on M3 Ultra from 2k to 64k keys.
    if kv_len > 16384:
        chunk = 512
    else:
        chunk = max(256, (-(-kv_len // 16) + 63) // 64 * 64)
    n_splits = -(-kv_len // chunk)
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    params = mx.array([kv_len, q_len, chunk, n_splits, scale_bits], dtype=mx.uint32)
    if not _GQA_KERNELS:
        _GQA_KERNELS["partial"] = mx.fast.metal_kernel(
            name="omlx_verify_attn_gqa_partial",
            input_names=["q", "k", "v", "params"],
            output_names=["o_part", "ml_part"],
            source=_GQA_PARTIAL,
            header=_GQA_HEADER,
            ensure_row_contiguous=False,
        )
        _GQA_KERNELS["combine"] = mx.fast.metal_kernel(
            name="omlx_verify_attn_gqa_combine",
            input_names=["o_part", "ml_part", "params"],
            output_names=["out"],
            source=_GQA_COMBINE,
        )
    template = [("T_", queries.dtype), ("G", group), ("KVH", kv_heads)]
    o_part, ml_part = _GQA_KERNELS["partial"](
        inputs=[fused, keys, values, params],
        template=template,
        grid=(kv_heads * 256, n_splits, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (n_splits, kv_heads, _WIDE_MAX_ROWS * group, dim),
            (n_splits, kv_heads, _WIDE_MAX_ROWS * group, 2),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    (out,) = _GQA_KERNELS["combine"](
        inputs=[o_part, ml_part, params],
        template=template,
        grid=(256, _WIDE_MAX_ROWS * group, kv_heads),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, heads, q_len, dim)],
        output_dtypes=[queries.dtype],
    )
    return out


_GQA_READY = None


def _gqa_ready() -> bool:
    """Whether this OS and GPU compile and run the tensor-op kernel."""
    global _GQA_READY
    if _GQA_READY is None:
        try:
            q = mx.zeros((1, 2, 2, 256), dtype=mx.bfloat16)
            kv = mx.zeros((1, 1, _GQA_MIN_KEYS, 256), dtype=mx.bfloat16)
            mx.eval(_gqa_causal_sdpa(q, kv, kv, 1.0))
            _GQA_READY = True
        except Exception:
            logger.info(
                "Tensor-op verify attention unavailable; using simdgroup kernels"
            )
            _GQA_READY = False
    return _GQA_READY


# One query row over a cache prefix plus a short tail held outside the cache.
# Draft chains keep their rows in the tail: writing them into a cache that
# in-flight steps still read makes MLX copy the whole cache every step.
_DECODE_PARTIAL = """
    constexpr int D = 256;
    constexpr int SGN = 8;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    // Heads vary fastest so the query heads of one KV head share key reads.
    int qh = int(threadgroup_position_in_grid.x);
    int split = int(threadgroup_position_in_grid.y);
    int h = qh / G;
    int P = int(params[0]);
    int Tn = int(params[1]);
    int chunk = int(params[2]);
    int n_splits = int(params[3]);
    float scale = as_type<float>(params[4]);

    float qv[8];
    for (int i = 0; i < 8; ++i)
        qv[i] = float(q[qh * D + lane * 8 + i]) * scale;
    float m = -INFINITY, l = 0.0f;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    const device T_* kb;
    const device T_* vb;
    int kst, vst, begin, end;
    if (split < n_splits) {
        kb = k + h * k_strides[1];
        vb = v + h * v_strides[1];
        kst = int(k_strides[2]);
        vst = int(v_strides[2]);
        begin = split * chunk;
        end = min(begin + chunk, P);
    } else {
        kb = kt + h * Tn * D;
        vb = vt + h * Tn * D;
        kst = D;
        vst = D;
        begin = 0;
        end = Tn;
    }
    for (int t = begin + int(sg); t < end; t += SGN) {
        const device T_* kr = kb + t * kst + lane * 8;
        float s = 0.0f;
        for (int i = 0; i < 8; ++i)
            s += qv[i] * float(kr[i]);
        s = simd_sum(s);
        float m_new = max(m, s);
        float alpha = exp(m - m_new);
        float p = exp(s - m_new);
        l = l * alpha + p;
        const device T_* vr = vb + t * vst + lane * 8;
        for (int i = 0; i < 8; ++i)
            acc[i] = acc[i] * alpha + p * float(vr[i]);
        m = m_new;
    }

    threadgroup float tm[SGN];
    threadgroup float tl[SGN];
    threadgroup float ta[SGN * D];
    if (lane == 0) {
        tm[sg] = m;
        tl[sg] = l;
    }
    for (int i = 0; i < 8; ++i)
        ta[sg * D + lane * 8 + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = -INFINITY;
    for (int j = 0; j < SGN; ++j)
        M = max(M, tm[j]);
    float o = 0.0f, L = 0.0f;
    for (int j = 0; j < SGN; ++j) {
        float w = tm[j] == -INFINITY ? 0.0f : exp(tm[j] - M);
        o += w * ta[j * D + tid];
        L += w * tl[j];
    }
    int idx = split * H + qh;
    o_part[idx * D + tid] = o;
    if (tid == 0) {
        ml_part[idx * 2] = M;
        ml_part[idx * 2 + 1] = L;
    }
"""

_DECODE_COMBINE = """
    constexpr int D = 256;
    uint d = thread_position_in_threadgroup.x;
    int qh = int(threadgroup_position_in_grid.y);
    int n = int(params[3]) + 1;
    float M = -INFINITY;
    for (int s = 0; s < n; ++s)
        M = max(M, ml_part[(s * H + qh) * 2]);
    float total = 0.0f, acc = 0.0f;
    for (int s = 0; s < n; ++s) {
        int idx = s * H + qh;
        float m = ml_part[idx * 2];
        if (m == -INFINITY) continue;
        float w = exp(m - M);
        total += w * ml_part[idx * 2 + 1];
        acc += w * o_part[idx * D + d];
    }
    out[qh * D + d] = T_(acc / total);
"""

# Keys per split; measured best on M3 Ultra from 2k to 64k keys.
_DECODE_CHUNK = 512
_DECODE_KERNELS: dict = {}


def _decode_kernels():
    if not _DECODE_KERNELS:
        _DECODE_KERNELS["partial"] = mx.fast.metal_kernel(
            name="omlx_chain_attn_partial",
            input_names=["q", "k", "v", "kt", "vt", "params"],
            output_names=["o_part", "ml_part"],
            source=_DECODE_PARTIAL,
            ensure_row_contiguous=False,
        )
        _DECODE_KERNELS["combine"] = mx.fast.metal_kernel(
            name="omlx_chain_attn_combine",
            input_names=["o_part", "ml_part", "params"],
            output_names=["out"],
            source=_DECODE_COMBINE,
        )
    return _DECODE_KERNELS


def prefix_tail_attention(queries, keys, values, tail_keys, tail_values, scale):
    """Attention of one query row over ``keys`` followed by ``tail_keys``."""
    _, heads, _, dim = queries.shape
    kv_heads, prefix = keys.shape[1], keys.shape[2]
    tail = tail_keys.shape[2]
    n_splits = -(-prefix // _DECODE_CHUNK)
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    params = mx.array(
        [prefix, tail, _DECODE_CHUNK, n_splits, scale_bits], dtype=mx.uint32
    )
    kernels = _decode_kernels()
    template = [("T_", queries.dtype), ("G", heads // kv_heads), ("H", heads)]
    o_part, ml_part = kernels["partial"](
        inputs=[
            mx.contiguous(queries),
            keys,
            values,
            mx.contiguous(tail_keys),
            mx.contiguous(tail_values),
            params,
        ],
        template=template,
        grid=(heads * 256, n_splits + 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_splits + 1, heads, dim), (n_splits + 1, heads, 2)],
        output_dtypes=[mx.float32, mx.float32],
    )
    (out,) = kernels["combine"](
        inputs=[o_part, ml_part, params],
        template=template,
        grid=(256, heads, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, heads, 1, dim)],
        output_dtypes=[queries.dtype],
    )
    return out


class ChainKVCache:
    """Draft-chain view of a ``KVCache``: new rows stay in a short tail."""

    def __init__(self, base):
        self.base = base
        self.keys = None
        self.values = None

    @property
    def offset(self) -> int:
        tail = 0 if self.keys is None else self.keys.shape[2]
        return self.base.offset + tail

    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values

    def attention(self, queries, scale, mask):
        base = self.base
        keys = base.keys[..., : base.offset, :]
        values = base.values[..., : base.offset, :]
        if (
            mask is None
            and queries.shape[0] == 1
            and queries.shape[2] == 1
            and queries.shape[-1] == 256
            and queries.dtype in (mx.bfloat16, mx.float16)
            and keys.dtype == queries.dtype
            and base.offset > 0
            and mx.default_device().type == mx.gpu
        ):
            return prefix_tail_attention(
                queries, keys, values, self.keys, self.values, scale
            )
        return mx.fast.scaled_dot_product_attention(
            queries,
            mx.concatenate([keys, self.keys], axis=2),
            mx.concatenate([values, self.values], axis=2),
            scale=scale,
            mask=mask,
        )


def install_chain_attention(attention_cls) -> bool:
    """Route ``ChainKVCache`` calls of this attention class to the tail path.

    The class must call its module's ``scaled_dot_product_attention`` with
    the cache after ``update_and_fetch``, as the Qwen3.5 attentions do.
    """
    import sys

    module = sys.modules.get(getattr(attention_cls, "__module__", ""))
    original = getattr(module, "scaled_dot_product_attention", None)
    if original is None:
        return False
    if getattr(original, "_omlx_chain_attention", False):
        return True

    def scaled_dot_product_attention(
        queries, keys, values, cache=None, scale=1.0, mask=None, **kwargs
    ):
        if isinstance(cache, ChainKVCache):
            return cache.attention(queries, scale, mask)
        return original(
            queries, keys, values, cache=cache, scale=scale, mask=mask, **kwargs
        )

    scaled_dot_product_attention._omlx_chain_attention = True
    module.scaled_dot_product_attention = scaled_dot_product_attention
    return True


def _eligible(queries, keys, cache) -> int:
    """Return the vector-kernel row limit (>0) when this call is ours."""
    # A turboquant-quantized cache hands back `_QuantizedStateProxy` objects
    # (exposes .shape, deliberately not .ndim, to avoid dequantizing — see
    # mlx_vlm.turboquant). That's exactly the "quantized caches" case this
    # patch already means to delegate to the original path below, but the
    # attribute access below used to run before the `hasattr(cache, "bits")`
    # guard could rule it out, so it crashed instead of falling through.
    if getattr(queries, "ndim", None) != 4 or getattr(keys, "ndim", None) != 4:
        return 0
    if queries.shape[0] != 1:
        return 0
    if cache is not None and hasattr(cache, "bits"):
        return 0
    if queries.dtype not in (mx.float16, mx.bfloat16):
        return 0
    if queries.dtype != keys.dtype:
        return 0
    if queries.shape[-1] != 256 or keys.shape[-1] != 256:
        return 0
    q_heads = queries.shape[-3]
    kv_heads = keys.shape[-3]
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        return 0
    limit = _VECTOR_ROW_BUDGET // (q_heads // kv_heads)
    if limit <= 0:
        return 0
    q_len = queries.shape[-2]
    if q_len <= 1 or q_len > keys.shape[-2]:
        return 0
    return limit


def apply_qwen35_verify_sdpa_split_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if not mx.metal.is_available():
        return False

    try:
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except ImportError:
        return False

    original = getattr(q35_lang, "_qwen3_5_left_padded_attention", None)
    if original is None:
        logger.debug("verify-split: target-verify seam not found; patch skipped")
        return False

    def patched_target_verify_attention(
        queries,
        keys,
        values,
        *,
        cache,
        scale,
        mask,
    ):
        # Only the batch-1 dense-cache shape is ours; a batch with real left
        # padding (or anything unexpected) keeps the original behavior.
        if mask is None or (isinstance(mask, str) and mask == "causal"):
            limit = _eligible(queries, keys, cache)
            if limit and getattr(cache, "left_padding", None) is None:
                try:
                    q_len = queries.shape[-2]
                    wide_from = min(limit + 1, _WIDE_MIN_ROWS)
                    if (
                        q_len <= _WIDE_MAX_ROWS
                        and keys.shape[-2] >= _GQA_MIN_KEYS
                        and queries.shape[1] // keys.shape[1] <= _GQA_MAX_GROUP
                        and _gqa_ready()
                    ):
                        out = _gqa_causal_sdpa(queries, keys, values, scale)
                    elif wide_from <= q_len <= _WIDE_MAX_ROWS and keys.shape[-2] >= 8:
                        out = _wide_causal_sdpa(queries, keys, values, scale)
                    else:
                        out = _chunked_causal_sdpa(queries, keys, values, scale, limit)
                    _log_engaged(queries.shape[-2])
                    return out
                except Exception:
                    logger.warning(
                        "verify-split attention failed; falling back",
                        exc_info=True,
                    )
        return original(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    q35_lang._qwen3_5_left_padded_attention = patched_target_verify_attention
    _PATCHED = True
    logger.info("Qwen3.5/3.6 verify-width causal vector attention patch applied")
    return True
