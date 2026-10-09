# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support TurboQuantKVCache.

When TurboQuantKVCache is detected, routes attention to:
  - One-sequence MSE decode and causal verify (L <= 15): a simdgroup-matrix
    2-pass kernel that serves every GQA repeat and verify row of a KV head
    from one unpack of each token (``_mma_mse_attention``)
  - Other decode (L=1): cache.decode_attention() — Metal kernel, no dequant
  - Other decode-shaped multi-row (1 < L <= 15, causal; MTP verify): a fused
    2-pass kernel that unpacks each KV token once and scores all L rows
    against it (MSE codecs, issue #2215); outside its envelope the L rows
    are folded into the GQA repeat dimension so the codecs' decode kernels
    apply, with the causal tail mask injected between key scoring and the
    value weighted sum — one lazy pass over the KV, no dequantize
  - Prefill (L>1): tiled quantized attention first for long contexts;
    cache.prefill_attention() first for short contexts; then dequantized SDPA
"""

import logging
from functools import cache
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_LONG_PREFILL_QUANTIZED_THRESHOLD = 8192
_LONG_PREFILL_QUERY_BLOCK_SIZE = 256
_LONG_PREFILL_KEY_CHUNK_SIZE = 16384
# MTP verify is a decode-shaped multi-row call (q_len = 1 + draft depth <= 9).
# Above this floor a multi-row call is genuine (chunked) prefill.
_DECODE_MULTIROW_MAX_Q_LEN = 15
# The repeat kernels unroll per-repeat register arrays, so folding is only a
# win while n_repeats * q_len stays under the register-pressure knee
# (measured: 24 fine, 30+ loses to single-chunk quantized_attention).
_MAX_FOLDED_REPEATS = 24
# Softmax-denominator floor, matching turboquant's quantized_attention.
_STATS_EPS = 1e-6
# Fused multi-row verify kernel envelope. Register arrays scale with QRows
# (q + o accumulators per row), so cap the rows; the adaptive MTP controller
# tops out at depth 3 (L=4). Below the token floor the fold path is already
# sub-0.2ms and the 2-pass block split has too few tokens per block.
_FUSED_MULTIROW_MAX_Q_ROWS = 4
_FUSED_MULTIROW_MIN_TOKENS = 2048


@cache
def _fused_mse_multirow_2pass1_kernel(key_bits: int, val_bits: int, dim: int):
    """Pass 1 of the fused multi-row MSE verify attention.

    Derived from turboquant's ``_fused_mse_decode_2pass_1_kernel`` with one
    structural change: each simdgroup unpacks a token's K/V codebook entries
    once and reuses them across all QRows query rows (per-row online softmax
    stats, causal tail applied inline). The upstream decode kernels re-unpack
    the KV per query row, so MTP verify paid the unpack ALU L times over
    (issue #2215).
    """
    from mlx_vlm import turboquant as _tq

    if not _tq._metal_available() or key_bits <= 0 or val_bits <= 0:
        return None
    if dim < 32 or dim % 32 != 0:
        return None

    elems_per_lane = dim // 32
    k_misaligned = (elems_per_lane * key_bits) % 8 != 0
    v_misaligned = (elems_per_lane * val_bits) % 8 != 0
    k_exprs = _tq._gen_unrolled_extract(
        key_bits, elems_per_lane, "key_codebook", "k_bit_off" if k_misaligned else ""
    )
    v_exprs = _tq._gen_unrolled_extract(
        val_bits, elems_per_lane, "val_codebook", "v_bit_off" if v_misaligned else ""
    )
    v_exprs = [e.replace("kb[", "vb[") for e in v_exprs]
    k_lines = "\n            ".join(
        f"k_el[{i}] = {expr};" for i, expr in enumerate(k_exprs)
    )
    v_lines = "\n            ".join(
        f"v_el[{i}] = {expr};" for i, expr in enumerate(v_exprs)
    )

    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = Dim / BD;
        constexpr int v_per_thread = Dim / BD;
        typedef float U;

        // Thread identity — matches turboquant's mse_sdpa_2pass_1 layout
        auto kv_head_idx = threadgroup_position_in_grid.x;
        auto batch_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto simd_lid = thread_index_in_simdgroup;
        auto gqa_idx = thread_position_in_threadgroup.y;

        auto token_count = key_norms_shape[2];
        auto kv_heads = key_norms_shape[1];
        auto bh = batch_idx * kv_heads + kv_head_idx;
        auto bqh = batch_idx * kv_heads * RepeatCount
            + kv_head_idx * RepeatCount + gqa_idx;

        auto k_nm = key_norms + bh * token_count;
        auto k_pk = key_packed + bh * token_count * KPackedWidth;
        auto v_nm = val_norms + bh * token_count;
        auto v_pk = val_packed + bh * token_count * VPackedWidth;

        // All QRows pre-rotated queries for this (kv_head, repeat) pair
        thread U q[QRows][qk_per_thread];
        for (int r = 0; r < QRows; r++) {{
            auto qr = queries + (bqh * QRows + r) * Dim
                + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; i++)
                q[r][i] = static_cast<U>(qr[i]);
        }}

        thread U o[QRows][v_per_thread] = {{}};
        U max_score[QRows];
        U sum_exp_score[QRows];
        for (int r = 0; r < QRows; r++) {{
            max_score[r] = -INFINITY;
            sum_exp_score[r] = 0;
        }}

        // Byte/bit offset for this lane's first element
        int k_bit_start = simd_lid * qk_per_thread * {key_bits};
        int v_bit_start = simd_lid * v_per_thread * {val_bits};
        int k_byte_base = k_bit_start >> 3;
        int v_byte_base = v_bit_start >> 3;
        {"int k_bit_off = k_bit_start & 7;" if k_misaligned else ""}
        {"int v_bit_off = v_bit_start & 7;" if v_misaligned else ""}

        // KV loop: unpack each token once, score all QRows rows against it
        for (int t = block_idx; t < (int)token_count; t += Blocks) {{
            U kn = static_cast<U>(k_nm[t]);
            auto kb = (const device uint8_t*)(k_pk + t * KPackedWidth)
                + k_byte_base;
            U k_el[qk_per_thread];
            {k_lines}

            auto vb = (const device uint8_t*)(v_pk + t * VPackedWidth)
                + v_byte_base;
            U vn = static_cast<U>(v_nm[t]);
            U v_el[v_per_thread];
            {v_lines}

            // Row r sits at global position token_count - QRows + r; token
            // t is invisible to rows r < t - (token_count - QRows).
            int first_row = t - (int)token_count + QRows;
            for (int r = 0; r < QRows; r++) {{
                U dot = 0;
                for (int i = 0; i < qk_per_thread; i++)
                    dot += q[r][i] * k_el[i];
                U score = simd_sum(dot) * kn;
                if (r >= first_row) {{
                    U new_max = max(max_score[r], score);
                    U factor = fast::exp(max_score[r] - new_max);
                    U exp_score = fast::exp(score - new_max);
                    max_score[r] = new_max;
                    sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;
                    for (int i = 0; i < v_per_thread; i++)
                        o[r][i] = o[r][i] * factor + exp_score * v_el[i] * vn;
                }}
            }}
        }}

        // Write per-row partial results for this block
        for (int r = 0; r < QRows; r++) {{
            auto row_out = bqh * QRows + r;
            if (simd_lid == 0) {{
                out_sums[row_out * Blocks + block_idx] = sum_exp_score[r];
                out_maxs[row_out * Blocks + block_idx] = max_score[r];
            }}
            for (int i = 0; i < v_per_thread; i++)
                out_acc[(row_out * Blocks + block_idx) * Dim
                    + simd_lid * v_per_thread + i] = static_cast<U>(o[r][i]);
        }}
    """

    return mx.fast.metal_kernel(
        name=f"omlx_tq_mse_multirow_2pass1_k{key_bits}_v{val_bits}_d{dim}",
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source=source,
    )


def _fused_multirow_mse_attention(
    real_cache, queries, keys_state, values_state, scale, total
):
    """Run MTP verify attention through the fused multi-row kernel.

    Returns None when the states/codecs are outside the kernel envelope
    (non-MSE codecs, fractional bits, mismatched dims); the caller falls
    back to the fold / one-shot paths.
    """
    from mlx_vlm import turboquant as _tq

    key_codec = getattr(real_cache, "key_codec", None)
    value_codec = getattr(real_cache, "value_codec", None)
    if not (
        isinstance(key_codec, _tq._TurboQuantMSECodec)
        and isinstance(value_codec, _tq._TurboQuantMSECodec)
    ):
        return None
    if not (
        isinstance(keys_state, _tq.TurboQuantMSEState)
        and isinstance(values_state, _tq.TurboQuantMSEState)
    ):
        return None
    if key_codec.bits != int(key_codec.bits) or value_codec.bits != int(
        value_codec.bits
    ):
        return None

    B, n_q_heads, L, D = queries.shape
    if key_codec.dim != D or value_codec.dim != D:
        return None
    if keys_state.norms.shape[0] != B:
        return None
    n_kv_heads = keys_state.norms.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    pass1 = _fused_mse_multirow_2pass1_kernel(
        int(key_codec.bits), int(value_codec.bits), D
    )
    pass2 = _tq._fused_mse_decode_2pass_2_kernel()
    if pass1 is None or pass2 is None:
        return None

    grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, L, D)
    q_rot = key_codec.prepare_queries(grouped)
    q_rot_flat = q_rot.reshape(B * n_kv_heads * n_repeats * L, D)

    # Same block split table as turboquant's 2-pass decode dispatch.
    if total <= 8192:
        num_blocks = 64
    elif total <= 32768:
        num_blocks = 128
    elif total <= 65536:
        num_blocks = 256
    else:
        num_blocks = 512

    n_rows = B * n_q_heads * L
    out_acc, out_sums, out_maxs = pass1(
        inputs=[
            q_rot_flat,
            keys_state.norms,
            keys_state.indices,
            key_codec.codebook,
            values_state.norms,
            values_state.indices,
            value_codec.codebook,
        ],
        template=[
            ("Dim", D),
            ("RepeatCount", n_repeats),
            ("QRows", L),
            ("Blocks", num_blocks),
            ("KPackedWidth", keys_state.indices.shape[-1]),
            ("VPackedWidth", values_state.indices.shape[-1]),
        ],
        grid=(n_kv_heads * 32, B * n_repeats, num_blocks),
        threadgroup=(32, n_repeats, 1),
        output_shapes=[
            (n_rows * num_blocks, D),
            (n_rows * num_blocks,),
            (n_rows * num_blocks,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = pass2(
        inputs=[out_acc, out_sums, out_maxs],
        template=[("Dim", D), ("Blocks", num_blocks)],
        grid=(n_rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_rows, D)],
        output_dtypes=[mx.float32],
    )[0]

    out_rotated = out.reshape(B, n_kv_heads, n_repeats, L, D)
    output = value_codec._rotate_inverse(out_rotated)
    return output.reshape(B, n_q_heads, L, D).astype(queries.dtype)


# One simdgroup-matrix kernel for decode and MTP verify over MSE states. A
# threadgroup owns 8 query rows of one KV head (GQA repeats x verify rows, so
# every repeat shares one unpack of each key/value token) and one block of
# tokens. Per 8-token tile its simdgroups each score a quarter of the head
# dimension (QK^T on 8x8 matrices whose key operand is decoded straight from
# the packed codes), sum the partial scores in threadgroup memory, run the
# online softmax, and accumulate their own quarter of the output (PV, the value
# norms folded into P). Pass 2 is turboquant's decode reduction. The states are
# read from the caches' step-allocated buffers, so no per-call copy of the
# sliced KV is made.
_MMA_SIMDGROUPS = 4

_MMA_SOURCE = """
    constexpr int NT = D / 8;
    constexpr int QT = NT / NS;
    constexpr uint KMASK = (1u << KB) - 1u;
    constexpr uint VMASK = (1u << VB) - 1u;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = sg * 32 + lane;
    const uint qid = lane >> 2;
    const uint fm = (qid & 4) | ((lane >> 1) & 3);
    const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
    const int head = int(threadgroup_position_in_grid.x);
    const int rtile = int(threadgroup_position_in_grid.y);
    const int block = int(threadgroup_position_in_grid.z);
    // Token count and buffer capacity are run-time values, so a growing
    // context reuses one pipeline.
    const int TOK = tokens[0];
    const long CAP = key_norms_shape[2];

    threadgroup float cbk[1 << KB];
    threadgroup float cbv[1 << VB];
    threadgroup half tq[8 * D];
    threadgroup float sred[2 * NS * 64];
    for (uint i = tid; i < (1u << KB); i += 32 * NS) cbk[i] = key_codebook[i];
    for (uint i = tid; i < (1u << VB); i += 32 * NS) cbv[i] = val_codebook[i];
    for (uint i = tid; i < 8 * D; i += 32 * NS) {
        const int r = rtile * 8 + int(i / D);
        tq[i] = r < R ? half(queries[(long(head) * R + r) * D + (i % D)]) : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Row rep * L + l sits at global position TOK - L + l.
    const int row = rtile * 8 + int(fm);
    const int last_visible = TOK - L + (min(row, R - 1) % L);

    simdgroup_matrix<float, 8, 8> om[QT];
    for (int j = 0; j < QT; ++j) om[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    float m_run = -1e30f;
    float l_run = 0.0f;
    int parity = 0;

    const int per = (TOK + Blocks - 1) / Blocks;
    const int t_begin = block * per;
    const int t_end = min(TOK, t_begin + per);
    const device uint32_t* kbase = key_packed + long(head) * CAP * KW;
    const device uint32_t* vbase = val_packed + long(head) * CAP * VW;
    const device half* knorm = key_norms + long(head) * CAP;
    const device half* vnorm = val_norms + long(head) * CAP;

    for (int t0 = t_begin; t0 < t_end; t0 += 8) {
        const int ta = min(t0 + int(fn), TOK - 1);
        const int tb = min(t0 + int(fn) + 1, TOK - 1);
        const device uint32_t* ka = kbase + long(ta) * KW;
        const device uint32_t* kb = kbase + long(tb) * KW;
        // This simdgroup's quarter of the scores, two accumulators for ILP.
        simdgroup_matrix<float, 8, 8> s0m = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> s1m = simdgroup_matrix<float, 8, 8>(0.0f);
        for (int kk = 0; kk < QT; ++kk) {
            const int kd = int(sg) * QT + kk;
            const int bit = (kd * 8 + int(fm)) * KB;
            simdgroup_matrix<half, 8, 8> qa, kt;
            simdgroup_load(qa, tq, D, ulong2(kd * 8, 0));
            kt.thread_elements()[0] = half(cbk[(ka[bit >> 5] >> (bit & 31)) & KMASK]);
            kt.thread_elements()[1] = half(cbk[(kb[bit >> 5] >> (bit & 31)) & KMASK]);
            if (kk & 1) {
                simdgroup_multiply_accumulate(s1m, qa, kt, s1m);
            } else {
                simdgroup_multiply_accumulate(s0m, qa, kt, s0m);
            }
        }
        float s0 = s0m.thread_elements()[0] + s1m.thread_elements()[0];
        float s1 = s0m.thread_elements()[1] + s1m.thread_elements()[1];
        // Double-buffered partial scores: one barrier per tile.
        threadgroup float* buf = sred + (parity * NS + sg) * 64;
        buf[fm * 8 + fn] = s0;
        buf[fm * 8 + fn + 1] = s1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = 0.0f;
        s1 = 0.0f;
        for (int i = 0; i < NS; ++i) {
            s0 += sred[(parity * NS + i) * 64 + fm * 8 + fn];
            s1 += sred[(parity * NS + i) * 64 + fm * 8 + fn + 1];
        }
        parity ^= 1;

        const int tok0 = t0 + int(fn);
        const bool ok0 = tok0 < t_end && tok0 <= last_visible;
        const bool ok1 = tok0 + 1 < t_end && tok0 + 1 <= last_visible;
        s0 = ok0 ? s0 * float(knorm[ta]) : -1e30f;
        s1 = ok1 ? s1 * float(knorm[tb]) : -1e30f;
        // Lanes of one row differ in lane bits 0 and 3.
        float mx_ = max(s0, s1);
        mx_ = max(mx_, simd_shuffle_xor(mx_, ushort(1)));
        mx_ = max(mx_, simd_shuffle_xor(mx_, ushort(8)));
        const float m_new = max(m_run, mx_);
        const float factor = fast::exp(m_run - m_new);
        const float p0 = ok0 ? fast::exp(s0 - m_new) : 0.0f;
        const float p1 = ok1 ? fast::exp(s1 - m_new) : 0.0f;
        float ps = p0 + p1;
        ps += simd_shuffle_xor(ps, ushort(1));
        ps += simd_shuffle_xor(ps, ushort(8));
        l_run = l_run * factor + ps;
        m_run = m_new;

        simdgroup_matrix<float, 8, 8> pm;
        pm.thread_elements()[0] = p0 * float(vnorm[ta]);
        pm.thread_elements()[1] = p1 * float(vnorm[tb]);
        const device uint32_t* vp = vbase + long(min(t0 + int(fm), TOK - 1)) * VW;
        for (int jj = 0; jj < QT; ++jj) {
            const int bit = ((int(sg) * QT + jj) * 8 + int(fn)) * VB;
            const uint w = vp[bit >> 5];
            simdgroup_matrix<float, 8, 8> vt;
            vt.thread_elements()[0] = cbv[(w >> (bit & 31)) & VMASK];
            vt.thread_elements()[1] = cbv[(w >> ((bit + VB) & 31)) & VMASK];
            om[jj].thread_elements()[0] *= factor;
            om[jj].thread_elements()[1] *= factor;
            simdgroup_multiply_accumulate(om[jj], pm, vt, om[jj]);
        }
    }

    if (row < R) {
        const long base = (long(head) * R + row) * Blocks + block;
        for (int jj = 0; jj < QT; ++jj) {
            const int d = (int(sg) * QT + jj) * 8 + int(fn);
            out_acc[base * D + d] = om[jj].thread_elements()[0];
            out_acc[base * D + d + 1] = om[jj].thread_elements()[1];
        }
        if (fn == 0 && sg == 0) {
            out_sums[base] = l_run;
            out_maxs[base] = m_run;
        }
    }
"""


@cache
def _mma_mse_attention_kernel():
    return mx.fast.metal_kernel(
        name="omlx_tq_mse_mma_attention",
        input_names=[
            "queries",
            "key_norms",
            "key_packed",
            "key_codebook",
            "val_norms",
            "val_packed",
            "val_codebook",
            "tokens",
        ],
        output_names=["out_acc", "out_sums", "out_maxs"],
        source=_MMA_SOURCE,
    )


def _mma_blocks(total: int) -> int:
    # Half of turboquant's 2-pass decode split, measured best on M4 Max.
    if total <= 8192:
        return 32
    if total <= 32768:
        return 64
    if total <= 65536:
        return 128
    return 256


def _full_state(cache_state, state, total):
    """The step-allocated buffer ``state`` is the ``[:total]`` prefix of."""
    from mlx_vlm import turboquant as _tq

    if (
        isinstance(cache_state, _tq.TurboQuantMSEState)
        and cache_state.norms.shape[:2] == state.norms.shape[:2]
        and cache_state.norms.shape[2] >= total
        and cache_state.indices.shape[-1] == state.indices.shape[-1]
    ):
        return cache_state
    return state


def _mma_mse_attention(real_cache, queries, keys, values, scale):
    """Decode and causal verify attention for one sequence of MSE states.

    Returns None outside the kernel's envelope (other codecs, batches,
    bit widths that straddle packed words, odd head dims).
    """
    from mlx_vlm import turboquant as _tq

    key_codec = getattr(real_cache, "key_codec", None)
    value_codec = getattr(real_cache, "value_codec", None)
    if not (
        isinstance(key_codec, _tq._TurboQuantMSECodec)
        and isinstance(value_codec, _tq._TurboQuantMSECodec)
        and isinstance(getattr(real_cache, "offset", None), int)
    ):
        return None
    keys_state = real_cache._unwrap(keys)
    values_state = real_cache._unwrap(values)
    if not (
        isinstance(keys_state, _tq.TurboQuantMSEState)
        and isinstance(values_state, _tq.TurboQuantMSEState)
    ):
        return None
    B, n_q_heads, L, D = queries.shape
    kb, vb = key_codec.bits, value_codec.bits
    if not (
        B == 1
        and keys_state.norms.shape[0] == 1
        and kb == int(kb)
        and vb == int(vb)
        and int(kb) in (1, 2, 4, 8)
        and int(vb) in (1, 2, 4, 8)
        and key_codec.dim == D
        and value_codec.dim == D
        and D % (8 * _MMA_SIMDGROUPS) == 0
        and keys_state.norms.dtype == mx.float16
        and values_state.norms.dtype == mx.float16
    ):
        return None
    n_kv_heads = keys_state.norms.shape[1]
    if n_q_heads % n_kv_heads:
        return None
    total = keys_state.norms.shape[2]
    if total < L:
        return None
    keys_full = _full_state(real_cache.keys, keys_state, total)
    values_full = _full_state(real_cache.values, values_state, total)
    if keys_full.norms.shape[2] != values_full.norms.shape[2]:
        keys_full, values_full = keys_state, values_state
    n_repeats = n_q_heads // n_kv_heads
    rows = n_repeats * L
    grouped = (queries * scale).reshape(B, n_kv_heads, n_repeats, L, D)
    q_rot = key_codec.prepare_queries(grouped).astype(mx.float32)
    num_blocks = _mma_blocks(total)
    n_rows = n_kv_heads * rows
    out_acc, out_sums, out_maxs = _mma_mse_attention_kernel()(
        inputs=[
            q_rot.reshape(n_rows, D),
            keys_full.norms,
            keys_full.indices,
            key_codec.codebook,
            values_full.norms,
            values_full.indices,
            value_codec.codebook,
            mx.array([total], dtype=mx.int32),
        ],
        template=[
            ("D", D),
            ("KB", int(kb)),
            ("VB", int(vb)),
            ("R", rows),
            ("L", L),
            ("Blocks", num_blocks),
            ("KW", keys_full.indices.shape[-1]),
            ("VW", values_full.indices.shape[-1]),
            ("NS", _MMA_SIMDGROUPS),
        ],
        grid=(32 * _MMA_SIMDGROUPS * n_kv_heads, (rows + 7) // 8, num_blocks),
        threadgroup=(32 * _MMA_SIMDGROUPS, 1, 1),
        output_shapes=[
            (n_rows * num_blocks, D),
            (n_rows * num_blocks,),
            (n_rows * num_blocks,),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = _tq._fused_mse_decode_2pass_2_kernel()(
        inputs=[out_acc, out_sums, out_maxs],
        template=[("Dim", D), ("Blocks", num_blocks)],
        grid=(n_rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_rows, D)],
        output_dtypes=[mx.float32],
    )[0]
    output = value_codec._rotate_inverse(out.reshape(B, n_kv_heads, n_repeats, L, D))
    return output.reshape(B, n_q_heads, L, D).astype(queries.dtype)


def _decode_multirow_quantized_attention(real_cache, queries, keys, values, scale):
    """Wider verify rows: one-shot quantized_attention over the whole KV.

    A single query block and a single key chunk turn quantized_attention's
    chunked online softmax into one pass — its einsum path amortizes the
    key unpack across rows, staying flat in q_len where the folded decode
    kernels hit register spill.
    """
    if not hasattr(real_cache, "quantized_attention"):
        return None
    old_query_block_size = getattr(real_cache, "prefill_query_block_size", None)
    old_key_chunk_size = getattr(real_cache, "prefill_key_chunk_size", None)
    try:
        real_cache.prefill_query_block_size = queries.shape[-2]
        real_cache.prefill_key_chunk_size = real_cache.decode_key_chunk_size
        return real_cache.quantized_attention(
            queries,
            keys_state=keys,
            values_state=values,
            scale=scale,
            mask="causal",
        )
    finally:
        if old_query_block_size is not None:
            real_cache.prefill_query_block_size = old_query_block_size
        if old_key_chunk_size is not None:
            real_cache.prefill_key_chunk_size = old_key_chunk_size


def _decode_multirow_attention(real_cache, queries, keys, values, scale):
    """Causal multi-row attention over TurboQuant states in one lazy pass.

    MTP verify would otherwise fall into the prefill fallbacks, which
    re-dequantize or chunk-scan the whole cache with per-chunk eval syncs
    on every verify cycle (issue #2127 class). MSE-codec states take the
    fused multi-row kernel (one KV unpack shared across the L rows, issue
    #2215); other codecs fold the L rows into the repeat dimension so the
    L==1 decode kernels stay applicable (repeat count is a kernel template
    parameter), with the causal tail mask applied on the raw scores before
    the value weighted sum. Returns None when the states don't fit; the
    caller falls back to the generic paths.
    """
    from mlx_vlm.turboquant import TurboQuantSplitState

    from ..turboquant_kv import _state_length

    result = _mma_mse_attention(real_cache, queries, keys, values, scale)
    if result is not None:
        return result
    keys_state = real_cache._unwrap(keys)
    values_state = real_cache._unwrap(values)
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = (
        keys_state.low.norms.shape[1]
        if isinstance(keys_state, TurboQuantSplitState)
        else keys_state.norms.shape[1]
    )
    n_repeats = n_q_heads // n_kv_heads
    total = _state_length(keys_state)
    if total < L:
        return None
    if L <= _FUSED_MULTIROW_MAX_Q_ROWS and total > _FUSED_MULTIROW_MIN_TOKENS:
        try:
            result = _fused_multirow_mse_attention(
                real_cache, queries, keys_state, values_state, scale, total
            )
        except Exception:
            logger.debug(
                "TurboQuant fused multi-row kernel failed; using fold path",
                exc_info=True,
            )
            result = None
        if result is not None:
            return result
    if n_repeats * L > _MAX_FOLDED_REPEATS:
        return _decode_multirow_quantized_attention(
            real_cache, queries, keys, values, scale
        )

    folded = (queries * scale).reshape(B, n_kv_heads, n_repeats * L, 1, D)
    prepared = real_cache.key_codec.prepare_queries(folded)
    scores = real_cache.key_codec.score_prepared(prepared, keys_state)

    # (B, H, R*L, 1, T): fold index r*L + i is the row at global position
    # total - L + i; mask the keys after it.
    scores = scores.reshape(B, n_kv_heads, n_repeats, L, total)
    q_pos = mx.arange(total - L, total)
    causal = mx.arange(total)[None, :] <= q_pos[:, None]
    scores = mx.where(causal, scores, mx.finfo(scores.dtype).min)
    scores = scores.reshape(B, n_kv_heads, n_repeats * L, 1, total)

    out, denom, _ = real_cache.value_codec.weighted_sum_stats_from_scores(
        scores, values_state
    )
    out = out / mx.maximum(denom[..., None], _STATS_EPS)
    out = out.reshape(B, n_q_heads, L, real_cache.value_codec.dim)
    return out.astype(queries.dtype)


def _patch_update_eval_policy() -> None:
    """Skip the per-layer eval for decode-shaped multi-row cache appends.

    Upstream ``update_and_fetch`` forces ``mx.eval`` whenever more than one
    token is appended — a graph-bounding measure sized for prefill chunks.
    MTP verify appends 2..9 rows per layer, so that policy serializes every
    layer of every verify cycle (~15 forced syncs/cycle). Raise the eval
    floor to prefill-sized appends; verify rows stay lazy and materialize
    at the cycle's sampling sync like the rest of the forward.
    """
    from mlx_vlm import turboquant as _tq

    cls = _tq.TurboQuantKVCache
    if getattr(cls, "_omlx_multirow_eval_patched", False):
        return

    def update_and_fetch(self, keys, values):
        # Mirror of upstream TurboQuantKVCache.update_and_fetch; the only
        # change is the eval gate (n_new > 1 -> prefill-sized appends).
        self._ensure_codecs(keys, values)

        new_keys, new_values = self._try_fused_kv_quantize(keys, values)
        if new_keys is None:
            new_keys = self.key_codec.quantize(keys)
            new_values = self.value_codec.quantize(values)

        new_end = self.offset + keys.shape[2]
        if self.keys is None:
            self.keys = _tq._allocate_state_like(new_keys, new_end)
            self.values = _tq._allocate_state_like(new_values, new_end)
        else:
            self.keys = _tq._reserve_state_capacity(
                self.keys, self.offset, new_end, self.cache_step
            )
            self.values = _tq._reserve_state_capacity(
                self.values, self.offset, new_end, self.cache_step
            )

        _tq._write_state(self.keys, new_keys, self.offset)
        _tq._write_state(self.values, new_values, self.offset)

        n_heads = keys.shape[1]
        n_new = keys.shape[2]

        self.offset = new_end
        self._cached_state = None
        self._cached_state_offset = -1
        if n_new > _DECODE_MULTIROW_MAX_Q_LEN or (self.offset % 50 == 0):
            mx.eval(self.keys, self.values)
        ks, vs = self.state
        return (
            _tq._QuantizedStateProxy(ks, self.offset, n_heads),
            _tq._QuantizedStateProxy(vs, self.offset, n_heads),
        )

    cls.update_and_fetch = update_and_fetch
    cls._omlx_multirow_eval_patched = True


def _patch_vlm_target_verify_attention() -> None:
    """Make mlx-vlm's qwen3_5 MTP verify attention TurboQuant-safe.

    The upstream verify path slices ``keys[:, :, : prefix + i + 1, :]`` per
    draft row before calling SDPA. With TurboQuant the fetched keys/values
    are packed ``_QuantizedStateProxy`` objects that are not subscriptable,
    so every verify forward crashes (issue #2139). Route TurboQuant caches
    through one causal SDPA call instead — the TurboQuant-patched dispatcher
    handles decode-shaped multi-row natively with identical semantics (row i
    attends the first ``prefix + i + 1`` positions).
    """
    try:
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except ImportError:
        return
    if getattr(q35_lang, "_omlx_tq_target_verify_patched", False):
        return
    original = getattr(q35_lang, "_qwen3_5_left_padded_attention", None)
    if original is None:
        return

    def patched(queries, keys, values, *, cache, scale, mask):
        from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

        from ..turboquant_kv import BatchTurboQuantKVCache

        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache
        if not isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            return original(queries, keys, values, cache=cache, scale=scale, mask=mask)

        sdpa = q35_lang.scaled_dot_product_attention
        if queries.shape[0] == 1 and not isinstance(mask, mx.array):
            return sdpa(
                queries, keys, values, cache=cache, scale=scale, mask="causal"
            )
        # Left-padded batches / explicit array masks: dequantize once and
        # replicate the caller's per-row causal slicing on dense arrays.
        dk, dv = real_cache.dequantize(keys_state=keys, values_state=values)
        dk = dk.astype(queries.dtype)
        dv = dv.astype(queries.dtype)
        L = queries.shape[2]
        prefix_len = dk.shape[-2] - L
        return mx.concatenate(
            [
                sdpa(
                    queries[:, :, i : i + 1, :],
                    dk[:, :, : prefix_len + i + 1, :],
                    dv[:, :, : prefix_len + i + 1, :],
                    cache=None,
                    scale=scale,
                    mask=(
                        mask[..., i : i + 1, : prefix_len + i + 1]
                        if isinstance(mask, mx.array) and mask.ndim >= 4
                        else None
                    ),
                )
                for i in range(L)
            ],
            axis=2,
        )

    q35_lang._qwen3_5_left_padded_attention = patched
    q35_lang._omlx_tq_target_verify_original = original
    q35_lang._omlx_tq_target_verify_patched = True


def apply_turboquant_attention_patch() -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for TurboQuant."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    try:
        _patch_update_eval_policy()
    except Exception:
        logger.debug("TurboQuant update eval-policy patch skipped", exc_info=True)

    try:
        _patch_vlm_target_verify_attention()
    except Exception:
        logger.debug(
            "TurboQuant VLM target-verify attention patch skipped", exc_info=True
        )

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
    ) -> mx.array:
        from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

        from ..turboquant_kv import BatchTurboQuantKVCache, _state_length

        # Detect underlying TQ cache (may be wrapped by proxy objects)
        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache

        if isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            if sinks is not None:
                # TurboQuant's quantized kernels do not implement attention
                # sinks. Preserve correctness by falling back to MLX's
                # sink-aware SDPA over dequantized states.
                dequantized_keys, dequantized_values = real_cache.dequantize(
                    keys_state=keys,
                    values_state=values,
                )
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    dequantized_keys.astype(queries.dtype),
                    dequantized_values.astype(queries.dtype),
                    scale=scale,
                    mask=mask,
                    sinks=sinks,
                )
            if queries.shape[-2] == 1 and mask is None:
                try:
                    result = _mma_mse_attention(
                        real_cache, queries, keys, values, scale
                    )
                except Exception:
                    logger.debug(
                        "TurboQuant MMA decode attention failed; "
                        "using decode_attention",
                        exc_info=True,
                    )
                    result = None
                if result is not None:
                    return result
            if queries.shape[-2] == 1:
                # Decode (B=1 and B>1). Continuous-batching decode passes a
                # per-request left-padding array mask; the masked decode_attention
                # path runs the quantized kernels directly (no full-batch
                # dequantize per step). The RHT masked-decode fix landed upstream
                # in mlx-vlm (Blaizzy/mlx-vlm#1244, in the pinned commit).
                return real_cache.decode_attention(
                    queries,
                    keys_state=keys,
                    values_state=values,
                    scale=scale,
                    mask=mask,
                )
            if (
                queries.shape[-2] <= _DECODE_MULTIROW_MAX_Q_LEN
                and isinstance(mask, str)
                and mask == "causal"
            ):
                # Decode-shaped multi-row (MTP verify) — see helper docstring.
                try:
                    result = _decode_multirow_attention(
                        real_cache, queries, keys, values, scale
                    )
                    if result is not None:
                        return result
                except Exception:
                    logger.debug(
                        "TurboQuant multi-row decode attention failed; "
                        "falling back to prefill paths",
                        exc_info=True,
                    )
            keys_state = getattr(keys, "_state", keys)
            try:
                total_tokens = _state_length(keys_state)
            except Exception:
                total_tokens = 0
            use_tiled_first = (
                total_tokens > _LONG_PREFILL_QUANTIZED_THRESHOLD
                and hasattr(real_cache, "quantized_attention")
            )
            if use_tiled_first:
                old_query_block_size = getattr(
                    real_cache, "prefill_query_block_size", None
                )
                old_key_chunk_size = getattr(
                    real_cache, "prefill_key_chunk_size", None
                )
                try:
                    real_cache.prefill_query_block_size = (
                        _LONG_PREFILL_QUERY_BLOCK_SIZE
                    )
                    real_cache.prefill_key_chunk_size = _LONG_PREFILL_KEY_CHUNK_SIZE
                    return real_cache.quantized_attention(
                        queries,
                        keys_state=keys,
                        values_state=values,
                        scale=scale,
                        mask=mask,
                    )
                except Exception:
                    logger.debug(
                        "TurboQuant quantized prefill attention failed; "
                        "falling back to prefill_attention / dequantize+SDPA",
                        exc_info=True,
                    )
                finally:
                    if old_query_block_size is not None:
                        real_cache.prefill_query_block_size = old_query_block_size
                    if old_key_chunk_size is not None:
                        real_cache.prefill_key_chunk_size = old_key_chunk_size
            result = real_cache.prefill_attention(
                queries,
                keys_state=keys,
                values_state=values,
                scale=scale,
                mask=mask,
            )
            if result is not None:
                return result
            dequantized_keys, dequantized_values = real_cache.dequantize(
                keys_state=keys,
                values_state=values,
            )
            return mx.fast.scaled_dot_product_attention(
                queries,
                dequantized_keys.astype(queries.dtype),
                dequantized_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )

        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Patch the module attribute
    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Also patch any model modules that already imported it locally
    # Covers both mlx_lm (LLM) and mlx_vlm (VLM) model modules
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is original_sdpa or func is not patched_sdpa:
                setattr(mod, "scaled_dot_product_attention", patched_sdpa)

    # Also patch mlx_vlm.models.base if loaded
    try:
        from mlx_vlm.models import base as vlm_base
        if hasattr(vlm_base, "scaled_dot_product_attention"):
            vlm_base.scaled_dot_product_attention = patched_sdpa
    except ImportError:
        pass

    _PATCHED = True
    logger.info("TurboQuant attention patch applied")
    return True
