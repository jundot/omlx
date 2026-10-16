# SPDX-License-Identifier: Apache-2.0
"""Exact gathered QSA for contiguous batch-one text prompts.

The native path reads selected four-token blocks directly from K/V. The MLX
fallback gathers the selected rows and causal tail. Eligible text-only Lightning
MTP verification uses this path too. Batched, padded, and multimodal requests
use mlx-vlm's general implementation.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import mlx.core as mx

from . import qsa_nax

IndexKeyNorm = Callable[[mx.array], mx.array]
IndexRoPE = Callable[[mx.array, mx.array], mx.array]


_NATIVE_QSA_SCORE_DISABLED = False
_NATIVE_QSA_SCORE_PROVEN = False
_NATIVE_QSA_TOPK_DISABLED = False
_NATIVE_QSA_TOPK_PROVEN = False
_NATIVE_QSA_MAIN_DISABLED = False
_NATIVE_QSA_MAIN_PROVEN = False
_NAX_QSA_MAIN_DISABLED = False
_NAX_QSA_MAIN_PROVEN = False

# Query rows from which each native kernel engages. Measured on M5 (NAX) and
# M3 Ultra alike; on M3 Ultra Flash-Next at 64k they took Lightning MTP from 48
# to 63-65 tok/s and decode from 50.8 to 52.3 tok/s.
# Indexer scores: MLX ops are faster below (NAX: 0.27 vs 0.36-0.77 ms, 1-16 rows).
_NATIVE_SCORE_MIN_ROWS = 32
# Top-k: argpartition ties or wins below (NAX: 0.26-0.31 vs 0.26 ms, one row).
_NATIVE_TOPK_MIN_ROWS = 8
# Sparse GQA: gathered SDPA is faster below (NAX: 0.7 vs 1.6 ms, verify width).
_NATIVE_MAIN_MIN_ROWS = 24


def contiguous_causal_query_chunk(key_tokens: int) -> int:
    """Keep long-context score sheets bounded without tiny launch overhead."""

    if key_tokens <= 4096:
        return 32
    if key_tokens <= 16384:
        return 64
    return 128


def _native_causal_query_chunk(key_tokens: int) -> int:
    """Amortize native QSA dispatches while bounding its FP32 score sheet."""

    if key_tokens <= 32768:
        return 4096
    if key_tokens <= 65536:
        return 2048
    return 1024


# One decode row's QSA block selection. The indexer picks the top block_topk
# complete blocks by score and the causal tail stays visible; the official
# (masked SDPA) arm widens that to a token mask, the gathered arm to the sorted
# token list it gathers. MLX builds either from the per-head scores with
# maximum/sum/divide, an argsort-backed argpartition (plus a sort) and a dozen
# to twenty small ops. This kernel scores the blocks with the same float
# operations in the same order, selects exactly the same set (argpartition's
# [-k:] is the last k of MLX's stable ascending merge sort: NaN above +inf,
# -0 == +0, ties by index) and writes the mask or token list in one launch.
# Tests set _DECODE_SELECT_DISABLED to compare against the MLX ops.
_DECODE_SELECT_DISABLED = False
_DECODE_SELECT_THREADS = 1024
# Keys held per thread; larger block banks keep the MLX ops.
_DECODE_SELECT_MAX_PER_THREAD = 32
_DECODE_SELECT_KERNELS: dict[str, object] = {}
_DECODE_SELECT_VALIDATED: set[tuple] = set()
# Failed template signatures and their errors. Register use grows with PER, so
# some GPUs cap larger-PER pipelines below TG threads; other signatures still run.
_DECODE_SELECT_FAILED: dict[tuple, str] = {}
_DECODE_SELECT_DIVISORS: dict[int, mx.array] = {}

_DECODE_SELECT_HEADER = r"""
// MLX's sum(maximum(head_scores, 0), axis=heads) / divisor for one block:
// maximum keeps NaN, the column reduce adds head by head onto 0, and the
// divide is IEEE.
inline float qsa_decode_block_score(
    const device float* head_scores, uint n, uint e, uint heads, float divisor) {
    float total = 0.0f;
    for (uint h = 0; h < heads; ++h) {
        const float v = head_scores[h * n + e];
        total = (metal::isnan(v) ? v : (v > 0.0f ? v : 0.0f)) + total;
    }
    return metal::precise::divide(total, divisor);
}

// Order of MLX's sort comparator as unsigned keys, for block scores: those
// are +0 (maximum turns -0 into +0), positive or NaN, and every NaN sorts
// equal and above +inf.
inline uint qsa_order_key(float score) {
    return metal::isnan(score) ? 0xFFFFFFFFu : as_type<uint>(score);
}
"""

_DECODE_SELECT_SOURCE = r"""
    constexpr uint TG = 1024;
    const uint n = uint(head_scores_shape[head_scores_ndim - 1]);
    const float div = divisor[0];
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup atomic_uint hist[256];
    threadgroup uint decided[2];
    threadgroup uint sg_ties[TG / 32];

    // Thread tid holds blocks tid, tid + TG, ... (slots past n are never counted).
    uint keys[PER];
    for (uint j = 0; j < PER; ++j) {
        const uint e = j * TG + tid;
        keys[j] = e < n
            ? qsa_order_key(qsa_decode_block_score(head_scores, n, e, H, div))
            : 0u;
    }

    // Radix-select the K-th largest key, eight bits per pass from the top:
    // `prefix` holds the decided high bits, `remaining` how many of the keys
    // sharing them are still to be selected.
    uint prefix = 0;
    uint remaining = K;
    for (uint pass = 0; pass < 4; ++pass) {
        const uint shift = 24 - 8 * pass;
        if (tid < 256) {
            atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < PER; ++j) {
            const uint e = j * TG + tid;
            if (e < n && (pass == 0 || (keys[j] >> (shift + 8)) == prefix)) {
                atomic_fetch_add_explicit(
                    &hist[(keys[j] >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            // Lane l owns digits 8l..8l+7; find the digit where the count of
            // larger keys first reaches `remaining`.
            uint counts[8];
            uint owned = 0;
            for (uint j = 0; j < 8; ++j) {
                counts[j] = atomic_load_explicit(&hist[8 * lane + j], memory_order_relaxed);
                owned += counts[j];
            }
            const uint above = simd_sum(owned) - simd_prefix_inclusive_sum(owned);
            if (above < remaining && remaining <= above + owned) {
                uint acc = above;
                for (int j = 7; j >= 0; --j) {
                    if (acc + counts[j] >= remaining) {
                        decided[0] = (prefix << 8) | (8 * lane + uint(j));
                        decided[1] = remaining - acc;
                        break;
                    }
                    acc += counts[j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = decided[0];
        remaining = decided[1];
    }

    // Keys equal to the threshold tie. The stable sort puts the highest-index
    // ties last, so the lowest `skip` of them stay unselected.
    const uint threshold = prefix;
    const uint skip =
        atomic_load_explicit(&hist[threshold & 255u], memory_order_relaxed) - remaining;
"""

_DECODE_SELECT_MASK_EPILOGUE = r"""
    if (skip == 0) {
        for (uint j = 0; j < PER; ++j) {
            const uint e = j * TG + tid;
            if (e < n) {
                const bool hit = keys[j] >= threshold;
                for (uint r = 0; r < R; ++r) {
                    mask[e * R + r] = hit;
                }
            }
        }
    } else {
        // Rank ties by block index: prefix counts in chunks of TG blocks.
        uint carry = 0;
        for (uint j = 0; j < PER; ++j) {
            const uint e = j * TG + tid;
            const uint tie = (e < n && keys[j] == threshold) ? 1u : 0u;
            const uint rank_in_simd = simd_prefix_exclusive_sum(tie);
            const uint simd_ties = simd_sum(tie);
            if (lane == 0) {
                sg_ties[sg] = simd_ties;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint before = 0;
            uint chunk = 0;
            for (uint s = 0; s < TG / 32; ++s) {
                const uint v = sg_ties[s];
                before += s < sg ? v : 0u;
                chunk += v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (e < n) {
                const bool hit =
                    keys[j] > threshold || (tie && carry + before + rank_in_simd >= skip);
                for (uint r = 0; r < R; ++r) {
                    mask[e * R + r] = hit;
                }
            }
            carry += chunk;
        }
    }
    // The incomplete tail block is always visible.
    if (tid < TAIL) {
        mask[n * R + tid] = true;
    }
"""

_DECODE_SELECT_TOKENS_EPILOGUE = r"""
    // Winning blocks in increasing index order, each widened to its R tokens,
    // then the tail tokens: prefix counts over chunks of TG blocks rank the
    // ties (as above) and place the winners.
    threadgroup uint sg_hits[TG / 32];
    uint tie_carry = 0;
    uint hit_carry = 0;
    for (uint j = 0; j < PER; ++j) {
        const uint e = j * TG + tid;
        const uint tie = (e < n && keys[j] == threshold) ? 1u : 0u;
        const uint tie_rank = simd_prefix_exclusive_sum(tie);
        const uint simd_ties = simd_sum(tie);
        if (lane == 0) {
            sg_ties[sg] = simd_ties;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tie_before = 0;
        uint tie_chunk = 0;
        for (uint s = 0; s < TG / 32; ++s) {
            const uint v = sg_ties[s];
            tie_before += s < sg ? v : 0u;
            tie_chunk += v;
        }
        const uint hit = (e < n && (keys[j] > threshold
            || (tie && tie_carry + tie_before + tie_rank >= skip))) ? 1u : 0u;
        const uint hit_rank = simd_prefix_exclusive_sum(hit);
        const uint simd_hits = simd_sum(hit);
        if (lane == 0) {
            sg_hits[sg] = simd_hits;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint hit_before = 0;
        uint hit_chunk = 0;
        for (uint s = 0; s < TG / 32; ++s) {
            const uint v = sg_hits[s];
            hit_before += s < sg ? v : 0u;
            hit_chunk += v;
        }
        if (hit) {
            const uint slot = (hit_carry + hit_before + hit_rank) * R;
            for (uint r = 0; r < R; ++r) {
                tokens[slot + r] = int(e * R + r);
            }
        }
        tie_carry += tie_chunk;
        hit_carry += hit_chunk;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < TAIL) {
        tokens[K * R + tid] = int(n * R + tid);
    }
"""


def _decode_block_selection(
    head_scores: mx.array,
    head_dim: int,
    key_tokens: int,
    compress_ratio: int,
    block_topk: int,
    tokens: bool,
) -> mx.array | None:
    if _DECODE_SELECT_DISABLED:
        return None
    blocks = int(head_scores.shape[-1])
    tail = key_tokens - blocks * compress_ratio
    if (
        head_scores.ndim != 4
        or head_scores.shape[0] != 1
        or 1 not in head_scores.shape[1:3]
        or head_scores.dtype != mx.float32
        or compress_ratio <= 0
        or not 0 <= tail < compress_ratio
        or not 0 < block_topk < blocks
    ):
        return None
    per_thread = 8
    while per_thread * _DECODE_SELECT_THREADS < blocks:
        per_thread *= 2
    if per_thread > _DECODE_SELECT_MAX_PER_THREAD:
        return None
    heads = int(head_scores.shape[1] * head_scores.shape[2])
    output = "tokens" if tokens else "mask"
    signature = (output, heads, block_topk, compress_ratio, tail, per_thread)
    if signature in _DECODE_SELECT_FAILED:
        return None
    try:
        kernel = _DECODE_SELECT_KERNELS.get(output)
        if kernel is None:
            kernel = mx.fast.metal_kernel(
                name=f"omlx_qwen4_qsa_decode_select_{output}",
                input_names=["head_scores", "divisor"],
                output_names=[output],
                header=_DECODE_SELECT_HEADER,
                source=_DECODE_SELECT_SOURCE
                + (_DECODE_SELECT_TOKENS_EPILOGUE if tokens else _DECODE_SELECT_MASK_EPILOGUE),
                ensure_row_contiguous=True,
            )
            _DECODE_SELECT_KERNELS[output] = kernel
        divisor = _DECODE_SELECT_DIVISORS.get(head_dim)
        if divisor is None:
            # The FP32 value MLX makes of the Python float in `scores / sqrt(d)`.
            divisor = mx.array([math.sqrt(head_dim)], dtype=mx.float32)
            mx.eval(divisor)
            _DECODE_SELECT_DIVISORS[head_dim] = divisor
        result = kernel(
            inputs=[head_scores, divisor],
            template=[
                ("H", heads),
                ("K", block_topk),
                ("R", compress_ratio),
                ("TAIL", tail),
                ("PER", per_thread),
            ],
            grid=(_DECODE_SELECT_THREADS, 1, 1),
            threadgroup=(_DECODE_SELECT_THREADS, 1, 1),
            output_shapes=[
                (1, block_topk * compress_ratio + tail) if tokens else (1, 1, 1, key_tokens)
            ],
            output_dtypes=[mx.int32 if tokens else mx.bool_],
        )[0]
        if signature not in _DECODE_SELECT_VALIDATED:
            # Surface a pipeline failure while the MLX ops can still take over.
            mx.eval(result)
            _DECODE_SELECT_VALIDATED.add(signature)
        return result
    except Exception as error:
        _DECODE_SELECT_FAILED[signature] = str(error)
        return None


def decode_block_selection_mask(
    head_scores: mx.array,
    *,
    head_dim: int,
    key_tokens: int,
    compress_ratio: int,
    block_topk: int,
) -> mx.array | None:
    """Token mask ``[1, 1, 1, key_tokens]`` of one aligned decode row, or None.

    ``head_scores`` are the row's FP32 query-head/block products
    ``[1, heads, 1, blocks]``, every complete block causal and more of them
    than ``block_topk``. The mask is what the official indexer returns: blocks
    scored ``sum(maximum(head_scores, 0), heads) / sqrt(head_dim)``, the top
    ``block_topk`` by ``mx.argpartition`` (ties resolved identically) widened
    to tokens, plus the incomplete tail. None when the kernel does not apply;
    the caller then keeps the MLX ops.
    """

    return _decode_block_selection(
        head_scores, head_dim, key_tokens, compress_ratio, block_topk, tokens=False
    )


def decode_block_selection_tokens(
    head_scores: mx.array,
    *,
    head_dim: int,
    key_tokens: int,
    compress_ratio: int,
    block_topk: int,
) -> mx.array | None:
    """Sorted selected token indices ``[1, block_topk * ratio + tail]`` (int32)
    of one decode row, or None: what the gathered arm builds from its
    argpartition, with ``head_scores`` ``[1, 1, heads, blocks]`` scored as in
    :func:`decode_block_selection_mask`."""

    return _decode_block_selection(
        head_scores, head_dim, key_tokens, compress_ratio, block_topk, tokens=True
    )


# One masked decode row's SDPA on the official arm. MLX runs its two-pass
# vector kernels there (sdpa_vector_2pass_1 with a bool mask, then
# sdpa_vector_2pass_2): key i goes to partition i % blocks, and every partition
# walks all of its positions reading one mask byte each. With ~2K of 24K keys
# selected that walk is most of the time. The kernels below are those two,
# transcribed, except that a partition first ballots its mask bytes 32 at a
# time and then visits only the selected keys, in the same increasing order:
# same partitions, same visiting order, same float operations, same bits.
# The partition count is MLX's rule for the GPU class; only the 'd' class is
# transcribed (and verified), other GPUs keep MLX.
_DECODE_SDPA_DISABLED = False
_DECODE_SDPA_KERNELS: list = []
_DECODE_SDPA_VALIDATED: set[tuple] = set()
_DECODE_SDPA_SCALES: dict[float, mx.array] = {}

_DECODE_SDPA_PASS1_SOURCE = r"""
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = V / BD;
    typedef float U;

    thread U q[qk_per_thread];
    thread U o[v_per_thread] = {0};

    const int kv_head_idx = threadgroup_position_in_grid.x;
    const int block_idx = threadgroup_position_in_grid.z;
    const int gqa_factor = threads_per_threadgroup.y;
    const int q_head_idx = gqa_factor * kv_head_idx + thread_position_in_threadgroup.y;
    const uint simd_lid = thread_index_in_simdgroup;
    const int N = int(keys_shape[2]);
    const int k_seq_stride = int(keys_strides[2]);
    const int v_seq_stride = int(values_strides[2]);
    const int mask_stride = int(mask_strides[3]);

    const device T* kp = keys + kv_head_idx * keys_strides[1]
        + block_idx * k_seq_stride + simd_lid * qk_per_thread * keys_strides[3];
    const device T* vp = values + kv_head_idx * values_strides[1]
        + block_idx * v_seq_stride + simd_lid * v_per_thread * values_strides[3];

    const float sc = scale[0];
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = static_cast<U>(sc) * queries[
            q_head_idx * queries_strides[1] + (simd_lid * qk_per_thread + i) * queries_strides[3]];
    }

    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;

    // This partition's positions are block_idx + t * BLOCKS.
    const int count = N > block_idx ? (N - block_idx + BLOCKS - 1) / BLOCKS : 0;
    for (int base = 0; base < count; base += 32) {
        const int t = base + int(simd_lid);
        const bool use = t < count && mask[(block_idx + t * BLOCKS) * mask_stride];
        uint64_t selected = uint64_t(simd_ballot(use));
        while (selected != 0) {
            const int step = base + int(ctz(selected));
            selected &= selected - 1;
            const device T* ki = kp + step * BLOCKS * k_seq_stride;
            const device T* vi = vp + step * BLOCKS * v_seq_stride;

            U score = 0;
            for (int i = 0; i < qk_per_thread; i++) {
                score += q[i] * ki[i * keys_strides[3]];
            }
            score = simd_sum(score);

            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);

            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            for (int i = 0; i < v_per_thread; i++) {
                o[i] = o[i] * factor + exp_score * vi[i * values_strides[3]];
            }
        }
    }

    const int o_offset = q_head_idx;
    if (simd_lid == 0) {
        sums[o_offset * BLOCKS + block_idx] = sum_exp_score;
        maxs[o_offset * BLOCKS + block_idx] = max_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
        partials[o_offset * BLOCKS * V + block_idx * V + simd_lid * v_per_thread + i] =
            static_cast<T>(o[i]);
    }
"""

_DECODE_SDPA_PASS2_SOURCE = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D / BD;
    typedef float U;

    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int q_offset = threadgroup_position_in_grid.x;
    const device T* pp = partials + q_offset * BLOCKS * D + simd_gid * D
        + simd_lid * elem_per_thread;
    const device float* sp = sums + q_offset * BLOCKS;
    const device float* mp = maxs + q_offset * BLOCKS;

    U sum_exp_score = 0.0;
    U max_score = Limits<U>::finite_min;

    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, mp[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * sp[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_gid] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(pp[i]);
        }
        mp += BN;
        sp += BN;
        pp += BN * D;
    }

    for (int i = 0; i < elem_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < elem_per_thread; i++) {
            out[q_offset * D + simd_gid * elem_per_thread + i] = static_cast<T>(o[i]);
        }
    }
"""


@functools.lru_cache(maxsize=None)
def _gpu_class() -> str:
    try:
        return str(mx.device_info().get("architecture", ""))[-1:]
    except Exception:
        return ""


def _two_pass_blocks(n_simds: int, keys: int) -> int | None:
    """MLX 0.32.2's sdpa_vector_2pass partition count; None where not transcribed."""

    if _gpu_class() != "d" or n_simds < 6:
        return None
    if keys < 16384:
        return 128
    if keys < 65536:
        return 512
    return 1024


def masked_decode_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    mask: mx.array,
    scale: float,
) -> mx.array | None:
    """``mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale,
    mask=mask)`` for one decode row with a bool key mask, bit for bit, visiting
    only the unmasked keys; None where MLX's kernel plan is not transcribed."""

    global _DECODE_SDPA_DISABLED
    if _DECODE_SDPA_DISABLED:
        return None
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4 or mask.ndim != 4:
        return None
    _, query_heads, rows, head_dim = queries.shape
    kv_heads, key_len = keys.shape[1], keys.shape[2]
    if (
        queries.shape[0] != 1
        or rows != 1
        or head_dim != 256
        or keys.shape != (1, kv_heads, key_len, head_dim)
        or values.shape != keys.shape
        or queries.dtype not in (mx.bfloat16, mx.float16)
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
        or mask.dtype != mx.bool_
        or mask.shape != (1, 1, 1, key_len)
        or key_len < 1024  # below it MLX may run the one-pass kernel
        or query_heads % kv_heads
    ):
        return None
    gqa_factor = query_heads // kv_heads
    blocks = _two_pass_blocks(gqa_factor * rows, key_len)
    if blocks is None:
        return None
    try:
        if not _DECODE_SDPA_KERNELS:
            _DECODE_SDPA_KERNELS.append(
                mx.fast.metal_kernel(
                    name="omlx_qwen4_masked_decode_sdpa_1",
                    input_names=["queries", "keys", "values", "mask", "scale"],
                    output_names=["partials", "sums", "maxs"],
                    source=_DECODE_SDPA_PASS1_SOURCE,
                    ensure_row_contiguous=False,
                )
            )
            _DECODE_SDPA_KERNELS.append(
                mx.fast.metal_kernel(
                    name="omlx_qwen4_masked_decode_sdpa_2",
                    input_names=["partials", "sums", "maxs"],
                    output_names=["out"],
                    source=_DECODE_SDPA_PASS2_SOURCE,
                )
            )
        pass_1, pass_2 = _DECODE_SDPA_KERNELS
        scale_array = _DECODE_SDPA_SCALES.get(scale)
        if scale_array is None:
            # MLX hands the kernel the FP32 value of the Python float.
            scale_array = mx.array([scale], dtype=mx.float32)
            mx.eval(scale_array)
            _DECODE_SDPA_SCALES[scale] = scale_array
        dtype = queries.dtype
        partials, sums, maxs = pass_1(
            inputs=[queries, keys, values, mask, scale_array],
            template=[("T", dtype), ("D", head_dim), ("V", head_dim), ("BLOCKS", blocks)],
            grid=(32 * kv_heads, gqa_factor, blocks),
            threadgroup=(32, gqa_factor, 1),
            output_shapes=[
                (query_heads, blocks, head_dim),
                (query_heads, blocks),
                (query_heads, blocks),
            ],
            output_dtypes=[dtype, mx.float32, mx.float32],
        )
        output = pass_2(
            inputs=[partials, sums, maxs],
            template=[("T", dtype), ("D", head_dim), ("BLOCKS", blocks)],
            grid=(query_heads * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(1, query_heads, 1, head_dim)],
            output_dtypes=[dtype],
        )[0]
        signature = (dtype, head_dim, gqa_factor, blocks)
        if signature not in _DECODE_SDPA_VALIDATED:
            mx.eval(output)
            _DECODE_SDPA_VALIDATED.add(signature)
        return output
    except Exception:
        _DECODE_SDPA_DISABLED = True
        return None


_TOKEN_MAJOR_MIN_QUERIES = 32
_TOKEN_MAJOR_MAX_TOKENS = 131072


def _gather_kv_rows(kv: mx.array, indices: mx.array) -> mx.array:
    """Gather token rows of the stored ``(B, H, N, D)`` cache: ``(B, S)`` -> ``(B, H, S, D)``,
    ``(B, T, S)`` -> ``(B, T, H, S, D)``. Prefill-width gathers copy token-major below 128k."""

    per_query = indices.shape[1] if indices.ndim == 3 else 1
    if per_query >= _TOKEN_MAJOR_MIN_QUERIES and kv.shape[2] < _TOKEN_MAJOR_MAX_TOKENS:
        return _gather_kv_rows_token_major(kv, indices)
    return _gather_kv_rows_stored(kv, indices)


def _gather_kv_rows_stored(kv: mx.array, indices: mx.array) -> mx.array:
    """Take along the stored token axis: cheapest for few queries, no copy of the cache."""

    batch, heads, _, dim = kv.shape
    flat = indices.astype(mx.int32).reshape(batch, -1)
    if batch == 1:
        rows = mx.take(kv, flat[0], axis=2)
    else:
        rows = mx.stack([mx.take(kv[b], flat[b], axis=1) for b in range(batch)])
    if indices.ndim == 2:
        return rows
    per_query, width = indices.shape[1], indices.shape[2]
    return rows.reshape(batch, heads, per_query, width, dim).transpose(0, 2, 1, 3, 4)


def _gather_kv_rows_token_major(kv: mx.array, indices: mx.array) -> mx.array:
    """Copy the cache token-major and gather flat rows: cheaper for many queries per token."""

    batch, heads, tokens, dim = kv.shape
    rows = kv.transpose(0, 2, 1, 3).reshape(batch * tokens, heads, dim)
    flat = indices.astype(mx.int32).reshape(batch, -1)
    if batch > 1:
        flat = flat + (mx.arange(batch, dtype=mx.int32) * tokens)[:, None]
    gathered = rows[flat.reshape(-1)].reshape(*indices.shape, heads, dim)
    axes = (0, 2, 1, 3) if indices.ndim == 2 else (0, 1, 3, 2, 4)
    return gathered.transpose(*axes)


def _portable_indexer_head_scores(
    queries: mx.array,
    pooled_keys: mx.array,
    head_dim: int,
) -> mx.array:
    """FP32 query-head/block products ``[B, T, H, blocks]``."""

    batch, query_tokens, query_heads, _ = queries.shape
    # Flatten the query-token and index-head axes so MLX emits one FP32 GEMM
    # for the chunk instead of a broadcasted batch of tiny matmuls.  Each
    # output dot product and the following head reduction are unchanged.
    return (
        queries.astype(mx.float32).reshape(
            batch, query_tokens * query_heads, head_dim
        )
        @ pooled_keys.astype(mx.float32).swapaxes(-1, -2)
    ).reshape(batch, query_tokens, query_heads, pooled_keys.shape[1])


def _portable_indexer_scores(
    queries: mx.array,
    pooled_keys: mx.array,
    head_dim: int,
) -> mx.array:
    """Current float32 MLX QSA score reference."""

    scores = _portable_indexer_head_scores(queries, pooled_keys, head_dim)
    return mx.sum(mx.maximum(scores, 0), axis=-2) / math.sqrt(head_dim)


def pool_completed_index_keys(
    index_keys: mx.array,
    index_position_ids: mx.array,
    *,
    compress_ratio: int,
    index_key_norm: IndexKeyNorm,
    apply_index_rope: IndexRoPE,
    start_block: int = 0,
    stop_block: int | None = None,
) -> mx.array:
    """Pool, normalize, and rotate a contiguous range of complete QSA blocks."""

    if index_keys.ndim != 3:
        raise ValueError("QSA raw index keys must have shape [B, S, D]")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    complete_blocks = index_keys.shape[1] // compress_ratio
    if stop_block is None:
        stop_block = complete_blocks
    if not 0 <= start_block <= stop_block <= complete_blocks:
        raise ValueError(
            "QSA pooled block range must lie within the complete raw-key prefix"
        )
    if index_position_ids.ndim not in {2, 3} or (
        index_position_ids.shape[-1] != index_keys.shape[1]
    ):
        raise ValueError("QSA index positions do not match raw index keys")

    block_count = stop_block - start_block
    raw_start = start_block * compress_ratio
    raw_stop = stop_block * compress_ratio
    pooled = index_keys[:, raw_start:raw_stop].reshape(
        index_keys.shape[0],
        block_count,
        compress_ratio,
        index_keys.shape[-1],
    )
    pooled = mx.mean(pooled.astype(mx.float32), axis=-2).astype(index_keys.dtype)
    pooled = index_key_norm(pooled)
    block_starts = mx.arange(
        raw_start,
        raw_stop,
        compress_ratio,
        dtype=mx.int32,
    )
    pooled_positions = index_position_ids[..., block_starts]
    return apply_index_rope(pooled[:, None], pooled_positions)[:, 0]


def _native_indexer_scores(
    queries: mx.array,
    pooled_keys: mx.array,
    *,
    head_dim: int,
    compress_ratio: int,
    mask_q_offset: int,
) -> mx.array | None:
    """Use the narrow native M3 score ABI or fail closed to the MLX path."""

    global _NATIVE_QSA_SCORE_DISABLED, _NATIVE_QSA_SCORE_PROVEN
    if _NATIVE_QSA_SCORE_DISABLED:
        return None
    if (
        queries.ndim != 4
        or queries.shape[0] != 1
        or queries.shape[-2:] != (4, 128)
        or pooled_keys.ndim != 3
        or pooled_keys.shape[0] != 1
        or pooled_keys.shape[-1] != 128
        or queries.dtype != pooled_keys.dtype
        or queries.dtype not in {mx.float16, mx.bfloat16}
        or head_dim != 128
        or compress_ratio != 4
        or mask_q_offset < 0
    ):
        return None
    if queries.shape[1] < _NATIVE_SCORE_MIN_ROWS:
        return None

    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        if not fast.is_native_available() or not fast.has_symbol(
            "qwen4_qsa_indexer_scores"
        ):
            _NATIVE_QSA_SCORE_DISABLED = True
            return None
        # The caller's [B,M,H,D] view transposes back to the GEMM-friendly
        # [B,H,M,D] ABI. The native wrapper only copies when the resulting
        # view is not row-contiguous (for example an offset query chunk).
        scores = fast.qwen4_qsa_indexer_scores(
            queries.transpose(0, 2, 1, 3),
            pooled_keys[:, None],
            mask_ratio=compress_ratio,
            mask_q_offset=mask_q_offset,
        )
        if not _NATIVE_QSA_SCORE_PROVEN:
            # MLX primitives are lazy, so a missing Metal pipeline would not
            # otherwise surface until the enclosing attention graph is
            # evaluated and can no longer fall back. Pay one process-wide
            # synchronization to prove the extension/pipeline pair.
            mx.eval(scores)
            _NATIVE_QSA_SCORE_PROVEN = True
        return scores
    except Exception:
        # A stale binary or rejected ABI should cost one attempt per process.
        # Shape misses were excluded above and remain eligible on later calls.
        _NATIVE_QSA_SCORE_DISABLED = True
        return None


def _native_topk_indices(scores: mx.array, topk: int) -> mx.array | None:
    """Use Qwen's exact FP32 top-k ABI or fail closed to argpartition."""

    global _NATIVE_QSA_TOPK_DISABLED, _NATIVE_QSA_TOPK_PROVEN
    if _NATIVE_QSA_TOPK_DISABLED:
        return None
    if (
        scores.ndim != 3
        or scores.shape[0] != 1
        or scores.shape[1] < 1
        or scores.shape[2] < topk
        or scores.dtype != mx.float32
        or topk != 512
    ):
        return None
    if scores.shape[1] < _NATIVE_TOPK_MIN_ROWS:
        return None
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        if not fast.is_native_available() or not fast.has_symbol(
            "qwen4_qsa_topk_indices"
        ):
            _NATIVE_QSA_TOPK_DISABLED = True
            return None
        indices = fast.qwen4_qsa_topk_indices(scores, topk=topk).astype(mx.int32)
        if not _NATIVE_QSA_TOPK_PROVEN:
            mx.eval(indices)
            _NATIVE_QSA_TOPK_PROVEN = True
        return indices
    except Exception:
        _NATIVE_QSA_TOPK_DISABLED = True
        return None


def _native_sparse_gqa_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    selected_blocks: mx.array,
    *,
    q_offset: int,
) -> mx.array | None:
    """Consume Qwen's selected rows directly in the exact native GQA kernel."""

    global _NATIVE_QSA_MAIN_DISABLED, _NATIVE_QSA_MAIN_PROVEN
    if _NATIVE_QSA_MAIN_DISABLED:
        return None
    if (
        queries.ndim != 4
        or queries.shape[0] != 1
        or queries.shape[1] != 24
        or queries.shape[-1] != 256
        or keys.ndim != 4
        or values.shape != keys.shape
        or keys.shape[0] != 1
        or keys.shape[1] != 2
        or keys.shape[-1] != 256
        or queries.dtype != keys.dtype
        or queries.dtype != values.dtype
        or queries.dtype not in {mx.float16, mx.bfloat16}
        or selected_blocks.ndim != 3
        or selected_blocks.shape != (1, queries.shape[2], 512)
        or q_offset < 0
        or q_offset + queries.shape[2] > keys.shape[2]
    ):
        return None
    if queries.shape[2] < _NATIVE_MAIN_MIN_ROWS:
        return None
    try:
        from omlx.custom_kernels.glm_moe_dsa import fast

        if not fast.is_native_available() or not fast.has_symbol(
            "qwen4_qsa_sparse_gqa_attention"
        ):
            _NATIVE_QSA_MAIN_DISABLED = True
            return None
        native_blocks = mx.contiguous(selected_blocks.astype(mx.uint32)[:, None])
        output = fast.qwen4_qsa_sparse_gqa_attention(
            queries,
            keys,
            values,
            native_blocks,
            queries.shape[-1] ** -0.5,
            q_offset,
            key_tile=64,
            dimension_tile=64,
        )
        if not _NATIVE_QSA_MAIN_PROVEN:
            # Prove the rebuilt extension and Metal pipeline before the lazy
            # graph advances cache state past a point where fallback is safe.
            mx.eval(output)
            _NATIVE_QSA_MAIN_PROVEN = True
        return output.transpose(0, 2, 1, 3)
    except Exception:
        _NATIVE_QSA_MAIN_DISABLED = True
        return None


def _nax_sparse_gqa_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    selected_blocks: mx.array,
    *,
    q_offset: int,
) -> mx.array | None:
    """Tensor-unit QSA (one query per threadgroup over its own blocks), or None.

    Same contract and geometry as :func:`_native_sparse_gqa_attention`
    (bf16 only); a compile or dispatch failure disables it for the process.
    """

    global _NAX_QSA_MAIN_DISABLED, _NAX_QSA_MAIN_PROVEN
    if _NAX_QSA_MAIN_DISABLED:
        return None
    if (
        queries.ndim != 4
        or queries.shape[0] != 1
        or queries.shape[1] != 24
        or queries.shape[-1] != 256
        or keys.ndim != 4
        or values.shape != keys.shape
        or keys.shape[0] != 1
        or keys.shape[1] != 2
        or keys.shape[-1] != 256
        or queries.dtype != mx.bfloat16
        or keys.dtype != mx.bfloat16
        or values.dtype != mx.bfloat16
        or selected_blocks.ndim != 3
        or selected_blocks.shape != (1, queries.shape[2], 512)
        or q_offset < 0
        or q_offset + queries.shape[2] > keys.shape[2]
    ):
        return None
    if queries.shape[2] < _NATIVE_MAIN_MIN_ROWS or not qsa_nax.nax_available():
        return None
    try:
        output = qsa_nax.sparse_gqa_attention(
            queries,
            keys,
            values,
            selected_blocks,
            q_offset=q_offset,
        )
        if not _NAX_QSA_MAIN_PROVEN:
            # Prove the JIT pipelines before later cache updates make a
            # fallback unsafe (same reasoning as the native kernel).
            mx.eval(output)
            _NAX_QSA_MAIN_PROVEN = True
        return output
    except Exception:
        _NAX_QSA_MAIN_DISABLED = True
        return None


def _decode_qsa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
) -> mx.array:
    """Run exact unmasked singleton SDPA through the narrow native seam.

    The gathered decode caller has already applied QSA selection, so there is
    no causal or sparse mask left to interpret.  Only the decode_fast ABI's
    explicitly supported shape/dtype contract may use the native primitive;
    missing/stale extensions and every other shape fail closed to MLX SDPA.
    Inputs are made contiguous by the caller, avoiding a lazy layout failure
    after the model caches have advanced.
    """

    try:
        from omlx.custom_kernels.decode_fast import fast

        extension = getattr(fast, "_ext", None)
        supported = getattr(extension, "sdpa_decode_supported", None)
        if (
            bool(getattr(fast, "NATIVE_AVAILABLE", False))
            and supported is not None
            and bool(supported(queries, keys, values))
        ):
            return fast.sdpa_decode(
                queries,
                keys,
                values,
                scale,
                causal=False,
            )
    except Exception:
        # Capability probing is eager and happens before a native primitive is
        # added to the lazy graph, so this fallback cannot leave partial work.
        pass

    return mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
    )


def contiguous_causal_gathered_qsa_decode(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    pooled_index_keys: mx.array,
    *,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    indexer_head_dim: int,
    compress_ratio: int,
    token_budget: int,
) -> mx.array:
    """Run exact batch-one QSA decode over only the selected K/V rows.

    This is the singleton counterpart to :func:`contiguous_causal_gathered_qsa`.
    The query is the final visible token, so every completed compressed block
    is causal.  QSA chooses ``token_budget / compress_ratio`` complete blocks;
    their token rows are gathered in chronological order and the zero-to-three
    incomplete tail rows are appended.  Main attention therefore remains
    bounded by ``token_budget + compress_ratio - 1`` instead of scanning a
    dense full-length mask.
    """

    if queries.ndim != 4 or queries.shape[:3] != (1, num_query_heads, 1):
        raise ValueError("gathered QSA decode requires [1, H, 1, D] queries")
    if queries.shape[-1] != head_dim:
        raise ValueError("QSA decode queries do not match the configured head dim")
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("QSA decode K/V must be matching rank-four arrays")
    if keys.shape[0] != 1 or keys.shape[1] != num_key_value_heads:
        raise ValueError("QSA decode K/V do not match the configured head count")
    if keys.shape[-1] != head_dim or keys.dtype != queries.dtype:
        raise ValueError("QSA decode K/V dtype or head dim does not match queries")
    if values.dtype != queries.dtype:
        raise ValueError("QSA decode values must match the query dtype")
    if (
        index_queries.ndim != 4
        or index_queries.shape[0] != 1
        or index_queries.shape[1] != 1
        or index_queries.shape[-1] != indexer_head_dim
    ):
        raise ValueError("QSA decode index queries must have shape [1, 1, H, D]")
    if compress_ratio <= 0 or token_budget <= 0 or token_budget % compress_ratio:
        raise ValueError("QSA decode token budget must contain complete blocks")
    if num_query_heads % num_key_value_heads:
        raise ValueError("QSA decode query heads must divide over K/V heads")

    key_tokens = int(keys.shape[2])
    max_blocks = key_tokens // compress_ratio
    block_budget = token_budget // compress_ratio
    if max_blocks <= block_budget:
        raise ValueError("gathered QSA decode requires a sparse block crossover")
    if pooled_index_keys.shape != (1, max_blocks, indexer_head_dim):
        raise ValueError("QSA decode pooled index-key cache has the wrong shape")

    block_scores = _native_indexer_scores(
        index_queries,
        pooled_index_keys,
        head_dim=indexer_head_dim,
        compress_ratio=compress_ratio,
        mask_q_offset=key_tokens - 1,
    )
    selected_tokens = None
    if block_scores is None:
        head_scores = _portable_indexer_head_scores(
            index_queries,
            pooled_index_keys,
            indexer_head_dim,
        )
        if index_queries.shape[1] < _NATIVE_TOPK_MIN_ROWS:
            # The argpartition path below, in one launch (same sorted tokens).
            selected_tokens = decode_block_selection_tokens(
                head_scores,
                head_dim=indexer_head_dim,
                key_tokens=key_tokens,
                compress_ratio=compress_ratio,
                block_topk=block_budget,
            )
        if selected_tokens is None:
            block_scores = mx.sum(mx.maximum(head_scores, 0), axis=-2) / math.sqrt(
                indexer_head_dim
            )

    if selected_tokens is None:
        selected_blocks = _native_topk_indices(block_scores, block_budget)
        if selected_blocks is None:
            selected_blocks = mx.argpartition(
                block_scores,
                kth=-block_budget,
                axis=-1,
            )[..., -block_budget:].astype(mx.int32)
        # Argpartition/native radix order is not chronological.  Sorting the
        # selected set preserves the official key order for deterministic SDPA.
        selected_blocks = mx.sort(selected_blocks, axis=-1)
        selected_tokens = (
            selected_blocks[..., None] * compress_ratio
            + mx.arange(compress_ratio, dtype=mx.int32)
        ).reshape(1, block_budget * compress_ratio)

        complete_key_len = max_blocks * compress_ratio
        if complete_key_len < key_tokens:
            tail = mx.arange(complete_key_len, key_tokens, dtype=mx.int32)[None]
            selected_tokens = mx.concatenate((selected_tokens, tail), axis=-1)

    selected_keys = _gather_kv_rows(keys, selected_tokens)
    selected_values = _gather_kv_rows(values, selected_tokens)
    output = _decode_qsa_sdpa(
        queries,
        selected_keys,
        selected_values,
        head_dim**-0.5,
    )
    return output.transpose(0, 2, 1, 3)


def contiguous_causal_gathered_qsa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    index_queries: mx.array,
    index_keys: mx.array,
    index_position_ids: mx.array,
    *,
    num_query_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    indexer_head_dim: int,
    compress_ratio: int,
    token_budget: int,
    index_key_norm: IndexKeyNorm,
    apply_index_rope: IndexRoPE,
    pooled_index_keys: mx.array | None = None,
    query_chunk: int | None = None,
) -> mx.array:
    """Run exact QSA over gathered K/V for one contiguous causal prompt.

    ``queries`` and ``keys`` must already carry their main-attention RoPE.
    ``index_queries`` must likewise be normalized and RoPE-rotated.  Raw
    indexer keys remain unrotated because Qwen pools each complete micro-block
    before applying its checkpoint k-norm and the block-start RoPE.
    """

    if queries.ndim != 4 or queries.shape[0] != 1 or queries.shape[2] <= 1:
        raise ValueError(
            "gathered QSA requires rank-four batch-one multi-token queries"
        )
    batch, actual_query_heads, query_tokens, actual_head_dim = queries.shape
    if actual_query_heads != num_query_heads or actual_head_dim != head_dim:
        raise ValueError("QSA queries do not match the configured geometry")
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("QSA keys and values must be matching rank-four arrays")
    if keys.shape[0] != batch or keys.shape[1:] != (
        num_key_value_heads,
        keys.shape[2],
        head_dim,
    ):
        raise ValueError("QSA K/V do not match the configured geometry")
    key_tokens = keys.shape[2]
    if query_tokens > key_tokens:
        raise ValueError("QSA query length cannot exceed cached key length")
    if index_queries.ndim != 4 or index_queries.shape[:2] != (
        batch,
        query_tokens,
    ):
        raise ValueError("QSA index queries do not match the current prompt")
    if index_queries.shape[-1] != indexer_head_dim:
        raise ValueError("QSA index queries have the wrong head dimension")
    if index_keys.shape != (batch, key_tokens, indexer_head_dim):
        raise ValueError("QSA raw index keys do not match cached K/V")
    if (
        index_position_ids.ndim not in {2, 3}
        or index_position_ids.shape[-1] != key_tokens
    ):
        raise ValueError("QSA index positions do not match cached K/V")
    if compress_ratio <= 0 or token_budget <= 0 or token_budget % compress_ratio:
        raise ValueError("QSA token budget must contain complete micro-blocks")
    if num_query_heads % num_key_value_heads:
        raise ValueError("QSA query heads must divide evenly over K/V heads")

    if query_chunk is None:
        query_chunk = contiguous_causal_query_chunk(key_tokens)
        # The direct-index main-attention kernel carries no per-query gathered
        # K/V tensor. Use wider tiles while the FP32 score sheet stays bounded
        # to 128 MiB through 64K keys, then preserve the 1,024-row long-context
        # tile (256 MiB at the model's 256K context limit).
        if (
            queries.shape[1:] == (24, query_tokens, 256)
            and keys.shape[1] == 2
            and queries.dtype in {mx.float16, mx.bfloat16}
            and not _NATIVE_QSA_MAIN_DISABLED
        ):
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast

                if fast.is_native_available() and fast.has_symbol(
                    "qwen4_qsa_sparse_gqa_attention"
                ):
                    query_chunk = max(
                        query_chunk, _native_causal_query_chunk(key_tokens)
                    )
            except Exception:
                pass
    if query_chunk <= 0:
        raise ValueError("QSA query chunk must be positive")

    ratio = compress_ratio
    max_blocks = key_tokens // ratio
    block_budget = token_budget // ratio
    query_start = key_tokens - query_tokens

    # A contiguous prompt shares the same block bank for every query.  The
    # caller can provide its cache of completed blocks; standalone users still
    # get the exact one-shot construction.
    if max_blocks:
        if pooled_index_keys is None:
            pooled = pool_completed_index_keys(
                index_keys,
                index_position_ids,
                compress_ratio=ratio,
                index_key_norm=index_key_norm,
                apply_index_rope=apply_index_rope,
            )
        else:
            if pooled_index_keys.shape != (batch, max_blocks, indexer_head_dim):
                raise ValueError("QSA pooled index-key cache has the wrong shape")
            pooled = pooled_index_keys
    else:
        if pooled_index_keys is not None and pooled_index_keys.shape != (
            batch,
            0,
            indexer_head_dim,
        ):
            raise ValueError("QSA pooled index-key cache has the wrong shape")
        pooled = None

    outputs: list[mx.array] = []
    groups = num_query_heads // num_key_value_heads
    for start in range(0, query_tokens, query_chunk):
        stop = min(start + query_chunk, query_tokens)
        chunk_tokens = stop - start
        absolute_queries = query_start + mx.arange(start, stop, dtype=mx.int32)
        visible_counts = mx.broadcast_to(
            (absolute_queries + 1)[None], (batch, chunk_tokens)
        )
        complete_counts = visible_counts // ratio

        if max_blocks:
            chunk_index_queries = index_queries[:, start:stop]
            block_scores = _native_indexer_scores(
                chunk_index_queries,
                pooled,
                head_dim=indexer_head_dim,
                compress_ratio=ratio,
                mask_q_offset=query_start + start,
            )
            if block_scores is None:
                block_scores = _portable_indexer_scores(
                    chunk_index_queries,
                    pooled,
                    indexer_head_dim,
                )
                valid_blocks = (
                    mx.arange(max_blocks)[None, None, :]
                    < complete_counts[..., None]
                )
                block_scores = mx.where(
                    valid_blocks,
                    block_scores,
                    mx.finfo(block_scores.dtype).min,
                )

            selected_width = min(max_blocks, block_budget)
            canonical = mx.broadcast_to(
                mx.arange(selected_width, dtype=mx.int32)[None, None],
                (batch, chunk_tokens, selected_width),
            )
            chronological = True
            if max_blocks > block_budget:
                ranked = _native_topk_indices(block_scores, block_budget)
                if ranked is None:
                    chronological = False
                    ranked = mx.argpartition(
                        block_scores,
                        kth=-block_budget,
                        axis=-1,
                    )[..., -block_budget:].astype(mx.int32)
                selected_block_rows = mx.where(
                    (complete_counts <= block_budget)[..., None],
                    canonical,
                    ranked,
                )
            else:
                selected_block_rows = canonical

            # argpartition's top-k set is unordered. Restore checkpoint/dense
            # token order before the portable gathered SDPA or the direct native
            # kernel performs its FP32 online-softmax reduction. The native
            # top-k already emits ascending block ids.
            if not chronological:
                selected_block_rows = mx.sort(selected_block_rows, axis=-1)

            selected_count = mx.minimum(complete_counts, block_budget)

            nax_output = _nax_sparse_gqa_attention(
                queries[:, :, start:stop],
                keys,
                values,
                selected_block_rows,
                q_offset=query_start + start,
            )
            if nax_output is not None:
                outputs.append(nax_output)
                continue

            native_output = _native_sparse_gqa_attention(
                queries[:, :, start:stop],
                keys,
                values,
                selected_block_rows,
                q_offset=query_start + start,
            )
            if native_output is not None:
                outputs.append(native_output)
                continue

            selected_indices = (
                selected_block_rows[..., None] * ratio
                + mx.arange(ratio, dtype=mx.int32)
            ).reshape(batch, chunk_tokens, selected_width * ratio)
            selected_valid = mx.broadcast_to(
                mx.arange(selected_width)[None, None, :, None]
                < selected_count[..., None, None],
                (batch, chunk_tokens, selected_width, ratio),
            ).reshape(batch, chunk_tokens, selected_width * ratio)
        else:
            selected_indices = mx.zeros(
                (batch, chunk_tokens, 0), dtype=mx.int32
            )
            selected_valid = mx.zeros(
                (batch, chunk_tokens, 0), dtype=mx.bool_
            )

        # The zero-to-three visible tokens after the final complete block are
        # always retained by the published QSA contract.
        tail_width = ratio - 1
        tail = complete_counts[..., None] * ratio + mx.arange(
            tail_width, dtype=mx.int32
        )
        tail_valid = tail < visible_counts[..., None]
        selected_indices = mx.concatenate((selected_indices, tail), axis=-1)
        selected_valid = mx.concatenate((selected_valid, tail_valid), axis=-1)

        safe_selected = mx.where(selected_valid, selected_indices, 0).astype(mx.int32)

        selected_keys = _gather_kv_rows(keys, safe_selected)
        selected_values = _gather_kv_rows(values, safe_selected)

        chunk_queries = queries[:, :, start:stop].transpose(0, 2, 1, 3)
        grouped_queries = chunk_queries.reshape(
            batch,
            chunk_tokens,
            num_key_value_heads,
            groups,
            head_dim,
        )
        scores = (
            grouped_queries.astype(mx.float32)
            @ selected_keys.astype(mx.float32).swapaxes(-1, -2)
        ) / math.sqrt(head_dim)
        scores = mx.where(
            selected_valid[:, :, None, None, :],
            scores,
            mx.finfo(scores.dtype).min,
        )
        probabilities = mx.softmax(scores, axis=-1).astype(chunk_queries.dtype)
        output = probabilities @ selected_values
        outputs.append(
            output.reshape(batch, chunk_tokens, num_query_heads, head_dim)
        )

    return mx.concatenate(outputs, axis=1)


def batched_causal_block_selection(
    index_queries: mx.array,
    pooled_index_keys: mx.array,
    visible: mx.array,
    *,
    compress_ratio: int,
    block_topk: int,
    indexer_head_dim: int,
) -> tuple[mx.array, mx.array]:
    """QSA key selection of every query of a batch of contiguous rows.

    ``index_queries`` ``[B, heads, L, D]`` are normalized and rotated;
    ``pooled_index_keys`` ``[B, blocks, D]`` holds each row's completed blocks
    left-aligned from its first token (zero past the row's own count);
    ``visible`` ``[B, L]`` is how many of its row's tokens each query sees.
    Returns row-relative token indices ``[B, L, block_topk * ratio + ratio - 1]``
    (int32, chronological) and their validity: the query's top ``block_topk``
    causal blocks by the FP32 score the indexer uses -- every causal block
    below the crossover -- then its incomplete tail. Invalid slots index 0.
    ``blocks`` must exceed ``block_topk``.
    """

    ratio = compress_ratio
    batch, _, length, _ = index_queries.shape
    blocks = pooled_index_keys.shape[1]
    complete = visible // ratio
    scores = index_queries.astype(mx.float32) @ pooled_index_keys[:, None].astype(
        mx.float32
    ).swapaxes(-1, -2)
    scores = mx.sum(mx.maximum(scores, 0), axis=1) / math.sqrt(indexer_head_dim)
    scores = mx.where(
        mx.arange(blocks)[None, None] < complete[..., None], scores, -mx.inf
    )
    # argpartition's top-k set is unordered; restore key order as the dense
    # path's SDPA reads keys.
    chosen = mx.sort(
        mx.argpartition(scores, kth=-block_topk, axis=-1)[..., -block_topk:].astype(
            mx.int32
        ),
        axis=-1,
    )
    tokens = (chosen[..., None] * ratio + mx.arange(ratio, dtype=mx.int32)).reshape(
        batch, length, block_topk * ratio
    )
    tail = complete[..., None] * ratio + mx.arange(ratio - 1, dtype=mx.int32)
    valid = mx.concatenate(
        [
            mx.repeat(chosen < complete[..., None], ratio, axis=-1),
            tail < visible[..., None],
        ],
        axis=-1,
    )
    return mx.where(valid, mx.concatenate([tokens, tail], axis=-1), 0), valid


__all__ = [
    "batched_causal_block_selection",
    "contiguous_causal_gathered_qsa",
    "contiguous_causal_gathered_qsa_decode",
    "contiguous_causal_query_chunk",
    "decode_block_selection_mask",
    "masked_decode_sdpa",
    "pool_completed_index_keys",
]
