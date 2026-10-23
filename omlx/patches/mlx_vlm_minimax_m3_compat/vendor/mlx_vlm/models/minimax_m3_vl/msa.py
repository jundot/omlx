from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx

try:
    from omlx.custom_kernels.sparse_index import fast as _index_fast
except Exception:  # pragma: no cover - optional native extension
    _index_fast = mx.fast


def _has_minimax_fast_symbol(name: str) -> bool:
    has_symbol = getattr(_index_fast, "has_symbol", None)
    if callable(has_symbol):
        return bool(has_symbol(name))
    return hasattr(_index_fast, name)


def _load_mlx_steel_attn_header() -> Optional[str]:
    include_root = Path(mx.__file__).parent / "include"
    if not include_root.exists():
        return None

    seen: set[Path] = set()

    def expand(include_path: str) -> str:
        path = include_root / include_path
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
        return expand("mlx/backend/metal/kernels/steel/attn/attn.h")
    except OSError:
        return None


def _make_msa_csr_k1_steel_mma_kernel():
    header = _load_mlx_steel_attn_header()
    if header is None:
        return None

    header += r"""

using namespace metal;
using namespace mlx::steel;

struct MSA_MaxOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return metal::max(x, y);
  }
};

struct MSA_SumOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return x + y;
  }
};

struct MSA_MulOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return x * y;
  }
};

struct MSA_Exp2SubOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return fast::exp2(x - y);
  }
};

struct MSA_DivOp {
  template <typename U>
  METAL_FUNC static constexpr U apply(U x, U y) {
    return x / y;
  }
};
"""

    return mx.fast.metal_kernel(
        name="minimax_m3_msa_csr_k1_steel_mma",
        input_names=["q", "k", "v", "row_ptr", "qsplit", "group_ptr", "scale"],
        output_names=["o_partial", "lse_partial"],
        header=header,
        source=r"""
            constexpr short BQ = M_BLOCK_SIZE;
            constexpr short BK = 128;
            constexpr short BD = 128;
            constexpr short WM = M_BLOCK_SIZE / 8;
            constexpr short WN = 1;
            constexpr short THREADS = WM * WN * 32;
            // One threadgroup per query group of a (KV head, block row) that
            // holds edges, in head, row, group order: the last row whose first
            // group is at or before this one owns it.
            int work = int(threadgroup_position_in_grid.x);
            int lo = 0;
            int hi = H_KV * TOTAL_ROWS - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) / 2;
                if (group_ptr[mid] <= work) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            int group_id = work - group_ptr[lo];
            int row = lo % TOTAL_ROWS;
            int hkv = lo / TOTAL_ROWS;
            int row_base = hkv * (TOTAL_ROWS + 1);
            int row_start = row_ptr[row_base + row];
            int row_end = row_ptr[row_base + row + 1];
            int edge_base = row_start + group_id * Q_TOKENS_PER_GROUP;
            if (edge_base >= row_end) {
                return;
            }

            uint tid_linear = thread_index_in_threadgroup;
            uint simd_lane_id = thread_index_in_simdgroup;
            uint simd_group_id = simdgroup_index_in_threadgroup;

            threadgroup int qidx_s[8];
            threadgroup int split_s[8];
            threadgroup int valid_s[8];
            threadgroup int any_valid_s;
            if (tid_linear == 0) {
                any_valid_s = 0;
                for (int qt = 0; qt < 8; ++qt) {
                    qidx_s[qt] = 0;
                    split_s[qt] = 0;
                    valid_s[qt] = 0;
                    if (qt < Q_TOKENS_PER_GROUP) {
                        int edge = edge_base + qt;
                        if (edge < row_end) {
                            int qsplit_value = qsplit[hkv * NNZ_CAP + edge];
                            if (qsplit_value >= 0) {
                                qidx_s[qt] = qsplit_value & 0x00FFFFFF;
                                split_s[qt] = (qsplit_value >> 24) & 0xFF;
                                valid_s[qt] = 1;
                                any_valid_s = 1;
                            }
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (any_valid_s == 0) {
                return;
            }

            int block_start = row * BLOCK_SIZE;

            constexpr short kFragSize = 8;
            using MMAFrag_acc_t = BaseMMAFrag<float, kFragSize, kFragSize>;
            constexpr int kNWarps = WM * WN;
            constexpr int TQ = BQ / (kNWarps * kFragSize);
            constexpr int TK = BK / kFragSize;
            constexpr int TD = BD / kFragSize;
            static_assert(TQ == 1, "MSA Steel MMA K1 expects one Q tile per warp.");

            MMATile<float, TQ, 1, MMAFrag_acc_t> Qtile;
            MMATile<float, 1, TK, MMAFrag_acc_t> Ktile;
            MMATile<float, TQ, TK, MMAFrag_acc_t> Stile;
            MMATile<float, 1, 1, MMAFrag_acc_t> Vtile;
            MMATile<float, TQ, TD, MMAFrag_acc_t> Otile;
            Otile.clear();

            const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
            const short sm = simd_coord.y;
            const short sn = simd_coord.x;
            const short tm = kFragSize * TQ * simd_group_id;

            constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

            float max_score[kRowsPT];
            float sum_score[kRowsPT] = {0};
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kRowsPT; ++i) {
                max_score[i] = -INFINITY;
            }

            Stile.clear();

            STEEL_PRAGMA_UNROLL
            for (short dd = 0; dd < TD; dd++) {
                simdgroup_barrier(mem_flags::mem_none);
                STEEL_PRAGMA_UNROLL
                for (short iq = 0; iq < TQ; iq++) {
                    int m = tm + sm + iq * kFragSize;
                    int qt = m / QHEAD_PER_KV;
                    int h_in_group = m - qt * QHEAD_PER_KV;
                    bool row_valid =
                        qt < Q_TOKENS_PER_GROUP && valid_s[qt] != 0;
                    int hq = hkv * QHEAD_PER_KV + h_in_group;
                    int q_idx = row_valid ? qidx_s[qt] : 0;
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
                        int d = dd * kFragSize + sn + jj;
                        Qtile.frag_at(iq, 0)[jj] = row_valid
                            ? float(q[(q_idx * H_Q + hq) * D + d])
                            : 0.0f;
                    }
                }
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    int d = dd * kFragSize + sm;
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
                        int k_pos = block_start + ik * kFragSize + sn + jj;
                        Ktile.frag_at(0, ik)[jj] = k_pos < TOTAL_K
                            ? float(k[(k_pos * H_KV + hkv) * D + d])
                            : 0.0f;
                    }
                }
                simdgroup_barrier(mem_flags::mem_none);
                tile_matmad(Stile, Qtile, Ktile, Stile);
            }

            float scale_log2 = float(scale) * 1.4426950408889634f;
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
                Stile.elems()[ii] *= scale_log2;
            }

            using stile_t = decltype(Stile);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < stile_t::kTileRows; i++) {
                int m = tm + sm + i * stile_t::kFragRows;
                int qt = m / QHEAD_PER_KV;
                bool row_valid = qt < Q_TOKENS_PER_GROUP && valid_s[qt] != 0;
                int q_abs = row_valid ? (Q_START + qidx_s[qt]) : -1;
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < stile_t::kTileCols; j++) {
                    int col_pos = block_start + sn + j * stile_t::kFragCols;
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
                        int k_pos = col_pos + jj;
                        if (!row_valid || k_pos >= TOTAL_K || k_pos > q_abs) {
                            Stile.frag_at(i, j)[jj] = -INFINITY;
                        }
                    }
                }
            }

            float new_max[kRowsPT];
            float factor[kRowsPT];
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kRowsPT; ++i) {
                new_max[i] = max_score[i];
            }
            Stile.template row_reduce<MSA_MaxOp>(new_max);
            Stile.template row_bin_op<MSA_Exp2SubOp>(new_max);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kRowsPT; ++i) {
                factor[i] = fast::exp2(max_score[i] - new_max[i]);
                max_score[i] = new_max[i];
            }

            float sum_score_tmp[kRowsPT] = {0};
            Stile.template row_reduce<MSA_SumOp>(sum_score_tmp);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < kRowsPT; ++i) {
                sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
            }
            Otile.template row_bin_op<MSA_MulOp>(factor);

            threadgroup_barrier(mem_flags::mem_threadgroup);
            STEEL_PRAGMA_UNROLL
            for (short iq = 0; iq < TQ; iq++) {
                STEEL_PRAGMA_UNROLL
                for (short id = 0; id < TD; id++) {
                    STEEL_PRAGMA_UNROLL
                    for (short ik = 0; ik < TK; ik++) {
                        simdgroup_barrier(mem_flags::mem_none);
                        STEEL_PRAGMA_UNROLL
                        for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
                            int k_pos = block_start + ik * kFragSize + sm;
                            int d = id * kFragSize + sn + jj;
                            Vtile.frag_at(0, 0)[jj] = k_pos < TOTAL_K
                                ? float(v[(k_pos * H_KV + hkv) * D + d])
                                : 0.0f;
                        }
                        simdgroup_barrier(mem_flags::mem_none);
                        MMAFrag_acc_t::mma(
                            Otile.frag_at(iq, id),
                            Stile.frag_at(iq, ik),
                            Vtile.frag_at(0, 0),
                            Otile.frag_at(iq, id));
                    }
                }
            }

            Otile.template row_bin_op<MSA_DivOp>(sum_score);
            threadgroup_barrier(mem_flags::mem_none);

            constexpr float LN2 = 0.6931471805599453f;
            STEEL_PRAGMA_UNROLL
            for (short iq = 0; iq < TQ; iq++) {
                int m = tm + sm + iq * kFragSize;
                int qt = m / QHEAD_PER_KV;
                int h_in_group = m - qt * QHEAD_PER_KV;
                if (qt < Q_TOKENS_PER_GROUP && valid_s[qt] != 0) {
                    int q_idx = qidx_s[qt];
                    int split_idx = split_s[qt];
                    int hq = hkv * QHEAD_PER_KV + h_in_group;
                    if (sn == 0) {
                        int lse_idx = (split_idx * TOTAL_Q + q_idx) * H_Q + hq;
                        lse_partial[lse_idx] = sum_score[iq] > 0.0f
                            ? (max_score[iq] + metal::log2(sum_score[iq])) * LN2
                            : -INFINITY;
                    }

                    STEEL_PRAGMA_UNROLL
                    for (short id = 0; id < TD; id++) {
                        int d_base = sn + id * kFragSize;
                        STEEL_PRAGMA_UNROLL
                        for (short jj = 0; jj < MMAFrag_acc_t::kElemCols; jj++) {
                            int d = d_base + jj;
                            int partial_idx =
                                ((split_idx * TOTAL_Q + q_idx) * H_Q + hq) * D + d;
                            o_partial[partial_idx] = T(Otile.frag_at(iq, id)[jj]);
                        }
                    }
                }
            }
        """,
    )


_MSA_CSR_K1_STEEL_MMA = (
    _make_msa_csr_k1_steel_mma_kernel() if mx.metal.is_available() else None
)


def _select_msa_topk_from_block_scores(
    block_scores: mx.array,
    *,
    q_start: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> mx.array:
    B, H_idx, L, num_blocks = block_scores.shape
    neg = mx.array(-float("inf"), dtype=mx.float32)
    qpos = mx.arange(q_start, q_start + L)
    blocks = mx.arange(num_blocks)
    cur_block = qpos // block_size
    valid_blocks = blocks[None, None, None, :] <= cur_block[None, None, :, None]
    valid_blocks = mx.broadcast_to(valid_blocks, (B, H_idx, L, num_blocks))

    selected_scores = mx.where(valid_blocks, block_scores, neg)
    if init_blocks > 0:
        init = blocks[None, None, None, :] < init_blocks
        selected_scores = mx.where(
            init & valid_blocks,
            mx.array(1e30, dtype=selected_scores.dtype),
            selected_scores,
        )
    if local_blocks > 0:
        local_start = mx.maximum(cur_block - local_blocks + 1, 0)
        local = (blocks[None, None, None, :] >= local_start[None, None, :, None]) & (
            blocks[None, None, None, :] <= cur_block[None, None, :, None]
        )
        selected_scores = mx.where(
            local & valid_blocks,
            mx.array(1e29, dtype=selected_scores.dtype),
            selected_scores,
        )

    topk_idx = mx.argpartition(-selected_scores, kth=topk - 1, axis=-1)[..., :topk]
    topk_valid = mx.take_along_axis(valid_blocks, topk_idx, axis=-1)
    topk_idx = mx.where(topk_valid, topk_idx, mx.array(num_blocks, dtype=topk_idx.dtype))
    sort_order = mx.argsort(topk_idx, axis=-1)
    topk_idx = mx.take_along_axis(topk_idx, sort_order, axis=-1)
    invalid = mx.array(-1, dtype=mx.int32)
    return mx.where(topk_idx < num_blocks, topk_idx.astype(mx.int32), invalid)


def _build_grouped_msa_topk_native(
    idx_queries: mx.array,
    idx_keys: mx.array,
    q_start: int,
    scale: float,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> Optional[mx.array]:
    if not _has_minimax_fast_symbol("msa_block_topk"):
        return None
    if not mx.metal.is_available():
        return None
    if (
        idx_queries.ndim != 4
        or idx_keys.ndim != 4
        or idx_keys.shape[1] != 1
        or idx_queries.shape[-1] != 128
        or idx_keys.shape[-1] != 128
        or block_size != 128
        or topk != 16
    ):
        return None

    try:
        return _index_fast.msa_block_topk(
            idx_queries,
            idx_keys,
            q_start=int(q_start),
            scale=float(scale),
            block_size=int(block_size),
            topk=int(topk),
            init_blocks=int(init_blocks),
            local_blocks=int(local_blocks),
        )
    except Exception:
        return None


_MSA_CSR_K2 = mx.fast.metal_kernel(
    name="minimax_m3_msa_csr_k2_combine",
    input_names=["o_partial", "lse_partial", "split_counts"],
    output_names=["out"],
    source=r"""
        uint elem = thread_position_in_grid.x;
        uint total = TOTAL_Q * H_Q * D;
        if (elem >= total) {
            return;
        }

        int d = elem % D;
        uint tmp = elem / D;
        int hq = tmp % H_Q;
        int q_idx = tmp / H_Q;
        int hkv = hq / QHEAD_PER_KV;
        int count = TOPK;
        if (FULL_SPLITS == 0) {
            count = split_counts[q_idx * H_KV + hkv];
        }

        if (count <= 0) {
            out[elem] = T(0);
            return;
        }

        float row_max = -INFINITY;
        for (int s = 0; s < TOPK; ++s) {
            if (s >= count) {
                break;
            }
            int lse_idx = (s * TOTAL_Q + q_idx) * H_Q + hq;
            row_max = max(row_max, lse_partial[lse_idx]);
        }

        float denom = 0.0f;
        float acc = 0.0f;
        for (int s = 0; s < TOPK; ++s) {
            if (s >= count) {
                break;
            }
            int lse_idx = (s * TOTAL_Q + q_idx) * H_Q + hq;
            float lse = lse_partial[lse_idx];
            float weight = metal::exp(lse - row_max);
            denom += weight;
            int partial_idx = ((s * TOTAL_Q + q_idx) * H_Q + hq) * D + d;
            acc += weight * o_partial[partial_idx];
        }
        out[elem] = T(acc / denom);
    """,
)


def build_grouped_msa_topk(
    idx_queries: mx.array,
    idx_keys: mx.array,
    q_start: int,
    scale: float,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    *,
    block_chunk_size: Optional[int] = None,
) -> mx.array:
    """MiniMax MSA q2k block top-k per KV group, ``[B, H_kv, L, topK]``.

    Invalid entries are ``-1``, as in the official sparse metadata contract.
    Without the native kernel, scores are computed ``block_chunk_size`` KV
    blocks at a time (all at once when None) and reduced to block scores right
    away, so a long context never holds the ``[B, H_kv, L, total_k]`` scores.
    """
    if idx_queries.ndim != 4 or idx_keys.ndim != 4:
        raise ValueError("idx_queries/idx_keys must have shape [B, H, L, D].")

    native = _build_grouped_msa_topk_native(
        idx_queries,
        idx_keys,
        q_start,
        scale,
        block_size,
        topk,
        init_blocks,
        local_blocks,
    )
    if native is not None:
        return native

    B, H_idx, L, _ = idx_queries.shape
    total_len = idx_keys.shape[2]
    num_blocks = (total_len + block_size - 1) // block_size
    chunk = block_chunk_size or num_blocks
    neg = mx.array(-float("inf"), dtype=mx.float32)

    q = idx_queries.astype(mx.float32)
    qpos = mx.arange(q_start, q_start + L)
    block_score_chunks = []
    for block_start in range(0, num_blocks, chunk):
        blocks_in_chunk = min(chunk, num_blocks - block_start)
        token_start = block_start * block_size
        token_end = min(total_len, token_start + blocks_in_chunk * block_size)
        token_count = token_end - token_start
        key_chunk = idx_keys[:, :, token_start:token_end, :].astype(mx.float32)
        scores = mx.matmul(q, key_chunk.swapaxes(-1, -2)) * scale

        kpos = mx.arange(token_start, token_end)
        causal = kpos[None, None, None, :] <= qpos[None, None, :, None]
        scores = mx.where(causal, scores, neg)

        pad = blocks_in_chunk * block_size - token_count
        if pad:
            pad_values = mx.full(
                (*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype
            )
            scores = mx.concatenate([scores, pad_values], axis=-1)

        scores = scores.reshape(B, H_idx, L, blocks_in_chunk, block_size)
        block_score_chunks.append(mx.max(scores, axis=-1))

    block_scores = mx.concatenate(block_score_chunks, axis=-1)
    block_scores = mx.where(block_scores == block_scores, block_scores, neg)

    return _select_msa_topk_from_block_scores(
        block_scores,
        q_start=q_start,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


def build_k2q_csr_b1(
    q2k_indices: mx.array,
    *,
    total_k: int,
    block_size: int,
    with_split_counts: bool = True,
) -> tuple[mx.array, mx.array, Optional[mx.array]]:
    """Official-style k2q CSR metadata for the B=1 prefill path.

    ``q2k_indices [H_kv, total_q, topK]`` holds batch-local KV block indices
    padded with ``-1``. Returns ``k2q_row_ptr [H_kv, total_rows + 1]``, the
    packed ``qsplit [H_kv, total_q * topK]`` (top-k slot << 24 | query) that K1
    reads, and ``split_counts [total_q, H_kv]`` for K2. A chunk with complete
    top-k splits skips the counts, since K2's ``FULL_SPLITS`` path uses
    ``TOPK`` directly.
    """
    if q2k_indices.ndim != 3:
        raise ValueError("q2k_indices must have shape [H_kv, total_q, topK].")
    if total_k <= 0:
        raise ValueError("total_k must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    h_kv, total_q, topk = q2k_indices.shape
    total_rows = (int(total_k) + int(block_size) - 1) // int(block_size)
    flat = q2k_indices.astype(mx.int32).reshape(h_kv, total_q * topk)
    valid = (flat >= 0) & (flat < total_rows)

    q_base = mx.arange(total_q, dtype=mx.int32)
    q_flat = mx.broadcast_to(q_base[None, :, None], (h_kv, total_q, topk))
    q_flat = q_flat.reshape(h_kv, total_q * topk)
    slot_base = mx.arange(topk, dtype=mx.int32)
    slot_flat = mx.broadcast_to(slot_base[None, None, :], (h_kv, total_q, topk))
    slot_flat = slot_flat.reshape(h_kv, total_q * topk)

    rows = mx.arange(total_rows, dtype=mx.int32)
    counts = mx.sum((flat[:, :, None] == rows[None, None, :]) & valid[:, :, None], axis=1)
    counts = counts.astype(mx.int32)
    zero = mx.zeros((h_kv, 1), dtype=mx.int32)
    row_ptr = mx.concatenate([zero, mx.cumsum(counts, axis=1)], axis=1)

    sort_stride = max(total_q, 1)
    invalid_key = mx.array(total_rows * sort_stride, dtype=mx.int32)
    sort_keys = mx.where(valid, flat * sort_stride + q_flat, invalid_key)
    sort_order = mx.argsort(sort_keys, axis=1)
    sorted_q = mx.take_along_axis(q_flat, sort_order, axis=1)
    sorted_slot = mx.take_along_axis(slot_flat, sort_order, axis=1)
    sorted_valid = mx.take_along_axis(valid, sort_order, axis=1)
    qsplit = sorted_slot * mx.array(1 << 24, dtype=mx.int32) + sorted_q
    qsplit = mx.where(sorted_valid, qsplit, mx.array(-1, dtype=mx.int32))
    split_counts = None
    if with_split_counts:
        split_counts = mx.sum(q2k_indices >= 0, axis=-1).astype(mx.int32).T
    return row_ptr, qsplit, split_counts


def msa_prefill_supported(head_dim: int, block_size: int, qhead_per_kv: int) -> bool:
    """Whether the K1 kernel covers this attention shape (MiniMax M3 and M3.1)."""
    return (
        _MSA_CSR_K1_STEEL_MMA is not None
        and head_dim == 128
        and block_size == 128
        and qhead_per_kv == 16
    )


def msa_sparse_attention_b1(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q2k_indices: mx.array,
    *,
    q_start: int,
    scale: float,
    block_size: int,
    full_splits: bool = False,
) -> mx.array:
    """MSA sparse attention for the B=1 prefill path (``total_q`` >= 8).

    The implementation follows the official K1/K2 contract:
    1. build ``k2q`` CSR from group-specific ``q2k``;
    2. K1 visits CSR KV-block rows and writes ``O_partial``/``LSE_partial`` by
       top-k split slot;
    3. K2 combines split partials with log-sum-exp correction.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, H_q, D].")
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError("k and v must have shape [total_k, H_kv, D].")
    if q2k_indices.ndim != 3:
        raise ValueError("q2k_indices must have shape [H_kv, total_q, topK].")

    total_q, h_q, dim = q.shape
    total_k, h_kv, k_dim = k.shape
    v_total_k, v_h_kv, v_dim = v.shape
    q2k_h, q2k_q, topk = q2k_indices.shape
    if (v_total_k, v_h_kv, k_dim, v_dim) != (total_k, h_kv, dim, dim):
        raise ValueError("K/V shapes must match q head dimension.")
    if (q2k_h, q2k_q) != (h_kv, total_q):
        raise ValueError("q2k_indices shape must match K/V heads and Q length.")
    if h_q % h_kv != 0:
        raise ValueError("H_q must be divisible by H_kv.")
    qhead_per_kv = h_q // h_kv
    if total_q < 8 or not msa_prefill_supported(dim, int(block_size), qhead_per_kv):
        raise RuntimeError(
            "MiniMax MSA prefill attention needs Metal with MLX's steel attention "
            "headers, D=128, block_size=128, 16 query heads per KV head and at "
            "least 8 query tokens."
        )

    row_ptr, qsplit, split_counts = build_k2q_csr_b1(
        q2k_indices,
        total_k=total_k,
        block_size=block_size,
        with_split_counts=not full_splits,
    )
    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    row_ptr = mx.contiguous(row_ptr)
    qsplit = mx.contiguous(qsplit)
    # K1 launches one threadgroup per group of 8 query edges of a block row,
    # only for rows that hold edges.
    q_tokens_per_group = 8
    groups = (row_ptr[:, 1:] - row_ptr[:, :-1] + q_tokens_per_group - 1) // (
        q_tokens_per_group
    )
    group_ptr = mx.concatenate(
        [mx.zeros((1,), dtype=mx.int32), mx.cumsum(groups.reshape(-1))]
    ).astype(mx.int32)
    if split_counts is None:
        split_counts_for_k2 = mx.zeros((1,), dtype=mx.int32)
        mx.eval(row_ptr, qsplit, group_ptr)
    else:
        split_counts_for_k2 = mx.contiguous(split_counts)
        mx.eval(row_ptr, qsplit, group_ptr, split_counts_for_k2)
    total_groups = max(int(group_ptr[-1].item()), 1)

    total_rows = (int(total_k) + int(block_size) - 1) // int(block_size)
    threadgroup_size = (q_tokens_per_group * qhead_per_kv // 8) * 32
    o_partial, lse_partial = _MSA_CSR_K1_STEEL_MMA(
        inputs=[q, k, v, row_ptr, qsplit, group_ptr, float(scale)],
        template=[
            ("T", q.dtype),
            ("TOTAL_Q", total_q),
            ("TOTAL_K", total_k),
            ("H_Q", h_q),
            ("H_KV", h_kv),
            ("D", dim),
            ("TOPK", topk),
            ("NNZ_CAP", total_q * topk),
            ("TOTAL_ROWS", total_rows),
            ("BLOCK_SIZE", int(block_size)),
            ("Q_START", int(q_start)),
            ("QHEAD_PER_KV", qhead_per_kv),
            ("THREADGROUP_SIZE", threadgroup_size),
            ("Q_TOKENS_PER_GROUP", q_tokens_per_group),
            ("M_BLOCK_SIZE", q_tokens_per_group * qhead_per_kv),
        ],
        grid=(total_groups * threadgroup_size, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[(topk, total_q, h_q, dim), (topk, total_q, h_q)],
        output_dtypes=[q.dtype, mx.float32],
    )

    return _MSA_CSR_K2(
        inputs=[o_partial, lse_partial, split_counts_for_k2],
        template=[
            ("T", q.dtype),
            ("TOTAL_Q", total_q),
            ("H_Q", h_q),
            ("H_KV", h_kv),
            ("D", dim),
            ("TOPK", topk),
            ("QHEAD_PER_KV", qhead_per_kv),
            ("FULL_SPLITS", 1 if full_splits else 0),
        ],
        grid=(q.size, 1, 1),
        threadgroup=(min(256, q.size), 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


# Block scores for decode and short verify calls. One threadgroup scores one
# key block of one row against every index head and query of that row, so
# each index key is read once and the [B, H, L, S] token scores never exist.
# Each thread owns one key of the block; positions follow the row's own
# coordinates (storage column = position + left padding). Scores are written
# negated so an ascending argpartition picks the top blocks directly.
_MSA_SMALL_L_BLOCK_SCORES = mx.fast.metal_kernel(
    name="minimax_m3_msa_small_l_block_scores",
    input_names=["q", "k", "qpos", "pad", "meta"],
    output_names=["out"],
    ensure_row_contiguous=False,
    source=r"""
        constexpr int HL = H * L;
        uint blk = threadgroup_position_in_grid.x;
        uint b = threadgroup_position_in_grid.y;
        uint t = thread_index_in_threadgroup;
        uint simd = t / 32;
        uint lane = t % 32;
        int total_len = meta[0];
        int num_blocks = meta[1];

        threadgroup T q_s[HL * D];
        threadgroup float red_s[(BLOCK / 32) * HL];

        for (uint i = t; i < uint(HL * D); i += BLOCK) {
            q_s[i] = q[b * HL * D + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int pos = int(blk) * BLOCK + int(t);
        int col = pos + (HAS_PAD ? int(pad[b]) : 0);
        bool stored = col < total_len;
        int key_offset = int(b) * k_strides[0] + col * k_strides[2];

        float acc[HL];
        for (int i = 0; i < HL; ++i) {
            acc[i] = 0.0f;
        }
        for (int d0 = 0; d0 < D; d0 += 32) {
            float kc[32];
            for (int j = 0; j < 32; ++j) {
                kc[j] = stored ? float(k[key_offset + (d0 + j) * k_strides[3]]) : 0.0f;
            }
            for (int i = 0; i < HL; ++i) {
                float s = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    s += float(q_s[i * D + d0 + j]) * kc[j];
                }
                acc[i] += s;
            }
        }

        for (int i = 0; i < HL; ++i) {
            int l = i % L;
            bool visible = stored && pos <= qpos[b * L + l];
            float score = visible ? acc[i] : -INFINITY;
            score = simd_max(score);
            if (lane == 0) {
                red_s[simd * HL + i] = score;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (t < uint(HL)) {
            int i = int(t);
            int h = i / L;
            int l = i % L;
            float score = -INFINITY;
            for (int s = 0; s < BLOCK / 32; ++s) {
                score = max(score, red_s[s * HL + i]);
            }
            if (score != score) {
                score = -INFINITY;
            }
            int cur = qpos[b * L + l] / BLOCK;
            int block = int(blk);
            if (block > cur) {
                score = -INFINITY;
            } else if (LOCAL_BLOCKS > 0 && block >= cur - LOCAL_BLOCKS + 1) {
                score = 1.0e29f;
            } else if (INIT_BLOCKS > 0 && block < INIT_BLOCKS) {
                score = 1.0e30f;
            }
            out[((int(b) * H + h) * L + l) * num_blocks + block] = -score;
        }
    """,
)

SMALL_L_BLOCK_SCORES_MAX_QUERIES = 16


def _row_pad(left_padding: Optional[mx.array], batch: int, dummy: mx.array):
    # Kernels read the padding only when HAS_PAD is set; without it any
    # existing int32 array fills the slot and no zeros array is built.
    if left_padding is None:
        return dummy
    return mx.contiguous(left_padding.reshape(-1)[:batch].astype(mx.int32))


def small_l_block_scores(
    idx_queries: mx.array,
    idx_keys: mx.array,
    qpos: mx.array,
    left_padding: Optional[mx.array],
    block_size: int,
    init_blocks: int,
    local_blocks: int,
) -> Optional[mx.array]:
    """Negated forced block scores ``[B, H_idx, L, num_blocks]``, or None.

    Blocks after the query's current block score ``-inf``; the local blocks
    score ``1e29`` and init blocks ``1e30``, matching the selection rule. The
    result is negated, ready for an ascending ``argpartition``.
    """
    B, H, L, D = idx_queries.shape
    if (
        not mx.metal.is_available()
        or L > SMALL_L_BLOCK_SCORES_MAX_QUERIES
        or idx_keys.shape[1] != 1
        or block_size % 32
        or block_size > 1024
        or D % 32
        or H * L * D * idx_queries.dtype.size > 16384
    ):
        return None
    total_len = idx_keys.shape[2]
    num_blocks = (total_len + block_size - 1) // block_size
    qpos = mx.contiguous(qpos.astype(mx.int32))
    return _MSA_SMALL_L_BLOCK_SCORES(
        inputs=[
            mx.contiguous(idx_queries),
            idx_keys,
            qpos,
            _row_pad(left_padding, B, qpos),
            mx.array([total_len, num_blocks], dtype=mx.int32),
        ],
        template=[
            ("T", idx_queries.dtype),
            ("H", H),
            ("L", L),
            ("D", D),
            ("BLOCK", block_size),
            ("INIT_BLOCKS", init_blocks),
            ("LOCAL_BLOCKS", local_blocks),
            ("HAS_PAD", left_padding is not None),
        ],
        grid=(num_blocks * block_size, B, 1),
        threadgroup=(block_size, 1, 1),
        output_shapes=[(B, H, L, num_blocks)],
        output_dtypes=[mx.float32],
    )[0]


# Sparse attention for decode and short verify calls, split over the selected
# blocks. Pass 1 runs one threadgroup per (row, KV head, query, selected
# block): it scores the block's keys for the KV head's query heads, keeps the
# block-local softmax max and sum, and accumulates P @ V. Pass 2 merges the
# per-block partials with log-sum-exp. Every query row is computed the same
# way whatever L is, so a verify row matches a one-token decode step.
_MSA_SMALL_L_PARTIALS = mx.fast.metal_kernel(
    name="minimax_m3_msa_small_l_partials",
    input_names=["q", "k", "v", "block_ids", "qpos", "pad", "meta", "scale", "vk"],
    output_names=["part_o", "part_m", "part_l"],
    ensure_row_contiguous=False,
    source=r"""
        uint slot = threadgroup_position_in_grid.x;
        uint kl = threadgroup_position_in_grid.y;
        uint b = threadgroup_position_in_grid.z;
        uint t = thread_index_in_threadgroup;
        uint simd = t / 32;
        uint lane = t % 32;
        int kv = int(kl) / L;
        int l = int(kl) % L;
        int total_len = meta[0];
        float sm_scale = scale[0];

        threadgroup float4 q_s[G * D / 4];
        threadgroup float p_s[G * BLOCK];
        threadgroup int col_s[BLOCK];
        threadgroup float m_s[G];
        threadgroup float l_s[G];

        for (uint i = t; i < uint(G * D / 4); i += BLOCK) {
            int h = int(i) / (D / 4);
            int q_off = ((int(b) * H_KV * G + kv * G + h) * L + l) * D + (int(i) % (D / 4)) * 4;
            q_s[i] = float4(
                float(q[q_off]), float(q[q_off + 1]), float(q[q_off + 2]), float(q[q_off + 3])) * sm_scale;
        }

        int row = (int(b) * H_KV + kv) * L + l;
        int blk = block_ids[row * TOPK + int(slot)];
        int pos = blk * BLOCK + int(t);
        int col = pos + (HAS_PAD ? int(pad[b]) : 0);
        bool valid = blk >= 0 && pos <= qpos[int(b) * L + l] && col < total_len;
        if (HAS_MASK && valid) {
            valid = vk[int(b) * vk_strides[0] + l * vk_strides[2] + col * vk_strides[3]];
        }
        // Skipped keys get probability 0 below; pointing them at column 0
        // keeps the value loads in bounds and branch-free.
        col_s[t] = valid ? col : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[G];
        for (int h = 0; h < G; ++h) {
            s[h] = 0.0f;
        }
        int k_off = int(b) * k_strides[0] + kv * k_strides[1] + col_s[t] * k_strides[2];
        int ks = k_strides[3];
        for (int d0 = 0; d0 < D; d0 += 32) {
            float4 kc[8];
            for (int j = 0; j < 8; ++j) {
                int o = k_off + (d0 + 4 * j) * ks;
                kc[j] = float4(float(k[o]), float(k[o + ks]), float(k[o + 2 * ks]), float(k[o + 3 * ks]));
            }
            for (int h = 0; h < G; ++h) {
                float acc = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    acc += dot(q_s[(h * D + d0) / 4 + j], kc[j]);
                }
                s[h] += acc;
            }
        }
        for (int h = 0; h < G; ++h) {
            p_s[h * BLOCK + int(t)] = valid ? s[h] : -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int h = int(simd); h < G; h += BLOCK / 32) {
            float m = -INFINITY;
            for (int j = int(lane); j < BLOCK; j += 32) {
                m = max(m, p_s[h * BLOCK + j]);
            }
            m = simd_max(m);
            float sum = 0.0f;
            for (int j = int(lane); j < BLOCK; j += 32) {
                float e = m == -INFINITY ? 0.0f : fast::exp(p_s[h * BLOCK + j] - m);
                p_s[h * BLOCK + j] = e;
                sum += e;
            }
            sum = simd_sum(sum);
            if (lane == 0) {
                m_s[h] = m;
                l_s[h] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Partials are stored in block order, so the merge does not depend on
        // the order the selection returned the blocks in.
        int rank = 0;
        for (int r = 0; r < TOPK; ++r) {
            int other = block_ids[row * TOPK + r];
            rank += other < blk || (other == blk && r < int(slot));
        }
        int part = (row * TOPK + rank) * G;
        for (int d = int(t); d < D; d += BLOCK) {
            float acc[G];
            for (int h = 0; h < G; ++h) {
                acc[h] = 0.0f;
            }
            int v_base = int(b) * v_strides[0] + kv * v_strides[1] + d * v_strides[3];
            for (int j0 = 0; j0 < BLOCK; j0 += 8) {
                float vv[8];
                for (int u = 0; u < 8; ++u) {
                    vv[u] = float(v[v_base + col_s[j0 + u] * v_strides[2]]);
                }
                for (int u = 0; u < 8; ++u) {
                    for (int h = 0; h < G; ++h) {
                        acc[h] += p_s[h * BLOCK + j0 + u] * vv[u];
                    }
                }
            }
            for (int h = 0; h < G; ++h) {
                part_o[(part + h) * D + d] = acc[h];
            }
        }
        if (t < uint(G)) {
            part_m[part + int(t)] = m_s[t];
            part_l[part + int(t)] = l_s[t];
        }
    """,
)

_MSA_SMALL_L_COMBINE = mx.fast.metal_kernel(
    name="minimax_m3_msa_small_l_combine",
    input_names=["part_o", "part_m", "part_l"],
    output_names=["out"],
    source=r"""
        uint d = thread_position_in_grid.x;
        uint hl = thread_position_in_grid.y;
        uint b = thread_position_in_grid.z;
        if (d >= uint(D)) {
            return;
        }
        int hq = int(hl) / L;
        int l = int(hl) % L;
        int kv = hq / G;
        int h = hq % G;
        int row = (int(b) * H_KV + kv) * L + l;
        float m_max = -INFINITY;
        for (int s = 0; s < TOPK; ++s) {
            m_max = max(m_max, part_m[(row * TOPK + s) * G + h]);
        }
        float num = 0.0f;
        float den = 0.0f;
        if (m_max != -INFINITY) {
            for (int s = 0; s < TOPK; ++s) {
                int p = (row * TOPK + s) * G + h;
                float w = fast::exp(part_m[p] - m_max);
                num += w * part_o[p * D + int(d)];
                den += w * part_l[p];
            }
        }
        out[((int(b) * H_KV * G + hq) * L + l) * D + int(d)] =
            T(den > 0.0f ? num / den : 0.0f);
    """,
)


def small_l_sparse_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    qpos: mx.array,
    left_padding: Optional[mx.array],
    block_size: int,
    scale: float,
    valid_keys: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """``[B, H, L, D]`` sparse attention over each query's own blocks, or None.

    ``block_ids`` is ``[B, H_kv, L, topk]`` (or one shared index head) in the
    row's own positions. Keys after the query position, outside the row or
    dropped by the ``[B, 1, L, S]`` ``valid_keys`` mask are skipped.
    """
    B, H, L, D = queries.shape
    _, H_kv, total_len, _ = keys.shape
    topk = block_ids.shape[-1]
    if (
        not mx.metal.is_available()
        or L > SMALL_L_BLOCK_SCORES_MAX_QUERIES
        or block_ids.shape[1] not in (1, H_kv)
        or H % H_kv
        or block_size % 32
        or block_size > 1024
        or D % 32
        or (H // H_kv) * (D + block_size) * 4 > 24576
    ):
        return None
    G = H // H_kv
    if block_ids.shape[1] != H_kv:
        block_ids = mx.broadcast_to(block_ids, (B, H_kv, L, topk))
    block_ids = mx.contiguous(block_ids.astype(mx.int32))
    qpos = mx.contiguous(qpos.astype(mx.int32))
    if valid_keys is not None:
        valid_keys = mx.broadcast_to(valid_keys, (B, 1, L, total_len))
    template = [
        ("T", queries.dtype),
        ("H_KV", H_kv),
        ("G", G),
        ("L", L),
        ("D", D),
        ("BLOCK", block_size),
        ("TOPK", topk),
        ("HAS_MASK", valid_keys is not None),
        ("HAS_PAD", left_padding is not None),
    ]
    rows = B * H_kv * L * topk * G
    part_o, part_m, part_l = _MSA_SMALL_L_PARTIALS(
        inputs=[
            mx.contiguous(queries),
            keys,
            values,
            block_ids,
            qpos,
            _row_pad(left_padding, B, qpos),
            mx.array([total_len], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
            qpos if valid_keys is None else valid_keys,
        ],
        template=template,
        grid=(topk * block_size, H_kv * L, B),
        threadgroup=(block_size, 1, 1),
        output_shapes=[(rows, D), (rows,), (rows,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _MSA_SMALL_L_COMBINE(
        inputs=[part_o, part_m, part_l],
        template=template,
        grid=(D, H * L, B),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, H, L, D)],
        output_dtypes=[queries.dtype],
    )[0]
