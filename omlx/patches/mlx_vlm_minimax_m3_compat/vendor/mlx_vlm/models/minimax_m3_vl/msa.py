from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mlx.core as mx

try:
    from omlx.custom_kernels.minimax_m3 import fast as _minimax_fast
except Exception:  # pragma: no cover - optional native extension
    _minimax_fast = mx.fast


_NATIVE_TOPK_SELECT_MODE = os.environ.get(
    "MLX_MINIMAX_MSA_NATIVE_TOPK_SELECT", "auto"
).lower()
_NATIVE_MSA_TOPK_MODE = os.environ.get(
    "MLX_MINIMAX_MSA_NATIVE_TOPK", "auto"
).lower()


def _has_minimax_fast_symbol(name: str) -> bool:
    has_symbol = getattr(_minimax_fast, "has_symbol", None)
    if callable(has_symbol):
        return bool(has_symbol(name))
    return hasattr(_minimax_fast, name)


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
        input_names=["q", "k", "v", "row_ptr", "qsplit", "scale"],
        output_names=["o_partial", "lse_partial"],
        header=header,
        source=r"""
            constexpr short BQ = M_BLOCK_SIZE;
            constexpr short BK = 128;
            constexpr short BD = 128;
            constexpr short WM = M_BLOCK_SIZE / 8;
            constexpr short WN = 1;
            constexpr short THREADS = WM * WN * 32;
            uint work = threadgroup_position_in_grid.x;
            if (work >= H_KV * TOTAL_ROWS * GROUPS_PER_ROW_CAP) {
                return;
            }

            int group_id = int(work % GROUPS_PER_ROW_CAP);
            int row = int((work / GROUPS_PER_ROW_CAP) % TOTAL_ROWS);
            int hkv = int(work / (GROUPS_PER_ROW_CAP * TOTAL_ROWS));
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
    """Select official-style sparse block Top-k from block scores."""
    if block_scores.ndim != 4:
        raise ValueError("block_scores must have shape [B, H, L, num_blocks].")
    if topk != 16:
        return _select_msa_topk_from_block_scores_fallback(
            block_scores,
            q_start=q_start,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
    if not mx.metal.is_available():
        return _select_msa_topk_from_block_scores_fallback(
            block_scores,
            q_start=q_start,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )

    B, H, L, num_blocks = block_scores.shape
    force_native = _NATIVE_TOPK_SELECT_MODE in {"1", "true", "yes", "on"}
    force_fallback = _NATIVE_TOPK_SELECT_MODE in {"0", "false", "no", "off"}
    use_native = force_native or (
        not force_fallback and num_blocks <= 256
    )
    if not use_native:
        return _select_msa_topk_from_block_scores_fallback(
            block_scores,
            q_start=q_start,
            block_size=block_size,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )

    threads = 256
    rows = B * H * L
    block_scores = mx.contiguous(block_scores.astype(mx.float32))
    return _MSA_TOPK_SELECT(
        inputs=[block_scores],
        template=[
            ("B", B),
            ("H", H),
            ("L", L),
            ("ROWS", rows),
            ("NUM_BLOCKS", num_blocks),
            ("TOPK", int(topk)),
            ("Q_START", int(q_start)),
            ("BLOCK_SIZE", int(block_size)),
            ("INIT_BLOCKS", int(init_blocks)),
            ("LOCAL_BLOCKS", int(local_blocks)),
            ("THREADS", threads),
        ],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(B, H, L, topk)],
        output_dtypes=[mx.int32],
    )[0]


def _select_msa_topk_from_block_scores_fallback(
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
    force_native = _NATIVE_MSA_TOPK_MODE in {"1", "true", "yes", "on"}
    force_fallback = _NATIVE_MSA_TOPK_MODE in {"0", "false", "no", "off"}
    if force_fallback:
        return None
    if not _has_minimax_fast_symbol("minimax_msa_topk"):
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
        return _minimax_fast.minimax_msa_topk(
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
        if force_native:
            raise
        return None


_MSA_CSR_K1_SCALAR = mx.fast.metal_kernel(
    name="minimax_m3_msa_csr_k1_scalar",
    input_names=["q", "k", "v", "row_ptr", "qsplit", "scale"],
    output_names=["o_partial", "lse_partial"],
    source=r"""
        uint work = threadgroup_position_in_grid.x;
        uint tid = thread_index_in_threadgroup;
        uint total_work = H_KV * NNZ_CAP;
        if (work >= total_work) {
            return;
        }

        int edge = work % NNZ_CAP;
        int hkv = work / NNZ_CAP;

        int qsplit_value = qsplit[hkv * NNZ_CAP + edge];
        if (qsplit_value < 0) {
            return;
        }

        int q_idx = qsplit_value & 0x00FFFFFF;
        int split_idx = (qsplit_value >> 24) & 0xFF;

        int row_base = hkv * (TOTAL_ROWS + 1);
        int left = 0;
        int right = TOTAL_ROWS;
        for (int step = 0; step < 32; ++step) {
            if (left < right) {
                int mid = (left + right) >> 1;
                int row_end = row_ptr[row_base + mid + 1];
                if (row_end <= edge) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
        }
        int kv_block = left;
        int block_start = kv_block * BLOCK_SIZE;
        int q_abs = Q_START + q_idx;

        threadgroup float scores[QHEAD_PER_KV * BLOCK_SIZE];
        threadgroup float row_maxes[QHEAD_PER_KV];
        threadgroup float denoms[QHEAD_PER_KV];

        int score_cells = QHEAD_PER_KV * BLOCK_SIZE;
        for (int cell = int(tid); cell < score_cells; cell += THREADGROUP_SIZE) {
            int h_in_group = cell / BLOCK_SIZE;
            int s = cell - h_in_group * BLOCK_SIZE;
            int k_pos = block_start + s;
            if (k_pos >= TOTAL_K || k_pos > q_abs) {
                scores[cell] = -INFINITY;
                continue;
            }
            int hq = hkv * QHEAD_PER_KV + h_in_group;
            int q_base = (q_idx * H_Q + hq) * D;
            int k_base = (k_pos * H_KV + hkv) * D;
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += float(q[q_base + d]) * float(k[k_base + d]);
            }
            scores[cell] = score * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < QHEAD_PER_KV) {
            int h_in_group = int(tid);
            int score_base = h_in_group * BLOCK_SIZE;
            float row_max = -INFINITY;
            for (int s = int(tid); s < BLOCK_SIZE; s += THREADGROUP_SIZE) {
                row_max = max(row_max, scores[score_base + s]);
            }
            for (int s = 0; s < BLOCK_SIZE; ++s) {
                row_max = max(row_max, scores[score_base + s]);
            }
            float denom = 0.0f;
            if (row_max != -INFINITY) {
                for (int s = 0; s < BLOCK_SIZE; ++s) {
                    float score = scores[score_base + s];
                    if (score != -INFINITY) {
                        denom += metal::exp(score - row_max);
                    }
                }
            }
            row_maxes[h_in_group] = row_max;
            denoms[h_in_group] = denom;
            int hq = hkv * QHEAD_PER_KV + h_in_group;
            int lse_idx = (split_idx * TOTAL_Q + q_idx) * H_Q + hq;
            lse_partial[lse_idx] = row_max == -INFINITY ? -INFINITY : row_max + metal::log(denom);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int out_cells = QHEAD_PER_KV * D;
        for (int cell = int(tid); cell < out_cells; cell += THREADGROUP_SIZE) {
            int h_in_group = cell / D;
            int d = cell - h_in_group * D;
            int hq = hkv * QHEAD_PER_KV + h_in_group;
            float row_max = row_maxes[h_in_group];
            float denom = denoms[h_in_group];
            float acc = 0.0f;
            if (denom > 0.0f) {
                int score_base = h_in_group * BLOCK_SIZE;
                for (int s = 0; s < BLOCK_SIZE; ++s) {
                    float score = scores[score_base + s];
                    if (score == -INFINITY) {
                        continue;
                    }
                    int k_pos = block_start + s;
                    int v_base = (k_pos * H_KV + hkv) * D;
                    float weight = metal::exp(score - row_max) / denom;
                    acc += weight * float(v[v_base + d]);
                }
            }
            int partial_idx = ((split_idx * TOTAL_Q + q_idx) * H_Q + hq) * D + d;
            o_partial[partial_idx] = T(acc);
        }
    """,
)


_MSA_CSR_K1_SIMD = mx.fast.metal_kernel(
    name="minimax_m3_msa_csr_k1_simd",
    input_names=["q", "k", "v", "row_ptr", "qsplit", "scale"],
    output_names=["o_partial", "lse_partial"],
    header="#include <metal_simdgroup>\nusing namespace metal;\n",
    source=r"""
        uint work = threadgroup_position_in_grid.x;
        uint total_work = H_KV * NNZ_CAP;
        if (work >= total_work) {
            return;
        }

        int edge = work % NNZ_CAP;
        int hkv = work / NNZ_CAP;
        int qsplit_value = qsplit[hkv * NNZ_CAP + edge];
        if (qsplit_value < 0) {
            return;
        }

        int q_idx = qsplit_value & 0x00FFFFFF;
        int split_idx = (qsplit_value >> 24) & 0xFF;

        int row_base = hkv * (TOTAL_ROWS + 1);
        int left = 0;
        int right = TOTAL_ROWS;
        for (int step = 0; step < 32; ++step) {
            if (left < right) {
                int mid = (left + right) >> 1;
                int row_end = row_ptr[row_base + mid + 1];
                if (row_end <= edge) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
        }
        int kv_block = left;
        int block_start = kv_block * BLOCK_SIZE;
        int q_abs = Q_START + q_idx;

        int h_in_group = int(simdgroup_index_in_threadgroup);
        int lane = int(thread_index_in_simdgroup);
        if (h_in_group >= QHEAD_PER_KV) {
            return;
        }
        int hq = hkv * QHEAD_PER_KV + h_in_group;

        thread float q_frag[D_PER_LANE];
        thread float out_frag[D_PER_LANE];
        int dim_base = lane * D_PER_LANE;
        int q_base = (q_idx * H_Q + hq) * D + dim_base;
        for (int d = 0; d < D_PER_LANE; ++d) {
            q_frag[d] = float(q[q_base + d]);
            out_frag[d] = 0.0f;
        }

        float row_max = -INFINITY;
        float denom = 0.0f;
        for (int s = 0; s < BLOCK_SIZE; ++s) {
            int k_pos = block_start + s;
            bool valid = k_pos < TOTAL_K && k_pos <= q_abs;
            float partial = 0.0f;
            if (valid) {
                int k_base = (k_pos * H_KV + hkv) * D + dim_base;
                for (int d = 0; d < D_PER_LANE; ++d) {
                    partial += q_frag[d] * float(k[k_base + d]);
                }
            }
            float score = simd_sum(partial) * scale;
            if (valid) {
                float new_max = max(row_max, score);
                float old_scale = fast::exp(row_max - new_max);
                float weight = fast::exp(score - new_max);
                int v_base = (k_pos * H_KV + hkv) * D + dim_base;
                for (int d = 0; d < D_PER_LANE; ++d) {
                    out_frag[d] =
                        out_frag[d] * old_scale + weight * float(v[v_base + d]);
                }
                denom = denom * old_scale + weight;
                row_max = new_max;
            }
        }

        int partial_base = ((split_idx * TOTAL_Q + q_idx) * H_Q + hq) * D + dim_base;
        float denom_rcp = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int d = 0; d < D_PER_LANE; ++d) {
            o_partial[partial_base + d] = T(out_frag[d] * denom_rcp);
        }
        if (lane == 0) {
            int lse_idx = (split_idx * TOTAL_Q + q_idx) * H_Q + hq;
            lse_partial[lse_idx] =
                denom > 0.0f ? row_max + metal::log(denom) : -INFINITY;
        }
    """,
)


_MSA_CSR_K1_SIMD_PACKED = mx.fast.metal_kernel(
    name="minimax_m3_msa_csr_k1_simd_packed",
    input_names=["q", "k", "v", "row_ptr", "qsplit", "scale"],
    output_names=["o_partial", "lse_partial"],
    header="#include <metal_simdgroup>\nusing namespace metal;\n",
    source=r"""
        uint work = threadgroup_position_in_grid.x;
        uint total_work = H_KV * TOTAL_ROWS * GROUPS_PER_ROW_CAP;
        if (work >= total_work) {
            return;
        }

        int group_id = int(work % GROUPS_PER_ROW_CAP);
        int row = int((work / GROUPS_PER_ROW_CAP) % TOTAL_ROWS);
        int hkv = int(work / (GROUPS_PER_ROW_CAP * TOTAL_ROWS));

        int row_base = hkv * (TOTAL_ROWS + 1);
        int row_start = row_ptr[row_base + row];
        int row_end = row_ptr[row_base + row + 1];
        int edge_base = row_start + group_id * Q_TOKENS_PER_GROUP;
        if (edge_base >= row_end) {
            return;
        }

        int block_start = row * BLOCK_SIZE;
        int h_in_group = int(simdgroup_index_in_threadgroup);
        int lane = int(thread_index_in_simdgroup);
        if (h_in_group >= QHEAD_PER_KV) {
            return;
        }
        int hq = hkv * QHEAD_PER_KV + h_in_group;
        int dim_base = lane * D_PER_LANE;

        thread float q_frag[D_PER_LANE];
        thread float out_frag[D_PER_LANE];

        for (int qt = 0; qt < Q_TOKENS_PER_GROUP; ++qt) {
            int edge = edge_base + qt;
            if (edge >= row_end) {
                break;
            }
            int qsplit_value = qsplit[hkv * NNZ_CAP + edge];
            if (qsplit_value < 0) {
                continue;
            }

            int q_idx = qsplit_value & 0x00FFFFFF;
            int split_idx = (qsplit_value >> 24) & 0xFF;
            int q_abs = Q_START + q_idx;
            int q_base = (q_idx * H_Q + hq) * D + dim_base;
            for (int d = 0; d < D_PER_LANE; ++d) {
                q_frag[d] = float(q[q_base + d]);
                out_frag[d] = 0.0f;
            }

            float row_max = -INFINITY;
            float denom = 0.0f;
            for (int s = 0; s < BLOCK_SIZE; ++s) {
                int k_pos = block_start + s;
                bool valid = k_pos < TOTAL_K && k_pos <= q_abs;
                float partial = 0.0f;
                if (valid) {
                    int k_base = (k_pos * H_KV + hkv) * D + dim_base;
                    for (int d = 0; d < D_PER_LANE; ++d) {
                        partial += q_frag[d] * float(k[k_base + d]);
                    }
                }
                float score = simd_sum(partial) * scale;
                if (valid) {
                    float new_max = max(row_max, score);
                    float old_scale = fast::exp(row_max - new_max);
                    float weight = fast::exp(score - new_max);
                    int v_base = (k_pos * H_KV + hkv) * D + dim_base;
                    for (int d = 0; d < D_PER_LANE; ++d) {
                        out_frag[d] =
                            out_frag[d] * old_scale + weight * float(v[v_base + d]);
                    }
                    denom = denom * old_scale + weight;
                    row_max = new_max;
                }
            }

            int partial_base =
                ((split_idx * TOTAL_Q + q_idx) * H_Q + hq) * D + dim_base;
            float denom_rcp = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (int d = 0; d < D_PER_LANE; ++d) {
                o_partial[partial_base + d] = T(out_frag[d] * denom_rcp);
            }
            if (lane == 0) {
                int lse_idx = (split_idx * TOTAL_Q + q_idx) * H_Q + hq;
                lse_partial[lse_idx] =
                    denom > 0.0f ? row_max + metal::log(denom) : -INFINITY;
            }
        }
    """,
)


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


_MSA_TOPK_SELECT = mx.fast.metal_kernel(
    name="minimax_m3_msa_topk_select",
    input_names=["block_scores"],
    output_names=["topk_idx"],
    source=r"""
        uint row = threadgroup_position_in_grid.x;
        uint tid = thread_index_in_threadgroup;
        if (row >= ROWS) {
            return;
        }

        int q_idx = int(row % L);
        int h = int((row / L) % H);
        int b = int(row / (L * H));
        int q_abs = Q_START + q_idx;
        int cur_block = q_abs / BLOCK_SIZE;
        int local_start = cur_block - LOCAL_BLOCKS + 1;
        if (local_start < 0) {
            local_start = 0;
        }

        threadgroup float scores_s[THREADS];
        threadgroup int indices_s[THREADS];
        threadgroup int selected_s[TOPK];

        if (tid < TOPK) {
            selected_s[tid] = NUM_BLOCKS;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int slot = 0; slot < TOPK; ++slot) {
            float best_score = -INFINITY;
            int best_idx = NUM_BLOCKS;
            for (int block = int(tid); block < NUM_BLOCKS; block += THREADS) {
                bool valid = block <= cur_block;
                bool already = false;
                for (int prev = 0; prev < slot; ++prev) {
                    already = already || (selected_s[prev] == block);
                }
                if (!valid || already) {
                    continue;
                }

                int offset = ((b * H + h) * L + q_idx) * NUM_BLOCKS + block;
                float score = float(block_scores[offset]);
                if (score != score) {
                    score = -INFINITY;
                }
                if (INIT_BLOCKS > 0 && block < INIT_BLOCKS) {
                    score = 1.0e30f;
                }
                if (
                    LOCAL_BLOCKS > 0
                    && block >= local_start
                    && block <= cur_block
                ) {
                    score = 1.0e29f;
                }
                if (score > best_score || (score == best_score && block < best_idx)) {
                    best_score = score;
                    best_idx = block;
                }
            }

            scores_s[tid] = best_score;
            indices_s[tid] = best_idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    float other_score = scores_s[tid + stride];
                    int other_idx = indices_s[tid + stride];
                    float cur_score = scores_s[tid];
                    int cur_idx = indices_s[tid];
                    if (
                        other_score > cur_score
                        || (other_score == cur_score && other_idx < cur_idx)
                    ) {
                        scores_s[tid] = other_score;
                        indices_s[tid] = other_idx;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                selected_s[slot] = indices_s[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            for (int i = 0; i < TOPK; ++i) {
                for (int j = i + 1; j < TOPK; ++j) {
                    if (selected_s[j] < selected_s[i]) {
                        int tmp = selected_s[i];
                        selected_s[i] = selected_s[j];
                        selected_s[j] = tmp;
                    }
                }
            }

            int out_base = row * TOPK;
            for (int i = 0; i < TOPK; ++i) {
                int idx = selected_s[i];
                topk_idx[out_base + i] = idx < NUM_BLOCKS ? idx : -1;
            }
        }
    """,
)


_MSA_DECODE_B1_SIMD = mx.fast.metal_kernel(
    name="minimax_m3_msa_decode_b1_simd",
    input_names=["q", "k", "v", "topk_idx", "topk_valid", "scale"],
    output_names=["out"],
    header="#include <metal_simdgroup>\nusing namespace metal;\n",
    source=r"""
        uint hq = threadgroup_position_in_grid.x;
        int lane = int(thread_index_in_simdgroup);
        if (hq >= H_Q) {
            return;
        }

        int hkv = int(hq) / QHEAD_PER_KV;
        int topk_head = TOPK_HEADS == 1 ? 0 : hkv;
        int dim_base = lane * D_PER_LANE;

        thread float q_frag[D_PER_LANE];
        thread float out_frag[D_PER_LANE];
        int q_base = int(hq) * D + dim_base;
        for (int d = 0; d < D_PER_LANE; ++d) {
            q_frag[d] = float(q[q_base + d]);
            out_frag[d] = 0.0f;
        }

        float row_max = -INFINITY;
        float denom = 0.0f;
        for (int slot = 0; slot < TOPK; ++slot) {
            int meta_idx = (topk_head * TOPK) + slot;
            bool block_valid = bool(topk_valid[meta_idx]);
            int block = topk_idx[meta_idx];
            if (!block_valid || block < 0) {
                continue;
            }
            int block_start = block * BLOCK_SIZE;
            for (int s = 0; s < BLOCK_SIZE; ++s) {
                int k_pos = block_start + s;
                bool valid = k_pos < TOTAL_K && k_pos <= Q_POS;
                float partial = 0.0f;
                if (valid) {
                    int k_base = (hkv * TOTAL_K + k_pos) * D + dim_base;
                    for (int d = 0; d < D_PER_LANE; ++d) {
                        partial += q_frag[d] * float(k[k_base + d]);
                    }
                }
                float score = simd_sum(partial) * scale;
                if (valid) {
                    float new_max = max(row_max, score);
                    float old_scale = fast::exp(row_max - new_max);
                    float weight = fast::exp(score - new_max);
                    int v_base = (hkv * TOTAL_K + k_pos) * D + dim_base;
                    for (int d = 0; d < D_PER_LANE; ++d) {
                        out_frag[d] =
                            out_frag[d] * old_scale + weight * float(v[v_base + d]);
                    }
                    denom = denom * old_scale + weight;
                    row_max = new_max;
                }
            }
        }

        float denom_rcp = denom > 0.0f ? 1.0f / denom : 0.0f;
        int out_base = int(hq) * D + dim_base;
        for (int d = 0; d < D_PER_LANE; ++d) {
            out[out_base + d] = T(out_frag[d] * denom_rcp);
        }
    """,
)


@mx.compile
def build_grouped_msa_topk(
    idx_queries: mx.array,
    idx_keys: mx.array,
    q_start: int,
    scale: float,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> mx.array:
    """Build MiniMax MSA q2k block indices per GQA/KV group.

    This matches the official sparse metadata contract:
    ``q2k_indices [B, H_kv, L, topK]`` with invalid entries padded by ``-1``.
    It covers the causal text-prefill path used by MiniMax M3 serving.
    """
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
    neg = mx.array(-float("inf"), dtype=mx.float32)

    scores = mx.matmul(
        idx_queries.astype(mx.float32),
        idx_keys.astype(mx.float32).swapaxes(-1, -2),
    )
    scores = scores * scale

    qpos = mx.arange(q_start, q_start + L)
    kpos = mx.arange(total_len)
    causal = kpos[None, None, None, :] <= qpos[None, None, :, None]
    scores = mx.where(causal, scores, neg)

    num_blocks = (total_len + block_size - 1) // block_size
    pad = num_blocks * block_size - total_len
    if pad:
        pad_values = mx.full(
            (*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype
        )
        scores = mx.concatenate([scores, pad_values], axis=-1)

    scores = scores.reshape(B, H_idx, L, num_blocks, block_size)
    block_scores = mx.max(scores, axis=-1)
    block_scores = mx.where(block_scores == block_scores, block_scores, neg)

    return _select_msa_topk_from_block_scores(
        block_scores,
        q_start=q_start,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


def build_grouped_msa_topk_blockwise(
    idx_queries: mx.array,
    idx_keys: mx.array,
    q_start: int,
    scale: float,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    *,
    block_chunk_size: int = 16,
) -> mx.array:
    """Memory-bounded MiniMax MSA q2k block top-k builder.

    The result matches :func:`build_grouped_msa_topk`, but it never materializes
    the full token score tensor ``[B, H_kv, L, total_k]``.  Instead, it computes
    scores for a small number of KV blocks, immediately reduces each block to
    a block score, then concatenates only ``[B, H_kv, L, num_blocks]`` scores.
    """
    if idx_queries.ndim != 4 or idx_keys.ndim != 4:
        raise ValueError("idx_queries/idx_keys must have shape [B, H, L, D].")
    if block_chunk_size <= 0:
        raise ValueError("block_chunk_size must be positive.")

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
    neg = mx.array(-float("inf"), dtype=mx.float32)

    q = idx_queries.astype(mx.float32)
    qpos = mx.arange(q_start, q_start + L)
    block_score_chunks = []
    for block_start in range(0, num_blocks, block_chunk_size):
        blocks_in_chunk = min(block_chunk_size, num_blocks - block_start)
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
            pad_values = mx.full((*scores.shape[:-1], pad), -float("inf"), dtype=scores.dtype)
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
    return_qsplits: bool = False,
    return_q_indices: bool = True,
    return_split_counts: bool = True,
) -> (
    tuple[mx.array, mx.array]
    | tuple[mx.array, Optional[mx.array], mx.array, Optional[mx.array]]
):
    """Build official-style k2q CSR metadata for the B=1 prefill hot path.

    Parameters follow the official MSA sparse-index contract:
    ``q2k_indices [H_kv, total_q, topK]`` contains batch-local KV block indices
    with ``-1`` padding. The returned base tensors are ``k2q_row_ptr
    [H_kv, total_rows + 1]`` and ``k2q_q_indices [H_kv, total_q * topK]``.
    If ``return_qsplits`` is true, also return the official packed
    ``qsplit`` payload and optionally ``split_counts [total_q, H_kv]`` used by
    K2 combine. Serving K1 consumes ``qsplit`` directly, so callers can skip
    ``q_indices`` materialization while metadata checks keep the full official
    tensor set. Long prefill chunks with complete top-k splits can also skip
    ``split_counts`` because K2's ``FULL_SPLITS`` path uses ``TOPK`` directly.
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
    invalid = mx.array(-1, dtype=mx.int32)
    q_indices = None
    if not return_qsplits or return_q_indices:
        q_indices = mx.where(sorted_valid, sorted_q, invalid)
    if not return_qsplits:
        if q_indices is None:
            raise RuntimeError("q_indices must be built when qsplit is not returned.")
        return row_ptr, q_indices

    qsplit = sorted_slot * mx.array(1 << 24, dtype=mx.int32) + sorted_q
    qsplit = mx.where(sorted_valid, qsplit, invalid)
    split_counts = None
    if return_split_counts:
        split_counts = mx.sum(q2k_indices >= 0, axis=-1).astype(mx.int32).T
    return row_ptr, q_indices, qsplit, split_counts


def msa_sparse_attention_b1_from_csr(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    row_ptr: mx.array,
    qsplit: mx.array,
    split_counts: Optional[mx.array],
    *,
    q_start: int,
    scale: float,
    block_size: int,
    topk: int,
    k1_impl: str = "auto",
    full_splits: bool = False,
) -> mx.array:
    """Run the MiniMax MSA K1/K2 kernels from prebuilt B=1 CSR metadata."""
    if not mx.metal.is_available():
        raise RuntimeError("MiniMax MSA sparse attention requires Metal.")
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, H_q, D].")
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError("k and v must have shape [total_k, H_kv, D].")

    total_q, h_q, dim = q.shape
    total_k, h_kv, k_dim = k.shape
    v_total_k, v_h_kv, v_dim = v.shape
    if (v_total_k, v_h_kv, k_dim, v_dim) != (total_k, h_kv, dim, dim):
        raise ValueError("K/V shapes must match q head dimension.")
    if h_q % h_kv != 0:
        raise ValueError("H_q must be divisible by H_kv.")
    if row_ptr.shape != (h_kv, (int(total_k) + int(block_size) - 1) // int(block_size) + 1):
        raise ValueError("row_ptr shape does not match K/V heads and block rows.")
    if qsplit.shape != (h_kv, total_q * topk):
        raise ValueError("qsplit shape must be [H_kv, total_q * topK].")
    if split_counts is None and not full_splits:
        raise ValueError("split_counts is required unless full_splits is true.")
    if split_counts is not None and split_counts.shape != (total_q, h_kv):
        raise ValueError("split_counts shape must be [total_q, H_kv].")

    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    row_ptr = mx.contiguous(row_ptr)
    qsplit = mx.contiguous(qsplit)
    if split_counts is None:
        split_counts_for_k2 = mx.zeros((1,), dtype=mx.int32)
    else:
        split_counts_for_k2 = mx.contiguous(split_counts)

    total_rows = (int(total_k) + int(block_size) - 1) // int(block_size)
    nnz_cap = total_q * topk
    qhead_per_kv = h_q // h_kv
    work_items = h_kv * nnz_cap
    if k1_impl not in {
        "auto",
        "scalar",
        "simd",
        "simd_packed",
        "steel_mma",
        "steel_mma_bq64",
    }:
        raise ValueError(f"Unsupported MiniMax MSA K1 implementation {k1_impl!r}.")

    use_simd_k1 = dim % 32 == 0 and qhead_per_kv <= 16
    use_packed_k1 = (
        use_simd_k1 and total_q >= 512 and dim == 128 and int(block_size) == 128
    )
    use_steel_mma_k1 = (
        k1_impl in {"auto", "steel_mma", "steel_mma_bq64"}
        and _MSA_CSR_K1_STEEL_MMA is not None
        and total_q >= 8
        and dim == 128
        and int(block_size) == 128
        and qhead_per_kv == 16
    )
    if k1_impl in {"steel_mma", "steel_mma_bq64"} and not use_steel_mma_k1:
        raise RuntimeError(
            "Steel MMA K1 requires Metal, D=128, block_size=128, "
            "qhead_per_kv=16, and total_q>=8."
        )
    if k1_impl == "scalar":
        use_simd_k1 = False
        use_packed_k1 = False
    elif k1_impl == "simd":
        use_packed_k1 = False
    elif k1_impl == "simd_packed" and not use_packed_k1:
        raise RuntimeError("SIMD packed K1 is not available for this shape.")

    threadgroup_size = qhead_per_kv * 32 if use_simd_k1 else 256

    if use_steel_mma_k1:
        q_tokens_per_group = 4 if k1_impl == "steel_mma_bq64" else 8
        groups_per_row_cap = (total_q + q_tokens_per_group - 1) // q_tokens_per_group
        work_items = h_kv * total_rows * groups_per_row_cap
        threadgroup_size = (q_tokens_per_group * qhead_per_kv // 8) * 32
        k1_kernel = _MSA_CSR_K1_STEEL_MMA
    elif use_packed_k1:
        q_tokens_per_group = 8
        groups_per_row_cap = (total_q + q_tokens_per_group - 1) // q_tokens_per_group
        work_items = h_kv * total_rows * groups_per_row_cap
        k1_kernel = _MSA_CSR_K1_SIMD_PACKED
    else:
        q_tokens_per_group = 1
        groups_per_row_cap = 0
        k1_kernel = _MSA_CSR_K1_SIMD if use_simd_k1 else _MSA_CSR_K1_SCALAR

    if split_counts is None:
        mx.eval(row_ptr, qsplit)
    else:
        mx.eval(row_ptr, qsplit, split_counts_for_k2)

    k1_template = [
        ("T", q.dtype),
        ("TOTAL_Q", total_q),
        ("TOTAL_K", total_k),
        ("H_Q", h_q),
        ("H_KV", h_kv),
        ("D", dim),
        ("TOPK", topk),
        ("NNZ_CAP", nnz_cap),
        ("TOTAL_ROWS", total_rows),
        ("BLOCK_SIZE", int(block_size)),
        ("Q_START", int(q_start)),
        ("QHEAD_PER_KV", qhead_per_kv),
        ("THREADGROUP_SIZE", threadgroup_size),
    ]
    if use_simd_k1:
        k1_template.append(("D_PER_LANE", dim // 32))
    if use_packed_k1 or use_steel_mma_k1:
        k1_template.extend(
            [
                ("Q_TOKENS_PER_GROUP", q_tokens_per_group),
                ("GROUPS_PER_ROW_CAP", groups_per_row_cap),
            ]
        )
    if use_steel_mma_k1:
        k1_template.append(("M_BLOCK_SIZE", q_tokens_per_group * qhead_per_kv))

    k1_inputs = [q, k, v, row_ptr, qsplit, float(scale)]
    o_partial, lse_partial = k1_kernel(
        inputs=k1_inputs,
        template=k1_template,
        grid=(work_items * threadgroup_size, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[(topk, total_q, h_q, dim), (topk, total_q, h_q)],
        output_dtypes=[q.dtype, mx.float32],
    )

    out = _MSA_CSR_K2(
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
    return out


def msa_sparse_attention_b1(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    q2k_indices: mx.array,
    *,
    q_start: int,
    scale: float,
    block_size: int,
    k1_impl: str = "auto",
    full_splits: bool = False,
) -> mx.array:
    """Official-structure MSA sparse attention for the B=1 prefill path.

    The implementation mirrors the official K1/K2 contract:
    1. build ``k2q`` CSR from group-specific ``q2k``;
    2. K1 visits CSR KV-block rows and writes ``O_partial``/``LSE_partial`` by
       top-k split slot;
    3. K2 combines split partials with log-sum-exp correction.

    This is a correctness-first Metal implementation of the official dataflow.
    It is intentionally not wired into serving until it beats the existing path.
    """
    if not mx.metal.is_available():
        raise RuntimeError("MiniMax MSA sparse attention requires Metal.")
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

    row_ptr, _, qsplit, split_counts = build_k2q_csr_b1(
        q2k_indices,
        total_k=total_k,
        block_size=block_size,
        return_qsplits=True,
        return_q_indices=False,
        return_split_counts=not full_splits,
    )
    return msa_sparse_attention_b1_from_csr(
        q,
        k,
        v,
        row_ptr,
        qsplit,
        split_counts,
        q_start=q_start,
        scale=scale,
        block_size=block_size,
        topk=topk,
        k1_impl=k1_impl,
        full_splits=full_splits,
    )


def msa_sparse_decode_b1(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    topk_idx: mx.array,
    topk_valid: mx.array,
    *,
    q_pos: int,
    scale: float,
    block_size: int,
) -> mx.array:
    """MiniMax MSA sparse decode for the B=1, one-token serving hot path.

    This avoids materializing compact K/V tensors for selected blocks.  It
    directly visits selected KV blocks and performs online softmax per Q head.
    """
    if not mx.metal.is_available():
        raise RuntimeError("MiniMax MSA sparse decode requires Metal.")
    if q.ndim != 4:
        raise ValueError("q must have shape [1, H_q, 1, D].")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must have shape [1, H_kv, total_k, D].")
    if topk_idx.ndim != 4 or topk_valid.ndim != 4:
        raise ValueError("topk_idx/topk_valid must have shape [1, H_meta, 1, topK].")

    bq, h_q, lq, dim = q.shape
    bk, h_kv, total_k, k_dim = k.shape
    bv, v_h_kv, v_total_k, v_dim = v.shape
    bt, topk_heads, lt, topk = topk_idx.shape
    if (bq, lq, bk, bv, bt, lt) != (1, 1, 1, 1, 1, 1):
        raise ValueError("msa_sparse_decode_b1 supports only B=1 and L=1.")
    if (v_h_kv, v_total_k, k_dim, v_dim) != (h_kv, total_k, dim, dim):
        raise ValueError("K/V shapes must match q head dimension.")
    if h_q % h_kv != 0:
        raise ValueError("H_q must be divisible by H_kv.")
    if topk_valid.shape != topk_idx.shape:
        raise ValueError("topk_valid shape must match topk_idx.")
    if topk_heads not in (1, h_kv):
        raise ValueError("topk metadata heads must be 1 or H_kv.")
    if dim % 32 != 0:
        raise ValueError("D must be divisible by 32 for simd decode.")

    q = mx.contiguous(q)
    k = mx.contiguous(k)
    v = mx.contiguous(v)
    topk_idx = mx.contiguous(topk_idx.astype(mx.int32))
    topk_valid = mx.contiguous(topk_valid.astype(mx.bool_))
    mx.eval(q, k, v, topk_idx, topk_valid)

    threadgroup_size = 32
    out = _MSA_DECODE_B1_SIMD(
        inputs=[q, k, v, topk_idx, topk_valid, float(scale)],
        template=[
            ("T", q.dtype),
            ("H_Q", h_q),
            ("H_KV", h_kv),
            ("D", dim),
            ("D_PER_LANE", dim // 32),
            ("TOTAL_K", total_k),
            ("TOPK", topk),
            ("TOPK_HEADS", topk_heads),
            ("BLOCK_SIZE", int(block_size)),
            ("QHEAD_PER_KV", h_q // h_kv),
            ("Q_POS", int(q_pos)),
        ],
        grid=(h_q * threadgroup_size, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
    return out
