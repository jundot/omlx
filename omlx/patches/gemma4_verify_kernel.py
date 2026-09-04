# SPDX-License-Identifier: Apache-2.0
#
# The kernel bodies below are adapted from MLX's sdpa_vector_2pass_1/2
# (mlx/backend/metal/kernels/sdpa_vector.h, Copyright © 2024 Apple Inc.,
# MIT license). MLX ships that kernel with multi-row query support but its
# C++ dispatch never routes q_len >= 2 at head_dim 512 to it; this module
# instantiates the same proven bodies through ``mx.fast.metal_kernel`` so
# the gemma4 verify path can call them directly.
"""Fused multi-row vector attention for gemma4 verify forwards (head_dim 512).

gemma4 global (full-attention) layers carry head_dim 512, which MLX's fused
SDPA covers only at L=1; every L >= 2 verify forward drops to the unfused
O(L x ctx) pass, and the decomposed per-token route caps at L=3 because it
pays one full-KV scan per row. This kernel scans the KV once for a block of
rows (vector-kernel tiling, online softmax, end-aligned causal), so verify
depth stops being context-bound.

Semantics match the decomposed route exactly: the mask argument is ignored,
every row attends keys [0 .. offset + row], left padding is assumed zero
(the caller gates on it). K/V are read from the cache's full backing buffer
plus a valid-length scalar so no contiguity copy of the KV ever happens.

Shape contract:
- queries [B, Hq, L, 512], keys/values buffers [B, Hkv, capacity, 512]
- Hq % Hkv == 0; threadgroup budget caps rows-per-dispatch at 32 // gqa
  (gemma4 26B/31B: gqa 8 -> 4 rows), longer L row-chunks with adjusted
  causal horizons — 2 dispatch pairs cover L=8.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_BLOCKS = 32  # KV interleave factor of pass 1 == simdgroups of pass 2

_HEADER = """
constant constexpr float NEG_MAX = -3.402823466e38f;
"""

# Threadgroup (32, gqa, S); grid threadgroups (Hkv, B, _BLOCKS). Each
# simdgroup owns R consecutive rows of one q head (R q rows and R output
# accumulators live in registers — R=2 is the largest spill-free point at
# head_dim 512), so a call covers S*R rows with S same-block KV re-reads
# instead of one per row. Odd row counts are front-padded by the host: the
# leading dummy rows get the *shortest* causal horizons, so they never
# touch keys beyond the valid range. q is pre-scaled on the host.
# params = [N, cap_k, cap_v] (int32).
_PASS1_SRC = """
    constexpr int BLOCKS = {blocks};
    constexpr int BD = 32;
    constexpr int qk_per_thread = {head_dim} / BD;
    constexpr int v_per_thread = {head_dim} / BD;

    const int N = params[0];
    const int cap_k = params[1];
    const int cap_v = params[2];

    const uint simd_lid = thread_index_in_simdgroup;
    const int kv_head_idx = threadgroup_position_in_grid.x;
    const int batch_idx = threadgroup_position_in_grid.y;
    const int block_idx = threadgroup_position_in_grid.z;
    const int gqa = threads_per_threadgroup.y;
    const int row_groups = threads_per_threadgroup.z;
    const int head_in_group = thread_position_in_threadgroup.y;
    const int row_group = thread_position_in_threadgroup.z;
    const int total_rows = row_groups * R;
    const int row_base = row_group * R;
    const int num_kv_heads = threadgroups_per_grid.x;
    const int num_q_heads = num_kv_heads * gqa;
    const int q_head_idx = gqa * kv_head_idx + head_in_group;
    const int q_bh = batch_idx * num_q_heads + q_head_idx;

    typedef float U;
    thread U qreg[R][qk_per_thread];
    thread U o[R][v_per_thread];
    thread U max_score[R];
    thread U sum_exp_score[R];

    const device T* qp = q
        + ((size_t)q_bh * total_rows + row_base) * {head_dim}
        + simd_lid * qk_per_thread;
    const device T* kp = k
        + ((size_t)(batch_idx * num_kv_heads + kv_head_idx) * cap_k + block_idx)
            * {head_dim}
        + simd_lid * qk_per_thread;
    const device T* vp = v
        + ((size_t)(batch_idx * num_kv_heads + kv_head_idx) * cap_v + block_idx)
            * {head_dim}
        + simd_lid * v_per_thread;

    for (int r = 0; r < R; r++) {{
        for (int i = 0; i < qk_per_thread; i++) {{
            qreg[r][i] = static_cast<U>(qp[(size_t)r * {head_dim} + i]);
        }}
        for (int i = 0; i < v_per_thread; i++) {{
            o[r][i] = 0;
        }}
        max_score[r] = NEG_MAX;
        sum_exp_score[r] = 0;
    }}

    // Global row (row_base + r) attends keys [0 .. N - total_rows + row].
    const int base_horizon = N - total_rows + row_base;
    for (int i = block_idx; i < N; i += BLOCKS) {{
        thread U kreg[qk_per_thread];
        for (int j = 0; j < qk_per_thread; j++) {{
            kreg[j] = static_cast<U>(kp[j]);
        }}
        for (int r = 0; r < R; r++) {{
            if (i <= base_horizon + r) {{
                U score = 0;
                for (int j = 0; j < qk_per_thread; j++) {{
                    score += qreg[r][j] * kreg[j];
                }}
                score = simd_sum(score);

                U new_max = max(max_score[r], score);
                U factor = metal::fast::exp(max_score[r] - new_max);
                U exp_score = metal::fast::exp(score - new_max);
                max_score[r] = new_max;
                sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;
                for (int j = 0; j < v_per_thread; j++) {{
                    o[r][j] = o[r][j] * factor + exp_score * static_cast<U>(vp[j]);
                }}
            }}
        }}
        kp += BLOCKS * {head_dim};
        vp += BLOCKS * {head_dim};
    }}

    for (int r = 0; r < R; r++) {{
        const size_t o_off = (size_t)q_bh * total_rows + row_base + r;
        if (simd_lid == 0) {{
            sums[o_off * BLOCKS + block_idx] = sum_exp_score[r];
            maxs[o_off * BLOCKS + block_idx] = max_score[r];
        }}
        // fp32 partials: a T-typed intermediate would round every block's
        // accumulator to bf16 (8-bit mantissa), adding per-layer verify
        // noise that measurably rejects near-tie draft tokens.
        device float* pp = partials
            + (o_off * BLOCKS + block_idx) * {head_dim}
            + simd_lid * v_per_thread;
        for (int i = 0; i < v_per_thread; i++) {{
            pp[i] = o[r][i];
        }}
    }}
"""

# Threadgroup (32 * _BLOCKS, 1, 1); grid threadgroups (B * Hq, rows, 1).
_PASS2_SRC = """
    constexpr int BLOCKS = {blocks};
    constexpr int BD = 32;
    constexpr int elem_per_thread = {head_dim} / BD;

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int bh = threadgroup_position_in_grid.x;
    const int row = threadgroup_position_in_grid.y;
    const int rows = threadgroups_per_grid.y;
    const int o_off = bh * rows + row;

    typedef float U;
    thread U o[elem_per_thread];
    threadgroup U outputs[BLOCKS * BD];

    const device float* pp = partials
        + (size_t)o_off * BLOCKS * {head_dim}
        + simd_gid * {head_dim} + simd_lid * elem_per_thread;
    const device float* sp = sums + (size_t)o_off * BLOCKS;
    const device float* mp = maxs + (size_t)o_off * BLOCKS;
    device T* op = out + (size_t)o_off * {head_dim} + simd_gid * elem_per_thread;

    U max_score = mp[simd_lid];
    max_score = simd_max(max_score);
    U sum_exp_score = simd_sum(metal::fast::exp(mp[simd_lid] - max_score) * sp[simd_lid]);

    U factor = metal::fast::exp(mp[simd_gid] - max_score);
    for (int i = 0; i < elem_per_thread; i++) {{
        o[i] = factor * pp[i];
    }}

    for (int i = 0; i < elem_per_thread; i++) {{
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (simd_lid == 0) {{
        for (int i = 0; i < elem_per_thread; i++) {{
            op[i] = static_cast<T>(o[i]);
        }}
    }}
"""

# Rows per simdgroup: R q rows and R fp32 output accumulators live in
# per-thread registers (head_dim/32 elements each); R=2 is the largest
# spill-free point at head_dim 512, measured — R>=3 collapses to stack
# traffic in the key loop.
_RPS = 2

_kernels: dict = {}
_availability: bool | None = None
# Per-dtype threadgroup thread budget of the pass-1 pipeline. Metal caps
# maxTotalThreadsPerThreadgroup per compiled pipeline by register usage,
# so the value is device-dependent (M3 Ultra allows 1024; the virtualized
# CI GPU allows 448) and must be probed, not assumed.
_tg_thread_cap: dict = {}


def _get_kernels(head_dim: int):
    key = head_dim
    got = _kernels.get(key)
    if got is None:
        p1 = mx.fast.metal_kernel(
            name=f"omlx_g4_verify_sdpa_p1_d{head_dim}",
            input_names=["q", "k", "v", "params"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER,
            source=_PASS1_SRC.format(head_dim=head_dim, blocks=_BLOCKS),
        )
        p2 = mx.fast.metal_kernel(
            name=f"omlx_g4_verify_sdpa_p2_d{head_dim}",
            input_names=["partials", "sums", "maxs"],
            output_names=["out"],
            header=_HEADER,
            source=_PASS2_SRC.format(head_dim=head_dim, blocks=_BLOCKS),
        )
        got = (p1, p2)
        _kernels[key] = got
    return got


def _dispatch(q_rows, k_buf, v_buf, n_keys: int):
    B, hq, rows, head_dim = q_rows.shape
    hkv, cap_k = k_buf.shape[1], k_buf.shape[2]
    cap_v = v_buf.shape[2]
    gqa = hq // hkv
    row_groups = rows // _RPS
    p1, p2 = _get_kernels(head_dim)

    params = mx.array([n_keys, cap_k, cap_v], dtype=mx.int32)
    partials, sums, maxs = p1(
        inputs=[q_rows, k_buf, v_buf, params],
        template=[("T", q_rows.dtype), ("R", _RPS)],
        grid=(32 * hkv, gqa * B, row_groups * _BLOCKS),
        threadgroup=(32, gqa, row_groups),
        output_shapes=[
            (B, hq, rows, _BLOCKS, head_dim),
            (B, hq, rows, _BLOCKS),
            (B, hq, rows, _BLOCKS),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = p2(
        inputs=[partials, sums, maxs],
        template=[("T", q_rows.dtype)],
        grid=(32 * _BLOCKS * B * hq, rows, 1),
        threadgroup=(32 * _BLOCKS, 1, 1),
        output_shapes=[(B, hq, rows, head_dim)],
        output_dtypes=[q_rows.dtype],
    )
    return out


def _probe_tg_cap(dtype) -> int:
    """Largest working pass-1 threadgroup size (in threads) for ``dtype``.

    Tries S in {4, 2, 1} row groups at gqa 8 (256 * S threads) with a tiny
    dispatch; the Metal validator rejects oversized threadgroups at eval
    time with a ValueError. Returns 0 when even 256 threads fail.
    """
    cap = _tg_thread_cap.get(dtype)
    if cap is not None:
        return cap
    cap = 0
    for s in (4, 2, 1):
        try:
            q = mx.zeros((1, 8, _RPS * s, 512), dtype=dtype)
            kv = mx.zeros((1, 1, 8, 512), dtype=dtype)
            mx.eval(_dispatch(q, kv, kv, 8))
            cap = 32 * 8 * s
            break
        except ValueError:
            continue
        except Exception:
            logger.warning("gemma4 verify kernel probe failed", exc_info=True)
            break
    _tg_thread_cap[dtype] = cap
    return cap


def kernel_max_rows(gqa: int, dtype) -> int:
    """Rows a single dispatch can carry for this geometry (0 = infeasible)."""
    cap = _probe_tg_cap(dtype)
    row_groups = min(cap, 1024) // (32 * gqa)
    return _RPS * row_groups


def fused_verify_sdpa(queries, k_buf, v_buf, n_keys: int, scale: float):
    """Multi-row causal SDPA over a linear KV buffer.

    queries [B, Hq, L, D]; k_buf/v_buf [B, Hkv, capacity, D] with the first
    ``n_keys`` positions valid (the last L of them are the block's own
    keys). Row j attends keys [0 .. n_keys - L + j]. Returns [B, Hq, L, D].
    """
    B, hq, L, head_dim = queries.shape
    gqa = hq // k_buf.shape[1]
    rows_cap = kernel_max_rows(gqa, queries.dtype)
    if rows_cap <= 0:
        raise ValueError(
            f"pass-1 threadgroup budget too small for gqa={gqa} "
            f"(needs {32 * gqa} threads)"
        )
    q = queries * scale if scale != 1.0 else queries

    outs = []
    r0 = 0
    while r0 < L:
        r1 = min(r0 + rows_cap, L)
        chunk = q[:, :, r0:r1, :]
        # Front-pad to a multiple of _RPS: leading dummy rows take the
        # shortest causal horizons, so they stay inside the valid keys and
        # are sliced back off below.
        pad = -(r1 - r0) % _RPS
        if pad:
            chunk = mx.concatenate([chunk[:, :, :1, :]] * pad + [chunk], axis=2)
        out = _dispatch(chunk, k_buf, v_buf, n_keys - (L - r1))
        if pad:
            out = out[:, :, pad:, :]
        outs.append(out)
        r0 = r1
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def is_available() -> bool:
    """Compile-and-run probe, cached for the process lifetime."""
    global _availability
    if _availability is None:
        try:
            if not mx.metal.is_available():
                _availability = False
            else:
                q = mx.zeros((1, 8, 2, 512), dtype=mx.float16)
                k = mx.zeros((1, 1, 8, 512), dtype=mx.float16)
                out = fused_verify_sdpa(q, k, k, 6, 1.0)
                mx.eval(out)
                _availability = True
        except Exception:
            logger.warning(
                "gemma4 verify kernel unavailable; falling back", exc_info=True
            )
            _availability = False
    return _availability
