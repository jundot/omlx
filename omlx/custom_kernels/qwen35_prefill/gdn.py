# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N806, UP006, UP035, UP045
"""Optimized Metal Gated DeltaNet prefill kernels for Qwen3.5/3.6.

The default production path is ``gated_delta_pipelined``: the exact
sequential recurrence used by mlx-lm with 8 lanes per value row, 16-row
threadgroups and a software-pipelined, unrolled 12-token block. Layouts it
does not cover fall back to ``gated_delta_blocked_seq`` (the same recurrence
with threadgroup-staged q/k/v blocks and register-resident state).

The module also keeps the FLA (flash-linear-attention) chunked
WY-representation experiment as two Metal kernels (mx.fast.metal_kernel JIT;
no mlx source modification):

- Kernel A (chunk-parallel): each (batch, v-head, chunk) threadgroup computes
  intra-chunk quantities — log-decay cumsum, A = beta*K.K^T*decay (strict
  lower), X = (I+A)^-1 via forward substitution, Kt = X*(beta*gamma*k),
  U0 = X*(beta*v), M = (q.k^T)*decay (lower incl.), MU0 = M*U0,
  Qeff = gamma*q - M*Kt.
- Kernel B (state-recurrent): each (batch, v-head, Dv-block) threadgroup
  walks chunks sequentially — U = U0 - Kt*S^T, y = MU0 + Qeff*S^T,
  S <- gamma_last*S + (gamma_last/gamma * U)^T * k. The state recurrence is
  fully independent along Dv, so the Dv split adds parallelism for free.

Memory-access design: all device inner loops stream rows (coalesced across
threads that own a fixed feature column d), never columns. Pair dots (KK^T,
qK^T) go through threadgroup-staged 32-wide d-tiles.

Numerics: all accumulation in fp32; state kept fp32 (production accuracy).
"""

import os
from typing import Optional, Tuple

import mlx.core as mx

CHUNK = 64
DV_BLK = 32

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_KERNEL_A_SRC = """
    constexpr int C = 64;
    const int tid = thread_position_in_threadgroup.x;   // 0..255
    const int c   = threadgroup_position_in_grid.x;      // chunk index
    const int hv  = threadgroup_position_in_grid.y;
    const int b   = threadgroup_position_in_grid.z;
    const int hk  = hv / (Hv / Hk);
    const int t0  = c * C;
    const int tt  = min(C, T - t0);
    const int nC  = (T + C - 1) / C;

    threadgroup float lcg_s[C];
    threadgroup float bet_s[C];
    threadgroup float bg_s[C];   // beta * exp(lcg)
    threadgroup float eg_s[C];   // exp(lcg)
    threadgroup float As[C][C + 1];
    threadgroup InT kst[C][34];  // staged 32-wide d-tile of k
    threadgroup InT qst[C][34];  // staged 32-wide d-tile of q

    // q,k: [B,T,Hk,Dk], v: [B,T,Hv,Dv], g,beta: [B,T,Hv]
    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv;
    const size_t row = (size_t)Hk * Dk;

    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < C; ++i) {
            float gi = (i < tt) ? g[((size_t)b * T + t0 + i) * Hv + hv] : 1.0f;
            acc += log(max(gi, 1e-6f));
            lcg_s[i] = acc;
        }
    }
    if (tid == 32) {
        for (int i = 0; i < C; ++i) {
            bet_s[i] = (i < tt) ? beta[((size_t)b * T + t0 + i) * Hv + hv] : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < C) {
        eg_s[tid] = exp(lcg_s[tid]);
        bg_s[tid] = bet_s[tid] * eg_s[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- A = KK^T and M = qK^T pair dots via staged 32-wide d-tiles ----
    // Each thread owns 16 (i,j) pairs; partial dots accumulate over 4 tiles.
    float accA[16];
    float accM[16];
    for (int r = 0; r < 16; ++r) { accA[r] = 0.0f; accM[r] = 0.0f; }
    for (int dt = 0; dt < Dk; dt += 32) {
        // stage k/q tiles: 64 rows x 32 cols, coalesced (thread -> (row, col))
        for (int p = tid; p < C * 32; p += 256) {
            const int r = p / 32, cc = p % 32;
            kst[r][cc] = k_base[(size_t)(t0 + r) * row + dt + cc];
            qst[r][cc] = q_base[(size_t)(t0 + r) * row + dt + cc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < 16; ++r) {
            const int p = tid * 16 + r;     // pair index
            const int i = p / C, j = p % C;
            if (j <= i) {
                float da = 0.0f, dm = 0.0f;
                for (int dd = 0; dd < 32; ++dd) {
                    const float kjd = (float)kst[j][dd];
                    da += (float)kst[i][dd] * kjd;
                    dm += (float)qst[i][dd] * kjd;
                }
                accA[r] += da;
                accM[r] += dm;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // write A (strict lower); stash M in qst-free time later — M recomputed
    // from accM after X is done, so keep accM in registers.
    for (int r = 0; r < 16; ++r) {
        const int p = tid * 16 + r;
        const int i = p / C, j = p % C;
        float a = 0.0f;
        if (j < i && i < tt)
            a = bet_s[i] * exp(lcg_s[i] - lcg_s[j]) * accA[r];
        As[i][j] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- X = (I+A)^{-1} forward substitution (row i overwrites As row i) ----
    for (int i = 0; i < C; ++i) {
        float x = 0.0f;
        if (tid < C) {
            x = (tid == i) ? 1.0f : 0.0f;
            for (int j = 0; j < i; ++j)
                x -= As[i][j] * As[j][tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < C) As[i][tid] = x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // out layouts: [B,Hv,nC,C,D]
    const size_t out_off = (((size_t)b * Hv + hv) * nC + c) * C;
    device InT* Kt_o  = Kt  + out_off * Dk;
    device InT* U0_o  = U0  + out_off * Dv;
    device InT* MU0_o = MU0 + out_off * Dv;
    device InT* Qe_o  = Qeff + out_off * Dk;
    device float* lcg_o = lcg + out_off;
    if (tid < C) lcg_o[tid] = lcg_s[tid];

    // ---- Kt = X @ (bg*k), U0 = X @ (beta*v): row-streaming ----
    // Thread owns column d = tid % Dk and half of the i range; k/v rows are
    // read coalesced once per j and broadcast into 32 register accumulators.
    {
        const int d = tid % Dk;
        const int i0 = (tid / Dk) * 32;   // 0 or 32
        float aK[32];
        float aV[32];
        for (int ii = 0; ii < 32; ++ii) { aK[ii] = 0.0f; aV[ii] = 0.0f; }
        for (int j = 0; j < tt; ++j) {
            const float kb = bg_s[j] * (float)k_base[(size_t)(t0 + j) * row + d];
            const float vb = bet_s[j] * (float)v_base[(size_t)(t0 + j) * Hv * Dv + d];
            for (int ii = 0; ii < 32; ++ii) {
                const float x = As[i0 + ii][j];
                aK[ii] += x * kb;
                aV[ii] += x * vb;
            }
        }
        for (int ii = 0; ii < 32; ++ii) {
            Kt_o[(size_t)(i0 + ii) * Dk + d] = (InT)aK[ii];
            U0_o[(size_t)(i0 + ii) * Dv + d] = (InT)aV[ii];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    // ---- write M into As (X no longer needed) ----
    for (int r = 0; r < 16; ++r) {
        const int p = tid * 16 + r;
        const int i = p / C, j = p % C;
        float m = 0.0f;
        if (j <= i && i < tt)
            m = accM[r] * exp(lcg_s[i] - lcg_s[j]);
        As[i][j] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- MU0 = M @ U0 ; Qeff = gamma*q - M @ Kt: row-streaming ----
    {
        const int d = tid % Dk;
        const int i0 = (tid / Dk) * 32;
        float aU[32];
        float aQ[32];
        for (int ii = 0; ii < 32; ++ii) { aU[ii] = 0.0f; aQ[ii] = 0.0f; }
        for (int j = 0; j < C; ++j) {
            const float u0 = (float)U0_o[(size_t)j * Dv + d];
            const float kt = (float)Kt_o[(size_t)j * Dk + d];
            for (int ii = 0; ii < 32; ++ii) {
                const float m = As[i0 + ii][j];
                aU[ii] += m * u0;
                aQ[ii] += m * kt;
            }
        }
        for (int ii = 0; ii < 32; ++ii) {
            const int i = i0 + ii;
            MU0_o[(size_t)i * Dv + d] = (InT)aU[ii];
            const float qg = (i < tt)
                ? eg_s[i] * (float)q_base[(size_t)(t0 + i) * row + d]
                : 0.0f;
            Qe_o[(size_t)i * Dk + d] = (InT)(qg - aQ[ii]);
        }
    }
"""

_KERNEL_B_SRC = """
    constexpr int C = 64;
    constexpr int DB = 32;                                 // Dv block
    const int tid = thread_position_in_threadgroup.x;     // 0..255
    const int blk = threadgroup_position_in_grid.x;        // Dv/DB block
    const int hv  = threadgroup_position_in_grid.y;
    const int b   = threadgroup_position_in_grid.z;
    const int nC  = (T + C - 1) / C;
    const int dv0 = blk * DB;

    threadgroup float S_s[DB][Dk + 4];   // state slice; +4 pad keeps float4 alignment, avoids conflicts
    threadgroup half  U_s[C][DB + 4];    // U (v_new) chunk slice
    threadgroup InT   kst[C][68];        // staged 64-wide d-tile of k
    threadgroup float lcg_s[C];
    threadgroup float r_s[C];            // exp(lcg_last - lcg)

    // state: [B,Hv,Dv,Dk]
    device const float* S_in = state_in + (((size_t)b * Hv + hv) * Dv + dv0) * Dk;
    device float* S_out      = state_out + (((size_t)b * Hv + hv) * Dv + dv0) * Dk;

    for (int p = tid; p < DB * Dk; p += 256)
        S_s[p / Dk][p % Dk] = S_in[((size_t)(p / Dk)) * Dk + (p % Dk)];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int hk = hv / (Hv / Hk);
    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    const size_t krow = (size_t)Hk * Dk;

    for (int c = 0; c < nC; ++c) {
        const int t0 = c * C;
        const int tt = min(C, T - t0);
        const size_t off = (((size_t)b * Hv + hv) * nC + c) * C;
        const device InT* Kt_c  = Kt  + off * Dk;
        const device InT* U0_c  = U0  + off * Dv;
        const device InT* MU0_c = MU0 + off * Dv;
        const device InT* Qe_c  = Qeff + off * Dk;

        if (tid < C) {
            lcg_s[tid] = lcg[off + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float lcg_last = lcg_s[C - 1];
        if (tid < C) r_s[tid] = exp(lcg_last - lcg_s[tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // U[i][dv] = U0 - Kt.S ; y[i][dv] = MU0 + Qeff.S
        // 2048 outputs / 256 threads = 8 each; float4 rows, S from smem.
        for (int p = tid; p < C * DB; p += 256) {
            const int i = p / DB, dv = p % DB;
            float4 aU4 = 0.0f, aY4 = 0.0f;
            const device vec<InT,4>* Kt4 = (const device vec<InT,4>*)(Kt_c + (size_t)i * Dk);
            const device vec<InT,4>* Qe4 = (const device vec<InT,4>*)(Qe_c + (size_t)i * Dk);
            const threadgroup float4* S4 =
                (const threadgroup float4*)&S_s[dv][0];
            for (int d4 = 0; d4 < Dk / 4; ++d4) {
                const float4 s4 = S4[d4];
                aU4 += float4(Kt4[d4]) * s4;
                aY4 += float4(Qe4[d4]) * s4;
            }
            const float accU = aU4.x + aU4.y + aU4.z + aU4.w;
            const float accY = aY4.x + aY4.y + aY4.z + aY4.w;
            U_s[i][dv] = (half)((float)U0_c[(size_t)i * Dv + dv0 + dv] - accU);
            if (i < tt) {
                const float yv = (float)MU0_c[(size_t)i * Dv + dv0 + dv] + accY;
                y[(((size_t)b * T + t0 + i) * Hv + hv) * Dv + dv0 + dv] = (InT)yv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S[dv][d] = exp(lcg_last)*S + sum_i r_i * U[i][dv] * k_i[d]
        // k is staged into smem in 64-wide d-tiles (coalesced rows), so the
        // i-loop below never touches device memory.
        const float gl = exp(lcg_last);
        for (int dt = 0; dt < Dk; dt += 64) {
            for (int p = tid; p < C * 64; p += 256) {
                const int i = p / 64, dd = p % 64;
                kst[i][dd] = k_base[(size_t)(t0 + i) * krow + dt + dd];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int p = tid; p < DB * 64; p += 256) {
                const int dv = p / 64, dd = p % 64;
                float acc = gl * S_s[dv][dt + dd];
                for (int i = 0; i < tt; ++i) {
                    acc += r_s[i] * (float)U_s[i][dv] * (float)kst[i][dd];
                }
                S_s[dv][dt + dd] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (int p = tid; p < DB * Dk; p += 256)
        S_out[((size_t)(p / Dk)) * Dk + (p % Dk)] = S_s[p / Dk][p % Dk];
"""

_kernel_a = None
_kernel_b = None


def _get_kernels():
    global _kernel_a, _kernel_b
    if _kernel_a is None:
        _kernel_a = mx.fast.metal_kernel(
            name="omlx_gdn_chunk_intra",
            input_names=["q", "k", "v", "g", "beta", "T"],
            output_names=["Kt", "U0", "MU0", "Qeff", "lcg"],
            source=_KERNEL_A_SRC,
            header=_HEADER,
        )
        _kernel_b = mx.fast.metal_kernel(
            name="omlx_gdn_chunk_scan",
            input_names=["k", "Kt", "U0", "MU0", "Qeff", "lcg", "state_in", "T"],
            output_names=["y", "state_out"],
            source=_KERNEL_B_SRC,
            header=_HEADER,
        )
    return _kernel_a, _kernel_b


SEGMENT = 4096  # tokens per A->B pipeline segment (bounds intermediate buffers)


def gated_delta_chunked_metal(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Segmented driver: run the fused chunked kernels on SEGMENT-token
    slices so intermediate buffers stay small and get reused."""
    T = q.shape[1]
    if T <= SEGMENT:
        return _gated_delta_chunked_metal_seg(q, k, v, g, beta, state)
    ys = []
    for s0 in range(0, T, SEGMENT):
        s1 = min(s0 + SEGMENT, T)
        y, state = _gated_delta_chunked_metal_seg(
            mx.contiguous(q[:, s0:s1]), mx.contiguous(k[:, s0:s1]),
            mx.contiguous(v[:, s0:s1]), mx.contiguous(g[:, s0:s1]),
            mx.contiguous(beta[:, s0:s1]), state,
        )
        ys.append(y)
    return mx.concatenate(ys, axis=1), state


def _gated_delta_chunked_metal_seg(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Chunked-parallel Gated DeltaNet prefill (scalar gating, fused Metal).

    q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv] fp32;
    state: [B,Hv,Dv,Dk] fp32. Returns y [B,T,Hv,Dv] (q.dtype) and fp32 state.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    nC = (T + CHUNK - 1) // CHUNK
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)

    ka, kb = _get_kernels()
    tmpl = [("InT", in_dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)]

    Kt, U0, MU0, Qeff, lcg = ka(
        inputs=[q, k, v, g, beta, T],
        template=tmpl,
        grid=(256 * nC, Hv, B),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (B, Hv, nC, CHUNK, Dk),
            (B, Hv, nC, CHUNK, Dv),
            (B, Hv, nC, CHUNK, Dv),
            (B, Hv, nC, CHUNK, Dk),
            (B, Hv, nC, CHUNK),
        ],
        output_dtypes=[in_dtype] * 4 + [mx.float32],
    )
    y, state_out = kb(
        inputs=[k, Kt, U0, MU0, Qeff, lcg, state, T],
        template=tmpl,
        grid=(256 * (Dv // DV_BLK), Hv, B),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[in_dtype, mx.float32],
    )
    return y, state_out


# ---------------------------------------------------------------------------
# Kernel S: blocked-sequential Gated DeltaNet prefill.
#
# Same recurrence as mlx_lm's sequential kernel (exact algorithm, no chunked
# reformulation -> half the FLOPs of the WY-chunked path), but restructured
# for Apple-GPU efficiency:
#   - k/q/v are staged into threadgroup memory in TB-token blocks with
#     coalesced cooperative loads. The stock kernel re-reads k/q from device
#     once per (Dv/4)-slice threadgroup => 32x redundant traffic (~13 GB per
#     16k-token layer). Here each v-head is split into Dv/16 blocks => 8x
#     fewer threadgroups touching the same k/q rows, and each row is read
#     from device exactly once per threadgroup.
#   - State stays in registers: thread owns (dv, 16-wide d-range) fragments;
#     the k.state and q.state contractions reduce across the 8 threads of a
#     dv row via simd_shuffle_down (no threadgroup barriers in the hot loop).
# ---------------------------------------------------------------------------

_KERNEL_S_SRC = """
    constexpr int TB = 32;                             // time block
    constexpr int DB = 32;                             // dv rows per threadgroup
    const int tid = thread_position_in_threadgroup.x;  // 0..255
    const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
    const int hv  = threadgroup_position_in_grid.y;
    const int b   = threadgroup_position_in_grid.z;
    const int hk  = hv / (Hv / Hk);
    const int dv0 = blk * DB;

    // thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in
    // the same simdgroup (lane = (dv%4)*8 + seg).
    const int dv  = tid / 8;            // 0..31
    const int seg = tid % 8;            // 0..7
    const int d0  = seg * 16;

    threadgroup InT k_s[TB][Dk + 8];
    threadgroup InT q_s[TB][Dk + 8];
    threadgroup InT v_s[TB][DB + 8];
    threadgroup float g_s[TB];
    threadgroup float b_s[TB];

    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
    const size_t krow = (size_t)Hk * Dk;

    // state fragment in registers: [dv0+dv][d0..d0+16]
    float4 st[4];
    {
        const device float4* S_in = (const device float4*)(
            state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
        for (int i = 0; i < 4; ++i) st[i] = S_in[i];
    }

    device InT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

    for (int t0 = 0; t0 < T; t0 += TB) {
        const int tt = min(TB, T - t0);
        // cooperative staging (coalesced): k/q rows, v slice, g/beta
        for (int p = tid; p < tt * Dk; p += 256) {
            const int r = p / Dk, d = p % Dk;
            k_s[r][d] = k_base[(size_t)(t0 + r) * krow + d];
            q_s[r][d] = q_base[(size_t)(t0 + r) * krow + d];
        }
        for (int p = tid; p < tt * DB; p += 256) {
            const int r = p / DB, d = p % DB;
            v_s[r][d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];
        }
        for (int p = tid; p < tt; p += 256) {
            g_s[p] = g[((size_t)b * T + t0 + p) * Hv + hv];
            b_s[p] = beta[((size_t)b * T + t0 + p) * Hv + hv];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < tt; ++t) {
            const float gt = g_s[t];
            const float bt = b_s[t];
            const threadgroup vec<InT,4>* k4 =
                (const threadgroup vec<InT,4>*)&k_s[t][d0];
            const threadgroup vec<InT,4>* q4 =
                (const threadgroup vec<InT,4>*)&q_s[t][d0];
            float4 kf[4];
            for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
            // kv_mem = (g*state) . k ; decay applied to state first
            float4 p4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] *= gt;
                p4 += st[i] * kf[i];
            }
            float part = p4.x + p4.y + p4.z + p4.w;
            // reduce across the 8 segment-threads of this dv row
            part += simd_shuffle_down(part, 4);
            part += simd_shuffle_down(part, 2);
            part += simd_shuffle_down(part, 1);
            const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
            const float delta = ((float)v_s[t][dv] - kv_mem) * bt;

            float4 o4 = 0.0f;
            for (int i = 0; i < 4; ++i) {
                st[i] += kf[i] * delta;
                o4 += st[i] * float4(q4[i]);
            }
            float out = o4.x + o4.y + o4.z + o4.w;
            out += simd_shuffle_down(out, 4);
            out += simd_shuffle_down(out, 2);
            out += simd_shuffle_down(out, 1);
            if (seg == 0) {
                y_base[(size_t)(t0 + t) * Hv * Dv + dv] = (InT)out;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        device float4* S_out = (device float4*)(
            state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
        for (int i = 0; i < 4; ++i) S_out[i] = st[i];
    }
"""

_SUPPORTED_BLOCK_T = (16, 32, 48)
_kernel_s_by_tb = {}


def _normalize_block_t(block_t: int | str | None, input_dtype=None) -> int:
    if block_t is None:
        configured_block_t = os.environ.get("OMLX_GDN_BLOCK_T")
        if configured_block_t is not None:
            block_t = configured_block_t
        else:
            # Qwen3.6 GDN inputs use float32 (mamba_ssm_dtype). TB=32 needs
            # 40,192 bytes for the 128/128, 16/32 layout, exceeding Metal's
            # 32 KiB threadgroup-memory limit. TB=16 needs 20,096 bytes.
            block_t = 16 if input_dtype == mx.float32 else 32
    block_t = int(block_t)
    if block_t not in _SUPPORTED_BLOCK_T:
        raise ValueError(
            f"OMLX_GDN_BLOCK_T must be one of {_SUPPORTED_BLOCK_T}, got {block_t}"
        )
    return block_t


def _get_kernel_s(block_t: int | str | None = None, input_dtype=None):
    block_t = _normalize_block_t(block_t, input_dtype)
    kernel = _kernel_s_by_tb.get(block_t)
    if kernel is None:
        source = _KERNEL_S_SRC.replace(
            "constexpr int TB = 32;", f"constexpr int TB = {block_t};"
        )
        kernel = mx.fast.metal_kernel(
            name=f"omlx_gdn_blocked_seq_tb{block_t}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            source=source,
            header=_HEADER,
        )
        _kernel_s_by_tb[block_t] = kernel
    return kernel


def gated_delta_blocked_seq(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    block_t: int | None = None,
) -> Tuple[mx.array, mx.array]:
    """Blocked-sequential Gated DeltaNet prefill (exact recurrence).

    q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv] fp32;
    state: [B,Hv,Dv,Dk] fp32. Returns y [B,T,Hv,Dv] (q.dtype), fp32 state.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    ks = _get_kernel_s(block_t, in_dtype)
    y, state_out = ks(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", in_dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(256 * (Dv // 32), Hv, B),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[in_dtype, mx.float32],
    )
    return y, state_out


# ---------------------------------------------------------------------------
# Kernel P: software-pipelined blocked-sequential Gated DeltaNet prefill.
#
# Same per-step math as mlx_lm's gated_delta_kernel and Kernel S; only the
# fp32 summation order of the two 128-wide dots differs:
#     S = S * g_t;  p = S.k_t;  delta = (v_t - p) * beta_t;
#     S = S + k_t * delta;  y_t = S.q_t
# The recurrence is issue- and latency-bound (the per-step dependency chain
# runs through two cross-lane reductions), so the layout trades a little
# reduction work for parallelism and hides the rest by scheduling:
#   - 8 lanes own a value row, 16 fp32 state channels each, held in
#     registers as four float4 granules at columns seg, seg+8, seg+16, seg+24
#     (bank-conflict-free threadgroup reads). A 128-thread threadgroup covers
#     16 rows, so 8 x Hv threadgroups spread evenly over the GPU cores.
#   - q/k (input dtype), v, g and beta are staged in threadgroup memory per
#     12-token block; the next block is prefetched into registers while the
#     current one runs.
#   - The 12 steps of a block are unrolled and software pipelined: step
#     t+1's k/g/beta/v are read during step t, and the q.S partials of four
#     steps are reduced together (a reduce-scatter that leaves each step's y
#     on its own lane pair, with the same pairwise tree as a per-step
#     reduction) after the next step's decay and k.S FMAs, so the in-order
#     issue does not stall on those shuffles.
#   - Dots accumulate in float2 (one horizontal add each) and the k.S dot is
#     all-reduced over the 8 lanes with xor shuffles; y is buffered per
#     block and written with 8-byte stores.
# ---------------------------------------------------------------------------

_PIPE_TB = 12  # tokens staged per block
_PIPE_DB = 16  # value rows per threadgroup
_PIPE_THREADS = _PIPE_DB * 8  # 8 lanes per row


def _pipe_load_vec(arr: str, dst: str, t: str) -> str:
    return "".join(
        f"{dst}[{i}] = float4(*(const threadgroup vec<InT, 4>*)"
        f"(&{arr}[{t}][4 * (seg + 8 * {i})]));\n"
        for i in range(4)
    )


def _pipe_reduce_q(t: str) -> str:
    # q.S partials of step t (carried in pq_prev) -> y_s[t][row] on lane 0.
    return (
        "{\n"
        "float yq = pq_prev;\n"
        "yq += simd_shuffle_down(yq, 4);\n"
        "yq += simd_shuffle_down(yq, 2);\n"
        "yq += simd_shuffle_down(yq, 1);\n"
        f"if (seg == 0) y_s[{t}][rg] = static_cast<InT>(yq);\n"
        "}\n"
    )


def _pipe_reduce_q4(s0: int) -> str:
    # q.S partials of steps s0..s0+3 (pq4[0..3]) -> y_s: a reduce-scatter
    # over the 8 lanes that leaves step s0 + 2*b2 + b1 on lanes (b2, b1, *).
    # Each sum uses the pairwise tree of _pipe_reduce_q (lane ^4, ^2, ^1),
    # so the result is bit-identical to reducing every step on its own.
    return (
        "{\n"
        "const bool qb2 = (seg & 4) != 0, qb1 = (seg & 2) != 0;\n"
        "float qa = qb2 ? pq4[2] : pq4[0];\n"
        "float qb = qb2 ? pq4[3] : pq4[1];\n"
        "qa += simd_shuffle_xor(qb2 ? pq4[0] : pq4[2], 4);\n"
        "qb += simd_shuffle_xor(qb2 ? pq4[1] : pq4[3], 4);\n"
        "float qx = qb1 ? qb : qa;\n"
        "qx += simd_shuffle_xor(qb1 ? qa : qb, 2);\n"
        "qx += simd_shuffle_xor(qx, 1);\n"
        f"if ((seg & 1) == 0) y_s[{s0} + (qb2 ? 2 : 0) + (qb1 ? 1 : 0)][rg] ="
        " static_cast<InT>(qx);\n"
        "}\n"
    )


def _pipe_step(t, load_next, reduce_prev) -> str:
    """One recurrence step at block-local index ``t``.

    Full blocks pass ``t`` as an int: step t+1's operands are read when
    ``load_next``, the q.S partial goes to pq4[t % 4], and the four steps
    before t are reduced here when t is a multiple of 4. The tail block
    passes C expressions (``t``, ``load_next``) and ``reduce_prev=None``:
    per-step q.S reduction of step t-1 when t > 0, partial in pq_prev.
    """
    nxt = (
        _pipe_load_vec("k_s", "k_next", f"({t}) + 1")
        + f"g_next = g_s[({t}) + 1]; b_next = b_s[({t}) + 1]; "
        f"v_next = v_s[({t}) + 1][rg];\n"
    )
    code = "{\n"
    code += "float4 kc[4];\nfor (int i = 0; i < 4; ++i) kc[i] = k_next[i];\n"
    code += "const float gc = g_next, bc = b_next, vc = v_next;\n"
    if load_next is True:
        code += nxt
    elif load_next is not False:
        code += f"if ({load_next}) {{\n{nxt}}}\n"
    # decay, then the k.S partial dot
    code += (
        "float2 a2 = 0.0f;\n"
        "for (int i = 0; i < 4; ++i) {\n"
        "    st[i] = st[i] * gc;\n"
        "    a2 = fma(st[i].xy, kc[i].xy, a2);\n"
        "    a2 = fma(st[i].zw, kc[i].zw, a2);\n"
        "}\n"
    )
    if reduce_prev is None:
        code += f"if (({t}) > 0) " + _pipe_reduce_q(f"({t}) - 1")
    elif reduce_prev:
        code += _pipe_reduce_q4(t - 4)
    code += (
        "float p = a2.x + a2.y;\n"
        "p += simd_shuffle_xor(p, 4);\n"
        "p += simd_shuffle_xor(p, 2);\n"
        "p += simd_shuffle_xor(p, 1);\n"
        "float4 qc[4];\n"
        + _pipe_load_vec("q_s", "qc", t)
        + "const float delta = (vc - p) * bc;\n"
        "float2 o2 = 0.0f;\n"
        "for (int i = 0; i < 4; ++i) {\n"
        "    st[i] = fma(kc[i], float4(delta), st[i]);\n"
        "    o2 = fma(st[i].xy, qc[i].xy, o2);\n"
        "    o2 = fma(st[i].zw, qc[i].zw, o2);\n"
        "}\n"
    )
    if reduce_prev is None:
        code += "pq_prev = o2.x + o2.y;\n"
    else:
        code += f"pq4[{t % 4}] = o2.x + o2.y;\n"
    return code + "}\n"


def _pipelined_source(tb: int = _PIPE_TB) -> str:
    assert tb % 4 == 0
    full_block = "".join(
        f"// step {t}\n" + _pipe_step(t, t + 1 < tb, t > 0 and t % 4 == 0)
        for t in range(tb)
    ) + _pipe_reduce_q4(tb - 4)
    tail_block = (
        "for (int t = 0; t < tt; ++t) "
        + _pipe_step("t", "t + 1 < tt", None)
        + _pipe_reduce_q("tt - 1")
    )
    first_operands = (
        _pipe_load_vec("k_s", "k_next", "0")
        + "g_next = g_s[0]; b_next = b_s[0]; v_next = v_s[0][rg];\n"
    )
    return f"""
    constexpr int TB = {tb};
    constexpr int DB = {_PIPE_DB};
    constexpr int NT = {_PIPE_THREADS};
    const int tid = thread_position_in_threadgroup.x;
    const int hv = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int hk = hv / (Hv / Hk);
    const int dv0 = threadgroup_position_in_grid.x * DB;
    const int rg = tid / 8;   // value row within the threadgroup
    const int seg = tid % 8;  // lane within the row

    threadgroup InT k_s[TB][Dk];
    threadgroup InT q_s[TB][Dk];
    threadgroup float v_s[TB][DB];
    threadgroup float g_s[TB];
    threadgroup float b_s[TB];
    threadgroup InT y_s[TB][DB];

    const size_t krow = (size_t)Hk * Dk;
    const size_t vrow = (size_t)Hv * Dv;
    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
    device InT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

    // fp32 state row fragment: channels 4*(seg + 8*i) .. +3, i = 0..3
    float4 st[4];
    {{
        const device float4* S_in = (const device float4*)(
            state_in + (((size_t)b * Hv + hv) * Dv + dv0 + rg) * Dk);
        for (int i = 0; i < 4; ++i) st[i] = S_in[seg + 8 * i];
    }}

    // register prefetch of one block (k/q rows, v slice, g, beta)
    constexpr int KQ4 = TB * Dk / 4;
    constexpr int NKQ = (KQ4 + NT - 1) / NT;
    constexpr int V4 = TB * DB / 4;
    constexpr int NV = (V4 + NT - 1) / NT;
    vec<InT, 4> pk[NKQ];
    vec<InT, 4> pq[NKQ];
    vec<InT, 4> pv[NV];
    float pg = 0.0f, pb = 0.0f;
#define GDN_PIPE_FETCH(T0N) {{ \\
        const int ttn = min(TB, T - (T0N)); \\
        for (int j = 0; j < NKQ; ++j) {{ \\
            const int p = tid + j * NT; \\
            if (p < ttn * (Dk / 4)) {{ \\
                const int r = p / (Dk / 4), c4 = p % (Dk / 4); \\
                const size_t off = (size_t)((T0N) + r) * krow + 4 * c4; \\
                pk[j] = *(const device vec<InT, 4>*)(k_base + off); \\
                pq[j] = *(const device vec<InT, 4>*)(q_base + off); \\
            }} \\
        }} \\
        for (int j = 0; j < NV; ++j) {{ \\
            const int p = tid + j * NT; \\
            if (p < ttn * (DB / 4)) {{ \\
                const int r = p / (DB / 4), c4 = p % (DB / 4); \\
                pv[j] = *(const device vec<InT, 4>*)( \\
                    v_base + (size_t)((T0N) + r) * vrow + 4 * c4); \\
            }} \\
        }} \\
        if (tid < ttn) {{ \\
            pg = g[((size_t)b * T + (T0N) + tid) * Hv + hv]; \\
            pb = beta[((size_t)b * T + (T0N) + tid) * Hv + hv]; \\
        }} \\
    }}
    GDN_PIPE_FETCH(0)

    float4 k_next[4];
    float g_next = 0.0f, b_next = 0.0f, v_next = 0.0f;
    float pq_prev = 0.0f;
    float pq4[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    for (int t0 = 0; t0 < T; t0 += TB) {{
        const int tt = min(TB, T - t0);
        for (int j = 0; j < NKQ; ++j) {{
            const int p = tid + j * NT;
            if (p < tt * (Dk / 4)) {{
                const int r = p / (Dk / 4), c4 = p % (Dk / 4);
                *(threadgroup vec<InT, 4>*)(&k_s[r][4 * c4]) = pk[j];
                *(threadgroup vec<InT, 4>*)(&q_s[r][4 * c4]) = pq[j];
            }}
        }}
        for (int j = 0; j < NV; ++j) {{
            const int p = tid + j * NT;
            if (p < tt * (DB / 4)) {{
                const int r = p / (DB / 4), c4 = p % (DB / 4);
                *(threadgroup float4*)(&v_s[r][4 * c4]) = float4(pv[j]);
            }}
        }}
        if (tid < tt) {{
            g_s[tid] = pg;
            b_s[tid] = pb;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t0 + TB < T) GDN_PIPE_FETCH(t0 + TB)
{first_operands}
        if (tt == TB) {{
{full_block}
        }} else {{
{tail_block}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = tid; p < tt * (DB / 4); p += NT) {{
            const int r = p / (DB / 4), c4 = p % (DB / 4);
            *(device vec<InT, 4>*)(y_base + (size_t)(t0 + r) * vrow + 4 * c4) =
                *(threadgroup vec<InT, 4>*)(&y_s[r][4 * c4]);
        }}
    }}
#undef GDN_PIPE_FETCH

    {{
        device float4* S_out = (device float4*)(
            state_out + (((size_t)b * Hv + hv) * Dv + dv0 + rg) * Dk);
        for (int i = 0; i < 4; ++i) S_out[seg + 8 * i] = st[i];
    }}
"""


_kernel_p = None


def _get_kernel_p():
    global _kernel_p
    if _kernel_p is None:
        _kernel_p = mx.fast.metal_kernel(
            name=f"omlx_gdn_pipelined_tb{_PIPE_TB}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            source=_pipelined_source(_PIPE_TB),
            header=_HEADER,
        )
    return _kernel_p


_PIPE_DTYPES = (mx.bfloat16, mx.float16, mx.float32)


def gated_delta_pipelined_supported(q: mx.array, k: mx.array, v: mx.array) -> bool:
    """True for the layouts Kernel P covers (128-wide keys, 16-row value blocks)."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    return (
        Dk == 128
        and Dv % _PIPE_DB == 0
        and Hk > 0
        and Hv % Hk == 0
        and T > 0
        and q.dtype in _PIPE_DTYPES
        and k.dtype == q.dtype
        and v.dtype == q.dtype
    )


def gated_delta_pipelined(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Software-pipelined blocked-sequential Gated DeltaNet prefill.

    q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; g,beta: [B,T,Hv]; state: [B,Hv,Dv,Dk]
    fp32. Returns y [B,T,Hv,Dv] (q.dtype) and the fp32 state. Layouts that
    ``gated_delta_pipelined_supported`` rejects run ``gated_delta_blocked_seq``.
    """
    if not gated_delta_pipelined_supported(q, k, v):
        return gated_delta_blocked_seq(q, k, v, g, beta, state)
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[2:]
    in_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    y, state_out = _get_kernel_p()(
        inputs=[
            q,
            k,
            v,
            g.astype(mx.float32),
            beta.astype(mx.float32),
            state.astype(mx.float32),
            T,
        ],
        template=[("InT", in_dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(_PIPE_THREADS * (Dv // _PIPE_DB), Hv, B),
        threadgroup=(_PIPE_THREADS, 1, 1),
        output_shapes=[(B, T, Hv, Dv), (B, Hv, Dv, Dk)],
        output_dtypes=[in_dtype, mx.float32],
    )
    return y, state_out
