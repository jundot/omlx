# SPDX-License-Identifier: Apache-2.0
"""Fused, batch-invariant hyper-connection kernels for GLM-5.3 prefill.

The canonical prefill hyper-connection (mlx-vlm ``deepseek_v4``) runs, per
call: a bf16->fp32 cast of the ``[L, 4, 4096]`` stream, an fp32 RMS norm over
16384, 256-row ``tiled_linear`` mix projections (a contiguous copy plus a
``[256, 16384] x [16384, 24]`` matmul per tile) and the sinkhorn/collapse
kernel; ``hc_expand`` casts the residual to fp32, runs a batched
``[4, 4] x [4, 4096]`` matmul and a compiled epilogue.  Both matmuls take
MLX's NAX path on M5-class GPUs, which runs fp32 inputs at TF32 precision
unless ``MLX_ENABLE_TF32=0``.

``hc_pre`` runs one threadgroup per 16-row tile:

- the fp32 sum of squares in MLX ``rms_looped``'s exact reduction order
  (1024 lanes, 4 reads each, ``simd_sum`` then a 32-way ``simd_sum``), so the
  inverse RMS and ``z = x * inv`` are bit-identical to ``mx.fast.rms_norm``;
- the 24 mix dot products in full fp32 on 8x8 simdgroup matrices, with a fixed
  K split (32 slices x 512) and a fixed sequential sum of the partials, so a
  row's result does not depend on the chunk length, its offset or the tile
  shape;
- sinkhorn and collapse transcribed from mlx-vlm's ``hc_sinkhorn_collapse``.

``hc_expand`` computes ``post * branch + comb^T @ residual`` with the
arithmetic of mlx-vlm's short-block ``exact_hc_expand`` (one fp32 8x8
simdgroup product per column tile, then a separately rounded
``post * branch`` product and one add) and writes bf16 once.

Both fail closed (return None, and stay off after the first failure) and the
caller keeps the canonical path.
Disable with OMLX_GLM_HC_PREFILL=0.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import mlx.core as mx

logger = logging.getLogger(__name__)

_DISABLED = os.environ.get("OMLX_GLM_HC_PREFILL", "1").strip().lower() in {
    "0",
    "false",
    "no",
    "off",
}
# Rows per pre threadgroup (a multiple of 8, at most 32).  Every threadgroup
# reads the whole [16384, 24] fp32 mix weight, so taller tiles cut that
# traffic; 16 rows measured fastest on M5 Ultra at 2047-4096-row chunks.
_ROWS = 16
# Threads per pre threadgroup (256, 512 or 1024).  Fewer threads emulate the
# 1024 reduction lanes, so results are identical for any _ROWS/_THREADS.
_THREADS = 1024
_KERNELS: dict[str, object] = {}
_VALIDATED: set[tuple] = set()

_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
"""

# x: [rows, HC*D] bf16, fnT: [HC*D, MIX] fp32 (the transposed mix weight).
# NT threads emulate rms_looped's 1024 lanes (VPER virtual simdgroups each);
# RM rows per threadgroup in RT 8-row simdgroup-matrix tiles.
_PRE_SOURCE = r"""
    constexpr int K = HC * D;
    constexpr int MIX = (2 + HC) * HC;
    constexpr int BASE_OFF = 2 * HC;
    constexpr int NSG = NT / 32;
    constexpr int VSG = 32;
    constexpr int VPER = VSG / NSG;
    constexpr int NSLICES = 32;
    constexpr int SLICE = K / NSLICES;
    constexpr int SPS = NSLICES / NSG;
    constexpr int RT = RM / 8;
    // Partial buffers per reduction round (<= 24 KB of threadgroup memory).
    constexpr int PB = (RM * MIX * 4 * NSLICES <= 24576) ? NSLICES
        : ((RM * MIX * 4 * 16 <= 24576) ? 16 : 8);
    constexpr float EPS = EPS_INT * 1e-9;

    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row0 = int(threadgroup_position_in_grid.x) * RM;
    const int nrows = int(n_rows[0]);

    threadgroup float lsum[RM][VSG];
    threadgroup float inv_sh[RM];
    threadgroup float part[PB][RM][MIX];
    threadgroup float mix_sh[RM][MIX];
    threadgroup float pre_sh[RM][HC];

    // Pass 1: sum of squares in rms_looped order (1024 lanes x 4 reads, a
    // simd_sum per 32 lanes, then a simd_sum over the 32 partials).
    for (int m = 0; m < RM; ++m) {
        if (row0 + m >= nrows) break;
        const device T* xr = x + (size_t)(row0 + m) * K;
        for (int i = 0; i < VPER; ++i) {
            const int vlid = (int(sg) + NSG * i) * 32 + int(lane);
            float acc = 0.0f;
            for (int r = 0; r < K; r += 4096) {
                for (int e = 0; e < 4; ++e) {
                    float xi = float(xr[r + vlid * 4 + e]);
                    acc += xi * xi;
                }
            }
            float s = simd_sum(acc);
            if (lane == 0) lsum[m][int(sg) + NSG * i] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int m = int(sg); m < RM; m += NSG) {
        if (row0 + m >= nrows) break;
        float s = simd_sum(lsum[m][lane]);
        if (lane == 0) inv_sh[m] = metal::precise::rsqrt(s / float(K) + norm_eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: mixes = (x * inv) @ fnT on 8x8 fp32 simdgroup matrices over 32
    // fixed 512-wide K slices (slice q on simdgroup q % NSG), then a
    // sequential sum of the 32 slice partials, independent of NT and RM.
    const short qid = short(lane / 4);
    const short fm = (qid & 4) + short((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
    bool row_ok[RT];
    float inv_r[RT];
    const device T* xa[RT];
    for (int t = 0; t < RT; ++t) {
        const int row = row0 + t * 8 + fm;
        row_ok[t] = row < nrows;
        inv_r[t] = row_ok[t] ? inv_sh[t * 8 + fm] : 0.0f;
        xa[t] = x + (size_t)(row_ok[t] ? row : row0) * K + fn;
    }
    const device float* fb = fnT + fm * MIX + fn;

    using T2 = vec<T, 2>;
    simdgroup_matrix<float, 8, 8> A;
    simdgroup_matrix<float, 8, 8> B0, B1, B2;
    simdgroup_matrix<float, 8, 8> C[SPS][RT][3];
    for (int q = 0; q < SPS; ++q) {
        for (int t = 0; t < RT; ++t) {
            for (int j = 0; j < 3; ++j) C[q][t][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
        const int kbeg = (int(sg) + NSG * q) * SLICE;
        for (int k0 = kbeg; k0 < kbeg + SLICE; k0 += 8) {
            const device float* fk = fb + k0 * MIX;
            reinterpret_cast<thread float2&>(B0.thread_elements()) = *(const device float2*)(fk);
            reinterpret_cast<thread float2&>(B1.thread_elements()) = *(const device float2*)(fk + 8);
            reinterpret_cast<thread float2&>(B2.thread_elements()) = *(const device float2*)(fk + 16);
            for (int t = 0; t < RT; ++t) {
                float2 a = float2(*(const device T2*)(xa[t] + k0));
                a.x = row_ok[t] ? a.x * inv_r[t] : 0.0f;
                a.y = row_ok[t] ? a.y * inv_r[t] : 0.0f;
                reinterpret_cast<thread float2&>(A.thread_elements()) = a;
                simdgroup_multiply_accumulate(C[q][t][0], A, B0, C[q][t][0]);
                simdgroup_multiply_accumulate(C[q][t][1], A, B1, C[q][t][1]);
                simdgroup_multiply_accumulate(C[q][t][2], A, B2, C[q][t][2]);
            }
        }
    }
    constexpr int OUT_PER = (RM * MIX + NT - 1) / NT;
    float msum[OUT_PER];
    for (int round = 0; round < NSLICES / PB; ++round) {
        for (int q = 0; q < SPS; ++q) {
            const int slice = int(sg) + NSG * q;
            if (slice / PB != round) continue;
            const int b = slice - round * PB;
            for (int t = 0; t < RT; ++t) {
                simdgroup_store(C[q][t][0], &part[b][t * 8][0], MIX);
                simdgroup_store(C[q][t][1], &part[b][t * 8][8], MIX);
                simdgroup_store(C[q][t][2], &part[b][t * 8][16], MIX);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < OUT_PER; ++i) {
            const uint o = tid + uint(i) * NT;
            if (o < (uint)(RM * MIX)) {
                const uint m = o / MIX;
                const uint j = o - m * MIX;
                int b = 0;
                if (round == 0) {
                    msum[i] = part[0][m][j];
                    b = 1;
                }
                for (; b < PB; ++b) msum[i] += part[b][m][j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < OUT_PER; ++i) {
        const uint o = tid + uint(i) * NT;
        if (o < (uint)(RM * MIX)) {
            const uint m = o / MIX;
            mix_sh[m][o - m * MIX] = msum[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Sinkhorn: one simdgroup per row (mlx-vlm hc_sinkhorn_collapse phase 1).
    for (int m = int(sg); m < RM; m += NSG) {
        const int row = row0 + m;
        if (row >= nrows) break;
        const threadgroup float* mix = mix_sh[m];
        device float* post_out = post + (size_t)row * HC;
        device float* comb_out = comb + (size_t)row * HC * HC;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint  llane  = metal::min(lane, (uint)(HC - 1));

        float pre_z  = mix[llane]      * pre_scale  + base[llane];
        float post_z = mix[HC + llane] * post_scale + base[HC + llane];
        float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

        if (lane < (uint)HC) {
            pre_sh[m][lane] = pre_v;
            post_out[lane]  = post_v;
        }

        float4 v = (*(const threadgroup float4*)(mix + BASE_OFF + llane * HC)
                        * comb_scale
                  + *(const device float4*)(base + BASE_OFF + llane * HC))
                 * active;

        float row_max = metal::max(metal::max(v.x, v.y),
                                   metal::max(v.z, v.w));
        float4 e = metal::fast::exp(v - row_max) * active;
        float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                 + EPS * active;

        float4 col_inv = 1.0f / (float4(
            simd_sum(r.x), simd_sum(r.y),
            simd_sum(r.z), simd_sum(r.w)
        ) + EPS);
        r *= col_inv;

        for (int iter = 1; iter < ITERS; ++iter) {
            r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
            col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + EPS);
            r *= col_inv;
        }

        if (lane < (uint)HC) {
            *(device float4*)(comb_out + lane * HC) = r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Collapse (mlx-vlm hc_sinkhorn_collapse phase 2).
    using T4 = vec<T, 4>;
    constexpr uint D4 = (uint)D / 4;
    for (int m = 0; m < RM; ++m) {
        const int row = row0 + m;
        if (row >= nrows) break;
        const float p0 = pre_sh[m][0];
        const float p1 = pre_sh[m][1];
        const float p2 = pre_sh[m][2];
        const float p3 = pre_sh[m][3];
        const device T* x_row = x + (size_t)row * K;
        const device T4* x_row0 = (const device T4*)(x_row + 0*D);
        const device T4* x_row1 = (const device T4*)(x_row + 1*D);
        const device T4* x_row2 = (const device T4*)(x_row + 2*D);
        const device T4* x_row3 = (const device T4*)(x_row + 3*D);
        device T4* out4 = (device T4*)(collapsed + (size_t)row * D);
        for (uint d4 = tid; d4 < D4; d4 += NT) {
            float4 x0 = float4(x_row0[d4]);
            float4 x1 = float4(x_row1[d4]);
            float4 x2 = float4(x_row2[d4]);
            float4 x3 = float4(x_row3[d4]);
            float4 result = fma(float4(p0), x0,
                            fma(float4(p1), x1,
                            fma(float4(p2), x2, float4(p3) * x3)));
            out4[d4] = T4(result);
        }
    }
"""


# Row-per-threadgroup expand: A = blockdiag(comb^T, comb^T) so one 8x8 product
# covers 16 columns (rows 0-3: columns c..c+7, rows 4-7: c+8..c+15).  Each
# output element sees the same four products as exact_hc_expand plus four
# exact zero terms, so values are identical (up to the sign of an exact zero).
_EXPAND_SOURCE = r"""
    constexpr int TPR = D / 16;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;

    const short qid = short(lane / 4);
    const short fm = (qid & 4) + short((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
    const short s = fm & 3;
    const short half_ = fm >> 2;

    // A[fm][k] = comb[row][t = k & 3][s = fm & 3] when (k >> 2) == (fm >> 2).
    float2 a_values = 0.0f;
    for (short e = 0; e < 2; ++e) {
        const short k = fn + e;
        if ((k >> 2) == half_) {
            a_values[e] = comb[(size_t)row * HC * HC + (k & 3) * HC + s];
        }
    }
    simdgroup_matrix<float, 8, 8> a_matrix;
    reinterpret_cast<thread float2&>(a_matrix.thread_elements()) = a_values;
    const float p = post[(size_t)row * HC + s];

    using T2 = vec<T, 2>;
    const device T* res_row = residual + (size_t)row * HC * D + s * D + half_ * 8 + fn;
    const device T* br_row = branch + (size_t)row * D + half_ * 8 + fn;
    device T* out_row = out + (size_t)row * HC * D + s * D + half_ * 8 + fn;

    for (int tile = int(sg); tile < TPR; tile += SIMDS) {
        const int c0 = tile * 16;
        float2 b_values = float2(*(const device T2*)(res_row + c0));
        simdgroup_matrix<float, 8, 8> b_matrix;
        simdgroup_matrix<float, 8, 8> d_matrix;
        reinterpret_cast<thread float2&>(b_matrix.thread_elements()) = b_values;
        simdgroup_multiply_accumulate(
            d_matrix, a_matrix, b_matrix,
            simdgroup_matrix<float, 8, 8>(0.0f));
        float2 values = reinterpret_cast<thread float2&>(d_matrix.thread_elements());
        const float2 br = float2(*(const device T2*)(br_row + c0));
        volatile float product0 = p * br.x;
        volatile float product1 = p * br.y;
        values.x = product0 + values.x;
        values.y = product1 + values.y;
        *(device T2*)(out_row + c0) = T2(values);
    }
"""


def enabled() -> bool:
    return not _DISABLED


def _kernel(name, inputs, outputs, source):
    kernel = _KERNELS.get(name)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=name,
            input_names=inputs,
            output_names=outputs,
            header=_HEADER,
            source=source,
            ensure_row_contiguous=True,
        )
        _KERNELS[name] = kernel
    return kernel


def _gpu_ok() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


def _module_constants(connection):
    """Transposed mix weight and the norm epsilon, cached on the module."""
    cached = getattr(connection, "_omlx_hc_prefill_consts", None)
    fn = connection.fn
    if cached is not None and cached.fn is fn:
        return cached.fn_t, cached.eps
    fn_t = mx.contiguous(fn.astype(mx.float32).T)
    eps = mx.array([connection.norm_eps], dtype=mx.float32)
    mx.eval(fn_t, eps)
    # A plain object is stored as an ordinary attribute, never as a parameter.
    connection._omlx_hc_prefill_consts = SimpleNamespace(fn=fn, fn_t=fn_t, eps=eps)
    return fn_t, eps


def pre_compatible(connection, x) -> bool:
    return (
        enabled()
        and isinstance(x, mx.array)
        and x.ndim == 4
        and x.dtype in (mx.bfloat16, mx.float16)
        and not getattr(connection, "training", False)
        and getattr(connection, "hc_mult", None) == 4
        and x.shape[2] == 4
        and x.shape[3] % 8 == 0
        and (x.shape[2] * x.shape[3]) % 4096 == 0
        and isinstance(getattr(connection, "fn", None), mx.array)
        and connection.fn.shape == (24, x.shape[2] * x.shape[3])
        and connection.fn.dtype == mx.float32
        and connection.base.shape == (24,)
        and connection.base.dtype == mx.float32
        and connection.scale.shape == (3,)
        and connection.scale.dtype == mx.float32
        and _gpu_ok()
    )


def _report_failure(exc):
    # Latch off: a persistent failure would otherwise rebuild and sync per call.
    global _DISABLED
    if not _DISABLED:
        _DISABLED = True
        logger.warning(
            "GLM fused prefill hyper-connection kernels failed closed; using "
            "the canonical path: %s",
            exc,
        )


def hc_pre(connection, x):
    """Return ``(collapsed, post, comb)`` like ``HyperConnection(x)``, or None."""
    if not pre_compatible(connection, x):
        return None
    try:
        batch, length, hc, width = x.shape
        rows = batch * length
        fn_t, eps = _module_constants(connection)
        n_rows = mx.array([rows], dtype=mx.uint32)
        eps_int = round(connection.hc_eps / 1e-9)
        collapsed, post, comb = _kernel(
            "omlx_glm_hc_prefill_pre",
            ["x", "fnT", "scale", "base", "norm_eps", "n_rows"],
            ["collapsed", "post", "comb"],
            _PRE_SOURCE,
        )(
            inputs=[
                x.reshape(rows, hc * width),
                fn_t,
                connection.scale,
                connection.base,
                eps,
                n_rows,
            ],
            template=[
                ("T", x.dtype),
                ("HC", hc),
                ("D", width),
                ("RM", _ROWS),
                ("NT", _THREADS),
                ("ITERS", connection.sinkhorn_iters),
                ("EPS_INT", eps_int),
            ],
            grid=(_THREADS * ((rows + _ROWS - 1) // _ROWS), 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(rows, width), (rows, hc), (rows, hc, hc)],
            output_dtypes=[x.dtype, mx.float32, mx.float32],
        )
        signature = (
            "pre", x.dtype, hc, width, connection.sinkhorn_iters, eps_int, _ROWS, _THREADS
        )
        if signature not in _VALIDATED:
            mx.eval(collapsed, post, comb)
            _VALIDATED.add(signature)
        return (
            collapsed.reshape(batch, length, width),
            post.reshape(batch, length, hc),
            comb.reshape(batch, length, hc, hc),
        )
    except Exception as exc:  # noqa: BLE001 - optional native path
        _report_failure(exc)
        return None


_EXPAND_SIMDS = 8


def hc_expand(branch, residual, post, comb):
    """Return ``hc_expand(branch, residual, post, comb)`` in one pass, or None."""
    if not (
        enabled()
        and _gpu_ok()
        and isinstance(branch, mx.array)
        and branch.ndim == 3
        and residual.ndim == 4
        and residual.shape[:2] == branch.shape[:2]
        and residual.shape[-1] == branch.shape[-1]
        and residual.dtype == branch.dtype
        and branch.dtype in (mx.bfloat16, mx.float16)
        and residual.shape[2] == 4
        and post.shape == residual.shape[:-1]
        and comb.shape == (*residual.shape[:-1], residual.shape[2])
        and post.dtype == mx.float32
        and comb.dtype == mx.float32
        and branch.shape[-1] % 16 == 0
    ):
        return None
    try:
        batch, length, width = branch.shape
        hc = residual.shape[2]
        rows = batch * length
        out = _kernel(
            "omlx_glm_hc_prefill_expand",
            ["branch", "residual", "post", "comb"],
            ["out"],
            _EXPAND_SOURCE,
        )(
            inputs=[branch, residual, post, comb],
            template=[
                ("T", branch.dtype),
                ("HC", hc),
                ("D", width),
                ("SIMDS", _EXPAND_SIMDS),
            ],
            grid=(32 * rows, _EXPAND_SIMDS, 1),
            threadgroup=(32, _EXPAND_SIMDS, 1),
            output_shapes=[residual.shape],
            output_dtypes=[branch.dtype],
        )[0]
        signature = ("expand", branch.dtype, hc, width)
        if signature not in _VALIDATED:
            mx.eval(out)
            _VALIDATED.add(signature)
        return out
    except Exception as exc:  # noqa: BLE001 - optional native path
        _report_failure(exc)
        return None


__all__ = ["enabled", "hc_expand", "hc_pre", "pre_compatible"]
