# Copyright © 2026 Apple Inc.

import logging
import os
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.deepseek_v4.decode_consistency import matmul as decode_matmul

_CANONICAL_WIDE_PREFILL = os.getenv(
    "OMLX_DSV4_CANONICAL_WIDE_PREFILL", "0"
).strip().lower() in ("1", "true", "on", "yes")
_COMPILED_HC_DECODE_PRODUCER = os.getenv(
    "OMLX_DSV4_COMPILED_HC_DECODE_PRODUCER", "0"
).strip().lower() in ("1", "true", "on", "yes")
_VERIFY_HC_PRENORM = os.getenv(
    "OMLX_DSV4_VERIFY_HC_PRENORM", "0"
).strip().lower() in ("1", "true", "on", "yes")
_PREFILL_HC_PRENORM = os.getenv(
    "OMLX_DSV4_PREFILL_HC_PRENORM", "0"
).strip().lower() in ("1", "true", "on", "yes")
_DECODE_HC_PRENORM = os.getenv(
    "OMLX_DSV4_DECODE_HC_PRENORM", "0"
).strip().lower() in ("1", "true", "on", "yes")
_VERIFY_HC_PRENORM_LOGGED = False


@mx.compile
def _compiled_hc_decode_mixes(x: mx.array, weight: mx.array) -> mx.array:
    y = x.astype(mx.float32)
    z = mx.fast.rms_norm(y.flatten(-2), None, 1e-6)
    return z @ weight.T


def _can_use_compiled_hc_decode_producer(module, x: mx.array) -> bool:
    return bool(
        _COMPILED_HC_DECODE_PRODUCER
        and not module.training
        and tuple(x.shape) == (1, 1, 4, 4096)
        and x.dtype == mx.bfloat16
        and tuple(module.fn.shape) == (24, 16384)
        and module.fn.dtype == mx.float32
        and float(module.norm_eps) == 1e-6
    )


def _make_hc_sinkhorn_collapse_kernel():
    """Fused sinkhorn + collapse: eliminates one dispatch per HC cycle.

    1. BRANCHLESS SINKHORN: all 32 lanes in simd group 0 execute identical
       instructions. Lanes >= HC use multiplicative mask (active=0) instead
       of divergent branches — eliminates SIMD serialization.
    2. PARALLEL SINKHORN: lanes 0-3 each own one comb row. Column norm
       via simd_sum() — free SIMD shuffle.
    3. NATIVE bfloat4 LOADS: single 64-bit load yields 4 bfloat16 values;
       cast to float4 is a free hardware conversion.
    4. FMA CHAINS: collapse uses fused multiply-add for 3 of 4 terms.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        uint tid  = thread_position_in_threadgroup.x;
        uint row  = threadgroup_position_in_grid.x;
        uint lane = tid % 32;
        uint sg   = tid / 32;

        constexpr int MIX      = (2 + HC) * HC;
        constexpr int BASE_OFF = 2 * HC;
        constexpr float EPS = EPS_INT * 1e-9;

        const device float* mix      = (const device float*)mixes + row * MIX;
        device float*       post_out = (device float*)post + row * HC;
        device float*       comb_out = (device float*)comb + row * HC * HC;

        threadgroup float pre_shared[HC];

        // ================================================================
        // PHASE 1: Branchless sinkhorn on simd group 0
        //   All 32 lanes execute identical instructions. Lanes >= HC
        //   compute on clamped indices but multiply by active=0, so they
        //   contribute zero to simd_sum. No divergent branches in the loop.
        // ================================================================
        if (sg == 0) {
            const float pre_scale  = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];

            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint  llane  = metal::min(lane, (uint)(HC - 1));

            // Pre/post sigmoids: all lanes compute, only active lanes write
            float pre_z  = mix[llane]      * pre_scale  + base[llane];
            float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            float pre_v  = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post_out[lane]   = post_v;
            }

            // Comb softmax: load + mask. Inactive lanes load row 0 (safe)
            // but multiply by active=0 so they hold zeros.
            float4 v = (*(const device float4*)(mix  + BASE_OFF + llane * HC)
                            * comb_scale
                      + *(const device float4*)(base + BASE_OFF + llane * HC))
                     * active;

            float row_max = metal::max(metal::max(v.x, v.y),
                                       metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                     + EPS * active;

            // Initial column normalization
            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)
            ) + EPS);
            r *= col_inv;

            // Sinkhorn iterations: zero branches in the loop body
            for (int iter = 1; iter < ITERS; ++iter) {
                // Row norm + re-clamp inactive lanes
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;

                // Col norm via simd_sum
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

        // ================================================================
        // PHASE 2: Collapse — all 256 threads, vectorized
        // ================================================================
        const float p0 = pre_shared[0];
        const float p1 = pre_shared[1];
        const float p2 = pre_shared[2];
        const float p3 = pre_shared[3];

        const device T* x_row  = (const device T*)x_in
                                         + row * (HC * D);
        device U*       out_row = (device U*)collapsed
                                         + row * D;

        using T4 = vec<T, 4>;
        using U4 = vec<U, 4>;
        const device T4* x_row0 = (const device T4*)(x_row + 0*D);
        const device T4* x_row1 = (const device T4*)(x_row + 1*D);
        const device T4* x_row2 = (const device T4*)(x_row + 2*D);
        const device T4* x_row3 = (const device T4*)(x_row + 3*D);
        device U4*       out4   = (device U4*)out_row;

        constexpr uint D4 = (uint)D / 4;

        for (uint d4 = tid; d4 < D4; d4 += 256) {
            float4 x0 = float4(x_row0[d4]);
            float4 x1 = float4(x_row1[d4]);
            float4 x2 = float4(x_row2[d4]);
            float4 x3 = float4(x_row3[d4]);

            float4 result = fma(float4(p0), x0,
                            fma(float4(p1), x1,
                            fma(float4(p2), x2, float4(p3) * x3)));

            out4[d4] = U4(result);
        }

        // Scalar tail for D not divisible by 4
        #if (D % 4) != 0
        for (uint d = D4 * 4 + tid; d < (uint)D; d += 256) {
            float val = p0*(float)x_row[0*D+d] + p1*(float)x_row[1*D+d]
                      + p2*(float)x_row[2*D+d] + p3*(float)x_row[3*D+d];
            out_row[d] = (U)val;
        }
        #endif
    """

    return mx.fast.metal_kernel(
        name="hc_sinkhorn_collapse",
        input_names=["x_in", "mixes", "scale", "base"],
        output_names=["collapsed", "post", "comb"],
        source=source,
        ensure_row_contiguous=True,
    )


_hc_sinkhorn_collapse_kernel = _make_hc_sinkhorn_collapse_kernel()


def _make_hc_sinkhorn_collapse_norm_kernel():
    """Fuse exact HC collapse with MLX's 4096-wide weighted RMSNorm.

    The first phase is the existing HC=4 sinkhorn/collapse arithmetic. The
    continuation casts every collapsed float4 to BF16 before reusing it, then
    reproduces ``rms_single_row<T, 4>`` exactly: 1024 threads, one four-scalar
    read per thread, 32 SIMD partials, ``metal::precise::rsqrt``, and the same
    normalize-to-BF16-before-weight multiplication boundary.
    """

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = r"""
        const uint tid  = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg   = simdgroup_index_in_threadgroup;
        const uint row  = threadgroup_position_in_grid.x;

        constexpr int HC = 4;
        constexpr int D = 4096;
        constexpr int MIX = 24;
        constexpr int BASE_OFF = 8;
        constexpr float EPS = EPS_INT * 1e-9f;
        constexpr float NORM_EPS = NORM_EPS_INT * 1e-9f;

        const device float* mix = (const device float*)mixes + row * MIX;
        device float* post_out = (device float*)post + row * HC;
        device float* comb_out = (device float*)comb + row * HC * HC;
        threadgroup float pre_shared[HC];
        threadgroup float local_sums[32];
        threadgroup float local_inv_mean[1];

        if (sg == 0) {
            const float pre_scale = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];
            const float active = lane < HC ? 1.0f : 0.0f;
            const uint llane = metal::min(lane, uint(HC - 1));

            const float pre_z = mix[llane] * pre_scale + base[llane];
            const float post_z = mix[HC + llane] * post_scale + base[HC + llane];
            const float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
            const float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
            if (lane < HC) {
                pre_shared[lane] = pre_v;
                post_out[lane] = post_v;
            }

            float4 v = (*(const device float4*)(mix + BASE_OFF + llane * HC)
                            * comb_scale
                        + *(const device float4*)(base + BASE_OFF + llane * HC))
                       * active;
            const float row_max = metal::max(
                metal::max(v.x, v.y), metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS))
                     + EPS * active;
            float4 col_inv = 1.0f / (float4(
                simd_sum(r.x), simd_sum(r.y),
                simd_sum(r.z), simd_sum(r.w)) + EPS);
            r *= col_inv;
            for (int iter = 1; iter < ITERS; ++iter) {
                r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
                col_inv = 1.0f / (float4(
                    simd_sum(r.x), simd_sum(r.y),
                    simd_sum(r.z), simd_sum(r.w)) + EPS);
                r *= col_inv;
            }
            if (lane < HC) {
                *(device float4*)(comb_out + lane * HC) = r;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        using T4 = vec<T, 4>;
        const device T* x_row = (const device T*)x_in + row * HC * D;
        const device T4* x0 = (const device T4*)(x_row + 0 * D);
        const device T4* x1 = (const device T4*)(x_row + 1 * D);
        const device T4* x2 = (const device T4*)(x_row + 2 * D);
        const device T4* x3 = (const device T4*)(x_row + 3 * D);
        device T4* collapsed4 = (device T4*)collapsed + row * (D / 4);
        device T* normed_row = (device T*)normed + row * D;

        const float4 result = fma(float4(pre_shared[0]), float4(x0[tid]),
                              fma(float4(pre_shared[1]), float4(x1[tid]),
                              fma(float4(pre_shared[2]), float4(x2[tid]),
                                  float4(pre_shared[3]) * float4(x3[tid]))));
        const T4 rounded = T4(result);
        collapsed4[tid] = rounded;

        float values[4];
        values[0] = float(rounded.x);
        values[1] = float(rounded.y);
        values[2] = float(rounded.z);
        values[3] = float(rounded.w);
        float acc = 0.0f;
        for (int idx = 0; idx < 4; ++idx) {
            acc += values[idx] * values[idx];
        }
        acc = simd_sum(acc);
        if (sg == 0) {
            local_sums[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            local_sums[sg] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            acc = simd_sum(local_sums[lane]);
            if (lane == 0) {
                local_inv_mean[0] = metal::precise::rsqrt(
                    acc / float(D) + NORM_EPS);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint base_index = tid * 4;
        for (int idx = 0; idx < 4; ++idx) {
            normed_row[base_index + idx] =
                norm_weight[base_index + idx]
                * T(values[idx] * local_inv_mean[0]);
        }
    """
    return mx.fast.metal_kernel(
        name="hc_sinkhorn_collapse_norm_verify",
        input_names=["x_in", "mixes", "scale", "base", "norm_weight"],
        output_names=["collapsed", "normed", "post", "comb"],
        source=source,
        ensure_row_contiguous=True,
    )


_hc_sinkhorn_collapse_norm_kernel = _make_hc_sinkhorn_collapse_norm_kernel()


def _hc_kernel(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    B, L, H, D = x.shape

    return _hc_sinkhorn_collapse_kernel(
        inputs=[x, mixes, scale, base],
        template=[
            ("T", x.dtype),
            ("U", x.dtype),
            ("HC", hc_mult),
            ("ITERS", sinkhorn_iters),
            ("D", D),
            ("EPS_INT", round(eps / 1e-9)),
        ],
        grid=(B * L * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(B, L, D), (B, L, hc_mult), (B, L, hc_mult, hc_mult)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


def _hc_norm_kernel(
    x,
    mixes,
    scale,
    base,
    norm_weight,
    sinkhorn_iters,
    eps,
    norm_eps,
):
    rows = int(x.shape[0]) * int(x.shape[1])
    return _hc_sinkhorn_collapse_norm_kernel(
        inputs=[x, mixes, scale, base, norm_weight],
        template=[
            ("T", x.dtype),
            ("ITERS", sinkhorn_iters),
            ("EPS_INT", round(eps / 1e-9)),
            ("NORM_EPS_INT", round(norm_eps / 1e-9)),
        ],
        grid=(rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[
            (x.shape[0], x.shape[1], 4096),
            (x.shape[0], x.shape[1], 4096),
            (x.shape[0], x.shape[1], 4),
            (x.shape[0], x.shape[1], 4, 4),
        ],
        output_dtypes=[x.dtype, x.dtype, mx.float32, mx.float32],
    )


@mx.compile
def _hc_split_sinkhorn_ops(
    mixes: mx.array,
    scale: mx.array,
    base: mx.array,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[mx.array, mx.array, mx.array]:
    mixes = mixes.astype(mx.float32)
    scale = scale.astype(mx.float32)
    base = base.astype(mx.float32)
    pre_scale, post_scale, comb_scale = scale[0], scale[1], scale[2]

    pre = mx.sigmoid(mixes[..., :hc_mult] * pre_scale + base[:hc_mult]) + eps
    post = 2 * mx.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * post_scale + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].reshape(
        *mixes.shape[:-1], hc_mult, hc_mult
    ) * comb_scale + base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _hc_ops(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    pre, post, comb = _hc_split_sinkhorn_ops(
        mixes, scale, base, hc_mult, sinkhorn_iters, eps
    )
    return (pre[..., None] * y).sum(axis=2).astype(x.dtype), post, comb


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def _call_one(self, x: mx.array):
        B, L, H, D = x.shape
        y = x.astype(mx.float32)
        if _can_use_compiled_hc_decode_producer(self, x):
            mixes = _compiled_hc_decode_mixes(x, self.fn)
        else:
            z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
            mixes = decode_matmul(z, self.fn.T)

        use_ops = (
            self.training
            or mx.default_device() != mx.gpu
            or not mx.metal.is_available()
        )
        hc_func = _hc_ops if use_ops else _hc_kernel

        return hc_func(
            x,
            y,
            mixes,
            self.scale,
            self.base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.hc_eps,
        )

    def __call__(self, x: mx.array):
        if (
            _CANONICAL_WIDE_PREFILL
            and not self.training
            and x.ndim == 4
            and tuple(x.shape[:2]) == (1, 2048)
            and x.shape[2:] == (self.hc_mult, 4096)
            and x.dtype == mx.bfloat16
        ):
            parts = (self._call_one(x[:, :1024]), self._call_one(x[:, 1024:]))
            return tuple(
                mx.concatenate([parts[0][index], parts[1][index]], axis=1)
                for index in range(3)
            )
        return self._call_one(x)

    def call_with_norm(self, x: mx.array, norm: nn.Module):
        """Return HC outputs plus the exact following RMSNorm, or ``None``."""

        weight = getattr(norm, "weight", None)
        norm_eps = getattr(norm, "eps", None)
        qualified_shape = tuple(x.shape) in (
            (1, 1, 4, 4096),
            (1, 6, 4, 4096),
            (1, 1024, 4, 4096),
        )
        enabled_shape = bool(
            (_DECODE_HC_PRENORM and tuple(x.shape) == (1, 1, 4, 4096))
            or (_VERIFY_HC_PRENORM and tuple(x.shape) == (1, 6, 4, 4096))
            or (_PREFILL_HC_PRENORM and tuple(x.shape) == (1, 1024, 4, 4096))
        )
        if not (
            enabled_shape
            and _hc_sinkhorn_collapse_norm_kernel is not None
            and not self.training
            and qualified_shape
            and x.dtype == mx.bfloat16
            and self.hc_mult == 4
            and tuple(self.fn.shape) == (24, 16384)
            and self.fn.dtype == mx.float32
            and weight is not None
            and tuple(weight.shape) == (4096,)
            and weight.dtype == mx.bfloat16
            and float(self.norm_eps) == 1e-6
            and float(norm_eps) == 1e-6
        ):
            return None
        global _VERIFY_HC_PRENORM_LOGGED
        if not _VERIFY_HC_PRENORM_LOGGED:
            _VERIFY_HC_PRENORM_LOGGED = True
            logging.getLogger(__name__).info(
                "deepseek_v4: using exact M=%d FP32-HC sinkhorn/collapse -> "
                "weighted RMSNorm continuation "
                "(verify/prefill HC pre-norm gates disable)",
                int(x.shape[1]),
            )
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = decode_matmul(z, self.fn.T)
        return _hc_norm_kernel(
            x,
            mixes,
            self.scale,
            self.base,
            weight,
            self.sinkhorn_iters,
            self.hc_eps,
            norm_eps,
        )


@mx.compile
def _hc_expand_op(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    y = y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))
    return y.astype(x.dtype)


def hc_expand(x, residual, post, comb):
    return _hc_expand_op(x, residual, post, comb)


@mx.compile
def _hc_residual_branch_op(residual, comb):
    """Independent FP32 residual branch of the exact HC expansion."""

    return mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))


@mx.compile
def _hc_merge_branch_op(x, post, residual_branch):
    """Merge after an attention/MoE collective with stock arithmetic order."""

    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    y = y + residual_branch
    return y.astype(x.dtype)


def hc_residual_branch(residual, comb):
    return _hc_residual_branch_op(residual, comb)


def hc_merge_branch(x, post, residual_branch):
    return _hc_merge_branch_op(x, post, residual_branch)


class HyperHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.fn = mx.zeros(
            (self.hc_mult, self.hc_mult * config.hidden_size), dtype=mx.float32
        )
        self.base = mx.zeros((self.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array):
        y = x.astype(mx.float32)
        z = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps)
        mixes = decode_matmul(z, self.fn.T)
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype)
