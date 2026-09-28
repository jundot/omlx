# SPDX-License-Identifier: Apache-2.0
#
# Layout adapted from Layr-Labs mlxfast-qwen38-125b-a6b-engine
# (Runner/FastModel/TrackFastGDNDecode.swift, "track_gdn_decode_complete"),
# without its deferred-state journal.
"""Single-launch Qwen4 GDN step for multi-row one-token decode (opt-in).

With two or three concurrent rows, a Qwen4 gated-delta layer decodes through
the stock mlx-vlm chain: after the four input projections it runs the conv
window update, SiLU, L2 q/k normalization, the g/beta gate math, the
recurrence and the gated RMSNorm as about twenty dispatches.  This kernel
does all of it in one launch: one threadgroup per (row, value head), 32
SIMD-groups each owning four value rows of the state.

Every rounding site mirrors the stock chain, so the output, conv state and
recurrent state are bit-identical to it.  B1 decode keeps the existing
three-kernel path and speculative verify keeps its own path; neither is
routed here.

The recurrent state is still written in full every call: unlike mlxfast we
keep no journal, so prefix-cache snapshots see ordinary state.  Enabled with
``OMLX_QWEN4_FUSED_GDN=1``; default off.
"""

from __future__ import annotations

import os

import mlx.core as mx

HK, HV, DK, DV = 16, 48, 128, 128
CONV_DIM = 2 * HK * DK + HV * DV
MIN_ROWS = 2
MAX_ROWS = 3

_KERNEL = None


def enabled() -> bool:
    return os.environ.get("OMLX_QWEN4_FUSED_GDN", "0") == "1"


_HEADER = """
    inline float omlx_log1p(float x) {
        float xp1 = 1.0f + x;
        if (xp1 == metal::numeric_limits<float>::max()) {
            return metal::numeric_limits<float>::max();
        }
        if (xp1 == 1.0f) {
            return x;
        }
        return x * (metal::log(xp1) / (xp1 - 1.0f));
    }
"""

_SOURCE = """
    static_assert(DK == 128 && DV == 128, "GDN head geometry");
    typedef InT T;
    const uint n = threadgroup_position_in_grid.z;
    const uint b_idx = n / uint(HV);
    const uint hv_idx = n % uint(HV);
    constexpr uint G = uint(HV) / uint(HK);
    const uint hk_idx = hv_idx / G;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;

    threadgroup T q_sh[DK];
    threadgroup T k_sh[DK];
    threadgroup T v_sh[DV];
    threadgroup T y_sh[DV];
    threadgroup float g_sh;
    threadgroup float beta_sh;

    // Issue the state loads first; they are the only large traffic.
    float4 st[4];
    const device float* st_in = state_in + (n * uint(DV) + sg * 4) * uint(DK) + lane * 4;
    for (int r = 0; r < 4; ++r) {
        st[r] = *reinterpret_cast<const device float4*>(st_in + r * DK);
    }

    const device T* qkv_b = qkv + b_idx * uint(C);
    const device T* cs_b = conv_state + b_idx * 3 * uint(C);
    auto win = [&](uint r, uint ch) -> T {
        return r < 3 ? cs_b[r * uint(C) + ch] : qkv_b[ch];
    };

    if (sg < 3) {
        const uint which = sg;
        const uint base = which == 0 ? hk_idx * uint(DK)
                        : (which == 1 ? uint(HK) * uint(DK) + hk_idx * uint(DK)
                                      : 2 * uint(HK) * uint(DK) + hv_idx * uint(DV));
        T activated[4];
        T l2acc = T(0);
        for (uint i = 0; i < 4; ++i) {
            const uint ch = base + lane * 4 + i;
            float acc = 0.0f;
            for (uint tap = 0; tap < 4; ++tap) {
                acc += float(win(tap, ch)) * float(conv_w[ch * 4 + tap]);
            }
            const T conv = T(acc);
            T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
            const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
            activated[i] = act;
            const T sqv = T(float(act) * float(act));
            l2acc = T(float(l2acc) + float(sqv));
        }
        if (which < 2) {
            // Stock Qwen4 L2 chain: x * rsqrt(sum(square(x), -1) + 1e-6),
            // then the dk^-0.5 query scale.
            threadgroup T* dst = which == 0 ? q_sh : k_sh;
            float tv = float(l2acc);
            tv += simd_shuffle_xor(tv, short(16));
            tv += simd_shuffle_xor(tv, short(8));
            tv += simd_shuffle_xor(tv, short(4));
            tv += simd_shuffle_xor(tv, short(2));
            tv += simd_shuffle_xor(tv, short(1));
            const T eps = T(float(T(tv)) + float(T(1e-6f)));
            const T inv = T(metal::precise::rsqrt(float(eps)));
            for (uint i = 0; i < 4; ++i) {
                const T l2 = T(float(activated[i]) * float(inv));
                dst[lane * 4 + i] = which == 0 ? T(float(l2) * float(q_scale)) : l2;
            }
        } else {
            for (uint i = 0; i < 4; ++i) {
                v_sh[lane * 4 + i] = activated[i];
            }
        }
    } else if (sg == 31 && lane == 0) {
        const uint gi = b_idx * uint(HV) + hv_idx;
        const T bv = b_in[gi];
        T by = T(1) / (T(1) + metal::exp(metal::abs(bv)));
        beta_sh = float((bv < T(0)) ? by : T(1) - by);
        const T apd = T(float(a_in[gi]) + float(dt_bias[hv_idx]));
        const T neg_abs = -metal::abs(apd);
        const T exp_term = T(metal::precise::exp(float(neg_abs)));
        const T log_term = T(omlx_log1p(float(exp_term)));
        const T positive = metal::max(apd, T(0));
        const T sp = T(float(positive) + float(log_term));
        float ea = metal::precise::exp(float(A_log[hv_idx]));
        g_sh = metal::precise::exp(-(ea * float(sp)));
    } else if (sg >= 28 && sg < 31) {
        // Next conv state: q/k channels are shared by the G value heads of a
        // key head, so only the first of them writes those.
        const uint which = sg - 28;
        if (which == 2 || hv_idx % G == 0) {
            const uint base = which == 0 ? hk_idx * uint(DK)
                            : (which == 1 ? uint(HK) * uint(DK) + hk_idx * uint(DK)
                                          : 2 * uint(HK) * uint(DK) + hv_idx * uint(DV));
            device T* o_conv = conv_out + b_idx * 3 * uint(C);
            for (uint i = 0; i < 4; ++i) {
                const uint ch = base + lane * 4 + i;
                for (uint j = 0; j < 3; ++j) {
                    o_conv[j * uint(C) + ch] = win(1 + j, ch);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float kf[4];
    float qf[4];
    for (int i = 0; i < 4; ++i) {
        kf[i] = float(k_sh[lane * 4 + i]);
        qf[i] = float(q_sh[lane * 4 + i]);
    }
    const float gt = g_sh;
    const float bt = beta_sh;
    for (int r = 0; r < 4; ++r) {
        const uint dv_idx = sg * 4 + r;
        float s[4] = {st[r].x, st[r].y, st[r].z, st[r].w};
        float kv_mem = 0.0f;
        for (int i = 0; i < 4; ++i) {
            s[i] = s[i] * gt;
            kv_mem += s[i] * kf[i];
        }
        kv_mem += simd_shuffle_xor(kv_mem, 1);
        kv_mem += simd_shuffle_xor(kv_mem, 2);
        kv_mem += simd_shuffle_xor(kv_mem, 4);
        kv_mem += simd_shuffle_xor(kv_mem, 8);
        kv_mem += simd_shuffle_xor(kv_mem, 16);
        const float delta = (float(v_sh[dv_idx]) - kv_mem) * bt;
        float yacc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            s[i] = s[i] + kf[i] * delta;
            yacc += s[i] * qf[i];
        }
        yacc += simd_shuffle_xor(yacc, 1);
        yacc += simd_shuffle_xor(yacc, 2);
        yacc += simd_shuffle_xor(yacc, 4);
        yacc += simd_shuffle_xor(yacc, 8);
        yacc += simd_shuffle_xor(yacc, 16);
        if (lane == 0) {
            y_sh[dv_idx] = T(yacc);
        }
        st[r] = float4(s[0], s[1], s[2], s[3]);
    }
    device float* st_out = state_out + (n * uint(DV) + sg * 4) * uint(DK) + lane * 4;
    for (int r = 0; r < 4; ++r) {
        *reinterpret_cast<device float4*>(st_out + r * DK) = st[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        const uint base = (b_idx * uint(HV) + hv_idx) * uint(DV) + lane * 4;
        float xs[4];
        float sumsq = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            xs[i] = float(y_sh[lane * 4 + i]);
            sumsq += xs[i] * xs[i];
        }
        sumsq = simd_sum(sumsq);
        float inv = metal::precise::rsqrt(sumsq / float(DV) + float(eps));
        for (uint i = 0; i < 4; ++i) {
            const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
            float zv = float(z[base + i]);
            float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
            float sig = zv < 0.0f ? sy : 1.0f - sy;
            out[base + i] = T(float(normed) * sig);
        }
    }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_gdn_fused_decode",
            input_names=[
                "qkv",
                "conv_state",
                "conv_w",
                "q_scale",
                "b_in",
                "a_in",
                "A_log",
                "dt_bias",
                "state_in",
                "z",
                "norm_w",
                "eps",
            ],
            output_names=["out", "state_out", "conv_out"],
            header=_HEADER,
            source=_SOURCE,
        )
    return _KERNEL


def fused_step(qkv, conv_state, conv_w, b, a, a_log, dt_bias, state, z, norm_w, eps):
    """Run conv prework, gates, recurrence and gated RMSNorm in one launch.

    qkv [B,1,C], conv_state [B,3,C], b/a [B,1,Hv], state [B,Hv,Dv,Dk] fp32,
    z [B,1,Hv*Dv].  Returns ``(out [B,1,Hv*Dv], state_out, conv_out [B,3,C])``.
    """
    batch, _, c_dim = qkv.shape
    return _kernel()(
        inputs=[
            qkv,
            conv_state,
            conv_w,
            mx.array(DK**-0.5, dtype=mx.bfloat16),
            b,
            a,
            a_log,
            dt_bias,
            state,
            z,
            norm_w,
            mx.array(eps, dtype=mx.float32),
        ],
        template=[
            ("InT", qkv.dtype),
            ("HK", HK),
            ("HV", HV),
            ("DK", DK),
            ("DV", DV),
            ("C", c_dim),
        ],
        grid=(32, 32, batch * HV),
        threadgroup=(32, 32, 1),
        output_shapes=[(batch, 1, HV * DV), (batch, HV, DV, DK), (batch, 3, c_dim)],
        output_dtypes=[qkv.dtype, mx.float32, qkv.dtype],
    )


def eligible(inputs, cache) -> bool:
    """Two or three bf16 one-token rows over plain Qwen4 recurrent state."""
    if not (
        isinstance(inputs, mx.array)
        and inputs.ndim == 3
        and MIN_ROWS <= inputs.shape[0] <= MAX_ROWS
        and inputs.shape[1] == 1
        and inputs.shape[2] == 2560
        and inputs.dtype == mx.bfloat16
        and mx.default_device() == mx.gpu
        and cache is not None
        and getattr(cache, "lengths", None) is None
        and not getattr(cache, "is_speculating", False)
        and len(getattr(cache, "cache", ())) == 2
    ):
        return False
    batch = inputs.shape[0]
    conv_state, recurrent_state = cache[0], cache[1]
    return (
        isinstance(conv_state, mx.array)
        and conv_state.shape == (batch, 3, CONV_DIM)
        and conv_state.dtype == mx.bfloat16
        and isinstance(recurrent_state, mx.array)
        and recurrent_state.shape == (batch, HV, DV, DK)
        and recurrent_state.dtype == mx.float32
    )


def run(layer, cache, mixed_qkv, z, b, a):
    """Advance one GDN layer's cache by one token; return the gated norm."""
    out, state, conv = fused_step(
        mixed_qkv,
        cache[0],
        layer.conv1d.weight,
        b,
        a,
        layer.A_log,
        layer.dt_bias,
        cache[1],
        z,
        layer.norm.weight,
        layer.norm.eps,
    )
    cache[0] = conv
    cache[1] = state
    return out
