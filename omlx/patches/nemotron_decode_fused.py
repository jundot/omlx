"""Fused Nemotron-H / Mamba2 decode-step kernels for MLX (Metal).

Replaces the ~13-dispatch per-layer decode glue (conv-state concat/slice,
depthwise Conv1d, silu, compute_dt, ssm_update_kernel, swiglu, grouped
rms_norm, plus the splits/reshapes around them) with two JIT Metal kernels:

  Kernel A  (mamba2_decode_fused):
      in_proj output -> conv-state update + depthwise conv + silu
      -> softplus(dt)+clip -> SSD state update + y  (S positions looped
      IN-KERNEL, per-thread register-resident SSM state)
      Also emits the updated conv/ssm caches and, optionally, per-position
      state captures for speculative-decode rollback.

  Kernel B  (mamba2_gatenorm_fused):
      swiglu(gate, y) -> grouped RMS norm -> * weight   (one dispatch)

The mixer decode step becomes: in_proj (qmm) -> A -> B -> out_proj (qmm).

Numerics deliberately mirror the stock mlx_lm path cast-for-cast:
  * conv accumulates fp32 (+bias), casts to the IO dtype, silu evaluated in
    fp32 from that rounded value, cast back to IO dtype, then widened to
    fp32 for the SSD math (matches Conv1d -> nn.silu -> kernel input chain).
  * dt = clip(softplus(dt_bf16 + dt_bias), limits) in fp32 (compute_dt).
  * SSD inner math identical op-order to mlx_lm ssm_update_kernel:
      dA = exp(-exp(A_log) * dt); state = dA*state + (x*dt)*B; y = C.state
      + D*x, fp32 accumulators, state stored in the state dtype.
  * gate/norm: val = bf16(silu(gate)*y); r = rsqrt(mean_fp32(val^2)+eps);
    out = weight * bf16(val*r)  (matches swiglu -> mx.fast.rms_norm ->
    weight-mul).

Gating: decode/verify shapes only (1 <= S <= 8), mask/lengths None. All
shapes templated: Dh in {32,64,128}, Ds multiple of 32, any H/groups whose
gated-norm group size is a multiple of 256. Unsupported shapes, masked or
ragged batches, and prefill chunks fall through to the previous call chain
(including the sequential chain-verify path this patch supersedes).
"""

import logging
import math
from typing import Optional, Tuple

import mlx.core as mx

logger = logging.getLogger(__name__)

_MARKER = "_omlx_nemotron_decode_fused"

_HEADER = """
#include <metal_stdlib>
using namespace metal;

// Precise exp variants: the stock path computes softplus (compute_dt) and
// silu (nn.silu, swiglu) with precise transcendentals; only the SSD decay
// terms use fast::exp (matching mlx_lm's ssm_update_kernel).
static inline float softplus_f(float x) {
    return (x > 20.0f) ? x : log1p(metal::exp(x));
}
static inline float silu_f(float x) {
    return x / (1.0f + metal::exp(-x));
}
"""

# --------------------------------------------------------------------------- #
# Kernel A: conv + dt + SSD step (S positions in-kernel)
# --------------------------------------------------------------------------- #
# Templates: T (io dtype), U (state dtype), S, Dh, Ds, H, G(=heads/group),
#            NG(=n_groups), CAPTURE (0/1), HAS_LIMIT (0/1)
# Grid: (32, Dh, H*N), threadgroup (32, 8, 1).
#
# Buffer layouts (all contiguous):
#   proj           [N, S, P]  P = 2*H*Dh_x? no: P = ID + CD + H   (see host)
#   conv_state_in  [N, K1, CD]      K1 = conv_kernel-1 (=3), CD = ID + 2*NG*Ds
#   ssm_state_in   [N, H, Dh, Ds]   (U)
#   conv_w         [CD, K, 1]       depthwise Conv1d weight
#   conv_b         [CD]
#   A_log          [H]   (fp32)     dt_bias [H] (fp32)   D [H] (T)
#   y_out          [N, S, ID]       ID = H*Dh
#   conv_state_out [N, K1, CD]
#   ssm_state_out  [N, H, Dh, Ds]   (U)
#   cap_state      [S, N, H, Dh, Ds] (U)   (CAPTURE only; else 1-elem dummy)
#   cap_conv       [S, N, K1, CD]          (CAPTURE only; else 1-elem dummy)
_KERNEL_A_SRC = """
    constexpr int K1  = 3;                    // conv_kernel - 1
    constexpr int ID  = H * Dh;               // intermediate size
    constexpr int CD  = ID + 2 * NG * Ds;     // conv dim
    constexpr int P   = ID + CD + H;          // in_proj row width
    constexpr int NPT = Ds / 32;              // states per thread

    const int n     = thread_position_in_grid.z;   // h + H*b
    const int h_idx = n % H;
    const int b     = n / H;
    const int d_idx = thread_position_in_grid.y;   // 0..Dh-1
    const int ds_t  = thread_position_in_threadgroup.x;  // 0..31
    const int tid   = thread_position_in_threadgroup.y * 32 + ds_t; // 0..255
    const int g_loc = h_idx / G;                   // group of this head

    // ---- per-head dt terms (uniform across the head's threads) --------
    const device T* prow0 = proj + ((size_t)b * S) * P;
    const float A = -fast::exp(A_log[h_idx]);

    // ---- x-channel conv rolling window (channel h_idx*Dh + d_idx) -----
    const int xc = h_idx * Dh + d_idx;
    float xw0 = static_cast<float>(conv_state_in[((size_t)b * K1 + 0) * CD + xc]);
    float xw1 = static_cast<float>(conv_state_in[((size_t)b * K1 + 1) * CD + xc]);
    float xw2 = static_cast<float>(conv_state_in[((size_t)b * K1 + 2) * CD + xc]);
    const float xk0 = static_cast<float>(conv_w[xc * 4 + 0]);
    const float xk1 = static_cast<float>(conv_w[xc * 4 + 1]);
    const float xk2 = static_cast<float>(conv_w[xc * 4 + 2]);
    const float xk3 = static_cast<float>(conv_w[xc * 4 + 3]);
    const float xbi = static_cast<float>(conv_b[xc]);

    // ---- B/C staging: 256 threads stage this head-group's 2*Ds chans --
    // channel: tid < Ds -> B chan, else C chan. Rolling window in regs.
    const int bc_rel  = (tid < Ds) ? tid : (tid - Ds);
    const int bc_chan = ((tid < Ds) ? (ID + g_loc * Ds)
                                    : (ID + NG * Ds + g_loc * Ds)) + bc_rel;
    float bw0 = static_cast<float>(conv_state_in[((size_t)b * K1 + 0) * CD + bc_chan]);
    float bw1 = static_cast<float>(conv_state_in[((size_t)b * K1 + 1) * CD + bc_chan]);
    float bw2 = static_cast<float>(conv_state_in[((size_t)b * K1 + 2) * CD + bc_chan]);
    const float bk0 = static_cast<float>(conv_w[bc_chan * 4 + 0]);
    const float bk1 = static_cast<float>(conv_w[bc_chan * 4 + 1]);
    const float bk2 = static_cast<float>(conv_w[bc_chan * 4 + 2]);
    const float bk3 = static_cast<float>(conv_w[bc_chan * 4 + 3]);
    const float bbi = static_cast<float>(conv_b[bc_chan]);

    threadgroup float tgB[Ds];
    threadgroup float tgC[Ds];

    // ---- SSM state: NPT states per thread, register resident ----------
    float st[NPT];
    {
        const device U* is = ssm_state_in + (size_t)n * Dh * Ds + (size_t)d_idx * Ds;
        for (int i = 0; i < NPT; ++i) {
            st[i] = static_cast<float>(is[NPT * ds_t + i]);
        }
    }

    const bool bc_writer = (h_idx % G == 0) && (d_idx < 8);
    const bool x_leader  = (ds_t == 0);

    for (int s = 0; s < S; ++s) {
        const device T* prow = prow0 + (size_t)s * P;

        // dt for this head at this position (uniform per head)
        float dtv = softplus_f(static_cast<float>(prow[ID + CD + h_idx])
                               + dt_bias[h_idx]);
        if (HAS_LIMIT) { dtv = clamp(dtv, lim[0], lim[1]); }
        const float dA = fast::exp(A * dtv);

        // x conv for this row's channel (all 32 lanes identical)
        float xin = static_cast<float>(prow[ID + xc]);
        float xcv = xk0 * xw0 + xk1 * xw1 + xk2 * xw2 + xk3 * xin + xbi;
        xw0 = xw1; xw1 = xw2; xw2 = xin;
        // cast chain: fp32 conv -> T -> silu(fp32) -> T -> fp32
        float x_ = static_cast<float>(
            static_cast<T>(silu_f(static_cast<float>(static_cast<T>(xcv)))));

        // B/C conv staged to threadgroup memory
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            float bin = static_cast<float>(prow[ID + bc_chan]);
            float bcv = bk0 * bw0 + bk1 * bw1 + bk2 * bw2 + bk3 * bin + bbi;
            bw0 = bw1; bw1 = bw2; bw2 = bin;
            float bcs = static_cast<float>(
                static_cast<T>(silu_f(static_cast<float>(static_cast<T>(bcv)))));
            if (tid < Ds) { tgB[bc_rel] = bcs; } else { tgC[bc_rel] = bcs; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // SSD step
        float acc = 0.0f;
        const float dtx = x_ * dtv;
        for (int i = 0; i < NPT; ++i) {
            const int s_idx = NPT * ds_t + i;
            st[i] = dA * st[i] + dtx * tgB[s_idx];
            acc += st[i] * tgC[s_idx];
        }
        acc = simd_sum(acc);
        if (x_leader) {
            y_out[(((size_t)b * S + s) * H + h_idx) * Dh + d_idx] =
                static_cast<T>(acc + x_ * static_cast<float>(D[h_idx]));
        }

        if (CAPTURE) {
            device U* cs = cap_state
                + (((size_t)s * NB + n) * Dh + d_idx) * Ds;
            for (int i = 0; i < NPT; ++i) {
                cs[NPT * ds_t + i] = static_cast<U>(st[i]);
            }
            device T* cc = cap_conv + ((size_t)s * (NB / H) + b) * K1 * CD;
            if (x_leader) {
                cc[0 * CD + xc] = static_cast<T>(xw0);
                cc[1 * CD + xc] = static_cast<T>(xw1);
                cc[2 * CD + xc] = static_cast<T>(xw2);
            }
            if (bc_writer) {
                cc[0 * CD + bc_chan] = static_cast<T>(bw0);
                cc[1 * CD + bc_chan] = static_cast<T>(bw1);
                cc[2 * CD + bc_chan] = static_cast<T>(bw2);
            }
        }
    }

    // ---- write back caches --------------------------------------------
    {
        device U* os = ssm_state_out + (size_t)n * Dh * Ds + (size_t)d_idx * Ds;
        for (int i = 0; i < NPT; ++i) {
            os[NPT * ds_t + i] = static_cast<U>(st[i]);
        }
    }
    if (x_leader) {
        device T* co = conv_state_out + (size_t)b * K1 * CD;
        co[0 * CD + xc] = static_cast<T>(xw0);
        co[1 * CD + xc] = static_cast<T>(xw1);
        co[2 * CD + xc] = static_cast<T>(xw2);
    }
    if (bc_writer) {
        device T* co = conv_state_out + (size_t)b * K1 * CD;
        co[0 * CD + bc_chan] = static_cast<T>(bw0);
        co[1 * CD + bc_chan] = static_cast<T>(bw1);
        co[2 * CD + bc_chan] = static_cast<T>(bw2);
    }
"""

# --------------------------------------------------------------------------- #
# Kernel B: swiglu + grouped RMS norm + weight
# --------------------------------------------------------------------------- #
# Templates: T, GS (norm group size), NGRP (ID // GS), EPS_DEN? passed via
#            template float EPS.
# Grid: (256, NGRP, S*N), threadgroup (256, 1, 1).
# Buffers: proj [N,S,P] (gate slice), y [N,S,ID], w [ID], out [N,S,ID]
_KERNEL_B_SRC = """
    constexpr int ID  = H * Dh;
    constexpr int CD  = ID + 2 * NG * Ds;
    constexpr int P   = ID + CD + H;
    constexpr int NT  = 256;
    constexpr int EPT = GS / NT;              // elements per thread

    const int tid  = thread_position_in_threadgroup.x;
    const int grp  = thread_position_in_grid.y;
    const int sn   = thread_position_in_grid.z;     // s + S*b
    const int s    = sn % S;
    const int b    = sn / S;

    const device T* grow = proj + ((size_t)b * S + s) * P + (size_t)grp * GS;
    const device T* yrow = y_in + ((size_t)b * S + s) * ID + (size_t)grp * GS;
    device T* orow = out + ((size_t)b * S + s) * ID + (size_t)grp * GS;

    float vals[EPT];
    float ss = 0.0f;
    for (int i = 0; i < EPT; ++i) {
        const int c = i * NT + tid;
        float gv = static_cast<float>(grow[c]);
        float yv = static_cast<float>(yrow[c]);
        // cast chain: swiglu output rounds to T before the norm reduction
        float v = static_cast<float>(static_cast<T>(silu_f(gv) * yv));
        vals[i] = v;
        ss += v * v;
    }
    ss = simd_sum(ss);
    threadgroup float red[NT / 32];
    if (thread_index_in_simdgroup == 0) {
        red[simdgroup_index_in_threadgroup] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_index_in_threadgroup == 0) {
        float t = (thread_index_in_simdgroup < NT / 32)
                    ? red[thread_index_in_simdgroup] : 0.0f;
        t = simd_sum(t);
        if (thread_index_in_simdgroup == 0) { red[0] = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float r = metal::rsqrt(red[0] / GS + eps_buf[0]);
    for (int i = 0; i < EPT; ++i) {
        const int c = i * NT + tid;
        orow[c] = static_cast<T>(static_cast<float>(norm_w[(size_t)grp * GS + c])
                                 * static_cast<float>(static_cast<T>(vals[i] * r)));
    }
"""

_kernel_a = None
_kernel_b = None
_lim_cache = {}
_eps_cache = {}


def _lim_array(lo, hi):
    key = (lo, hi)
    if key not in _lim_cache:
        _lim_cache[key] = mx.array([lo, hi], dtype=mx.float32)
    return _lim_cache[key]


def _eps_array(eps):
    if eps not in _eps_cache:
        _eps_cache[eps] = mx.array([eps], dtype=mx.float32)
    return _eps_cache[eps]


def _get_kernels():
    global _kernel_a, _kernel_b
    if _kernel_a is None:
        _kernel_a = mx.fast.metal_kernel(
            name="mamba2_decode_fused",
            input_names=[
                "proj", "conv_state_in", "ssm_state_in", "conv_w", "conv_b",
                "A_log", "dt_bias", "D", "lim",
            ],
            output_names=[
                "y_out", "conv_state_out", "ssm_state_out",
                "cap_state", "cap_conv",
            ],
            header=_HEADER,
            source=_KERNEL_A_SRC,
        )
        _kernel_b = mx.fast.metal_kernel(
            name="mamba2_gatenorm_fused",
            input_names=["proj", "y_in", "norm_w", "eps_buf"],
            output_names=["out"],
            header=_HEADER,
            source=_KERNEL_B_SRC,
        )
    return _kernel_a, _kernel_b


def supported(num_heads, head_dim, state_size, n_groups, conv_kernel, group_size):
    return (
        conv_kernel == 4
        and head_dim in (32, 64, 128)
        and 32 <= state_size <= 256 and state_size % 32 == 0
        and num_heads % n_groups == 0
        and (num_heads * head_dim) % group_size == 0
        and group_size % 256 == 0
    )


def mamba2_decode_step(
    proj: mx.array,            # [N, S, P] in_proj output
    conv_state: mx.array,      # [N, K-1, CD]
    ssm_state: mx.array,       # [N, H, Dh, Ds]
    conv_w: mx.array,          # [CD, K, 1]
    conv_b: mx.array,          # [CD]
    A_log: mx.array,           # [H] fp32
    dt_bias: mx.array,         # [H] fp32
    D: mx.array,               # [H]
    norm_w: mx.array,          # [ID]
    *,
    num_heads: int,
    head_dim: int,
    state_size: int,
    n_groups: int,
    eps: float,
    group_size: int,
    time_step_limit: Optional[Tuple[float, float]] = None,
    capture: bool = False,
):
    """Fused decode step. Returns (out, conv_state_out, ssm_state_out,
    cap_state, cap_conv); cap_* are dummies unless capture=True."""
    ka, kb = _get_kernels()
    N, S, P = proj.shape
    H, Dh, Ds = num_heads, head_dim, state_size
    ID = H * Dh
    CD = ID + 2 * n_groups * Ds
    tdt = proj.dtype
    has_limit = (
        time_step_limit is not None
        and not (time_step_limit[0] == 0.0 and math.isinf(time_step_limit[1]))
    )
    lo = float(time_step_limit[0]) if has_limit else 0.0
    hi = float(time_step_limit[1]) if has_limit else 0.0
    lim = _lim_array(lo, hi)
    cap_state_shape = (S, N * H, Dh, Ds) if capture else (1, 1, 1, 1)
    cap_conv_shape = (S, N, 3, CD) if capture else (1, 1, 1, 1)

    tmpl_a = [
        ("T", tdt), ("U", ssm_state.dtype), ("S", S), ("Dh", Dh), ("Ds", Ds),
        ("H", H), ("G", H // n_groups), ("NG", n_groups), ("NB", N * H),
        ("CAPTURE", 1 if capture else 0), ("HAS_LIMIT", 1 if has_limit else 0),
    ]
    y, conv_out, ssm_out, cap_state, cap_conv = ka(
        inputs=[proj, conv_state, ssm_state, conv_w, conv_b, A_log, dt_bias, D, lim],
        template=tmpl_a,
        grid=(32, Dh, H * N),
        threadgroup=(32, 8, 1),
        output_shapes=[
            (N, S, ID), conv_state.shape, ssm_state.shape,
            cap_state_shape, cap_conv_shape,
        ],
        output_dtypes=[tdt, tdt, ssm_state.dtype, ssm_state.dtype, tdt],
    )
    (out,) = kb(
        inputs=[proj, y, norm_w, _eps_array(float(eps))],
        template=[
            ("T", tdt), ("H", H), ("Dh", Dh), ("Ds", Ds), ("NG", n_groups),
            ("S", S), ("GS", group_size),
        ],
        grid=(256, ID // group_size, S * N),
        threadgroup=(256, 1, 1),
        output_shapes=[(N, S, ID)],
        output_dtypes=[tdt],
    )
    return out, conv_out, ssm_out, cap_state, cap_conv


# --------------------------------------------------------------------------- #
# Mixer.__call__ wrap: route decode + MTP verify windows to the fused kernels
# --------------------------------------------------------------------------- #
def _mixer_fused_ok(mixer):
    ok = getattr(mixer, "_fused_dims_ok", None)
    if ok is None:
        ok = supported(
            mixer.num_heads,
            mixer.head_dim,
            mixer.ssm_state_size,
            mixer.n_groups,
            mixer.conv_kernel_size,
            mixer.norm.group_size,
        )
        mixer._fused_dims_ok = ok
    return ok


def _run_fused(mixer, hidden_states, cache, capture):
    proj = mixer.in_proj(hidden_states)
    y, conv_out, ssm_out, cap_s, cap_c = mamba2_decode_step(
        proj,
        cache[0],
        cache[1],
        mixer.conv1d.weight,
        mixer.conv1d.bias,
        mixer.A_log,
        mixer.dt_bias,
        mixer.D.astype(hidden_states.dtype),
        mixer.norm.weight,
        num_heads=mixer.num_heads,
        head_dim=mixer.head_dim,
        state_size=mixer.ssm_state_size,
        n_groups=mixer.n_groups,
        eps=mixer.norm.eps,
        group_size=mixer.norm.group_size,
        time_step_limit=mixer.time_step_limit,
        capture=capture,
    )
    cache[0], cache[1] = conv_out, ssm_out
    cache.advance(hidden_states.shape[1])
    return mixer.out_proj(y), cap_s, cap_c


def apply_nemotron_decode_fused_patch() -> bool:
    """Wrap NemotronHMamba2Mixer.__call__ (on top of whatever MTP patches are
    installed) so decode steps and small MTP verify windows run the fused
    kernel pair. Falls through for prefill chunks, masked/ragged batches,
    uninitialized caches, and unsupported dims."""
    if not (mx.metal.is_available() and mx.default_device() == mx.gpu):
        return False
    try:
        from mlx_lm.models import nemotron_h as nh
    except Exception:
        return False

    Mixer = nh.NemotronHMamba2Mixer
    prev_call = Mixer.__call__
    if getattr(prev_call, _MARKER, False):
        return True
    # Does the installed call chain understand n_confirmed (omlx MTP patches)?
    prev_takes_confirmed = getattr(prev_call, "_omlx_nh_mtp", False)

    def __call__(self, hidden_states, mask, cache=None, n_confirmed=0):
        S = hidden_states.shape[1]
        if (
            cache is not None
            and mask is None
            and 1 <= S <= 8
            and cache[0] is not None
            and cache[1] is not None
            and getattr(cache, "lengths", None) is None
            and hidden_states.dtype in (mx.bfloat16, mx.float16)
            and _mixer_fused_ok(self)
        ):
            if 0 < n_confirmed < S:
                # MTP chain verify window: capture per-position states so
                # partial rollback is a pure ref restore.
                conv0, ssm0 = cache[0], cache[1]
                out, cap_s, cap_c = _run_fused(self, hidden_states, cache, True)
                state_shape = ssm0.shape
                cache._mtp_pos_states = [
                    (cap_c[i], cap_s[i].reshape(state_shape)) for i in range(S)
                ]
                cache.rollback_state = (conv0, ssm0)
                cache._mtp_draft_stash = hidden_states
                return out
            out, _, _ = _run_fused(self, hidden_states, cache, False)
            return out
        if prev_takes_confirmed:
            return prev_call(self, hidden_states, mask, cache, n_confirmed)
        return prev_call(self, hidden_states, mask, cache)

    setattr(__call__, _MARKER, True)
    __call__._omlx_nh_mtp = prev_takes_confirmed
    Mixer.__call__ = __call__
    logger.info(
        "Nemotron-H fused decode patch applied "
        "(fused conv+dt+SSD and gate+norm kernels, S<=8, in-kernel captures)"
    )
    return True
