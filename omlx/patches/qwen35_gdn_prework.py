# SPDX-License-Identifier: Apache-2.0
#
# Kernel adapted from mlx-serve (src/transformer.zig, GDN_PREWORK_SOURCE),
# itself a port of the mlxfast-challenge qwen35_packed_gdn_prework kernel.
"""Fused GDN prework for Qwen3.5/3.6 MTP verify widths (S in 2..9).

The composed target-verify prework in mlx-vlm's ``Qwen3_5GatedDeltaNet`` —
conv-state concat + depthwise conv1d + SiLU + q/k/v split + reshapes + two
ones-weight RMS norms + two scalar scales + the next conv-state slice — is
~10 small dispatches per GDN layer per verify cycle. On the 27B that is 48
layers x ~0.29 ms (measured sync-mode at S=4 on M3 Ultra), the largest
remaining verify-cycle cost after the attention split.

One Metal launch replaces the whole chain. Numerics notes carried from the
donor kernel: the in-kernel sigmoid uses MLX's own unary formula
(exp-of-abs), which the challenge swept bit-exact over all finite bf16
inputs; the RMS applies the ones-weight rounding then the separate scalar
multiply's rounding — the composed chain's two casts.

For S=2, the next conv state retains one row from the old conv state.
Longer verify windows fill the entire next state from the new qkv rows.
Only the target-verify arm routes here; decode (S=1) and prefill keep the
stock path.
"""

from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_PATCHED = False
_ENGAGED_LOGGED = False
_KERNEL = None
_QWEN4_DECODE_KERNEL = None
_QWEN4_NORM_GATE_KERNEL = None
_QWEN4_DECODE_ENGAGED_LOGGED = False

_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint batch_idx = threadgroup_position_in_grid.y / uint(S);
    uint row = threadgroup_position_in_grid.y % uint(S);
    uint logical_head = threadgroup_position_in_grid.z;
    constexpr uint q_heads = uint(HK);
    constexpr uint k_head_base = uint(HK);
    constexpr uint v_head_base = 2 * uint(HK);
    bool is_q = logical_head < q_heads;
    bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    uint head = is_q ? logical_head
               : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
    uint channel_base = is_q ? head * uint(DK)
                       : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                               : 2 * uint(HK) * uint(DK) + head * uint(DV));
    T activated[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint channel = channel_base + lane * 4 + i;
        float acc = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            uint input_row = row + tap;
            const T xv = input_row < uint(NKEEP)
                ? conv_state[(batch_idx * uint(NKEEP) + input_row) * uint(C) + channel]
                : qkv[(batch_idx * uint(S) + input_row - uint(NKEEP)) * uint(C) + channel];
            acc += float(xv) * float(conv_w[channel * 4 + tap]);
        }
        const T conv = T(acc);
        T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
        const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
        activated[i] = act;
        float value = float(act);
        sumsq += value * value;
    }
    if (is_q || is_k) {
        sumsq = simd_sum(sumsq);
        float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
        const T scale = is_q ? q_scale : k_scale;
        uint out_base = ((batch_idx * uint(S) + row) * uint(HK) + head) * uint(DK) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            const T rms = T(1) * T(float(activated[i]) * inv);
            const T value = scale * rms;
        if (is_q) {
            q_out[out_base + i] = value;
        } else {
            k_out[out_base + i] = value;
        }
        }
    } else {
        uint out_base = ((batch_idx * uint(S) + row) * uint(HV) + head) * uint(DV) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            v_out[out_base + i] = activated[i];
        }
    }
    if (S < NKEEP && row == 0) {
        for (uint old_row = 0; old_row < uint(NKEEP - S); ++old_row) {
            uint dst = (batch_idx * uint(NKEEP) + old_row) * uint(C)
                       + channel_base + lane * 4;
            uint src = (batch_idx * uint(NKEEP) + old_row + uint(S)) * uint(C)
                       + channel_base + lane * 4;
            for (uint i = 0; i < 4; ++i) {
                conv_out[dst + i] = conv_state[src + i];
            }
        }
    }
    if (row + uint(NKEEP) >= uint(S)) {
        uint state_row = row + uint(NKEEP) - uint(S);
        uint raw_base = (batch_idx * uint(S) + row) * uint(C) + channel_base + lane * 4;
        uint state_base = (batch_idx * uint(NKEEP) + state_row) * uint(C) + channel_base + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            conv_out[state_base + i] = qkv[raw_base + i];
        }
    }
"""


# Copyright (c) 2026 David Dalcu.  The Qwen4 decode prework and norm-gate
# kernels below are adapted from ddalcu/mlx-serve's MIT-licensed
# ``src/transformer.zig`` at tag ``v26.8.11-pre-release.1``.  Preserve this
# scoped notice with those kernels.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Qwen4 decode-only continuation of the donor kernel.  This is deliberately a
# separate ABI from the verify kernel above: the production Qwen4 recurrence
# consumes an FP32 forget gate, while the older donor stores that gate as BF16.
# Keeping distinct outputs lets this path match mlx-vlm's current arithmetic
# exactly instead of silently changing the recurrent state update.
_QWEN4_DECODE_HEADER = """
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


_QWEN4_DECODE_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint logical_head = threadgroup_position_in_grid.z;
    constexpr uint q_heads = uint(HK);
    constexpr uint k_head_base = uint(HK);
    constexpr uint v_head_base = 2 * uint(HK);
    bool is_q = logical_head < q_heads;
    bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    uint head = is_q ? logical_head
               : (is_k ? logical_head - k_head_base
                       : logical_head - v_head_base);
    uint channel_base = is_q ? head * uint(DK)
                       : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                               : 2 * uint(HK) * uint(DK) + head * uint(DV));

    T activated[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint channel = channel_base + lane * 4 + i;
        float acc = 0.0f;
        for (uint tap = 0; tap < 3; ++tap) {
            acc += float(conv_state[tap * uint(C) + channel])
                 * float(conv_w[channel * 4 + tap]);
        }
        acc += float(qkv[channel]) * float(conv_w[channel * 4 + 3]);
        const T conv = T(acc);
        T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
        const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
        activated[i] = act;
        float value = float(act);
        sumsq += value * value;

        // T=1: [old0, old1, old2, qkv] -> [old1, old2, qkv].  Each
        // channel is owned by exactly one thread, so no synchronization is
        // needed between these three stores.
        conv_out[channel] = conv_state[uint(C) + channel];
        conv_out[uint(C) + channel] = conv_state[2 * uint(C) + channel];
        conv_out[2 * uint(C) + channel] = qkv[channel];
    }

    if (is_q || is_k) {
        sumsq = simd_sum(sumsq);
        float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
        float scale = is_q ? float(q_scale) : float(k_scale);
        uint out_base = head * uint(DK) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            // rms_norm returns T; multiplying by the Python scalar also
            // returns T, so preserve both rounding sites.
            const T rms = T(float(activated[i]) * inv);
            const T value = T(float(rms) * scale);
            if (is_q) {
                q_out[out_base + i] = value;
            } else {
                k_out[out_base + i] = value;
            }
        }
    } else {
        uint out_base = head * uint(DV) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            v_out[out_base + i] = activated[i];
        }
        if (lane == 0) {
            const T bv = b_in[head];
            T by = T(1) / (T(1) + metal::exp(metal::abs(bv)));
            beta_out[head] = (bv < T(0)) ? by : T(1) - by;

            // compute_g casts A_log to FP32 but keeps softplus(a+dt_bias)
            // in BF16 before the FP32 multiply and outer exp.
            const T apd = T(float(a_in[head]) + float(dt_bias[head]));
            const T neg_abs = -metal::abs(apd);
            const T exp_term = T(metal::precise::exp(float(neg_abs)));
            const T log_term = T(omlx_log1p(float(exp_term)));
            const T positive = metal::max(apd, T(0));
            const T sp = T(float(positive) + float(log_term));
            float ea = metal::precise::exp(float(A_log[head]));
            g_out[head] = metal::precise::exp(-(ea * float(sp)));
        }
    }
"""


_QWEN4_NORM_GATE_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint head = threadgroup_position_in_grid.z;
    uint base = head * uint(DV) + lane * 4;
    float xs[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        xs[i] = float(y[base + i]);
        sumsq += xs[i] * xs[i];
    }
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DV) + float(eps));
    for (uint i = 0; i < 4; ++i) {
        // mx.fast.rms_norm materializes BF16 before Qwen4 casts it back to
        // FP32 for the sigmoid product.
        const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
        float zv = float(z[base + i]);
        float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
        float sig = zv < 0.0f ? sy : 1.0f - sy;
        out[base + i] = T(float(normed) * sig);
    }
"""


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen35_gdn_prework",
            input_names=["qkv", "conv_state", "conv_w", "q_scale", "k_scale"],
            output_names=["q_out", "k_out", "v_out", "conv_out"],
            source=_SOURCE,
        )
    return _KERNEL


def gdn_prework_fused(qkv, conv_state, conv_w, q_scale, k_scale, hk, hv, dk, dv):
    """One fused dispatch. qkv [B,S,C], conv_state [B,3,C], conv_w [C,4,1]."""
    batch_size = qkv.shape[0]
    s_len = qkv.shape[1]
    c_dim = qkv.shape[2]
    outs = _kernel()(
        inputs=[qkv, conv_state, conv_w, q_scale, k_scale],
        template=[
            ("T", qkv.dtype),
            ("HK", hk),
            ("HV", hv),
            ("DK", dk),
            ("DV", dv),
            ("NKEEP", 3),
            ("C", c_dim),
            ("S", s_len),
        ],
        grid=(32, batch_size * s_len, 2 * hk + hv),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (batch_size, s_len, hk, dk),
            (batch_size, s_len, hk, dk),
            (batch_size, s_len, hv, dv),
            (batch_size, 3, c_dim),
        ],
        output_dtypes=[qkv.dtype] * 4,
    )
    return outs


def _qwen4_decode_kernel():
    global _QWEN4_DECODE_KERNEL
    if _QWEN4_DECODE_KERNEL is None:
        _QWEN4_DECODE_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_gdn_decode_prework",
            input_names=[
                "qkv",
                "conv_state",
                "conv_w",
                "q_scale",
                "k_scale",
                "b_in",
                "a_in",
                "A_log",
                "dt_bias",
            ],
            output_names=[
                "q_out",
                "k_out",
                "v_out",
                "conv_out",
                "g_out",
                "beta_out",
            ],
            header=_QWEN4_DECODE_HEADER,
            source=_QWEN4_DECODE_SOURCE,
        )
    return _QWEN4_DECODE_KERNEL


def qwen4_decode_prework_fused(
    qkv,
    conv_state,
    conv_w,
    q_scale,
    k_scale,
    b,
    a,
    A_log,
    dt_bias,
    hk,
    hv,
    dk,
    dv,
):
    """Fuse the exact Qwen4 B1/T1 GDN prework into one Metal dispatch."""

    c_dim = int(qkv.shape[-1])
    return _qwen4_decode_kernel()(
        inputs=[
            qkv,
            conv_state,
            conv_w,
            q_scale,
            k_scale,
            b,
            a,
            A_log,
            dt_bias,
        ],
        template=[
            ("T", qkv.dtype),
            ("HK", hk),
            ("HV", hv),
            ("DK", dk),
            ("DV", dv),
            ("C", c_dim),
        ],
        grid=(32, 1, 2 * hk + hv),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, 1, hk, dk),
            (1, 1, hk, dk),
            (1, 1, hv, dv),
            (1, 3, c_dim),
            (1, 1, hv),
            (1, 1, hv),
        ],
        output_dtypes=[
            qkv.dtype,
            qkv.dtype,
            qkv.dtype,
            qkv.dtype,
            mx.float32,
            qkv.dtype,
        ],
    )


def _qwen4_norm_gate_kernel():
    global _QWEN4_NORM_GATE_KERNEL
    if _QWEN4_NORM_GATE_KERNEL is None:
        _QWEN4_NORM_GATE_KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_gdn_decode_norm_gate",
            input_names=["y", "z", "norm_w", "eps"],
            output_names=["out"],
            source=_QWEN4_NORM_GATE_SOURCE,
        )
    return _QWEN4_NORM_GATE_KERNEL


def qwen4_decode_norm_gate_fused(y, z, norm_w, *, hv, dv, eps):
    """Fuse Qwen4's BF16 RMSNorm + FP32 sigmoid-gate at B1/T1."""

    return _qwen4_norm_gate_kernel()(
        inputs=[y, z, norm_w, mx.array(eps, dtype=mx.float32)],
        template=[
            ("T", y.dtype),
            ("HV", hv),
            ("DV", dv),
        ],
        grid=(32, 1, hv),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, 1, hv * dv)],
        output_dtypes=[y.dtype],
    )[0]


def _qwen4_decode_recurrence(q, k, v, g, beta, state):
    from mlx_lm.models.gated_delta import gated_delta_kernel

    return gated_delta_kernel(q, k, v, g, beta, state, None)


def _qwen4_decode_static_eligible(module) -> bool:
    """Fail closed unless this is the shipped Qwen4 oQe decode geometry."""

    conv_dim = 2 * 16 * 128 + 48 * 128
    if (
        type(module).__name__ != "Qwen4ExpGatedDeltaNet"
        or type(module).__module__ != "mlx_vlm.models.qwen4_exp.language"
        or module.training
        or module.num_k_heads != 16
        or module.num_v_heads != 48
        or module.head_k_dim != 128
        or module.head_v_dim != 128
        or module.conv_kernel_size != 4
    ):
        return False

    conv = getattr(module, "conv1d", None)
    norm = getattr(module, "norm", None)
    if (
        conv is None
        or getattr(conv, "bias", None) is not None
        or conv.weight.shape != (conv_dim, 4, 1)
        or conv.weight.dtype != mx.bfloat16
        or norm is None
        or getattr(norm, "activation", None) != "sigmoid"
        or norm.weight.shape != (128,)
        or norm.weight.dtype != mx.bfloat16
        or module.A_log.shape != (48,)
        or module.A_log.dtype != mx.bfloat16
        or module.dt_bias.shape != (48,)
        or module.dt_bias.dtype != mx.bfloat16
    ):
        return False

    def canonical_projection(linear, rows, signatures):
        if type(linear) is not nn.QuantizedLinear or linear.mode != "affine":
            return False
        signature = (linear.bits, linear.group_size)
        if signature not in signatures:
            return False
        bits, group_size = signature
        packed_cols = 2560 * bits // 32
        scale_cols = 2560 // group_size
        return (
            linear.weight.shape == (rows, packed_cols)
            and linear.weight.dtype == mx.uint32
            and linear.scales.shape == (rows, scale_cols)
            and linear.scales.dtype == mx.bfloat16
            and linear.biases is not None
            and linear.biases.shape == (rows, scale_cols)
            and linear.biases.dtype == mx.bfloat16
            and "bias" not in linear
        )

    # The shipped oQe allocation is intentionally mixed per tensor.  This
    # kernel begins after those projections, so accept only the exact
    # canonical layouts emitted by the converter rather than demanding that
    # all four happen to share layer 0's q6/g64 allocation.
    if not canonical_projection(
        module.in_proj_qkv,
        conv_dim,
        {(4, 64), (5, 64), (6, 64)},
    ):
        return False
    for linear, rows in (
        (module.in_proj_z, 48 * 128),
        (module.in_proj_b, 48),
        (module.in_proj_a, 48),
    ):
        if not canonical_projection(linear, rows, {(5, 128), (6, 64)}):
            return False

    out = module.out_proj
    return (
        type(out) is nn.QuantizedLinear
        and out.bits == 5
        and out.group_size == 128
        and out.mode == "affine"
        and out.weight.shape == (2560, 960)
        and out.weight.dtype == mx.uint32
        and out.scales.shape == (2560, 48)
        and out.scales.dtype == mx.bfloat16
        and out.biases is not None
        and out.biases.shape == (2560, 48)
        and out.biases.dtype == mx.bfloat16
        and "bias" not in out
    )


def _qwen4_decode_dynamic_eligible(
    module,
    inputs,
    mask,
    cache,
    gdn_sink,
    target_verify,
) -> bool:
    if (
        target_verify
        or gdn_sink is not None
        or mask is not None
        or cache is None
        or not isinstance(inputs, mx.array)
        or inputs.shape != (1, 1, 2560)
        or inputs.dtype != mx.bfloat16
        or getattr(cache, "lengths", None) is not None
        or getattr(cache, "left_padding", None) is not None
    ):
        return False
    conv_state = cache[0]
    recurrent_state = cache[1]
    return (
        isinstance(conv_state, mx.array)
        and conv_state.shape == (1, 3, 10240)
        and conv_state.dtype == mx.bfloat16
        and isinstance(recurrent_state, mx.array)
        and recurrent_state.shape == (1, 48, 128, 128)
        and recurrent_state.dtype == mx.float32
        and _qwen4_decode_static_eligible(module)
    )


def apply_qwen35_gdn_prework_patch() -> bool:
    """Install fused prework at ordinary decode and speculative entry points."""
    global _PATCHED
    if _PATCHED:
        return True
    if not mx.metal.is_available():
        return False

    from mlx_vlm.models.qwen3_5 import language as q35
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward
    from mlx_vlm.speculative.ops.linear import _target_verify_linears

    cls = q35.Qwen3_5GatedDeltaNet
    original = cls.__call__
    original_verify = Qwen3_5BatchInvariantForward._gated_delta

    def decode(self, inputs, mask=None, cache=None):
        if not _qwen4_decode_dynamic_eligible(self, inputs, mask, cache, None, False):
            return original(self, inputs, mask=mask, cache=cache)
        mixed_qkv, z, b, a = _target_verify_linears(
            (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a), inputs
        )
        inv = self.head_k_dim**-0.5
        q, k, v, conv_state, g, beta = qwen4_decode_prework_fused(
            mixed_qkv,
            cache[0],
            self.conv1d.weight,
            mx.array(inv * inv, dtype=mx.bfloat16),
            mx.array(inv, dtype=mx.bfloat16),
            b,
            a,
            self.A_log,
            self.dt_bias,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        )
        out, state = _qwen4_decode_recurrence(q, k, v, g, beta, cache[1])
        flat = qwen4_decode_norm_gate_fused(
            out,
            z,
            self.norm.weight,
            hv=self.num_v_heads,
            dv=self.head_v_dim,
            eps=self.norm.eps,
        )
        result = self.out_proj(flat)
        cache[0], cache[1] = conv_state, state
        if hasattr(cache, "advance"):
            cache.advance(1)
            q35._qwen3_5_advance_left_padding_info(cache, 1)
            q35._qwen3_5_advance_lengths_info(cache, 1)
        global _QWEN4_DECODE_ENGAGED_LOGGED
        if not _QWEN4_DECODE_ENGAGED_LOGGED:
            _QWEN4_DECODE_ENGAGED_LOGGED = True
            logger.info("Qwen4 fused B1/T1 GDN decode prework and norm-gate engaged")
        return result

    def verify(verifier, layer, inputs, mask, cache):
        length = inputs.shape[1]
        # The fused prework uses Qwen3.5 RMS scaling, not Qwen4 L2 scaling.
        compatible_norm = (
            type(verifier)._normalize_gated_delta_qk
            is Qwen3_5BatchInvariantForward._normalize_gated_delta_qk
        )
        if not (
            compatible_norm
            and cache is not None
            and cache.is_speculating
            and 2 <= length <= 9
            and mask is None
            and inputs.dtype == mx.bfloat16
            and layer.conv_kernel_size == 4
            and layer.head_k_dim == 128
            and layer.head_v_dim == 128
            and cache.lengths is None
            and cache[0] is not None
            and cache[0].shape[0] == inputs.shape[0]
            and cache[0].dtype == mx.bfloat16
            and layer.conv1d.weight.dtype == mx.bfloat16
            and getattr(layer.conv1d, "bias", None) is None
        ):
            return original_verify(verifier, layer, inputs, mask, cache)
        mixed_qkv, z, b, a = verifier._linears(
            (layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a),
            inputs,
        )
        inv = layer.head_k_dim**-0.5
        q, k, v, conv_state = gdn_prework_fused(
            mixed_qkv,
            cache[0],
            layer.conv1d.weight,
            mx.array(inv * inv, dtype=mx.bfloat16),
            mx.array(inv, dtype=mx.bfloat16),
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_k_dim,
            layer.head_v_dim,
        )
        conv_input = mx.concatenate([cache[0], mixed_qkv], axis=1)
        cache.record_speculative_window(0, conv_input, layer.conv_kernel_size - 1)
        cache[0] = conv_state
        out, _ = q35.gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            layer.A_log,
            layer.dt_bias,
            cache=cache,
            use_kernel=not layer.training,
        )
        if hasattr(cache, "advance"):
            cache.advance(length)
            q35._qwen3_5_advance_left_padding_info(cache, length)
            q35._qwen3_5_advance_lengths_info(cache, length)
        out = layer.norm(out, z.reshape(inputs.shape[0], length, -1, layer.head_v_dim))
        result = verifier._linear(
            layer.out_proj, out.reshape(inputs.shape[0], length, -1)
        )
        global _ENGAGED_LOGGED
        if not _ENGAGED_LOGGED:
            _ENGAGED_LOGGED = True
            logger.info("[gdn-prework] fused verify prework engaged (S=%d)", length)
        return result

    cls.__call__ = decode
    cls._omlx_gdn_prework_patched = True
    Qwen3_5BatchInvariantForward._gated_delta = verify
    _PATCHED = True
    logger.info("Qwen fused GDN prework patch applied")
    return True
