"""Fused KDA (Glm5NextLinearAttention) prefill prework and norm-gate.

Port of the Qwen4 fused-GDN-prefill route to GLM-5.3-Flash. The stock
linear-attention prefill runs conv1d + SiLU + split + L2-normalize + scale
and the gated RMSNorm epilogue as a chain of separate elementwise ops over
every chunk row; on 8192-row wide chunks that is a launch- and
bandwidth-bound tail around the recurrent kernel. This module fuses each
side into one Metal dispatch:

* prework: short conv (4 taps) + SiLU + L2(q/k) + q scale + conv-state
  rolling, one simdgroup per (row, q/k/v head).
* norm-gate: fp32 RMSNorm + weight + sigmoid gate, one simdgroup per
  (row, head).

Both kernels mirror the stock rounding sites so the fused path is
bit-compatible with the unfused stock path (fp32 L2 sums, a single cast
back to the activation dtype, bf16-domain conv/SiLU). The forget gate and
the recurrent delta kernel are untouched; the driver calls the stock
``gated_delta_update`` on the fused outputs.

Eligibility is fail-closed (see ``glm53_kda_prefill_eligible``); anything
unexpected runs the stock path. Kill switch:
``OMLX_GLM53_KDA_PREFILL_FUSED=0``.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_GLM53_KDA_PREFILL_ENABLED = (
    os.environ.get("OMLX_GLM53_KDA_PREFILL_FUSED", "1") != "0"
)
_GLM53_KDA_PREFILL_MIN_ROWS = 64

_PREWORK_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint lh = threadgroup_position_in_grid.z;
    const uint S_rt = uint(s_len);
    bool is_q = lh < uint(H);
    bool is_k = lh >= uint(H) && lh < 2u * uint(H);
    uint head = is_q ? lh : (is_k ? lh - uint(H) : lh - 2u * uint(H));
    uint channel_base = head * uint(D)
        + (is_q ? 0u : (is_k ? uint(H) * uint(D) : 2u * uint(H) * uint(D)));

    T activated[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        uint channel = channel_base + lane * 4 + i;
        float acc = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            uint input_row = row + tap;
            const T xv = input_row < uint(NKEEP)
                ? conv_state[input_row * uint(C) + channel]
                : qkv[(input_row - uint(NKEEP)) * uint(C) + channel];
            acc += float(xv) * float(conv_w[channel * 4 + tap]);
        }
        const T conv = T(acc);
        T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
        const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
        activated[i] = act;
        if (is_q || is_k) {
            const float f = float(act);
            sumsq += f * f;
        }
    }

    if (is_q || is_k) {
        // Stock L2 chain: x * rsqrt(sum(square(x), -1) + 1e-6) in FP32 with
        // the dk^-0.5 query scale applied after the normalize, cast back to
        // T once. Mirror the per-lane sequential partial sums and the fp32
        // xor reduction tree of mx.sum over the head axis.
        float tv = sumsq;
        tv += simd_shuffle_xor(tv, short(16));
        tv += simd_shuffle_xor(tv, short(8));
        tv += simd_shuffle_xor(tv, short(4));
        tv += simd_shuffle_xor(tv, short(2));
        tv += simd_shuffle_xor(tv, short(1));
        const float inv = metal::precise::rsqrt(tv + 1e-6f);
        uint out_base = (row * uint(H) + head) * uint(D) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            const float l2 = float(activated[i]) * inv;
            const T value = is_q ? T(l2 * float(q_scale)) : T(l2);
            if (is_q) {
                q_out[out_base + i] = value;
            } else {
                k_out[out_base + i] = value;
            }
        }
    } else {
        uint out_base = (row * uint(H) + head) * uint(D) + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            v_out[out_base + i] = activated[i];
        }
    }

    if (S_rt < uint(NKEEP) && row == 0) {
        for (uint old_row = 0; old_row < uint(NKEEP) - S_rt; ++old_row) {
            uint dst = old_row * uint(C) + channel_base + lane * 4;
            uint src = (old_row + S_rt) * uint(C) + channel_base + lane * 4;
            for (uint i = 0; i < 4; ++i) {
                conv_out[dst + i] = conv_state[src + i];
            }
        }
    }
    if (row + uint(NKEEP) >= S_rt) {
        uint state_row = row + uint(NKEEP) - S_rt;
        uint raw_base = row * uint(C) + channel_base + lane * 4;
        uint state_base = state_row * uint(C) + channel_base + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            conv_out[state_base + i] = qkv[raw_base + i];
        }
    }
"""

_NORM_GATE_SOURCE = """
    uint lane = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint head = threadgroup_position_in_grid.z;
    uint base = (row * uint(H) + head) * uint(D) + lane * 4;
    float xs[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        xs[i] = float(y[base + i]);
        sumsq += xs[i] * xs[i];
    }
    // Glm5NextRMSNormGated runs entirely in FP32 and casts once at the end:
    // x * rsqrt(mean(x^2) + eps), then weight, then sigmoid(gate).
    float tv = sumsq;
    tv += simd_shuffle_xor(tv, short(16));
    tv += simd_shuffle_xor(tv, short(8));
    tv += simd_shuffle_xor(tv, short(4));
    tv += simd_shuffle_xor(tv, short(2));
    tv += simd_shuffle_xor(tv, short(1));
    float inv = metal::precise::rsqrt(tv / float(D) + float(eps));
    for (uint i = 0; i < 4; ++i) {
        float t = float(norm_w[lane * 4 + i]) * (xs[i] * inv);
        float zv = float(z[base + i]);
        float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(zv)));
        float sig = zv < 0.0f ? sy : 1.0f - sy;
        out[base + i] = T(t * sig);
    }
"""

_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = (
            mx.fast.metal_kernel(
                name="omlx_glm53_kda_prefill_prework",
                input_names=["qkv", "conv_state", "conv_w", "q_scale", "s_len"],
                output_names=["q_out", "k_out", "v_out", "conv_out"],
                source=_PREWORK_SOURCE,
            ),
            mx.fast.metal_kernel(
                name="omlx_glm53_kda_prefill_norm_gate",
                input_names=["y", "z", "norm_w", "eps"],
                output_names=["out"],
                source=_NORM_GATE_SOURCE,
            ),
        )
    return _KERNELS


def kda_prework_fused(mixed, conv_state, conv_w, q_scale, length, heads, dim):
    """Fused conv+SiLU+L2 prework for one prefill chunk.

    mixed [1,S,3*heads*dim] in the activation dtype, conv_state [1,3,C],
    conv_w [C,1,4]. Returns (q, k, v, next_conv).
    """
    prework, _ = _kernels()
    c_dim = 3 * heads * dim
    return prework(
        inputs=[mixed, conv_state, conv_w, q_scale, mx.array(length, dtype=mx.int32)],
        template=[
            ("T", mixed.dtype),
            ("H", heads),
            ("D", dim),
            ("C", c_dim),
            ("NKEEP", 3),
        ],
        grid=(32, length, 3 * heads),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, length, heads, dim),
            (1, length, heads, dim),
            (1, length, heads, dim),
            (1, 3, c_dim),
        ],
        output_dtypes=[mixed.dtype] * 4,
    )


def kda_norm_gate_fused(y, gate, norm_w, eps, heads, dim):
    """Fused Glm5NextRMSNormGated epilogue. y, gate [1,S,heads,dim]."""
    _, norm_gate = _kernels()
    return norm_gate(
        inputs=[y, gate, norm_w, mx.array(eps, dtype=mx.float32)],
        template=[("T", y.dtype), ("H", heads), ("D", dim)],
        grid=(32, y.shape[1], heads),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, y.shape[1], heads * dim)],
        output_dtypes=[y.dtype],
    )[0]


def glm53_kda_prefill_eligible(module, inputs, mask, cache) -> bool:
    if (
        not _GLM53_KDA_PREFILL_ENABLED
        or mask is not None
        or cache is None
        or not isinstance(inputs, mx.array)
        or inputs.ndim != 3
        or inputs.shape[0] != 1
        or inputs.shape[1] < _GLM53_KDA_PREFILL_MIN_ROWS
        or inputs.dtype != mx.bfloat16
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or getattr(module, "conv_kernel_size", 0) != 4
        or getattr(module, "head_dim", 0) != 128
        or not getattr(module, "num_heads", 0)
        or module.qkv_dim != module.num_heads * module.head_dim
        or len(getattr(cache, "cache", ())) != 2
        or getattr(cache, "is_speculating", True)
        or getattr(cache, "history_capacity", 0)
        or getattr(cache, "lengths", None) is not None
        or getattr(cache, "left_padding", None) is not None
    ):
        return False
    conv_state, recurrent_state = cache[0], cache[1]
    if conv_state is not None and not (
        isinstance(conv_state, mx.array)
        and conv_state.shape == (1, 3, module.conv_dim)
        and conv_state.dtype == inputs.dtype
    ):
        return False
    if recurrent_state is None:
        return True
    return bool(
        isinstance(recurrent_state, mx.array)
        and recurrent_state.shape
        == (1, module.num_heads, module.head_dim, module.head_dim)
        and recurrent_state.dtype == mx.float32
    )


_GLM53_KDA_ENGAGED_LOGGED = False


def glm53_kda_prefill(module, inputs, cache):
    """Stock GLM-5.3 KDA prefill with the conv/L2 prework and norm-gate fused."""
    from mlx_vlm.models.glm5_next import language as lang

    global _GLM53_KDA_ENGAGED_LOGGED
    length = inputs.shape[1]
    heads, dim = module.num_heads, module.head_dim
    if module.fuse_in:
        q_o, k_o, v_o, fa_o, ga_o, b_o = module._fused_in_proj(inputs)
        mixed = mx.concatenate([q_o, k_o, v_o], axis=-1)
    else:
        mixed = mx.concatenate(
            [module.q_proj(inputs), module.k_proj(inputs), module.v_proj(inputs)],
            axis=-1,
        )
        fa_o = module.forget_gate.f_a_proj(inputs)
        ga_o = module.g_a_proj(inputs)
        b_o = module.b_proj(inputs)

    conv_state = cache[0]
    if conv_state is None:
        conv_state = mx.zeros((1, 3, module.conv_dim), dtype=inputs.dtype)
    q, k, v, next_conv = kda_prework_fused(
        mixed,
        conv_state,
        module.conv1d.weight,
        mx.array(dim**-0.5, dtype=mx.float32),
        length,
        heads,
        dim,
    )
    cache[0] = next_conv

    fg = module.forget_gate
    a = lang.linear_forward(fg.f_b_proj, fa_o).reshape(1, length, heads, dim)
    state = cache[1]
    out, state = lang.gated_delta_update(
        q,
        k,
        v,
        a,
        b_o,
        fg.A_log.reshape(heads, 1),
        fg.dt_bias.reshape(heads, dim),
        state=state,
        lower_bound=fg.safe_gate_lower_bound,
    )
    cache[1] = state
    cache.advance(length)

    gate = lang.linear_forward(module.g_b_proj, ga_o).reshape(1, length, heads, dim)
    flat = kda_norm_gate_fused(
        out, gate, module.o_norm.weight, module.o_norm.eps, heads, dim
    )
    if not _GLM53_KDA_ENGAGED_LOGGED:
        _GLM53_KDA_ENGAGED_LOGGED = True
        logger.info("GLM-5.3 fused KDA prefill prework and norm-gate engaged")
    return lang.linear_forward(module.o_proj, flat)
