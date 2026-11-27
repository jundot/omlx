# SPDX-License-Identifier: Apache-2.0
"""Fused split affine expert GEMVs and clamped SwiGLU for Hy4 verify."""

import logging
from functools import cache

import mlx.core as mx

from ...custom_kernels.quant_matmul import fast as quant_fast
from ..deepseek_v4.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _sort_threshold,
)
from ..moe_verify_gather import _HEADER, gather_qmv, pair_order

logger = logging.getLogger(__name__)


_GATE_UP = r"""
    const int p = int(order[threadgroup_position_in_grid.y]);
    const int lane = int(thread_index_in_simdgroup);
    const int out = int(threadgroup_position_in_grid.z) * COLS * GROUPS
        + int(simdgroup_index_in_threadgroup) * COLS;
    const size_t e = size_t(rhs[p]);
    const int ws = K * BYTES_PER_PACK / PACK_FACTOR;
    const int ss = K / GS;
    const size_t offset = e * N + out;
    const size_t wp = offset * ws + lane * PACKS_PER_THREAD * BYTES_PER_PACK;
    const size_t sp = offset * ss + lane / SCALE_STEP_PER_THREAD;
    const device T* xp = x + (p / TOP) * K + lane * VALUES_PER_THREAD;
    float gate[COLS] = {0}, up[COLS] = {0};

    for (int k = 0; k < K; k += BLOCK_SIZE) {
        float values[VALUES_PER_THREAD];
        const float sum = load_vector(xp + k, values);
        for (int c = 0; c < COLS; ++c) {
            const size_t wi = wp + c * ws + k * BYTES_PER_PACK / PACK_FACTOR;
            const size_t si = sp + c * ss + k / GS;
            gate[c] += qdot_n(
                (const device uchar*)gw + wi,
                values, gs[si], gb[si], sum, VALUES_PER_THREAD);
            up[c] += qdot_n(
                (const device uchar*)uw + wi,
                values, us[si], ub[si], sum, VALUES_PER_THREAD);
        }
    }
    for (int c = 0; c < COLS; ++c) {
        const float g1 = simd_sum(gate[c]);
        const float u1 = simd_sum(up[c]);
        if (lane == 0) {
            const T g = metal::min(T(g1), limit[0]);
            const T u = metal::min(metal::max(T(u1), T(-limit[0])), limit[0]);
            const auto a = 1 / (1 + metal::precise::exp(metal::abs(g)));
            const T sigmoid = (g < 0) ? a : 1 - a;
            y[p * N + out + c] = T(T(g * sigmoid) * u);
        }
    }
"""


@cache
def _gate_up_kernel(bits, group):
    header = (
        _HEADER.replace("__BITS__", str(bits))
        .replace("__GS__", str(group))
        .replace("__FAST__", "1")
    )
    logger.debug(
        "Hy4 short verify: fused affine expert gate/up (bits=%d, group=%d)", bits, group
    )
    return mx.fast.metal_kernel(
        name=f"hy4_decode_split_gate_up_b{bits}_g{group}",
        input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "rhs", "order", "limit"],
        output_names=["y"],
        header=header,
        source=_GATE_UP,
    )


def _gate_up(x, gate, up, rhs, order, top, limit):
    cols, groups = 2, 2

    def operands(proj):
        return [proj["weight"], proj["scales"], proj["biases"]]

    n = gate.output_dims
    return _gate_up_kernel(gate.bits, gate.group_size)(
        inputs=[
            x,
            *operands(gate),
            *operands(up),
            rhs,
            order,
            mx.array([limit], dtype=x.dtype),
        ],
        template=[
            ("T", x.dtype),
            ("K", x.shape[-1]),
            ("N", n),
            ("TOP", top),
            ("COLS", cols),
            ("GROUPS", groups),
        ],
        grid=(32 * groups, rhs.size, n // (cols * groups)),
        threadgroup=(32 * groups, 1, 1),
        output_shapes=[(rhs.size, n)],
        output_dtypes=[x.dtype],
    )[0]


def routed_experts(switch, routed, indices, scores, limit):
    """Preserve router slot order and the stock expert reduction."""
    if (
        type(switch) is not SwitchGLU
        or switch.training
        or routed.shape[-1] != 1536
        or indices.shape != (routed.shape[0], 16)
        or "gate_up_proj" in switch
        or "oqh_experts" in switch
        or routed.dtype not in (mx.bfloat16, mx.float16)
    ):
        return None
    projections = switch.projections()
    if not all(
        type(proj) is QuantizedSwitchLinear
        and proj.bits in (3, 4)
        and proj.group_size in (32, 64, 128)
        and proj.input_dims == proj.output_dims == 1536
        and proj._has_affine_metadata_dtype(routed.dtype)
        and "bias" not in proj
        and "global_scale" not in proj
        for proj in projections
    ):
        return None
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    if (gate.bits, gate.group_size, gate.weight.shape) != (
        up.bits,
        up.group_size,
        up.weight.shape,
    ):
        return None

    do_sort = indices.size >= _sort_threshold(*projections)
    # MLX 0.32.4 GatherQMM::eval_gpu switches sorted one-row gathers to a
    # matrix kernel at four routes per expert bank entry. Its rounding is
    # different from the GEMV arithmetic reproduced below.
    if do_sort and any(indices.size >= 4 * proj.num_experts for proj in projections):
        return None

    rhs = indices.astype(mx.uint32).reshape(-1)
    order = pair_order(rhs)
    hidden = _gate_up(routed, gate, up, rhs, order, indices.shape[-1], limit)
    values = gather_qmv(down, hidden, rhs, 1, order)
    if (
        scores.dtype == mx.float32
        and do_sort
        and scores.shape[-1] in quant_fast.moe_weighted_sum_top_ks()
        and quant_fast.has_symbol("moe_weighted_sum")
    ):
        identity = mx.arange(rhs.size, dtype=mx.uint32)
        return quant_fast.moe_weighted_sum(
            values.reshape(rhs.size, 1, -1), identity, scores
        )
    return values.reshape(*indices.shape, down.output_dims)
