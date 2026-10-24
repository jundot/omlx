"""Routed NVFP4 experts for decode and short verify calls.

``scaled_switch_glu`` runs ``MiniMaxScaledSwitchGLU`` for a few token rows on
the shared FP4 expert kernels (``omlx.patches.fp4_expert_rows``), bit-exact to
MLX's per-pair ``gather_qmv``. The epilogue repeats ``_scaled_swiglu_oai`` op
for op in float32 on the rounded projection outputs.
"""

from __future__ import annotations

import mlx.core as mx

from omlx.patches import fp4_expert_rows

MAX_ROWS = 8

_SCALED_SWIGLU_OAI = fp4_expert_rows.Activation(
    name="scaled_swiglu_oai",
    source=r"""
    const float limit = params[1];
    float x_glu = float(static_cast<T>(g)) * float(gate_scale[e]);
    x_glu = fp4_minimum(x_glu, limit);
    float x_linear = float(static_cast<T>(u)) * float(up_scale[e]);
    x_linear = fp4_minimum(fp4_maximum(x_linear, -limit), limit);
    const float sig = fp4_sigmoid(params[0] * x_glu);
    act[out] = static_cast<T>(x_glu * sig * (x_linear + params[2]));
""",
    inputs=("gate_scale", "up_scale", "params"),
)


def _nvfp4(linear, k: int) -> bool:
    return (
        getattr(linear, "mode", None) == "nvfp4"
        and getattr(linear, "bits", None) == 4
        and getattr(linear, "group_size", None) == 16
        and linear.get("biases") is None
        and linear.get("bias") is None
        and linear.weight.dtype == mx.uint32
        and linear.scales.dtype == mx.uint8
        and linear.weight.shape[-1] * 8 == k
        and k % fp4_expert_rows.BLOCK == 0
    )


def scaled_switch_glu(switch, x: mx.array, indices: mx.array) -> mx.array | None:
    """Routed experts of ``MiniMaxScaledSwitchGLU`` for a few rows, or None."""
    hidden = x.shape[-1]
    topk = indices.shape[-1]
    pairs = indices.size
    rows = pairs // topk
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    inter = down.weight.shape[-1] * 8
    act_fn = switch.activation
    if not (
        rows <= MAX_ROWS
        and x.size == rows * hidden
        and x.dtype in (mx.bfloat16, mx.float16)
        and _nvfp4(gate, hidden)
        and _nvfp4(up, hidden)
        and _nvfp4(down, inter)
        and gate.weight.shape[1] == up.weight.shape[1] == inter
        and inter % 8 == 0
        and hidden % 8 == 0
        and mx.metal.is_available()
    ):
        return None
    inds = indices.reshape(rows, topk).astype(mx.uint32)
    params = mx.array([act_fn.alpha, act_fn.limit, act_fn.beta], dtype=mx.float32)
    # Each member row keeps RPS accumulators. On these shapes, calls of more
    # than four rows run faster with fewer output rows per simdgroup.
    rps = 4 if rows <= 4 else 2
    act = fp4_expert_rows.gate_up(
        x.reshape(rows, hidden),
        inds,
        gate.weight,
        gate.scales,
        up.weight,
        up.scales,
        n_out=inter,
        scale_group_size=16,
        activation=_SCALED_SWIGLU_OAI,
        extra=(switch.gate_scale, switch.up_scale, params),
        rps=rps,
    )
    y = fp4_expert_rows.down(
        act, inds, down.weight, down.scales, scale_group_size=16, rps=rps
    )
    return y.reshape(*indices.shape, hidden)
