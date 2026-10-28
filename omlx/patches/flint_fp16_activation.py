# SPDX-License-Identifier: Apache-2.0
"""Range-safe SwiGLU for Flint in float16.

In a float16 Flint pack (an oQ pack built with ``dtype=float16``, or a bf16
pack recast at load), a few routed experts' ``silu(gate) * up`` reach ~3e6
on ordinary prompts (layer 46 of a short coding prompt), past float16's
65504, while the down projection's output stays ~1e4. The infinities turn
into NaN logits and replies of ``!!!!`` (issue #4784). bf16 has the range
and is unaffected.

This patch forms the product in fp32, divides it by one scale per call
(1 unless the product leaves float16's range), runs the down projection on
the scaled product and multiplies its output back. The down projection is
linear, so the result matches the fp32 reference up to rounding.

The max over the product splits a compiled graph into several launches, so
decode-sized calls (up to ``_KERNEL_MAX`` elements) run one Metal kernel:
product, threadgroup max and scaled store in a single threadgroup. One-token
decode also undoes the scales inside the Flint decode combine
(``flint_decode.fused_combine``) instead of in separate multiplies. Larger
calls (prefill) use the compiled graph.

Covered: the routed experts (``SwitchGLU``'s activation, read by every
oMLX MoE route), the shared expert and the dense MLP layers. Only models
whose activations are float16 are touched; bf16 and fp32 keep mlx-vlm's
modules unchanged.
"""

from __future__ import annotations

import logging
from functools import cache, partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

logger = logging.getLogger(__name__)

# Peak |silu(gate) * up| kept after scaling: headroom under 65504 for the
# down projection's accumulation in float16.
_FP16_ACT_LIMIT = 16384.0


# One threadgroup loops over the whole tensor: decode and verify widths
# (a 16-row verify block of routed experts is 16 * 6 * 512 = 49152).
_KERNEL_MAX = 65536
_KERNEL_THREADS = 1024

_SCALED_SWIGLU_SOURCE = """
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    constexpr int NSG = TG / 32;
    threadgroup float red[NSG];

    // silu(g) * u in fp32 with MLX's Sigmoid formula; read twice instead of
    // staging the product.
    auto product = [&](int i) {
        float g = float(gate[elem_to_loc(i, gate_shape, gate_strides, gate_ndim)]);
        float u = float(up[elem_to_loc(i, up_shape, up_strides, up_ndim)]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(g)));
        return g * ((g < 0.0f) ? y : 1.0f - y) * u;
    };

    float m = 0.0f;
    for (int i = int(t); i < N; i += TG) {
        m = metal::max(m, metal::abs(product(i)));
    }
    m = simd_max(m);
    if (lane == 0) { red[sg] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    for (int k = 1; k < NSG; ++k) { m = metal::max(m, red[k]); }
    const float s = metal::max(m / float(LIMIT), 1.0f);
    if (t == 0) { scale[0] = static_cast<half>(s); }
    for (int i = int(t); i < N; i += TG) {
        out[i] = static_cast<half>(product(i) / s);
    }
"""


@cache
def _scaled_swiglu_kernel():
    return mx.fast.metal_kernel(
        name="omlx_flint_scaled_swiglu",
        input_names=["gate", "up"],
        output_names=["out", "scale"],
        source=_SCALED_SWIGLU_SOURCE,
        ensure_row_contiguous=False,
    )


def _scaled_swiglu_launch(gate, up):
    out, scale = _scaled_swiglu_kernel()(
        inputs=[gate, up],
        template=[
            ("N", up.size),
            ("TG", _KERNEL_THREADS),
            ("LIMIT", int(_FP16_ACT_LIMIT)),
        ],
        grid=(_KERNEL_THREADS, 1, 1),
        threadgroup=(_KERNEL_THREADS, 1, 1),
        output_shapes=[up.shape, (1,)],
        output_dtypes=[mx.float16, mx.float16],
    )
    return out, scale


@partial(mx.compile, shapeless=True)
def _scaled_swiglu_fp16(gate, up):
    a = nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)
    s = mx.maximum(mx.abs(a).max() / _FP16_ACT_LIMIT, 1.0)
    return (a / s).astype(mx.float16), s.astype(mx.float16)


def scaled_swiglu(gate: mx.array, up: mx.array) -> tuple[mx.array, Any]:
    """``(silu(gate) * up / s, s)``; ``s`` is None outside float16."""
    if up.dtype != mx.float16:
        return swiglu(gate, up), None
    if up.size <= _KERNEL_MAX and mx.metal.is_available():
        return _scaled_swiglu_launch(gate, up)
    return _scaled_swiglu_fp16(gate, up)


class ScaledSwiGLU:
    """``SwitchGLU`` activation that records the scale its caller undoes.

    A plain callable, not a module: the scale is per-call state, not a
    parameter.
    """

    def __init__(self):
        self.scale = None

    def __call__(self, x_up, x_gate):
        out, self.scale = scaled_swiglu(x_gate, x_up)
        return out


def routed_scale(moe: Any) -> Any:
    """The scale of ``moe.switch_mlp``'s last call, or None."""
    return getattr(moe.switch_mlp.activation, "scale", None)


def unscale_routed(moe: Any, routed: mx.array) -> mx.array:
    """Undo the activation scale on ``moe.switch_mlp``'s output."""
    scale = routed_scale(moe)
    return routed if scale is None else routed * scale


def mlp_parts(mlp: Any, x: mx.array) -> tuple[mx.array, Any]:
    """``(down_proj output, scale to undo or None)`` of a SwiGLU MLP."""
    parts = getattr(mlp, "scaled_parts", None)
    return parts(x) if parts is not None else (mlp(x), None)


@cache
def _scaled_mlp_class():
    from mlx_vlm.models.mlp import SwiGLUMLP

    class ScaledSwiGLUMLP(SwiGLUMLP):
        def scaled_parts(self, x):
            act, scale = scaled_swiglu(self.gate_proj(x), self.up_proj(x))
            return self.down_proj(act), scale

        def __call__(self, x):
            y, scale = self.scaled_parts(x)
            return y if scale is None else y * scale

    return ScaledSwiGLUMLP


def _moe_call(self, x):
    # mlx-vlm's SparseMoeBlock.__call__ with the routed scale undone.
    indices, weights = self.route(x)
    routed = unscale_routed(self, self.switch_mlp(x, indices))
    out = (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(x.dtype)
    if self.shared_expert is not None:
        out = out + self.shared_expert(x)
    return out


def _activation_dtype(layers) -> Any:
    weight = getattr(getattr(layers[0], "input_layernorm", None), "weight", None)
    return None if weight is None else weight.dtype


def apply_flint_fp16_activation_patch(model: Any) -> bool:
    """Install the range-safe SwiGLU on a loaded float16 Flint model."""
    try:
        from mlx_vlm.models.flint import language as flint
        from mlx_vlm.models.mlp import SwiGLUMLP
    except ImportError:
        return False
    layers = getattr(model, "layers", None)
    if not layers or _activation_dtype(layers) != mx.float16:
        return False

    cls = flint.SparseMoeBlock
    if not getattr(cls, "_omlx_flint_fp16_activation", False):
        cls._omlx_flint_original_call = cls.__call__
        cls.__call__ = _moe_call
        cls._omlx_flint_fp16_activation = True

    mlp_cls = _scaled_mlp_class()
    for layer in layers:
        mlp = layer.mlp
        if isinstance(mlp, cls):
            mlp.switch_mlp.activation = ScaledSwiGLU()
            mlp = mlp.shared_expert
        if type(mlp) is SwiGLUMLP:
            mlp.__class__ = mlp_cls
    logger.info("Flint float16 range-safe SwiGLU patch applied")
    return True
