# SPDX-License-Identifier: Apache-2.0
"""Decode / short-verify routed-expert kernels for MXFP4 experts (MiMo V2).

``gate_up_swiglu`` and ``down_proj`` run a ``SwitchGLU`` of a few token rows
on the shared FP4 expert kernels (``omlx.patches.fp4_expert_rows``), bit-exact
to MLX's per-pair ``gather_qmv``. The SwiGLU epilogue is MLX's compiled
``silu(gate) * up`` in the activation dtype, op for op.
"""

from __future__ import annotations

import mlx.core as mx

from omlx.patches import fp4_expert_rows

BLOCK = fp4_expert_rows.BLOCK

_SWIGLU = fp4_expert_rows.Activation(
    name="swiglu",
    source=r"""
    const T gt = static_cast<T>(g);
    const T ut = static_cast<T>(u);
    const T sig = fp4_sigmoid(gt);
    const T silu = gt * sig;
    act[out] = silu * ut;
""",
)


def supported(linear, k: int, n: int) -> bool:
    """A ``QuantizedSwitchLinear`` these kernels handle (MXFP4, gs 32)."""
    return (
        getattr(linear, "mode", None) == "mxfp4"
        and getattr(linear, "bits", None) == 4
        and getattr(linear, "group_size", None) == 32
        and linear.get("biases") is None
        and linear.get("bias") is None
        and k % BLOCK == 0
        and linear.weight.dtype == mx.uint32
        and linear.scales.dtype == mx.uint8
    )


def gate_up_swiglu(x, inds, wg, sg, wu, su, *, n_out, up_offset=0):
    """``swiglu(gate(x), up(x))`` per (row, expert) pair, shape ``(*inds.shape, n_out)``.

    ``x (..., K)`` rows matching ``inds (..., TOPK)``; ``wg``/``wu`` packed
    MXFP4 ``(E, rows, K / 8)`` with up rows starting at ``up_offset`` (a
    fused ``[gate; up]`` tensor passes itself twice with ``up_offset=n_out``).
    """
    act = fp4_expert_rows.gate_up(
        x,
        inds,
        wg,
        sg,
        wu,
        su,
        n_out=n_out,
        scale_group_size=32,
        activation=_SWIGLU,
        up_offset=up_offset,
    )
    return act.reshape(*inds.shape, n_out)


def down_proj(a, inds, w, s):
    """Per-pair down projection: ``a (..., TOPK, K)`` -> ``(..., TOPK, N)``."""
    y = fp4_expert_rows.down(a, inds, w, s, scale_group_size=32)
    return y.reshape(*inds.shape, -1)
