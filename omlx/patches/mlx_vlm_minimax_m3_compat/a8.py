# SPDX-License-Identifier: Apache-2.0
"""INT8-activation (A8) prefill for MiniMax M3 / M3.1 on GPUs with tensor units.

Dense projections take the DeepSeek V4.1 A8 plans: affine GS64 Q4/Q5/Q6/Q8 run
the oQ A8 GEMM and MXFP8 runs as a one-expert INT8 gather. Every activation is
scaled per GS64 group: the dense MLP inputs carry channels about 40x their row
RMS, which a row-wide INT8 scale would round away (9% round-trip error, 1.4%
per group). The sparse-attention indexer projections keep the A16 path: they
select the attended blocks and carry about 1% of the prefill FLOPs.

Routed experts with affine GS64 Q2/Q3/Q4 gate/up run the sorted A8 gather with
the SwiGLU-OAI epilogue, and so does an affine down projection. NVFP4 experts
keep the A16 path: their per-16 E4M3 scales need twice the float corrections
of MXFP4, which made the A8 gather slower than MLX's NVFP4 gather on M5 Max.
An NVFP4 down after affine gate/up runs the A16 gather on the sorted rows.
"""

from __future__ import annotations

import mlx.core as mx

from omlx.custom_kernels.quant_matmul import fast as quant_fast

from ..deepseek_v41.a8 import (
    MIN_TOKENS_FLOOR,
    _affine_gs64,
    dense_a8,
    tag_dense_a8,
)
from ..m5_gather_qmm_a8 import sorted_gather_qmm_a8

# The min_tokens tag on each routed expert bank.
_ROUTED = "_omlx_minimax_a8_min_tokens"
_INDEXER = ("index_q_proj", "index_k_proj")

dense = dense_a8


def _backbone(model):
    """The module holding the decoder ``layers`` of a MiniMax model."""
    lm = getattr(model, "language_model", model)
    return getattr(lm, "model", lm)


def tag(model, min_tokens: int) -> int:
    """Tag the decoder layers of ``model`` for A8 prefill; returns how many
    dense projections and routed expert banks were tagged. Nothing is tagged
    without the A8 kernels."""
    if not quant_fast.oq_a8_available():
        return 0
    min_tokens = max(int(min_tokens), MIN_TOKENS_FLOOR)
    backbone = _backbone(model)
    tagged = tag_dense_a8(backbone, min_tokens, skip=_INDEXER, group_scales=True)
    for layer in backbone.layers:
        switch = getattr(getattr(layer, "block_sparse_moe", None), "switch_mlp", None)
        # Only the scaled bank (every oQ checkpoint) routes through ``routed``.
        if switch is not None and "down_scale" in switch and _affine(switch.gate_proj):
            object.__setattr__(switch, _ROUTED, min_tokens)
            tagged += 1
    return tagged


def _affine(projection, dtype=None) -> bool:
    """Affine GS64 Q2/Q3/Q4 expert codes, with metadata of ``dtype`` if given."""
    return _affine_gs64(projection, (2, 3, 4), 3) and (
        dtype is None or projection.scales.dtype == dtype
    )


def routed(switch, x, indices):
    """Unweighted routed outputs ``[..., k, dim]`` of a scaled MiniMax expert
    bank, or None to keep the A16 path."""
    min_tokens = getattr(switch, _ROUTED, None)
    if min_tokens is None or x.dtype not in (mx.float16, mx.bfloat16):
        return None
    dim = x.shape[-1]
    tokens = x.size // dim
    if tokens < min_tokens:
        return None
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    if not (_affine(gate, x.dtype) and _affine(up, x.dtype) and gate.bits == up.bits):
        return None

    act = switch.activation
    # The A16 activation always clamps; the kernel reads limit 0 as no clamp.
    if not act.limit:
        return None
    flat = indices.reshape(-1)
    order = mx.argsort(flat)
    sorted_ids = flat[order].astype(mx.uint32)
    hidden = sorted_gather_qmm_a8(
        x.reshape(tokens, 1, dim),
        gate.weight,
        gate.scales,
        gate.biases,
        sorted_ids,
        (order // indices.shape[-1]).astype(mx.uint32),
        group_size=64,
        bits=gate.bits,
        up=(up.weight, up.scales, up.biases),
        swiglu=True,
        swiglu_limit=act.limit,
        swiglu_alpha=act.alpha,
        swiglu_beta=act.beta,
        expert_scales=(switch.gate_scale, switch.up_scale),
    )
    if hidden is None:
        return None
    y = None
    if _affine(down, x.dtype):
        y = sorted_gather_qmm_a8(
            hidden,
            down.weight,
            down.scales,
            down.biases,
            sorted_ids,
            mx.arange(sorted_ids.size, dtype=mx.uint32),
            group_size=64,
            bits=down.bits,
            # A row-wide scale would spread the SwiGLU-OAI outliers.
            group_scales=True,
        )
    if y is None:
        y = down(hidden, sorted_ids, sorted_indices=True)
    return y[mx.argsort(order)].reshape(*indices.shape, -1)
