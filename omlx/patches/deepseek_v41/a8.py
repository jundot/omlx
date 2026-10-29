# SPDX-License-Identifier: MIT
"""INT8-activation (A8) prefill on GPUs with tensor units.

Stage A quantizes each activation row to INT8 once and replaces the FP8
activation emulation of the A16 path. Routed experts use the sorted A8 gather,
whose gate/up epilogue applies the clamped SwiGLU and the route weight; dense
projections use the oQ A8 GEMM with a plan tagged on each eligible module at
load. Only affine GS64 weights qualify (Q2/Q3/Q4 routed, Q4/Q5/Q6/Q8 dense).
"""

import mlx.core as mx
import mlx.nn as nn

from ..m5_gather_qmm_a8 import sorted_gather_qmm_a8
from ..qwen35_oq_a8 import _plan_for, apply_plan, stage_a

# (plan, min_tokens) on each tagged dense projection.
_PLAN = "_omlx_a8_plan"


def _affine_gs64(projection, bits, ndim):
    weight = projection.get("weight")
    return (
        getattr(projection, "mode", None) == "affine"
        and getattr(projection, "group_size", None) == 64
        and getattr(projection, "bits", None) in bits
        and projection.get("biases") is not None
        and weight is not None
        and weight.ndim == ndim
    )


def _tokens(x, min_tokens):
    tokens = x.size // x.shape[-1]
    if tokens < min_tokens or x.dtype not in (mx.bfloat16, mx.float16):
        return 0
    return tokens


def tag_dense_a8(language_model, min_tokens) -> int:
    """Attach an A8 plan to every eligible dense projection of the backbone
    layers. Returns how many were tagged; the draft stages are not visited."""
    tagged = 0
    for layer in language_model.layers:
        for _, module in layer.named_modules():
            if not _affine_gs64(module, (4, 5, 6, 8), 2):
                continue
            plan = _plan_for(module.bits, module.weight.shape[0])
            if plan is not None:
                object.__setattr__(module, _PLAN, (plan, min_tokens))
                tagged += 1
    return tagged


def _plan(projection, x):
    tag = getattr(projection, _PLAN, None)
    if tag is None or not _tokens(x, tag[1]):
        return None
    return tag[0]


def dense_a8(projection, x):
    """``x @ W.T`` through the A8 GEMM, or None to keep the A16 path."""
    plan = _plan(projection, x)
    if plan is None:
        return None
    return apply_plan(projection, stage_a(x, plan.act_mode, plan.layout), plan)


def _shared_stage(projections, x):
    """One Stage A for projections of the same input, or None."""
    plans = [_plan(p, x) for p in projections]
    if None in plans:
        return None
    if len({(p.act_mode, p.layout) for p in plans}) != 1:
        return None
    return plans, stage_a(x, plans[0].act_mode, plans[0].layout)


def dense_a8_pair(first, second, x):
    """Two projections of one input sharing Stage A, or None."""
    shared = _shared_stage((first, second), x)
    if shared is None:
        return None
    (p1, p2), stage = shared
    return apply_plan(first, stage, p1), apply_plan(second, stage, p2)


@mx.compile
def _clamped_swiglu(gate, up, limit):
    """silu(min(gate, limit)) * clip(up, -limit, limit) in float, as one kernel."""
    g, u = gate.astype(mx.float32), up.astype(mx.float32)
    if limit:
        g = mx.minimum(g, limit)
        u = mx.clip(u, -limit, limit)
    return (nn.silu(g) * u).astype(gate.dtype)


def shared_expert_a8(expert, x):
    """The shared expert: gate/up share Stage A, then the clamped SwiGLU in
    float and the down projection. None keeps the A16 path."""
    shared = _shared_stage((expert.w1, expert.w3), x)
    down = _plan(expert.w2, x)
    if shared is None or down is None:
        return None
    (p1, p3), stage = shared
    gate = apply_plan(expert.w1, stage, p1)
    up = apply_plan(expert.w3, stage, p3)
    hidden = _clamped_swiglu(gate, up, expert._limit or 0.0)
    return apply_plan(expert.w2, stage_a(hidden, down.act_mode, down.layout), down)


def routed_experts_a8(experts, x, flat, order, weights, min_tokens):
    """Sorted routed outputs ``[M, 1, dim]``, or None to keep the A16 path.

    ``x`` holds the token rows, ``flat[order]`` the sorted expert ids and
    ``weights`` the route weights in token order.
    """
    tokens = _tokens(x, min_tokens)
    w1, w3, w2 = experts.w1, experts.w3, experts.w2
    if not tokens or not all(_affine_gs64(p, (2, 3, 4), 3) for p in (w1, w3, w2)):
        return None
    if w1.bits != w3.bits:
        return None
    indices = flat[order].astype(mx.uint32)
    row_map = (order // (flat.size // tokens)).astype(mx.uint32)
    hidden = sorted_gather_qmm_a8(
        x.reshape(tokens, 1, x.shape[-1]),
        w1.weight,
        w1.scales,
        w1.biases,
        indices,
        row_map,
        group_size=64,
        bits=w1.bits,
        swiglu=True,
        up=(w3.weight, w3.scales, w3.biases),
        swiglu_limit=experts._limit or None,
        route_weights=weights.reshape(-1)[order],
    )
    if hidden is None:
        return None
    return sorted_gather_qmm_a8(
        hidden,
        w2.weight,
        w2.scales,
        w2.biases,
        indices,
        mx.arange(indices.size, dtype=mx.uint32),
        group_size=64,
        bits=w2.bits,
    )
