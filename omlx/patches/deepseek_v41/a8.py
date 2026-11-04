# SPDX-License-Identifier: MIT
"""INT8-activation (A8) prefill on GPUs with tensor units.

Stage A quantizes each activation row to INT8 once and replaces the FP8
activation emulation of the A16 path. Routed experts use the sorted A8 gather,
whose gate/up epilogue applies the clamped SwiGLU and the route weight; dense
projections use the oQ A8 GEMM with a plan tagged on each eligible module at
load. Affine GS64 weights qualify (Q2/Q3/Q4 routed, Q4/Q5/Q6/Q8 dense), and so
do the source-precision MXFP4 routed experts and MXFP8 dense projections. An
MXFP8 weight is converted to INT8 codes on every A8 call, so no second copy
stays resident.
"""

from functools import partial

import mlx.core as mx
import mlx.nn as nn

from ..m5_gather_qmm_a8 import (
    _FMT_INT8,
    _FMT_MXFP4,
    _MIN_TOKENS_FLOOR,
    mx_shape_supported,
    mxfp8_to_int8,
    sorted_gather_qmm_a8,
    sorted_gather_qmm_a8_mx,
)
from ..qwen35_oq_a8 import _plan_for, apply_plan, stage_a

# (plan, min_tokens) on each tagged dense projection.
_PLAN = "_omlx_a8_plan"
# The plan of an MXFP8 projection, which runs through the INT8 gather.
_MXFP8 = "mxfp8"
# One expert holds every row, so taller tiles reuse each weight tile more: on
# M5 Max BM128 beat BM32 by up to 30% at 2K-8K rows.
_DENSE_BM = 128
# Decode and verify blocks stay on the A16 path; the setting can only raise it.
MIN_TOKENS_FLOOR = _MIN_TOKENS_FLOOR


def _affine_gs64(projection, bits, ndim):
    """Affine GS64 codes with matching fp16/bf16 scales and biases, as the
    A8 kernels read them."""
    weight = projection.get("weight")
    scales, biases = projection.get("scales"), projection.get("biases")
    bits_ok = getattr(projection, "bits", None) in bits
    return (
        getattr(projection, "mode", None) == "affine"
        and getattr(projection, "group_size", None) == 64
        and bits_ok
        and "bias" not in projection
        and weight is not None
        and weight.ndim == ndim
        and weight.dtype == mx.uint32
        and scales is not None
        and biases is not None
        and scales.dtype in (mx.float16, mx.bfloat16)
        and biases.dtype == scales.dtype
        and scales.shape == biases.shape
        and scales.shape[:-1] == weight.shape[:-1]
        and weight.shape[-1] * 32 == scales.shape[-1] * 64 * projection.bits
    )


def _mx(projection, mode, ndim):
    """Source-precision MX codes with E8M0 scales per 32 and no bias."""
    weight, scales = projection.get("weight"), projection.get("scales")
    return (
        getattr(projection, "mode", None) == mode
        and getattr(projection, "group_size", None) == 32
        and "bias" not in projection
        and projection.get("biases") is None
        and weight is not None
        and weight.ndim == ndim
        and weight.dtype == mx.uint32
        and scales is not None
        and scales.dtype == mx.uint8
        and scales.shape[:-1] == weight.shape[:-1]
        and weight.shape[-1] == scales.shape[-1] * projection.bits
    )


def _tokens(x, min_tokens, dtype):
    """Rows of ``x`` for the A8 path, or 0. The output takes the dtype of the
    scales, so ``x`` must already have it."""
    tokens = x.size // x.shape[-1]
    if tokens < min_tokens or x.dtype != dtype:
        return 0
    return tokens


def tag_dense_a8(language_model, min_tokens) -> int:
    """Attach an A8 plan to every eligible dense projection of the backbone
    layers. Returns how many were tagged; the draft stages are not visited."""
    tagged = 0
    for layer in language_model.layers:
        for _, module in layer.named_modules():
            if _mx(module, "mxfp8", 2):
                n, k = module.weight.shape[0], module.scales.shape[-1] * 32
                if mx_shape_supported(n, k):
                    object.__setattr__(module, _PLAN, (_MXFP8, min_tokens))
                    tagged += 1
                continue
            if not _affine_gs64(module, (4, 5, 6, 8), 2):
                continue
            plan = _plan_for(module.bits, module.weight.shape[0])
            if plan is not None:
                object.__setattr__(module, _PLAN, (plan, min_tokens))
                tagged += 1
    return tagged


def _plan(projection, x):
    tag = getattr(projection, _PLAN, None)
    if tag is None:
        return None
    # MX scales are E8M0 bytes; the output takes the activation dtype.
    dtype = x.dtype if tag[0] is _MXFP8 else projection.scales.dtype
    if x.dtype not in (mx.float16, mx.bfloat16) or not _tokens(x, tag[1], dtype):
        return None
    return tag[0]


def _mxfp8_gather(x, projections, **kwargs):
    """Dense MXFP8 projections of ``x`` as a one-expert INT8 gather; a second
    projection is the up half of a paired SwiGLU."""
    rows = x.size // x.shape[-1]
    (w, s), *up = [mxfp8_to_int8(p.weight[None], p.scales[None]) for p in projections]
    y = sorted_gather_qmm_a8_mx(
        x.reshape(rows, 1, x.shape[-1]),
        w,
        s,
        mx.zeros((rows,), mx.uint32),
        mx.arange(rows, dtype=mx.uint32),
        fmt=_FMT_INT8,
        swiglu=bool(up),
        up=up[0] if up else None,
        group_scales=True,
        bm=_DENSE_BM,
        **kwargs,
    )
    return None if y is None else y.reshape(*x.shape[:-1], y.shape[-1])


def dense_a8(projection, x):
    """``x @ W.T`` through the A8 GEMM, or None to keep the A16 path."""
    plan = _plan(projection, x)
    if plan is None:
        return None
    if plan is _MXFP8:
        return _mxfp8_gather(x, (projection,))
    return apply_plan(projection, stage_a(x, plan.act_mode, plan.layout), plan)


def _shared_stage(projections, x):
    """One Stage A for projections of the same input, or None."""
    plans = [_plan(p, x) for p in projections]
    if None in plans or _MXFP8 in plans:
        return None
    if len({(p.act_mode, p.layout) for p in plans}) != 1:
        return None
    return plans, stage_a(x, plans[0].act_mode, plans[0].layout)


def dense_a8_pair(first, second, x):
    """Two projections of one input, or None. Affine projections share Stage
    A; MXFP8 ones each run their own gather."""
    if _plan(first, x) is _MXFP8 and _plan(second, x) is _MXFP8:
        pair = _mxfp8_gather(x, (first,)), _mxfp8_gather(x, (second,))
        return None if any(y is None for y in pair) else pair
    shared = _shared_stage((first, second), x)
    if shared is None:
        return None
    (p1, p2), stage = shared
    return apply_plan(first, stage, p1), apply_plan(second, stage, p2)


@partial(mx.compile, shapeless=True)
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
    if all(_plan(p, x) is _MXFP8 for p in (expert.w1, expert.w3, expert.w2)):
        hidden = _mxfp8_gather(
            x, (expert.w1, expert.w3), swiglu_limit=expert._limit or None
        )
        return None if hidden is None else _mxfp8_gather(hidden, (expert.w2,))
    shared = _shared_stage((expert.w1, expert.w3), x)
    down = _plan(expert.w2, x)
    if shared is None or down is None:
        return None
    (p1, p3), stage = shared
    gate = apply_plan(expert.w1, stage, p1)
    up = apply_plan(expert.w3, stage, p3)
    hidden = _clamped_swiglu(gate, up, expert._limit or 0.0)
    if down is _MXFP8:
        return _mxfp8_gather(hidden, (expert.w2,))
    return apply_plan(expert.w2, stage_a(hidden, down.act_mode, down.layout), down)


def routed_experts_a8(experts, x, flat, order, weights, min_tokens):
    """Sorted routed outputs ``[M, 1, dim]``, or None to keep the A16 path.

    ``x`` holds the token rows, ``flat[order]`` the sorted expert ids and
    ``weights`` the route weights in token order.
    """
    # Offloaded experts stream their weights and keep the A16 path.
    projections = [getattr(experts, name, None) for name in ("w1", "w3", "w2")]
    if all(p is not None and _mx(p, "mxfp4", 3) for p in projections):
        mxfp4 = True
    elif all(p is not None and _affine_gs64(p, (2, 3, 4), 3) for p in projections):
        mxfp4 = False
    else:
        return None
    w1, w3, w2 = projections
    if x.dtype not in (mx.float16, mx.bfloat16):
        return None
    tokens = _tokens(x, min_tokens, x.dtype if mxfp4 else w1.scales.dtype)
    if not tokens or w1.bits != w3.bits:
        return None
    indices = flat[order].astype(mx.uint32)
    row_map = (order // (flat.size // tokens)).astype(mx.uint32)
    rows = x.reshape(tokens, 1, x.shape[-1])
    swiglu = {
        "swiglu": True,
        "swiglu_limit": experts._limit or None,
        "route_weights": weights.reshape(-1)[order],
    }
    sequential = mx.arange(indices.size, dtype=mx.uint32)
    if mxfp4:
        hidden = sorted_gather_qmm_a8_mx(
            rows,
            w1.weight,
            w1.scales,
            indices,
            row_map,
            fmt=_FMT_MXFP4,
            up=(w3.weight, w3.scales),
            **swiglu,
        )
        if hidden is None:
            return None
        return sorted_gather_qmm_a8_mx(
            hidden,
            w2.weight,
            w2.scales,
            indices,
            sequential,
            fmt=_FMT_MXFP4,
            # The route-weighted SwiGLU has outliers that a row-wide scale
            # would spread over the whole row.
            group_scales=True,
        )
    hidden = sorted_gather_qmm_a8(
        rows,
        w1.weight,
        w1.scales,
        w1.biases,
        indices,
        row_map,
        group_size=64,
        bits=w1.bits,
        up=(w3.weight, w3.scales, w3.biases),
        **swiglu,
    )
    if hidden is None:
        return None
    return sorted_gather_qmm_a8(
        hidden,
        w2.weight,
        w2.scales,
        w2.biases,
        indices,
        sequential,
        group_size=64,
        bits=w2.bits,
    )
