# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 fused short-window kernels against their reference expressions.

Every kernel in ``omlx.patches.inkling2.fast_decode`` is a drop-in for an
expression in ``model.py``. These tests pin that equivalence on random inputs,
including the rounding points the reference relies on (the conv result is
rounded to the weight dtype before the residual add, the loop-controller
down-projection is rounded to bf16 before the gate matmul, the NVFP4 expert
post-scale is applied in the activation dtype).
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:  # pragma: no cover - MLX is a hard runtime dependency
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture(scope="module", autouse=True)
def _gpu():
    if mx.default_device() != mx.gpu:
        pytest.skip("fused kernels require a Metal GPU")


@pytest.fixture(scope="module")
def model_module():
    from omlx.patches.inkling2 import model

    return model


@pytest.fixture(scope="module")
def fd():
    from omlx.patches.inkling2 import fast_decode

    if not fast_decode.enabled():
        pytest.skip("fused decode path disabled (OMLX_INKLING2_FAST_DECODE=0)")
    return fast_decode


def _assert_close(name, got, want, tol):
    diff = mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))).item()
    assert diff <= tol, f"{name}: max |diff| {diff:.6g} > {tol}"


def _assert_close_rel(name, got, want, rtol):
    """For reductions whose sum order intentionally differs (expert GEMVs)."""
    want_f = want.astype(mx.float32)
    diff = mx.max(mx.abs(got.astype(mx.float32) - want_f)).item()
    scale = max(float(mx.max(mx.abs(want_f)).item()), 1.0)
    assert diff <= rtol * scale, f"{name}: max |diff| {diff:.6g} > {rtol} * {scale:.3g}"


@pytest.mark.parametrize("length", [1, 2, 5, 8])
@pytest.mark.parametrize("channels", [1024, 2304])
def test_conv_short_matches_static_conv(model_module, fd, length, channels):
    mx.random.seed(0)
    kernel = 4
    x = mx.random.normal((1, length, channels)).astype(mx.bfloat16)
    w = mx.random.normal((channels, kernel)).astype(mx.bfloat16)
    state = mx.random.normal((1, kernel - 1, channels)).astype(mx.float32)
    scale = 1.0 / (kernel + 1) ** 0.5

    want_out, want_window = model_module._static_conv(x, w, state, scale)
    got_out, got_state, got_window = fd.conv_short(x, w, state, scale)

    _assert_close("conv out", got_out, want_out, 0.0)
    _assert_close("conv window", got_window, want_window, 0.0)
    _assert_close("conv state", got_state, want_window[:, -(kernel - 1) :], 0.0)


@pytest.mark.parametrize(
    "heads,dim,per_head_scale",
    [(24, 128, True), (8, 128, False), (24, 16, True)],
)
def test_l2_clip_matches_reference(model_module, fd, heads, dim, per_head_scale):
    mx.random.seed(1)
    x = mx.random.normal((1, heads, dim)).astype(mx.bfloat16)
    scale = (
        mx.random.uniform(0.5, 2.0, (heads, dim)).astype(mx.float32)
        if per_head_scale
        else mx.array(1.0)
    )
    threshold = 22.627416997969522
    want = model_module._l2_clip(
        x.reshape(1, 1, heads, dim),
        scale if per_head_scale else mx.array(1.0),
        mx.array(threshold),
    )
    got = fd.l2_clip_heads(x, scale, threshold).reshape(1, 1, heads, dim)
    _assert_close("l2 clip", got, want, 1e-3)


def test_unit_rms_heads_matches_reference(model_module, fd):
    mx.random.seed(2)
    x = mx.random.normal((1, 8, 128)).astype(mx.float32)
    want = model_module._unit_rms(x)
    got = fd.unit_rms_heads(x).astype(mx.float32)
    _assert_close("unit rms", got, want, 2e-3)


@pytest.mark.parametrize("num_inputs", [1, 3])
def test_loop_controller_matches_reference(model_module, fd, num_inputs):
    mx.random.seed(3)
    rows, hidden, rank = 1, 256, 8
    x = mx.random.normal((num_inputs, 1, rows, hidden)).astype(mx.bfloat16)
    message = mx.random.normal((num_inputs, 1, rows, hidden)).astype(mx.bfloat16)
    down = (mx.random.normal((num_inputs, 2, hidden, rank)) * 0.1).astype(mx.float32)
    gain_forget = (mx.random.normal((num_inputs, rank, 2 * hidden)) * 0.1).astype(
        mx.float32
    )

    want = model_module._mix_messages(x, message, down, gain_forget)

    z = fd.messages_prep(
        x.reshape(num_inputs, rows, hidden), message.reshape(num_inputs, rows, hidden)
    ).astype(mx.float32)
    low = fd.messages_low(
        z.reshape(2 * num_inputs, rows, hidden)
        @ down.reshape(2 * num_inputs, hidden, rank),
        rows,
    )
    gates = low.reshape(num_inputs, rows, rank) @ gain_forget
    got = fd.messages_apply(
        x.reshape(num_inputs, rows, hidden),
        message.reshape(num_inputs, rows, hidden),
        gates,
    ).reshape(num_inputs, 1, rows, hidden)
    _assert_close("loop controller", got, want, 1e-2)


def test_swiglu_scaled_matches_reference(model_module, fd):
    mx.random.seed(4)
    tokens, top_k, inter = 1, 8, 64
    y = mx.random.normal((tokens, top_k, 2 * inter)).astype(mx.bfloat16)
    s13 = mx.random.uniform(0.5, 1.5, (tokens, top_k)).astype(mx.float32)
    scaled = (y * s13[..., None]).astype(y.dtype)
    want = model_module._softcapped_swiglu(
        scaled[..., 0::2], scaled[..., 1::2], mx.array(4.0), mx.array(25.0)
    )
    got = fd.swiglu_scaled(y, s13, 4.0, 25.0)
    _assert_close("swiglu", got, want, 1e-2)


def test_moe_combine_matches_reference(fd):
    mx.random.seed(5)
    tokens, top_k, hidden = 1, 8, 256
    z = mx.random.normal((tokens, top_k, hidden)).astype(mx.bfloat16)
    s2 = mx.random.uniform(0.5, 1.5, (tokens, top_k)).astype(mx.float32)
    weights = mx.random.uniform(0.0, 1.0, (tokens, top_k)).astype(mx.float32)
    shared = mx.random.normal((tokens, hidden)).astype(mx.bfloat16)
    want = (
        mx.sum(z.astype(mx.float32) * (weights * s2)[..., None], axis=-2)
        + shared.astype(mx.float32)
    ).astype(mx.bfloat16)
    got = fd.moe_combine(z, s2, weights, shared)
    _assert_close("moe combine", got, want, 1e-2)


@pytest.mark.parametrize("keys", [7, 1024, 1031])
def test_decode_mask_matches_reference(fd, keys):
    mx.random.seed(6)
    heads, extent = 24, 1024
    rel = mx.random.normal((1, 1, heads, extent)).astype(mx.bfloat16)
    used = min(extent, keys)
    tail = rel[0, 0, :, :used][:, ::-1]
    if used < keys:
        tail = mx.concatenate(
            [mx.zeros((heads, keys - used), dtype=tail.dtype), tail], axis=-1
        )
    want = tail.reshape(1, heads, 1, keys).astype(mx.bfloat16)
    got = fd.decode_mask(rel, keys)
    _assert_close("decode mask", got, want, 0.0)


def test_nvfp4_experts_match_dequantized_reference(model_module, fd):
    """The fused NVFP4 expert GEMV pair reproduces the gather_qmm arithmetic."""
    mx.random.seed(7)
    from omlx.patches.inkling2.model import ExpertProj

    experts, inter, hidden = 8, 16, 32
    spec = {"mode": "nvfp4", "bits": 4, "group_size": 16}

    def _projection(out_dims, in_dims):
        proj = ExpertProj(experts, out_dims, in_dims, spec)
        w = mx.random.normal((experts, out_dims, in_dims)).astype(mx.bfloat16)
        weight, scales = mx.quantize(w, group_size=16, bits=4, mode="nvfp4")
        proj.weight = weight
        proj.scales = scales
        # post_scale then divides by 448*6, i.e. the tensor scale is 1.0.
        proj.global_scale = mx.full((experts,), 448.0 * 6.0, mx.float32)
        return proj

    class _Experts:
        w13 = _projection(2 * inter, hidden)
        w2 = _projection(hidden, inter)
        _gcap = mx.array(4.0)
        _ucap = mx.array(25.0)

    assert fd.nvfp4_decode_eligible(_Experts)

    x = mx.random.normal((1, hidden)).astype(mx.bfloat16)
    inds = mx.array([[1, 3, 5, 7, 0, 2, 4, 6]], dtype=mx.uint32)
    weights = mx.random.uniform(0.0, 1.0, (1, 8)).astype(mx.float32)

    got = fd.moe_decode_nvfp4(x, _Experts, inds, weights)

    def _dequant(proj):
        return mx.dequantize(
            proj.weight, proj.scales, None, group_size=16, bits=4, mode="nvfp4"
        )

    flat = inds.reshape(-1)
    scale13 = _Experts.w13.post_scale(flat).reshape(1, 8)
    y = mx.stack([x[0] @ _dequant(_Experts.w13)[int(e)].T for e in flat], 0)
    y = (y.reshape(1, 8, 2 * inter) * scale13[..., None]).astype(mx.bfloat16)
    act = model_module._softcapped_swiglu(
        y[..., 0::2], y[..., 1::2], mx.array(4.0), mx.array(25.0)
    )
    scale2 = _Experts.w2.post_scale(flat).reshape(1, 8)
    z = mx.stack(
        [act[0, k] @ _dequant(_Experts.w2)[int(flat[k])].T for k in range(8)], 0
    ).reshape(1, 8, hidden)
    want = mx.sum(z.astype(mx.float32) * (weights * scale2)[..., None], axis=-2)

    # A different FP32 reduction order; the reference path is 1-2% either way.
    _assert_close_rel("nvfp4 experts", got, want, 2e-2)


def test_saturating_logits_stay_finite(model_module, fd):
    """Fast-math tanh/exp turn NaN far outside the linear range; the kernels clamp."""
    y = mx.array([[[1000.0, 1000.0], [-1000.0, -1000.0]]], dtype=mx.bfloat16)
    s13 = mx.ones((1, 2), dtype=mx.float32)
    got = fd.swiglu_scaled(y, s13, 4.0, 25.0)
    assert mx.isfinite(got).all(), "swiglu produced a non-finite activation"

    scaled = (y * s13[..., None]).astype(y.dtype)
    want = model_module._softcapped_swiglu(
        scaled[..., 0::2], scaled[..., 1::2], mx.array(4.0), mx.array(25.0)
    )
    _assert_close("saturating swiglu", got, want, 0.05)
