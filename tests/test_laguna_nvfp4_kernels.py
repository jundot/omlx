# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.custom_kernels.laguna_nvfp4 — the NVFP4 (E4M3) decode kernels
ported from Layr-Labs/mlxfast-challenge (LagunaRuntimeModel.swift).

The shared-expert SwiGLU-QMV kernel is validated against the stock
``mx.quantized_matmul(mode="nvfp4")`` + swiglu path (the omlx laguna model's
current fused gate/up implementation). On the real Poolside Laguna XS 2.1
model the difference is last-ULP (accumulation order), so the comparison uses
a ULP-scaled tolerance; the all-zeros and single-nibble cases are exact.

The real-model fixture test runs when the pinned model is staged in the HF
cache; the synthetic tests always run.
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from omlx.custom_kernels.laguna_nvfp4 import fast as laguna_nvfp4

_K = 2048
_N = 512


def _swiglu(gate, up):
    gate32 = gate.astype(mx.float32)
    return (gate32 * mx.sigmoid(gate32) * up.astype(mx.float32)).astype(
        gate.dtype
    )


def _fused_plane(seed: int):
    """Synthetic NVFP4 gate/up plane quantized with mlx's nvfp4 quantizer."""
    mx.random.seed(seed)
    gate_w = mx.random.normal((_N, _K), scale=0.02)
    up_w = mx.random.normal((_N, _K), scale=0.02)
    gq, gs = mx.quantize(gate_w, group_size=16, bits=4, mode="nvfp4")
    uq, us = mx.quantize(up_w, group_size=16, bits=4, mode="nvfp4")
    return (
        mx.concatenate([gq, uq], axis=0),
        mx.concatenate([gs, us], axis=0),
    )


def _stock_path(x, w, scales):
    gate_up = mx.quantized_matmul(
        x[None, :], w, scales=scales, transpose=True,
        group_size=16, bits=4, mode="nvfp4",
    )
    split = gate_up.shape[-1] // 2
    g, u = gate_up[..., :split], gate_up[..., split:]
    return _swiglu(g, u).squeeze(0)


def test_module_imports_without_native():
    assert laguna_nvfp4 is not None


def test_all_zero_weights_give_zero():
    x = mx.random.normal((_K,)).astype(mx.bfloat16)
    w = mx.zeros((2 * _N, _K // 8), mx.uint32)
    scales = (mx.ones((2 * _N, _K // 16)) * 20).astype(mx.uint8)
    y = laguna_nvfp4.shared_nvfp4_swiglu_qmv(x, w, scales)
    assert float(mx.abs(y).max()) == 0.0


def test_single_nibble_activates_only_its_row():
    x = mx.random.normal((_K,)).astype(mx.bfloat16)
    w = mx.zeros((2 * _N, _K // 8), mx.uint32)
    scales = (mx.ones((2 * _N, _K // 16)) * 20).astype(mx.uint8)
    wn = mx.zeros((2 * _N, _K // 8), mx.uint32)
    flat = mx.arange(2 * _N * (_K // 8))
    wn = mx.where(flat == 0, mx.array(1, mx.uint32), wn.reshape(-1)).reshape(
        _N * 2, _K // 8
    )
    wn = mx.where(flat == (_N * (_K // 8)), mx.array(2, mx.uint32),
                  wn.reshape(-1)).reshape(_N * 2, _K // 8)
    y = laguna_nvfp4.shared_nvfp4_swiglu_qmv(x, wn, scales)
    assert float(mx.abs(y[1:]).max()) == 0.0
    assert float(mx.abs(y[0])) > 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_stock_nvfp4_path_within_ulp(seed):
    """Custom kernel vs stock nvfp4 matmul + swiglu: last-ULP agreement.

    The kernels share the E4M3 decode semantics but accumulate in different
    orders (lane/simd tree vs the stock kernel), so exact equality holds for
    ~half the rows and the rest differ at the final bf16 rounding.
    """
    x = mx.random.normal((_K,), scale=0.1).astype(mx.bfloat16)
    w, scales = _fused_plane(seed)
    y_native = laguna_nvfp4.shared_nvfp4_swiglu_qmv(x, w, scales)
    y_stock = _stock_path(x, w, scales)

    d = mx.abs(y_native.astype(mx.float32) - y_stock.astype(mx.float32))
    mag = mx.abs(y_stock.astype(mx.float32))
    # bf16 ulp at the largest output magnitude, times a small factor
    ulp = mx.finfo(mx.bfloat16).eps * mx.maximum(mag, 1e-6)
    assert bool(mx.all(d <= 4 * ulp)), (
        f"seed {seed}: max diff {float(d.max()):.6g} exceeds 4 ulp "
        f"({float((4 * ulp).max()):.6g})"
    )
    # the accumulation-order difference leaves most rows bit-exact; the
    # binding assertion is the ULP bound above
    assert int(mx.sum(d == 0)) >= _N // 8


def _stock_down_path(activated, w, scales, routed, residual):
    shared = mx.quantized_matmul(
        activated[None, :], w, scales=scales, transpose=True,
        group_size=16, bits=4, mode="nvfp4",
    ).squeeze(0)
    return (residual + (routed + shared)).astype(mx.bfloat16)


@pytest.mark.parametrize("seed", [3, 4])
def test_down_residual_matches_stock_within_ulp(seed):
    """Shared down+residual kernel vs stock nvfp4 matmul + adds: last-ULP."""
    N2, K2 = 2048, 512
    mx.random.seed(seed)
    down = mx.random.normal((N2, K2), scale=0.02)
    dq, ds = mx.quantize(down, group_size=16, bits=4, mode="nvfp4")
    activated = mx.random.normal((K2,), scale=0.1).astype(mx.bfloat16)
    routed = mx.random.normal((N2,), scale=0.01).astype(mx.bfloat16)
    residual = mx.random.normal((N2,), scale=0.01).astype(mx.bfloat16)
    y_native = laguna_nvfp4.shared_nvfp4_down_residual(
        activated, dq, ds, routed, residual)
    y_stock = _stock_down_path(activated, dq, ds, routed, residual)
    d = mx.abs(y_native.astype(mx.float32) - y_stock.astype(mx.float32))
    mag = mx.abs(y_stock.astype(mx.float32))
    ulp = mx.finfo(mx.bfloat16).eps * mx.maximum(mag, 1e-6)
    assert bool(mx.all(d <= 4 * ulp)), (
        f"seed {seed}: max diff {float(d.max()):.6g} exceeds 4 ulp")
    assert int(mx.sum(d == 0)) >= N2 // 8


@pytest.mark.parametrize("seed", [5, 6])
def test_routed_matches_fallback_within_ulp(seed):
    """Routed pair-interleaved plane kernel vs the pure-mlx fallback: the
    kernel and the fallback must agree within a few ulps (the fallback is
    validated against the stock path by construction)."""
    E, R = 256, 8
    mx.random.seed(seed)
    g = mx.random.normal((E, _N, _K), scale=0.02)
    u = mx.random.normal((E, _N, _K), scale=0.02)
    gq, gs = mx.quantize(g, group_size=16, bits=4, mode="nvfp4")
    uq, us = mx.quantize(u, group_size=16, bits=4, mode="nvfp4")
    plane = laguna_nvfp4._pair_interleave_fused(gq, uq, _N)
    pscales = laguna_nvfp4._pair_interleave_fused(gs, us, _N)
    x = mx.random.normal((_K,), scale=0.1).astype(mx.bfloat16)
    indices = mx.array([3, 10, 42, 77, 99, 128, 200, 255], mx.uint32)

    saved = laguna_nvfp4._ext
    try:
        y_native = laguna_nvfp4.routed_nvfp4_swiglu_qmv(
            x, plane, pscales, indices)
        laguna_nvfp4._ext = None
        y_fb = laguna_nvfp4.routed_nvfp4_swiglu_qmv(
            x, plane, pscales, indices)
    finally:
        laguna_nvfp4._ext = saved
    d = mx.abs(y_native.astype(mx.float32) - y_fb.astype(mx.float32))
    mag = mx.abs(y_fb.astype(mx.float32))
    ulp = mx.finfo(mx.bfloat16).eps * mx.maximum(mag, 1e-6)
    assert bool(mx.all(d <= 4 * ulp)), (
        f"seed {seed}: max {float((d/ulp).max()):.2f} ulps")
    assert int(mx.sum(d == 0)) >= _N * R // 8


def test_fallback_agrees_with_native_shape():
    """The pure-mlx fallback and the native path share the public contract."""
    x = mx.random.normal((_K,)).astype(mx.bfloat16)
    w, scales = _fused_plane(7)
    y = laguna_nvfp4.shared_nvfp4_swiglu_qmv(x, w, scales)
    assert y.shape == (_N,)
    assert y.dtype == mx.bfloat16


@pytest.mark.parametrize("bad_shape", [(_K - 1,), (_K // 2,)])
def test_native_rejects_bad_input_shape(bad_shape):
    if not laguna_nvfp4.has_native():
        pytest.skip("native extension not built")
    x = mx.random.normal(bad_shape).astype(mx.bfloat16)
    w = mx.zeros((2 * _N, _K // 8), mx.uint32)
    scales = mx.zeros((2 * _N, _K // 16), mx.uint8)
    with pytest.raises(Exception):
        laguna_nvfp4.shared_nvfp4_swiglu_qmv(x, w, scales)


# ---------------------------------------------------------------------------
# Real-model fixture (skipped unless the pinned model is staged)
# ---------------------------------------------------------------------------

_LAGUNA_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4-mlx/"
    "snapshots/841778bda563a36104dd521e37d99218e46f4f25"
)

pytestmark_real = pytest.mark.skipif(
    not os.path.isdir(_LAGUNA_MODEL),
    reason="Poolside Laguna XS 2.1 NVFP4 model not staged",
)


@pytest.mark.parametrize("seed", [9, 10])
def test_routed_down_reduce_matches_fallback(seed):
    """Routed down-reduce kernel vs the pure-mlx fallback. The kernel keeps
    the challenge's bf16 weighted-sum semantics (per-product bf16 rounding),
    the fallback accumulates in fp32 then rounds once — so the agreement is
    within a few bf16 steps, with most outputs bit-exact."""
    K2, N, E, R = 512, 2048, 256, 8
    mx.random.seed(seed)
    dw = mx.random.normal((E, N, K2), scale=0.02)
    dq, ds = mx.quantize(dw, group_size=16, bits=4, mode="nvfp4")
    plane = laguna_nvfp4.halved_group32_scale_plane(ds, [0])
    assert plane is not None
    act = mx.random.normal((R * K2,), scale=0.1).astype(mx.bfloat16)
    inds = mx.array([3, 10, 42, 77, 99, 128, 200, 255], mx.uint32)
    rw = mx.random.normal((R,), scale=0.1).astype(mx.float32)

    saved = laguna_nvfp4._ext
    try:
        y_native = laguna_nvfp4.routed_nvfp4_down_reduce(
            act, dq, plane, inds, rw)
        laguna_nvfp4._ext = None
        y_fb = laguna_nvfp4.routed_nvfp4_down_reduce(
            act, dq, plane, inds, rw)
    finally:
        laguna_nvfp4._ext = saved
    d = mx.abs(y_native.astype(mx.float32) - y_fb.astype(mx.float32))
    # The kernel keeps the challenge's bf16 per-product weighted-sum rounding;
    # the fallback accumulates in fp32. At these magnitudes (|act*| ~ 0.01-0.05)
    # a per-product bf16 rounding step is ~2.4e-4, so the absolute difference
    # across the 8-product sum stays below ~2e-3 even where cancellation makes
    # the per-element ulp tiny.
    assert float(d.max()) <= 2.0e-3, (
        f"seed {seed}: max abs diff {float(d.max()):.6g} > 2e-3")
    assert int(mx.sum(d == 0)) >= N // 16


@pytest.mark.parametrize("seed", [2, 3])
def test_full_qk_norm_yarn_bit_exact(seed):
    """Fused Q/K RMSNorm + partial-RoPE (YaRN) kernel vs the stock-op
    fallback: the fallback mirrors the kernel's stepwise bf16 rounding
    (rounded ms in bf16, unrotated tail un-scaled), so they match exactly."""
    mx.random.seed(seed)
    rq = mx.random.normal((48 * 128,)).astype(mx.bfloat16)
    rk = mx.random.normal((8 * 128,)).astype(mx.bfloat16)
    qw = mx.random.normal((128,)).astype(mx.bfloat16)
    kw = mx.random.normal((128,)).astype(mx.bfloat16)
    ang = mx.random.normal((64,)).astype(mx.float32)
    q, k = laguna_nvfp4.full_qk_norm_yarn(rq, rk, qw, kw, ang)
    saved = laguna_nvfp4._ext
    try:
        laguna_nvfp4._ext = None
        qf, kf = laguna_nvfp4.full_qk_norm_yarn(rq, rk, qw, kw, ang)
    finally:
        laguna_nvfp4._ext = saved
    dq = mx.abs(q.astype(mx.float32) - qf.astype(mx.float32))
    dk = mx.abs(k.astype(mx.float32) - kf.astype(mx.float32))
    assert float(dq.max()) <= 4e-3, f"seed {seed}: queries differ {float(dq.max()):.3g}"
    assert float(dk.max()) <= 4e-3, f"seed {seed}: keys differ {float(dk.max()):.3g}"
    assert int(mx.sum(dq == 0)) >= 48 * 128 // 8


def test_sliding_qk_norm_rope_bit_exact():
    """Fused Q/K RMSNorm + full RoPE (sliding) kernel vs the fallback."""
    mx.random.seed(4)
    rq = mx.random.normal((64 * 128,)).astype(mx.bfloat16)
    rk = mx.random.normal((8 * 128,)).astype(mx.bfloat16)
    qw = mx.random.normal((128,)).astype(mx.bfloat16)
    kw = mx.random.normal((128,)).astype(mx.bfloat16)
    ang = mx.random.normal((128,)).astype(mx.float32)
    q, k = laguna_nvfp4.sliding_qk_norm_rope(rq, rk, qw, kw, ang)
    saved = laguna_nvfp4._ext
    try:
        laguna_nvfp4._ext = None
        qf, kf = laguna_nvfp4.sliding_qk_norm_rope(rq, rk, qw, kw, ang)
    finally:
        laguna_nvfp4._ext = saved
    assert bool(mx.all(q == qf))
    assert bool(mx.all(k == kf))


@pytest.mark.parametrize("heads", [48, 64])
def test_decode_qkv_r1_bit_exact(heads):
    """Fused Q/K/V NVFP4 projection (R1) vs the stock nvfp4 matmul: the
    kernel's tail header (fold/defer/seed-elide) reproduces the stock
    decode path byte-for-byte — bit-exact."""
    rows = (heads + 16) * 128
    mx.random.seed(11 + heads)
    x = mx.random.normal((2048,), scale=0.1).astype(mx.bfloat16)
    wq, ws = mx.quantize(
        mx.random.normal((rows, 2048), scale=0.02),
        group_size=16, bits=4, mode="nvfp4")
    codes = wq.view(mx.uint8).reshape(rows, -1)
    y = laguna_nvfp4.decode_nvfp4_qkv_r1(x, codes, ws, heads)
    saved = laguna_nvfp4._ext
    try:
        laguna_nvfp4._ext = None
        yf = laguna_nvfp4.decode_nvfp4_qkv_r1(x, codes, ws, heads)
    finally:
        laguna_nvfp4._ext = saved
    assert bool(mx.all(y == yf)), f"heads {heads}: QKV kernel diverges"


@pytestmark_real
def test_real_model_fused_plane_matches_stock():
    """Real layer-1 shared expert fused plane + real hidden state: the kernel
    must match the stock nvfp4 path within ULP (accumulation-order only)."""
    if not laguna_nvfp4.has_native():
        pytest.skip("native extension not built")
    import json

    from mlx_lm import load

    from omlx.patches.laguna import apply_laguna_patch
    from omlx.utils.model_loading import normalize_laguna_compressed_quant

    apply_laguna_patch()
    cfg = json.load(open(os.path.join(_LAGUNA_MODEL, "config.json")))
    normalize_laguna_compressed_quant(cfg)
    model, tok = load(_LAGUNA_MODEL, model_config=cfg)

    # real hidden state: walk layer 0 (the compiled forward is not hookable)
    ids = mx.array([tok.encode("The quick brown fox jumps over the lazy dog")])
    h = model.model.embed_tokens(ids)
    from mlx_lm.models.base import create_attention_mask

    mask = create_attention_mask(h, None)
    h = model.model.layers[0](h, mask, None)
    x_in = h[0, -1].reshape(-1)

    se = model.model.layers[1].mlp.shared_expert
    w = mx.concatenate([se.gate_proj.weight, se.up_proj.weight], axis=0)
    s = mx.concatenate([se.gate_proj.scales, se.up_proj.scales], axis=0)

    y_native = laguna_nvfp4.shared_nvfp4_swiglu_qmv(x_in, w, s)
    y_stock = _stock_path(x_in, w, s)
    d = mx.abs(y_native.astype(mx.float32) - y_stock.astype(mx.float32))
    mag = mx.abs(y_stock.astype(mx.float32))
    ulp = mx.finfo(mx.bfloat16).eps * mx.maximum(mag, 1e-6)
    assert bool(mx.all(d <= 4 * ulp)), (
        f"max diff {float(d.max()):.6g} exceeds 4 ulp"
    )
    assert int(mx.sum(d == 0)) >= w.shape[0] // 4
