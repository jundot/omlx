# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused one-token Flint MoE decode path."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
flint = pytest.importorskip("mlx_vlm.models.flint.language")

from mlx_vlm.models.flint.config import ModelConfig  # noqa: E402

from omlx.patches import flint_decode  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Flint decode kernels need Metal"
)


def _stock_route(logits, bias, top_k):
    scores = mx.sigmoid(logits.astype(mx.float32))
    choice = scores + bias.astype(mx.float32)
    indices = mx.argpartition(choice, kth=-top_k, axis=-1)[..., -top_k:]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    return indices, weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)


@pytest.mark.parametrize("num_experts,top_k", [(256, 6), (64, 4), (384, 8)])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("biased", [False, True])
def test_router_matches_stock_route(num_experts, top_k, ties, biased):
    mx.random.seed(num_experts + top_k)
    for _ in range(20):
        logits = mx.random.normal((num_experts,)).astype(mx.bfloat16)
        if ties:
            # bf16 logits collide often; force collisions around the k-th place.
            logits = mx.round(logits.astype(mx.float32) * 2) / 2
            logits = logits.astype(mx.bfloat16)
        bias = (
            mx.random.uniform(-0.05, 0.05, (num_experts,))
            if biased
            else mx.zeros((num_experts,))
        )
        ref_idx, ref_w = _stock_route(logits, bias, top_k)
        idx, w = flint_decode.fused_router(logits, bias, top_k)
        assert sorted(idx.tolist()) == sorted(ref_idx.tolist())
        by_expert = dict(zip(ref_idx.tolist(), ref_w.tolist()))
        for e, value in zip(idx.tolist(), w.tolist()):
            assert value == pytest.approx(by_expert[e], abs=1e-6)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_combine_matches_stock_roundings(dtype):
    mx.random.seed(3)
    top_k, dim = 6, 2048
    routed = mx.random.normal((1, 1, top_k, dim)).astype(dtype)
    weights = mx.softmax(mx.random.normal((top_k,)))
    shared = mx.random.normal((1, 1, dim)).astype(dtype)
    residual = mx.random.normal((1, 1, dim)).astype(dtype)
    moe = (routed.astype(mx.float32) * weights[:, None]).sum(axis=-2).astype(dtype)
    ref = residual + (moe + shared)
    out = flint_decode.fused_combine(routed, weights, shared, residual)
    assert out.dtype == dtype
    assert mx.array_equal(out, ref).item()


def _tiny_layer():
    config = ModelConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=128,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=16,
        num_experts=32,
        num_experts_per_tok=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=128,
    )
    mx.random.seed(7)
    layer = flint.DecoderLayer(config, 1)
    layer.mlp.expert_bias = mx.random.uniform(-0.1, 0.1, (config.num_experts,))
    layer.set_dtype(mx.bfloat16)
    mx.eval(layer.parameters())
    return layer


def test_patched_layer_matches_stock_decode():
    layer = _tiny_layer()
    x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
    assert flint_decode.apply_flint_decode_patch()
    ref = flint.DecoderLayer._omlx_flint_original_call(layer, x)
    out = layer(x)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    # Only the k-sum order differs from the composed path.
    assert mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item() < 0.05


@pytest.mark.parametrize("rows", [2, 5, 16])
def test_router_rows_match_one_row_router(rows):
    mx.random.seed(rows)
    logits = mx.random.normal((1, rows, 256)).astype(mx.bfloat16)
    bias = mx.random.uniform(-0.05, 0.05, (256,))
    idx, w = flint_decode.fused_router(logits, bias, 6)
    assert idx.shape == (1, rows, 6) and w.shape == (1, rows, 6)
    for r in range(rows):
        one_idx, one_w = flint_decode.fused_router(logits[0, r], bias, 6)
        assert idx[0, r].tolist() == one_idx.tolist()
        assert mx.array_equal(w[0, r], one_w).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_combine_rows_matches_stock_roundings(dtype):
    mx.random.seed(5)
    rows, top_k, dim = 5, 6, 2048
    routed = mx.random.normal((1, rows, top_k, dim)).astype(dtype)
    weights = mx.softmax(mx.random.normal((1, rows, top_k)), axis=-1)
    shared = mx.random.normal((1, rows, dim)).astype(dtype)
    residual = mx.random.normal((1, rows, dim)).astype(dtype)
    moe = (routed.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(dtype)
    ref = residual + (moe + shared)
    out = flint_decode.fused_combine(routed, weights, shared, residual)
    assert out.shape == ref.shape and out.dtype == dtype
    assert mx.array_equal(out, ref).item()


def test_patched_verify_rows_match_one_token_steps():
    """A verify block routes and combines each row like its one-token step."""
    layer = _tiny_layer()
    assert flint_decode.apply_flint_decode_patch()
    x = mx.random.normal((1, 4, 256)).astype(mx.bfloat16)
    block = layer(x, mask="causal")
    ref = flint.DecoderLayer._omlx_flint_original_call(layer, x, mask="causal")
    assert block.shape == ref.shape and block.dtype == ref.dtype
    assert mx.abs(block.astype(mx.float32) - ref.astype(mx.float32)).max().item() < 0.05


def test_patched_layer_keeps_prefill_path(monkeypatch):
    layer = _tiny_layer()
    assert flint_decode.apply_flint_decode_patch()

    def fail(*args, **kwargs):
        raise AssertionError("prefill calls must keep the stock body")

    monkeypatch.setattr(flint_decode, "fused_router", fail)
    rows = flint_decode._MAX_ROWS + 1
    out = layer(mx.random.normal((1, rows, 256)).astype(mx.bfloat16), mask="causal")
    assert out.shape == (1, rows, 256)
    out = layer(
        mx.random.normal((2, rows // 2 + 1, 256)).astype(mx.bfloat16), mask="causal"
    )
    assert out.shape == (2, rows // 2 + 1, 256)


@pytest.mark.parametrize("shape", [(2, 1), (3, 1), (3, 5)])
def test_patched_batch_rows_match_stock(shape):
    """Batched decode / verify rows take the fused path and match the stock body."""
    layer = _tiny_layer()
    assert flint_decode.apply_flint_decode_patch()
    x = mx.random.normal((*shape, 256)).astype(mx.bfloat16)
    mask = "causal" if shape[1] > 1 else None
    calls = []
    router = flint_decode.fused_router

    def counted(*args, **kwargs):
        calls.append(args[0].shape)
        return router(*args, **kwargs)

    flint_decode.fused_router = counted
    try:
        out = layer(x, mask=mask)
    finally:
        flint_decode.fused_router = router
    assert calls, "batched rows should use the fused router"
    ref = flint.DecoderLayer._omlx_flint_original_call(layer, x, mask=mask)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item() < 0.05
    # Each sequence's rows route as they would alone.
    for b in range(shape[0]):
        alone = layer(x[b : b + 1], mask=mask)
        assert (
            mx.abs(alone.astype(mx.float32) - out[b : b + 1].astype(mx.float32))
            .max()
            .item()
            < 0.05
        )
