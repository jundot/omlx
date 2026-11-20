# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused Gemma 4 MoE router and combine."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
language = pytest.importorskip("mlx_vlm.models.gemma4.language")

from mlx_vlm.models.gemma4.config import TextConfig  # noqa: E402

from omlx.patches import gemma4_moe_decode  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="Gemma 4 MoE decode kernels need Metal"
)

_DTYPES = [mx.bfloat16, mx.float16]


def _stock_route(scores, scale, top_k):
    idx = mx.argpartition(scores, kth=-top_k, axis=-1)[..., -top_k:]
    weights = mx.softmax(mx.take_along_axis(scores, idx, axis=-1), axis=-1)
    return idx, weights * scale[idx]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("num_experts,top_k", [(128, 8), (256, 4)])
@pytest.mark.parametrize("ties", [False, True])
def test_router_is_bit_identical_to_stock(dtype, num_experts, top_k, ties):
    mx.random.seed(num_experts + top_k)
    for rows in (1, 3, 16):
        scores = mx.random.normal((1, rows, num_experts)) * 2
        if ties:
            # bf16 logits collide often; argpartition keeps the highest index.
            scores = mx.round(scores * 2) / 2
        scores = scores.astype(dtype)
        scale = mx.random.uniform(0.3, 2.0, (num_experts,)).astype(dtype)
        ref_idx, ref_w = _stock_route(scores, scale, top_k)
        idx, w = gemma4_moe_decode.fused_router(scores, scale, top_k)
        assert w.dtype == dtype
        assert mx.array_equal(idx, ref_idx).item()
        assert mx.array_equal(w, ref_w).item()


@pytest.mark.parametrize("dtype", _DTYPES)
def test_combine_is_bit_identical_to_stock(dtype):
    mx.random.seed(3)
    routed = (mx.random.normal((2, 3, 8, 704)) * 0.7).astype(dtype)
    weights = mx.softmax(mx.random.normal((2, 3, 8)), axis=-1).astype(dtype)
    ref = (mx.expand_dims(weights, -1) * routed).sum(-2)
    out = gemma4_moe_decode.fused_combine(routed, weights)
    assert out.dtype == dtype
    assert mx.array_equal(out, ref).item()


def _moe_modules(dtype, top_k=8):
    config = TextConfig.from_dict(
        {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "num_experts": 128,
            "top_k_experts": top_k,
            "vocab_size": 128,
            "enable_moe_block": True,
            "layer_types": ["sliding_attention", "full_attention"],
        }
    )
    mx.random.seed(7)
    router, experts = language.Router(config), language.Experts(config)
    router.per_expert_scale = mx.random.uniform(0.3, 2.0, (128,))
    for module in (router, experts):
        module.set_dtype(dtype)
    mx.eval(router.parameters(), experts.parameters())
    return router, experts


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(1, 1), (3, 5)])
def test_patched_moe_matches_stock(dtype, shape):
    router, experts = _moe_modules(dtype)
    x = mx.random.normal((*shape, 256)).astype(dtype)
    assert gemma4_moe_decode.apply_gemma4_moe_decode_patch()
    idx, w = router(x)
    out = experts(x, idx, w)
    ref_idx, ref_w = language.Router._omlx_gemma4_original_call(router, x)
    ref_out = language.Experts._omlx_gemma4_original_call(experts, x, ref_idx, ref_w)
    for a, b in ((idx, ref_idx), (w, ref_w), (out, ref_out)):
        assert mx.array_equal(a, b).item()


def test_prefill_sized_blocks_keep_the_stock_path(monkeypatch):
    router, experts = _moe_modules(mx.bfloat16)
    assert gemma4_moe_decode.apply_gemma4_moe_decode_patch()

    def boom(*args, **kwargs):
        raise AssertionError("fused kernel used for a prefill-sized block")

    monkeypatch.setattr(gemma4_moe_decode, "fused_router", boom)
    monkeypatch.setattr(gemma4_moe_decode, "fused_combine", boom)
    x = mx.random.normal((1, gemma4_moe_decode._MAX_ROWS + 1, 256)).astype(mx.bfloat16)
    idx, w = router(x)
    assert experts(x, idx, w).shape == x.shape


def test_scale_in_another_dtype_keeps_the_stock_route():
    router, _ = _moe_modules(mx.bfloat16)
    router.per_expert_scale = router.per_expert_scale.astype(mx.float32)
    assert gemma4_moe_decode.apply_gemma4_moe_decode_patch()
    x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    idx, w = router(x)
    ref_idx, ref_w = language.Router._omlx_gemma4_original_call(router, x)
    assert w.dtype == ref_w.dtype == mx.float32
    assert mx.array_equal(idx, ref_idx).item() and mx.array_equal(w, ref_w).item()


def test_ties_across_the_thread_stride_keep_the_highest_index():
    scores = mx.full((1, 1, 256), -1.0).astype(mx.bfloat16)
    scores[..., 5] = scores[..., 133] = 2.0
    scores[..., 17] = scores[..., 145] = 1.0
    scale = mx.ones((256,), dtype=mx.bfloat16)
    ref_idx, ref_w = _stock_route(scores, scale, 3)
    idx, w = gemma4_moe_decode.fused_router(scores, scale, 3)
    assert 145 in idx.tolist()[0][0] and 17 not in idx.tolist()[0][0]
    assert mx.array_equal(idx, ref_idx).item() and mx.array_equal(w, ref_w).item()


def test_applying_the_patch_twice_keeps_the_true_original():
    assert gemma4_moe_decode.apply_gemma4_moe_decode_patch()
    original = language.Router._omlx_gemma4_original_call
    assert gemma4_moe_decode.apply_gemma4_moe_decode_patch()
    assert language.Router._omlx_gemma4_original_call is original


def test_norm_weight_follows_a_replaced_scale():
    router, _ = _moe_modules(mx.bfloat16)
    assert gemma4_moe_decode.apply_gemma4_moe_decode_patch()
    x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    router(x)
    router.scale = router.scale * 3
    idx, w = router(x)
    ref_idx, ref_w = language.Router._omlx_gemma4_original_call(router, x)
    assert mx.array_equal(idx, ref_idx).item() and mx.array_equal(w, ref_w).item()


def test_top_k_above_the_two_lane_limit_keeps_the_stock_route():
    router, _ = _moe_modules(mx.bfloat16, top_k=12)
    assert gemma4_moe_decode.apply_gemma4_moe_decode_patch()
    x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    idx, w = router(x)
    ref_idx, ref_w = language.Router._omlx_gemma4_original_call(router, x)
    assert mx.array_equal(idx, ref_idx).item() and mx.array_equal(w, ref_w).item()
