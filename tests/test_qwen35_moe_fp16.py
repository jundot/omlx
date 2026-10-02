# SPDX-License-Identifier: Apache-2.0
"""FP16 router kernels preserve the composed inference arithmetic."""

import mlx.core as mx
import pytest

from omlx.patches import qwen35_moe_router as router

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


@pytest.mark.parametrize("experts,width", [(128, 1024), (256, 2048), (512, 2560)])
def test_fp16_gate_matches_stock(experts, width):
    for seed in range(20):
        mx.random.seed(seed)
        weight = (mx.random.normal((experts, width)) * 0.02).astype(mx.float16)
        x = (mx.random.normal((1, 1, width)) * (1 + seed % 4)).astype(mx.float16)
        out = router.router_logits_row(x, weight)
        expected = x @ weight.T
        mx.eval(out, expected)
        assert mx.array_equal(out.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.parametrize("experts", [128, 256, 512, 1024])
@pytest.mark.parametrize("top_k", [8, 10])
@pytest.mark.parametrize("kind", ["random", "ties", "adjacent", "saturated"])
def test_fp16_softmax_topk_preserves_routes_and_scores(experts, top_k, kind):
    for seed in range(20):
        mx.random.seed(seed)
        if kind == "ties":
            logits = mx.zeros((1, 1, experts), mx.float16)
        elif kind == "adjacent":
            codes = mx.array(4.0, mx.float16).view(mx.uint16)
            logits = (
                codes - mx.random.randint(0, 4, (1, 1, experts)).astype(mx.uint16)
            ).view(mx.float16)
        else:
            logits = (
                mx.random.normal((1, 1, experts)) * (80 if kind == "saturated" else 3)
            ).astype(mx.float16)
        expected_i, expected_s = router.fused_router_topk(
            mx.softmax(logits, axis=-1, precise=True), top_k
        )
        out_i, out_s = router.softmax_topk_row(logits, top_k)
        mx.eval(out_i, out_s, expected_i, expected_s)
        assert mx.array_equal(out_i, expected_i).item()
        assert mx.array_equal(out_s.view(mx.uint16), expected_s.view(mx.uint16)).item()


@pytest.mark.parametrize("top_k", [8, 10])
def test_fp16_combine_matches_every_gate_encoding(top_k):
    mx.random.seed(9)
    routed = mx.random.normal((1, 1, top_k, 64)).astype(mx.float16)
    scores = mx.softmax(mx.random.normal((1, 1, top_k)), axis=-1).astype(mx.float16)
    shared = mx.random.normal((1, 1, 64)).astype(mx.float16)
    gates = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(mx.float16)
    for start in range(0, gates.size, 2048):
        chunk = [gates[i].reshape(1, 1, 1) for i in range(start, start + 2048)]
        out = mx.stack(
            [router.fused_moe_combine(routed, scores, shared, g) for g in chunk]
        )
        expected = mx.stack(
            [
                (routed * scores[..., None]).sum(-2) + mx.sigmoid(g) * shared
                for g in chunk
            ]
        )
        mx.eval(out, expected)
        assert mx.array_equal(out.view(mx.uint16), expected.view(mx.uint16)).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_decode_block_uses_fused_router_and_invalidates_replaced_gate(
    monkeypatch, dtype
):
    from types import SimpleNamespace

    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    assert router.apply_qwen35_moe_router_patch()
    block = Qwen3_5MoeSparseMoeBlock(
        SimpleNamespace(
            hidden_size=1024,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts=128,
            num_experts_per_tok=8,
        )
    )
    block.set_dtype(dtype)
    block.eval()
    x = mx.random.normal((1, 1, 1024)).astype(dtype)
    calls = []
    build = router._router_gate_plan
    monkeypatch.setattr(
        router,
        "_router_gate_plan",
        lambda gate: calls.append(gate["weight"]) or build(gate),
    )
    for _step in range(2):
        with monkeypatch.context() as off:
            off.setattr(router, "router_gemv", lambda weight: None)
            off.setattr(router, "softmax_topk_row", lambda *args: None)
            off.setattr(router, "fused_moe_combine", lambda *args: None)
            # Do not populate the cached plan with the deliberately disabled launcher.
            off.setattr(router, "cached_per_module", lambda *args, **kwargs: None)
            expected = block(x)
            mx.eval(expected)
        out = block(x)
        mx.eval(out)
        assert mx.array_equal(out.view(mx.uint16), expected.view(mx.uint16)).item()
        assert calls[-1] is block.gate["weight"]
        assert block.gate.__dict__["_omlx_router_gate_plan"][-1] is not None
        cached = block.gate.__dict__["_omlx_router_gate_plan"]
        mx.eval(block(x))
        assert block.gate.__dict__["_omlx_router_gate_plan"] is cached
        block.gate["weight"] = block.gate["weight"] * 0.5
    assert len(calls) == 2
