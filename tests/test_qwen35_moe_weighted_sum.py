# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5/3.6 MoE weighted-sum prefill patch."""

from __future__ import annotations

import types

import mlx.core as mx
import pytest


class _Switch:
    up_proj = object()
    gate_proj = object()
    down_proj = object()


class _Block:
    top_k = 8
    sharding_group = None
    switch_mlp = _Switch()


@pytest.fixture(autouse=True)
def _fresh_moe_patch(monkeypatch):
    import omlx.patches.qwen35_moe_weighted_sum as patch

    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)
    monkeypatch.delenv("OMLX_QWEN35_MOE_WEIGHTED_SUM", raising=False)
    monkeypatch.delenv("OMLX_QWEN35_MOE_WEIGHTED_SUM_MIN_TOKENS", raising=False)

    originals = []
    try:
        import mlx_lm.models.qwen3_moe as qwen3_moe

        originals.append(
            (
                qwen3_moe.Qwen3MoeSparseMoeBlock,
                qwen3_moe.Qwen3MoeSparseMoeBlock.__call__,
            )
        )
    except Exception:
        pass
    yield
    monkeypatch.setattr(patch, "_PATCHED", False, raising=False)
    for cls, original in originals:
        cls.__call__ = original
        for name in (
            "_omlx_qwen_moe_weighted_sum_patched",
            "_omlx_qwen_moe_weighted_sum_original_call",
        ):
            if hasattr(cls, name):
                delattr(cls, name)


def test_moe_weighted_sum_route_gate(monkeypatch):
    import omlx.patches.qwen35_moe_weighted_sum as patch

    monkeypatch.setattr(patch.mx.metal, "is_available", lambda: True)
    x = mx.zeros((1, 1024, 128), dtype=mx.bfloat16)
    assert patch._should_route(_Block(), x, False, min_tokens=1024)
    assert not patch._should_route(_Block(), x[:, :1], False, min_tokens=1024)
    assert not patch._should_route(_Block(), x, True, min_tokens=1024)

    bad_topk = _Block()
    bad_topk.top_k = 2
    assert not patch._should_route(bad_topk, x, False, min_tokens=1024)

    sharded = _Block()
    sharded.sharding_group = object()
    assert not patch._should_route(sharded, x, False, min_tokens=1024)


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal is required")
def test_qwen3_moe_patch_matches_stock_and_skips_decode(monkeypatch):
    from mlx_lm.models import qwen3_moe

    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.patches.qwen35_moe_weighted_sum import (
        apply_qwen35_moe_weighted_sum_patch,
    )

    if not fast.has_symbol("qwen35_moe_weighted_sum"):
        pytest.skip("qwen35_moe_weighted_sum native kernel unavailable")

    monkeypatch.setenv("OMLX_QWEN35_MOE_WEIGHTED_SUM", "1")
    monkeypatch.setenv("OMLX_QWEN35_MOE_WEIGHTED_SUM_MIN_TOKENS", "16")

    args = types.SimpleNamespace(
        hidden_size=128,
        moe_intermediate_size=64,
        num_experts=16,
        num_experts_per_tok=8,
        norm_topk_prob=True,
    )
    block = qwen3_moe.Qwen3MoeSparseMoeBlock(args)
    x = mx.random.normal((1, 32, 128)).astype(mx.bfloat16)
    orig_call = qwen3_moe.Qwen3MoeSparseMoeBlock.__call__
    y_ref = orig_call(block, x)
    mx.eval(y_ref)

    calls = {"count": 0}
    orig_weighted_sum = fast.qwen35_moe_weighted_sum

    def spy(*args, **kwargs):
        calls["count"] += 1
        return orig_weighted_sum(*args, **kwargs)

    monkeypatch.setattr(fast, "qwen35_moe_weighted_sum", spy)
    assert apply_qwen35_moe_weighted_sum_patch() is True
    y = block(x)
    mx.eval(y)
    assert calls["count"] == 1

    diff = mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32))
    mx.eval(diff)
    assert mx.max(diff).item() <= 2e-2

    calls["count"] = 0
    y_decode = block(x[:, :1])
    mx.eval(y_decode)
    assert calls["count"] == 0
