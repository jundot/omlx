# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness and routing tests for the fused one-token routed experts."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

import omlx.patches.qwen35_moe_routed_decode as routed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

EXPERTS = 32


class _FakeQwen4Model:
    pass


_FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"


@pytest.fixture(autouse=True)
def _patched_block(monkeypatch):
    """Apply router + routed patches, restore the class afterwards."""
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    from omlx.patches.qwen35_moe_router import apply_qwen35_moe_router_patch

    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    apply_qwen35_moe_router_patch()  # process-wide and idempotent
    assert cls._omlx_router_fused
    call = cls.__call__
    original = getattr(call, "_omlx_routed_decode_original", call)
    monkeypatch.setattr(routed, "_DISABLED", False)
    monkeypatch.setattr(routed, "_PROVEN", False)
    cls.__call__ = original
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")
    assert routed.apply_qwen35_moe_routed_decode_patch()
    yield cls
    cls.__call__ = original
    cls._omlx_router_fused = True
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")


def _block(hidden, inter, top_k=10, bits=4, group_size=64, seed=0):
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter,
        num_experts=EXPERTS,
        num_experts_per_tok=top_k,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(sm, name, getattr(sm, name).to_quantized(group_size, bits))
    block.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("mlp.switch_mlp", sm)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    mx.eval(block.parameters())
    return block


def _pair(block, x):
    routed._DISABLED = True
    ref = block(x)
    mx.eval(ref)
    routed._DISABLED = False
    out = block(x)
    mx.eval(out)
    return ref, out


@pytest.mark.parametrize("hidden,inter", [(2560, 640), (1024, 320)])
def test_fused_decode_is_bit_identical(hidden, inter, monkeypatch):
    calls = []
    fused = routed.routed_decode
    monkeypatch.setattr(
        routed, "routed_decode", lambda *a: calls.append(1) or fused(*a)
    )
    block = _block(hidden, inter)
    for step in range(8):
        x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
        assert routed.routed_decode_eligible(block, x)
        ref, out = _pair(block, x)
        assert mx.array_equal(ref, out).item()
    assert len(calls) == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 1024, "inter": 512},  # down would take qmv_fast
        {"hidden": 960, "inter": 320},  # gate+up would take qmv
        {"hidden": 1024, "inter": 320, "top_k": 8},
        {"hidden": 1024, "inter": 320, "bits": 8},
        {"hidden": 1024, "inter": 320, "group_size": 32},
    ],
)
def test_ineligible_shapes_keep_the_composed_body(kwargs):
    block = _block(**kwargs)
    hidden = kwargs["hidden"]
    x = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
    assert not routed.routed_decode_eligible(block, x)


def test_prefill_verify_and_float_rows_keep_the_composed_body():
    block = _block(1024, 320)
    assert not routed.routed_decode_eligible(
        block, mx.zeros((1, 4, 1024), dtype=mx.bfloat16)
    )
    assert not routed.routed_decode_eligible(
        block, mx.zeros((2, 1, 1024), dtype=mx.bfloat16)
    )
    assert not routed.routed_decode_eligible(
        block, mx.zeros((1, 1, 1024), dtype=mx.float16)
    )
    x = mx.random.normal((1, 5, 1024)).astype(mx.bfloat16)
    ref, out = _pair(block, x)
    assert mx.array_equal(ref, out).item()


def test_block_without_gate_up_fusion_is_not_routed():
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    args = SimpleNamespace(
        hidden_size=1024,
        moe_intermediate_size=320,
        shared_expert_intermediate_size=320,
        num_experts=EXPERTS,
        num_experts_per_tok=10,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    x = mx.zeros((1, 1, 1024), dtype=mx.bfloat16)
    assert not routed.routed_decode_eligible(block, x)


def test_kernel_failure_falls_back_once(monkeypatch):
    block = _block(1024, 320)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    routed._DISABLED = True
    ref = block(x)
    routed._DISABLED = False

    def broken(*args):
        raise RuntimeError("no pipeline")

    monkeypatch.setattr(routed, "routed_decode", broken)
    out = block(x)
    mx.eval(ref, out)
    assert mx.array_equal(ref, out).item()
    assert routed._DISABLED
    assert not routed.routed_decode_eligible(block, x)


def test_apply_requires_the_fused_router(_patched_block):
    cls = _patched_block
    del cls._omlx_routed_decode
    cls._omlx_router_fused = False
    assert not routed.apply_qwen35_moe_routed_decode_patch()


def test_apply_is_idempotent(_patched_block):
    cls = _patched_block
    call = cls.__call__
    assert routed.apply_qwen35_moe_routed_decode_patch()
    assert cls.__call__ is call
