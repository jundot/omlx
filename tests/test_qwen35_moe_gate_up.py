# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5/3.6 MoE gate+up fusion patch (issue #2238)."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

import omlx.patches.qwen35_moe_gate_up as patch_mod
from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

E, TOPK, HIDDEN, INTER = 8, 2, 64, 32


class _FakeQwenModel:
    # Module path carries the family token used by the gate.
    pass


_FakeQwenModel.__module__ = "mlx_lm.models.qwen3_5_moe"


class _FakeOtherModel:
    pass


_FakeOtherModel.__module__ = "mlx_lm.models.deepseek_v3"


def _make_model(quantize=True, model_cls=_FakeQwenModel, n_blocks=2):
    mx.random.seed(7)
    blocks = []
    for _ in range(n_blocks):
        glu = SwitchGLU(HIDDEN, INTER, E)
        if quantize:
            glu.gate_proj = glu.gate_proj.to_quantized(32, 4)
            glu.up_proj = glu.up_proj.to_quantized(32, 4)
            glu.down_proj = glu.down_proj.to_quantized(32, 4)
        blocks.append(glu)
    model = model_cls()
    model.blocks = blocks
    model.named_modules = lambda: [(f"blocks.{i}", b) for i, b in enumerate(blocks)]
    return model


@pytest.fixture(autouse=True)
def _restore_call(monkeypatch):
    monkeypatch.delenv("OMLX_QWEN35_MOE_GATE_UP", raising=False)
    orig = getattr(SwitchGLU, "_omlx_gate_up_original_call", SwitchGLU.__call__)
    yield
    SwitchGLU.__call__ = orig
    for attr in ("_omlx_gate_up_fused_call", "_omlx_gate_up_original_call"):
        if hasattr(SwitchGLU, attr):
            delattr(SwitchGLU, attr)
    patch_mod._CALL_PATCHED = False


def _forward_all(model, x, indices):
    return [blk(x, indices) for blk in model.blocks]


@pytest.mark.parametrize("quantize", [True, False])
def test_fused_output_bit_exact(quantize):
    model = _make_model(quantize=quantize)
    x = (mx.random.normal(shape=(1, 1, HIDDEN)) * 0.5).astype(mx.bfloat16)
    idx_decode = mx.random.randint(0, E, shape=(1, 1, TOPK))
    # 40 tokens x top-2 = 80 indices >= 64 exercises the sorted branch.
    x_sorted = (mx.random.normal(shape=(1, 40, HIDDEN)) * 0.5).astype(mx.bfloat16)
    idx_sorted = mx.random.randint(0, E, shape=(1, 40, TOPK))

    ref_decode = _forward_all(model, x, idx_decode)
    ref_prefill = _forward_all(model, x_sorted, idx_sorted)
    mx.eval(ref_decode, ref_prefill)

    fused = apply_qwen35_moe_gate_up_fusion(model)
    assert fused == 2
    for blk in model.blocks:
        assert hasattr(blk, "gate_up_proj")
        assert not hasattr(blk, "gate_proj")
        assert not hasattr(blk, "up_proj")

    out_decode = _forward_all(model, x, idx_decode)
    out_prefill = _forward_all(model, x_sorted, idx_sorted)
    mx.eval(out_decode, out_prefill)

    for ref, out in zip(ref_decode, out_decode):
        assert mx.array_equal(ref, out).item()
    for ref, out in zip(ref_prefill, out_prefill):
        assert mx.array_equal(ref, out).item()


def test_env_kill_switch(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN35_MOE_GATE_UP", "0")
    model = _make_model()
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert hasattr(model.blocks[0], "gate_proj")


def test_unsupported_family_skipped():
    model = _make_model(model_cls=_FakeOtherModel)
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert hasattr(model.blocks[0], "gate_proj")


def test_idempotent():
    model = _make_model()
    assert apply_qwen35_moe_gate_up_fusion(model) == 2
    assert apply_qwen35_moe_gate_up_fusion(model) == 0


def test_mismatched_quant_params_skipped():
    model = _make_model(quantize=False, n_blocks=1)
    glu = model.blocks[0]
    glu.gate_proj = glu.gate_proj.to_quantized(32, 4)
    glu.up_proj = glu.up_proj.to_quantized(32, 8)
    glu.down_proj = glu.down_proj.to_quantized(32, 4)
    assert apply_qwen35_moe_gate_up_fusion(model) == 0
    assert hasattr(glu, "gate_proj")


def test_vlm_target_verify_fused_bit_exact():
    lang = pytest.importorskip("mlx_vlm.models.qwen3_5_moe.language")

    model = _make_model(n_blocks=1)
    glu = model.blocks[0]
    x = (mx.random.normal(shape=(2, 3, HIDDEN)) * 0.5).astype(mx.bfloat16)
    idx = mx.random.randint(0, E, shape=(2, 3, TOPK))

    ref = lang._target_verify_switch_glu(glu, x, idx, True)
    mx.eval(ref)

    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    out = lang._target_verify_switch_glu(glu, x, idx, True)
    mx.eval(out)

    assert ref.shape == out.shape
    assert mx.array_equal(ref, out).item()


def test_weighted_sum_route_accepts_fused_layout():
    from omlx.patches.qwen35_moe_weighted_sum import _should_route

    model = _make_model(n_blocks=1)
    apply_qwen35_moe_gate_up_fusion(model)

    class _Block:
        top_k = 8
        sharding_group = None
        switch_mlp = model.blocks[0]

    if not mx.metal.is_available():
        pytest.skip("Metal required for _should_route")
    x = mx.zeros((1, 2048, HIDDEN), dtype=mx.bfloat16)
    assert _should_route(_Block(), x, target_verify=False, min_tokens=1024)
