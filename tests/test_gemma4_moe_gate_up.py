# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma 4 MoE gate/up fusion."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.gemma4_text import GeGLU
from mlx_lm.models.switch_layers import SwitchGLU

import omlx.patches.qwen35_moe_gate_up as patch_mod
from omlx.patches.qwen35_moe_gate_up import apply_gemma4_moe_gate_up_fusion

E, TOPK, HIDDEN, INTER = 128, 8, 64, 64


class _FakeGemma4Model:
    pass


_FakeGemma4Model.__module__ = "mlx_vlm.models.gemma4.gemma4"


class _FakeOtherModel:
    pass


_FakeOtherModel.__module__ = "mlx_vlm.models.other"


def _make_gemma4_moe_model(*, n_blocks: int = 30, model_class=_FakeGemma4Model):
    mx.random.seed(19)
    blocks = []
    for _ in range(n_blocks):
        glu = SwitchGLU(
            HIDDEN,
            INTER,
            E,
            activation=GeGLU(),
            bias=False,
        )
        glu.gate_proj = glu.gate_proj.to_quantized(64, 4, mode="affine")
        glu.up_proj = glu.up_proj.to_quantized(64, 4, mode="affine")
        glu.down_proj = glu.down_proj.to_quantized(64, 4, mode="affine")
        blocks.append(glu)

    model = model_class()
    model.config = SimpleNamespace(
        model_type="gemma4",
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            enable_moe_block=True,
            num_experts=E,
            top_k_experts=TOPK,
            num_hidden_layers=n_blocks,
            hidden_size=2816,
            moe_intermediate_size=704,
        ),
    )
    model.blocks = blocks
    model.named_modules = lambda: [
        (f"blocks.{index}", block) for index, block in enumerate(blocks)
    ]
    return model


@pytest.fixture(autouse=True)
def _restore_switch_glu(monkeypatch):
    monkeypatch.delenv("OMLX_GEMMA4_MOE_GATE_UP", raising=False)
    original = getattr(
        SwitchGLU,
        "_omlx_gate_up_original_call",
        SwitchGLU.__call__,
    )
    yield
    SwitchGLU.__call__ = original
    for attribute in (
        "_omlx_gate_up_fused_call",
        "_omlx_gate_up_original_call",
    ):
        if hasattr(SwitchGLU, attribute):
            delattr(SwitchGLU, attribute)
    patch_mod._CALL_PATCHED = False


@pytest.mark.parametrize("token_count", [1, 16], ids=["decode", "prefill"])
def test_gemma4_quantized_fusion_is_bit_exact(token_count):
    model = _make_gemma4_moe_model()
    x = (mx.random.normal(shape=(1, token_count, HIDDEN)) * 0.5).astype(mx.bfloat16)
    indices = mx.random.randint(0, E, shape=(1, token_count, TOPK))

    reference = [block(x, indices) for block in model.blocks]
    mx.eval(reference)

    assert apply_gemma4_moe_gate_up_fusion(model) == 30
    assert model._omlx_gemma4_gate_up_fused_count == 30
    output = [block(x, indices) for block in model.blocks]
    mx.eval(output)

    for block, expected, actual in zip(model.blocks, reference, output):
        assert hasattr(block, "gate_up_proj")
        assert not hasattr(block, "gate_proj")
        assert not hasattr(block, "up_proj")
        assert mx.array_equal(expected, actual).item()


def test_gemma4_fusion_is_idempotent():
    model = _make_gemma4_moe_model()

    assert apply_gemma4_moe_gate_up_fusion(model) == 30
    assert apply_gemma4_moe_gate_up_fusion(model) == 0
    assert model._omlx_gemma4_gate_up_fused_count == 30


@pytest.mark.parametrize(
    ("mutate", "model_class"),
    [
        (
            lambda model: setattr(model.config.text_config, "enable_moe_block", False),
            _FakeGemma4Model,
        ),
        (
            lambda model: setattr(model.config.text_config, "num_experts", 16),
            _FakeGemma4Model,
        ),
        (
            lambda model: setattr(model.config.text_config, "top_k_experts", 2),
            _FakeGemma4Model,
        ),
        (
            lambda model: setattr(model.config.text_config, "hidden_size", 1536),
            _FakeGemma4Model,
        ),
        (lambda model: None, _FakeOtherModel),
    ],
    ids=[
        "dense",
        "wrong-expert-count",
        "wrong-top-k",
        "wrong-hidden-size",
        "wrong-family",
    ],
)
def test_gemma4_fusion_excludes_unsupported_models(mutate, model_class):
    model = _make_gemma4_moe_model(model_class=model_class)
    mutate(model)

    assert apply_gemma4_moe_gate_up_fusion(model) == 0
    assert all(hasattr(block, "gate_proj") for block in model.blocks)
    assert not hasattr(model, "_omlx_gemma4_gate_up_fused_count")


def test_gemma4_fusion_skips_atomically_when_layer_count_does_not_match():
    model = _make_gemma4_moe_model(n_blocks=29)
    model.config.text_config.num_hidden_layers = 30

    assert apply_gemma4_moe_gate_up_fusion(model) == 0
    assert all(hasattr(block, "gate_proj") for block in model.blocks)


def test_gemma4_fusion_kill_switch_leaves_model_unchanged(monkeypatch):
    model = _make_gemma4_moe_model()
    monkeypatch.setenv("OMLX_GEMMA4_MOE_GATE_UP", "0")

    assert apply_gemma4_moe_gate_up_fusion(model) == 0
    assert all(hasattr(block, "gate_proj") for block in model.blocks)
    assert not hasattr(model, "_omlx_gemma4_gate_up_fused_count")
