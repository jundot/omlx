# SPDX-License-Identifier: Apache-2.0
"""Tests for the Step 3.7 mlx-lm monkey-patch (PR 1325 port)."""

import importlib
import sys

import mlx.core as mx
import pytest


def _text_config(**overrides):
    cfg = dict(
        model_type="step3p5",
        hidden_size=256,
        num_hidden_layers=4,
        vocab_size=1024,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=64,
        intermediate_size=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        sliding_window=64,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        partial_rotary_factors=[0.5, 1.0, 1.0, 0.5],
        attention_other_setting={
            "num_attention_heads": 8,
            "num_attention_groups": 2,
        },
        use_head_wise_attn_gate=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=256,
        share_expert_dim=256,
        moe_layers_enum="1,2,3",
    )
    cfg.update(overrides)
    return cfg


def test_apply_registers_step3p7_module():
    from omlx.patches.step3p7 import apply_step3p7_patch

    apply_step3p7_patch()

    assert "mlx_lm.models.step3p7" in sys.modules
    mod = importlib.import_module("mlx_lm.models.step3p7")
    assert mod.__package__ == "mlx_lm.models"

    import mlx_lm.models as models_pkg

    assert models_pkg.step3p7 is mod


def test_apply_is_idempotent():
    from omlx.patches.step3p7 import apply_step3p7_patch, is_applied

    first = apply_step3p7_patch()
    second = apply_step3p7_patch()

    assert is_applied() is True
    assert second is False
    assert first in (True, False)


def test_get_classes_resolves_step3p7():
    from omlx.patches.step3p7 import apply_step3p7_patch

    apply_step3p7_patch()

    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(
        {"model_type": "step3p7", "text_config": _text_config()}
    )

    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"


def test_step3p7_wrapper_delegates_cache_and_forward():
    from omlx.patches.step3p7 import apply_step3p7_patch

    apply_step3p7_patch()

    from mlx_lm.models import step3p7
    from mlx_lm.models.cache import RotatingKVCache

    args = step3p7.ModelArgs(model_type="step3p7", text_config=_text_config())
    model = step3p7.Model(args)

    cache = model.make_cache()
    assert isinstance(cache[1], RotatingKVCache)

    logits = model(mx.array([[1, 2, 3]]))
    assert logits.shape == (1, 3, 1024)
    assert model.layers is model.language_model.layers


def test_step3p7_sanitize_drops_vision_and_nests_text_weights():
    from omlx.patches.step3p7 import apply_step3p7_patch

    apply_step3p7_patch()

    from mlx_lm.models import step3p7

    args = step3p7.ModelArgs(
        model_type="step3p7",
        text_config=_text_config(
            rope_theta=10000.0,
            partial_rotary_factors=[1.0] * 4,
        ),
    )
    model = step3p7.Model(args)

    weights = {
        "vision_model.conv1.weight": mx.zeros((4, 4)),
        "vision_model.transformer.resblocks.0.ln_1.weight": mx.zeros((4,)),
        "vit_large_projector.weight": mx.zeros((4, 4)),
        "model.embed_tokens.weight": mx.zeros((1024, 256)),
        "lm_head.weight": mx.zeros((1024, 256)),
        "model.norm.weight": mx.ones((256,)),
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((256, 256)),
        "model.layers.0.self_attn.q_norm.weight": mx.zeros((64,)),
        "model.layers.1.moe.gate.weight": mx.zeros((4, 256)),
        "model.layers.1.moe.router_bias": mx.zeros((4,)),
        "model.layers.1.moe.gate_proj.weight": mx.zeros((4, 256, 256)),
        "model.layers.4.enorm.weight": mx.zeros((256,)),
        "model.layers.4.self_attn.q_proj.weight": mx.zeros((256, 256)),
    }

    out = model.sanitize(weights)

    assert not any(k.startswith("vision_model") for k in out)
    assert not any("vit_large_projector" in k for k in out)
    assert not any("layers.4." in k for k in out)

    assert all(k.startswith("language_model.") for k in out)
    assert "language_model.lm_head.weight" in out
    assert "language_model.model.embed_tokens.weight" in out
    assert "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight" in out
    assert "language_model.model.layers.1.mlp.gate.gate.weight" in out
    assert "language_model.model.layers.1.mlp.gate.router_bias" in out

    assert mx.allclose(
        out["language_model.model.norm.weight"],
        mx.full((256,), 2.0),
    )


def test_pre_load_dispatch_applies_step3p7_patch(tmp_path):
    from omlx.patches import step3p7

    step3p7._APPLIED = False
    sys.modules.pop("mlx_lm.models.step3p7", None)
    import mlx_lm.models as models_pkg

    if hasattr(models_pkg, "step3p7"):
        delattr(models_pkg, "step3p7")

    (tmp_path / "config.json").write_text(
        '{"model_type": "step3p7", "text_config": {"model_type": "step3p5"}}'
    )

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert step3p7.is_applied() is True
    assert "mlx_lm.models.step3p7" in sys.modules


@pytest.fixture
def step3p7_mtp_model():
    from omlx.patches.mlx_lm_mtp import (
        is_mtp_active,
        set_mtp_active,
        step3p7_model,
    )
    from omlx.patches.step3p7 import apply_step3p7_patch

    apply_step3p7_patch()
    previous = is_mtp_active()
    set_mtp_active(True)
    try:
        assert step3p7_model.apply() is True

        from mlx_lm.models import step3p7

        text_config = _text_config(
            hidden_size=32,
            vocab_size=64,
            num_attention_heads=4,
            num_attention_groups=2,
            head_dim=8,
            intermediate_size=64,
            layer_types=[
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            partial_rotary_factors=[1.0] * 5,
            attention_other_setting={
                "num_attention_heads": 4,
                "num_attention_groups": 2,
            },
            moe_intermediate_size=16,
            share_expert_dim=16,
            num_nextn_predict_layers=1,
        )
        args = step3p7.ModelArgs.from_dict(
            {"model_type": "step3p7", "text_config": text_config}
        )
        yield step3p7.Model(args)
    finally:
        set_mtp_active(previous)


def test_step3p7_mtp_sanitize_shifts_raw_hf_norms(step3p7_mtp_model):
    weights = {
        "language_model.model.layers.0.input_layernorm.weight": mx.zeros((32,)),
        "language_model.model.layers.1.moe.gate_proj.weight": mx.zeros((1,)),
        "language_model.model.layers.4.enorm.weight": mx.zeros((32,)),
        "language_model.model.layers.4.hnorm.weight": mx.zeros((32,)),
        "language_model.model.layers.4.input_layernorm.weight": mx.zeros((32,)),
        "language_model.model.layers.4.post_attention_layernorm.weight": mx.zeros(
            (32,)
        ),
        "language_model.model.layers.4.self_attn.q_norm.weight": mx.zeros((8,)),
        "language_model.model.layers.4.self_attn.k_norm.weight": mx.zeros((8,)),
        "language_model.model.layers.4.transformer.shared_head.norm.weight": (
            mx.zeros((32,))
        ),
    }

    out = step3p7_mtp_model.sanitize(weights)

    assert mx.allclose(
        out["language_model.model.layers.0.input_layernorm.weight"],
        mx.ones((32,)),
    )
    for key in (
        "language_model.mtp.enorm.weight",
        "language_model.mtp.hnorm.weight",
        "language_model.mtp.block.input_layernorm.weight",
        "language_model.mtp.block.post_attention_layernorm.weight",
        "language_model.mtp.block.self_attn.q_norm.weight",
        "language_model.mtp.block.self_attn.k_norm.weight",
        "language_model.mtp.shared_head_norm.weight",
    ):
        assert mx.allclose(out[key], mx.ones(out[key].shape)), key


def test_step3p7_mtp_forward_returns_finite_logits(step3p7_mtp_model):
    inputs = mx.array([[1, 2]])
    logits, hidden = step3p7_mtp_model(inputs, return_hidden=True)
    mtp_logits = step3p7_mtp_model.mtp_forward(
        hidden[:, -1:],
        mx.array([[3]]),
        step3p7_mtp_model.make_mtp_cache(),
    )
    mx.eval(logits, mtp_logits)

    assert logits.shape == (1, 2, 64)
    assert mtp_logits.shape == (1, 1, 64)
    assert bool(mx.all(mx.isfinite(mtp_logits)).item())


def test_step3p7_mtp_sanitize_does_not_double_shift_converted_norms(
    step3p7_mtp_model,
):
    weights = {
        "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight": mx.zeros((1,)),
        "language_model.model.layers.4.enorm.weight": mx.ones((32,)),
        "language_model.model.layers.4.hnorm.weight": mx.ones((32,)),
        "language_model.model.layers.4.transformer.shared_head.norm.weight": (
            mx.ones((32,))
        ),
    }

    out = step3p7_mtp_model.sanitize(weights)

    for key in (
        "language_model.mtp.enorm.weight",
        "language_model.mtp.hnorm.weight",
        "language_model.mtp.shared_head_norm.weight",
    ):
        assert mx.allclose(out[key], mx.ones(out[key].shape)), key


@pytest.mark.parametrize(
    "prefix",
    (
        "model.layers.4",
        "language_model.model.layers.4",
        "model.language_model.layers.4",
    ),
)
def test_step3p7_mtp_sanitize_accepts_nextn_prefixes(
    step3p7_mtp_model,
    prefix,
):
    out = step3p7_mtp_model.sanitize({f"{prefix}.enorm.weight": mx.ones((32,))})

    assert mx.allclose(
        out["language_model.mtp.enorm.weight"],
        mx.ones((32,)),
    )


def test_step3p7_mtp_sanitize_tracks_streaming_norm_transforms(
    step3p7_mtp_model,
):
    from omlx.oq import _TrackedTensor

    raw = step3p7_mtp_model.sanitize(
        {
            "language_model.model.layers.1.moe.gate_proj.weight": _TrackedTensor(
                (1,), "F16"
            ),
            "language_model.model.layers.4.enorm.weight": _TrackedTensor((32,), "F16"),
        }
    )
    converted = step3p7_mtp_model.sanitize(
        {
            "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight": (
                _TrackedTensor((1,), "F16")
            ),
            "language_model.model.layers.4.enorm.weight": _TrackedTensor((32,), "F16"),
        }
    )

    assert raw["language_model.mtp.enorm.weight"].transform == "add"
    assert (
        converted["language_model.mtp.enorm.weight"].transform == "add_if_mean_lt_0_5"
    )
