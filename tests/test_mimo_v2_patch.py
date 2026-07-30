# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiMo V2.5 mlx-lm monkey-patch (PR 1219 port)."""

import importlib
import json
import sys

import mlx.core as mx
import pytest


def _minimal_config(**overrides):
    config = {
        "model_type": "mimo_v2",
        "architectures": ["MiMoV2ForCausalLM"],
        "vocab_size": 1000,
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "v_head_dim": 24,
        "rope_theta": 1000.0,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 32,
        "swa_v_head_dim": 24,
        "swa_rope_theta": 1000.0,
        "sliding_window_size": 32,
        "add_full_attention_sink_bias": False,
        "add_swa_attention_sink_bias": True,
        "hybrid_layer_pattern": [0, 1, 1, 0],
        "moe_layer_freq": [0, 1, 1, 1],
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        "partial_rotary_factor": 0.5,
        "attention_bias": False,
        "layernorm_epsilon": 1e-5,
        "max_position_embeddings": 1000,
        "attention_value_scale": 0.707,
    }
    config.update(overrides)
    return config


def _load_patch_module():
    from omlx.patches.mimo_v2 import apply_mimo_v2_patch

    apply_mimo_v2_patch()
    return importlib.import_module("mlx_lm.models.mimo_v2")


def test_apply_registers_mimo_v2_module():
    module = _load_patch_module()

    assert module.__package__ == "mlx_lm.models"
    assert sys.modules["mlx_lm.models.mimo_v2"] is module

    import mlx_lm.models as models_pkg

    assert models_pkg.mimo_v2 is module


def test_apply_is_idempotent():
    from omlx.patches.mimo_v2 import apply_mimo_v2_patch, is_applied

    first = apply_mimo_v2_patch()
    second = apply_mimo_v2_patch()

    assert is_applied() is True
    assert second is False
    assert first in (True, False)


def test_get_classes_resolves_mimo_v2():
    _load_patch_module()

    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(_minimal_config())

    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"


def test_mixed_cache_forward_and_continuous_batching():
    mimo_v2 = _load_patch_module()
    from mlx_lm.generate import BatchGenerator

    model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(_minimal_config()))
    cache = model.make_cache()

    assert [type(layer).__name__ for layer in cache] == [
        "KVCache",
        "RotatingKVCache",
        "RotatingKVCache",
        "KVCache",
    ]

    prefill = model(mx.array([[1, 2, 3], [4, 5, 6]]), cache=cache)
    decode = model(mx.array([[7], [8]]), cache=cache)
    mx.eval(prefill, decode)

    assert prefill.shape == (2, 3, 1000)
    assert decode.shape == (2, 1, 1000)

    generator = BatchGenerator(
        model,
        max_tokens=2,
        prefill_batch_size=2,
        completion_batch_size=2,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )
    uids = generator.insert([[1, 2, 3], [4, 5, 6]], max_tokens=[2, 2])
    finished = []
    for _ in range(8):
        _, generation_responses = generator.next()
        finished.extend(
            response
            for response in generation_responses
            if response.finish_reason is not None
        )
        if len(finished) == 2:
            break

    assert uids == [0, 1]
    assert {response.uid for response in finished} == {0, 1}
    assert all(response.finish_reason == "length" for response in finished)


def test_sanitize_handles_fused_fp8_and_text_only_weights():
    mimo_v2 = _load_patch_module()
    config = _minimal_config(
        num_hidden_layers=2,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 1],
    )
    model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))

    weights = {
        "model.layers.0.self_attn.qkv_proj.weight": mx.to_fp8(mx.ones((240, 128))),
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv": mx.ones((2, 1)),
        "model.layers.0.self_attn.o_proj.weight": mx.to_fp8(mx.ones((128, 96))),
        "model.layers.0.self_attn.o_proj.weight_scale_inv": mx.ones((1, 1)),
        "visual.ignored": mx.ones((1,)),
        "audio_encoder.ignored": mx.ones((1,)),
        "speech_embeddings.ignored": mx.ones((1,)),
        "model.mtp.ignored": mx.ones((1,)),
    }
    for projection, shape in (
        ("gate_proj", (64, 128)),
        ("up_proj", (64, 128)),
        ("down_proj", (128, 64)),
    ):
        for expert in range(2):
            weights[f"model.layers.1.mlp.experts.{expert}.{projection}.weight"] = (
                mx.ones(shape)
            )

    sanitized = model.sanitize(weights)

    assert sanitized["model.layers.0.self_attn.q_proj.weight"].shape == (128, 128)
    assert sanitized["model.layers.0.self_attn.k_proj.weight"].shape == (64, 128)
    assert sanitized["model.layers.0.self_attn.v_proj.weight"].shape == (48, 128)
    assert sanitized["model.layers.0.self_attn.o_proj.weight"].shape == (128, 96)
    assert sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape == (
        2,
        64,
        128,
    )
    assert not any(
        key.startswith(
            ("visual.", "audio_encoder.", "speech_embeddings.", "model.mtp.")
        )
        for key in sanitized
    )


def test_pre_load_dispatch_calls_mimo_patch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "omlx.patches.mimo_v2.apply_mimo_v2_patch",
        lambda: calls.append(True) or True,
    )
    (tmp_path / "config.json").write_text(json.dumps(_minimal_config()))

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert calls == [True]


def test_multimodal_mimo_is_explicitly_routed_to_text_engine(tmp_path, caplog):
    from omlx.model_discovery import detect_model_type

    config = _minimal_config(
        vision_config={"hidden_size": 32},
        audio_config={"hidden_size": 16},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))

    with caplog.at_level("WARNING"):
        assert detect_model_type(tmp_path) == "llm"

    assert "text-only" in caplog.text


def test_oq_uses_mlx_lm_sanitizer_for_multimodal_mimo(monkeypatch):
    import mlx_vlm.utils as vlm_utils

    from omlx.oq import _build_model_sanitizer

    monkeypatch.setattr(
        vlm_utils,
        "get_model_and_args",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("mlx-vlm lookup must be skipped")
        ),
    )
    config = _minimal_config(
        num_hidden_layers=2,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 1],
        vision_config={"hidden_size": 32},
        audio_config={"hidden_size": 16},
    )

    sanitize = _build_model_sanitizer(config, text_only=False)

    assert sanitize is not None
    assert sanitize({"visual.ignored": mx.ones((1,))}) == {}


def _neutralize_sensitivity_deps(monkeypatch):
    """Stub _measure_sensitivity's non-routing dependencies.

    Leaves the ``is_vlm``-driven loader selection intact so a test can assert
    which load path a config takes, without loading a real model or running
    calibration.
    """
    import omlx.oq as oq
    import omlx.utils.model_loading as ml

    monkeypatch.setattr(ml, "_checkpoint_has_mtp_weights", lambda *_a, **_k: False)
    monkeypatch.setattr(ml, "_has_mtp_heads", lambda *_a, **_k: False)
    monkeypatch.setattr(ml, "maybe_apply_pre_load_patches", lambda *_a, **_k: None)
    monkeypatch.setattr(
        oq,
        "_measure_sensitivity_from_model",
        lambda *_a, **_k: {"model.layers.0": 1.0},
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "qwen2_vl", "vision_config": {"hidden_size": 32}}, True),
        ({"model_type": "mimo_v2", "vision_config": {"hidden_size": 32}}, False),
        ({"model_type": "mimo-v2", "vision_config": {"hidden_size": 32}}, False),
        ({"model_type": "llama"}, False),
        ({"model_type": "mimo_v2"}, False),
    ],
    ids=[
        "genuine_vlm_is_vlm",
        "text_only_mimo_with_vision_is_not_vlm",
        "dashed_model_type_normalizes",
        "plain_llm_is_not_vlm",
        "mimo_text_only_quant_is_not_vlm",
    ],
)
def test_is_vlm_load_predicate(config, expected):
    from omlx.oq import _is_vlm_load

    assert _is_vlm_load(config) is expected


def test_measure_sensitivity_routes_multimodal_mimo_to_mlx_lm(monkeypatch):
    # Exception path: a text-only-served mimo base ships a vision_config but must
    # load via mlx-lm, not fall through to the mlx-vlm drafter lookup.
    # _measure_sensitivity wraps the load in try/except -> {}, so record the
    # loader calls rather than raising (a raise would be swallowed).
    import mlx_vlm.utils as vlm_utils

    import omlx.utils.model_loading as ml
    from omlx.oq import _measure_sensitivity

    _neutralize_sensitivity_deps(monkeypatch)
    vlm_calls, lm_calls = [], []
    monkeypatch.setattr(
        vlm_utils, "load_model", lambda *_a, **_k: vlm_calls.append(True) or object()
    )
    monkeypatch.setattr(
        ml,
        "lm_load_compat",
        lambda *_a, **_k: lm_calls.append(True) or (object(), object()),
    )

    config = _minimal_config(
        vision_config={"hidden_size": 32},
        audio_config={"hidden_size": 16},
    )
    result = _measure_sensitivity("/unused/path", config, oq_level=4)

    assert vlm_calls == []
    assert lm_calls == [True]
    assert result == {"model.layers.0": 1.0}


def test_measure_sensitivity_routes_genuine_vlm_to_mlx_vlm(monkeypatch):
    # Happy path: a real VLM (vision_config + non-text-only model_type) still
    # loads through mlx-vlm.
    import mlx_lm.tokenizer_utils as tok_utils
    import mlx_vlm.utils as vlm_utils

    import omlx.utils.model_loading as ml
    from omlx.oq import _measure_sensitivity

    _neutralize_sensitivity_deps(monkeypatch)
    vlm_calls, lm_calls = [], []
    monkeypatch.setattr(
        vlm_utils, "load_model", lambda *_a, **_k: vlm_calls.append(True) or object()
    )
    monkeypatch.setattr(tok_utils, "load", lambda *_a, **_k: object())
    monkeypatch.setattr(
        ml,
        "lm_load_compat",
        lambda *_a, **_k: lm_calls.append(True) or (object(), object()),
    )

    config = {"model_type": "qwen2_vl", "vision_config": {"hidden_size": 32}}
    result = _measure_sensitivity("/unused/path", config, oq_level=4)

    assert vlm_calls == [True]
    assert lm_calls == []
    assert result == {"model.layers.0": 1.0}
