# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiMo V2.5 mlx-lm monkey-patch (PR 1219 port)."""

import importlib
import json
import sys
import types

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
    assert sys.modules["mlx_lm.models.mimo_v2_flash"] is module

    import mlx_lm.models as models_pkg

    assert models_pkg.mimo_v2 is module
    assert models_pkg.mimo_v2_flash is module


def test_apply_is_idempotent():
    from omlx.patches.mimo_v2 import apply_mimo_v2_patch, is_applied

    first = apply_mimo_v2_patch()
    second = apply_mimo_v2_patch()

    assert is_applied() is True
    assert second is False
    assert first in (True, False)


def test_apply_replaces_upstream_module(monkeypatch):
    import mlx_lm.models as models_pkg

    import omlx.patches.mimo_v2 as patch

    upstream = types.ModuleType("mlx_lm.models.mimo_v2")
    upstream.__file__ = "/tmp/upstream/mlx_lm/models/mimo_v2.py"
    monkeypatch.setitem(sys.modules, "mlx_lm.models.mimo_v2", upstream)
    monkeypatch.setattr(models_pkg, "mimo_v2", upstream, raising=False)
    monkeypatch.setattr(patch, "_APPLIED", False)

    assert patch.apply_mimo_v2_patch() is True
    registered = sys.modules["mlx_lm.models.mimo_v2"]
    assert registered is not upstream
    assert registered.__file__.endswith("omlx/patches/mimo_v2/mimo_v2_model.py")
    assert models_pkg.mimo_v2 is registered
    assert sys.modules["mlx_lm.models.mimo_v2_flash"] is registered
    assert models_pkg.mimo_v2_flash is registered


@pytest.mark.parametrize("model_type", ["mimo_v2", "mimo_v2_flash"])
def test_get_classes_resolves_mimo_v2(model_type):
    _load_patch_module()

    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(_minimal_config(model_type=model_type))

    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"


def test_router_preserves_fp32_score_difference():
    module = _load_patch_module()
    args = module.ModelArgs.from_dict(_minimal_config(hidden_size=2))
    gate = module.MoEGate(args)
    gate.weight = mx.array([[1.0, 0.0], [1.0, 1.0]], dtype=mx.bfloat16)
    gate.e_score_correction_bias = mx.zeros((2,))
    hidden = mx.array([[[1.0, 1.0 / 256]]], dtype=mx.bfloat16)

    experts, _ = gate(hidden)

    assert experts.item() == 1


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
        num_nextn_predict_layers=1,
    )
    from omlx.patches.mlx_lm_mtp import set_mtp_active

    set_mtp_active(True)
    try:
        model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)

    weights = {
        "model.layers.0.self_attn.qkv_proj.weight": mx.to_fp8(mx.ones((240, 128))),
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv": mx.ones((2, 1)),
        "model.layers.0.self_attn.o_proj.weight": mx.to_fp8(mx.ones((128, 96))),
        "model.layers.0.self_attn.o_proj.weight_scale_inv": mx.ones((1, 1)),
        "visual.ignored": mx.ones((1,)),
        "audio_encoder.ignored": mx.ones((1,)),
        "speech_embeddings.ignored": mx.ones((1,)),
        "model.mtp.ignored": mx.ones((1,)),
        "model.mtp.layers.0.self_attn.qkv_proj.weight": mx.to_fp8(mx.ones((240, 128))),
        "model.mtp.layers.0.self_attn.qkv_proj.weight_scale_inv": mx.ones((2, 1)),
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
    assert "model.mtp.ignored" in sanitized
    assert sanitized["model.mtp.layers.0.self_attn.q_proj.weight"].shape == (
        128,
        128,
    )
    assert sanitized["model.mtp.layers.0.self_attn.k_proj.weight"].shape == (64, 128)
    assert sanitized["model.mtp.layers.0.self_attn.v_proj.weight"].shape == (48, 128)
    assert not any(
        key.startswith(("visual.", "audio_encoder.", "speech_embeddings."))
        for key in sanitized
    )

    inactive = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    assert "model.mtp.ignored" not in inactive.sanitize(
        {"model.mtp.ignored": mx.ones((1,))}
    )


def test_sanitize_loads_and_splits_quantized_mtp_sidecar(monkeypatch):
    mimo_v2 = _load_patch_module()
    from omlx.patches.mlx_lm_mtp import set_mtp_active

    sidecar_path = "/models/mimo/mtp/model_mtp.safetensors"
    config = _minimal_config(
        num_nextn_predict_layers=1,
        omlx_mtp_sidecar=sidecar_path,
    )
    set_mtp_active(True)
    try:
        model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)

    prefix = "model.mtp.layers.0.self_attn.qkv_proj"
    sidecar = {
        f"{prefix}.weight": mx.ones((240, 16), dtype=mx.uint32),
        f"{prefix}.scales": mx.ones((240, 2)),
        f"{prefix}.biases": mx.ones((240, 2)),
    }
    loaded = []

    def fake_load(path):
        loaded.append(path)
        return sidecar

    monkeypatch.setattr(mimo_v2.mx, "load", fake_load)
    sanitized = model.sanitize({})

    assert loaded == [sidecar_path]
    for suffix, width in (("weight", 16), ("scales", 2), ("biases", 2)):
        assert sanitized[f"model.mtp.layers.0.self_attn.q_proj.{suffix}"].shape == (
            128,
            width,
        )
        assert sanitized[f"model.mtp.layers.0.self_attn.k_proj.{suffix}"].shape == (
            64,
            width,
        )
        assert sanitized[f"model.mtp.layers.0.self_attn.v_proj.{suffix}"].shape == (
            48,
            width,
        )


def test_lightning_mtp_heads_forward_and_adapter_contract():
    mimo_v2 = _load_patch_module()
    from omlx.patches.mimo_v2.omnimodal import MiMoLanguageAdapter
    from omlx.patches.mlx_lm_mtp import (
        set_mtp_active,
        set_mtp_depth,
    )

    set_mtp_active(True)
    set_mtp_depth(3)
    try:
        args = mimo_v2.ModelArgs.from_dict(_minimal_config(num_nextn_predict_layers=3))
        model = mimo_v2.Model(args)
    finally:
        set_mtp_active(False)
        set_mtp_depth(1)

    assert len(model.mtp.layers) == 3
    assert model._omlx_mtp_decode_enabled is True
    assert model._omlx_mtp_depth == 3

    cache = model.make_cache()
    logits, hidden = model(mx.array([[1, 2]]), cache=cache, return_hidden=True)
    assert logits.shape == (1, 2, 1000)
    assert hidden.shape == (1, 2, 128)

    mtp_cache = model.make_mtp_cache()
    assert len(mtp_cache) == 3
    model.mtp_begin_cycle(mtp_cache, 3)
    first_logits, first_hidden = model.mtp_forward(
        hidden,
        mx.array([[2, 3]]),
        mtp_cache,
        return_hidden=True,
        logits_keep=1,
    )
    assert first_logits.shape == (1, 1, 1000)
    assert first_hidden.shape == (1, 2, 128)
    assert mtp_cache.layer_idx == 1

    second_logits = model.mtp_forward(first_hidden[:, -1:], mx.array([[4]]), mtp_cache)
    mx.eval(logits, first_logits, second_logits)
    assert second_logits.shape == (1, 1, 1000)
    assert mtp_cache.layer_idx == 2

    changed_hidden = mx.concatenate([hidden[:, :1], hidden[:, 1:] + 5], axis=1)
    original_logits = model.mtp_forward(
        hidden, mx.array([[2, 3]]), model.make_mtp_cache()
    )
    changed_logits = model.mtp_forward(
        changed_hidden, mx.array([[2, 3]]), model.make_mtp_cache()
    )
    assert mx.allclose(original_logits[:, :1], changed_logits[:, :1]).item()

    adapter = MiMoLanguageAdapter(model)
    assert adapter._omlx_mtp_decode_enabled is True
    assert adapter._omlx_mtp_chain is True
    adapter.mtp_begin_cycle(mtp_cache, 3)
    assert mtp_cache.layer_idx == 0
    adapter_logits, adapter_hidden = adapter(
        mx.array([[5]]), cache=model.make_cache(), return_hidden=True
    )
    mx.eval(adapter_logits, adapter_hidden)
    assert adapter_logits.shape == (1, 1, 1000)
    assert adapter_hidden.shape == (1, 1, 128)


@pytest.mark.parametrize("model_type", ["mimo_v2", "mimo_v2_flash"])
def test_pre_load_dispatch_calls_mimo_patch(tmp_path, monkeypatch, model_type):
    calls = []
    monkeypatch.setattr(
        "omlx.patches.mimo_v2.apply_mimo_v2_patch",
        lambda: calls.append(True) or True,
    )
    (tmp_path / "config.json").write_text(
        json.dumps(_minimal_config(model_type=model_type))
    )

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert calls == [True]


def test_mtp_sidecar_counts_as_checkpoint_weights(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from omlx.utils.model_loading import _checkpoint_has_mtp_weights

    sidecar = tmp_path / "mtp" / "model_mtp.safetensors"
    sidecar.parent.mkdir()
    save_file({"model.mtp.layers.0.weight": np.ones((1,), dtype=np.float32)}, sidecar)

    assert _checkpoint_has_mtp_weights(tmp_path) is True


def test_load_text_model_injects_mtp_sidecar(tmp_path, monkeypatch):
    import omlx.utils.model_loading as ml

    sidecar = tmp_path / "mtp" / "model_mtp.safetensors"
    sidecar.parent.mkdir()
    sidecar.touch()
    captured = {}
    monkeypatch.setattr(ml, "maybe_apply_pre_load_patches", lambda *_a, **_k: None)

    def fake_load(model_name, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(ml, "lm_load_compat", fake_load)
    ml.load_text_model(str(tmp_path))

    assert captured["model_name"] == str(tmp_path)
    assert captured["model_config"] == {"omlx_mtp_sidecar": str(sidecar)}


def test_multimodal_mimo_is_explicitly_routed_to_text_engine(tmp_path, caplog):
    from omlx.model_discovery import detect_model_type

    config = _minimal_config(
        vision_config={"hidden_size": 32},
        audio_config={"hidden_size": 16},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))

    with caplog.at_level("WARNING"):
        assert detect_model_type(tmp_path) == "llm"

    assert "no supported vision sidecar" in caplog.text


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


@pytest.mark.parametrize("model_type", ["mimo_v2", "mimo_v2_flash"])
def test_official_mxfp4_checkpoint_loads_without_requantizing(tmp_path, model_type):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    module = _load_patch_module()
    config = _minimal_config(model_type=model_type)
    model = module.Model(module.ModelArgs.from_dict(config))
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        mode="mxfp4",
        class_predicate=lambda path, layer: ".switch_mlp." in path
        and hasattr(layer, "to_quantized"),
    )
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if ".switch_mlp." not in name:
            weights[name] = value
            continue
        prefix, projection = name.split(".switch_mlp.")
        projection, suffix = projection.rsplit(".", 1)
        for expert, tensor in enumerate(value):
            key = f"{prefix}.experts.{expert}.{projection}.weight"
            if suffix == "scales":
                weights[key + "_scale"] = tensor
            else:
                weights[key] = tensor.view(mx.uint8)
    mx.eval(weights)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    config["quantization_config"] = {"quant_method": "fp8", "store_dtype": "mxfp4"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    maybe_apply_pre_load_patches(str(tmp_path))

    loaded, loaded_config = load_model(tmp_path)
    ids = mx.array([[1, 2, 3]])
    expected, actual = model(ids), loaded(ids)
    mx.eval(expected, actual)

    assert mx.array_equal(expected, actual).item()
    assert loaded_config["quantization"]["mode"] == "mxfp4"
    original = model.layers[1].mlp.switch_mlp.gate_proj
    restored = loaded.layers[1].mlp.switch_mlp.gate_proj
    assert mx.array_equal(original.weight, restored.weight).item()
    assert mx.array_equal(original.scales, restored.scales).item()

    from omlx.oq import quantize_oq_streaming

    output = tmp_path / "oq"
    quantize_oq_streaming(
        str(tmp_path),
        str(output),
        4,
        sensitivity_map_override={i: 1.0 for i in range(4)},
    )
    converted, _ = load_model(output)
    expert = converted.layers[1].mlp.switch_mlp.gate_proj
    assert mx.array_equal(original.weight, expert.weight).item()
    assert mx.array_equal(original.scales, expert.scales).item()
    assert mx.isfinite(converted(ids)).all().item()


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_mtp_partial_rollback_after_rotation(accepted):
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch, set_mtp_active
    from omlx.patches.mlx_lm_mtp.batch_generator import _call_backbone

    module = _load_patch_module()
    apply_mlx_lm_mtp_patch()
    set_mtp_active(True)
    try:
        model = module.Model(
            module.ModelArgs.from_dict(
                _minimal_config(num_nextn_predict_layers=3, sliding_window_size=8)
            )
        )
    finally:
        set_mtp_active(False)
    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    verify = mx.array([[10, 11, 12, 13]])
    speculative, reference = model.make_cache(), model.make_cache()
    model(prompt, cache=speculative)
    model(prompt, cache=reference)
    _call_backbone(model, verify, speculative, n_confirmed=1)
    assert model.mtp_partial_rollback(speculative, accepted, 3)
    model(verify[:, : accepted + 1], cache=reference)
    continuation = mx.array([[14]])
    actual, expected = model(continuation, cache=speculative), model(
        continuation, cache=reference
    )
    mx.eval(actual, expected)
    assert [c.offset for c in speculative] == [c.offset for c in reference]
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()


def test_mtp_draft_clone_preserves_head_index_and_isolates_cache():
    from omlx.patches.mlx_lm_mtp import set_mtp_active
    from omlx.patches.mlx_lm_mtp.batch_generator import _clone_mtp_head_cache

    module = _load_patch_module()
    set_mtp_active(True)
    try:
        model = module.Model(
            module.ModelArgs.from_dict(_minimal_config(num_nextn_predict_layers=3))
        )
    finally:
        set_mtp_active(False)
    cache = model.make_mtp_cache()
    hidden = mx.ones((1, 1, 128))
    ids = mx.array([[1]])
    model.mtp_begin_cycle(cache, 3)
    model.mtp_forward(hidden, ids, cache)
    clone = _clone_mtp_head_cache(cache)
    model.mtp_forward(hidden, ids, clone)
    assert clone.layer_idx == 2
    assert cache.layer_idx == 1
    assert clone[1].offset == 1
    assert cache[1].offset == 0


@pytest.mark.parametrize("layout", ["root", "nested"])
def test_oq_preserves_mtp_shards_and_calibrates_all_heads(tmp_path, layout):
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    from omlx.oq import (
        OQImatrixCollector,
        _collect_mtp_head_imatrix,
        quantize_oq_streaming,
    )
    from omlx.patches.mlx_lm_mtp import set_mtp_active

    module = _load_patch_module()
    config = _minimal_config(
        num_nextn_predict_layers=3,
        v_head_dim=32,
        swa_v_head_dim=32,
        vocab_size=1024,
    )
    set_mtp_active(True)
    try:
        model = module.Model(module.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)
    weights = dict(tree_flatten(model.parameters()))
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    trunk = {k: v for k, v in weights.items() if not k.startswith("model.mtp.")}
    head = {k: v for k, v in weights.items() if k.startswith("model.mtp.")}
    mx.save_safetensors(str(source / "model.safetensors"), trunk)
    sidecar = source / (
        "model_mtp.safetensors" if layout == "root" else "mtp/model_mtp.safetensors"
    )
    sidecar.parent.mkdir(exist_ok=True)
    mx.save_safetensors(str(sidecar), head)
    output = tmp_path / "output"
    quantize_oq_streaming(
        str(source),
        str(output),
        4,
        group_size=32,
        preserve_mtp=True,
        sensitivity_map_override={i: 1.0 for i in range(4)},
    )
    set_mtp_active(True)
    try:
        loaded, output_config = load_model(output)
    finally:
        set_mtp_active(False)
    assert output_config["num_nextn_predict_layers"] == 3
    assert len(loaded.mtp.layers) == 3
    ids = mx.array([[1, 2, 3, 4]])
    loaded.model.norm.weight = mx.linspace(0.5, 1.5, config["hidden_size"])
    _, hidden = loaded(ids, return_hidden=True)
    head = loaded.mtp.layers[0]
    expected_input = mx.concatenate(
        [head.enorm(loaded.model.embed_tokens(ids[:, 1:])), head.hnorm(hidden[:, :-1])],
        axis=-1,
    )
    expected_energy = mx.sum(expected_input.astype(mx.float32) ** 2, axis=(0, 1))
    collector = OQImatrixCollector()
    collector.install(loaded)
    try:
        assert _collect_mtp_head_imatrix(loaded, ids, hidden)
        for index in range(3):
            assert f"model.mtp.layers.{index}.eh_proj" in collector.entries
        actual_energy = collector.entries["model.mtp.layers.0.eh_proj"].in_sum2
        assert mx.allclose(mx.array(actual_energy), expected_energy, atol=1e-5).item()
    finally:
        collector.restore(loaded)
