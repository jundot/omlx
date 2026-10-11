# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiMo V2.5 mlx-lm monkey-patch (PR 1219 port)."""

import importlib
import json
import sys
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_causal_mask

from omlx.engine import batched as batched_engine
from omlx.patches.mimo_v2 import decode_fast as df
from omlx.patches.mimo_v2 import moe_decode as md
from omlx.patches.mimo_v2 import sdpa_flash as sf
from omlx.patches.specprefill import _OffsetAdjustedRoPE, _PositionMappedRoPE
from omlx.utils import fast_attention, nax_attention


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


def _disable_fast_attention(monkeypatch, mimo_v2):
    """Route every attention layer to the masked full SDPA reference."""
    monkeypatch.setattr(mimo_v2, "window_query_padding", lambda n: 0)
    monkeypatch.setattr(
        mimo_v2, "blocked_sliding_window_attention", lambda *a, **k: None
    )
    monkeypatch.setattr(mimo_v2, "mixed_head_dim_sdpa", lambda *a, **k: None)


def test_window_layers_pad_the_projection_input_not_the_queries(monkeypatch):
    """Padding the q_proj input for the blocked window path is bit-exact."""
    mimo_v2 = _load_patch_module()

    mx.random.seed(5)
    config = _minimal_config(sliding_window_size=128, hybrid_layer_pattern=[1, 1, 0, 1])
    model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)
    first_chunk = mx.random.randint(0, 1000, (1, 300))  # 84 padding rows
    second_chunk = mx.random.randint(0, 1000, (1, 257))  # 127, after a prefix
    real_pad = mimo_v2.window_query_padding
    asked = []

    def run(pad_fn):
        monkeypatch.setattr(mimo_v2, "window_query_padding", pad_fn)
        cache = model.make_cache()
        out = [model(first_chunk, cache=cache), model(second_chunk, cache=cache)]
        mx.eval(out)
        return out

    padded = run(lambda n: asked.append(n) or real_pad(n))
    assert set(asked) == {300, 257}
    unpadded = run(lambda n: 0)  # the blocked path pads the queries itself
    for a, b in zip(padded, unpadded):
        assert mx.array_equal(a, b).item()

    _disable_fast_attention(monkeypatch, mimo_v2)
    reference = run(lambda n: 0)
    for a, b in zip(padded, reference):
        assert mx.allclose(
            a.astype(mx.float32), b.astype(mx.float32), atol=5e-2, rtol=5e-2
        ).item()


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


def test_batched_engine_load_kwargs_carry_mtp_sidecar(tmp_path):
    assert batched_engine._mtp_sidecar_load_kwargs(str(tmp_path)) == {}
    sidecar = tmp_path / "mtp" / "model_mtp.safetensors"
    sidecar.parent.mkdir()
    sidecar.touch()

    assert batched_engine._mtp_sidecar_load_kwargs(str(tmp_path)) == {
        "model_config": {"omlx_mtp_sidecar": str(sidecar)}
    }


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


def _quantized_moe(mimo, T, top_k=8):
    cfg = mimo.ModelArgs.from_dict(
        _minimal_config(n_routed_experts=16, num_experts_per_tok=top_k)
    )
    moe = mimo.MoE(cfg)
    mx.random.seed(0)
    moe.gate.weight = mx.random.normal(moe.gate.weight.shape) * 0.1
    moe.gate.e_score_correction_bias = mx.zeros_like(moe.gate.e_score_correction_bias)
    nn.quantize(moe.switch_mlp, group_size=64, bits=4)
    x = mx.random.normal((1, T, 128)).astype(mx.bfloat16)
    mx.eval(moe.parameters(), x)
    return moe, x


def _unfused_combine(moe, x):
    inds, scores = moe.gate(x)
    y = moe.switch_mlp(x, inds)
    if y.ndim == x.ndim + 1:
        y = (y * scores[..., None]).sum(axis=-2)
    return y.astype(x.dtype)


@pytest.mark.parametrize("T", [4, 96])
def test_moe_fused_combine_matches_unfused_combine(T):
    """Top-8 routing combines through glm_moe_weighted_sum (sorted prefill)."""
    moe, x = _quantized_moe(_load_patch_module(), T)
    assert moe._fused_combine
    out = moe(x)
    ref = _unfused_combine(moe, x)
    mx.eval(out, ref)
    assert out.shape == ref.shape == x.shape
    assert out.dtype == x.dtype
    assert mx.allclose(
        out.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2
    ).item()


def test_moe_unsupported_top_k_uses_plain_switch_glu():
    moe, x = _quantized_moe(_load_patch_module(), 96, top_k=2)
    assert not moe._fused_combine
    out = moe(x)
    ref = _unfused_combine(moe, x)
    mx.eval(out, ref)
    assert mx.allclose(
        out.astype(mx.float32), ref.astype(mx.float32), atol=2e-2, rtol=2e-2
    ).item()


# --- Prefill attention fast paths (omlx.utils.fast_attention / nax_attention) ---

requires_nax = pytest.mark.skipif(
    not mx.metal.is_available() or not nax_attention._nax_available(),
    reason="NAX (M5) GPU required",
)


def _window_mask(prefix, S, window):
    qpos = mx.arange(prefix, S)[:, None]
    kpos = mx.arange(S)[None, :]
    return (kpos <= qpos) & (kpos > qpos - window)


@pytest.mark.parametrize(
    "prefix,dims,L",
    [(0, (192, 128), 512), (127, (192, 128), 511), (300, (64, 64), 300)],
)
def test_blocked_window_attention_matches_masked(prefix, dims, L):
    mx.random.seed(prefix + L)
    qk_dim, v_dim = dims
    H, Hk, window = 8, 2, 128
    S = prefix + L
    q = mx.random.normal((1, H, L, qk_dim))
    k = mx.random.normal((1, Hk, S, qk_dim))
    v = mx.random.normal((1, Hk, S, v_dim))
    sinks = mx.random.normal((H,))
    scale = qk_dim**-0.5
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=_window_mask(prefix, S, window), sinks=sinks
    )
    out = fast_attention.blocked_sliding_window_attention(
        q, k, v, scale=scale, window=window, sinks=sinks
    )
    assert out is not None and out.shape == ref.shape
    assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()


def test_blocked_window_attention_honours_padded_bool_mask():
    mx.random.seed(37)
    H, Hk, L, window, prefix, pad = 4, 2, 384, 128, 100, 37
    S = prefix + L
    q = mx.random.normal((1, H, L, 64))
    k = mx.random.normal((1, Hk, S, 64))
    v = mx.random.normal((1, Hk, S, 64))
    mask = (_window_mask(prefix, S, window) & (mx.arange(S)[None, :] >= pad))[
        None, None
    ]
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125, mask=mask)
    out = fast_attention.blocked_sliding_window_attention(
        q, k, v, scale=0.125, window=window, mask=mask
    )
    assert out is not None
    assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()


def test_blocked_window_attention_declines_unsupported_layouts():
    f = fast_attention.blocked_sliding_window_attention
    batched = mx.zeros((2, 4, 512, 64))
    assert f(batched, batched, batched, scale=1.0, window=128) is None
    short = mx.zeros((1, 4, 200, 64))
    assert f(short, short, short, scale=1.0, window=128) is None


def test_window_layers_skip_query_padding_under_wrapped_rope(monkeypatch):
    """SpecPrefill maps RoPE positions per query row; padded window queries
    would outgrow the position slice on the last sparse chunk."""
    mimo_v2 = _load_patch_module()
    mx.random.seed(6)
    config = _minimal_config(sliding_window_size=128, hybrid_layer_pattern=[1, 1, 0, 1])
    model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    positions = mx.arange(301) * 3
    tokens = mx.random.randint(0, 1000, (1, 301))
    outs = []
    for fast in (True, False):
        if not fast:
            _disable_fast_attention(monkeypatch, mimo_v2)
        try:
            for layer in model.model.layers:
                attn = layer.self_attn
                attn.rope = _PositionMappedRoPE(attn.rope, positions, cache_start=0)
            cache = model.make_cache()
            model(tokens[:, :300], cache=cache)
            outs.append(model(tokens[:, 300:], cache=cache))
            mx.eval(outs[-1])
        finally:
            for layer in model.model.layers:
                layer.self_attn.rope = layer.self_attn.rope._original
    assert mx.allclose(outs[0], outs[1], atol=1e-3, rtol=1e-3).item()


def _nax_reference(q, k, v, scale, mask=None, sinks=None):
    """fp32 attention with an exact softmax (optional sinks)."""
    B, H, qL, D = q.shape
    Hk, kL = k.shape[1], k.shape[2]
    g = H // Hk
    s = (q.astype(mx.float32).reshape(B, Hk, g, qL, D) * scale) @ k.astype(mx.float32)[
        :, :, None
    ].swapaxes(-1, -2)
    if isinstance(mask, str):
        m = (mx.arange(qL)[:, None] + (kL - qL)) >= mx.arange(kL)[None]
        s = mx.where(m, s, -mx.inf)
    elif mask is not None:
        m = mx.broadcast_to(mask, (B, H, qL, kL)).reshape(B, Hk, g, qL, kL)
        s = mx.where(m, s, -mx.inf)
    top = s.max(-1, keepdims=True)
    if sinks is not None:
        sk = sinks.astype(mx.float32).reshape(1, Hk, g, 1, 1)
        top = mx.maximum(top, sk)
        p = mx.exp(s - top)
        den = p.sum(-1, keepdims=True) + mx.exp(sk - top)
    else:
        p = mx.exp(s - top)
        den = p.sum(-1, keepdims=True)
    return ((p @ v.astype(mx.float32)[:, :, None]) / den).reshape(B, H, qL, -1)


def _nax_inputs(B, H, Hk, qL, kL, seed=0):
    mx.random.seed(seed)
    q = (0.5 * mx.random.normal((B, H, qL, 192))).astype(mx.bfloat16)
    k = (0.5 * mx.random.normal((B, Hk, kL, 192))).astype(mx.bfloat16)
    v = (0.5 * mx.random.normal((B, Hk, kL, 128))).astype(mx.bfloat16)
    return q, k, v


def _nax_case(case):
    B, H, Hk, qL, kL, mask_kind, has_sinks = case
    q, k, v = _nax_inputs(B, H, Hk, qL, kL, seed=qL + kL)
    mask = mask_kind
    if mask_kind == "array":
        mask = mx.random.uniform(shape=(B, 1, qL, kL)) > 0.3
        mask[..., 0] = True
    sinks = (2 * mx.random.normal((H,))).astype(mx.bfloat16) if has_sinks else None
    return q, k, v, mask, sinks


def _max_err(out, ref):
    return mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()


# (B, H, Hk, qL, kL, mask, sinks): unaligned tails, a prefix, bool masks,
# sinks and a batch.
_NAX_CASES = [
    (1, 8, 2, 2049, 2049, "causal", True),
    (1, 8, 2, 1031, 4096, "causal", False),
    (2, 8, 4, 300, 700, "array", True),
]


@requires_nax
@pytest.mark.parametrize("case", _NAX_CASES)
def test_nax_attention_matches_fp32_reference(case):
    q, k, v, mask, sinks = _nax_case(case)
    scale = 192**-0.5
    out = nax_attention.nax_mixed_head_dim_attention(
        q, k, v, scale=scale, mask=mask, sinks=sinks
    )
    assert out is not None and out.shape == (*q.shape[:3], 128)
    assert _max_err(out, _nax_reference(q, k, v, scale, mask, sinks)) < 1e-2


@requires_nax
@pytest.mark.parametrize("case", [_NAX_CASES[1], _NAX_CASES[2]])
@pytest.mark.parametrize("dsplit", [0, 2])
def test_nax_attention_key_range_passes_are_bit_identical(case, dsplit):
    """Several key-range dispatches resume the exact fp32 row state."""
    q, k, v, mask, sinks = _nax_case(case)
    B, H, kL = q.shape[0], q.shape[1], k.shape[2]
    scale = 192**-0.5
    run = nax_attention._run
    one = run(q, k, v, scale, mask, sinks, pass_keys=0, dsplit=dsplit)
    pass_keys = kL // 3
    assert len(nax_attention._pass_edges(B * H, (kL + 31) // 32, pass_keys, 0)) == 4
    many = run(
        q, k, v, scale, mask, sinks, pass_keys=pass_keys, min_groups=0, dsplit=dsplit
    )
    assert mx.array_equal(one, many).item()


@requires_nax
def test_nax_attention_head_dim_split_changes_only_summation_order():
    q, k, v, mask, sinks = _nax_case(_NAX_CASES[0])
    scale = 192**-0.5
    run = nax_attention._run
    one_sg = run(q, k, v, scale, mask, sinks, pass_keys=0, dsplit=0)
    split = run(q, k, v, scale, mask, sinks, pass_keys=0, dsplit=2)
    assert mx.mean((split == one_sg).astype(mx.float32)).item() > 0.98
    assert _max_err(split, _nax_reference(q, k, v, scale, mask, sinks)) < 1e-2


@requires_nax
def test_blocked_window_attention_uses_jit_kernel(monkeypatch):
    """MiMo's window layers (192/128, sinks, block masks) on stock MLX."""
    monkeypatch.setattr(
        fast_attention, "_native_mixed_dims_supported", lambda *a: False
    )
    calls = []
    real = fast_attention.nax_mixed_head_dim_attention

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(fast_attention, "nax_mixed_head_dim_attention", spy)
    H, Hk, L, window, prefix = 8, 2, 511, 128, 100
    S = prefix + L
    q, k, v = _nax_inputs(1, H, Hk, L, S, seed=prefix)
    sinks = mx.random.normal((H,)).astype(mx.bfloat16)
    scale = 192**-0.5
    out = fast_attention.blocked_sliding_window_attention(
        q, k, v, scale=scale, window=window, sinks=sinks
    )
    assert calls == [True]
    mask = _window_mask(prefix, S, window)[None, None]
    assert _max_err(out, _nax_reference(q, k, v, scale, mask, sinks)) < 1e-2


def test_nax_attention_declines_unsupported_inputs(monkeypatch):
    monkeypatch.setattr(nax_attention, "_nax_available", lambda: True)
    monkeypatch.setattr(nax_attention, "_self_check_passed", lambda: True)
    q, k, v = _nax_inputs(1, 4, 2, 64, 64)
    f = nax_attention.nax_mixed_head_dim_attention
    assert f(q[..., :128], k[..., :128], v, scale=1.0) is None  # 128/128
    assert (
        f(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=1.0)
        is None
    )
    assert f(q[:, :, :8], k, v, scale=1.0) is None  # decode-shaped
    assert f(q, k, v, scale=1.0, mask=mx.zeros((64, 64), dtype=mx.bfloat16)) is None
    assert f(q, k, v, scale=1.0, sinks=mx.zeros((3,))) is None
    assert f(q[:, :3], k, v, scale=1.0) is None  # heads not a multiple


def test_nax_attention_failed_self_check_disables_route(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("Unable to build metal library from source")

    monkeypatch.setattr(nax_attention, "_nax_available", lambda: True)
    monkeypatch.setattr(nax_attention, "_run_edges", boom)
    nax_attention._self_check_passed.cache_clear()
    try:
        q, k, v = _nax_inputs(1, 4, 2, 64, 64)
        assert nax_attention.nax_mixed_head_dim_attention(q, k, v, scale=1.0) is None
    finally:
        nax_attention._self_check_passed.cache_clear()


# --- Decode / short-verify fast path (omlx.patches.mimo_v2.decode_fast) ---

_DECODE_CONFIG = {
    "model_type": "mimo_v2",
    "vocab_size": 512,
    "hidden_size": 1024,
    "intermediate_size": 512,
    "moe_intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 48,
    "v_head_dim": 32,
    "rope_theta": 1e7,
    "swa_num_attention_heads": 8,
    "swa_num_key_value_heads": 4,
    "swa_head_dim": 48,
    "swa_v_head_dim": 32,
    "swa_rope_theta": 10000.0,
    "sliding_window_size": 32,
    "add_full_attention_sink_bias": False,
    "add_swa_attention_sink_bias": True,
    "hybrid_layer_pattern": [0, 1, 1, 0],
    "moe_layer_freq": [0, 1, 1, 1],
    "n_routed_experts": 64,
    "num_experts_per_tok": 8,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "topk_method": "noaux_tc",
    "partial_rotary_factor": 0.334,
    "attention_bias": False,
    "layernorm_epsilon": 1e-6,
    "max_position_embeddings": 4096,
    "attention_value_scale": 0.707,
}


def _mismatches(a, b):
    a = np.array(a.astype(mx.float32))
    b = np.array(b.astype(mx.float32))
    assert a.shape == b.shape
    return int((a != b).sum())


def _decode_model(seed=3, **overrides):
    """8-bit affine attention/dense and MXFP4 experts (MiMo-V2.6-Flash)."""
    m = _load_patch_module()
    mx.random.seed(seed)
    model = m.Model(m.ModelArgs.from_dict({**_DECODE_CONFIG, **overrides}))
    updates = []
    for key, value in tree_flatten(model.parameters()):
        if key.endswith("gate.weight"):
            updates.append((key, mx.random.normal(value.shape) * 0.05))
        elif key.endswith("e_score_correction_bias"):
            updates.append((key, mx.random.normal(value.shape) * 0.02))
        elif key.endswith("attention_sink_bias"):
            updates.append((key, mx.random.normal(value.shape)))
        elif "norm" in key:
            updates.append((key, 1 + 0.1 * mx.random.normal(value.shape)))
    model.load_weights(updates, strict=False)
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda p, mod: isinstance(mod, nn.Linear)
        and "switch_mlp" not in p,
    )
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        mode="mxfp4",
        class_predicate=lambda p, mod: "switch_mlp" in p
        and hasattr(mod, "to_quantized"),
    )
    casts = [
        (k, v.astype(mx.bfloat16))
        for k, v in tree_flatten(model.parameters())
        if v.dtype == mx.float32 and "e_score_correction_bias" not in k
    ]
    model.load_weights(casts, strict=False)
    mx.eval(model.parameters())
    return model


def _per_row_router(monkeypatch, m):
    """Reference router with every token row through MLX's M=1 gemv, the
    arithmetic the fast path uses for verify rows."""
    orig = m.MoEGate.__call__

    def call(self, x):
        if x.shape[-2] * x.shape[0] == 1:
            return orig(self, x)
        w32 = self.weight.astype(mx.float32)
        rows = [
            mx.concatenate(
                [
                    x[b : b + 1, i : i + 1].astype(mx.float32) @ w32.T
                    for i in range(x.shape[1])
                ],
                axis=1,
            )
            for b in range(x.shape[0])
        ]
        return m.group_expert_select(
            mx.concatenate(rows, axis=0),
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )

    monkeypatch.setattr(m.MoEGate, "__call__", call)


def _clone(caches):
    # Fresh arrays: KV caches write their buffers in place.
    out = []
    for c in caches:
        n = type(c).__new__(type(c))
        n.__dict__.update(
            {
                k: mx.array(v) if isinstance(v, mx.array) else v
                for k, v in c.__dict__.items()
            }
        )
        out.append(n)
    return out


def _forward(model, tokens, cache, fast, monkeypatch):
    monkeypatch.setattr(df, "enabled", lambda: fast)
    out = model(tokens, cache=cache)
    mx.eval(out, [c.state for c in cache])
    return out


def _count_fast_runs(monkeypatch):
    calls = {"n": 0, "experts": 0}
    orig, orig_experts = df.combine_rms, df._experts

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    def counted_experts(*a, **k):
        calls["experts"] += 1
        return orig_experts(*a, **k)

    monkeypatch.setattr(df, "combine_rms", counted)
    monkeypatch.setattr(df, "_experts", counted_experts)
    return calls


@requires_nax
def test_decode_fast_forward_matches_reference(monkeypatch):
    model = _decode_model()
    _per_row_router(monkeypatch, _load_patch_module())
    calls = _count_fast_runs(monkeypatch)
    tokens = mx.random.randint(0, 512, (1, 70))
    cache = model.make_cache()
    _forward(model, tokens[:, :40], cache, False, monkeypatch)
    pos = 40
    for L in [1, 2, 3, 1, 7, 1]:
        step = tokens[:, pos : pos + L]
        ref_cache, fast_cache = _clone(cache), _clone(cache)
        ref = _forward(model, step, ref_cache, False, monkeypatch)
        before, before_experts = calls["n"], calls["experts"]
        fast = _forward(model, step, fast_cache, True, monkeypatch)
        assert calls["n"] > before and calls["experts"] > before_experts
        assert _mismatches(ref, fast) == 0, f"logits differ at L={L}"
        for a, b in zip(ref_cache, fast_cache):
            assert a.offset == b.offset
            assert _mismatches(a.state[0], b.state[0]) == 0
            assert _mismatches(a.state[1], b.state[1]) == 0
        cache = fast_cache
        pos += L


@requires_nax
def test_decode_fast_forward_matches_reference_batch_caches(monkeypatch):
    model = _decode_model(seed=4)
    _per_row_router(monkeypatch, _load_patch_module())
    calls = _count_fast_runs(monkeypatch)
    tokens = mx.random.randint(0, 512, (1, 48))
    ca, cb = model.make_cache(), model.make_cache()
    _forward(model, tokens[:, :45], ca, False, monkeypatch)
    _forward(model, tokens[:, 3:40], cb, False, monkeypatch)
    batch = [type(a).merge([a, b]) for a, b in zip(ca, cb)]
    for L in [1, 3]:
        step = mx.random.randint(0, 512, (2, L))
        ref_cache, fast_cache = _clone(batch), _clone(batch)
        ref = _forward(model, step, ref_cache, False, monkeypatch)
        before = calls["n"]
        fast = _forward(model, step, fast_cache, True, monkeypatch)
        assert calls["n"] > before
        assert _mismatches(ref, fast) == 0
        batch = fast_cache


@requires_nax
def test_decode_fast_gqa16_chunked_sdpa_matches_reference(monkeypatch):
    """GQA 16 (MiMo's full-attention layers): 3+ verify rows run the vector
    SDPA in row chunks, compared with a reference using the same chunks."""
    m = _load_patch_module()
    model = _decode_model(
        seed=11,
        num_attention_heads=16,
        num_key_value_heads=1,
        swa_num_attention_heads=16,
        swa_num_key_value_heads=2,
    )
    _per_row_router(monkeypatch, m)
    orig_sdpa = m.scaled_dot_product_attention
    chunked = {"n": 0}

    def ref_sdpa(q, k, v, cache=None, scale=1.0, mask=None, sinks=None):
        rep, L = q.shape[1] // k.shape[1], q.shape[2]
        if 1 < L <= 8 and L * rep > 32:
            chunked["n"] += 1
            out = df._sdpa_row_chunks(
                orig_sdpa, q, k, v, cache, scale, mask, sinks, 32 // rep
            )
            return out.reshape(q.shape[0], L, q.shape[1], -1).swapaxes(1, 2)
        return orig_sdpa(q, k, v, cache=cache, scale=scale, mask=mask, sinks=sinks)

    monkeypatch.setattr(m, "scaled_dot_product_attention", ref_sdpa)
    tokens = mx.random.randint(0, 512, (1, 60))
    cache = model.make_cache()
    _forward(model, tokens[:, :40], cache, False, monkeypatch)
    pos = 40
    for L in [1, 3, 7]:
        step = tokens[:, pos : pos + L]
        ref_cache, fast_cache = _clone(cache), _clone(cache)
        before = chunked["n"]
        ref = _forward(model, step, ref_cache, False, monkeypatch)
        assert (chunked["n"] > before) == (L >= 3)
        fast = _forward(model, step, fast_cache, True, monkeypatch)
        assert _mismatches(ref, fast) == 0, f"logits differ at L={L}"
        cache = fast_cache
        pos += L


@requires_nax
def test_decode_fast_declines_unsupported_forwards(monkeypatch):
    model = _decode_model(seed=5)
    inner = model.model
    cache = model.make_cache()
    monkeypatch.setattr(df, "enabled", lambda: True)
    h1 = inner.embed_tokens(mx.array([[1]]))
    # 8 rows x top-8 would take SwitchGLU's sorted path.
    assert (
        df.run_layers(inner, inner.embed_tokens(mx.array([[1] * 8])), cache, None, None)
        is None
    )
    assert df.run_layers(inner, h1, [None] * len(cache), None, None) is None
    assert df.run_layers(inner, h1.astype(mx.float32), cache, None, None) is None
    monkeypatch.setattr(df, "enabled", lambda: False)
    assert df.run_layers(inner, h1, cache, None, None) is None


@requires_nax
def test_decode_fast_runs_the_reference_under_wrapped_rope(monkeypatch):
    """The fused q/k/v path reads RoPE parameters directly: SpecPrefill's
    wrapped ropes run the reference, and the path arms once they are gone."""
    model = _decode_model(seed=6)
    inner = model.model
    cache = model.make_cache()
    _forward(model, mx.random.randint(0, 512, (1, 20)), cache, False, monkeypatch)
    step = mx.array([[7]])
    originals = [layer.self_attn.rope for layer in inner.layers]
    for layer, rope in zip(inner.layers, originals):
        layer.self_attn.rope = _PositionMappedRoPE(
            rope, mx.arange(100, 200), cache_start=0
        )
    h = inner.embed_tokens(step)
    assert df.run_layers(inner, h, _clone(cache), None, None) is None
    assert "_omlx_decode_fast_state" not in inner.__dict__  # not latched off
    for layer, rope in zip(inner.layers, originals):
        layer.self_attn.rope = _OffsetAdjustedRoPE(rope, 1000)
    ref = _forward(model, step, _clone(cache), False, monkeypatch)
    fast = _forward(model, step, _clone(cache), True, monkeypatch)
    assert _mismatches(ref, fast) == 0
    for layer, rope in zip(inner.layers, originals):
        layer.self_attn.rope = rope
    monkeypatch.setattr(df, "enabled", lambda: True)
    assert df.run_layers(inner, h, _clone(cache), None, None) is not None


@requires_nax
def test_decode_fast_follows_weight_and_module_changes(monkeypatch):
    """The fused q/k/v buffer and the float32 router bias built on the first
    fast forward follow later weight loads."""
    model = _decode_model(seed=13)
    _per_row_router(monkeypatch, _load_patch_module())
    inner = model.model
    tokens = mx.random.randint(0, 512, (1, 40))
    cache = model.make_cache()
    _forward(model, tokens[:, :30], cache, False, monkeypatch)

    def check(step):
        ref = _forward(model, step, _clone(cache), False, monkeypatch)
        fast = _forward(model, step, _clone(cache), True, monkeypatch)
        assert _mismatches(ref, fast) == 0

    check(tokens[:, 30:31])  # arms the fast path
    attn = inner.layers[1].self_attn
    old_fused = attn.__dict__["_omlx_qkv"]
    attn.q_proj.weight = mx.array(np.array(attn.q_proj.weight)[::-1].copy())
    check(tokens[:, 31:33])
    assert attn.__dict__["_omlx_qkv"] is not old_fused
    gate = inner.layers[1].mlp.gate
    gate.e_score_correction_bias = gate.e_score_correction_bias + 0.05
    check(tokens[:, 33:34])
    assert gate.__dict__["_omlx_gate"].source is gate.e_score_correction_bias


def _mxfp4(e, n, k, seed):
    mx.random.seed(seed)
    w = (mx.random.normal((e, n, k)) * 0.05).astype(mx.bfloat16)
    return mx.quantize(w, group_size=32, bits=4, mode="mxfp4")


def _gather(x, w, sc, inds):
    return mx.gather_qmm(
        x,
        w,
        sc,
        None,
        rhs_indices=inds,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )


@requires_nax
@pytest.mark.parametrize("layout", ["split", "fused"])
def test_decode_expert_kernels_match_gather_qmm(layout):
    """gate/up + SwiGLU and down per (row, expert), bit-exact to gather_qmv."""
    E, D, F, rows = 12, 1024, 512, 3
    g, u, d = _mxfp4(E, F, D, 1), _mxfp4(E, F, D, 2), _mxfp4(E, D, F, 3)
    rng = np.random.default_rng(rows)
    inds = mx.array(
        np.stack([rng.choice(E, 8, replace=False) for _ in range(rows)])[None].astype(
            np.uint32
        )
    )
    mx.random.seed(rows)
    x = (mx.random.normal((1, rows, D)) * 2).astype(mx.bfloat16)
    xe = mx.expand_dims(x, (-2, -3))
    ref_act = swiglu(_gather(xe, *g, inds), _gather(xe, *u, inds))
    ref_y = _gather(ref_act, *d, inds).squeeze(-2)
    if layout == "fused":
        gu_w = mx.concatenate([g[0], u[0]], axis=1)
        gu_s = mx.concatenate([g[1], u[1]], axis=1)
        act = md.gate_up_swiglu(x, inds, gu_w, gu_s, gu_w, gu_s, n_out=F, up_offset=F)
    else:
        act = md.gate_up_swiglu(x, inds, g[0], g[1], u[0], u[1], n_out=F)
    y = md.down_proj(act, inds, d[0], d[1])
    assert _mismatches(act, ref_act.squeeze(-2)) == 0
    assert _mismatches(y, ref_y) == 0


def test_decode_expert_kind_rejects_wrapped_switch_modules():
    """An expert-offload style wrapper is not a stock SwitchGLU."""
    model = _decode_model(seed=14)
    sw = model.model.layers[1].mlp.switch_mlp

    class OffloadSwitchGLU(nn.Module):
        def __init__(self, glu):
            super().__init__()
            self.activation = glu.activation
            self.down_proj = glu.down_proj

    assert df._expert_kind(OffloadSwitchGLU(sw)) is None


# --- Long-context decode attention (omlx.patches.mimo_v2.sdpa_flash) ---


def _flash_inputs(B, H, Hk, L, S, mask_kind, with_sinks, seed):
    mx.random.seed(seed)
    q = (mx.random.normal((B, L, H, 192)) * 3).astype(mx.bfloat16).swapaxes(1, 2)
    # KV-cache views: the head stride exceeds the key count.
    k = mx.random.normal((B, Hk, S + 256, 192)).astype(mx.bfloat16)[:, :, :S]
    v = mx.random.normal((B, Hk, S + 256, 128)).astype(mx.bfloat16)[:, :, :S]
    sinks = mx.random.normal((H,)).astype(mx.bfloat16) if with_sinks else None
    if mask_kind == "none":
        mask = None
    elif mask_kind == "causal":
        mask = "causal"
    elif mask_kind == "window":
        mask = create_causal_mask(L, offset=S - L, window_size=S // 3)
    else:  # left-padded batch
        mask = create_causal_mask(L, offset=S - L, left_padding=mx.array([0, 700][:B]))
    return q, k, v, mask, sinks


def _mlx_decode_attention(q, k, v, scale, mask, sinks):
    """MLX's vector SDPA as the decode path runs it (B, L, H * Dv)."""
    B, H, L, _ = q.shape

    def sdpa(q, k, v, cache=None, scale=1.0, mask=None, sinks=None):
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask, sinks=sinks
        )

    rep = H // k.shape[1]
    if L * rep > 32:
        return df._sdpa_row_chunks(sdpa, q, k, v, None, scale, mask, sinks, 32 // rep)
    return (
        sdpa(q, k, v, scale=scale, mask=mask, sinks=sinks)
        .swapaxes(1, 2)
        .reshape(B, L, -1)
    )


@requires_nax
@pytest.mark.parametrize(
    "case",
    [
        (1, 64, 4, 1, 1100, "none", True),
        (1, 64, 4, 3, 1100, "causal", True),
        (1, 64, 4, 3, 17000, "window", True),
        (2, 64, 4, 3, 5000, "padded", False),
    ],
)
def test_sdpa_flash_matches_mlx_to_summation_order(case):
    """MLX's float32 attention in another summation order: within two bf16
    ULPs of MLX's kernel at each head vector's scale."""
    B, H, Hk, L, S, mask_kind, with_sinks = case
    q, k, v, mask, sinks = _flash_inputs(
        B, H, Hk, L, S, mask_kind, with_sinks, seed=L * S
    )
    scale = 192**-0.5
    out = sf.sdpa_flash(q, k, v, scale, mask, sinks)
    assert out is not None and out.shape == (B, L, H * 128)
    ref = np.array(
        _mlx_decode_attention(q, k, v, scale, mask, sinks).astype(mx.float32)
    )
    new = np.array(out.astype(mx.float32))
    assert np.isfinite(new).all()
    vec = np.abs(ref).reshape(B, L, H, 128).max(axis=-1, keepdims=True)
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(vec, 1e-30))) - 7)
    ulp = np.broadcast_to(ulp, (B, L, H, 128)).reshape(B, L, -1)
    assert (np.abs(new - ref) <= ulp * 2.0001).all()


@requires_nax
def test_sdpa_flash_verify_rows_equal_one_row_decodes():
    """A causal verify computes each row exactly like the one-row decode at
    that row's position."""
    S, L = 17000, 3
    q, k, v, _, sinks = _flash_inputs(1, 64, 4, L, S, "causal", True, seed=17)
    scale = 192**-0.5
    verify = sf.sdpa_flash(q, k, v, scale, "causal", sinks)
    rows = [
        sf.sdpa_flash(
            q[:, :, r : r + 1],
            k[:, :, : S - L + r + 1],
            v[:, :, : S - L + r + 1],
            scale,
            None,
            sinks,
        )
        for r in range(L)
    ]
    assert mx.array_equal(verify, mx.concatenate(rows, axis=1)).item()


def test_sdpa_flash_declines_unsupported_shapes():
    bf16 = mx.bfloat16
    k = mx.zeros((1, 4, 5000, 192), bf16)
    v = mx.zeros((1, 4, 5000, 128), bf16)
    assert (
        sf.sdpa_flash(mx.zeros((1, 12, 1, 192), bf16), k, v, 1.0, None, None) is None
    )  # GQA 3
    assert (
        sf.sdpa_flash(mx.zeros((1, 64, 5, 192), bf16), k, v, 1.0, "causal", None)
        is None
    )
    q32 = mx.zeros((1, 64, 1, 192), mx.float32)
    assert (
        sf.sdpa_flash(q32, k.astype(mx.float32), v.astype(mx.float32), 1.0, None, None)
        is None
    )


def _fp8_fused_qkv_sidecar(prefix, *, tp, n_h=4, n_kv=2, hd=32, vhd=24, cols=128):
    """A pre-sharded FP8 fused qkv whose rows say which shard/part they are (values chosen to be exact in e4m3)."""
    from omlx.patches.mimo_v2.fused_qkv_layout import (
        FUSED_QKV_BLOCK_SIZE,
        fused_qkv_part_rows,
        fused_qkv_shard_rows,
    )

    q_pr, k_pr, v_pr = fused_qkv_part_rows(n_h, n_kv, hd, vhd, tp)
    actual_pr, padded_pr = fused_qkv_shard_rows(n_h, n_kv, hd, vhd, tp)
    # On disk the shards are stored back to back without padding; only the
    # block scales are laid out on the per-shard padded grid.
    rows = []
    for shard in range(tp):
        rows.extend(
            [1.0 + shard] * q_pr + [10.0 + shard] * k_pr + [32.0 + 4 * shard] * v_pr
        )
    assert len(rows) == tp * actual_pr
    weight = mx.to_fp8(mx.array(rows, dtype=mx.float32)[:, None] * mx.ones((1, cols)))
    scale = mx.ones(
        (tp * padded_pr // FUSED_QKV_BLOCK_SIZE, cols // FUSED_QKV_BLOCK_SIZE)
    )
    sidecar = {f"{prefix}.weight": weight, f"{prefix}.weight_scale_inv": scale}
    return sidecar, (q_pr, k_pr, v_pr)


def _sanitized_qkv_rows(sidecar, monkeypatch):
    mimo_v2 = _load_patch_module()
    from omlx.patches.mlx_lm_mtp import set_mtp_active

    config = _minimal_config(
        num_nextn_predict_layers=1,
        omlx_mtp_sidecar="/models/mimo/mtp/model_mtp.safetensors",
    )
    set_mtp_active(True)
    try:
        model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(config))
    finally:
        set_mtp_active(False)
    monkeypatch.setattr(mimo_v2.mx, "load", lambda path: sidecar)
    out = model.sanitize({})
    prefix = "model.mtp.layers.0.self_attn"
    return {
        name: out[f"{prefix}.{name}_proj.weight"][:, 0].astype(mx.float32).tolist()
        for name in ("q", "k", "v")
    }


def test_fp8_sidecar_qkv_assumes_the_official_tp4_layout_when_main_is_split(
    monkeypatch,
):
    prefix = "model.mtp.layers.0.self_attn.qkv_proj"
    sidecar, (q_pr, k_pr, v_pr) = _fp8_fused_qkv_sidecar(prefix, tp=4)
    rows = _sanitized_qkv_rows(sidecar, monkeypatch)
    # Each projection is the concatenation of its per-shard parts, in order.
    assert rows["q"] == [1.0 + s for s in range(4) for _ in range(q_pr)]
    assert rows["k"] == [10.0 + s for s in range(4) for _ in range(k_pr)]
    assert rows["v"] == [32.0 + 4 * s for s in range(4) for _ in range(v_pr)]
