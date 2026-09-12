# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the pinned mlx-vlm Qwen4-Exp compatibility overlay."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


def _tiny_config():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models import qwen4_exp

    text = qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    vision = qwen4_exp.VisionConfig(
        model_type="qwen4_exp",
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    return qwen4_exp.ModelConfig(
        text_config=text,
        vision_config=vision,
        model_type="qwen4_exp",
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
        vocab_size=64,
    )


def test_qwen4_exp_compat_registers_model_and_media_formatter():
    assert compat.apply_mlx_vlm_qwen4_exp_compat_patch() in {True, False}
    from mlx_vlm.models import qwen4_exp
    from mlx_vlm.prompt_utils import get_message_json

    assert qwen4_exp.ModelConfig is not None
    message = get_message_json(
        "qwen4_exp", "inspect", "user", num_images=1, skip_image_token=False
    )
    assert message["content"][0]["type"] == "image"


def test_qwen4_exp_config_normalizes_reference_layer_type():
    config = _tiny_config()
    assert config.text_config.layer_types == [
        "linear_attention",
        "qwen_sparse_attention",
    ]
    assert config.text_config.rope_parameters["type"] == "default"


@pytest.mark.parametrize("quantized", [False, True])
def test_qwen4_small_hyper_connection_fusion_fails_closed(quantized):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedResidual,
        compile_hyper_connections,
        fuse_hyper_connection_projections,
    )

    config = SimpleNamespace(
        hc_count=2,
        hidden_size=32,
        hc_lowrank=32,
        rms_norm_eps=1e-6,
    )
    mx.random.seed(17)
    module = Qwen4ExpGatedResidual(config)
    if quantized:
        module.input_mix_weight_down = module.input_mix_weight_down.to_quantized(
            32, 4
        )
        module.block_inject_weight = module.block_inject_weight.to_quantized(32, 4)

    inputs = mx.random.normal((2, 3, 64)).astype(mx.bfloat16)
    eager = module(inputs)
    verify_eager = module(inputs, target_verify=True)
    mx.eval(*eager, *verify_eager)

    assert fuse_hyper_connection_projections(module) == 0
    assert fuse_hyper_connection_projections(module) == 0
    fused = module(inputs)
    verify_fused = module(inputs, target_verify=True)
    mx.eval(*fused, *verify_fused)

    assert not hasattr(module, "input_inject_weight")
    assert hasattr(module, "input_mix_weight_down")
    assert hasattr(module, "block_inject_weight")
    for expected, actual in zip(eager, fused):
        assert mx.array_equal(expected, actual).item()
    for expected, actual in zip(verify_eager, verify_fused):
        assert mx.array_equal(expected, actual).item()

    assert compile_hyper_connections(module) == 1
    assert compile_hyper_connections(module) == 0
    prefill = module(inputs)
    decode_inputs = inputs[:1, :1]
    decode_eager = module._forward(decode_inputs)
    decode_compiled = module(decode_inputs)
    verify_compiled = module(inputs, target_verify=True)
    mx.eval(*prefill, *decode_eager, *decode_compiled, *verify_compiled)
    for expected, actual in zip(fused, prefill):
        assert mx.array_equal(expected, actual).item()
    for expected, actual in zip(decode_eager, decode_compiled):
        assert mx.array_equal(expected, actual).item()
    for expected, actual in zip(verify_fused, verify_compiled):
        assert mx.array_equal(expected, actual).item()

    compiled_forward = module._compiled_forward
    module._compiled_forward = MagicMock(
        side_effect=AssertionError("target verification entered compiled decode")
    )
    verify_decode = module(decode_inputs, target_verify=True)
    mx.eval(*verify_decode)
    module._compiled_forward.assert_not_called()
    module._compiled_forward = compiled_forward


def test_qwen4_hyper_connection_optimizations_fail_closed():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedResidual,
        compile_hyper_connections,
        fuse_hyper_connection_projections,
    )

    config = SimpleNamespace(
        hc_count=2,
        hidden_size=32,
        hc_lowrank=32,
        rms_norm_eps=1e-6,
    )
    incompatible = Qwen4ExpGatedResidual(config)
    incompatible.block_inject_weight = incompatible.block_inject_weight.to_quantized(
        32, 4
    )
    assert fuse_hyper_connection_projections(incompatible) == 0
    assert hasattr(incompatible, "input_mix_weight_down")
    assert hasattr(incompatible, "block_inject_weight")

    already_compiled = Qwen4ExpGatedResidual(config)
    assert compile_hyper_connections(already_compiled) == 1
    assert fuse_hyper_connection_projections(already_compiled) == 0


def test_qwen4_hyper_connection_compile_covers_uncombined_mixer():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpGatedResidual,
        compile_hyper_connections,
    )

    config = SimpleNamespace(
        hc_count=2,
        hidden_size=32,
        hc_lowrank=32,
        rms_norm_eps=1e-6,
    )
    mixer = Qwen4ExpGatedResidual(config, use_combine=False)
    inputs = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
    eager = mixer._forward(inputs)
    assert compile_hyper_connections(mixer) == 1
    compiled = mixer(inputs)
    mx.eval(eager, compiled)

    assert mx.allclose(eager, compiled, rtol=1e-5, atol=1e-6).item()


def test_qwen4_resident_ple_fuses_packed_shards_exactly():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx.nn as nn
    from mlx_vlm.models.qwen4_exp.language import ShardedEmbedding

    mx.random.seed(43)
    embedding = ShardedEmbedding(32, 64, 4)
    embedding.shards = [
        nn.QuantizedEmbedding.from_embedding(
            shard,
            group_size=32,
            bits=4,
            mode="affine",
        )
        for shard in embedding.shards
    ]
    indices = mx.array([[0, 9, 17, 31, 9]], dtype=mx.int32)
    expected = embedding(indices)
    mx.eval(expected)

    assert embedding.fuse_quantized_shards() is True
    assert embedding.fuse_quantized_shards() is False
    assert embedding.shards == []
    # The fused arm performs one device gather and no longer consults the host
    # shard boundaries after load.
    embedding.shard_offsets = ()
    actual = embedding(indices)
    mx.eval(actual)

    assert mx.array_equal(actual, expected).item()


def test_qwen4_single_packed_ple_aliases_without_join(monkeypatch):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx.nn as nn
    from mlx_vlm.models.qwen4_exp.language import ShardedEmbedding

    embedding = ShardedEmbedding(32, 64, 1)
    packed = nn.QuantizedEmbedding.from_embedding(
        embedding.shards[0], group_size=32, bits=4, mode="affine"
    )
    embedding.shards = [packed]
    embedding.weight_scale = mx.array([0.5], dtype=mx.bfloat16)
    indices = mx.array([[0, 31, 3, 3]], dtype=mx.int32)
    expected = embedding(indices)
    mx.eval(expected)

    def unexpected_join(*args, **kwargs):
        raise AssertionError("Single-shard fusion must not allocate a joined table")

    monkeypatch.setattr(mx, "concatenate", unexpected_join)
    assert embedding.fuse_quantized_shards() is True
    assert embedding.fused is packed
    assert embedding.shards == []
    assert embedding.fuse_quantized_shards() is False
    assert mx.array_equal(embedding(indices), expected).item()


@pytest.mark.parametrize(
    "memory_gib,mode,shards,quantized,expected",
    [
        (128, "resident", 1, True, 1),
        (128, "resident", 4, True, 0),
        (256, "resident", 4, True, 1),
        (128, "resident", 1, False, 0),
        (128, "mmap", 1, True, 0),
    ],
)
def test_qwen4_resident_ple_join_memory_admission(
    monkeypatch, memory_gib, mode, shards, quantized, expected
):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx.nn as nn
    from mlx_vlm.models.qwen4_exp import language

    embedding = language.ShardedEmbedding(32, 64, shards)
    if quantized:
        embedding.shards = [
            nn.QuantizedEmbedding.from_embedding(
                shard, group_size=32, bits=4, mode="affine"
            )
            for shard in embedding.shards
        ]
    layer = SimpleNamespace(
        ple=SimpleNamespace(ple_embedding=SimpleNamespace(ngram_embedding=embedding))
    )
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    sysconf = language.os.sysconf
    values = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": memory_gib * 2**30 // 4096}
    monkeypatch.setattr(language.os, "sysconf", lambda key: values[key] if key in values else sysconf(key))
    monkeypatch.setattr(language, "get_ple_runtime_mode", lambda: mode)

    assert language.fuse_resident_ple_embeddings(model) == expected
    assert (getattr(embedding, "fused", None) is not None) == bool(expected)


def test_qwen4_exp_load_enables_hyper_connection_optimizations(monkeypatch, caplog):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5 import Model as Qwen3_5Model
    from mlx_vlm.models.qwen4_exp import qwen4_exp as model_module

    model = model_module.Model.__new__(model_module.Model)
    nn.Module.__init__(model)
    base_load = MagicMock(return_value="loaded")
    fuse = MagicMock(return_value=96)
    fuse_ple = MagicMock(return_value=1)
    compile_connections = MagicMock(return_value=97)
    monkeypatch.setattr(Qwen3_5Model, "load_weights", base_load)
    monkeypatch.setattr(model_module, "fuse_hyper_connection_projections", fuse)
    monkeypatch.setattr(model_module, "fuse_resident_ple_embeddings", fuse_ple)
    monkeypatch.setattr(model_module, "compile_hyper_connections", compile_connections)
    monkeypatch.setattr(
        model_module,
        "get_mtp_runtime",
        MagicMock(return_value=SimpleNamespace(enabled=False)),
    )
    caplog.set_level("INFO", logger=model_module.__name__)

    weights = [("language_model.model.embed_tokens.weight", object())]
    result = model.load_weights(weights, strict=False)

    assert result == "loaded"
    base_load.assert_called_once_with(weights, strict=False)
    fuse.assert_called_once_with(model)
    fuse_ple.assert_called_once_with(model)
    compile_connections.assert_called_once_with(model)
    assert (
        "96 exact hybrid projection pairs, 97 compiled decode paths"
        in caplog.text
    )
    assert "Fused 1 resident Qwen4-Exp PLE table" in caplog.text


def test_qwen4_exp_load_skips_projection_fusion_during_mtp_verify(
    monkeypatch, caplog
):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5 import Model as Qwen3_5Model
    from mlx_vlm.models.qwen4_exp import qwen4_exp as model_module

    model = model_module.Model.__new__(model_module.Model)
    nn.Module.__init__(model)
    base_load = MagicMock(return_value=model)
    fuse = MagicMock(return_value=96)
    fuse_ple = MagicMock(return_value=1)
    compile_connections = MagicMock(return_value=100)
    monkeypatch.setattr(Qwen3_5Model, "load_weights", base_load)
    monkeypatch.setattr(model_module, "fuse_hyper_connection_projections", fuse)
    monkeypatch.setattr(model_module, "fuse_resident_ple_embeddings", fuse_ple)
    monkeypatch.setattr(model_module, "compile_hyper_connections", compile_connections)
    monkeypatch.setattr(
        model_module,
        "get_mtp_runtime",
        MagicMock(return_value=SimpleNamespace(enabled=True)),
    )
    caplog.set_level("INFO", logger=model_module.__name__)

    assert model.load_weights([], strict=False) is model

    fuse.assert_not_called()
    fuse_ple.assert_called_once_with(model)
    compile_connections.assert_called_once_with(model)
    assert "Skipped Qwen4-Exp exact hybrid projections" in caplog.text
    assert (
        "0 exact hybrid projection pairs, 100 compiled decode paths"
        in caplog.text
    )


def test_qwen4_exp_sanitize_keeps_converted_norm_values():
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    norm = mx.array([0.25, -0.5], dtype=mx.float32)
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(tie_word_embeddings=False, num_hidden_layers=0)
        )
    )
    result = Model.sanitize(
        model, {"model.language_model.norm.weight": norm}
    )
    assert mx.array_equal(result["language_model.model.norm.weight"], norm).item()


def _qwen4_rmsnorm_sanitize_fixture(*, include_mtp=False):
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpRMSNorm,
        Qwen4ExpRMSNormGated,
    )

    modules = []
    weights = {}
    target_keys = set()
    anchor_residuals = (
        -0.30,
        -0.20,
        -0.10,
        -0.05,
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.75,
    )
    for layer_idx, residual in enumerate(anchor_residuals):
        path = (
            f"language_model.model.layers.{layer_idx}."
            "attn_hyper_connection.hc_norm"
        )
        modules.append((path, Qwen4ExpRMSNorm(4)))
        key = f"{path}.weight"
        weights[key] = mx.full((4,), residual, dtype=mx.bfloat16)
        target_keys.add(key)

    path = "language_model.model.hyper_connection_mixer.hc_norm"
    modules.append((path, Qwen4ExpRMSNorm(4)))
    weights[f"{path}.weight"] = mx.array([-0.25, 0.0, 0.25, 0.5], dtype=mx.bfloat16)
    target_keys.add(f"{path}.weight")

    gated_path = "language_model.model.layers.0.linear_attn.norm"
    modules.append((gated_path, Qwen4ExpRMSNormGated(4, eps=1e-6, activation="silu")))
    weights[f"{gated_path}.weight"] = mx.array(
        [0.95, 1.0, 1.05, 1.1], dtype=mx.bfloat16
    )

    if include_mtp:
        for path, values in (
            (
                "mtp.hyper_connection_mixer.hc_norm",
                [3.75, 3.875, 4.0, 4.125],
            ),
            (
                "mtp.pre_fc_norm_embedding",
                [-0.8, -0.75, -0.7, -0.65],
            ),
        ):
            modules.append((path, Qwen4ExpRMSNorm(4)))
            weights[f"{path}.weight"] = mx.array(values, dtype=mx.bfloat16)
            target_keys.add(f"{path}.weight")

    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=0,
                num_experts=0,
                ple_layer_ids=(),
            )
        ),
        named_modules=lambda: iter(modules),
    )
    return model, modules, weights, target_keys


def test_qwen4_exp_sanitize_keeps_zero_centered_rmsnorms():
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    model, _, weights, target_keys = _qwen4_rmsnorm_sanitize_fixture()
    result = Model.sanitize(model, dict(weights))

    for key in target_keys:
        assert mx.array_equal(result[key], weights[key]).item()
        assert result[key].dtype == mx.bfloat16


def test_qwen4_exp_sanitize_recenters_ones_centered_base_and_mtp(tmp_path, caplog):
    from mlx_vlm.models.qwen4_exp.language import configure_mtp_runtime
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    configure_mtp_runtime(tmp_path, enabled=True)
    try:
        model, modules, canonical, target_keys = _qwen4_rmsnorm_sanitize_fixture(
            include_mtp=True
        )
        shifted = {
            key: (
                value + mx.array(1.0, dtype=value.dtype)
                if key in target_keys
                else value
            )
            for key, value in canonical.items()
        }

        with caplog.at_level("INFO"):
            result = Model.sanitize(model, dict(shifted))

        for key in target_keys:
            assert result[key].dtype == mx.float32
            assert mx.array_equal(
                1.0 + result[key], shifted[key].astype(mx.float32)
            ).item()

        gated_key = "language_model.model.layers.0.linear_attn.norm.weight"
        assert mx.array_equal(result[gated_key], shifted[gated_key]).item()
        assert "Canonicalized 15 ones-centered" in caplog.text

        mtp_norm = dict(modules)["mtp.pre_fc_norm_embedding"]
        mtp_norm.weight = result["mtp.pre_fc_norm_embedding.weight"]
        x = mx.array([[1.0, -2.0, 3.0, -4.0]], dtype=mx.float32)
        actual = mtp_norm(x)
        rms = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        expected = rms * shifted["mtp.pre_fc_norm_embedding.weight"].astype(mx.float32)
        assert mx.array_equal(actual, expected).item()

        second = Model.sanitize(model, dict(result))
        for key in target_keys:
            assert mx.array_equal(second[key], result[key]).item()
    finally:
        configure_mtp_runtime(tmp_path, enabled=False)


def test_qwen4_exp_sanitize_leaves_ambiguous_rmsnorms_unchanged(caplog):
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    model, _, weights, target_keys = _qwen4_rmsnorm_sanitize_fixture()
    for index, key in enumerate(
        sorted(key for key in target_keys if ".layers." in key)
    ):
        weights[key] = mx.full(
            weights[key].shape,
            0.0 if index < 6 else 1.0,
            dtype=mx.bfloat16,
        )

    with caplog.at_level("WARNING"):
        result = Model.sanitize(model, dict(weights))

    for key in target_keys:
        assert mx.array_equal(result[key], weights[key]).item()
    assert "RMSNorm checkpoint centering is ambiguous" in caplog.text


def test_qwen4_exp_sanitize_keeps_converted_ple_shared_scale():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    key = (
        "language_model.model.layers.1.ple.ple_embedding."
        "ngram_embedding.weight_scale"
    )
    scale = mx.array([0.0002], dtype=mx.bfloat16)
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=0,
                num_experts=0,
                ple_layer_ids=[2],
            )
        )
    )

    result = Model.sanitize(model, {key: scale})

    assert list(name for name in result if name.endswith("weight_scale")) == [key]
    assert mx.array_equal(result[key], scale).item()


def test_qwen4_exp_sanitize_adds_unit_ple_scale_for_bf16_checkpoint():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    key = (
        "language_model.model.layers.1.ple.ple_embedding."
        "ngram_embedding.weight_scale"
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=0,
                num_experts=0,
                ple_layer_ids=[2],
            )
        )
    )

    result = Model.sanitize(model, {})

    assert list(name for name in result if name.endswith("weight_scale")) == [key]
    assert mx.array_equal(
        result[key], mx.ones((1,), dtype=mx.bfloat16)
    ).item()


def test_qwen4_quantization_sanitize_keeps_mmap_ple_shards(tmp_path):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import configure_ple_runtime
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    config = SimpleNamespace(
        text_config=SimpleNamespace(
            tie_word_embeddings=False,
            num_hidden_layers=0,
            num_experts=0,
            ple_layer_ids=(),
        )
    )
    key = (
        "model.language_model.layers.1.ple.ple_embedding."
        "ngram_embedding.shard_0.weight"
    )
    base = key.removesuffix(".weight")
    weights = {
        key: mx.zeros((2, 20), dtype=mx.uint32),
        f"{base}.scales": mx.ones((2, 5), dtype=mx.bfloat16),
        f"{base}.biases": mx.zeros((2, 5), dtype=mx.bfloat16),
    }
    runtime_model = SimpleNamespace(config=config)
    quantization_proxy = SimpleNamespace(
        config=config,
        _omlx_preserve_qwen4_ple_for_quantization=True,
    )

    configure_ple_runtime(tmp_path, mode="mmap")
    try:
        assert Model.sanitize(runtime_model, dict(weights)) == {}
        sanitized = Model.sanitize(quantization_proxy, dict(weights))
    finally:
        configure_ple_runtime(tmp_path, mode="resident")

    sanitized_key = (
        "language_model.model.layers.1.ple.ple_embedding."
        "ngram_embedding.shards.0.weight"
    )
    assert sanitized_key in sanitized
    assert sanitized[sanitized_key].shape == (2, 20)
    assert sanitized_key.removesuffix(".weight") + ".scales" in sanitized
    assert sanitized_key.removesuffix(".weight") + ".biases" in sanitized


def test_qwen4_exp_tiny_text_prefill_and_decode():
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    cache = model.make_cache()
    logits = model(mx.array([[2, 3, 4]], dtype=mx.int32), cache=cache)
    next_logits = model(mx.array([[5]], dtype=mx.int32), cache=cache)
    mx.eval(logits.logits, next_logits.logits)
    assert logits.logits.shape == (1, 3, 64)
    assert next_logits.logits.shape == (1, 1, 64)


def test_qwen4_gathered_qsa_prefill_matches_official_mask_path(monkeypatch):
    config = _tiny_config()
    import mlx_vlm.models.qwen4_exp.language as language
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache, Qwen4ExpAttention

    attention = Qwen4ExpAttention(config.text_config)
    mx.eval(attention.parameters())
    hidden = mx.random.normal((1, 20, config.text_config.hidden_size))

    calls = []
    gathered = language.contiguous_causal_gathered_qsa

    def tracked(*args, **kwargs):
        calls.append((args[0].shape, args[1].shape))
        return gathered(*args, **kwargs)

    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa", tracked)
    fast_cache = QSAKVCache()
    actual = attention(hidden, mask="causal", cache=fast_cache)

    monkeypatch.setattr(
        Qwen4ExpAttention,
        "_gathered_text_prefill_eligible",
        staticmethod(lambda *args, **kwargs: False),
    )
    reference_cache = QSAKVCache()
    expected = attention(hidden, mask="causal", cache=reference_cache)
    mx.eval(actual, expected)

    assert calls == [((1, 4, 20, 8), (1, 2, 20, 8))]
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
    assert fast_cache.offset == reference_cache.offset == 20
    assert mx.array_equal(fast_cache.index_keys, reference_cache.index_keys).item()
    assert mx.array_equal(
        fast_cache.index_position_ids,
        reference_cache.index_position_ids,
    ).item()


def test_qwen4_rank_two_text_positions_match_identical_mrope_plane_logits(
    monkeypatch,
):
    """Canonical text positions preserve exact values and unlock gathered QSA."""
    config = _tiny_config()
    import mlx_vlm.models.qwen4_exp.language as language
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache, Qwen4ExpAttention

    mx.random.seed(831)
    attention = Qwen4ExpAttention(config.text_config)
    projection = mx.random.normal((config.text_config.hidden_size, 17))
    hidden = mx.random.normal((1, 20, config.text_config.hidden_size))
    text_positions = mx.arange(20, dtype=mx.int32)[None]
    identical_mrope = mx.broadcast_to(text_positions[None], (3, 1, 20))
    mx.eval(attention.parameters(), projection, hidden)

    assert attention._gathered_text_prefill_eligible(
        hidden,
        "causal",
        QSAKVCache(),
        text_positions,
        None,
        False,
    )
    assert attention._gathered_text_prefill_eligible(
        hidden,
        "causal",
        QSAKVCache(),
        identical_mrope,
        None,
        False,
    )
    unequal_mrope = mx.concatenate(
        [text_positions[None], text_positions[None], text_positions[None] + 1],
        axis=0,
    )
    assert not attention._gathered_text_prefill_eligible(
        hidden,
        "causal",
        QSAKVCache(),
        unequal_mrope,
        None,
        False,
    )

    gathered_calls = []
    original_gathered = language.contiguous_causal_gathered_qsa

    def tracked_gathered(*args, **kwargs):
        gathered_calls.append(True)
        return original_gathered(*args, **kwargs)

    monkeypatch.setattr(
        language,
        "contiguous_causal_gathered_qsa",
        tracked_gathered,
    )
    fast_cache = QSAKVCache()
    reference_cache = QSAKVCache()
    actual = attention(
        hidden,
        mask="causal",
        cache=fast_cache,
        position_ids=text_positions,
    )
    expected = attention(
        hidden,
        mask="causal",
        cache=reference_cache,
        position_ids=identical_mrope,
    )
    actual_logits = actual @ projection
    expected_logits = expected @ projection
    mx.eval(actual_logits, expected_logits)

    # Both the 2-D and the broadcast 3-D text positions take the gathered arm.
    assert gathered_calls == [True, True]
    assert mx.allclose(actual, expected, rtol=2e-5, atol=2e-5).item()
    assert mx.allclose(
        actual_logits,
        expected_logits,
        rtol=2e-5,
        atol=2e-5,
    ).item()
    assert mx.array_equal(
        mx.argmax(actual_logits, axis=-1),
        mx.argmax(expected_logits, axis=-1),
    ).item()
    assert fast_cache.offset == reference_cache.offset == 20
    fast_keys, fast_values, fast_index_keys, _ = fast_cache.state
    reference_keys, reference_values, reference_index_keys, _ = reference_cache.state
    for fast_value, reference_value in (
        (fast_keys, reference_keys),
        (fast_values, reference_values),
    ):
        assert mx.allclose(
            fast_value,
            reference_value,
            rtol=2e-5,
            atol=2e-5,
        ).item()
    assert mx.allclose(
        fast_index_keys,
        reference_index_keys,
        rtol=2e-5,
        atol=2e-5,
    ).item()


def test_qwen4_gathered_qsa_fails_closed_for_multimodal_positions(monkeypatch):
    config = _tiny_config()
    import mlx_vlm.models.qwen4_exp.language as language
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache, Qwen4ExpAttention

    def must_not_run(*args, **kwargs):
        raise AssertionError("multimodal positions must use mlx-vlm's general QSA")

    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa", must_not_run)
    attention = Qwen4ExpAttention(config.text_config)
    hidden = mx.random.normal((1, 3, config.text_config.hidden_size))
    position_ids = mx.array(
        [
            [[0, 1, 2]],
            [[0, 1, 2]],
            [[0, 0, 1]],
        ],
        dtype=mx.int32,
    )

    output = attention(
        hidden,
        mask="causal",
        cache=QSAKVCache(),
        position_ids=position_ids,
    )
    mx.eval(output)

    assert output.shape == hidden.shape


def test_qwen4_gathered_qsa_fails_closed_for_batched_prefill(monkeypatch):
    config = _tiny_config()
    import mlx_vlm.models.qwen4_exp.language as language
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache, Qwen4ExpAttention

    def must_not_run(*args, **kwargs):
        raise AssertionError("batched requests must use mlx-vlm's general QSA")

    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa", must_not_run)
    attention = Qwen4ExpAttention(config.text_config)
    hidden = mx.random.normal((2, 3, config.text_config.hidden_size))

    output = attention(hidden, mask="causal", cache=QSAKVCache())
    mx.eval(output)

    assert output.shape == hidden.shape


def test_qwen4_gathered_qsa_keeps_official_path_at_sparse_budget(monkeypatch):
    config = _tiny_config()
    import mlx_vlm.models.qwen4_exp.language as language
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache, Qwen4ExpAttention

    def must_not_run(*args, **kwargs):
        raise AssertionError("at-budget prefill must use the official full path")

    monkeypatch.setattr(language, "contiguous_causal_gathered_qsa", must_not_run)
    attention = Qwen4ExpAttention(config.text_config)
    budget = attention.indexer.token_budget
    hidden = mx.random.normal((1, budget, config.text_config.hidden_size))

    output = attention(hidden, mask="causal", cache=QSAKVCache())
    mx.eval(output)

    assert output.shape == hidden.shape


def test_qwen4_gathered_qsa_chunk_grows_with_context():
    _tiny_config()
    from mlx_vlm.models.qwen4_exp.qsa_fast import contiguous_causal_query_chunk

    assert contiguous_causal_query_chunk(4096) == 32
    assert contiguous_causal_query_chunk(4097) == 64
    assert contiguous_causal_query_chunk(16384) == 64
    assert contiguous_causal_query_chunk(16385) == 128


def test_qwen4_adapter_cache_only_prefill_skips_vocab_projection():
    from mlx_vlm.models.qwen4_exp import Model

    from omlx.models.vlm import VLMModelAdapter

    model = VLMModelAdapter(Model(_tiny_config()))
    cache = model.make_cache()
    result = model(
        mx.array([[2, 3, 4, 5]], dtype=mx.int32),
        cache=cache,
        skip_lm_head=True,
    )
    mx.eval([member.state for member in cache])

    assert result is None
    offsets = [member.offset for member in cache if hasattr(member, "offset")]
    assert offsets and max(offsets) == 4



def test_qwen4_batch_factory_honors_model_owned_cache_conversion():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, QSAKVCache

    import omlx.scheduler  # noqa: F401  (installs BatchGenerator cache patches)

    qsa_cache = QSAKVCache()
    qsa_cache.state = (
        mx.arange(24, dtype=mx.float32).reshape(1, 2, 3, 4),
        mx.arange(24, 48, dtype=mx.float32).reshape(1, 2, 3, 4),
        mx.arange(12, dtype=mx.float32).reshape(1, 3, 4),
        mx.array([[5, 6, 7]], dtype=mx.int32),
    )

    class Model:
        layers = (object(),)

        def make_cache(self):
            return [qsa_cache]

    generate = importlib.import_module("mlx_lm.generate")
    caches = generate._make_cache(Model(), [0], None)

    assert len(caches) == 1
    assert isinstance(caches[0], BatchQSAKVCache)
    mx.eval(caches[0].offset, caches[0].index_keys, caches[0].index_position_ids)
    assert caches[0].offset.tolist() == [3]
    assert caches[0].index_offset == 3
    assert mx.array_equal(caches[0].index_keys, qsa_cache.index_keys).item()
    assert mx.array_equal(
        caches[0].index_position_ids, qsa_cache.index_position_ids
    ).item()


def test_qwen4_fp8_ple_dequantizes_only_selected_rows():
    _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import ShardedEmbedding

    embedding = ShardedEmbedding(num_embeddings=4, dims=2, num_shards=2)
    embedding.shards[0].weight = mx.to_fp8(
        mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    )
    embedding.shards[1].weight = mx.to_fp8(
        mx.array([[5.0, 6.0], [7.0, 8.0]], dtype=mx.float32)
    )
    embedding.weight_scale = mx.array([0.25], dtype=mx.bfloat16)

    result = embedding(mx.array([[1, 2]], dtype=mx.int32))
    expected = mx.array([[[0.75, 1.0], [1.25, 1.5]]], dtype=mx.bfloat16)

    assert mx.array_equal(result, expected).item()


def test_qwen4_qsa_cache_round_trip_preserves_greedy_decode():
    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    from omlx.cache.type_registry import CacheTypeRegistry

    model = LanguageModel(config.text_config, config)
    full_cache = model.make_cache()
    prefix_cache = model.make_cache()

    full = model(mx.array([[2, 3, 4, 5]], dtype=mx.int32), cache=full_cache)
    model(mx.array([[2, 3, 4]], dtype=mx.int32), cache=prefix_cache)

    restored = []
    for cache in prefix_cache:
        handler = CacheTypeRegistry.get_handler_by_class_name(type(cache).__name__)
        state = handler.extract_state(cache)
        if type(cache).__name__ == "ArraysCache":
            restored.append(handler.reconstruct_cache(state, token_count=3))
        else:
            restored.append(handler.reconstruct_cache(state))

    resumed = model(mx.array([[5]], dtype=mx.int32), cache=restored)
    expected = mx.argmax(full.logits[:, -1], axis=-1)
    actual = mx.argmax(resumed.logits[:, -1], axis=-1)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()
    assert restored[1].index_keys.shape[1] == 4
    assert restored[1].index_position_ids.shape[-1] == 4


def test_qwen4_verify_matches_singleton_greedy_and_rolls_back_qsa():
    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    model = LanguageModel(config.text_config, config)
    verify_cache = model.make_cache()
    singleton_cache = model.make_cache()
    prefix = mx.array([[2, 3, 4]], dtype=mx.int32)
    model(prefix, cache=verify_cache)
    model(prefix, cache=singleton_cache)

    verified = model(
        mx.array([[5, 6]], dtype=mx.int32),
        cache=verify_cache,
        return_hidden=True,
    )
    first = model(mx.array([[5]], dtype=mx.int32), cache=singleton_cache)
    second = model(mx.array([[6]], dtype=mx.int32), cache=singleton_cache)
    singleton_logits = mx.concatenate([first.logits, second.logits], axis=1)
    verified_tokens = mx.argmax(verified.logits, axis=-1)
    singleton_tokens = mx.argmax(singleton_logits, axis=-1)
    mx.eval(verified_tokens, singleton_tokens)

    assert mx.array_equal(verified_tokens, singleton_tokens).item()
    assert verified.hidden_states[0].shape == (1, 2, 64)
    assert len(verified.gdn_states) == 1

    model.rollback_speculative_cache(
        verify_cache,
        verified.gdn_states,
        accepted=0,
        block_size=2,
    )
    qsa_cache = verify_cache[1]
    assert qsa_cache.offset == 4
    assert qsa_cache.index_keys.shape[1] == 4
    assert qsa_cache.index_position_ids.shape[-1] == 4


def _assert_ple_state_matches(actual_cache, expected_cache):
    mx.eval(
        actual_cache[2],
        actual_cache[3],
        expected_cache[2],
        expected_cache[3],
    )
    assert mx.array_equal(actual_cache[3], expected_cache[3]).item()
    assert mx.allclose(actual_cache[2], expected_cache[2], rtol=1e-3, atol=1e-3).item()


def test_qwen4_ple_rollback_keeps_only_committed_verify_prefix():
    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    model = LanguageModel(config.text_config, config)
    verify_cache = model.make_cache()
    replay_cache = model.make_cache()
    prefix = mx.array([[2, 3, 4]], dtype=mx.int32)
    confirmed = mx.array([[5]], dtype=mx.int32)
    draft = mx.array([[6]], dtype=mx.int32)
    next_token = mx.array([[7]], dtype=mx.int32)
    model(prefix, cache=verify_cache)
    model(prefix, cache=replay_cache)

    verified = model(
        mx.concatenate([confirmed, draft], axis=1),
        cache=verify_cache,
        return_hidden=True,
    )
    model(confirmed, cache=replay_cache)
    model.rollback_speculative_cache(
        verify_cache,
        verified.gdn_states,
        accepted=0,
        block_size=2,
    )

    _assert_ple_state_matches(verify_cache[0], replay_cache[0])
    rolled_back = model(next_token, cache=verify_cache)
    replayed = model(next_token, cache=replay_cache)
    mx.eval(rolled_back.logits, replayed.logits)
    assert mx.allclose(rolled_back.logits, replayed.logits, rtol=1e-3, atol=1e-3).item()


def test_qwen4_ple_partial_rollback_and_accept_match_sequential_replay():
    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    model = LanguageModel(config.text_config, config)
    prefix = mx.array([[2, 3, 4]], dtype=mx.int32)
    verify_tokens = mx.array([[5, 6, 7, 8]], dtype=mx.int32)

    accepted_cache = model.make_cache()
    sequential_cache = model.make_cache()
    model(prefix, cache=accepted_cache)
    model(prefix, cache=sequential_cache)
    model(verify_tokens, cache=accepted_cache, return_hidden=True)
    for token in (5, 6, 7, 8):
        model(mx.array([[token]], dtype=mx.int32), cache=sequential_cache)
    _assert_ple_state_matches(accepted_cache[0], sequential_cache[0])

    partial_cache = model.make_cache()
    replay_cache = model.make_cache()
    model(prefix, cache=partial_cache)
    model(prefix, cache=replay_cache)
    verified = model(verify_tokens, cache=partial_cache, return_hidden=True)
    for token in (5, 6, 7):
        model(mx.array([[token]], dtype=mx.int32), cache=replay_cache)
    model.rollback_speculative_cache(
        partial_cache,
        verified.gdn_states,
        accepted=2,
        block_size=4,
    )

    _assert_ple_state_matches(partial_cache[0], replay_cache[0])


def test_qwen4_ple_ordinary_forward_disarms_stale_snapshot():
    """A fully accepted verify cycle never calls rollback. The snapshot it armed
    must be dropped by the next ordinary forward so it cannot be mistaken for the
    current committed position by a later rollback."""
    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    model = LanguageModel(config.text_config, config)
    cache = model.make_cache()
    ple_cache = cache[0]
    model(mx.array([[2, 3, 4]], dtype=mx.int32), cache=cache)

    model(mx.array([[5, 6]], dtype=mx.int32), cache=cache, return_hidden=True)
    assert getattr(ple_cache, "_qwen4_exp_ple_speculative_state", None) is not None

    # ordinary decode forward (no verify): the stale snapshot must be gone
    model(mx.array([[7]], dtype=mx.int32), cache=cache)
    assert getattr(ple_cache, "_qwen4_exp_ple_speculative_state", None) is None


def test_qwen4_ple_rollback_validates_accepted_count_before_qsa_mutation():
    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    model = LanguageModel(config.text_config, config)
    cache = model.make_cache()
    model(mx.array([[2, 3, 4]], dtype=mx.int32), cache=cache)
    verified = model(
        mx.array([[5, 6]], dtype=mx.int32), cache=cache, return_hidden=True
    )
    before_offset = int(cache[1].offset)

    with pytest.raises(ValueError, match="outside the verify window"):
        model.rollback_speculative_cache(
            cache, verified.gdn_states, accepted=2, block_size=2
        )
    assert int(cache[1].offset) == before_offset
    assert getattr(cache[0], "_qwen4_exp_ple_speculative_state", None) is None


def test_qwen4_lightning_mtp_rejects_nextn_only_layout(tmp_path):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import configure_mtp_runtime

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {
                    "num_hidden_layers": 2,
                    "num_nextn_predict_layers": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.2.self_attn.q_proj.weight": "model.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )

    runtime = configure_mtp_runtime(tmp_path, enabled=True)
    try:
        assert runtime.enabled is False
        assert runtime.checkpoint_prefix is None
    finally:
        configure_mtp_runtime(tmp_path, enabled=False)


def test_qwen4_lightning_mtp_fusion_and_runtime_attachment(tmp_path):
    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpMTPModule,
        configure_mtp_runtime,
    )
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    configure_mtp_runtime(tmp_path, enabled=True)
    try:
        model = Model(config)
        assert isinstance(model.mtp, Qwen4ExpMTPModule)
        assert model.language_model.get_mtp_module() is model.mtp
        assert model.language_model._omlx_mtp_decode_enabled is True

        head = model.mtp
        head.fc_embedding.weight = mx.eye(config.text_config.hidden_size)
        head.fc_hidden.weight = mx.eye(config.text_config.hidden_size)
        embedding = mx.arange(1, 33, dtype=mx.float32).reshape(1, 1, 32)
        hidden = mx.arange(1, 65, dtype=mx.float32).reshape(1, 1, 64)
        actual = head.fuse_inputs(embedding, hidden)
        expected_embedding = embedding * mx.rsqrt(
            mx.mean(embedding * embedding, axis=-1, keepdims=True)
            + config.text_config.rms_norm_eps
        )
        expected_hidden = hidden * mx.rsqrt(
            mx.mean(hidden * hidden, axis=-1, keepdims=True)
            + config.text_config.rms_norm_eps
        )
        expected = (
            expected_embedding[..., None, :] + expected_hidden.reshape(1, 1, 2, 32)
        ).reshape(1, 1, 64)

        assert mx.allclose(actual, expected, atol=2e-5).item()
    finally:
        configure_mtp_runtime(tmp_path, enabled=False)


def test_qwen4_sanitize_dequantizes_and_stacks_fp8_experts(tmp_path):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import configure_mtp_runtime
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    configure_mtp_runtime(tmp_path, enabled=True)
    try:
        model = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(
                    tie_word_embeddings=False,
                    num_hidden_layers=1,
                    num_experts=4,
                )
            )
        )
        weights = {}
        for root in (
            "model.language_model.layers.0.mlp",
            "mtp.layers.0.mlp",
        ):
            for expert in range(4):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    key = f"{root}.experts.{expert}.{projection}.weight"
                    weights[key] = mx.to_fp8(mx.ones((2, 2), dtype=mx.float32))
                    weights[f"{key}_scale_inv"] = mx.ones((1, 1))

        result = Model.sanitize(model, weights)

        base_key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
        mtp_key = "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
        assert result[base_key].shape == (4, 2, 2)
        assert result[mtp_key].shape == (4, 2, 2)
        assert result[base_key].dtype == mx.bfloat16
        assert not any(key.endswith("weight_scale_inv") for key in result)
    finally:
        configure_mtp_runtime(tmp_path, enabled=False)


def test_disk_backed_bf16_ple_reads_only_requested_rows(tmp_path):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding

    prefix = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    )
    tensors = {
        f"{prefix}.shard_0.weight": mx.arange(16, dtype=mx.float32)
        .reshape(4, 4)
        .astype(mx.bfloat16),
        f"{prefix}.shard_1.weight": mx.arange(16, 32, dtype=mx.float32)
        .reshape(4, 4)
        .astype(mx.bfloat16),
    }
    filename = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(tmp_path / filename), tensors, metadata={"format": "mlx"})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: filename for key in tensors}}),
        encoding="utf-8",
    )

    embedding = DiskBackedShardedEmbedding(
        tmp_path, prefix, num_embeddings=8, dims=4, num_shards=2
    )
    values = embedding(mx.array([[1, 6]], dtype=mx.int32))
    mx.eval(values)
    expected = mx.stack([tensors[f"{prefix}.shard_0.weight"][1], tensors[f"{prefix}.shard_1.weight"][2]])[None]
    assert mx.array_equal(values, expected).item()
    assert embedding.last_touched_shards == (0, 1)
    assert embedding.rows_read == 2
    embedding.close()


def test_disk_bf16_upload_preserves_every_bit_pattern():
    import numpy as np
    from mlx_vlm.models.qwen4_exp.language import _SafeTensorMMap

    bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    uploaded = _SafeTensorMMap.to_mx(bits, "BF16")
    assert uploaded.dtype == mx.bfloat16
    assert np.array_equal(np.asarray(uploaded.view(mx.uint16)), bits)


@pytest.fixture
def early_gather_model(tmp_path):
    config = _tiny_config()
    config.text_config.num_hidden_layers = 3
    config.text_config.layer_types = ["linear_attention", "linear_attention", "full_attention"]
    config.text_config.ple_layer_ids = [2]
    from mlx_vlm.models.qwen4_exp.language import (
        DiskBackedShardedEmbedding,
        LanguageModel,
    )

    mx.random.seed(1789194451)
    model = LanguageModel(config.text_config, config)
    embedding = model.model.layers[1].ple.ple_embedding
    resident = embedding.ngram_embedding
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    tensors = {
        f"{prefix}.shard_{index}.weight": shard.weight.astype(mx.bfloat16)
        for index, shard in enumerate(resident.shards)
    }
    filename = "ple.safetensors"
    mx.save_safetensors(str(tmp_path / filename), tensors, metadata={"format": "mlx"})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: filename for name in tensors}})
    )
    disk = DiskBackedShardedEmbedding(
        tmp_path, prefix, resident.shard_offsets[-1], resident.dims, len(resident.shards)
    )
    embedding.ngram_embedding = disk
    mx.eval(model.parameters())
    try:
        yield model, embedding, disk
    finally:
        disk.close()


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_early_ple_gather_matches_late_gather_across_rollback(early_gather_model, monkeypatch, accepted):
    from mlx_vlm.models.qwen4_exp import language

    model, _, disk = early_gather_model
    early_cache, late_cache = model.make_cache(), model.make_cache()
    prefix = mx.array([[2, 1, 3]], dtype=mx.int32)
    verify_ids = mx.array([[4, 5, 1, 6]], dtype=mx.int32)
    outputs = []
    for enabled, cache in ((True, early_cache), (False, late_cache)):
        monkeypatch.setattr(language, "_PLE_EARLY_GATHER", enabled)
        model(prefix, cache=cache)
        verified = model(verify_ids, cache=cache, return_hidden=True)
        assert disk.last_prefetch_hit is enabled
        mx.eval(verified.logits)
        model.rollback_speculative_cache(cache, verified.gdn_states, accepted=accepted, block_size=4)
        resumed = model(mx.array([[7]], dtype=mx.int32), cache=cache)
        mx.eval(resumed.logits)
        outputs.append((verified.logits, resumed.logits))
    for component in (0, 1):
        assert mx.array_equal(outputs[0][component], outputs[1][component]).item()
    _assert_ple_state_matches(early_cache[1], late_cache[1])


def test_early_ple_gather_starts_before_first_layer_without_advancing_history(early_gather_model, monkeypatch):
    from mlx_vlm.models.qwen4_exp import language

    model, embedding, disk = early_gather_model
    cache = model.make_cache()
    monkeypatch.setattr(language, "_PLE_EARLY_GATHER", True)
    prefix = mx.array([[2, 3, 4]], dtype=mx.int32)
    mx.eval(model(prefix, cache=cache).logits)
    history = cache[1][3]
    next_ids = mx.array([[5, 1, 6]], dtype=mx.int32)
    prepared = embedding.prepare_indices(next_ids, cache[1])
    assert cache[1][3] is history
    assert disk._pending
    observed = []
    original = language.Qwen4ExpDecoderLayer.__call__

    def traced(layer, *args, **kwargs):
        if layer is model.model.layers[0]:
            observed.append(bool(disk._pending))
        return original(layer, *args, **kwargs)

    monkeypatch.setattr(language.Qwen4ExpDecoderLayer, "__call__", traced)
    mx.eval(model(next_ids, cache=cache).logits)
    assert prepared.shape == (1, 3, embedding.ngram_heads)
    assert observed == [True]
    assert disk.last_prefetch_hit
    assert not disk._pending


@pytest.fixture
def disk_ple_reader(tmp_path):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import language as ple

    # 166-byte rows exercise reads that straddle page boundaries.
    dense = (mx.arange(4096 * 83).reshape(4096, 83) / 3.0).astype(mx.bfloat16)
    path = tmp_path / "ple.safetensors"
    mx.save_safetensors(str(path), {"weight": dense})
    reader = ple._SafeTensorMMap(path)
    try:
        yield ple, reader, dense
    finally:
        reader.close()


@pytest.mark.parametrize("short_reads", [False, True])
def test_disk_backed_ple_page_prefetch_returns_identical_rows(
    disk_ple_reader, monkeypatch, short_reads
):
    """Read every required page once, retry short reads, and skip warm prefetch."""
    ple, reader, dense = disk_ple_reader
    page_size = ple._PLE_PAGE_SIZE
    row_bytes = 83 * 2
    base = reader._data_start + reader._header["weight"]["data_offsets"][0]
    crossing = next(
        row
        for row in range(4096)
        if (base + row * row_bytes) % page_size + row_bytes > page_size
    )
    indices = [crossing, crossing, 0, 1, 1000, 2000, 3000, 4000, 4095]
    expected = dense[mx.array(indices, dtype=mx.int32)]
    # Enumerate the byte spans independently of the prefetch endpoint formula.
    needed_pages = {
        offset // page_size
        for row in indices
        for offset in range(base + row * row_bytes, base + (row + 1) * row_bytes)
    }
    reads = []
    original_pread = ple.os.pread

    def record_pread(fd, size, offset):
        limit = min(size, page_size // 3) if short_reads else size
        data = original_pread(fd, limit, offset)
        reads.append((offset, len(data)))
        return data

    monkeypatch.setattr(ple.os, "pread", record_pread)
    assert mx.array_equal(reader.rows("weight", indices), expected).item()
    assert {offset // page_size for offset, _ in reads} == needed_pages
    file_size = reader.path.stat().st_size
    for page in needed_pages:
        spans = sorted(
            (offset, length)
            for offset, length in reads
            if offset // page_size == page and length
        )
        cursor = page * page_size
        for offset, length in spans:
            assert offset == cursor
            cursor += length
        assert cursor == min((page + 1) * page_size, file_size)
    assert all(reader._seen_pages[page] for page in needed_pages)

    # Fix the measured time so a busy test host cannot trigger reactivation.
    monkeypatch.setattr(ple, "time", SimpleNamespace(perf_counter=lambda: 0.0))
    reads.clear()
    assert mx.array_equal(reader.rows("weight", indices), expected).item()
    assert reads == []


@pytest.mark.parametrize("row_count", [0, 1, 8])
def test_disk_backed_ple_small_gathers_skip_prefetch(
    disk_ple_reader, monkeypatch, row_count
):
    _, reader, dense = disk_ple_reader
    prefetch = MagicMock(side_effect=AssertionError("Unexpected prefetch"))
    monkeypatch.setattr(reader, "_prefetch_missing_pages", prefetch)
    indices = list(range(row_count))
    expected = dense[mx.array(indices, dtype=mx.int32)]
    assert mx.array_equal(reader.rows("weight", indices), expected).item()
    prefetch.assert_not_called()


def test_disk_backed_ple_rearms_seen_bitmap_on_slow_gather(
    disk_ple_reader, monkeypatch, caplog
):
    """Control both clocks to verify the budget, retry, and interval boundary."""
    ple, reader, dense = disk_ple_reader
    indices = list(range(16))
    expected = dense[mx.array(indices, dtype=mx.int32)]
    pread = MagicMock(wraps=ple.os.pread)
    monkeypatch.setattr(ple.os, "pread", pread)
    budget = (
        ple._PLE_REARM_FLOOR_SECONDS + len(indices) * ple._PLE_REARM_PER_ROW_SECONDS
    )
    interval = ple._PLE_REARM_MIN_INTERVAL_SECONDS
    caplog.set_level("INFO", logger=ple.__name__)

    def gather(elapsed, now):
        ticks = iter((0.0, elapsed))
        monkeypatch.setattr(
            ple,
            "time",
            SimpleNamespace(perf_counter=lambda: next(ticks), monotonic=lambda: now),
        )
        values = reader.rows("weight", indices)
        assert mx.array_equal(values, expected).item()

    gather(0.0, 1000.0)
    assert pread.call_count > 0
    pread.reset_mock()
    gather(budget / 2, 1000.0)
    assert reader._rearm_count == 0
    assert any(reader._seen_pages)
    pread.assert_not_called()

    gather(budget * 2, 1000.0)
    assert reader._rearm_count == 1
    assert not any(reader._seen_pages)
    assert "re-armed seen-page bitmap" in caplog.text
    pread.assert_not_called()

    gather(0.0, 1001.0)
    assert pread.call_count > 0
    assert any(reader._seen_pages)
    pread.reset_mock()
    gather(budget * 2, 1000.0 + interval - 1.0)
    assert reader._rearm_count == 1
    assert any(reader._seen_pages)
    gather(budget * 2, 1000.0 + interval)
    assert reader._rearm_count == 2
    assert not any(reader._seen_pages)
    pread.assert_not_called()


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_disk_backed_affine_ple_supports_all_oq_bits(tmp_path, bits):
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding

    source_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    stored_prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
    tensors = {}
    expected_rows = []
    for shard_index in range(2):
        dense = (
            mx.arange(4 * 160, dtype=mx.float32).reshape(4, 160) / 97
            + shard_index * 10
        ).astype(mx.bfloat16)
        weight, scales, biases = mx.quantize(
            dense, group_size=32, bits=bits, mode="affine"
        )
        base = f"{stored_prefix}.shards.{shard_index}"
        tensors[f"{base}.weight"] = weight
        tensors[f"{base}.scales"] = scales
        tensors[f"{base}.biases"] = biases
        expected_rows.append(
            mx.dequantize(
                weight,
                scales,
                biases,
                group_size=32,
                bits=bits,
                mode="affine",
            )
        )

    filename = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(tmp_path / filename), tensors, metadata={"format": "mlx"})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: filename for key in tensors}}),
        encoding="utf-8",
    )

    embedding = DiskBackedShardedEmbedding(
        tmp_path,
        source_prefix,
        num_embeddings=8,
        dims=160,
        num_shards=2,
    )
    values = embedding(mx.array([[1, 6]], dtype=mx.int32))
    mx.eval(values)
    expected = mx.stack([expected_rows[0][1], expected_rows[1][2]])[None]

    assert mx.allclose(values, expected, atol=2e-2, rtol=2e-2).item()
    assert embedding.last_touched_shards == (0, 1)
    assert embedding.rows_read == 2
    assert embedding._shard_specs[0][3:] == (bits, 32)
    embedding.close()

# ---------------------------------------------------------------------------
# Continuous-batching join regressions (issue #3245, PR #3246)
# ---------------------------------------------------------------------------


def _warm_qsa_row(length: int, start: int, index_dim: int = 4):
    """Build a warm singleton QSAKVCache holding ``length`` cached tokens.

    Adapted from the fixture in PR #3215 (DiscoStew6082).
    """
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    cache = QSAKVCache()
    values = mx.arange(start, start + 2 * length * 4, dtype=mx.float32).reshape(
        1, 2, length, 4
    )
    cache.state = (
        values,
        values + 100,
        mx.arange(start, start + length * index_dim, dtype=mx.float32).reshape(
            1, length, index_dim
        ),
        mx.arange(start, start + length, dtype=mx.int32)[None],
    )
    return cache


def test_qwen4_cache_extension_promotes_singletons_to_model_owned_batch():
    """A warm QSA singleton joining a running batch must be promoted via the
    model-owned ``to_batch`` before ``extend`` — previously the join path
    raised AttributeError ('QSAKVCache' object has no attribute 'extend').

    Adapted from PR #3215 (DiscoStew6082), which fixes the same seam.
    """
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache

    import omlx.scheduler  # noqa: F401  (installs BatchGenerator cache patches)

    left = _warm_qsa_row(3, 10)
    right = _warm_qsa_row(1, 30)
    generate = importlib.import_module("mlx_lm.generate")

    caches = generate._extend_cache([left], [right])

    assert len(caches) == 1
    assert isinstance(caches[0], BatchQSAKVCache)
    mx.eval(caches[0].offset, caches[0].index_keys, caches[0].index_position_ids)
    assert caches[0].offset.tolist() == [3, 1]
    assert caches[0].index_offset == 3
    assert caches[0].extract(0).offset == 3
    assert caches[0].extract(1).offset == 1
    assert mx.array_equal(
        caches[0].extract(1).index_position_ids, right.index_position_ids
    ).item()


def test_qwen4_cache_extension_accepts_existing_model_owned_batch():
    """A singleton QSA row can join an existing model-owned batch.

    Adapted from PR #3214 (HaloFour).
    """
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache

    import omlx.scheduler  # noqa: F401  (installs BatchGenerator cache patches)

    left = _warm_qsa_row(3, 10)
    right_rows = [_warm_qsa_row(2, 30), _warm_qsa_row(1, 50)]
    right = BatchQSAKVCache.merge(right_rows)
    generate = importlib.import_module("mlx_lm.generate")

    caches = generate._extend_cache([left], [right])

    assert len(caches) == 1
    assert isinstance(caches[0], BatchQSAKVCache)
    mx.eval(caches[0].offset, caches[0].left_padding, caches[0].index_keys)
    assert caches[0].offset.tolist() == [3, 2, 1]
    assert caches[0].left_padding.tolist() == [0, 1, 2]
    assert [caches[0].extract(i).offset for i in range(3)] == [3, 2, 1]


def test_qwen4_cache_extension_keeps_existing_batch_in_place():
    """Extending two batched QSA caches must retain the left batch object.

    Adapted from PR #3214 (HaloFour).
    """
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache

    import omlx.scheduler  # noqa: F401  (installs BatchGenerator cache patches)

    left = BatchQSAKVCache.merge([_warm_qsa_row(3, 10)])
    right = BatchQSAKVCache.merge([_warm_qsa_row(2, 30)])
    generate = importlib.import_module("mlx_lm.generate")

    caches = generate._extend_cache([left], [right])

    assert caches[0] is left
    mx.eval(left.offset, left.left_padding, left.index_keys)
    assert left.offset.tolist() == [3, 2]
    assert left.left_padding.tolist() == [0, 1]
    assert [left.extract(i).offset for i in range(2)] == [3, 2]


def test_qwen4_qsa_indexer_handles_ragged_batch_offsets():
    """``from_projected`` on a batched cache whose ``offset`` is a per-row
    array must keep the mask math on aligned-column scalars — previously the
    (batch,) offsets broadcast into the seq axis and produced selected_tokens
    of shape (batch, batch, key_len), crashing mx.concatenate."""
    config = _tiny_config().text_config
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, Qwen4ExpQSAIndexer

    class _PassthroughRope:
        @staticmethod
        def apply_rotary(q, k, position_ids, unsqueeze_dim=1):
            return q, k

    indexer = Qwen4ExpQSAIndexer(config, _PassthroughRope())
    index_dim = config.indexer_head_dim

    batch = _warm_qsa_row(12, 0, index_dim=index_dim).to_batch([0])
    batch.extend(_warm_qsa_row(4, 100, index_dim=index_dim).to_batch([0]))
    assert isinstance(batch, BatchQSAKVCache)
    assert isinstance(batch.offset, mx.array)  # ragged per-row KV offsets
    assert batch.offset.tolist() == [12, 4]

    total = (config.indexer_n_heads + config.indexer_kv_heads) * index_dim
    qk = (mx.arange(2 * total, dtype=mx.float32) / total).reshape(2, 1, total)
    positions = mx.array([[12], [4]], dtype=mx.int32)

    selected = indexer.from_projected(qk, batch, positions)

    assert selected is not None
    mx.eval(selected)
    key_len = batch.index_offset
    assert key_len == 13
    assert selected.shape == (2, 1, 1, key_len)
    assert selected.dtype == mx.bool_


def test_qwen4_batch_qsa_trim_slices_indexer_arrays():
    """``BatchQSAKVCache.trim`` must slice the physical indexer arrays like the
    singleton trim does — previously only ``index_offset`` was decremented, so
    a later ``update_indexer`` resynced it to the stale physical width and the
    rejected draft columns fossilized inside the index."""
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()

    batch = _warm_qsa_row(6, 0).to_batch([0])
    assert batch.index_offset == 6

    trimmed = batch.trim(2)

    assert trimmed == 2
    assert batch.offset.tolist() == [4]
    assert batch.index_offset == 4
    assert batch.index_keys.shape[1] == 4
    assert batch.index_position_ids.shape[-1] == 4

    # The next indexer update must resync against the trimmed length, not the
    # stale physical width.
    batch.update_indexer(
        mx.zeros((1, 1, 4), dtype=mx.float32),
        mx.array([[4]], dtype=mx.int32),
    )
    assert batch.index_offset == 5
    assert batch.index_keys.shape[1] == 5
    assert batch.index_position_ids.shape[-1] == 5
def _make_bound_qwen4_language_model(config):
    from mlx_vlm.models.qwen4_exp.language import LanguageModel, Qwen4ExpMTPModule

    class MTPOwner:
        pass

    model = LanguageModel(config.text_config, config)
    owner = MTPOwner()
    owner.mtp = Qwen4ExpMTPModule(config.text_config)
    model.bind_mtp_owner(owner)
    return model, owner


def _assert_qwen4_lightning_mtp_hidden_width(model, text_config):
    expected_width = text_config.hc_count * text_config.hidden_size
    output = model(
        mx.array([[2, 3, 4]], dtype=mx.int32),
        cache=model.make_cache(),
        return_hidden=True,
    )
    hidden = output.hidden_states[-1]
    logits, head_hidden = model.mtp_forward(
        hidden[:, -1:],
        mx.array([[7]], dtype=mx.uint32),
        model.make_mtp_cache(),
        return_hidden=True,
        logits_keep=1,
    )
    mx.eval(hidden, logits, head_hidden)

    assert hidden.shape == (1, 3, expected_width)
    assert logits.shape == (1, 1, text_config.vocab_size)
    assert head_hidden.shape == (1, 1, expected_width)


def test_qwen4_lightning_mtp_isolated_from_dense_qwen35_runtime_patch():
    """Qwen3.5 patching must preserve resident and later Qwen4 MTP models."""
    from omlx.patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_runtime_patch

    config = _tiny_config()
    resident_model, resident_owner = _make_bound_qwen4_language_model(config)
    _assert_qwen4_lightning_mtp_hidden_width(resident_model, config.text_config)

    apply_mlx_vlm_mtp_runtime_patch()
    _assert_qwen4_lightning_mtp_hidden_width(resident_model, config.text_config)

    later_model, later_owner = _make_bound_qwen4_language_model(config)
    _assert_qwen4_lightning_mtp_hidden_width(later_model, config.text_config)

    assert resident_owner.mtp is not None
    assert later_owner.mtp is not None


@pytest.mark.parametrize("tied", [False, True])
def test_qwen4_dedicated_mtp_head_preserves_target_and_hidden(tied):
    """A proposal head may change logits, never the target or MTP cache math."""
    config = _tiny_config()
    config.text_config.tie_word_embeddings = tied
    config.text_config.mtp_use_dedicated_lm_head = True
    model, owner = _make_bound_qwen4_language_model(config)
    tokens = mx.array([[2, 3, 4]], dtype=mx.int32)
    before = model(tokens, cache=model.make_cache(), return_hidden=True)
    hidden = before.hidden_states[-1]
    next_ids = mx.array([[5, 6, 7]], dtype=mx.uint32)
    mixed, expected_hidden = owner.mtp(
        hidden, next_ids, model.model.embed_tokens, model.make_mtp_cache()
    )
    expected = owner.mtp.lm_head(mixed[:, -2:])
    actual, actual_hidden = model.mtp_forward(
        hidden, next_ids, model.make_mtp_cache(),
        return_hidden=True, logits_keep=2,
    )
    mx.eval(before.logits, expected, actual, expected_hidden, actual_hidden)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(actual_hidden, expected_hidden).item()

    # A conspicuous proposal-only change must remain invisible to target logits.
    owner.mtp.lm_head.weight = mx.zeros_like(owner.mtp.lm_head.weight)
    proposal = model.mtp_forward(
        hidden, next_ids, model.make_mtp_cache(), logits_keep=2
    )
    after = model(tokens, cache=model.make_cache(), return_hidden=True).logits
    mx.eval(proposal, after)
    assert mx.array_equal(proposal, mx.zeros_like(proposal)).item()
    assert not mx.array_equal(actual, proposal).item()
    assert mx.array_equal(before.logits, after).item()

    # A declared head must not silently fall back if its module disappears.
    del owner.mtp.lm_head
    with pytest.raises(AttributeError):
        model.mtp_forward(hidden, next_ids, model.make_mtp_cache())


def test_qwen4_default_mtp_still_uses_target_head():
    config = _tiny_config()
    model, owner = _make_bound_qwen4_language_model(config)
    assert not config.text_config.mtp_use_dedicated_lm_head
    assert not hasattr(owner.mtp, "lm_head")
    hidden = mx.ones((1, 1, 64))
    tokens = mx.array([[7]], dtype=mx.uint32)
    mixed, _ = owner.mtp(
        hidden, tokens, model.model.embed_tokens, model.make_mtp_cache()
    )
    expected = model.lm_head(mixed)
    actual = model.mtp_forward(hidden, tokens, model.make_mtp_cache())
    mx.eval(expected, actual)
    assert mx.array_equal(actual, expected).item()


def test_qwen4_quantized_dedicated_mtp_head_roundtrip(tmp_path):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpMTPModule

    config = _tiny_config()
    config.text_config.mtp_use_dedicated_lm_head = True
    head = Qwen4ExpMTPModule(config.text_config)
    nn.quantize(head, group_size=32, bits=4, class_predicate=lambda p, m: p == "lm_head")
    weights = dict(tree_flatten(head.parameters()))
    path = tmp_path / "mtp.safetensors"
    mx.save_safetensors(str(path), weights)
    restored = Qwen4ExpMTPModule(config.text_config)
    nn.quantize(restored, group_size=32, bits=4, class_predicate=lambda p, m: p == "lm_head")
    restored.load_weights(list(mx.load(str(path)).items()), strict=True)
    assert restored.lm_head.weight.dtype == mx.uint32
    inputs = mx.random.normal((1, 3, 32))
    before, after = head.lm_head(inputs), restored.lm_head(inputs)
    mx.eval(before, after)
    assert mx.array_equal(before, after).item()


def test_qwen4_declared_draft_head_requires_checkpoint_weights(tmp_path):
    config = _tiny_config()
    config.text_config.mtp_use_dedicated_lm_head = True
    from mlx_vlm.models.qwen4_exp.language import configure_mtp_runtime
    from mlx_vlm.models.qwen4_exp.qwen4_exp import Model

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc_hidden.weight": "model.safetensors"}})
    )
    configure_mtp_runtime(tmp_path, enabled=True)
    try:
        model = Model(config)
        with pytest.raises(ValueError, match="requires checkpoint weights"):
            model.sanitize({"mtp.fc_hidden.weight": mx.eye(32)})
        # Source-prefixed head keys are normalized before the requirement check.
        sanitized = model.sanitize({
            "model.language_model.mtp.lm_head.weight": mx.zeros((64, 32)),
        })
        assert "mtp.lm_head.weight" in sanitized
    finally:
        configure_mtp_runtime(tmp_path, enabled=False)

    # Turning MTP off still permits ordinary target-only serving of this file.
    model = Model(config)
    assert not hasattr(model, "mtp")
    sanitized = model.sanitize({"mtp.lm_head.weight": mx.zeros((64, 32))})
    assert "mtp.lm_head.weight" not in sanitized


def _disk_ple(tmp_path, *, shards, rows, dims, bits=None):
    """Write a sharded PLE table (affine-packed when ``bits`` is set, else dense bf16) and open it from disk."""
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding

    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    tensors, dense_rows = {}, []
    for shard_index in range(shards):
        dense = (mx.random.normal((rows, dims)) * (shard_index + 1)).astype(mx.bfloat16)
        base = f"{prefix}.shard_{shard_index}"
        if bits is None:
            tensors[f"{base}.weight"] = dense
            dense_rows.append(dense)
        else:
            weight, scales, biases = mx.quantize(dense, group_size=32, bits=bits, mode="affine")
            tensors[f"{base}.weight"], tensors[f"{base}.scales"], tensors[f"{base}.biases"] = weight, scales, biases
            dense_rows.append(mx.dequantize(weight, scales, biases, group_size=32, bits=bits, mode="affine").astype(mx.bfloat16))
    mx.eval(*tensors.values(), *dense_rows)
    filename = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(tmp_path / filename), tensors, metadata={"format": "mlx"})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: filename for key in tensors}}), encoding="utf-8")
    table = mx.concatenate(dense_rows)
    return DiskBackedShardedEmbedding(tmp_path, prefix, num_embeddings=shards * rows, dims=dims, num_shards=shards), table


@pytest.mark.parametrize("bits", [None, 4, 5])
def test_disk_backed_ple_gathers_a_chunk_with_one_upload_per_tensor(tmp_path, bits):
    """A chunk's rows are assembled on the host and uploaded once per tensor, not once per shard."""
    embedding, table = _disk_ple(tmp_path, shards=3, rows=8, dims=64, bits=bits)
    mx.random.seed(11)
    indices = mx.random.randint(0, 24, (2, 40)).astype(mx.int32)
    mx.eval(indices)
    values = embedding(indices)
    mx.eval(values)
    expected = mx.take(table, indices.reshape(-1), axis=0).reshape(2, 40, 64)
    assert values.dtype == mx.bfloat16
    assert mx.array_equal(values, expected).item()
    assert embedding.rows_read == 80
    assert embedding.last_touched_shards == (0, 1, 2)
    assert embedding.last_uploads == (1 if bits is None else 3)
    embedding.close()


def test_disk_backed_ple_rejects_out_of_range_indices(tmp_path):
    embedding, _ = _disk_ple(tmp_path, shards=2, rows=4, dims=32, bits=4)
    with pytest.raises(IndexError):
        embedding(mx.array([[1, 8]], dtype=mx.int32))
    embedding.close()


def test_small_ple_gather_reads_pages_in_parallel_across_shards(tmp_path, monkeypatch):
    import threading
    import time

    from mlx_vlm.models.qwen4_exp import language

    embedding, table = _disk_ple(tmp_path, shards=16, rows=128, dims=160)
    indices = mx.arange(16, dtype=mx.int32)[None] * 128
    original = language.os.pread
    lock = threading.Lock()
    counts = {"active": 0, "peak": 0, "calls": 0}

    def traced(*args):
        with lock:
            counts["active"] += 1
            counts["calls"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        try:
            time.sleep(0.005)  # Model independent SSD latency, not CPU work.
            return original(*args)
        finally:
            with lock:
                counts["active"] -= 1

    monkeypatch.setattr(language.os, "pread", traced)
    try:
        actual = embedding(indices)
        expected = mx.take(table, indices, axis=0)
        mx.eval(actual, expected)
        assert mx.array_equal(actual, expected).item()
        assert counts["peak"] > 1
        cold_calls = counts["calls"]
        mx.eval(embedding(indices))
        assert counts["calls"] == cold_calls
    finally:
        embedding.close()


def test_cross_shard_ple_read_failure_drains_other_reads(tmp_path, monkeypatch):
    import threading
    import time

    from mlx_vlm.models.qwen4_exp import language

    embedding, _ = _disk_ple(tmp_path, shards=16, rows=128, dims=160)
    original = language._SafeTensorMMap._touch_page
    lock = threading.Lock()
    counts = {"started": 0, "finished": 0}

    def failing(reader, page):
        with lock:
            counts["started"] += 1
            fail = counts["started"] == 1
        try:
            if fail:
                raise OSError("injected read failure")
            time.sleep(0.01)
            original(reader, page)
        finally:
            with lock:
                counts["finished"] += 1

    monkeypatch.setattr(language._SafeTensorMMap, "_touch_page", failing)
    try:
        with pytest.raises(OSError, match="injected read failure"):
            embedding(mx.arange(16, dtype=mx.int32)[None] * 128)
        assert counts["started"] > 8
        assert counts["started"] == counts["finished"]
    finally:
        embedding.close()


def test_disk_backed_ple_prefetch_serves_the_matching_call(tmp_path):
    """A prefetched chunk is assembled off the main thread and consumed by the next call with the same indices."""
    embedding, table = _disk_ple(tmp_path, shards=3, rows=8, dims=64, bits=5)
    first = mx.array([[3, 17, 5, 22, 3, 9]], dtype=mx.int32)
    other = mx.array([[1, 2]], dtype=mx.int32)
    embedding.prefetch(first)
    # The prefill loops announce chunk k+1 before chunk k gathers: both must stay pending until consumed.
    embedding.prefetch(other)
    values = embedding(first)
    mx.eval(values)
    assert embedding.last_prefetch_hit is True
    values_other = embedding(other)
    mx.eval(values_other)
    assert embedding.last_prefetch_hit is True
    assert mx.array_equal(values_other, mx.take(table, other.reshape(-1), axis=0)[None]).item()
    assert embedding(other) is not None and embedding.last_prefetch_hit is False
    values = embedding(first)
    mx.eval(values)
    assert embedding.last_prefetch_hit is False
    assert mx.array_equal(values, mx.take(table, first.reshape(-1), axis=0)[None]).item()
    values = embedding(other)
    mx.eval(values)
    assert embedding.last_prefetch_hit is False
    assert mx.array_equal(values, mx.take(table, other.reshape(-1), axis=0)[None]).item()
    embedding.close()


def test_prompt_loop_prefetches_the_next_chunk(monkeypatch):
    """While a chunk runs, the model is told the next chunk and the chunk it follows."""
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_lm.generate import PromptProcessingBatch

    seen, forwards = [], []

    class Model:
        def __call__(self, tokens, cache=None):
            forwards.append(tokens.tolist()[0])

        def prefetch_ple(self, next_ids, current_ids):
            seen.append((next_ids.tolist()[0], current_ids.tolist()[0]))

    batch = PromptProcessingBatch(Model(), uids=[0], caches=[[]], prefill_step_size=4)
    batch.prompt([list(range(10, 20))])
    assert forwards == [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19]]
    assert seen == [([14, 15, 16, 17], [10, 11, 12, 13]), ([18, 19], [14, 15, 16, 17])]


def test_prompt_loop_without_a_prefetch_hook_is_unchanged():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_lm.generate import PromptProcessingBatch

    forwards = []

    class Model:
        def __call__(self, tokens, cache=None):
            forwards.append(tokens.shape[1])

    PromptProcessingBatch(Model(), uids=[0], caches=[[]], prefill_step_size=3).prompt([list(range(7))])
    assert forwards == [3, 3, 1]


def test_ngram_prefetch_computes_the_next_chunks_indices():
    """prefetch(next, tail of current) hashes exactly the rows the next real call will gather."""
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpNGramEmbedding

    config = _tiny_config()
    config = getattr(config, "text_config", config)
    embedding = Qwen4ExpNGramEmbedding(config, config.ple_embed_dim, 1, 0)
    inner = embedding.ngram_embedding
    seen = {}

    class Recorder:
        def __call__(self, indices):
            seen["call"] = indices
            return inner(indices)

        def prefetch(self, indices):
            seen["prefetch"] = indices

    embedding.ngram_embedding = Recorder()
    cache = [None, None, None, None]
    chunk1 = mx.array([[5, 9, 2, 7, 1, 4, 4, 8]], dtype=mx.int64)
    chunk2 = mx.array([[3, 3, 6, 1, 9]], dtype=mx.int64)
    mx.eval(embedding(chunk1, cache))
    embedding.prefetch(chunk2, chunk1[:, -embedding.context_len :])
    mx.eval(embedding(chunk2, cache))
    assert seen["prefetch"].shape == seen["call"].shape
    assert mx.array_equal(seen["prefetch"], seen["call"]).item()


def test_prompt_lookahead_keeps_the_schedulers_mrope_hook(monkeypatch):
    """The scheduler wraps prompt() to set mRoPE deltas first; the lookahead loop must run under it, not over it."""
    import omlx.scheduler as scheduler  # installs the wrapper

    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_lm.generate import PromptProcessingBatch

    order = []
    monkeypatch.setattr(scheduler, "_prepare_mrope_prompt", lambda self: order.append("before"))

    class Model:
        def __call__(self, tokens, cache=None):
            order.append(("forward", tokens.shape[1]))

        def prefetch_ple(self, next_ids, current_ids):
            order.append(("prefetch", next_ids.shape[1]))

    PromptProcessingBatch(Model(), uids=[0], caches=[[]], prefill_step_size=4).prompt([list(range(6))])
    assert order == ["before", ("prefetch", 2), ("forward", 4), ("forward", 2)]


def test_disk_ple_cancels_displaced_queued_prefetches(tmp_path, monkeypatch):
    from threading import Event

    embedding, _ = _disk_ple(tmp_path, shards=3, rows=8, dims=64, bits=4)
    entered, release = Event(), Event()
    assemble = embedding._assemble

    def paused_assemble(host, plan):
        entered.set()
        assert release.wait(10)
        return assemble(host, plan)

    monkeypatch.setattr(embedding, "_assemble", paused_assemble)
    futures = []
    try:
        for index in range(10):
            embedding.prefetch(mx.array([[index]], dtype=mx.int32))
            futures.append(list(embedding._pending.values())[-1][1])
            if index == 0:
                assert entered.wait(10)
        assert len(embedding._pending) == 2
        assert futures[0].running()
        assert all(future.cancelled() for future in futures[1:8])
        assert sum(not future.done() for future in futures) == 3
    finally:
        release.set()
        embedding.close()


def test_disk_ple_close_drains_displaced_running_read(tmp_path, monkeypatch):
    from threading import Event, Thread

    embedding, _ = _disk_ple(tmp_path, shards=3, rows=8, dims=64, bits=4)
    entered, release, closing, closed = Event(), Event(), Event(), Event()
    assemble = embedding._assemble
    shutdown = embedding._prefetch_executor.shutdown
    readers = list(embedding._readers.values())
    errors = []

    def paused_assemble(host, plan):
        entered.set()
        assert release.wait(10)
        return assemble(host, plan)

    def observed_shutdown(*args, **kwargs):
        closing.set()
        return shutdown(*args, **kwargs)

    def close():
        try:
            embedding.close()
        except Exception as exc:
            errors.append(exc)
        finally:
            closed.set()

    monkeypatch.setattr(embedding, "_assemble", paused_assemble)
    monkeypatch.setattr(embedding._prefetch_executor, "shutdown", observed_shutdown)
    thread = Thread(target=close)
    try:
        embedding.prefetch(mx.array([[1]], dtype=mx.int32))
        active = next(iter(embedding._pending.values()))[1]
        assert entered.wait(10)
        embedding.prefetch(mx.array([[2]], dtype=mx.int32))
        embedding.prefetch(mx.array([[3]], dtype=mx.int32))
        assert all(future is not active for _, future in embedding._pending.values())
        thread.start()
        assert closing.wait(10)
        assert not closed.is_set()
        assert all(reader._mapping is not None for reader in readers)
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(timeout=10)
        else:
            embedding.close()
    assert closed.is_set() and not errors
    assert active.result(timeout=10)
    assert all(reader._mapping is None for reader in readers)
    assert not embedding._pending
    embedding.prefetch(mx.array([[4]], dtype=mx.int32))
    assert not embedding._pending
    embedding.close()
