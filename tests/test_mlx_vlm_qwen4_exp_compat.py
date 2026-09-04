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
    assert not attention._gathered_text_prefill_eligible(
        hidden,
        "causal",
        QSAKVCache(),
        identical_mrope,
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

    assert gathered_calls == [True]
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
