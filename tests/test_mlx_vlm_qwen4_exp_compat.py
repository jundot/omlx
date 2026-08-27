# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the pinned mlx-vlm Qwen4-Exp compatibility overlay."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

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


def test_external_ple_path_is_bounded_and_ssd_alias_resolves(tmp_path):
    compute = tmp_path / "compute"
    ple = tmp_path / "ple"
    compute.mkdir()
    ple.mkdir()
    (compute / "config.json").write_text(
        json.dumps(
            {
                "qwen4_exp_artifact": {
                    "ple_artifact": "../ple",
                    "ple_residency": "ssd_mmap",
                }
            }
        ),
        encoding="utf-8",
    )
    try:
        assert compat.configure_qwen4_exp_runtime(compute) == "mmap"
    finally:
        from mlx_vlm.models.qwen4_exp.language import configure_ple_runtime

        configure_ple_runtime(compute, mode="resident")
