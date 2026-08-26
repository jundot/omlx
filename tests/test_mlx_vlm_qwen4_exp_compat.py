# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the vendored Qwen4-Exp mlx-vlm runtime."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_qwen4_exp_compat_registers_model_type():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    assert apply_mlx_vlm_qwen4_exp_compat_patch() is True

    from mlx_vlm.utils import get_model_and_args

    module, model_type = get_model_and_args(
        {
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
        }
    )

    assert model_type == "qwen4_exp"
    assert module.__name__ == "mlx_vlm.models.qwen4_exp"


def test_qwen4_exp_config_preserves_runtime_specific_fields():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp import ModelConfig

    config = ModelConfig.from_dict(
        {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "shared_expert_intermediate_size": 32,
                "moe_intermediate_size": 16,
                "rms_norm_eps": 1e-6,
                "vocab_size": 128,
                "num_key_value_heads": 1,
                "max_position_embeddings": 1024,
                "head_dim": 16,
                "rope_parameters": {
                    "rope_type": "default",
                    "mrope_section": [2, 1, 1],
                    "rope_theta": 10000,
                    "partial_rotary_factor": 0.25,
                },
                "full_attention_interval": 4,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
                "hc_count": 4,
                "hc_lowrank": 8,
                "ple_layer_ids": [2],
                "ple_embed_dim": 64,
                "ple_conv_kernel_size": 4,
                "ngram_size": 3,
                "heads_per_ngram": 2,
                "ngram_vocab_size_base": 101,
                "make_ngram_vocab_size_divisible_by": 16,
                "split_ngram_parts": 4,
                "indexer_n_heads": 2,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 16,
                "indexer_budget": 8,
                "indexer_compress_ratio": 4,
                "output_gate_type": "sigmoid",
            },
            "vision_config": {
                "model_type": "qwen4_exp",
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "out_hidden_size": 64,
                "num_heads": 4,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
        }
    )

    assert config.text_config.hc_count == 4
    assert config.text_config.ple_layer_ids == [2]
    assert config.text_config.split_ngram_parts == 4
    assert config.text_config.indexer_budget == 8
    assert config.text_config.rope_parameters["type"] == "default"


def test_gated_residual_matches_hyper_connection_equations():
    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import GatedResidual

    args = SimpleNamespace(
        hc_count=2,
        hidden_size=4,
        hc_lowrank=2,
        rms_norm_eps=1e-6,
    )
    module = GatedResidual(args)
    module.input_mix_weight_down.weight = mx.zeros((2, 8))
    module.input_mix_weight_up.weight = mx.zeros((8, 2))
    module.block_inject_weight.weight = mx.zeros((2, 8))

    x = mx.arange(1, 17, dtype=mx.float32).reshape(1, 2, 8)
    mixed, hyper_input, injection = module(x)

    streams = x.reshape(1, 2, 2, 4)
    normalized = streams * mx.rsqrt(
        mx.mean(streams * streams, axis=-1, keepdims=True) + 1e-6
    )
    expected_mixed = 0.5 * mx.mean(normalized, axis=-2)

    assert mx.allclose(mixed, expected_mixed, atol=1e-6).item()
    assert mx.array_equal(hyper_input, x).item()
    assert mx.array_equal(injection, mx.ones((1, 2, 2))).item()


@pytest.mark.parametrize("quantized", [False, True])
def test_hyper_connection_input_and_injection_projection_fusion_is_bit_exact(
    quantized,
):
    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import (
        GatedResidual,
        compile_hyper_connections,
        fuse_hyper_connection_projections,
    )

    args = SimpleNamespace(
        hc_count=2,
        hidden_size=32,
        hc_lowrank=32,
        rms_norm_eps=1e-6,
    )
    mx.random.seed(17)
    module = GatedResidual(args)
    if quantized:
        module.input_mix_weight_down = module.input_mix_weight_down.to_quantized(32, 4)
        module.block_inject_weight = module.block_inject_weight.to_quantized(32, 4)
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    reference = module(x)
    mx.eval(*reference)

    assert fuse_hyper_connection_projections(module) == 1
    actual = module(x)
    mx.eval(*actual)

    assert hasattr(module, "input_inject_weight")
    assert not hasattr(module, "input_mix_weight_down")
    assert not hasattr(module, "block_inject_weight")
    for expected, observed in zip(reference, actual):
        assert mx.array_equal(expected, observed).item()

    assert compile_hyper_connections(module) == 1
    # Prefill stays eager because MLX shapeless compilation cannot infer the
    # symbolic reshape for a changing sequence dimension. Decode (T=1) uses
    # the compiled graph.
    prefill = module(x)
    decode_x = x[:, :1]
    decode_reference = module._forward(decode_x)
    compiled = module(decode_x)
    mx.eval(*prefill, *decode_reference, *compiled)
    for expected, observed in zip(reference, prefill):
        assert mx.array_equal(expected, observed).item()
    for expected, observed in zip(decode_reference, compiled):
        assert mx.array_equal(expected, observed).item()


def test_ngram_hasher_matches_transformers_and_resets_at_eos():
    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import NGramHasher

    args = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=2,
        vocab_size=128,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=4,
        seed=1234,
        eos_token_id=99,
    )
    hasher = NGramHasher(args, ple_layer_index=0)

    actual = hasher.compute_ids(mx.array([[10, 11, 12, 99, 13]], dtype=mx.int64))
    expected = mx.array(
        [
            [
                [3, 134, 213, 368],
                [63, 112, 262, 323],
                [68, 158, 221, 406],
                [69, 115, 255, 411],
                [42, 157, 236, 336],
            ]
        ],
        dtype=mx.int64,
    )

    assert hasher.total_vocab_size == 420
    assert hasher.padded_vocab_size == 432
    assert hasher.shard_rows == 108
    assert mx.array_equal(actual, expected).item()


def test_sharded_quantized_embedding_gathers_only_addressed_rows():
    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import ShardedQuantizedEmbedding

    table = mx.arange(384, dtype=mx.float32).reshape(12, 32) / 13
    module = ShardedQuantizedEmbedding(
        num_embeddings=12,
        dims=32,
        num_shards=4,
        group_size=32,
        bits=4,
    )
    for shard_index in range(4):
        start = shard_index * 3
        weight, scales, biases = mx.quantize(
            table[start : start + 3], group_size=32, bits=4
        )
        shard = getattr(module, f"shard_{shard_index}")
        shard.weight = weight
        shard.scales = scales
        shard.biases = biases

    ids = mx.array([[0, 4, 11, 7]], dtype=mx.int64)
    actual = module(ids)
    weight, scales, biases = mx.quantize(table, group_size=32, bits=4)
    reference = mx.dequantize(
        weight[ids],
        scales=scales[ids],
        biases=biases[ids],
        group_size=32,
        bits=4,
    )

    assert actual.shape == (1, 4, 32)
    assert mx.allclose(actual, reference, atol=1e-6).item()
    assert module.last_touched_shards == (0, 1, 2, 3)
    assert (
        module.shard_0.to_quantized(group_size=32, bits=4, mode="affine")
        is module.shard_0
    )


def test_disk_backed_quantized_embedding_reads_only_requested_rows(tmp_path):
    import json

    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import (
        DiskBackedQuantizedEmbedding,
    )

    prefix = "language_model.model.layers.1.ple.ple_embedding." "ngram_embedding"
    table = mx.arange(384, dtype=mx.float32).reshape(12, 32) / 13
    tensors = {}
    weight_map = {}
    weights = []
    scales = []
    biases = []
    filename = "model-00001-of-00001.safetensors"
    for shard_index in range(4):
        start = shard_index * 3
        weight, scale, bias = mx.quantize(
            table[start : start + 3], group_size=32, bits=4
        )
        weights.append(weight)
        scales.append(scale)
        biases.append(bias)
        for suffix, value in (
            ("weight", weight),
            ("scales", scale),
            ("biases", bias),
        ):
            key = f"{prefix}.shard_{shard_index}.{suffix}"
            tensors[key] = value
            weight_map[key] = filename

    mx.save_safetensors(tmp_path / filename, tensors, metadata={"format": "mlx"})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )

    module = DiskBackedQuantizedEmbedding(
        model_path=tmp_path,
        prefix=prefix,
        num_embeddings=12,
        dims=32,
        num_shards=4,
        group_size=32,
        bits=4,
    )
    ids = mx.array([[0, 4, 11, 7]], dtype=mx.int64)
    actual = module(ids)
    reference = mx.dequantize(
        mx.concatenate(weights, axis=0)[ids],
        scales=mx.concatenate(scales, axis=0)[ids],
        biases=mx.concatenate(biases, axis=0)[ids],
        group_size=32,
        bits=4,
    )
    mx.eval(actual, reference)

    assert actual.shape == (1, 4, 32)
    assert mx.allclose(actual, reference, atol=1e-6).item()
    assert module.last_touched_shards == (0, 1, 2, 3)
    assert module.rows_read == 12
    assert module.parameters() == {}


def test_ngram_embedding_cache_matches_single_prefill():
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import NGramEmbedding

    args = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=2,
        vocab_size=128,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=4,
        seed=1234,
        eos_token_id=99,
        ple_embed_dim=128,
    )
    module = NGramEmbedding(args, layer_idx=1, ple_layer_index=0)
    assert module.ngram_heads_vocab_sizes.tolist() == [101, 103, 107, 109]
    assert module.ngram_heads_offsets.tolist() == [0, 101, 204, 311]
    table = mx.arange(432 * 32, dtype=mx.float32).reshape(432, 32) / 97
    for shard_index in range(4):
        start = shard_index * 108
        weight, scales, biases = mx.quantize(
            table[start : start + 108], group_size=32, bits=4
        )
        shard = getattr(module.ngram_embedding, f"shard_{shard_index}")
        shard.weight = weight
        shard.scales = scales
        shard.biases = biases

    tokens = mx.array([[10, 11, 12, 99, 13]], dtype=mx.int64)
    full = module(tokens, cache=ArraysCache(size=4))

    split_cache = ArraysCache(size=4)
    first = module(tokens[:, :3], cache=split_cache)
    second = module(tokens[:, 3:], cache=split_cache)

    assert full.shape == (1, 5, 128)
    assert mx.allclose(mx.concatenate([first, second], axis=1), full, atol=1e-6).item()
    assert mx.array_equal(split_cache[3], mx.array([[99, 13]])).item()


def test_ple_layer_dilated_conv_cache_matches_single_prefill():
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import PLELayer

    args = SimpleNamespace(
        hidden_size=32,
        hc_count=2,
        rms_norm_eps=1e-6,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        vocab_size=128,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=4,
        seed=1234,
        eos_token_id=99,
        ple_embed_dim=128,
    )
    mx.random.seed(7)
    module = PLELayer(args, layer_idx=1, ple_layer_index=0)
    table = mx.arange(432 * 32, dtype=mx.float32).reshape(432, 32) / 97
    for shard_index in range(4):
        start = shard_index * 108
        weight, scales, biases = mx.quantize(
            table[start : start + 108], group_size=32, bits=4
        )
        shard = getattr(module.ple_embedding.ngram_embedding, f"shard_{shard_index}")
        shard.weight = weight
        shard.scales = scales
        shard.biases = biases
    module.key_proj.weight = mx.random.normal((64, 128)) * 0.01
    module.value_proj.weight = mx.random.normal((32, 128)) * 0.01
    module.conv1d.weight = mx.random.normal((64, 1, 4)) * 0.01

    tokens = mx.array([[10, 11, 12, 14, 15]], dtype=mx.int64)
    hidden = mx.random.normal((1, 5, 64))
    full = module(hidden, tokens, cache=ArraysCache(size=4))

    split_cache = ArraysCache(size=4)
    first = module(hidden[:, :3], tokens[:, :3], cache=split_cache)
    second = module(hidden[:, 3:], tokens[:, 3:], cache=split_cache)

    assert module.conv1d.weight.shape == (64, 1, 4)
    assert full.shape == hidden.shape
    assert mx.allclose(mx.concatenate([first, second], axis=1), full, atol=2e-5).item()
    assert split_cache[2].shape == (1, 9, 64)


def test_text_first_weight_filter_normalizes_native_mlx_ple_conv_layout():
    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.qwen4_exp import filter_text_first_weights

    native_mlx = mx.arange(32, dtype=mx.float32).reshape(8, 4, 1)
    [(name, normalized)] = filter_text_first_weights(
        [("language_model.model.layers.1.ple.conv1d.weight", native_mlx)]
    )

    assert name == "language_model.model.layers.1.ple.conv1d.weight"
    assert normalized.shape == (8, 1, 4)
    assert mx.array_equal(normalized, mx.moveaxis(native_mlx, 1, 2)).item()


def test_qsa_indexer_selects_best_complete_block_and_keeps_tail():
    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import QSAIndexer

    args = SimpleNamespace(
        hidden_size=4,
        indexer_n_heads=1,
        indexer_kv_heads=1,
        indexer_head_dim=2,
        indexer_budget=2,
        indexer_compress_ratio=2,
        rms_norm_eps=1e-6,
        head_dim=4,
        max_position_embeddings=128,
        rope_parameters={
            "partial_rotary_factor": 0.0,
            "rope_theta": 10000,
            "mrope_section": [0, 0, 0],
        },
    )
    indexer = QSAIndexer(args, layer_idx=3)
    query = mx.array([[1.0, 0.0]])
    raw_keys = mx.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    selected = indexer.select_token_indices(
        query,
        raw_keys,
        visible_length=7,
        query_position=6,
    )

    assert selected.tolist() == [0, 1, 6]


def test_tiny_qwen4_exp_language_model_prefill_and_decode():
    import mlx.core as mx

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp import TextConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    args = TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        eos_token_id=99,
        head_dim=16,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
        full_attention_interval=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[2],
        ple_embed_dim=128,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        output_gate_type="sigmoid",
    )
    outer_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=119,
    )
    model = LanguageModel(args, config=outer_config)
    cache = model.make_cache()

    prefill = model(mx.array([[10, 11, 12, 13]]), cache=cache)
    decode = model(mx.array([[14]]), cache=cache)
    mx.eval(prefill.logits, decode.logits)

    assert prefill.logits.shape == (1, 4, 128)
    assert decode.logits.shape == (1, 1, 128)
    assert mx.all(mx.isfinite(prefill.logits)).item()
    assert mx.all(mx.isfinite(decode.logits)).item()
    assert cache[3].offset == 5
    assert cache[3].indexer_offset == 5


def test_qwen4_exp_package_exports_complete_vlm_surface():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp import (
        LanguageModel,
        Model,
        ModelConfig,
        TextConfig,
        VisionConfig,
        VisionModel,
    )

    assert Model.__name__ == "Model"
    assert ModelConfig.__name__ == "ModelConfig"
    assert LanguageModel.__name__ == "LanguageModel"
    assert TextConfig.__name__ == "TextConfig"
    assert VisionModel.__name__ == "VisionModel"
    assert VisionConfig.__name__ == "VisionConfig"


def test_qwen4_exp_preload_dispatch_applies_vlm_compat(tmp_path, monkeypatch):
    import json

    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "text_config": {"model_type": "qwen4_exp_text"},
                "vision_config": {"model_type": "qwen4_exp"},
            }
        )
    )
    apply_patch = MagicMock(return_value=True)
    configure_runtime = MagicMock(return_value="resident")
    monkeypatch.setattr(compat, "apply_mlx_vlm_qwen4_exp_compat_patch", apply_patch)
    monkeypatch.setattr(compat, "configure_qwen4_exp_runtime", configure_runtime)

    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)

    apply_patch.assert_called_once_with()
    configure_runtime.assert_called_once_with(str(tmp_path))


def test_qwen4_exp_auto_ple_mode_uses_4bit_checkpoint_as_ram_boundary():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.language import resolve_ple_runtime_mode

    physical_memory = 128 * (1 << 30)

    assert (
        resolve_ple_runtime_mode(
            "auto", checkpoint_bytes=68 * (1 << 30), physical_memory=physical_memory
        )
        == "resident"
    )
    assert (
        resolve_ple_runtime_mode(
            "auto",
            checkpoint_bytes=104 * (1 << 30),
            physical_memory=physical_memory,
        )
        == "mmap"
    )
    assert (
        resolve_ple_runtime_mode(
            "mmap", checkpoint_bytes=1, physical_memory=physical_memory
        )
        == "mmap"
    )


def test_text_first_weight_filter_drops_deferred_towers_and_optional_ple():
    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen4_exp.qwen4_exp import filter_text_first_weights

    weights = [
        ("language_model.model.embed_tokens.weight", object()),
        (
            "language_model.model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight",
            object(),
        ),
        ("mtp.fc_embedding.weight", object()),
        ("mtp.layers.0.self_attn.q_proj.weight", object()),
        ("vision_tower.blocks.0.attn.qkv.weight", object()),
    ]

    assert [name for name, _ in filter_text_first_weights(weights)] == [
        "language_model.model.embed_tokens.weight",
        (
            "language_model.model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight"
        ),
    ]
    assert [name for name, _ in filter_text_first_weights(weights, drop_ple=True)] == [
        "language_model.model.embed_tokens.weight",
    ]


def test_qwen4_exp_load_enables_hyper_connection_optimizations(monkeypatch):
    import mlx.nn as nn

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )

    apply_mlx_vlm_qwen4_exp_compat_patch()

    from mlx_vlm.models.qwen3_5 import Model as Qwen3_5Model
    from mlx_vlm.models.qwen4_exp import qwen4_exp as model_module

    model = model_module.Model.__new__(model_module.Model)
    nn.Module.__init__(model)
    model._disk_backed_ple = False
    base_load = MagicMock(return_value="loaded")
    fuse = MagicMock(return_value=96)
    compile_connections = MagicMock(return_value=97)
    monkeypatch.setattr(Qwen3_5Model, "load_weights", base_load)
    monkeypatch.setattr(model_module, "fuse_hyper_connection_projections", fuse)
    monkeypatch.setattr(model_module, "compile_hyper_connections", compile_connections)

    result = model.load_weights(
        [("language_model.model.embed_tokens.weight", object())]
    )

    assert result == "loaded"
    base_load.assert_called_once()
    fuse.assert_called_once_with(model)
    compile_connections.assert_called_once_with(model)
