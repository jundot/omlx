# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLX-independent fixed KV-cache memory planner."""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

import omlx.fixed_kv_memory as fixed_memory
from omlx.fixed_kv_memory import (
    CacheTensorDescriptor,
    FixedKVPlanningError,
    build_cache_manifest,
    estimate_cache_tensors_from_config,
    estimate_model_memory,
)


def _model(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def _generic_config(**updates):
    config = {
        "model_type": "test_gqa",
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 64,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
    }
    config.update(updates)
    return config


def test_tensor_descriptor_to_dict_is_stable_and_json_safe():
    tensor = CacheTensorDescriptor(
        path="layers.0.keys",
        cache_kind="KVCache",
        role="keys",
        shape=(1, 2, 256, 8),
        dtype="float16",
        dtype_bytes=2,
        nbytes=8192,
        logical_tokens=17,
        physical_tokens=256,
        capacity_kind="linear",
        provenance="test",
    )

    payload = tensor.to_dict()

    assert list(payload) == [
        "schema_version",
        "path",
        "cache_kind",
        "role",
        "shape",
        "dtype",
        "dtype_bytes",
        "bytes",
        "logical_tokens",
        "physical_tokens",
        "capacity_kind",
        "provenance",
        "notes",
    ]
    assert payload["shape"] == [1, 2, 256, 8]
    json.dumps(payload)


def test_generic_gqa_rounds_each_linear_cache_to_256_tokens(tmp_path):
    model = _model(tmp_path, _generic_config())

    plan = estimate_model_memory(
        model,
        257,
        weights_bytes=1_000,
        requested_session_slots=1,
        available_memory_bytes=10**9,
    )

    # 2 layers × K/V × 2 KV heads × 512 physical tokens × 8 dims × 2 bytes.
    assert plan.per_session_kv_bytes == 65_536
    assert len(plan.cache_tensors) == 4
    assert {tensor.physical_tokens for tensor in plan.cache_tensors} == {512}
    assert {tensor.shape[1] for tensor in plan.cache_tensors} == {2}


def test_generic_mha_defaults_kv_heads_to_attention_heads():
    tensors = estimate_cache_tensors_from_config(
        _generic_config(num_hidden_layers=1, num_key_value_heads=None),
        1,
    )

    assert tensors[0].shape == (1, 8, 256, 8)


def test_native_mtp_full_attention_cache_is_part_of_every_session(tmp_path):
    model = _model(
        tmp_path,
        _generic_config(num_hidden_layers=2, mtp_num_hidden_layers=1),
    )

    ordinary = estimate_model_memory(
        model,
        257,
        weights_bytes=0,
        requested_session_slots=2,
        available_memory_bytes=10**9,
    )
    mtp = estimate_model_memory(
        model,
        257,
        weights_bytes=0,
        requested_session_slots=2,
        available_memory_bytes=10**9,
        native_mtp_enabled=True,
    )

    assert mtp.native_mtp_kv_bytes_per_session == ordinary.per_session_kv_bytes // 2
    assert mtp.per_session_kv_bytes == ordinary.per_session_kv_bytes * 3 // 2
    assert {tensor.path for tensor in mtp.cache_tensors if tensor.path.startswith("mtp.")} == {
        "mtp.layers.0.keys",
        "mtp.layers.0.values",
    }


def test_deepseek_dspark_auxiliary_rings_are_in_fixed_plan(tmp_path):
    model = _model(
        tmp_path,
        {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 1,
            "head_dim": 4,
            "index_head_dim": 2,
            "sliding_window": 8,
            "compress_ratios": [0],
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [40, 41, 42],
            "num_nextn_predict_layers": 1,
            "torch_dtype": "float16",
            "max_position_embeddings": 4096,
        },
    )

    plan = estimate_model_memory(
        model,
        16,
        weights_bytes=0,
        requested_session_slots=2,
        available_memory_bytes=10**9,
        native_mtp_enabled=True,
    )

    dspark = [
        tensor
        for tensor in plan.cache_tensors
        if tensor.cache_kind == "DSparkContextCache"
    ]
    assert len(dspark) == 3
    assert all(tensor.shape == (1, 1, 8, 4) for tensor in dspark)
    assert plan.native_mtp_kv_bytes_per_session == 3 * 8 * 4 * 2
    assert plan.to_dict()["native_mtp_kv_bytes_per_session"] == 192


def test_deepseek_dspark_explicit_stage_count_overrides_target_count(tmp_path):
    model = _model(
        tmp_path,
        {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 1,
            "head_dim": 4,
            "index_head_dim": 2,
            "sliding_window": 8,
            "compress_ratios": [0],
            "dspark_block_size": 5,
            "dspark_target_layer_ids": [40, 41, 42],
            "n_mtp_layers": 2,
            "torch_dtype": "float16",
            "max_position_embeddings": 4096,
        },
    )

    plan = estimate_model_memory(
        model,
        16,
        weights_bytes=0,
        requested_session_slots=1,
        available_memory_bytes=10**9,
        native_mtp_enabled=True,
    )

    dspark = [
        tensor
        for tensor in plan.cache_tensors
        if tensor.cache_kind == "DSparkContextCache"
    ]
    assert [tensor.path for tensor in dspark] == [
        "mtp.layers.0.keys",
        "mtp.layers.1.keys",
    ]


def test_session_concurrency_is_capped_by_binding_memory(tmp_path):
    model = _model(tmp_path, _generic_config(num_hidden_layers=1))
    per_session = 32_768
    base = 1_000 + 200

    plan = estimate_model_memory(
        model,
        257,
        weights_bytes=1_000,
        other_fixed_bytes=200,
        requested_session_slots=5,
        available_memory_bytes=base + 4 * per_session,
        memory_ceiling_bytes=base + 3 * per_session + 10,
    )

    assert plan.per_session_kv_bytes == per_session
    assert plan.max_feasible_session_slots == 2
    assert plan.requested_session_slots == 5
    assert plan.reserved_session_slots == 2
    assert plan.fixed_kv_cache_bytes == 2 * per_session
    assert plan.pool_scratch_bytes == per_session
    assert plan.binding_memory_source == (
        "caller supplied Metal/admission memory ceiling"
    )
    assert plan.configured_concurrency_capped is True
    assert plan.fits is True
    assert plan.requested_configuration_fits is False
    assert plan.fit_reason is not None
    assert "capped to 2" in plan.fit_reason
    assert "Lower the context window" in plan.fit_reason


def test_million_token_pool_caps_eight_requested_sessions_to_four(tmp_path):
    model = _model(
        tmp_path,
        _generic_config(
            num_hidden_layers=42,
            num_attention_heads=32,
            num_key_value_heads=2,
            hidden_size=4096,
            max_position_embeddings=1_048_576,
        ),
    )

    plan = estimate_model_memory(
        model,
        1_000_000,
        weights_bytes=32 * 2**30,
        requested_session_slots=8,
        available_memory_bytes=220 * 2**30,
    )

    assert plan.context_window == 1_000_000
    assert plan.requested_session_slots == 8
    assert plan.max_feasible_session_slots == 4
    assert plan.reserved_session_slots == 4
    assert plan.configured_concurrency_capped is True
    assert plan.requested_configuration_fits is False
    assert plan.fits is True
    assert plan.fit_reason is not None
    assert "Requested concurrency 8" in plan.fit_reason
    assert "capped to 4" in plan.fit_reason


def test_no_session_fit_returns_actionable_reason(tmp_path):
    model = _model(tmp_path, _generic_config(num_hidden_layers=1))

    plan = estimate_model_memory(
        model,
        256,
        weights_bytes=10_000,
        other_fixed_bytes=1_000,
        requested_session_slots=2,
        available_memory_bytes=10_999,
    )

    assert plan.max_feasible_session_slots == 0
    assert plan.reserved_session_slots == 0
    assert plan.fixed_kv_cache_bytes == 0
    assert plan.fits is False
    assert plan.configured_concurrency_capped is True
    assert plan.fit_reason is not None
    assert "No session slot fits" in plan.fit_reason
    assert "smaller model" in plan.fit_reason


def test_fit_is_unknown_without_a_memory_measurement(tmp_path, monkeypatch):
    model = _model(tmp_path, _generic_config(num_hidden_layers=1))
    monkeypatch.setattr(
        fixed_memory,
        "_detect_system_memory",
        lambda _available: fixed_memory._SystemMemory(None, None, None, None),
    )

    plan = estimate_model_memory(
        model,
        256,
        weights_bytes=1,
        requested_session_slots=3,
    )

    assert plan.reserved_session_slots == 3
    assert plan.max_feasible_session_slots is None
    assert plan.fits is None
    assert plan.fit_reason is not None
    assert "fit is unknown" in plan.fit_reason


def test_model_plan_to_dict_contains_session_contract(tmp_path):
    plan = estimate_model_memory(
        _model(tmp_path, _generic_config(num_hidden_layers=1)),
        256,
        weights_bytes=100,
        requested_session_slots=2,
        available_memory_bytes=10**9,
        other_fixed_bytes=50,
    )

    payload = plan.to_dict()

    assert payload["context_window"] == 256
    assert payload["per_session_kv_bytes"] == plan.per_session_kv_bytes
    assert payload["fixed_kv_cache_bytes"] == 2 * plan.per_session_kv_bytes
    assert payload["pool_scratch_bytes"] == plan.per_session_kv_bytes
    assert payload["other_fixed_bytes"] == 50 + plan.pool_scratch_bytes
    assert payload["estimated_total_bytes"] == (
        100 + payload["other_fixed_bytes"] + payload["fixed_kv_cache_bytes"]
    )
    assert payload["cache_tensors"][0]["schema_version"] == 1
    assert set(field.name for field in fields(type(plan))) >= {
        "requested_session_slots",
        "reserved_session_slots",
        "max_feasible_session_slots",
    }
    json.dumps(payload)


def test_deepseek_v3_uses_latent_mla_layout_at_200k():
    config = {
        "model_type": "deepseek_v3",
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 200_192,
        "torch_dtype": "float16",
    }

    tensors = estimate_cache_tensors_from_config(config, 200_000)

    assert sum(tensor.nbytes for tensor in tensors) == 14_067_892_224
    assert tensors[0].shape == (1, 1, 200_192, 512)
    assert tensors[1].shape == (1, 1, 200_192, 64)


def test_dsa_manifest_includes_index_key_and_zero_width_value():
    config = {
        "model_type": "deepseek_v32",
        "num_hidden_layers": 2,
        "kv_lora_rank": 16,
        "qk_rope_head_dim": 4,
        "index_head_dim": 8,
        "torch_dtype": "float16",
    }

    tensors = estimate_cache_tensors_from_config(config, 257)

    assert len(tensors) == 8
    index_values = [
        tensor for tensor in tensors if tensor.path.endswith("index.values")
    ]
    assert len(index_values) == 2
    assert all(tensor.shape == (1, 1, 512, 0) for tensor in index_values)
    assert all(tensor.nbytes == 0 for tensor in index_values)
    expected = 2 * ((16 + 4 + 8) * 512 * 2)
    assert sum(tensor.nbytes for tensor in tensors) == expected


def test_minimax_m3_manifest_includes_sparse_index_keys():
    config = {
        "model_type": "minimax_m3",
        "num_hidden_layers": 5,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "sparse_index_dim": 12,
            "sparse_attention_freq": [0, 0, 1, 0, 1],
        },
    }

    tensors = estimate_cache_tensors_from_config(config, 257)

    assert [tensor.path for tensor in tensors if tensor.role == "index_keys"] == [
        "layers.2.index_keys",
        "layers.4.index_keys",
    ]
    assert {
        tensor.cache_kind
        for tensor in tensors
        if tensor.path.startswith(("layers.2.", "layers.4."))
    } == {"MiniMaxM3KVCache"}
    assert all(
        tensor.shape == (1, 1, 512, 12)
        for tensor in tensors
        if tensor.role == "index_keys"
    )


def test_minimax_m3_omitted_sparse_flag_matches_runtime_dense_default():
    config = {
        "model_type": "minimax_m3",
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "torch_dtype": "bfloat16",
        "sparse_attention_config": {
            "sparse_index_dim": 12,
            "sparse_attention_freq": [1, 1],
        },
    }

    tensors = estimate_cache_tensors_from_config(config, 257)

    assert not [tensor for tensor in tensors if tensor.role == "index_keys"]
    assert {tensor.cache_kind for tensor in tensors} == {"KVCache"}


@pytest.mark.parametrize(
    ("extra", "indexed_layers"),
    [
        ({}, {3, 4}),
        (
            {
                "layer_types": [
                    "minimax_m3_sparse",
                    "minimax_m3_sparse",
                    "minimax_m3_sparse",
                    "minimax_m3_sparse",
                    "minimax_m3_sparse",
                ]
            },
            {0, 1, 2, 3, 4},
        ),
        (
            {
                "sparse_attention_config": {
                    "sparse_disable_index_value": [0, 0, 0, 1, 1]
                }
            },
            {3, 4},
        ),
        (
            {"sparse_attention_config": {"use_sparse_attention": True}},
            set(),
        ),
        (
            {
                "sparse_attention_config": {
                    "sparse_attention_freq": [0, 0, 1, 1, 1],
                    "sparse_disable_index_value": [0, 0, 0, 1, 1],
                }
            },
            set(),
        ),
    ],
)
def test_minimax_m3_sparse_defaults_match_runtime_post_init(extra, indexed_layers):
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()
    from mlx_vlm.models.minimax_m3_vl.config import TextConfig

    config = {
        "model_type": "minimax_m3",
        "num_hidden_layers": 5,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "index_head_dim": 12,
        "torch_dtype": "bfloat16",
        **extra,
    }

    tensors = estimate_cache_tensors_from_config(config, 257)
    indexed = {
        int(tensor.path.split(".")[1])
        for tensor in tensors
        if tensor.role == "index_keys"
    }
    runtime = TextConfig(
        num_hidden_layers=5,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        index_head_dim=12,
        **extra,
    )
    runtime_indexed = {
        layer for layer in range(5) if runtime.has_sparse_index(layer)
    }

    assert indexed == indexed_layers
    assert indexed == runtime_indexed


def test_minimax_m3_sparse_config_exhaustive_runtime_parity():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()
    from mlx_vlm.models.minimax_m3_vl.config import TextConfig

    missing = object()
    sparse_layers = ["minimax_m3_sparse"] * 5
    frequency = [0, 0, 1, 1, 1]
    disable = [0, 0, 0, 1, 1]
    for layer_types in (None, sparse_layers):
        for use_sparse in (missing, False, True):
            for configured_frequency in (missing, frequency):
                for disable_frequency in (missing, disable):
                    sparse = {}
                    if use_sparse is not missing:
                        sparse["use_sparse_attention"] = use_sparse
                    if configured_frequency is not missing:
                        sparse["sparse_attention_freq"] = configured_frequency
                    if disable_frequency is not missing:
                        sparse["sparse_disable_index_value"] = disable_frequency
                    extra = {"sparse_attention_config": sparse}
                    if layer_types is not None:
                        extra["layer_types"] = layer_types
                    config = {
                        "model_type": "minimax_m3",
                        "num_hidden_layers": 5,
                        "num_attention_heads": 8,
                        "num_key_value_heads": 2,
                        "head_dim": 16,
                        "index_head_dim": 12,
                        "torch_dtype": "bfloat16",
                        **extra,
                    }

                    tensors = estimate_cache_tensors_from_config(config, 257)
                    planned = {
                        int(tensor.path.split(".")[1])
                        for tensor in tensors
                        if tensor.role == "index_keys"
                    }
                    runtime = TextConfig(
                        num_hidden_layers=5,
                        num_attention_heads=8,
                        num_key_value_heads=2,
                        head_dim=16,
                        index_head_dim=12,
                        **extra,
                    )
                    live = {
                        layer
                        for layer in range(5)
                        if runtime.has_sparse_index(layer)
                    }

                    assert planned == live, extra


def test_unlimited_ocr_uses_ring_sliding_cache_layout():
    config = {
        "model_type": "unlimited_ocr",
        "text_config": {
            "model_type": "deepseek_v2",
            "num_hidden_layers": 2,
            "num_attention_heads": 10,
            "num_key_value_heads": 10,
            "hidden_size": 1280,
            "sliding_window_size": 128,
            "use_mla": False,
            "max_position_embeddings": 4096,
            "torch_dtype": "float16",
        },
    }

    tensors = estimate_cache_tensors_from_config(
        {**config["text_config"], "_root_model_type": config["model_type"]},
        257,
    )

    assert {tensor.cache_kind for tensor in tensors} == {"RingSlidingKVCache"}
    assert {tensor.physical_tokens for tensor in tensors} == {512}
    assert {tensor.logical_tokens for tensor in tensors} == {257}


def test_bailing_hybrid_manifest_mixes_mla_and_recurrent_state():
    config = {
        "model_type": "bailing_hybrid",
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "kv_lora_rank": 8,
        "qk_rope_head_dim": 4,
        "layer_group_size": 2,
        "short_conv_kernel_size": 3,
        "max_position_embeddings": 4096,
        "torch_dtype": "float16",
    }

    tensors = estimate_cache_tensors_from_config(config, 257)

    recurrent = [tensor for tensor in tensors if tensor.cache_kind == "ArraysCache"]
    attention = [tensor for tensor in tensors if tensor.cache_kind == "KVCache"]
    assert [tensor.shape for tensor in recurrent] == [
        (1, 2, 8, 8),
        (1, 16, 3),
        (1, 16, 3),
        (1, 16, 3),
    ]
    assert recurrent[0].dtype == "float32"
    assert [tensor.shape for tensor in attention] == [
        (1, 1, 512, 8),
        (1, 1, 512, 4),
    ]


@pytest.mark.parametrize("model_type", ["qwen3_6", "qwen3_8"])
def test_qwen_aliases_use_gated_delta_manifest(model_type):
    config = {
        "model_type": model_type,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "hidden_size": 128,
        "full_attention_interval": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 4096,
        "torch_dtype": "float16",
    }

    tensors = estimate_cache_tensors_from_config(config, 257)

    assert sum(tensor.cache_kind == "ArraysCache" for tensor in tensors) == 6


def test_nemotron_nas_all_linear_blocks_have_an_exact_zero_byte_cache(tmp_path):
    model = _model(
        tmp_path,
        {
            "model_type": "nemotron-nas",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "max_position_embeddings": 64,
            "block_configs": [
                {
                    "attention": {"replace_with_linear": True},
                    "ffn": {"replace_with_linear": True},
                },
                {
                    "attention": {"replace_with_linear": True},
                    "ffn": {"replace_with_linear": True},
                },
            ],
            "torch_dtype": "float16",
        },
    )

    plan = estimate_model_memory(
        model,
        32,
        weights_bytes=1_000,
        requested_session_slots=8,
        available_memory_bytes=10_000,
    )

    assert plan.cache_tensors == ()
    assert plan.per_session_kv_bytes == 0
    assert plan.fixed_kv_cache_bytes == 0
    assert plan.reserved_session_slots == 8
    assert plan.fits is True


def test_nemotron_h_manifest_tracks_only_cache_bearing_blocks():
    config = {
        "model_type": "nemotron_h",
        "num_hidden_layers": 4,
        "layers_block_type": ["mamba", "moe", "attention", "mlp"],
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "mamba_num_heads": 4,
        "mamba_head_dim": 8,
        "ssm_state_size": 6,
        "n_groups": 2,
        "conv_kernel": 4,
        "mamba_ssm_cache_dtype": "float32",
        "max_position_embeddings": 4096,
        "dtype": "bfloat16",
    }

    tensors = estimate_cache_tensors_from_config(config, 257)

    assert [tensor.cache_kind for tensor in tensors] == [
        "ArraysCache",
        "ArraysCache",
        "KVCache",
        "KVCache",
    ]
    assert [tensor.shape for tensor in tensors] == [
        (1, 3, 56),
        (1, 4, 8, 6),
        (1, 2, 512, 16),
        (1, 2, 512, 16),
    ]
    assert tensors[1].dtype == "float32"


def test_plamo2_attention_uses_runtime_hidden_size_per_head():
    config = {
        "model_type": "plamo2",
        "num_hidden_layers": 2,
        "hidden_size": 28,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 99,
        "hidden_size_per_head": 7,
        "mamba_num_heads": 4,
        "mamba_d_state": 3,
        "mamba_d_conv": 4,
        "mamba_step": 2,
        "torch_dtype": "float16",
    }

    tensors = estimate_cache_tensors_from_config(config, 257)
    attention = [
        tensor
        for tensor in tensors
        if tensor.path.startswith("layers.1.") and tensor.cache_kind == "KVCache"
    ]

    assert [tensor.shape for tensor in attention] == [
        (1, 2, 512, 7),
        (1, 2, 512, 7),
    ]


def test_inkling_manifest_includes_per_layer_conv_state_and_kv_shapes():
    config = {
        "model_type": "inkling",
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 3,
        "swa_head_dim": 4,
        "hidden_size": 32,
        "layer_types": ["hybrid_sliding", "full_attention"],
        "sconv_kernel_size": 4,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
    }

    tensors = estimate_cache_tensors_from_config(config, 257)

    kv_keys = [tensor for tensor in tensors if tensor.role == "keys"]
    assert [tensor.shape for tensor in kv_keys] == [
        (1, 3, 512, 4),
        (1, 2, 512, 8),
    ]
    layer_zero_state = [
        tensor.shape
        for tensor in tensors
        if tensor.path.startswith("layers.0.state")
    ]
    assert layer_zero_state == [
        (1, 3, 12),
        (1, 3, 12),
        (1, 3, 32),
        (1, 3, 32),
    ]


def test_baichuan_m1_manifest_combines_conv_and_mixed_attention_caches():
    config = {
        "model_type": "baichuan_m1",
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_swa_attention_heads": 8,
        "num_swa_key_value_heads": 1,
        "hidden_size": 32,
        "sliding_window": 128,
        "sliding_window_layers": [0],
        "conv_window": 2,
        "max_position_embeddings": 4096,
        "torch_dtype": "float16",
    }

    tensors = estimate_cache_tensors_from_config(config, 257)

    keys = [tensor for tensor in tensors if tensor.role == "keys"]
    assert [tensor.cache_kind for tensor in keys] == [
        "RotatingKVCache",
        "KVCache",
    ]
    assert [tensor.shape for tensor in keys] == [
        (1, 1, 128, 4),
        (1, 2, 512, 8),
    ]
    conv = [tensor.shape for tensor in tensors if tensor.cache_kind == "ArraysCache"]
    assert conv == [
        (1, 1, 1, 4),
        (1, 1, 1, 4),
        (1, 2, 1, 8),
        (1, 2, 1, 8),
    ]


def test_glm_dsa_allocates_runtime_index_cache_for_every_layer():
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 2,
        "kv_lora_rank": 16,
        "qk_rope_head_dim": 4,
        "index_head_dim": 8,
        "index_topk_pattern": "FS",
        "torch_dtype": "float16",
    }

    tensors = estimate_cache_tensors_from_config(config, 256)

    index = [tensor for tensor in tensors if ".index." in tensor.path]
    assert len(index) == 4
    assert {tensor.path.split(".")[1] for tensor in index} == {"0", "1"}


def test_deepseek_v2_is_not_misclassified_as_latent_mla():
    config = {
        "model_type": "deepseek_v2",
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 4,
        "v_head_dim": 6,
        "torch_dtype": "float16",
    }

    tensors = estimate_cache_tensors_from_config(config, 1)

    assert tensors[0].shape == (1, 4, 256, 12)
    assert tensors[1].shape == (1, 4, 256, 6)


def test_deepseek_v4_counts_rotating_pool_buffers_and_overlap_carry():
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 3,
        "head_dim": 16,
        "index_head_dim": 4,
        "sliding_window": 128,
        # Some checkpoints append auxiliary/MTP ratios. The serving model
        # truncates this list to num_hidden_layers during ModelArgs parsing.
        "compress_ratios": [0, 4, 128, 4, 0],
        "torch_dtype": "float16",
    }

    tensors = estimate_cache_tensors_from_config(config, 1_000)

    local_keys = [tensor for tensor in tensors if tensor.path.endswith("local.keys")]
    assert len(local_keys) == 3
    assert all(tensor.physical_tokens == 128 for tensor in local_keys)
    assert any(
        tensor.path.endswith("main_pool.previous_window_kv") for tensor in tensors
    )
    assert any(tensor.path.endswith("index_pool.buffer_gate") for tensor in tensors)
    assert not any(
        "layers.2.main_pool.previous_window" in tensor.path for tensor in tensors
    )
    # Matches the model's resident-cache formula for ratios 0, 4, and 128.
    assert sum(tensor.nbytes for tensor in tensors) == 31_984


def test_plan_accepts_deepseek_v4_pooled_layout(tmp_path):
    model = _model(
        tmp_path,
        {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 3,
            "head_dim": 16,
            "index_head_dim": 4,
            "sliding_window": 128,
            "compress_ratios": [0, 4, 0],
            "max_position_embeddings": 4096,
            "torch_dtype": "float16",
        },
    )

    plan = estimate_model_memory(
        model,
        256,
        weights_bytes=1,
        requested_session_slots=1,
        available_memory_bytes=10**9,
    )

    assert "PoolingCache" in {tensor.cache_kind for tensor in plan.cache_tensors}
    assert plan.fixed_kv_cache_bytes > 0


def test_explicit_hybrid_layer_types_cap_only_sliding_layers():
    config = _generic_config(
        num_hidden_layers=3,
        layer_types=["full_attention", "sliding_attention", "global_attention"],
        sliding_window=128,
    )

    tensors = estimate_cache_tensors_from_config(config, 1_000)
    key_capacities = [
        tensor.physical_tokens for tensor in tensors if tensor.role == "keys"
    ]

    assert key_capacities == [1_024, 128, 1_024]


def test_gemma_pattern_and_shared_kv_reduce_physical_layers():
    config = _generic_config(
        model_type="gemma4",
        num_hidden_layers=6,
        num_kv_shared_layers=2,
        sliding_window_pattern=2,
        sliding_window=128,
    )

    tensors = estimate_cache_tensors_from_config(config, 1_000)

    assert len(tensors) == 8
    assert [tensor.physical_tokens for tensor in tensors if tensor.role == "keys"] == [
        128,
        1_024,
        128,
        1_024,
    ]


def test_llama4_chunked_layers_include_one_prefill_chunk_of_headroom():
    config = _generic_config(
        model_type="llama4",
        num_hidden_layers=4,
        attention_chunk_size=768,
        max_position_embeddings=20_000,
    )

    tensors = estimate_cache_tensors_from_config(
        config,
        10_000,
        prefill_step_size=2_048,
    )

    assert [tensor.capacity_kind for tensor in tensors if tensor.role == "keys"] == [
        "chunked",
        "chunked",
        "chunked",
        "linear",
    ]
    assert [tensor.physical_tokens for tensor in tensors if tensor.role == "keys"] == [
        2_816,
        2_816,
        2_816,
        10_240,
    ]


def test_llama4_plan_caps_concurrency_to_serialized_runtime(tmp_path):
    model = _model(
        tmp_path,
        _generic_config(
            model_type="llama4",
            num_hidden_layers=4,
            attention_chunk_size=768,
        ),
    )

    plan = estimate_model_memory(
        model,
        2_000,
        weights_bytes=0,
        requested_session_slots=8,
        available_memory_bytes=1 << 40,
    )

    assert plan.cache_layout_max_session_slots == 1
    assert plan.max_feasible_session_slots == 1
    assert plan.reserved_session_slots == 1
    assert plan.configured_concurrency_capped is True
    assert plan.fit_reason is not None
    assert "serialized" in plan.fit_reason


@pytest.mark.parametrize(
    "config, message",
    [
        (
            _generic_config(layer_types=["full_attention", "linear_attention"]),
            "live cache probe",
        ),
        (
            _generic_config(model_type="unknown_mla", kv_lora_rank=16),
            "no verified fixed-cache layout adapter",
        ),
        (
            _generic_config(model_type="qwen3_next"),
            "Qwen gated-delta cache planning requires",
        ),
        (
            _generic_config(layer_types=["full_attention"]),
            "length does not match",
        ),
    ],
)
def test_unknown_special_layouts_fail_closed(config, message):
    with pytest.raises(FixedKVPlanningError, match=message):
        estimate_cache_tensors_from_config(config, 256)


def test_context_above_native_limit_fails_before_planning():
    with pytest.raises(FixedKVPlanningError, match="exceeds the model limit"):
        estimate_cache_tensors_from_config(_generic_config(), 4_097)


class _FakeDType:
    itemsize = 2

    def __str__(self):
        return "float16"


class _FakeArray:
    def __init__(self, shape, *, nbytes=None):
        self.shape = tuple(shape)
        self.dtype = _FakeDType()
        self.nbytes = (
            nbytes if nbytes is not None else 2 * __import__("math").prod(shape)
        )


class _FakeKVCache:
    def __init__(self):
        self.keys = _FakeArray((1, 2, 256, 8))
        self.values = _FakeArray((1, 2, 256, 4))
        self.offset = 17

    @property
    def state(self):
        raise AssertionError("backing arrays must be preferred over state slices")


class _FakePoolingCache:
    def __init__(self):
        self._pool_buf = _FakeArray((1, 32, 8))
        self.buf_kv = _FakeArray((1, 4, 16))
        self.buf_gate = _FakeArray((1, 4, 16))


class _FakeCacheList:
    def __init__(self, *caches):
        self.caches = list(caches)


class _FakeArraysCache:
    def __init__(self):
        self.cache = [_FakeArray((1, 4, 8)), _FakeArray((1, 2, 8))]


def test_live_probe_manifest_reads_backing_arrays_without_mlx():
    tree = [_FakeCacheList(_FakeKVCache(), _FakePoolingCache()), _FakeArraysCache()]

    tensors = build_cache_manifest(tree)

    assert len(tensors) == 7
    keys = next(tensor for tensor in tensors if tensor.role == "keys")
    assert keys.shape == (1, 2, 256, 8)
    assert keys.logical_tokens == 17
    assert keys.physical_tokens == 256
    assert keys.nbytes == 8192
    pool = next(tensor for tensor in tensors if tensor.role == "_pool_buf")
    assert pool.capacity_kind == "pooled"
    assert pool.physical_tokens == 32
    assert any(tensor.cache_kind == "_FakeArraysCache" for tensor in tensors)


def test_live_probe_deduplicates_shared_array_references():
    shared = _FakeArray((1, 2, 3))

    class SharedCache:
        keys = shared
        values = shared
        offset = 3

    tensors = build_cache_manifest([SharedCache()])

    assert len(tensors) == 1


def test_live_probe_refuses_an_unmaterialized_cache():
    class EmptyCache:
        keys = None
        values = None

    with pytest.raises(FixedKVPlanningError, match="Run one model step"):
        build_cache_manifest([EmptyCache()])
