# SPDX-License-Identifier: Apache-2.0
"""Tests for oQ (oMLX Universal Dynamic Quantization)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.oq import (
    _LEVEL_BITS,
    _LEVEL_EXPERT_DOWN_BOOST,
    _MAX_MODEL_RAM_FRACTION,
    _OQ_BPW_TARGETS,
    _PROXY_QUANT_BITS,
    _PROXY_QUANT_GROUP_SIZE,
    _ROUTED_LAYER_BOOST_LEVELS,
    OQ_LEVELS,
    _bpw_targets_for_level,
    _build_proxy_for_sensitivity,
    _build_quant_plan,
    _build_streaming_proxy_for_sensitivity,
    _config_expects_moe_expert_counts,
    _discover_sanitize_plan,
    _DiscoveredPlan,
    _extract_layer_index,
    _format_size,
    _forward_layer,
    _forward_layer_result,
    _get_predicate_bits,
    _ImatrixCaptureWrapper,
    _imatrix_expert_coverage_stats,
    _imatrix_expert_coverage_sufficient,
    _imatrix_requires_expert_counts,
    _is_audio_tensor,
    _is_moe_router,
    _is_vision_tensor,
    _LazyTensorIndex,
    _measure_sensitivity,
    _measure_sensitivity_from_quantized_model,
    _normalize_quant_path,
    _perturb_bits_for,
    _progress_total_bytes,
    _quantize_chunked,
    _sensitivity_lm_config_override,
    _should_quantize_tensor,
    _TrackedTensor,
    _validate_oq_dtype_for_model,
    OQImatrixCollector,
    OQImatrixEntry,
    estimate_bpw_and_size,
    estimate_memory,
    make_predicate,
    quantize_oq_streaming,
    resolve_output_name,
    universal_quant_predicate,
    validate_quantizable,
)

# =============================================================================
# Test universal_quant_predicate
# =============================================================================


class TestUniversalQuantPredicate:
    """Test the universal quantization predicate with various tensor paths."""

    @pytest.fixture
    def dense_config(self):
        return {"num_hidden_layers": 32, "hidden_size": 4096}

    @pytest.fixture
    def moe_config(self):
        return {
            "num_hidden_layers": 48,
            "num_local_experts": 256,
            "hidden_size": 3072,
        }

    @pytest.fixture
    def large_moe_config(self):
        return {
            "num_hidden_layers": 48,
            "num_local_experts": 512,
            "hidden_size": 4096,
        }

    @pytest.fixture
    def module(self):
        return MagicMock(spec=[])

    # Stage 0: Non-quantization (should return False)

    def test_moe_router_fp16(self, moe_config, module):
        result = universal_quant_predicate(
            "model.layers.0.mlp.gate", module, moe_config
        )
        assert (
            result is False
        )  # MoE router gates kept fp16 (some models lack to_quantized)

    def test_shared_expert_gate_8bit(self, moe_config, module):
        result = universal_quant_predicate(
            "model.layers.0.shared_expert_gate", module, moe_config
        )
        assert isinstance(result, dict) and result["bits"] == 8

    def test_non_quantizable_module_skipped(self, dense_config, module):
        cfg = {
            **dense_config,
            "_oq_non_quantizable": {
                "language_model.model.per_layer_model_projection",
            },
        }
        assert (
            universal_quant_predicate(
                "language_model.model.per_layer_model_projection.weight", module, cfg
            )
            is False
        )

    def test_non_quantizable_set_does_not_affect_other_paths(
        self, dense_config, module
    ):
        cfg = {
            **dense_config,
            "_oq_non_quantizable": {
                "language_model.model.per_layer_model_projection",
            },
        }
        result = universal_quant_predicate(
            "language_model.model.layers.0.per_layer_input_gate.weight", module, cfg
        )
        assert result is not False

    def test_empty_non_quantizable_set_is_noop(self, dense_config, module):
        cfg = {**dense_config, "_oq_non_quantizable": set()}
        result = universal_quant_predicate(
            "model.layers.0.self_attn.q_proj.weight", module, cfg
        )
        assert result is not False

    def test_vision_encoder_not_quantized(self, dense_config, module):
        assert (
            universal_quant_predicate(
                "visual.encoder.layers.0.self_attn.q_proj", module, dense_config
            )
            is False
        )

    def test_patch_embed_not_quantized(self, dense_config, module):
        assert (
            universal_quant_predicate("model.patch_embed.proj", module, dense_config)
            is False
        )

    def test_ssm_alpha_not_quantized(self, dense_config, module):
        assert (
            universal_quant_predicate("model.layers.0.ssm_alpha", module, dense_config)
            is False
        )

    def test_ssm_beta_not_quantized(self, dense_config, module):
        assert (
            universal_quant_predicate("model.layers.0.ssm_beta", module, dense_config)
            is False
        )

    def test_a_log_not_quantized(self, dense_config, module):
        assert (
            universal_quant_predicate("model.layers.0.a_log", module, dense_config)
            is False
        )

    def test_mamba_d_not_quantized(self, dense_config, module):
        assert (
            universal_quant_predicate("model.layers.0.mixer.D", module, dense_config)
            is False
        )

    def test_time_decay_not_quantized(self, dense_config, module):
        assert (
            universal_quant_predicate("model.layers.0.time_decay", module, dense_config)
            is False
        )

    # Qwen3_5 hybrid (GatedDeltaNet) — issue #913 regression guards.
    # Real weight names use capital `A_log`, so the skip check must be case-insensitive.

    def test_qwen35_A_log_not_quantized(self, dense_config, module):
        path = "model.language_model.layers.0.linear_attn.A_log"
        assert universal_quant_predicate(path, module, dense_config) is False

    def test_qwen35_dt_bias_not_quantized(self, dense_config, module):
        path = "model.language_model.layers.0.linear_attn.dt_bias"
        assert universal_quant_predicate(path, module, dense_config) is False

    def test_qwen35_linear_attn_conv1d_8bit(self, dense_config, module):
        path = "model.language_model.layers.0.linear_attn.conv1d.weight"
        result = universal_quant_predicate(path, module, dense_config)
        assert isinstance(result, dict)
        assert result["bits"] == 8

    def test_qwen35_linear_attn_out_proj_5bit(self, dense_config, module):
        path = "model.language_model.layers.0.linear_attn.out_proj.weight"
        result = universal_quant_predicate(path, module, dense_config)
        assert isinstance(result, dict)
        assert result["bits"] == 5

    def test_qwen35_linear_attn_in_proj_qkv_quantized(self, dense_config, module):
        # Regression guard: existing behavior should still return a quant dict/True, not skip.
        path = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        result = universal_quant_predicate(path, module, dense_config)
        assert result is not False

    # Stage 1: High-precision protection

    def test_ssm_output_8bit(self, dense_config, module):
        result = universal_quant_predicate(
            "model.layers.0.ssm_output", module, dense_config
        )
        assert isinstance(result, dict)
        assert result["bits"] == 8

    def test_lm_head_6bit(self, dense_config, module):
        result = universal_quant_predicate("lm_head", module, dense_config)
        assert isinstance(result, dict)
        assert result["bits"] == 6

    def test_mla_kv_b_proj_6bit(self, dense_config, module):
        result = universal_quant_predicate(
            "model.layers.0.self_attn.kv_b_proj", module, dense_config
        )
        assert isinstance(result, dict)
        assert result["bits"] == 6

    def test_dense_o_proj_5bit(self, dense_config, module):
        result = universal_quant_predicate(
            "model.layers.5.self_attn.o_proj", module, dense_config
        )
        assert isinstance(result, dict)
        assert result["bits"] == 5

    # Stage 2: MoE-specific

    def test_shared_expert_body_high_bits(self, moe_config, module):
        result = universal_quant_predicate(
            "model.layers.0.mlp.shared_expert.gate_proj", module, moe_config
        )
        assert isinstance(result, dict)
        assert result["bits"] == 8

    def test_512_expert_gate_proj_floor(self, large_moe_config, module):
        result = universal_quant_predicate(
            "model.layers.0.mlp.switch_mlp.gate_proj", module, large_moe_config
        )
        assert isinstance(result, dict)
        assert result["bits"] >= 4

    def test_512_expert_down_proj_floor(self, large_moe_config, module):
        result = universal_quant_predicate(
            "model.layers.0.mlp.switch_mlp.down_proj", module, large_moe_config
        )
        assert isinstance(result, dict)
        assert result["bits"] >= 3

    # Stage 3: Layer position strategy

    def test_v_proj_sensitive_layer_6bit(self, dense_config, module):
        # Layer 0 is in first 12.5% (0 < 32//8 = 4)
        result = universal_quant_predicate(
            "model.layers.0.self_attn.v_proj", module, dense_config
        )
        assert isinstance(result, dict)
        assert result["bits"] == 6

    def test_v_proj_non_sensitive_layer_base(self, dense_config, module):
        # Layer 10 is not sensitive → returns True (base bits)
        result = universal_quant_predicate(
            "model.layers.10.self_attn.v_proj", module, dense_config
        )
        assert result is True

    def test_down_proj_always_protected(self, dense_config, module):
        # Non-sensitive layer should still get 5-bit (Super Weights)
        result = universal_quant_predicate(
            "model.layers.10.mlp.down_proj", module, dense_config
        )
        assert isinstance(result, dict)
        assert result["bits"] >= 5

    def test_q_proj_sensitive_5bit(self, dense_config, module):
        result = universal_quant_predicate(
            "model.layers.0.self_attn.q_proj", module, dense_config
        )
        assert isinstance(result, dict)
        assert result["bits"] == 5

    # Stage 4: SSM/GatedDeltaNet

    def test_gated_deltanet_in_proj_z_5bit(self, dense_config, module):
        result = universal_quant_predicate(
            "model.layers.0.attn.in_proj_z", module, dense_config
        )
        assert isinstance(result, dict)
        assert result["bits"] == 5

    def test_mamba_mixer_in_proj_5bit(self, dense_config, module):
        result = universal_quant_predicate(
            "model.layers.0.mixer.in_proj", module, dense_config
        )
        assert isinstance(result, dict)
        assert result["bits"] == 5

    # Stage 6: FFN/MLP (default bits)

    def test_gate_proj_default(self, dense_config, module):
        result = universal_quant_predicate(
            "model.layers.10.mlp.gate_proj", module, dense_config
        )
        assert result is True

    def test_up_proj_default(self, dense_config, module):
        result = universal_quant_predicate(
            "model.layers.10.mlp.up_proj", module, dense_config
        )
        assert result is True

    # Group size

    def test_moe_router_fp16_group_size(self, moe_config, module):
        result = universal_quant_predicate(
            "model.layers.0.mlp.gate", module, moe_config
        )
        assert result is False  # MoE router gates kept fp16

    def test_150_expert_group_size_128(self, module):
        config = {
            "num_hidden_layers": 32,
            "num_local_experts": 200,
            "hidden_size": 2048,
        }
        result = universal_quant_predicate(
            "model.layers.10.mlp.gate_proj", module, config
        )
        # gate_proj returns True (default), but when a dict is returned,
        # group_size should be 128 for 150+ experts
        # gate_proj is in stage 6, returns True, so no dict to check
        assert result is True

    # VLM nested config support

    def test_vlm_nested_config_moe_detection(self, module):
        """VLM models have text model config nested under text_config."""
        vlm_config = {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "num_hidden_layers": 40,
                "num_experts": 256,
                "hidden_size": 2048,
            },
            "vision_config": {"hidden_size": 1152},
        }
        # Expert down_proj should be base bits (routed expert in MoE)
        result = universal_quant_predicate(
            "model.layers.10.mlp.experts.0.down_proj", module, vlm_config
        )
        assert result is True  # base bits, NOT 5-bit

    def test_vlm_nested_config_sensitive_layers(self, module):
        """Sensitive layer calculation uses correct num_hidden_layers from text_config."""
        vlm_config = {
            "text_config": {
                "num_hidden_layers": 40,
                "num_experts": 256,
                "hidden_size": 2048,
            },
        }
        # Layer 10 should NOT be sensitive (40 layers: first 5 and last 5)
        result = universal_quant_predicate(
            "model.layers.10.self_attn.v_proj", module, vlm_config
        )
        assert result is True  # base bits (not sensitive)

    def test_vlm_nested_config_num_local_experts(self, module):
        """Also handles num_local_experts in text_config."""
        vlm_config = {
            "text_config": {
                "num_hidden_layers": 32,
                "num_local_experts": 64,
                "hidden_size": 4096,
            },
        }
        result = universal_quant_predicate(
            "model.layers.10.mlp.experts.0.down_proj", module, vlm_config
        )
        assert result is True  # routed expert → base bits

    def test_null_num_experts_dense_model(self, module):
        """Gemma 4 dense models have explicit num_experts: null in config."""
        config = {
            "num_hidden_layers": 60,
            "hidden_size": 6144,
            "text_config": {"num_experts": None},
        }
        result = universal_quant_predicate(
            "model.layers.10.self_attn.q_proj", module, config
        )
        assert result is True  # should not crash on None > 0


# =============================================================================
# Test helper functions
# =============================================================================


class TestHelpers:
    def test_is_moe_router_mlp_gate(self):
        assert _is_moe_router("model.layers.0.mlp.gate") is True

    def test_is_moe_router_router(self):
        assert _is_moe_router("model.layers.0.block_sparse_moe.router") is True

    def test_is_moe_router_gate_proj_not_router(self):
        assert _is_moe_router("model.layers.0.mlp.gate_proj") is False

    def test_is_moe_router_shared_expert_gate_proj_not_router(self):
        assert _is_moe_router("model.layers.0.mlp.shared_expert.gate_proj") is False

    def test_extract_layer_index(self):
        assert _extract_layer_index("model.layers.5.self_attn.q_proj") == 5

    def test_extract_layer_index_no_match(self):
        assert _extract_layer_index("lm_head") == -1

    def test_extract_layer_index_large(self):
        assert _extract_layer_index("model.layers.47.mlp.gate_proj") == 47

    def test_normalize_quant_path_weight(self):
        assert _normalize_quant_path("model.layers.0.self_attn.q_proj.weight") == (
            "model.layers.0.self_attn.q_proj"
        )

    def test_normalize_quant_path_scales(self):
        assert _normalize_quant_path("lm_head.scales") == "lm_head"

    def test_is_audio_tensor_audio_tower(self):
        assert (
            _is_audio_tensor(
                "audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight"
            )
            is True
        )

    def test_is_audio_tensor_embed_audio_not_excluded(self):
        # embed_audio.embedding_projection is the projection from audio output
        # to text hidden — should be quantizable like embed_vision counterpart.
        assert _is_audio_tensor("embed_audio.embedding_projection.weight") is False

    def test_is_audio_tensor_language_model(self):
        assert (
            _is_audio_tensor("language_model.model.layers.0.self_attn.q_proj.weight")
            is False
        )

    def test_is_audio_tensor_vision_tower(self):
        assert (
            _is_audio_tensor("vision_tower.layers.0.self_attn.k_proj.weight") is False
        )

    def test_universal_quant_predicate_skips_audio_tower(self):
        # audio_tower tensors must be kept in fp16 (return False from predicate)
        # — same treatment as vision_tower.
        result = universal_quant_predicate(
            "audio_tower.layers.0.self_attn.k_proj", None, {}, oq_level=4
        )
        assert result is False

    def test_universal_quant_predicate_quantizes_embed_audio(self):
        # embed_audio.embedding_projection should NOT be skipped — it's a
        # quantizable Linear, mirroring how embed_vision is treated.
        result = universal_quant_predicate(
            "embed_audio.embedding_projection", None, {}, oq_level=4
        )
        assert result is not False


# =============================================================================
# Test resolve_output_name
# =============================================================================


class TestResolveOutputName:
    def test_basic(self):
        assert resolve_output_name("Qwen3.5-122B-A10B", 4) == "Qwen3.5-122B-A10B-oQ4"

    def test_deepseek_v4_oq8_mtp(self):
        assert (
            resolve_output_name("DeepSeek-V4-Flash", 8, "bfloat16", preserve_mtp=True)
            == "DeepSeek-V4-Flash-oQ8-mtp"
        )

    def test_strip_existing_bit_suffix(self):
        assert (
            resolve_output_name("Qwen3.5-122B-A10B-8bit", 4) == "Qwen3.5-122B-A10B-oQ4"
        )

    def test_strip_existing_oq_suffix(self):
        assert (
            resolve_output_name("Qwen3.5-122B-A10B-oQ6", 2) == "Qwen3.5-122B-A10B-oQ2"
        )

    def test_strip_existing_enhanced_suffix(self):
        assert (
            resolve_output_name("Qwen3.5-122B-A10B-oQ4e", 2) == "Qwen3.5-122B-A10B-oQ2"
        )

    def test_enhanced_appends_e_suffix(self):
        assert (
            resolve_output_name("Qwen3.5-122B-A10B", 4, enhanced=True)
            == "Qwen3.5-122B-A10B-oQ4e"
        )

    def test_all_levels(self):
        for level in OQ_LEVELS:
            result = resolve_output_name("Model-7B", level)
            assert result == f"Model-7B-oQ{level}"

    def test_bfloat16_default_no_suffix(self):
        assert resolve_output_name("Llama-3-8B", 4, "bfloat16") == "Llama-3-8B-oQ4"

    def test_float16_appends_fp16_suffix(self):
        assert resolve_output_name("Llama-3-8B", 4, "float16") == "Llama-3-8B-oQ4-fp16"

    def test_float16_strips_existing_dtype_suffix(self):
        assert resolve_output_name("Model-oQ6-fp16", 4, "float16") == "Model-oQ4-fp16"

    def test_bfloat16_strips_chained_suffixes(self):
        assert resolve_output_name("Model-oQ6-fp16", 4, "bfloat16") == "Model-oQ4"

    def test_strips_bf16_suffix(self):
        assert resolve_output_name("Model-bf16", 4, "bfloat16") == "Model-oQ4"

    def test_float16_with_bitwidth_suffix(self):
        assert resolve_output_name("Model-8bit", 3, "float16") == "Model-oQ3-fp16"

    def test_preserve_mtp_appends_mtp_suffix(self):
        assert (
            resolve_output_name("Qwen3.5-27B", 4, "bfloat16", preserve_mtp=True)
            == "Qwen3.5-27B-oQ4-mtp"
        )

    def test_preserve_mtp_with_fp16(self):
        assert (
            resolve_output_name("Llama-3-8B", 4, "float16", preserve_mtp=True)
            == "Llama-3-8B-oQ4-fp16-mtp"
        )

    def test_preserve_mtp_strips_existing_mtp_suffix(self):
        # Re-quantizing an already-mtp output keeps the suffix only when the
        # caller asks for it; without preserve_mtp the suffix is dropped.
        assert (
            resolve_output_name("Model-oQ6-mtp", 4, "bfloat16", preserve_mtp=False)
            == "Model-oQ4"
        )
        assert (
            resolve_output_name("Model-oQ6-mtp", 4, "bfloat16", preserve_mtp=True)
            == "Model-oQ4-mtp"
        )


class TestOqDtypeModelSupport:
    def test_rejects_deepseek_v4_float16(self):
        with pytest.raises(ValueError, match="dtype=float16.*deepseek_v4"):
            _validate_oq_dtype_for_model({"model_type": "deepseek_v4"}, "float16")

    def test_rejects_deepseek_v4_architecture_float16(self):
        with pytest.raises(ValueError, match="dtype=float16.*deepseek_v4"):
            _validate_oq_dtype_for_model(
                {"architectures": ["DeepseekV4ForCausalLM"]}, "float16"
            )

    def test_allows_deepseek_v4_bfloat16(self):
        _validate_oq_dtype_for_model({"model_type": "deepseek_v4"}, "bfloat16")

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_streaming_rejects_before_output_dir_is_created(self, tmp_path):
        src = tmp_path / "DeepSeek-V4-Flash"
        src.mkdir()
        (src / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4"}),
            encoding="utf-8",
        )

        out = tmp_path / "DeepSeek-V4-Flash-oQ4-fp16"
        with pytest.raises(ValueError, match="dtype=float16.*deepseek_v4"):
            quantize_oq_streaming(str(src), str(out), oq_level=4, dtype="float16")

        assert not out.exists()


class TestShouldSkipTensor:
    def test_default_skips_mtp(self):
        from omlx.oq import _should_skip_tensor

        assert _should_skip_tensor("mtp.fc.weight") is True
        assert _should_skip_tensor("language_model.mtp.layers.0.foo") is True

    def test_preserve_mtp_keeps_mtp(self):
        from omlx.oq import _should_skip_tensor

        assert _should_skip_tensor("mtp.fc.weight", preserve_mtp=True) is False
        assert (
            _should_skip_tensor("language_model.mtp.layers.0.foo", preserve_mtp=True)
            is False
        )

    def test_non_mtp_tensors_never_skipped(self):
        from omlx.oq import _should_skip_tensor

        assert _should_skip_tensor("model.layers.0.attn.q_proj.weight") is False
        assert (
            _should_skip_tensor("model.layers.0.attn.q_proj.weight", preserve_mtp=True)
            is False
        )


class TestMtpFcFullPrecision:
    """Critical MTP projections (Qwen3.5 mtp.fc + DeepSeek-V4 e_proj/h_proj
    + hc_head.*) must stay in full precision. Mirrors PR 990's quant_predicate
    extended to PR 15's DeepSeek-V4 MTPBlock layout."""

    def test_qwen_mtp_fc_top_level_returns_none(self):
        from omlx.oq import _get_predicate_bits

        bits, gs, mode = _get_predicate_bits("mtp.fc.weight", {}, 4, 64)
        assert bits is None and gs is None and mode is None

    def test_qwen_mtp_fc_nested_returns_none(self):
        from omlx.oq import _get_predicate_bits

        bits, gs, mode = _get_predicate_bits("language_model.mtp.fc.weight", {}, 4, 64)
        assert bits is None and gs is None and mode is None

    def test_deepseek_e_proj_protected(self):
        from omlx.oq import _get_predicate_bits

        bits, _, _ = _get_predicate_bits("mtp.0.e_proj.weight", {}, 4, 64)
        assert bits is None

    def test_deepseek_h_proj_protected(self):
        from omlx.oq import _get_predicate_bits

        bits, _, _ = _get_predicate_bits("mtp.0.h_proj.weight", {}, 4, 64)
        assert bits is None

    def test_deepseek_hc_head_sanitized_protected(self):
        from omlx.oq import _get_predicate_bits

        for k in ("mtp.0.hc_head.fn", "mtp.0.hc_head.base", "mtp.0.hc_head.scale"):
            bits, _, _ = _get_predicate_bits(k, {}, 4, 64)
            assert bits is None, f"{k} should be full precision"

    def test_deepseek_hc_head_raw_hf_protected(self):
        from omlx.oq import _get_predicate_bits

        # Raw HF form (before sanitize) — covered too.
        for k in ("mtp.0.hc_head_fn", "mtp.0.hc_head_base", "mtp.0.hc_head_scale"):
            bits, _, _ = _get_predicate_bits(k, {}, 4, 64)
            assert bits is None, f"{k} should be full precision"

    def test_other_mtp_tensors_still_quantized(self):
        from omlx.oq import _get_predicate_bits

        bits, _, _ = _get_predicate_bits(
            "mtp.layers.0.self_attn.q_proj.weight", {}, 4, 64
        )
        assert bits is not None and bits >= 4

    def test_deepseek_block_attn_still_quantized(self):
        from omlx.oq import _get_predicate_bits

        # MTPBlock 의 내부 attention/ffn 은 backbone 과 같은 양자화 정책
        bits, _, _ = _get_predicate_bits("mtp.0.block.attn.wq_a.weight", {}, 4, 64)
        assert bits is not None

    def test_normal_weights_unaffected(self):
        from omlx.oq import _get_predicate_bits

        bits, _, _ = _get_predicate_bits("model.layers.0.attn.q_proj.weight", {}, 4, 64)
        assert bits is not None

    def test_non_mtp_e_proj_not_protected(self):
        from omlx.oq import _get_predicate_bits

        # e_proj 가 mtp 밖 (가상 케이스) 이면 보호 안 함
        bits, _, _ = _get_predicate_bits("model.layers.0.e_proj.weight", {}, 4, 64)
        assert bits is not None


class TestNormalizeMtpInConfig:
    def test_zeros_top_level_mtp_fields(self):
        from omlx.oq import _normalize_mtp_in_config

        cfg = {"mtp_num_hidden_layers": 1, "num_nextn_predict_layers": 2}
        _normalize_mtp_in_config(cfg)
        assert cfg["mtp_num_hidden_layers"] == 0
        assert cfg["num_nextn_predict_layers"] == 0

    def test_zeros_nested_text_config_fields(self):
        from omlx.oq import _normalize_mtp_in_config

        cfg = {
            "model_type": "qwen3_5",
            "text_config": {"mtp_num_hidden_layers": 1, "num_hidden_layers": 64},
        }
        _normalize_mtp_in_config(cfg)
        assert cfg["text_config"]["mtp_num_hidden_layers"] == 0
        # Non-mtp fields untouched.
        assert cfg["text_config"]["num_hidden_layers"] == 64

    def test_no_mtp_fields_is_noop(self):
        from omlx.oq import _normalize_mtp_in_config

        cfg = {"model_type": "llama"}
        _normalize_mtp_in_config(cfg)
        assert cfg == {"model_type": "llama"}


# =============================================================================
# Test validate_quantizable
# =============================================================================


class TestValidateQuantizable:
    def test_non_quantized(self):
        assert validate_quantizable({"model_type": "llama"}) is True

    def test_already_quantized(self):
        assert validate_quantizable({"quantization": {"bits": 4}}) is False

    def test_quantization_config_with_known_method(self):
        # Real HF quantization configs always carry quant_method
        assert (
            validate_quantizable(
                {"quantization_config": {"quant_method": "gptq", "bits": 4}}
            )
            is False
        )

    def test_fp8_native_is_quantizable(self):
        # Native FP8 models (MiniMax, DeepSeek) should be quantizable
        assert (
            validate_quantizable({"quantization_config": {"quant_method": "fp8"}})
            is True
        )

    def test_non_fp8_quantization_config(self):
        # Other quant methods (gptq, awq) are already quantized
        assert (
            validate_quantizable({"quantization_config": {"quant_method": "gptq"}})
            is False
        )

    def test_qat_no_quant_method_is_quantizable(self):
        # QAT models have quantization_config with no quant_method — full-precision weights
        assert (
            validate_quantizable({"quantization_config": {"quant_type": "q4_0"}})
            is True
        )

    def test_empty_quantization_config_not_quantizable(self):
        # Empty config has no quant_type — not a QAT config, not quantizable
        assert validate_quantizable({"quantization_config": {}}) is False

    def test_legacy_bits_config_not_quantizable(self):
        # Legacy configs with only {"bits": N} lack quant_type and are not QAT
        assert validate_quantizable({"quantization_config": {"bits": 4}}) is False

    def test_awq_not_quantizable(self):
        assert (
            validate_quantizable({"quantization_config": {"quant_method": "awq"}})
            is False
        )


# =============================================================================
# Test _sensitivity_lm_config_override
# =============================================================================


class TestSensitivityLmConfigOverride:
    def test_no_quantization_config(self):
        assert _sensitivity_lm_config_override({"model_type": "llama"}) is None

    def test_qat_config_top_level(self):
        # QAT config with no quant_method → should override
        result = _sensitivity_lm_config_override(
            {"quantization_config": {"quant_type": "q4_0"}}
        )
        assert result == {"quantization_config": None}

    def test_qat_config_in_text_config(self):
        # QAT config nested in text_config (VLM layout) → should override
        result = _sensitivity_lm_config_override(
            {"text_config": {"quantization_config": {"quant_type": "q4_0"}}}
        )
        assert result == {"quantization_config": None}

    def test_fp8_config_no_override(self):
        # FP8 has quant_method set — not a QAT config, no override needed
        assert (
            _sensitivity_lm_config_override(
                {"quantization_config": {"quant_method": "fp8"}}
            )
            is None
        )

    def test_known_method_no_override(self):
        assert (
            _sensitivity_lm_config_override(
                {"quantization_config": {"quant_method": "gptq"}}
            )
            is None
        )

    def test_non_qat_config_without_quant_method_no_override(self):
        # Legacy bits-only config has no quant_type — not a QAT config, no override
        assert (
            _sensitivity_lm_config_override({"quantization_config": {"bits": 4}})
            is None
        )

    def test_empty_quantization_config(self):
        # Empty dict: no quant_method but also nothing to process — no override
        assert _sensitivity_lm_config_override({"quantization_config": {}}) is None


# =============================================================================
# Test make_predicate
# =============================================================================


class TestMakePredicate:
    def test_returns_callable(self):
        config = {"num_hidden_layers": 32}
        pred = make_predicate(config)
        assert callable(pred)

    def test_predicate_works(self):
        config = {"num_hidden_layers": 32}
        pred = make_predicate(config)
        module = MagicMock(spec=[])
        result = pred("lm_head", module)
        assert isinstance(result, dict)
        assert result["bits"] == 6

    @pytest.mark.parametrize("oq_level", [3, 4, 5])
    def test_budget_plan_disables_static_lm_head_boost_without_override(self, oq_level):
        config = {"num_hidden_layers": 32, "_oq_use_budget_plan": True}
        pred = make_predicate(config, oq_level=oq_level)
        module = MagicMock(spec=[])
        assert pred("lm_head", module) is True

    def test_budget_plan_uses_boost_override(self):
        config = {
            "num_hidden_layers": 32,
            "_oq_use_budget_plan": True,
            "_oq_boost_map": {
                "lm_head": {"bits": 6, "group_size": 64, "mode": "affine"}
            },
        }
        pred = make_predicate(config, oq_level=4)
        module = MagicMock(spec=[])
        result = pred("lm_head.weight", module)
        assert isinstance(result, dict)
        assert result["bits"] == 6


# =============================================================================
# Test estimate_memory
# =============================================================================


class TestEstimateMemory:
    def test_streaming_includes_buffer(self):
        size = 100 * 1024**3  # 100GB model
        result = estimate_memory(size)
        # Streaming: source + 6GB buffer
        assert result["peak_bytes"] > size
        assert result["peak_bytes"] < size * 1.2

    def test_has_formatted(self):
        result = estimate_memory(10 * 1024**3)
        assert "peak_formatted" in result
        assert "GB" in result["peak_formatted"]


# =============================================================================
# Test streaming quantization helpers
# =============================================================================


class TestStreamingHelpers:
    def test_should_quantize_2d_weight(self):
        assert (
            _should_quantize_tensor(
                "model.layers.0.self_attn.q_proj.weight", (4096, 4096)
            )
            is True
        )

    def test_should_not_quantize_1d(self):
        assert (
            _should_quantize_tensor("model.layers.0.input_layernorm.weight", (4096,))
            is False
        )

    def test_should_not_quantize_bias(self):
        assert (
            _should_quantize_tensor("model.layers.0.self_attn.q_proj.bias", (4096,))
            is False
        )

    def test_should_not_quantize_norm(self):
        assert (
            _should_quantize_tensor("model.layers.0.rmsnorm.weight", (4096, 4096))
            is False
        )

    def test_get_predicate_bits_lm_head(self):
        config = {"num_hidden_layers": 32}
        bits, gs, mode = _get_predicate_bits("lm_head", config, 4, 64)
        assert bits == 6
        # 6-bit → affine (no mxfp mode for 6-bit)
        assert mode == "affine"

    def test_get_predicate_bits_router_fp16(self):
        config = {"num_hidden_layers": 32, "num_local_experts": 8}
        bits, gs, mode = _get_predicate_bits("model.layers.0.mlp.gate", config, 4, 64)
        assert bits is None  # Router → fp16 (not quantized)

    def test_get_predicate_bits_default_affine4(self):
        config = {"num_hidden_layers": 32}
        bits, gs, mode = _get_predicate_bits(
            "model.layers.10.mlp.gate_proj.weight", config, 4, 64
        )
        assert bits == 4
        assert gs == 64
        assert mode == "affine"

    def test_get_predicate_bits_3bit_affine(self):
        config = {"num_hidden_layers": 32}
        bits, gs, mode = _get_predicate_bits(
            "model.layers.10.mlp.gate_proj.weight", config, 3, 64
        )
        # oQ3 → base 3-bit → affine
        assert bits == 3
        assert mode == "affine"

    def test_get_predicate_bits_8bit(self):
        config = {"num_hidden_layers": 32}
        bits, gs, mode = _get_predicate_bits(
            "model.layers.10.mlp.gate_proj.weight", config, 8, 64
        )
        # oQ8 → base 8-bit, always affine mode to minimize kernel combos
        assert bits == 8
        assert gs == 64
        assert mode == "affine"

    def test_build_quant_plan_respects_hard_cap(self):
        named_shapes = {
            "lm_head": (4096, 4096),
            "model.layers.0.self_attn.v_proj": (4096, 4096),
            "model.layers.0.self_attn.o_proj": (4096, 4096),
            "model.layers.1.mlp.down_proj": (4096, 14336),
            "model.layers.1.mlp.gate_proj": (14336, 4096),
            "model.layers.1.mlp.up_proj": (14336, 4096),
        }
        config = {"num_hidden_layers": 32, "_oq_use_budget_plan": True}
        plan = _build_quant_plan(
            named_shapes, config, 4, target_bpw=4.6, hard_cap_bpw=4.7
        )
        assert plan.effective_bpw <= 4.7
        assert plan.boost_map

    def test_format_size(self):
        assert "GB" in _format_size(5 * 1024**3)
        assert "MB" in _format_size(500 * 1024**2)
        assert "KB" in _format_size(500 * 1024)


# =============================================================================
# Test level-specific budget plan
# =============================================================================


class TestLevelBudgetPlan:
    """Tests for per-level target_bpw and budget plan activation."""

    def test_bpw_targets_for_level_returns_correct_values(self):
        assert _bpw_targets_for_level(2.5) == (3.1, 3.3)
        assert _bpw_targets_for_level(2.7) == (3.25, 3.35)
        assert _bpw_targets_for_level(3) == (3.5, 3.7)
        assert _bpw_targets_for_level(3.5) == (3.8, 4.0)
        assert _bpw_targets_for_level(4) == (4.6, 4.7)
        assert _bpw_targets_for_level(5) == (5.5, 5.7)
        assert _bpw_targets_for_level(6) == (6.5, 6.7)
        assert _bpw_targets_for_level(2.8) is None

    def test_oq2_fractional_base_bits_are_2(self):
        assert _LEVEL_BITS[2.5] == 2
        assert _LEVEL_BITS[2.7] == 2

    def test_oq35_mandatory_expert_down_proj_boost(self):
        """oQ3.5 protects routed expert down_proj above base bits
        even with negligible sensitivity scores."""
        oq_level = 3.5
        named_shapes = {
            "model.layers.0.mlp.switch_mlp.down_proj": (8, 256, 256),
            "model.layers.0.mlp.switch_mlp.gate_proj": (8, 256, 256),
            "model.layers.0.self_attn.q_proj": (64, 64),
        }
        config = {
            "num_hidden_layers": 1,
            "num_experts": 8,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": {"0": 0.01},
        }
        target, cap = _OQ_BPW_TARGETS[oq_level]
        plan = _build_quant_plan(
            named_shapes, config, oq_level, target_bpw=target, hard_cap_bpw=cap
        )
        boost = plan.boost_map.get("model.layers.0.mlp.switch_mlp.down_proj")
        assert boost is not None
        assert boost["bits"] == 4

    def test_oq35_predicate_floor_for_expert_down_proj(self):
        """The non-budget predicate floor mirrors the oQ3.5 mandatory boost."""
        config = {
            "num_hidden_layers": 32,
            "num_experts": 8,
            "hidden_size": 1024,
        }
        result = universal_quant_predicate(
            "model.layers.5.mlp.switch_mlp.down_proj", None, config, 3.5
        )
        assert isinstance(result, dict)
        assert result["bits"] == 4

    @pytest.mark.parametrize("oq_level", [2.5, 2.7])
    def test_oq2x_predicate_keeps_routed_down_proj_at_base_without_plan(self, oq_level):
        """oQ2.5/oQ2.7 use budget-planned routed boosts."""
        config = {
            "num_hidden_layers": 32,
            "num_experts": 8,
            "hidden_size": 1024,
        }
        result = universal_quant_predicate(
            "model.layers.5.mlp.switch_mlp.down_proj", None, config, oq_level
        )
        assert result is True

    def test_bpw_targets_for_level_returns_none_for_minimal(self):
        assert _bpw_targets_for_level(8) is None

    def test_level_bits_covers_all_oq_levels(self):
        for level in OQ_LEVELS:
            assert level in _LEVEL_BITS

    def test_budget_plan_oq2_enabled(self):
        assert 2 in _OQ_BPW_TARGETS
        assert _bpw_targets_for_level(2) == (2.8, 3.0)

    def test_oq2_fractional_levels_use_routed_layer_boosts(self):
        for level in (2.5, 2.7):
            assert level in OQ_LEVELS
            assert level in _ROUTED_LAYER_BOOST_LEVELS
            assert level not in _LEVEL_EXPERT_DOWN_BOOST
        assert 2.8 not in OQ_LEVELS
        assert 2.8 not in _ROUTED_LAYER_BOOST_LEVELS

    def test_budget_plan_oq8_not_enabled(self):
        assert 8 not in _OQ_BPW_TARGETS

    def test_budget_plan_oq3_respects_cap(self):
        named_shapes = {
            "lm_head": (4096, 4096),
            "model.layers.0.self_attn.v_proj": (4096, 4096),
            "model.layers.0.self_attn.o_proj": (4096, 4096),
            "model.layers.1.mlp.down_proj": (4096, 14336),
            "model.layers.1.mlp.gate_proj": (14336, 4096),
            "model.layers.1.mlp.up_proj": (14336, 4096),
        }
        config = {"num_hidden_layers": 32, "_oq_use_budget_plan": True}
        plan = _build_quant_plan(
            named_shapes, config, 3, target_bpw=3.5, hard_cap_bpw=3.7
        )
        assert plan.effective_bpw <= 3.7

    @pytest.mark.parametrize(
        "oq_level,target,cap",
        [(3, 3.5, 3.7), (4, 4.6, 4.7), (5, 5.5, 5.7)],
    )
    def test_budget_plan_respects_level_cap(self, oq_level, target, cap):
        named_shapes = {
            "lm_head": (4096, 4096),
            "model.layers.0.self_attn.v_proj": (4096, 4096),
            "model.layers.0.self_attn.o_proj": (4096, 4096),
            "model.layers.1.mlp.down_proj": (4096, 14336),
            "model.layers.1.mlp.gate_proj": (14336, 4096),
            "model.layers.1.mlp.up_proj": (14336, 4096),
        }
        config = {"num_hidden_layers": 32, "_oq_use_budget_plan": True}
        plan = _build_quant_plan(
            named_shapes,
            config,
            oq_level,
            target_bpw=target,
            hard_cap_bpw=cap,
        )
        assert plan.effective_bpw <= cap

    def test_build_quant_plan_mandatory_lm_head(self):
        # lm_head gets mandatory 8-bit boost (consensus-critical)
        named_shapes = {"lm_head": (4096, 32000)}
        for i in range(32):
            named_shapes[f"model.layers.{i}.self_attn.v_proj"] = (4096, 4096)
            named_shapes[f"model.layers.{i}.self_attn.q_proj"] = (4096, 4096)
            named_shapes[f"model.layers.{i}.mlp.gate_proj"] = (14336, 4096)
            named_shapes[f"model.layers.{i}.mlp.up_proj"] = (14336, 4096)
            named_shapes[f"model.layers.{i}.mlp.down_proj"] = (4096, 14336)
        config = {"num_hidden_layers": 32, "_oq_use_budget_plan": True}
        plan = _build_quant_plan(
            named_shapes, config, 4, target_bpw=4.6, hard_cap_bpw=4.7
        )
        assert "lm_head" in plan.boost_map
        assert plan.boost_map["lm_head"]["bits"] == 8

    def test_build_quant_plan_sensitivity_driven(self):
        # Sensitive layers get more bits, insensitive get fewer
        named_shapes = {"lm_head": (4096, 32000)}
        for i in range(32):
            named_shapes[f"model.layers.{i}.self_attn.v_proj"] = (4096, 4096)
            named_shapes[f"model.layers.{i}.self_attn.q_proj"] = (4096, 4096)
            named_shapes[f"model.layers.{i}.mlp.gate_proj"] = (14336, 4096)
            named_shapes[f"model.layers.{i}.mlp.up_proj"] = (14336, 4096)
            named_shapes[f"model.layers.{i}.mlp.down_proj"] = (4096, 14336)
        sensitivity = {"0": 0.05, "1": 0.003, "31": 0.002}
        config = {
            "num_hidden_layers": 32,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": sensitivity,
        }
        plan = _build_quant_plan(
            named_shapes, config, 4, target_bpw=4.6, hard_cap_bpw=4.7
        )
        # L0 (highest sensitivity) should get boosted
        l0_boosts = [k for k in plan.boost_map if "layers.0." in k]
        assert len(l0_boosts) > 0
        # L0 should get more bits than L1 (if L1 boosted at all)
        l0_bits = max(plan.boost_map[k]["bits"] for k in l0_boosts)
        l1_boosts = [k for k in plan.boost_map if "layers.1." in k]
        if l1_boosts:
            l1_bits = max(plan.boost_map[k]["bits"] for k in l1_boosts)
            assert l0_bits >= l1_bits

    def test_build_quant_plan_skips_routed_experts(self):
        # Routed experts should never be boosted
        named_shapes = {
            "lm_head": (4096, 32000),
            "model.layers.0.self_attn.v_proj": (4096, 4096),
            "model.layers.0.mlp.switch_mlp.gate_proj": (256, 512, 4096),
            "model.layers.0.mlp.switch_mlp.up_proj": (256, 512, 4096),
        }
        config = {
            "num_hidden_layers": 32,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": {"0": 0.05},
        }
        plan = _build_quant_plan(
            named_shapes, config, 4, target_bpw=4.6, hard_cap_bpw=4.7
        )
        for k in plan.boost_map:
            assert "switch_mlp" not in k

    @pytest.mark.parametrize("oq_level", [2.5, 2.7])
    def test_oq2x_boosts_routed_down_proj_by_layer_sensitivity(self, oq_level):
        """oQ2.5/oQ2.7 boost routed projections by whole layer modules."""
        named_shapes = {}
        for i in range(2):
            named_shapes[f"model.layers.{i}.ffn.switch_mlp.gate_proj"] = (8, 64, 64)
            named_shapes[f"model.layers.{i}.ffn.switch_mlp.up_proj"] = (8, 64, 64)
            named_shapes[f"model.layers.{i}.ffn.switch_mlp.down_proj"] = (8, 64, 64)
        config = {
            "num_hidden_layers": 2,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": {"0": 1.0, "1": 0.1},
        }
        plan = _build_quant_plan(
            named_shapes, config, oq_level, target_bpw=2.65, hard_cap_bpw=2.7
        )
        assert plan.boost_map["model.layers.0.ffn.switch_mlp.down_proj"]["bits"] == 3
        assert "model.layers.1.ffn.switch_mlp.down_proj" not in plan.boost_map

    @pytest.mark.parametrize("oq_level", [2.5, 2.7])
    def test_oq2x_boosts_gate_up_pair_after_routed_down_proj(self, oq_level):
        """After routed w2/down_proj, oQ2.5/oQ2.7 boost gate+up as a pair."""
        named_shapes = {
            "model.layers.0.ffn.switch_mlp.gate_proj": (8, 64, 64),
            "model.layers.0.ffn.switch_mlp.up_proj": (8, 64, 64),
            "model.layers.0.ffn.switch_mlp.down_proj": (8, 64, 64),
        }
        config = {
            "num_hidden_layers": 1,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": {"0": 1.0},
        }
        plan = _build_quant_plan(
            named_shapes, config, oq_level, target_bpw=3.4, hard_cap_bpw=3.6
        )
        assert plan.boost_map["model.layers.0.ffn.switch_mlp.down_proj"]["bits"] == 3
        assert plan.boost_map["model.layers.0.ffn.switch_mlp.gate_proj"]["bits"] == 3
        assert plan.boost_map["model.layers.0.ffn.switch_mlp.up_proj"]["bits"] == 3

    @pytest.mark.parametrize("oq_level", [2.5, 2.7])
    def test_oq2x_prioritizes_dense_greedy_before_routed_fallback(self, oq_level):
        """oQ2.5/oQ2.7 spend target budget on dense sensitivity first."""
        named_shapes = {
            "model.layers.0.ffn.switch_mlp.down_proj": (8, 64, 64),
            "model.layers.0.mlp.gate_proj": (8, 64, 64),
        }
        config = {
            "num_hidden_layers": 2,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": {"0": 1.0, "1": 0.1},
        }
        plan = _build_quant_plan(
            named_shapes, config, oq_level, target_bpw=3.0, hard_cap_bpw=3.01
        )
        assert plan.boost_map["model.layers.0.mlp.gate_proj"]["bits"] == 3
        assert "model.layers.0.ffn.switch_mlp.down_proj" not in plan.boost_map

    def test_oq2_budget_plan_respects_cap(self):
        """oQ2 with budget plan should stay within hard cap."""
        named_shapes = {"lm_head": (4096, 32000)}
        for i in range(32):
            named_shapes[f"model.layers.{i}.self_attn.v_proj"] = (4096, 4096)
            named_shapes[f"model.layers.{i}.self_attn.q_proj"] = (4096, 4096)
            named_shapes[f"model.layers.{i}.mlp.gate_proj"] = (14336, 4096)
            named_shapes[f"model.layers.{i}.mlp.up_proj"] = (14336, 4096)
            named_shapes[f"model.layers.{i}.mlp.down_proj"] = (4096, 14336)
        sensitivity = {str(i): 0.1 / (i + 1) for i in range(32)}
        config = {
            "num_hidden_layers": 32,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": sensitivity,
        }
        plan = _build_quant_plan(
            named_shapes, config, 2, target_bpw=2.8, hard_cap_bpw=3.0
        )
        assert plan.effective_bpw <= 3.0
        assert plan.boost_map

    def test_oq2_moe_protection_floor(self):
        """oQ2 MoE: protection floor boosts attention, experts stay 2bit."""
        named_shapes = {"lm_head": (4096, 32000)}
        n_layers = 52
        n_experts = 64
        for i in range(n_layers):
            named_shapes[f"model.layers.{i}.self_attn.v_proj"] = (1024, 4096)
            named_shapes[f"model.layers.{i}.self_attn.q_proj"] = (4096, 4096)
            named_shapes[f"model.layers.{i}.self_attn.k_proj"] = (1024, 4096)
            named_shapes[f"model.layers.{i}.self_attn.o_proj"] = (4096, 1024)
        for i in range(n_layers):
            for e in range(n_experts):
                named_shapes[f"model.layers.{i}.mlp.experts.{e}.down_proj"] = (
                    4096,
                    1024,
                )
                named_shapes[f"model.layers.{i}.mlp.experts.{e}.up_proj"] = (1024, 4096)
                named_shapes[f"model.layers.{i}.mlp.experts.{e}.gate_proj"] = (
                    1024,
                    4096,
                )
        sensitivity = {str(i): 0.1 / (i + 1) for i in range(n_layers)}
        config = {
            "num_hidden_layers": n_layers,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": sensitivity,
        }
        plan = _build_quant_plan(
            named_shapes, config, 2, target_bpw=2.8, hard_cap_bpw=3.0
        )
        assert plan.effective_bpw <= 3.0
        # Attention tensors should be boosted via protection floor
        attn_boosts = [k for k in plan.boost_map if "self_attn" in k]
        assert len(attn_boosts) > 0, "Expected attention protection floor boosts"
        # Routed experts should NOT be boosted
        expert_boosts = [k for k in plan.boost_map if "experts" in k]
        assert len(expert_boosts) == 0, "Routed experts should stay at base bits"

    def test_oq2_moe_protection_floor_switch_mlp(self):
        """oQ2 MoE with switch_mlp naming: experts stay 2bit, attention boosted."""
        named_shapes = {"lm_head": (4096, 32000)}
        n_layers = 52
        for i in range(n_layers):
            named_shapes[f"backbone.layers.{i}.mixer.q_proj"] = (4096, 2688)
            named_shapes[f"backbone.layers.{i}.mixer.k_proj"] = (1024, 2688)
            named_shapes[f"backbone.layers.{i}.mixer.v_proj"] = (1024, 2688)
            named_shapes[f"backbone.layers.{i}.mixer.in_proj"] = (10304, 2688)
            named_shapes[f"backbone.layers.{i}.mixer.out_proj"] = (2688, 4096)
            named_shapes[f"backbone.layers.{i}.mixer.shared_experts.up_proj"] = (
                3712,
                2688,
            )
            named_shapes[f"backbone.layers.{i}.mixer.shared_experts.down_proj"] = (
                2688,
                3712,
            )
        for i in range(n_layers):
            named_shapes[f"backbone.layers.{i}.mixer.switch_mlp.fc1"] = (
                128,
                1856,
                2688,
            )
            named_shapes[f"backbone.layers.{i}.mixer.switch_mlp.fc2"] = (
                128,
                2688,
                1856,
            )
        sensitivity = {str(i): 0.1 / (i + 1) for i in range(n_layers)}
        config = {
            "num_hidden_layers": n_layers,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": sensitivity,
        }
        plan = _build_quant_plan(
            named_shapes, config, 2, target_bpw=2.8, hard_cap_bpw=3.0
        )
        assert (
            plan.effective_bpw >= 2.7
        ), f"Expected bpw >= 2.7, got {plan.effective_bpw:.2f}"
        assert plan.effective_bpw <= 3.0
        # Attention should be boosted via protection floor
        attn_boosts = [k for k in plan.boost_map if "q_proj" in k or "v_proj" in k]
        assert len(attn_boosts) > 0, "Expected attention protection floor boosts"
        # switch_mlp experts should NOT be boosted
        expert_boosts = [k for k in plan.boost_map if "switch_mlp" in k]
        assert len(expert_boosts) == 0, "Routed experts should stay at base bits"


# =============================================================================
# Test _forward_layer tuple unwrapping
# =============================================================================


class TestForwardLayer:
    """Test _forward_layer tuple unwrapping for Gemma4/Hunyuan-style models."""

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_returns_tensor_when_block_returns_tensor(self):
        tensor = mx.ones((2, 4, 8))
        block = lambda x, mask, cache, pos: x * 2
        result = _forward_layer(block, tensor, None, None)
        assert isinstance(result, mx.array)
        assert result.shape == (2, 4, 8)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_unwraps_3tuple_gemma4_style(self):
        tensor = mx.ones((2, 4, 8))
        block = lambda x, mask, cache, pos: (x * 2, None, 0)
        result = _forward_layer(block, tensor, None, None)
        assert isinstance(result, mx.array)
        assert result.shape == (2, 4, 8)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_unwraps_2tuple_hunyuan_style(self):
        tensor = mx.ones((2, 4, 8))
        block = lambda x, mask, cache, pos: (x * 2, None)
        result = _forward_layer(block, tensor, None, None)
        assert isinstance(result, mx.array)
        assert result.shape == (2, 4, 8)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_returns_none_when_all_signatures_fail(self):
        def bad_block(*args, **kwargs):
            raise TypeError("unsupported")

        result = _forward_layer(bad_block, mx.ones((2, 4)), None, None)
        assert result is None

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_fallback_signature_with_tuple(self):
        tensor = mx.ones((2, 4, 8))

        def block_only_one_arg(x):
            return (x * 3, {"cache": True})

        result = _forward_layer(block_only_one_arg, tensor, None, None)
        assert isinstance(result, mx.array)

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_glm_state_signature_returns_aux(self):
        tensor = mx.ones((2, 4, 8))
        seen = []

        def glm_block(x, mask, cache, prev_topk):
            seen.append((mask, cache, prev_topk))
            return x + 1, "next_topk"

        state = {"kind": "glm_moe_dsa", "prev_topk_indices": "prev_topk"}
        result, aux = _forward_layer_result(glm_block, tensor, "mask", state)

        assert isinstance(result, mx.array)
        assert aux == "next_topk"
        assert seen == [("mask", None, "prev_topk")]


# =============================================================================
# Test _LazyTensorIndex
# =============================================================================


def _write_safetensors(path, tensors):
    """Write a minimal safetensors file from {name: np.ndarray} dict.

    Values can be np.ndarray (auto-dtype) or (raw_bytes, shape, sf_dtype) tuples
    for dtypes numpy doesn't support (F8_E4M3, F8_E8M0, I8)."""
    import json
    import struct

    header = {}
    data_parts = []
    offset = 0
    dtype_map = {np.float16: "F16", np.float32: "F32", np.dtype("<f2"): "F16"}
    for name, val in tensors.items():
        if isinstance(val, tuple):
            raw, shape, sf_dtype = val
        else:
            raw = val.tobytes()
            shape = list(val.shape)
            sf_dtype = dtype_map.get(val.dtype, "F16")
        header[name] = {
            "dtype": sf_dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for part in data_parts:
            f.write(part)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestLazyTensorIndex:
    @pytest.fixture
    def sf_file(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        tensors = {
            "layer.0.weight": np.random.randn(4, 8).astype(np.float16),
            "layer.1.weight": np.random.randn(2, 8).astype(np.float16),
            "embed.weight": np.random.randn(16, 8).astype(np.float16),
        }
        _write_safetensors(str(path), tensors)
        return str(path), tensors

    def test_keys_and_len(self, sf_file):
        path, tensors = sf_file
        idx = _LazyTensorIndex([path])
        assert set(idx.keys()) == set(tensors.keys())
        assert len(idx) == len(tensors)

    def test_contains(self, sf_file):
        path, _ = sf_file
        idx = _LazyTensorIndex([path])
        assert "layer.0.weight" in idx
        assert "nonexistent" not in idx

    def test_getitem_roundtrip(self, sf_file):
        path, tensors = sf_file
        idx = _LazyTensorIndex([path])
        for name, expected in tensors.items():
            result = idx[name]
            assert isinstance(result, mx.array)
            np.testing.assert_allclose(
                np.array(result.astype(mx.float32)),
                expected.astype(np.float32),
                atol=1e-3,
            )

    def test_pop_returns_mx_array(self, sf_file):
        path, tensors = sf_file
        idx = _LazyTensorIndex([path])
        result = idx.pop("layer.0.weight")
        assert isinstance(result, mx.array)
        assert "layer.0.weight" not in idx

    def test_pop_missing_raises(self, sf_file):
        path, _ = sf_file
        idx = _LazyTensorIndex([path])
        with pytest.raises(KeyError):
            idx.pop("nonexistent")

    def test_pop_missing_default(self, sf_file):
        path, _ = sf_file
        idx = _LazyTensorIndex([path])
        assert idx.pop("nonexistent", None) is None

    def test_setitem_override(self, sf_file):
        path, _ = sf_file
        idx = _LazyTensorIndex([path])
        override = mx.ones((3, 3))
        idx["custom_key"] = override
        assert "custom_key" in idx
        assert "custom_key" in list(idx.keys())

    def test_iter_includes_overrides(self, sf_file):
        path, tensors = sf_file
        idx = _LazyTensorIndex([path])
        idx["override_key"] = mx.zeros((2,))
        all_keys = list(idx)
        assert "override_key" in all_keys
        for k in tensors:
            assert k in all_keys

    def test_delitem(self, sf_file):
        path, _ = sf_file
        idx = _LazyTensorIndex([path])
        del idx["layer.0.weight"]
        assert "layer.0.weight" not in idx


# =============================================================================
# Test _quantize_chunked
# =============================================================================


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestQuantizeChunked:
    def test_matches_mx_quantize(self):
        w = mx.random.normal((32, 64))
        mx.eval(w)
        qw_ref, scales_ref, *rest_ref = mx.quantize(w, group_size=64, bits=4)
        biases_ref = rest_ref[0] if rest_ref else None

        qw, scales, biases = _quantize_chunked(w, group_size=64, bits=4, mode="affine")

        np.testing.assert_array_equal(np.array(qw), np.array(qw_ref))
        np.testing.assert_array_equal(np.array(scales), np.array(scales_ref))
        if biases is not None and biases_ref is not None:
            np.testing.assert_array_equal(np.array(biases), np.array(biases_ref))

    def test_output_shapes(self):
        w = mx.random.normal((16, 128))
        mx.eval(w)
        qw, scales, biases = _quantize_chunked(w, group_size=64, bits=4, mode="affine")
        assert qw.shape[0] == 16
        assert scales.shape[0] == 16

    def test_uniform_importance_matches_mx_quantize(self):
        w = mx.random.normal((8, 64)).astype(mx.float16)
        mx.eval(w)
        qw_ref, scales_ref, *rest_ref = mx.quantize(
            w, group_size=64, bits=4, mode="affine"
        )
        biases_ref = rest_ref[0] if rest_ref else None

        qw, scales, biases = _quantize_chunked(
            w,
            group_size=64,
            bits=4,
            mode="affine",
            importance=mx.ones((64,), dtype=mx.float32),
        )

        np.testing.assert_array_equal(np.array(qw), np.array(qw_ref))
        np.testing.assert_array_equal(np.array(scales), np.array(scales_ref))
        np.testing.assert_array_equal(np.array(biases), np.array(biases_ref))

    def test_weighted_importance_reduces_weighted_error(self):
        vals = np.array(
            [
                8.0,
                -0.9835515,
                -1.0129286,
                -0.9208264,
                -0.933982,
                -0.96833235,
                -1.1101755,
                -0.99739856,
                -0.96581566,
                -0.9498019,
                -0.62327445,
                0.04132598,
            ]
            + [0.0] * 52,
            dtype=np.float32,
        )
        w = mx.array(vals.reshape(1, 64), dtype=mx.float16)
        importance = np.full((64,), 0.1, dtype=np.float32)
        importance[1:10] = 100.0
        imp = mx.array(importance)

        qw_ref, scales_ref, biases_ref = mx.quantize(
            w, group_size=64, bits=2, mode="affine"
        )
        y_ref = mx.dequantize(
            qw_ref,
            scales_ref,
            biases_ref,
            group_size=64,
            bits=2,
            mode="affine",
        )

        qw, scales, biases = _quantize_chunked(
            w,
            group_size=64,
            bits=2,
            mode="affine",
            importance=imp,
        )
        y_weighted = mx.dequantize(
            qw,
            scales,
            biases,
            group_size=64,
            bits=2,
            mode="affine",
        )
        ref_err = mx.sum(((w - y_ref) ** 2) * imp)
        weighted_err = mx.sum(((w - y_weighted) ** 2) * imp)
        mx.eval(ref_err, weighted_err)
        assert weighted_err.item() < ref_err.item()

    def test_weighted_3d_expert_importance_chunked(self, monkeypatch):
        monkeypatch.setattr("omlx.oq._QUANTIZE_CHUNK_BYTES", 128)
        w = mx.random.normal((4, 2, 64)).astype(mx.float16)
        importance = mx.arange(4 * 64, dtype=mx.float32).reshape(4, 64) + 1.0
        mx.eval(w, importance)

        qw, scales, biases = _quantize_chunked(
            w,
            group_size=64,
            bits=4,
            mode="affine",
            importance=importance,
        )

        assert qw.shape == (4, 2, 8)
        assert scales.shape == (4, 2, 1)
        assert biases.shape == (4, 2, 1)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestOQImatrixCollector:
    def test_capture_wrapper_delegates_without_init_recursion(self):
        module = nn.Linear(4, 3, bias=False)
        collector = OQImatrixCollector()

        wrapper = _ImatrixCaptureWrapper(module, "linear", collector)
        y = wrapper(mx.ones((2, 4)))
        mx.eval(y)

        assert wrapper.weight is module.weight
        assert "linear" in collector.entries
        assert collector.entries["linear"].counts[0] == 2

    def test_switch_topk_capture_accumulates_per_expert(self):
        class SwitchModule:
            weight = mx.zeros((3, 2, 4))

        collector = OQImatrixCollector()
        x = mx.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=mx.float32,
        )
        indices = mx.array([[0, 1], [1, 2]], dtype=mx.int32)

        collector.collect_switch("switch", SwitchModule(), x, indices)

        entry = collector.entries["switch"]
        expected_sq = np.asarray(
            [
                [1.0, 4.0, 9.0, 16.0],
                [26.0, 40.0, 58.0, 80.0],
                [25.0, 36.0, 49.0, 64.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(entry.counts, np.array([1, 2, 1]))
        np.testing.assert_allclose(entry.in_sum2, expected_sq)

    def test_expert_coverage_stats_gate_adaptive_collection(self):
        insufficient = {
            "experts": OQImatrixEntry(
                in_sum2=np.zeros((4, 8), dtype=np.float32),
                counts=np.array([0, 16, 32, 48], dtype=np.int64),
            )
        }
        stats = _imatrix_expert_coverage_stats(insufficient)

        assert stats["has_expert_counts"] is True
        assert stats["zero_count_experts"] == 1
        assert _imatrix_expert_coverage_sufficient(stats) is False

        sufficient = {
            "experts": OQImatrixEntry(
                in_sum2=np.zeros((4, 8), dtype=np.float32),
                counts=np.array([16, 16, 32, 48], dtype=np.int64),
            )
        }
        stats = _imatrix_expert_coverage_stats(sufficient)

        assert stats["zero_count_experts"] == 0
        assert stats["p05_count"] >= 16
        assert _imatrix_expert_coverage_sufficient(stats) is True

    def test_expert_coverage_requires_counts_for_moe_config(self):
        dense_only = {
            "dense": OQImatrixEntry(
                in_sum2=np.zeros((8,), dtype=np.float32),
                counts=np.array([1024], dtype=np.int64),
            )
        }
        stats = _imatrix_expert_coverage_stats(dense_only)

        assert stats["has_expert_counts"] is False
        assert _imatrix_expert_coverage_sufficient(stats) is True
        assert (
            _imatrix_expert_coverage_sufficient(stats, require_expert_counts=True)
            is False
        )
        assert _config_expects_moe_expert_counts({"n_routed_experts": 256}) is True
        assert _imatrix_requires_expert_counts({"n_routed_experts": 256}, 0) is False
        assert _imatrix_requires_expert_counts({"n_routed_experts": 256}, 3) is True

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
    def test_quantized_switch_linear_capture_uses_logical_input_dims(self):
        class QuantizedSwitchLinear:
            weight = mx.zeros((3, 2, 1), dtype=mx.uint32)

            @property
            def input_dims(self):
                return 4

            @property
            def num_experts(self):
                return 3

        module = QuantizedSwitchLinear()
        collector = OQImatrixCollector()
        x = mx.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        indices = mx.array([[[0, 2], [2, 1]]], dtype=mx.int32)

        assert collector._is_capture_module(module) is True
        collector.collect_switch("experts", module, x, indices)

        entry = collector.entries["experts"]
        np.testing.assert_array_equal(entry.counts, np.array([1, 1, 2]))
        np.testing.assert_allclose(entry.in_sum2[0], np.array([1.0, 4.0, 9.0, 16.0]))
        np.testing.assert_allclose(entry.in_sum2[1], np.array([25.0, 36.0, 49.0, 64.0]))
        np.testing.assert_allclose(entry.in_sum2[2], np.array([26.0, 40.0, 58.0, 80.0]))


class TestOQECalibrationData:
    @staticmethod
    def _rough_est_tokens(text: str) -> int:
        total = 0.0
        for ch in text:
            o = ord(ch)
            if 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF:
                total += 1 / 1.3
            elif ch.isspace():
                continue
            elif o < 128:
                total += 1 / 4
            else:
                total += 1 / 2
        return int(total)

    def test_oqe_calibration_json_is_balanced_and_multilingual(self):
        p = Path(__file__).parent.parent / "omlx" / "oqe_calibration_data.json"
        with open(p, encoding="utf-8") as f:
            data = json.load(f)

        required = {
            "tool_calling",
            "chat",
            "mixed",
            "reasoning",
            "code",
            "en",
            "ko",
            "zh",
            "ja",
            "bartowski",
        }
        assert required.issubset(data.keys())

        tokens = {
            key: sum(self._rough_est_tokens(text) for text in data[key])
            for key in required
        }
        total = sum(tokens.values())
        shares = {key: value / total for key, value in tokens.items()}

        multilingual_share = sum(shares[key] for key in ("en", "ko", "zh", "ja"))
        assert multilingual_share >= 0.25
        assert shares["tool_calling"] <= 0.18
        assert max(shares.values()) <= 0.18


# =============================================================================
# Test _TrackedTensor
# =============================================================================


class TestTrackedTensor:
    def test_shape_preserved(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        assert t.shape == (4, 8)
        assert t.ndim == 2

    def test_reshape(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        r = t.reshape(2, 16)
        assert r.shape == (2, 16)
        assert r.transform == "reshape"

    def test_reshape_infer_dim(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        r = t.reshape(-1, 4)
        assert r.shape == (8, 4)

    def test_getitem_int(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        r = t[0]
        assert r.shape == (8,)

    def test_getitem_slice(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        r = t[1:3]
        assert r.shape == (2, 8)
        assert r.transform == "slice"

    def test_getitem_half_split(self):
        t = _TrackedTensor((256, 2048, 384), "F16", sources=["gate_up"])
        first = t[:, :1024, :]
        assert first.shape == (256, 1024, 384)
        assert first.transform == "split_0_2"
        assert first.axis == 1
        second = t[:, 1024:, :]
        assert second.transform == "split_1_2"
        # bare-slice path (axis 0)
        t2 = _TrackedTensor((8, 4), "F16", sources=["a"])
        assert t2[:4].transform == "split_0_2"

    def test_getitem_non_half_stays_slice(self):
        t = _TrackedTensor((256, 2048, 384), "F16", sources=["a"])
        assert t[:, :512, :].transform == "slice"

    def test_getitem_none_broadcast(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        r = t[:, None, :]
        assert r.shape == (4, 1, 8)

    def test_astype(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        r = t.astype("BF16")
        assert r.dtype == "BF16"
        assert r.shape == (4, 8)

    def test_arithmetic_preserves_sources(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        r = t + 1.0
        assert r.sources == ["a"]
        assert r.transform == "add"

    def test_transpose_property(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        r = t.T
        assert r.shape == (8, 4)

    def test_moveaxis_method(self):
        t = _TrackedTensor((2, 3, 4), "F16", sources=["a"])
        assert t.moveaxis(0, 2).shape == (3, 4, 2)
        assert t.moveaxis(0, 2).transform == "moveaxis_0_2"
        # negative axes normalized
        assert t.moveaxis(-1, 0).transform == "moveaxis_2_0"

    def test_transpose_method(self):
        t = _TrackedTensor((2, 3, 4), "F16", sources=["a"])
        assert t.transpose(2, 0, 1).shape == (4, 2, 3)
        assert t.transpose(2, 0, 1).transform == "transpose_2_0_1"
        # no-args reverses all axes
        assert t.transpose().transform == "transpose_2_1_0"

    def test_swapaxes_method(self):
        t = _TrackedTensor((2, 3, 4), "F16", sources=["a"])
        r = t.swapaxes(-1, -2)
        assert r.shape == (2, 4, 3)
        assert r.transform == "transpose_0_2_1"
        assert r.sources == ["a"]
        assert r.recipe == [("transpose", (0, 2, 1))]

    def test_expand_dims_method(self):
        t = _TrackedTensor((2, 3), "F16", sources=["a"])
        r = t.expand_dims(axis=0)
        assert r.shape == (1, 2, 3)
        assert r.transform == "expand_dims"
        assert r.recipe == [("expand_dims", (0,))]

    def test_expand_dims_multiple_axes(self):
        t = _TrackedTensor((2, 3), "F16", sources=["a"])
        r = t.expand_dims(axis=(0, -2))
        assert r.shape == (1, 2, 1, 3)
        assert r.recipe == [("expand_dims", (0, 2))]

    def test_getitem_ellipsis_half_split(self):
        # Sanitize patterns like gate_up[..., :mid, :] must round-trip through
        # the tracked-tensor dry run so streaming discovery covers low-RAM
        # quantization paths (see #1204).
        t = _TrackedTensor((256, 2048, 384), "F16", sources=["gate_up"])
        first = t[..., :1024, :]
        assert first.shape == (256, 1024, 384)
        assert first.transform == "split_0_2"
        assert first.axis == 1
        second = t[..., 1024:, :]
        assert second.transform == "split_1_2"

    def test_getitem_ellipsis_trailing(self):
        t = _TrackedTensor((4, 8, 16), "F16", sources=["a"])
        r = t[..., :4]
        assert r.shape == (4, 8, 4)
        assert r.transform == "slice"

    def test_getitem_ellipsis_zero_pad(self):
        # Ellipsis with no axes to fill (rank already covered)
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        r = t[..., 0:4, :]
        assert r.shape == (4, 8)

    def test_getitem_ellipsis_middle(self):
        t = _TrackedTensor((2, 3, 4, 5), "F16", sources=["a"])
        r = t[0, ..., 2:4]
        assert r.shape == (3, 4, 2)

    def test_getitem_multiple_ellipsis_raises(self):
        t = _TrackedTensor((2, 3, 4), "F16", sources=["a"])
        with pytest.raises(ValueError):
            t[..., :2, ...]

    def test_size_property(self):
        t = _TrackedTensor((4, 8), "F16", sources=["a"])
        assert t.size == 32


# =============================================================================
# Test _discover_sanitize_plan
# =============================================================================


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestDiscoverSanitizePlan:
    @pytest.fixture
    def sf_file(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(8, 8).astype(
                np.float16
            ),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(4, 8).astype(
                np.float16
            ),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(16, 8).astype(
                np.float16
            ),
            "model.embed_tokens.weight": np.random.randn(32, 8).astype(np.float16),
        }
        _write_safetensors(str(path), tensors)
        return str(path), tensors

    def test_passthrough_sanitize(self, sf_file):
        path, tensors = sf_file
        idx = _LazyTensorIndex([path])

        def identity_sanitize(weights):
            return weights

        plan = _discover_sanitize_plan(identity_sanitize, idx)
        assert plan is not None
        assert set(plan.keys()) == set(tensors.keys())
        for k, info in plan.items():
            assert info["transform"] == "passthrough"
            assert info["sources"] == [k]

    def test_rename_sanitize(self, sf_file):
        path, tensors = sf_file
        idx = _LazyTensorIndex([path])

        def rename_sanitize(weights):
            return {k.replace("model.", "renamed."): v for k, v in weights.items()}

        plan = _discover_sanitize_plan(rename_sanitize, idx)
        assert plan is not None
        for k in plan:
            assert k.startswith("renamed.")

    def test_drop_key_sanitize(self, sf_file):
        path, tensors = sf_file
        idx = _LazyTensorIndex([path])

        def drop_sanitize(weights):
            return {k: v for k, v in weights.items() if "embed" not in k}

        plan = _discover_sanitize_plan(drop_sanitize, idx)
        assert plan is not None
        assert "model.embed_tokens.weight" not in plan
        assert len(plan) == len(tensors) - 1

    def test_swapaxes_sanitize(self, sf_file):
        path, _tensors = sf_file
        idx = _LazyTensorIndex([path])

        def swapaxes_sanitize(weights):
            key = "model.layers.0.self_attn.q_proj.weight"
            return {"q_swapped.weight": weights[key].swapaxes(-1, -2)}

        plan = _discover_sanitize_plan(swapaxes_sanitize, idx)
        assert plan["q_swapped.weight"]["shape"] == (8, 8)
        assert plan["q_swapped.weight"]["transform"] == "transpose_1_0"

    def test_slice_sanitize_replays(self, sf_file):
        path, tensors = sf_file
        idx = _LazyTensorIndex([path])

        def slice_sanitize(weights):
            return {k: v[:, :3] for k, v in weights.items()}

        plan = _discover_sanitize_plan(slice_sanitize, idx)
        discovered = _DiscoveredPlan(plan, idx)
        key = "model.layers.0.self_attn.q_proj.weight"
        arr = discovered.pop(key)
        np.testing.assert_allclose(
            np.array(arr),
            tensors[key][:, :3],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_reshape_slice_swapaxes_sanitize_replays(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        tensor = np.arange(2 * 6 * 4, dtype=np.float16).reshape(12, 4)
        _write_safetensors(str(path), {"kv_b_proj.weight": tensor})
        idx = _LazyTensorIndex([str(path)])

        def glm_like_sanitize(weights):
            v = weights["kv_b_proj.weight"].reshape(2, 6, -1)
            return {
                "embed_q.weight": v[:, :2, :].swapaxes(-1, -2),
                "unembed_out.weight": v[:, 2:, :],
            }

        plan = _discover_sanitize_plan(glm_like_sanitize, idx)
        discovered = _DiscoveredPlan(plan, idx)

        expected = tensor.reshape(2, 6, 4)
        embed_q = discovered.pop("embed_q.weight")
        unembed_out = discovered.pop("unembed_out.weight")

        np.testing.assert_allclose(
            np.array(embed_q),
            expected[:, :2, :].swapaxes(-1, -2),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.array(unembed_out),
            expected[:, 2:, :],
            rtol=1e-3,
            atol=1e-3,
        )

    def test_expand_dims_sanitize_replays(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        tensor = np.arange(6, dtype=np.float16).reshape(2, 3)
        _write_safetensors(str(path), {"shared_down.weight": tensor})
        idx = _LazyTensorIndex([str(path)])

        def sanitize(weights):
            return {
                "shared_down.weight": mx.expand_dims(
                    weights["shared_down.weight"], axis=0
                )
            }

        plan = _discover_sanitize_plan(sanitize, idx)
        assert plan["shared_down.weight"]["shape"] == (1, 2, 3)
        assert plan["shared_down.weight"]["recipe"] == [("expand_dims", (0,))]

        result = _DiscoveredPlan(plan, idx).pop("shared_down.weight")
        np.testing.assert_array_equal(np.array(result), tensor[None, :, :])

    def test_minimax_shared_expert_sanitize_replays(self, tmp_path):
        path = tmp_path / "weights.safetensors"
        tensors = {}
        for name, start in (
            ("experts.0.w1.weight", 0),
            ("experts.1.w1.weight", 10),
            ("experts.0.w3.weight", 20),
            ("experts.1.w3.weight", 30),
            ("shared.gate.weight", 40),
            ("shared.up.weight", 50),
            ("experts.0.w2.weight", 60),
            ("experts.1.w2.weight", 70),
            ("shared.down.weight", 80),
        ):
            tensors[name] = (np.arange(6, dtype=np.float16) + start).reshape(2, 3)
        _write_safetensors(str(path), tensors)
        idx = _LazyTensorIndex([str(path)])

        def sanitize(weights):
            weights = dict(weights)
            gate = mx.stack(
                [weights.pop("experts.0.w1.weight"), weights.pop("experts.1.w1.weight")]
            )
            up = mx.stack(
                [weights.pop("experts.0.w3.weight"), weights.pop("experts.1.w3.weight")]
            )
            routed_gate_up = mx.concatenate([gate, up], axis=1)
            shared_gate_up = mx.expand_dims(
                mx.concatenate(
                    [
                        weights.pop("shared.gate.weight"),
                        weights.pop("shared.up.weight"),
                    ],
                    axis=0,
                ),
                axis=0,
            )
            down = mx.stack(
                [weights.pop("experts.0.w2.weight"), weights.pop("experts.1.w2.weight")]
            )
            shared_down = mx.expand_dims(weights.pop("shared.down.weight"), axis=0)
            return {
                "switch.gate_up.weight": mx.concatenate(
                    [routed_gate_up, shared_gate_up], axis=0
                ),
                "switch.down.weight": mx.concatenate([down, shared_down], axis=0),
            }

        plan = _discover_sanitize_plan(sanitize, idx)
        assert plan["switch.gate_up.weight"]["transform"] == "expr"
        assert plan["switch.gate_up.weight"]["shape"] == (3, 4, 3)
        assert plan["switch.down.weight"]["transform"] == "expr"
        assert plan["switch.down.weight"]["shape"] == (3, 2, 3)

        discovered = _DiscoveredPlan(plan, idx)
        gate = np.stack(
            [tensors["experts.0.w1.weight"], tensors["experts.1.w1.weight"]]
        )
        up = np.stack([tensors["experts.0.w3.weight"], tensors["experts.1.w3.weight"]])
        expected_gate_up = np.concatenate(
            [
                np.concatenate([gate, up], axis=1),
                np.concatenate(
                    [tensors["shared.gate.weight"], tensors["shared.up.weight"]],
                    axis=0,
                )[None, :, :],
            ],
            axis=0,
        )
        expected_down = np.concatenate(
            [
                np.stack(
                    [
                        tensors["experts.0.w2.weight"],
                        tensors["experts.1.w2.weight"],
                    ]
                ),
                tensors["shared.down.weight"][None, :, :],
            ],
            axis=0,
        )

        np.testing.assert_array_equal(
            np.array(discovered.pop("switch.gate_up.weight")), expected_gate_up
        )
        np.testing.assert_array_equal(
            np.array(discovered.pop("switch.down.weight")), expected_down
        )

    def test_conditional_mtp_norm_add_materializes_by_mean(self, tmp_path):
        path = tmp_path / "mtp_norms.safetensors"
        tensors = {
            "raw.weight": np.full((8,), 0.04, dtype=np.float16),
            "shifted.weight": np.full((8,), 1.27, dtype=np.float16),
        }
        _write_safetensors(str(path), tensors)
        idx = _LazyTensorIndex([str(path)])

        plan = {
            "raw.weight": {
                "sources": ["raw.weight"],
                "transform": "add_if_mean_lt_0_5",
                "shape": (8,),
                "axis": None,
            },
            "shifted.weight": {
                "sources": ["shifted.weight"],
                "transform": "add_if_mean_lt_0_5",
                "shape": (8,),
                "axis": None,
            },
        }
        discovered = _DiscoveredPlan(plan, idx)

        raw = discovered.pop("raw.weight")
        shifted = discovered.pop("shifted.weight")

        assert float(raw.astype(mx.float32)[0].item()) == pytest.approx(1.04, abs=1e-3)
        assert float(shifted.astype(mx.float32)[0].item()) == pytest.approx(
            1.27, abs=1e-3
        )


# =============================================================================
# Test _model_exceeds_ram guard
# =============================================================================


class TestModelExceedsRamGuard:
    """Tests for the OOM guard that skips memory-intensive paths when a model
    is larger than system RAM."""

    @pytest.fixture
    def sf_file(self, tmp_path):
        if not HAS_MLX:
            pytest.skip("mlx not available")
        from safetensors.numpy import save_file as np_save

        tensors = {
            "weight_a": np.zeros((128, 256), dtype=np.float32),
            "weight_b": np.zeros((64, 128), dtype=np.float32),
        }
        path = tmp_path / "test.safetensors"
        np_save(tensors, str(path))
        expected_bytes = (128 * 256 + 64 * 128) * 4
        return path, expected_bytes

    def test_lazy_index_nbytes_matches_tensor_sizes(self, sf_file):
        path, expected_bytes = sf_file
        idx = _LazyTensorIndex([path])
        assert idx.nbytes() == expected_bytes

    def test_guard_boundary(self, sf_file):
        """Guard uses strict > with _MAX_MODEL_RAM_FRACTION of system RAM."""
        path, expected_bytes = sf_file
        idx = _LazyTensorIndex([path])
        nbytes = idx.nbytes()
        # Exceeds when "system RAM" is small enough
        small_ram = int(nbytes / _MAX_MODEL_RAM_FRACTION) - 1
        assert nbytes > int(small_ram * _MAX_MODEL_RAM_FRACTION)
        # Does not exceed when system RAM is large
        large_ram = int(nbytes / _MAX_MODEL_RAM_FRACTION) + 1
        assert not (nbytes > int(large_ram * _MAX_MODEL_RAM_FRACTION))


class TestQuantProgressTotalBytes:
    def test_uses_logical_plan_when_larger_than_source(self, tmp_path):
        source = tmp_path / "model"
        source.mkdir()
        (source / "model.safetensors").write_bytes(b"x" * 100)

        class FakeWeights:
            _plan = {"large.weight": {"shape": (50, 4)}}

            def nbytes(self):
                return 100

        assert _progress_total_bytes(FakeWeights(), source) == 400


class TestBuildProxyForSensitivity:
    """Tests for the auto-built sensitivity proxy.

    The proxy is created when the source model exceeds available RAM and the
    user has not supplied a pre-quantized model via sensitivity_model_path.
    Without it, quantize_oq_streaming aborts with a RuntimeError.
    """

    def test_invokes_streaming_proxy_builder(self, tmp_path, monkeypatch):
        """Proxy build uses oQ's streaming writer, not mlx_lm.convert."""
        from omlx import oq as _oq

        calls = []

        def _fake_build(
            model_path,
            output_path,
            *,
            dtype,
            trust_remote_code=False,
            preserve_mtp=False,
        ):
            calls.append(
                (model_path, output_path, dtype, trust_remote_code, preserve_mtp)
            )
            output_path.mkdir()

        monkeypatch.setattr(_oq, "_build_streaming_proxy_for_sensitivity", _fake_build)
        proxy_dir = _build_proxy_for_sensitivity(
            str(tmp_path / "src_model"),
            dtype="bfloat16",
            trust_remote_code=True,
        )

        assert calls == [
            (str(tmp_path / "src_model"), proxy_dir, "bfloat16", True, False)
        ]
        assert proxy_dir.exists()

    def test_returns_path_under_system_temp(self, tmp_path):
        """Proxy lives under the system temp dir, not next to the source."""
        import tempfile

        from omlx import oq as _oq

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            _oq,
            "_build_streaming_proxy_for_sensitivity",
            lambda _model, output, **_kwargs: output.mkdir(),
        )
        try:
            proxy_dir = _build_proxy_for_sensitivity(
                str(tmp_path / "src_model"), dtype="bfloat16"
            )
        finally:
            monkeypatch.undo()
        # tempfile.gettempdir() is the system temp root (e.g. /tmp).
        assert str(proxy_dir).startswith(tempfile.gettempdir())
        assert proxy_dir.name.startswith("omlx_oq_proxy_")

    def test_caller_is_responsible_for_cleanup(self, tmp_path):
        """The helper does not auto-delete the proxy; caller cleans up."""
        from omlx import oq as _oq

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            _oq,
            "_build_streaming_proxy_for_sensitivity",
            lambda _model, output, **_kwargs: output.mkdir(),
        )
        try:
            proxy_dir = _build_proxy_for_sensitivity(
                str(tmp_path / "src_model"), dtype="bfloat16"
            )
        finally:
            monkeypatch.undo()
        # The directory should still exist after the helper returns.
        assert proxy_dir.exists()

    def test_propagates_dtype_argument(self, tmp_path):
        """dtype is forwarded so the proxy matches the target output dtype."""
        from omlx import oq as _oq

        captured = {}

        def _fake_build(_model, output, **kwargs):
            captured.update(kwargs)
            output.mkdir()

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(_oq, "_build_streaming_proxy_for_sensitivity", _fake_build)
        try:
            _build_proxy_for_sensitivity(str(tmp_path / "src_model"), dtype="float16")
        finally:
            monkeypatch.undo()
        assert captured["dtype"] == "float16"

    def test_working_dir_pins_proxy_to_output_volume(self, tmp_path):
        """working_dir sets where mkdtemp anchors the proxy.

        Critical on Linux where /tmp can be tmpfs (RAM-backed): the caller
        passes the output volume so the proxy lands on actual disk and the
        OOM-driven proxy build does not defeat itself.
        """
        anchor = tmp_path / "out_volume"
        anchor.mkdir()
        from omlx import oq as _oq

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            _oq,
            "_build_streaming_proxy_for_sensitivity",
            lambda _model, output, **_kwargs: output.mkdir(),
        )
        try:
            proxy_dir = _build_proxy_for_sensitivity(
                str(tmp_path / "src_model"),
                dtype="bfloat16",
                working_dir=str(anchor),
            )
        finally:
            monkeypatch.undo()
        assert proxy_dir.parent == anchor

    @pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
    def test_streaming_proxy_writes_loadable_quantized_config(self, tmp_path):
        """The streaming proxy can quantize from safetensors without convert()."""
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        out = tmp_path / "proxy"
        src.mkdir()
        (src / "config.json").write_text(
            json.dumps({"model_type": "llama", "num_hidden_layers": 1}),
            encoding="utf-8",
        )
        np_save(
            {
                "model.layers.0.self_attn.q_proj.weight": np.ones(
                    (8, 64), dtype=np.float16
                ),
                "model.layers.0.input_layernorm.weight": np.ones(
                    (64,), dtype=np.float16
                ),
            },
            str(src / "model.safetensors"),
        )

        with (
            patch("omlx.oq._build_model_sanitizer", return_value=None),
            patch("omlx.oq._build_non_quantizable_set", return_value=set()),
        ):
            _build_streaming_proxy_for_sensitivity(str(src), out, dtype="bfloat16")

        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert config["quantization"]["bits"] == _PROXY_QUANT_BITS
        assert config["quantization"]["group_size"] == _PROXY_QUANT_GROUP_SIZE
        assert (out / "model.safetensors").exists()


class TestSensitivityRequiredEnforcement:
    """Regression tests: quantize_oq_streaming must abort when sensitivity
    measurement cannot run, rather than silently producing an output that
    skipped the data-driven step.
    """

    def test_opt_out_with_model_exceeding_ram_raises(self, tmp_path, monkeypatch):
        """auto_proxy_sensitivity=False + model > RAM -> RuntimeError."""
        if not HAS_MLX:
            pytest.skip("mlx not available")
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        src.mkdir()
        np_save(
            {"w": np.zeros((128, 256), dtype=np.float32)}, str(src / "w.safetensors")
        )
        (src / "config.json").write_text('{"model_type": "llama"}')

        # Force OOM by pretending system has 0 bytes of RAM.
        from omlx import settings as _settings

        monkeypatch.setattr(_settings, "get_system_memory", lambda: 0)

        with pytest.raises(RuntimeError, match="auto_proxy_sensitivity is disabled"):
            quantize_oq_streaming(
                str(src),
                str(tmp_path / "out"),
                4,
                auto_proxy_sensitivity=False,
            )

    def test_streaming_discovery_failure_with_model_exceeding_ram_raises(
        self, tmp_path, monkeypatch
    ):
        """#1204: discovery failure + model > RAM must hard-fail. The old
        behaviour was a silent ``logger.error`` followed by the source weight
        names landing in the output, which loaded with "Received N parameters
        not in model".

        Sensitivity measurement runs before sanitize-plan discovery, so
        reaching the discovery block with the model over RAM means the
        auto-proxy sensitivity path has to succeed first; the proxy build and
        measurement are stubbed so the run gets as far as discovery."""
        if not HAS_MLX:
            pytest.skip("mlx not available")
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        src.mkdir()
        np_save(
            {"w": np.zeros((128, 256), dtype=np.float32)},
            str(src / "w.safetensors"),
        )
        (src / "config.json").write_text('{"model_type": "llama"}')

        from omlx import settings as _settings

        monkeypatch.setattr(_settings, "get_system_memory", lambda: 0)

        from omlx import oq as _oq

        # Stub the auto-proxy sensitivity path so the run reaches discovery.
        monkeypatch.setattr(
            _oq, "_build_proxy_for_sensitivity", lambda *a, **k: tmp_path / "proxy"
        )
        monkeypatch.setattr(
            _oq, "_measure_sensitivity_from_quantized_model", lambda *a, **k: {0: 0.1}
        )

        # Force a sanitize_fn that fails during the tracked-tensor dry run,
        # mimicking an indexing pattern _TrackedTensor cannot trace.
        def _broken_sanitize(weights):
            raise NotImplementedError("simulated unsupported indexing pattern")

        monkeypatch.setattr(
            _oq, "_build_model_sanitizer", lambda *a, **k: _broken_sanitize
        )

        with pytest.raises(
            RuntimeError, match="streaming sanitize-plan discovery failed"
        ):
            quantize_oq_streaming(
                str(src),
                str(tmp_path / "out"),
                4,
                auto_proxy_sensitivity=True,
            )

    def test_proxy_build_failure_raises(self, tmp_path, monkeypatch):
        """auto_proxy_sensitivity=True + proxy build fails -> RuntimeError."""
        if not HAS_MLX:
            pytest.skip("mlx not available")
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        src.mkdir()
        np_save(
            {"w": np.zeros((128, 256), dtype=np.float32)}, str(src / "w.safetensors")
        )
        (src / "config.json").write_text('{"model_type": "llama"}')

        from omlx import settings as _settings

        monkeypatch.setattr(_settings, "get_system_memory", lambda: 0)
        from omlx import oq as _oq

        monkeypatch.setattr(
            _oq,
            "_build_streaming_proxy_for_sensitivity",
            MagicMock(side_effect=RuntimeError("simulated build fail")),
        )

        with pytest.raises(RuntimeError, match="auto-proxy sensitivity failed"):
            quantize_oq_streaming(
                str(src),
                str(tmp_path / "out"),
                4,
                auto_proxy_sensitivity=True,
            )


# =============================================================================
# Test on-the-fly FP8 dequant in _LazyTensorIndex
# =============================================================================


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestOnTheFlyFp8Dequant:
    def test_vllm_scale_inv_convention(self, tmp_path):
        """vLLM convention: weight (F8_E4M3) + weight_scale_inv (F32)."""
        w = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        s = np.ones((1, 1), dtype=np.float32)
        path = str(tmp_path / "vllm.safetensors")
        _write_safetensors(
            path,
            {
                "layer.weight": (w.tobytes(), [128, 128], "F8_E4M3"),
                "layer.weight_scale_inv": s,
            },
        )
        idx = _LazyTensorIndex([path])
        assert len(idx._fp8_pairs) == 1
        assert "layer.weight" in idx
        assert "layer.weight_scale_inv" not in idx
        result = idx["layer.weight"]
        assert result.dtype == mx.bfloat16

    def test_mxfp_dot_scale_convention(self, tmp_path):
        """MXFP convention: key.weight (F8_E4M3) + key.scale (F8_E8M0)."""
        w = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        s = np.full((1, 1), 127, dtype=np.uint8)  # E8M0 127 = 2^0 = 1.0
        path = str(tmp_path / "mxfp.safetensors")
        _write_safetensors(
            path,
            {
                "layer.weight": (w.tobytes(), [128, 128], "F8_E4M3"),
                "layer.scale": (s.tobytes(), [1, 1], "F8_E8M0"),
            },
        )
        idx = _LazyTensorIndex([path])
        assert "layer.weight" in idx
        assert "layer.scale" not in idx
        result = idx.pop("layer.weight")
        assert result.dtype == mx.bfloat16
        assert "layer.weight" not in idx._index

    def test_mxfp_partial_block_scale_convention(self, tmp_path):
        """FP8 block scales may use ceil(rows / 128) partial tail blocks."""
        w = np.random.randint(0, 255, (129, 256), dtype=np.uint8)
        s = np.full((2, 2), 127, dtype=np.uint8)
        path = str(tmp_path / "mxfp_partial.safetensors")
        _write_safetensors(
            path,
            {
                "layer.weight": (w.tobytes(), [129, 256], "F8_E4M3"),
                "layer.scale": (s.tobytes(), [2, 2], "F8_E8M0"),
            },
        )
        idx = _LazyTensorIndex([path])
        result = idx.pop("layer.weight")
        assert result.shape == (129, 256)
        assert result.dtype == mx.bfloat16

    def test_i8_with_e8m0_scale_is_fp4_packed(self, tmp_path):
        """I8 bytes with a (rows, byte_cols/16) E8M0 scale are FP4-packed
        (DeepSeek V4 expert layout): each byte holds two fp4 values, so the
        dequant must unpack via mxfp4 instead of reading the bytes as int8
        values."""
        w = mx.random.normal((32, 64)).astype(mx.bfloat16)
        qw, scales = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
        path = str(tmp_path / "i8.safetensors")
        _write_safetensors(
            path,
            {
                "expert.weight": (
                    np.array(qw).view(np.int8).tobytes(),
                    [32, 32],
                    "I8",
                ),
                "expert.scale": (np.array(scales).tobytes(), [32, 2], "F8_E8M0"),
            },
        )
        idx = _LazyTensorIndex([path])
        result = idx["expert.weight"]
        expected = mx.dequantize(qw, scales, None, group_size=32, bits=4, mode="mxfp4")
        assert result.shape == (32, 64)
        assert result.dtype == mx.bfloat16
        assert mx.allclose(
            result.astype(mx.float32), expected.astype(mx.float32)
        ).item()

    def test_i8_with_block_e8m0_scale_plain_dequant(self, tmp_path):
        """I8 with a non-fp4 scale layout (16x16 blocks) stays plain int8
        block dequant."""
        w = np.random.randint(-128, 127, (32, 32), dtype=np.int8)
        s = np.full((2, 2), 127, dtype=np.uint8)  # 16x16 blocking, scale=1.0
        path = str(tmp_path / "i8_block.safetensors")
        _write_safetensors(
            path,
            {
                "expert.weight": (w.tobytes(), [32, 32], "I8"),
                "expert.scale": (s.tobytes(), [2, 2], "F8_E8M0"),
            },
        )
        idx = _LazyTensorIndex([path])
        assert idx.source_quant_info("expert.weight") is None
        result = idx["expert.weight"]
        expected = mx.array(w.astype(np.float32)).astype(mx.bfloat16)
        assert mx.allclose(result, expected, atol=0.1).item()

    def test_no_scale_keys_no_pairs(self, tmp_path):
        path = str(tmp_path / "plain.safetensors")
        _write_safetensors(
            path,
            {
                "layer.weight": np.random.randn(4, 8).astype(np.float16),
            },
        )
        idx = _LazyTensorIndex([path])
        assert len(idx._fp8_pairs) == 0
        assert len(idx) == 1


# =============================================================================
# Pre-quantized source passthrough (DeepSeek V4 fp4/fp8 checkpoints)
# =============================================================================


def _write_fp4_pair(tensors, name, rows, cols):
    """Quantize a random tensor to mxfp4 and add it to a fixture dict in the
    DeepSeek V4 raw layout (I8 packed bytes + E8M0 scale). Returns the
    reference (qw, scales) pair."""
    w = mx.random.normal((rows, cols)).astype(mx.bfloat16)
    qw, scales = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
    tensors[f"{name}.weight"] = (
        np.array(qw).view(np.int8).tobytes(),
        [rows, cols // 2],
        "I8",
    )
    tensors[f"{name}.scale"] = (
        np.array(scales).tobytes(),
        [rows, cols // 32],
        "F8_E8M0",
    )
    return qw, scales


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestPreQuantizedSource:
    def test_fp4_detection_and_logical_shape(self, tmp_path):
        path = str(tmp_path / "fp4.safetensors")
        tensors = {}
        _write_fp4_pair(tensors, "experts.0.w1", 8, 64)
        _write_safetensors(path, tensors)
        idx = _LazyTensorIndex([path])
        info = idx.source_quant_info("experts.0.w1.weight")
        assert info == {
            "kind": "mxfp4",
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
        }
        logical = idx.logical_metadata()
        assert logical["experts.0.w1.weight"] == ((8, 64), "BF16")
        assert "experts.0.w1.scale" not in logical

    def test_fp8_block_classification_and_load_packed(self, tmp_path):
        rows, cols = 256, 128
        w = np.random.randint(0, 255, (rows, cols), dtype=np.uint8)
        s = np.full((2, 1), 127, dtype=np.uint8)
        path = str(tmp_path / "fp8.safetensors")
        _write_safetensors(
            path,
            {
                "attn.wq_a.weight": (w.tobytes(), [rows, cols], "F8_E4M3"),
                "attn.wq_a.scale": (s.tobytes(), [2, 1], "F8_E8M0"),
            },
        )
        idx = _LazyTensorIndex([path])
        info = idx.source_quant_info("attn.wq_a.weight")
        assert info == {
            "kind": "fp8_block",
            "bits": 8,
            "group_size": 32,
            "mode": "mxfp8",
        }
        packed, scales = idx._load_packed("attn.wq_a.weight")
        assert packed.dtype == mx.uint32
        assert packed.shape == (rows, cols // 4)
        assert scales.dtype == mx.uint8
        assert scales.shape == (rows, cols // 32)
        ref = mx.repeat(mx.repeat(mx.array(s), 4, -1), 128, 0)
        assert mx.array_equal(scales, ref).item()

    def test_fp4_load_packed_roundtrip(self, tmp_path):
        path = str(tmp_path / "fp4.safetensors")
        tensors = {}
        qw, scales = _write_fp4_pair(tensors, "experts.0.w1", 8, 64)
        _write_safetensors(path, tensors)
        idx = _LazyTensorIndex([path])
        packed, sc = idx._load_packed("experts.0.w1.weight")
        assert mx.array_equal(packed, qw).item()
        assert mx.array_equal(sc, scales).item()

    def test_reshape_astype_replay(self, tmp_path):
        path = str(tmp_path / "w.safetensors")
        _write_safetensors(
            path,
            {
                "wo_a.weight": np.arange(32, dtype=np.float16).reshape(4, 8),
                "tid2eid": np.arange(8, dtype=np.float16).reshape(2, 4),
            },
        )
        idx = _LazyTensorIndex([path])

        def sanitize(weights):
            out = dict(weights)
            out["wo_a.weight"] = out["wo_a.weight"].reshape(2, 2, -1)
            out["tid2eid"] = out["tid2eid"].astype(mx.int32)
            return out

        plan = _discover_sanitize_plan(sanitize, idx)
        dp = _DiscoveredPlan(plan, idx)
        wo_a = dp.pop("wo_a.weight")
        assert wo_a.shape == (2, 2, 8)
        assert mx.array_equal(
            wo_a, mx.arange(32, dtype=mx.float16).reshape(2, 2, 8)
        ).item()
        tid = dp.pop("tid2eid")
        assert tid.dtype == mx.int32
        assert mx.array_equal(tid, mx.arange(8, dtype=mx.int32).reshape(2, 4)).item()

    def test_stack_pop_packed(self, tmp_path):
        path = str(tmp_path / "experts.safetensors")
        tensors = {}
        refs = [_write_fp4_pair(tensors, f"experts.{e}.w1", 8, 64) for e in range(4)]
        _write_safetensors(path, tensors)
        idx = _LazyTensorIndex([path])

        def sanitize(weights):
            stacked = [weights.pop(f"experts.{e}.w1.weight") for e in range(4)]
            weights["switch.w1.weight"] = mx.stack(stacked)
            return weights

        plan = _discover_sanitize_plan(sanitize, idx)
        dp = _DiscoveredPlan(plan, idx)
        assert dp.source_quant_info("switch.w1.weight") == {
            "kind": "mxfp4",
            "bits": 4,
            "group_size": 32,
            "mode": "mxfp4",
        }
        assert dp.plan_shape("switch.w1.weight") == (4, 8, 64)
        w, s = dp.pop_packed("switch.w1.weight")
        assert w.dtype == mx.uint32 and w.shape == (4, 8, 8)
        assert s.dtype == mx.uint8 and s.shape == (4, 8, 2)
        for e, (qw, scales) in enumerate(refs):
            assert mx.array_equal(w[e], qw).item()
            assert mx.array_equal(s[e], scales).item()
        assert "switch.w1.weight" not in dp

    def test_mixed_kind_stack_no_passthrough(self, tmp_path):
        path = str(tmp_path / "mixed.safetensors")
        tensors = {}
        _write_fp4_pair(tensors, "a", 8, 64)
        w = np.random.randint(0, 255, (8, 64), dtype=np.uint8)
        s = np.full((1, 2), 127, dtype=np.uint8)
        tensors["b.weight"] = (w.tobytes(), [8, 64], "F8_E4M3")
        tensors["b.scale"] = (s.tobytes(), [1, 2], "F8_E8M0")
        _write_safetensors(path, tensors)
        idx = _LazyTensorIndex([path])

        def sanitize(weights):
            weights["mixed.weight"] = mx.stack(
                [weights.pop("a.weight"), weights.pop("b.weight")]
            )
            return weights

        plan = _discover_sanitize_plan(sanitize, idx)
        dp = _DiscoveredPlan(plan, idx)
        assert dp.source_quant_info("mixed.weight") is None

    def test_float_source_no_passthrough(self, tmp_path):
        path = str(tmp_path / "plain.safetensors")
        _write_safetensors(
            path,
            {"layer.weight": np.random.randn(4, 8).astype(np.float16)},
        )
        idx = _LazyTensorIndex([path])
        assert idx.source_quant_info("layer.weight") is None

        plan = _discover_sanitize_plan(lambda w: dict(w), idx)
        dp = _DiscoveredPlan(plan, idx)
        assert dp.source_quant_info("layer.weight") is None


class TestPerturbBitsFor:
    def test_snap_below(self):
        assert _perturb_bits_for(8) == 6
        assert _perturb_bits_for(6) == 5
        assert _perturb_bits_for(4) == 3
        assert _perturb_bits_for(3) == 2
        assert _perturb_bits_for(2) is None


class TestShouldQuantizeTensorWeightGuard:
    def test_non_weight_2d_params_not_quantized(self):
        assert not _should_quantize_tensor("model.layers.0.attn_hc.fn", (24, 16384))
        assert not _should_quantize_tensor("model.layers.0.hc_head.fn", (4, 16384))
        assert not _should_quantize_tensor(
            "model.layers.0.attn.compressor.ape", (4, 1024)
        )
        assert not _should_quantize_tensor("mtp.0.hc_head.base", (4, 16384))

    def test_weight_tensors_still_quantized(self):
        assert _should_quantize_tensor("model.layers.0.attn.wq_a.weight", (1024, 4096))
        assert _should_quantize_tensor("lm_head.weight", (1024, 4096))


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestBuildQuantPlanFixedOverrides:
    def _shapes(self):
        # The routed expert dominates the params so non-expert boosts fit
        # comfortably under the hard bpw cap.
        return {
            "model.layers.0.self_attn.q_proj": (64, 64),
            "model.layers.0.ffn.switch_mlp.gate_proj": (256, 64, 64),
        }

    def _config(self):
        return {
            "num_hidden_layers": 1,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": {"0": 1.0},
        }

    def test_fixed_paths_excluded_from_boosts(self):
        fixed = {
            "model.layers.0.self_attn.q_proj": {
                "bits": 8,
                "group_size": 32,
                "mode": "mxfp8",
            }
        }
        baseline = _build_quant_plan(
            self._shapes(), self._config(), 4, target_bpw=4.6, hard_cap_bpw=4.7
        )
        assert "model.layers.0.self_attn.q_proj" in baseline.boost_map
        plan = _build_quant_plan(
            self._shapes(),
            self._config(),
            4,
            target_bpw=4.6,
            hard_cap_bpw=4.7,
            fixed_overrides=fixed,
        )
        assert "model.layers.0.self_attn.q_proj" not in plan.boost_map


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestEstimateBpwHeaderOnly:
    def _make_model_dir(self, tmp_path, with_mtp=False):
        d = tmp_path / "model"
        d.mkdir()
        tensors = {}
        _write_fp4_pair(tensors, "layers.0.ffn.experts.0.w1", 8, 64)
        w = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        s = np.full((1, 2), 127, dtype=np.uint8)
        tensors["layers.0.attn.wq_a.weight"] = (w.tobytes(), [64, 64], "F8_E4M3")
        tensors["layers.0.attn.wq_a.scale"] = (s.tobytes(), [1, 2], "F8_E8M0")
        if with_mtp:
            tensors["mtp.0.e_proj.weight"] = (w.tobytes(), [64, 64], "F8_E4M3")
            tensors["mtp.0.e_proj.scale"] = (s.tobytes(), [1, 2], "F8_E8M0")
        _write_safetensors(str(d / "model.safetensors"), tensors)
        config = {
            "model_type": "deepseek_v4",
            "num_hidden_layers": 1,
            "quantization_config": {"quant_method": "fp8"},
        }
        (d / "config.json").write_text(json.dumps(config))
        index = {
            "metadata": {},
            "weight_map": {k: "model.safetensors" for k in tensors},
        }
        (d / "model.safetensors.index.json").write_text(json.dumps(index))
        return d

    def test_fp8_source_estimates_without_mx_load(self, tmp_path):
        """F8_E8M0 scales crash mx.load; the header-only scan must not."""
        d = self._make_model_dir(tmp_path)
        result = estimate_bpw_and_size(str(d), 8)
        # fp4 expert passthrough: 8x64 logical at 4 bits + 1B e8m0 per group.
        expert_bytes = (8 * 64 * 4) // 8 + 8 * (64 // 32)
        # fp8 attn passthrough: 64x64 at 8 bits + 1B e8m0 per group.
        attn_bytes = 64 * 64 + 64 * (64 // 32)
        assert result["output_size_bytes"] == expert_bytes + attn_bytes
        assert result["effective_bpw"] > 0

    def test_preserve_mtp_counts_protected_fp8_as_bf16(self, tmp_path):
        d = self._make_model_dir(tmp_path, with_mtp=True)
        without = estimate_bpw_and_size(str(d), 8, preserve_mtp=False)
        with_mtp = estimate_bpw_and_size(str(d), 8, preserve_mtp=True)
        # e_proj is MTP-protected -> full precision bf16 in the output.
        assert (
            with_mtp["output_size_bytes"] - without["output_size_bytes"] == 64 * 64 * 2
        )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestQuantizeOqStreamingPassthroughDtypes:
    def test_float16_keeps_vision_audio_passthrough_tensors_float32(self, tmp_path):
        """Protected VLM/audio tensors must not be saved as FP16."""
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        src.mkdir()
        hidden = 64
        np_save(
            {
                "vision_tower.layers.0.self_attn.k_proj.weight": np.ones(
                    (hidden, hidden), dtype=np.float32
                ),
                "multi_modal_projector.linear.weight": np.ones(
                    (hidden, hidden), dtype=np.float32
                ),
                "audio_tower.layers.0.self_attn.k_proj.weight": np.ones(
                    (hidden, hidden), dtype=np.float32
                ),
                "vision_tower.layers.0.input_layernorm.weight": np.ones(
                    hidden, dtype=np.float32
                ),
                "model.layers.0.input_layernorm.weight": np.ones(
                    hidden, dtype=np.float32
                ),
                "model.layers.0.self_attn.q_proj.weight": np.ones(
                    (hidden, hidden), dtype=np.float32
                ),
            },
            str(src / "model.safetensors"),
        )
        (src / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["TestModelForCausalLM"],
                    "model_type": "test_passthrough",
                    "num_hidden_layers": 1,
                    "hidden_size": hidden,
                    "vocab_size": 256,
                }
            ),
            encoding="utf-8",
        )
        (src / "oq_sensitivity_map.json").write_text(
            json.dumps({"0": 0.1}), encoding="utf-8"
        )

        out = tmp_path / "out"
        quantize_oq_streaming(str(src), str(out), oq_level=4, dtype="float16")

        tensors = {}
        for sf in out.glob("*.safetensors"):
            tensors.update(mx.load(str(sf)))

        assert (
            tensors["vision_tower.layers.0.self_attn.k_proj.weight"].dtype == mx.float32
        )
        assert tensors["multi_modal_projector.linear.weight"].dtype == mx.float32
        assert (
            tensors["audio_tower.layers.0.self_attn.k_proj.weight"].dtype == mx.float32
        )
        assert (
            tensors["vision_tower.layers.0.input_layernorm.weight"].dtype == mx.float32
        )
        assert tensors["model.layers.0.input_layernorm.weight"].dtype == mx.float16
        assert tensors["model.layers.0.self_attn.q_proj.weight"].dtype == mx.uint32


# =============================================================================
# End-to-end: quantize_oq_streaming with FP8 sources
# =============================================================================


def _make_fp8_model(
    model_dir, n_layers=2, hidden=128, n_experts=0, fp8_convention="mxfp"
):
    """Create a synthetic FP8 model directory for integration testing.

    Returns the path and total raw bytes of FP8 weight data.
    """
    import json

    config = {
        "architectures": ["TestModelForCausalLM"],
        "model_type": "test_fp8",
        "num_hidden_layers": n_layers,
        "hidden_size": hidden,
        "vocab_size": 256,
    }
    if n_experts:
        config["num_local_experts"] = n_experts

    tensors = {}

    # Embedding (plain F16 — not FP8)
    tensors["model.embed_tokens.weight"] = np.random.randn(256, hidden).astype(
        np.float16
    )

    for i in range(n_layers):
        pfx = f"model.layers.{i}"

        # Attention weights (FP8 + scale)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            w = np.random.randint(0, 255, (hidden, hidden), dtype=np.uint8)
            if fp8_convention == "mxfp":
                s = np.full((1, 1), 127, dtype=np.uint8)  # E8M0 scale=1.0
                tensors[f"{pfx}.self_attn.{proj}.weight"] = (
                    w.tobytes(),
                    [hidden, hidden],
                    "F8_E4M3",
                )
                tensors[f"{pfx}.self_attn.{proj}.scale"] = (
                    s.tobytes(),
                    [1, 1],
                    "F8_E8M0",
                )
            else:  # vllm
                s = np.ones((1, 1), dtype=np.float32)
                tensors[f"{pfx}.self_attn.{proj}.weight"] = (
                    w.tobytes(),
                    [hidden, hidden],
                    "F8_E4M3",
                )
                tensors[f"{pfx}.self_attn.{proj}.weight_scale_inv"] = s

        # MLP weights (FP8 + scale)
        for proj in ["gate_proj", "up_proj"]:
            w = np.random.randint(0, 255, (hidden * 4, hidden), dtype=np.uint8)
            if fp8_convention == "mxfp":
                s = np.full((1, 1), 127, dtype=np.uint8)
                tensors[f"{pfx}.mlp.{proj}.weight"] = (
                    w.tobytes(),
                    [hidden * 4, hidden],
                    "F8_E4M3",
                )
                tensors[f"{pfx}.mlp.{proj}.scale"] = (s.tobytes(), [1, 1], "F8_E8M0")
            else:
                s = np.ones((1, 1), dtype=np.float32)
                tensors[f"{pfx}.mlp.{proj}.weight"] = (
                    w.tobytes(),
                    [hidden * 4, hidden],
                    "F8_E4M3",
                )
                tensors[f"{pfx}.mlp.{proj}.weight_scale_inv"] = s

        # down_proj (FP8)
        w = np.random.randint(0, 255, (hidden, hidden * 4), dtype=np.uint8)
        if fp8_convention == "mxfp":
            s = np.full((1, 1), 127, dtype=np.uint8)
            tensors[f"{pfx}.mlp.down_proj.weight"] = (
                w.tobytes(),
                [hidden, hidden * 4],
                "F8_E4M3",
            )
            tensors[f"{pfx}.mlp.down_proj.scale"] = (s.tobytes(), [1, 1], "F8_E8M0")
        else:
            s = np.ones((1, 1), dtype=np.float32)
            tensors[f"{pfx}.mlp.down_proj.weight"] = (
                w.tobytes(),
                [hidden, hidden * 4],
                "F8_E4M3",
            )
            tensors[f"{pfx}.mlp.down_proj.weight_scale_inv"] = s

        # Layer norms (plain F16)
        tensors[f"{pfx}.input_layernorm.weight"] = np.ones(hidden, dtype=np.float16)
        tensors[f"{pfx}.post_attention_layernorm.weight"] = np.ones(
            hidden, dtype=np.float16
        )

    # LM head (plain F16)
    tensors["lm_head.weight"] = np.random.randn(256, hidden).astype(np.float16)

    sf_path = str(model_dir / "model.safetensors")
    _write_safetensors(sf_path, tensors)

    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    return model_dir


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestQuantizeOqStreamingFp8:
    """End-to-end tests for quantize_oq_streaming with FP8 source models.

    These tests exercise the FP8 dequant + streaming write path on synthetic
    safetensors data, so the source models cannot be loaded by mlx_lm.load.
    Real sensitivity measurement would fail; we mock it out per class to keep
    the focus on the FP8 dequant path. The sensitivity-required contract is
    covered separately by TestSensitivityRequiredEnforcement.
    """

    @pytest.fixture(autouse=True)
    def _mock_sensitivity(self, monkeypatch):
        """Bypass real sensitivity measurement for synthetic FP8 fixtures."""
        from omlx import oq as _oq

        def _fake_measure(model_path, config, oq_level, **_kw):
            n = (
                config.get("num_hidden_layers")
                or config.get("text_config", {}).get("num_hidden_layers")
                or 4
            )
            return {i: 0.1 for i in range(n)}

        monkeypatch.setattr(_oq, "_measure_sensitivity", _fake_measure)
        monkeypatch.setattr(
            _oq, "_measure_sensitivity_from_quantized_model", _fake_measure
        )
        # Auto-proxy path: skip the actual mlx_lm.convert build since
        # synthetic FP8 fixtures cannot be loaded; treat the proxy as a no-op
        # and let the mocked measurement above produce the scores.
        monkeypatch.setattr(
            _oq,
            "_build_proxy_for_sensitivity",
            lambda *a, **k: Path("/dev/null"),
        )

    def test_mxfp_source_produces_output(self, tmp_path):
        """MXFP (.scale suffix) FP8 model quantizes without error."""
        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, fp8_convention="mxfp")
        out = tmp_path / "out"

        quantize_oq_streaming(str(src), str(out), oq_level=4)

        assert (out / "config.json").exists()
        out_shards = list(out.glob("*.safetensors"))
        assert len(out_shards) > 0

    def test_vllm_source_produces_output(self, tmp_path):
        """vLLM (_scale_inv suffix) FP8 model quantizes without error."""
        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, fp8_convention="vllm")
        out = tmp_path / "out"

        quantize_oq_streaming(str(src), str(out), oq_level=4)

        assert (out / "config.json").exists()
        out_shards = list(out.glob("*.safetensors"))
        assert len(out_shards) > 0

    def test_no_scale_keys_in_output(self, tmp_path):
        """Scale keys are consumed by dequant, never written to output."""
        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, fp8_convention="mxfp")
        out = tmp_path / "out"

        quantize_oq_streaming(str(src), str(out), oq_level=4)

        from safetensors import safe_open

        for sf in out.glob("*.safetensors"):
            with safe_open(str(sf), framework="numpy") as f:
                for k in f.keys():
                    assert not k.endswith(".scale"), f"scale key leaked: {k}"
                    assert not k.endswith("_scale_inv"), f"scale_inv key leaked: {k}"

    def test_output_tensors_are_bf16_or_quantized(self, tmp_path):
        """All output tensors are either quantized (uint32) or bf16."""
        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, fp8_convention="mxfp")
        out = tmp_path / "out"

        quantize_oq_streaming(str(src), str(out), oq_level=4)

        allowed = {mx.bfloat16, mx.float16, mx.float32, mx.uint32, mx.uint8}
        for sf in out.glob("*.safetensors"):
            tensors = mx.load(str(sf))
            for k, t in tensors.items():
                assert t.dtype in allowed, f"{k}: unexpected dtype {t.dtype}"

    def test_exceeds_ram_skips_eager_sanitize(self, tmp_path):
        """When model exceeds simulated RAM, eager sanitize is skipped."""
        from unittest.mock import patch

        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, n_layers=2, hidden=128, fp8_convention="mxfp")
        out = tmp_path / "out"

        # Patch system RAM to 1 byte — any model exceeds it
        with patch("omlx.settings.get_system_memory", return_value=1):
            quantize_oq_streaming(str(src), str(out), oq_level=4)

        assert (out / "config.json").exists()
        out_shards = list(out.glob("*.safetensors"))
        assert len(out_shards) > 0

    def test_exceeds_ram_no_scratch_files(self, tmp_path):
        """On-the-fly dequant produces zero scratch/temp shard files."""
        import os
        import tempfile
        from unittest.mock import patch

        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, fp8_convention="mxfp")
        out = tmp_path / "out"

        # List temp files before
        tmpdir = tempfile.gettempdir()
        before = set(os.listdir(tmpdir))

        with patch("omlx.settings.get_system_memory", return_value=1):
            quantize_oq_streaming(str(src), str(out), oq_level=4)

        # No new safetensors scratch files in tmp
        after = set(os.listdir(tmpdir))
        new_files = after - before
        scratch = [f for f in new_files if "safetensors" in f or "dequant" in f]
        assert scratch == [], f"scratch files created: {scratch}"

    def test_fp8_dequant_with_sanitize_plan(self, tmp_path):
        """When sanitize discovery succeeds, FP8 dequant works through
        _DiscoveredPlan._materialize_source."""
        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, n_layers=1, hidden=128, fp8_convention="mxfp")

        idx = _LazyTensorIndex([str(src / "model.safetensors")])
        assert len(idx._fp8_pairs) > 0

        def rename_sanitize(weights):
            return {k.replace("model.", "m."): v for k, v in weights.items()}

        plan = _discover_sanitize_plan(rename_sanitize, idx)
        assert plan is not None

        dp = _DiscoveredPlan(plan, idx)
        # pop a renamed FP8 tensor — should dequant via _materialize_source
        renamed_key = None
        for k in dp.keys():
            if "q_proj" in k:
                renamed_key = k
                break
        assert renamed_key is not None
        arr = dp.pop(renamed_key)
        assert arr.dtype == mx.bfloat16
        assert arr.shape == (128, 128)

    def test_logical_metadata_hides_scales_reports_bf16(self, tmp_path):
        """logical_metadata() hides scale keys and reports FP8 weights as BF16."""
        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, n_layers=1, hidden=64, fp8_convention="mxfp")
        idx = _LazyTensorIndex([str(src / "model.safetensors")])

        meta = idx.logical_metadata()
        for k in meta:
            assert not k.endswith(".scale"), f"scale key visible: {k}"
        for k, (_shape, dtype) in meta.items():
            if "self_attn" in k or "mlp" in k:
                if k.endswith(".weight"):
                    assert dtype == "BF16", f"{k}: dtype={dtype}, expected BF16"

    def test_mixed_fp8_and_plain_tensors(self, tmp_path):
        """Model with both FP8 and plain (F16) tensors handles both correctly."""
        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, n_layers=1, hidden=128, fp8_convention="mxfp")
        out = tmp_path / "out"

        quantize_oq_streaming(str(src), str(out), oq_level=4)

        from safetensors import safe_open

        out_keys = set()
        for sf in out.glob("*.safetensors"):
            with safe_open(str(sf), framework="numpy") as f:
                out_keys.update(f.keys())

        # Embedding and norms should be present (not quantized, just passed through)
        assert any("embed" in k for k in out_keys)
        assert any("layernorm" in k for k in out_keys)
        # Attention weights should be quantized (have .scales)
        assert any("self_attn" in k and k.endswith(".scales") for k in out_keys)

    def test_streaming_sensitivity_proxy_handles_fp8_source(
        self, tmp_path, monkeypatch
    ):
        """The sensitivity proxy writer handles FP8 sources without convert()."""
        src = tmp_path / "src"
        src.mkdir()
        _make_fp8_model(src, n_layers=1, hidden=64, fp8_convention="mxfp")
        out = tmp_path / "proxy"

        monkeypatch.setattr("omlx.oq._build_model_sanitizer", lambda *_a, **_k: None)
        monkeypatch.setattr("omlx.oq._build_non_quantizable_set", lambda _config: set())

        _build_streaming_proxy_for_sensitivity(str(src), out, dtype="bfloat16")

        proxy_config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        assert proxy_config["quantization"]["bits"] == _PROXY_QUANT_BITS
        assert proxy_config["quantization"]["group_size"] == _PROXY_QUANT_GROUP_SIZE

        from safetensors import safe_open

        out_keys = set()
        for sf in out.glob("*.safetensors"):
            with safe_open(str(sf), framework="numpy") as f:
                out_keys.update(f.keys())

        assert out_keys
        assert not any(k.endswith(".scale") for k in out_keys)
        assert any(k.endswith(".scales") for k in out_keys)

    def test_i8_expert_weights_with_mxfp_scale(self, tmp_path):
        """I8 expert weights with E8M0 microscaling (1x16 block) dequant
        correctly through the full quantize pipeline."""
        import json

        src = tmp_path / "src"
        src.mkdir()

        hidden = 64
        tensors = {
            "model.embed_tokens.weight": np.random.randn(256, hidden).astype(
                np.float16
            ),
            "lm_head.weight": np.random.randn(256, hidden).astype(np.float16),
            "model.layers.0.input_layernorm.weight": np.ones(hidden, dtype=np.float16),
        }
        # I8 weight with 1x16 blocking
        w_i8 = np.random.randint(-128, 127, (hidden, hidden), dtype=np.int8)
        bs_col = 16
        sn = hidden // bs_col
        s_e8m0 = np.full((hidden, sn), 127, dtype=np.uint8)
        tensors["model.layers.0.self_attn.q_proj.weight"] = (
            w_i8.tobytes(),
            [hidden, hidden],
            "I8",
        )
        tensors["model.layers.0.self_attn.q_proj.scale"] = (
            s_e8m0.tobytes(),
            [hidden, sn],
            "F8_E8M0",
        )

        _write_safetensors(str(src / "model.safetensors"), tensors)
        config = {
            "architectures": ["TestModelForCausalLM"],
            "model_type": "test_i8",
            "num_hidden_layers": 1,
            "hidden_size": hidden,
            "vocab_size": 256,
        }
        with open(src / "config.json", "w") as f:
            json.dump(config, f)

        out = tmp_path / "out"
        quantize_oq_streaming(str(src), str(out), oq_level=4)

        assert (out / "config.json").exists()
        from safetensors import safe_open

        out_keys = set()
        for sf in out.glob("*.safetensors"):
            with safe_open(str(sf), framework="numpy") as f:
                out_keys.update(f.keys())
        assert not any(k.endswith(".scale") for k in out_keys)

    def test_bf16_weight_with_scale_key_not_paired(self, tmp_path):
        """BF16 weight + .scale key must NOT be treated as FP8 pair."""
        src = tmp_path / "src"
        src.mkdir()
        hidden = 64
        tensors = {
            "model.embed_tokens.weight": np.random.randn(256, hidden).astype(
                np.float16
            ),
            "lm_head.weight": np.random.randn(256, hidden).astype(np.float16),
            "model.layers.0.input_layernorm.weight": np.ones(hidden, dtype=np.float16),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(
                hidden, hidden
            ).astype(np.float16),
            "model.layers.0.self_attn.q_proj.scale": np.ones(
                (1, hidden), dtype=np.float32
            ),
        }
        _write_safetensors(str(src / "model.safetensors"), tensors)
        import json

        config = {
            "architectures": ["TestModelForCausalLM"],
            "model_type": "test_bf16_scale",
            "num_hidden_layers": 1,
            "hidden_size": hidden,
            "vocab_size": 256,
        }
        with open(src / "config.json", "w") as f:
            json.dump(config, f)

        idx = _LazyTensorIndex([str(src / "model.safetensors")])
        assert len(idx._fp8_pairs) == 0, "BF16 weight should not pair with .scale"
        assert (
            "model.layers.0.self_attn.q_proj.scale" in idx
        ), "scale key must remain visible"


# =============================================================================
# Test _build_model_sanitizer text_only VLM bypass
# =============================================================================


class TestBuildModelSanitizerTextOnly:
    """When text_only=True, _build_model_sanitizer must use the mlx-lm (LLM)
    sanitize path — never the mlx-vlm (VLM) path — even when the model config
    lists a ForConditionalGeneration architecture.

    Without this, VLM sanitize uses a _Proxy that lacks self.mtp, silently
    stripping all mtp.* tensors from the oQ output despite preserve_mtp=True.
    """

    VLM_CONFIG = {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen3_5",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
    }

    LLM_CONFIG = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen3_5",
        "num_hidden_layers": 28,
        "hidden_size": 3584,
    }

    def test_vlm_config_without_text_only_attempts_vlm_path(self):
        """Baseline: VLM config without text_only should try the VLM path."""
        from unittest.mock import patch

        from omlx.oq import _build_model_sanitizer

        with patch("omlx.oq.logger") as mock_logger:
            _build_model_sanitizer(self.VLM_CONFIG, text_only=False)

        debug_messages = [str(c) for c in mock_logger.debug.call_args_list]
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        all_messages = " ".join(debug_messages + info_messages)
        assert "mlx-vlm" in all_messages or "mlx-lm" in all_messages

    def test_vlm_config_with_text_only_skips_vlm_path(self):
        """With text_only=True, the VLM path must be skipped entirely."""
        from unittest.mock import patch

        from omlx.oq import _build_model_sanitizer

        with patch("omlx.oq.logger") as mock_logger:
            _build_model_sanitizer(self.VLM_CONFIG, text_only=True)

        debug_messages = [str(c) for c in mock_logger.debug.call_args_list]
        info_messages = [str(c) for c in mock_logger.info.call_args_list]
        all_messages = " ".join(debug_messages + info_messages)
        assert "mlx-vlm full sanitize" not in all_messages

    def test_llm_config_unaffected_by_text_only(self):
        """LLM configs (no ForConditionalGeneration) should always use the
        mlx-lm path regardless of text_only."""
        from unittest.mock import patch

        from omlx.oq import _build_model_sanitizer

        for text_only in (True, False):
            with patch("omlx.oq.logger") as mock_logger:
                _build_model_sanitizer(self.LLM_CONFIG, text_only=text_only)

            debug_messages = [str(c) for c in mock_logger.debug.call_args_list]
            info_messages = [str(c) for c in mock_logger.info.call_args_list]
            all_messages = " ".join(debug_messages + info_messages)
            assert "mlx-vlm full sanitize" not in all_messages


class TestBuildModelSanitizerMiniMaxCompat:
    def test_minimax_vlm_applies_compat_before_model_lookup(self, monkeypatch):
        pytest.importorskip("mlx_vlm.utils")
        from types import SimpleNamespace

        import mlx_vlm.utils as vlm_utils

        from omlx.oq import _build_model_sanitizer

        class _Cfg:
            def __init__(self, **fields):
                self.__dict__.update(fields)

            @classmethod
            def from_dict(cls, fields):
                return cls(**fields)

        class _FakeModel:
            @staticmethod
            def sanitize(proxy, weights):
                assert proxy.config.text_config.num_hidden_layers == 1
                return weights

        fake_module = SimpleNamespace(
            Model=_FakeModel,
            ModelConfig=_Cfg,
            VisionConfig=_Cfg,
            TextConfig=_Cfg,
            VisionModel=object,
            LanguageModel=object,
        )

        def unsupported_get_model_and_args(_config):
            raise ValueError("Model type minimax_m3_vl not supported")

        def apply_compat_patch():
            vlm_utils.get_model_and_args = lambda _config: (
                fake_module,
                "minimax_m3_vl",
            )
            return True

        monkeypatch.setattr(
            vlm_utils,
            "get_model_and_args",
            unsupported_get_model_and_args,
        )
        monkeypatch.setattr(
            "omlx.patches.mlx_vlm_minimax_m3_compat."
            "apply_mlx_vlm_minimax_m3_compat_patch",
            apply_compat_patch,
        )
        monkeypatch.setattr(
            "mlx_vlm.utils.sanitize_weights",
            lambda _model, weights, _config: weights,
        )

        config = {
            "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
            "model_type": "minimax_m3_vl",
            "text_config": {"num_hidden_layers": 1, "hidden_size": 16},
            "vision_config": {"hidden_size": 8},
        }
        sanitize = _build_model_sanitizer(config, text_only=False)

        assert sanitize is not None
        assert sanitize({"weight": 1}) == {"weight": 1}


# =============================================================================
# Test _vlm_sanitize proxy exposes the gemma-4 audio-guard attributes
# =============================================================================


class TestVlmSanitizeProxyAudioAttrs:
    """oq's _vlm_sanitize _Proxy must expose BOTH audio-guard attributes the
    gemma-4 family reads: gemma4 guards audio weights on ``self.audio_tower``,
    gemma4_unified on ``self.embed_audio``. A proxy missing the attribute a
    model's ``sanitize()`` reads raises AttributeError, aborting sanitize so oQ
    ships raw ``model.``-prefixed VLM keys that mlx-vlm cannot load.
    """

    @pytest.mark.parametrize("audio_attr", ["audio_tower", "embed_audio"])
    def test_proxy_exposes_audio_guard_attr(self, monkeypatch, audio_attr):
        pytest.importorskip("mlx_vlm.utils")
        from types import SimpleNamespace

        from omlx.oq import _build_model_sanitizer

        class _Cfg:
            def __init__(self, **fields):
                self.__dict__.update(fields)

            @classmethod
            def from_dict(cls, fields):
                return cls(**fields)

        class _FakeModel:
            @staticmethod
            def sanitize(proxy, weights):
                # The real Gemma4(Unified).sanitize reads this guard attr off
                # self; the proxy must expose it or getattr raises AttributeError
                # and oq's whole sanitize pass is silently dropped.
                getattr(proxy, audio_attr)
                return weights

        fake_module = SimpleNamespace(
            Model=_FakeModel,
            ModelConfig=_Cfg,
            VisionConfig=_Cfg,
            TextConfig=_Cfg,
            VisionModel=object,
            LanguageModel=object,
        )
        monkeypatch.setattr(
            "mlx_vlm.utils.get_model_and_args",
            lambda config: (fake_module, None),
        )
        monkeypatch.setattr(
            "mlx_vlm.utils.sanitize_weights",
            lambda _model, weights, _config: weights,
        )

        config = {
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "model_type": "gemma4_unified",
            "text_config": {"num_hidden_layers": 4, "hidden_size": 64},
            "vision_config": {"hidden_size": 32},
            "audio_config": {"hidden_size": 16},
        }

        sanitize = _build_model_sanitizer(config, text_only=False)
        assert sanitize is not None

        # Before the fix the proxy lacked ``embed_audio`` → this call raised
        # AttributeError for the gemma4_unified guard, dropping the sanitize pass.
        weights = {"model.embed_audio.proj.weight": 1}
        assert sanitize(weights) == weights


# =============================================================================
# Test _build_proxy_for_sensitivity MTP patch integration
# =============================================================================


class TestBuildProxyForSensitivityMtpPatch:
    """Regression tests for MTP responsibility in proxy building.

    _build_proxy_for_sensitivity is now a thin wrapper around the streaming
    proxy writer. It must not toggle global MTP state itself; MTP attach/restore
    belongs to the sanitizer and sensitivity-load paths.
    """

    def test_wrapper_does_not_toggle_mtp_state(self, tmp_path, monkeypatch):
        mtp_mod = MagicMock(
            apply_mlx_lm_mtp_patch=MagicMock(return_value=True),
            is_mtp_active=MagicMock(return_value=False),
            set_mtp_active=MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "omlx.patches.mlx_lm_mtp", mtp_mod)

        build_mock = MagicMock(side_effect=lambda _m, out, **_kw: out.mkdir())
        monkeypatch.setattr(
            "omlx.oq._build_streaming_proxy_for_sensitivity",
            build_mock,
        )

        result = _build_proxy_for_sensitivity(
            "/my/model",
            dtype="bfloat16",
            working_dir=str(tmp_path),
            trust_remote_code=True,
        )

        assert isinstance(result, Path)
        assert result.name.startswith("omlx_oq_proxy_")
        assert result.parent == tmp_path

        mtp_mod.apply_mlx_lm_mtp_patch.assert_not_called()
        mtp_mod.is_mtp_active.assert_not_called()
        mtp_mod.set_mtp_active.assert_not_called()
        build_mock.assert_called_once()
        assert build_mock.call_args.kwargs["dtype"] == "bfloat16"
        assert build_mock.call_args.kwargs["trust_remote_code"] is True

    def test_streaming_helper_error_propagates(self, tmp_path, monkeypatch):
        build_mock = MagicMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(
            "omlx.oq._build_streaming_proxy_for_sensitivity",
            build_mock,
        )
        with pytest.raises(RuntimeError, match="boom"):
            _build_proxy_for_sensitivity(
                "/fake/model",
                dtype="float16",
                working_dir=str(tmp_path),
            )

        build_mock.assert_called_once()


# =============================================================================
# Test _measure_sensitivity MTP patch integration (VLM path)
# =============================================================================


class TestMeasureSensitivityVlmMtp:
    """_measure_sensitivity must attach the MTP head for VLM checkpoints that
    declare MTP heads.

    mlx-vlm skips Model.sanitize for MLX-format checkpoints, so the
    language_model.mtp.* weights stay in the weight dict. Without an attached
    MTP head load_weights(strict=True) rejects them and the measurement
    silently returns {}. The function must apply the mlx-vlm runtime MTP
    patch and toggle mtp_active True for the load, then restore the previous
    state. The text path needs no toggle (the patched qwen35_model.sanitize
    self-consistently strips mtp.* when no head is attached).
    """

    def _patch_common(
        self, monkeypatch, has_mtp, has_mtp_weights=None, prev_active=False
    ):
        from omlx import oq as oq_mod

        if has_mtp_weights is None:
            has_mtp_weights = has_mtp
        mock_apply_patch = MagicMock()
        mock_apply_runtime = MagicMock()
        mock_set_active = MagicMock()
        mock_is_active = MagicMock(return_value=prev_active)

        monkeypatch.setitem(
            sys.modules,
            "omlx.utils.model_loading",
            MagicMock(
                maybe_apply_pre_load_patches=MagicMock(),
                _has_mtp_heads=MagicMock(return_value=has_mtp),
                _checkpoint_has_mtp_weights=MagicMock(return_value=has_mtp_weights),
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_lm_mtp",
            MagicMock(is_mtp_active=mock_is_active, set_mtp_active=mock_set_active),
        )
        monkeypatch.setitem(
            sys.modules,
            "omlx.patches.mlx_vlm_mtp",
            MagicMock(
                apply_mlx_vlm_mtp_patch=mock_apply_patch,
                apply_mlx_vlm_mtp_runtime_patch=mock_apply_runtime,
            ),
        )
        monkeypatch.setitem(sys.modules, "mlx_vlm", MagicMock())
        monkeypatch.setitem(
            sys.modules,
            "mlx_vlm.utils",
            MagicMock(load_model=MagicMock(return_value=MagicMock())),
        )
        monkeypatch.setitem(sys.modules, "mlx_lm", MagicMock())
        monkeypatch.setitem(
            sys.modules,
            "mlx_lm.tokenizer_utils",
            MagicMock(load=MagicMock(return_value=MagicMock())),
        )
        monkeypatch.setattr(
            oq_mod,
            "_measure_sensitivity_from_model",
            MagicMock(return_value={0: 0.1}),
        )
        return mock_apply_patch, mock_apply_runtime, mock_set_active

    def test_vlm_with_mtp_heads_attaches_head(self, monkeypatch):
        """VLM + MTP heads → runtime patch applied, mtp_active toggled True for load."""
        mock_apply_patch, mock_apply_runtime, mock_set_active = self._patch_common(
            monkeypatch,
            has_mtp=True,
        )

        result = _measure_sensitivity(
            "/fake/vlm-mtp",
            {"vision_config": {}},
            6,
        )

        assert result == {0: 0.1}
        mock_apply_patch.assert_called_once()
        mock_apply_runtime.assert_called_once()
        assert mock_set_active.call_args_list[0] == ((True,),)
        assert mock_set_active.call_args_list[-1] == ((False,),)

    def test_vlm_load_forwards_trust_remote_code(self, monkeypatch):
        self._patch_common(monkeypatch, has_mtp=True)

        _measure_sensitivity(
            "/fake/vlm-mtp",
            {"vision_config": {}},
            6,
            trust_remote_code=True,
        )

        load_model = sys.modules["mlx_vlm.utils"].load_model
        assert load_model.call_args.kwargs["trust_remote_code"] is True

    @pytest.mark.parametrize("prev_active", [False, True])
    def test_mtp_active_restored_after_load(self, monkeypatch, prev_active):
        """The previous mtp_active state is restored once the load returns."""
        _, _, mock_set_active = self._patch_common(
            monkeypatch,
            has_mtp=True,
            prev_active=prev_active,
        )

        _measure_sensitivity("/fake/vlm-mtp", {"vision_config": {}}, 6)

        assert mock_set_active.call_args_list[-1] == ((prev_active,),)

    def test_vlm_without_mtp_heads_no_toggle(self, monkeypatch):
        """VLM without MTP heads → no runtime patch, no mtp_active toggle."""
        mock_apply_patch, mock_apply_runtime, mock_set_active = self._patch_common(
            monkeypatch,
            has_mtp=False,
        )

        _measure_sensitivity("/fake/vlm", {"vision_config": {}}, 6)

        mock_apply_patch.assert_not_called()
        mock_apply_runtime.assert_not_called()
        mock_set_active.assert_not_called()

    def test_vlm_declares_mtp_without_weights_no_toggle(self, monkeypatch):
        """Config-only MTP declarations must not attach a missing MTP head."""
        mock_apply_patch, mock_apply_runtime, mock_set_active = self._patch_common(
            monkeypatch,
            has_mtp=True,
            has_mtp_weights=False,
        )

        _measure_sensitivity("/fake/vlm-mtp-config-only", {"vision_config": {}}, 6)

        mock_apply_patch.assert_not_called()
        mock_apply_runtime.assert_not_called()
        mock_set_active.assert_not_called()

    def test_text_model_no_vlm_toggle(self, monkeypatch):
        """Text checkpoint → VLM MTP toggling is skipped entirely."""
        mock_apply_patch, mock_apply_runtime, mock_set_active = self._patch_common(
            monkeypatch,
            has_mtp=True,
        )
        monkeypatch.setitem(
            sys.modules,
            "mlx_lm",
            MagicMock(load=MagicMock(return_value=(MagicMock(), MagicMock()))),
        )

        _measure_sensitivity("/fake/text", {}, 6)

        mock_apply_patch.assert_not_called()
        mock_apply_runtime.assert_not_called()
        mock_set_active.assert_not_called()

    def test_text_load_forwards_trust_remote_code(self, monkeypatch):
        """Text sensitivity load forwards the mlx-lm custom-code opt-in when
        the installed mlx-lm supports it."""
        import omlx.utils.model_loading as real_ml

        self._patch_common(monkeypatch, has_mtp=True)
        mock_load = MagicMock(return_value=(MagicMock(), MagicMock()))
        monkeypatch.setitem(sys.modules, "mlx_lm", MagicMock(load=mock_load))
        # _patch_common swapped model_loading for a MagicMock; oq imports
        # lm_load_compat from it. Expose the real shim and pin the capability
        # flag so forwarding is deterministic regardless of installed mlx-lm.
        monkeypatch.setattr(real_ml, "_LM_LOAD_ACCEPTS_TRC", True)
        sys.modules["omlx.utils.model_loading"].lm_load_compat = real_ml.lm_load_compat

        _measure_sensitivity(
            "/fake/text",
            {},
            6,
            trust_remote_code=True,
        )

        assert mock_load.call_args.kwargs["trust_remote_code"] is True


class TestMeasureSensitivityQuantizedVlm:
    def test_quantized_vlm_proxy_uses_vlm_loader(self, monkeypatch):
        from omlx import oq as oq_mod

        maybe_apply = MagicMock()
        monkeypatch.setitem(
            sys.modules,
            "omlx.utils.model_loading",
            MagicMock(
                maybe_apply_pre_load_patches=maybe_apply,
                _has_mtp_heads=MagicMock(return_value=False),
                _checkpoint_has_mtp_weights=MagicMock(return_value=False),
            ),
        )

        vlm_load = MagicMock(return_value=MagicMock())
        tokenizer_load = MagicMock(return_value=MagicMock())
        lm_load = MagicMock(side_effect=AssertionError("mlx-lm loader used for VLM"))

        monkeypatch.setitem(sys.modules, "mlx_vlm", MagicMock())
        monkeypatch.setitem(
            sys.modules,
            "mlx_vlm.utils",
            MagicMock(load_model=vlm_load),
        )
        monkeypatch.setitem(sys.modules, "mlx_lm", MagicMock(load=lm_load))
        monkeypatch.setitem(
            sys.modules,
            "mlx_lm.tokenizer_utils",
            MagicMock(load=tokenizer_load),
        )
        monkeypatch.setattr(
            oq_mod,
            "_load_calibration_data",
            MagicMock(return_value=None),
        )

        result = _measure_sensitivity_from_quantized_model(
            "/fake/minimax-proxy",
            {"vision_config": {}, "model_type": "minimax_m3_vl"},
            3.5,
            trust_remote_code=True,
        )

        assert result == {}
        maybe_apply.assert_called_once_with("/fake/minimax-proxy", for_vlm=True)
        vlm_load.assert_called_once()
        assert vlm_load.call_args.args[0] == Path("/fake/minimax-proxy")
        assert vlm_load.call_args.kwargs["lazy"] is True
        assert vlm_load.call_args.kwargs["trust_remote_code"] is True
        tokenizer_load.assert_called_once_with(Path("/fake/minimax-proxy"))
        lm_load.assert_not_called()


# =============================================================================
# Test pre-computed sensitivity map loading (oq_sensitivity_map.json)
# =============================================================================


class TestPrecomputedSensitivityMap:
    """Tests for the oq_sensitivity_map.json disk cache feature.

    When a pre-computed sensitivity map file exists at
    ``{model_path}/oq_sensitivity_map.json``, quantize_oq_streaming loads it
    directly and skips the entire sensitivity measurement pipeline
    (proxy building, model loading, calibration, etc.).
    """

    def test_loads_existing_sensitivity_map_and_skips_measurement(
        self, tmp_path, monkeypatch
    ):
        """When oq_sensitivity_map.json exists, it is loaded and measurement
        functions are never called."""
        if not HAS_MLX:
            pytest.skip("mlx not available")
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        src.mkdir()
        np_save(
            {"w": np.zeros((128, 256), dtype=np.float32)},
            str(src / "w.safetensors"),
        )
        (src / "config.json").write_text('{"model_type": "llama"}')

        sensitivity_map = {"0": 0.05, "1": 0.03, "2": 0.01}
        (src / "oq_sensitivity_map.json").write_text(
            json.dumps(sensitivity_map), encoding="utf-8"
        )

        from omlx import oq as _oq

        # Stub all measurement functions — they should NOT be called
        monkeypatch.setattr(
            _oq,
            "_measure_sensitivity",
            MagicMock(side_effect=RuntimeError("should not call")),
        )
        monkeypatch.setattr(
            _oq,
            "_measure_sensitivity_from_quantized_model",
            MagicMock(side_effect=RuntimeError("should not call")),
        )
        monkeypatch.setattr(
            _oq,
            "_build_proxy_for_sensitivity",
            MagicMock(side_effect=RuntimeError("should not call")),
        )

        out = tmp_path / "out"
        quantize_oq_streaming(str(src), str(out), oq_level=4)

        _oq._measure_sensitivity.assert_not_called()
        _oq._measure_sensitivity_from_quantized_model.assert_not_called()
        _oq._build_proxy_for_sensitivity.assert_not_called()

    def test_sensitivity_map_used_in_quant_plan(self, tmp_path, monkeypatch):
        """The loaded sensitivity map is stored in config['_oq_sensitivity_map']
        and flows into _build_quant_plan."""
        if not HAS_MLX:
            pytest.skip("mlx not available")
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        src.mkdir()
        np_save(
            {"w": np.zeros((128, 256), dtype=np.float32)},
            str(src / "w.safetensors"),
        )
        (src / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "num_hidden_layers": 32,
                    "hidden_size": 128,
                    "intermediate_size": 256,
                    "num_attention_heads": 8,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 256,
                }
            )
        )

        sensitivity_map = {str(i): 0.1 / (i + 1) for i in range(32)}
        (src / "oq_sensitivity_map.json").write_text(
            json.dumps(sensitivity_map), encoding="utf-8"
        )

        from omlx import oq as _oq

        # Capture the config that flows into _build_quant_plan
        captured_configs = []
        original_build_plan = _oq._build_quant_plan

        def _capture_build_plan(named_shapes, config, oq_level, **kwargs):
            captured_configs.append(dict(config))
            return original_build_plan(named_shapes, config, oq_level, **kwargs)

        monkeypatch.setattr(_oq, "_build_quant_plan", _capture_build_plan)

        out = tmp_path / "out"
        quantize_oq_streaming(str(src), str(out), oq_level=4)

        assert len(captured_configs) == 1
        config = captured_configs[0]
        assert "_oq_sensitivity_map" in config
        loaded_sens = config["_oq_sensitivity_map"]
        assert loaded_sens == sensitivity_map

    def test_no_sensitivity_map_falls_back_to_measurement(self, tmp_path, monkeypatch):
        """When oq_sensitivity_map.json does NOT exist, measurement runs."""
        if not HAS_MLX:
            pytest.skip("mlx not available")
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        src.mkdir()
        np_save(
            {"w": np.zeros((128, 256), dtype=np.float32)},
            str(src / "w.safetensors"),
        )
        (src / "config.json").write_text('{"model_type": "llama"}')

        from omlx import oq as _oq

        monkeypatch.setattr(
            _oq,
            "_measure_sensitivity",
            MagicMock(return_value={"0": 0.05, "1": 0.03}),
        )

        out = tmp_path / "out"
        quantize_oq_streaming(
            str(src),
            str(out),
            oq_level=4,
            trust_remote_code=True,
        )

        _oq._measure_sensitivity.assert_called_once()
        assert _oq._measure_sensitivity.call_args.kwargs["trust_remote_code"] is True

    @pytest.mark.parametrize(
        ("content,expected_exc,expected_match"),
        [
            ("{}", RuntimeError, "sensitivity measurement produced no scores"),
            ("not valid json", ValueError, None),
        ],
    )
    def test_sensitivity_map_file_errors(
        self,
        tmp_path,
        monkeypatch,
        content,
        expected_exc,
        expected_match,
    ):
        """Sensitivity map file issues (empty JSON or malformed JSON) should raise."""
        if not HAS_MLX:
            pytest.skip("mlx not available")
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        src.mkdir()
        np_save(
            {"w": np.zeros((128, 256), dtype=np.float32)},
            str(src / "w.safetensors"),
        )
        (src / "config.json").write_text('{"model_type": "llama"}')
        (src / "oq_sensitivity_map.json").write_text(content, encoding="utf-8")

        with pytest.raises(expected_exc, match=expected_match or ".*"):
            quantize_oq_streaming(str(src), str(tmp_path / "out"), oq_level=4)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestReplayChainGuards:
    """Chained transforms should replay in order instead of silently
    materializing only the final transform."""

    def _idx(self, tmp_path):
        path = str(tmp_path / "w.safetensors")
        _write_safetensors(
            path,
            {"w.weight": np.arange(32, dtype=np.float16).reshape(4, 8)},
        )
        return _LazyTensorIndex([path])

    def test_reshape_then_astype_replays(self, tmp_path):
        idx = self._idx(tmp_path)

        def sanitize(weights):
            out = dict(weights)
            out["w.weight"] = out["w.weight"].reshape(2, 2, -1).astype(mx.int32)
            return out

        plan = _discover_sanitize_plan(sanitize, idx)
        info = plan["w.weight"]
        assert info["recipe"][0][0] == "reshape"
        assert info["recipe"][1][0] == "astype"

        result = _DiscoveredPlan(plan, idx).pop("w.weight")
        assert result.shape == (2, 2, 8)
        assert result.dtype == mx.int32
        np.testing.assert_array_equal(
            np.array(result),
            np.arange(32, dtype=np.int32).reshape(2, 2, 8),
        )

    def test_astype_then_reshape_replays(self, tmp_path):
        idx = self._idx(tmp_path)

        def sanitize(weights):
            out = dict(weights)
            out["w.weight"] = out["w.weight"].astype(mx.int32).reshape(2, 2, -1)
            return out

        plan = _discover_sanitize_plan(sanitize, idx)
        info = plan["w.weight"]
        assert info["recipe"][0][0] == "astype"
        assert info["recipe"][1][0] == "reshape"

        result = _DiscoveredPlan(plan, idx).pop("w.weight")
        assert result.shape == (2, 2, 8)
        assert result.dtype == mx.int32
        np.testing.assert_array_equal(
            np.array(result),
            np.arange(32, dtype=np.int32).reshape(2, 2, 8),
        )


# =============================================================================
# End-to-end: oQ2.5 routed-layer boost
# =============================================================================


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestQuantizeOqStreamingOq25:
    def test_oq25_end_to_end_synthetic_moe(self, tmp_path):
        """oQ2.5 output: 2-bit affine base with routed expert down_proj
        selected at 3-bit through the routed-layer budget plan."""
        from safetensors.numpy import save_file as np_save

        src = tmp_path / "src"
        src.mkdir()
        h = 128
        np_save(
            {
                "model.layers.0.mlp.switch_mlp.down_proj.weight": np.random.randn(
                    8, h, h
                ).astype(np.float32),
                "model.layers.0.mlp.switch_mlp.gate_proj.weight": np.random.randn(
                    8, h, h
                ).astype(np.float32),
                "model.layers.0.mlp.switch_mlp.up_proj.weight": np.random.randn(
                    8, h, h
                ).astype(np.float32),
                "model.layers.0.self_attn.q_proj.weight": np.random.randn(h, h).astype(
                    np.float32
                ),
                "model.layers.0.input_layernorm.weight": np.ones(h, dtype=np.float32),
            },
            str(src / "model.safetensors"),
        )
        (src / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["TestModelForCausalLM"],
                    "model_type": "test_oq25",
                    "num_hidden_layers": 1,
                    "hidden_size": h,
                    "num_experts": 8,
                    "vocab_size": 256,
                }
            ),
            encoding="utf-8",
        )
        (src / "oq_sensitivity_map.json").write_text(
            json.dumps({"0": 0.1}), encoding="utf-8"
        )

        out = tmp_path / "out"
        quantize_oq_streaming(str(src), str(out), oq_level=2.5)

        config = json.loads((out / "config.json").read_text())
        q = config["quantization"]
        assert q["bits"] == 2
        assert q["group_size"] == 64
        assert q["mode"] == "affine"
        down = q.get("model.layers.0.mlp.switch_mlp.down_proj")
        assert down is not None
        assert down["bits"] == 3

        tensors = {}
        for sf in out.glob("*.safetensors"):
            tensors.update(mx.load(str(sf)))
        assert "model.layers.0.mlp.switch_mlp.down_proj.scales" in tensors
        assert "model.layers.0.mlp.switch_mlp.gate_proj.scales" in tensors
        assert "model.layers.0.mlp.switch_mlp.up_proj.scales" in tensors
