# SPDX-License-Identifier: Apache-2.0
"""Tests for oQ (oMLX Universal Dynamic Quantization)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from omlx.oq import (
    OQ_LEVELS,
    _LEVEL_BITS,
    _OQ_BPW_TARGETS,
    _TrackedTensor,
    _bpw_targets_for_level,
    _build_quant_plan,
    _discover_sanitize_plan,
    _extract_layer_index,
    _format_size,
    _forward_layer,
    _get_predicate_bits,
    _is_audio_tensor,
    _is_moe_router,
    _is_vision_tensor,
    _LazyTensorIndex,
    _normalize_quant_path,
    _quantize_chunked,
    _should_quantize_tensor,
    estimate_memory,
    make_predicate,
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
        result = universal_quant_predicate("model.layers.0.mlp.gate", module, moe_config)
        assert result is False  # MoE router gates kept fp16 (some models lack to_quantized)

    def test_shared_expert_gate_8bit(self, moe_config, module):
        result = universal_quant_predicate("model.layers.0.shared_expert_gate", module, moe_config)
        assert isinstance(result, dict) and result["bits"] == 8

    def test_non_quantizable_module_skipped(self, dense_config, module):
        cfg = {**dense_config, "_oq_non_quantizable": {
            "language_model.model.per_layer_model_projection",
        }}
        assert universal_quant_predicate(
            "language_model.model.per_layer_model_projection.weight", module, cfg
        ) is False

    def test_non_quantizable_set_does_not_affect_other_paths(self, dense_config, module):
        cfg = {**dense_config, "_oq_non_quantizable": {
            "language_model.model.per_layer_model_projection",
        }}
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
        assert universal_quant_predicate("visual.encoder.layers.0.self_attn.q_proj", module, dense_config) is False

    def test_patch_embed_not_quantized(self, dense_config, module):
        assert universal_quant_predicate("model.patch_embed.proj", module, dense_config) is False

    def test_ssm_alpha_not_quantized(self, dense_config, module):
        assert universal_quant_predicate("model.layers.0.ssm_alpha", module, dense_config) is False

    def test_ssm_beta_not_quantized(self, dense_config, module):
        assert universal_quant_predicate("model.layers.0.ssm_beta", module, dense_config) is False

    def test_a_log_not_quantized(self, dense_config, module):
        assert universal_quant_predicate("model.layers.0.a_log", module, dense_config) is False

    def test_mamba_d_not_quantized(self, dense_config, module):
        assert universal_quant_predicate("model.layers.0.mixer.D", module, dense_config) is False

    def test_time_decay_not_quantized(self, dense_config, module):
        assert universal_quant_predicate("model.layers.0.time_decay", module, dense_config) is False

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
        result = universal_quant_predicate("model.layers.0.ssm_output", module, dense_config)
        assert isinstance(result, dict)
        assert result["bits"] == 8

    def test_lm_head_6bit(self, dense_config, module):
        result = universal_quant_predicate("lm_head", module, dense_config)
        assert isinstance(result, dict)
        assert result["bits"] == 6

    def test_mla_kv_b_proj_6bit(self, dense_config, module):
        result = universal_quant_predicate("model.layers.0.self_attn.kv_b_proj", module, dense_config)
        assert isinstance(result, dict)
        assert result["bits"] == 6

    def test_dense_o_proj_5bit(self, dense_config, module):
        result = universal_quant_predicate("model.layers.5.self_attn.o_proj", module, dense_config)
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
        result = universal_quant_predicate("model.layers.0.mlp.gate", module, moe_config)
        assert result is False  # MoE router gates kept fp16

    def test_150_expert_group_size_128(self, module):
        config = {"num_hidden_layers": 32, "num_local_experts": 200, "hidden_size": 2048}
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
        assert _is_audio_tensor(
            "audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight"
        ) is True

    def test_is_audio_tensor_embed_audio_not_excluded(self):
        # embed_audio.embedding_projection is the projection from audio output
        # to text hidden — should be quantizable like embed_vision counterpart.
        assert _is_audio_tensor(
            "embed_audio.embedding_projection.weight"
        ) is False

    def test_is_audio_tensor_language_model(self):
        assert _is_audio_tensor(
            "language_model.model.layers.0.self_attn.q_proj.weight"
        ) is False

    def test_is_audio_tensor_vision_tower(self):
        assert _is_audio_tensor(
            "vision_tower.layers.0.self_attn.k_proj.weight"
        ) is False

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

    def test_strip_existing_bit_suffix(self):
        assert resolve_output_name("Qwen3.5-122B-A10B-8bit", 4) == "Qwen3.5-122B-A10B-oQ4"

    def test_strip_existing_oq_suffix(self):
        assert resolve_output_name("Qwen3.5-122B-A10B-oQ6", 2) == "Qwen3.5-122B-A10B-oQ2"

    def test_strip_existing_enhanced_suffix(self):
        assert resolve_output_name("Qwen3.5-122B-A10B-oQ4e", 2) == "Qwen3.5-122B-A10B-oQ2"

    def test_all_levels(self):
        for level in OQ_LEVELS:
            result = resolve_output_name("Model-7B", level)
            assert result == f"Model-7B-oQ{level}"

    def test_bfloat16_default_no_suffix(self):
        assert resolve_output_name("Llama-3-8B", 4, "bfloat16") == "Llama-3-8B-oQ4"

    def test_float16_appends_fp16_suffix(self):
        assert resolve_output_name("Llama-3-8B", 4, "float16") == "Llama-3-8B-oQ4-fp16"

    def test_float16_strips_existing_dtype_suffix(self):
        assert (
            resolve_output_name("Model-oQ6-fp16", 4, "float16")
            == "Model-oQ4-fp16"
        )

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

        bits, gs, mode = _get_predicate_bits(
            "language_model.mtp.fc.weight", {}, 4, 64
        )
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
        bits, _, _ = _get_predicate_bits(
            "mtp.0.block.attn.wq_a.weight", {}, 4, 64
        )
        assert bits is not None

    def test_normal_weights_unaffected(self):
        from omlx.oq import _get_predicate_bits

        bits, _, _ = _get_predicate_bits(
            "model.layers.0.attn.q_proj.weight", {}, 4, 64
        )
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

    def test_quantization_config(self):
        assert validate_quantizable({"quantization_config": {"bits": 4}}) is False

    def test_fp8_native_is_quantizable(self):
        # Native FP8 models (MiniMax, DeepSeek) should be quantizable
        assert validate_quantizable({"quantization_config": {"quant_method": "fp8"}}) is True

    def test_non_fp8_quantization_config(self):
        # Other quant methods (gptq, awq) are already quantized
        assert validate_quantizable({"quantization_config": {"quant_method": "gptq"}}) is False


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
            "_oq_boost_map": {"lm_head": {"bits": 6, "group_size": 64, "mode": "affine"}},
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
        assert _should_quantize_tensor("model.layers.0.self_attn.q_proj.weight", (4096, 4096)) is True

    def test_should_not_quantize_1d(self):
        assert _should_quantize_tensor("model.layers.0.input_layernorm.weight", (4096,)) is False

    def test_should_not_quantize_bias(self):
        assert _should_quantize_tensor("model.layers.0.self_attn.q_proj.bias", (4096,)) is False

    def test_should_not_quantize_norm(self):
        assert _should_quantize_tensor("model.layers.0.rmsnorm.weight", (4096, 4096)) is False

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
        bits, gs, mode = _get_predicate_bits("model.layers.10.mlp.gate_proj.weight", config, 4, 64)
        assert bits == 4
        assert gs == 64
        assert mode == "affine"

    def test_get_predicate_bits_3bit_affine(self):
        config = {"num_hidden_layers": 32}
        bits, gs, mode = _get_predicate_bits("model.layers.10.mlp.gate_proj.weight", config, 3, 64)
        # oQ3 → base 3-bit → affine
        assert bits == 3
        assert mode == "affine"

    def test_get_predicate_bits_8bit(self):
        config = {"num_hidden_layers": 32}
        bits, gs, mode = _get_predicate_bits("model.layers.10.mlp.gate_proj.weight", config, 8, 64)
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
        plan = _build_quant_plan(named_shapes, config, 4, target_bpw=4.6, hard_cap_bpw=4.7)
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
        assert _bpw_targets_for_level(3) == (3.5, 3.7)
        assert _bpw_targets_for_level(3.5) == (3.8, 4.0)
        assert _bpw_targets_for_level(4) == (4.6, 4.7)
        assert _bpw_targets_for_level(5) == (5.5, 5.7)
        assert _bpw_targets_for_level(6) == (6.5, 6.7)

    def test_bpw_targets_for_level_returns_none_for_minimal(self):
        assert _bpw_targets_for_level(8) is None

    def test_level_bits_covers_all_oq_levels(self):
        for level in OQ_LEVELS:
            assert level in _LEVEL_BITS

    def test_budget_plan_oq2_enabled(self):
        assert 2 in _OQ_BPW_TARGETS
        assert _bpw_targets_for_level(2) == (2.8, 3.0)

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
            named_shapes, config, oq_level,
            target_bpw=target, hard_cap_bpw=cap,
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
                named_shapes[f"model.layers.{i}.mlp.experts.{e}.down_proj"] = (4096, 1024)
                named_shapes[f"model.layers.{i}.mlp.experts.{e}.up_proj"] = (1024, 4096)
                named_shapes[f"model.layers.{i}.mlp.experts.{e}.gate_proj"] = (1024, 4096)
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
            named_shapes[f"backbone.layers.{i}.mixer.shared_experts.up_proj"] = (3712, 2688)
            named_shapes[f"backbone.layers.{i}.mixer.shared_experts.down_proj"] = (2688, 3712)
        for i in range(n_layers):
            named_shapes[f"backbone.layers.{i}.mixer.switch_mlp.fc1"] = (128, 1856, 2688)
            named_shapes[f"backbone.layers.{i}.mixer.switch_mlp.fc2"] = (128, 2688, 1856)
        sensitivity = {str(i): 0.1 / (i + 1) for i in range(n_layers)}
        config = {
            "num_hidden_layers": n_layers,
            "_oq_use_budget_plan": True,
            "_oq_sensitivity_map": sensitivity,
        }
        plan = _build_quant_plan(
            named_shapes, config, 2, target_bpw=2.8, hard_cap_bpw=3.0
        )
        assert plan.effective_bpw >= 2.7, (
            f"Expected bpw >= 2.7, got {plan.effective_bpw:.2f}"
        )
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


# =============================================================================
# Test _LazyTensorIndex
# =============================================================================


def _write_safetensors(path, tensors):
    """Write a minimal safetensors file from {name: np.ndarray} dict."""
    import json
    import struct

    header = {}
    data_parts = []
    offset = 0
    dtype_map = {np.float16: "F16", np.float32: "F32", np.dtype("<f2"): "F16"}
    for name, arr in tensors.items():
        raw = arr.tobytes()
        sf_dtype = dtype_map.get(arr.dtype, "F16")
        header[name] = {
            "dtype": sf_dtype,
            "shape": list(arr.shape),
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
                np.array(result.astype(mx.float32)), expected.astype(np.float32),
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
        assert r.transform == "slice"

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
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(8, 8).astype(np.float16),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(4, 8).astype(np.float16),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(16, 8).astype(np.float16),
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


# =============================================================================
# Test GPTQ quantization
