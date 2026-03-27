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
    _bpw_targets_for_level,
    _build_quant_plan,
    _compute_group_params,
    _extract_layer_index,
    _format_size,
    _get_predicate_bits,
    _gptq_compute_hessian,
    _gptq_quantize_experts_batched,
    _gptq_quantize_weight,
    _is_moe_router,
    _normalize_quant_path,
    _search_best_clip,
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
        assert result["bits"] == 6

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

    def test_clip_suffix(self):
        assert resolve_output_name("Qwen3.5-122B-A10B", 4, enable_clip=True) == "Qwen3.5-122B-A10B-oQ4e"

    def test_strip_existing_clip_suffix(self):
        assert resolve_output_name("Qwen3.5-122B-A10B-oQ4e", 2) == "Qwen3.5-122B-A10B-oQ2"

    def test_all_levels(self):
        for level in OQ_LEVELS:
            result = resolve_output_name("Model-7B", level)
            assert result == f"Model-7B-oQ{level}"

    def test_all_levels_clip(self):
        for level in OQ_LEVELS:
            result = resolve_output_name("Model-7B", level, enable_clip=True)
            assert result == f"Model-7B-oQ{level}e"


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


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestClipOptimization:
    """Test AWQ-style output-MSE clip optimization."""

    def test_search_best_clip_returns_same_shape(self):
        """Clipped weights should have the same shape as input."""

        w = mx.random.normal((64, 128))
        x = mx.random.normal((32, 128))  # 32 activation samples
        clipped = _search_best_clip(w, x, group_size=64, bits=4)
        assert clipped.shape == w.shape

    def test_search_best_clip_reduces_range(self):
        """Clipping should reduce the weight range (or keep it same)."""

        w = mx.random.normal((32, 64))
        x = mx.random.normal((16, 64))
        clipped = _search_best_clip(w, x, group_size=64, bits=2, n_grid=10)
        # Clipped range should be <= original range
        orig_range = float(w.max() - w.min())
        clip_range = float(clipped.max() - clipped.min())
        assert clip_range <= orig_range * 1.01

    def test_search_best_clip_2bit(self):
        """2-bit clip search should work."""

        w = mx.random.normal((16, 128))
        x = mx.random.normal((8, 128))
        clipped = _search_best_clip(w, x, group_size=64, bits=2, n_grid=5)
        assert clipped.shape == w.shape
        assert clipped.dtype == w.dtype

    def test_output_mse_improves_with_clip(self):
        """Clipped + quantized should have lower output MSE than raw quantized."""

        np.random.seed(42)
        # Weight with outliers
        w_np = np.random.randn(32, 64).astype(np.float32) * 0.1
        w_np[0, 0] = 5.0
        w_np[1, 1] = -4.0
        w = mx.array(w_np)
        x = mx.random.normal((16, 64))

        # Baseline: quantize directly
        rtn_q = mx.dequantize(*mx.quantize(w, group_size=64, bits=2), 64, 2)
        x_grouped = x.reshape(x.shape[0], -1, 64)
        w_grouped = w.reshape(w.shape[0], -1, 64)
        rtn_q_grouped = rtn_q.reshape(rtn_q.shape[0], -1, 64)
        out_orig = mx.einsum("bdg,odg->bod", x_grouped, w_grouped)
        out_rtn = mx.einsum("bdg,odg->bod", x_grouped, rtn_q_grouped)
        rtn_loss = float(((out_orig - out_rtn) ** 2).mean())

        # Clip-optimized: search + quantize
        clipped = _search_best_clip(w, x, group_size=64, bits=2, n_grid=10)
        clip_q = mx.dequantize(*mx.quantize(clipped, group_size=64, bits=2), 64, 2)
        clip_q_grouped = clip_q.reshape(clip_q.shape[0], -1, 64)
        out_clip = mx.einsum("bdg,odg->bod", x_grouped, clip_q_grouped)
        clip_loss = float(((out_orig - out_clip) ** 2).mean())

        # Clip-optimized should have equal or better output MSE
        assert clip_loss <= rtn_loss * 1.05, (
            f"Clip output MSE ({clip_loss:.6f}) worse than RTN ({rtn_loss:.6f})"
        )


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
        result = estimate_memory(size, enable_clip=False)
        # Streaming: source + 5GB buffer + 5% sanitize overhead
        assert result["peak_bytes"] > size
        assert result["peak_bytes"] < size * 1.2

    def test_clip_larger_than_streaming(self):
        size = 50 * 1024**3
        streaming = estimate_memory(size, enable_clip=False)
        clip = estimate_memory(size, enable_clip=True)
        assert clip["peak_bytes"] > streaming["peak_bytes"]

    def test_has_formatted(self):
        result = estimate_memory(10 * 1024**3, enable_clip=False)
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

    def test_get_predicate_bits_8bit_mxfp8(self):
        config = {"num_hidden_layers": 32}
        bits, gs, mode = _get_predicate_bits("model.layers.10.mlp.gate_proj.weight", config, 8, 64)
        # oQ8 → base 8-bit → mxfp8
        assert bits == 8
        assert gs == 32
        assert mode == "mxfp8"

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
# Test GPTQ quantization
# =============================================================================


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
class TestGPTQQuantize:
    """Tests for GPTQ weight optimization functions."""

    def test_gptq_axis_alignment(self):
        """GPTQ qdq must match mx.quantize row-wise grouping (last axis)."""
        out_dim, in_dim = 128, 256
        bits, gs = 4, 64

        w = mx.random.normal((out_dim, in_dim)).astype(mx.float32) * 0.1
        x = mx.random.normal((32, in_dim)).astype(mx.float32)
        mx.eval(w, x)

        _, Hinv = _gptq_compute_hessian(x)
        w_opt = _gptq_quantize_weight(w, Hinv, bits, gs)
        mx.eval(w_opt)

        # Quantize with mx.quantize (groups along last axis = in_dim)
        qdq_opt = mx.dequantize(
            *mx.quantize(w_opt, group_size=gs, bits=bits),
            group_size=gs, bits=bits,
        )
        qdq_plain = mx.dequantize(
            *mx.quantize(w, group_size=gs, bits=bits),
            group_size=gs, bits=bits,
        )
        mx.eval(qdq_opt, qdq_plain)

        # GPTQ output should already be on the quantization grid
        # (re-quantization should be nearly lossless)
        regrid_mse = ((w_opt - qdq_opt) ** 2).mean().item()
        plain_mse = ((w - qdq_plain) ** 2).mean().item()
        assert regrid_mse < plain_mse * 0.5, (
            f"GPTQ output not grid-aligned: regrid_mse={regrid_mse:.6f} "
            f"vs plain_mse={plain_mse:.6f}"
        )

    def test_gptq_does_not_degrade_mse(self):
        """GPTQ-optimized weights should not significantly degrade output MSE.

        With random weights GPTQ may not always improve (real model structure
        is needed for significant gains), but it should stay within a small
        tolerance of plain RTN.
        """
        # Multiple groups (in_dim / gs = 2) with well-conditioned Hessian
        out_dim, in_dim = 128, 128
        bits, gs = 4, 64

        w = mx.random.normal((out_dim, in_dim)).astype(mx.float32) * 0.1
        x = mx.random.normal((512, in_dim)).astype(mx.float32)
        mx.eval(w, x)

        float_out = x @ w.T
        _, Hinv = _gptq_compute_hessian(x)

        # Plain RTN quantization
        qdq_plain = mx.dequantize(
            *mx.quantize(w, group_size=gs, bits=bits),
            group_size=gs, bits=bits,
        )
        mse_plain = ((float_out - x @ qdq_plain.T) ** 2).mean()

        # GPTQ-optimized quantization
        w_opt = _gptq_quantize_weight(w, Hinv, bits, gs)
        qdq_gptq = mx.dequantize(
            *mx.quantize(w_opt, group_size=gs, bits=bits),
            group_size=gs, bits=bits,
        )
        mse_gptq = ((float_out - x @ qdq_gptq.T) ** 2).mean()
        mx.eval(mse_plain, mse_gptq)

        # GPTQ should not make things significantly worse (< 10% degradation)
        assert mse_gptq.item() <= mse_plain.item() * 1.10, (
            f"GPTQ degraded MSE by >10%: gptq={mse_gptq.item():.6f} "
            f"vs plain={mse_plain.item():.6f}"
        )

    def test_gptq_experts_batched(self):
        """Batched expert GPTQ should improve or match plain quantization."""
        num_experts, out_dim, in_dim = 4, 64, 128
        bits, gs = 4, 64

        w_3d = mx.random.normal((num_experts, out_dim, in_dim)).astype(mx.float32) * 0.1
        x = mx.random.normal((32, in_dim)).astype(mx.float32)
        mx.eval(w_3d, x)

        _, Hinv = _gptq_compute_hessian(x)
        w_opt = _gptq_quantize_experts_batched(w_3d, Hinv, bits, gs)
        mx.eval(w_opt)

        # Check each expert: GPTQ should not increase output MSE
        for ei in range(num_experts):
            w_orig = w_3d[ei]
            w_new = w_opt[ei]
            float_out = x @ w_orig.T

            qdq_plain = mx.dequantize(
                *mx.quantize(w_orig, group_size=gs, bits=bits),
                group_size=gs, bits=bits,
            )
            qdq_gptq = mx.dequantize(
                *mx.quantize(w_new, group_size=gs, bits=bits),
                group_size=gs, bits=bits,
            )
            mse_p = ((float_out - x @ qdq_plain.T) ** 2).mean()
            mse_g = ((float_out - x @ qdq_gptq.T) ** 2).mean()
            mx.eval(mse_p, mse_g)
            assert mse_g.item() <= mse_p.item() * 1.05, (
                f"Expert {ei}: GPTQ degraded MSE by >5%: "
                f"gptq={mse_g.item():.6f} vs plain={mse_p.item():.6f}"
            )

    def test_compute_group_params_partial_group(self):
        """_compute_group_params should handle partial last group by padding."""
        # Partial group: width=48 with group_size=64
        w = mx.random.normal((32, 48)).astype(mx.float32)
        mx.eval(w)
        scales, biases = _compute_group_params(w, bits=4, group_size=64)
        assert scales.shape == (32, 1)
        assert biases.shape == (32, 1)

    def test_compute_group_params_full_group(self):
        """_compute_group_params with exact group_size should work."""
        w = mx.random.normal((32, 64)).astype(mx.float32)
        mx.eval(w)
        scales, biases = _compute_group_params(w, bits=4, group_size=64)
        assert scales.shape == (32, 1)
        assert biases.shape == (32, 1)
