# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen4-Exp protected quantization floor.

The Qwen4-Exp (Qwen3.8-Flash-Next) sparse-attention indexer, MoE gate and
gated-residual mixers decide discrete things, so their quantization error is
not averaged out the way it is in a weighted sum. They stay in full precision,
and the attention / gated-residual / PLE key-value / MTP fusion projections are
pinned to a fixed 8-bit affine format at every oQ level.
"""

import pytest

from omlx.oq import (
    _build_quant_plan,
    _qwen4_exp_protected_floor,
    universal_quant_predicate,
)

QWEN4_CONFIG = {
    "model_type": "qwen4_exp",
    "num_hidden_layers": 48,
    "num_local_experts": 512,
    "hidden_size": 2560,
}

# Paths the floor keeps in full precision (never quantized).
FULL_PRECISION_PATHS = [
    "language_model.model.layers.11.self_attn.indexer.index_qk_proj",
    "mtp.layers.0.self_attn.indexer.index_qk_proj",
    "language_model.model.layers.11.mlp.shared_expert_gate",
    "mtp.layers.0.mlp.shared_expert_gate",
]

# Paths the floor pins to 8-bit affine.
Q8_PATHS = [
    "language_model.model.layers.11.self_attn.q_proj",
    "language_model.model.layers.11.self_attn.k_proj",
    "language_model.model.layers.11.self_attn.v_proj",
    "language_model.model.layers.11.self_attn.o_proj",
    "language_model.model.layers.11.ple.key_proj",
    "language_model.model.layers.11.ple.value_proj",
    "language_model.model.layers.11.attn_hyper_connection.block_inject_weight",
    "language_model.model.layers.11.attn_hyper_connection.input_mix_weight_up",
    "language_model.model.layers.11.attn_hyper_connection.input_mix_weight_down",
    "language_model.model.layers.11.mlp_hyper_connection.block_inject_weight",
    "language_model.model.layers.11.mlp_hyper_connection.input_mix_weight_up",
    "language_model.model.layers.11.mlp_hyper_connection.input_mix_weight_down",
    "language_model.model.hyper_connection_mixer.input_mix_weight_up",
    "language_model.model.hyper_connection_mixer.input_mix_weight_down",
    # MTP copies stay on the same format as the backbone.
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down",
    "mtp.hyper_connection_mixer.input_mix_weight_up",
    "mtp.fc_embedding",
    "mtp.fc_hidden",
]

# Paths the floor must not touch.
UNCOVERED_PATHS = [
    "language_model.model.layers.11.mlp.gate",
    "language_model.model.layers.11.mlp.switch_mlp.gate_proj",
    "language_model.model.layers.11.mlp.switch_mlp.down_proj",
    "language_model.model.layers.11.linear_attn.in_proj_qkv",
    "language_model.model.layers.11.ple.ple_embedding.ngram_embedding.shards.7",
    "language_model.model.layers.11.mlp.shared_expert.down_proj",
    "language_model.model.layers.11.self_attn.q_norm",
    "lm_head",
    "language_model.model.embed_tokens",
]

Q8_SPEC = {"bits": 8, "group_size": 64, "mode": "affine"}


class TestQwen4ExpProtectedFloor:
    """Unit tests for _qwen4_exp_protected_floor."""

    @pytest.mark.parametrize("path", FULL_PRECISION_PATHS)
    def test_indexer_and_gate_stay_full_precision(self, path):
        assert _qwen4_exp_protected_floor(path, QWEN4_CONFIG) is False

    @pytest.mark.parametrize("path", Q8_PATHS)
    def test_quality_critical_paths_are_q8(self, path):
        assert _qwen4_exp_protected_floor(path, QWEN4_CONFIG) == Q8_SPEC

    @pytest.mark.parametrize("path", UNCOVERED_PATHS)
    def test_uncovered_paths_get_no_opinion(self, path):
        assert _qwen4_exp_protected_floor(path, QWEN4_CONFIG) is None

    def test_weight_suffix_is_normalized(self):
        # The predicate is also called with raw tensor names.
        assert (
            _qwen4_exp_protected_floor(
                "language_model.model.layers.11.self_attn.q_proj.weight",
                QWEN4_CONFIG,
            )
            == Q8_SPEC
        )
        assert (
            _qwen4_exp_protected_floor(
                "language_model.model.layers.11.self_attn.q_proj.scales",
                QWEN4_CONFIG,
            )
            == Q8_SPEC
        )

    def test_other_model_types_are_untouched(self):
        for config in (
            {"model_type": "qwen3_5_moe"},
            {"model_type": "glm5_next"},
            {"model_type": "deepseek_v41"},
            {},
        ):
            for path in FULL_PRECISION_PATHS + Q8_PATHS:
                assert _qwen4_exp_protected_floor(path, config) is None

    def test_text_config_model_type_is_honored(self):
        config = {
            "model_type": "qwen3_5_vl",
            "text_config": {"model_type": "qwen4_exp"},
        }
        assert (
            _qwen4_exp_protected_floor(
                "language_model.model.layers.11.self_attn.q_proj", config
            )
            == Q8_SPEC
        )


class TestQwen4ExpFloorInPredicate:
    """The floor must win over the per-level policy and the boost map."""

    @pytest.mark.parametrize("level", [2, 2.5, 2.7, 3, 3.5, 4, 5, 6, 8])
    def test_floor_holds_at_every_level(self, level):
        for path in FULL_PRECISION_PATHS:
            assert universal_quant_predicate(path, None, QWEN4_CONFIG, level) is False
        for path in Q8_PATHS:
            assert universal_quant_predicate(path, None, QWEN4_CONFIG, level) == Q8_SPEC

    def test_floor_beats_base_bits_at_oq2(self):
        path = "language_model.model.layers.11.self_attn.q_proj"
        result = universal_quant_predicate(path, None, QWEN4_CONFIG, 2)
        assert result == Q8_SPEC

    def test_floor_beats_boost_map(self):
        path = "language_model.model.layers.11.self_attn.q_proj"
        config = {
            **QWEN4_CONFIG,
            "_oq_boost_map": {path: {"bits": 2, "group_size": 64}},
        }
        assert universal_quant_predicate(path, None, config, 4) == Q8_SPEC

    def test_non_qwen4_models_keep_their_level_policy(self):
        path = "language_model.model.layers.11.self_attn.q_proj"
        generic = {"model_type": "qwen3_5_moe"}
        assert universal_quant_predicate(path, None, generic, 4) != Q8_SPEC


class TestQwen4ExpFloorInBudgetPlan:
    """The plan must price the floor up front instead of letting the cap drop it."""

    def _named_shapes(self):
        return {
            "language_model.model.layers.11.self_attn.q_proj": (12288, 2560),
            "language_model.model.layers.11.attn_hyper_connection.input_mix_weight_up": (
                10240,
                2560,
            ),
            "language_model.model.layers.11.self_attn.indexer.index_qk_proj": (
                640,
                2560,
            ),
            "language_model.model.layers.11.mlp.switch_mlp.gate_proj": (512, 640, 2560),
        }

    @pytest.mark.parametrize("level", [2, 4, 6])
    def test_plan_seeds_qwen4_floor(self, level):
        plan = _build_quant_plan(
            self._named_shapes(),
            QWEN4_CONFIG,
            level,
            target_bpw={2: 2.9, 4: 4.6, 6: 6.5}[level],
            hard_cap_bpw={2: 3.0, 4: 4.7, 6: 6.6}[level],
        )
        assert plan.boost_map["language_model.model.layers.11.self_attn.q_proj"] == (
            Q8_SPEC
        )
        assert (
            plan.boost_map[
                "language_model.model.layers.11.attn_hyper_connection.input_mix_weight_up"
            ]
            == Q8_SPEC
        )
        # Full-precision members are never quantized, so they must not be seeded.
        assert (
            "language_model.model.layers.11.self_attn.indexer.index_qk_proj"
            not in plan.boost_map
        )
