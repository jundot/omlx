# SPDX-License-Identifier: Apache-2.0
"""Tests for the small-but-critical role floor.

Roles that decide something discrete - a sparse-attention top-k selection, a
gated residual write, an attention read pattern - do not average their
quantization error away. On a fine-grained MoE they are a rounding error of the
checkpoint, so they are held above the level's base bits: hard selectors and the
MTP fusion projections in full precision, attention and gated-residual roles at
8-bit.

Attention and gated-residual roles pass a cost gate: each candidate is priced
against the per-level policy and pinned only while its own extra bytes and its
role's total stay inside a budget, so a dense model keeps the allocator's
decision instead of being repriced.
"""

import pytest

from omlx import oq
from omlx.oq import (
    _build_quant_plan,
    _is_mtp_path,
    _is_mtp_protected_tensor,
    _role_floor,
    _role_floor_gated_overrides,
    universal_quant_predicate,
)

# Hard top-k selectors stay in full precision, across family spellings.
HARD_SELECTOR_PATHS = [
    "language_model.model.layers.11.self_attn.indexer.index_qk_proj",
    "language_model.model.layers.11.self_attn.indexer.wk",
    "language_model.model.layers.11.self_attn.indexer.wq_b",
    "language_model.model.layers.11.self_attn.indexer.weights_proj",
    "mtp.layers.0.self_attn.indexer.index_qk_proj",
]

# MTP fusion projections stay in full precision, like the Qwen3.5/3.6 ``mtp.fc``
# precedent: aggressively quantizing them collapses draft acceptance.
MTP_FUSION_PATHS = [
    "mtp.fc_embedding",
    "mtp.fc_hidden",
]

# Unconditionally pinned to 8-bit: tiny by construction.
UNCONDITIONAL_Q8_PATHS = [
    "language_model.model.layers.11.ple.key_proj",
    "language_model.model.layers.11.ple.value_proj",
    # MTP copies of the backbone roles stay on the same format.
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down",
    "mtp.hyper_connection_mixer.input_mix_weight_up",
]

# Paths the floor must not touch.
UNCOVERED_PATHS = [
    "language_model.model.layers.11.mlp.gate",
    "language_model.model.layers.11.mlp.switch_mlp.gate_proj",
    "language_model.model.layers.11.mlp.switch_mlp.down_proj",
    "language_model.model.layers.11.ple.ple_embedding.ngram_embedding.shards.7",
    "language_model.model.layers.11.mlp.shared_expert.down_proj",
    "language_model.model.layers.11.self_attn.q_norm",
    # Not a selector: a substring match on "indexer" would have swept this in.
    "language_model.model.layers.11.mlp.my_indexer_thing",
    "lm_head",
    "language_model.model.embed_tokens",
]

Q8_SPEC = {"bits": 8, "group_size": 64, "mode": "affine"}


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


# A fine-grained MoE: experts dominate, attention is a rounding error.
MOE_SHAPES = {
    **{
        f"model.layers.{i}.mlp.switch_mlp.gate_proj": (512, 640, 2560)
        for i in range(12)
    },
    "model.layers.0.self_attn.q_proj": (12288, 2560),
    "model.layers.0.self_attn.k_proj": (512, 2560),
    "model.layers.0.linear_attn.in_proj_qkv": (10240, 2560),
    "model.layers.0.attn_hyper_connection.input_mix_weight_up": (10240, 2560),
    "model.layers.0.mlp.gate": (512, 2560),
}

# A dense model: attention is a large share, so the gate must not fire.
DENSE_SHAPES = {
    "model.layers.0.self_attn.q_proj": (4096, 4096),
    "model.layers.0.self_attn.k_proj": (4096, 4096),
    "model.layers.0.self_attn.v_proj": (4096, 4096),
    "model.layers.0.self_attn.o_proj": (4096, 4096),
    "model.layers.0.mlp.down_proj": (11008, 4096),
    "model.embed_tokens": (32000, 4096),
}


class TestRoleFloorUnconditional:
    """Roles that are tiny by construction, at any level and model."""

    @pytest.mark.parametrize("path", HARD_SELECTOR_PATHS)
    def test_hard_selectors_stay_full_precision(self, path):
        assert _role_floor(path, {}) is False

    @pytest.mark.parametrize("path", MTP_FUSION_PATHS)
    def test_mtp_fusion_stays_full_precision(self, path):
        assert _role_floor(path, {}) is False
        assert _is_mtp_protected_tensor(f"{path}.weight") is True

    @pytest.mark.parametrize("path", UNCONDITIONAL_Q8_PATHS)
    def test_tiny_roles_are_q8(self, path):
        assert _role_floor(path, {}) == Q8_SPEC

    @pytest.mark.parametrize("path", UNCOVERED_PATHS)
    def test_uncovered_paths_get_no_opinion(self, path):
        assert _role_floor(path, {}) is None

    def test_weight_suffix_is_normalized(self):
        assert _role_floor("model.layers.0.indexer.wk.weight", {}) is False
        assert _role_floor("mtp.layers.0.self_attn.q_proj.scales", {}) == Q8_SPEC

    def test_gated_roles_are_not_unconditional(self):
        # These need the cost gate, so the unconditional half must not claim
        # them.
        assert _role_floor("model.layers.0.self_attn.q_proj", {}) is None
        assert _role_floor("model.layers.0.linear_attn.in_proj_qkv", {}) is None
        assert (
            _role_floor("model.layers.0.attn_hyper_connection.block_inject_weight", {})
            is None
        )

    def test_fused_family_invariants_are_left_to_their_own_rules(self):
        # Inkling fuses Q/K/V/R and its loader requires Q8, so the floor must
        # not claim it as a full-precision hard selector.
        assert _role_floor("model.layers.0.self_attn.qkvr_proj", {}) is None
        assert (
            universal_quant_predicate(
                "model.layers.0.self_attn.qkvr_proj",
                None,
                {"model_type": "inkling"},
                4,
            )
            == Q8_SPEC
        )


class TestSensitivityProxyStaysUniform:
    """The measuring proxy must not inherit the policy it is measuring."""

    @pytest.mark.parametrize("level", [2, 4, 6])
    def test_proxy_ignores_the_floor(self, level):
        config = {"_oq_proxy": True}
        for path in (
            "model.layers.0.self_attn.indexer.wk",
            "mtp.fc_hidden",
            "mtp.fc_embedding",
            "model.layers.0.self_attn.q_proj",
        ):
            result = universal_quant_predicate(path, None, config, level)
            assert result is not False
            assert result != Q8_SPEC


class TestMtpPathConsistency:
    """Every caller must agree on what counts as MTP."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("mtp.layers.0.self_attn.q_proj", True),
            ("mtp.fc_embedding", True),
            ("language_model.model.mtp.layers.0.self_attn.q_proj", True),
            ("model.layers.0.self_attn.q_proj", False),
            # A substring match on "mtp" would wrongly claim this one.
            ("model.layers.0.mtmp_thing", False),
        ],
    )
    def test_is_mtp_path(self, path, expected):
        assert _is_mtp_path(path) is expected

    def test_mtp_paths_are_left_to_the_unconditional_half(self):
        shapes = {
            "mtp.layers.0.self_attn.q_proj": (12288, 2560),
            "model.layers.0.self_attn.q_proj": (12288, 2560),
            **{
                f"model.layers.{i}.mlp.switch_mlp.gate_proj": (512, 640, 2560)
                for i in range(12)
            },
        }
        overrides = _role_floor_gated_overrides(shapes, {}, 4)
        assert "mtp.layers.0.self_attn.q_proj" not in overrides
        assert "model.layers.0.self_attn.q_proj" in overrides


class TestRoleFloorInPredicate:
    """The floor must win over the per-level policy and the boost map."""

    @pytest.mark.parametrize("level", [2, 2.5, 2.7, 3, 3.5, 4, 5, 6, 8])
    def test_floor_holds_at_every_level(self, level):
        for path in HARD_SELECTOR_PATHS + MTP_FUSION_PATHS:
            assert universal_quant_predicate(path, None, {}, level) is False
        for path in UNCONDITIONAL_Q8_PATHS:
            assert universal_quant_predicate(path, None, {}, level) == Q8_SPEC

    def test_floor_beats_base_bits_at_oq2(self):
        assert (
            universal_quant_predicate("model.layers.11.indexer.wk", None, {}, 2)
            is False
        )

    def test_floor_beats_boost_map(self):
        path = "mtp.layers.0.self_attn.q_proj"
        config = {"_oq_boost_map": {path: {"bits": 2, "group_size": 64}}}
        assert universal_quant_predicate(path, None, config, 4) == Q8_SPEC

    def test_glm_indexer_invariant_still_wins(self):
        # GLM must keep Q8 (fused loader), not the floor's full precision.
        path = "model.layers.11.self_attn.indexer.wk"
        config = {"model_type": "glm5_next"}
        assert universal_quant_predicate(path, None, config, 4) == Q8_SPEC

    @pytest.mark.parametrize(
        "name",
        [
            "mtp.fc.weight",
            "mtp.0.hc_head.fn",
            "mtp.0.e_proj.weight",
            "mtp.0.markov_head.w",
        ],
    )
    def test_existing_mtp_full_precision_is_never_downgraded(self, name):
        # The floor may only ever raise precision: names the MTP guard already
        # keeps in full precision must stay there at every level.
        assert _role_floor(name, {}) is None
        assert _is_mtp_protected_tensor(name) is True
        for level in (2, 3, 4, 6, 8):
            assert oq._get_predicate_bits(name, {}, level, 64)[0] is None


class TestRoleFloorCostGate:
    """Attention and gated-residual roles are pinned only while they are cheap."""

    def test_moe_pins_attention_mixer_and_linear_attention(self):
        overrides = _role_floor_gated_overrides(MOE_SHAPES, {}, 4)
        assert overrides["model.layers.0.self_attn.q_proj"] == Q8_SPEC
        assert overrides["model.layers.0.self_attn.k_proj"] == Q8_SPEC
        assert overrides["model.layers.0.linear_attn.in_proj_qkv"] == Q8_SPEC
        assert (
            overrides["model.layers.0.attn_hyper_connection.input_mix_weight_up"]
            == Q8_SPEC
        )
        # Routed experts and the MoE gate are never touched.
        assert not any("switch_mlp" in path for path in overrides)
        assert "model.layers.0.mlp.gate" not in overrides

    def test_moe_role_cost_is_inside_the_budget(self):
        # The premise of the case above: the role really is cheap here.
        overrides = _role_floor_gated_overrides(MOE_SHAPES, {}, 4)
        total = sum(_numel(shape) for shape in MOE_SHAPES.values())
        attention = sum(_numel(MOE_SHAPES[path]) for path in overrides)
        assert attention / total < 0.02

    def test_dense_model_is_left_to_the_allocator(self):
        assert _role_floor_gated_overrides(DENSE_SHAPES, {}, 4) == {}

    def test_mla_low_rank_pairs_are_attention(self):
        shapes = {
            "model.layers.0.self_attn.q_a_proj": (1536, 7168),
            "model.layers.0.self_attn.kv_b_proj": (32768, 512),
            **{
                f"model.layers.{i}.mlp.switch_mlp.gate_proj": (512, 640, 2560)
                for i in range(12)
            },
        }
        overrides = _role_floor_gated_overrides(shapes, {}, 4)
        assert overrides["model.layers.0.self_attn.q_a_proj"] == Q8_SPEC
        assert overrides["model.layers.0.self_attn.kv_b_proj"] == Q8_SPEC

    def test_role_budget(self):
        assert oq._ROLE_FLOOR_MAX_COST_SHARE == 0.035

    def test_role_budget_boundary(self, monkeypatch):
        # Inside the budget the role is pinned; outside it drops as a whole.
        assert _role_floor_gated_overrides(MOE_SHAPES, {}, 4)
        monkeypatch.setattr(oq, "_ROLE_FLOOR_MAX_COST_SHARE", 0.0)
        assert _role_floor_gated_overrides(MOE_SHAPES, {}, 4) == {}

    def test_coarse_moe_fits_the_budget(self):
        # A coarse MoE (few large experts) costs more than a fine-grained one
        # but still fits, which is what raising the budget buys.
        shapes = {
            "model.layers.0.self_attn.q_proj": (4096, 4096),
            "model.layers.0.self_attn.k_proj": (1024, 4096),
            "model.layers.0.self_attn.v_proj": (1024, 4096),
            "model.layers.0.self_attn.o_proj": (4096, 4096),
            **{
                f"model.layers.{i}.mlp.experts.gate_proj": (8, 14336, 4096)
                for i in range(32)
            },
        }
        overrides = _role_floor_gated_overrides(shapes, {}, 4)
        assert overrides["model.layers.0.self_attn.q_proj"] == Q8_SPEC

    def test_structural_pricing_keeps_the_budget_honest(self):
        # The Qwen4-Exp PLE rows are 160 elements wide, so pricing them at the
        # default group size 64 makes them look unquantizable and counts them at
        # 16 bits. That inflates the budget enough to admit a role that does not
        # actually fit, which is why the invariant layout is priced in.
        tables = {
            f"language_model.model.ple.ple_embedding.ngram_embedding.shards.{i}": (
                2500012,
                160,
            )
            for i in range(4)
        }
        shapes = {
            **tables,
            **{f"model.layers.{i}.self_attn.q_proj": (4096, 4096) for i in range(13)},
            **{
                f"model.layers.{i}.mlp.experts.gate_proj": (8, 14336, 4096)
                for i in range(4)
            },
        }
        invariant = {
            path: {"bits": 4, "group_size": 32, "mode": "affine"} for path in tables
        }
        # Priced at group 32 the attention role really is over budget.
        assert _role_floor_gated_overrides(shapes, {}, 4, invariant) == {}
        # Priced at group 64 the table is counted at 16 bits and the role slips
        # through on a budget that does not exist.
        assert _role_floor_gated_overrides(shapes, {}, 4) != {}

    def test_per_tensor_budget_boundary(self, monkeypatch):
        # A single expensive member cannot hide behind the role budget.
        monkeypatch.setattr(oq, "_ROLE_FLOOR_MAX_TENSOR_COST_SHARE", 0.005)
        assert "model.layers.0.self_attn.q_proj" in _role_floor_gated_overrides(
            MOE_SHAPES, {}, 4
        )
        monkeypatch.setattr(oq, "_ROLE_FLOOR_MAX_TENSOR_COST_SHARE", 0.0)
        assert _role_floor_gated_overrides(MOE_SHAPES, {}, 4) == {}

    def test_cost_is_level_aware(self):
        # The same roles are cheap at a high level and not at the lowest one.
        assert _role_floor_gated_overrides(MOE_SHAPES, {}, 6)
        assert _role_floor_gated_overrides(DENSE_SHAPES, {}, 6) == {}

    def test_empty_shapes_is_a_noop(self):
        assert _role_floor_gated_overrides({}, {}, 4) == {}


class TestRoleFloorInBudgetPlan:
    """The plan must price the floor up front instead of letting the cap drop it."""

    @pytest.mark.parametrize("level", [2, 4, 6])
    def test_plan_seeds_the_floor(self, level):
        plan = _build_quant_plan(
            MOE_SHAPES,
            {},
            level,
            target_bpw={2: 2.9, 4: 4.6, 6: 6.5}[level],
            hard_cap_bpw={2: 3.0, 4: 4.7, 6: 6.6}[level],
        )
        assert plan.boost_map["model.layers.0.self_attn.q_proj"] == Q8_SPEC
        assert (
            plan.boost_map["model.layers.0.attn_hyper_connection.input_mix_weight_up"]
            == Q8_SPEC
        )
        # Routed experts keep the per-level base bits.
        assert all(
            plan.boost_map.get(path, {}).get("bits", 0) != 8
            for path in MOE_SHAPES
            if "switch_mlp" in path
        )

    def test_plan_leaves_a_dense_model_alone(self):
        plan = _build_quant_plan(DENSE_SHAPES, {}, 4, target_bpw=4.6, hard_cap_bpw=4.7)
        for path in DENSE_SHAPES:
            # The sensitivity allocator may still boost these to 5/6 bits; the
            # floor must not pin them to Q8 the way it does for a checkpoint
            # where attention is a rounding error.
            assert plan.boost_map.get(path, {}).get("bits", 0) != 8

    def test_over_budget_plan_matches_the_base_branch(self, monkeypatch):
        # An over-budget role changes nothing: nothing is repinned and no other
        # tensor moves, so the plan is identical to one built with no gated
        # floor at all.
        with_floor = _build_quant_plan(
            DENSE_SHAPES, {}, 4, target_bpw=4.6, hard_cap_bpw=4.7
        )
        monkeypatch.setattr(oq, "_role_floor_gated_overrides", lambda *a, **k: {})
        without_floor = _build_quant_plan(
            DENSE_SHAPES, {}, 4, target_bpw=4.6, hard_cap_bpw=4.7
        )
        assert with_floor.boost_map == without_floor.boost_map
        assert with_floor.effective_bpw == without_floor.effective_bpw
