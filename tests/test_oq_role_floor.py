# SPDX-License-Identifier: Apache-2.0
"""Tests for the small-but-critical role floor.

The floor is the last consumer of the budget plan's byte budget: after the
format invariants, the mandatory consensus boosts, the protection floor and
the discretionary sensitivity stages have spent what they spent, the floored
roles - hard top-k selectors, PLE key/value reads, the MTP head's fusion
projections and its copies of the backbone roles, attention projections and
gated-residual mixers - may still buy Q8 with whatever band is left under
the level cap.

The contract the tests pin down, one clause per objection it answers:

  * raise-only: no tensor ever ends below what the same plan without the
    floor assigned it (the floor displaces no allocation);
  * cap-bound: the plan's effective bpw never exceeds ``hard_cap_bpw``, and a
    role that would cross the cap is dropped whole (the floor never overshoots
    the level, and role coverage stays uniform);
  * invariants and mandatory boosts first: ``lm_head`` / ``embed_tokens`` and
    family Q8 rules are untouched whatever the floor does;
  * MoE-only: a checkpoint without routed experts gets a plan that is
    byte-identical to the base branch's;
  * Q8, not full precision: floored roles are raised to the same 8-bit format
    the GLM indexer rule ships, and the predicate itself is left untouched.
"""

import pytest

from omlx import oq
from omlx.oq import (
    _build_quant_plan,
    _is_mtp_path,
    _is_mtp_protected_tensor,
    _role_floor_role,
    universal_quant_predicate,
)

Q8_SPEC = {"bits": 8, "group_size": 64, "mode": "affine"}

CLASSIFIED_ROLES = [
    # hard top-k selectors, matched per dot-segment
    ("language_model.model.layers.11.self_attn.indexer.index_qk_proj", "selector"),
    ("language_model.model.layers.11.self_attn.indexer.wk", "selector"),
    ("language_model.model.layers.11.self_attn.indexer.wq_b", "selector"),
    ("language_model.model.layers.11.self_attn.indexer.weights_proj", "selector"),
    # exact-match PLE reads
    ("language_model.model.layers.11.ple.key_proj", "ple_kv"),
    ("language_model.model.layers.11.ple.value_proj", "ple_kv"),
    # the split Qwen4-Exp MTP fusion pair
    ("mtp.fc_embedding", "mtp_fusion"),
    ("language_model.model.mtp.fc_hidden", "mtp_fusion"),
    # the MTP head's copies of the backbone roles
    ("mtp.layers.0.self_attn.q_proj", "mtp_attention"),
    ("mtp.layers.0.linear_attn.in_proj_qkv", "mtp_attention"),
    ("mtp.layers.0.attn_hyper_connection.input_mix_weight_up", "mtp_mixer"),
    ("mtp.hyper_connection_mixer.input_mix_weight_up", "mtp_mixer"),
    # attention, all spellings: softmax, MLA low-rank, linear, state-space
    ("model.layers.3.self_attn.q_proj", "attention"),
    ("model.layers.3.self_attn.o_proj", "attention"),
    ("model.layers.3.self_attn.q_a_proj", "attention"),
    ("model.layers.3.self_attn.kv_a_proj_with_mqa", "attention"),
    ("model.layers.3.linear_attn.in_proj_a", "attention"),
    ("model.layers.3.mamba.in_proj", "attention"),
    ("model.layers.3.mamba.dt_proj", "attention"),
    ("backbone.layers.3.mixer.in_proj", "attention"),
    ("backbone.layers.3.mixer.x_proj", "attention"),
    # gated residual / hyper-connection mixers across families
    ("model.layers.3.attn_hyper_connection.block_inject_weight", "mixer"),
    ("model.layers.3.mlp_hyper_connection.input_mix_weight_down", "mixer"),
    ("model.layers.3.hc_attn_fn", "mixer"),
    ("model.layers.3.hc_ffn.base", "mixer"),
    ("model.layers.3.attn_hc.scale", "mixer"),
    ("model.layers.3.ffn_hc.fn", "mixer"),
]

NOT_FLORED = [
    "model.layers.3.mlp.gate",
    "model.layers.3.mlp.switch_mlp.gate_proj",
    "model.layers.3.mlp.experts.gate_proj",
    "model.layers.3.mlp.shared_expert.down_proj",
    "model.layers.3.ple.ple_embedding.ngram_embedding.shards.7",
    "model.layers.3.self_attn.q_norm",
    # a substring match on "indexer" would have swept this in
    "model.layers.3.mlp.my_indexer_thing",
    # Nemotron-H names its MLP under ``mixer.`` too: not a mixing path
    "backbone.layers.3.mixer.up_proj",
    "backbone.layers.3.mixer.down_proj",
    "model.layers.3.mamba.A_log",
    "lm_head",
    "language_model.model.embed_tokens",
    # Inkling fuses Q/K/V/R and its loader requires Q8 by its own family rule
    "model.layers.0.self_attn.qkvr_proj",
]


class TestRoleClassification:
    @pytest.mark.parametrize("path,role", CLASSIFIED_ROLES)
    def test_floored_roles(self, path, role):
        assert _role_floor_role(path) == role

    @pytest.mark.parametrize("path", NOT_FLORED)
    def test_other_roles_are_not_claimed(self, path):
        assert _role_floor_role(path) is None

    def test_weight_suffix_is_normalized(self):
        assert _role_floor_role("model.layers.0.indexer.wk.weight") == "selector"

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("mtp.layers.0.self_attn.q_proj", True),
            ("mtp.fc_embedding", True),
            ("language_model.model.mtp.layers.0.self_attn.q_proj", True),
            ("model.layers.0.self_attn.q_proj", False),
            # a substring match on "mtp" would wrongly claim this one
            ("model.layers.0.mtmp_thing", False),
        ],
    )
    def test_is_mtp_path(self, path, expected):
        assert _is_mtp_path(path) is expected


class TestPredicateCarriesNoFloor:
    """The floor lives in the budget plan, so the predicate stays the policy."""

    @pytest.mark.parametrize("level", [2, 2.5, 3, 4, 5, 6, 8])
    def test_floored_paths_keep_the_per_level_policy(self, level):
        for path in (
            "model.layers.0.self_attn.indexer.wk",
            "model.layers.0.ple.key_proj",
            "mtp.fc_embedding",
            "mtp.fc_hidden",
        ):
            result = universal_quant_predicate(path, None, {}, level)
            assert result is not False
            assert result != Q8_SPEC

    def test_glm_indexer_invariant_still_wins(self):
        # GLM's fused loader keeps its own Q8 rule at the predicate level;
        # the floor neither needs nor shadows it.
        result = universal_quant_predicate(
            "model.layers.11.self_attn.indexer.wk",
            None,
            {"model_type": "glm5_next"},
            4,
        )
        assert result == Q8_SPEC

    def test_mtp_fusion_pair_is_not_carved_out_of_quantization(self):
        # The Qwen4-Exp split fusion gets Q8 from the floor, not full
        # precision: on the base branch it ships at the level's bits, and the
        # floor may only raise it inside the budget.
        assert _is_mtp_protected_tensor("mtp.fc_embedding.weight") is False
        assert _is_mtp_protected_tensor("mtp.fc_hidden.weight") is False
        assert _is_mtp_protected_tensor("mtp.fc.weight") is True


# A fine-grained MoE: experts dominate, the floored roles are a rounding error.
MOE_SHAPES = {
    **{
        f"model.layers.{i}.mlp.switch_mlp.gate_proj": (512, 640, 2560)
        for i in range(12)
    },
    "model.layers.0.self_attn.q_proj": (4096, 2560),
    "model.layers.0.self_attn.k_proj": (512, 2560),
    "model.layers.0.self_attn.v_proj": (512, 2560),
    "model.layers.0.self_attn.o_proj": (2560, 4096),
    "model.layers.0.linear_attn.in_proj_qkv": (5120, 2560),
    "model.layers.0.linear_attn.in_proj_z": (5120, 2560),
    "model.layers.0.self_attn.indexer.wk": (640, 2560),
    "model.layers.0.ple.key_proj": (4096, 2560),
    "model.layers.0.attn_hyper_connection.input_mix_weight_up": (5120, 2560),
    "mtp.layers.0.self_attn.q_proj": (4096, 2560),
    "mtp.fc_embedding": (2560, 2560),
    "mtp.fc_hidden": (2560, 2560),
    "model.layers.0.mlp.gate": (512, 2560),
    "model.embed_tokens": (151936, 2560),
    "lm_head": (151936, 2560),
}

# A dense checkpoint: no routed experts, so the floor must be a no-op.
DENSE_SHAPES = {
    "model.layers.0.self_attn.q_proj": (4096, 4096),
    "model.layers.0.self_attn.k_proj": (1024, 4096),
    "model.layers.0.self_attn.v_proj": (1024, 4096),
    "model.layers.0.self_attn.o_proj": (4096, 4096),
    "model.layers.0.mlp.gate_proj": (11008, 4096),
    "model.layers.0.mlp.down_proj": (4096, 11008),
    "model.embed_tokens": (151936, 4096),
    "lm_head": (151936, 4096),
}

FLOORED_IN_FIXTURE = [
    path
    for path in MOE_SHAPES
    if _role_floor_role(path) is not None and not path.startswith("mtp.")
] + ["mtp.layers.0.self_attn.q_proj", "mtp.fc_embedding", "mtp.fc_hidden"]


def _plan(shapes, level=4, target=4.6, cap=4.7):
    return _build_quant_plan(
        shapes, {}, level, target_bpw=target, hard_cap_bpw=cap, fixed_overrides=None
    )


def _no_floor_plan(shapes, monkeypatch, level=4, target=4.6, cap=4.7):
    def _disabled(
        named_shapes,
        config,
        oq_level,
        boost_map,
        fixed_overrides,
        total_bits_f,
        total_params,
        current_bpw,
        hard_cap_bpw,
    ):
        return total_bits_f, current_bpw, 0

    monkeypatch.setattr(oq, "_apply_role_floor", _disabled)
    return _plan(shapes, level, target, cap)


class TestFloorInBudgetPlan:
    def test_floored_roles_reach_q8_with_headroom(self):
        plan = _plan(MOE_SHAPES, level=2, target=2.8, cap=3.0)
        floored = [p for p in FLOORED_IN_FIXTURE if MOE_SHAPES[p]]
        for path in floored:
            entry = plan.boost_map.get(path)
            assert entry is not None, path
            assert entry["bits"] == 8, path
        assert plan.effective_bpw <= 3.0

    @pytest.mark.parametrize("level,target,cap", [(2, 2.8, 3.0), (4, 4.6, 4.7)])
    def test_raise_only(self, level, target, cap, monkeypatch):
        base = _no_floor_plan(MOE_SHAPES, monkeypatch, level, target, cap)
        floored = _plan(MOE_SHAPES, level, target, cap)
        for path in MOE_SHAPES:
            if path in base.boost_map and path not in floored.boost_map:
                pytest.fail(f"floor dropped {path}")
            got = floored.boost_map.get(path, base.boost_map.get(path))
            had = base.boost_map.get(path)
            if had is not None and got is not None:
                assert int(got["bits"]) >= int(had["bits"]), path

    @pytest.mark.parametrize(
        "level,target,cap", [(2, 2.8, 2.9), (2, 2.8, 3.0), (4, 4.5, 4.6), (4, 4.6, 4.7)]
    )
    def test_cap_is_never_exceeded(self, level, target, cap):
        plan = _plan(MOE_SHAPES, level=level, target=target, cap=cap)
        assert plan.effective_bpw <= cap + 1e-9

    @pytest.mark.parametrize(
        "level,target,cap", [(2, 2.8, 3.0), (3, 3.5, 3.7), (4, 4.6, 4.7)]
    )
    def test_invariants_and_mandatory_are_untouched(
        self, level, target, cap, monkeypatch
    ):
        base = _no_floor_plan(MOE_SHAPES, monkeypatch, level, target, cap)
        floored = _plan(MOE_SHAPES, level, target, cap)
        for key in ("lm_head", "model.embed_tokens"):
            assert floored.boost_map.get(key) == base.boost_map.get(key)

    def test_routed_experts_are_never_floored(self):
        plan = _plan(MOE_SHAPES, level=2, target=2.8, cap=3.0)
        assert not any(
            "switch_mlp" in path and spec["bits"] == 8
            for path, spec in plan.boost_map.items()
        )

    def test_dense_checkpoint_plan_is_identical(self, monkeypatch):
        base = _no_floor_plan(DENSE_SHAPES, monkeypatch, level=4, target=4.6, cap=4.7)
        floored = _plan(DENSE_SHAPES, 4, 4.6, 4.7)
        assert floored.boost_map == base.boost_map
        assert floored.effective_bpw == base.effective_bpw

    def test_role_over_cap_is_dropped_whole(self, monkeypatch):
        # A cap the base plan already sits on: the floor can buy nothing, and
        # because a role is applied whole or not at all, no floored tensor is
        # left at a partial 5/6-bit mixture either.
        base = _no_floor_plan(MOE_SHAPES, monkeypatch, level=4, target=4.6, cap=4.7)
        # The base plan lands on the cap: nothing is left for the floor.
        plan = _plan(MOE_SHAPES, level=4, target=4.6, cap=base.effective_bpw + 1e-9)
        for path in FLOORED_IN_FIXTURE:
            assert plan.boost_map.get(path) == base.boost_map.get(path), path

    def test_role_budget_switch_off_disables_the_attention_role(self, monkeypatch):
        base = _no_floor_plan(MOE_SHAPES, monkeypatch, level=2, target=2.8, cap=3.0)
        monkeypatch.setattr(oq, "_ROLE_FLOOR_MAX_COST_SHARE", 0.0)
        floored = _plan(MOE_SHAPES, level=2, target=2.8, cap=3.0)
        for path in FLOORED_IN_FIXTURE:
            assert floored.boost_map.get(path) == base.boost_map.get(path), path

    def test_tensor_budget_switch_off_disables_the_floor(self, monkeypatch):
        base = _no_floor_plan(MOE_SHAPES, monkeypatch, level=2, target=2.8, cap=3.0)
        monkeypatch.setattr(oq, "_ROLE_FLOOR_MAX_TENSOR_COST_SHARE", 0.0)
        floored = _plan(MOE_SHAPES, level=2, target=2.8, cap=3.0)
        assert floored.boost_map == base.boost_map

    def test_plan_is_deterministic(self):
        assert (
            _plan(MOE_SHAPES, 2, 2.8, 3.0).boost_map
            == _plan(MOE_SHAPES, 2, 2.8, 3.0).boost_map
        )


class TestApplyRoleFloorDirectly:
    """Priority under a contended band, and the all-or-nothing role unit."""

    def _totals(self, shapes, bits=4):
        params = sum(
            s[0] * s[1] if len(s) == 2 else s[0] * s[1] * s[2] for s in shapes.values()
        )
        return bits * params, params

    def _q8_delta_bits(self, shape, cur_bits=4):
        q8 = oq._tensor_quantized_bytes(shape, 8, 64, "affine")
        cur = oq._tensor_quantized_bytes(shape, cur_bits, 64, "affine")
        return 8 * (q8 - cur)

    SHAPES = {
        "model.layers.0.self_attn.indexer.wk": (640, 2560),
        "model.layers.0.self_attn.q_proj": (4096, 2560),
        "model.layers.0.self_attn.v_proj": (512, 2560),
        **{
            f"model.layers.{i}.mlp.switch_mlp.gate_proj": (512, 640, 2560)
            for i in range(6)
        },
    }

    def test_selector_role_wins_a_contended_band(self):
        total_bits_f, total_params = self._totals(self.SHAPES)
        current = total_bits_f / total_params
        selector_delta = self._q8_delta_bits(
            self.SHAPES["model.layers.0.self_attn.indexer.wk"]
        )
        attn_deltas = self._q8_delta_bits(
            self.SHAPES["model.layers.0.self_attn.q_proj"]
        ) + self._q8_delta_bits(self.SHAPES["model.layers.0.self_attn.v_proj"])
        # A band that pays for the selector alone.
        cap = current + 1.5 * selector_delta / total_params
        boost_map = {}
        new_bits, new_bpw, bumps = oq._apply_role_floor(
            self.SHAPES, {}, 4, boost_map, {}, total_bits_f, total_params, current, cap
        )
        assert boost_map["model.layers.0.self_attn.indexer.wk"] == Q8_SPEC
        assert "model.layers.0.self_attn.q_proj" not in boost_map
        assert "model.layers.0.self_attn.v_proj" not in boost_map
        assert new_bpw <= cap
        # The band is big enough for the attention role too: it is applied
        # whole - both members, not a partial layer mix.
        cap2 = current + (selector_delta + attn_deltas + 1) / total_params
        boost_map = {}
        _, bpw2, _ = oq._apply_role_floor(
            self.SHAPES, {}, 4, boost_map, {}, total_bits_f, total_params, current, cap2
        )
        assert boost_map["model.layers.0.self_attn.q_proj"] == Q8_SPEC
        assert boost_map["model.layers.0.self_attn.v_proj"] == Q8_SPEC
        assert bpw2 <= cap2

    def test_noop_inputs(self):
        total_bits_f, total_params = self._totals(self.SHAPES)
        boost_map = {}
        out = oq._apply_role_floor({}, {}, 4, boost_map, {}, total_bits_f, 0, 4.0, 4.7)
        assert out == (total_bits_f, 4.0, 0)
