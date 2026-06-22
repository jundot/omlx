# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Nudge.tier and ValidationResult.budget extensions.

These cover the additive fields added for Forge retry support:
  * Nudge.tier (int, default 0) — backward compatible.
  * ValidationResult.budget (ErrorBudget | None, default None).
  * Optional ``tier`` parameter on all nudge generator functions.
  * Serialization of ``tier`` and ``budget`` in ValidationResult.to_dict().
"""
from omlx.api.guardrails.budget import ErrorBudget
from omlx.api.guardrails.nudge import (
    missing_params_nudge,
    retry_nudge,
    tool_arg_validation_nudge,
    unknown_tool_nudge,
)
from omlx.api.guardrails.types import (
    KIND_RETRY,
    KIND_TOOL_ARG_VALIDATION,
    KIND_UNKNOWN_TOOL,
    Nudge,
    ValidationResult,
)


class TestNudgeTierField:
    def test_default_tier_is_zero(self):
        n = Nudge(role="user", content="x", kind=KIND_RETRY)
        assert n.tier == 0

    def test_tier_can_be_set(self):
        n = Nudge(role="user", content="x", kind=KIND_RETRY, tier=2)
        assert n.tier == 2

    def test_tier_accepts_zero_explicitly(self):
        n = Nudge(role="tool", content="x", kind=KIND_UNKNOWN_TOOL, tier=0)
        assert n.tier == 0

    def test_is_still_frozen(self):
        n = Nudge(role="user", content="x", kind=KIND_RETRY, tier=1)
        try:
            n.tier = 5  # type: ignore
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestNudgeGeneratorsAcceptTier:
    def test_retry_nudge_default_tier(self):
        assert retry_nudge().tier == 0

    def test_retry_nudge_accepts_tier(self):
        assert retry_nudge(tier=1).tier == 1

    def test_unknown_tool_nudge_default_tier(self):
        assert unknown_tool_nudge("bad", ["a"]).tier == 0

    def test_unknown_tool_nudge_accepts_tier(self):
        assert unknown_tool_nudge("bad", ["a"], tier=3).tier == 3

    def test_tool_arg_validation_nudge_default_tier(self):
        assert tool_arg_validation_nudge("t", "v", "str").tier == 0

    def test_tool_arg_validation_nudge_accepts_tier(self):
        assert tool_arg_validation_nudge("t", "v", "str", tier=2).tier == 2

    def test_missing_params_nudge_default_tier(self):
        assert missing_params_nudge("t", ["p"]).tier == 0

    def test_missing_params_nudge_accepts_tier(self):
        assert missing_params_nudge("t", ["p"], tier=2).tier == 2

    def test_retry_nudge_other_fields_unchanged_with_tier(self):
        n = retry_nudge(tier=1)
        assert n.role == "user"
        assert n.kind == KIND_RETRY
        assert "tool" in n.content.lower()


class TestValidationResultBudget:
    def test_budget_defaults_to_none(self):
        vr = ValidationResult(checks=[], passed=True)
        assert vr.budget is None

    def test_budget_can_be_set(self):
        b = ErrorBudget(max_retries=5, max_tool_errors=1)
        vr = ValidationResult(checks=[], passed=True, budget=b)
        assert vr.budget is b
        assert vr.budget.max_retries == 5

    def test_to_dict_without_budget_omits_key(self):
        vr = ValidationResult(checks=[], passed=True)
        d = vr.to_dict()
        assert "budget" not in d

    def test_to_dict_with_budget_serializes(self):
        b = ErrorBudget(max_retries=5, max_tool_errors=1)
        vr = ValidationResult(checks=[], passed=True, budget=b)
        d = vr.to_dict()
        assert d["budget"] == {
            "max_retries": 5,
            "max_tool_errors": 1,
            "recommended_action": "retry",
        }


class TestValidationResultToDictWithTier:
    def test_nudge_dict_includes_tier(self):
        nudge = Nudge(
            role="tool", content="retry", kind=KIND_UNKNOWN_TOOL, tier=2
        )
        vr = ValidationResult(checks=[], nudge=nudge, passed=False)
        d = vr.to_dict()
        assert d["nudge"]["tier"] == 2

    def test_nudge_dict_includes_default_tier(self):
        nudge = Nudge(role="user", content="x", kind=KIND_RETRY)
        vr = ValidationResult(checks=[], nudge=nudge, passed=False)
        d = vr.to_dict()
        assert d["nudge"]["tier"] == 0

    def test_to_dict_with_nudge_tier_and_budget(self):
        nudge = Nudge(
            role="tool", content="fix", kind=KIND_TOOL_ARG_VALIDATION, tier=1
        )
        budget = ErrorBudget()
        vr = ValidationResult(checks=[], nudge=nudge, passed=False, budget=budget)
        d = vr.to_dict()
        assert d["nudge"]["tier"] == 1
        assert "budget" in d
        assert d["budget"]["max_retries"] == 3
