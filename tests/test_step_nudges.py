# SPDX-License-Identifier: Apache-2.0
"""Unit tests for step_nudge generator (3-tier escalation)."""
from omlx.api.guardrails.nudge import step_nudge
from omlx.api.guardrails.types import KIND_STEP


class TestStepNudge:
    def test_role_is_user(self):
        n = step_nudge("respond", ["search", "read"], tier=1)
        assert n.role == "user"

    def test_kind_is_step(self):
        n = step_nudge("respond", ["search"], tier=1)
        assert n.kind == KIND_STEP

    def test_tier1_polite(self):
        n = step_nudge("respond", ["search", "read"], tier=1)
        assert n.tier == 1
        assert "respond" in n.content
        assert "search" in n.content
        assert "read" in n.content
        # Polite tone
        assert "cannot" in n.content.lower() or "must first" in n.content.lower()

    def test_tier2_direct(self):
        n = step_nudge("respond", ["search"], tier=2)
        assert n.tier == 2
        assert "search" in n.content
        # More direct — shorter, imperative
        assert "pick one" in n.content.lower()

    def test_tier3_aggressive(self):
        n = step_nudge("respond", ["search"], tier=3)
        assert n.tier == 3
        assert "search" in n.content
        # Aggressive — STOP / MUST
        content_upper = n.content.upper()
        assert "STOP" in content_upper or "MUST" in content_upper

    def test_tier_clamped_to_3(self):
        n = step_nudge("respond", ["search"], tier=5)
        assert n.tier == 3

    def test_tier_clamped_to_1(self):
        n = step_nudge("respond", ["search"], tier=0)
        assert n.tier == 1


class TestStepNudgeTierProgression:
    """Verify tier escalation matches Forge's pattern."""

    def test_tier_progresses_1_2_3(self):
        tiers = []
        for attempts in range(0, 5):
            tier = min(attempts + 1, 3)
            n = step_nudge("respond", ["search"], tier=tier)
            tiers.append(n.tier)
        assert tiers == [1, 2, 3, 3, 3]

    def test_each_tier_content_is_distinct(self):
        contents = set()
        for tier in (1, 2, 3):
            n = step_nudge("respond", ["search"], tier=tier)
            contents.add(n.content)
        assert len(contents) == 3  # all distinct

    def test_tier3_is_strongest(self):
        n1 = step_nudge("respond", ["search"], tier=1)
        n3 = step_nudge("respond", ["search"], tier=3)
        # Tier 3 content is longer / more emphatic
        assert len(n3.content) >= len(n1.content)
