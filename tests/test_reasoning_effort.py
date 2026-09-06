# SPDX-License-Identifier: Apache-2.0
"""Tests for reasoning_effort field in ModelSettings."""

from __future__ import annotations

from omlx.model_settings import ModelSettings


class TestReasoningEffortField:
    """Tests for reasoning_effort field in ModelSettings."""

    def test_default_is_none(self):
        """reasoning_effort defaults to None (use model default)."""
        s = ModelSettings()
        assert s.reasoning_effort is None

    def test_set_low(self):
        """reasoning_effort can be set to 'low'."""
        s = ModelSettings(reasoning_effort="low")
        assert s.reasoning_effort == "low"

    def test_set_medium(self):
        """reasoning_effort can be set to 'medium'."""
        s = ModelSettings(reasoning_effort="medium")
        assert s.reasoning_effort == "medium"

    def test_set_high(self):
        """reasoning_effort can be set to 'high'."""
        s = ModelSettings(reasoning_effort="high")
        assert s.reasoning_effort == "high"

    def test_set_xhigh(self):
        """reasoning_effort can be set to 'xhigh'."""
        s = ModelSettings(reasoning_effort="xhigh")
        assert s.reasoning_effort == "xhigh"

    def test_set_max(self):
        """reasoning_effort can be set to 'max'."""
        s = ModelSettings(reasoning_effort="max")
        assert s.reasoning_effort == "max"

    def test_to_dict_includes_reasoning_effort(self):
        """reasoning_effort is included in to_dict when set."""
        s = ModelSettings(reasoning_effort="high")
        d = s.to_dict()
        assert d["reasoning_effort"] == "high"

    def test_to_dict_omits_none(self):
        """reasoning_effort is omitted from to_dict when None."""
        s = ModelSettings()
        d = s.to_dict()
        assert "reasoning_effort" not in d

    def test_from_dict_with_reasoning_effort(self):
        """reasoning_effort is restored from dict."""
        data = {"reasoning_effort": "xhigh"}
        s = ModelSettings.from_dict(data)
        assert s.reasoning_effort == "xhigh"

    def test_from_dict_without_reasoning_effort(self):
        """reasoning_effort defaults to None when missing from dict."""
        data = {"temperature": 0.7}
        s = ModelSettings.from_dict(data)
        assert s.reasoning_effort is None

    def test_roundtrip(self):
        """reasoning_effort survives to_dict/from_dict roundtrip."""
        s = ModelSettings(reasoning_effort="medium")
        d = s.to_dict()
        restored = ModelSettings.from_dict(d)
        assert restored.reasoning_effort == "medium"

    def test_roundtrip_none(self):
        """None reasoning_effort survives roundtrip."""
        s = ModelSettings()
        d = s.to_dict()
        restored = ModelSettings.from_dict(d)
        assert restored.reasoning_effort is None


class TestReasoningEffortInProfiles:
    """Tests for reasoning_effort in profile fields."""

    def test_in_universal_profile_fields(self):
        """reasoning_effort is in UNIVERSAL_PROFILE_FIELDS."""
        from omlx.model_profiles import UNIVERSAL_PROFILE_FIELDS

        assert "reasoning_effort" in UNIVERSAL_PROFILE_FIELDS

    def test_not_in_model_specific_fields(self):
        """reasoning_effort is not in MODEL_SPECIFIC_PROFILE_FIELDS."""
        from omlx.model_profiles import MODEL_SPECIFIC_PROFILE_FIELDS

        assert "reasoning_effort" not in MODEL_SPECIFIC_PROFILE_FIELDS
