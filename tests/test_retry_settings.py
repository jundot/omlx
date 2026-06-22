"""Unit tests for retry/compaction settings on ForgeGuardrailsSettings (Task 4)."""
from omlx.settings import GlobalSettings, ForgeGuardrailsSettings


class TestRetryDefaults:
    def test_max_retries_default(self):
        s = ForgeGuardrailsSettings()
        assert s.max_retries == 3

    def test_max_tool_errors_default(self):
        s = ForgeGuardrailsSettings()
        assert s.max_tool_errors == 2

    def test_compaction_strategy_default(self):
        s = ForgeGuardrailsSettings()
        assert s.compaction_strategy == "none"


class TestRetryToDict:
    def test_to_dict_includes_new_fields(self):
        s = ForgeGuardrailsSettings()
        d = s.to_dict()
        assert d["max_retries"] == 3
        assert d["max_tool_errors"] == 2
        assert d["compaction_strategy"] == "none"

    def test_to_dict_with_custom_values(self):
        s = ForgeGuardrailsSettings(
            max_retries=5, max_tool_errors=4, compaction_strategy="sliding_window"
        )
        d = s.to_dict()
        assert d["max_retries"] == 5
        assert d["max_tool_errors"] == 4
        assert d["compaction_strategy"] == "sliding_window"


class TestRetryFromDict:
    def test_from_dict_with_new_fields(self):
        d = {
            "validation_enabled": True,
            "max_retries": 7,
            "max_tool_errors": 6,
            "compaction_strategy": "tiered",
        }
        s = ForgeGuardrailsSettings.from_dict(d)
        assert s.max_retries == 7
        assert s.max_tool_errors == 6
        assert s.compaction_strategy == "tiered"

    def test_from_dict_backward_compat_defaults(self):
        d = {"validation_enabled": True}
        s = ForgeGuardrailsSettings.from_dict(d)
        assert s.max_retries == 3
        assert s.max_tool_errors == 2
        assert s.compaction_strategy == "none"

    def test_from_dict_empty(self):
        s = ForgeGuardrailsSettings.from_dict({})
        assert s.max_retries == 3
        assert s.max_tool_errors == 2
        assert s.compaction_strategy == "none"


class TestRetryRoundTrip:
    def test_round_trip_custom_values(self):
        original = ForgeGuardrailsSettings(
            max_retries=10,
            max_tool_errors=5,
            compaction_strategy="tiered",
            validation_enabled=True,
        )
        d = original.to_dict()
        restored = ForgeGuardrailsSettings.from_dict(d)
        assert restored == original

    def test_round_trip_defaults(self):
        original = ForgeGuardrailsSettings()
        d = original.to_dict()
        restored = ForgeGuardrailsSettings.from_dict(d)
        assert restored == original


class TestGlobalSettingsRetryIntegration:
    def test_global_has_new_fields(self):
        gs = GlobalSettings()
        assert gs.forge_guardrails.max_retries == 3
        assert gs.forge_guardrails.max_tool_errors == 2
        assert gs.forge_guardrails.compaction_strategy == "none"

    def test_global_to_dict_includes_new_fields(self):
        gs = GlobalSettings()
        d = gs.to_dict()
        assert d["forge_guardrails"]["max_retries"] == 3
        assert d["forge_guardrails"]["max_tool_errors"] == 2
        assert d["forge_guardrails"]["compaction_strategy"] == "none"

    def test_global_from_dict_with_new_fields(self):
        d = {
            "forge_guardrails": {
                "max_retries": 8,
                "max_tool_errors": 4,
                "compaction_strategy": "sliding_window",
            }
        }
        gs = GlobalSettings.from_dict(d)
        assert gs.forge_guardrails.max_retries == 8
        assert gs.forge_guardrails.max_tool_errors == 4
        assert gs.forge_guardrails.compaction_strategy == "sliding_window"
