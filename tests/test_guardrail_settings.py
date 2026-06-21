"""Unit tests for ForgeGuardrailsSettings integration."""
from omlx.settings import GlobalSettings, ForgeGuardrailsSettings


class TestForgeGuardrailsSettings:
    def test_defaults_all_off(self):
        s = ForgeGuardrailsSettings()
        assert s.validation_enabled is False
        assert s.strict_tool_args is False
        assert s.include_validation_metadata is False

    def test_to_dict(self):
        s = ForgeGuardrailsSettings(validation_enabled=True)
        d = s.to_dict()
        assert d["validation_enabled"] is True
        assert d["strict_tool_args"] is False
        assert d["include_validation_metadata"] is False

    def test_from_dict(self):
        d = {
            "validation_enabled": True,
            "strict_tool_args": True,
            "include_validation_metadata": True,
        }
        s = ForgeGuardrailsSettings.from_dict(d)
        assert s.validation_enabled is True
        assert s.strict_tool_args is True
        assert s.include_validation_metadata is True

    def test_from_dict_defaults(self):
        s = ForgeGuardrailsSettings.from_dict({})
        assert s.validation_enabled is False

    def test_round_trip(self):
        original = ForgeGuardrailsSettings(
            validation_enabled=True, strict_tool_args=True, include_validation_metadata=True
        )
        d = original.to_dict()
        restored = ForgeGuardrailsSettings.from_dict(d)
        assert restored == original


class TestGlobalSettingsIntegration:
    def test_global_has_forge_guardrails(self):
        gs = GlobalSettings()
        assert hasattr(gs, "forge_guardrails")
        assert isinstance(gs.forge_guardrails, ForgeGuardrailsSettings)

    def test_global_defaults_off(self):
        gs = GlobalSettings()
        assert gs.forge_guardrails.validation_enabled is False

    def test_global_to_dict_includes_forge_guardrails(self):
        gs = GlobalSettings()
        d = gs.to_dict()
        assert "forge_guardrails" in d
        assert "validation_enabled" in d["forge_guardrails"]

    def test_global_from_dict(self):
        d = {"forge_guardrails": {"validation_enabled": True}}
        gs = GlobalSettings.from_dict(d)
        assert gs.forge_guardrails.validation_enabled is True
