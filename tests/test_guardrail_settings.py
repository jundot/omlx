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


class TestNewForgeSettings:
    def test_inject_respond_tool_defaults_false(self):
        s = ForgeGuardrailsSettings()
        assert s.inject_respond_tool is False

    def test_enforce_mcp_prerequisites_defaults_false(self):
        s = ForgeGuardrailsSettings()
        assert s.enforce_mcp_prerequisites is False

    def test_to_dict_includes_new_fields(self):
        s = ForgeGuardrailsSettings(inject_respond_tool=True)
        d = s.to_dict()
        assert d["inject_respond_tool"] is True
        assert "enforce_mcp_prerequisites" in d

    def test_from_dict_reads_new_fields(self):
        d = {
            "inject_respond_tool": True,
            "enforce_mcp_prerequisites": True,
        }
        s = ForgeGuardrailsSettings.from_dict(d)
        assert s.inject_respond_tool is True
        assert s.enforce_mcp_prerequisites is True

    def test_round_trip_with_new_fields(self):
        original = ForgeGuardrailsSettings(
            inject_respond_tool=True,
            enforce_mcp_prerequisites=True,
        )
        d = original.to_dict()
        restored = ForgeGuardrailsSettings.from_dict(d)
        assert restored == original


class TestAdminRoutesWiring:
    def test_admin_routes_has_inject_respond_tool_field(self):
        from pathlib import Path

        routes_path = Path(__file__).resolve().parent.parent / "omlx" / "admin" / "routes.py"
        src = routes_path.read_text()
        assert "forge_guardrails_inject_respond_tool" in src

    def test_admin_routes_has_enforce_mcp_prerequisites_field(self):
        from pathlib import Path

        routes_path = Path(__file__).resolve().parent.parent / "omlx" / "admin" / "routes.py"
        src = routes_path.read_text()
        assert "forge_guardrails_enforce_mcp_prerequisites" in src
