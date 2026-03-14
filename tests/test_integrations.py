"""Tests for the integrations module."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from omlx.integrations import get_integration, list_integrations
from omlx.integrations.codex import CodexIntegration
from omlx.integrations.opencode import OpenCodeIntegration
from omlx.integrations.openclaw import OpenClawIntegration


class TestIntegrationRegistry:
    def test_list_integrations(self):
        integrations = list_integrations()
        assert len(integrations) == 3
        names = {i.name for i in integrations}
        assert names == {"codex", "opencode", "openclaw"}

    def test_get_integration(self):
        assert get_integration("codex") is not None
        assert get_integration("opencode") is not None
        assert get_integration("openclaw") is not None
        assert get_integration("nonexistent") is None


class TestCodexIntegration:
    def test_get_command(self):
        codex = CodexIntegration()
        cmd = codex.get_command(port=8000, api_key="test-key", model="qwen3.5")
        assert "omlx launch codex" in cmd
        assert "--model qwen3.5" in cmd

    def test_get_command_no_model(self):
        codex = CodexIntegration()
        cmd = codex.get_command(port=8000, api_key="", model="")
        assert "select-a-model" in cmd

    def test_configure(self, tmp_path):
        codex = CodexIntegration()
        config_path = tmp_path / "codex" / "config.toml"
        with patch.object(CodexIntegration, "CONFIG_PATH", config_path):
            codex.configure(port=8000, api_key="test-key", model="qwen3.5")

        assert config_path.exists()
        content = config_path.read_text()
        assert 'model = "qwen3.5"' in content
        assert 'model_provider = "omlx"' in content
        assert 'base_url = "http://127.0.0.1:8000/v1"' in content
        assert 'env_key = "OMLX_API_KEY"' in content

    def test_configure_custom_host(self, tmp_path):
        codex = CodexIntegration()
        config_path = tmp_path / "codex" / "config.toml"
        with patch.object(CodexIntegration, "CONFIG_PATH", config_path):
            codex.configure(port=9000, api_key="key", model="test", host="192.168.1.100")

        content = config_path.read_text()
        assert 'base_url = "http://192.168.1.100:9000/v1"' in content

    def test_configure_creates_backup(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text('model = "old"')

        codex = CodexIntegration()
        with patch.object(CodexIntegration, "CONFIG_PATH", config_path):
            codex.configure(port=8000, api_key="", model="new")

        backups = list(tmp_path.glob("config.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == 'model = "old"'

    def test_type(self):
        codex = CodexIntegration()
        assert codex.type == "config_file"
        assert codex.display_name == "Codex"


class TestOpenCodeIntegration:
    def test_get_command(self):
        oc = OpenCodeIntegration()
        cmd = oc.get_command(port=8000, api_key="key", model="qwen3.5")
        assert "omlx launch opencode" in cmd
        assert "--model qwen3.5" in cmd

    def test_configure_new_file(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"

        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(port=8000, api_key="test-key", model="qwen3.5")

        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["provider"]["omlx"]["options"]["baseURL"] == "http://127.0.0.1:8000/v1"
        assert config["provider"]["omlx"]["npm"] == "@ai-sdk/openai-compatible"
        assert config["provider"]["omlx"]["options"]["apiKey"] == "test-key"
        assert config["provider"]["omlx"]["models"]["qwen3.5"]["name"] == "qwen3.5"
        assert config["model"] == "omlx/qwen3.5"

    def test_configure_custom_host(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"
        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(port=9000, api_key="key", model="test", host="10.0.0.5")

        config = json.loads(config_path.read_text())
        assert config["provider"]["omlx"]["options"]["baseURL"] == "http://10.0.0.5:9000/v1"

    def test_configure_preserves_existing(self, tmp_path):
        config_path = tmp_path / "opencode.json"
        existing = {
            "provider": {
                "ollama": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": "http://localhost:11434/v1",
                    },
                }
            },
            "logLevel": "INFO",
        }
        config_path.write_text(json.dumps(existing))

        oc = OpenCodeIntegration()
        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(port=9000, api_key="", model="llama")

        config = json.loads(config_path.read_text())
        # Existing provider preserved
        assert "ollama" in config["provider"]
        assert config["provider"]["ollama"]["options"]["baseURL"] == "http://localhost:11434/v1"
        # omlx provider added
        assert "omlx" in config["provider"]
        assert config["provider"]["omlx"]["options"]["baseURL"] == "http://127.0.0.1:9000/v1"
        # Other keys preserved
        assert config["logLevel"] == "INFO"

    def test_configure_creates_backup(self, tmp_path):
        config_path = tmp_path / "opencode.json"
        config_path.write_text('{"existing": true}')

        oc = OpenCodeIntegration()
        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(port=8000, api_key="", model="test")

        # Check backup was created
        backups = list(tmp_path.glob("opencode.*.bak"))
        assert len(backups) == 1
        backup_content = json.loads(backups[0].read_text())
        assert backup_content == {"existing": True}

    def test_configure_handles_invalid_json(self, tmp_path):
        config_path = tmp_path / "opencode.json"
        config_path.write_text("not valid json {{{")

        oc = OpenCodeIntegration()
        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(port=8000, api_key="key", model="test")

        # Should create new config despite invalid existing file
        config = json.loads(config_path.read_text())
        assert "omlx" in config["provider"]

    def test_configure_with_limits(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"

        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(
                port=8000, api_key="key", model="qwen3.5",
                context_window=32768, max_tokens=8192,
            )

        config = json.loads(config_path.read_text())
        model_config = config["provider"]["omlx"]["models"]["qwen3.5"]
        assert model_config["limit"]["context"] == 32768
        assert model_config["limit"]["output"] == 8192

    def test_configure_with_context_window_only(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"

        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(
                port=8000, api_key="key", model="qwen3.5", context_window=32768
            )

        config = json.loads(config_path.read_text())
        model_config = config["provider"]["omlx"]["models"]["qwen3.5"]
        assert model_config["limit"]["context"] == 32768
        assert model_config["limit"]["output"] == 32768

    def test_configure_without_limits(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"

        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(port=8000, api_key="key", model="qwen3.5")

        config = json.loads(config_path.read_text())
        model_config = config["provider"]["omlx"]["models"]["qwen3.5"]
        assert "limit" not in model_config

    def test_type(self):
        oc = OpenCodeIntegration()
        assert oc.type == "config_file"
        assert oc.display_name == "OpenCode"


class TestOpenClawIntegration:
    def test_get_command(self):
        ocl = OpenClawIntegration()
        cmd = ocl.get_command(port=8000, api_key="key", model="qwen3.5")
        assert "omlx launch openclaw" in cmd
        assert "--model qwen3.5" in cmd

    def test_configure_new_file(self, tmp_path):
        config_path = tmp_path / "openclaw" / "openclaw.json"

        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "CONFIG_PATH", config_path):
            ocl.configure(port=8000, api_key="test-key", model="qwen3.5")

        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["models"]["providers"]["omlx"]["baseUrl"] == "http://127.0.0.1:8000/v1"
        assert config["models"]["providers"]["omlx"]["api"] == "openai-completions"
        assert config["models"]["providers"]["omlx"]["apiKey"] == "test-key"
        assert config["agents"]["defaults"]["model"]["primary"] == "omlx/qwen3.5"
        assert config["tools"]["profile"] == "coding"

    def test_configure_custom_host(self, tmp_path):
        config_path = tmp_path / "openclaw" / "openclaw.json"
        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "CONFIG_PATH", config_path):
            ocl.configure(port=9000, api_key="key", model="test", host="192.168.1.100")

        config = json.loads(config_path.read_text())
        assert config["models"]["providers"]["omlx"]["baseUrl"] == "http://192.168.1.100:9000/v1"

    def test_configure_preserves_existing(self, tmp_path):
        config_path = tmp_path / "openclaw.json"
        existing = {
            "models": {
                "providers": {
                    "ollama": {"baseUrl": "http://localhost:11434"}
                }
            },
            "channels": {"telegram": {"enabled": True}},
        }
        config_path.write_text(json.dumps(existing))

        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "CONFIG_PATH", config_path):
            ocl.configure(port=9000, api_key="key", model="llama")

        config = json.loads(config_path.read_text())
        # Existing preserved
        assert "ollama" in config["models"]["providers"]
        assert config["channels"]["telegram"]["enabled"] is True
        # omlx added
        assert "omlx" in config["models"]["providers"]
        assert config["models"]["providers"]["omlx"]["baseUrl"] == "http://127.0.0.1:9000/v1"

    def test_configure_exec_approvals_coding(self, tmp_path):
        approvals_path = tmp_path / "exec-approvals.json"
        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "EXEC_APPROVALS_PATH", approvals_path):
            ocl.configure_exec_approvals(tools_profile="coding")

        config = json.loads(approvals_path.read_text())
        assert config["defaults"]["security"] == "full"
        assert config["defaults"]["ask"] == "off"

    def test_configure_exec_approvals_messaging(self, tmp_path):
        approvals_path = tmp_path / "exec-approvals.json"
        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "EXEC_APPROVALS_PATH", approvals_path):
            ocl.configure_exec_approvals(tools_profile="messaging")

        config = json.loads(approvals_path.read_text())
        assert config["defaults"]["security"] == "allowlist"
        assert config["defaults"]["ask"] == "on-miss"

    def test_configure_exec_approvals_preserves_existing(self, tmp_path):
        approvals_path = tmp_path / "exec-approvals.json"
        existing = {
            "version": 1,
            "socket": {"path": "/tmp/test.sock", "token": "abc"},
            "defaults": {"security": "deny", "ask": "always"},
        }
        approvals_path.write_text(json.dumps(existing))
        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "EXEC_APPROVALS_PATH", approvals_path):
            ocl.configure_exec_approvals(tools_profile="full")

        config = json.loads(approvals_path.read_text())
        assert config["defaults"]["security"] == "full"
        assert config["defaults"]["ask"] == "off"
        # Existing fields preserved
        assert config["version"] == 1
        assert config["socket"]["token"] == "abc"

    def test_configure_tools_profile(self, tmp_path):
        config_path = tmp_path / "openclaw" / "openclaw.json"
        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "CONFIG_PATH", config_path):
            ocl.configure(port=8000, api_key="key", model="test", tools_profile="full")

        config = json.loads(config_path.read_text())
        assert config["tools"]["profile"] == "full"

    def test_type(self):
        ocl = OpenClawIntegration()
        assert ocl.type == "config_file"
        assert ocl.display_name == "OpenClaw"


class TestIntegrationSettings:
    def test_settings_dataclass(self):
        from omlx.settings import IntegrationSettings

        settings = IntegrationSettings()
        assert settings.codex_model is None
        assert settings.opencode_model is None
        assert settings.openclaw_model is None
        assert settings.openclaw_tools_profile == "coding"

    def test_to_dict(self):
        from omlx.settings import IntegrationSettings

        settings = IntegrationSettings(codex_model="qwen3.5")
        d = settings.to_dict()
        assert d["codex_model"] == "qwen3.5"
        assert d["opencode_model"] is None
        assert d["openclaw_tools_profile"] == "coding"

    def test_from_dict(self):
        from omlx.settings import IntegrationSettings

        settings = IntegrationSettings.from_dict(
            {"codex_model": "llama", "opencode_model": "qwen"}
        )
        assert settings.codex_model == "llama"
        assert settings.opencode_model == "qwen"
        assert settings.openclaw_model is None

    def test_from_dict_empty(self):
        from omlx.settings import IntegrationSettings

        settings = IntegrationSettings.from_dict({})
        assert settings.codex_model is None
