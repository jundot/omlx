"""Tests for the integrations module."""

import json
import plistlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from omlx.integrations import get_integration, list_integrations
from omlx.integrations.base import IntegrationContext
from omlx.integrations.claude import ClaudeCodeIntegration
from omlx.integrations.codex import (
    CodexIntegration,
    codex_config_args,
    write_codex_config,
)
from omlx.integrations.codex_app import CodexAppIntegration, find_codex_app_bundle
from omlx.integrations.copilot import CopilotIntegration
from omlx.integrations.dsh import (
    DshConfigShapeError,
    DshIntegration,
    find_dsh_app_bundle,
    write_credentials_ref,
    write_dsh_patch,
)
from omlx.integrations.hermes import HermesIntegration
from omlx.integrations.openclaw import OpenClawIntegration
from omlx.integrations.opencode import OpenCodeIntegration
from omlx.integrations.pi import PiIntegration, _get_agent_dir


def ctx(**overrides) -> IntegrationContext:
    defaults = {
        "host": "127.0.0.1",
        "port": 8000,
        "api_key": "",
        "model": "",
    }
    defaults.update(overrides)
    return IntegrationContext(**defaults)


class TestIntegrationRegistry:
    def test_list_integrations(self):
        integrations = list_integrations()
        assert len(integrations) == 9
        names = {i.name for i in integrations}
        assert names == {
            "claude",
            "codex",
            "codex_app",
            "copilot",
            "opencode",
            "openclaw",
            "hermes",
            "pi",
            "dsh",
        }

    def test_get_integration(self):
        assert get_integration("claude") is not None
        assert get_integration("codex") is not None
        assert get_integration("codex_app") is not None
        assert get_integration("copilot") is not None
        assert get_integration("opencode") is not None
        assert get_integration("openclaw") is not None
        assert get_integration("hermes") is not None
        assert get_integration("pi") is not None
        assert get_integration("dsh") is not None
        assert get_integration("nonexistent") is None


class TestIntegrationCommands:
    def test_commands_quote_full_app_cli_prefix(self):
        with patch(
            "omlx.utils.install.get_cli_prefix",
            return_value="/Users/me/My Apps/oMLX.app/Contents/MacOS/omlx-cli",
        ):
            cmd = ClaudeCodeIntegration().get_command(ctx())

        assert (
            cmd == "'/Users/me/My Apps/oMLX.app/Contents/MacOS/omlx-cli' launch claude"
        )


class TestCodexIntegration:
    def test_get_command(self):
        codex = CodexIntegration()
        cmd = codex.get_command(ctx(port=8000, api_key="test-key", model="qwen3.5"))
        assert "omlx launch codex" in cmd
        assert "--model qwen3.5" in cmd

    def test_get_command_no_model(self):
        codex = CodexIntegration()
        cmd = codex.get_command(ctx(port=8000, api_key="", model=""))
        assert "select-a-model" in cmd

    def test_config_args_use_process_scoped_provider(self):
        args = codex_config_args(
            ctx(
                host="192.168.1.100",
                port=9000,
                model="deepseek-v3.1",
                context_window=240_000,
            )
        )

        assert 'model_provider="omlx"' in args
        assert (
            'model_providers.omlx.base_url="http://192.168.1.100:9000/v1"' in args
        )
        assert 'model_providers.omlx.env_key="OMLX_API_KEY"' in args
        assert "model_context_window=240000" in args
        assert not any("model_auto_compact_token_limit" in arg for arg in args)

    def test_config_args_reasoning_uses_resolved_metadata(self):
        reasoning_args = codex_config_args(
            ctx(port=8000, model="custom-model", reasoning=True)
        )
        non_reasoning_args = codex_config_args(
            ctx(port=8000, model="deepseek-r1", reasoning=False)
        )

        assert 'model_reasoning_effort="high"' in reasoning_args
        assert not any("model_reasoning_effort" in arg for arg in non_reasoning_args)

    def test_configure_does_not_write_codex_config(self):
        with patch("omlx.integrations.codex.write_codex_config") as writer:
            CodexIntegration().configure(ctx(port=8000, model="new-model"))

        writer.assert_not_called()

    def test_type(self):
        codex = CodexIntegration()
        assert codex.type == "env_var"
        assert codex.display_name == "Codex"

    def test_launch_forwards_extra_args(self):
        codex = CodexIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv
            captured["env"] = env

        base_env = {
            "PATH": "/usr/bin",
            "PYTHONHOME": "/bundle/python",
            "PYTHONPATH": "/bundle/lib",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with (
            patch("omlx.integrations.codex.write_codex_config") as writer,
            patch("omlx.integrations.codex.os.environ", base_env),
            patch("omlx.integrations.codex.os.execvpe", side_effect=fake_execvpe),
        ):
            codex.launch(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen3.5",
                    extra_args=("--yolo",),
                )
            )

        assert captured["argv"][-3:] == ["-m", "qwen3.5", "--yolo"]
        assert 'model_provider="omlx"' in captured["argv"]
        assert "model_context_window=240000" not in captured["argv"]
        assert captured["env"]["OMLX_API_KEY"] == "key"
        assert "PYTHONHOME" not in captured["env"]
        assert "PYTHONPATH" not in captured["env"]
        assert "PYTHONDONTWRITEBYTECODE" not in captured["env"]
        writer.assert_not_called()


class TestCodexConfigWriter:
    """Keep coverage for the config writer used by Codex App."""

    def test_writes_provider_config(self, tmp_path):
        config_path = tmp_path / "config.toml"

        write_codex_config(
            config_path,
            ctx(
                host="192.168.1.100",
                port=9000,
                api_key="test-key",
                model="qwen3.5",
            ),
        )

        content = config_path.read_text()
        assert 'model = "qwen3.5"' in content
        assert 'model_provider = "omlx"' in content
        assert 'base_url = "http://192.168.1.100:9000/v1"' in content
        assert 'env_key = "OMLX_API_KEY"' in content

    def test_creates_backup(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text('model = "old"')

        write_codex_config(config_path, ctx(port=8000, model="new"))

        backups = list(tmp_path.glob("config.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == 'model = "old"'

    def test_preserves_existing_sections(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'model = "old-model"\n'
            'other_key = "value"\n'
            "\n"
            "[agents]\n"
            "max_concurrent_threads_per_session = 4\n"
            "\n"
            "[model_providers.omlx]\n"
            'name = "old-omlx"\n'
        )

        write_codex_config(config_path, ctx(port=8000, model="new-model"))

        content = config_path.read_text()
        assert 'model = "new-model"' in content
        assert 'other_key = "value"' in content
        assert "[agents]" in content
        assert "max_concurrent_threads_per_session = 4" in content
        assert 'name = "oMLX"' in content
        assert "old-omlx" not in content

    def test_clears_stale_reasoning_effort(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'model = "old-thinking-model"\n'
            'model_provider = "omlx"\n'
            'model_reasoning_effort = "high"\n'
        )

        write_codex_config(
            config_path,
            ctx(port=8000, model="llama-3.1-8b", reasoning=False),
        )

        content = config_path.read_text()
        assert 'model = "llama-3.1-8b"' in content
        assert "model_reasoning_effort" not in content


def make_app_bundle(
    root: Path, name: str, bundle_id: str, with_cli: bool = True
) -> Path:
    """Create a fake macOS app bundle with an Info.plist and optional CLI."""
    contents = root / name / "Contents"
    contents.mkdir(parents=True)
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump({"CFBundleIdentifier": bundle_id}, f)
    if with_cli:
        resources = contents / "Resources"
        resources.mkdir()
        cli = resources / "codex"
        cli.write_text("#!/bin/sh\n")
        cli.chmod(0o755)
    return root / name


class TestCodexAppIntegration:
    def test_get_command(self):
        codex_app = CodexAppIntegration()
        cmd = codex_app.get_command(ctx(port=8000, api_key="key", model="qwen3.5"))
        assert "omlx launch codex_app" in cmd
        assert "--model qwen3.5" in cmd

    def test_configure(self, tmp_path):
        codex_app = CodexAppIntegration()
        config_path = tmp_path / "codex" / "config.toml"
        with patch.object(CodexAppIntegration, "CONFIG_PATH", config_path):
            codex_app.configure(ctx(port=8000, api_key="test-key", model="qwen3.5"))

        assert config_path.exists()
        content = config_path.read_text()
        assert 'model = "qwen3.5"' in content
        assert 'model_provider = "omlx"' in content
        assert 'base_url = "http://127.0.0.1:8000/v1"' in content
        assert 'env_key = "OMLX_API_KEY"' in content

    def test_launch_app(self, tmp_path):
        codex_app = CodexAppIntegration()
        config_path = tmp_path / "codex" / "config.toml"
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv
            captured["env"] = env

        base_env = {
            "PATH": "/usr/bin",
            "PYTHONHOME": "/bundle/python",
            "PYTHONPATH": "/bundle/lib",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with (
            patch.object(CodexAppIntegration, "CONFIG_PATH", config_path),
            patch("omlx.integrations.codex_app.os.environ", base_env),
            patch("omlx.integrations.codex_app.os.execvpe", side_effect=fake_execvpe),
            patch(
                "omlx.integrations.codex_app.resolve_codex_binary",
                return_value="/opt/homebrew/bin/codex",
            ),
        ):
            codex_app.launch(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen3.5",
                    extra_args=(),
                )
            )

        # Codex App should launch with "app" subcommand, not "-m <model>"
        assert captured["argv"] == ["/opt/homebrew/bin/codex", "app"]
        assert captured["env"]["OMLX_API_KEY"] == "key"
        assert "PYTHONHOME" not in captured["env"]
        assert "PYTHONPATH" not in captured["env"]
        assert "PYTHONDONTWRITEBYTECODE" not in captured["env"]

    def test_is_installed_with_app_bundle_only(self, tmp_path):
        # DMG-only install: no codex CLI on PATH, old bundle folder name
        bundle = make_app_bundle(tmp_path, "Codex.app", "com.openai.codex")
        with (
            patch("omlx.integrations.codex_app.shutil.which", return_value=None),
            patch("omlx.integrations.codex_app._APP_BUNDLE_ROOTS", (tmp_path,)),
        ):
            assert find_codex_app_bundle() == bundle
            assert CodexAppIntegration().is_installed()

    def test_is_installed_with_renamed_chatgpt_bundle(self, tmp_path):
        # Post-rename fresh install: ChatGPT.app folder, codex bundle id
        make_app_bundle(tmp_path, "ChatGPT.app", "com.openai.codex")
        with (
            patch("omlx.integrations.codex_app.shutil.which", return_value=None),
            patch("omlx.integrations.codex_app._APP_BUNDLE_ROOTS", (tmp_path,)),
        ):
            assert CodexAppIntegration().is_installed()

    def test_legacy_chatgpt_chat_app_not_matched(self, tmp_path):
        # The old ChatGPT chat app has a different bundle id and no codex CLI
        make_app_bundle(tmp_path, "ChatGPT.app", "com.openai.chat")
        with (
            patch("omlx.integrations.codex_app.shutil.which", return_value=None),
            patch("omlx.integrations.codex_app._APP_BUNDLE_ROOTS", (tmp_path,)),
        ):
            assert find_codex_app_bundle() is None
            assert not CodexAppIntegration().is_installed()

    def test_not_installed_without_cli_or_bundle(self, tmp_path):
        with (
            patch("omlx.integrations.codex_app.shutil.which", return_value=None),
            patch("omlx.integrations.codex_app._APP_BUNDLE_ROOTS", (tmp_path,)),
        ):
            assert not CodexAppIntegration().is_installed()

    def test_launch_falls_back_to_bundled_cli(self, tmp_path):
        bundle = make_app_bundle(tmp_path, "Codex.app", "com.openai.codex")
        config_path = tmp_path / "codex" / "config.toml"
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["binary"] = binary
            captured["argv"] = argv

        with (
            patch.object(CodexAppIntegration, "CONFIG_PATH", config_path),
            patch("omlx.integrations.codex_app.shutil.which", return_value=None),
            patch("omlx.integrations.codex_app._APP_BUNDLE_ROOTS", (tmp_path,)),
            patch("omlx.integrations.codex_app.os.execvpe", side_effect=fake_execvpe),
        ):
            CodexAppIntegration().launch(ctx(port=8000, api_key="key", model="q"))

        bundled = str(bundle / "Contents" / "Resources" / "codex")
        assert captured["binary"] == bundled
        assert captured["argv"] == [bundled, "app"]

    def test_type(self):
        codex_app = CodexAppIntegration()
        assert codex_app.type == "config_file"
        assert codex_app.display_name == "Codex App"
        assert codex_app.name == "codex_app"


class TestOpenCodeIntegration:
    def test_get_command(self):
        oc = OpenCodeIntegration()
        cmd = oc.get_command(ctx(port=8000, api_key="key", model="qwen3.5"))
        assert "omlx launch opencode" in cmd
        assert "--model qwen3.5" in cmd

    def test_configure_new_file(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"

        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(ctx(port=8000, api_key="test-key", model="qwen3.5"))

        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert (
            config["provider"]["omlx"]["options"]["baseURL"]
            == "http://127.0.0.1:8000/v1"
        )
        assert config["provider"]["omlx"]["npm"] == "@ai-sdk/openai-compatible"
        assert config["provider"]["omlx"]["options"]["apiKey"] == "test-key"
        assert config["provider"]["omlx"]["models"]["qwen3.5"]["name"] == "qwen3.5"
        assert config["provider"]["omlx"]["models"]["qwen3.5"]["modalities"] == {
            "input": ["text"],
            "output": ["text"],
        }
        assert config["model"] == "omlx/qwen3.5"

    def test_configure_custom_host(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"
        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(ctx(port=9000, api_key="key", model="test", host="10.0.0.5"))

        config = json.loads(config_path.read_text())
        assert (
            config["provider"]["omlx"]["options"]["baseURL"]
            == "http://10.0.0.5:9000/v1"
        )

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
            oc.configure(ctx(port=9000, api_key="", model="llama"))

        config = json.loads(config_path.read_text())
        # Existing provider preserved
        assert "ollama" in config["provider"]
        assert (
            config["provider"]["ollama"]["options"]["baseURL"]
            == "http://localhost:11434/v1"
        )
        # omlx provider added
        assert "omlx" in config["provider"]
        assert (
            config["provider"]["omlx"]["options"]["baseURL"]
            == "http://127.0.0.1:9000/v1"
        )
        # Other keys preserved
        assert config["logLevel"] == "INFO"

    def test_configure_creates_backup(self, tmp_path):
        config_path = tmp_path / "opencode.json"
        config_path.write_text('{"existing": true}')

        oc = OpenCodeIntegration()
        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(ctx(port=8000, api_key="", model="test"))

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
            oc.configure(ctx(port=8000, api_key="key", model="test"))

        # Should create new config despite invalid existing file
        config = json.loads(config_path.read_text())
        assert "omlx" in config["provider"]

    def test_configure_with_limits(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"

        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen3.5",
                    context_window=32768,
                    max_tokens=8192,
                )
            )

        config = json.loads(config_path.read_text())
        model_config = config["provider"]["omlx"]["models"]["qwen3.5"]
        assert model_config["limit"]["context"] == 32768
        assert model_config["limit"]["output"] == 8192

    def test_configure_vlm_modalities(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"

        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen2.5-vl",
                    model_type="vlm",
                )
            )

        config = json.loads(config_path.read_text())
        model_config = config["provider"]["omlx"]["models"]["qwen2.5-vl"]
        assert model_config["attachment"] is True
        assert model_config["modalities"] == {
            "input": ["text", "image"],
            "output": ["text"],
        }

    def test_configure_with_context_window_only(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"

        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen3.5",
                    context_window=32768,
                )
            )

        config = json.loads(config_path.read_text())
        model_config = config["provider"]["omlx"]["models"]["qwen3.5"]
        assert model_config["limit"]["context"] == 32768
        assert model_config["limit"]["output"] == 32768

    def test_configure_without_limits(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"

        with patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path):
            oc.configure(ctx(port=8000, api_key="key", model="qwen3.5"))

        config = json.loads(config_path.read_text())
        model_config = config["provider"]["omlx"]["models"]["qwen3.5"]
        assert "limit" not in model_config

    def test_launch_scrubs_python_env(self, tmp_path):
        oc = OpenCodeIntegration()
        config_path = tmp_path / "opencode" / "opencode.json"
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv
            captured["env"] = env

        base_env = {
            "PATH": "/usr/bin",
            "PYTHONHOME": "/bundle/python",
            "PYTHONPATH": "/bundle/lib",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with (
            patch.object(OpenCodeIntegration, "CONFIG_PATH", config_path),
            patch("omlx.integrations.opencode.os.environ", base_env),
            patch("omlx.integrations.opencode.os.execvpe", side_effect=fake_execvpe),
        ):
            oc.launch(ctx(port=8000, api_key="key", model="qwen3.5"))

        assert captured["argv"] == ["opencode"]
        assert "PYTHONHOME" not in captured["env"]
        assert "PYTHONPATH" not in captured["env"]
        assert "PYTHONDONTWRITEBYTECODE" not in captured["env"]

    def test_type(self):
        oc = OpenCodeIntegration()
        assert oc.type == "config_file"
        assert oc.display_name == "OpenCode"


class TestOpenClawIntegration:
    def test_get_command(self):
        ocl = OpenClawIntegration()
        cmd = ocl.get_command(ctx(port=8000, api_key="key", model="qwen3.5"))
        assert "omlx launch openclaw" in cmd
        assert "--model qwen3.5" in cmd

    def test_configure_new_file(self, tmp_path):
        config_path = tmp_path / "openclaw" / "openclaw.json"

        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "CONFIG_PATH", config_path):
            ocl.configure(ctx(port=8000, api_key="test-key", model="qwen3.5"))

        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert (
            config["models"]["providers"]["omlx"]["baseUrl"]
            == "http://127.0.0.1:8000/v1"
        )
        assert config["models"]["providers"]["omlx"]["api"] == "openai-completions"
        assert config["models"]["providers"]["omlx"]["apiKey"] == "test-key"
        assert config["agents"]["defaults"]["model"]["primary"] == "omlx/qwen3.5"
        assert config["tools"]["profile"] == "coding"

    def test_configure_model_metadata_from_context(self, tmp_path):
        config_path = tmp_path / "openclaw" / "openclaw.json"
        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "CONFIG_PATH", config_path):
            ocl.configure(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen2.5-vl",
                    model_type="vlm",
                    reasoning=True,
                    context_window=32768,
                    max_tokens=8192,
                )
            )

        model_config = json.loads(config_path.read_text())["models"]["providers"][
            "omlx"
        ]["models"][0]
        assert model_config["reasoning"] is True
        assert model_config["input"] == ["text", "image"]
        assert model_config["contextWindow"] == 32768
        assert model_config["maxTokens"] == 8192

    def test_configure_omits_unknown_limits(self, tmp_path):
        config_path = tmp_path / "openclaw" / "openclaw.json"
        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "CONFIG_PATH", config_path):
            ocl.configure(ctx(port=8000, api_key="key", model="llama"))

        model_config = json.loads(config_path.read_text())["models"]["providers"][
            "omlx"
        ]["models"][0]
        assert model_config["reasoning"] is False
        assert model_config["input"] == ["text"]
        assert "contextWindow" not in model_config
        assert "maxTokens" not in model_config

    def test_configure_custom_host(self, tmp_path):
        config_path = tmp_path / "openclaw" / "openclaw.json"
        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "CONFIG_PATH", config_path):
            ocl.configure(
                ctx(port=9000, api_key="key", model="test", host="192.168.1.100")
            )

        config = json.loads(config_path.read_text())
        assert (
            config["models"]["providers"]["omlx"]["baseUrl"]
            == "http://192.168.1.100:9000/v1"
        )

    def test_configure_preserves_existing(self, tmp_path):
        config_path = tmp_path / "openclaw.json"
        existing = {
            "models": {"providers": {"ollama": {"baseUrl": "http://localhost:11434"}}},
            "channels": {"telegram": {"enabled": True}},
        }
        config_path.write_text(json.dumps(existing))

        ocl = OpenClawIntegration()
        with patch.object(OpenClawIntegration, "CONFIG_PATH", config_path):
            ocl.configure(ctx(port=9000, api_key="key", model="llama"))

        config = json.loads(config_path.read_text())
        # Existing preserved
        assert "ollama" in config["models"]["providers"]
        assert config["channels"]["telegram"]["enabled"] is True
        # omlx added
        assert "omlx" in config["models"]["providers"]
        assert (
            config["models"]["providers"]["omlx"]["baseUrl"]
            == "http://127.0.0.1:9000/v1"
        )

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
            ocl.configure(
                ctx(port=8000, api_key="key", model="test", tools_profile="full")
            )

        config = json.loads(config_path.read_text())
        assert config["tools"]["profile"] == "full"

    def test_launch_scrubs_python_env(self, tmp_path):
        ocl = OpenClawIntegration()
        config_path = tmp_path / "openclaw.json"
        approvals_path = tmp_path / "exec-approvals.json"
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["exec_env"] = env

        def fake_run(args, **kwargs):
            captured["run_env"] = kwargs.get("env")
            return None

        base_env = {
            "PATH": "/usr/bin",
            "PYTHONHOME": "/bundle/python",
            "PYTHONPATH": "/bundle/lib",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with (
            patch.object(OpenClawIntegration, "CONFIG_PATH", config_path),
            patch.object(OpenClawIntegration, "EXEC_APPROVALS_PATH", approvals_path),
            patch.object(OpenClawIntegration, "_is_onboarded", return_value=True),
            patch.object(
                OpenClawIntegration, "_gateway_info", return_value=("localhost", 9999)
            ),
            # Gateway already running: hits the daemon-restart subprocess.run
            # branch and skips gateway start, so execvpe is reached.
            patch.object(OpenClawIntegration, "_port_open", return_value=True),
            patch.object(OpenClawIntegration, "_wait_for_port", return_value=True),
            patch("omlx.integrations.openclaw.os.environ", base_env),
            patch("omlx.integrations.openclaw.subprocess.run", side_effect=fake_run),
            patch("omlx.integrations.openclaw.os.execvpe", side_effect=fake_execvpe),
        ):
            ocl.launch(ctx(port=8000, api_key="key", model="qwen3.5"))

        # Both the daemon-restart subprocess and the TUI exec get scrubbed env.
        for env in (captured["run_env"], captured["exec_env"]):
            assert "PYTHONHOME" not in env
            assert "PYTHONPATH" not in env
            assert "PYTHONDONTWRITEBYTECODE" not in env

    def test_type(self):
        ocl = OpenClawIntegration()
        assert ocl.type == "config_file"
        assert ocl.display_name == "OpenClaw"


class TestHermesIntegration:
    def test_get_command(self):
        hermes = HermesIntegration()
        cmd = hermes.get_command(ctx(port=8000, api_key="key", model="qwen3.5"))
        assert "omlx launch hermes" in cmd
        assert "--model qwen3.5" in cmd

    def test_get_command_no_model(self):
        hermes = HermesIntegration()
        cmd = hermes.get_command(ctx(port=8000, api_key="", model=""))
        assert "select-a-model" in cmd

    def test_configure_new_file(self, tmp_path):
        config_path = tmp_path / "hermes" / "config.yaml"

        hermes = HermesIntegration()
        with patch.object(HermesIntegration, "CONFIG_PATH", config_path):
            hermes.configure(
                ctx(
                    port=8000,
                    api_key="test-key",
                    model="qwen3.5",
                    context_window=131072,
                    max_tokens=8192,
                )
            )

        assert config_path.exists()
        config = yaml.safe_load(config_path.read_text())
        provider = config["providers"]["omlx"]
        assert provider["name"] == "oMLX"
        assert provider["base_url"] == "http://127.0.0.1:8000/v1"
        assert provider["api_key"] == "test-key"
        assert provider["api_mode"] == "chat_completions"
        assert provider["default_model"] == "qwen3.5"
        assert config["model"]["provider"] == "omlx"
        assert config["model"]["default"] == "qwen3.5"
        assert config["model"]["context_length"] == 131072
        assert config["model"]["max_tokens"] == 8192

    def test_configure_custom_host(self, tmp_path):
        config_path = tmp_path / "config.yaml"

        hermes = HermesIntegration()
        with patch.object(HermesIntegration, "CONFIG_PATH", config_path):
            hermes.configure(ctx(port=9000, api_key="", model="llama", host="10.0.0.5"))

        provider = yaml.safe_load(config_path.read_text())["providers"]["omlx"]
        assert provider["base_url"] == "http://10.0.0.5:9000/v1"
        assert provider["api_key"] == "omlx"

    def test_configure_preserves_existing(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "theme": "dark",
                    "providers": {
                        "anthropic": {"base_url": "https://api.anthropic.com"},
                        "omlx": {"timeout": 120},
                    },
                    "model": {
                        "temperature": 0.2,
                        "base_url": "https://inference-api.nousresearch.com/v1",
                        "api_key": "old-key",
                    },
                },
                sort_keys=False,
            )
        )

        hermes = HermesIntegration()
        with patch.object(HermesIntegration, "CONFIG_PATH", config_path):
            hermes.configure(ctx(port=8000, api_key="key", model="qwen3.5"))

        config = yaml.safe_load(config_path.read_text())
        assert config["theme"] == "dark"
        assert (
            config["providers"]["anthropic"]["base_url"] == "https://api.anthropic.com"
        )
        assert config["providers"]["omlx"]["timeout"] == 120
        assert config["providers"]["omlx"]["base_url"] == "http://127.0.0.1:8000/v1"
        assert config["model"]["temperature"] == 0.2
        assert config["model"]["provider"] == "omlx"
        assert config["model"]["default"] == "qwen3.5"
        assert "base_url" not in config["model"]
        assert "api_key" not in config["model"]

    def test_configure_creates_backup(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("existing: true\n")

        hermes = HermesIntegration()
        with patch.object(HermesIntegration, "CONFIG_PATH", config_path):
            hermes.configure(ctx(port=8000, api_key="", model="test"))

        backups = list(tmp_path.glob("config.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == "existing: true\n"

    def test_configure_clears_stale_limits_when_unknown(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "provider": "omlx",
                        "default": "old",
                        "context_length": 32768,
                        "max_tokens": 8192,
                    }
                }
            )
        )

        hermes = HermesIntegration()
        with patch.object(HermesIntegration, "CONFIG_PATH", config_path):
            hermes.configure(ctx(port=8000, api_key="key", model="new"))

        model_config = yaml.safe_load(config_path.read_text())["model"]
        assert model_config["default"] == "new"
        assert "context_length" not in model_config
        assert "max_tokens" not in model_config

    def test_configure_preserves_actual_context_length(self, tmp_path):
        config_path = tmp_path / "config.yaml"

        hermes = HermesIntegration()
        with patch.object(HermesIntegration, "CONFIG_PATH", config_path):
            hermes.configure(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen3.5",
                    context_window=32768,
                )
            )

        model_config = yaml.safe_load(config_path.read_text())["model"]
        assert model_config["context_length"] == 32768

    def test_model_disabled_reason_below_64k(self):
        hermes = HermesIntegration()

        reason = hermes.model_disabled_reason(
            {"id": "small-model", "max_context_window": 32768}
        )

        assert reason is not None
        assert "at least 64K" in reason
        assert "32,768" in reason

    def test_model_disabled_reason_allows_64k(self):
        hermes = HermesIntegration()

        assert (
            hermes.model_disabled_reason(
                {"id": "supported-model", "max_context_window": 64000}
            )
            is None
        )

    def test_select_model_rejects_disabled_choice(self, capsys):
        hermes = HermesIntegration()

        with (
            patch("omlx.integrations.base.sys.stdout.isatty", return_value=False),
            patch("builtins.input", side_effect=["1", "2"]),
        ):
            selected = hermes.select_model(
                [
                    {"id": "small-model", "max_context_window": 32768},
                    {"id": "supported-model", "max_context_window": 64000},
                ]
            )

        assert selected == "supported-model"
        output = capsys.readouterr().out
        assert "small-model" in output
        assert "unavailable" in output
        assert "Cannot select small-model" in output

    def test_launch_rejects_context_below_64k_and_records_actual_value(
        self, tmp_path, capsys
    ):
        config_path = tmp_path / "config.yaml"
        hermes = HermesIntegration()

        with (
            patch.object(HermesIntegration, "CONFIG_PATH", config_path),
            patch("omlx.integrations.hermes.os.execvpe") as execvpe,
            pytest.raises(SystemExit) as exc,
        ):
            hermes.launch(
                ctx(
                    port=8000,
                    api_key="secret",
                    model="small-model",
                    context_window=32768,
                    max_tokens=8192,
                )
            )

        assert exc.value.code == 1
        execvpe.assert_not_called()
        output = capsys.readouterr().out
        assert "Cannot launch Hermes Agent" in output
        assert "at least 64K" in output

        config = yaml.safe_load(config_path.read_text())
        assert config["model"]["context_length"] == 32768

    def test_launch_sets_config_and_execs(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        hermes = HermesIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["binary"] = binary
            captured["argv"] = argv
            captured["env"] = env

        base_env = {
            "PATH": "/usr/bin",
            "PYTHONHOME": "/bundle/python",
            "PYTHONPATH": "/bundle/lib",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with (
            patch.object(HermesIntegration, "CONFIG_PATH", config_path),
            patch("omlx.integrations.hermes.os.environ", base_env),
            patch("omlx.integrations.hermes.os.execvpe", side_effect=fake_execvpe),
        ):
            hermes.launch(
                ctx(
                    port=8000,
                    api_key="secret",
                    model="qwen3.5",
                    context_window=131072,
                    max_tokens=8192,
                )
            )

        assert captured["binary"] == "hermes"
        assert captured["argv"] == [
            "hermes",
            "chat",
            "--tui",
            "-m",
            "qwen3.5",
        ]
        assert "PYTHONHOME" not in captured["env"]
        assert "PYTHONPATH" not in captured["env"]
        assert "PYTHONDONTWRITEBYTECODE" not in captured["env"]

        config = yaml.safe_load(config_path.read_text())
        assert config["providers"]["omlx"]["api_key"] == "secret"
        assert config["model"]["context_length"] == 131072
        assert config["model"]["max_tokens"] == 8192

    def test_launch_without_model(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        hermes = HermesIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv

        with (
            patch.object(HermesIntegration, "CONFIG_PATH", config_path),
            patch("omlx.integrations.hermes.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.hermes.os.execvpe", side_effect=fake_execvpe),
        ):
            hermes.launch(ctx(port=8000, api_key="", model=""))

        assert captured["argv"] == ["hermes", "chat", "--tui"]

    def test_launch_forwards_extra_args(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        hermes = HermesIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv

        with (
            patch.object(HermesIntegration, "CONFIG_PATH", config_path),
            patch("omlx.integrations.hermes.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.hermes.os.execvpe", side_effect=fake_execvpe),
        ):
            hermes.launch(
                ctx(port=8000, api_key="", model="qwen3.5", extra_args=("--continue",))
            )

        assert captured["argv"] == [
            "hermes",
            "chat",
            "--tui",
            "-m",
            "qwen3.5",
            "--continue",
        ]

    def test_type(self):
        hermes = HermesIntegration()
        assert hermes.type == "config_file"
        assert hermes.display_name == "Hermes Agent"
        assert hermes.install_check == "hermes"


class TestPiIntegration:
    def test_get_agent_dir_default(self, tmp_path, monkeypatch):
        """Default agent dir is ~/.pi/agent when env var is not set."""
        monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
        result = _get_agent_dir()
        assert result == Path.home() / ".pi" / "agent"

    def test_get_agent_dir_custom_env(self, tmp_path, monkeypatch):
        """PI_CODING_AGENT_DIR env var overrides the default path."""
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "custom_pi"))
        result = _get_agent_dir()
        assert result == tmp_path / "custom_pi"

    def test_get_agent_dir_expands_user(self, tmp_path, monkeypatch):
        """PI_CODING_AGENT_DIR ~ is expanded to the home directory."""
        monkeypatch.setenv("PI_CODING_AGENT_DIR", "~/my-agent")
        result = _get_agent_dir()
        assert result == Path.home() / "my-agent"

    def test_get_command(self):
        pi = PiIntegration()
        cmd = pi.get_command(ctx(port=8000, api_key="key", model="qwen3.5"))
        assert "omlx launch pi" in cmd
        assert "--model qwen3.5" in cmd

    def test_get_command_no_model(self):
        pi = PiIntegration()
        cmd = pi.get_command(ctx(port=8000, api_key="", model=""))
        assert "select-a-model" in cmd

    def test_configure_new_files(self, tmp_path):
        models_path = tmp_path / "pi" / "agent" / "models.json"
        settings_path = tmp_path / "pi" / "agent" / "settings.json"

        pi = PiIntegration()
        with (
            patch.object(PiIntegration, "MODELS_PATH", models_path),
            patch.object(PiIntegration, "SETTINGS_PATH", settings_path),
        ):
            pi.configure(ctx(port=8000, api_key="test-key", model="qwen3.5"))

        models_config = json.loads(models_path.read_text())
        provider = models_config["providers"]["omlx"]
        assert provider["baseUrl"] == "http://127.0.0.1:8000/v1"
        assert provider["api"] == "openai-completions"
        assert provider["apiKey"] == "test-key"
        assert provider["authHeader"] is True
        assert provider["models"][0]["id"] == "qwen3.5"
        assert provider["models"][0]["input"] == ["text"]

        settings_config = json.loads(settings_path.read_text())
        assert settings_config["defaultProvider"] == "omlx"
        assert settings_config["defaultModel"] == "qwen3.5"

    def test_configure_custom_host(self, tmp_path):
        models_path = tmp_path / "models.json"
        settings_path = tmp_path / "settings.json"

        pi = PiIntegration()
        with (
            patch.object(PiIntegration, "MODELS_PATH", models_path),
            patch.object(PiIntegration, "SETTINGS_PATH", settings_path),
        ):
            pi.configure(
                ctx(port=9000, api_key="key", model="test", host="192.168.1.100")
            )

        provider = json.loads(models_path.read_text())["providers"]["omlx"]
        assert provider["baseUrl"] == "http://192.168.1.100:9000/v1"

    def test_configure_creates_backup(self, tmp_path):
        models_path = tmp_path / "models.json"
        settings_path = tmp_path / "settings.json"
        models_path.write_text('{"providers": {"old": {}}}')
        settings_path.write_text('{"defaultProvider": "old"}')

        pi = PiIntegration()
        with (
            patch.object(PiIntegration, "MODELS_PATH", models_path),
            patch.object(PiIntegration, "SETTINGS_PATH", settings_path),
        ):
            pi.configure(ctx(port=8000, api_key="", model="test"))

        model_backups = list(tmp_path.glob("models.*.bak"))
        settings_backups = list(tmp_path.glob("settings.*.bak"))
        assert len(model_backups) == 1
        assert len(settings_backups) == 1
        assert json.loads(model_backups[0].read_text()) == {"providers": {"old": {}}}
        assert json.loads(settings_backups[0].read_text()) == {"defaultProvider": "old"}

    def test_configure_vlm_model(self, tmp_path):
        models_path = tmp_path / "models.json"
        settings_path = tmp_path / "settings.json"

        pi = PiIntegration()
        with (
            patch.object(PiIntegration, "MODELS_PATH", models_path),
            patch.object(PiIntegration, "SETTINGS_PATH", settings_path),
        ):
            pi.configure(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen2.5-vl",
                    model_type="vlm",
                    context_window=32768,
                    max_tokens=8192,
                )
            )

        provider = json.loads(models_path.read_text())["providers"]["omlx"]
        model_config = provider["models"][0]
        assert model_config["input"] == ["text", "image"]
        assert model_config["contextWindow"] == 32768
        assert model_config["maxTokens"] == 8192

    def test_configure_reasoning_true_overrides_slug(self, tmp_path):
        models_path = tmp_path / "models.json"
        settings_path = tmp_path / "settings.json"

        pi = PiIntegration()
        with (
            patch.object(PiIntegration, "MODELS_PATH", models_path),
            patch.object(PiIntegration, "SETTINGS_PATH", settings_path),
        ):
            pi.configure(ctx(port=8000, model="qwen3.6", reasoning=True))

        model_config = json.loads(models_path.read_text())["providers"]["omlx"][
            "models"
        ][0]
        assert model_config["reasoning"] is True

    def test_configure_reasoning_false_overrides_slug(self, tmp_path):
        models_path = tmp_path / "models.json"
        settings_path = tmp_path / "settings.json"

        pi = PiIntegration()
        with (
            patch.object(PiIntegration, "MODELS_PATH", models_path),
            patch.object(PiIntegration, "SETTINGS_PATH", settings_path),
        ):
            pi.configure(ctx(port=8000, model="some-thinking-model", reasoning=False))

        model_config = json.loads(models_path.read_text())["providers"]["omlx"][
            "models"
        ][0]
        assert model_config["reasoning"] is False

    def test_configure_reasoning_falls_back_to_slug(self, tmp_path):
        models_path = tmp_path / "models.json"
        settings_path = tmp_path / "settings.json"

        pi = PiIntegration()
        with (
            patch.object(PiIntegration, "MODELS_PATH", models_path),
            patch.object(PiIntegration, "SETTINGS_PATH", settings_path),
        ):
            pi.configure(ctx(port=8000, model="qwen3-thinking"))

        model_config = json.loads(models_path.read_text())["providers"]["omlx"][
            "models"
        ][0]
        assert model_config["reasoning"] is True

    def test_configure_preserves_existing(self, tmp_path):
        models_path = tmp_path / "models.json"
        settings_path = tmp_path / "settings.json"
        models_path.write_text(
            json.dumps(
                {"providers": {"anthropic": {"baseUrl": "https://api.anthropic.com"}}}
            )
        )
        settings_path.write_text(json.dumps({"theme": "dark"}))

        pi = PiIntegration()
        with (
            patch.object(PiIntegration, "MODELS_PATH", models_path),
            patch.object(PiIntegration, "SETTINGS_PATH", settings_path),
        ):
            pi.configure(ctx(port=9000, api_key="", model="llama"))

        models_config = json.loads(models_path.read_text())
        assert "anthropic" in models_config["providers"]
        assert models_config["providers"]["omlx"]["apiKey"] == "omlx"

        settings_config = json.loads(settings_path.read_text())
        assert settings_config["theme"] == "dark"
        assert settings_config["defaultProvider"] == "omlx"
        assert settings_config["defaultModel"] == "llama"

    def test_launch_scrubs_python_env(self, tmp_path):
        pi = PiIntegration()
        models_path = tmp_path / "models.json"
        settings_path = tmp_path / "settings.json"
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv
            captured["env"] = env

        base_env = {
            "PATH": "/usr/bin",
            "PYTHONHOME": "/bundle/python",
            "PYTHONPATH": "/bundle/lib",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with (
            patch.object(PiIntegration, "MODELS_PATH", models_path),
            patch.object(PiIntegration, "SETTINGS_PATH", settings_path),
            patch("omlx.integrations.pi.os.environ", base_env),
            patch("omlx.integrations.pi.os.execvpe", side_effect=fake_execvpe),
        ):
            pi.launch(ctx(port=8000, api_key="key", model="qwen3.5"))

        assert captured["argv"] == ["pi", "--model", "omlx/qwen3.5"]
        assert "PYTHONHOME" not in captured["env"]
        assert "PYTHONPATH" not in captured["env"]
        assert "PYTHONDONTWRITEBYTECODE" not in captured["env"]

    def test_type(self):
        pi = PiIntegration()
        assert pi.type == "config_file"
        assert pi.display_name == "Pi"


class TestClaudeCodeIntegration:
    def test_get_command(self):
        cc = ClaudeCodeIntegration()
        cmd = cc.get_command(ctx(port=8000, api_key="key", model="qwen3.5"))
        assert "omlx launch claude" in cmd

    def test_get_command_ignores_model(self):
        # Claude integration uses TUI selection so the rendered command
        # is the same regardless of model arg.
        cc = ClaudeCodeIntegration()
        assert cc.get_command(ctx(port=8000, api_key="", model="")) == cc.get_command(
            ctx(port=8000, api_key="key", model="qwen3.5")
        )

    def test_type(self):
        cc = ClaudeCodeIntegration()
        assert cc.type == "env_var"
        assert cc.display_name == "Claude Code"
        assert cc.install_check == "claude"

    def test_find_claude_binary_in_path(self):
        cc = ClaudeCodeIntegration()
        with patch(
            "omlx.integrations.claude.shutil.which", return_value="/usr/bin/claude"
        ):
            assert cc._find_claude_binary() == "claude"

    def test_find_claude_binary_local_fallback(self, tmp_path):
        cc = ClaudeCodeIntegration()
        local_claude = tmp_path / ".claude" / "local" / "claude"
        local_claude.parent.mkdir(parents=True)
        local_claude.write_text("#!/bin/sh\n")
        with (
            patch("omlx.integrations.claude.shutil.which", return_value=None),
            patch("omlx.integrations.claude.Path.home", return_value=tmp_path),
        ):
            assert cc._find_claude_binary() == str(local_claude)

    def test_find_claude_binary_not_found(self, tmp_path):
        cc = ClaudeCodeIntegration()
        with (
            patch("omlx.integrations.claude.shutil.which", return_value=None),
            patch("omlx.integrations.claude.Path.home", return_value=tmp_path),
        ):
            # Falls back to the bare name so the os.execvpe error surfaces clearly.
            assert cc._find_claude_binary() == "claude"

    def test_launch_sets_anthropic_env(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["binary"] = binary
            captured["argv"] = argv
            captured["env"] = env

        base_env = {
            "PATH": "/usr/bin",
            "PYTHONHOME": "/bundle/python",
            "PYTHONPATH": "/bundle/lib",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with (
            patch("omlx.integrations.claude.os.environ", base_env),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(
                    port=8000,
                    api_key="secret",
                    model="qwen3.5",
                    context_window=131072,
                )
            )

        env = captured["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "secret"
        assert env["ANTHROPIC_API_KEY"] == ""
        assert env["ANTHROPIC_MODEL"] == "qwen3.5"
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "qwen3.5"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "qwen3.5"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "qwen3.5"
        assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "qwen3.5"
        assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "131072"
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
        # Bundled-python vars must be stripped so claude code subprocess hooks
        # don't inherit our cpython-3.11 stack.
        assert "PYTHONHOME" not in env
        assert "PYTHONPATH" not in env
        assert "PYTHONDONTWRITEBYTECODE" not in env

    def test_model_disabled_reason_below_48k(self):
        cc = ClaudeCodeIntegration()

        reason = cc.model_disabled_reason(
            {"id": "small-model", "max_context_window": 32768}
        )

        assert reason is not None
        assert "at least 48K" in reason
        assert "32,768" in reason

    def test_model_disabled_reason_allows_48k(self):
        cc = ClaudeCodeIntegration()

        assert (
            cc.model_disabled_reason(
                {"id": "supported-model", "max_context_window": 49152}
            )
            is None
        )

    def test_select_model_rejects_disabled_choice(self, capsys):
        cc = ClaudeCodeIntegration()

        with (
            patch("omlx.integrations.base.sys.stdout.isatty", return_value=False),
            patch("builtins.input", side_effect=["1", "2"]),
        ):
            selected = cc.select_model(
                [
                    {"id": "small-model", "max_context_window": 32768},
                    {"id": "supported-model", "max_context_window": 49152},
                ]
            )

        assert selected == "supported-model"
        output = capsys.readouterr().out
        assert "small-model" in output
        assert "unavailable" in output
        assert "Cannot select small-model" in output

    def test_launch_rejects_max_context_tokens_below_48k(self, capsys):
        cc = ClaudeCodeIntegration()

        with pytest.raises(SystemExit) as exc:
            cc.launch(
                ctx(
                    port=8000,
                    api_key="secret",
                    model="qwen3.5-32k",
                    context_window=32768,
                )
            )

        assert exc.value.code == 1
        output = capsys.readouterr().out
        assert "Cannot launch Claude Code" in output
        assert "at least 48K" in output

    def test_launch_sets_max_context_tokens_48k(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(
                    port=8000,
                    api_key="secret",
                    model="qwen3.5",
                    sonnet_model="mlx-community/Qwen3-30B-A3B-4bit",
                    context_window=49152,
                )
            )

        env = captured["env"]
        assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "49152"
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "49152"

    def test_launch_sets_max_context_tokens_64k(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(
                    port=8000,
                    api_key="secret",
                    model="qwen3.5",
                    sonnet_model="mlx-community/Qwen3-30B-A3B-4bit",
                    context_window=65536,
                )
            )

        env = captured["env"]
        assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "65536"
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"

    def test_launch_sets_max_context_tokens_for_canonical_claude_alias(self):
        """oMLX always sets CLAUDE_CODE_MAX_CONTEXT_TOKENS the same way
        regardless of the configured model name — whether Claude Code's own
        CLI then honors it for a "claude-*"-canonicalized model name is that
        binary's internal behavior (confirmed live: it does not, for names
        that canonicalize to "claude-*"), not something this integration can
        special-case. This only guards that oMLX's side of the contract is
        unconditional."""
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(
                    port=8000,
                    api_key="secret",
                    model="claude-3-5-sonnet-20241022",
                    sonnet_model="claude-3-5-sonnet-20241022",
                    context_window=131072,
                )
            )

        env = captured["env"]
        assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "131072"
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"

    def test_launch_omits_max_context_tokens_when_context_window_unset(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(
                    port=8000,
                    api_key="secret",
                    model="qwen3.5",
                )
            )

        env = captured["env"]
        assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW" not in env
        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env

    def test_launch_disables_nonessential_traffic_by_default(self):
        # Cross-session messaging is opt-in (--cross-session) because
        # CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC is the privacy-oriented
        # default for a launch aimed at a local model.
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.claude.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(ctx(port=8000, api_key="key", model="qwen3.5"))

        env = captured["env"]
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        for var in (
            "DISABLE_AUTOUPDATER",
            "DISABLE_ERROR_REPORTING",
            "DISABLE_FEEDBACK_COMMAND",
            "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY",
        ):
            assert var not in env

    def test_launch_cross_session_enables_messaging_traffic(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.claude.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(port=8000, api_key="key", model="qwen3.5", cross_session=True)
            )

        env = captured["env"]
        assert "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC" not in env
        assert env["DISABLE_AUTOUPDATER"] == "1"
        assert env["DISABLE_ERROR_REPORTING"] == "1"
        assert env["DISABLE_FEEDBACK_COMMAND"] == "1"
        assert env["CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY"] == "1"

    @pytest.mark.parametrize(
        ("name", "value"),
        (
            ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
            ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "0"),
            ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "false"),
            ("DISABLE_TELEMETRY", "1"),
            ("DISABLE_TELEMETRY", "0"),
            ("DISABLE_TELEMETRY", "false"),
            ("DO_NOT_TRACK", "1"),
            ("DISABLE_GROWTHBOOK", "true"),
        ),
    )
    def test_launch_cross_session_warns_when_user_env_blocks_messaging(
        self, capsys, name, value
    ):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch(
                "omlx.integrations.claude.os.environ",
                {"PATH": "/usr/bin", name: value},
            ),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(port=8000, api_key="key", model="qwen3.5", cross_session=True)
            )

        # The user's own opt-out is preserved, not silently overridden.
        assert captured["env"][name] == value
        output = capsys.readouterr().out
        assert name in output
        assert "cross-session messaging will remain unavailable" in output

    @pytest.mark.parametrize(
        ("name", "value"),
        (
            ("DO_NOT_TRACK", "0"),
            ("DO_NOT_TRACK", "false"),
            ("DISABLE_GROWTHBOOK", "0"),
            ("DISABLE_GROWTHBOOK", "false"),
        ),
    )
    def test_launch_cross_session_accepts_false_boolean_opt_outs(
        self, capsys, name, value
    ):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch(
                "omlx.integrations.claude.os.environ",
                {"PATH": "/usr/bin", name: value},
            ),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(port=8000, api_key="key", model="qwen3.5", cross_session=True)
            )

        assert captured["env"][name] == value
        assert (
            "cross-session messaging will remain unavailable"
            not in capsys.readouterr().out
        )

    def test_launch_sets_distinct_claude_tier_models(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.claude.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(
                    port=8000,
                    api_key="key",
                    model="fallback",
                    opus_model="opus-local",
                    sonnet_model="sonnet-local",
                    haiku_model="haiku-local",
                )
            )

        env = captured["env"]
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "opus-local"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "sonnet-local"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "haiku-local"
        assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "haiku-local"

    def test_launch_open_server_uses_omlx_token(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.claude.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(ctx(port=8000, api_key="", model="qwen3.5"))

        # Empty api_key means an open server, claude code still needs
        # *some* token so we ship a placeholder.
        assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "omlx"

    def test_launch_without_model(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.claude.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(ctx(port=8000, api_key="key", model=""))

        env = captured["env"]
        assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env
        assert "CLAUDE_CODE_SUBAGENT_MODEL" not in env

    def test_launch_forwards_extra_args(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv

        with (
            patch("omlx.integrations.claude.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen3.5",
                    extra_args=("--resume", "abc123"),
                )
            )

        assert captured["argv"] == [
            "claude",
            "--disallowedTools",
            "LSP",
            "--settings",
            '{"useAutoModeDuringPlan":false}',
            "--resume",
            "abc123",
        ]

    def test_launch_forwards_short_resume(self):
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv

        with (
            patch("omlx.integrations.claude.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen3.5",
                    extra_args=("-r", "xyz"),
                )
            )

        assert captured["argv"] == [
            "claude",
            "--disallowedTools",
            "LSP",
            "--settings",
            '{"useAutoModeDuringPlan":false}',
            "-r",
            "xyz",
        ]

    def test_launch_denies_lsp_by_default(self):
        """LSP's schema joins the tools array mid-session and re-prefills the
        whole conversation on a caching server (#2349); the launcher denies it
        so the tools array stays stable."""
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv

        with (
            patch("omlx.integrations.claude.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(ctx(port=8000, api_key="key", model="qwen3.5"))

        assert captured["argv"] == [
            "claude",
            "--disallowedTools",
            "LSP",
            "--settings",
            '{"useAutoModeDuringPlan":false}',
        ]

    def test_launch_respects_user_disallowed_tools(self):
        """A caller-supplied --disallowedTools takes over: don't inject ours
        on top (would duplicate the flag / fight their choice)."""
        cc = ClaudeCodeIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["argv"] = argv

        with (
            patch("omlx.integrations.claude.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.claude.os.execvpe", side_effect=fake_execvpe),
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            cc.launch(
                ctx(
                    port=8000,
                    api_key="key",
                    model="qwen3.5",
                    extra_args=("--disallowedTools", "Bash"),
                )
            )

        assert captured["argv"] == [
            "claude",
            "--settings",
            '{"useAutoModeDuringPlan":false}',
            "--disallowedTools",
            "Bash",
        ]

    @pytest.mark.parametrize(
        "settings_args",
        [
            ("--settings", '{"useAutoModeDuringPlan":true}'),
            ("--settings", "/tmp/custom-claude-settings.json"),
            ('--settings={"useAutoModeDuringPlan":true}',),
        ],
    )
    def test_launch_preserves_explicit_settings(self, settings_args):
        with (
            patch("omlx.integrations.claude.os.execvpe") as execute,
            patch.object(
                ClaudeCodeIntegration, "_find_claude_binary", return_value="claude"
            ),
        ):
            ClaudeCodeIntegration().launch(
                ctx(port=8000, api_key="key", model="qwen3.5", extra_args=settings_args)
            )

        assert execute.call_args.args[1] == [
            "claude",
            "--disallowedTools",
            "LSP",
            *settings_args,
        ]


class TestCopilotIntegration:
    def test_get_command(self):
        copilot = CopilotIntegration()
        cmd = copilot.get_command(ctx(port=8000, api_key="key", model="qwen3.5"))
        assert "omlx launch copilot" in cmd
        assert "--model qwen3.5" in cmd

    def test_get_command_no_model(self):
        copilot = CopilotIntegration()
        cmd = copilot.get_command(ctx(port=8000, api_key="", model=""))
        assert "select-a-model" in cmd

    def test_type(self):
        copilot = CopilotIntegration()
        assert copilot.type == "env_var"
        assert copilot.display_name == "Copilot CLI"
        assert copilot.install_check == "copilot"

    def test_launch_sets_provider_env(self):
        copilot = CopilotIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["binary"] = binary
            captured["argv"] = argv
            captured["env"] = env

        base_env = {
            "PATH": "/usr/bin",
            "PYTHONHOME": "/bundle/python",
            "PYTHONPATH": "/bundle/lib",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        with (
            patch("omlx.integrations.copilot.os.environ", base_env),
            patch("omlx.integrations.copilot.os.execvpe", side_effect=fake_execvpe),
        ):
            copilot.launch(
                ctx(
                    port=8000,
                    api_key="secret",
                    model="qwen3.5",
                    context_window=131072,
                    max_tokens=8192,
                )
            )

        env = captured["env"]
        assert captured["binary"] == "copilot"
        assert captured["argv"] == ["copilot"]
        assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8000/v1"
        assert env["COPILOT_PROVIDER_TYPE"] == "openai"
        assert env["COPILOT_PROVIDER_WIRE_API"] == "responses"
        assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "secret"
        assert env["COPILOT_MODEL"] == "qwen3.5"
        assert env["COPILOT_PROVIDER_MODEL_ID"] == "qwen3.5"
        assert env["COPILOT_PROVIDER_WIRE_MODEL"] == "qwen3.5"
        assert env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] == "131072"
        assert env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] == "8192"
        assert "PYTHONHOME" not in env
        assert "PYTHONPATH" not in env
        assert "PYTHONDONTWRITEBYTECODE" not in env

    def test_launch_open_server_uses_omlx_token(self):
        copilot = CopilotIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.copilot.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.copilot.os.execvpe", side_effect=fake_execvpe),
        ):
            copilot.launch(ctx(port=8000, api_key="", model="qwen3.5"))

        assert captured["env"]["COPILOT_PROVIDER_BEARER_TOKEN"] == "omlx"

    def test_launch_without_model_or_limits(self):
        copilot = CopilotIntegration()
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env

        with (
            patch("omlx.integrations.copilot.os.environ", {"PATH": "/usr/bin"}),
            patch("omlx.integrations.copilot.os.execvpe", side_effect=fake_execvpe),
        ):
            copilot.launch(ctx(port=8000, api_key="key", model=""))

        env = captured["env"]
        assert "COPILOT_MODEL" not in env
        assert "COPILOT_PROVIDER_MODEL_ID" not in env
        assert "COPILOT_PROVIDER_WIRE_MODEL" not in env
        assert "COPILOT_PROVIDER_MAX_PROMPT_TOKENS" not in env
        assert "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS" not in env


class TestIntegrationSettings:
    def test_settings_dataclass(self):
        from omlx.settings import IntegrationSettings

        settings = IntegrationSettings()
        assert settings.copilot_model is None
        assert settings.codex_model is None
        assert settings.opencode_model is None
        assert settings.openclaw_model is None
        assert settings.hermes_model is None
        assert settings.pi_model is None
        assert settings.dsh_model is None
        assert settings.openclaw_tools_profile == "coding"

    def test_to_dict(self):
        from omlx.settings import IntegrationSettings

        settings = IntegrationSettings(codex_model="qwen3.5", dsh_model="Qwen3.8")
        d = settings.to_dict()
        assert d["copilot_model"] is None
        assert d["codex_model"] == "qwen3.5"
        assert d["opencode_model"] is None
        assert d["hermes_model"] is None
        assert d["pi_model"] is None
        assert d["dsh_model"] == "Qwen3.8"
        assert d["openclaw_tools_profile"] == "coding"

    def test_from_dict(self):
        from omlx.settings import IntegrationSettings

        settings = IntegrationSettings.from_dict(
            {
                "copilot_model": "gpt-oss",
                "codex_model": "llama",
                "opencode_model": "qwen",
                "hermes_model": "hermes-qwen",
                "dsh_model": "Qwen3.8",
            }
        )
        assert settings.copilot_model == "gpt-oss"
        assert settings.codex_model == "llama"
        assert settings.opencode_model == "qwen"
        assert settings.openclaw_model is None
        assert settings.hermes_model == "hermes-qwen"
        assert settings.pi_model is None
        assert settings.dsh_model == "Qwen3.8"

    def test_from_dict_empty(self):
        from omlx.settings import IntegrationSettings

        settings = IntegrationSettings.from_dict({})
        assert settings.codex_model is None
        assert settings.dsh_model is None


# ---------------------------------------------------------------------------
# DeepSeek Harness (dsh)
# ---------------------------------------------------------------------------

DSH_MODELS = [
    {
        "id": "Qwen3.8-27B-oQ5e-mtp",
        "name": "Qwen3.8-27B-oQ5e-mtp",
        "contextWindow": 131072,
        "maxTokens": 32768,
        "input": ["text"],
    },
    {
        "id": "qwen-vl",
        "name": "qwen-vl",
        "contextWindow": 32768,
        "maxTokens": 8192,
        "input": ["text", "image"],
    },
]


def make_dsh_bundle(root: Path, with_cli: bool = False) -> Path:
    """Create a fake DeepSeek Harness app bundle in ``root``."""
    return make_app_bundle(
        root, "DeepSeek Harness.app", bundle_id="com.deepseek.dsh", with_cli=with_cli
    )


def _route(patch_path: Path, protocol: str = "openai-responses") -> tuple[dict, dict]:
    data = yaml.safe_load(patch_path.read_text())
    pi_ai = next(e for e in data if e.get("id") == "llm-pi-ai")
    route = pi_ai["config"]["providers"]["omlx"]
    assert route["api"] == protocol
    return route, data


class TestDshIntegration:
    def test_get_command(self):
        dsh = DshIntegration()
        cmd = dsh.get_command(ctx(model="Qwen3.8-27B-oQ5e-mtp"))
        assert "omlx launch dsh" in cmd
        assert "--model Qwen3.8-27B-oQ5e-mtp" in cmd

    def test_get_command_no_model(self):
        dsh = DshIntegration()
        assert dsh.get_command(ctx(model="")) == "omlx launch dsh"

    def test_type(self):
        assert DshIntegration().type == "config_file"

    def test_requires_no_interactive_model_selection(self):
        # dsh registers the whole model catalog; the model flag only seeds
        # the harness session default, so launch must not force a picker.
        assert DshIntegration().requires_model_selection is False
        # Every other integration keeps the picker semantics.
        assert OpenCodeIntegration().requires_model_selection is True
        assert PiIntegration().requires_model_selection is True

    def test_is_installed_with_app_bundle(self, tmp_path, monkeypatch):
        make_dsh_bundle(tmp_path)
        monkeypatch.setattr("omlx.integrations.dsh._APP_BUNDLE_ROOTS", (tmp_path,))
        assert DshIntegration().is_installed() is True
        assert find_dsh_app_bundle() == tmp_path / "DeepSeek Harness.app"

    def test_not_installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("omlx.integrations.dsh._APP_BUNDLE_ROOTS", (tmp_path,))
        assert DshIntegration().is_installed() is False
        assert find_dsh_app_bundle() is None

    def test_launch_without_bundle_exits(self, monkeypatch):
        monkeypatch.setattr("omlx.integrations.dsh.find_dsh_app_bundle", lambda: None)
        monkeypatch.setattr(DshIntegration, "configure", lambda self, c: None)
        with pytest.raises(SystemExit):
            DshIntegration().launch(ctx())

    def test_configure_writes_route_and_credential(self, tmp_path, monkeypatch):
        patch_path = tmp_path / "profiles" / "desktop" / "cordis.patch.yml"
        creds_path = tmp_path / ".credentials.yaml"
        monkeypatch.setattr("omlx.integrations.dsh.patch_file_path", lambda: patch_path)
        monkeypatch.setattr(
            "omlx.integrations.dsh.credentials_file_path", lambda: creds_path
        )
        monkeypatch.setattr(
            "omlx.integrations.dsh.web_patch_file_path",
            lambda: tmp_path / "profiles" / "web" / "cordis.patch.yml",
        )
        monkeypatch.setattr(
            DshIntegration,
            "_fetch_models",
            lambda self, c: [dict(m) for m in DSH_MODELS],
        )

        DshIntegration().configure(ctx(api_key="sk-test"))

        route, data = _route(patch_path)
        assert route["baseURL"] == "http://127.0.0.1:8000/v1"
        assert route["apiKeyEnv"] == "OMLX_API_KEY"
        assert [m["id"] for m in route["models"]] == [m["id"] for m in DSH_MODELS]
        assert route["models"][0]["contextWindow"] == 131072
        assert route["models"][1]["input"] == ["text", "image"]
        # No default model was named: the session default stays untouched.
        assert not any(e.get("id") == "agent-default-model" for e in data)

        creds = yaml.safe_load(creds_path.read_text())
        assert creds["refs"]["OMLX_API_KEY"] == "sk-test"

    def test_configure_sets_default_model(self, tmp_path, monkeypatch):
        patch_path = tmp_path / "cordis.patch.yml"
        creds_path = tmp_path / ".credentials.yaml"
        monkeypatch.setattr("omlx.integrations.dsh.patch_file_path", lambda: patch_path)
        monkeypatch.setattr(
            "omlx.integrations.dsh.credentials_file_path", lambda: creds_path
        )
        monkeypatch.setattr(
            "omlx.integrations.dsh.web_patch_file_path",
            lambda: tmp_path / "profiles" / "web" / "cordis.patch.yml",
        )
        monkeypatch.setattr(
            DshIntegration,
            "_fetch_models",
            lambda self, c: [dict(m) for m in DSH_MODELS],
        )

        DshIntegration().configure(ctx(model="Qwen3.8-27B-oQ5e-mtp", api_key=""))

        _, data = _route(patch_path)
        agent = next(e for e in data if e.get("id") == "agent-default-model")
        assert agent["config"] == {
            "provider": "omlx",
            "model": "Qwen3.8-27B-oQ5e-mtp",
        }
        # No API key configured: the ref still resolves (the dummy token the
        # base context supplies) so the route never fails MISSING_CREDENTIAL.
        creds = yaml.safe_load(creds_path.read_text())
        assert creds["refs"]["OMLX_API_KEY"] == "omlx"

    def test_configure_updates_web_profile_too(self, tmp_path, monkeypatch):
        # `dsh web` boots profiles/web — a separate profile from desktop's.
        # The same provider route must land in both patch layers.
        monkeypatch.setattr(
            "omlx.integrations.dsh.patch_file_path",
            lambda: tmp_path / "profiles" / "desktop" / "cordis.patch.yml",
        )
        monkeypatch.setattr(
            "omlx.integrations.dsh.web_patch_file_path",
            lambda: tmp_path / "profiles" / "web" / "cordis.patch.yml",
        )
        monkeypatch.setattr(
            "omlx.integrations.dsh.credentials_file_path",
            lambda: tmp_path / ".credentials.yaml",
        )
        monkeypatch.setattr(
            DshIntegration,
            "_fetch_models",
            lambda self, c: [dict(m) for m in DSH_MODELS],
        )

        DshIntegration().configure(ctx())

        for target in (
            tmp_path / "profiles" / "desktop" / "cordis.patch.yml",
            tmp_path / "profiles" / "web" / "cordis.patch.yml",
        ):
            route, _ = _route(target)
            assert route["baseURL"] == "http://127.0.0.1:8000/v1"
            assert [m["id"] for m in route["models"]] == [m["id"] for m in DSH_MODELS]
        creds = yaml.safe_load((tmp_path / ".credentials.yaml").read_text())
        assert "OMLX_API_KEY" in creds["refs"]

    def test_configure_exits_when_server_has_no_models(self, monkeypatch):
        monkeypatch.setattr(DshIntegration, "_fetch_models", lambda self, c: [])
        with pytest.raises(SystemExit):
            DshIntegration().configure(ctx())

    def test_fetch_models_reads_capacity_and_modalities(self):
        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {
                    "id": "qwen-llm",
                    "max_context_window": 65536,
                    "max_tokens": 4096,
                    "model_type": "llm",
                },
                {"id": "qwen-vl", "model_type": "vlm"},
            ]
        }
        models_response = MagicMock()
        models_response.json.return_value = {
            "data": [{"id": "qwen-llm"}, {"id": "qwen-vl"}]
        }

        with patch("requests.get", side_effect=[status_response, models_response]):
            models = DshIntegration()._fetch_models(ctx(api_key="k"))

        assert [m["id"] for m in models] == ["qwen-llm", "qwen-vl"]
        assert models[0]["contextWindow"] == 65536
        assert models[0]["maxTokens"] == 4096
        assert models[0]["input"] == ["text"]
        assert models[1]["input"] == ["text", "image"]

    def test_fetch_models_skips_non_chat_model_types(self):
        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {"id": "chat-llm", "model_type": "llm"},
                {"id": "embed-bert", "model_type": "embedding"},
                {"id": "asr", "model_type": "audio_sts"},
            ]
        }
        models_response = MagicMock()
        models_response.json.return_value = {
            # The list endpoint leaves model_type unset; filtering happens
            # against the status map. Unknown types stay in.
            "data": [
                {"id": "chat-llm"},
                {"id": "embed-bert"},
                {"id": "asr"},
                {"id": "untyped"},
            ]
        }

        with patch("requests.get", side_effect=[status_response, models_response]):
            models = DshIntegration()._fetch_models(ctx())

        assert [m["id"] for m in models] == ["chat-llm", "untyped"]

    def test_fetch_models_falls_back_to_selected_model(self):
        status_response = MagicMock()
        status_response.ok = False
        with patch("requests.get", side_effect=[status_response, Exception("down")]):
            models = DshIntegration()._fetch_models(ctx(model="only-model"))

        assert models == [{"id": "only-model", "name": "only-model", "input": ["text"]}]

    def test_launch_opens_app_and_scrubs_session_env(self, monkeypatch):
        calls = {}

        def fake_run(cmd, env=None, **kwargs):
            calls["cmd"] = cmd
            calls["env"] = env
            return MagicMock(returncode=0)

        monkeypatch.setattr(
            "omlx.integrations.dsh.find_dsh_app_bundle",
            lambda: Path("/Applications/DeepSeek Harness.app"),
        )
        monkeypatch.setattr("omlx.integrations.dsh._app_is_running", lambda: False)
        monkeypatch.setattr("omlx.integrations.dsh.subprocess.run", fake_run)
        monkeypatch.setattr(DshIntegration, "configure", lambda self, c: None)

        integration = DshIntegration()
        integration.launch(ctx())

        assert calls["cmd"] == [
            "open",
            "/Applications/DeepSeek Harness.app",
        ]
        for key in ("DSH_HOME", "DSH_SESSION_ID", "DSH_SESSION_JSONL", "DSH_SHELL"):
            assert key not in calls["env"]


class TestDshPatchWriter:
    def test_creates_new_file(self, tmp_path):
        path = tmp_path / "cordis.patch.yml"
        write_dsh_patch(path, "http://127.0.0.1:8000/v1", DSH_MODELS)

        route, data = _route(path)
        assert route["baseURL"] == "http://127.0.0.1:8000/v1"
        assert [m["id"] for m in route["models"]] == [m["id"] for m in DSH_MODELS]
        assert isinstance(data, list) and data[0]["id"] == "llm-pi-ai"

    def test_replaces_route_and_preserves_everything_else(self, tmp_path):
        path = tmp_path / "cordis.patch.yml"
        path.write_text(
            "# header comment kept\n"
            "- id: llm-pi-ai\n"
            '  name: "@deepseek-ai/dsh-llm-pi-ai"\n'
            "  config:\n"
            "    providers:\n"
            "      # 用户的旧注释\n"
            "      omlx:\n"
            "        displayName: omlx\n"
            "        api: openai-completions\n"
            '        baseURL: "http://127.0.0.1:8000/v1"\n'
            "        models:\n"
            "          - id: Stale-Model\n"
            "            name: Stale-Model\n"
            "      opencode-go1:\n"
            "        displayName: OpenCode-Go\n"
            "        # 分组＝协议\n"
            "        api: openai-completions\n"
            '        baseURL: "https://opencode.ai/zen/go/v1"\n'
            "        models:\n"
            "          - id: mimo-v2.6-flash\n"
            "            name: mimo-v2.6-flash\n"
            "- id: ui-chat\n"
            '  name: "@deepseek-ai/dsh-client-ui-chat"\n'
            "  config:\n"
            "    transcriptView: standard\n"
        )

        write_dsh_patch(path, "http://127.0.0.1:9000/v1", DSH_MODELS)
        out = path.read_text()

        # The managed route was replaced with the full catalog.
        route, data = _route(path)
        assert route["baseURL"] == "http://127.0.0.1:9000/v1"
        assert [m["id"] for m in route["models"]] == [m["id"] for m in DSH_MODELS]
        assert "Stale-Model" not in out

        # Everything the user did not ask us to manage survives.
        assert "# header comment kept" in out
        assert "# 分组＝协议" in out
        assert "opencode-go1" in yaml.safe_load(out)[0]["config"]["providers"]
        assert data[-1]["id"] == "ui-chat"
        assert data[-1]["config"] == {"transcriptView": "standard"}

        # A timestamped backup of the previous file exists.
        assert any(tmp_path.glob("cordis.patch.*.bak"))

    def test_no_default_model_leaves_agent_entry_untouched(self, tmp_path):
        path = tmp_path / "cordis.patch.yml"
        path.write_text(
            "- id: agent-default-model\n"
            '  name: "@deepseek-ai/dsh-agent-default-model"\n'
            "  config:\n"
            "    provider: opencode-go1\n"
            '    model: "mimo-v2.6-flash"\n'
        )

        write_dsh_patch(path, "http://127.0.0.1:8000/v1", DSH_MODELS)

        data = yaml.safe_load(path.read_text())
        agent = next(e for e in data if e.get("id") == "agent-default-model")
        assert agent["config"]["provider"] == "opencode-go1"
        assert agent["config"]["model"] == "mimo-v2.6-flash"

    def test_default_model_rewrites_existing_entry(self, tmp_path):
        path = tmp_path / "cordis.patch.yml"
        path.write_text(
            "- id: agent-default-model\n"
            '  name: "@deepseek-ai/dsh-agent-default-model"\n'
            "  config:\n"
            "    provider: opencode-go1\n"
            '    model: "mimo-v2.6-flash"\n'
            "- id: llm-pi-ai\n"
            '  name: "@deepseek-ai/dsh-llm-pi-ai"\n'
            "  config:\n"
            "    providers:\n"
            "      omlx:\n"
            "        api: openai-completions\n"
            '        baseURL: "http://127.0.0.1:8000/v1"\n'
            "        models:\n"
            "          - id: x\n"
        )

        write_dsh_patch(
            path,
            "http://127.0.0.1:8000/v1",
            DSH_MODELS,
            default_model="Qwen3.8-27B-oQ5e-mtp",
        )

        data = yaml.safe_load(path.read_text())
        agent = next(e for e in data if e.get("id") == "agent-default-model")
        assert agent["config"] == {
            "provider": "omlx",
            "model": "Qwen3.8-27B-oQ5e-mtp",
        }

    def test_empty_flow_config_is_rewritten(self, tmp_path):
        path = tmp_path / "cordis.patch.yml"
        path.write_text(
            "- id: llm-pi-ai\n"
            '  name: "@deepseek-ai/dsh-llm-pi-ai"\n'
            "  config: {}\n"
        )

        write_dsh_patch(path, "http://127.0.0.1:8000/v1", DSH_MODELS)

        route, _ = _route(path)
        assert [m["id"] for m in route["models"]] == [m["id"] for m in DSH_MODELS]

    def test_empty_flow_providers_is_rewritten(self, tmp_path):
        path = tmp_path / "cordis.patch.yml"
        path.write_text(
            "- id: llm-pi-ai\n"
            '  name: "@deepseek-ai/dsh-llm-pi-ai"\n'
            "  config:\n"
            "    providers: {}\n"
        )

        write_dsh_patch(path, "http://127.0.0.1:8000/v1", DSH_MODELS)

        route, _ = _route(path)
        assert route["baseURL"] == "http://127.0.0.1:8000/v1"

    def test_populated_inline_config_is_refused(self, tmp_path):
        path = tmp_path / "cordis.patch.yml"
        path.write_text(
            "- id: llm-pi-ai\n"
            '  name: "@deepseek-ai/dsh-llm-pi-ai"\n'
            "  config: {retryPolicy: {mode: normal}}\n"
        )

        with pytest.raises(DshConfigShapeError):
            write_dsh_patch(path, "http://127.0.0.1:8000/v1", DSH_MODELS)

        # The refusal never overwrote the user's file.
        assert "retryPolicy" in path.read_text()

    def test_refuses_empty_model_list(self, tmp_path):
        path = tmp_path / "cordis.patch.yml"
        with pytest.raises(DshConfigShapeError):
            write_dsh_patch(path, "http://127.0.0.1:8000/v1", [])
        assert not path.exists()

    def test_protocol_env_override(self, tmp_path, monkeypatch):
        path = tmp_path / "cordis.patch.yml"
        monkeypatch.setenv("OMLX_DSH_API", "openai-completions")
        write_dsh_patch(path, "http://127.0.0.1:8000/v1", DSH_MODELS)
        _route(path, protocol="openai-completions")

        monkeypatch.setenv("OMLX_DSH_API", "bogus")
        with pytest.raises(DshConfigShapeError):
            write_dsh_patch(path, "http://127.0.0.1:8000/v1", DSH_MODELS)

    def test_js_tags_do_not_break_parsing(self, tmp_path):
        # The patch layer allows `!!js` expressions; a route update must not
        # choke on them or drop them.
        path = tmp_path / "cordis.patch.yml"
        path.write_text(
            "- id: llm-pi-ai\n"
            '  name: "@deepseek-ai/dsh-llm-pi-ai"\n'
            "  config:\n"
            "    providers:\n"
            "      omlx:\n"
            "        api: openai-completions\n"
            '        baseURL: "http://127.0.0.1:8000/v1"\n'
            "        models:\n"
            "          - id: x\n"
        )

        write_dsh_patch(path, "http://127.0.0.1:8000/v1", DSH_MODELS)
        assert "omlx" in path.read_text()


class TestDshCredentialsRef:
    def test_updates_existing_ref_and_keeps_others(self, tmp_path):
        path = tmp_path / ".credentials.yaml"
        path.write_text(
            "version: 1\n"
            "records:\n"
            "  deepseek-account-platform/default:\n"
            "    kind: token\n"
            "    payload:\n"
            '      token: "secret"\n'
            "refs:\n"
            '  DEEPSEEK_API_KEY: "dk"\n'
            '  OMLX_API_KEY: "stale"\n'
        )

        write_credentials_ref(path, "OMLX_API_KEY", "sk-fresh")

        data = yaml.safe_load(path.read_text())
        assert data["version"] == 1
        assert data["refs"]["OMLX_API_KEY"] == "sk-fresh"
        assert data["refs"]["DEEPSEEK_API_KEY"] == "dk"
        assert (
            data["records"]["deepseek-account-platform/default"]["payload"]["token"]
            == "secret"
        )
        assert any(tmp_path.glob(".credentials.*.bak"))

    def test_appends_missing_ref_to_existing_section(self, tmp_path):
        path = tmp_path / ".credentials.yaml"
        path.write_text('version: 1\nrecords: {}\nrefs:\n  DEEPSEEK_API_KEY: "dk"\n')

        write_credentials_ref(path, "OMLX_API_KEY", "sk-new")

        data = yaml.safe_load(path.read_text())
        assert data["refs"] == {"DEEPSEEK_API_KEY": "dk", "OMLX_API_KEY": "sk-new"}

    def test_appends_refs_section_when_missing(self, tmp_path):
        path = tmp_path / ".credentials.yaml"
        path.write_text("version: 1\nrecords: {}\n")

        write_credentials_ref(path, "OMLX_API_KEY", "sk-new")

        data = yaml.safe_load(path.read_text())
        assert data["refs"]["OMLX_API_KEY"] == "sk-new"
        assert data["records"] == {}

    def test_creates_missing_file(self, tmp_path):
        path = tmp_path / ".credentials.yaml"
        write_credentials_ref(path, "OMLX_API_KEY", "sk-new")

        data = yaml.safe_load(path.read_text())
        assert data == {
            "version": 1,
            "records": {},
            "refs": {"OMLX_API_KEY": "sk-new"},
        }
