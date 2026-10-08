# SPDX-License-Identifier: Apache-2.0
"""Mistral Vibe launch/configuration regressions."""

import json
import os
from unittest.mock import patch

from omlx.integrations.base import IntegrationContext
from omlx.integrations.vibe import VibeIntegration
from omlx.settings import IntegrationSettings


def test_launch_scopes_configuration_and_preserves_parent(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    original = 'active_model = "cloud"\n# custom configuration\n'
    config.write_text(original)
    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    monkeypatch.setenv("VIBE_ACTIVE_MODEL", "cloud")
    monkeypatch.setenv("PYTHONHOME", "/bundled/python")
    monkeypatch.setenv("PYTHONPATH", "/bundled/modules")
    monkeypatch.setenv("MISTRAL_API_KEY", "cloud-secret")
    ctx = IntegrationContext(
        host="127.0.0.1",
        port=8000,
        api_key="local-secret",
        model='model with "quotes"',
        context_window=32768,
        model_type="vlm",
        extra_args=("--resume", "session-id"),
    )
    with patch("omlx.integrations.vibe.os.execvpe") as execute:
        VibeIntegration().launch(ctx)
    command, args, env = execute.call_args.args
    assert command == "vibe"
    assert args == ["vibe", "--resume", "session-id"]
    assert "local-secret" not in str(args)
    assert env["OMLX_API_KEY"] == "local-secret"
    assert env["VIBE_HOME"] == str(tmp_path)
    assert env["MISTRAL_API_KEY"] == "cloud-secret"
    assert "PYTHONHOME" not in env and "PYTHONPATH" not in env
    provider = json.loads(env["VIBE_PROVIDERS"])[0]
    assert provider["api_base"] == "http://127.0.0.1:8000/v1"
    assert provider["api_key_env_var"] == "OMLX_API_KEY"
    assert provider["backend"] == "generic"
    models = json.loads(env["VIBE_MODELS"])
    selected = next(
        model for model in models if model["alias"] == env["VIBE_ACTIVE_MODEL"]
    )
    assert selected["name"] == ctx.model
    assert selected["provider"] == provider["name"]
    assert selected["max_context_length"] == 32768
    assert selected["supports_images"] is True
    assert selected["input_price"] == selected["output_price"] == 0
    assert config.read_text() == original
    assert os.environ["VIBE_ACTIVE_MODEL"] == "cloud"


def test_keyless_launch_and_unknown_capacity():
    with patch("omlx.integrations.vibe.os.execvpe") as execute:
        VibeIntegration().launch(
            IntegrationContext(host="localhost", port=8000, model="local")
        )
    env = execute.call_args.args[2]
    assert env["OMLX_API_KEY"] == "omlx"
    model = json.loads(env["VIBE_MODELS"])[0]
    assert "max_context_length" not in model
    assert model["supports_images"] is False


def test_command_quotes_model():
    command = VibeIntegration().get_command(
        IntegrationContext(host="localhost", port=8000, model="local model")
    )
    assert command.endswith("launch vibe --model 'local model'")


def test_default_model_roundtrip():
    settings = IntegrationSettings.from_dict({"vibe_model": "local"})
    assert settings.vibe_model == "local"
    assert settings.to_dict()["vibe_model"] == "local"
    assert IntegrationSettings().vibe_model is None


def test_admin_saves_and_clears_default_without_affecting_other_tools(
    tmp_path, monkeypatch
):
    import asyncio
    import importlib

    importlib.import_module("omlx.server")  # Initialize admin callbacks first.
    from omlx.admin import routes
    from omlx.settings import GlobalSettings

    settings = GlobalSettings(base_path=tmp_path)
    settings.integrations.pi_model = "existing-pi-model"
    monkeypatch.setattr(routes, "_get_global_settings", lambda: settings)
    for model in ("local-vibe-model", None):
        request = routes.GlobalSettingsRequest(integrations_vibe_model=model)
        result = asyncio.run(
            routes.update_global_settings(request=request, is_admin=True)
        )
        assert result["success"] is True
        assert GlobalSettings.load(base_path=tmp_path).integrations.vibe_model == model
        response = asyncio.run(routes.get_global_settings(is_admin=True))
        assert response["integrations"]["vibe_model"] == model
        assert settings.integrations.pi_model == "existing-pi-model"


def test_dashboard_command_uses_saved_model_with_shell_quoting():
    import shutil
    import subprocess
    from pathlib import Path

    import pytest

    if not shutil.which("node"):
        pytest.skip("Node.js is needed for dashboard command execution")
    source = Path(__file__).resolve().parents[1] / "omlx/admin/static/js/dashboard.js"
    script = r"""
const assert = require('node:assert/strict');
const source = require('node:fs').readFileSync(process.argv[1], 'utf8');
function member(start, end) {
    return source.split(start)[1].split(end)[0];
}
const app = Function('return {' +
    'shellQuote(value) {' + member('shellQuote(value) {', 'shellEnvAssign(name, value) {') +
    '_launchCmd(tool) {' + member('_launchCmd(tool) {', 'get claudeCommand() {') +
    'get vibeCommand() {' + member('get vibeCommand() {', 'get dshCommand() {') +
    '}')();
app.stats = {cli_prefix: 'omlx'};
app.globalSettings = {integrations: {vibe_model: null}};
assert.equal(app.vibeCommand, 'omlx launch vibe');
app.stats.cli_prefix = '/My Apps/oMLX.app/omlx-cli';
app.globalSettings.integrations.vibe_model = "local model's name";
assert.equal(app.vibeCommand, "'/My Apps/oMLX.app/omlx-cli' launch vibe --model 'local model'\"'\"'s name'");
"""
    subprocess.run(
        ["node", "-e", script, str(source)], check=True, capture_output=True, text=True
    )


def test_alias_survives_resume_and_separates_models_and_servers():
    aliases = []
    contexts = [
        IntegrationContext(host="localhost", port=8000, model="local"),
        IntegrationContext(host="localhost", port=8000, model="local"),
        IntegrationContext(host="localhost", port=8000, model="other"),
        IntegrationContext(host="localhost", port=8001, model="local"),
    ]
    for context in contexts:
        with patch("omlx.integrations.vibe.os.execvpe") as execute:
            VibeIntegration().launch(context)
        aliases.append(execute.call_args.args[2]["VIBE_ACTIVE_MODEL"])
    assert aliases[0] == aliases[1]
    assert len(set(aliases)) == 3
