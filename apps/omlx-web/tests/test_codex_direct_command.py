# SPDX-License-Identifier: Apache-2.0
"""Run the production dashboard recipe through a shell and parse its TOML args."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

JAVASCRIPT = Path(__file__).resolve().parents[1] / "omlx_web/static/js/dashboard.js"


def recipe(model="", host="127.0.0.1", context=None):
    if not shutil.which("node"):
        pytest.skip("node is required to execute the dashboard command builder")
    fixture = dict(
        model=model,
        host=host,
        context=context,
    )
    script = """
const fs = require('fs');
const vm = require('vm');
const fixture = JSON.parse(process.argv[2]);
const sandbox = {
    window: {t: key => key},
    localStorage: {getItem: () => null},
    console,
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const component = sandbox.dashboard();
component.globalSettings.integrations.codex_model = fixture.model;
component.stats = {host: fixture.host, port: 8000, api_key: 'synthetic-private-key'};
component.models = fixture.model ? [{id: fixture.model, model_context_length: 131072,
    settings: {max_context_window: fixture.context}}] : [];
process.stdout.write(component.codexDirectCommand);
"""
    result = subprocess.run(
        ["node", "-e", script, str(JAVASCRIPT), json.dumps(fixture)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.mark.parametrize(
    "model", ["", "example-model", "a'$(touch SHOULD_NOT_EXIST)\\\"b"]
)
@pytest.mark.parametrize("host", ["127.0.0.1", "[::1]", "0.0.0.0", "::"])
def test_direct_command_round_trips_shell_and_toml(tmp_path, model, host):
    command = recipe(model=model, host=host, context=32768)
    stub = tmp_path / "codex"
    stub.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + ' -c \'import json,os,sys; print(json.dumps({"args":sys.argv[1:],'
        + '"key":os.environ.get("OMLX_API_KEY")}))\' "$@"\n'
    )
    stub.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    captured = json.loads(result.stdout)
    args = captured["args"]
    assert args[:2] == ["-m", model or "select-a-model"]
    config = tomllib.loads("\n".join(args[3::2]))
    provider = config["model_providers"]["omlx"]
    expected_host = "[::1]" if host == "[::1]" else "127.0.0.1"
    assert provider["base_url"] == f"http://{expected_host}:8000/v1"
    assert provider["wire_api"] == "responses"
    assert provider["env_key"] == "OMLX_API_KEY"
    assert provider["requires_openai_auth"] is False
    assert provider["supports_websockets"] is False
    assert provider["stream_idle_timeout_ms"] == 3000000
    assert captured["key"] == "your-omlx-api-key"
    assert "synthetic-private-key" not in command
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()
    if model:
        assert config["model_context_window"] == 32768
    else:
        assert "model_context_window" not in config
