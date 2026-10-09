# SPDX-License-Identifier: Apache-2.0
"""Mistral Vibe dashboard command regression."""


def test_dashboard_command_uses_saved_model_with_shell_quoting():
    import shutil
    import subprocess
    from pathlib import Path

    import pytest

    if not shutil.which("node"):
        pytest.skip("Node.js is needed for dashboard command execution")
    source = Path(__file__).resolve().parents[1] / "omlx_web/static/js/dashboard.js"
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
