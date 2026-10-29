# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess
from pathlib import Path

import pytest


def test_image_settings_never_serialize_hidden_chat_options():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js required")
    root = Path(__file__).resolve().parents[1] / "omlx_web"
    script = "globalThis.localStorage = {getItem: () => null}; globalThis.window = {t: k => k};\n"
    script += (root / "static/js/dashboard.js").read_text()
    script += """
const assert = require('node:assert/strict');
const app = dashboard();
app.selectedModel = {model_type: 'image'};
const state = app.buildModelSettingsState(app.selectedModel, {
    model_alias: ' boat ', ttl_seconds: 30, model_type_override: 'llm',
    temperature: 0.7, dflash_enabled: true, external_engine: 'vllm-metal'
});
assert.deepEqual(app.buildModelSettingsPayload(state), {model_alias: 'boat', ttl_seconds: 30});
app.selectedModel = {model_type: 'llm'};
assert.ok('temperature' in app.buildModelSettingsPayload(state));
"""
    result = subprocess.run([node], input=script, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
