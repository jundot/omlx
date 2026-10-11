# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Downloaded badge in the model browser."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_BUTTON = re.compile(
    r'<button @click="(?:downloadRecommended|downloadMsModel)\((\w+\.repo_id)\)'
)


def _models_template() -> str:
    return (ROOT / "omlx_web/templates/dashboard/_models.html").read_text()


def test_every_browse_download_button_yields_to_the_badge():
    html = _models_template()
    buttons = list(DOWNLOAD_BUTTON.finditer(html))
    assert len(buttons) == 9

    for button in buttons:
        repo = button.group(1)
        tag = html[button.start() : html.index(">", button.start())]
        assert f'x-show="!isModelDownloaded({repo})"' in tag
        before = html[max(0, button.start() - 800) : button.start()]
        assert f'isModelDownloaded({repo})" x-cloak' in before
        assert "models.browse.downloaded" in before


def test_every_locale_names_the_badge():
    for locale in sorted((ROOT / "omlx_web/i18n").glob("*.json")):
        data = json.loads(locale.read_text(encoding="utf-8"))
        assert data.get("models.browse.downloaded"), locale.name


def test_is_model_downloaded_matches_the_full_repo_id():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync('omlx_web/static/js/dashboard.js', 'utf8');
const state = vm.runInNewContext(source + '\n dashboard;', {
    localStorage: {getItem: () => null},
    THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
    window: {t: key => key}, navigator: {language: 'en'}, document: {},
})();
state.hfModels = [
    {name: 'Qwen3-8B-4bit', repo_id: 'mlx-community/Qwen3-8B-4bit'},
    {name: 'local-only', repo_id: null},
];

assert.equal(state.isModelDownloaded('mlx-community/Qwen3-8B-4bit'), true);
assert.equal(state.isModelDownloaded('MLX-Community/qwen3-8b-4bit'), true);
assert.equal(state.isModelDownloaded('someone-else/Qwen3-8B-4bit'), false);
assert.equal(state.isModelDownloaded('someone/local-only'), false);
assert.equal(state.isModelDownloaded(undefined), false);
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
