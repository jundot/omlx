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
    assert buttons

    for button in buttons:
        repo = button.group(1)
        tag = html[button.start() : html.index(">", button.start())]
        assert f'x-show="!isModelDownloaded({repo})"' in tag
    assert html.count("t('models.browse.downloaded')") == len(buttons)


def test_every_locale_names_the_badge():
    for locale in sorted((ROOT / "omlx_web/i18n").glob("*.json")):
        data = json.loads(locale.read_text(encoding="utf-8"))
        assert data.get("models.browse.downloaded"), locale.name


def test_is_model_downloaded_needs_a_finished_local_repo():
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

state.hfTasks = [{repo_id: 'mlx-community/Qwen3-8B-4bit', status: 'failed'}];
assert.equal(state.isModelDownloaded('mlx-community/Qwen3-8B-4bit'), false);
state.hfTasks.push({repo_id: 'mlx-community/Qwen3-8B-4bit', status: 'completed'});
assert.equal(state.isModelDownloaded('mlx-community/Qwen3-8B-4bit'), true);
state.msTasks = [{repo_id: 'mlx-community/Qwen3-8B-4bit', status: 'downloading'}];
assert.equal(state.isModelDownloaded('mlx-community/Qwen3-8B-4bit'), false);
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_recommended_rows_and_first_run_banner_follow_downloads():
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
    window: {t: key => key}, navigator: {language: 'en'}, document: {},
})();
const repo = 'Jundot/Qwen3.8-27B-oQ4e-mtp';

// The MarkItDown entry is virtual and does not count as a chat model.
state.models = [{id: 'markitdown', virtual: true}];
state.onboardingStartedEmpty = !state.hasUserModels;
assert.equal(state.onboardingState, 'first');
assert.equal(state.recState(repo), 'idle');

state.hfTasks = [{repo_id: repo, status: 'downloading', progress: 40}];
assert.equal(state.onboardingState, 'downloading');
assert.equal(state.recState(repo), 'downloading');

state.hfTasks = [{repo_id: repo, status: 'completed'}];
state.hfModels = [{name: 'Qwen3.8-27B-oQ4e-mtp', repo_id: repo}];
state.models.push({id: 'Qwen3.8-27B-oQ4e-mtp', display_name: 'Qwen3.8-27B-oQ4e-mtp'});
assert.equal(state.recState(repo), 'downloaded');
assert.equal(state.onboardingState, 'ready');
assert.equal(state.onboardingReadyName, 'Qwen3.8-27B-oQ4e-mtp');
state.onboardingReadyDismissed = true;
assert.equal(state.onboardingState, null);

// A dashboard that opened with models never shows the banner.
state.onboardingStartedEmpty = false;
state.onboardingReadyDismissed = false;
assert.equal(state.onboardingState, null);
"""
    result = subprocess.run(
        [node, "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
