# SPDX-License-Identifier: Apache-2.0
"""Usage history dashboard rendering and its i18n keys."""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omlx_web import routes as webui

from omlx.admin.auth import require_admin
from omlx.admin.routes import router
from omlx.server_metrics import ServerMetrics
from omlx.usage_history import UsageHistory

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "omlx_web/i18n"
USAGE_HISTORY_I18N_KEYS = {
    "settings.usage.section_label",
    "settings.usage.history",
    "settings.usage.history_hint",
    "usage.disabled",
    "usage.open_settings",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    metrics = ServerMetrics()
    metrics.usage_history = UsageHistory(tmp_path / "usage.sqlite3")
    metrics.record_request_complete(100, 20, 60, 0.5, 1.0, "canonical-model", 2.0)
    metrics.usage_history.flush()
    monkeypatch.setattr(
        webui,
        "_host",
        webui.WebUIHost(
            version="test",
            require_admin=require_admin,
            is_admin=lambda request: True,
            ui_language=lambda: "en",
            main_api_key=lambda: None,
        ),
    )
    app = FastAPI()
    app.include_router(webui.router)
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: True
    app.dependency_overrides[webui.require_admin] = lambda: True
    with (
        patch("omlx.server_metrics.get_server_metrics", return_value=metrics),
        TestClient(app) as client,
    ):
        yield client, metrics
    metrics.close()


def test_usage_template_renders_with_localized_labels(client):
    client, _ = client
    response = client.get("/admin/dashboard")
    assert response.status_code == 200
    assert 'x-data="usageHistory()"' in response.text
    assert "Usage History" in response.text
    assert "js/usage.js" in response.text
    assert (
        'id="usage-heading" class="text-xl font-bold">Usage History</h3>'
        in response.text
    )


def test_dashboard_renders_usage_history_switch_and_disabled_notice(client):
    client, _ = client
    html = client.get("/admin/dashboard").text
    assert "globalSettings.usage.usage_history" in html
    assert "Record usage history" in html
    assert "Usage history is off. Turn it on in Settings" in html
    assert "setSettingsTab('global')" in html
    javascript = (ROOT / "omlx_web/static/js/dashboard.js").read_text(encoding="utf-8")
    assert "usage: { usage_history: true }" in javascript
    assert "usage_history: this.globalSettings.usage.usage_history" in javascript


def test_usage_history_i18n_keys_present_in_every_locale():
    locales = sorted(I18N_DIR.glob("*.json"))
    assert len(locales) == 10
    for locale_path in locales:
        locale = json.loads(locale_path.read_text(encoding="utf-8"))
        missing = {key for key in USAGE_HISTORY_I18N_KEYS if not locale.get(key)}
        assert not missing, f"{locale_path.name}: missing {sorted(missing)}"


def test_dashboard_counts_use_wan_and_yi_only_in_chinese():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('omlx_web/static/js/dashboard.js', 'utf8');
const render = (lang, fn, value) => {
    const context = {localStorage: {getItem: () => null}, window: {t: key => key},
                     document: {documentElement: {lang}}};
    return vm.runInNewContext(source + '\n dashboard;', context)()[fn](value);
};
assert.equal(render('zh', 'formatNumber', 9999), '9,999');
assert.equal(render('zh', 'formatNumber', 19290), '1.9万');
assert.equal(render('zh', 'formatNumber', 12345678), '1,234.6万');
assert.equal(render('zh', 'formatNumber', 99999999), '1亿');
assert.equal(render('zh', 'formatTokenCount', 1.2e12), '1.2万亿');
assert.equal(render('zh-TW', 'formatDownloads', 123456789), '1.2億');
// Every other language keeps its current output.
assert.equal(render('en', 'formatNumber', 12345678), '12.3M');
assert.equal(render('ko', 'formatTokenCount', 19290), '19.3k');
assert.equal(render('ja', 'formatDownloads', 19290), '19.3K');
"""
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
