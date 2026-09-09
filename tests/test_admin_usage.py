# SPDX-License-Identifier: Apache-2.0
"""Authenticated local admin usage endpoint, including degraded storage."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin.auth import require_admin
from omlx.admin.routes import router
from omlx.server_metrics import ServerMetrics
from omlx.usage_history import UsageHistory

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "omlx/admin/i18n"
USAGE_HISTORY_I18N_KEYS = {
    "settings.usage.section_label",
    "settings.usage.history",
    "settings.usage.history_hint",
    "usage.disabled",
    "usage.open_settings",
}


@pytest.fixture
def client(tmp_path):
    metrics = ServerMetrics()
    metrics.usage_history = UsageHistory(tmp_path / "usage.sqlite3")
    metrics.record_request_complete(100, 20, 60, 0.5, 1.0, "canonical-model", 2.0)
    metrics.usage_history.flush()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_admin] = lambda: True
    with (
        patch("omlx.server_metrics.get_server_metrics", return_value=metrics),
        TestClient(app) as client,
    ):
        yield client, metrics
    metrics.close()


def test_usage_endpoint(client):
    client, _ = client
    response = client.get("/admin/api/usage?range=7d&model=canonical-model")
    assert response.status_code == 200
    data = response.json()
    assert data["totals"]["total_tokens"] == 120
    assert data["models"][0]["model_id"] == "canonical-model"
    assert len(data["heatmap"]) == 7
    assert not any(
        key in data for key in ("api_key", "host", "messages", "prompt", "response")
    )
    assert client.get("/admin/api/usage?range=invalid").status_code == 422
    assert (
        client.get("/admin/api/usage?model=unloaded-model").json()["totals"]["requests"]
        == 0
    )


def test_usage_requires_admin(client):
    client, _ = client
    client.app.dependency_overrides.clear()
    with (
        patch("omlx.admin.auth.verify_session", return_value=False),
        patch("omlx.admin.auth._get_global_settings", return_value=None),
    ):
        assert client.get("/admin/api/usage").status_code == 401


def test_storage_failure_is_generic_503(client):
    client, metrics = client
    with patch.object(
        metrics.usage_history, "query", side_effect=OSError("private path")
    ):
        response = client.get("/admin/api/usage")
    assert response.status_code == 503
    assert response.json() == {"detail": "Usage history unavailable"}
    metrics.record_request_complete(20, 10, model_id="canonical-model")
    assert metrics.get_snapshot()["total_requests"] == 2


@pytest.mark.asyncio
async def test_completed_stream_records_once_and_analytics_failure_preserves_response(
    tmp_path,
):
    import json
    from types import SimpleNamespace

    from omlx import server
    from omlx.api.openai_models import CompletionRequest
    from omlx.engine.base import GenerationOutput

    async def generate(**kwargs):
        yield GenerationOutput(
            text="private output",
            new_text="private output",
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=60,
            finished=True,
        )

    engine = SimpleNamespace(tokenizer=None, stream_generate=generate)
    request = CompletionRequest(
        model="display-alias",
        prompt="private input",
        stream=True,
        stream_options={"include_usage": True},
    )
    metrics = ServerMetrics()
    metrics.usage_history = UsageHistory(tmp_path / "usage.sqlite3")
    try:
        with patch.object(server, "get_server_metrics", return_value=metrics):
            chunks = [
                chunk
                async for chunk in server.stream_completion(
                    engine, request.prompt, request, resolved_model="canonical-model"
                )
            ]
        assert chunks[-1] == "data: [DONE]\n\n"
        wire_usage = [
            json.loads(chunk[6:]) for chunk in chunks[:-1] if '"usage"' in chunk
        ][0]["usage"]
        metrics.usage_history.flush()
        result = metrics.usage_history.query()
        assert result["models"][0]["model_id"] == "canonical-model"
        assert result["totals"]["requests"] == 1
        assert result["totals"]["total_tokens"] == wire_usage["total_tokens"] == 120
        assert (
            result["totals"]["cached_tokens"]
            == wire_usage["prompt_tokens_details"]["cached_tokens"]
            == 60
        )
        assert result["totals"]["timed_requests"] == 1
        assert b"private input" not in metrics.usage_history.path.read_bytes()
        assert b"private output" not in metrics.usage_history.path.read_bytes()
        with (
            patch.object(server, "get_server_metrics", return_value=metrics),
            patch.object(
                metrics.usage_history, "record", side_effect=OSError("storage failed")
            ),
        ):
            chunks = [
                chunk
                async for chunk in server.stream_completion(
                    engine, request.prompt, request, resolved_model="canonical-model"
                )
            ]
        assert chunks[-1] == "data: [DONE]\n\n"
        assert metrics.get_snapshot()["total_requests"] == 2
    finally:
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


def test_usage_endpoint_reports_disabled_state(client):
    client, metrics = client
    metrics.usage_history.set_enabled(False)
    response = client.get("/admin/api/usage?range=7d")
    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is False
    assert data["totals"]["requests"] == 0
    assert data["models"] == []
    assert len(data["heatmap"]) == 7
    metrics.usage_history.set_enabled(True)
    data = client.get("/admin/api/usage?range=7d").json()
    assert data["enabled"] is True
    assert data["totals"]["requests"] == 1


def test_global_settings_toggle_applies_at_runtime(client, tmp_path, monkeypatch):
    from omlx.admin import routes as admin_routes
    from omlx.settings import GlobalSettings

    client, metrics = client
    base_path = tmp_path / "settings"
    gs = GlobalSettings(base_path=base_path)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)

    current = asyncio.run(admin_routes.get_global_settings(is_admin=True))
    assert current["usage"] == {"usage_history": True}

    result = asyncio.run(
        admin_routes.update_global_settings(
            request=admin_routes.GlobalSettingsRequest(usage_history=False),
            is_admin=True,
        )
    )
    assert result["success"] is True
    assert "usage_history" in result["runtime_applied"]
    assert gs.usage.usage_history is False
    assert GlobalSettings.load(base_path=base_path).usage.usage_history is False
    assert metrics.usage_history.enabled is False

    # Serving statistics keep counting; history does not.
    metrics.record_request_complete(100, 20, 60, 0.5, 1.0, "canonical-model", 2.0)
    metrics.usage_history.flush()
    assert metrics.get_snapshot()["total_requests"] == 2
    assert client.get("/admin/api/usage").json()["enabled"] is False

    result = asyncio.run(
        admin_routes.update_global_settings(
            request=admin_routes.GlobalSettingsRequest(usage_history=True),
            is_admin=True,
        )
    )
    assert result["success"] is True
    assert gs.usage.usage_history is True
    metrics.record_request_complete(100, 20, 60, 0.5, 1.0, "canonical-model", 2.0)
    metrics.usage_history.flush()
    data = client.get("/admin/api/usage").json()
    assert data["enabled"] is True
    assert data["totals"]["requests"] == 2

    untouched = asyncio.run(
        admin_routes.update_global_settings(
            request=admin_routes.GlobalSettingsRequest(), is_admin=True
        )
    )
    assert "usage_history" not in untouched["runtime_applied"]
    assert gs.usage.usage_history is True


def test_dashboard_renders_usage_history_switch_and_disabled_notice(client):
    client, _ = client
    html = client.get("/admin/dashboard").text
    assert "globalSettings.usage.usage_history" in html
    assert "Record usage history" in html
    assert "Usage history is off. Turn it on in Settings" in html
    assert "setSettingsTab('global')" in html
    javascript = (ROOT / "omlx/admin/static/js/dashboard.js").read_text(
        encoding="utf-8"
    )
    assert "usage: { usage_history: true }" in javascript
    assert "usage_history: this.globalSettings.usage.usage_history" in javascript


def test_usage_history_i18n_keys_present_in_every_locale():
    locales = sorted(I18N_DIR.glob("*.json"))
    assert len(locales) == 9
    for locale_path in locales:
        locale = json.loads(locale_path.read_text(encoding="utf-8"))
        missing = {key for key in USAGE_HISTORY_I18N_KEYS if not locale.get(key)}
        assert not missing, f"{locale_path.name}: missing {sorted(missing)}"
