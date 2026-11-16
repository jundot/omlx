# SPDX-License-Identifier: Apache-2.0
"""Secure Server mode: state reading, the 403 guard, and telemetry skipping.

The OS-level enforcement (pf, loopback only) is the privileged helper's job
and is verified on a real Mac. These tests cover the parts the Python server
owns: the read-only status flag, the egress denials that turn a dropped
packet into an actionable 403, and the telemetry guards.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from omlx import secure_mode
from omlx.admin.auth import require_admin
from omlx.server import app, verify_api_key


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    """A temp state file, wired in through the documented env override."""

    path = tmp_path / "state.json"
    monkeypatch.setenv(secure_mode.STATE_PATH_ENV, str(path))
    secure_mode.reset_cache()
    yield path
    secure_mode.reset_cache()


def _write(path, payload):
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    secure_mode.reset_cache()


class TestLayoutContract:
    """The helper writes these exact paths; the app and this server read them."""

    def test_production_paths(self):
        assert (
            secure_mode.CONFIG_DIR == "/Library/Application Support/oMLX/secure-server"
        )
        assert secure_mode.STATE_PATH == secure_mode.CONFIG_DIR + "/state.json"
        assert secure_mode.PF_ANCHOR_PATH == secure_mode.CONFIG_DIR + "/pf-anchor.conf"
        assert secure_mode.PF_ANCHOR == "com.apple/900.omlx.secure"

    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv(secure_mode.STATE_PATH_ENV, str(tmp_path / "s.json"))
        assert secure_mode.state_path() == str(tmp_path / "s.json")


class TestIsEnabled:
    def test_enabled_true(self, state_file):
        _write(state_file, {"enabled": True})
        assert secure_mode.is_enabled() is True

    def test_enabled_false(self, state_file):
        _write(state_file, {"enabled": False})
        assert secure_mode.is_enabled() is False

    def test_missing_file_reads_as_off(self, state_file):
        assert not state_file.exists()
        assert secure_mode.is_enabled() is False

    def test_malformed_json_reads_as_off(self, state_file):
        _write(state_file, "{not json")
        assert secure_mode.is_enabled() is False

    def test_picks_up_changes_without_cache_reset(self, state_file):
        _write(state_file, {"enabled": False})
        assert secure_mode.is_enabled() is False
        _write(state_file, {"enabled": True})
        assert secure_mode.is_enabled() is True


def _client():
    app.dependency_overrides[require_admin] = lambda: True
    app.dependency_overrides[verify_api_key] = lambda: True
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client():
    secure_mode.reset_cache()
    yield _client()
    app.dependency_overrides.clear()


EGRESS_CALLS = [
    ("POST", "/admin/api/hf/download"),
    ("POST", "/admin/api/hf/retry/task-1"),
    ("GET", "/admin/api/hf/recommended"),
    ("GET", "/admin/api/hf/search?q=qwen"),
    ("GET", "/admin/api/hf/model-info?repo_id=a/b"),
    ("POST", "/admin/api/ms/download"),
    ("POST", "/admin/api/ms/retry/task-1"),
    ("GET", "/admin/api/ms/recommended"),
    ("GET", "/admin/api/ms/search?q=qwen"),
    ("GET", "/admin/api/ms/model-info?model_id=a/b"),
    ("POST", "/admin/api/upload/start"),
    ("POST", "/admin/api/upload/validate-token"),
    ("POST", "/admin/api/presets/refresh"),
    ("POST", "/admin/api/web-search/test"),
    ("GET", "/admin/api/update-check"),
    ("POST", "/admin/api/oq/start"),
    ("POST", "/v1/web/search"),
    ("POST", "/v1/web/fetch"),
]


class TestEgressGuard:
    @pytest.mark.parametrize("method,path", EGRESS_CALLS)
    def test_egress_endpoints_are_refused(self, client, state_file, method, path):
        _write(state_file, {"enabled": True})
        response = client.request(method, path)
        assert response.status_code == 403
        assert response.json() == {"detail": secure_mode.SECURE_MODE_DETAIL}

    def test_download_allowed_when_mode_is_off(self, client, state_file):
        _write(state_file, {"enabled": False})
        response = client.post("/admin/api/hf/download", json={})
        assert response.status_code != 403
        assert response.json().get("detail") != secure_mode.SECURE_MODE_DETAIL

    def test_inference_is_not_gated(self, client, state_file):
        _write(state_file, {"enabled": True})
        response = client.post("/v1/chat/completions", json={})
        assert response.status_code != 403

    def test_health_is_not_gated(self, client, state_file):
        _write(state_file, {"enabled": True})
        assert client.get("/health").status_code != 403


class TestStatusField:
    def test_global_settings_exposes_read_only_flag(
        self, client, state_file, tmp_path, monkeypatch
    ):
        from omlx.admin import routes as admin_routes
        from omlx.settings import init_settings, reset_settings

        settings = init_settings(base_path=tmp_path)
        monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)
        try:
            _write(state_file, {"enabled": True})
            body = client.get("/admin/api/global-settings").json()
            assert body["secure_server"] == {"enabled": True}
        finally:
            reset_settings()

    def test_flag_is_absent_from_the_writable_schema(self):
        from omlx.admin.routes import GlobalSettingsRequest

        assert "secure_server" not in GlobalSettingsRequest.model_fields


class TestTelemetryGuards:
    def test_throughput_upload_skips_network(self, state_file):
        from omlx.admin import benchmark

        _write(state_file, {"enabled": True})
        run = MagicMock()
        with patch("requests.post") as post:
            import asyncio

            asyncio.run(benchmark._upload_to_omlx_ai(run, MagicMock()))
        post.assert_not_called()

    def test_accuracy_upload_skips_network(self, state_file):
        from omlx.admin import accuracy_upload

        _write(state_file, {"enabled": True})
        result = {"total": 500, "benchmark": "mmlu"}
        with patch("requests.post") as post:
            import asyncio

            outcome = asyncio.run(
                accuracy_upload.upload_intelligence_result(
                    MagicMock(), {"model_id": "m"}, result
                )
            )
        post.assert_not_called()
        assert outcome == {"skipped": "secure_server"}
