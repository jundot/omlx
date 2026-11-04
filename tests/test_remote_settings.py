# SPDX-License-Identifier: Apache-2.0
"""Remote-backend settings stay client-side: never exposed through the admin API."""

import asyncio
import json

import pytest

from omlx.admin import routes as admin_routes
from omlx.settings import GlobalSettings, RemoteSettings

SECRET = "remote-secret-key"


def _write_settings(base_path, **extra):
    base_path.mkdir(parents=True, exist_ok=True)
    data = {
        "version": "1.0",
        "remote": {
            "enabled": True,
            "url": "https://studio.local:8443",
            "api_key": SECRET,
        },
        **extra,
    }
    (base_path / "settings.json").write_text(json.dumps(data))


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OMLX_REMOTE_URL", raising=False)
    monkeypatch.delenv("OMLX_REMOTE_API_KEY", raising=False)


def test_remote_section_survives_load_save_round_trip(tmp_path):
    _write_settings(tmp_path)
    gs = GlobalSettings.load(base_path=tmp_path)
    assert gs.remote == RemoteSettings(
        enabled=True, url="https://studio.local:8443", api_key=SECRET
    )
    gs.server.port = 8123  # unrelated change, then save
    gs.save()
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["remote"] == {
        "enabled": True,
        "url": "https://studio.local:8443",
        "api_key": SECRET,
    }
    assert GlobalSettings.load(base_path=tmp_path).remote == gs.remote


def test_global_to_dict_excludes_remote(tmp_path):
    _write_settings(tmp_path)
    gs = GlobalSettings.load(base_path=tmp_path)
    assert "remote" not in gs.to_dict()


def test_admin_get_global_settings_excludes_remote(tmp_path, monkeypatch):
    _write_settings(tmp_path)
    gs = GlobalSettings.load(base_path=tmp_path)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)

    payload = asyncio.run(admin_routes.get_global_settings(is_admin=True))

    assert "remote" not in payload
    assert SECRET not in json.dumps(payload, default=str)
    assert "studio.local" not in json.dumps(payload, default=str)


def test_admin_defaults_payload_excludes_remote(tmp_path, monkeypatch):
    _write_settings(tmp_path)
    gs = GlobalSettings.load(base_path=tmp_path)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)

    payload = asyncio.run(admin_routes.get_global_settings_defaults(is_admin=True))

    assert "remote" not in payload


def test_admin_request_model_has_no_remote_fields():
    fields = admin_routes.GlobalSettingsRequest.model_fields
    assert not [name for name in fields if name.startswith("remote")]


def test_admin_post_cannot_touch_remote_and_preserves_it(tmp_path, monkeypatch):
    _write_settings(tmp_path)
    gs = GlobalSettings.load(base_path=tmp_path)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: gs)

    # Unknown/extra keys such as "remote" are not part of the request schema.
    request = admin_routes.GlobalSettingsRequest.model_validate(
        {"remote": {"enabled": False, "url": "evil.local"}, "log_level": "debug"}
    )
    assert not hasattr(request, "remote")
    result = asyncio.run(
        admin_routes.update_global_settings(request=request, is_admin=True)
    )
    assert result["success"] is True

    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["remote"]["enabled"] is True
    assert saved["remote"]["url"] == "https://studio.local:8443"
    assert saved["remote"]["api_key"] == SECRET


# --- integrations: scheme-aware context and gateway URL ---------------------


def test_integration_context_base_url_uses_scheme():
    from omlx.integrations import IntegrationContext

    assert IntegrationContext(host="127.0.0.1", port=8000).base_url == (
        "http://127.0.0.1:8000"
    )
    ctx = IntegrationContext(host="studio.local", port=8443, scheme="https")
    assert ctx.base_url == "https://studio.local:8443"
    assert ctx.openai_base_url == "https://studio.local:8443/v1"
    v6 = IntegrationContext(host="::1", port=8000)
    assert v6.base_url == "http://[::1]:8000"


def test_bracket_host_still_importable_from_integrations_base():
    from omlx.integrations.base import bracket_host

    assert bracket_host("::1") == "[::1]"
    assert bracket_host("studio.local") == "studio.local"


def test_configure_omlx_gateway_base_url_override(tmp_path, monkeypatch):
    from omlx.integrations import claude_desktop

    monkeypatch.setattr(claude_desktop, "_is_macos", lambda: True)
    paths = claude_desktop._paths(tmp_path)

    assert claude_desktop.configure_omlx_gateway(8000, "k", home=tmp_path) is True
    profile = json.loads(paths["profile"].read_text())
    assert profile["inferenceGatewayBaseUrl"] == "http://127.0.0.1:8000"

    claude_desktop.configure_omlx_gateway(
        8443, "k", home=tmp_path, base_url="https://studio.local:8443"
    )
    profile = json.loads(paths["profile"].read_text())
    assert profile["inferenceGatewayBaseUrl"] == "https://studio.local:8443"


def test_claude_desktop_app_passes_ctx_base_url(monkeypatch):
    from omlx.integrations import IntegrationContext, claude_desktop_app

    calls = []
    monkeypatch.setattr(
        claude_desktop_app,
        "configure_omlx_gateway",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    ctx = IntegrationContext(
        host="studio.local", port=8443, api_key="k", scheme="https", is_remote=True
    )
    claude_desktop_app.ClaudeDesktopAppIntegration().configure(ctx)
    assert calls == [
        {"port": 8443, "api_key": "k", "base_url": "https://studio.local:8443"}
    ]
    # A local server keeps the loopback gateway URL that is_configured() expects.
    calls.clear()
    claude_desktop_app.ClaudeDesktopAppIntegration().configure(
        IntegrationContext(host="localhost", port=8000, api_key="k")
    )
    assert calls == [{"port": 8000, "api_key": "k", "base_url": None}]
