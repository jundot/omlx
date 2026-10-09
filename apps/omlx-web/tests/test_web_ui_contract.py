# SPDX-License-Identifier: Apache-2.0
"""HTTP contract for the browser admin pages and the session flows they use."""

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from omlx_web import routes as webui

from omlx import server
from omlx._version import __version__
from omlx.admin import auth
from omlx.admin import routes as admin_routes
from omlx.settings import GlobalSettings, SubKeyEntry

MAIN_KEY = "main-key-1234"
SUB_KEY = "sub-key-5678"
HTML = {"Accept": "text/html"}
JSON = {"Accept": "application/json"}


@pytest.fixture
def web(monkeypatch, tmp_path):
    settings = GlobalSettings(base_path=tmp_path)
    settings.server.host = "127.0.0.1"
    settings.auth.api_key = MAIN_KEY
    settings.auth.sub_keys = [SubKeyEntry(key=SUB_KEY, name="sub")]
    monkeypatch.setattr(server._server_state, "global_settings", settings)
    monkeypatch.setattr(server._server_state, "api_key", MAIN_KEY)
    monkeypatch.setattr(server._server_state, "bind_host", "127.0.0.1")
    monkeypatch.setattr(auth, "_get_global_settings", lambda: settings)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: settings)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: server._server_state)
    return TestClient(server.app, follow_redirects=False), settings


def _login(client, remember=False):
    resp = client.post(
        "/admin/api/login", json={"api_key": MAIN_KEY, "remember": remember}
    )
    assert resp.status_code == 200, resp.text
    return resp


class TestLoginPage:
    @pytest.mark.parametrize("path", ["/admin", "/admin/"])
    def test_login_form_when_key_configured(self, web, path):
        client, _ = web
        resp = client.get(path, headers=HTML)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "loginForm(true)" in resp.text

    def test_setup_form_without_key(self, web):
        client, settings = web
        settings.auth.api_key = None
        resp = client.get("/admin", headers=HTML)
        assert resp.status_code == 200
        assert "loginForm(false)" in resp.text

    def test_session_redirects_to_dashboard(self, web):
        client, _ = web
        _login(client)
        resp = client.get("/admin", headers=HTML)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/admin/dashboard"

    def test_skip_auth_on_loopback_redirects_to_dashboard(self, web):
        client, settings = web
        settings.auth.skip_api_key_verification = True
        resp = client.get("/admin", headers=HTML)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/admin/dashboard"

    def test_skip_auth_on_network_bind_shows_login(self, web, monkeypatch):
        client, settings = web
        settings.auth.skip_api_key_verification = True
        monkeypatch.setattr(server._server_state, "bind_host", "0.0.0.0")
        resp = client.get("/admin", headers=HTML)
        assert resp.status_code == 200
        assert "loginForm(true)" in resp.text


class TestProtectedPages:
    @pytest.mark.parametrize("path", ["/admin/dashboard", "/admin/chat"])
    def test_browser_without_session_redirects_to_login(self, web, path):
        client, _ = web
        resp = client.get(path, headers=HTML)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/admin"

    @pytest.mark.parametrize("path", ["/admin/dashboard", "/admin/chat"])
    def test_json_without_session_is_401(self, web, path):
        client, _ = web
        resp = client.get(path, headers=JSON)
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Admin authentication required"}

    def test_admin_api_html_without_session_redirects_to_login(self, web):
        client, _ = web
        resp = client.get("/admin/api/global-settings", headers=HTML)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/admin"

    def test_dashboard_renders_locale_version_and_static_urls(self, web):
        client, _ = web
        _login(client)
        resp = client.get("/admin/dashboard", headers=HTML)
        assert resp.status_code == 200
        assert '<html lang="en">' in resp.text
        assert "window._t = {" in resp.text
        assert "Dashboard - oMLX Admin" in resp.text
        assert __version__ in resp.text
        assert "/admin/static/js/dashboard.js?v=" in resp.text

    def test_chat_injects_main_key(self, web):
        client, _ = web
        _login(client)
        resp = client.get("/admin/chat", headers=HTML)
        assert resp.status_code == 200
        assert f'const serverApiKey = "{MAIN_KEY}";' in resp.text

    def test_language_change_applies_to_next_render(self, web):
        client, _ = web
        _login(client)
        try:
            resp = client.post("/admin/api/global-settings", json={"ui_language": "ko"})
            assert resp.status_code == 200, resp.text
            page = client.get("/admin/dashboard", headers=HTML).text
            assert '<html lang="ko">' in page
            assert "대시보드 - oMLX 관리자" in page
        finally:
            client.post("/admin/api/global-settings", json={"ui_language": "en"})
        page = client.get("/admin/dashboard", headers=HTML).text
        assert '<html lang="en">' in page


class TestStaticFiles:
    @pytest.mark.parametrize(
        "path,media_type",
        [
            ("js/dashboard.js", "application/javascript"),
            ("css/tailwind.css", "text/css"),
            ("favicon.svg", "image/svg+xml"),
            ("omlx_preset.json", "application/octet-stream"),
        ],
    )
    def test_media_types(self, web, path, media_type):
        client, _ = web
        resp = client.get(f"/admin/static/{path}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(media_type)

    @pytest.mark.parametrize(
        "path", ["missing.js", "..%2f..%2fsettings.py", "%2e%2e/routes.py"]
    )
    def test_missing_or_traversal_is_404(self, web, path):
        client, _ = web
        assert client.get(f"/admin/static/{path}").status_code == 404


class TestSessionFlows:
    def test_login_cookie_attributes(self, web):
        client, _ = web
        cookie = _login(client).headers["set-cookie"]
        assert "omlx_admin_session=" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=lax" in cookie
        assert "Max-Age=86400" in cookie

    def test_remember_login_extends_cookie(self, web):
        client, _ = web
        cookie = _login(client, remember=True).headers["set-cookie"]
        assert "Max-Age=2592000" in cookie

    def test_auto_login_with_main_key(self, web):
        client, _ = web
        resp = client.get(
            "/admin/auto-login",
            params={"key": MAIN_KEY, "redirect": "/admin/chat"},
        )
        assert resp.status_code == 302
        assert resp.headers["location"] == "/admin/chat"
        assert "omlx_admin_session=" in resp.headers["set-cookie"]

    @pytest.mark.parametrize("key", [SUB_KEY, "wrong-key", ""])
    def test_auto_login_rejects_other_keys(self, web, key):
        client, _ = web
        resp = client.get("/admin/auto-login", params={"key": key})
        assert resp.status_code == 302
        assert resp.headers["location"] == "/admin"
        assert "set-cookie" not in resp.headers

    def test_auto_login_rejects_foreign_redirect(self, web):
        client, _ = web
        resp = client.get(
            "/admin/auto-login",
            params={"key": MAIN_KEY, "redirect": "https://example.com"},
        )
        assert resp.status_code == 400


def test_root_is_not_routed(web):
    client, _ = web
    assert client.get("/").status_code == 404


def _imports_omlx(module: str) -> bool:
    return module == "omlx" or module.startswith("omlx.")


def test_web_ui_package_does_not_import_omlx():
    for source in Path(webui.__file__).parent.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Relative imports must stay inside the package.
                assert node.level <= 1, f"{source.name}: {ast.unparse(node)}"
                if node.level == 0:
                    assert not _imports_omlx(node.module or ""), ast.unparse(node)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not _imports_omlx(alias.name), ast.unparse(node)
