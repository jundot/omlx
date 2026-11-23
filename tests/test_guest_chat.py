"""Anonymous (guest) chat: who can open it, and with which credential.

Guest chat serves the normal chat page to someone with no admin session.
The whole feature rests on one substitution: the page is handed
``auth.guest_chat_key`` instead of ``auth.api_key``. Get that wrong and an
anonymous visitor walks away holding the credential that opens the admin
panel, so the key-substitution tests here are the load-bearing ones.

The rest pins the surrounding contract: the route is absent (404, not a
login redirect) until an operator enables it, enabling always mints a key
so an enabled flag can never mean "no credential needed", rotation
invalidates open pages, and nothing about this widens /admin access.
"""

import importlib.util

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.admin.auth import SESSION_COOKIE_NAME, create_session_token
from omlx.settings import AuthSettings, SubKeyEntry

requires_mlx = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="omlx.server imports mlx; covered in CI",
)

MAIN_KEY = "main-key-must-never-reach-a-guest"
GUEST_KEY = "guest-key-abc123"


class _FakeSettings:
    """Stands in for GlobalSettings: just the auth block plus save/validate."""

    def __init__(self, auth: AuthSettings):
        self.auth = auth
        self.saves = 0
        self.save_error: Exception | None = None
        self.validation_errors: list[str] = []

    def validate(self) -> list[str]:
        return list(self.validation_errors)

    def save(self) -> None:
        if self.save_error is not None:
            raise self.save_error
        self.saves += 1


def _settings(**kwargs) -> _FakeSettings:
    auth = AuthSettings(api_key=MAIN_KEY, **kwargs)
    return _FakeSettings(auth)


@pytest.fixture
def client(monkeypatch):
    """Admin router on a bare app, with settings swapped per test."""
    app = FastAPI()
    app.include_router(admin_routes.router)
    holder: dict[str, _FakeSettings | None] = {"settings": None}

    monkeypatch.setattr(
        admin_routes, "_get_global_settings", lambda: holder["settings"]
    )

    with TestClient(app) as test_client:
        test_client.use_settings = lambda s: holder.update(settings=s)
        yield test_client


# ---------------------------------------------------------------------------
# The credential itself
# ---------------------------------------------------------------------------


class TestGuestCredential:
    def test_off_by_default(self):
        assert AuthSettings().guest_chat_enabled is False
        assert AuthSettings().guest_chat_credential() is None

    def test_key_without_enabled_flag_is_not_live(self):
        # Disabling must revoke access even though the key is retained so
        # that re-enabling resumes the same URL.
        auth = AuthSettings(guest_chat_enabled=False, guest_chat_key=GUEST_KEY)
        assert auth.guest_chat_credential() is None

    def test_enabled_without_key_is_not_live(self):
        # A flag with no credential behind it must never read as "open".
        auth = AuthSettings(guest_chat_enabled=True, guest_chat_key=None)
        assert auth.guest_chat_credential() is None

    def test_blank_key_is_not_live(self):
        auth = AuthSettings(guest_chat_enabled=True, guest_chat_key="   ")
        assert auth.guest_chat_credential() is None

    def test_enabled_with_key(self):
        auth = AuthSettings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        assert auth.guest_chat_credential() == GUEST_KEY

    def test_round_trip_through_settings_file(self):
        auth = AuthSettings(
            api_key=MAIN_KEY, guest_chat_enabled=True, guest_chat_key=GUEST_KEY
        )
        restored = AuthSettings.from_dict(auth.to_dict())
        assert restored.guest_chat_enabled is True
        assert restored.guest_chat_key == GUEST_KEY
        assert restored.guest_chat_credential() == GUEST_KEY

    def test_absent_from_older_settings_file(self):
        restored = AuthSettings.from_dict({"api_key": MAIN_KEY})
        assert restored.guest_chat_enabled is False
        assert restored.guest_chat_key is None


# ---------------------------------------------------------------------------
# The guest page
# ---------------------------------------------------------------------------


class TestGuestChatPage:
    def test_absent_when_disabled(self, client):
        client.use_settings(_settings())
        resp = client.get("/admin/chat/guest", follow_redirects=False)
        # 404 rather than a login redirect: a server that never enabled
        # guest chat should not advertise that the feature exists.
        assert resp.status_code == 404

    def test_absent_when_enabled_without_key(self, client):
        client.use_settings(_settings(guest_chat_enabled=True))
        assert client.get("/admin/chat/guest").status_code == 404

    def test_absent_when_server_uninitialized(self, client):
        client.use_settings(None)
        assert client.get("/admin/chat/guest").status_code == 404

    def test_served_without_any_session_cookie(self, client):
        client.use_settings(
            _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        )
        resp = client.get("/admin/chat/guest")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_page_carries_the_guest_key_not_the_main_key(self, client):
        client.use_settings(
            _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        )
        body = client.get("/admin/chat/guest").text

        assert GUEST_KEY in body
        # The one invariant that matters most.
        assert MAIN_KEY not in body

    def test_page_knows_it_is_in_guest_mode(self, client):
        client.use_settings(
            _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        )
        body = client.get("/admin/chat/guest").text

        assert "const GUEST_MODE = true" in body
        # Admin endpoints are routed through the guard that skips them.
        assert "fetch('/admin/api/" not in body
        assert "adminFetch('/admin/api/" in body

    def test_guest_page_does_not_link_into_the_admin_panel(self, client):
        client.use_settings(
            _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        )
        body = client.get("/admin/chat/guest").text
        assert 'href="/admin/dashboard"' not in body

    def test_admin_page_is_unchanged(self, client):
        # The operator's own chat page keeps the main key and the dashboard
        # link; guest mode must not have altered it.
        settings = _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        client.use_settings(settings)
        token = create_session_token()
        client.cookies.set(SESSION_COOKIE_NAME, token)

        body = client.get("/admin/chat").text

        assert MAIN_KEY in body
        assert "const GUEST_MODE = false" in body
        assert 'href="/admin/dashboard"' in body
        client.cookies.clear()


# ---------------------------------------------------------------------------
# Operator controls
# ---------------------------------------------------------------------------


class TestGuestChatControls:
    def test_controls_require_admin(self, client):
        client.use_settings(_settings())

        assert client.get("/admin/api/guest-chat").status_code == 401
        assert (
            client.post("/admin/api/guest-chat", json={"enabled": True}).status_code
            == 401
        )

    def _as_admin(self, client):
        client.cookies.set(
            SESSION_COOKIE_NAME, create_session_token()
        )

    def test_enabling_mints_a_key(self, client):
        settings = _settings()
        client.use_settings(settings)
        self._as_admin(client)

        resp = client.post("/admin/api/guest-chat", json={"enabled": True})

        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        minted = body["guest_chat_key"]
        assert len(minted) >= 20
        assert minted != MAIN_KEY
        assert settings.auth.guest_chat_credential() == minted
        assert settings.saves == 1

    def test_enabling_keeps_an_existing_key(self, client):
        settings = _settings(guest_chat_key=GUEST_KEY)
        client.use_settings(settings)
        self._as_admin(client)

        resp = client.post("/admin/api/guest-chat", json={"enabled": True})

        # Re-enabling resumes the same link rather than invalidating one the
        # operator has already shared.
        assert resp.json()["guest_chat_key"] == GUEST_KEY

    def test_disabling_retains_the_key(self, client):
        settings = _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        client.use_settings(settings)
        self._as_admin(client)

        resp = client.post("/admin/api/guest-chat", json={"enabled": False})

        assert resp.json()["enabled"] is False
        assert settings.auth.guest_chat_key == GUEST_KEY
        assert settings.auth.guest_chat_credential() is None

    def test_rotation_invalidates_the_old_key(self, client):
        settings = _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        client.use_settings(settings)
        self._as_admin(client)

        resp = client.post("/admin/api/guest-chat", json={"rotate": True})

        rotated = resp.json()["guest_chat_key"]
        assert rotated != GUEST_KEY
        assert settings.auth.guest_chat_credential() == rotated
        assert resp.json()["enabled"] is True

    def test_failed_save_rolls_back(self, client):
        settings = _settings()
        settings.save_error = OSError("disk full")
        client.use_settings(settings)
        self._as_admin(client)

        resp = client.post("/admin/api/guest-chat", json={"enabled": True})

        assert resp.status_code == 500
        # Nothing may be live that was not persisted.
        assert settings.auth.guest_chat_enabled is False
        assert settings.auth.guest_chat_key is None
        assert settings.auth.guest_chat_credential() is None

    def test_rejected_validation_rolls_back(self, client):
        settings = _settings()
        settings.validation_errors = ["nope"]
        client.use_settings(settings)
        self._as_admin(client)

        resp = client.post("/admin/api/guest-chat", json={"enabled": True})

        assert resp.status_code == 400
        assert settings.auth.guest_chat_enabled is False
        assert settings.auth.guest_chat_key is None

    def test_status_reports_state_without_leaking_the_key(self, client):
        settings = _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        client.use_settings(settings)
        self._as_admin(client)

        body = client.get("/admin/api/guest-chat").json()

        assert body["enabled"] is True
        assert body["key_set"] is True
        assert body["path"] == "/chat"
        assert GUEST_KEY not in str(body)


# ---------------------------------------------------------------------------
# What the guest key may authenticate
# ---------------------------------------------------------------------------


class TestGuestChatUi:
    """The dashboard switch that turns this on, and its strings."""

    I18N_KEYS = [
        "settings.auth.guest_chat",
        "settings.auth.guest_chat_hint",
        "settings.auth.guest_chat_warning",
        "settings.auth.guest_chat_link",
        "settings.auth.guest_chat_copy",
        "settings.auth.guest_chat_copied",
        "settings.auth.guest_chat_rotate",
        "settings.auth.guest_chat_rotate_confirm",
    ]

    @staticmethod
    def _repo_root():
        from pathlib import Path

        return Path(__file__).resolve().parents[1]

    @pytest.mark.parametrize(
        "lang", ["en", "es", "fr", "ja", "ko", "pt-BR", "ru", "zh", "zh-TW"]
    )
    def test_every_locale_has_the_strings(self, lang):
        # A missing key renders as the key itself, so the switch would ship
        # reading "settings.auth.guest_chat" in that language.
        import json

        locale = json.loads(
            (self._repo_root() / f"omlx/admin/i18n/{lang}.json").read_text(
                encoding="utf-8"
            )
        )
        missing = [k for k in self.I18N_KEYS if k not in locale]
        assert missing == []
        assert all(str(locale[k]).strip() for k in self.I18N_KEYS)

    def test_settings_panel_is_wired_to_the_endpoint(self):
        markup = (
            self._repo_root() / "omlx/admin/templates/dashboard/_settings.html"
        ).read_text(encoding="utf-8")

        assert "toggleGuestChat()" in markup
        assert "rotateGuestChatKey()" in markup
        assert "guestChatUrl()" in markup
        assert "globalSettings.auth.guest_chat_enabled" in markup

    def test_dashboard_defines_the_handlers(self):
        script = (
            self._repo_root() / "omlx/admin/static/js/dashboard.js"
        ).read_text(encoding="utf-8")

        for handler in (
            "async toggleGuestChat()",
            "async rotateGuestChatKey()",
            "guestChatUrl()",
            "/admin/api/guest-chat",
        ):
            assert handler in script


class TestGuestKeyResolution:
    """The resolver the API verifier uses to decide if guest access is live.

    Tested here rather than only through verify_api_key() because these are
    the inputs that must never raise: importing omlx.server needs mlx, so
    this path would otherwise be exercised only on Apple silicon.
    """

    def test_uninitialized_settings(self):
        from omlx.admin.auth import resolve_guest_chat_key

        assert resolve_guest_chat_key(None) is None

    def test_live_key(self):
        from omlx.admin.auth import resolve_guest_chat_key

        settings = _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        assert resolve_guest_chat_key(settings) == GUEST_KEY

    def test_disabled(self):
        from omlx.admin.auth import resolve_guest_chat_key

        settings = _settings(guest_chat_enabled=False, guest_chat_key=GUEST_KEY)
        assert resolve_guest_chat_key(settings) is None

    def test_mock_settings_object_yields_no_access(self):
        # Existing auth tests substitute MagicMock settings. A truthy mock
        # attribute must not be taken for a key: it would reach compare_keys
        # and raise on every authenticated request.
        from unittest.mock import MagicMock

        from omlx.admin.auth import resolve_guest_chat_key

        assert resolve_guest_chat_key(MagicMock()) is None

    def test_settings_without_the_accessor(self):
        from omlx.admin.auth import resolve_guest_chat_key

        class Old:
            auth = object()

        assert resolve_guest_chat_key(Old()) is None

    @pytest.mark.parametrize("stored", [1, True, [], {}, object()])
    def test_non_string_key_in_settings_file(self, stored):
        from omlx.admin.auth import resolve_guest_chat_key

        settings = _settings(guest_chat_enabled=True)
        settings.auth.guest_chat_key = stored
        assert resolve_guest_chat_key(settings) is None

    def test_raising_accessor_is_contained(self):
        from omlx.admin.auth import resolve_guest_chat_key

        class Boom:
            class auth:  # noqa: N801
                @staticmethod
                def guest_chat_credential():
                    raise RuntimeError("settings unreadable")

        assert resolve_guest_chat_key(Boom()) is None


@requires_mlx
class TestGuestKeyScope:
    """The guest key opens the inference API and nothing else."""

    @staticmethod
    def _call(monkeypatch, *, provided, settings, main_key, host="0.0.0.0"):
        import asyncio

        from fastapi.security import HTTPAuthorizationCredentials

        from omlx import server as omlx_server

        monkeypatch.setattr(omlx_server._server_state, "api_key", main_key)
        monkeypatch.setattr(
            omlx_server._server_state, "global_settings", settings, raising=False
        )
        monkeypatch.setattr(
            omlx_server._server_state, "bind_host", host, raising=False
        )

        class _Req:
            headers: dict[str, str] = {}

        credentials = (
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=provided)
            if provided is not None
            else None
        )
        return asyncio.run(
            omlx_server.verify_api_key(request=_Req(), credentials=credentials)
        )

    def test_guest_key_authenticates_the_api(self, monkeypatch):
        settings = _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        assert (
            self._call(
                monkeypatch,
                provided=GUEST_KEY,
                settings=settings,
                main_key=MAIN_KEY,
            )
            is True
        )

    def test_guest_key_is_refused_once_disabled(self, monkeypatch):
        from fastapi import HTTPException

        settings = _settings(guest_chat_enabled=False, guest_chat_key=GUEST_KEY)
        with pytest.raises(HTTPException) as excinfo:
            self._call(
                monkeypatch,
                provided=GUEST_KEY,
                settings=settings,
                main_key=MAIN_KEY,
            )
        assert excinfo.value.status_code == 401

    def test_rotated_away_key_is_refused(self, monkeypatch):
        from fastapi import HTTPException

        settings = _settings(guest_chat_enabled=True, guest_chat_key="rotated-new")
        with pytest.raises(HTTPException):
            self._call(
                monkeypatch,
                provided=GUEST_KEY,
                settings=settings,
                main_key=MAIN_KEY,
            )

    def test_guest_key_works_when_no_main_key_is_configured(self, monkeypatch):
        # A LAN bind with no main key otherwise refuses everything; the
        # guest page's own requests must still authenticate.
        settings = _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        assert (
            self._call(
                monkeypatch, provided=GUEST_KEY, settings=settings, main_key=None
            )
            is True
        )

    def test_no_key_still_refused_on_a_lan_bind(self, monkeypatch):
        from fastapi import HTTPException

        settings = _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        with pytest.raises(HTTPException):
            self._call(
                monkeypatch, provided=None, settings=settings, main_key=None
            )

    def test_operator_sub_keys_still_work(self, monkeypatch):
        settings = _settings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        settings.auth.sub_keys = [SubKeyEntry(key="sub-key-1", name="laptop")]
        assert (
            self._call(
                monkeypatch,
                provided="sub-key-1",
                settings=settings,
                main_key=MAIN_KEY,
            )
            is True
        )

    def test_guest_key_is_not_stored_among_operator_sub_keys(self):
        # Revoking guest access must be one switch, never confused with the
        # operator's own keys.
        auth = AuthSettings(guest_chat_enabled=True, guest_chat_key=GUEST_KEY)
        assert auth.sub_keys == []
        assert GUEST_KEY not in [sk.key for sk in auth.sub_keys]
