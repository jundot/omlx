# SPDX-License-Identifier: Apache-2.0
"""Tests for admin authentication and chat page API key injection."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.server  # noqa: F401 — ensure server module is imported first
import omlx.admin.auth as admin_auth
import omlx.admin.routes as admin_routes


def _mock_global_settings(api_key=None):
    """Create a mock GlobalSettings with the given API key."""
    mock = MagicMock()
    mock.auth.api_key = api_key
    return mock


def _patch_getter(mock_settings):
    """Replace the module-level _get_global_settings with a lambda returning mock."""
    original = admin_routes._get_global_settings
    admin_routes._get_global_settings = lambda: mock_settings
    return original


def _restore_getter(original):
    """Restore the original _get_global_settings."""
    admin_routes._get_global_settings = original


class TestAutoLogin:
    """Tests for GET /admin/auto-login endpoint."""

    def test_auto_login_success_redirects_to_dashboard(self):
        """Valid API key should redirect to the specified path with session cookie."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="test-key", redirect="/admin/dashboard")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin/dashboard"
            # Check that session cookie is set
            cookie_header = result.headers.get("set-cookie", "")
            assert "omlx_admin_session" in cookie_header
        finally:
            _restore_getter(original)

    def test_auto_login_success_redirects_to_chat(self):
        """Valid API key should redirect to chat page."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="test-key", redirect="/admin/chat")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin/chat"
        finally:
            _restore_getter(original)

    def test_auto_login_invalid_key_redirects_to_login(self):
        """Invalid API key should redirect to login page without session cookie."""
        mock_settings = _mock_global_settings(api_key="correct-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="wrong-key", redirect="/admin/dashboard")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
            cookie_header = result.headers.get("set-cookie", "")
            assert "omlx_admin_session" not in cookie_header
        finally:
            _restore_getter(original)

    def test_auto_login_empty_key_redirects_to_login(self):
        """Empty API key should redirect to login page."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="", redirect="/admin/dashboard")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter(original)

    def test_auto_login_no_server_key_redirects_to_login(self):
        """No server API key configured should redirect to login page."""
        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="any-key", redirect="/admin/dashboard")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter(original)

    def test_auto_login_invalid_redirect_returns_400(self):
        """Redirect path not starting with /admin should return 400."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.auto_login(
                        key="test-key", redirect="https://evil.com"
                    )
                )
            assert exc_info.value.status_code == 400
            assert "Invalid redirect path" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_auto_login_redirect_to_admin_root(self):
        """Redirect to /admin (exact match) should be allowed."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="test-key", redirect="/admin")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter(original)


class TestChatPageApiKeyInjection:
    """Tests for GET /admin/chat API key template injection."""

    def test_chat_page_passes_api_key_in_context(self):
        """Chat page should include API key in template context."""
        mock_settings = _mock_global_settings(api_key="test-chat-key")
        original = _patch_getter(mock_settings)
        try:
            mock_request = MagicMock()
            with patch.object(admin_routes, "templates") as mock_templates:
                mock_templates.TemplateResponse.return_value = MagicMock()
                asyncio.run(
                    admin_routes.chat_page(request=mock_request, is_admin=True)
                )
                mock_templates.TemplateResponse.assert_called_once_with(
                    "chat.html",
                    {"request": mock_request, "api_key": "test-chat-key"},
                )
        finally:
            _restore_getter(original)

    def test_chat_page_passes_empty_when_no_key(self):
        """Chat page should pass empty string when no API key is configured."""
        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            mock_request = MagicMock()
            with patch.object(admin_routes, "templates") as mock_templates:
                mock_templates.TemplateResponse.return_value = MagicMock()
                asyncio.run(
                    admin_routes.chat_page(request=mock_request, is_admin=True)
                )
                call_args = mock_templates.TemplateResponse.call_args
                context = call_args[0][1]
                assert context["api_key"] == ""
        finally:
            _restore_getter(original)

    def test_chat_page_passes_empty_when_no_settings(self):
        """Chat page should pass empty string when global settings is None."""
        original = admin_routes._get_global_settings
        admin_routes._get_global_settings = lambda: None
        try:
            mock_request = MagicMock()
            with patch.object(admin_routes, "templates") as mock_templates:
                mock_templates.TemplateResponse.return_value = MagicMock()
                asyncio.run(
                    admin_routes.chat_page(request=mock_request, is_admin=True)
                )
                call_args = mock_templates.TemplateResponse.call_args
                context = call_args[0][1]
                assert context["api_key"] == ""
        finally:
            admin_routes._get_global_settings = original


class TestInitAuth:
    """Tests for init_auth() persistent secret key initialization."""

    def test_init_auth_sets_serializer(self):
        """init_auth should update the serializer with the provided key."""
        original_serializer = admin_auth._serializer
        try:
            admin_auth.init_auth("test-persistent-secret-key")
            # Create a token with the new serializer
            token = admin_auth.create_session_token()
            assert admin_auth.verify_session_token(token) is True
        finally:
            admin_auth._serializer = original_serializer

    def test_init_auth_env_var_takes_priority(self):
        """OMLX_SECRET_KEY env var should take priority over provided key."""
        original_serializer = admin_auth._serializer
        original_secret = admin_auth.SECRET_KEY
        try:
            with patch.dict("os.environ", {"OMLX_SECRET_KEY": "env-secret-key"}):
                admin_auth.init_auth("settings-secret-key")
                assert admin_auth.SECRET_KEY == "env-secret-key"
        finally:
            admin_auth._serializer = original_serializer
            admin_auth.SECRET_KEY = original_secret

    def test_init_auth_uses_provided_key_when_no_env(self):
        """Should use provided key when no OMLX_SECRET_KEY env var."""
        original_serializer = admin_auth._serializer
        original_secret = admin_auth.SECRET_KEY
        try:
            with patch.dict("os.environ", {}, clear=True):
                # Remove OMLX_SECRET_KEY if it exists
                import os

                os.environ.pop("OMLX_SECRET_KEY", None)
                admin_auth.init_auth("my-persistent-key")
                assert admin_auth.SECRET_KEY == "my-persistent-key"
        finally:
            admin_auth._serializer = original_serializer
            admin_auth.SECRET_KEY = original_secret

    def test_tokens_survive_reinit_with_same_key(self):
        """Tokens created before re-init should still be valid with same key."""
        original_serializer = admin_auth._serializer
        original_secret = admin_auth.SECRET_KEY
        try:
            key = "persistent-key-for-test"
            admin_auth.init_auth(key)
            token = admin_auth.create_session_token()

            # Re-initialize with same key (simulates server restart)
            admin_auth.init_auth(key)
            assert admin_auth.verify_session_token(token) is True
        finally:
            admin_auth._serializer = original_serializer
            admin_auth.SECRET_KEY = original_secret

    def test_tokens_invalid_after_reinit_with_different_key(self):
        """Tokens should be invalid after re-init with a different key."""
        original_serializer = admin_auth._serializer
        original_secret = admin_auth.SECRET_KEY
        try:
            admin_auth.init_auth("key-one")
            token = admin_auth.create_session_token()

            admin_auth.init_auth("key-two")
            assert admin_auth.verify_session_token(token) is False
        finally:
            admin_auth._serializer = original_serializer
            admin_auth.SECRET_KEY = original_secret


class TestRememberMe:
    """Tests for remember me session token functionality."""

    def test_create_token_default_no_remember(self):
        """Default token should not have remember flag."""
        token = admin_auth.create_session_token()
        # Verify it works with default max_age
        assert admin_auth.verify_session_token(token) is True

    def test_create_token_with_remember(self):
        """Token with remember=True should be valid."""
        token = admin_auth.create_session_token(remember=True)
        assert admin_auth.verify_session_token(token) is True

    def test_remember_token_has_extended_max_age(self):
        """Remember token should use 30-day max_age for verification."""
        token = admin_auth.create_session_token(remember=True)
        # Manually load the payload to check the remember flag
        data = admin_auth._serializer.loads(token, max_age=None)
        assert data["remember"] is True
        assert data["admin"] is True

    def test_non_remember_token_payload(self):
        """Non-remember token should have remember=False in payload."""
        token = admin_auth.create_session_token(remember=False)
        data = admin_auth._serializer.loads(token, max_age=None)
        assert data["remember"] is False
        assert data["admin"] is True

    def test_remember_me_max_age_constant(self):
        """REMEMBER_ME_MAX_AGE should be 30 days."""
        assert admin_auth.REMEMBER_ME_MAX_AGE == 2592000  # 30 * 24 * 60 * 60

    def test_session_max_age_constant(self):
        """SESSION_MAX_AGE should be 24 hours."""
        assert admin_auth.SESSION_MAX_AGE == 86400  # 24 * 60 * 60


# =============================================================================
# Update Check
# =============================================================================


def _make_async_return(value):
    """Create a coroutine function that returns the given value."""

    async def _coro(*args, **kwargs):
        return value

    return _coro


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json


class TestCheckUpdate:
    """Tests for update-check version filtering."""

    def setup_method(self):
        admin_routes._update_cache = None
        admin_routes._update_cache_time = 0.0

    @pytest.mark.asyncio
    async def test_prerelease_not_shown(self):
        """Dev/pre-release GitHub releases should not trigger update notification."""
        fake_resp = _FakeResponse(
            200,
            {
                "tag_name": "v99.0.0.dev1",
                "html_url": "https://github.com/jundot/omlx/releases/tag/v99.0.0.dev1",
            },
        )
        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _make_async_return(fake_resp)
            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is False
        assert result["latest_version"] is None

    @pytest.mark.asyncio
    async def test_stable_version_shown(self):
        """Stable GitHub releases should trigger update notification."""
        fake_resp = _FakeResponse(
            200,
            {
                "tag_name": "v99.0.0",
                "html_url": "https://github.com/jundot/omlx/releases/tag/v99.0.0",
            },
        )
        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _make_async_return(fake_resp)
            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is True
        assert result["latest_version"] == "99.0.0"

    @pytest.mark.asyncio
    async def test_rc_not_shown(self):
        """RC releases should not trigger update notification."""
        fake_resp = _FakeResponse(
            200,
            {
                "tag_name": "v99.0.0rc1",
                "html_url": "https://github.com/jundot/omlx/releases/tag/v99.0.0rc1",
            },
        )
        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _make_async_return(fake_resp)
            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is False
        assert result["latest_version"] is None
