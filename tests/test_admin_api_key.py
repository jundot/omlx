# SPDX-License-Identifier: Apache-2.0
"""Tests for admin API key management (validation, setup, login, settings update)."""

import asyncio
from dataclasses import fields as dataclass_fields
from unittest.mock import MagicMock, patch

import pytest

from omlx.admin.auth import validate_api_key, verify_any_api_key, verify_api_key
from omlx.model_settings import ModelSettings
import omlx.server  # noqa: F401 — ensure server module is imported first (triggers set_admin_getters)
import omlx.admin.routes as admin_routes


class TestListModelsSettings:
    """Tests for list_models() settings completeness."""

    def test_list_models_includes_all_model_settings_fields(self):
        """Ensure list_models response includes all ModelSettings fields."""
        mock_engine_pool = MagicMock()
        mock_engine_pool.get_status.return_value = {
            "models": [
                {
                    "id": "test-model",
                    "loaded": True,
                    "estimated_size": 1000,
                    "pinned": False,
                    "engine_type": "batched",
                    "model_type": "llm",
                }
            ]
        }

        test_settings = ModelSettings(
            max_context_window=8192,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            presence_penalty=0.3,
            force_sampling=True,
            is_pinned=True,
            is_default=False,
            display_name="Test Model",
            description="A test model",
        )

        mock_settings_manager = MagicMock()
        mock_settings_manager.get_all_settings.return_value = {
            "test-model": test_settings
        }

        mock_server_state = MagicMock()
        mock_server_state.default_model = None

        with (
            patch.object(admin_routes, "_get_engine_pool", return_value=mock_engine_pool),
            patch.object(admin_routes, "_get_settings_manager", return_value=mock_settings_manager),
            patch.object(admin_routes, "_get_server_state", return_value=mock_server_state),
        ):
            result = asyncio.run(admin_routes.list_models(is_admin=True))

        model = result["models"][0]
        assert "settings" in model

        settings_dict = model["settings"]
        expected_fields = {f.name for f in dataclass_fields(ModelSettings)}
        actual_fields = set(settings_dict.keys())
        assert expected_fields == actual_fields, (
            f"Missing fields: {expected_fields - actual_fields}, "
            f"Extra fields: {actual_fields - expected_fields}"
        )

        # Verify specific values
        assert settings_dict["max_context_window"] == 8192
        assert settings_dict["max_tokens"] == 4096
        assert settings_dict["temperature"] == 0.7


class TestValidateApiKey:
    """Tests for validate_api_key() format validation."""

    def test_valid_key_simple(self):
        is_valid, msg = validate_api_key("abcd")
        assert is_valid is True
        assert msg == ""

    def test_valid_key_long(self):
        is_valid, msg = validate_api_key("sk-1234567890abcdef")
        assert is_valid is True

    def test_valid_key_special_chars(self):
        is_valid, msg = validate_api_key("a!@#$%^&*()-_=+[]{}|;:',.<>?/~`")
        assert is_valid is True

    def test_too_short_empty(self):
        is_valid, msg = validate_api_key("")
        assert is_valid is False
        assert "at least 4" in msg

    def test_too_short_one_char(self):
        is_valid, msg = validate_api_key("a")
        assert is_valid is False
        assert "at least 4" in msg

    def test_too_short_three_chars(self):
        is_valid, msg = validate_api_key("abc")
        assert is_valid is False
        assert "at least 4" in msg

    def test_exactly_four_chars(self):
        is_valid, msg = validate_api_key("abcd")
        assert is_valid is True

    def test_whitespace_space(self):
        is_valid, msg = validate_api_key("ab cd")
        assert is_valid is False
        assert "whitespace" in msg

    def test_whitespace_tab(self):
        is_valid, msg = validate_api_key("ab\tcd")
        assert is_valid is False
        assert "whitespace" in msg

    def test_whitespace_newline(self):
        is_valid, msg = validate_api_key("ab\ncd")
        assert is_valid is False
        assert "whitespace" in msg

    def test_whitespace_leading(self):
        is_valid, msg = validate_api_key(" abcd")
        assert is_valid is False
        assert "whitespace" in msg

    def test_whitespace_trailing(self):
        is_valid, msg = validate_api_key("abcd ")
        assert is_valid is False
        assert "whitespace" in msg

    def test_control_char_null(self):
        is_valid, msg = validate_api_key("ab\x00cd")
        assert is_valid is False
        assert "printable" in msg

    def test_control_char_bell(self):
        is_valid, msg = validate_api_key("ab\x07cd")
        assert is_valid is False
        assert "printable" in msg


class TestVerifyApiKeyAdmin:
    """Tests for verify_api_key() constant-time comparison."""

    def test_matching_keys(self):
        assert verify_api_key("secret123", "secret123") is True

    def test_non_matching_keys(self):
        assert verify_api_key("wrong", "secret123") is False

    def test_empty_api_key(self):
        assert verify_api_key("", "secret123") is False

    def test_empty_server_key(self):
        assert verify_api_key("secret123", "") is False

    def test_both_empty(self):
        assert verify_api_key("", "") is False


class TestVerifyAnyApiKey:
    """Tests for verify_any_api_key() checking main key + sub keys."""

    def test_matches_main_key(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1"), SubKeyEntry(key="sub2")]
        assert verify_any_api_key("main-key", "main-key", sub_keys) is True

    def test_matches_sub_key(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1"), SubKeyEntry(key="sub2")]
        assert verify_any_api_key("sub2", "main-key", sub_keys) is True

    def test_no_match(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1")]
        assert verify_any_api_key("wrong", "main-key", sub_keys) is False

    def test_empty_api_key(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1")]
        assert verify_any_api_key("", "main-key", sub_keys) is False

    def test_no_main_key_matches_sub(self):
        from omlx.settings import SubKeyEntry
        sub_keys = [SubKeyEntry(key="sub1")]
        assert verify_any_api_key("sub1", "", sub_keys) is True

    def test_empty_sub_keys(self):
        assert verify_any_api_key("main-key", "main-key", []) is True

    def test_no_match_empty_sub_keys(self):
        assert verify_any_api_key("wrong", "main-key", []) is False


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


class TestSetupApiKeyEndpoint:
    """Tests for POST /admin/api/setup-api-key endpoint logic."""

    def test_setup_rejects_when_key_already_set(self):
        """Setup should fail if API key is already configured."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key="existing-key")
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.SetupApiKeyRequest(
                api_key="newkey", api_key_confirm="newkey"
            )
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.setup_api_key(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "already configured" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_setup_rejects_mismatched_keys(self):
        """Setup should fail if api_key and api_key_confirm don't match."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.SetupApiKeyRequest(
                api_key="key1", api_key_confirm="key2"
            )
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.setup_api_key(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "do not match" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_setup_rejects_short_key(self):
        """Setup should fail if key is too short."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.SetupApiKeyRequest(
                api_key="abc", api_key_confirm="abc"
            )
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.setup_api_key(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "at least 4" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_setup_rejects_whitespace_key(self):
        """Setup should fail if key contains whitespace."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.SetupApiKeyRequest(
                api_key="ab cd", api_key_confirm="ab cd"
            )
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.setup_api_key(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "whitespace" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_setup_success_saves_key(self):
        """Successful setup should save key to settings and server state."""
        from unittest.mock import patch

        mock_settings = _mock_global_settings(api_key=None)
        mock_response = MagicMock()
        mock_server_state = MagicMock()
        mock_server_state.api_key = None

        original = _patch_getter(mock_settings)
        try:
            with patch("omlx.server._server_state", mock_server_state):
                request = admin_routes.SetupApiKeyRequest(
                    api_key="validkey123", api_key_confirm="validkey123"
                )
                result = asyncio.run(
                    admin_routes.setup_api_key(request, mock_response)
                )

                assert result["success"] is True
                assert mock_settings.auth.api_key == "validkey123"
                assert mock_server_state.api_key == "validkey123"
                mock_settings.save.assert_called_once()
                mock_response.set_cookie.assert_called_once()
        finally:
            _restore_getter(original)


class TestLoginEndpoint:
    """Tests for POST /admin/api/login endpoint logic."""

    def test_login_rejects_when_no_key_configured(self):
        """Login should fail with 400 when no API key is configured."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.LoginRequest(api_key="anykey")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.login(request, MagicMock()))
            assert exc_info.value.status_code == 400
            assert "No API key configured" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_login_rejects_invalid_key(self):
        """Login should fail with 401 for wrong API key."""
        from fastapi import HTTPException

        mock_settings = _mock_global_settings(api_key="correct-key")
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.LoginRequest(api_key="wrong-key")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.login(request, MagicMock()))
            assert exc_info.value.status_code == 401
        finally:
            _restore_getter(original)

    def test_login_success(self):
        """Login should succeed with correct API key."""
        mock_settings = _mock_global_settings(api_key="correct-key")
        mock_response = MagicMock()
        original = _patch_getter(mock_settings)
        try:
            request = admin_routes.LoginRequest(api_key="correct-key")
            result = asyncio.run(admin_routes.login(request, mock_response))
            assert result["success"] is True
            mock_response.set_cookie.assert_called_once()
        finally:
            _restore_getter(original)
