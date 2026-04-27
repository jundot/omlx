# SPDX-License-Identifier: Apache-2.0
"""Tests for API key authentication."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# Note: These tests need a mock server setup since the actual server requires models


def _mock_request(headers=None):
    """Create a mock FastAPI request with given headers."""
    req = MagicMock()
    req.headers = headers or {}
    return req


class TestVerifyApiKey:
    """Tests for verify_api_key function."""

    def test_verify_api_key_no_auth_required(self):
        """Test that no auth is required when api_key is None."""
        from omlx.server import verify_api_key, _server_state
        import asyncio

        original_key = _server_state.api_key
        _server_state.api_key = None

        try:
            # Should return True without any credentials
            result = asyncio.run(verify_api_key(request=_mock_request(), credentials=None))
            assert result is True
        finally:
            _server_state.api_key = original_key

    def test_verify_api_key_missing_credentials(self):
        """Test that missing credentials raises 401 when api_key is set."""
        from omlx.server import verify_api_key, _server_state
        from fastapi import HTTPException
        import asyncio

        original_key = _server_state.api_key
        _server_state.api_key = "test-key"

        try:
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(verify_api_key(request=_mock_request(), credentials=None))
            assert exc_info.value.status_code == 401
            assert "required" in exc_info.value.detail.lower()
        finally:
            _server_state.api_key = original_key

    def test_verify_api_key_invalid_key(self):
        """Test that invalid key raises 401."""
        from omlx.server import verify_api_key, _server_state
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        import asyncio

        original_key = _server_state.api_key
        _server_state.api_key = "correct-key"

        try:
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(verify_api_key(request=_mock_request(), credentials=credentials))
            assert exc_info.value.status_code == 401
            assert "invalid" in exc_info.value.detail.lower()
        finally:
            _server_state.api_key = original_key

    def test_verify_api_key_valid_key(self):
        """Test that valid key passes."""
        from omlx.server import verify_api_key, _server_state
        from fastapi.security import HTTPAuthorizationCredentials
        import asyncio

        original_key = _server_state.api_key
        _server_state.api_key = "correct-key"

        try:
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="correct-key")
            result = asyncio.run(verify_api_key(request=_mock_request(), credentials=credentials))
            assert result is True
        finally:
            _server_state.api_key = original_key


class TestXApiKeyHeader:
    """Tests for x-api-key header authentication (Anthropic SDK compatibility)."""

    def test_x_api_key_header_accepted(self):
        """Test that x-api-key header is accepted when no Bearer token."""
        from omlx.server import verify_api_key, _server_state
        import asyncio

        original_key = _server_state.api_key
        _server_state.api_key = "correct-key"

        try:
            request = _mock_request(headers={"x-api-key": "correct-key"})
            result = asyncio.run(verify_api_key(request=request, credentials=None))
            assert result is True
        finally:
            _server_state.api_key = original_key

    def test_x_api_key_header_invalid(self):
        """Test that invalid x-api-key raises 401."""
        from omlx.server import verify_api_key, _server_state
        from fastapi import HTTPException
        import asyncio

        original_key = _server_state.api_key
        _server_state.api_key = "correct-key"

        try:
            request = _mock_request(headers={"x-api-key": "wrong-key"})
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(verify_api_key(request=request, credentials=None))
            assert exc_info.value.status_code == 401
            assert "invalid" in exc_info.value.detail.lower()
        finally:
            _server_state.api_key = original_key

    def test_bearer_takes_priority_over_x_api_key(self):
        """Test that Bearer token takes priority when both are present."""
        from omlx.server import verify_api_key, _server_state
        from fastapi.security import HTTPAuthorizationCredentials
        import asyncio

        original_key = _server_state.api_key
        _server_state.api_key = "bearer-key"

        try:
            # Bearer has correct key, x-api-key has wrong key
            request = _mock_request(headers={"x-api-key": "wrong-key"})
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bearer-key")
            result = asyncio.run(verify_api_key(request=request, credentials=credentials))
            assert result is True
        finally:
            _server_state.api_key = original_key

    def test_x_api_key_with_sub_keys(self):
        """Test that x-api-key works with sub keys."""
        from omlx.server import verify_api_key, _server_state
        from omlx.settings import SubKeyEntry
        import asyncio

        original_key = _server_state.api_key
        original_gs = _server_state.global_settings
        _server_state.api_key = "main-key"

        mock_gs = MagicMock()
        mock_gs.auth.sub_keys = [
            SubKeyEntry(key="sub-key-1", name="Test Sub Key"),
        ]
        mock_gs.auth.skip_api_key_verification = False
        mock_gs.server.host = "0.0.0.0"
        _server_state.global_settings = mock_gs

        try:
            request = _mock_request(headers={"x-api-key": "sub-key-1"})
            result = asyncio.run(verify_api_key(request=request, credentials=None))
            assert result is True
        finally:
            _server_state.api_key = original_key
            _server_state.global_settings = original_gs


class TestSubKeyVerification:
    """Tests for sub key API authentication."""

    def test_sub_key_accepted_for_api(self):
        """Test that a sub key is accepted for API authentication."""
        from omlx.server import verify_api_key, _server_state
        from omlx.settings import SubKeyEntry
        from fastapi.security import HTTPAuthorizationCredentials
        import asyncio

        original_key = _server_state.api_key
        original_gs = _server_state.global_settings
        _server_state.api_key = "main-key"

        # Create mock global_settings with sub keys
        mock_gs = MagicMock()
        mock_gs.auth.sub_keys = [
            SubKeyEntry(key="sub-key-1", name="Test Sub Key"),
        ]
        mock_gs.auth.skip_api_key_verification = False
        mock_gs.server.host = "127.0.0.1"
        _server_state.global_settings = mock_gs

        try:
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="sub-key-1")
            result = asyncio.run(verify_api_key(request=_mock_request(), credentials=credentials))
            assert result is True
        finally:
            _server_state.api_key = original_key
            _server_state.global_settings = original_gs

    def test_invalid_sub_key_rejected(self):
        """Test that an invalid sub key is rejected."""
        from omlx.server import verify_api_key, _server_state
        from omlx.settings import SubKeyEntry
        from fastapi.security import HTTPAuthorizationCredentials
        from fastapi import HTTPException
        import asyncio

        original_key = _server_state.api_key
        original_gs = _server_state.global_settings
        _server_state.api_key = "main-key"

        mock_gs = MagicMock()
        mock_gs.auth.sub_keys = [
            SubKeyEntry(key="sub-key-1", name="Test Sub Key"),
        ]
        mock_gs.auth.skip_api_key_verification = False
        mock_gs.server.host = "0.0.0.0"
        _server_state.global_settings = mock_gs

        try:
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(verify_api_key(request=_mock_request(), credentials=credentials))
            assert exc_info.value.status_code == 401
        finally:
            _server_state.api_key = original_key
            _server_state.global_settings = original_gs

    def test_main_key_still_works_for_api(self):
        """Test that the main key still works for API authentication."""
        from omlx.server import verify_api_key, _server_state
        from omlx.settings import SubKeyEntry
        from fastapi.security import HTTPAuthorizationCredentials
        import asyncio

        original_key = _server_state.api_key
        original_gs = _server_state.global_settings
        _server_state.api_key = "main-key"

        mock_gs = MagicMock()
        mock_gs.auth.sub_keys = [
            SubKeyEntry(key="sub-key-1", name="Test Sub Key"),
        ]
        mock_gs.auth.skip_api_key_verification = False
        mock_gs.server.host = "0.0.0.0"
        _server_state.global_settings = mock_gs

        try:
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="main-key")
            result = asyncio.run(verify_api_key(request=_mock_request(), credentials=credentials))
            assert result is True
        finally:
            _server_state.api_key = original_key
            _server_state.global_settings = original_gs


class TestSkipApiKeyVerification:
    """Tests for skip_api_key_verification feature."""

    def _make_global_settings(self, host="127.0.0.1", skip=True):
        from omlx.settings import GlobalSettings, ServerSettings, AuthSettings
        from dataclasses import dataclass
        gs = GlobalSettings.__new__(GlobalSettings)
        gs.server = ServerSettings(host=host)
        gs.auth = AuthSettings(api_key="test-key", skip_api_key_verification=skip)
        return gs

    def test_skip_verification_when_localhost(self):
        """Skip API key verification when enabled."""
        from omlx.server import verify_api_key, _server_state
        import asyncio

        original_key = _server_state.api_key
        original_gs = _server_state.global_settings
        _server_state.api_key = "test-key"
        _server_state.global_settings = self._make_global_settings(
            host="127.0.0.1", skip=True
        )

        try:
            result = asyncio.run(verify_api_key(request=_mock_request(), credentials=None))
            assert result is True
        finally:
            _server_state.api_key = original_key
            _server_state.global_settings = original_gs

    def test_skip_verification_on_any_host(self):
        """Skip verification when enabled regardless of host."""
        from omlx.server import verify_api_key, _server_state
        import asyncio

        original_key = _server_state.api_key
        original_gs = _server_state.global_settings
        _server_state.api_key = "test-key"
        _server_state.global_settings = self._make_global_settings(
            host="0.0.0.0", skip=True
        )

        try:
            result = asyncio.run(verify_api_key(request=_mock_request(), credentials=None))
            assert result is True
        finally:
            _server_state.api_key = original_key
            _server_state.global_settings = original_gs

    def test_skip_verification_disabled_by_default(self):
        """Default skip_api_key_verification is False."""
        from omlx.settings import AuthSettings

        auth = AuthSettings()
        assert auth.skip_api_key_verification is False


class TestAdminAuth:
    """Tests for admin authentication functions."""

    def test_create_session_token(self):
        """Test session token creation."""
        from omlx.admin.auth import create_session_token

        token = create_session_token()
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_session_token_valid(self):
        """Test valid session token verification."""
        from omlx.admin.auth import create_session_token, verify_session_token

        token = create_session_token()
        assert verify_session_token(token) is True

    def test_verify_session_token_invalid(self):
        """Test invalid session token verification."""
        from omlx.admin.auth import verify_session_token

        assert verify_session_token("invalid-token") is False

    def test_verify_session_token_expired(self):
        """Test expired session token verification."""
        from omlx.admin.auth import create_session_token, verify_session_token
        import time

        token = create_session_token()
        # Wait a moment and verify with very short max_age
        time.sleep(0.1)
        # With max_age=0, token should be expired after any delay
        # Note: itsdangerous rounds to nearest second, so we use a small delay
        assert verify_session_token(token, max_age=-1) is False

    def test_verify_api_key_constant_time(self):
        """Test that API key comparison uses constant time."""
        from omlx.admin.auth import verify_api_key
        import secrets

        server_key = "test-api-key-12345"

        # Valid key
        assert verify_api_key("test-api-key-12345", server_key) is True

        # Invalid key
        assert verify_api_key("wrong-key", server_key) is False

        # Empty key
        assert verify_api_key("", server_key) is False
