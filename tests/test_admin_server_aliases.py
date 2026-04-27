# SPDX-License-Identifier: Apache-2.0
"""Tests for server alias support: /admin/api/server-info endpoint and
``server_aliases`` save/validate path in /admin/api/global-settings."""

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import omlx.server  # noqa: F401 — ensure server module is imported first (triggers set_admin_getters)
import omlx.admin.routes as admin_routes
from omlx.admin.routes import GlobalSettingsRequest
from omlx.utils.network import (
    detect_server_aliases,
    is_valid_alias,
    is_valid_hostname,
    is_valid_ip,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_global_settings(server_aliases: list[str] | None = None, host: str = "127.0.0.1"):
    """Build a MagicMock GlobalSettings with the fields the alias paths touch."""
    gs = MagicMock()
    gs.server.host = host
    gs.server.port = 8000
    gs.server.log_level = "info"
    gs.server.server_aliases = list(server_aliases or [])
    # Validation is invoked at the end of update_global_settings; return no errors.
    gs.validate.return_value = []
    gs.save.return_value = None
    return gs


@contextmanager
def _patched_global_settings(gs):
    """Patch the module-level _get_global_settings getter without disturbing others."""
    original = admin_routes._get_global_settings
    admin_routes._get_global_settings = lambda: gs
    try:
        yield
    finally:
        admin_routes._get_global_settings = original


# =============================================================================
# Unit tests for omlx.utils.network
# =============================================================================


class TestNetworkValidation:
    """Validation primitives used by the alias save path."""

    def test_valid_ipv4(self):
        assert is_valid_ip("192.168.1.10")
        assert is_valid_ip("127.0.0.1")

    def test_valid_ipv6(self):
        assert is_valid_ip("::1")
        assert is_valid_ip("fe80::1")

    def test_rejects_unspecified_ipv4(self):
        """0.0.0.0 parses as a valid IP but is not routable as an alias."""
        assert not is_valid_ip("0.0.0.0")

    def test_rejects_unspecified_ipv6(self):
        """:: is the IPv6 unspecified address — also not usable as an alias."""
        assert not is_valid_ip("::")

    def test_rejects_garbage(self):
        assert not is_valid_ip("not-an-ip")
        assert not is_valid_ip("999.999.999.999")

    def test_valid_hostname(self):
        assert is_valid_hostname("example.local")
        assert is_valid_hostname("my-mac")
        assert is_valid_hostname("a.b.c.d")

    def test_rejects_invalid_hostname(self):
        assert not is_valid_hostname("")
        assert not is_valid_hostname("with space")
        assert not is_valid_hostname("-leading-dash")
        assert not is_valid_hostname("a" * 300)

    def test_alias_accepts_either(self):
        assert is_valid_alias("localhost")
        assert is_valid_alias("192.168.1.10")
        assert is_valid_alias("foo.local")
        assert is_valid_alias("::1")

    def test_alias_rejects_unspecified(self):
        assert not is_valid_alias("0.0.0.0")
        assert not is_valid_alias("::")

    def test_alias_rejects_non_string(self):
        assert not is_valid_alias(None)  # type: ignore[arg-type]
        assert not is_valid_alias(123)  # type: ignore[arg-type]


class TestDetectServerAliases:
    """Auto-detection should always return at least loopback when bound to localhost."""

    def test_localhost_includes_loopback(self):
        aliases = detect_server_aliases(host="127.0.0.1")
        assert "localhost" in aliases
        assert "127.0.0.1" in aliases

    def test_no_unspecified_in_output(self):
        """Even when bound to 0.0.0.0, detection should not return 0.0.0.0 itself."""
        aliases = detect_server_aliases(host="0.0.0.0")
        assert "0.0.0.0" not in aliases
        assert "::" not in aliases

    def test_returns_unique_values(self):
        aliases = detect_server_aliases()
        assert len(aliases) == len(set(aliases))


# =============================================================================
# /admin/api/server-info endpoint
# =============================================================================


class TestServerInfoEndpoint:
    """get_server_info: returns persisted aliases or falls back to detection."""

    def test_returns_persisted_aliases(self):
        gs = _make_global_settings(
            server_aliases=["my-mac.local", "192.168.1.10", "localhost"],
            host="127.0.0.1",
        )
        with _patched_global_settings(gs):
            result = asyncio.run(admin_routes.get_server_info(is_admin=True))

        assert result["host"] == "127.0.0.1"
        assert result["port"] == 8000
        assert result["aliases"] == ["my-mac.local", "192.168.1.10", "localhost"]

    def test_falls_back_to_detection_when_empty(self):
        """Empty persisted list → live auto-detection kicks in."""
        gs = _make_global_settings(server_aliases=[], host="127.0.0.1")
        with _patched_global_settings(gs):
            result = asyncio.run(admin_routes.get_server_info(is_admin=True))

        # Auto-detection always returns at least the loopback pair.
        assert "localhost" in result["aliases"]
        assert "127.0.0.1" in result["aliases"]

    def test_returns_503_when_settings_unavailable(self):
        with _patched_global_settings(None):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(admin_routes.get_server_info(is_admin=True))
        assert exc_info.value.status_code == 503


# =============================================================================
# /admin/api/global-settings save path for server_aliases
# =============================================================================


class TestUpdateGlobalSettingsAliases:
    """update_global_settings: saving server_aliases with validation."""

    def test_saves_valid_aliases(self):
        gs = _make_global_settings(server_aliases=[])
        request = GlobalSettingsRequest(server_aliases=["custom.local", "10.0.0.5"])

        with _patched_global_settings(gs):
            result = asyncio.run(
                admin_routes.update_global_settings(request=request, is_admin=True)
            )

        assert result["success"] is True
        assert "server_aliases" in result["runtime_applied"]
        assert gs.server.server_aliases == ["custom.local", "10.0.0.5"]
        gs.save.assert_called_once()

    def test_strips_whitespace_and_dedupes(self):
        gs = _make_global_settings(server_aliases=[])
        request = GlobalSettingsRequest(
            server_aliases=["  foo.local  ", "foo.local", "10.0.0.5", "  "],
        )

        with _patched_global_settings(gs):
            asyncio.run(
                admin_routes.update_global_settings(request=request, is_admin=True)
            )

        assert gs.server.server_aliases == ["foo.local", "10.0.0.5"]

    def test_rejects_invalid_alias_with_400(self):
        gs = _make_global_settings(server_aliases=[])
        request = GlobalSettingsRequest(
            server_aliases=["valid.local", "not valid!!!"],
        )

        with _patched_global_settings(gs):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.update_global_settings(request=request, is_admin=True)
                )

        assert exc_info.value.status_code == 400
        assert "not valid!!!" in exc_info.value.detail
        gs.save.assert_not_called()

    def test_rejects_unspecified_address_with_400(self):
        """0.0.0.0 must be rejected — bind address, not a routable URL host."""
        gs = _make_global_settings(server_aliases=[])
        request = GlobalSettingsRequest(server_aliases=["0.0.0.0"])

        with _patched_global_settings(gs):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.update_global_settings(request=request, is_admin=True)
                )

        assert exc_info.value.status_code == 400
        assert "0.0.0.0" in exc_info.value.detail
        gs.save.assert_not_called()

    def test_rejects_ipv6_unspecified_with_400(self):
        gs = _make_global_settings(server_aliases=[])
        request = GlobalSettingsRequest(server_aliases=["::"])

        with _patched_global_settings(gs):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.update_global_settings(request=request, is_admin=True)
                )

        assert exc_info.value.status_code == 400

    def test_accepts_ipv6_loopback(self):
        gs = _make_global_settings(server_aliases=[])
        request = GlobalSettingsRequest(server_aliases=["::1"])

        with _patched_global_settings(gs):
            asyncio.run(
                admin_routes.update_global_settings(request=request, is_admin=True)
            )

        assert gs.server.server_aliases == ["::1"]

    def test_empty_list_clears_aliases(self):
        gs = _make_global_settings(server_aliases=["existing.local"])
        request = GlobalSettingsRequest(server_aliases=[])

        with _patched_global_settings(gs):
            asyncio.run(
                admin_routes.update_global_settings(request=request, is_admin=True)
            )

        assert gs.server.server_aliases == []
