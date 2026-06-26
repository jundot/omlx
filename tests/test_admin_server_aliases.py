# SPDX-License-Identifier: Apache-2.0
"""Tests for server alias support: /admin/api/server-info endpoint and
``server_aliases`` save/validate path in /admin/api/global-settings."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — ensure server module is imported first (triggers set_admin_getters)
from omlx.admin.routes import GlobalSettingsRequest
from omlx.utils.network import (
    detect_server_aliases,
    is_valid_alias,
    is_valid_bind_host,
    is_valid_hostname,
    is_valid_ip,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_global_settings(
    server_aliases: list[str] | None = None, host: str = "127.0.0.1"
):
    """Build a MagicMock GlobalSettings with the fields the alias paths touch."""
    gs = MagicMock()
    gs.server.host = host
    gs.server.port = 8000
    gs.server.log_level = "info"
    gs.server.server_aliases = list(server_aliases or [])
    gs.server.preserve_mid_system_cache = True
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
        assert is_valid_hostname("web1.local")

    def test_rejects_invalid_hostname(self):
        assert not is_valid_hostname("")
        assert not is_valid_hostname("with space")
        assert not is_valid_hostname("-leading-dash")
        assert not is_valid_hostname("a" * 300)

    def test_rejects_all_numeric_last_label_in_dotted_names(self):
        # For dotted (multi-label) names: IANA never delegates numeric TLDs,
        # so an all-digit rightmost label signals an IP-shaped string.
        # Mirrors the approach used by the ``validators`` PyPI library.
        assert not is_valid_hostname("999.999.999.999")
        assert not is_valid_hostname("1.2.3.4")
        assert not is_valid_hostname("host.123")

    def test_accepts_single_label_without_letters(self):
        # Single-label names (no dots) are local hostnames — no TLD constraint.
        assert is_valid_hostname("192-168-1-1")
        assert is_valid_hostname("web1")

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


class TestIsValidBindHost:
    """is_valid_bind_host() accepts IPs (including unspecified) and hostnames,
    but must reject IP-shaped values that fail IP parsing."""

    # ------------------------------------------------------------------
    # Valid IPv4 — including unspecified/wildcard addresses that are
    # rejected by is_valid_alias() but are legitimate bind targets.
    # ------------------------------------------------------------------

    def test_accepts_regular_ipv4(self):
        assert is_valid_bind_host("127.0.0.1")
        assert is_valid_bind_host("192.168.1.10")
        assert is_valid_bind_host("10.0.0.255")
        assert is_valid_bind_host("255.255.255.255")

    def test_accepts_wildcard_ipv4(self):
        assert is_valid_bind_host("0.0.0.0")

    # ------------------------------------------------------------------
    # Valid IPv6 — the ip-shaped guard uses a digit+dot regex so it
    # never interferes with colon-containing IPv6 addresses.
    # ------------------------------------------------------------------

    def test_accepts_wildcard_ipv6(self):
        assert is_valid_bind_host("::")

    def test_accepts_loopback_ipv6(self):
        assert is_valid_bind_host("::1")

    def test_accepts_link_local_ipv6(self):
        assert is_valid_bind_host("fe80::1")

    def test_accepts_full_ipv6(self):
        assert is_valid_bind_host("2001:db8::1")

    def test_accepts_ipv4_mapped_ipv6(self):
        # ::ffff:192.168.1.1 is valid IPv6 and contains dots, but the
        # colon means it reaches ipaddress.ip_address() first and parses fine.
        assert is_valid_bind_host("::ffff:192.168.1.1")

    # ------------------------------------------------------------------
    # Valid hostnames
    # ------------------------------------------------------------------

    def test_accepts_simple_hostname(self):
        assert is_valid_bind_host("localhost")
        assert is_valid_bind_host("my-host")

    def test_accepts_fqdn(self):
        assert is_valid_bind_host("my-host.local")
        assert is_valid_bind_host("example.com")

    def test_accepts_hostname_with_leading_digit_label(self):
        # Numeric-prefixed labels are valid hostnames (e.g. "web1.local")
        assert is_valid_bind_host("web1.local")

    def test_accepts_hostname_with_dashes_instead_of_dots(self):
        # "192-168-1-1" looks IP-like but uses dashes — valid hostname,
        # does not match the digit-dot regex.
        assert is_valid_bind_host("192-168-1-1")

    def test_accepts_all_letter_dotted_hostname(self):
        # Labels a.b.c.d contain letters so they don't match the ip-shaped guard.
        assert is_valid_bind_host("a.b.c.d")

    # ------------------------------------------------------------------
    # Rejected: IP-shaped strings that fail IP parsing
    # The bug: ipaddress.ip_address() raises ValueError, and digit-only
    # dotted labels also match the hostname regex — so without the guard
    # they would be accepted silently.
    # ------------------------------------------------------------------

    def test_rejects_ipv4_all_octets_out_of_range(self):
        assert not is_valid_bind_host("999.999.999.999")

    def test_rejects_ipv4_first_octet_out_of_range(self):
        assert not is_valid_bind_host("256.0.0.1")

    def test_rejects_ipv4_last_octet_out_of_range(self):
        assert not is_valid_bind_host("1.2.3.999")

    def test_rejects_ip_shaped_too_few_octets(self):
        # 3-part and 2-part dotted numeric strings look IP-shaped but are
        # not valid IPs and must not slip through as hostnames.
        assert not is_valid_bind_host("1.2.3")
        assert not is_valid_bind_host("1.2")

    def test_rejects_ip_shaped_too_many_octets(self):
        assert not is_valid_bind_host("1.2.3.4.5")

    # ------------------------------------------------------------------
    # Rejected: malformed hostnames
    # ------------------------------------------------------------------

    def test_rejects_leading_dash(self):
        assert not is_valid_bind_host("-bad-host")

    def test_rejects_hostname_with_space(self):
        assert not is_valid_bind_host("with space")

    def test_rejects_label_too_long(self):
        assert not is_valid_bind_host("a" * 64 + ".local")

    def test_rejects_value_too_long(self):
        assert not is_valid_bind_host("a." * 127 + "b")

    def test_rejects_invalid_ipv6_form(self):
        # Colon-containing but not a valid IP — falls through to hostname,
        # which rejects colons.
        assert not is_valid_bind_host(":invalid:")
        assert not is_valid_bind_host("[::1]")

    # ------------------------------------------------------------------
    # Rejected: empty / wrong types
    # ------------------------------------------------------------------

    def test_rejects_empty_string(self):
        assert not is_valid_bind_host("")

    def test_rejects_whitespace_only(self):
        assert not is_valid_bind_host("   ")

    def test_rejects_non_string(self):
        assert not is_valid_bind_host(None)  # type: ignore[arg-type]
        assert not is_valid_bind_host(8080)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Whitespace stripping
    # ------------------------------------------------------------------

    def test_strips_surrounding_whitespace(self):
        assert is_valid_bind_host("  127.0.0.1  ")
        assert is_valid_bind_host("  localhost  ")


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

    def test_comma_separated_host_includes_loopback(self):
        """Comma-separated bind hosts containing a loopback must not drop localhost aliases."""
        aliases = detect_server_aliases(host="127.0.0.1, ::1")
        assert "localhost" in aliases
        assert "127.0.0.1" in aliases

    def test_comma_separated_wildcard_includes_loopback(self):
        aliases = detect_server_aliases(host="0.0.0.0, ::1")
        assert "localhost" in aliases
        assert "127.0.0.1" in aliases

    def test_comma_separated_non_loopback_skips_loopback(self):
        """If no part of the comma-separated host is a loopback/wildcard, no loopback aliases."""
        aliases = detect_server_aliases(host="192.168.1.10, 10.0.0.1")
        assert "localhost" not in aliases


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


class TestUpdateGlobalSettingsHotCache:
    """update_global_settings: validate hot cache size before runtime apply."""

    def test_rejects_hot_cache_auto_with_400(self):
        gs = _make_global_settings()
        gs.cache.enabled = True
        gs.cache.hot_cache_max_size = "0"
        request = GlobalSettingsRequest(cache_enabled=False, hot_cache_max_size="auto")

        with _patched_global_settings(gs):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.update_global_settings(request=request, is_admin=True)
                )

        assert exc_info.value.status_code == 400
        assert "hot_cache_max_size" in exc_info.value.detail
        assert "auto" in exc_info.value.detail
        assert gs.cache.enabled is True
        assert gs.cache.hot_cache_max_size == "0"
        gs.save.assert_not_called()


class TestUpdateGlobalSettingsMidSystemCache:
    """update_global_settings: save the mid-system prefix-cache fallback toggle."""

    def test_saves_disabled_mid_system_cache_fallback(self):
        gs = _make_global_settings()
        request = GlobalSettingsRequest(preserve_mid_system_cache=False)

        with _patched_global_settings(gs):
            result = asyncio.run(
                admin_routes.update_global_settings(request=request, is_admin=True)
            )

        assert result["success"] is True
        assert "preserve_mid_system_cache" in result["runtime_applied"]
        assert gs.server.preserve_mid_system_cache is False
        gs.save.assert_called_once()


class TestUpdateGlobalSettingsSampling:
    """update_global_settings: saving and hot-applying sampling defaults."""

    @staticmethod
    def _make_sampling_settings(policy: int | None = None):
        return SimpleNamespace(
            max_context_window=32768,
            max_context_window_policy=policy,
            max_tokens=32768,
            temperature=1.0,
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.0,
        )

    def test_saves_and_hot_applies_context_window_policy(self):
        gs = MagicMock()
        gs.sampling = self._make_sampling_settings()
        gs.validate.return_value = []
        gs.save.return_value = None
        server_state = SimpleNamespace(sampling=self._make_sampling_settings())
        request = GlobalSettingsRequest(sampling_max_context_window_policy=128000)

        with (
            _patched_global_settings(gs),
            patch.object(
                omlx.server,
                "_server_state",
                server_state,
            ),
        ):
            result = asyncio.run(
                admin_routes.update_global_settings(request=request, is_admin=True)
            )

        assert result["success"] is True
        assert "sampling" in result["runtime_applied"]
        assert gs.sampling.max_context_window_policy == 128000
        assert server_state.sampling.max_context_window_policy == 128000
        gs.save.assert_called_once()

    def test_explicit_null_clears_context_window_policy(self):
        gs = MagicMock()
        gs.sampling = self._make_sampling_settings(policy=128000)
        gs.validate.return_value = []
        gs.save.return_value = None
        server_state = SimpleNamespace(
            sampling=self._make_sampling_settings(policy=128000)
        )
        request = GlobalSettingsRequest(sampling_max_context_window_policy=None)

        with (
            _patched_global_settings(gs),
            patch.object(
                omlx.server,
                "_server_state",
                server_state,
            ),
        ):
            result = asyncio.run(
                admin_routes.update_global_settings(request=request, is_admin=True)
            )

        assert "sampling_max_context_window_policy" in request.model_fields_set
        assert result["success"] is True
        assert "sampling" in result["runtime_applied"]
        assert gs.sampling.max_context_window_policy is None
        assert server_state.sampling.max_context_window_policy is None
        gs.save.assert_called_once()


class TestUpdateGlobalSettingsEmbeddingBatchSize:
    """update_global_settings: saving and hot-applying embedding batch size."""

    def _make_scheduler_settings(self):
        return SimpleNamespace(
            max_concurrent_requests=8,
            embedding_batch_size=32,
            chunked_prefill=False,
        )

    def test_saves_and_hot_applies_embedding_batch_size(self):
        gs = MagicMock()
        gs.scheduler = self._make_scheduler_settings()
        gs.validate.return_value = []
        gs.save.return_value = None

        pool = SimpleNamespace(apply_embedding_batch_size=AsyncMock())
        server_state = SimpleNamespace(engine_pool=pool)
        request = GlobalSettingsRequest(embedding_batch_size=5)

        with (
            _patched_global_settings(gs),
            patch.object(
                omlx.server,
                "_server_state",
                server_state,
            ),
        ):
            result = asyncio.run(
                admin_routes.update_global_settings(request=request, is_admin=True)
            )

        assert result["success"] is True
        assert "embedding_batch_size" in result["runtime_applied"]
        assert gs.scheduler.embedding_batch_size == 5
        pool.apply_embedding_batch_size.assert_awaited_once_with(5)
        gs.save.assert_called_once()

    def test_rejects_invalid_embedding_batch_size(self):
        gs = MagicMock()
        gs.scheduler = self._make_scheduler_settings()
        gs.validate.return_value = []
        gs.save.return_value = None
        request = GlobalSettingsRequest(embedding_batch_size=0)

        with _patched_global_settings(gs):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.update_global_settings(request=request, is_admin=True)
                )

        assert exc_info.value.status_code == 400
        assert "embedding_batch_size" in exc_info.value.detail
        gs.save.assert_not_called()

    def test_does_not_hot_apply_embedding_batch_size_when_validation_fails(self):
        gs = MagicMock()
        gs.scheduler = self._make_scheduler_settings()
        gs.validate.return_value = ["invalid unrelated setting"]
        gs.save.return_value = None
        pool = SimpleNamespace(apply_embedding_batch_size=AsyncMock())
        server_state = SimpleNamespace(engine_pool=pool)
        request = GlobalSettingsRequest(embedding_batch_size=5)

        with (
            _patched_global_settings(gs),
            patch.object(
                omlx.server,
                "_server_state",
                server_state,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.update_global_settings(request=request, is_admin=True)
                )

        assert exc_info.value.status_code == 400
        pool.apply_embedding_batch_size.assert_not_awaited()
        assert gs.scheduler.embedding_batch_size == 32
        gs.save.assert_not_called()

    def test_does_not_mutate_embedding_batch_size_when_api_key_is_invalid(self):
        gs = MagicMock()
        gs.scheduler = self._make_scheduler_settings()
        gs.validate.return_value = []
        gs.save.return_value = None
        pool = SimpleNamespace(apply_embedding_batch_size=AsyncMock())
        server_state = SimpleNamespace(engine_pool=pool)
        request = GlobalSettingsRequest(embedding_batch_size=5, api_key="abc")

        with (
            _patched_global_settings(gs),
            patch.object(
                omlx.server,
                "_server_state",
                server_state,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.update_global_settings(request=request, is_admin=True)
                )

        assert exc_info.value.status_code == 400
        pool.apply_embedding_batch_size.assert_not_awaited()
        assert gs.scheduler.embedding_batch_size == 32
        gs.save.assert_not_called()

    def test_does_not_hot_apply_embedding_batch_size_when_save_fails(self):
        gs = MagicMock()
        gs.scheduler = self._make_scheduler_settings()
        gs.validate.return_value = []
        gs.save.side_effect = OSError("disk full")
        pool = SimpleNamespace(apply_embedding_batch_size=AsyncMock())
        server_state = SimpleNamespace(engine_pool=pool)
        request = GlobalSettingsRequest(embedding_batch_size=5)

        with (
            _patched_global_settings(gs),
            patch.object(
                omlx.server,
                "_server_state",
                server_state,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.update_global_settings(request=request, is_admin=True)
                )

        assert exc_info.value.status_code == 500
        pool.apply_embedding_batch_size.assert_not_awaited()
        assert gs.scheduler.embedding_batch_size == 32
