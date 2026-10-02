# SPDX-License-Identifier: Apache-2.0
"""Tests for admin update check endpoint."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import omlx.admin.routes as admin_routes


class _FakeResponse:
    """Minimal mock for requests.Response."""

    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self._data = data if data is not None else []

    def json(self):
        return self._data


def _reset_cache():
    """Reset module-level update cache between tests."""
    admin_routes._update_cache = {}
    admin_routes._update_cache_time = {}
    admin_routes._UPDATE_PREFS_PATH = Path("/tmp/omlx-test-missing-update-prefs.json")


class TestCheckUpdate:
    """Tests for /admin/api/update-check endpoint."""

    def setup_method(self):
        _reset_cache()

    def teardown_method(self):
        _reset_cache()

    @pytest.mark.asyncio
    async def test_update_available(self):
        """Should return update_available=True when newer version exists."""
        fake_resp = _FakeResponse(200, [{
            "tag_name": "v99.0.0",
            "html_url": "https://github.com/jundot/omlx/releases/tag/v99.0.0",
        }])

        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _make_async_return(fake_resp)

            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is True
        assert result["latest_version"] == "99.0.0"
        assert "releases/tag" in result["release_url"]

    @pytest.mark.asyncio
    async def test_no_update(self):
        """Should return update_available=False when current version is latest."""
        fake_resp = _FakeResponse(200, [{
            "tag_name": "v0.0.1",
            "html_url": "https://github.com/jundot/omlx/releases/tag/v0.0.1",
        }])

        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _make_async_return(fake_resp)

            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is False
        assert result["latest_version"] is None

    @pytest.mark.asyncio
    async def test_github_api_failure(self):
        """Should return update_available=False on HTTP error."""
        fake_resp = _FakeResponse(403)

        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = _make_async_return(fake_resp)

            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is False

    @pytest.mark.asyncio
    async def test_network_error(self):
        """Should return update_available=False on network exception."""

        async def raise_error(*args, **kwargs):
            raise ConnectionError("no network")

        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = raise_error

            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is False

    @pytest.mark.asyncio
    async def test_cache_is_used(self):
        """Should not call GitHub API again within TTL window."""
        fake_resp = _FakeResponse(200, [{
            "tag_name": "v99.0.0",
            "html_url": "https://github.com/jundot/omlx/releases/tag/v99.0.0",
        }])

        call_count = 0

        async def counting_to_thread(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fake_resp

        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = counting_to_thread

            # First call - should hit API
            await admin_routes.check_update(is_admin=True)
            assert call_count == 1

            # Second call - should use cache
            result = await admin_routes.check_update(is_admin=True)
            assert call_count == 1
            assert result["update_available"] is True

    @pytest.mark.asyncio
    async def test_cache_expires(self):
        """Should call GitHub API again after TTL expires."""
        fake_resp = _FakeResponse(200, [{
            "tag_name": "v99.0.0",
            "html_url": "https://github.com/jundot/omlx/releases/tag/v99.0.0",
        }])

        call_count = 0

        async def counting_to_thread(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fake_resp

        with patch("omlx.admin.routes.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = counting_to_thread

            # First call
            await admin_routes.check_update(is_admin=True)
            assert call_count == 1

            # Expire cache
            admin_routes._update_cache_time["stable"] = time.time() - 90000

            # Second call - should hit API again
            await admin_routes.check_update(is_admin=True)
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_rc_channel_shows_newer_rc(self):
        """Release Candidate channel should show newer RC releases."""
        fake_resp = _FakeResponse(
            200,
            [
                {
                    "tag_name": "v0.4.0rc2",
                    "html_url": "https://github.com/jundot/omlx/releases/tag/v0.4.0rc2",
                }
            ],
        )

        with (
            patch("omlx.admin.routes._omlx_version", "0.4.0rc1"),
            patch(
                "omlx.admin.routes._read_update_channel",
                return_value="release_candidate",
            ),
            patch("omlx.admin.routes.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = _make_async_return(fake_resp)

            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is True
        assert result["latest_version"] == "0.4.0rc2"
        assert result["update_channel"] == "release_candidate"

    @pytest.mark.asyncio
    async def test_stable_channel_hides_rc(self):
        """Stable channel should not show RC-only releases."""
        fake_resp = _FakeResponse(
            200,
            [
                {
                    "tag_name": "v0.4.0rc2",
                    "html_url": "https://github.com/jundot/omlx/releases/tag/v0.4.0rc2",
                }
            ],
        )

        with (
            patch("omlx.admin.routes._omlx_version", "0.4.0rc1"),
            patch("omlx.admin.routes._read_update_channel", return_value="stable"),
            patch("omlx.admin.routes.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = _make_async_return(fake_resp)

            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is False
        assert result["latest_version"] is None
        assert result["update_channel"] == "stable"

    @pytest.mark.asyncio
    async def test_rc_channel_shows_final_release(self):
        """Release Candidate channel should show final stable releases."""
        fake_resp = _FakeResponse(
            200,
            [
                {
                    "tag_name": "v0.4.0",
                    "html_url": "https://github.com/jundot/omlx/releases/tag/v0.4.0",
                }
            ],
        )

        with (
            patch("omlx.admin.routes._omlx_version", "0.4.0rc2"),
            patch(
                "omlx.admin.routes._read_update_channel",
                return_value="release_candidate",
            ),
            patch("omlx.admin.routes.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = _make_async_return(fake_resp)

            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is True
        assert result["latest_version"] == "0.4.0"

    @pytest.mark.asyncio
    async def test_dev_channel_shows_dev_release(self):
        """Dev channel should show dev releases."""
        fake_resp = _FakeResponse(
            200,
            [
                {
                    "tag_name": "v0.4.1.dev1",
                    "html_url": "https://github.com/jundot/omlx/releases/tag/v0.4.1.dev1",
                }
            ],
        )

        with (
            patch("omlx.admin.routes._omlx_version", "0.4.0"),
            patch("omlx.admin.routes._read_update_channel", return_value="dev"),
            patch("omlx.admin.routes.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = _make_async_return(fake_resp)

            result = await admin_routes.check_update(is_admin=True)

        assert result["update_available"] is True
        assert result["latest_version"] == "0.4.1.dev1"
        assert result["update_channel"] == "dev"

    @pytest.mark.asyncio
    async def test_cache_is_per_channel(self):
        """Stable and RC update-check cache entries should not be shared."""
        fake_resp = _FakeResponse(
            200,
            [
                {
                    "tag_name": "v0.4.0rc2",
                    "html_url": "https://github.com/jundot/omlx/releases/tag/v0.4.0rc2",
                }
            ],
        )

        call_count = 0
        channels = iter(["stable", "release_candidate"])

        async def counting_to_thread(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fake_resp

        with (
            patch("omlx.admin.routes._omlx_version", "0.4.0rc1"),
            patch(
                "omlx.admin.routes._read_update_channel",
                side_effect=lambda: next(channels),
            ),
            patch("omlx.admin.routes.asyncio") as mock_asyncio,
        ):
            mock_asyncio.to_thread = counting_to_thread

            stable_result = await admin_routes.check_update(is_admin=True)
            rc_result = await admin_routes.check_update(is_admin=True)

        assert call_count == 2
        assert stable_result["update_available"] is False
        assert rc_result["update_available"] is True


def _make_async_return(value):
    """Create an async function that returns the given value."""

    async def _async_return(*args, **kwargs):
        return value

    return _async_return
