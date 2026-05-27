# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin update-check endpoint.

flyto-mlx disables the upstream-release check (see
packaging/omlx_app/app.py::_check_for_updates and
omlx/admin/routes.py::check_update). The endpoint must always report
no update so the dashboard banner stays dark, regardless of upstream
jundot/omlx tags.
"""

import pytest

import omlx.admin.routes as admin_routes


class TestCheckUpdateDisabled:
    @pytest.mark.asyncio
    async def test_endpoint_always_reports_no_update(self):
        result = await admin_routes.check_update(is_admin=True)

        assert result == {
            "update_available": False,
            "latest_version": None,
            "release_url": None,
        }

    @pytest.mark.asyncio
    async def test_endpoint_does_not_call_github(self, monkeypatch):
        called = {"n": 0}

        async def fail_to_thread(*args, **kwargs):
            called["n"] += 1
            raise AssertionError(
                "check_update must not hit the network in flyto-mlx"
            )

        monkeypatch.setattr(admin_routes.asyncio, "to_thread", fail_to_thread)

        result = await admin_routes.check_update(is_admin=True)

        assert called["n"] == 0
        assert result["update_available"] is False
