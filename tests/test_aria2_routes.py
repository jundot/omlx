# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, patch

import pytest

import omlx.admin.routes as admin_routes


@pytest.mark.asyncio
async def test_aria2_status_route_returns_dependency_state() -> None:
    expected = {
        "installed": True,
        "path": "/opt/homebrew/bin/aria2c",
        "version": "aria2 version 1.37.0",
    }
    with patch(
        "omlx.admin.routes.get_aria2_status",
        new=AsyncMock(return_value=expected),
    ):
        assert await admin_routes.aria2_status(is_admin=True) == expected


@pytest.mark.asyncio
async def test_aria2_install_route_uses_fixed_installer() -> None:
    result = {"installed": True, "path": "/opt/homebrew/bin/aria2c"}
    with patch.object(
        admin_routes._aria2_installer,
        "install",
        new=AsyncMock(return_value=result),
    ) as install:
        assert await admin_routes.install_aria2(is_admin=True) == result

    install.assert_awaited_once_with()
