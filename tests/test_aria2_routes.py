# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes
from omlx.admin.aria2_downloader import Aria2UnavailableError


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


@pytest.mark.asyncio
async def test_hf_download_reports_missing_aria2_as_service_unavailable() -> None:
    downloader = AsyncMock()
    downloader.start_download.side_effect = Aria2UnavailableError("aria2 missing")
    request = admin_routes.HFDownloadRequest(repo_id="owner/model")

    with (
        patch.object(admin_routes, "_hf_downloader", downloader),
        pytest.raises(HTTPException) as exc_info,
    ):
        await admin_routes.start_hf_download(request=request, is_admin=True)

    assert exc_info.value.status_code == 503
    assert "aria2 missing" in exc_info.value.detail
