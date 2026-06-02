# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/api/mcp_routes.py — the HTTP layer over MCPClientManager.

The manager itself is covered by tests/test_mcp_manager.py; here we only
verify the route handlers: response shape, alias handling, and the
no-manager fallbacks that ship in each endpoint.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.api import mcp_routes
from omlx.mcp.types import (
    MCPServerState,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)


@pytest.fixture
def app_client():
    """TestClient mounting only the MCP router."""
    app = FastAPI()
    app.include_router(mcp_routes.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_mcp_manager_getter():
    """Each test gets a clean ``_get_mcp_manager`` slot.

    Routes consult a module-global callback; without this fixture a prior
    test's getter would leak into the next test's no-manager path.
    """
    original = mcp_routes._get_mcp_manager
    mcp_routes._get_mcp_manager = None
    yield
    mcp_routes._get_mcp_manager = original


def _make_status(name, state=MCPServerState.CONNECTED, tools_count=0, error=None):
    status = MagicMock()
    status.name = name
    status.state = state
    status.transport = MCPTransport.STDIO
    status.tools_count = tools_count
    status.error = error
    return status


class TestSetMcpManagerGetter:
    def test_getter_is_installed_and_invoked(self):
        sentinel = object()
        mcp_routes.set_mcp_manager_getter(lambda: sentinel)
        assert mcp_routes._get_manager() is sentinel

    def test_unset_getter_returns_none(self):
        """No getter wired → _get_manager returns None.

        Routes lean on this to short-circuit into the empty-list / 503
        branches when the server starts without --mcp-config.
        """
        assert mcp_routes._get_manager() is None

    def test_getter_returning_none_propagates(self):
        """Manager is unset *because the getter says so*, not just because
        no getter was registered. Routes must treat both identically."""
        mcp_routes.set_mcp_manager_getter(lambda: None)
        assert mcp_routes._get_manager() is None


class TestListMcpTools:
    def test_returns_empty_when_no_manager(self, app_client):
        r = app_client.get("/v1/mcp/tools")
        assert r.status_code == 200
        assert r.json() == {"tools": [], "count": 0}

    def test_serializes_tools_from_manager(self, app_client):
        tool_a = MCPTool(
            server_name="srv1",
            name="add",
            description="Add two numbers",
            input_schema={"type": "object", "properties": {}},
        )
        tool_b = MCPTool(
            server_name="srv2",
            name="search",
            description="Search the web",
            input_schema={"type": "object"},
        )
        mgr = MagicMock()
        mgr.get_all_tools.return_value = [tool_a, tool_b]
        mcp_routes.set_mcp_manager_getter(lambda: mgr)

        r = app_client.get("/v1/mcp/tools")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert body["tools"][0] == {
            "name": "srv1__add",  # namespaced via MCPTool.full_name
            "description": "Add two numbers",
            "server": "srv1",
            "parameters": {"type": "object", "properties": {}},
        }
        assert body["tools"][1]["name"] == "srv2__search"
        assert body["tools"][1]["server"] == "srv2"

    def test_zero_tools_returns_count_zero(self, app_client):
        mgr = MagicMock()
        mgr.get_all_tools.return_value = []
        mcp_routes.set_mcp_manager_getter(lambda: mgr)

        r = app_client.get("/v1/mcp/tools")
        assert r.status_code == 200
        assert r.json() == {"tools": [], "count": 0}


class TestListMcpServers:
    def test_returns_empty_when_no_manager(self, app_client):
        r = app_client.get("/v1/mcp/servers")
        assert r.status_code == 200
        assert r.json() == {"servers": []}

    def test_serializes_state_enum_to_string(self, app_client):
        """The route flattens ``MCPServerState`` → its ``.value`` so JSON
        clients see a plain string ("connected") not an enum repr."""
        mgr = MagicMock()
        mgr.get_server_status.return_value = [
            _make_status("primary", state=MCPServerState.CONNECTED, tools_count=3)
        ]
        mcp_routes.set_mcp_manager_getter(lambda: mgr)

        r = app_client.get("/v1/mcp/servers")
        assert r.status_code == 200
        servers = r.json()["servers"]
        assert len(servers) == 1
        assert servers[0]["name"] == "primary"
        assert servers[0]["state"] == "connected"  # enum.value
        assert servers[0]["transport"] == "stdio"
        assert servers[0]["tools_count"] == 3
        assert servers[0]["error"] is None

    def test_propagates_error_field(self, app_client):
        mgr = MagicMock()
        mgr.get_server_status.return_value = [
            _make_status(
                "broken",
                state=MCPServerState.ERROR,
                tools_count=0,
                error="connection refused",
            )
        ]
        mcp_routes.set_mcp_manager_getter(lambda: mgr)

        r = app_client.get("/v1/mcp/servers")
        body = r.json()["servers"][0]
        assert body["state"] == "error"
        assert body["error"] == "connection refused"


class TestExecuteMcpTool:
    def test_returns_503_when_no_manager(self, app_client):
        r = app_client.post(
            "/v1/mcp/execute",
            json={"tool_name": "srv__add", "arguments": {"a": 1, "b": 2}},
        )
        assert r.status_code == 503
        assert "MCP not configured" in r.json()["detail"]
        assert "--mcp-config" in r.json()["detail"]

    def test_success_returns_result_payload(self, app_client):
        result = MCPToolResult(
            tool_name="srv__add",
            content="3",
            is_error=False,
            error_message=None,
        )
        mgr = MagicMock()
        mgr.execute_tool = AsyncMock(return_value=result)
        mcp_routes.set_mcp_manager_getter(lambda: mgr)

        r = app_client.post(
            "/v1/mcp/execute",
            json={"tool_name": "srv__add", "arguments": {"a": 1, "b": 2}},
        )
        assert r.status_code == 200
        assert r.json() == {
            "tool_name": "srv__add",
            "content": "3",
            "is_error": False,
            "error_message": None,
        }
        mgr.execute_tool.assert_awaited_once_with("srv__add", {"a": 1, "b": 2})

    def test_error_result_propagates_is_error_and_message(self, app_client):
        """A handled tool error returns 200 with is_error=True — only
        unconfigured-manager raises 5xx. Lets clients distinguish 'tool
        ran and failed' from 'server can't run tools at all'."""
        result = MCPToolResult(
            tool_name="srv__broken",
            content=None,
            is_error=True,
            error_message="upstream timeout",
        )
        mgr = MagicMock()
        mgr.execute_tool = AsyncMock(return_value=result)
        mcp_routes.set_mcp_manager_getter(lambda: mgr)

        r = app_client.post(
            "/v1/mcp/execute",
            json={"tool_name": "srv__broken", "arguments": {}},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["is_error"] is True
        assert body["error_message"] == "upstream timeout"
        assert body["content"] is None

    def test_accepts_tool_alias_field(self, app_client):
        """MCPExecuteRequest declares ``tool_name`` with
        ``AliasChoices('tool_name', 'tool')`` — both wire formats must
        reach the manager identically. Upstream PR #1285 added the
        ``tool`` alias for compatibility with some external MCP clients;
        without this test a future refactor could silently drop it.
        """
        result = MCPToolResult(tool_name="srv__add", content="ok", is_error=False)
        mgr = MagicMock()
        mgr.execute_tool = AsyncMock(return_value=result)
        mcp_routes.set_mcp_manager_getter(lambda: mgr)

        r = app_client.post(
            "/v1/mcp/execute",
            json={"tool": "srv__add", "arguments": {"x": 1}},
        )
        assert r.status_code == 200
        mgr.execute_tool.assert_awaited_once_with("srv__add", {"x": 1})

    def test_arguments_default_to_empty_dict(self, app_client):
        """``arguments`` is optional — omitting it must yield ``{}``,
        not None, otherwise the manager's signature would break."""
        result = MCPToolResult(tool_name="srv__noop", content=None)
        mgr = MagicMock()
        mgr.execute_tool = AsyncMock(return_value=result)
        mcp_routes.set_mcp_manager_getter(lambda: mgr)

        r = app_client.post(
            "/v1/mcp/execute",
            json={"tool_name": "srv__noop"},
        )
        assert r.status_code == 200
        mgr.execute_tool.assert_awaited_once_with("srv__noop", {})

    def test_missing_tool_name_returns_422(self, app_client):
        """Pydantic validation must reject payloads with neither key."""
        mgr = MagicMock()
        mgr.execute_tool = AsyncMock()
        mcp_routes.set_mcp_manager_getter(lambda: mgr)

        r = app_client.post("/v1/mcp/execute", json={"arguments": {}})
        assert r.status_code == 422
        mgr.execute_tool.assert_not_awaited()
