# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP client manager (omlx/mcp/manager.py).
"""

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from omlx.mcp.client import MCPClient
from omlx.mcp.manager import MCPClientManager
from omlx.mcp.types import (
    MCPConfig,
    MCPServerConfig,
    MCPServerState,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)


class TestMCPClientManagerInit:
    """Tests for MCPClientManager initialization."""

    def test_init_empty_config(self):
        """Test initialization with empty config."""
        config = MCPConfig()
        manager = MCPClientManager(config)

        assert manager.is_started is False
        assert len(manager._clients) == 0

    def test_init_with_servers(self):
        """Test initialization creates clients for each server."""
        config = MCPConfig.from_dict({
            "servers": {
                "server1": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "server1"],
                },
                "server2": {
                    "transport": "sse",
                    "url": "http://localhost:3000",
                },
            }
        })
        manager = MCPClientManager(config)

        assert len(manager._clients) == 2
        assert "server1" in manager._clients
        assert "server2" in manager._clients
        assert isinstance(manager._clients["server1"], MCPClient)
        assert isinstance(manager._clients["server2"], MCPClient)


class TestMCPClientManagerStartStop:
    """Tests for MCPClientManager start/stop."""

    @pytest.fixture
    def manager_with_clients(self) -> MCPClientManager:
        """Create a manager with mock clients."""
        config = MCPConfig.from_dict({
            "servers": {
                "server1": {
                    "transport": "stdio",
                    "command": "python",
                },
                "server2": {
                    "transport": "stdio",
                    "command": "node",
                },
            }
        })
        return MCPClientManager(config)

    @pytest.mark.asyncio
    async def test_start_connects_all_servers(self, manager_with_clients: MCPClientManager):
        """Test start connects to all enabled servers."""
        # Mock all client connect methods
        for client in manager_with_clients._clients.values():
            client.connect = AsyncMock(return_value=True)
            client._state = MCPServerState.CONNECTED

        await manager_with_clients.start()

        assert manager_with_clients.is_started is True
        for client in manager_with_clients._clients.values():
            client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, manager_with_clients: MCPClientManager):
        """Test start is idempotent when already started."""
        manager_with_clients._started = True

        # If start is called again, it should return immediately
        await manager_with_clients.start()

        # No connection attempts should be made
        for client in manager_with_clients._clients.values():
            assert not hasattr(client, "connect") or not isinstance(
                client.connect, AsyncMock
            )

    @pytest.mark.asyncio
    async def test_start_handles_connection_failure(self):
        """Test start handles connection failures gracefully."""
        config = MCPConfig.from_dict({
            "servers": {
                "good": {
                    "transport": "stdio",
                    "command": "python",
                },
                "bad": {
                    "transport": "stdio",
                    "command": "nonexistent",
                },
            }
        })
        manager = MCPClientManager(config)

        # Mock connections
        manager._clients["good"].connect = AsyncMock(return_value=True)
        manager._clients["bad"].connect = AsyncMock(
            side_effect=ConnectionError("Failed")
        )

        await manager.start()

        assert manager.is_started is True
        # Manager should still be started even if some connections fail

    @pytest.mark.asyncio
    async def test_start_skips_disabled_servers(self):
        """Test start skips disabled servers."""
        config = MCPConfig.from_dict({
            "servers": {
                "enabled": {
                    "transport": "stdio",
                    "command": "python",
                    "enabled": True,
                },
                "disabled": {
                    "transport": "stdio",
                    "command": "python",
                    "enabled": False,
                },
            }
        })
        manager = MCPClientManager(config)

        manager._clients["enabled"].connect = AsyncMock(return_value=True)
        manager._clients["disabled"].connect = AsyncMock(return_value=False)

        await manager.start()

        manager._clients["enabled"].connect.assert_called_once()
        # Disabled server should not have connect called
        # (it's skipped in the list comprehension)

    @pytest.mark.asyncio
    async def test_stop_disconnects_all_servers(self, manager_with_clients: MCPClientManager):
        """Test stop disconnects from all servers."""
        manager_with_clients._started = True
        for client in manager_with_clients._clients.values():
            client.disconnect = AsyncMock()

        await manager_with_clients.stop()

        assert manager_with_clients.is_started is False
        for client in manager_with_clients._clients.values():
            client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, manager_with_clients: MCPClientManager):
        """Test stop is idempotent when not started."""
        manager_with_clients._started = False

        await manager_with_clients.stop()

        assert manager_with_clients.is_started is False


class TestMCPClientManagerTools:
    """Tests for MCPClientManager tool methods."""

    @pytest.fixture
    def manager_with_tools(self) -> MCPClientManager:
        """Create a manager with mock tools."""
        config = MCPConfig.from_dict({
            "servers": {
                "weather": {
                    "transport": "stdio",
                    "command": "python",
                },
                "search": {
                    "transport": "stdio",
                    "command": "node",
                },
            }
        })
        manager = MCPClientManager(config)

        # Set up mock tools
        manager._clients["weather"]._state = MCPServerState.CONNECTED
        manager._clients["weather"]._tools = [
            MCPTool(
                server_name="weather",
                name="get_weather",
                description="Get weather info",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ]

        manager._clients["search"]._state = MCPServerState.CONNECTED
        manager._clients["search"]._tools = [
            MCPTool(
                server_name="search",
                name="web_search",
                description="Search the web",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
            MCPTool(
                server_name="search",
                name="image_search",
                description="Search images",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
        ]

        return manager

    def test_get_all_tools(self, manager_with_tools: MCPClientManager):
        """Test get_all_tools returns tools from all connected servers."""
        tools = manager_with_tools.get_all_tools()

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "get_weather" in tool_names
        assert "web_search" in tool_names
        assert "image_search" in tool_names

    def test_get_all_tools_only_connected(self, manager_with_tools: MCPClientManager):
        """Test get_all_tools only returns tools from connected servers."""
        manager_with_tools._clients["weather"]._state = MCPServerState.DISCONNECTED

        tools = manager_with_tools.get_all_tools()

        assert len(tools) == 2
        assert all(t.server_name == "search" for t in tools)

    def test_get_all_tools_openai(self, manager_with_tools: MCPClientManager):
        """Test get_all_tools_openai returns OpenAI format."""
        tools = manager_with_tools.get_all_tools_openai()

        assert len(tools) == 3
        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_get_merged_tools_no_user_tools(self, manager_with_tools: MCPClientManager):
        """Test get_merged_tools without user tools."""
        tools = manager_with_tools.get_merged_tools()

        assert len(tools) == 3

    def test_get_merged_tools_with_user_tools(self, manager_with_tools: MCPClientManager):
        """Test get_merged_tools with user tools."""
        user_tools = [
            {
                "type": "function",
                "function": {
                    "name": "custom_tool",
                    "description": "A custom tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        tools = manager_with_tools.get_merged_tools(user_tools)

        assert len(tools) == 4
        tool_names = [t["function"]["name"] for t in tools]
        assert "custom_tool" in tool_names

    def test_get_merged_tools_user_override(self, manager_with_tools: MCPClientManager):
        """Test get_merged_tools where user tool overrides MCP tool."""
        # Create a user tool with same name as MCP tool
        user_tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather__get_weather",  # Same as MCP tool full name
                    "description": "Overridden description",
                    "parameters": {"type": "object"},
                },
            }
        ]

        tools = manager_with_tools.get_merged_tools(user_tools)

        # Should have 3 tools (user tool overrides MCP tool)
        assert len(tools) == 3
        weather_tool = next(
            t for t in tools if t["function"]["name"] == "weather__get_weather"
        )
        assert weather_tool["function"]["description"] == "Overridden description"


class TestMCPClientManagerServerStatus:
    """Tests for MCPClientManager server status methods."""

    @pytest.fixture
    def manager(self) -> MCPClientManager:
        """Create a manager with servers."""
        config = MCPConfig.from_dict({
            "servers": {
                "server1": {"transport": "stdio", "command": "python"},
                "server2": {"transport": "sse", "url": "http://test.com"},
            }
        })
        return MCPClientManager(config)

    def test_get_server_status(self, manager: MCPClientManager):
        """Test get_server_status returns status for all servers."""
        statuses = manager.get_server_status()

        assert len(statuses) == 2
        names = [s.name for s in statuses]
        assert "server1" in names
        assert "server2" in names

    def test_get_client(self, manager: MCPClientManager):
        """Test get_client returns correct client."""
        client = manager.get_client("server1")

        assert client is not None
        assert client.name == "server1"

    def test_get_client_not_found(self, manager: MCPClientManager):
        """Test get_client returns None for unknown server."""
        client = manager.get_client("nonexistent")

        assert client is None


class TestMCPClientManagerExecuteTool:
    """Tests for MCPClientManager execute_tool methods."""

    @pytest.fixture
    def manager_with_connected_client(self) -> MCPClientManager:
        """Create a manager with a connected client."""
        config = MCPConfig.from_dict({
            "servers": {
                "test": {
                    "transport": "stdio",
                    "command": "python",
                },
            },
            "default_timeout": 30.0,
        })
        manager = MCPClientManager(config)
        manager._clients["test"]._state = MCPServerState.CONNECTED
        manager._clients["test"]._tools = [
            MCPTool(server_name="test", name="my_tool", description="Test tool"),
        ]
        return manager

    @pytest.mark.asyncio
    async def test_execute_tool_with_full_name(
        self, manager_with_connected_client: MCPClientManager
    ):
        """Test execute_tool with server__tool format."""
        manager_with_connected_client._clients["test"].call_tool = AsyncMock(
            return_value=MCPToolResult(
                tool_name="my_tool",
                content="Success",
            )
        )

        result = await manager_with_connected_client.execute_tool(
            "test__my_tool", {"arg": "value"}
        )

        assert result.is_error is False
        assert result.content == "Success"

    @pytest.mark.asyncio
    async def test_execute_tool_without_server_prefix(
        self, manager_with_connected_client: MCPClientManager
    ):
        """Test execute_tool finds server when no prefix given."""
        manager_with_connected_client._clients["test"].call_tool = AsyncMock(
            return_value=MCPToolResult(
                tool_name="my_tool",
                content="Found it",
            )
        )

        result = await manager_with_connected_client.execute_tool("my_tool", {})

        assert result.is_error is False
        assert result.content == "Found it"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(
        self, manager_with_connected_client: MCPClientManager
    ):
        """Test execute_tool returns error when tool not found."""
        result = await manager_with_connected_client.execute_tool(
            "nonexistent_tool", {}
        )

        assert result.is_error is True
        assert "not found" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_tool_server_not_found(
        self, manager_with_connected_client: MCPClientManager
    ):
        """Test execute_tool returns error when server not found."""
        result = await manager_with_connected_client.execute_tool(
            "unknown_server__tool", {}
        )

        assert result.is_error is True
        assert "not found" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_tool_server_not_connected(
        self, manager_with_connected_client: MCPClientManager
    ):
        """Test execute_tool returns error when server not connected."""
        manager_with_connected_client._clients["test"]._state = MCPServerState.DISCONNECTED

        result = await manager_with_connected_client.execute_tool(
            "test__my_tool", {}
        )

        assert result.is_error is True
        assert "not connected" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_tool_call(
        self, manager_with_connected_client: MCPClientManager
    ):
        """Test execute_tool_call with OpenAI format."""
        manager_with_connected_client._clients["test"].call_tool = AsyncMock(
            return_value=MCPToolResult(
                tool_name="my_tool",
                content="Result",
            )
        )

        tool_call = {
            "id": "call_123",
            "function": {
                "name": "test__my_tool",
                "arguments": '{"key": "value"}',
            },
        }

        result = await manager_with_connected_client.execute_tool_call(tool_call)

        assert result.is_error is False
        assert result.content == "Result"


class TestMCPClientManagerRefreshReconnect:
    """Tests for MCPClientManager refresh and reconnect methods."""

    @pytest.fixture
    def manager(self) -> MCPClientManager:
        """Create a manager with clients."""
        config = MCPConfig.from_dict({
            "servers": {
                "server1": {"transport": "stdio", "command": "python"},
                "server2": {"transport": "stdio", "command": "node"},
            }
        })
        manager = MCPClientManager(config)
        for client in manager._clients.values():
            client._state = MCPServerState.CONNECTED
            client.refresh_tools = AsyncMock()
            client.disconnect = AsyncMock()
            client.connect = AsyncMock(return_value=True)
        return manager

    @pytest.mark.asyncio
    async def test_refresh_tools(self, manager: MCPClientManager):
        """Test refresh_tools refreshes all connected clients."""
        await manager.refresh_tools()

        for client in manager._clients.values():
            client.refresh_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_tools_skips_disconnected(self, manager: MCPClientManager):
        """Test refresh_tools skips disconnected clients."""
        manager._clients["server1"]._state = MCPServerState.DISCONNECTED

        await manager.refresh_tools()

        manager._clients["server1"].refresh_tools.assert_not_called()
        manager._clients["server2"].refresh_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_specific_server(self, manager: MCPClientManager):
        """Test reconnect to specific server."""
        await manager.reconnect("server1")

        manager._clients["server1"].disconnect.assert_called_once()
        manager._clients["server1"].connect.assert_called_once()
        manager._clients["server2"].disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconnect_all_servers(self, manager: MCPClientManager):
        """Test reconnect to all servers."""
        await manager.reconnect()

        for client in manager._clients.values():
            client.disconnect.assert_called_once()
            client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_unknown_server(self, manager: MCPClientManager):
        """Test reconnect with unknown server name does nothing."""
        await manager.reconnect("nonexistent")

        # Should not raise and no clients should be affected
        for client in manager._clients.values():
            client.disconnect.assert_not_called()


class TestInitMCPGracefulFallback:
    """Tests for init_mcp() graceful fallback on config errors (issue #474)."""

    @pytest.mark.asyncio
    async def test_invalid_json_does_not_crash(self, tmp_path):
        """init_mcp should not raise on malformed JSON config."""
        bad_config = tmp_path / "mcp.json"
        bad_config.write_text("{invalid json")

        from omlx.server import init_mcp

        # Should return gracefully, not raise
        await init_mcp(str(bad_config))

    @pytest.mark.asyncio
    async def test_missing_file_does_not_crash(self):
        """init_mcp should not raise on nonexistent config path."""
        from omlx.server import init_mcp

        await init_mcp("/nonexistent/mcp.json")

    @pytest.mark.asyncio
    async def test_invalid_config_structure_does_not_crash(self, tmp_path):
        """init_mcp should not raise on invalid config structure."""
        bad_config = tmp_path / "mcp.json"
        bad_config.write_text(json.dumps("not a dict"))

        from omlx.server import init_mcp

        await init_mcp(str(bad_config))
