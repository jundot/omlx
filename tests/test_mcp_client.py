# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP client (omlx/mcp/client.py).

These tests mock the MCP SDK to test client logic without
requiring actual MCP server connections.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from omlx.mcp.client import MCPClient
from omlx.mcp.types import (
    MCPServerConfig,
    MCPServerState,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)


class TestMCPClientInit:
    """Tests for MCPClient initialization."""

    def test_client_init_stdio(self):
        """Test client initialization with stdio config."""
        config = MCPServerConfig(
            name="test-server",
            transport=MCPTransport.STDIO,
            command="python",
            args=["-m", "server"],
        )
        client = MCPClient(config)

        assert client.name == "test-server"
        assert client.state == MCPServerState.DISCONNECTED
        assert client.is_connected is False
        assert client.tools == []

    def test_client_init_sse(self):
        """Test client initialization with SSE config."""
        config = MCPServerConfig(
            name="sse-server",
            transport=MCPTransport.SSE,
            url="http://localhost:3000/mcp",
        )
        client = MCPClient(config)

        assert client.name == "sse-server"
        assert client.config.url == "http://localhost:3000/mcp"

    def test_client_init_streamable_http(self):
        """Test client initialization with streamable_http config."""
        config = MCPServerConfig(
            name="streamable-server",
            transport=MCPTransport.STREAMABLE_HTTP,
            url="http://localhost:3000/mcp",
        )
        client = MCPClient(config)

        assert client.name == "streamable-server"
        assert client.config.url == "http://localhost:3000/mcp"


class TestMCPClientProperties:
    """Tests for MCPClient properties."""

    @pytest.fixture
    def client(self) -> MCPClient:
        """Create a test client."""
        config = MCPServerConfig(
            name="test",
            transport=MCPTransport.STDIO,
            command="echo",
        )
        return MCPClient(config)

    def test_name_property(self, client: MCPClient):
        """Test name property."""
        assert client.name == "test"

    def test_state_property(self, client: MCPClient):
        """Test state property."""
        assert client.state == MCPServerState.DISCONNECTED

        client._state = MCPServerState.CONNECTED
        assert client.state == MCPServerState.CONNECTED

    def test_is_connected_property(self, client: MCPClient):
        """Test is_connected property."""
        assert client.is_connected is False

        client._state = MCPServerState.CONNECTED
        assert client.is_connected is True

        client._state = MCPServerState.ERROR
        assert client.is_connected is False

    def test_tools_property(self, client: MCPClient):
        """Test tools property."""
        assert client.tools == []

        tool = MCPTool(
            server_name="test",
            name="my_tool",
            description="Test tool",
        )
        client._tools = [tool]
        assert len(client.tools) == 1
        assert client.tools[0].name == "my_tool"


class TestMCPClientStatus:
    """Tests for MCPClient.get_status()."""

    def test_get_status_disconnected(self):
        """Test get_status when disconnected."""
        config = MCPServerConfig(
            name="status-test",
            transport=MCPTransport.STDIO,
            command="python",
        )
        client = MCPClient(config)
        status = client.get_status()

        assert status.name == "status-test"
        assert status.state == MCPServerState.DISCONNECTED
        assert status.transport == MCPTransport.STDIO
        assert status.tools_count == 0
        assert status.error is None
        assert status.last_connected is None

    def test_get_status_connected_with_tools(self):
        """Test get_status when connected with tools."""
        config = MCPServerConfig(
            name="connected",
            transport=MCPTransport.SSE,
            url="http://test.com",
        )
        client = MCPClient(config)
        client._state = MCPServerState.CONNECTED
        client._tools = [
            MCPTool(server_name="connected", name="tool1", description=""),
            MCPTool(server_name="connected", name="tool2", description=""),
        ]
        client._last_connected = 1234567890.0

        status = client.get_status()

        assert status.state == MCPServerState.CONNECTED
        assert status.tools_count == 2
        assert status.last_connected == 1234567890.0

    def test_get_status_with_error(self):
        """Test get_status when in error state."""
        config = MCPServerConfig(
            name="error-server",
            transport=MCPTransport.STDIO,
            command="bad",
        )
        client = MCPClient(config)
        client._state = MCPServerState.ERROR
        client._error = "Connection failed"

        status = client.get_status()

        assert status.state == MCPServerState.ERROR
        assert status.error == "Connection failed"


class TestMCPClientConnect:
    """Tests for MCPClient.connect()."""

    @pytest.fixture
    def stdio_client(self) -> MCPClient:
        """Create a stdio client for testing."""
        config = MCPServerConfig(
            name="stdio-test",
            transport=MCPTransport.STDIO,
            command="python",
            args=["-m", "mcp_server"],
        )
        return MCPClient(config)

    @pytest.fixture
    def sse_client(self) -> MCPClient:
        """Create an SSE client for testing."""
        config = MCPServerConfig(
            name="sse-test",
            transport=MCPTransport.SSE,
            url="http://localhost:3000/mcp",
        )
        return MCPClient(config)

    @pytest.fixture
    def streamable_http_client(self) -> MCPClient:
        """Create a streamable_http client for testing."""
        config = MCPServerConfig(
            name="streamable-http-test",
            transport=MCPTransport.STREAMABLE_HTTP,
            url="http://localhost:3000/mcp",
        )
        return MCPClient(config)

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, stdio_client: MCPClient):
        """Test connect returns True when already connected."""
        stdio_client._state = MCPServerState.CONNECTED

        result = await stdio_client.connect()

        assert result is True
        assert stdio_client.state == MCPServerState.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_disabled_server(self):
        """Test connect returns False for disabled server."""
        config = MCPServerConfig(
            name="disabled",
            transport=MCPTransport.STDIO,
            command="python",
            enabled=False,
        )
        client = MCPClient(config)

        result = await client.connect()

        assert result is False
        assert client.state == MCPServerState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_stdio_success(self, stdio_client: MCPClient):
        """Test successful stdio connection."""
        # Mock the internal methods instead of trying to patch imports
        with patch.object(stdio_client, "_connect_stdio", new_callable=AsyncMock) as mock_connect, \
             patch.object(stdio_client, "_initialize_session", new_callable=AsyncMock), \
             patch.object(stdio_client, "_discover_tools", new_callable=AsyncMock):
            mock_connect.return_value = None
            # Set up session to pass the check
            stdio_client._session = MagicMock()

            result = await stdio_client.connect()

        assert result is True
        assert stdio_client.state == MCPServerState.CONNECTED
        assert stdio_client._last_connected is not None

    @pytest.mark.asyncio
    async def test_connect_sse_success(self, sse_client: MCPClient):
        """Test successful SSE connection."""
        with patch.object(sse_client, "_connect_sse", new_callable=AsyncMock), \
             patch.object(sse_client, "_initialize_session", new_callable=AsyncMock), \
             patch.object(sse_client, "_discover_tools", new_callable=AsyncMock):
            sse_client._session = MagicMock()

            result = await sse_client.connect()

        assert result is True
        assert sse_client.state == MCPServerState.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_streamable_http_success(
        self, streamable_http_client: MCPClient
    ):
        """Test successful streamable_http connection."""
        with (
            patch.object(
                streamable_http_client,
                "_connect_streamable_http",
                new_callable=AsyncMock,
            ),
            patch.object(
                streamable_http_client, "_initialize_session", new_callable=AsyncMock
            ),
            patch.object(
                streamable_http_client, "_discover_tools", new_callable=AsyncMock
            ),
        ):
            streamable_http_client._session = MagicMock()

            result = await streamable_http_client.connect()

        assert result is True
        assert streamable_http_client.state == MCPServerState.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_failure(self, stdio_client: MCPClient):
        """Test connection failure sets error state."""
        with patch.object(
            stdio_client, "_connect_stdio", new_callable=AsyncMock
        ) as mock_connect:
            mock_connect.side_effect = ConnectionError("Connection refused")

            result = await stdio_client.connect()

        assert result is False
        assert stdio_client.state == MCPServerState.ERROR
        assert stdio_client._error == "Connection refused"

    @pytest.mark.asyncio
    async def test_connect_import_error(self, stdio_client: MCPClient):
        """Test connection fails gracefully when MCP SDK not installed."""
        # Remove _connect_stdio to test original implementation
        original = MCPClient._connect_stdio

        async def mock_connect_stdio(self):
            raise ImportError("MCP SDK required")

        with patch.object(MCPClient, "_connect_stdio", mock_connect_stdio):
            result = await stdio_client.connect()

        assert result is False
        assert stdio_client.state == MCPServerState.ERROR
        assert "MCP SDK required" in stdio_client._error


class TestMCPClientDisconnect:
    """Tests for MCPClient.disconnect()."""

    @pytest.fixture
    def connected_client(self) -> MCPClient:
        """Create a connected client for testing."""
        config = MCPServerConfig(
            name="connected",
            transport=MCPTransport.STDIO,
            command="python",
        )
        client = MCPClient(config)
        client._state = MCPServerState.CONNECTED
        client._session = AsyncMock()
        client._tools = [MCPTool(server_name="connected", name="tool", description="")]
        return client

    @pytest.mark.asyncio
    async def test_disconnect_already_disconnected(self):
        """Test disconnect when already disconnected."""
        config = MCPServerConfig(
            name="test",
            transport=MCPTransport.STDIO,
            command="python",
        )
        client = MCPClient(config)

        await client.disconnect()

        assert client.state == MCPServerState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_clears_session(self, connected_client: MCPClient):
        """Test disconnect clears session and tools."""
        await connected_client.disconnect()

        assert connected_client.state == MCPServerState.DISCONNECTED
        assert connected_client._session is None
        assert connected_client.tools == []

    @pytest.mark.asyncio
    async def test_disconnect_handles_session_error(self, connected_client: MCPClient):
        """Test disconnect handles session cleanup errors."""
        connected_client._session.__aexit__.side_effect = RuntimeError("Cleanup error")

        # Should not raise
        await connected_client.disconnect()

        assert connected_client.state == MCPServerState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_cleans_stdio_client(self, connected_client: MCPClient):
        """Test disconnect cleans up stdio client."""
        mock_stdio = AsyncMock()
        connected_client._stdio_client = mock_stdio

        await connected_client.disconnect()

        mock_stdio.__aexit__.assert_called_once()
        assert connected_client._stdio_client is None

    @pytest.mark.asyncio
    async def test_disconnect_cleans_sse_client(self):
        """Test disconnect cleans up SSE client."""
        config = MCPServerConfig(
            name="sse",
            transport=MCPTransport.SSE,
            url="http://test.com",
        )
        client = MCPClient(config)
        client._state = MCPServerState.CONNECTED
        client._session = AsyncMock()
        mock_sse = AsyncMock()
        client._sse_client = mock_sse

        await client.disconnect()

        mock_sse.__aexit__.assert_called_once()
        assert client._sse_client is None

    @pytest.mark.asyncio
    async def test_disconnect_cleans_streamable_http_client(self):
        """Test disconnect cleans up streamable_http client."""
        config = MCPServerConfig(
            name="streamable-http",
            transport=MCPTransport.STREAMABLE_HTTP,
            url="http://test.com",
        )
        client = MCPClient(config)
        client._state = MCPServerState.CONNECTED
        client._session = AsyncMock()
        mock_streamable_http = AsyncMock()
        mock_http_client = AsyncMock()
        client._streamable_http_client = mock_streamable_http
        client._http_client = mock_http_client

        await client.disconnect()

        mock_streamable_http.__aexit__.assert_called_once()
        mock_http_client.__aexit__.assert_called_once()
        assert client._streamable_http_client is None
        assert client._http_client is None

    @pytest.mark.asyncio
    async def test_connect_failure_cleans_up_resources(self):
        """Test that connect() cleans up partial resources on failure."""
        config = MCPServerConfig(
            name="cleanup-test",
            transport=MCPTransport.STDIO,
            command="python",
        )
        client = MCPClient(config)

        mock_stdio = AsyncMock()

        with (
            patch.object(client, "_connect_stdio", new_callable=AsyncMock) as mock_connect,
            patch.object(client, "_initialize_session", new_callable=AsyncMock) as mock_init,
        ):
            async def setup_partial_resources():
                client._stdio_client = mock_stdio
                client._session = AsyncMock()

            mock_connect.side_effect = setup_partial_resources
            mock_init.side_effect = RuntimeError("Init failed")

            result = await client.connect()

        assert result is False
        assert client.state == MCPServerState.ERROR
        # Partial resources should be cleaned up
        assert client._session is None

    @pytest.mark.asyncio
    async def test_streamable_http_partial_connect_cleanup(self):
        """Test streamable-http cleans up http_client on session failure."""
        config = MCPServerConfig(
            name="partial-cleanup",
            transport=MCPTransport.STREAMABLE_HTTP,
            url="http://localhost:3000/mcp",
        )
        client = MCPClient(config)

        mock_http_client = AsyncMock()
        mock_streamable = AsyncMock()

        with (
            patch("omlx.mcp.client.MCPClient._connect_streamable_http") as mock_connect,
            patch.object(client, "_initialize_session", new_callable=AsyncMock) as mock_init,
        ):
            async def setup_and_fail():
                client._http_client = mock_http_client
                client._streamable_http_client = mock_streamable
                client._session = AsyncMock()
                raise RuntimeError("Connection failed midway")

            mock_connect.side_effect = setup_and_fail

            result = await client.connect()

        assert result is False
        # Resources should be cleaned up via _cleanup_resources
        assert client._http_client is None
        assert client._streamable_http_client is None


class TestMCPClientCallTool:
    """Tests for MCPClient.call_tool()."""

    @pytest.fixture
    def connected_client(self) -> MCPClient:
        """Create a connected client with mock session."""
        config = MCPServerConfig(
            name="tool-test",
            transport=MCPTransport.STDIO,
            command="python",
            timeout=5.0,
        )
        client = MCPClient(config)
        client._state = MCPServerState.CONNECTED
        client._session = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        """Test call_tool fails when not connected."""
        config = MCPServerConfig(
            name="test",
            transport=MCPTransport.STDIO,
            command="python",
        )
        client = MCPClient(config)

        result = await client.call_tool("my_tool", {"arg": "value"})

        assert result.is_error is True
        assert "Not connected" in result.error_message

    @pytest.mark.asyncio
    async def test_call_tool_no_session(self, connected_client: MCPClient):
        """Test call_tool fails when session is None."""
        connected_client._session = None

        result = await connected_client.call_tool("my_tool", {})

        assert result.is_error is True
        assert "Session not initialized" in result.error_message

    @pytest.mark.asyncio
    async def test_call_tool_success(self, connected_client: MCPClient):
        """Test successful tool call."""
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="Tool output")]
        mock_result.isError = False
        connected_client._session.call_tool.return_value = mock_result

        result = await connected_client.call_tool("get_data", {"id": 123})

        assert result.is_error is False
        assert result.content == "Tool output"
        assert result.tool_name == "get_data"
        connected_client._session.call_tool.assert_called_with("get_data", {"id": 123})

    @pytest.mark.asyncio
    async def test_call_tool_timeout(self, connected_client: MCPClient):
        """Test tool call timeout."""
        async def slow_call(*args, **kwargs):
            await asyncio.sleep(10)

        connected_client._session.call_tool = slow_call

        result = await connected_client.call_tool("slow_tool", {}, timeout=0.1)

        assert result.is_error is True
        assert "timed out" in result.error_message

    @pytest.mark.asyncio
    async def test_call_tool_uses_config_timeout(self, connected_client: MCPClient):
        """Test tool call uses config timeout when not specified."""
        mock_result = MagicMock()
        mock_result.content = []
        mock_result.isError = False
        connected_client._session.call_tool.return_value = mock_result

        await connected_client.call_tool("tool", {})

        # Should use the config timeout (5.0)
        # The actual timeout is handled by asyncio.wait_for

    @pytest.mark.asyncio
    async def test_call_tool_exception(self, connected_client: MCPClient):
        """Test tool call exception handling."""
        connected_client._session.call_tool.side_effect = RuntimeError("Tool error")

        result = await connected_client.call_tool("bad_tool", {})

        assert result.is_error is True
        assert result.error_message == "Tool error"

    @pytest.mark.asyncio
    async def test_call_tool_multiple_content_items(self, connected_client: MCPClient):
        """Test tool call with multiple content items."""
        mock_result = MagicMock()
        mock_result.content = [
            MagicMock(text="Line 1"),
            MagicMock(text="Line 2"),
        ]
        mock_result.isError = False
        connected_client._session.call_tool.return_value = mock_result

        result = await connected_client.call_tool("multi_tool", {})

        assert result.is_error is False
        assert result.content == ["Line 1", "Line 2"]

    @pytest.mark.asyncio
    async def test_call_tool_data_content(self, connected_client: MCPClient):
        """Test tool call with data content."""
        mock_result = MagicMock()
        mock_item = MagicMock(spec=["data"])
        mock_item.data = {"key": "value"}
        del mock_item.text  # Remove text attribute
        mock_result.content = [mock_item]
        mock_result.isError = False
        connected_client._session.call_tool.return_value = mock_result

        result = await connected_client.call_tool("data_tool", {})

        assert result.content == {"key": "value"}

    @pytest.mark.asyncio
    async def test_call_tool_structured_content(self, connected_client: MCPClient):
        """Test tool call with structuredContent fallback."""
        mock_result = MagicMock(spec=[])
        mock_result.content = []
        mock_result.structuredContent = {"results": ["Result 1", "Result 2"]}
        mock_result.isError = False
        connected_client._session.call_tool.return_value = mock_result

        result = await connected_client.call_tool("web_search", {"query": "test"})

        assert result.is_error is False
        assert result.content == {"results": ["Result 1", "Result 2"]}
        connected_client._session.call_tool.assert_called_with(
            "web_search", {"query": "test"}
        )


class TestMCPClientRefreshTools:
    """Tests for MCPClient.refresh_tools()."""

    @pytest.mark.asyncio
    async def test_refresh_tools_not_connected(self):
        """Test refresh_tools does nothing when not connected."""
        config = MCPServerConfig(
            name="test",
            transport=MCPTransport.STDIO,
            command="python",
        )
        client = MCPClient(config)
        client._tools = [MCPTool(server_name="test", name="old", description="")]

        await client.refresh_tools()

        # Tools should remain unchanged
        assert len(client.tools) == 1

    @pytest.mark.asyncio
    async def test_refresh_tools_connected(self):
        """Test refresh_tools updates tools when connected."""
        config = MCPServerConfig(
            name="test",
            transport=MCPTransport.STDIO,
            command="python",
        )
        client = MCPClient(config)
        client._state = MCPServerState.CONNECTED
        client._session = AsyncMock()

        # Create a proper mock tool object with correct attributes
        mock_tool = MagicMock()
        mock_tool.name = "new_tool"  # Set name explicitly, not as MagicMock
        mock_tool.description = "New"
        mock_tool.inputSchema = {}

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        client._session.list_tools.return_value = mock_result

        await client.refresh_tools()

        assert len(client.tools) == 1
        assert client.tools[0].name == "new_tool"
