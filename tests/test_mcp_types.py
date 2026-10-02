# SPDX-License-Identifier: Apache-2.0
"""
Tests for MCP type definitions (omlx/mcp/types.py).
"""

import pytest

from omlx.mcp.types import (
    MCPConfig,
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)


class TestMCPTransport:
    """Tests for MCPTransport enum."""

    def test_transport_values(self):
        """Test transport enum values."""
        assert MCPTransport.STDIO.value == "stdio"
        assert MCPTransport.SSE.value == "sse"

    def test_transport_is_string_enum(self):
        """Test transport enum is string-based."""
        assert str(MCPTransport.STDIO) == "MCPTransport.STDIO"
        assert MCPTransport.STDIO == "stdio"


class TestMCPServerState:
    """Tests for MCPServerState enum."""

    def test_state_values(self):
        """Test server state enum values."""
        assert MCPServerState.DISCONNECTED.value == "disconnected"
        assert MCPServerState.CONNECTING.value == "connecting"
        assert MCPServerState.CONNECTED.value == "connected"
        assert MCPServerState.ERROR.value == "error"

    def test_state_comparison(self):
        """Test state enum comparison."""
        assert MCPServerState.CONNECTED == MCPServerState.CONNECTED
        assert MCPServerState.CONNECTED != MCPServerState.DISCONNECTED


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_stdio_config_valid(self):
        """Test valid stdio transport configuration."""
        config = MCPServerConfig(
            name="test-server",
            transport=MCPTransport.STDIO,
            command="python",
            args=["-m", "mcp_server"],
            env={"API_KEY": "secret"},
        )
        assert config.name == "test-server"
        assert config.transport == MCPTransport.STDIO
        assert config.command == "python"
        assert config.args == ["-m", "mcp_server"]
        assert config.env == {"API_KEY": "secret"}
        assert config.enabled is True
        assert config.timeout == 30.0

    def test_sse_config_valid(self):
        """Test valid SSE transport configuration."""
        config = MCPServerConfig(
            name="sse-server",
            transport=MCPTransport.SSE,
            url="http://localhost:3000/mcp",
        )
        assert config.name == "sse-server"
        assert config.transport == MCPTransport.SSE
        assert config.url == "http://localhost:3000/mcp"

    def test_stdio_config_requires_command(self):
        """Test stdio transport requires command."""
        with pytest.raises(ValueError, match="stdio transport requires 'command'"):
            MCPServerConfig(
                name="bad-stdio",
                transport=MCPTransport.STDIO,
                # Missing command
            )

    def test_sse_config_requires_url(self):
        """Test SSE transport requires URL."""
        with pytest.raises(ValueError, match="sse transport requires 'url'"):
            MCPServerConfig(
                name="bad-sse",
                transport=MCPTransport.SSE,
                # Missing url
            )

    def test_transport_string_conversion(self):
        """Test transport string is converted to enum."""
        config = MCPServerConfig(
            name="test",
            transport="stdio",  # type: ignore
            command="echo",
        )
        assert config.transport == MCPTransport.STDIO

    def test_disabled_config(self):
        """Test disabled server configuration."""
        config = MCPServerConfig(
            name="disabled-server",
            transport=MCPTransport.STDIO,
            command="python",
            enabled=False,
        )
        assert config.enabled is False

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        config = MCPServerConfig(
            name="slow-server",
            transport=MCPTransport.STDIO,
            command="python",
            timeout=120.0,
        )
        assert config.timeout == 120.0

    def test_stdio_config_with_cwd(self):
        """Test stdio config accepts a working directory (#1111)."""
        config = MCPServerConfig(
            name="cwd-server",
            transport=MCPTransport.STDIO,
            command="node",
            args=["server.js"],
            cwd="/srv/mcp/cwd-server",
        )
        assert config.cwd == "/srv/mcp/cwd-server"

    def test_cwd_defaults_to_none(self):
        """Test cwd defaults to None when omitted."""
        config = MCPServerConfig(
            name="no-cwd",
            transport=MCPTransport.STDIO,
            command="python",
        )
        assert config.cwd is None


class TestMCPConfig:
    """Tests for MCPConfig dataclass."""

    def test_default_config(self):
        """Test default MCP configuration."""
        config = MCPConfig()
        assert config.servers == {}
        assert config.max_tool_calls == 10
        assert config.default_timeout == 30.0

    def test_config_with_servers(self):
        """Test configuration with servers."""
        server = MCPServerConfig(
            name="test",
            transport=MCPTransport.STDIO,
            command="python",
        )
        config = MCPConfig(
            servers={"test": server},
            max_tool_calls=5,
            default_timeout=60.0,
        )
        assert len(config.servers) == 1
        assert "test" in config.servers
        assert config.max_tool_calls == 5
        assert config.default_timeout == 60.0

    def test_from_dict_empty(self):
        """Test from_dict with empty data."""
        config = MCPConfig.from_dict({})
        assert config.servers == {}
        assert config.max_tool_calls == 10
        assert config.default_timeout == 30.0

    def test_from_dict_with_servers(self):
        """Test from_dict with server definitions."""
        data = {
            "servers": {
                "python-server": {
                    "transport": "stdio",
                    "command": "python",
                    "args": ["-m", "mcp_server"],
                },
                "web-server": {
                    "transport": "sse",
                    "url": "http://localhost:3000/mcp",
                },
            },
            "max_tool_calls": 20,
            "default_timeout": 45.0,
        }
        config = MCPConfig.from_dict(data)

        assert len(config.servers) == 2
        assert "python-server" in config.servers
        assert "web-server" in config.servers
        assert config.servers["python-server"].command == "python"
        assert config.servers["web-server"].url == "http://localhost:3000/mcp"
        assert config.max_tool_calls == 20
        assert config.default_timeout == 45.0

    def test_from_dict_server_with_cwd(self):
        """Test 'cwd' survives the **server_data expansion (#1111).

        Before the fix the unknown kwarg made MCPServerConfig raise
        TypeError, which disabled MCP for the whole config.
        """
        data = {
            "servers": {
                "local-tools": {
                    "transport": "stdio",
                    "command": "node",
                    "args": ["server.js"],
                    "cwd": "/srv/mcp/local-tools",
                },
            },
        }
        config = MCPConfig.from_dict(data)
        assert config.servers["local-tools"].cwd == "/srv/mcp/local-tools"


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_tool_creation(self):
        """Test basic tool creation."""
        tool = MCPTool(
            server_name="test-server",
            name="get_weather",
            description="Get weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )
        assert tool.server_name == "test-server"
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a location"
        assert "location" in tool.input_schema["properties"]

    def test_tool_default_schema(self):
        """Test tool with default empty schema."""
        tool = MCPTool(
            server_name="server",
            name="simple_tool",
            description="A simple tool",
        )
        assert tool.input_schema == {}

    def test_full_name(self):
        """Test full_name property."""
        tool = MCPTool(
            server_name="my_server",
            name="my_tool",
            description="Test",
        )
        assert tool.full_name == "my_server__my_tool"

    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        tool = MCPTool(
            server_name="weather",
            name="get_forecast",
            description="Get weather forecast",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        )
        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "weather__get_forecast"
        assert openai_format["function"]["description"] == "Get weather forecast"
        assert openai_format["function"]["parameters"]["properties"]["city"]["type"] == "string"


class TestMCPToolResult:
    """Tests for MCPToolResult dataclass."""

    def test_successful_result(self):
        """Test successful tool result."""
        result = MCPToolResult(
            tool_name="get_weather",
            content="Sunny, 25C",
            is_error=False,
        )
        assert result.tool_name == "get_weather"
        assert result.content == "Sunny, 25C"
        assert result.is_error is False
        assert result.error_message is None

    def test_error_result(self):
        """Test error tool result."""
        result = MCPToolResult(
            tool_name="get_weather",
            content=None,
            is_error=True,
            error_message="City not found",
        )
        assert result.is_error is True
        assert result.error_message == "City not found"

    def test_to_message_success_string(self):
        """Test to_message with string content."""
        result = MCPToolResult(
            tool_name="test_tool",
            content="Result text",
        )
        message = result.to_message("call-123")

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call-123"
        assert message["content"] == "Result text"

    def test_to_message_success_dict(self):
        """Test to_message with dict content."""
        result = MCPToolResult(
            tool_name="test_tool",
            content={"temperature": 25, "unit": "celsius"},
        )
        message = result.to_message("call-456")

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call-456"
        # Content should be JSON string
        assert '"temperature": 25' in message["content"]

    def test_to_message_error(self):
        """Test to_message with error."""
        result = MCPToolResult(
            tool_name="test_tool",
            content=None,
            is_error=True,
            error_message="Something went wrong",
        )
        message = result.to_message("call-789")

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call-789"
        assert message["content"] == "Error: Something went wrong"


class TestMCPServerStatus:
    """Tests for MCPServerStatus dataclass."""

    def test_status_creation(self):
        """Test server status creation."""
        status = MCPServerStatus(
            name="test-server",
            state=MCPServerState.CONNECTED,
            transport=MCPTransport.STDIO,
            tools_count=5,
            last_connected=1234567890.0,
        )
        assert status.name == "test-server"
        assert status.state == MCPServerState.CONNECTED
        assert status.transport == MCPTransport.STDIO
        assert status.tools_count == 5
        assert status.error is None
        assert status.last_connected == 1234567890.0

    def test_status_with_error(self):
        """Test server status with error."""
        status = MCPServerStatus(
            name="failed-server",
            state=MCPServerState.ERROR,
            transport=MCPTransport.SSE,
            error="Connection refused",
        )
        assert status.state == MCPServerState.ERROR
        assert status.error == "Connection refused"

    def test_to_dict(self):
        """Test status to_dict conversion."""
        status = MCPServerStatus(
            name="server",
            state=MCPServerState.CONNECTED,
            transport=MCPTransport.STDIO,
            tools_count=3,
            last_connected=9999.0,
        )
        result = status.to_dict()

        assert result["name"] == "server"
        assert result["state"] == "connected"
        assert result["transport"] == "stdio"
        assert result["tools_count"] == 3
        assert result["error"] is None
        assert result["last_connected"] == 9999.0
