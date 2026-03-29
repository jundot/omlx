# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Type definitions for MCP client support.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class MCPTransport(str, Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


class MCPServerState(str, Enum):
    """MCP server connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    name: str
    transport: MCPTransport = MCPTransport.STDIO

    # For stdio transport
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    # For SSE transport
    url: Optional[str] = None

    # For streamable-http transport
    headers: Optional[Dict[str, str]] = None

    # Common options
    enabled: bool = True
    timeout: float = 30.0

    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.transport, str):
            self.transport = MCPTransport(self.transport)

        if self.transport == MCPTransport.STDIO:
            if not self.command:
                raise ValueError(f"MCP server '{self.name}': stdio transport requires 'command'")
        elif self.transport == MCPTransport.SSE:
            if not self.url:
                raise ValueError(f"MCP server '{self.name}': sse transport requires 'url'")
        elif self.transport == MCPTransport.STREAMABLE_HTTP:
            if not self.url:
                raise ValueError(f"MCP server '{self.name}': streamable-http transport requires 'url'")


@dataclass
class MCPConfig:
    """Root configuration for MCP client."""

    servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    max_tool_calls: int = 10
    default_timeout: float = 30.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPConfig":
        """Create config from dictionary."""
        servers = {}
        for name, server_data in data.get("servers", {}).items():
            server_data["name"] = name
            servers[name] = MCPServerConfig(**server_data)

        return cls(
            servers=servers,
            max_tool_calls=data.get("max_tool_calls", 10),
            default_timeout=data.get("default_timeout", 30.0),
        )


@dataclass
class MCPTool:
    """Normalized tool representation from MCP server."""

    server_name: str
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get namespaced tool name (server__tool)."""
        return f"{self.server_name}__{self.name}"

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.full_name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        }


@dataclass
class MCPToolResult:
    """Result from a tool execution."""

    tool_name: str
    content: Any
    is_error: bool = False
    error_message: Optional[str] = None

    def to_message(self, tool_call_id: str) -> Dict[str, Any]:
        """Convert to OpenAI tool result message format."""
        if self.is_error:
            content = f"Error: {self.error_message}"
        elif isinstance(self.content, str):
            content = self.content
        else:
            import json
            content = json.dumps(self.content)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }


@dataclass
class MCPServerStatus:
    """Status of an MCP server connection."""

    name: str
    state: MCPServerState
    transport: MCPTransport
    tools_count: int = 0
    error: Optional[str] = None
    last_connected: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "state": self.state.value,
            "transport": self.transport.value,
            "tools_count": self.tools_count,
            "error": self.error,
            "last_connected": self.last_connected,
        }
