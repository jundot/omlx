# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
MCP client for connecting to individual MCP servers.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .types import (
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for connecting to a single MCP server.

    Supports both stdio and SSE transports.
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize MCP client.

        Args:
            config: Server configuration
        """
        self.config = config
        self._session = None
        self._read = None
        self._write = None
        self._tools: List[MCPTool] = []
        self._state = MCPServerState.DISCONNECTED
        self._error: Optional[str] = None
        self._last_connected: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Get server name."""
        return self.config.name

    @property
    def state(self) -> MCPServerState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._state == MCPServerState.CONNECTED

    @property
    def tools(self) -> List[MCPTool]:
        """Get discovered tools."""
        return self._tools

    def get_status(self) -> MCPServerStatus:
        """Get server status."""
        return MCPServerStatus(
            name=self.name,
            state=self._state,
            transport=self.config.transport,
            tools_count=len(self._tools),
            error=self._error,
            last_connected=self._last_connected,
        )

    async def connect(self) -> bool:
        """
        Connect to the MCP server.

        Returns:
            True if connection successful, False otherwise
        """
        async with self._lock:
            if self._state == MCPServerState.CONNECTED:
                return True

            if not self.config.enabled:
                logger.info(f"MCP server '{self.name}' is disabled")
                return False

            self._state = MCPServerState.CONNECTING
            self._error = None

            try:
                if self.config.transport == MCPTransport.STDIO:
                    await self._connect_stdio()
                elif self.config.transport == MCPTransport.SSE:
                    await self._connect_sse()
                elif self.config.transport == MCPTransport.STREAMABLE_HTTP:
                    await self._connect_streamable_http()
                else:
                    raise ValueError(f"Unknown transport: {self.config.transport}")

                # Initialize session
                await self._initialize_session()

                # Discover tools
                await self._discover_tools()

                self._state = MCPServerState.CONNECTED
                self._last_connected = time.time()
                logger.info(
                    f"Connected to MCP server '{self.name}' "
                    f"({len(self._tools)} tools available)"
                )
                return True

            except Exception as e:
                self._state = MCPServerState.ERROR
                self._error = str(e)
                logger.error(f"Failed to connect to MCP server '{self.name}': {e}")
                await self._cleanup_resources()
                return False

    async def _connect_stdio(self):
        """Connect via stdio transport."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env,
        )

        # Create stdio client context
        self._stdio_client = stdio_client(server_params)
        self._read, self._write = await self._stdio_client.__aenter__()

        try:
            self._session = ClientSession(self._read, self._write)
            await self._session.__aenter__()
        except Exception:
            await self._stdio_client.__aexit__(None, None, None)
            self._stdio_client = None
            raise

    async def _connect_sse(self):
        """Connect via SSE transport."""
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        # Create SSE client context
        self._sse_client = sse_client(self.config.url)
        self._read, self._write = await self._sse_client.__aenter__()

        try:
            self._session = ClientSession(self._read, self._write)
            await self._session.__aenter__()
        except Exception:
            await self._sse_client.__aexit__(None, None, None)
            self._sse_client = None
            raise

    async def _connect_streamable_http(self):
        """Connect via streamable_http transport."""
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            import httpx
        except ImportError:
            raise ImportError(
                "MCP SDK required for MCP support. Install with: pip install mcp"
            )

        headers = self.config.headers or {}
        self._http_client = httpx.AsyncClient(headers=headers)
        await self._http_client.__aenter__()

        try:
            self._streamable_http_client = streamable_http_client(
                url=self.config.url, http_client=self._http_client
            )
            self._read, self._write, _ = await self._streamable_http_client.__aenter__()
            self._session = ClientSession(self._read, self._write)
            await self._session.__aenter__()
        except Exception:
            if (
                hasattr(self, "_streamable_http_client")
                and self._streamable_http_client
            ):
                await self._streamable_http_client.__aexit__(None, None, None)
                self._streamable_http_client = None
            await self._http_client.__aexit__(None, None, None)
            self._http_client = None
            raise

    async def _initialize_session(self):
        """Initialize the MCP session."""
        if self._session is None:
            raise RuntimeError("Session not created")

        # Initialize with capabilities
        result = await self._session.initialize()
        logger.debug(
            f"MCP server '{self.name}' initialized: "
            f"protocol={result.protocolVersion}, "
            f"server={result.serverInfo.name if result.serverInfo else 'unknown'}"
        )

    async def _discover_tools(self):
        """Discover available tools from the server."""
        if self._session is None:
            raise RuntimeError("Session not initialized")

        try:
            result = await self._session.list_tools()
            self._tools = []

            for tool in result.tools:
                mcp_tool = MCPTool(
                    server_name=self.name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema
                    if hasattr(tool, "inputSchema")
                    else {},
                )
                self._tools.append(mcp_tool)
                logger.debug(f"Discovered tool: {mcp_tool.full_name}")

        except Exception as e:
            logger.warning(f"Failed to discover tools from '{self.name}': {e}")
            self._tools = []

    async def _cleanup_resources(self):
        """Clean up connection resources without acquiring lock."""
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
                self._session = None

            if hasattr(self, "_stdio_client") and self._stdio_client:
                await self._stdio_client.__aexit__(None, None, None)
                self._stdio_client = None

            if hasattr(self, "_sse_client") and self._sse_client:
                await self._sse_client.__aexit__(None, None, None)
                self._sse_client = None

            if (
                hasattr(self, "_streamable_http_client")
                and self._streamable_http_client
            ):
                await self._streamable_http_client.__aexit__(None, None, None)
                self._streamable_http_client = None

            if hasattr(self, "_http_client") and self._http_client:
                await self._http_client.__aexit__(None, None, None)
                self._http_client = None

        except Exception as e:
            logger.warning(f"Error cleaning up resources for '{self.name}': {e}")

    async def disconnect(self):
        """Disconnect from the MCP server."""
        async with self._lock:
            if self._state == MCPServerState.DISCONNECTED:
                return

            try:
                await self._cleanup_resources()
            finally:
                self._state = MCPServerState.DISCONNECTED
                self._tools = []
                logger.info(f"Disconnected from MCP server '{self.name}'")

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> MCPToolResult:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool (without server prefix)
            arguments: Tool arguments
            timeout: Optional timeout in seconds

        Returns:
            MCPToolResult with the result or error
        """
        if not self.is_connected:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Not connected to server '{self.name}'",
            )

        if self._session is None:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message="Session not initialized",
            )

        try:
            # Call with timeout
            timeout = timeout or self.config.timeout

            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=timeout,
            )

            # Extract content from result
            content = self._extract_content(result)

            return MCPToolResult(
                tool_name=tool_name,
                content=content,
                is_error=result.isError if hasattr(result, "isError") else False,
            )

        except asyncio.TimeoutError:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=f"Tool call timed out after {timeout}s",
            )
        except Exception as e:
            return MCPToolResult(
                tool_name=tool_name,
                content=None,
                is_error=True,
                error_message=str(e),
            )

    def _extract_content(self, result) -> Any:
        """Extract content from MCP tool result."""
        if not hasattr(result, "content") or not result.content:
            # Fall back to structuredContent if available
            if hasattr(result, "structuredContent") and result.structuredContent:
                return result.structuredContent
            return None

        # Handle list of content items
        contents = []
        for item in result.content:
            if hasattr(item, "text"):
                contents.append(item.text)
            elif hasattr(item, "data"):
                contents.append(item.data)
            else:
                contents.append(str(item))

        # Return single item or list
        if len(contents) == 1:
            return contents[0]
        return contents

    async def refresh_tools(self):
        """Refresh the list of available tools."""
        if not self.is_connected:
            return

        await self._discover_tools()
