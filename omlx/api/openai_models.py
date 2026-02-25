# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Pydantic models for OpenAI-compatible API.

These models define the request and response schemas for:
- Chat completions
- Text completions
- Tool calling
- MCP (Model Context Protocol) integration
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from omlx.api.shared_models import (
    BaseUsage,
    IDPrefix,
    generate_id,
    get_unix_timestamp,
)


# =============================================================================
# Content Types
# =============================================================================

class ContentPart(BaseModel):
    """
    A part of a message content array.

    Supports:
    - text: Plain text content
    """
    type: str  # "text"
    text: Optional[str] = None


# =============================================================================
# Messages
# =============================================================================

class Message(BaseModel):
    """
    A message in a chat conversation.

    Supports:
    - Simple text messages (role + content string)
    - Content array messages (role + content list with text parts)
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool" with tool_call_id)
    """
    role: str
    content: Optional[Union[str, List[ContentPart], List[dict]]] = None
    # For assistant messages with tool calls
    tool_calls: Optional[List[dict]] = None
    # For tool response messages (role="tool")
    tool_call_id: Optional[str] = None


# =============================================================================
# Tool Calling
# =============================================================================

class FunctionCall(BaseModel):
    """A function call with name and arguments."""
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from the model."""
    id: str
    type: str = "function"
    function: FunctionCall


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""
    type: str = "function"
    function: dict


# =============================================================================
# Structured Output (JSON Schema)
# =============================================================================

class ResponseFormatJsonSchema(BaseModel):
    """JSON Schema definition for structured output."""
    name: str
    description: Optional[str] = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    strict: Optional[bool] = False

    class Config:
        populate_by_name = True


class ResponseFormat(BaseModel):
    """
    Response format specification for structured output.

    Supports:
    - "text": Default text output (no structure enforcement)
    - "json_object": Forces valid JSON output
    - "json_schema": Forces JSON matching a specific schema
    """
    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: Optional[ResponseFormatJsonSchema] = None


# =============================================================================
# Chat Completion
# =============================================================================

class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    model: str
    messages: List[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[List[str]] = None
    # Tool calling
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, dict]] = None  # "auto", "none", or specific tool
    # Structured output
    response_format: Optional[Union[ResponseFormat, dict]] = None
    # Chat template kwargs (e.g. enable_thinking, reasoning_effort)
    chat_template_kwargs: Optional[Dict[str, Any]] = None


class AssistantMessage(BaseModel):
    """Response message from the assistant."""
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""
    index: int = 0
    message: AssistantMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseUsage):
    """Token usage statistics for OpenAI API."""

    pass


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.CHAT_COMPLETION))
    object: str = "chat.completion"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Text Completion
# =============================================================================

class CompletionRequest(BaseModel):
    """Request for text completion."""
    model: str
    prompt: Union[str, List[str]]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    """A single choice in text completion response."""
    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.COMPLETION))
    object: str = "text_completion"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: List[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Models List
# =============================================================================

class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=get_unix_timestamp)
    owned_by: str = "omlx"


class ModelsResponse(BaseModel):
    """Response for listing models."""
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# MCP (Model Context Protocol)
# =============================================================================

class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""
    name: str
    description: str
    server: str
    parameters: dict = Field(default_factory=dict)


class MCPToolsResponse(BaseModel):
    """Response for listing MCP tools."""
    tools: List[MCPToolInfo]
    count: int


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""
    name: str
    state: str
    transport: str
    tools_count: int
    error: Optional[str] = None


class MCPServersResponse(BaseModel):
    """Response for listing MCP servers."""
    servers: List[MCPServerInfo]


class MCPExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""
    tool_name: str
    arguments: dict = Field(default_factory=dict)


class MCPExecuteResponse(BaseModel):
    """Response from executing an MCP tool."""
    tool_name: str
    content: Optional[Union[str, list, dict]] = None
    is_error: bool = False
    error_message: Optional[str] = None


# =============================================================================
# Streaming (for SSE responses)
# =============================================================================

class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[dict]] = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.CHAT_COMPLETION))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[Usage] = None  # Present on last chunk when include_usage=true
