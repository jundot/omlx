# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for Anthropic Messages API.

These models define the request and response schemas for:
- Anthropic Messages API (/v1/messages)
- Streaming events
- Tool calling in Anthropic format
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from omlx.api.shared_models import IDPrefix, generate_id

# =============================================================================
# Content Blocks
# =============================================================================


class ContentBlockText(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ContentBlockImage(BaseModel):
    """Image content block with source data."""

    type: Literal["image"] = "image"
    source: dict[str, Any]  # {"type": "base64"|"url", "media_type": "...", "data"|"url": "..."}


class ContentBlockToolUse(BaseModel):
    """Tool use content block (model requesting a tool call)."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(BaseModel):
    """Tool result content block (user providing tool output)."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[dict[str, Any]] | dict[str, Any] | list[Any] | Any
    is_error: bool | None = None


class ContentBlockThinking(BaseModel):
    """Thinking content block for reasoning models (e.g., Claude Opus 4.6)."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


# Union type for all content blocks
ContentBlock = (
    ContentBlockText
    | ContentBlockImage
    | ContentBlockToolUse
    | ContentBlockToolResult
    | ContentBlockThinking
)


# =============================================================================
# System Content
# =============================================================================


class SystemContent(BaseModel):
    """System message content block."""

    type: Literal["text"] = "text"
    text: str
    cache_control: dict[str, str] | None = None


# =============================================================================
# Messages
# =============================================================================


class AnthropicMessage(BaseModel):
    """A message in an Anthropic conversation."""

    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]


# =============================================================================
# Tools
# =============================================================================


class AnthropicTool(BaseModel):
    """Tool definition in Anthropic format."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]
    cache_control: dict[str, str] | None = None


class ToolChoice(BaseModel):
    """Tool choice specification."""

    type: Literal["auto", "any", "tool"]
    name: str | None = None  # Required when type="tool"


# =============================================================================
# Thinking Configuration
# =============================================================================


class ThinkingConfig(BaseModel):
    """Configuration for extended thinking/reasoning."""

    type: Literal["enabled", "disabled"] = "enabled"
    budget_tokens: int | None = None


# =============================================================================
# Request
# =============================================================================


class MessagesRequest(BaseModel):
    """Request for Anthropic Messages API."""

    model: str
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: ToolChoice | dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None


# =============================================================================
# Token Counting
# =============================================================================


class TokenCountRequest(BaseModel):
    """Request for token counting (Anthropic format)."""

    model: str
    messages: list[AnthropicMessage]
    system: str | list[SystemContent] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: ToolChoice | dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None


class TokenCountResponse(BaseModel):
    """Response for token counting."""

    input_tokens: int


# =============================================================================
# Response
# =============================================================================


class AnthropicUsage(BaseModel):
    """Token usage statistics for Anthropic API."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    """Response for Anthropic Messages API."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.MESSAGE))
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[ContentBlockText | ContentBlockToolUse]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# =============================================================================
# Streaming Events
# =============================================================================


class MessageStartEvent(BaseModel):
    """Event sent at the start of a message."""

    type: Literal["message_start"] = "message_start"
    message: dict[str, Any]  # Partial MessagesResponse


class ContentBlockStartEvent(BaseModel):
    """Event sent at the start of a content block."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: dict[str, Any]  # Partial content block


class TextDelta(BaseModel):
    """Text delta for streaming."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(BaseModel):
    """JSON input delta for tool use streaming."""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ContentBlockDeltaEvent(BaseModel):
    """Event sent for content block updates."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: TextDelta | InputJsonDelta | dict[str, Any]


class ContentBlockStopEvent(BaseModel):
    """Event sent when a content block ends."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    """Event sent for message-level updates (stop_reason, usage)."""

    type: Literal["message_delta"] = "message_delta"
    delta: dict[str, Any]  # {"stop_reason": "...", "stop_sequence": ...}
    usage: dict[str, int]  # {"output_tokens": N}


class MessageStopEvent(BaseModel):
    """Event sent when the message ends."""

    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    """Ping event for keeping connection alive."""

    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    """Error event for streaming errors."""

    type: Literal["error"] = "error"
    error: dict[str, Any]  # {"type": "...", "message": "..."}


# Union type for all streaming events
StreamingEvent = (
    MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageDeltaEvent
    | MessageStopEvent
    | PingEvent
    | ErrorEvent
)


# =============================================================================
# Error Response
# =============================================================================


class AnthropicErrorDetail(BaseModel):
    """Error detail in Anthropic format."""

    type: str  # "invalid_request_error", "authentication_error", "api_error", etc.
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response in Anthropic format."""

    type: Literal["error"] = "error"
    error: AnthropicErrorDetail
