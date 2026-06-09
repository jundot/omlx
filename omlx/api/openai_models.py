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

import json
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from omlx.api.shared_models import (
    BaseUsage,
    IDPrefix,
    generate_id,
    get_unix_timestamp,
)

# =============================================================================
# Content Types
# =============================================================================


class ImageURL(BaseModel):
    """Image URL or base64 data URI for vision model input."""

    url: str  # "https://..." or "data:image/jpeg;base64,..."
    detail: str | None = "auto"  # "low", "high", "auto"


class InputAudio(BaseModel):
    """Audio input data for multimodal models (OpenAI format)."""

    data: str  # Base64-encoded audio or data URI
    format: str = "wav"  # Audio format: wav, mp3, etc.


class FileContent(BaseModel):
    """File input for attachment preprocessing.

    ``file_data`` matches OpenAI Chat Completions file content parts.
    ``data`` is accepted as an oMLX legacy alias for dashboard clients.
    """

    filename: str | None = None
    mime_type: str | None = None
    file_data: str | None = None
    data: str | None = None
    file_id: str | None = None


class ContentPart(BaseModel):
    """
    A part of a message content array.

    Supports:
    - text: Plain text content
    - image_url: Image input for vision models
    - input_audio: Audio input for multimodal audio models
    - file: Document or text input for attachment preprocessing
    """

    type: str  # "text", "image_url", "input_audio", or "file"
    text: str | None = None
    image_url: ImageURL | None = None
    input_audio: InputAudio | None = None
    file: FileContent | None = None


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
    content: str | list[ContentPart] | list[dict] | None = None
    # Reasoning/thinking content from <think> blocks (OpenAI reasoning_content field)
    reasoning_content: str | None = None
    # For assistant messages with tool calls
    tool_calls: list[dict] | None = None
    # For tool response messages (role="tool")
    tool_call_id: str | None = None
    # Participant name, rendered into chat template (e.g. Kimi K2/K2.5 named assistants)
    name: str | None = None
    # Continue from this message instead of starting a new turn (prefill / partial mode)
    partial: bool = False

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _validate_tool_call_arguments(cls, v: Any) -> Any:
        """Validate arguments on each tool_call before the raw dict is stored.

        tool_calls is typed as List[dict] for flexibility, which bypasses
        FunctionCall's own validator. Re-run the same coercion here so
        malformed arguments surface as 422 instead of crashing the chat
        template on the next turn.
        """
        if not isinstance(v, list):
            return v
        for tc in v:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function")
            if not isinstance(func, dict) or "arguments" not in func:
                continue
            func["arguments"] = _coerce_tool_call_arguments(func["arguments"])
        return v


# =============================================================================
# Tool Calling
# =============================================================================


def _coerce_tool_call_arguments(v: Any) -> str:
    """Normalize a tool_call.arguments value to a JSON-object string.

    Native tool-calling chat templates (Qwen3.5/3.6, GLM-4.x, MiniMax)
    iterate `arguments.items()`, which requires the echoed value to parse
    back into a dict. Rejecting malformed inputs here turns the silent 500
    in downstream template rendering into a clear 422 that tells the client
    what to fix. Dict inputs (non-spec but common) are coerced to JSON
    strings, empty/whitespace strings normalize to ``"{}"``, and any value
    that can't round-trip into a JSON object raises ValueError.
    """
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if not isinstance(v, str):
        raise ValueError(
            f"arguments must be a JSON-encoded string, got {type(v).__name__}. "
            "Per the OpenAI spec tool_call.arguments is a string containing JSON, "
            'not a dict/list/number. Example: \'{"location": "Tokyo"}\'.'
        )
    stripped = v.strip()
    if not stripped:
        return "{}"
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as e:
        snippet = stripped if len(stripped) <= 120 else stripped[:117] + "..."
        raise ValueError(
            f"arguments must be valid JSON, got parse error: {e}. "
            "This usually means the client echoed a previous tool call "
            "with a malformed arguments value. Send arguments as a "
            'JSON-encoded object string like \'{"location": "Tokyo"}\'. '
            f"Received: {snippet!r}"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"arguments must be a JSON object, got {type(parsed).__name__}. "
            "Tool-call arguments cannot be a list, number, or bare string. "
            'Example: \'{"location": "Tokyo"}\'.'
        )
    return v


class FunctionCall(BaseModel):
    """A function call with name and arguments."""

    name: str
    arguments: str  # JSON string

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, v: Any) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("arguments", mode="before")
    @classmethod
    def _validate_arguments_json(cls, v: Any) -> str:
        return _coerce_tool_call_arguments(v)


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
    description: str | None = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    strict: bool | None = False

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
    json_schema: ResponseFormatJsonSchema | None = None


class StructuredOutputOptions(BaseModel):
    """vLLM-compatible structured output options.

    Exactly one field should be set. When passed via ``extra_body`` in the
    OpenAI client, the key is ``structured_outputs``.

    Supports:
    - json: JSON schema (dict or string) for logit-level enforcement
    - regex: Regular expression the output must match
    - choice: List of allowed string values (output will be exactly one)
    - grammar: EBNF/GBNF context-free grammar string
    """

    model_config = {"populate_by_name": True}

    json_schema: str | dict | None = Field(None, alias="json")
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None


# =============================================================================
# Chat Completion
# =============================================================================


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = Field(
        default=None,
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
    )
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: list[str] | None = None
    min_p: float | None = None
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    # Tool calling
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific tool
    # Structured output
    response_format: ResponseFormat | dict | None = None
    # vLLM-compatible structured output (grammar, regex, choice, json)
    structured_outputs: StructuredOutputOptions | dict | None = None
    # vLLM/OpenAI-compatible grammar alias, normalized to structured_outputs
    guided_grammar: str | None = None
    # Chat template kwargs (e.g. enable_thinking, reasoning_effort)
    chat_template_kwargs: dict[str, Any] | None = None
    # Thinking budget (max thinking tokens, None = unlimited)
    thinking_budget: int | None = Field(default=None, ge=0)
    # SpecPrefill: per-request enable/disable (None = use model setting)
    specprefill: bool | None = None
    # SpecPrefill: per-request keep percentage (0.1-0.5, None = use model setting)
    specprefill_keep_pct: float | None = None
    # SpecPrefill: per-request threshold override (min tokens to trigger, None = use model setting)
    specprefill_threshold: int | None = None
    # Seed for reproducible generation (best-effort)
    seed: int | None = None
    # Per-token logprobs (OpenAI-compatible; #1549)
    logprobs: bool | None = False
    top_logprobs: int | None = None

    @field_validator("stop", mode="before")
    @classmethod
    def coerce_stop(cls, v):
        """Accept stop as a single string (OpenAI compat) and wrap in a list."""
        if isinstance(v, str):
            return [v]
        return v

    @model_validator(mode="after")
    def _validate_logprobs(self):
        """Enforce OpenAI logprobs rules: top_logprobs needs logprobs, range 0-20."""
        if self.top_logprobs is not None:
            if not self.logprobs:
                raise ValueError("`top_logprobs` requires `logprobs` to be true")
            if not 0 <= self.top_logprobs <= 20:
                raise ValueError("`top_logprobs` must be between 0 and 20")
        return self


class AssistantMessage(BaseModel):
    """Response message from the assistant."""

    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class TopLogprob(BaseModel):
    """A candidate token and its logprob at one position (OpenAI shape)."""
    token: str
    logprob: float
    bytes: list[int] | None = None


class ChatCompletionTokenLogprob(BaseModel):
    """Logprob info for a single generated token (OpenAI shape)."""
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] = Field(default_factory=list)


class ChoiceLogprobs(BaseModel):
    """Per-choice logprobs payload (OpenAI ``choices[].logprobs``)."""
    content: list[ChatCompletionTokenLogprob] | None = None


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = "stop"
    logprobs: ChoiceLogprobs | None = None


class PromptTokensDetails(BaseModel):
    """Breakdown of prompt tokens used."""

    cached_tokens: int | None = None
    audio_tokens: int | None = None


class Usage(BaseUsage):
    """Token usage statistics for OpenAI API.

    Extends BaseUsage with optional timing metrics (oMLX extension).
    When present, timing values are in seconds.
    """

    prompt_tokens_details: PromptTokensDetails | None = None
    # Timing metrics (oMLX extension, seconds)
    model_load_duration: float | None = None
    time_to_first_token: float | None = None
    total_time: float | None = None
    prompt_eval_duration: float | None = None
    generation_duration: float | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.CHAT_COMPLETION))
    object: str = "chat.completion"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Text Completion
# =============================================================================


class CompletionRequest(BaseModel):
    """Request for text completion."""

    model: str
    prompt: str | list[str]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: list[str] | None = None
    min_p: float | None = None
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    # Seed for reproducible generation (best-effort)
    seed: int | None = None
    # Cap reasoning/thinking tokens (parity with /v1/chat/completions)
    thinking_budget: int | None = Field(default=None, ge=0)

    @field_validator("stop", mode="before")
    @classmethod
    def coerce_stop(cls, v):
        """Accept stop as a single string (OpenAI compat) and wrap in a list."""
        if isinstance(v, str):
            return [v]
        return v


class CompletionChoice(BaseModel):
    """A single choice in text completion response."""

    index: int = 0
    text: str
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.COMPLETION))
    object: str = "text_completion"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: list[CompletionChoice]
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
    # vLLM-compatible extension: lets OpenAI-style clients discover the
    # effective context window from the listing without a separate call
    # to /v1/models/status (see #1308).
    max_model_len: int | None = None


class ModelsResponse(BaseModel):
    """Response for listing models."""

    object: str = "list"
    data: list[ModelInfo]


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

    tools: list[MCPToolInfo]
    count: int


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    name: str
    state: str
    transport: str
    tools_count: int
    error: str | None = None


class MCPServersResponse(BaseModel):
    """Response for listing MCP servers."""

    servers: list[MCPServerInfo]


class MCPExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""

    model_config = {"populate_by_name": True}

    tool_name: str = Field(validation_alias=AliasChoices("tool_name", "tool"))
    arguments: dict = Field(default_factory=dict)


class MCPExecuteResponse(BaseModel):
    """Response from executing an MCP tool."""

    tool_name: str
    content: str | list | dict | None = None
    is_error: bool = False
    error_message: str | None = None


# =============================================================================
# Streaming (for SSE responses)
# =============================================================================


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.CHAT_COMPLETION))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # Present on last chunk when include_usage=true
