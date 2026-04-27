# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the OpenAI Responses API (/v1/responses)."""

import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .shared_models import IDPrefix, generate_id, get_unix_timestamp


# =============================================================================
# Request Models
# =============================================================================


class InputItem(BaseModel):
    """A single item in the Responses API input array.

    Supports EasyInputMessage (no type field), message, function_call,
    function_call_output, and many other types from the Responses API.
    """

    # type is optional — EasyInputMessage omits it
    type: Optional[str] = None
    # message fields
    role: Optional[str] = None
    content: Optional[Union[str, List[Any]]] = None
    # function_call fields
    id: Optional[str] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    # function_call_output fields
    output: Optional[Union[str, List[Any], Dict[str, Any]]] = None
    # status field (present on many item types)
    status: Optional[str] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _serialize_complex_output(cls, data: Any) -> Any:
        """Serialize list/dict output to JSON string for compatibility.

        Agent frameworks may send multimodal tool outputs (e.g. images) as
        lists or dicts. Convert them to JSON strings so downstream code that
        expects ``str`` keeps working.
        """
        if isinstance(data, dict):
            output = data.get("output")
            if isinstance(output, (list, dict)):
                data = {**data, "output": json.dumps(output)}
        return data


class ResponsesTool(BaseModel):
    """Tool definition in Responses API format.

    Supports function, local_shell, mcp, web_search, and other tool types.
    """

    type: str = "function"
    # function tool fields
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    model_config = {"extra": "allow"}


class TextFormatConfig(BaseModel):
    """Text format configuration."""

    type: str = "text"  # "text", "json_object", "json_schema"
    name: Optional[str] = None
    description: Optional[str] = None
    schema_: Optional[Dict[str, Any]] = Field(None, alias="schema")
    strict: Optional[bool] = None

    model_config = {"extra": "allow", "populate_by_name": True}


class TextConfig(BaseModel):
    """Text configuration wrapper."""

    format: Optional[TextFormatConfig] = None
    verbosity: Optional[str] = None  # "low", "medium", "high"

    model_config = {"extra": "allow"}


class ResponsesRequest(BaseModel):
    """Request body for POST /v1/responses."""

    model: str
    input: Optional[Union[str, List[InputItem]]] = None
    instructions: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[ResponsesTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    text: Optional[TextConfig] = None
    previous_response_id: Optional[str] = None
    store: Optional[bool] = None
    truncation: Optional[str] = None  # "auto" or "disabled"
    metadata: Optional[Dict[str, str]] = None
    reasoning: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None
    # Fields that Codex CLI sends
    include: Optional[List[str]] = None
    service_tier: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[str] = None
    user: Optional[str] = None
    top_logprobs: Optional[int] = None
    background: Optional[bool] = None
    conversation: Optional[Any] = None
    max_tool_calls: Optional[int] = None
    stream_options: Optional[Dict[str, Any]] = None
    # Seed for reproducible generation (best-effort)
    seed: Optional[int] = None

    model_config = {"extra": "allow"}


# =============================================================================
# Response Models
# =============================================================================


class OutputContent(BaseModel):
    """Content block within an output message item."""

    type: str = "output_text"
    text: str = ""
    annotations: List[Any] = Field(default_factory=list)


class OutputItem(BaseModel):
    """A single item in the response output array.

    Can be a message or a function_call.
    """

    type: str  # "message" or "function_call"
    id: str
    status: str = "completed"
    # message fields
    role: Optional[str] = None
    content: Optional[List[OutputContent]] = None
    # function_call fields
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None


class OutputTokensDetails(BaseModel):
    """Details about output token usage."""

    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    """Token usage for Responses API."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    output_tokens_details: OutputTokensDetails = Field(
        default_factory=OutputTokensDetails
    )

    def model_post_init(self, __context) -> None:
        if self.total_tokens == 0 and (self.input_tokens > 0 or self.output_tokens > 0):
            object.__setattr__(
                self,
                "total_tokens",
                self.input_tokens + self.output_tokens,
            )


class ResponseObject(BaseModel):
    """Full response object for the Responses API."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.RESPONSE))
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=get_unix_timestamp)
    model: str
    status: str = "completed"  # "completed", "in_progress", "failed", "incomplete"
    output: List[OutputItem] = Field(default_factory=list)
    usage: Optional[ResponseUsage] = None
    text: Optional[TextConfig] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    tools: List[ResponsesTool] = Field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    metadata: Optional[Dict[str, str]] = Field(default_factory=dict)
    truncation: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
