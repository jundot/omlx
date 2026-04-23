# SPDX-License-Identifier: Apache-2.0
"""
OpenAI API adapter for oMLX.

This adapter handles conversion between OpenAI API format and the internal
request/response format used by the inference engine.
"""

import json
import time
import uuid
from typing import Any, List, Optional

from .base import (
    BaseAdapter,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    StreamChunk,
)
from ..openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    PromptTokensDetails,
    Usage,
)
from ..thinking import extract_thinking
from ..utils import clean_special_tokens, extract_text_content
from ..tool_calling import convert_tools_for_template


class OpenAIAdapter(BaseAdapter):
    """
    Adapter for OpenAI API format.

    Handles conversion between OpenAI chat completion requests/responses
    and the internal format used by the inference engine.
    """

    @property
    def name(self) -> str:
        return "openai"

    def parse_request(self, request: ChatCompletionRequest) -> InternalRequest:
        """
        Convert an OpenAI ChatCompletionRequest to internal format.

        Args:
            request: OpenAI chat completion request.

        Returns:
            InternalRequest in unified format.
        """
        # Extract text content from messages
        messages = extract_text_content(request.messages)

        # Convert to internal messages
        internal_messages = [
            InternalMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
            )
            for msg in messages
        ]

        # Convert tools if provided
        tools = None
        if request.tools:
            tools = convert_tools_for_template(request.tools)

        return InternalRequest(
            messages=internal_messages,
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature if request.temperature is not None else 1.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            min_p=request.min_p if request.min_p is not None else 0.0,
            presence_penalty=request.presence_penalty if request.presence_penalty is not None else 0.0,
            frequency_penalty=request.frequency_penalty if request.frequency_penalty is not None else 0.0,
            stream=request.stream or False,
            stop=request.stop if isinstance(request.stop, list) else (
                [request.stop] if request.stop else None
            ),
            tools=tools,
            tool_choice=request.tool_choice,
            response_format=request.response_format,
            model=request.model,
            request_id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        )

    def format_response(
        self,
        response: InternalResponse,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        Convert an internal response to OpenAI ChatCompletionResponse.

        Args:
            response: Internal response object.
            request: Original OpenAI request.

        Returns:
            ChatCompletionResponse in OpenAI format.
        """
        # Separate thinking from content
        raw_text = clean_special_tokens(response.text) if response.text else ""
        thinking_content, regular_content = extract_thinking(raw_text)
        content = regular_content.strip() if regular_content else None

        # Determine finish reason
        finish_reason = (
            "tool_calls" if response.tool_calls else response.finish_reason
        )

        return ChatCompletionResponse(
            id=response.request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(
                        content=content,
                        reasoning_content=thinking_content if thinking_content else None,
                        tool_calls=response.tool_calls,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.prompt_tokens + response.completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=getattr(response, "cached_tokens", 0),
                ),
            ),
        )

    def format_stream_chunk(
        self,
        chunk: StreamChunk,
        request: ChatCompletionRequest,
    ) -> str:
        """
        Format a streaming chunk for SSE output in OpenAI format.

        Args:
            chunk: The stream chunk to format.
            request: Original OpenAI request.

        Returns:
            SSE-formatted string.
        """
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        delta = ChatCompletionChunkDelta(
            content=chunk.text if chunk.text else None,
            reasoning_content=chunk.reasoning_content if chunk.reasoning_content else None,
            tool_calls=chunk.tool_call_delta,
        )

        # Add role on first chunk
        if chunk.is_first:
            delta.role = "assistant"

        response = ChatCompletionChunk(
            id=request_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=delta,
                    finish_reason=chunk.finish_reason,
                )
            ],
        )

        # Add usage on last chunk if available
        if chunk.is_last and (chunk.prompt_tokens > 0 or chunk.completion_tokens > 0):
            response.usage = Usage(
                prompt_tokens=chunk.prompt_tokens,
                completion_tokens=chunk.completion_tokens,
                total_tokens=chunk.prompt_tokens + chunk.completion_tokens,
                prompt_tokens_details=PromptTokensDetails(
                    cached_tokens=getattr(chunk, "cached_tokens", 0),
                ),
            )

        return f"data: {response.model_dump_json(exclude_none=True)}\n\n"

    def format_stream_end(self, request: ChatCompletionRequest) -> str:
        """
        Format the stream end marker for OpenAI format.

        Args:
            request: Original OpenAI request.

        Returns:
            SSE-formatted end marker.
        """
        return "data: [DONE]\n\n"

    def create_error_response(
        self,
        error: str,
        error_type: str = "server_error",
        status_code: int = 500,
    ) -> dict:
        """
        Create an error response in OpenAI format.

        Args:
            error: Error message.
            error_type: Type of error.
            status_code: HTTP status code.

        Returns:
            Error response dict.
        """
        return {
            "error": {
                "message": error,
                "type": error_type,
                "param": None,
                "code": status_code,
            }
        }
