# SPDX-License-Identifier: Apache-2.0
"""
Anthropic API adapter for oMLX.

This adapter handles conversion between Anthropic Messages API format and the
internal request/response format used by the inference engine.
"""

import json
import uuid
from typing import Any, List, Optional

from .base import (
    BaseAdapter,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    StreamChunk,
)
from ..anthropic_models import (
    MessagesRequest as AnthropicMessagesRequest,
)
from ..anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
    create_content_block_start_event,
    create_content_block_stop_event,
    create_error_event,
    create_input_json_delta_event,
    create_message_delta_event,
    create_message_start_event,
    create_message_stop_event,
    create_text_delta_event,
    map_finish_reason_to_stop_reason,
)
from ...context.bootstrap import extract_context_selector, prepend_bootstrap_message
from ...context.runtime import apply_skill_defaults, resolve_agent_runtime_request
from ...settings import get_settings


class AnthropicAdapter(BaseAdapter):
    """
    Adapter for Anthropic Messages API format.

    Handles conversion between Anthropic message requests/responses
    and the internal format used by the inference engine.
    """

    @property
    def name(self) -> str:
        return "anthropic"

    def parse_request(self, request: AnthropicMessagesRequest) -> InternalRequest:
        """
        Convert an Anthropic MessagesRequest to internal format.

        Args:
            request: Anthropic messages request.

        Returns:
            InternalRequest in unified format.
        """
        # Convert messages to internal format (includes system message handling)
        messages = convert_anthropic_to_internal(request)

        # Convert to InternalMessage objects
        internal_messages = []
        for msg in messages:
            internal_messages.append(
                InternalMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                )
            )
        try:
            runtime = resolve_agent_runtime_request(request)
            runtime = apply_skill_defaults(get_settings(), runtime)
            internal_messages = prepend_bootstrap_message(
                internal_messages,
                get_settings(),
                selector=extract_context_selector(request),
                skill_names=runtime.skills,
            )
        except RuntimeError:
            pass

        # Convert tools if provided
        tools = None
        if request.tools:
            tools = convert_anthropic_tools_to_internal(request.tools)

        return InternalRequest(
            messages=internal_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature is not None else 1.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            top_k=request.top_k if request.top_k is not None else 0,
            stream=request.stream or False,
            stop=request.stop_sequences,
            tools=tools,
            model=request.model,
            request_id=f"msg_{uuid.uuid4().hex[:24]}",
        )

    def format_response(
        self,
        response: InternalResponse,
        request: AnthropicMessagesRequest,
    ) -> dict:
        """
        Convert an internal response to Anthropic Messages format.

        Args:
            response: Internal response object.
            request: Original Anthropic request.

        Returns:
            Response dict in Anthropic format.
        """
        return convert_internal_to_anthropic_response(
            text=response.text,
            finish_reason=response.finish_reason,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model=request.model,
            tool_calls=response.tool_calls,
        )

    def format_stream_chunk(
        self,
        chunk: StreamChunk,
        request: AnthropicMessagesRequest,
    ) -> str:
        """
        Format a streaming chunk for SSE output in Anthropic format.

        Args:
            chunk: The stream chunk to format.
            request: Original Anthropic request.

        Returns:
            SSE-formatted string.
        """
        events = []

        # First chunk: send message_start and content_block_start
        if chunk.is_first:
            message_id = f"msg_{uuid.uuid4().hex[:24]}"
            events.append(create_message_start_event(message_id, request.model))
            events.append(create_content_block_start_event(0, "text"))

        # Text delta
        if chunk.text:
            events.append(create_text_delta_event(0, chunk.text))

        # Tool call delta
        if chunk.tool_call_delta:
            partial_json = json.dumps(chunk.tool_call_delta)
            events.append(create_input_json_delta_event(0, partial_json))

        # Last chunk: send content_block_stop, message_delta, message_stop
        if chunk.is_last:
            events.append(create_content_block_stop_event(0))
            events.append(
                create_message_delta_event(
                    stop_reason=map_finish_reason_to_stop_reason(
                        chunk.finish_reason, bool(chunk.tool_call_delta)
                    ),
                    output_tokens=chunk.completion_tokens,
                )
            )
            events.append(create_message_stop_event())

        return "".join(events)

    def format_stream_end(self, request: AnthropicMessagesRequest) -> str:
        """
        Format the stream end marker for Anthropic format.

        Anthropic doesn't use [DONE] - the message_stop event is the end.

        Args:
            request: Original Anthropic request.

        Returns:
            Empty string (no additional marker needed).
        """
        return ""

    def create_error_response(
        self,
        error: str,
        error_type: str = "api_error",
        status_code: int = 500,
    ) -> dict:
        """
        Create an error response in Anthropic format.

        Args:
            error: Error message.
            error_type: Type of error (e.g., "invalid_request_error").
            status_code: HTTP status code (not used in Anthropic format).

        Returns:
            Error response dict in Anthropic format.
        """
        return {
            "type": "error",
            "error": {
                "type": error_type,
                "message": error,
            },
        }

    def format_error_event(
        self,
        error: str,
        error_type: str = "api_error",
    ) -> str:
        """
        Create an error event in Anthropic SSE format.

        Args:
            error: Error message.
            error_type: Type of error.

        Returns:
            SSE-formatted error event.
        """
        return create_error_event(error_type, error)
