# SPDX-License-Identifier: Apache-2.0
"""
Base adapter interface for API format conversion.

This module defines the abstract interface that all API adapters must implement,
plus internal data structures for request/response handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union


@dataclass
class InternalMessage:
    """Internal representation of a chat message."""

    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class InternalRequest:
    """
    Internal request format used by the inference engine.

    This provides a unified format that all adapters convert to/from.
    """

    # Required fields
    messages: List[InternalMessage]

    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = False

    # Stop conditions
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None

    # Tool calling
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Response format
    response_format: Optional[Dict[str, Any]] = None

    # Model
    model: Optional[str] = None

    # Metadata
    request_id: Optional[str] = None


@dataclass
class InternalResponse:
    """
    Internal response format from the inference engine.

    This provides a unified format that all adapters convert from.
    """

    # Generated content
    text: str
    finish_reason: Optional[str] = None
    reasoning_content: Optional[str] = None

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    # Tool calls (parsed)
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Metadata
    request_id: Optional[str] = None
    model: Optional[str] = None


@dataclass
class StreamChunk:
    """A single chunk in a streaming response."""

    text: str = ""
    reasoning_content: Optional[str] = None
    finish_reason: Optional[str] = None
    tool_call_delta: Optional[Dict[str, Any]] = None
    is_first: bool = False
    is_last: bool = False

    # Token counts (usually only on last chunk)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class BaseAdapter(ABC):
    """
    Abstract base class for API adapters.

    Adapters handle conversion between external API formats (OpenAI, Anthropic)
    and the internal request/response format used by the inference engine.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adapter name (e.g., 'openai', 'anthropic')."""
        pass

    @abstractmethod
    def parse_request(self, request: Any) -> InternalRequest:
        """
        Convert an external API request to internal format.

        Args:
            request: The external API request object.

        Returns:
            InternalRequest in unified format.
        """
        pass

    @abstractmethod
    def format_response(
        self,
        response: InternalResponse,
        request: Any,
    ) -> Any:
        """
        Convert an internal response to external API format.

        Args:
            response: The internal response object.
            request: The original external request (for context).

        Returns:
            Response in the external API format.
        """
        pass

    @abstractmethod
    def format_stream_chunk(
        self,
        chunk: StreamChunk,
        request: Any,
    ) -> str:
        """
        Format a streaming chunk for SSE output.

        Args:
            chunk: The stream chunk to format.
            request: The original external request (for context).

        Returns:
            SSE-formatted string.
        """
        pass

    @abstractmethod
    def format_stream_end(self, request: Any) -> str:
        """
        Format the stream end marker.

        Args:
            request: The original external request (for context).

        Returns:
            SSE-formatted end marker.
        """
        pass

    @abstractmethod
    def create_error_response(
        self,
        error: str,
        error_type: str = "server_error",
        status_code: int = 500,
    ) -> dict:
        """
        Create an error response in the adapter's format.

        Args:
            error: Error message.
            error_type: Type of error (e.g., "invalid_request_error").
            status_code: HTTP status code.

        Returns:
            Error response dict in the adapter's format.
        """
        pass
