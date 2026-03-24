# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible Embeddings API.

These models define the request and response schemas for:
- /v1/embeddings endpoint
"""

import time
import uuid
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class EmbeddingInputItem(BaseModel):
    """Structured input item for multimodal embeddings."""

    text: Optional[str] = None
    image: Optional[str] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_fields(self) -> "EmbeddingInputItem":
        """Require at least one supported field."""
        if self.text is None and self.image is None:
            raise ValueError("Embedding input item must include text or image")
        return self


class EmbeddingRequest(BaseModel):
    """
    Request for creating embeddings.

    OpenAI-compatible request format for the /v1/embeddings endpoint.
    """

    input: Optional[Union[str, List[str]]] = None
    """Input text(s) to embed. Can be a single string or list of strings."""

    items: Optional[List[EmbeddingInputItem]] = None
    """Structured embedding items for multimodal inputs."""

    model: str
    """ID of the model to use."""

    encoding_format: Literal["float", "base64"] = "float"
    """
    The format to return embeddings in.
    - "float": Returns a list of floats (default)
    - "base64": Returns a base64-encoded string of little-endian floats
    """

    dimensions: Optional[int] = None
    """
    The number of dimensions the output embeddings should have.
    Only supported by some models. If not supported, returns full dimensions.
    """

    @model_validator(mode="after")
    def validate_input_source(self) -> "EmbeddingRequest":
        """Require exactly one input source."""
        if self.input is None and self.items is None:
            raise ValueError("Either input or items must be provided")
        if self.input is not None and self.items is not None:
            raise ValueError("input and items cannot be provided together")
        if self.items is not None and len(self.items) == 0:
            raise ValueError("items cannot be empty")
        return self


class EmbeddingData(BaseModel):
    """A single embedding result."""

    object: str = "embedding"
    """The object type, always "embedding"."""

    index: int
    """The index of the embedding in the input list."""

    embedding: Union[List[float], str]
    """
    The embedding vector.
    - List[float] when encoding_format="float"
    - str (base64) when encoding_format="base64"
    """


class EmbeddingUsage(BaseModel):
    """Token usage statistics for embedding request."""

    prompt_tokens: int
    """Number of tokens in the input."""

    total_tokens: int
    """Total number of tokens used (same as prompt_tokens for embeddings)."""


class EmbeddingResponse(BaseModel):
    """
    Response from creating embeddings.

    OpenAI-compatible response format for the /v1/embeddings endpoint.
    """

    object: str = "list"
    """The object type, always "list"."""

    data: List[EmbeddingData]
    """List of embedding objects."""

    model: str
    """The model used for embedding."""

    usage: EmbeddingUsage
    """Usage statistics."""
