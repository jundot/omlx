# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for Cohere/Jina-compatible Rerank API.

These models define the request and response schemas for:
- /v1/rerank endpoint
"""

import uuid

from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    """
    Request for reranking documents.

    Cohere/Jina-compatible request format for the /v1/rerank endpoint.
    """

    model: str
    """ID of the model to use."""

    query: str | dict[str, str]
    """
    The search query to compare documents against. String for text-only
    rerankers. Dict with 'text' and/or 'image' (URL, base64 data URI, or
    local path) for multimodal rerankers like Qwen3-VL-Reranker.
    """

    documents: list[str] | list[dict[str, str]]
    """
    Documents to rerank. Can be:
    - List of strings
    - List of dicts with 'text' field (and optional 'image' for multimodal
      rerankers). Image values accept URL, base64 data URI, or local path.
    """

    top_n: int | None = None
    """
    Number of top results to return.
    If not specified, returns all documents.
    """

    return_documents: bool = True
    """Whether to include document text in the response."""

    max_chunks_per_doc: int | None = None
    """
    Maximum chunks per document (for long documents).
    Currently not implemented.
    """


class RerankResult(BaseModel):
    """A single rerank result."""

    index: int
    """Original index of the document in the input list."""

    relevance_score: float
    """Relevance score between 0 and 1."""

    document: dict[str, str] | None = None
    """
    The document (if return_documents=True). For text-only rerankers or
    string inputs, format is {"text": "..."}. For multimodal inputs, the
    original dict (including 'image') is returned as-is.
    """


class RerankUsage(BaseModel):
    """Token usage statistics for rerank request."""

    total_tokens: int
    """Total number of tokens processed."""


class RerankResponse(BaseModel):
    """
    Response from reranking documents.

    Cohere/Jina-compatible response format for the /v1/rerank endpoint.
    """

    id: str = Field(default_factory=lambda: f"rerank-{uuid.uuid4().hex[:8]}")
    """Unique identifier for the rerank request."""

    results: list[RerankResult]
    """Reranked results sorted by relevance score (descending)."""

    model: str
    """The model used for reranking."""

    usage: RerankUsage | None = None
    """Token usage statistics."""
