# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for oMLX inference.

Provides multiple engine implementations:
- BatchedEngine: Continuous batching for multiple concurrent users
- VLMBatchedEngine: Vision-language model engine with image support
- EmbeddingEngine: Batch embedding generation using mlx-embeddings
- RerankerEngine: Document reranking using SequenceClassification models

Also re-exports core engine components for backwards compatibility.
"""

# Re-export from parent engine.py for backwards compatibility
from ..engine_core import AsyncEngineCore, EngineConfig, EngineCore
from .base import BaseEngine, BaseNonStreamingEngine, GenerationOutput
from .batched import BatchedEngine
from .embedding import EmbeddingEngine
from .reranker import RerankerEngine
from .vlm import VLMBatchedEngine

__all__ = [
    "BaseEngine",
    "BaseNonStreamingEngine",
    "GenerationOutput",
    "BatchedEngine",
    "VLMBatchedEngine",
    "EmbeddingEngine",
    "RerankerEngine",
    # Core engine components
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]
