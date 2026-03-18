# SPDX-License-Identifier: Apache-2.0
"""
Reranker engine for oMLX.

This module provides an engine for document reranking using
SequenceClassification and CausalLM-based reranker models.
Unlike LLM engines, reranker engines don't support streaming
or chat completion.
"""

import asyncio
import gc
import logging
from typing import Any, Dict

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..models.reranker import MLXRerankerModel, RerankOutput
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


class RerankerEngine(BaseNonStreamingEngine):
    """
    Engine for document reranking.

    This engine wraps MLXRerankerModel and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support streaming or chat
    since reranking is computed in a single forward pass.
    """

    def __init__(self, model_name: str):
        """
        Initialize the reranker engine.

        Args:
            model_name: HuggingFace model name or local path
        """
        self._model_name = model_name
        self._model: MLXRerankerModel | None = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def processor(self) -> Any:
        """Get the processor/tokenizer."""
        return self._model.processor if self._model else None

    @property
    def num_labels(self) -> int | None:
        """Get the number of classification labels."""
        return self._model.num_labels if self._model else None

    async def start(self) -> None:
        """Start the engine (load model if not loaded).

        Model loading runs on the global MLX executor to avoid Metal
        command buffer races with concurrent BatchGenerator steps.
        """
        if self._model is not None:
            return

        logger.info(f"Starting reranker engine: {self._model_name}")
        self._model = MLXRerankerModel(self._model_name)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), self._model.load)
        logger.info(f"Reranker engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping reranker engine: {self._model_name}")
        self._model = None

        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info(f"Reranker engine stopped: {self._model_name}")

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        max_length: int | None = None,
    ) -> RerankOutput:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of top results to return (None = all)
            max_length: Maximum token length for each query-document pair.
                If None, uses model-appropriate default (512 for encoder,
                8192 for CausalLM).

        Returns:
            RerankOutput with scores, sorted indices, and token count
        """
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        model = self._model

        def _rerank_sync():
            return model.rerank(
                query=query,
                documents=documents,
                max_length=max_length,
            )

        loop = asyncio.get_running_loop()
        output = await loop.run_in_executor(
            get_mlx_executor(), _rerank_sync
        )

        # Apply top_n filtering if specified
        if top_n is not None and top_n < len(output.indices):
            top_indices = output.indices[:top_n]
            # Keep original scores but note which indices are in top_n
            return RerankOutput(
                scores=output.scores,
                indices=top_indices,
                total_tokens=output.total_tokens,
            )

        return output

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "num_labels": self.num_labels,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"loaded": False, "model_name": self._model_name}
        return self._model.get_model_info()

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<RerankerEngine model={self._model_name} status={status}>"
