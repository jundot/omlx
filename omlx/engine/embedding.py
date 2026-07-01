# SPDX-License-Identifier: Apache-2.0
"""
Embedding engine for oMLX.

This module provides an engine for generating text embeddings using
mlx-embeddings. Unlike LLM engines, embedding engines don't support
streaming or chat completion.
"""

import asyncio
import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..models.embedding import EmbeddingOutput, MLXEmbeddingModel
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


@dataclass
class _EmbedWork:
    """Single embedding request pending dispatch."""

    texts: List  # List[str | Dict[str, str]]
    max_length: int
    padding: bool
    truncation: bool
    future: asyncio.Future = field(repr=False)


def _approx_len(item: Any) -> int:
    """Approximate token-length of one text item by character count."""
    if isinstance(item, str):
        return len(item)
    if isinstance(item, dict):
        return len(item.get("text", "")) + len(item.get("image", "") or "")
    return 0


class EmbeddingEngine(BaseNonStreamingEngine):
    """
    Engine for generating text embeddings.

    Features a coalescing dispatch loop that:
    - Collects all concurrently-pending requests before each executor call
    - Sorts ALL texts by approximate token length across requests
    - Runs one forward pass per batch with homogeneous sequence lengths
      (reduces padding waste by up to 3x for variable-length inputs)
    - Issues a single mx.synchronize()/clear_cache() per dispatch cycle
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        batch_size: int | None = None,
        *,
        scheduler_config: Any | None = None,
    ):
        """
        Initialize the embedding engine.

        Args:
            model_name: HuggingFace model name or local path
            trust_remote_code: Allow loaders to execute custom Python shipped
                with the model repo. Off by default for security (issue #926).
            batch_size: Explicit per-forward input chunk size override.
            scheduler_config: Shared scheduler configuration. Embedding uses
                embedding_batch_size as its per-forward input chunk size.
        """
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        if batch_size is None:
            batch_size = (
                getattr(scheduler_config, "embedding_batch_size", 64)
                if scheduler_config is not None
                else 64
            )
        self._batch_size = max(1, int(batch_size))  # type: ignore[arg-type]
        self._model: Optional[MLXEmbeddingModel] = None
        self._work_queue: Optional[asyncio.Queue] = None
        self._dispatch_task: Optional[asyncio.Task] = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def processor(self) -> Any:
        """Get the processor/tokenizer."""
        return self._model.processor if self._model else None

    @property
    def hidden_size(self) -> Optional[int]:
        """Get the embedding dimension."""
        return self._model.hidden_size if self._model else None

    async def start(self) -> None:
        """Start the engine (load model if not loaded).

        Model loading runs on the global MLX executor to avoid Metal
        command buffer races with concurrent BatchGenerator steps.
        """
        if self._model is not None:
            return

        logger.info(f"Starting embedding engine: {self._model_name}")
        self._model = MLXEmbeddingModel(
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), self._model.load)

        self._work_queue = asyncio.Queue()
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())

        logger.info(f"Embedding engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping embedding engine: {self._model_name}")

        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
            self._dispatch_task = None

        # Drain and fail any remaining work
        if self._work_queue is not None:
            while not self._work_queue.empty():
                try:
                    work, _ = self._work_queue.get_nowait()
                    if not work.future.done():
                        work.future.set_exception(
                            RuntimeError("Embedding engine stopped")
                        )
                except asyncio.QueueEmpty:
                    break
            self._work_queue = None

        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info(f"Embedding engine stopped: {self._model_name}")

    # ------------------------------------------------------------------
    # Public embed API
    # ------------------------------------------------------------------

    async def embed(
        self,
        texts: Union[List[str], List[Dict[str, str]]],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> EmbeddingOutput:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of input texts
            max_length: Maximum token length for each text. If omitted, the
                model resolves its configured limit.
            padding: Whether to pad shorter sequences
            truncation: Whether to truncate longer sequences

        Returns:
            EmbeddingOutput with embeddings and token count
        """
        if self._model is None or self._work_queue is None:
            raise RuntimeError("Engine not started. Call start() first.")

        input_items: List = [texts] if isinstance(texts, str) else list(texts)
        if not input_items:
            return EmbeddingOutput(embeddings=[], total_tokens=0, dimensions=0)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        work = _EmbedWork(
            texts=input_items,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            future=fut,
        )
        activity_id = self._begin_activity(
            "embedding",
            detail="Embedding",
            total_items=len(input_items),
            metadata={"input_count": len(input_items), "batch_size": self._batch_size},
        )
        await self._work_queue.put((work, activity_id))
        try:
            return await fut
        finally:
            self._end_activity(activity_id)

    # ------------------------------------------------------------------
    # Coalescing dispatch loop
    # ------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """
        Background coroutine: drain the work queue, coalesce concurrent
        requests, sort all texts by length, and run one executor call.
        """
        queue = self._work_queue
        assert queue is not None
        while True:
            try:
                first = await queue.get()
                pending = [first]

                # Drain any immediately-available concurrent requests
                while True:
                    try:
                        pending.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                await self._run_coalesced(pending)

            except asyncio.CancelledError:
                # Fail any items we already drained but haven't processed
                break
            except Exception as exc:
                logger.error(
                    f"Embedding dispatch loop error: {exc}", exc_info=True
                )

    async def _run_coalesced(
        self, pending: List[tuple]  # List[(work, activity_id)]
    ) -> None:
        """
        Process a coalesced group of requests in a single executor call,
        resolving each request's future as soon as its last text is processed.

        Steps:
        1. Flatten all texts from all requests with position tracking
        2. Sort globally by approximate token length (reduces padding waste)
        3. Split into fixed-size forward-pass batches
        4. Run all batches in one executor call; after each batch check which
           requests are now complete and resolve their futures immediately via
           loop.call_soon_threadsafe() — so shorter-text requests return before
           longer-text ones without waiting for the whole group to finish
        5. Any requests not resolved mid-run are resolved after the final batch

        GPU work is always fully interleaved (no per-request sequential processing)
        so throughput is unchanged vs. the flat-coalescing approach.
        """
        model = self._model
        assert model is not None
        batch_size = self._batch_size
        loop = asyncio.get_running_loop()

        # --- 1. Flatten with origin tracking ---------------------------------
        # flat[i] = (req_idx, text_idx_in_req, text_item)
        flat: List[tuple] = []
        for req_idx, (work, _) in enumerate(pending):
            for text_idx, text in enumerate(work.texts):
                flat.append((req_idx, text_idx, text))

        # --- 2. Sort globally by approximate token length --------------------
        flat.sort(key=lambda x: _approx_len(x[2]))

        first_work: _EmbedWork = pending[0][0]
        sorted_texts = [x[2] for x in flat]
        batches = [
            sorted_texts[i : i + batch_size]
            for i in range(0, len(sorted_texts), batch_size)
        ]

        # Per-request accounting
        n_reqs = len(pending)
        req_total = [len(pending[i][0].texts) for i in range(n_reqs)]
        req_embs: List[Dict[int, List[float]]] = [{} for _ in range(n_reqs)]
        req_chars = [0] * n_reqs
        total_chars = 0
        for req_idx, _text_idx, text in flat:
            c = _approx_len(text)
            req_chars[req_idx] += c
            total_chars += c

        # Mutable state shared between _run_sync (MLX thread) and
        # _resolve_req (event-loop thread).  Python's GIL is sufficient —
        # _state is written only by _run_sync and read only by _resolve_req,
        # which is scheduled *after* the write via call_soon_threadsafe.
        _state: Dict[str, Any] = {"total_tokens": 0, "dimensions": 0}
        _resolved: set = set()

        def _resolve_req(req_idx: int) -> None:
            """Scheduled on the event-loop thread; safe to call set_result."""
            work, activity_id = pending[req_idx]
            emb_map = req_embs[req_idx]
            embeddings = [emb_map[i] for i in range(len(work.texts))]
            total_tokens = _state["total_tokens"]
            dimensions = _state["dimensions"]
            req_tokens = (
                round(req_chars[req_idx] / total_chars * total_tokens)
                if total_chars > 0
                else 0
            )
            result = EmbeddingOutput(
                embeddings=embeddings,
                total_tokens=req_tokens,
                dimensions=dimensions,
            )
            self._update_activity(
                activity_id,
                completed_items=len(work.texts),
                token_count=req_tokens,
                dimensions=dimensions,
            )
            if not work.future.done():
                work.future.set_result(result)

        # --- 3 & 4. Run all batches in one executor call ---------------------
        def _run_sync() -> None:
            sorted_pos = 0
            try:
                for batch in batches:
                    out = model.embed(
                        inputs=batch,
                        max_length=first_work.max_length,
                        padding=first_work.padding,
                        truncation=first_work.truncation,
                    )
                    _state["total_tokens"] += out.total_tokens
                    if out.dimensions:
                        _state["dimensions"] = out.dimensions

                    for k, embedding in enumerate(out.embeddings):
                        ri, ti, _ = flat[sorted_pos + k]
                        req_embs[ri][ti] = embedding
                    sorted_pos += len(batch)

                    # Free Metal pool after every batch (prevents 1GB→10GB RAM
                    # growth; safe because model.embed() calls mx.eval() inside).
                    mx.clear_cache()

                    # Resolve any request whose last text was in this batch
                    for i in range(n_reqs):
                        if i not in _resolved and len(req_embs[i]) == req_total[i]:
                            _resolved.add(i)
                            loop.call_soon_threadsafe(_resolve_req, i)
            finally:
                mx.synchronize()
                mx.clear_cache()

        try:
            await loop.run_in_executor(get_mlx_executor(), _run_sync)
        except Exception as exc:
            for i, (work, _) in enumerate(pending):
                if i not in _resolved and not work.future.done():
                    work.future.set_exception(exc)
            return

        # Resolve stragglers (defensive; normally all resolved inside _run_sync)
        for i in range(n_reqs):
            if i not in _resolved:
                _resolve_req(i)

    # ------------------------------------------------------------------
    # Stats / info
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "hidden_size": self.hidden_size,
            "batch_size": self._batch_size,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"loaded": False, "model_name": self._model_name}
        return self._model.get_model_info()

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<EmbeddingEngine model={self._model_name} status={status}>"
