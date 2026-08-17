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
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..models.embedding import EmbeddingOutput, MLXEmbeddingModel
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)

EmbeddingInput = Union[str, Dict[str, str]]


def _input_length(item: EmbeddingInput) -> int:
    """Approximate input cost used only to group similarly sized items."""
    if isinstance(item, str):
        return len(item)
    if isinstance(item, dict):
        return sum(len(v) for v in item.values() if isinstance(v, str))
    return 0


@dataclass(eq=False)
class _EmbeddingJob:
    """One caller's request while it is being split across forward passes."""

    inputs: List[EmbeddingInput]
    max_length: int | None
    padding: bool
    truncation: bool
    batch_size: int
    future: asyncio.Future[EmbeddingOutput]
    activity_id: str
    pending_indices: deque[int] = field(init=False)
    embeddings: List[Optional[List[float]]] = field(init=False)
    token_counts: List[int] = field(init=False)
    completed_items: int = 0
    dimensions: int = 0

    def __post_init__(self) -> None:
        ordered_indices = sorted(
            range(len(self.inputs)), key=lambda index: _input_length(self.inputs[index])
        )
        self.pending_indices = deque(ordered_indices)
        self.embeddings = [None] * len(self.inputs)
        self.token_counts = [0] * len(self.inputs)

    @property
    def compatibility_key(self) -> tuple[bool, int | None, bool, bool, int]:
        """Parameters that must match before requests can share a forward pass."""
        structured_inputs = isinstance(self.inputs[0], dict)
        return (
            structured_inputs,
            self.max_length,
            self.padding,
            self.truncation,
            self.batch_size,
        )


class EmbeddingEngine(BaseNonStreamingEngine):
    """
    Engine for generating text embeddings.

    This engine wraps MLXEmbeddingModel and provides async methods
    for integration with the oMLX server.

    Unlike BaseEngine, this doesn't support streaming or chat
    since embeddings are computed in a single forward pass.
    """

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        batch_size: int | None = None,
        *,
        scheduler_config: Any | None = None,
        microbatch_wait_ms: float = 2.0,
        max_pending_requests: int | None = None,
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
            microbatch_wait_ms: Maximum time to collect compatible concurrent
                requests before dispatching a partially filled forward pass.
            max_pending_requests: Maximum requests waiting or running in the
                embedding dispatcher. Defaults to the scheduler concurrency.
        """
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        if batch_size is None:
            batch_size = (
                getattr(scheduler_config, "embedding_batch_size", 32)
                if scheduler_config is not None
                else 32
            )
        self._batch_size = max(1, int(batch_size))
        self._microbatch_wait_s = max(0.0, float(microbatch_wait_ms) / 1000.0)
        if max_pending_requests is None:
            max_pending_requests = (
                getattr(scheduler_config, "max_num_seqs", 256)
                if scheduler_config is not None
                else 256
            )
        self._max_pending_requests = max(1, int(max_pending_requests))
        self._queue_slots = asyncio.Semaphore(self._max_pending_requests)
        self._model: Optional[MLXEmbeddingModel] = None
        self._pending_jobs: deque[_EmbeddingJob] = deque()
        self._active_jobs: set[_EmbeddingJob] = set()
        self._dispatcher_task: Optional[asyncio.Task[None]] = None
        self._arrival_waiter: Optional[asyncio.Future[None]] = None
        self._stopping = False

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

        self._stopping = False
        logger.info(f"Starting embedding engine: {self._model_name}")
        self._model = MLXEmbeddingModel(
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), self._model.load)
        logger.info(f"Embedding engine started: {self._model_name}")

    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        if self._model is None:
            return

        logger.info(f"Stopping embedding engine: {self._model_name}")
        self._stopping = True
        dispatcher = self._dispatcher_task
        if dispatcher is not None and not dispatcher.done():
            dispatcher.cancel()
            with suppress(asyncio.CancelledError):
                await dispatcher
        self._fail_jobs(
            list(self._active_jobs),
            RuntimeError("Embedding engine stopped before the request completed"),
        )
        self._pending_jobs.clear()

        model = self._model
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), model.close)
        self._model = None

        gc.collect()
        logger.info(f"Embedding engine stopped: {self._model_name}")

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
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        input_items = [texts] if isinstance(texts, str) else list(texts)

        if not input_items:
            return EmbeddingOutput(
                embeddings=[], total_tokens=0, dimensions=0, token_counts=[]
            )

        await self._queue_slots.acquire()
        try:
            batch_size = self._batch_size
            activity_id = self._begin_activity(
                "embedding",
                detail="Embedding",
                total_items=len(input_items),
                metadata={"input_count": len(input_items), "batch_size": batch_size},
            )
            loop = asyncio.get_running_loop()
            future: asyncio.Future[EmbeddingOutput] = loop.create_future()
            job = _EmbeddingJob(
                inputs=input_items,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                batch_size=batch_size,
                future=future,
                activity_id=activity_id,
            )

            try:
                if self._stopping:
                    raise RuntimeError("Embedding engine is stopping")
                self._pending_jobs.append(job)
                self._active_jobs.add(job)
                self._notify_arrival()
                try:
                    self._ensure_dispatcher(loop)
                except Exception:
                    self._pending_jobs.remove(job)
                    self._active_jobs.discard(job)
                    future.cancel()
                    raise
                return await future
            finally:
                if future.cancelled():
                    job.pending_indices.clear()
                    self._active_jobs.discard(job)
                self._end_activity(activity_id)
        finally:
            self._queue_slots.release()

    def _ensure_dispatcher(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start an ephemeral dispatcher for the current burst of requests."""
        task = self._dispatcher_task
        if task is not None and not task.done():
            if task.get_loop() is not loop:
                raise RuntimeError("Embedding dispatcher belongs to another event loop")
            return
        self._dispatcher_task = loop.create_task(
            self._dispatch_loop(),
            name=f"embedding-microbatch-{self._model_name}",
        )

    async def _dispatch_loop(self) -> None:
        """Combine compatible jobs into bounded, length-sorted forward passes."""
        current_task = asyncio.current_task()
        try:
            while True:
                self._discard_cancelled_jobs()
                if not self._pending_jobs:
                    return

                await self._wait_for_microbatch()
                self._discard_cancelled_jobs()
                if not self._pending_jobs:
                    continue

                batch, selected_jobs, compatibility_key = self._take_microbatch()
                if not batch:
                    continue
                await self._execute_microbatch(
                    batch, selected_jobs, compatibility_key
                )
        except asyncio.CancelledError:
            self._fail_jobs(
                list(self._active_jobs),
                RuntimeError(
                    "Embedding dispatcher stopped before the request completed"
                ),
            )
            raise
        except Exception as exc:
            logger.exception("Embedding microbatch dispatcher failed")
            self._fail_jobs(list(self._active_jobs), exc)
            self._pending_jobs.clear()
        finally:
            if self._dispatcher_task is current_task:
                self._dispatcher_task = None

    def _notify_arrival(self) -> None:
        waiter = self._arrival_waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    async def _wait_for_microbatch(self) -> None:
        """Wait until the first compatible batch is full or its deadline expires."""
        if self._microbatch_wait_s <= 0:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._microbatch_wait_s
        while self._pending_jobs:
            first = self._pending_jobs[0]
            compatible_items = sum(
                len(job.pending_indices)
                for job in self._pending_jobs
                if not job.future.done()
                and job.compatibility_key == first.compatibility_key
            )
            if compatible_items >= first.batch_size:
                return

            remaining = deadline - loop.time()
            if remaining <= 0:
                return

            waiter: asyncio.Future[None] = loop.create_future()
            self._arrival_waiter = waiter
            try:
                await asyncio.wait_for(waiter, timeout=remaining)
            except asyncio.TimeoutError:
                return
            finally:
                if self._arrival_waiter is waiter:
                    self._arrival_waiter = None
            self._discard_cancelled_jobs()

    def _discard_cancelled_jobs(self) -> None:
        retained: deque[_EmbeddingJob] = deque()
        while self._pending_jobs:
            job = self._pending_jobs.popleft()
            if job.future.done():
                job.pending_indices.clear()
                self._active_jobs.discard(job)
            else:
                retained.append(job)
        self._pending_jobs = retained

    def _take_microbatch(
        self,
    ) -> tuple[
        List[tuple[_EmbeddingJob, int, EmbeddingInput]],
        List[_EmbeddingJob],
        tuple[bool, int | None, bool, bool, int],
    ]:
        """Take at most one forward pass, rotating partial jobs for fairness."""
        first = self._pending_jobs.popleft()
        compatibility_key = first.compatibility_key
        capacity = first.batch_size
        candidates = deque([first])
        candidates.extend(self._pending_jobs)
        self._pending_jobs.clear()
        deferred: deque[_EmbeddingJob] = deque()

        batch: List[tuple[_EmbeddingJob, int, EmbeddingInput]] = []
        selected_jobs: List[_EmbeddingJob] = []
        while candidates:
            job = candidates.popleft()
            if job.future.done():
                job.pending_indices.clear()
                self._active_jobs.discard(job)
                continue
            if (
                len(batch) >= capacity
                or job.compatibility_key != compatibility_key
            ):
                deferred.append(job)
                continue
            selected_jobs.append(job)
            while job.pending_indices and len(batch) < capacity:
                index = job.pending_indices.popleft()
                batch.append((job, index, job.inputs[index]))

        self._pending_jobs = deferred
        return batch, selected_jobs, compatibility_key

    async def _execute_microbatch(
        self,
        batch: List[tuple[_EmbeddingJob, int, EmbeddingInput]],
        selected_jobs: List[_EmbeddingJob],
        compatibility_key: tuple[bool, int | None, bool, bool, int],
    ) -> None:
        model = self._model
        if model is None:
            self._fail_jobs(selected_jobs, RuntimeError("Embedding engine is stopped"))
            return

        ordered_batch = sorted(batch, key=lambda item: _input_length(item[2]))
        combined_inputs = [item[2] for item in ordered_batch]
        _, max_length, padding, truncation, _ = compatibility_key
        logger.debug(
            "Embedding microbatch: model=%s requests=%d inputs=%d",
            self._model_name,
            len(selected_jobs),
            len(combined_inputs),
        )

        def _embed_sync() -> EmbeddingOutput:
            try:
                return model.embed(
                    inputs=combined_inputs,
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                )
            finally:
                mx.synchronize()
                mx.clear_cache()

        try:
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(get_mlx_executor(), _embed_sync)
            if len(output.embeddings) != len(ordered_batch):
                raise RuntimeError(
                    "Embedding model returned "
                    f"{len(output.embeddings)} vectors for "
                    f"{len(ordered_batch)} inputs"
                )

            token_counts = getattr(output, "token_counts", None)
            if (
                token_counts is None
                or len(token_counts) != len(ordered_batch)
                or sum(int(count) for count in token_counts) != output.total_tokens
            ):
                token_counts = self._distribute_token_count(
                    output.total_tokens, len(ordered_batch)
                )

            for (job, index, _), embedding, token_count in zip(
                ordered_batch, output.embeddings, token_counts
            ):
                job.embeddings[index] = embedding
                job.token_counts[index] = int(token_count)
                job.completed_items += 1
                if output.dimensions:
                    job.dimensions = output.dimensions
                elif embedding:
                    job.dimensions = len(embedding)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_jobs(selected_jobs, exc)
            return

        for job in selected_jobs:
            if job.future.done():
                job.pending_indices.clear()
                self._active_jobs.discard(job)
                continue

            total_tokens = sum(job.token_counts)
            self._update_activity(
                job.activity_id,
                completed_items=job.completed_items,
                token_count=total_tokens,
                dimensions=job.dimensions,
            )
            if job.pending_indices:
                self._pending_jobs.append(job)
                continue

            if any(embedding is None for embedding in job.embeddings):
                self._fail_jobs(
                    [job],
                    RuntimeError("Embedding request completed with missing vectors"),
                )
                continue

            result_embeddings = [
                embedding for embedding in job.embeddings if embedding is not None
            ]
            job.future.set_result(
                EmbeddingOutput(
                    embeddings=result_embeddings,
                    total_tokens=total_tokens,
                    dimensions=job.dimensions,
                    token_counts=list(job.token_counts),
                )
            )
            self._active_jobs.discard(job)

    @staticmethod
    def _distribute_token_count(total_tokens: int, input_count: int) -> List[int]:
        """Compatibility fallback for model adapters without per-input counts."""
        if input_count <= 0:
            return []
        quotient, remainder = divmod(max(0, int(total_tokens)), input_count)
        return [quotient + (index < remainder) for index in range(input_count)]

    def _fail_jobs(self, jobs: List[_EmbeddingJob], exc: Exception) -> None:
        for job in jobs:
            job.pending_indices.clear()
            if not job.future.done():
                job.future.set_exception(exc)
            self._active_jobs.discard(job)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "hidden_size": self.hidden_size,
            "batch_size": self._batch_size,
            "microbatch_wait_ms": self._microbatch_wait_s * 1000.0,
            "max_pending_requests": self._max_pending_requests,
            "active_requests": len(self._active_jobs),
            "queued_requests": len(self._pending_jobs),
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"loaded": False, "model_name": self._model_name}
        return self._model.get_model_info()

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<EmbeddingEngine model={self._model_name} status={status}>"
