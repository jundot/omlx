# SPDX-License-Identifier: Apache-2.0
"""
Decision engine for oMLX.

Serves decision models (Clef, OpenJev) through ``/v1/systemone``. A request
runs causal prefills and no decoding. The engine steps the model's prefill
generator one chunk per executor call, so active chat decodes get the GPU
between chunks.
"""

import asyncio
import gc
import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..model_discovery import decision_kind
from ..models.clef import ClefModel
from ..models.openjev import OpenJevModel
from ..scheduler import _CONTENDED_CHUNK_FLOOR, SchedulerConfig
from .base import BaseNonStreamingEngine
from .forward_fairness import ForwardFairnessGate

logger = logging.getLogger(__name__)

_MODEL_CLASSES = {"clef": ClefModel, "openjev": OpenJevModel}


class DecisionEngine(BaseNonStreamingEngine):
    """Engine for ``/v1/systemone`` decision models."""

    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool = False,
        scheduler_config: SchedulerConfig | None = None,
    ):
        super().__init__()
        self._model_name = model_name
        self._trust_remote_code = trust_remote_code
        self._prefill_step = max(
            1, int(getattr(scheduler_config, "prefill_step_size", 0) or 2048)
        )
        self._model: ClefModel | OpenJevModel | None = None
        self._kind: str | None = None
        # One request at a time: requests share one GPU executor, so running
        # them interleaved only multiplies the activation memory.
        self._lock = asyncio.Lock()
        self._fairness = ForwardFairnessGate(
            f"decision:{model_name}:{id(self):x}", scheduler_config
        )

    def set_memory_soft_limit(self, soft_limit_bytes: int) -> None:
        self._fairness.set_memory_soft_limit(soft_limit_bytes)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def kind(self) -> str | None:
        return self._kind

    async def start(self) -> None:
        """Load the model on the global MLX executor."""
        if self._model is not None:
            return
        kind = decision_kind(Path(self._model_name))
        if kind is None:
            raise ValueError(
                f"{self._model_name} is not a supported decision model "
                "(Clef or OpenJev)"
            )
        logger.info(f"Starting decision engine ({kind}): {self._model_name}")
        model = _MODEL_CLASSES[kind](
            self._model_name, trust_remote_code=self._trust_remote_code
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), model.load)
        self._model = model
        self._kind = kind
        logger.info(f"Decision engine started: {self._model_name}")

    async def stop(self) -> None:
        if self._model is None:
            return
        logger.info(f"Stopping decision engine: {self._model_name}")
        model = self._model
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(get_mlx_executor(), model.close)
        self._model = None
        gc.collect()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info(f"Decision engine stopped: {self._model_name}")

    def _require_model(self) -> ClefModel | OpenJevModel:
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")
        return self._model

    async def encode(self, request: dict, truncate: bool = True) -> Any:
        """Validate and tokenize a request off the event loop.

        Raises the model's request errors before any GPU work starts.
        """
        model = self._require_model()
        return await asyncio.to_thread(model.encode, request, truncate)

    async def systemone(self, plan: Any) -> dict:
        """Run an encoded request and return ``answers`` and ``input_tokens``."""
        model = self._require_model()
        async with self._lock:
            activity_id = self._begin_activity(
                "decision",
                detail="Deciding",
                total_items=len(plan.questions),
                metadata={"question_count": len(plan.questions)},
            )
            loop = asyncio.get_running_loop()
            steps = model.run(plan, self._chunk_size)
            done_tokens = 0
            try:
                while True:
                    finished, value = await loop.run_in_executor(
                        get_mlx_executor(), self._step, steps
                    )
                    if finished:
                        return value
                    done_tokens += value
                    self._update_activity(activity_id, token_count=done_tokens)
            except BaseException:
                await loop.run_in_executor(get_mlx_executor(), steps.close)
                raise
            finally:
                await self._finish_activity(activity_id)

    def _chunk_size(self) -> int:
        # Called on the executor before each prefill chunk.
        cap = self._fairness.chunk_cap()
        if cap is None:
            return self._prefill_step
        return max(_CONTENDED_CHUNK_FLOOR, min(self._prefill_step, cap))

    def _step(self, steps: Generator[int, None, dict]) -> tuple[bool, Any]:
        """Advance the request by one prefill chunk on the executor."""
        contended = self._fairness.wait_turn()
        start = time.perf_counter()
        tokens = 0
        try:
            tokens = next(steps)
            return False, tokens
        except StopIteration as stop:
            return True, stop.value
        finally:
            mx.synchronize()
            self._fairness.settle(time.perf_counter() - start, tokens, contended)

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "decision_kind": self._kind,
        }

    def get_model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "loaded": self._model is not None,
            "model_name": self._model_name,
        }
        if self._model is not None:
            info["decision_kind"] = self._kind
            info["vision"] = self._model.backbone.has_vision
        return info

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<DecisionEngine model={self._model_name} status={status}>"
