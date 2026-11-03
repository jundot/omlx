# SPDX-License-Identifier: Apache-2.0
"""A connected helper as the scheduler sees it.

The scheduler starts one job per accelerated request and takes one chunk per
grid step. Jobs whose request disappeared from the scheduler (abort, failure,
retry) are cancelled on the next step, so they never hold the helper's queue.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .adapter import Arrays, StageAdapter
from .client import Chunk, HelperClient, HelperError, HelperJob

logger = logging.getLogger(__name__)

# Shorter prompts finish locally before the first chunk would come back.
MIN_TOKENS = 2048


@dataclass
class HelperPrefill:
    """One request's progress through its helper job."""

    session: PrefillHelperSession
    job: HelperJob
    grid: list[int]
    index: int = 0
    chunk: Chunk | None = None

    def next_size(self) -> int:
        return self.grid[self.index] if self.index < len(self.grid) else 0


@dataclass
class PrefillHelperSession:
    client: HelperClient
    adapter: StageAdapter
    layers: int
    name: str
    min_tokens: int = MIN_TOKENS
    last_error: str | None = None
    last_prompt: dict[str, Any] | None = None
    _jobs: dict[str, HelperJob] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def alive(self) -> bool:
        return self.client.alive

    def accepts(self, tokens: int) -> bool:
        return self.alive and tokens >= self.min_tokens

    def start(
        self,
        request_id: str,
        tokens: np.ndarray,
        grid: list[int],
        prefix: Arrays | None,
    ) -> HelperPrefill | None:
        try:
            job = self.client.submit(tokens, grid, prefix)
        except HelperError as error:
            self.note_failure(str(error))
            return None
        with self._lock:
            self._jobs[request_id] = job
        return HelperPrefill(self, job, list(grid))

    def finish(self, request_id: str, tokens: int, seconds: float) -> None:
        with self._lock:
            self._jobs.pop(request_id, None)
        self.last_prompt = {
            "tokens": tokens,
            "seconds": seconds,
            "at": time.time(),
        }

    def drop(self, request_id: str, reason: str) -> None:
        with self._lock:
            job = self._jobs.pop(request_id, None)
        if job is not None:
            job.cancel()
        logger.warning(
            "Prefill helper %s left request %s to this Mac: %s",
            self.name,
            request_id,
            reason,
        )
        self.note_failure(reason)

    def sync(self, live_request_ids: Iterable[str]) -> None:
        """Cancel jobs whose request the scheduler no longer prefills."""
        live = set(live_request_ids)
        with self._lock:
            gone = [rid for rid in self._jobs if rid not in live]
            jobs = [self._jobs.pop(rid) for rid in gone]
        for job in jobs:
            job.cancel()

    def note_failure(self, reason: str) -> None:
        self.last_error = reason

    def close(self) -> None:
        self.client.close()
