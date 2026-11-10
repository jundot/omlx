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

from .adapter import StageAdapter
from .client import Chunk, HelperClient, HelperJob

logger = logging.getLogger(__name__)

# Shorter prompts finish locally before the first chunk would come back.
MIN_TOKENS = 2048
# Short chunks (tails, block slivers) are dominated by fixed costs.
_OBSERVE_TOKENS = 1024
_EWMA = 0.5


@dataclass
class HelperPrefill:
    """One request's progress through its helper job."""

    session: PrefillHelperSession
    job: HelperJob
    grid: list[int]
    layers: int
    # Block drafter layers and the first job row its window keeps.
    capture: tuple[list[int], int] | None = None
    index: int = 0
    chunk: Chunk | None = None

    def next_size(self) -> int:
        return self.grid[self.index] if self.index < len(self.grid) else 0


@dataclass
class PrefillHelperSession:
    """``layers`` is how many the helper holds; each job uses the first
    ``pick_layers()`` of them. Without a fixed choice the split follows the
    measured seconds per unit of work on each Mac (``work`` per layer)."""

    client: HelperClient
    adapter: StageAdapter
    lm: Any
    layers: int
    name: str
    work: list[float]
    fixed_layers: int | None = None
    min_tokens: int = MIN_TOKENS
    last_error: str | None = None
    last_prompt: dict[str, Any] | None = None
    unit_helper: float | None = None
    unit_host: float | None = None
    _jobs: dict[str, HelperJob] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        hello = self.client.info
        seconds, tokens = hello.get("warm_seconds"), hello.get("warm_tokens")
        if seconds and tokens and self.unit_helper is None:
            self.unit_helper = seconds / (self._work(0, self.layers) * tokens)

    def _work(self, start: int, end: int) -> float:
        return max(sum(self.work[start:end]), 1e-6)

    def pick_layers(self) -> int:
        if self.fixed_layers:
            return min(self.fixed_layers, self.layers)
        helper, host = self.unit_helper, self.unit_host
        if helper is None or host is None:
            # Until both Macs are timed the helper, usually the faster Mac,
            # takes all it holds; its first chunk times both.
            return self.layers
        return min(
            range(1, self.layers + 1),
            key=lambda k: max(
                helper * self._work(0, k), host * self._work(k, len(self.work))
            ),
        )

    def observe(self, layers: int, tokens: int, helper_s: float, host_s: float) -> None:
        """Fold one chunk's timings into the per-unit estimates."""
        if tokens < _OBSERVE_TOKENS:
            return
        helper = helper_s / (self._work(0, layers) * tokens)
        host = host_s / (self._work(layers, len(self.work)) * tokens)
        self.unit_helper = (
            helper
            if self.unit_helper is None
            else (_EWMA * helper + (1 - _EWMA) * self.unit_helper)
        )
        self.unit_host = (
            host
            if self.unit_host is None
            else (_EWMA * host + (1 - _EWMA) * self.unit_host)
        )

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
        cache: list | None,
        capture: tuple[list[int], int] | None = None,
    ) -> HelperPrefill | None:
        """Queue a job; ``cache`` holds a restored prefix (None without one).

        ``capture`` names a block drafter's layers and the first job row of
        its window; the helper returns the ones it runs.
        """
        with self._lock:
            stale = self._jobs.pop(request_id, None)
        if stale is not None:
            # A retried request must not leave its old job holding the queue.
            stale.cancel()
        layers = self.pick_layers()
        wanted = None
        if capture is not None:
            below = [i for i in capture[0] if i < layers]
            wanted = {"layers": below, "from": int(capture[1])} if below else None
        try:
            prefix = self.adapter.prefix_state(cache, layers) if cache else None
            feed = self.adapter.job_feed(self.lm, cache, tokens, grid, layers)
            job = self.client.submit(
                tokens, grid, prefix, layers=layers, feed=feed, capture=wanted
            )
        except Exception as error:  # the request prefills here instead
            logger.warning("Prefill helper %s did not take a job: %s", self.name, error)
            self.note_failure(str(error))
            return None
        with self._lock:
            self._jobs[request_id] = job
        return HelperPrefill(self, job, list(grid), layers, capture)

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
