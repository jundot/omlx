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
# Until a prompt has been timed on this Mac alone, a split predicted to save
# less than this share of the time alone leaves one prompt here to time it.
_TIME_ALONE_BELOW = 0.1


def _blend(old: float | None, new: float) -> float:
    return new if old is None else _EWMA * new + (1 - _EWMA) * old


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
    # Seconds per token a split adds on this Mac besides its layers: taking
    # in the helper's state and finishing the chunk.
    unit_extra: float | None = None
    # A prompt on this Mac alone over its estimate from the split's timings,
    # which assume ``work`` is right for the layers the helper takes.
    alone_ratio: float | None = None
    _jobs: dict[str, HelperJob] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        hello = self.client.info
        seconds, tokens = hello.get("warm_seconds"), hello.get("warm_tokens")
        if seconds and tokens and self.unit_helper is None:
            self.unit_helper = seconds / (self._work(0, self.layers) * tokens)

    def _work(self, start: int, end: int) -> float:
        return max(sum(self.work[start:end]), 1e-6)

    def _link_seconds(self, k: int) -> float:
        """Seconds per token the link needs to carry a split at ``k``; 0 until
        a chunk measured it."""
        rate = getattr(self.client, "link_bps", None)
        sizes = getattr(self.client, "token_bytes", None)
        if not rate or not sizes:
            return 0.0
        # A layer no job has used yet costs about what the measured ones do.
        known = [size for layer, size in sizes.items() if layer >= 0]
        typical = sum(known) / len(known) if known else 0.0
        state = sum(sizes.get(layer, typical) for layer in range(k))
        return (sizes.get(-1, 0.0) + state) / rate

    def pick_layers(self, grid: list[int] | None = None) -> int:
        """Layers the next job uses; 0 leaves the prompt on this Mac.

        ``grid`` is the prompt's chunk sizes. With it the choice also counts
        the wait for the first chunk, during which this Mac is idle.
        """
        if self.fixed_layers:
            return min(self.fixed_layers, self.layers)
        helper, host = self.unit_helper, self.unit_host
        if helper is None or host is None:
            # Until both Macs are timed the helper, usually the faster Mac,
            # takes all it holds; its first chunk times both.
            return self.layers
        tokens, first = (sum(grid), grid[0]) if grid else (1, 0)

        def seconds(k: int, compute: float) -> float:
            mine, link = compute * self._work(0, k), self._link_seconds(k)
            # The helper, the link and this Mac work on different chunks at
            # once, so the slowest of the three sets the speed. Before that,
            # the first chunk has to be computed there and carried here.
            rest = host * self._work(k, len(self.work)) + (self.unit_extra or 0.0)
            pace = max(mine, rest, link)
            return first * (mine + link) + tokens * pace

        best = min(range(1, self.layers + 1), key=lambda k: seconds(k, helper))
        alone = tokens * host * self._work(0, len(self.work))
        if self.alone_ratio is None:
            alone *= 1 - _TIME_ALONE_BELOW
        else:
            alone *= self.alone_ratio
        # Only the link takes the helper out. A helper that timed slow keeps
        # a job, so its next chunks can correct the estimate.
        return 0 if seconds(best, 0.0) > alone else best

    def observe(
        self,
        layers: int,
        tokens: int,
        helper_s: float,
        host_s: float,
        extra_s: float = 0.0,
    ) -> None:
        """Fold one chunk's timings into the per-unit estimates. ``host_s`` is
        this Mac's layers, ``extra_s`` the rest of its work on the chunk."""
        if tokens < _OBSERVE_TOKENS:
            return
        helper = helper_s / (self._work(0, layers) * tokens)
        host = host_s / (self._work(layers, len(self.work)) * tokens)
        self.unit_helper = _blend(self.unit_helper, helper)
        self.unit_host = _blend(self.unit_host, host)
        self.unit_extra = _blend(self.unit_extra, extra_s / tokens)

    def observe_alone(self, tokens: int, seconds: float) -> None:
        """Fold a prompt this Mac prefilled alone into its estimate."""
        if tokens < _OBSERVE_TOKENS or self.unit_host is None:
            return
        estimate = tokens * self.unit_host * self._work(0, len(self.work))
        self.alone_ratio = _blend(self.alone_ratio, seconds / estimate)

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
        layers = self.pick_layers(grid)
        if not layers:
            # Every split is slower than this Mac alone (a slow link).
            return None
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
