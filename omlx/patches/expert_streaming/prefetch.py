# SPDX-License-Identifier: Apache-2.0
"""Async expert prefetch (colibri PILOT-style router lookahead).

A worker pool reads the *numpy* slice for predicted expert ids in the
background while the GPU computes the current layer. Prediction never
changes output — it only warms a private staging buffer that the
inference thread drains before falling back to the synchronous load.

Design note — why staging is separate from the expert LRU:
    An earlier revision had workers call ``_load_expert_bundle`` directly,
    which pushed prefetched bundles into the shared LRU. That backfired:
    the working set per decode step (~1.3k bundles) far exceeds the LRU,
    so prefetched entries evicted the bundles the forward pass had just
    demand-loaded, and the main thread re-read them (double I/O, measurably
    slower than no prefetch at all). Staging is a small FIFO (default 64
    bundles ≈ 2.5 GB for 13 MB experts) that only the consumer removes;
    the LRU stays demand-populated exactly as in the no-prefetch path.

The worker is numpy-only: it must never allocate MLX arrays. MLX op
allocation off the inference thread binds arrays to a non-existent
default stream and later breaks the forward pass with
"There is no Stream(gpu, N) in current thread".
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class ExpertPrefetcher:
    """Thread pool that stages predicted expert bundles ahead of the forward pass."""

    def __init__(self, cache: Any, workers: int = 2, staging_cap: int = 64):
        self._cache = cache
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=2048)
        self._closed = threading.Event()
        self._staging: dict[Any, tuple] = {}
        self._staging_lock = threading.Lock()
        self._staging_order: deque = deque()
        self._staging_cap = max(8, staging_cap)
        self._workers = [
            threading.Thread(target=self._run, name=f"expert-prefetch-{i}", daemon=True)
            for i in range(max(1, workers))
        ]
        self.stats = {
            "submissions": 0,
            "bundles_loaded": 0,
            "staged_dropped": 0,
            "staged_consumed": 0,
            "skipped_hits": 0,
        }

    def start(self) -> None:
        for w in self._workers:
            w.start()

    def submit(self, linear: Any, expert_ids: Sequence[int]) -> None:
        """Enqueue a per-projection batch of expert ids to stage.

        *linear* must expose ``_load_expert_np`` (numpy-only loader) and
        ``bundle_key``. Idempotent; safe to race the caller.
        """
        ids = list(dict.fromkeys(int(e) for e in expert_ids if e is not None))
        if not ids:
            return
        try:
            self._queue.put((linear, ids), timeout=0.05)
            self.stats["submissions"] += 1
        except queue.Full:
            pass

    def take(self, key: Any):
        """Main-thread consumer: pop a staged np bundle or None."""
        with self._staging_lock:
            bundle = self._staging.pop(key, None)
            if bundle is not None:
                try:
                    self._staging_order.remove(key)
                except ValueError:  # pragma: no cover - defensive
                    pass
                self.stats["staged_consumed"] += 1
            return bundle

    def _stage(self, key: Any, bundle: tuple) -> None:
        with self._staging_lock:
            if key in self._staging:
                return
            self._staging[key] = bundle
            self._staging_order.append(key)
            while len(self._staging_order) > self._staging_cap:
                old = self._staging_order.popleft()
                self._staging.pop(old, None)
                self.stats["staged_dropped"] += 1

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                linear, ids = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            for eid in ids:
                try:
                    bundle = linear._load_expert_np(eid)
                    if bundle is None:
                        continue
                    self._stage(linear.bundle_key(eid), bundle)
                    self.stats["bundles_loaded"] += 1
                except Exception as exc:  # pragma: no cover - debug path
                    self.stats.setdefault("errors", 0)
                    self.stats["errors"] += 1
                    if self.stats["errors"] <= 3:
                        logger.warning("prefetch load failed (eid=%s): %s", eid, exc)
            self._queue.task_done()

    def stop(self) -> None:
        self._closed.set()
        for w in self._workers:
            try:
                w.join(timeout=1.0)
            except Exception:
                pass

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.stop()
        except Exception:
            pass
