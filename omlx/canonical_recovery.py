# SPDX-License-Identifier: Apache-2.0
"""Canonical state recovery: bookkeeping for a scheduler-owned dense re-prefill.

A SpecPrefill turn stores no reusable prefix (_cleanup_finished skips requests
with specprefill_indices set), so later turns keep paying for the suffix it did
not store. Recovery re-reads that range densely while the engine is idle and
publishes ordinary cache blocks.

This module is the policy side only, with no MLX, so it can be tested without a
model. Execution lives in Scheduler. Publishing happens only at block
boundaries, and a growing session extends its one live job.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# A yield is a chunk the throttle or an abort refused. Retrying forever would
# keep an idle engine awake.
MAX_CONSECUTIVE_YIELDS = 8

# The allowance refills per window, so one early overrun delays the job instead
# of ending it.
DEFAULT_BUDGET_WINDOW_S = 30.0


def safe_publish_boundary(*, tokens_committed: int, block_size: int) -> int:
    """Largest block boundary at or below *tokens_committed*.

    Non-sliceable layers only have real state at a block boundary.
    """
    if block_size <= 0 or tokens_committed <= 0:
        return 0
    return (tokens_committed // block_size) * block_size


@dataclass
class CanonicalRecoveryBudget:
    """A replenishing share of wall time, shared by every engine in the process.

    Each window of ``window_s`` grants ``pct/100 * window_s`` seconds. Unused
    allowance is dropped at the roll. An overrun is charged to the next window,
    capped at one allowance, since a chunk cannot be interrupted.
    """

    pct: float = 0.0
    window_s: float = DEFAULT_BUDGET_WINDOW_S
    service_s: float = 0.0
    wall_start_s: float = field(default_factory=time.perf_counter)
    window_start_s: float = field(default_factory=time.perf_counter)
    window_service_s: float = 0.0
    windows: int = 1
    overshoot_s: float = 0.0
    # The pool's process-wide budget. Its owners must never reset it; see `reset`.
    shared: bool = False
    owners: set[str] = field(default_factory=set)
    # A slice cannot be interrupted, so it takes this claim before it starts.
    claim_key: str | None = None
    claim_at_s: float = 0.0
    # Frees the claim if its holder dies mid-slice. A real slice takes seconds.
    claim_ttl_s: float = 120.0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            self.window_s = DEFAULT_BUDGET_WINDOW_S

    @property
    def allowance_s(self) -> float:
        """Seconds of service granted at the start of each window."""
        if self.pct <= 0:
            return 0.0
        return (self.pct / 100.0) * self.window_s

    # Registration grants nothing, so unloading and reloading cannot buy a fresh window.

    def register(self, owner: str) -> None:
        with self._lock:
            self.owners.add(owner)

    def deregister(self, owner: str) -> None:
        with self._lock:
            self.owners.discard(owner)
            if self.claim_key == owner:
                self.claim_key = None

    def try_claim(self, owner: str, now: float | None = None) -> bool:
        """Take the process-wide right to execute a recovery slice.

        Acquired before the state build rather than after the first chunk,
        because the gap between those two is exactly the window in which a
        second engine sees an idle process and starts a slice of its own.
        """
        now = now if now is not None else time.perf_counter()
        with self._lock:
            held = self.claim_key
            if (
                held is not None
                and held != owner
                and now - self.claim_at_s < self.claim_ttl_s
            ):
                return False
            self.claim_key = owner
            self.claim_at_s = now
            return True

    def release_claim(self, owner: str) -> None:
        with self._lock:
            if self.claim_key == owner:
                self.claim_key = None

    def claim_held_by_other(self, owner: str, now: float | None = None) -> bool:
        now = now if now is not None else time.perf_counter()
        with self._lock:
            held = self.claim_key
            if held is None or held == owner:
                return False
            return (now - self.claim_at_s) < self.claim_ttl_s

    def elapsed_s(self, now: float | None = None) -> float:
        now = now if now is not None else time.perf_counter()
        return max(0.0, now - self.wall_start_s)

    def share(self, now: float | None = None) -> float:
        """Measured lifetime share of wall time spent on recovery work."""
        elapsed = self.elapsed_s(now)
        return (self.service_s / elapsed) if elapsed > 0 else 0.0

    def roll(self, now: float | None = None) -> None:
        """Advance to the current window, carrying at most one allowance of debt."""
        with self._lock:
            self._roll_locked(now if now is not None else time.perf_counter())

    def _roll_locked(self, now: float) -> None:
        elapsed = now - self.window_start_s
        if elapsed < self.window_s:
            return
        skipped = int(elapsed // self.window_s)
        allowance = self.allowance_s
        # Carry the overrun, capped at one allowance. Idle windows do not pay it
        # off: a gap cannot tell "nothing to do" from "foreground was busy".
        self.window_service_s = min(
            max(0.0, self.window_service_s - allowance), allowance
        )
        self.window_start_s += skipped * self.window_s
        self.windows += skipped

    def allows(self, now: float | None = None) -> bool:
        """Whether the current window still has allowance left.

        Roll and check under one lock so two engines cannot both spend the same
        allowance.
        """
        if self.pct <= 0:
            return False
        with self._lock:
            self._roll_locked(now if now is not None else time.perf_counter())
            return self.window_service_s < self.allowance_s

    def note_service(self, seconds: float) -> None:
        """Charge service to the shared budget, not to the owner."""
        seconds = max(0.0, seconds)
        with self._lock:
            self.service_s += seconds
            before = self.window_service_s
            self.window_service_s += seconds
            allowance = self.allowance_s
            if self.window_service_s > allowance:
                self.overshoot_s += self.window_service_s - max(before, allowance)

    def reset(self, now: float | None = None) -> None:
        """Start the accounting over. Refuses on a shared budget."""
        if self.shared:
            raise RuntimeError(
                "refusing to reset a shared recovery budget: one owner cannot "
                "discard the service every other owner has already spent"
            )
        now = now if now is not None else time.perf_counter()
        with self._lock:
            self.service_s = 0.0
            self.wall_start_s = now
            self.window_start_s = now
            self.window_service_s = 0.0
            self.windows = 1
            self.overshoot_s = 0.0


@dataclass
class CanonicalRecoveryCounters:
    """What a recovery job did, for the tests that assert on it."""

    publishes: int = 0
    chunks: int = 0
    yielded_steps: int = 0
    service_s: float = 0.0
    # States given back while the job stayed alive, as opposed to slices skipped.
    states_retired: int = 0


@dataclass
class CanonicalRecoveryJob:
    """One session's dense re-read, extended rather than replaced as it grows."""

    session_key: str
    tokens: list[int]
    target_tokens: int
    block_size: int
    committed_tokens: int = 0       # longest published canonical prefix
    processed_tokens: int = 0       # dense tokens consumed this job, published or not
    published_boundaries: list[int] = field(default_factory=list)
    cancelled: bool = False
    consecutive_yields: int = 0
    # Set explicitly: a `>=` on target_tokens misses when the last token is held back.
    reached_target: bool = False
    prefill_state: object | None = None

    def note_reached_target(self) -> None:
        self.reached_target = True

    def extend(self, tokens: list[int]) -> bool:
        """Grow the target append-only. Returns False if *tokens* is not an append."""
        if len(tokens) <= self.target_tokens:
            return False
        if tokens[: self.target_tokens] != self.tokens[: self.target_tokens]:
            return False
        self.tokens = tokens
        self.target_tokens = len(tokens)
        self.reached_target = False
        return True

    def publishable_boundary(self) -> int:
        """The boundary to publish now, or 0 if there is nothing new to publish."""
        if self.cancelled:
            return 0
        boundary = safe_publish_boundary(
            tokens_committed=self.processed_tokens, block_size=self.block_size
        )
        return boundary if boundary > self.committed_tokens else 0

    def note_published(self, boundary: int) -> None:
        if boundary > self.committed_tokens:
            self.committed_tokens = boundary
            self.published_boundaries.append(boundary)

    def note_ground_lost(self, restorable: int) -> None:
        """Lower the watermark to what the serving cache can still restore.

        Published blocks can be evicted (under hot_cache_only they are dropped), and
        since publishable_boundary refuses anything at or below the watermark, a stale
        watermark would stop the job republishing the lost range.
        """
        restorable = max(0, restorable)
        if restorable >= self.committed_tokens:
            return
        self.committed_tokens = restorable
        self.published_boundaries = [
            boundary for boundary in self.published_boundaries
            if boundary <= restorable
        ]

    @property
    def done(self) -> bool:
        return self.reached_target or self.processed_tokens >= self.target_tokens


def canonical_recovery_slice_cap(slice_tokens: int, request: object, n: int) -> int:
    """Cap a recovery slice, separately from the publication block.

    A slice cannot be interrupted, so its size bounds how long an arriving request
    waits. Publishing still only happens at block boundaries. Takes the cap rather
    than the config because the config is shared across engines in the pool.
    """
    if not getattr(request, "is_canonical_recovery", False):
        return n
    cap = int(slice_tokens or 0)
    return min(n, cap) if cap > 0 else n


def canonical_recovery_is_runnable(
    *,
    enabled: bool,
    budget: CanonicalRecoveryBudget,
    has_job: bool,
    waiting_requests: int,
    running_requests: int,
    prefilling_requests: int,
    specprefill_active: bool,
    inbound_requests: int,
    consecutive_idle_steps: int,
    foreign_engine_busy: bool = False,
    min_idle_steps: int = 2,
    now: float | None = None,
) -> bool:
    """Whether a recovery chunk may start on this step.

    - ``specprefill_active``: a SpecPrefill RoPE wrapper on the shared model
      would give a dense forward the wrong positions.
    - ``inbound_requests``: a request is invisible to the scheduler until its
      admission runs on the executor.
    - ``consecutive_idle_steps``: two idle steps leave a gap for an arriving
      request to announce itself.
    - ``foreign_engine_busy``: another engine in the process has foreground work.
    """
    if not enabled or not has_job:
        return False
    if specprefill_active:
        return False
    if waiting_requests or running_requests or prefilling_requests or inbound_requests:
        return False
    if foreign_engine_busy:
        return False
    if consecutive_idle_steps < min_idle_steps:
        return False
    return budget.allows(now)


def apply_canonical_recovery_settings(
    scheduler_config: object, model_settings: object
) -> None:
    """Copy a model's recovery settings onto the shared SchedulerConfig at load.

    Only the enable flag and slice size are per model. The budget is server-level.
    """
    scheduler_config.canonical_state_recovery_enabled = bool(
        getattr(model_settings, "canonical_state_recovery_enabled", False)
    )
    scheduler_config.canonical_state_recovery_slice_tokens = int(
        getattr(model_settings, "canonical_state_recovery_slice_tokens", 0) or 0
    )
    # Enabled with a zero server budget never runs, and nothing else logs it.
    if scheduler_config.canonical_state_recovery_enabled and float(
        getattr(scheduler_config, "canonical_state_recovery_global_budget_pct", 0.0) or 0.0
    ) <= 0.0:
        logger.warning(
            "canonical state recovery is enabled for %s but the server-level recovery "
            "budget (scheduler.canonical_state_recovery_global_budget_pct) is 0%%, so "
            "recovery will never be scheduled; both grants are required, "
            "raise the budget to let recovery run",
            getattr(scheduler_config, "model_name", None) or "this model",
        )
