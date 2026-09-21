# SPDX-License-Identifier: Apache-2.0
"""Shadow prefill: bookkeeping for a scheduler-owned dense re-prefill.

A sparse (SpecPrefill) prefill serves its request and leaves nothing the prefix
cache will accept: ``_cleanup_finished`` refuses to extract a cache whose
``specprefill_indices`` is set, and the non-sliceable layers of a hybrid model
only carry real state at a captured block boundary. The reusable dense prefix
therefore stops advancing, and every later turn in the session pays to
recompute the suffix the sparse turn did not canonicalize.

This module holds the *decision* half of a shadow prefill: a dense re-read of a
token range the session has already been served, run as scheduler-owned work,
which publishes ordinary canonical cache state. It deliberately contains no MLX
and touches no cache, so the policy can be tested without a model. The
execution half lives in ``Scheduler``.

Two things it exists to get right:

**Publication happens at every safe boundary.** Canonical state for a
non-sliceable layer exists only at a cache block boundary, and a partial block
is not a smaller win but a corrupt one. Publishing at each boundary as it is
reached is what makes an interrupted job worth the prefix it got to.

**Growth is append-only and single-flight.** A session that adds a turn extends
the live job's target rather than starting a second one, because two jobs on
one session would recompute the same prefix twice and race to publish it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

# How many consecutive yields a job may take before it is given up on. A yield
# means the chunk did not run — the memory throttle wanted headroom, or the
# chunk was aborted — and nothing the shadow itself does will change that, so
# retrying forever only keeps an idle engine awake.
MAX_CONSECUTIVE_YIELDS = 8

# How many consecutive idle steps a job may be *allowed* to run and still not
# run before it is given up on. This is a different condition from a yield: a
# yield is raised from inside a chunk, and the case this bounds never reaches
# one. The engine is idle, the window grants service, and the runnable
# predicate still says no — which on this runtime means a SpecPrefill RoPE
# wrapper is installed with no request behind it, a state `_unwrap_rope`
# documents as expected. Nothing the recovery job does takes it off, so
# without a deadline the job never runs, never finishes and never gives up,
# on a loop that keeps stepping for it. At the 50 ms step interval this is
# about ten seconds.
MAX_BLOCKED_IDLE_STEPS = 200

# The scheduling window a recovery allowance is granted in. The budget has to
# replenish: measured over a job's whole life instead, one chunk that overran
# the allowance early put the share above the cap for as long as the
# denominator took to grow back, which at 5% on an 80K session was the rest of
# the run. A job that overshoots once must be late, not finished.
DEFAULT_BUDGET_WINDOW_S = 30.0


def safe_publish_boundary(*, tokens_committed: int, block_size: int) -> int:
    """The largest boundary at or below *tokens_committed* that may be published.

    Non-sliceable layers carry real state only at a block boundary; anything
    between two boundaries restores as a placeholder and is rejected or walked
    back. Publishing a partial block is therefore not a smaller win, it is a
    corrupt one, so the floor is the block below.
    """
    if block_size <= 0 or tokens_committed <= 0:
        return 0
    return (tokens_committed // block_size) * block_size


@dataclass
class ShadowBudget:
    """A replenishing bounded share of wall time, aggregated over the process.

    There is one of these per process, created by ``EnginePool`` and adopted by
    every scheduler. It rations one accelerator, and a process holds several
    engines that share it, so a budget per engine would grant M loaded models M
    times the configured ceiling with nothing adding them up.

    Service is granted per tumbling window of ``window_s``:

    - each window opens with an allowance of ``pct/100 * window_s`` seconds;
    - unused allowance is discarded at the roll, so the job cannot bank credit
      and spend it as one long slice in front of the foreground;
    - a chunk that overruns the allowance is charged to the next window, which
      is what makes the cap hold across windows rather than only inside one;
    - that carried overshoot is capped at a single allowance, so one overrun
      can cost at most one window of service and never a permanent lockout.

    The chunk grain is the reason the last two rules are not symmetric. A
    chunk cannot be interrupted once it is handed to the model, so when the
    allowance is smaller than one chunk every chunk overshoots by
    construction. The cap then bounds the lockout rather than the share.
    """

    pct: float = 0.0
    # A non-positive window is a misconfiguration and is corrected in
    # __post_init__ rather than interpreted. Left alone it made the allowance
    # zero beside a non-zero measured share, which is state that contradicts
    # itself.
    window_s: float = DEFAULT_BUDGET_WINDOW_S
    service_s: float = 0.0
    wall_start_s: float = field(default_factory=time.perf_counter)
    window_start_s: float = field(default_factory=time.perf_counter)
    window_service_s: float = 0.0
    windows: int = 1
    overshoot_s: float = 0.0
    # True when this object is the process-global budget the engine pool
    # created and every scheduler adopted. A shared budget must never be
    # reset by one of its owners: see `reset`.
    shared: bool = False
    # Owner keys currently registered. Registration grants nothing — it
    # exists so a departing engine can deregister, and so a reader can say
    # how many engines a share is being divided between.
    owners: set[str] = field(default_factory=set)
    # The single execution claim. A recovery slice is uninterruptible and
    # invisible from outside while it runs, so "is anyone recovering right
    # now" cannot be answered by watching side effects; it is answered here,
    # before the work starts.
    claim_key: str | None = None
    claim_at_s: float = 0.0
    # A holder that dies mid-slice would otherwise hold the claim forever.
    # Generous, because a legitimate slice is seconds.
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

    # -- ownership ------------------------------------------------------
    # Registration is deliberately inert. An engine that loads, unloads and
    # loads again must not find a fresh window waiting for it, or unloading
    # becomes a way to discharge a debt — and under memory pressure the pool
    # unloads and reloads on every eviction cycle.

    def register(self, owner: str) -> None:
        with self._lock:
            self.owners.add(owner)

    def deregister(self, owner: str) -> None:
        with self._lock:
            self.owners.discard(owner)
            if self.claim_key == owner:
                self.claim_key = None

    # -- the execution claim --------------------------------------------

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

    # -- accounting -----------------------------------------------------

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
        # Carry the overrun, capped at one allowance. It is not discharged by
        # windows the job did not run in: the gap between two `allows` calls
        # cannot tell "the job had nothing to do" from "the foreground was
        # busy for a minute", and the second is the case the budget exists
        # for. Discharging on the gap meant any busy period longer than two
        # windows wiped the debt, so the cap stopped holding on exactly the
        # sessions it was written for.
        self.window_service_s = min(
            max(0.0, self.window_service_s - allowance), allowance
        )
        self.window_start_s += skipped * self.window_s
        self.windows += skipped

    def allows(self, now: float | None = None) -> bool:
        """Whether the current window still has allowance left.

        The roll and the comparison are one critical section. Split, two
        engines can both clear the test on allowance only one of them has —
        and the charge for a slice necessarily lands after the grant, so
        nothing downstream would catch it.
        """
        if self.pct <= 0:
            return False
        with self._lock:
            self._roll_locked(now if now is not None else time.perf_counter())
            return self.window_service_s < self.allowance_s

    def note_service(self, seconds: float) -> None:
        """Charge service, globally.

        The overshoot this accrues belongs to the budget rather than to the
        owner that caused it. Per-owner debt would let the aggregate overrun
        scale with the number of engines, which is the property this object
        exists to remove.
        """
        seconds = max(0.0, seconds)
        with self._lock:
            self.service_s += seconds
            before = self.window_service_s
            self.window_service_s += seconds
            allowance = self.allowance_s
            if self.window_service_s > allowance:
                self.overshoot_s += self.window_service_s - max(before, allowance)

    def reset(self, now: float | None = None) -> None:
        """Start the accounting over. Never valid on a shared budget.

        One engine resetting a budget its peers are charging against would
        forgive their spent allowance mid-window, erase the carried overshoot
        and splice their accounting onto a new clock. `Scheduler.reset` checks
        `shared` before calling this; the guard here is the backstop.
        """
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
class ShadowCounters:
    """What a recovery job did, for the tests that assert on it."""

    publishes: int = 0
    chunks: int = 0
    yielded_steps: int = 0
    service_s: float = 0.0


@dataclass
class ShadowJob:
    """One session's dense re-read, extended rather than replaced as it grows."""

    session_key: str
    tokens: list[int]
    target_tokens: int
    block_size: int
    committed_tokens: int = 0       # longest published canonical prefix
    processed_tokens: int = 0       # dense tokens consumed this job, published or not
    published_boundaries: list[int] = field(default_factory=list)
    # Identity of the BlockAwarePrefixCache that served the originating
    # request. A job is bound to it for its whole life: one served model can
    # present more than one prefix-cache instance, and state published into the
    # instance that did not serve the request is valid, durable and invisible.
    serving_cache_id: int | None = None
    # A weak reference to that same instance. The id alone answers "is this
    # still the cache that served me", which is all the publish decision
    # needs. Releasing the job's blocks needs the object: on the one path
    # where the instance has changed, the current one is by construction not
    # the one holding them.
    serving_cache_ref: object | None = None
    cancelled: bool = False
    # Consecutive chunks that yielded without doing work. Bounded, because a
    # yield that nothing ever satisfies is not a pause: the job stays live, the
    # engine loop keeps stepping to serve it, and an idle server spins forever
    # holding the job's whole prefill state resident.
    consecutive_yields: int = 0
    # Set when the dense pass has consumed the whole target. Tracked explicitly
    # rather than compared against target_tokens: the prefill path holds back
    # the final token for the generation kickoff, so processed_tokens legitimately
    # tops out one short and a `>=` comparison never fires.
    reached_target: bool = False
    # Set by the scheduler; the opaque _PrefillState driving the dense chunks.
    prefill_state: object | None = None

    def note_reached_target(self) -> None:
        self.reached_target = True

    def extend(self, tokens: list[int]) -> bool:
        """Grow the target append-only. Returns False if *tokens* is not a growth.

        A turn that rewrites history rather than appending to it invalidates
        everything already computed, so it is refused here and the caller
        starts a new job instead of silently publishing state for a prefix the
        session no longer has.
        """
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

    @property
    def done(self) -> bool:
        return self.reached_target or self.processed_tokens >= self.target_tokens


def shadow_slice_cap(config: object, request: object, n: int) -> int:
    """Cap a recovery slice, which is not the same thing as its block.

    Recovery publishes at a cache block boundary because that is the only
    place canonical state exists for a non-sliceable layer. It does not have
    to *compute* a whole block at a time, and the two were the same number
    only because nothing had separated them.

    The one a foreground request waits for is this one. A slice cannot be
    interrupted once it is handed to the model, so a whole-block recovery unit
    can delay an arriving request for the duration of that unit. Smaller
    execution slices bound that blocking interval, independently of the
    canonical publication grain.

    Three controls, three different jobs:

    - the recovery budget governs how *often* recovery collides with a
      foreground request;
    - the execution slice governs how *long* that request is blocked when it
      does;
    - the publication grain governs *when* reusable canonical state may be
      committed, and is fixed by the cache layout rather than chosen.

    Shrinking the slice changes nothing about publication.
    ``clamp_prefill_chunk_to_boundary`` already stops a slice overshooting a
    boundary, ``should_emit_prefill_boundary`` fires only on exact block
    multiples, and ``safe_publish_boundary`` floors publication to them. More
    slices reach the same boundaries; they simply leave a gap in between for a
    request to arrive in.

    A free function because it is a property of the config and the request
    rather than of the scheduler, and the chunk path should not acquire a new
    reason to reach through ``self`` for it.
    """
    if not getattr(request, "is_shadow", False):
        return n
    cap = int(getattr(config, "shadow_prefill_slice_tokens", 0) or 0)
    return min(n, cap) if cap > 0 else n


def shadow_is_runnable(
    *,
    enabled: bool,
    budget: ShadowBudget,
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
    """Whether a shadow chunk may start on this step.

    Every clause is a defect from the earlier prototype's safety review, or a
    constraint read out of the runtime, written down as a condition rather than
    as a comment:

    - ``specprefill_active`` — SpecPrefill installs ``_OffsetAdjustedRoPE`` on
      the *shared* model and keeps it installed until generation ends. A dense
      forward taken while it is installed reads that request's position offset,
      so the shadow must not run in that window at all.
    - ``inbound_requests`` — a request is invisible to the scheduler until its
      admission runs on the single-worker executor. Idleness judged from the
      admitted lists alone starts a slice in front of a request that has
      already arrived.
    - ``consecutive_idle_steps`` — a slice holds the interpreter lock for its
      whole duration, so back-to-back slices leave no window in which a request
      can announce itself. Requiring two idle steps buys that window back at a
      cost of one step interval per slice.
    - ``foreign_engine_busy`` — every other clause here reads one scheduler's
      own lists, and a process can hold several engines sharing one GPU. An
      engine whose own lists are empty is idle; the *machine* it is about to
      take a slice on may not be. Recovery is the lowest-priority work in the
      process, so it stands down for foreground work anywhere in it.
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


def apply_shadow_prefill_settings(
    scheduler_config: object, model_settings: object
) -> None:
    """Carry a model's shadow-prefill knobs onto a shared ``SchedulerConfig``.

    Two of them, and neither is a ceiling: a model chooses whether to recover
    and how large a slice it does it in. How much of the accelerator recovery
    may have is one server-level number, because every engine in the pool
    shares one accelerator and this config object is rewritten per load.

    Mirrors how ``model_name``/``model_path`` are wired per model at engine
    load: shadow prefill is scheduler-owned, not per-request, so its settings
    live on the config object rather than flowing through per-call kwargs.
    """
    scheduler_config.shadow_prefill_enabled = bool(
        getattr(model_settings, "shadow_prefill_enabled", False)
    )
    scheduler_config.shadow_prefill_slice_tokens = int(
        getattr(model_settings, "shadow_prefill_slice_tokens", 0) or 0
    )
