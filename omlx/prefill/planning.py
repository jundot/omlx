"""Pure planning for equal-length chunks of ordered prefill requests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


@dataclass(frozen=True)
class PrefillCandidate:
    """One prepared request; remaining tokens exclude generation kickoff."""

    request_id: str
    remaining_tokens: int
    max_chunk_tokens: int
    priority: int = 0
    cache_tokens: int = 0
    compatible: bool = True


@dataclass(frozen=True)
class PrefillEstimate:
    """Estimated additional memory and forward duration for one batch shape."""

    memory_bytes: int
    duration_seconds: float | None = None


@dataclass(frozen=True)
class PrefillPlan:
    """An ordered batch with an equal, unpadded chunk in every row."""

    request_ids: tuple[str, ...]
    chunk_tokens: int
    estimate: PrefillEstimate | None = None

    @property
    def total_tokens(self) -> int:
        return len(self.request_ids) * self.chunk_tokens


class PrefillReason(StrEnum):
    DISABLED = "disabled"
    INVALID_LIMITS = "invalid_limits"
    INSUFFICIENT_ROWS = "insufficient_rows"
    TOKEN_BUDGET = "token_budget"
    ESTIMATE_UNAVAILABLE = "estimate_unavailable"
    MEMORY_LIMIT = "memory_limit"
    DURATION_LIMIT = "duration_limit"


@dataclass(frozen=True)
class PrefillRun:
    plan: PrefillPlan


@dataclass(frozen=True)
class PrefillDefer:
    reason: PrefillReason


@dataclass(frozen=True)
class PrefillFallback:
    reason: PrefillReason


PrefillDecision = PrefillRun | PrefillDefer | PrefillFallback


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonnegative_number(value: object) -> bool:
    return _nonnegative_int(value) or (
        isinstance(value, float) and value >= 0 and isfinite(value)
    )


def _estimate_rejection(
    estimate: PrefillEstimate | None,
    max_memory_bytes: int | None,
    max_duration_seconds: float | None,
) -> PrefillReason | None:
    if estimate is None:
        return (
            None
            if max_memory_bytes is None and max_duration_seconds is None
            else PrefillReason.ESTIMATE_UNAVAILABLE
        )
    if not _nonnegative_int(estimate.memory_bytes):
        return PrefillReason.ESTIMATE_UNAVAILABLE
    if max_memory_bytes is not None and estimate.memory_bytes > max_memory_bytes:
        return PrefillReason.MEMORY_LIMIT
    duration = estimate.duration_seconds
    if duration is not None and not _nonnegative_number(duration):
        return PrefillReason.ESTIMATE_UNAVAILABLE
    if max_duration_seconds is not None:
        if duration is None:
            return PrefillReason.ESTIMATE_UNAVAILABLE
        if duration > max_duration_seconds:
            return PrefillReason.DURATION_LIMIT
    return None


def plan_prefill_batch(
    candidates: Sequence[PrefillCandidate],
    *,
    max_batch_size: int = 1,
    token_budget: int,
    estimate: Callable[[int, int, int], PrefillEstimate | None] | None = None,
    max_memory_bytes: int | None = None,
    max_duration_seconds: float | None = None,
    allow_width_reduction: bool = True,
) -> PrefillDecision:
    """Choose the widest feasible contiguous prefix, then its largest chunk.

    Candidates already follow scheduler order. An incompatible request,
    priority change, or cache-length change ends the prefix; later requests
    never bypass it. New groups may fall back to guarded singleton execution;
    existing groups must defer on memory pressure rather than allocate scalar
    caches without headroom. Existing groups disable width reduction because
    removing live rows requires a separate, admitted cache transition.

    The estimator receives ``(batch_size, chunk_tokens, cache_tokens)`` and
    must be nondecreasing in chunk length for a fixed batch. Memory estimates
    represent additional allocation, with current occupancy already subtracted
    from the supplied limit. Time limits constrain estimates, not wall time.
    A required but unavailable estimate prevents forming a batch.
    """
    if (
        not _nonnegative_int(max_batch_size)
        or not _nonnegative_int(token_budget)
        or not isinstance(allow_width_reduction, bool)
    ):
        return PrefillFallback(PrefillReason.INVALID_LIMITS)
    if max_batch_size < 2:
        return PrefillFallback(PrefillReason.DISABLED)
    if max_memory_bytes is not None and not _nonnegative_int(max_memory_bytes):
        return PrefillFallback(PrefillReason.INVALID_LIMITS)
    if max_duration_seconds is not None and not _nonnegative_number(
        max_duration_seconds
    ):
        return PrefillFallback(PrefillReason.INVALID_LIMITS)
    if estimate is None and (
        max_memory_bytes is not None or max_duration_seconds is not None
    ):
        return PrefillFallback(PrefillReason.ESTIMATE_UNAVAILABLE)

    if not candidates:
        return PrefillFallback(PrefillReason.INSUFFICIENT_ROWS)

    head = candidates[0]
    eligible = []
    request_ids = set()
    for candidate in candidates[:max_batch_size]:
        if (
            candidate.compatible is not True
            or not _nonnegative_int(candidate.remaining_tokens)
            or not _nonnegative_int(candidate.max_chunk_tokens)
            or not _nonnegative_int(candidate.cache_tokens)
            or candidate.remaining_tokens < 1
            or candidate.max_chunk_tokens < 1
            or candidate.cache_tokens < 0
            or candidate.priority != head.priority
            or candidate.cache_tokens != head.cache_tokens
        ):
            break
        if candidate.request_id in request_ids:
            raise ValueError("A prefill batch cannot contain duplicate request IDs")
        request_ids.add(candidate.request_id)
        eligible.append(candidate)

    if len(eligible) < 2:
        return PrefillFallback(PrefillReason.INSUFFICIENT_ROWS)
    minimum_width = 2 if allow_width_reduction else len(candidates)
    if len(eligible) < minimum_width:
        return PrefillFallback(PrefillReason.INSUFFICIENT_ROWS)
    if token_budget < minimum_width:
        return PrefillDefer(PrefillReason.TOKEN_BUDGET)

    rejections = set()
    for batch_size in range(min(len(eligible), token_budget), minimum_width - 1, -1):
        selected = eligible[:batch_size]
        chunk_limit = min(
            token_budget // batch_size,
            *(min(row.remaining_tokens, row.max_chunk_tokens) for row in selected),
        )
        lower = 1
        upper = chunk_limit
        accepted_chunk = 0
        accepted_estimate = None
        while lower <= upper:
            chunk_tokens = chunk_limit if upper == chunk_limit else (lower + upper) // 2
            shape_estimate = (
                estimate(batch_size, chunk_tokens, head.cache_tokens)
                if estimate is not None
                else None
            )
            rejection = _estimate_rejection(
                shape_estimate, max_memory_bytes, max_duration_seconds
            )
            if rejection is None:
                accepted_chunk = chunk_tokens
                accepted_estimate = shape_estimate
                lower = chunk_tokens + 1
            else:
                rejections.add(rejection)
                upper = chunk_tokens - 1
        if accepted_chunk:
            return PrefillRun(
                PrefillPlan(
                    request_ids=tuple(row.request_id for row in selected),
                    chunk_tokens=accepted_chunk,
                    estimate=accepted_estimate,
                )
            )
    if PrefillReason.ESTIMATE_UNAVAILABLE in rejections:
        return PrefillFallback(PrefillReason.ESTIMATE_UNAVAILABLE)
    if PrefillReason.DURATION_LIMIT in rejections:
        return PrefillFallback(PrefillReason.DURATION_LIMIT)
    return PrefillDefer(PrefillReason.MEMORY_LIMIT)
