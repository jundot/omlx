"""Engine-thread ownership of staged rows and live prefill groups.

This module has no request queues, sampling policy, or scheduler callbacks.
Logical ownership changes precede fallible cache compaction. A failure can
therefore affect only rows which have not been committed to another owner.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .execution import BatchedPrefillGroup, PrefillGroupStepResult

logger = logging.getLogger(__name__)


@dataclass
class _GroupOwnership:
    members: dict[str, None]
    pending_handoffs: dict[str, list[Any]] = field(default_factory=dict)
    error: Exception | None = None


class PrefillBatchRuntime:
    """Own membership until decode commit, scalar demotion, or cancellation.

    All calls are serialized on the engine thread. Failed groups retain their
    logical members until the scheduler consumes them through discard(). Cache
    release failures remain registered for a later close_all() attempt.
    """

    def __init__(self) -> None:
        self._candidates: dict[str, tuple[int, ...]] = {}
        self._groups: dict[BatchedPrefillGroup, _GroupOwnership] = {}
        self._by_request: dict[str, BatchedPrefillGroup] = {}

    @property
    def has_groups(self) -> bool:
        return bool(self._groups)

    def stage(self, request_id: str, tokens: Sequence[int]) -> None:
        if request_id in self._candidates or request_id in self._by_request:
            raise ValueError("Prefill request already has a runtime owner")
        if not tokens:
            raise ValueError("A staged prefill must contain tokens")
        self._candidates[request_id] = tuple(tokens)

    def is_candidate(self, request_id: str) -> bool:
        return request_id in self._candidates

    def unstage(self, request_id: str) -> None:
        self._candidates.pop(request_id, None)

    def group_for(self, request_id: str) -> BatchedPrefillGroup | None:
        return self._by_request.get(request_id)

    def owned_ids(self, group: BatchedPrefillGroup) -> tuple[str, ...]:
        ownership = self._groups.get(group)
        return tuple(ownership.members) if ownership is not None else ()

    def error_for(self, group: BatchedPrefillGroup) -> Exception | None:
        ownership = self._groups.get(group)
        return ownership.error if ownership is not None else None

    def needs_compaction(self, group: BatchedPrefillGroup) -> bool:
        return group.request_ids != self.owned_ids(group)

    def create(
        self,
        model: Any,
        request_ids: Sequence[str],
        *,
        stream: Any,
        skip_lm_head: bool = False,
    ) -> BatchedPrefillGroup:
        group = BatchedPrefillGroup(
            model,
            [(request_id, self._candidates[request_id]) for request_id in request_ids],
            stream,
            skip_lm_head=skip_lm_head,
        )
        self._groups[group] = _GroupOwnership(dict.fromkeys(request_ids))
        for request_id in request_ids:
            self._candidates.pop(request_id)
            self._by_request[request_id] = group
        return group

    def advance(
        self, group: BatchedPrefillGroup, chunk_tokens: int
    ) -> PrefillGroupStepResult:
        if group.request_ids != self.owned_ids(group):
            raise RuntimeError("Prefill ownership must be compacted before advancing")
        if self._groups[group].pending_handoffs:
            raise RuntimeError("Cannot advance a group with pending handoffs")
        try:
            return group.step(chunk_tokens)
        except Exception as error:
            if not group.valid:
                self.invalidate(group, error)
            raise

    def prepare_handoff(self, group: BatchedPrefillGroup, request_id: str) -> list[Any]:
        ownership = self._groups[group]
        if request_id not in ownership.members:
            raise KeyError(request_id)
        if group.remaining_tokens[request_id] != 0:
            raise ValueError("Only completed prefill rows can enter decode")
        if request_id not in ownership.pending_handoffs:
            ownership.pending_handoffs[request_id] = group.extract(request_id)
        return ownership.pending_handoffs[request_id]

    def commit_handoff(self, group: BatchedPrefillGroup, request_id: str) -> None:
        """Acknowledge successful decode registration without GPU operations."""
        ownership = self._groups[group]
        if request_id not in ownership.pending_handoffs:
            raise ValueError("Prefill handoff was not prepared")
        ownership.pending_handoffs.pop(request_id)
        ownership.members.pop(request_id)
        self._by_request.pop(request_id)

    def compact(self, group: BatchedPrefillGroup) -> None:
        """Physically remove committed/cancelled rows; never undo their commit."""
        ownership = self._groups[group]
        removed = [
            request_id
            for request_id in group.request_ids
            if request_id not in ownership.members
        ]
        try:
            group.remove(removed)
        except Exception as error:
            self.invalidate(group, error)
            raise
        if not ownership.members:
            self._groups.pop(group)

    def cancel(self, request_id: str) -> BatchedPrefillGroup | None:
        """Cancel ownership; the caller admits survivor compaction separately."""
        self.unstage(request_id)
        group = self._by_request.pop(request_id, None)
        if group is None:
            return None
        ownership = self._groups[group]
        ownership.members.pop(request_id)
        ownership.pending_handoffs.pop(request_id, None)
        if not ownership.members:
            self.discard(group)
        return group

    def demote(self, group: BatchedPrefillGroup) -> dict[str, list[Any]]:
        """Return scalar caches only after every extraction and close succeeds.

        The scheduler must admit the transition's temporary allocation first.
        Partial extraction never transfers ownership to the caller.
        """
        ownership = self._groups[group]
        if ownership.pending_handoffs:
            raise RuntimeError("Cannot demote a group with pending handoffs")
        try:
            caches = {
                request_id: group.extract(request_id)
                for request_id in ownership.members
            }
            group.close()
        except Exception as error:
            self.invalidate(group, error)
            raise
        for request_id in ownership.members:
            self._by_request.pop(request_id)
        self._groups.pop(group)
        return caches

    def invalidate(self, group: BatchedPrefillGroup, error: Exception) -> None:
        ownership = self._groups[group]
        ownership.error = error
        ownership.pending_handoffs.clear()
        try:
            group.close()
        except Exception:
            logger.warning("Failed to drain invalid prefill group", exc_info=True)

    def discard(self, group: BatchedPrefillGroup) -> tuple[str, ...]:
        """Release owned requests for recovery, retaining no committed rows."""
        ownership = self._groups.get(group)
        if ownership is None:
            return ()
        request_ids = tuple(ownership.members)
        for request_id in request_ids:
            self._by_request.pop(request_id)
        ownership.members.clear()
        ownership.pending_handoffs.clear()
        try:
            group.close()
        except Exception:
            logger.warning("Failed to drain discarded prefill group", exc_info=True)
        else:
            self._groups.pop(group)
        return request_ids

    def close_all(self) -> None:
        self._candidates.clear()
        for group in tuple(self._groups):
            self.discard(group)
