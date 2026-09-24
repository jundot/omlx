"""Logical ownership contracts, independent of GPU numerical execution."""

from unittest.mock import Mock

import pytest

from omlx.prefill.runtime import PrefillBatchRuntime


class FakeGroup:
    def __init__(self, model, rows, stream, skip_lm_head=False):
        self.request_ids = tuple(request_id for request_id, _ in rows)
        self.remaining_tokens = {request_id: len(tokens) for request_id, tokens in rows}
        self.valid = True
        self.extracted = []
        self.removed = []
        self.closed = 0

    def step(self, chunk_tokens):
        for request_id in self.request_ids:
            self.remaining_tokens[request_id] -= chunk_tokens
        return tuple(self.request_ids)

    def extract(self, request_id):
        self.extracted.append(request_id)
        return [object()]

    def remove(self, request_ids):
        self.removed.append(tuple(request_ids))
        self.request_ids = tuple(
            request_id
            for request_id in self.request_ids
            if request_id not in request_ids
        )
        if not self.request_ids:
            self.close()

    def close(self):
        self.valid = False
        self.request_ids = ()
        self.closed += 1


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr("omlx.prefill.runtime.BatchedPrefillGroup", FakeGroup)
    instance = PrefillBatchRuntime()
    yield instance
    instance.close_all()


def create_group(runtime, *request_ids):
    for request_id in request_ids:
        runtime.stage(request_id, [1, 2, 3])
    return runtime.create(None, request_ids, stream=object())


def test_staging_has_one_owner_and_failed_creation_keeps_candidates(
    runtime, monkeypatch
):
    runtime.stage("first", [1])
    with pytest.raises(ValueError, match="already has"):
        runtime.stage("first", [2])
    with pytest.raises(ValueError, match="contain tokens"):
        runtime.stage("empty", [])
    monkeypatch.setattr(
        "omlx.prefill.runtime.BatchedPrefillGroup", Mock(side_effect=MemoryError)
    )
    with pytest.raises(MemoryError):
        runtime.create(None, ["first"], stream=object())
    assert runtime.is_candidate("first")
    assert runtime.group_for("first") is None
    assert not runtime.has_groups
    runtime.cancel("first")
    assert not runtime.is_candidate("first")


def test_prepared_handoff_keeps_ownership_and_materializes_once(runtime):
    group = create_group(runtime, "first", "second")
    with pytest.raises(ValueError, match="completed"):
        runtime.prepare_handoff(group, "first")
    runtime.advance(group, 3)
    cache = runtime.prepare_handoff(group, "first")
    assert runtime.prepare_handoff(group, "first") is cache
    assert group.extracted == ["first"]
    assert runtime.owned_ids(group) == ("first", "second")
    with pytest.raises(RuntimeError, match="pending handoffs"):
        runtime.advance(group, 1)
    with pytest.raises(RuntimeError, match="pending handoffs"):
        runtime.demote(group)


def test_commit_is_gpu_free_and_not_undone_by_compaction_failure(runtime, monkeypatch):
    group = create_group(runtime, "first", "second", "third")
    runtime.advance(group, 3)
    runtime.prepare_handoff(group, "first")
    failure = RuntimeError("filter failed after partial mutation")
    monkeypatch.setattr(group, "remove", Mock(side_effect=failure))
    monkeypatch.setattr(group, "extract", Mock(side_effect=AssertionError))

    runtime.commit_handoff(group, "first")

    assert group.removed == []
    assert runtime.group_for("first") is None
    assert runtime.owned_ids(group) == ("second", "third")
    with pytest.raises(RuntimeError, match="compacted"):
        runtime.advance(group, 1)
    with pytest.raises(RuntimeError, match="filter failed"):
        runtime.compact(group)
    assert runtime.error_for(group) is failure
    assert not group.valid
    assert runtime.discard(group) == ("second", "third")
    assert not runtime.has_groups


def test_commit_requires_preparation_and_cannot_happen_twice(runtime):
    group = create_group(runtime, "first", "second")
    with pytest.raises(ValueError, match="not prepared"):
        runtime.commit_handoff(group, "first")
    runtime.advance(group, 3)
    runtime.prepare_handoff(group, "first")
    runtime.commit_handoff(group, "first")
    with pytest.raises(ValueError, match="not prepared"):
        runtime.commit_handoff(group, "first")
    with pytest.raises(KeyError):
        runtime.prepare_handoff(group, "first")
    runtime.compact(group)
    assert group.request_ids == ("second",)
    assert not runtime.needs_compaction(group)


def test_cancellation_commits_before_survivor_allocation(runtime, monkeypatch):
    group = create_group(runtime, "first", "cancelled", "last")
    monkeypatch.setattr(group, "remove", Mock(side_effect=MemoryError))
    assert runtime.cancel("cancelled") is group
    assert runtime.group_for("cancelled") is None
    group.remove.assert_not_called()
    with pytest.raises(MemoryError):
        runtime.compact(group)
    assert runtime.discard(group) == ("first", "last")


def test_failed_extraction_does_not_prepare_or_transfer_a_row(runtime, monkeypatch):
    group = create_group(runtime, "first", "second")
    runtime.advance(group, 3)
    original_extract = group.extract
    monkeypatch.setattr(group, "extract", Mock(side_effect=MemoryError))
    with pytest.raises(MemoryError):
        runtime.prepare_handoff(group, "first")
    assert group.valid
    assert runtime.owned_ids(group) == ("first", "second")
    with pytest.raises(ValueError, match="not prepared"):
        runtime.commit_handoff(group, "first")
    monkeypatch.setattr(group, "extract", original_extract)
    runtime.prepare_handoff(group, "first")
    runtime.commit_handoff(group, "first")


def test_failed_demotion_never_partially_transfers_ownership(runtime, monkeypatch):
    group = create_group(runtime, "first", "second")
    monkeypatch.setattr(group, "extract", Mock(side_effect=[[object()], MemoryError()]))
    with pytest.raises(MemoryError):
        runtime.demote(group)
    assert not group.valid
    assert runtime.owned_ids(group) == ("first", "second")
    assert runtime.discard(group) == ("first", "second")


def test_successful_demotion_releases_only_its_group(runtime):
    group = create_group(runtime, "first", "second")
    unrelated = create_group(runtime, "third", "fourth")
    runtime.stage("later", [4])
    runtime.advance(group, 1)
    caches = runtime.demote(group)
    assert tuple(caches) == ("first", "second")
    assert runtime.group_for("first") is None
    assert runtime.owned_ids(group) == ()
    assert not group.valid
    assert runtime.group_for("third") is unrelated
    assert runtime.is_candidate("later")


def test_failed_forward_keeps_only_uncommitted_rows_for_recovery(runtime, monkeypatch):
    group = create_group(runtime, "first", "second")
    failure = RuntimeError("forward failed")

    def fail(chunk_tokens):
        group.valid = False
        raise failure

    monkeypatch.setattr(group, "step", fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        runtime.advance(group, 1)
    assert runtime.error_for(group) is failure
    assert group.closed == 1
    assert runtime.discard(group) == ("first", "second")


def test_cleanup_failure_retains_cache_owner_for_later_drain(runtime, monkeypatch):
    group = create_group(runtime, "first", "second")
    original_close = group.close
    monkeypatch.setattr(group, "close", Mock(side_effect=RuntimeError("sync failed")))
    assert runtime.discard(group) == ("first", "second")
    assert runtime.group_for("first") is None
    assert runtime.has_groups
    monkeypatch.setattr(group, "close", original_close)
    runtime.close_all()
    assert not runtime.has_groups
    assert group.closed == 1


def test_all_rows_committed_or_cancelled_close_without_recovery(runtime):
    group = create_group(runtime, "first", "second")
    runtime.advance(group, 3)
    runtime.prepare_handoff(group, "first")
    runtime.commit_handoff(group, "first")
    runtime.cancel("second")
    assert not runtime.has_groups
    assert runtime.discard(group) == ()
    assert group.closed == 1


def test_reset_clears_staged_rows_and_multiple_group_indexes(runtime):
    first = create_group(runtime, "first", "second")
    second = create_group(runtime, "third", "fourth")
    runtime.stage("later", [4])
    runtime.close_all()
    assert not runtime.has_groups
    assert not runtime.is_candidate("later")
    assert all(
        runtime.group_for(request_id) is None
        for request_id in ("first", "second", "third", "fourth")
    )
    assert first.closed == second.closed == 1
