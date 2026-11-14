# SPDX-License-Identifier: Apache-2.0
"""The persisted apply/start operation.

These pin the qualities an operation record must have before a UI can trust it:

* states move forward only: a late event cannot undo ``ready`` or revive
  ``failed``;
* one apply/start at a time, and the refusal names the running operation;
* a record survives a restart, and a job that was still running becomes
  ``failed`` rather than a ghost the UI polls forever;
* the file is atomic, bounded, mode 0600 and fails closed;
* nothing secret is stored — no SSH target, address, key, argv or model file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from omlx.cluster.assistant import operations as operations_module
from omlx.cluster.assistant.operations import (
    ACTION_APPLY,
    ACTION_DIAGNOSTICS,
    ACTION_START,
    KIND_APPLY,
    KIND_START,
    MAX_OPERATION_RECORDS,
    STATE_APPLIED,
    STATE_APPLYING,
    STATE_FAILED,
    STATE_LOADING,
    STATE_QUEUED,
    STATE_READY,
    STATE_STARTING,
    OperationEndpoint,
    OperationInProgressError,
    OperationProgress,
    OperationStore,
    new_operation_id,
)

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


def endpoint() -> OperationEndpoint:
    return OperationEndpoint(node="m5", host="172.16.1.11", port=8099)


def store(tmp_path) -> OperationStore:
    return OperationStore(tmp_path)


def applied_operation(tmp_path, *, store_instance: OperationStore | None = None):
    operations = store_instance or store(tmp_path)
    record = operations.start(
        kind=KIND_APPLY,
        measurement_id="ms-1",
        plan_signature="a" * 64,
        coordinator_node="m5",
        rank_order=("m5", "m1"),
        context_tokens=8192,
        endpoint=endpoint(),
        now=NOW,
    )
    operations.advance(record.operation_id, STATE_APPLYING, now=NOW)
    operations.advance(record.operation_id, STATE_APPLIED, now=NOW)
    return operations, operations.get(record.operation_id)


# ---------------------------------------------------------------------------
# Identity and persistence
# ---------------------------------------------------------------------------


def test_operation_ids_are_opaque_ordered_and_kind_tagged() -> None:
    first = new_operation_id(KIND_APPLY, now=NOW)
    second = new_operation_id(KIND_START, now=NOW)

    assert first.startswith("op-20261103T120000Z-apply-")
    assert second.startswith("op-20261103T120000Z-start-")
    assert first != second
    assert len(first) == len("op-20261103T120000Z-apply-") + 8


def test_an_applied_operation_survives_a_restart_intact(tmp_path) -> None:
    operations, record = applied_operation(tmp_path)
    assert record is not None
    assert operations.path == tmp_path / "cluster" / "assistant-operations.json"

    reopened = OperationStore(tmp_path)
    assert reopened.load_error is None
    loaded = reopened.get(record.operation_id)
    assert loaded is not None
    assert loaded.state == STATE_APPLIED
    assert loaded.kind == KIND_APPLY
    assert loaded.measurement_id == "ms-1"
    assert loaded.plan_signature == "a" * 64
    assert loaded.coordinator_node == "m5"
    assert loaded.rank_order == ("m5", "m1")
    assert loaded.context_tokens == 8192
    assert loaded.endpoint == OperationEndpoint("m5", "172.16.1.11", 8099)
    assert loaded.error is None
    # An applied record is finished, so it may not become "interrupted".
    assert reopened.recovered == 0


def test_the_file_is_private_atomic_and_bounded(tmp_path) -> None:
    operations, _ = applied_operation(tmp_path)

    assert operations.path.stat().st_mode & 0o777 == 0o600
    assert json.loads(operations.path.read_text())["schema_version"] == 1
    assert list(operations.path.parent.glob(".assistant-operations.*.tmp")) == []

    for _ in range(MAX_OPERATION_RECORDS + 4):
        record = operations.start(kind=KIND_APPLY, measurement_id="ms-2", now=NOW)
        operations.fail(
            record.operation_id,
            code="staging_failed",
            message="no",
            action=ACTION_APPLY,
            now=NOW,
        )

    assert len(operations.list()) == MAX_OPERATION_RECORDS
    reopened = OperationStore(tmp_path)
    assert len(reopened.list()) == MAX_OPERATION_RECORDS


def test_a_corrupt_file_fails_closed_instead_of_stopping_omlx(tmp_path) -> None:
    path = tmp_path / "cluster" / "assistant-operations.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")

    operations = OperationStore(tmp_path)

    assert operations.load_error is not None
    assert operations.list() == ()
    assert operations.get("op-anything") is None


def test_a_running_operation_is_recovered_as_failed(tmp_path) -> None:
    operations = store(tmp_path)
    apply_record = operations.start(kind=KIND_APPLY, measurement_id="ms-1", now=NOW)
    operations.advance(apply_record.operation_id, STATE_APPLYING, now=NOW)
    operations.fail(
        apply_record.operation_id,
        code="staging_failed",
        message="no",
        action=ACTION_APPLY,
        now=NOW,
    )
    start_record = operations.start(kind=KIND_START, measurement_id="ms-2", now=NOW)
    operations.advance(start_record.operation_id, STATE_STARTING, now=NOW)
    operations.advance(start_record.operation_id, STATE_LOADING, now=NOW)

    reopened = OperationStore(tmp_path)

    assert reopened.recovered == 1
    recovered = reopened.get(start_record.operation_id)
    assert recovered is not None
    assert recovered.state == STATE_FAILED
    assert recovered.error is not None
    assert recovered.error.code == "operation_interrupted"
    assert recovered.error.action == ACTION_START


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


def test_states_move_forward_only(tmp_path) -> None:
    operations, record = applied_operation(tmp_path)
    assert record is not None
    operation_id = record.operation_id

    # applied is terminal: a start belongs to its own record, and a late
    # failure cannot rewrite the finished apply.
    assert operations.advance(operation_id, STATE_STARTING).state == STATE_APPLIED
    assert (
        operations.fail(
            operation_id, code="late", message="late", action=ACTION_DIAGNOSTICS
        ).state
        == STATE_APPLIED
    )

    start_record = operations.start(kind=KIND_START, measurement_id="ms-1", now=NOW)
    operations.advance(start_record.operation_id, STATE_STARTING, now=NOW)
    ready = operations.advance(
        start_record.operation_id,
        STATE_READY,
        progress=OperationProgress(phase="ready", message="done"),
        endpoint=OperationEndpoint("m5", "172.16.1.11", 8099, ready=True),
        now=NOW,
    )
    assert ready.state == STATE_READY

    # A late "applying" may not follow "ready".
    late = operations.advance(start_record.operation_id, STATE_APPLYING)
    assert late.state == STATE_READY
    assert late.endpoint is not None and late.endpoint.ready is True
    assert (
        operations.fail(
            start_record.operation_id,
            code="late",
            message="late",
            action=ACTION_DIAGNOSTICS,
        ).state
        == STATE_READY
    )


def test_progress_updates_inside_a_running_state_are_kept(tmp_path) -> None:
    operations = store(tmp_path)
    record = operations.start(kind=KIND_APPLY, measurement_id="ms-1", now=NOW)
    operations.advance(
        record.operation_id,
        STATE_APPLYING,
        progress=OperationProgress(
            phase="staging", message="staged 1 of 2 Macs", nodes_ready=1, nodes_total=2
        ),
        placement_signature="b" * 16,
        now=NOW,
    )

    updated = operations.get(record.operation_id)

    assert updated is not None
    assert updated.state == STATE_APPLYING
    assert updated.progress.nodes_ready == 1
    assert updated.progress.nodes_total == 2
    assert updated.placement_signature == "b" * 16


def test_one_apply_or_start_at_a_time(tmp_path) -> None:
    operations = store(tmp_path)
    running = operations.start(kind=KIND_APPLY, measurement_id="ms-1", now=NOW)

    try:
        operations.start(kind=KIND_START, measurement_id="ms-1", now=NOW)
    except OperationInProgressError as exc:
        assert exc.operation_id == running.operation_id
        assert exc.kind == KIND_APPLY
        assert "poll it" in str(exc)
    else:  # pragma: no cover - the refusal is the contract
        raise AssertionError("a second operation must be refused, not joined")

    # applied frees the slot: staging finished, so a start may follow.
    operations.advance(running.operation_id, STATE_APPLYING, now=NOW)
    operations.advance(running.operation_id, STATE_APPLIED, now=NOW)
    assert operations.active() is None
    follow_up = operations.start(kind=KIND_START, measurement_id="ms-1", now=NOW)
    assert follow_up.kind == KIND_START


def test_the_applied_lookup_matches_every_identity(tmp_path) -> None:
    operations, record = applied_operation(tmp_path)
    assert record is not None

    assert (
        operations.applied(
            measurement_id="ms-1",
            plan_signature="a" * 64,
            coordinator_node="m5",
        )
        is not None
    )
    assert (
        operations.applied(
            measurement_id="ms-1",
            plan_signature="a" * 64,
            coordinator_node="m1",
        )
        is None
    )
    assert (
        operations.applied(
            measurement_id="ms-other",
            plan_signature="a" * 64,
            coordinator_node="m5",
        )
        is None
    )


def test_the_applied_lookup_returns_the_newest_match(tmp_path) -> None:
    """Resume/stop must build on the latest applied plan, not a stale one (B7)."""

    operations = store(tmp_path)
    identities = {
        "measurement_id": "ms-1",
        "plan_signature": "a" * 64,
        "coordinator_node": "m5",
    }
    older = operations.start(kind=KIND_APPLY, now=NOW, **identities)
    operations.advance(older.operation_id, STATE_APPLYING, now=NOW)
    operations.advance(older.operation_id, STATE_APPLIED, now=NOW)

    later = NOW + timedelta(seconds=60)
    newer = operations.start(kind=KIND_APPLY, now=later, **identities)
    operations.advance(newer.operation_id, STATE_APPLYING, now=later)
    operations.advance(newer.operation_id, STATE_APPLIED, now=later)

    found = operations.applied(**identities)
    assert found is not None
    assert found.operation_id == newer.operation_id

    # A still newer apply with a different identity must not shadow the match.
    newest = operations.start(
        kind=KIND_APPLY,
        measurement_id="ms-1",
        plan_signature="b" * 64,
        coordinator_node="m5",
        now=later + timedelta(seconds=60),
    )
    operations.advance(newest.operation_id, STATE_APPLYING, now=later)
    operations.advance(newest.operation_id, STATE_APPLIED, now=later)
    found = operations.applied(**identities)
    assert found is not None
    assert found.operation_id == newer.operation_id


def test_nothing_secret_is_persisted(tmp_path) -> None:
    operations, _ = applied_operation(tmp_path)

    raw = operations.path.read_text()

    for forbidden in ("ssh", "api_key", "argv", "password", "@"):
        assert forbidden not in raw
    # One endpoint host and port are identifiers a client must dial, not
    # credentials — and the record stores exactly one of them. The checkpoint
    # path is likewise an identifier the reloaded card needs, never a key.
    assert "172.16.1.11" in raw
    assert raw.count('"host"') == 1
    assert "model_path" in raw


def test_unknown_states_are_refused(tmp_path) -> None:
    operations = store(tmp_path)
    record = operations.start(kind=KIND_APPLY, measurement_id="ms-1", now=NOW)

    for state in ("done", STATE_FAILED):
        try:
            operations.advance(record.operation_id, state)
        except ValueError:
            pass
        else:  # pragma: no cover - fail() is the only producer of failed
            raise AssertionError(f"{state} must not be settable through advance")


def test_the_record_reports_its_own_state_truthfully(tmp_path) -> None:
    operations = store(tmp_path)
    queued = operations.start(kind=KIND_APPLY, measurement_id="ms-1", now=NOW)
    assert queued.state == STATE_QUEUED
    assert queued.running is True
    assert queued.finished is False

    operations.advance(queued.operation_id, STATE_APPLYING, now=NOW)
    failed = operations.fail(
        queued.operation_id,
        code="apply_failed",
        message="boom",
        action=ACTION_DIAGNOSTICS,
        now=NOW,
    )

    assert failed.running is False
    assert failed.finished is True
    assert failed.error is not None
    assert set(failed.to_dict()["error"]) == {"code", "message", "action"}


# ---------------------------------------------------------------------------
# Mirrors: the cluster-wide lock on a remote coordinator (B5, CHAPTER-5 §2.2)
# ---------------------------------------------------------------------------


def test_a_running_mirror_takes_the_lock(tmp_path) -> None:
    operations = store(tmp_path)
    # The real clock: ``active()`` and ``start()`` read it, and a mirror dated
    # in the past would already be past the stale TTL.
    mirrored = operations.mirror(
        "op-origin-a",
        KIND_APPLY,
        "m5",
        STATE_APPLYING,
        model_path="/models/qwen",
        context_tokens=8192,
    )

    assert mirrored.mirrored is True
    assert mirrored.operation_id == "op-origin-a"
    assert mirrored.model_path == "/models/qwen"
    assert mirrored.context_tokens == 8192
    active = operations.active()
    assert active is not None
    assert active.operation_id == "op-origin-a"

    # The coordinator's own start is refused while the mirror runs.
    with pytest.raises(OperationInProgressError) as excinfo:
        operations.start(kind=KIND_START, measurement_id="ms-1", now=NOW)
    assert excinfo.value.operation_id == "op-origin-a"
    # A second origin's running mirror is refused with the running ID too.
    with pytest.raises(OperationInProgressError) as excinfo:
        operations.mirror("op-origin-b", KIND_START, "m5", STATE_STARTING, now=NOW)
    assert excinfo.value.operation_id == "op-origin-a"


def test_a_running_mirror_updates_forward(tmp_path) -> None:
    operations = store(tmp_path)
    operations.mirror("op-origin-a", KIND_START, "m5", STATE_STARTING)

    updated = operations.mirror("op-origin-a", KIND_START, "m5", STATE_LOADING)

    assert updated.state == STATE_LOADING
    assert updated.mirrored is True
    active = operations.active()
    assert active is not None
    assert active.operation_id == "op-origin-a"


def test_a_terminal_mirror_releases_the_lock(tmp_path) -> None:
    operations = store(tmp_path)
    operations.mirror("op-origin-a", KIND_APPLY, "m5", STATE_APPLYING, now=NOW)

    released = operations.mirror("op-origin-a", KIND_APPLY, "m5", STATE_APPLIED, now=NOW)

    assert released.state == STATE_APPLIED
    assert released.mirrored is True
    assert operations.active() is None
    follow_up = operations.start(kind=KIND_START, measurement_id="ms-1", now=NOW)
    assert follow_up.running is True


def test_a_failed_mirror_releases_the_lock(tmp_path) -> None:
    operations = store(tmp_path)
    operations.mirror("op-origin-a", KIND_START, "m5", STATE_LOADING, now=NOW)

    failed = operations.mirror("op-origin-a", KIND_START, "m5", STATE_FAILED, now=NOW)

    assert failed.state == STATE_FAILED
    assert failed.error is not None
    assert operations.active() is None


def test_a_terminal_mirror_without_a_running_record_is_inserted(tmp_path) -> None:
    """A release that arrives after a coordinator restart still lands."""

    operations = store(tmp_path)
    operations.mirror("op-origin-a", KIND_START, "m5", STATE_READY, now=NOW)

    assert operations.active() is None
    record = operations.get("op-origin-a")
    assert record is not None
    assert record.state == STATE_READY
    assert record.mirrored is True


def test_mirrors_are_forward_only(tmp_path) -> None:
    operations = store(tmp_path)
    operations.mirror("op-origin-a", KIND_START, "m5", STATE_STARTING, now=NOW)
    operations.mirror("op-origin-a", KIND_START, "m5", STATE_READY, now=NOW)

    # A late running mirror cannot resurrect a terminal record.
    late = operations.mirror("op-origin-a", KIND_START, "m5", STATE_LOADING, now=NOW)
    assert late.state == STATE_READY
    # A late failure cannot undo ready.
    late_failure = operations.mirror(
        "op-origin-a", KIND_START, "m5", STATE_FAILED, now=NOW
    )
    assert late_failure.state == STATE_READY
    assert operations.active() is None


def test_latest_session_and_applied_skip_mirrors(tmp_path) -> None:
    """Mirrors are lock markers, not the coordinator's own sessions."""

    operations = store(tmp_path)
    operations.mirror("op-origin-a", KIND_START, "m5", STATE_READY, now=NOW)
    operations.mirror("op-origin-b", KIND_APPLY, "m5", STATE_APPLIED, now=NOW)

    assert operations.latest_session() is None
    assert (
        operations.applied(
            measurement_id="", plan_signature="", coordinator_node="m5"
        )
        is None
    )

    # A real session of the coordinator's own still wins.
    own = operations.start(kind=KIND_APPLY, measurement_id="ms-1", now=NOW)
    operations.advance(own.operation_id, STATE_APPLYING, now=NOW)
    operations.advance(own.operation_id, STATE_APPLIED, now=NOW)
    session = operations.latest_session()
    assert session is not None
    assert session.operation_id == own.operation_id


def test_a_stale_mirror_stops_counting_as_running(tmp_path) -> None:
    """A crashed origin must not wedge the coordinator's lock forever."""

    ttl = operations_module.MIRROR_TTL_SECONDS
    operations = store(tmp_path)
    operations.mirror("op-origin-a", KIND_APPLY, "m5", STATE_APPLYING, now=NOW)

    fresh = NOW + timedelta(seconds=ttl - 1)
    active = operations.active(now=fresh)
    assert active is not None
    assert active.operation_id == "op-origin-a"
    stale = NOW + timedelta(seconds=ttl + 1)
    assert operations.active(now=stale) is None

    # With a mirror old enough on the real clock, a new operation may start.
    old = datetime.now(UTC) - timedelta(seconds=ttl + 60)
    operations.mirror("op-crashed", KIND_APPLY, "m5", STATE_APPLYING, now=old)
    assert operations.active() is None
    record = operations.start(kind=KIND_APPLY, measurement_id="ms-1")
    assert record.running is True


def test_a_mirror_survives_a_restart_and_recovers_as_failed(tmp_path) -> None:
    operations = store(tmp_path)
    operations.mirror("op-origin-a", KIND_APPLY, "m5", STATE_APPLYING, now=NOW)

    reopened = OperationStore(tmp_path)
    loaded = reopened.get("op-origin-a")

    assert loaded is not None
    assert loaded.mirrored is True
    # A job cannot outlive its process — a mirror neither.
    assert loaded.state == STATE_FAILED
    assert loaded.error is not None
    assert loaded.error.code == "operation_interrupted"
    assert reopened.active() is None


def test_a_mirror_never_rewrites_the_coordinators_own_record(tmp_path) -> None:
    operations = store(tmp_path)
    own = operations.start(kind=KIND_APPLY, measurement_id="ms-1", now=NOW)
    operations.advance(own.operation_id, STATE_APPLYING, now=NOW)
    operations.advance(own.operation_id, STATE_APPLIED, now=NOW)

    with pytest.raises(ValueError):
        operations.mirror(own.operation_id, KIND_APPLY, "m5", STATE_FAILED, now=NOW)

    kept = operations.get(own.operation_id)
    assert kept is not None
    assert kept.state == STATE_APPLIED
    assert kept.mirrored is False


def test_a_mirror_validates_kind_and_state(tmp_path) -> None:
    operations = store(tmp_path)

    with pytest.raises(ValueError):
        operations.mirror("op-origin-a", "deploy", "m5", STATE_APPLYING, now=NOW)
    with pytest.raises(ValueError):
        operations.mirror("op-origin-a", KIND_APPLY, "m5", "done", now=NOW)
    assert operations.list() == ()


def test_a_terminal_mirror_is_kept_when_the_save_fails(tmp_path, monkeypatch) -> None:
    """Slice 4's rule holds for mirrors: a terminal state survives a bad save."""

    operations = store(tmp_path)
    operations.mirror("op-origin-a", KIND_APPLY, "m5", STATE_APPLYING, now=NOW)

    def broken_save() -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(operations, "_save", broken_save)
    with pytest.raises(OSError):
        operations.mirror("op-origin-a", KIND_APPLY, "m5", STATE_APPLIED, now=NOW)

    kept = operations.get("op-origin-a")
    assert kept is not None
    assert kept.state == STATE_APPLIED
    assert operations.active() is None
