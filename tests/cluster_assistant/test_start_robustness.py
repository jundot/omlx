# SPDX-License-Identifier: Apache-2.0
"""Slice 8: the start job's robustness edges (B13 deadline, B6 save failures).

Fake-driven: a staging job that never finishes must fail with one action rather
than spin forever, and a persistence failure must not crash the job task. No
real Mac, model or SSH.
"""

from __future__ import annotations

from types import SimpleNamespace

from omlx.cluster.assistant import start as assistant_start


def _plan():
    return SimpleNamespace(
        rank_order=("a", "b"), coordinator=SimpleNamespace(node_id="a")
    )


class RecordingStore:
    def __init__(self) -> None:
        self.advanced: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def advance(self, operation_id, state, **kwargs) -> None:
        self.advanced.append(state)

    def fail(self, operation_id, code, message, action) -> None:
        self.failed.append((code, action))


class OSErrorStore:
    def advance(self, *args, **kwargs) -> None:
        raise OSError("disk full")

    def fail(self, *args, **kwargs) -> None:
        raise OSError("disk full")


def test_poll_staging_fails_after_the_deadline() -> None:
    store = RecordingStore()
    ops = SimpleNamespace(
        staging_status=lambda job_id: {"status": "running", "nodes": {}}
    )
    slept: list[float] = []
    assistant_start._poll_staging(
        store,
        "op-1",
        ops,
        "job-1",
        _plan(),
        [],
        1.0,
        lambda seconds: slept.append(seconds),
        deadline_seconds=3.0,
    )
    assert store.failed == [("staging_timeout", assistant_start.ACTION_APPLY)]
    # Three sleeps advance the fake clock to the deadline, then it fails.
    assert len(slept) == 3


def test_poll_staging_unknown_job_also_times_out() -> None:
    # A lost job answers None forever; the deadline still bounds the wait.
    store = RecordingStore()
    ops = SimpleNamespace(staging_status=lambda job_id: None)
    assistant_start._poll_staging(
        store,
        "op-1",
        ops,
        "job-1",
        _plan(),
        [],
        1.0,
        lambda seconds: None,
        deadline_seconds=2.0,
    )
    assert store.failed == [("staging_timeout", assistant_start.ACTION_APPLY)]


def test_advance_survives_a_save_failure() -> None:
    # Must not raise: the true state stays in memory (slice 4) and the job lives.
    assistant_start._advance(
        OSErrorStore(), "op-1", "applying", SimpleNamespace(phase="staging")
    )


def test_fail_survives_a_save_failure() -> None:
    assistant_start._fail(OSErrorStore(), "op-1", "staging_failed", "msg", "apply")
