# SPDX-License-Identifier: Apache-2.0
"""Tests for the server-owned cluster incident store."""

import json

from omlx.cluster import incidents as incidents_module
from omlx.cluster.incidents import (
    IncidentStore,
    Severity,
    configure_cluster_incidents,
    get_cluster_incidents,
)


def _record(
    store: IncidentStore,
    message: str = "boom",
    *,
    severity: Severity = Severity.ERROR,
    source: str = "coordinator",
    state_code: str = "activation_launch_failed",
    **kwargs,
):
    return store.record(severity, source, state_code, message, **kwargs)


def test_seq_strictly_increases_across_records(tmp_path):
    store = IncidentStore(tmp_path)

    sequences = [_record(store, f"failure {index}").seq for index in range(5)]

    assert sequences == sorted(sequences)
    assert len(set(sequences)) == 5
    assert store.latest_seq() == sequences[-1]


def test_ring_cap_evicts_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(incidents_module, "_MAX_INCIDENTS", 5)
    store = IncidentStore(tmp_path)

    for index in range(8):
        _record(store, f"failure {index}")

    listed = store.list()
    assert len(listed) == 5
    assert [incident.message for incident in listed] == [
        f"failure {index}" for index in range(3, 8)
    ]
    # Eviction never reuses sequence numbers.
    assert listed[0].seq == 4


def test_dismiss_persists_across_store_reload(tmp_path):
    store = IncidentStore(tmp_path)
    incident = _record(store)
    assert store.dismiss(incident.id) is True

    reloaded = IncidentStore(tmp_path)

    persisted = {item.id: item for item in reloaded.list()}
    assert persisted[incident.id].dismissed_at is not None
    # Dismissal marks; it never deletes.
    assert len(reloaded.list()) == 1
    assert reloaded.dismiss("no-such-incident") is False


def test_corrupt_incident_log_fails_closed(tmp_path):
    path = tmp_path / "cluster" / "incidents.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    store = IncidentStore(tmp_path)

    assert store.list() == ()
    assert store.load_error is not None
    # The next record clears the error and starts a fresh, valid log.
    _record(store, "after corruption")
    assert store.load_error is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["incidents"][0]["message"] == "after corruption"


def test_epoch_survives_reload_but_changes_on_a_corrupt_reset(tmp_path):
    store = IncidentStore(tmp_path)
    _record(store, "first")
    original_epoch = store.epoch

    # A clean reload keeps the numbering identity: open tabs keep their
    # cursor.
    reloaded = IncidentStore(tmp_path)
    assert reloaded.epoch == original_epoch

    # A corrupt-log reset restarts seq at 1 — a cursor from the old numbering
    # would silence the feed forever, so the epoch must change to tell
    # clients to restart from 0.
    path = tmp_path / "cluster" / "incidents.json"
    path.write_text("{not json", encoding="utf-8")
    reset = IncidentStore(tmp_path)
    assert reset.epoch != original_epoch
    _record(reset, "after corruption")
    assert IncidentStore(tmp_path).epoch == reset.epoch


def test_list_since_seq_excludes_already_seen_records(tmp_path):
    store = IncidentStore(tmp_path)
    first = _record(store, "first")
    second = _record(store, "second")

    assert [item.message for item in store.list()] == ["first", "second"]
    assert [item.message for item in store.list(since_seq=first.seq)] == ["second"]
    assert store.list(since_seq=second.seq) == ()


def test_supersede_marks_but_does_not_delete(tmp_path):
    store = IncidentStore(tmp_path)
    old_attempt = _record(store, "attempt 3 failed", job_id="job-3")
    unrelated = _record(store, "different job", job_id="job-9")
    replacement = _record(store, "attempt 4 failed", job_id="job-3")

    changed = store.supersede("job-3", replacement.id)

    assert changed == 1
    by_id = {item.id: item for item in store.list()}
    assert len(by_id) == 3
    assert by_id[old_attempt.id].superseded_by == replacement.id
    assert by_id[unrelated.id].superseded_by is None
    assert by_id[replacement.id].superseded_by is None


def test_reload_continues_sequence_without_reuse(tmp_path):
    store = IncidentStore(tmp_path)
    last = _record(store, "before restart").seq

    reloaded = IncidentStore(tmp_path)
    after = _record(reloaded, "after restart").seq

    assert after > last


def test_configure_and_get_module_pair(tmp_path):
    configured = configure_cluster_incidents(tmp_path)
    try:
        assert get_cluster_incidents() is configured
    finally:
        incidents_module._configured_incidents = None
