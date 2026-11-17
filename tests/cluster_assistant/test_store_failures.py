# SPDX-License-Identifier: Apache-2.0
"""Store write failures must not wedge the assistant (review B6).

An unwritable store directory used to roll a finished measurement or operation
back to its last persisted *running* state: the job was over, but the record
polled forever and every new job was refused as "already running". These tests
pin the recovery contract against a real unwritable directory (mode 0o500, no
filesystem mocks):

* a terminal state (ready/failed, applied/ready/failed) stays in memory even
  when the persistence write fails, and the save error still reaches the
  caller instead of being hidden by a rollback;
* the measurement job surfaces the failure as ``failed`` with one action
  instead of swallowing the OSError, and the status poll reports it;
* once the directory is writable again, the next job persists normally.

Fake-driven per AGENTS.md: the probes are the same shape as
``test_measure_api``'s, so no test touches a Mac, a model or an endpoint.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.measurement import measure_budget_payloads
from omlx.cluster.assistant.models import AssistantMeasureRequest
from omlx.cluster.assistant.operations import (
    KIND_APPLY,
    KIND_START,
    STATE_APPLIED,
    STATE_APPLYING,
    STATE_FAILED,
    STATE_LOADING,
    STATE_QUEUED,
    STATE_READY,
    STATE_STARTING,
    OperationEndpoint,
    OperationStore,
)
from omlx.cluster.assistant.state import (
    MEASUREMENT_FAILED,
    MEASUREMENT_READY,
    MEASUREMENT_RUNNING,
    MeasurementEndpoint,
    MeasurementStore,
    ModelIdentity,
)
from omlx.cluster.node_role import NodeBudgetSuggestion
from omlx.cluster.transport import TransportInfo, TransportMatrix

GIB = 1024**3
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
MEASURED_AT = NOW.isoformat()
MODEL = ModelIdentity(model_size_bytes=40 * GIB, layer_count=20)
STATUS_URL = "/admin/api/cluster/assistant/status"
SSH = {"m5": "192.0.2.11", "m1": "192.0.2.14"}
CAPACITY_GIB = {"m5": 107.5, "m1": 60.0}
USABLE_GIB = {"m5": 96.75, "m1": 54.0}


@contextlib.contextmanager
def unwritable(directory: Path):
    """Make *directory* unwritable, proving the write really fails.

    Skips when the process can write anyway (e.g. root): a green test that
    never exercised the failure path would be worse than no test. The mode is
    restored even when the test fails, so pytest can clean the tmp directory.
    """

    original = directory.stat().st_mode & 0o777
    directory.chmod(0o500)
    try:
        probe = directory / ".write-probe"
        try:
            probe.touch()
        except OSError:
            pass
        else:
            probe.unlink()
            pytest.skip("directory modes do not restrict this process (root?)")
        yield
    finally:
        directory.chmod(original)


def measured():
    """One finished measurement's facts, built from fake budgets."""

    return measure_budget_payloads(
        [
            {
                "node_id": "m5",
                "capacity_bytes": int(107.5 * GIB),
                "reserve_bytes": int((107.5 - 96.75) * GIB),
                "usable_bytes": int(96.75 * GIB),
                "role": "headless",
                "capacity_source": "admission_ceiling",
                "endpoint_host": SSH["m5"],
                "endpoint_port": 8099,
                "disk_free_bytes": 300 * GIB,
            }
        ],
        measured_at=MEASURED_AT,
    )


ENDPOINTS = {"m5": MeasurementEndpoint(host=SSH["m5"], port=8099)}


# ---------------------------------------------------------------------------
# Measurement store: a terminal state survives a failed save
# ---------------------------------------------------------------------------


def test_a_ready_measurement_is_not_rolled_back_by_a_failed_save(tmp_path) -> None:
    store = MeasurementStore(tmp_path)
    record = store.start(model=MODEL, node_ids=["m5"], now=NOW)

    with unwritable(store.path.parent):
        with pytest.raises(OSError):
            store.finish(
                record.measurement_id,
                measurement=measured(),
                endpoints=ENDPOINTS,
                now=NOW,
            )
        kept = store.get(record.measurement_id)
        assert kept is not None
        # The job finished; the unwritable file may not pretend it is running.
        assert kept.state == MEASUREMENT_READY
        assert kept.measured_at == MEASURED_AT
        assert store.active() is None

    # Recovery: once the directory is writable again the same finish persists.
    store.finish(
        record.measurement_id,
        measurement=measured(),
        endpoints=ENDPOINTS,
        now=NOW,
    )
    reopened = MeasurementStore(tmp_path)
    loaded = reopened.get(record.measurement_id)
    assert loaded is not None and loaded.state == MEASUREMENT_READY


def test_a_failed_measurement_is_not_rolled_back_by_a_failed_save(tmp_path) -> None:
    store = MeasurementStore(tmp_path)
    record = store.start(model=MODEL, node_ids=["m5"], now=NOW)

    with unwritable(store.path.parent):
        with pytest.raises(OSError):
            store.fail(
                record.measurement_id, code="probe_failed", message="no route"
            )
        kept = store.get(record.measurement_id)
        assert kept is not None
        assert kept.state == MEASUREMENT_FAILED
        assert kept.error is not None
        assert kept.error.code == "probe_failed"
        # The wedge: a rolled-back "running" record refuses every new measure.
        assert store.active() is None

    follow_up = store.start(model=MODEL, node_ids=["m5"], now=NOW)
    assert follow_up.state == MEASUREMENT_RUNNING


# ---------------------------------------------------------------------------
# Operation store: the same rollback pattern, the same contract
# ---------------------------------------------------------------------------


def test_an_applied_operation_is_not_rolled_back_by_a_failed_save(tmp_path) -> None:
    operations = OperationStore(tmp_path)
    record = operations.start(
        kind=KIND_APPLY,
        measurement_id="ms-1",
        plan_signature="a" * 64,
        coordinator_node="m5",
        now=NOW,
    )
    operations.advance(record.operation_id, STATE_APPLYING, now=NOW)

    with unwritable(operations.path.parent):
        with pytest.raises(OSError):
            operations.advance(record.operation_id, STATE_APPLIED, now=NOW)
        kept = operations.get(record.operation_id)
        assert kept is not None
        assert kept.state == STATE_APPLIED
        assert operations.active() is None
        assert (
            operations.applied(
                measurement_id="ms-1",
                plan_signature="a" * 64,
                coordinator_node="m5",
            )
            is not None
        )


def test_a_ready_operation_is_not_rolled_back_by_a_failed_save(tmp_path) -> None:
    operations = OperationStore(tmp_path)
    record = operations.start(kind=KIND_START, measurement_id="ms-1", now=NOW)
    operations.advance(record.operation_id, STATE_STARTING, now=NOW)
    operations.advance(record.operation_id, STATE_LOADING, now=NOW)

    with unwritable(operations.path.parent):
        with pytest.raises(OSError):
            operations.advance(
                record.operation_id,
                STATE_READY,
                endpoint=OperationEndpoint("m5", SSH["m5"], 8099, ready=True),
                now=NOW,
            )
        kept = operations.get(record.operation_id)
        assert kept is not None
        assert kept.state == STATE_READY
        assert kept.endpoint is not None and kept.endpoint.ready is True
        assert operations.active() is None


def test_a_failed_operation_is_not_rolled_back_by_a_failed_save(tmp_path) -> None:
    operations = OperationStore(tmp_path)
    record = operations.start(kind=KIND_APPLY, measurement_id="ms-1", now=NOW)

    with unwritable(operations.path.parent):
        with pytest.raises(OSError):
            operations.fail(
                record.operation_id,
                code="staging_failed",
                message="no",
                action="apply",
                now=NOW,
            )
        kept = operations.get(record.operation_id)
        assert kept is not None
        assert kept.state == STATE_FAILED
        assert kept.error is not None
        assert kept.error.code == "staging_failed"
        # A failed operation frees the slot even when the file cannot be read
        # back; otherwise every later apply/start is refused forever.
        assert operations.active() is None


def test_a_stop_is_not_undone_by_a_failed_save(tmp_path) -> None:
    operations = OperationStore(tmp_path)
    record = operations.start(kind=KIND_START, measurement_id="ms-1", now=NOW)
    operations.advance(record.operation_id, STATE_STARTING, now=NOW)
    operations.advance(record.operation_id, STATE_READY, now=NOW)

    with unwritable(operations.path.parent):
        with pytest.raises(OSError):
            operations.mark_stopped(record.operation_id, now=NOW)
        kept = operations.get(record.operation_id)
        assert kept is not None
        assert kept.state == STATE_READY
        # The deployment really was stopped; a failed save may not present it
        # as running again.
        assert kept.stopped is True


def test_an_unpersisted_running_transition_still_rolls_back(tmp_path) -> None:
    """Only terminal states are kept; a running transition stays honest."""

    operations = OperationStore(tmp_path)
    record = operations.start(kind=KIND_APPLY, measurement_id="ms-1", now=NOW)

    with unwritable(operations.path.parent):
        with pytest.raises(OSError):
            operations.advance(record.operation_id, STATE_APPLYING, now=NOW)
        kept = operations.get(record.operation_id)
        assert kept is not None
        assert kept.state == STATE_QUEUED


# ---------------------------------------------------------------------------
# The measurement job surfaces the failure instead of swallowing it
# ---------------------------------------------------------------------------


def fake_probes() -> assistant_routes.MeasureProbes:
    """The test_measure_api probe shapes, without a gate: no Mac is reached."""

    def admission_ceiling(ssh: str, python_executable: str | None = None) -> int:
        node_id = next(key for key, value in SSH.items() if value == ssh)
        return int(CAPACITY_GIB[node_id] * GIB)

    def budget(**kwargs) -> NodeBudgetSuggestion:
        ssh = kwargs.get("ssh_target")
        node_id = next(key for key, value in SSH.items() if value == ssh)
        capacity = int(CAPACITY_GIB[node_id] * GIB)
        return NodeBudgetSuggestion(
            capacity_bytes=capacity,
            reserve_bytes=int((CAPACITY_GIB[node_id] - USABLE_GIB[node_id]) * GIB),
            role=kwargs.get("role", "headless"),
            capacity_source="admission_ceiling",
        )

    return assistant_routes.MeasureProbes(
        admission_ceiling=admission_ceiling,
        budget=budget,
        transports=lambda hosts: TransportMatrix(
            transports=(
                TransportInfo(
                    kind="thunderbolt",
                    interface="en5",
                    peer_node_id=SSH["m1"],
                    source_node_id=SSH["m5"],
                    link_speed_gbps=80,
                ),
            ),
            backend="jaccl-ring",
        ),
        model_inventory=lambda ssh, python_executable=None: [
            {"model_path": "/models/qwen", "id": "qwen"}
        ],
        disk_free=lambda path, ssh_target=None: 300 * GIB,
        device_record=lambda node_id: (
            {"last_addrs": [SSH[node_id]]} if node_id in SSH else None
        ),
    )


def hardware_only_request() -> AssistantMeasureRequest:
    """The Phase C3 entry measurement: no model, so no layout read either."""

    return AssistantMeasureRequest.model_validate(
        {
            "nodes": [
                {"node_id": node_id, "ssh": ssh} for node_id, ssh in SSH.items()
            ],
            "roles": {"m5": "headless", "m1": "headless"},
        }
    )


@pytest.fixture()
def wired_routes(tmp_path):
    """Point the assistant module at tmp stores; restore the globals after."""

    store = MeasurementStore(tmp_path / "store")
    assistant_routes.configure_assistant_measurements(store, probes=fake_probes())
    assistant_routes.set_assistant_getters(
        lambda: SimpleNamespace(base_path=tmp_path),
        lambda: SimpleNamespace(port=8099, host="127.0.0.1"),
    )
    yield store
    assistant_routes._store = None
    assistant_routes._probes = None
    assistant_routes._performance = None
    assistant_routes.set_assistant_getters(None, None, None)


def test_the_measure_job_surfaces_an_unwritable_store(
    tmp_path, wired_routes, caplog
) -> None:
    store = wired_routes
    request = hardware_only_request()
    node_ids = [node.node_id for node in request.nodes]
    record = store.start(model=request.model_identity(), node_ids=node_ids, now=NOW)

    with unwritable(store.path.parent):
        with caplog.at_level(logging.WARNING):
            # The worker-thread body, run synchronously: the probes succeed,
            # only the persistence of the result fails.
            assistant_routes._measure_job(
                store, fake_probes(), record.measurement_id, request
            )
        kept = store.get(record.measurement_id)
        assert kept is not None
        # Not stuck "running": the failure is explicit, with one next step.
        assert kept.state == MEASUREMENT_FAILED
        assert kept.error is not None
        assert kept.error.code == "measurement_not_stored"
        assert kept.error.action == "measure"
        assert store.active() is None

    # The persistence failure is logged, not silent.
    assert any(
        item.name == "omlx.cluster.assistant.routes"
        and "could not persist" in item.message
        and record.measurement_id in item.message
        for item in caplog.records
    )

    # The status route surfaces the same failure to a polling card.
    app = FastAPI()
    app.include_router(assistant_routes.router)
    with TestClient(app) as api:
        response = api.get(
            STATUS_URL, params={"measurement_id": record.measurement_id}
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "failed"
    assert body["error"]["code"] == "measurement_not_stored"
    assert body["error"]["action"] == "measure"

    # Recovery: writable again, the next measurement runs and persists ready.
    second = store.start(model=request.model_identity(), node_ids=node_ids, now=NOW)
    assistant_routes._measure_job(store, fake_probes(), second.measurement_id, request)
    assert store.get(second.measurement_id).state == MEASUREMENT_READY
    reopened = MeasurementStore(tmp_path / "store")
    loaded = reopened.get(second.measurement_id)
    assert loaded is not None and loaded.state == MEASUREMENT_READY


def test_fail_job_stays_silent_for_a_pruned_record(wired_routes, caplog) -> None:
    """A pruned record is still nothing to tell the caller — and nothing to log."""

    with caplog.at_level(logging.WARNING):
        assistant_routes._fail_job(
            wired_routes, "ms-gone", code="x", message="y", action="measure"
        )
    assert not [
        item
        for item in caplog.records
        if item.name == "omlx.cluster.assistant.routes"
    ]
