# SPDX-License-Identifier: Apache-2.0
"""Tests for the server-owned Start Cluster job (B2).

The orchestration is driven directly with injected phase fakes — the same
seams ``routes._run_cluster_start_job`` wires to ``_autoconfigure`` /
``_run_staging_job`` / ``_activate`` — so these tests prove the ladder,
the failure narration and the store semantics without a server.
"""

import asyncio

import pytest

from omlx.cluster.incidents import Severity
from omlx.cluster.start_job import (
    StartJobConflictError,
    StartJobPhase,
    StartJobStore,
    run_start_job,
)


class _IncidentLog:
    """Records every incident the runner narrates; returns synthetic ids."""

    def __init__(self):
        self.records = []

    def __call__(
        self,
        severity,
        state_code,
        message,
        *,
        job_id=None,
        deployment_id=None,
    ):
        self.records.append(
            {
                "severity": severity,
                "state_code": state_code,
                "message": message,
                "job_id": job_id,
                "deployment_id": deployment_id,
            }
        )
        return f"incident-{len(self.records)}"


def _proposal(**overrides):
    proposal = {
        "fabric_ready": True,
        "fabric_blocker": "",
        "staging": None,
        "ready_to_activate": True,
        "preflight": "",
        "activation": {"model_path": "/models/nemotron"},
    }
    proposal.update(overrides)
    return proposal


def _create_job(store, model_path="/models/nemotron"):
    job, superseded = store.create(
        model_path=model_path,
        hosts=["127.0.0.1", "studio.local"],
    )
    return job["job_id"], superseded


def _run(store, job_id, *, incidents=None, **overrides):
    incidents = incidents if incidents is not None else _IncidentLog()

    async def link_setup():
        return None

    async def autoconfigure():
        return _proposal()

    async def start_staging(proposal):
        return "a" * 24

    async def wait_staging(staging_job_id):
        return {"ready": True, "status": "completed", "error": ""}

    async def activate(activation):
        return {"ok": True, "deployment": {"deployment_id": "dep-1"}}

    kwargs = {
        "link_setup": link_setup,
        "autoconfigure": autoconfigure,
        "start_staging": start_staging,
        "wait_staging": wait_staging,
        "activate": activate,
        "record_incident": incidents,
    }
    kwargs.update(overrides)
    asyncio.run(run_start_job(store, job_id, **kwargs))
    return incidents


def _watch_phases(store, job_id):
    phases = [store.get(job_id)["phase"]]
    original_update = store.update

    def logging_update(updated_job_id, **fields):
        if "phase" in fields and updated_job_id == job_id:
            phases.append(str(fields["phase"]))
        return original_update(updated_job_id, **fields)

    store.update = logging_update
    return phases


def test_happy_path_walks_queued_to_ready_and_records_an_info_incident():
    store = StartJobStore()
    job_id, _ = _create_job(store)
    phases = _watch_phases(store, job_id)

    incidents = _run(store, job_id)

    assert phases == [
        "queued",
        "link_setup",
        "autoconfigure",
        "activating",
        "ready",
    ]
    record = store.get(job_id)
    assert record["phase"] == "ready"
    assert record["error"] == ""
    assert record["result"] == {
        "ok": True,
        "deployment": {"deployment_id": "dep-1"},
    }
    assert record["incident_id"] == "incident-1"
    assert incidents.records == [
        {
            "severity": Severity.INFO,
            "state_code": "start_job_ready",
            "message": (
                "Cluster start finished: /models/nemotron is ready."
            ),
            "job_id": job_id,
            "deployment_id": "dep-1",
        }
    ]


def test_staging_phase_runs_when_the_model_is_not_ready_everywhere():
    store = StartJobStore()
    job_id, _ = _create_job(store)
    phases = _watch_phases(store, job_id)
    calls = {"autoconfigure": 0}

    async def autoconfigure():
        calls["autoconfigure"] += 1
        if calls["autoconfigure"] == 1:
            return _proposal(staging={"ready": False}, ready_to_activate=False)
        return _proposal(staging={"ready": True})

    _run(store, job_id, autoconfigure=autoconfigure)

    assert phases == [
        "queued",
        "link_setup",
        "autoconfigure",
        "staging",
        "autoconfigure",
        "activating",
        "ready",
    ]
    assert calls["autoconfigure"] == 2
    record = store.get(job_id)
    assert record["staging_job_id"] == "a" * 24
    assert record["phase"] == "ready"


@pytest.mark.parametrize(
    ("state_code", "message", "overrides"),
    [
        (
            "start_link_setup_failed",
            "no cable",
            {"link_setup": None},
        ),
        (
            "start_autoconfigure_failed",
            "planner refused",
            {"autoconfigure": None},
        ),
        (
            "start_staging_failed",
            "copy failed on small",
            {"wait_staging": None},
        ),
        (
            "start_activation_failed",
            "canary failed",
            {"activate": None},
        ),
    ],
)
def test_each_phase_failure_parks_the_job_with_that_phases_incident(
    state_code, message, overrides
):
    store = StartJobStore()
    job_id, _ = _create_job(store)

    async def boom(*_args, **_kwargs):
        raise RuntimeError(message)

    fakes = {key: boom for key in overrides}
    if "wait_staging" in fakes:
        # A staging failure needs the staging phase to run at all.
        async def autoconfigure():
            return _proposal(staging={"ready": False})

        fakes["autoconfigure"] = autoconfigure
        # The failure under test is the copy outcome, not an exception.
        async def wait_staging(staging_job_id):
            return {"ready": False, "status": "failed", "error": message}

        fakes["wait_staging"] = wait_staging

    incidents = _run(store, job_id, **fakes)

    record = store.get(job_id)
    assert record["phase"] == "failed"
    assert record["error"] == message
    assert record["incident_id"] == "incident-1"
    assert incidents.records[0]["severity"] == Severity.ERROR
    assert incidents.records[0]["state_code"] == state_code
    assert incidents.records[0]["message"] == message
    assert incidents.records[0]["job_id"] == job_id


def test_fabric_blocker_fails_the_autoconfigure_phase():
    store = StartJobStore()
    job_id, _ = _create_job(store)

    async def autoconfigure():
        return _proposal(fabric_ready=False, fabric_blocker="cable unplugged")

    incidents = _run(store, job_id, autoconfigure=autoconfigure)

    record = store.get(job_id)
    assert record["phase"] == "failed"
    assert record["error"] == "cable unplugged"
    assert incidents.records[0]["state_code"] == "start_autoconfigure_failed"


def test_a_proposal_that_is_not_ready_fails_with_its_preflight_text():
    store = StartJobStore()
    job_id, _ = _create_job(store)

    async def autoconfigure():
        return _proposal(
            ready_to_activate=False,
            preflight="2 peers cannot import mlx",
        )

    incidents = _run(store, job_id, autoconfigure=autoconfigure)

    record = store.get(job_id)
    assert record["phase"] == "failed"
    assert record["error"] == "2 peers cannot import mlx"
    assert incidents.records[0]["state_code"] == "start_autoconfigure_failed"


def test_concurrent_job_for_the_same_model_is_refused():
    store = StartJobStore()
    job_id, _ = _create_job(store)

    with pytest.raises(StartJobConflictError):
        _create_job(store)

    # A different model may start; the refusal is per model.
    other_id, _ = _create_job(store, model_path="/models/other")
    assert other_id != job_id

    # Once the first job is terminal, the same model may try again.
    store.update(job_id, phase=StartJobPhase.FAILED, error="boom")
    retry_id, superseded = _create_job(store)
    assert retry_id not in {job_id, other_id}
    assert superseded == (job_id,)


def test_supersession_chain_keeps_every_failed_attempt():
    store = StartJobStore()
    first, _ = _create_job(store)
    store.update(first, phase=StartJobPhase.FAILED, error="first")

    second, superseded = _create_job(store)
    assert superseded == (first,)
    assert store.get(first)["superseded_by"] == second
    assert store.get(second)["attempt"] == 2

    store.update(second, phase=StartJobPhase.FAILED, error="second")
    third, superseded = _create_job(store)
    assert superseded == (second,)
    assert store.get(second)["superseded_by"] == third
    assert store.get(third)["attempt"] == 3
    # The chain is history, not replacement: the first record still stands.
    assert store.get(first)["superseded_by"] == second


def test_finished_record_stays_readable_with_no_cleanup_race():
    store = StartJobStore()
    job_id, _ = _create_job(store)
    _run(store, job_id)

    # The task has finished; the record must still answer GETs.
    record = store.get(job_id)
    assert record is not None
    assert record["phase"] == "ready"
    assert record["result"]["ok"] is True
    listed = store.list()
    assert listed[0]["job_id"] == job_id

    # Capacity evicts finished records oldest-first, never a running job.
    running_id, _ = _create_job(store, model_path="/models/running")
    for index in range(40):
        extra, _ = _create_job(store, model_path=f"/models/extra-{index}")
        store.update(extra, phase=StartJobPhase.FAILED, error="x")
    listed = store.list()
    assert len(listed) <= 17
    assert any(job["job_id"] == running_id for job in listed)


async def test_cancellation_records_a_failed_incident():
    store = StartJobStore()
    job_id, _ = _create_job(store)
    incidents = _IncidentLog()

    async def link_setup():
        return None

    async def autoconfigure():
        return _proposal()

    async def start_staging(proposal):
        return "a" * 24

    async def wait_staging(staging_job_id):
        return {"ready": True}

    async def activate(activation):
        await asyncio.Event().wait()

    task = asyncio.get_running_loop().create_task(
        run_start_job(
            store,
            job_id,
            link_setup=link_setup,
            autoconfigure=autoconfigure,
            start_staging=start_staging,
            wait_staging=wait_staging,
            activate=activate,
            record_incident=incidents,
        )
    )
    for _ in range(200):
        if store.get(job_id)["phase"] == "activating":
            break
        await asyncio.sleep(0.01)
    assert store.get(job_id)["phase"] == "activating"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    record = store.get(job_id)
    assert record["phase"] == "failed"
    assert record["error"] == (
        "The server shut down before the start job finished."
    )
    assert incidents.records[-1]["state_code"] == "start_job_cancelled"
    assert incidents.records[-1]["severity"] == Severity.ERROR
