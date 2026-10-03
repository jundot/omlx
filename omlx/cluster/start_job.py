# SPDX-License-Identifier: Apache-2.0
"""Start Cluster as a persisted, server-owned job (design B.3, plan item B2).

Clicking Start used to drive a purely client-side ladder (link setup →
autoconfigure → staging → activation) whose state lived in the browser tab.
A reload lost all of it and left a dead button. Here the ladder runs on the
server as one job record the dashboard merely renders: ``POST /start-jobs``
creates it, ``GET /start-jobs/{id}`` polls it, and a reloaded page re-attaches
to whatever job is still running.

**Persistence decision (explicit):** the job store is in-memory — a dict
behind a lock, exactly like the staging-job registry in ``routes.py``. That
satisfies the design's stated requirement ("if the browser reloads, the job
is still there"): the job must survive the *browser*, not the server. Durable
history across server restarts is provided elsewhere — every phase failure
and every READY transition records an Incident in the persisted B1 incident
store, and the deployment registry is the durable record of a successful
start. Adding a second persisted job file here would only duplicate those.

Library purity: this module is the job runner the shared conventions single
out — it may convert outcomes into incidents, but it does so through an
injected ``record_incident`` callable so the module itself stays importable
(and testable) without a configured store or the routes layer.
"""

from __future__ import annotations

import asyncio
import copy
import secrets
import threading
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from .incidents import Severity

_MAX_START_JOBS = 16


class StartJobPhase(StrEnum):
    """The ladder ``startCluster()`` ran client-side, now server-owned."""

    QUEUED = "queued"
    LINK_SETUP = "link_setup"
    AUTOCONFIGURE = "autoconfigure"
    STAGING = "staging"
    ACTIVATING = "activating"
    READY = "ready"
    FAILED = "failed"


TERMINAL_PHASES = frozenset({StartJobPhase.READY, StartJobPhase.FAILED})


class StartJobConflictError(RuntimeError):
    """A non-terminal start job already exists for this model."""


class StartJobStore:
    """In-memory registry of Start Cluster jobs (dict + lock).

    Mirrors the ``_STAGING_JOBS`` pattern in ``routes.py``: snapshots leave as
    deep copies so no caller can mutate shared state, and eviction only ever
    removes finished records — a running job is never dropped for capacity.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(
        self,
        *,
        model_path: str,
        hosts: list[str],
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Create one job; supersede any earlier FAILED attempts for the model.

        Returns ``(snapshot, superseded_job_ids)``. Raises
        :class:`StartJobConflictError` when a non-terminal job already exists
        for the same model — attempt #4 replaces #3, it never races it.
        """

        with self._lock:
            for job in self._jobs.values():
                if (
                    job["model_path"] == model_path
                    and job["phase"] not in TERMINAL_PHASES
                ):
                    raise StartJobConflictError(
                        "A Start Cluster job "
                        f"({job['job_id']}) is already running for this model."
                    )
            superseded = [
                job
                for job in self._jobs.values()
                if job["model_path"] == model_path
                and job["phase"] == StartJobPhase.FAILED
                and not job.get("superseded_by")
            ]
            attempt = 1 + max(
                (
                    int(job.get("attempt") or 1)
                    for job in self._jobs.values()
                    if job["model_path"] == model_path
                ),
                default=0,
            )
            now = time.time()
            job = {
                "job_id": secrets.token_hex(12),
                "created_at": now,
                "updated_at": now,
                "phase": str(StartJobPhase.QUEUED),
                "model_path": model_path,
                "hosts": list(hosts),
                "staging_job_id": None,
                "attempt": attempt,
                "superseded_by": None,
                "incident_id": None,
                "error": "",
                "result": None,
            }
            for old in superseded:
                old["superseded_by"] = job["job_id"]
                old["updated_at"] = now
            if len(self._jobs) >= _MAX_START_JOBS:
                finished = [
                    key
                    for key, value in self._jobs.items()
                    if value["phase"] in TERMINAL_PHASES
                ]
                if finished:
                    self._jobs.pop(finished[0], None)
            self._jobs[job["job_id"]] = job
            return copy.deepcopy(job), tuple(item["job_id"] for item in superseded)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job is not None else None

    def list(self) -> list[dict[str, Any]]:
        """Recent records, newest first (the store is already capped)."""

        with self._lock:
            return [copy.deepcopy(job) for job in reversed(self._jobs.values())]

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            if "phase" in fields:
                fields["phase"] = str(fields["phase"])
            job.update(fields)
            job["updated_at"] = time.time()
            return copy.deepcopy(job)

    def active_job(self, model_path: str | None = None) -> dict[str, Any] | None:
        """The newest non-terminal job — for one model, or for any."""

        with self._lock:
            for job in reversed(self._jobs.values()):
                if job["phase"] in TERMINAL_PHASES:
                    continue
                if model_path is None or job["model_path"] == model_path:
                    return copy.deepcopy(job)
            return None


# One store per server process. In-memory by design (see module docstring),
# so no base_path configuration is needed; the reset hook exists for tests.
_store_lock = threading.Lock()
_store: StartJobStore | None = None


def get_start_job_store() -> StartJobStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = StartJobStore()
        return _store


def reset_start_job_store() -> StartJobStore:
    global _store
    with _store_lock:
        _store = StartJobStore()
        return _store


async def run_start_job(
    store: StartJobStore,
    job_id: str,
    *,
    link_setup: Callable[[], Awaitable[None]],
    autoconfigure: Callable[[], Awaitable[dict[str, Any]]],
    start_staging: Callable[[dict[str, Any]], Awaitable[str]],
    wait_staging: Callable[[str], Awaitable[dict[str, Any]]],
    activate: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    record_incident: Callable[..., str | None],
    describe_error: Callable[[BaseException], str] = str,
) -> None:
    """Run exactly the ladder ``startCluster()`` ran client-side.

    Must execute as a task on the server's event loop — ``activate`` is async
    and touches the engine pool. Each phase transition updates the store; each
    failure records an Incident (source ``"coordinator"``, ``job_id`` set) and
    parks the job in FAILED. ``record_incident(severity, state_code, message,
    job_id=..., deployment_id=...)`` returns the incident id or ``None``.
    """

    def _incident(
        severity: Severity,
        state_code: str,
        message: str,
        *,
        deployment_id: str | None = None,
    ) -> str | None:
        try:
            return record_incident(
                severity,
                state_code,
                message,
                job_id=job_id,
                deployment_id=deployment_id,
            )
        except Exception:  # noqa: BLE001 - narration never outranks the job
            return None

    def _fail(state_code: str, message: str) -> None:
        incident_id = _incident(Severity.ERROR, state_code, message)
        store.update(
            job_id,
            phase=StartJobPhase.FAILED,
            error=message,
            incident_id=incident_id,
        )

    def _not_ready_message(proposal: dict[str, Any]) -> str:
        staging = proposal.get("staging")
        if isinstance(staging, dict):
            if staging.get("error"):
                return str(staging["error"])
            if not staging.get("ready"):
                return "The model files could not be prepared on every worker."
        return (
            str(proposal.get("preflight") or "")
            or "Resolve the setup checks below, then try Start Cluster again."
        )

    try:
        store.update(job_id, phase=StartJobPhase.LINK_SETUP)
        try:
            await link_setup()
        except Exception as exc:  # noqa: BLE001 - every outcome becomes state
            _fail("start_link_setup_failed", describe_error(exc))
            return

        store.update(job_id, phase=StartJobPhase.AUTOCONFIGURE)
        try:
            proposal = await autoconfigure()
        except Exception as exc:  # noqa: BLE001
            _fail("start_autoconfigure_failed", describe_error(exc))
            return
        if proposal.get("fabric_ready") is False:
            _fail(
                "start_autoconfigure_failed",
                str(
                    proposal.get("fabric_blocker")
                    or "The workers do not have a verified cluster route yet."
                ),
            )
            return

        staging = proposal.get("staging")
        if isinstance(staging, dict) and not staging.get("ready"):
            store.update(job_id, phase=StartJobPhase.STAGING)
            try:
                staging_job_id = await start_staging(proposal)
                store.update(job_id, staging_job_id=staging_job_id)
                outcome = await wait_staging(staging_job_id)
            except Exception as exc:  # noqa: BLE001
                _fail("start_staging_failed", describe_error(exc))
                return
            if not outcome.get("ready"):
                _fail(
                    "start_staging_failed",
                    str(
                        outcome.get("error")
                        or "The model files could not be prepared on every worker."
                    ),
                )
                return
            # Staged files change what the planner can propose; the client
            # ladder re-ran autoconfigure here for the same reason.
            store.update(job_id, phase=StartJobPhase.AUTOCONFIGURE)
            try:
                proposal = await autoconfigure()
            except Exception as exc:  # noqa: BLE001
                _fail("start_autoconfigure_failed", describe_error(exc))
                return
            if proposal.get("fabric_ready") is False:
                _fail(
                    "start_autoconfigure_failed",
                    str(
                        proposal.get("fabric_blocker")
                        or "The workers do not have a verified cluster route yet."
                    ),
                )
                return

        if not proposal.get("ready_to_activate"):
            _fail("start_autoconfigure_failed", _not_ready_message(proposal))
            return
        activation = proposal.get("activation")
        if not isinstance(activation, dict):
            _fail(
                "start_autoconfigure_failed",
                "Automatic setup did not return a safe activation request.",
            )
            return

        store.update(job_id, phase=StartJobPhase.ACTIVATING)
        try:
            result = await activate(activation)
        except Exception as exc:  # noqa: BLE001
            _fail("start_activation_failed", describe_error(exc))
            return

        record = store.get(job_id) or {}
        deployment_id = None
        if isinstance(result, dict):
            deployment_id = (result.get("deployment") or {}).get("deployment_id")
        incident_id = _incident(
            Severity.INFO,
            "start_job_ready",
            f"Cluster start finished: {record.get('model_path', '')} is ready.",
            deployment_id=deployment_id,
        )
        store.update(
            job_id,
            phase=StartJobPhase.READY,
            result=result,
            incident_id=incident_id,
            error="",
        )
    except asyncio.CancelledError:
        # Server shutdown mid-job. The registry rollback inside the activation
        # path already restored the previous deployment; here the record and
        # the durable incident say what happened instead of a dead button.
        message = "The server shut down before the start job finished."
        try:
            incident_id = _incident(Severity.ERROR, "start_job_cancelled", message)
            store.update(
                job_id,
                phase=StartJobPhase.FAILED,
                error=message,
                incident_id=incident_id,
            )
        except Exception:  # noqa: BLE001 - a narration failure at teardown
            # must never leave the record stuck non-terminal *and* mask the
            # cancellation; nothing here may outrank the re-raise below.
            pass
        raise
