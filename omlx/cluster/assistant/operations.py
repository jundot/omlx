# SPDX-License-Identifier: Apache-2.0
"""Persisted apply/start operations for the Cluster Assistant.

``apply`` and ``start`` are jobs, not calls. Staging copies tens of gigabytes
and the coordinator loads weights for minutes; a request that waited for either
would time out long before the answer exists. This module is the record those
jobs write, and the only place that decides whether an operation may still move
forward.

The store mirrors ``state.MeasurementStore`` and ``cluster/registry.py``: one
JSON file under ``<base_path>/cluster/``, one re-entrant lock, atomic
replacement, mode 0600, and a load that fails closed. A corrupt file must never
stop the local oMLX server; it only means no operation is known, and the
routes answer as if the operation never existed.

Four rules are deliberate.

1. **States are forward-only.** ``queued -> applying -> applied`` is one
   apply's life, ``queued -> starting -> loading -> ready`` is one start's, and
   ``failed`` may end either. A late event — a staging poll that arrives after
   the caller's machine woke up, a second worker on an old ID — may never move a
   record backwards: the transition is dropped, not applied. ``applied`` and
   ``ready`` are terminal, so "ready" cannot be undone by a stale failure.
2. **One apply or start at a time.** A second request is refused with the
   running operation's ID and the one action ``status`` instead of joining it,
   for the same reason a second measurement is refused: joining would hand the
   caller progress that belongs to a different plan.
3. **A job cannot outlive its process.** Loaded operations that were still
   running are recovered as ``failed`` (``operation_interrupted``) rather than
   reported as running forever.
4. **No secrets.** No SSH targets, addresses, API keys, argv or model file
   lists are stored. Node IDs, the plan signature, the placement signature and
   one endpoint are identifiers, not credentials; the targets are resolved from
   the device registry at request time and never persisted.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPERATION_SCHEMA_VERSION = 1
OPERATION_FILE_NAME = "assistant-operations.json"

# Newest wins; older records are dropped on the next start. Bounded for the
# same reason the measurement file is: this is a scratch record, not a history,
# and it is rewritten whole on every transition.
MAX_OPERATION_RECORDS = 8

# A running *mirror* older than this stops counting as running in ``active()``.
# Generous on purpose: staging copies tens of gigabytes and a weight load takes
# minutes, so the bound sits well above any normal apply/start. It exists only
# so an origin that crashed mid-operation cannot wedge the coordinator's lock
# forever (the apply/start design notes); a live origin mirrors the terminal state long
# before this expires.
MIRROR_TTL_SECONDS = 2 * 3600.0

KIND_APPLY = "apply"
KIND_START = "start"
_OPERATION_KINDS = (KIND_APPLY, KIND_START)

STATE_QUEUED = "queued"
STATE_APPLYING = "applying"
STATE_APPLIED = "applied"
STATE_STARTING = "starting"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_FAILED = "failed"

# The only forward order there is. ``failed`` is deliberately absent: it ends an
# operation from any running state and no state follows it.
STATE_ORDER = (
    STATE_QUEUED,
    STATE_APPLYING,
    STATE_APPLIED,
    STATE_STARTING,
    STATE_LOADING,
    STATE_READY,
)
_STATE_RANK = {state: rank for rank, state in enumerate(STATE_ORDER)}

# Terminal states an operation never leaves, and the running ones that make a
# second apply/start a conflict.
TERMINAL_STATES = frozenset({STATE_APPLIED, STATE_READY, STATE_FAILED})
RUNNING_STATES = frozenset(
    {STATE_QUEUED, STATE_APPLYING, STATE_STARTING, STATE_LOADING}
)

# The next step for each failure. One action per failure (the project design notes,
# §12.4); the route never invents a second one.
ACTION_APPLY = "apply"
ACTION_START = "start"
ACTION_PLAN = "plan"
ACTION_MEASURE = "measure"
ACTION_STATUS = "status"
ACTION_ADD_MAC = "add_mac"
ACTION_PAIR_MAC = "pair_mac"
ACTION_COPY_MODEL = "copy_model"
#: Free space on a Mac before its stage can be copied there.
ACTION_FREE_DISK_SPACE = "free_disk_space"
ACTION_CONFIGURE_ENDPOINT = "configure_endpoint"
ACTION_RETRY_COORDINATOR = "retry_coordinator"
ACTION_DIAGNOSTICS = "diagnostics"
#: The one next step when ``iogpu.wired_limit_mb`` — not RAM and not live
#: pressure — is what caps a Mac's admitted ceiling. The app applies the
#: recommended limit from Performance → "Apply with Admin…"; the web page can
#: only show the command, so this remains an app-side click.
ACTION_RAISE_WIRED_LIMIT = "raise_wired_limit"

OPERATION_IN_PROGRESS_CODE = "operation_in_progress"
OPERATION_NOT_FOUND_CODE = "operation_not_found"

_ID_LOCK = threading.Lock()
_ID_SEQUENCE = 0


class OperationInProgressError(RuntimeError):
    """Another apply/start is running; the caller polls that one instead."""

    def __init__(self, operation_id: str, kind: str) -> None:
        super().__init__(
            f"{kind} operation {operation_id} is still running; poll it instead "
            "of starting a second one"
        )
        self.operation_id = operation_id
        self.kind = kind


@dataclass(frozen=True)
class OperationProgress:
    """Where one operation is, in terms both a person and a client can read.

    ``phase`` is a short machine-readable label (``staging``, ``starting``,
    ``loading``, ``ready``); ``message`` is the sentence the UI shows. The node
    counts are staging progress and are zero when they do not apply.
    """

    phase: str = STATE_QUEUED
    message: str = ""
    nodes_ready: int = 0
    nodes_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "message": self.message,
            "nodes_ready": self.nodes_ready,
            "nodes_total": self.nodes_total,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OperationProgress:
        if not isinstance(payload, Mapping):
            raise ValueError("operation progress is malformed")
        return cls(
            phase=str(payload.get("phase") or STATE_QUEUED),
            message=str(payload.get("message") or ""),
            nodes_ready=_optional_int(payload.get("nodes_ready")) or 0,
            nodes_total=_optional_int(payload.get("nodes_total")) or 0,
        )


@dataclass(frozen=True)
class OperationEndpoint:
    """The coordinator's endpoint, and whether a client may dial it yet.

    ``ready`` is false until health and one smoke completion have succeeded.
    The host and port come from the stored measurement — never invented, never
    zero — so an unresolved endpoint is ``None`` on the record, not a URL
    nobody answers.
    """

    node: str
    host: str
    port: int
    ready: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("endpoint port must be between 1 and 65535")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "host": self.host,
            "port": self.port,
            "url": self.url,
            "ready": self.ready,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OperationEndpoint:
        if not isinstance(payload, Mapping):
            raise ValueError("operation endpoint is malformed")
        return cls(
            node=str(payload.get("node") or ""),
            host=str(payload.get("host") or ""),
            port=_optional_int(payload.get("port")) or 0,
            ready=bool(payload.get("ready", False)),
        )


@dataclass(frozen=True)
class OperationObserved:
    """One real end-to-end measurement of the started cluster.

    This is the only place a rate may be reported, and only by a measurement
    the coordinator actually served: a non-streaming completion whose own
    timing fields oMLX computed. ``None`` on the record means "not measured",
    never zero, and ``predicted`` on the plan is never rewritten from it.

    ``measured_at`` is the coordinator's ISO timestamp for the completion.
    """

    prefill_tok_per_s: float | None = None
    decode_tok_per_s: float | None = None
    ttft_ms: float | None = None
    measured_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefill_tok_per_s": self.prefill_tok_per_s,
            "decode_tok_per_s": self.decode_tok_per_s,
            "ttft_ms": self.ttft_ms,
            "measured_at": self.measured_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OperationObserved:
        if not isinstance(payload, Mapping):
            raise ValueError("operation observed block is malformed")
        return cls(
            prefill_tok_per_s=_optional_float(payload.get("prefill_tok_per_s")),
            decode_tok_per_s=_optional_float(payload.get("decode_tok_per_s")),
            ttft_ms=_optional_float(payload.get("ttft_ms")),
            measured_at=str(payload.get("measured_at") or ""),
        )


@dataclass(frozen=True)
class OperationWarning:
    """Something the caller should know before trusting this operation."""

    code: str
    message: str
    node_id: str = ""
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "node_id": self.node_id,
            "message": self.message,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OperationWarning:
        if not isinstance(payload, Mapping):
            raise ValueError("operation warning is malformed")
        return cls(
            code=str(payload.get("code") or ""),
            message=str(payload.get("message") or ""),
            node_id=str(payload.get("node_id") or ""),
            action=str(payload.get("action") or ""),
        )


@dataclass(frozen=True)
class OperationFailure:
    """Why an operation ended, and the one thing to do about it."""

    code: str
    message: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "action": self.action}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OperationFailure:
        if not isinstance(payload, Mapping):
            raise ValueError("operation failure is malformed")
        return cls(
            code=str(payload.get("code") or ""),
            message=str(payload.get("message") or ""),
            action=str(payload.get("action") or ACTION_DIAGNOSTICS),
        )


@dataclass(frozen=True)
class OperationRecord:
    """One apply or start, from ``queued`` to ``ready`` or ``failed``.

    ``plan_signature`` is the election's plan hash the caller approved;
    ``placement_signature`` is the coordinator's own signature for the
    deployment that staging prepared. Both are recorded because they answer
    different questions: the first is "which plan did the user approve", the
    second is "which rank map did the coordinator actually accept".
    """

    operation_id: str
    kind: str
    state: str
    created_at: str
    updated_at: str
    measurement_id: str
    plan_signature: str = ""
    placement_signature: str = ""
    coordinator_node: str = ""
    #: Whether the elected coordinator was this Mac when the operation ran.
    #: Recorded so stopping never has to guess (or re-elect from a stale
    #: measurement) which seam to use.
    coordinator_local: bool = False
    rank_order: tuple[str, ...] = ()
    context_tokens: int = 0
    #: The context the user asked for, before the solver reduced it.
    target_context_tokens: int = 0
    #: The checkpoint this operation runs. Not a secret; it lets a reloaded
    #: card offer Start again without re-selecting a model.
    model_path: str = ""
    rank0_extra_bytes: int = 0
    microbatch_size: int = 1
    mtp: bool = False
    execution_profile: str = "balanced"
    #: The user's explicit consent to the 2-Mac fallback. It is part of the
    #: plan signature, so start must carry it when it re-proves the plan.
    two_mac_fallback: bool = False
    #: The rank-0 pin the plan was approved with (and with it the endpoint
    #: host). Stored so ``start`` re-elects the same plan without trusting the
    #: request body; empty means "automatic".
    preferred_coordinator: str = ""
    #: The resource policy the approved plan used (engram/ple, MoE offload,
    #: tier, context, admission). Persisted so ``start`` re-elects the same plan
    #: after a reload without trusting the request body.
    policy: Mapping[str, Any] = field(default_factory=dict)
    progress: OperationProgress = field(default_factory=OperationProgress)
    endpoint: OperationEndpoint | None = None
    #: The coordinator's deployment ID, known after a start reports ready.
    deployment_id: str = ""
    #: True after a successful stop. ``state`` stays ``ready``: the operation
    #: did become ready, and no state machine value reports a stop.
    stopped: bool = False
    #: True when another origin mirrored this record onto this Mac through the
    #: admin seam (the cluster-wide lock, the apply/start design notes). A mirror is a lock
    #: marker, not this Mac's own assistant session: ``latest_session`` and
    #: ``applied`` skip it, while ``active`` reports a running one.
    mirrored: bool = False
    #: Measured throughput, filled only after ``ready`` by a real completion.
    observed: OperationObserved | None = None
    warnings: tuple[OperationWarning, ...] = ()
    error: OperationFailure | None = None

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def running(self) -> bool:
        return self.state in RUNNING_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "kind": self.kind,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "measurement_id": self.measurement_id,
            "plan_signature": self.plan_signature,
            "placement_signature": self.placement_signature,
            "coordinator_node": self.coordinator_node,
            "coordinator_local": self.coordinator_local,
            "rank_order": list(self.rank_order),
            "context_tokens": self.context_tokens,
            "target_context_tokens": self.target_context_tokens,
            "model_path": self.model_path,
            "rank0_extra_bytes": self.rank0_extra_bytes,
            "microbatch_size": self.microbatch_size,
            "mtp": self.mtp,
            "execution_profile": self.execution_profile,
            "two_mac_fallback": self.two_mac_fallback,
            "preferred_coordinator": self.preferred_coordinator,
            "policy": dict(self.policy),
            "progress": self.progress.to_dict(),
            "endpoint": None if self.endpoint is None else self.endpoint.to_dict(),
            "deployment_id": self.deployment_id,
            "stopped": self.stopped,
            "mirrored": self.mirrored,
            "observed": None if self.observed is None else self.observed.to_dict(),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "error": None if self.error is None else self.error.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> OperationRecord:
        if not isinstance(payload, Mapping):
            raise ValueError("operation record is malformed")
        operation_id = str(payload.get("operation_id") or "")
        if not operation_id:
            raise ValueError("operation record is missing an ID")
        kind = str(payload.get("kind") or "")
        if kind not in _OPERATION_KINDS:
            raise ValueError(f"operation {operation_id} has an unknown kind")
        state = str(payload.get("state") or "")
        if state not in _STATE_RANK and state != STATE_FAILED:
            raise ValueError(f"operation {operation_id} has an unknown state")
        raw_warnings = payload.get("warnings") or []
        if not isinstance(raw_warnings, list):
            raise ValueError(f"operation {operation_id} has malformed warnings")
        endpoint = payload.get("endpoint")
        error = payload.get("error")
        observed = payload.get("observed")
        return cls(
            operation_id=operation_id,
            kind=kind,
            state=state,
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            measurement_id=str(payload.get("measurement_id") or ""),
            plan_signature=str(payload.get("plan_signature") or ""),
            placement_signature=str(payload.get("placement_signature") or ""),
            coordinator_node=str(payload.get("coordinator_node") or ""),
            coordinator_local=bool(payload.get("coordinator_local", False)),
            rank_order=tuple(str(item) for item in (payload.get("rank_order") or ())),
            context_tokens=_optional_int(payload.get("context_tokens")) or 0,
            target_context_tokens=_optional_int(
                payload.get("target_context_tokens")
            )
            or 0,
            model_path=str(payload.get("model_path") or ""),
            rank0_extra_bytes=_optional_int(payload.get("rank0_extra_bytes")) or 0,
            microbatch_size=_optional_int(payload.get("microbatch_size")) or 1,
            mtp=bool(payload.get("mtp", False)),
            execution_profile=str(payload.get("execution_profile") or "balanced"),
            two_mac_fallback=bool(payload.get("two_mac_fallback", False)),
            preferred_coordinator=str(payload.get("preferred_coordinator") or ""),
            policy=_policy_from_dict(payload.get("policy")),
            progress=OperationProgress.from_dict(payload.get("progress") or {}),
            endpoint=(
                None if endpoint is None else OperationEndpoint.from_dict(endpoint)
            ),
            deployment_id=str(payload.get("deployment_id") or ""),
            stopped=bool(payload.get("stopped", False)),
            mirrored=bool(payload.get("mirrored", False)),
            observed=(
                None if observed is None else OperationObserved.from_dict(observed)
            ),
            warnings=tuple(
                OperationWarning.from_dict(item)
                for item in raw_warnings
                if isinstance(item, Mapping)
            ),
            error=None if error is None else OperationFailure.from_dict(error),
        )


def new_operation_id(
    kind: str, *, now: datetime | None = None, existing: Sequence[str] = ()
) -> str:
    """A stable, opaque, never-reused ID: ``op-<UTC>-<kind>-<8 hex>``.

    The counter keeps IDs generated inside one second ordered, so "the newest
    operation" is the last one by ID as well as by time; the random half keeps
    them opaque.
    """

    if kind not in _OPERATION_KINDS:
        raise ValueError(f"unknown operation kind: {kind!r}")
    global _ID_SEQUENCE
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    known = set(existing)
    for _ in range(64):
        with _ID_LOCK:
            _ID_SEQUENCE += 1
            sequence = _ID_SEQUENCE
        candidate = f"op-{stamp}-{kind}-{sequence & 0xFFFF:04x}{secrets.token_hex(2)}"
        if candidate not in known:
            return candidate
    raise RuntimeError("could not generate a unique operation ID")


class OperationStore:
    """Thread-safe, atomic, bounded store of apply/start operations."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / OPERATION_FILE_NAME
        self._lock = threading.RLock()
        self._records: dict[str, OperationRecord] = {}
        self.load_error: str | None = None
        #: Running records that were recovered as failed on load.
        self.recovered = 0
        try:
            self._load()
        except (KeyError, TypeError, ValueError) as exc:
            # A corrupt scratch file must never prevent the local oMLX server
            # and GUI from starting. Fail closed: no operation is known.
            self._records = {}
            self.load_error = str(exc)

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._records = {}
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"could not read assistant operations: {exc}") from exc
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != OPERATION_SCHEMA_VERSION
            ):
                raise ValueError("unsupported assistant operation schema")
            raw_records = payload.get("operations")
            if not isinstance(raw_records, list):
                raise ValueError("assistant operation file is malformed")
            records = [OperationRecord.from_dict(item) for item in raw_records]
            if len({record.operation_id for record in records}) != len(records):
                raise ValueError("assistant operation file has duplicate IDs")
            # A job cannot survive the process that ran it. Reporting it as
            # running would leave the UI polling a ghost forever, and would
            # also keep refusing new applies with "already running".
            recovered = [
                _replaced(
                    record,
                    state=STATE_FAILED,
                    error=OperationFailure(
                        code="operation_interrupted",
                        message=(
                            "the operation was interrupted by a server restart "
                            "and never finished"
                        ),
                        action=(
                            ACTION_START if record.kind == KIND_START else ACTION_APPLY
                        ),
                    ),
                )
                if record.running
                else record
                for record in records
            ]
            self.recovered = sum(
                1
                for before, after in zip(records, recovered)
                if before.state != after.state
            )
            self._records = {record.operation_id: record for record in recovered}
            self._prune()
            if self.recovered:
                with suppress(OSError):
                    self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": OPERATION_SCHEMA_VERSION,
            "operations": [
                self._records[operation_id].to_dict()
                for operation_id in sorted(self._records)
            ],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".assistant-operations.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def _prune(self) -> int:
        """Drop the oldest records beyond the cap. Returns how many went."""

        with self._lock:
            surplus = len(self._records) - MAX_OPERATION_RECORDS
            if surplus <= 0:
                return 0
            for operation_id in sorted(self._records)[:surplus]:
                del self._records[operation_id]
            return surplus

    # -- job lifecycle ------------------------------------------------------

    def start(
        self,
        *,
        kind: str,
        measurement_id: str,
        plan_signature: str = "",
        coordinator_node: str = "",
        coordinator_local: bool = False,
        rank_order: Sequence[str] = (),
        context_tokens: int = 0,
        target_context_tokens: int = 0,
        model_path: str = "",
        rank0_extra_bytes: int = 0,
        microbatch_size: int = 1,
        mtp: bool = False,
        execution_profile: str = "balanced",
        two_mac_fallback: bool = False,
        preferred_coordinator: str = "",
        policy: Mapping[str, Any] | None = None,
        endpoint: OperationEndpoint | None = None,
        now: datetime | None = None,
    ) -> OperationRecord:
        """Create a ``queued`` record and persist it before any work runs.

        One apply/start at a time: a second one raises ``OperationInProgressError``
        with the running ID rather than joining it.
        """

        moment = now or datetime.now(UTC)
        stamp = moment.isoformat()
        with self._lock:
            running = self.active()
            if running is not None:
                raise OperationInProgressError(running.operation_id, running.kind)
            operation_id = new_operation_id(
                kind, now=moment, existing=tuple(self._records)
            )
            previous = dict(self._records)
            record = OperationRecord(
                operation_id=operation_id,
                kind=kind,
                state=STATE_QUEUED,
                created_at=stamp,
                updated_at=stamp,
                measurement_id=measurement_id,
                plan_signature=plan_signature,
                coordinator_node=coordinator_node,
                coordinator_local=coordinator_local,
                rank_order=tuple(rank_order),
                context_tokens=context_tokens,
                target_context_tokens=target_context_tokens,
                model_path=model_path,
                rank0_extra_bytes=rank0_extra_bytes,
                microbatch_size=microbatch_size,
                mtp=mtp,
                execution_profile=execution_profile,
                two_mac_fallback=two_mac_fallback,
                preferred_coordinator=preferred_coordinator,
                policy=dict(policy or {}),
                progress=OperationProgress(
                    phase=STATE_QUEUED,
                    message=f"{kind} queued",
                ),
                endpoint=endpoint,
            )
            self._records[operation_id] = record
            self._prune()
            try:
                self._save()
            except Exception:
                self._records = previous
                raise
            self.load_error = None
            return record

    def advance(
        self,
        operation_id: str,
        state: str,
        *,
        progress: OperationProgress | None = None,
        endpoint: OperationEndpoint | None = None,
        warnings: Sequence[OperationWarning] | None = None,
        placement_signature: str | None = None,
        deployment_id: str | None = None,
        now: datetime | None = None,
    ) -> OperationRecord:
        """Move one operation forward, or leave it exactly as it is.

        A transition that is not strictly forward — or that would leave a
        terminal state — is dropped rather than applied, so a late event can
        never undo "ready" or resurrect "failed".
        """

        if state == STATE_FAILED or state not in _STATE_RANK:
            # ``failed`` carries an error and has exactly one producer: fail().
            raise ValueError(f"unknown operation state: {state!r}")
        with self._lock:
            record = self._require(operation_id)
            if not _may_advance(record.state, state):
                return record
            updated = _replaced(
                record,
                state=state,
                progress=progress
                or OperationProgress(
                    phase=state,
                    message=record.progress.message,
                    nodes_ready=record.progress.nodes_ready,
                    nodes_total=record.progress.nodes_total,
                ),
                endpoint=endpoint,
                warnings=warnings,
                placement_signature=placement_signature,
                deployment_id=deployment_id,
            )
            self._commit(updated)
            return updated

    def observe(
        self,
        operation_id: str,
        *,
        observed: OperationObserved | None = None,
        warning: OperationWarning | None = None,
        now: datetime | None = None,
    ) -> OperationRecord:
        """Attach a post-``ready`` measurement, or the reason none arrived.

        ``ready`` is terminal, so this is deliberately *not* a state
        transition: it only adds the measured ``observed`` block and/or one
        warning to the record the poll already returns. A record that is not
        ready is left untouched — a probe may never move an operation.
        """

        if observed is None and warning is None:
            with self._lock:
                return self._require(operation_id)
        with self._lock:
            record = self._require(operation_id)
            if record.state != STATE_READY:
                return record
            warnings = (
                record.warnings
                if warning is None
                else (*record.warnings, warning)
            )
            updated = replace(
                record,
                observed=record.observed if observed is None else observed,
                warnings=warnings,
                updated_at=(now or datetime.now(UTC)).isoformat(),
            )
            self._commit(updated)
            return updated

    def mark_stopped(
        self,
        operation_id: str,
        *,
        warning: OperationWarning | None = None,
        now: datetime | None = None,
    ) -> OperationRecord:
        """Record that the ready deployment was stopped.

        ``ready`` stays the operation's state: it did become ready, and the
        frozen state machine has no "stopped" value. ``stopped`` is the boolean
        the Screen-6 surface reads, so a reload cannot present a stopped
        deployment as running.
        """

        with self._lock:
            record = self._require(operation_id)
            if record.state != STATE_READY or record.stopped:
                return record
            warnings = (
                record.warnings if warning is None else (*record.warnings, warning)
            )
            updated = replace(
                record,
                stopped=True,
                warnings=warnings,
                updated_at=(now or datetime.now(UTC)).isoformat(),
            )
            self._commit(updated)
            return updated

    def fail(
        self,
        operation_id: str,
        *,
        code: str,
        message: str,
        action: str,
        now: datetime | None = None,
    ) -> OperationRecord:
        """End a running operation in ``failed`` with exactly one next step."""

        with self._lock:
            record = self._require(operation_id)
            if record.finished:
                return record
            updated = _replaced(
                record,
                state=STATE_FAILED,
                progress=OperationProgress(
                    phase=STATE_FAILED,
                    message=record.progress.message,
                    nodes_ready=record.progress.nodes_ready,
                    nodes_total=record.progress.nodes_total,
                ),
                error=OperationFailure(code=code, message=message, action=action),
            )
            self._commit(updated)
            return updated

    def mirror(
        self,
        operation_id: str,
        kind: str,
        coordinator_node: str,
        state: str,
        *,
        model_path: str = "",
        context_tokens: int = 0,
        now: datetime | None = None,
    ) -> OperationRecord:
        """Register or release another origin's operation on this coordinator.

        The cluster-wide lock (the apply/start design notes): a remote apply/start drives
        this Mac's raw cluster routes, so the origin's record lives only in
        the origin's store. The origin mirrors its operation here through the
        admin seam — a running mirror takes the lock (``active`` reports it
        and ``start`` refuses a second one), a terminal mirror releases it.
        The caller supplies the operation ID so both stores name the same
        operation.

        Mirrors are lock markers, not this Mac's own sessions:
        ``latest_session`` and ``applied`` skip them. The forward-only rule
        holds unchanged — a late mirror never resurrects or rewrites a
        terminal record — and a running mirror older than
        ``MIRROR_TTL_SECONDS`` stops counting as running, so a crashed origin
        cannot wedge the lock forever.
        """

        if kind not in _OPERATION_KINDS:
            raise ValueError(f"unknown operation kind: {kind!r}")
        if state not in _STATE_RANK and state != STATE_FAILED:
            raise ValueError(f"unknown operation state: {state!r}")
        moment = now or datetime.now(UTC)
        stamp = moment.isoformat()
        with self._lock:
            existing = self._records.get(operation_id)
            if existing is not None and not existing.mirrored:
                # This Mac's own record is not a lock marker; a remote origin
                # may never rewrite it. IDs are opaque and collision-free, so
                # this is a guard, not a branch callers reach.
                raise ValueError(f"operation {operation_id} is not a mirror")
            if existing is not None and existing.finished:
                # Forward-only: a late mirror cannot resurrect or rewrite a
                # terminal record (a stale failure may not undo "ready").
                return existing
            if state in RUNNING_STATES:
                running = self.active(now=moment)
                if running is not None and running.operation_id != operation_id:
                    raise OperationInProgressError(
                        running.operation_id, running.kind
                    )
            if existing is None:
                record = OperationRecord(
                    operation_id=operation_id,
                    kind=kind,
                    state=state,
                    created_at=stamp,
                    updated_at=stamp,
                    measurement_id="",
                    coordinator_node=coordinator_node,
                    model_path=model_path,
                    context_tokens=context_tokens,
                    mirrored=True,
                    progress=OperationProgress(
                        phase=state,
                        message=f"{kind} operation mirrored from another origin",
                    ),
                    error=(
                        OperationFailure(
                            code="operation_failed",
                            message=(
                                "the origin that ran this operation reported "
                                "it failed"
                            ),
                            action=ACTION_STATUS,
                        )
                        if state == STATE_FAILED
                        else None
                    ),
                )
            else:
                if state != STATE_FAILED and not _may_advance(existing.state, state):
                    return existing
                record = _replaced(
                    existing,
                    state=state,
                    progress=OperationProgress(
                        phase=state, message=existing.progress.message
                    ),
                    error=(
                        OperationFailure(
                            code="operation_failed",
                            message=(
                                "the origin that ran this operation reported "
                                "it failed"
                            ),
                            action=ACTION_STATUS,
                        )
                        if state == STATE_FAILED and existing.error is None
                        else existing.error
                    ),
                    now=moment,
                )
            self._commit(record)
            self.load_error = None
            return record

    def _commit(self, record: OperationRecord) -> None:
        previous = dict(self._records)
        self._records[record.operation_id] = record
        try:
            self._save()
        except Exception:
            # A terminal state (applied/ready/failed) is the truth about the
            # operation even when the file could not be written; rolling it
            # back left the record running forever and wedged every later
            # apply/start (review B6). Only an unpersisted running transition
            # is undone. The save error still reaches the caller.
            if record.state not in TERMINAL_STATES:
                self._records = previous
            raise

    def _require(self, operation_id: str) -> OperationRecord:
        record = self._records.get(operation_id)
        if record is None:
            raise KeyError(operation_id)
        return record

    # -- reads --------------------------------------------------------------

    def get(self, operation_id: str) -> OperationRecord | None:
        with self._lock:
            return self._records.get(operation_id)

    def list(self) -> tuple[OperationRecord, ...]:
        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def active(self, *, now: datetime | None = None) -> OperationRecord | None:
        """The apply/start still running, if any.

        ``applied`` is not running: applying finished, and a start may follow.
        A running *mirror* older than ``MIRROR_TTL_SECONDS`` no longer counts:
        an origin that crashed mid-operation must not wedge the lock forever.
        """

        moment = now or datetime.now(UTC)
        with self._lock:
            return next(
                (
                    record
                    for record in self._records.values()
                    if record.running and not _mirror_expired(record, moment)
                ),
                None,
            )

    def latest_session(self) -> OperationRecord | None:
        """The newest deployment a reloaded card can show or start again.

        A ready start wins over an applied plan; both are terminal enough to
        describe "the last thing the assistant did". Failed and running records
        are not a session: the card shows the failure from its own poll, and a
        running operation is polled, not resumed. Mirrors are another origin's
        lock markers, never this Mac's session.
        """

        with self._lock:
            records = sorted(self._records.values(), key=_sort_key)
        for record in reversed(records):
            if record.mirrored:
                continue
            if record.kind == KIND_START and record.state == STATE_READY:
                return record
        for record in reversed(records):
            if record.mirrored:
                continue
            if record.kind == KIND_APPLY and record.state == STATE_APPLIED:
                return record
        return None

    def applied(
        self,
        *,
        measurement_id: str,
        plan_signature: str,
        coordinator_node: str,
    ) -> OperationRecord | None:
        """The applied plan a start may build on, or ``None``.

        Matching on all three identities is what keeps "start" from launching a
        deployment that a different plan (or a different measurement) staged.
        The newest match wins, like ``latest_session``: a stale apply must not
        shadow the one the caller just approved. Mirrors are another origin's
        lock markers, never an applied plan this Mac may start.
        """

        with self._lock:
            records = sorted(self._records.values(), key=_sort_key)
        for record in reversed(records):
            if record.mirrored:
                continue
            if (
                record.kind == KIND_APPLY
                and record.state == STATE_APPLIED
                and record.measurement_id == measurement_id
                and record.plan_signature == plan_signature
                and record.coordinator_node == coordinator_node
            ):
                return record
        return None


def _mirror_expired(record: OperationRecord, now: datetime) -> bool:
    """Whether a running mirror is too old to still count as the lock.

    Only mirrors expire: this Mac's own running records are recovered as
    failed on restart, so they cannot outlive their job. An unreadable
    timestamp cannot prove the mirror is fresh, so it counts as expired.
    """

    if not record.mirrored:
        return False
    try:
        updated = datetime.fromisoformat(record.updated_at)
    except ValueError:
        return True
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return (now - updated).total_seconds() > MIRROR_TTL_SECONDS


def _may_advance(current: str, target: str) -> bool:
    """Forward only, or a metadata update inside a running state.

    A repeated state is how staging progress and the placement signature are
    recorded; a *backwards* state, or anything after a terminal state, is
    dropped. ``applied``/``ready`` are terminal, so a late start event cannot
    move an applied record and a late failure cannot undo "ready".
    """

    if current in TERMINAL_STATES:
        return False
    if current == target:
        return True
    return _STATE_RANK.get(target, -1) > _STATE_RANK.get(current, -1)


def _replaced(
    record: OperationRecord,
    *,
    state: str,
    progress: OperationProgress | None = None,
    endpoint: OperationEndpoint | None = None,
    warnings: Sequence[OperationWarning] | None = None,
    placement_signature: str | None = None,
    deployment_id: str | None = None,
    error: OperationFailure | None = None,
    now: datetime | None = None,
) -> OperationRecord:
    return OperationRecord(
        operation_id=record.operation_id,
        kind=record.kind,
        state=state,
        created_at=record.created_at,
        updated_at=(now or datetime.now(UTC)).isoformat(),
        measurement_id=record.measurement_id,
        plan_signature=record.plan_signature,
        placement_signature=(
            record.placement_signature
            if placement_signature is None
            else placement_signature
        ),
        coordinator_node=record.coordinator_node,
        coordinator_local=record.coordinator_local,
        rank_order=record.rank_order,
        context_tokens=record.context_tokens,
        target_context_tokens=record.target_context_tokens,
        model_path=record.model_path,
        rank0_extra_bytes=record.rank0_extra_bytes,
        microbatch_size=record.microbatch_size,
        mtp=record.mtp,
        execution_profile=record.execution_profile,
        policy=record.policy,
        progress=record.progress if progress is None else progress,
        endpoint=record.endpoint if endpoint is None else endpoint,
        deployment_id=(
            record.deployment_id if deployment_id is None else deployment_id
        ),
        stopped=record.stopped,
        mirrored=record.mirrored,
        # ``observed`` is terminal evidence and is only ever written by
        # ``OperationStore.observe``; a forward transition never clears it.
        observed=record.observed,
        warnings=record.warnings if warnings is None else tuple(warnings),
        error=error,
    )


def _sort_key(record: OperationRecord) -> tuple[str, str]:
    return record.created_at, record.operation_id


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _policy_from_dict(value: Any) -> dict[str, Any]:
    """A persisted resource-policy snapshot, copied into plain JSON values."""

    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(key, str)
        and isinstance(item, (str, int, float, bool, type(None)))
    }


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
