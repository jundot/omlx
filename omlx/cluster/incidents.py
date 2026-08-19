# SPDX-License-Identifier: Apache-2.0
"""Server-owned incident log: every abnormal end state becomes a durable record.

The dashboard polls with a ``since`` cursor and merges — it can add to this
record but never erase it, which makes "error state wiped by the next poll"
structurally impossible. Dismissal is the only removal path, and supersession
("attempt #4 replaces #3") flags but keeps the earlier record.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

_MAX_INCIDENTS = 200


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class Incident:
    """One abnormal (or notable) cluster end state, already redacted."""

    seq: int
    id: str
    ts: float
    severity: Severity
    source: str
    state_code: str
    message: str
    guidance_code: str | None = None
    job_id: str | None = None
    deployment_id: str | None = None
    bundle_ref: str | None = None
    dismissed_at: float | None = None
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = str(self.severity)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Incident:
        if not isinstance(data, dict):
            raise ValueError("incident record is malformed")
        try:
            return cls(
                seq=int(data["seq"]),
                id=str(data["id"]),
                ts=float(data["ts"]),
                severity=Severity(str(data["severity"])),
                source=str(data["source"]),
                state_code=str(data["state_code"]),
                message=str(data["message"]),
                guidance_code=(
                    str(data["guidance_code"])
                    if data.get("guidance_code") is not None
                    else None
                ),
                job_id=(
                    str(data["job_id"]) if data.get("job_id") is not None else None
                ),
                deployment_id=(
                    str(data["deployment_id"])
                    if data.get("deployment_id") is not None
                    else None
                ),
                bundle_ref=(
                    str(data["bundle_ref"])
                    if data.get("bundle_ref") is not None
                    else None
                ),
                dismissed_at=(
                    float(data["dismissed_at"])
                    if data.get("dismissed_at") is not None
                    else None
                ),
                superseded_by=(
                    str(data["superseded_by"])
                    if data.get("superseded_by") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"incident record is malformed: {exc}") from exc


class IncidentStore:
    """Thread-safe, persisted FIFO ring of the last ``_MAX_INCIDENTS`` incidents.

    ``seq`` is store-assigned and strictly monotonic — it is the poll cursor
    the dashboard sends back as ``since``, which makes monotonic merge a
    server-enforced property rather than client discipline.
    """

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / "incidents.json"
        self._lock = threading.RLock()
        self._incidents: list[Incident] = []
        self._next_seq = 1
        self._epoch = uuid.uuid4().hex[:16]
        self.load_error: str | None = None
        try:
            self._load()
        except ValueError as exc:
            # A corrupt optional incident log must never prevent the local
            # oMLX server and GUI from starting. Fail closed: report none.
            # ``_next_seq`` resets to 1 here, so the fresh ``_epoch`` from
            # ``__init__`` stays — it is what tells an open dashboard tab its
            # cursor no longer means anything (a cursor above the reset seq
            # would otherwise silence the feed for that tab forever).
            self._incidents = []
            self._next_seq = 1
            self.load_error = str(exc)

    @property
    def epoch(self) -> str:
        """Identity of this seq numbering; changes when the log is reset."""

        with self._lock:
            return self._epoch

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._incidents = []
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"could not read cluster incident log: {exc}") from exc
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("unsupported cluster incident log schema")
            raw_incidents = payload.get("incidents")
            if not isinstance(raw_incidents, list):
                raise ValueError("cluster incident log is malformed")
            incidents = [Incident.from_dict(item) for item in raw_incidents]
            incidents.sort(key=lambda incident: incident.seq)
            if len({incident.id for incident in incidents}) != len(incidents):
                raise ValueError("cluster incident log has duplicate incident IDs")
            self._incidents = incidents[-_MAX_INCIDENTS:]
            highest = max((incident.seq for incident in incidents), default=0)
            self._next_seq = max(int(payload.get("next_seq", 0)), highest + 1, 1)
            stored_epoch = payload.get("epoch")
            if isinstance(stored_epoch, str) and stored_epoch:
                self._epoch = stored_epoch

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "next_seq": self._next_seq,
            "epoch": self._epoch,
            "incidents": [incident.to_dict() for incident in self._incidents],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".incidents.",
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

    def record(
        self,
        severity: Severity,
        source: str,
        state_code: str,
        message: str,
        *,
        guidance_code: str | None = None,
        job_id: str | None = None,
        deployment_id: str | None = None,
        bundle_ref: str | None = None,
    ) -> Incident:
        """Append one incident; the only I/O beyond the list is the save."""

        with self._lock:
            incident = Incident(
                seq=self._next_seq,
                id=uuid.uuid4().hex,
                ts=time.time(),
                severity=Severity(severity),
                source=source,
                state_code=state_code,
                message=message,
                guidance_code=guidance_code,
                job_id=job_id,
                deployment_id=deployment_id,
                bundle_ref=bundle_ref,
            )
            previous = list(self._incidents)
            previous_seq = self._next_seq
            self._incidents.append(incident)
            del self._incidents[:-_MAX_INCIDENTS]
            self._next_seq += 1
            try:
                self._save()
            except Exception:
                self._incidents = previous
                self._next_seq = previous_seq
                raise
            self.load_error = None
            return incident

    def list(self, since_seq: int = 0) -> tuple[Incident, ...]:
        with self._lock:
            return tuple(
                incident
                for incident in self._incidents
                if incident.seq > since_seq
            )

    def latest_seq(self) -> int:
        with self._lock:
            return self._incidents[-1].seq if self._incidents else 0

    def dismiss(self, incident_id: str) -> bool:
        with self._lock:
            index = next(
                (
                    position
                    for position, incident in enumerate(self._incidents)
                    if incident.id == incident_id
                ),
                None,
            )
            if index is None:
                return False
            previous = list(self._incidents)
            if self._incidents[index].dismissed_at is None:
                self._incidents[index] = replace(
                    self._incidents[index], dismissed_at=time.time()
                )
            try:
                self._save()
            except Exception:
                self._incidents = previous
                raise
            return True

    def supersede(self, job_id: str, by_incident_id: str) -> int:
        """Flag every incident of ``job_id`` as superseded — kept, not deleted."""

        with self._lock:
            previous = list(self._incidents)
            changed = 0
            for position, incident in enumerate(self._incidents):
                if (
                    incident.job_id == job_id
                    and incident.id != by_incident_id
                    and incident.superseded_by is None
                ):
                    self._incidents[position] = replace(
                        incident, superseded_by=by_incident_id
                    )
                    changed += 1
            if not changed:
                return 0
            try:
                self._save()
            except Exception:
                self._incidents = previous
                raise
            return changed

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "incidents": [incident.to_dict() for incident in self._incidents],
                "latest_seq": self.latest_seq(),
                "load_error": self.load_error,
            }


_incidents_lock = threading.Lock()
_configured_incidents: IncidentStore | None = None


def configure_cluster_incidents(base_path: Path) -> IncidentStore:
    global _configured_incidents
    with _incidents_lock:
        _configured_incidents = IncidentStore(base_path)
        return _configured_incidents


def get_cluster_incidents() -> IncidentStore:
    with _incidents_lock:
        if _configured_incidents is None:
            raise RuntimeError("cluster incident store is not configured")
        return _configured_incidents
