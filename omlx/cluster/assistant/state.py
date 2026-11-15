# SPDX-License-Identifier: Apache-2.0
"""Persisted measurement sets for the Cluster Assistant.

``/assistant/measure`` is a job, not a call: the probes it runs take seconds
over SSH, so the result has to outlive the request that started it. This module
is that record, and the only place that decides whether a stored measurement is
still usable for a plan.

The store mirrors ``cluster/registry.py``: one JSON file under
``<base_path>/cluster/``, one re-entrant lock, atomic replacement, and a load
that fails closed. A corrupt or unwritable file must never stop the local oMLX
server from starting — it only means no measurement is available, and the plan
answers ``409 stale_measurement`` with the one action ``measure``.

Two rules are deliberate.

1. **A measurement is dated, not refreshed.** ``measured_at`` is when the
   probes ran. Reading a record never rewrites it, and a record older than
   ``MEASUREMENT_TTL_SECONDS`` is stale even if it is the newest one.
2. **A measurement answers one question.** The plan must ask the same one: the
   same Macs, the same checkpoint identity, the same MTP setting. A plan for a
   different cluster is refused instead of answered from numbers that describe
   something else.

Context length and the rank-0 head/coordinator bytes stay plan-time inputs.
They are bounded by the admission the measurement took and are cheap to vary,
so requiring a new measurement for every context change would only teach people
to skip measuring. That is a decision, not an omission.

No secrets are stored: no SSH targets, no API keys, no raw probe output.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .election import NodeFacts
from .measurement import MeasurementSet, MeasurementWarning

MEASUREMENT_SCHEMA_VERSION = 1
MEASUREMENT_FILE_NAME = "assistant-measurements.json"

# A measurement older than this is stale. Long enough to walk from "measure"
# to "review the plan" in the UI; short enough that a plan cannot be approved
# against a laptop whose free memory changed while someone compiled something.
MEASUREMENT_TTL_SECONDS = 15 * 60

# Newest wins; older records are dropped on the next measure. Bounded because
# this is a scratch record, not a history: an unbounded file would be rewritten
# on every measure.
MAX_MEASUREMENT_RECORDS = 8

MEASUREMENT_RUNNING = "running"
MEASUREMENT_READY = "ready"
MEASUREMENT_FAILED = "failed"

# The one next step for a measurement that cannot be planned against.
ACTION_MEASURE = "measure"
# The one next step while another measurement is already running.
ACTION_STATUS = "status"

STALE_MEASUREMENT_CODE = "stale_measurement"

# Why a stored measurement cannot answer this plan. Machine-readable so the API
# does not have to be parsed as prose.
REFUSAL_NOT_READY = "state"
REFUSAL_STALE = "age"
REFUSAL_NODE_SET = "node_set"
REFUSAL_MTP = "mtp"
REFUSAL_MODEL = "model"

_ID_LOCK = threading.Lock()
_ID_SEQUENCE = 0


@dataclass(frozen=True)
class MeasurementEndpoint:
    """Where one Mac's endpoint lives. ``port`` 0 means "not resolved".

    The host is the private address discovery and pairing reported, never
    loopback: ``127.0.0.1`` is not an endpoint another Mac can dial. The port is
    that Mac's own ``ServerSettings.port``; this record only carries it so the
    pure plan can report it without reading settings itself.
    """

    host: str = ""
    port: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.port <= 65535:
            raise ValueError("endpoint port must be between 0 and 65535")

    @property
    def resolved(self) -> bool:
        """Whether this names an endpoint a client could actually dial."""

        return bool(self.host) and 1 <= self.port <= 65535

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MeasurementEndpoint:
        if not isinstance(payload, Mapping):
            raise ValueError("measurement endpoint is malformed")
        return cls(
            host=str(payload.get("host") or ""),
            port=_optional_int(payload.get("port")) or 0,
        )


@dataclass(frozen=True)
class ModelIdentity:
    """The checkpoint one measurement covers.

    Exactly one of ``model_path`` or ``model_size_bytes`` + ``layer_count``,
    mirroring the plan request: a real checkpoint on disk, or the pre-download
    size. ``hardware_only`` is the third, deliberately empty identity: the
    measurement answers "what hardware is here?" before a model is chosen, and
    a plan refuses it until the chosen model is measured again.
    """

    model_path: str = ""
    model_size_bytes: int | None = None
    layer_count: int = 0
    mtp: bool = False
    hardware_only: bool = False

    def __post_init__(self) -> None:
        path = self.model_path.strip()
        if self.hardware_only:
            if path or self.model_size_bytes is not None or self.mtp:
                raise ValueError(
                    "a hardware-only identity cannot name a model or MTP"
                )
            object.__setattr__(self, "model_path", "")
            object.__setattr__(self, "layer_count", 0)
            return
        if path:
            if self.model_size_bytes is not None:
                raise ValueError(
                    "provide exactly one of model_path or model_size_bytes"
                )
            object.__setattr__(self, "model_path", path)
            object.__setattr__(self, "layer_count", 0)
            return
        if self.model_size_bytes is None:
            raise ValueError("provide exactly one of model_path or model_size_bytes")
        if self.model_size_bytes <= 0:
            raise ValueError("model_size_bytes must be positive")
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")

    @property
    def identity_key(self) -> str:
        """What makes two measurements the same model, ignoring MTP."""

        if self.hardware_only:
            return "hardware"
        if self.model_path:
            return f"path:{os.path.normpath(os.path.expanduser(self.model_path))}"
        return f"size:{self.model_size_bytes}:{self.layer_count}"

    @property
    def label(self) -> str:
        """A stable, human-readable name for the plan object."""

        if self.hardware_only:
            return "hardware only (no model measured)"
        if self.model_path:
            return self.model_path
        size_gib = (self.model_size_bytes or 0) / 1024**3
        return f"synthetic model: {size_gib:.1f} GiB, {self.layer_count} layers"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_size_bytes": self.model_size_bytes,
            "layer_count": self.layer_count,
            "mtp": self.mtp,
            "hardware_only": self.hardware_only,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelIdentity:
        if not isinstance(payload, Mapping):
            raise ValueError("model identity is malformed")
        return cls(
            model_path=str(payload.get("model_path") or ""),
            model_size_bytes=_optional_int(payload.get("model_size_bytes")),
            layer_count=_optional_int(payload.get("layer_count")) or 0,
            mtp=bool(payload.get("mtp", False)),
            hardware_only=bool(payload.get("hardware_only", False)),
        )


@dataclass(frozen=True)
class MeasurementFailure:
    """Why a measurement job ended, and the one thing to do about it."""

    code: str
    message: str
    action: str = ACTION_MEASURE

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "action": self.action}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MeasurementFailure:
        if not isinstance(payload, Mapping):
            raise ValueError("measurement failure is malformed")
        return cls(
            code=str(payload.get("code") or ""),
            message=str(payload.get("message") or ""),
            action=str(payload.get("action") or ACTION_MEASURE),
        )


@dataclass(frozen=True)
class MeasurementRecord:
    """One measurement job and, when it finished, the facts it produced."""

    measurement_id: str
    state: str
    created_at: str
    updated_at: str
    model: ModelIdentity
    measured_at: str = ""
    requested_node_ids: tuple[str, ...] = ()
    endpoints: Mapping[str, MeasurementEndpoint] = field(default_factory=dict)
    #: ``ModelLayout.to_dict()`` of the checkpoint at measurement time. The plan
    #: is built from *what was measured*: re-reading a checkpoint whose shard
    #: files changed while the server ran made the same measurement elect a
    #: different coordinator between apply and start.
    layout: dict[str, Any] | None = None
    nodes: tuple[dict[str, Any], ...] = ()
    #: node_id -> the model list that Mac's own catalogue reported while it was
    #: measured: ``{"model_id", "model_path", "size_bytes"}`` entries. It is how
    #: the free-space affordance names a model the user may delete on a Mac that
    #: is short of disk; ``size_bytes`` 0 means that Mac reported no size. Empty
    #: means "no inventory was measured", never "this Mac holds nothing".
    inventories: Mapping[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    #: The saved per-model settings the origin could read at measurement time:
    #: memory tier, Engram/PLE switch, MoE offload and the model context window.
    #: The resource policy starts from these; a plan applies a change only
    #: through Start setup and shows it first. Empty means "not read".
    model_settings: Mapping[str, Any] = field(default_factory=dict)
    #: Measured point-to-point links inside this group, as sorted node-ID
    #: pairs. ``fabric_measured`` is False when the transport probe did not
    #: answer: an empty set means "unknown", never "no links".
    fabric_links: tuple[tuple[str, str], ...] = ()
    fabric_measured: bool = False
    warnings: tuple[MeasurementWarning, ...] = ()
    error: MeasurementFailure | None = None

    @property
    def ready(self) -> bool:
        return self.state == MEASUREMENT_READY

    @property
    def node_ids(self) -> tuple[str, ...]:
        """The Macs this measurement actually covers, in measured order."""

        return tuple(str(node.get("node_id") or "") for node in self.nodes)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def endpoint_for(self, node_id: str) -> MeasurementEndpoint | None:
        return self.endpoints.get(node_id)

    def measurement_set(self) -> MeasurementSet:
        """The stored facts back in the type the election consumes."""

        return MeasurementSet(
            nodes=tuple(NodeFacts.from_dict(node) for node in self.nodes),
            measured_at=self.measured_at,
            warnings=self.warnings,
        )

    def model_layout(self) -> Any:
        """The measured ``ModelLayout``, or ``None`` for a record without one."""

        if not self.layout:
            return None
        from ..planner import ModelLayout

        try:
            return ModelLayout.from_dict(dict(self.layout))
        except (KeyError, TypeError, ValueError):
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "measured_at": self.measured_at,
            "model": self.model.to_dict(),
            "requested_node_ids": list(self.requested_node_ids),
            "endpoints": {
                node_id: endpoint.to_dict()
                for node_id, endpoint in sorted(self.endpoints.items())
            },
            "layout": None if self.layout is None else dict(self.layout),
            "nodes": [dict(node) for node in self.nodes],
            "inventories": {
                node_id: [dict(item) for item in items]
                for node_id, items in sorted(self.inventories.items())
            },
            "model_settings": dict(self.model_settings),
            "fabric_links": [list(pair) for pair in self.fabric_links],
            "fabric_measured": self.fabric_measured,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "error": None if self.error is None else self.error.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MeasurementRecord:
        if not isinstance(payload, Mapping):
            raise ValueError("measurement record is malformed")
        measurement_id = str(payload.get("measurement_id") or "")
        if not measurement_id:
            raise ValueError("measurement record is missing an ID")
        state = str(payload.get("state") or "")
        if state not in (MEASUREMENT_RUNNING, MEASUREMENT_READY, MEASUREMENT_FAILED):
            raise ValueError(f"measurement {measurement_id} has an unknown state")
        raw_endpoints = payload.get("endpoints") or {}
        if not isinstance(raw_endpoints, Mapping):
            raise ValueError(f"measurement {measurement_id} has malformed endpoints")
        raw_nodes = payload.get("nodes") or []
        if not isinstance(raw_nodes, list):
            raise ValueError(f"measurement {measurement_id} has malformed nodes")
        raw_warnings = payload.get("warnings") or []
        if not isinstance(raw_warnings, list):
            raise ValueError(f"measurement {measurement_id} has malformed warnings")
        warnings = tuple(
            MeasurementWarning(
                code=str(item.get("code") or ""),
                message=str(item.get("message") or ""),
                node_id=str(item.get("node_id") or ""),
            )
            for item in raw_warnings
            if isinstance(item, Mapping)
        )
        if len(warnings) != len(raw_warnings):
            raise ValueError(f"measurement {measurement_id} has malformed warnings")
        error = payload.get("error")
        nodes = tuple(dict(node) for node in raw_nodes)
        inventories = _inventories_from_dict(
            payload.get("inventories"), measurement_id=measurement_id
        )
        # Validate the stored facts once, on load: a record that cannot be
        # turned back into ``NodeFacts`` is not a measurement, and the file
        # fails closed rather than handing a corrupt plan to the election.
        for node in nodes:
            try:
                NodeFacts.from_dict(node)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"measurement {measurement_id} has invalid node facts: {exc}"
                ) from exc
        return cls(
            measurement_id=measurement_id,
            state=state,
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            measured_at=str(payload.get("measured_at") or ""),
            model=ModelIdentity.from_dict(payload.get("model") or {}),
            requested_node_ids=tuple(
                str(item) for item in (payload.get("requested_node_ids") or ())
            ),
            endpoints={
                str(node_id): MeasurementEndpoint.from_dict(endpoint)
                for node_id, endpoint in raw_endpoints.items()
            },
            layout=(
                dict(payload["layout"])
                if isinstance(payload.get("layout"), Mapping)
                else None
            ),
            nodes=tuple(dict(node) for node in raw_nodes),
            inventories=inventories,
            model_settings=_settings_from_dict(payload.get("model_settings")),
            fabric_links=_fabric_links_from_dict(
                payload.get("fabric_links"), measurement_id=measurement_id
            ),
            fabric_measured=bool(payload.get("fabric_measured", False)),
            warnings=warnings,
            error=None if error is None else MeasurementFailure.from_dict(error),
        )


@dataclass(frozen=True)
class MeasurementRefusal:
    """Why a stored measurement cannot answer a plan, and the one next step."""

    reason: str
    message: str


def measurement_refusal(
    record: MeasurementRecord,
    *,
    node_ids: Sequence[str] | None = None,
    model: ModelIdentity,
    now: datetime | None = None,
    ttl_seconds: int = MEASUREMENT_TTL_SECONDS,
    hardware_only_ok: bool = False,
) -> MeasurementRefusal | None:
    """Whether this record may answer a plan for exactly this question.

    ``None`` means usable. Checking state first, then age, then identity keeps
    the reported reason the most fundamental one: a failed job is not "stale",
    and a stale record is not blamed on the model.
    """

    if not record.ready:
        if record.error is not None:
            return MeasurementRefusal(
                REFUSAL_NOT_READY,
                f"measurement {record.measurement_id} did not finish: "
                f"{record.error.message}",
            )
        return MeasurementRefusal(
            REFUSAL_NOT_READY,
            f"measurement {record.measurement_id} is still {record.state}; "
            "poll its status until it is ready",
        )

    taken = _parse_timestamp(record.measured_at) or _parse_timestamp(record.created_at)
    if taken is None:
        return MeasurementRefusal(
            REFUSAL_STALE,
            f"measurement {record.measurement_id} has no usable timestamp",
        )
    age = ((now or datetime.now(UTC)) - taken).total_seconds()
    if age > ttl_seconds:
        return MeasurementRefusal(
            REFUSAL_STALE,
            f"measurement {record.measurement_id} is {_describe_age(age)} old; "
            f"measurements expire after {ttl_seconds // 60} minutes",
        )

    measured_nodes = set(record.node_ids)
    requested_nodes = set(node_ids) if node_ids is not None else measured_nodes
    # The measurement covers every paired Mac (read-only, and it is what answers
    # "what is available"); the plan binds to the selected subset. A requested
    # node that was never measured is still a refusal: no plan may cover a Mac
    # whose admission nobody measured.
    missing = sorted(requested_nodes - measured_nodes)
    if missing:
        return MeasurementRefusal(
            REFUSAL_NODE_SET,
            f"measurement {record.measurement_id} covers "
            f"{', '.join(sorted(measured_nodes)) or 'no Macs'}; not measured: "
            + ", ".join(missing),
        )

    if record.model.mtp != model.mtp:
        return MeasurementRefusal(
            REFUSAL_MTP,
            "the measurement was taken with Lightning MTP "
            f"{'on' if record.model.mtp else 'off'}, and this plan asks for it "
            f"{'on' if model.mtp else 'off'}",
        )

    if record.model.hardware_only:
        if hardware_only_ok:
            return None
        return MeasurementRefusal(
            REFUSAL_MODEL,
            "the measurement covers hardware only; measure the chosen model "
            "before planning",
        )

    if record.model.identity_key != model.identity_key:
        return MeasurementRefusal(
            REFUSAL_MODEL,
            f"measurement {record.measurement_id} covers {record.model.label}, "
            f"not {model.label}",
        )

    return None


def new_measurement_id(
    *, now: datetime | None = None, existing: Sequence[str] = ()
) -> str:
    """A stable, opaque, never-reused ID: ``ms-<UTC>-<8 hex>``.

    The counter in the hex suffix keeps IDs generated inside one second
    ordered, so "the newest measurement" is the last one by ID as well as by
    time. The random half keeps them opaque and unguessable.
    """

    global _ID_SEQUENCE
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    known = set(existing)
    for _ in range(64):
        with _ID_LOCK:
            _ID_SEQUENCE += 1
            sequence = _ID_SEQUENCE
        candidate = f"ms-{stamp}-{sequence & 0xFFFF:04x}{secrets.token_hex(2)}"
        if candidate not in known:
            return candidate
    raise RuntimeError("could not generate a unique measurement ID")


class MeasurementStore:
    """Thread-safe, atomic, bounded store of measurement records.

    Deliberately not a history: ``start`` prunes to the newest
    ``MAX_MEASUREMENT_RECORDS``. The file is tiny, rewritten whole, and read
    once at startup.
    """

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / MEASUREMENT_FILE_NAME
        self._lock = threading.RLock()
        self._records: dict[str, MeasurementRecord] = {}
        self.load_error: str | None = None
        #: Running records that were recovered as failed on load.
        self.recovered = 0
        try:
            self._load()
        except (KeyError, TypeError, ValueError) as exc:
            # A corrupt scratch file must never prevent the local oMLX server
            # and GUI from starting. Fail closed: no measurement is available.
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
                raise ValueError(
                    f"could not read assistant measurements: {exc}"
                ) from exc
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != MEASUREMENT_SCHEMA_VERSION
            ):
                raise ValueError("unsupported assistant measurement schema")
            raw_records = payload.get("measurements")
            if not isinstance(raw_records, list):
                raise ValueError("assistant measurement file is malformed")
            records = [MeasurementRecord.from_dict(item) for item in raw_records]
            if len({record.measurement_id for record in records}) != len(records):
                raise ValueError("assistant measurement file has duplicate IDs")
            # A job cannot survive the process that ran it. Reporting it as
            # running would leave the UI polling a ghost forever.
            recovered = [
                _with_state(
                    record,
                    state=MEASUREMENT_FAILED,
                    updated_at=record.updated_at,
                    error=MeasurementFailure(
                        code="measurement_interrupted",
                        message=(
                            "the measurement was interrupted by a server restart "
                            "and never finished"
                        ),
                    ),
                )
                if record.state == MEASUREMENT_RUNNING
                else record
                for record in records
            ]
            self.recovered = sum(
                1
                for before, after in zip(records, recovered)
                if before.state != after.state
            )
            self._records = {record.measurement_id: record for record in recovered}
            self._prune()
            if self.recovered:
                # Persist the recovered states once; never let an unwritable
                # file block startup.
                with suppress(OSError):
                    self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MEASUREMENT_SCHEMA_VERSION,
            "measurements": [
                self._records[measurement_id].to_dict()
                for measurement_id in sorted(self._records)
            ],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".assistant-measurements.",
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
            surplus = len(self._records) - MAX_MEASUREMENT_RECORDS
            if surplus <= 0:
                return 0
            for measurement_id in sorted(self._records)[:surplus]:
                del self._records[measurement_id]
            return surplus

    # -- job lifecycle ------------------------------------------------------

    def start(
        self,
        *,
        model: ModelIdentity,
        node_ids: Sequence[str],
        now: datetime | None = None,
    ) -> MeasurementRecord:
        """Create a ``running`` record and persist it before any probe runs."""

        moment = now or datetime.now(UTC)
        stamp = moment.isoformat()
        with self._lock:
            measurement_id = new_measurement_id(
                now=moment, existing=tuple(self._records)
            )
            previous = dict(self._records)
            record = MeasurementRecord(
                measurement_id=measurement_id,
                state=MEASUREMENT_RUNNING,
                created_at=stamp,
                updated_at=stamp,
                model=model,
                requested_node_ids=tuple(node_ids),
            )
            self._records[measurement_id] = record
            self._prune()
            try:
                self._save()
            except Exception:
                self._records = previous
                raise
            self.load_error = None
            return record

    def finish(
        self,
        measurement_id: str,
        *,
        measurement: MeasurementSet,
        endpoints: Mapping[str, MeasurementEndpoint],
        now: datetime | None = None,
        layout: Mapping[str, Any] | None = None,
        inventories: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        model_settings: Mapping[str, Any] | None = None,
        fabric_links: Sequence[Sequence[str]] | None = None,
        fabric_measured: bool | None = None,
    ) -> MeasurementRecord:
        """Store the facts a finished job measured."""

        stamp = (now or datetime.now(UTC)).isoformat()
        with self._lock:
            record = self._require(measurement_id)
            updated = _with_state(
                record,
                state=MEASUREMENT_READY,
                updated_at=stamp,
                measured_at=measurement.measured_at or record.measured_at,
                endpoints=dict(endpoints),
                nodes=tuple(node.to_dict() for node in measurement.nodes),
                inventories=_inventories_from_dict(
                    None
                    if inventories is None
                    else {
                        node_id: [dict(item) for item in items]
                        for node_id, items in inventories.items()
                    },
                    measurement_id=measurement_id,
                ),
                model_settings=_settings_from_dict(model_settings),
                fabric_links=(
                    None
                    if fabric_links is None
                    else tuple((str(pair[0]), str(pair[1])) for pair in fabric_links)
                ),
                fabric_measured=fabric_measured,
                warnings=tuple(measurement.warnings),
                error=None,
                layout=None if layout is None else dict(layout),
            )
            self._commit(updated)
            return updated

    def fail(
        self,
        measurement_id: str,
        *,
        code: str,
        message: str,
        action: str = ACTION_MEASURE,
        now: datetime | None = None,
    ) -> MeasurementRecord:
        """End a job in ``failed`` with exactly one actionable next step."""

        stamp = (now or datetime.now(UTC)).isoformat()
        with self._lock:
            record = self._require(measurement_id)
            updated = _with_state(
                record,
                state=MEASUREMENT_FAILED,
                updated_at=stamp,
                error=MeasurementFailure(code=code, message=message, action=action),
            )
            self._commit(updated)
            return updated

    def _commit(self, record: MeasurementRecord) -> None:
        # Only terminal states (ready/failed) are committed here. A terminal
        # state is the truth about the job even when the file could not be
        # written: rolling it back left the record "running" forever, refused
        # every new measurement and wedged the assistant (review B6). Keep it
        # in memory and let the save error reach the caller, which surfaces it.
        self._records[record.measurement_id] = record
        self._save()

    def _require(self, measurement_id: str) -> MeasurementRecord:
        record = self._records.get(measurement_id)
        if record is None:
            raise KeyError(measurement_id)
        return record

    # -- reads --------------------------------------------------------------

    def get(self, measurement_id: str) -> MeasurementRecord | None:
        with self._lock:
            return self._records.get(measurement_id)

    def list(self) -> tuple[MeasurementRecord, ...]:
        with self._lock:
            return tuple(self._records[key] for key in sorted(self._records))

    def active(self) -> MeasurementRecord | None:
        """The measurement still running, if any.

        One at a time on purpose: a second measure would either overwrite this
        job's record or measure a different node set under a new ID, and a UI
        polling the first one would never learn which.
        """

        with self._lock:
            return next(
                (
                    record
                    for record in self._records.values()
                    if record.state == MEASUREMENT_RUNNING
                ),
                None,
            )


def _with_state(
    record: MeasurementRecord,
    *,
    state: str,
    updated_at: str,
    measured_at: str | None = None,
    endpoints: Mapping[str, MeasurementEndpoint] | None = None,
    nodes: tuple[dict[str, Any], ...] | None = None,
    inventories: Mapping[str, tuple[dict[str, Any], ...]] | None = None,
    warnings: tuple[MeasurementWarning, ...] | None = None,
    error: MeasurementFailure | None = None,
    layout: Mapping[str, Any] | None = None,
    model_settings: Mapping[str, Any] | None = None,
    fabric_links: tuple[tuple[str, str], ...] | None = None,
    fabric_measured: bool | None = None,
) -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=record.measurement_id,
        state=state,
        created_at=record.created_at,
        updated_at=updated_at,
        measured_at=record.measured_at if measured_at is None else measured_at,
        model=record.model,
        requested_node_ids=record.requested_node_ids,
        endpoints=record.endpoints if endpoints is None else endpoints,
        nodes=record.nodes if nodes is None else nodes,
        inventories=(
            record.inventories if inventories is None else dict(inventories)
        ),
        model_settings=(
            record.model_settings if model_settings is None else dict(model_settings)
        ),
        fabric_links=(
            record.fabric_links if fabric_links is None else fabric_links
        ),
        fabric_measured=(
            record.fabric_measured
            if fabric_measured is None
            else bool(fabric_measured)
        ),
        warnings=record.warnings if warnings is None else warnings,
        error=error,
        layout=record.layout if layout is None else dict(layout),
    )


def _fabric_links_from_dict(
    payload: Any, *, measurement_id: str
) -> tuple[tuple[str, str], ...]:
    """Parse measured link pairs, failing closed on a malformed record."""

    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError(f"measurement {measurement_id} has malformed fabric links")
    links: list[tuple[str, str]] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"measurement {measurement_id} has malformed fabric links"
            )
        first, second = str(item[0]).strip(), str(item[1]).strip()
        if not first or not second or first == second:
            raise ValueError(
                f"measurement {measurement_id} has malformed fabric links"
            )
        links.append((first, second))
    return tuple(links)


def _inventories_from_dict(
    payload: Any, *, measurement_id: str
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Parse the stored per-Mac model lists, failing closed on a bad one.

    The entries are display and delete evidence for the free-space affordance,
    so a malformed one is a broken record, not a model list to guess at.
    """

    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"measurement {measurement_id} has malformed inventories")
    inventories: dict[str, tuple[dict[str, Any], ...]] = {}
    for node_id, items in payload.items():
        if not isinstance(node_id, str) or not isinstance(items, list):
            raise ValueError(f"measurement {measurement_id} has malformed inventories")
        entries: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"measurement {measurement_id} has malformed model entries"
                )
            model_id = str(item.get("model_id") or "").strip()
            model_path = str(item.get("model_path") or "").strip()
            if not model_id or not model_path:
                raise ValueError(
                    f"measurement {measurement_id} has an unidentifiable model entry"
                )
            entries.append(
                {
                    "model_id": model_id,
                    "model_path": model_path,
                    "size_bytes": max(0, _optional_int(item.get("size_bytes")) or 0),
                }
            )
        inventories[node_id] = tuple(entries)
    return inventories


def _settings_from_dict(payload: Any) -> dict[str, Any]:
    """A saved-settings snapshot, copied into plain JSON values.

    The settings file is user-editable, so an odd record is dropped rather than
    failing the whole measurement: the plan then starts from the documented
    defaults and the warnings still describe what was measured.
    """

    if not isinstance(payload, Mapping):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(key, str)
        and isinstance(value, (str, int, float, bool, type(None)))
    }


def _parse_timestamp(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _describe_age(seconds: float) -> str:
    if seconds < 120:
        return f"{int(seconds)} seconds"
    if seconds < 7200:
        return f"{int(seconds // 60)} minutes"
    return f"{seconds / 3600:.1f} hours"


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
