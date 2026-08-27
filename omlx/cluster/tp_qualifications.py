# SPDX-License-Identifier: Apache-2.0
"""Persisted, exact-match qualifications for heterogeneous TP layouts.

Unequal tensor partitions are an executable memory contract, not a tuning
hint.  A vector measured on one model/runtime/fabric combination must never be
silently reused after a checkpoint, kernel, JACCL library, node order, batch
profile, or MTP graph changes.  This module stores only full-model,
parity-qualified records and returns a candidate only when every key field is
an exact match.

The store deliberately has no "nearest" lookup.  Missing evidence, an old
schema, or a corrupt file produces no qualification and leaves the planner on
its equal-shard fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .strategy_benchmarks import context_bucket

_SCHEMA_VERSION = 1
_MAX_RECORDS = 256
_MAX_TEXT = 2_000
_MAX_RATE = 1e9
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_BACKENDS = frozenset({"ring", "jaccl", "jaccl-ring"})
_PROFILES = frozenset({"interactive", "balanced", "throughput"})


def _strict_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ValueError(
            f"{label} fields are invalid (missing={missing}, extra={extra})"
        )


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    value = value.strip()
    if (not value and not allow_empty) or len(value) > _MAX_TEXT or "\x00" in value:
        raise ValueError(f"{label} is invalid")
    return value


def _digest(value: Any, label: str) -> str:
    value = _text(value, label)
    if _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, label: str, *, maximum: int = 2**63 - 1) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_rate(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite positive rate") from exc
    if not math.isfinite(result) or not 0 < result <= _MAX_RATE:
        raise ValueError(f"{label} must be a finite positive rate")
    return result


def _canonical_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class TPNodeFingerprint:
    """Stable rank-ordered hardware, runtime, kernel, and fabric identity."""

    node_id: str
    chip_name: str
    physical_memory_bytes: int
    accelerator: str
    fabric_identifier: str
    omlx_version: str
    mlx_version: str
    mlx_lm_version: str
    python_version: str
    os_name: str
    os_version: str
    jaccl_identifier: str
    kernel_identifier: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or _NODE_ID.fullmatch(self.node_id) is None:
            raise ValueError("qualification node_id is invalid")
        object.__setattr__(
            self,
            "physical_memory_bytes",
            _positive_int(self.physical_memory_bytes, "physical memory"),
        )
        for name in (
            "chip_name",
            "accelerator",
            "omlx_version",
            "mlx_version",
            "mlx_lm_version",
            "python_version",
            "os_name",
            "os_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "fabric_identifier",
            "jaccl_identifier",
            "kernel_identifier",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "chip_name": self.chip_name,
            "physical_memory_bytes": self.physical_memory_bytes,
            "accelerator": self.accelerator,
            "fabric_identifier": self.fabric_identifier,
            "omlx_version": self.omlx_version,
            "mlx_version": self.mlx_version,
            "mlx_lm_version": self.mlx_lm_version,
            "python_version": self.python_version,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "jaccl_identifier": self.jaccl_identifier,
            "kernel_identifier": self.kernel_identifier,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TPNodeFingerprint:
        if not isinstance(payload, dict):
            raise ValueError("qualification node fingerprint must be an object")
        expected = {
            "node_id",
            "chip_name",
            "physical_memory_bytes",
            "accelerator",
            "fabric_identifier",
            "omlx_version",
            "mlx_version",
            "mlx_lm_version",
            "python_version",
            "os_name",
            "os_version",
            "jaccl_identifier",
            "kernel_identifier",
        }
        _strict_keys(payload, expected, "qualification node fingerprint")
        return cls(**payload)


@dataclass(frozen=True)
class TPQualificationKey:
    """Everything that must match before an unequal TP vector is reusable."""

    model_identity: str
    nodes: tuple[TPNodeFingerprint, ...]
    backend: str
    tensor_parallel_size: int
    context_bucket: int
    execution_profile: str
    microbatch_size: int
    decode_concurrency: int
    prompt_concurrency: int
    prefill_step_size: int
    auto_tune: bool
    mtp_enabled: bool
    mtp_depth: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "model_identity", _digest(self.model_identity, "model identity")
        )
        if not isinstance(self.nodes, tuple):
            object.__setattr__(self, "nodes", tuple(self.nodes))
        if len(self.nodes) < 2 or len(self.nodes) > 64:
            raise ValueError("qualification requires 2..64 ordered nodes")
        if len({item.node_id for item in self.nodes}) != len(self.nodes):
            raise ValueError("qualification node IDs must be unique")
        if self.backend not in _BACKENDS:
            raise ValueError("qualification backend is invalid")
        size = _positive_int(
            self.tensor_parallel_size,
            "tensor parallel size",
            maximum=len(self.nodes),
        )
        if size != len(self.nodes):
            raise ValueError("layout qualification currently requires pure TP")
        expected_bucket = context_bucket(self.context_bucket)
        if expected_bucket != self.context_bucket:
            raise ValueError("qualification context must be a canonical bucket")
        if self.execution_profile not in _PROFILES:
            raise ValueError("qualification execution profile is invalid")
        for name in (
            "microbatch_size",
            "decode_concurrency",
            "prompt_concurrency",
            "prefill_step_size",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name, maximum=1_048_576),
            )
        if self.prompt_concurrency > self.decode_concurrency:
            raise ValueError("qualification prompt concurrency exceeds decode")
        if self.microbatch_size > self.decode_concurrency:
            raise ValueError("qualification microbatch exceeds decode concurrency")
        if not isinstance(self.auto_tune, bool) or not isinstance(self.mtp_enabled, bool):
            raise ValueError("qualification execution flags must be booleans")
        if self.mtp_enabled:
            object.__setattr__(
                self,
                "mtp_depth",
                _positive_int(self.mtp_depth, "MTP depth", maximum=8),
            )
        elif self.mtp_depth is not None:
            raise ValueError("qualification MTP depth requires MTP enabled")

    @property
    def qualification_id(self) -> str:
        return _canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_identity": self.model_identity,
            "nodes": [item.to_dict() for item in self.nodes],
            "backend": self.backend,
            "tensor_parallel_size": self.tensor_parallel_size,
            "context_bucket": self.context_bucket,
            "execution_profile": self.execution_profile,
            "microbatch_size": self.microbatch_size,
            "decode_concurrency": self.decode_concurrency,
            "prompt_concurrency": self.prompt_concurrency,
            "prefill_step_size": self.prefill_step_size,
            "auto_tune": self.auto_tune,
            "mtp_enabled": self.mtp_enabled,
            "mtp_depth": self.mtp_depth,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TPQualificationKey:
        if not isinstance(payload, dict):
            raise ValueError("qualification key must be an object")
        expected = {
            "model_identity",
            "nodes",
            "backend",
            "tensor_parallel_size",
            "context_bucket",
            "execution_profile",
            "microbatch_size",
            "decode_concurrency",
            "prompt_concurrency",
            "prefill_step_size",
            "auto_tune",
            "mtp_enabled",
            "mtp_depth",
        }
        _strict_keys(payload, expected, "qualification key")
        raw_nodes = payload["nodes"]
        if not isinstance(raw_nodes, list):
            raise ValueError("qualification nodes must be an array")
        return cls(
            **{key: value for key, value in payload.items() if key != "nodes"},
            nodes=tuple(TPNodeFingerprint.from_dict(item) for item in raw_nodes),
        )


@dataclass(frozen=True)
class TPRateEvidence:
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    samples: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prefill_tokens_per_second",
            _positive_rate(self.prefill_tokens_per_second, "prefill throughput"),
        )
        object.__setattr__(
            self,
            "decode_tokens_per_second",
            _positive_rate(self.decode_tokens_per_second, "decode throughput"),
        )
        object.__setattr__(
            self, "samples", _positive_int(self.samples, "sample count", maximum=10_000)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TPRateEvidence:
        if not isinstance(payload, dict):
            raise ValueError("qualification rate evidence must be an object")
        _strict_keys(
            payload,
            {
                "prefill_tokens_per_second",
                "decode_tokens_per_second",
                "samples",
            },
            "qualification rate evidence",
        )
        return cls(**payload)


@dataclass(frozen=True)
class TPLayoutQualification:
    key: TPQualificationKey
    shard_weights: tuple[int, ...]
    equal_control: TPRateEvidence
    candidate: TPRateEvidence
    exact: bool
    parity_sha256: str
    promotable: bool
    reason: str
    qualified_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.shard_weights, tuple):
            object.__setattr__(self, "shard_weights", tuple(self.shard_weights))
        if len(self.shard_weights) != self.key.tensor_parallel_size:
            raise ValueError("qualification vector length does not match TP size")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= 4096
            for value in self.shard_weights
        ):
            raise ValueError("qualification vector must contain positive integers")
        if len(set(self.shard_weights)) < 2:
            raise ValueError("qualification vector must be asymmetric")
        if not isinstance(self.exact, bool) or not isinstance(self.promotable, bool):
            raise ValueError("qualification exact/promotable flags must be booleans")
        object.__setattr__(
            self, "parity_sha256", _digest(self.parity_sha256, "parity hash")
        )
        object.__setattr__(self, "reason", _text(self.reason, "qualification reason"))
        qualified = _text(self.qualified_at, "qualification timestamp")
        try:
            datetime.fromisoformat(qualified.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("qualification timestamp is invalid") from exc
        if self.promotable and not self.exact:
            raise ValueError("an inexact qualification cannot be promotable")
        if self.promotable and min(
            self.equal_control.samples,
            self.candidate.samples,
        ) < 2:
            raise ValueError("a promotable qualification needs two matched samples")

    @property
    def qualification_id(self) -> str:
        return self.key.qualification_id

    @property
    def record_digest(self) -> str:
        return _canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "shard_weights": list(self.shard_weights),
            "equal_control": self.equal_control.to_dict(),
            "candidate": self.candidate.to_dict(),
            "exact": self.exact,
            "parity_sha256": self.parity_sha256,
            "promotable": self.promotable,
            "reason": self.reason,
            "qualified_at": self.qualified_at,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TPLayoutQualification:
        if not isinstance(payload, dict):
            raise ValueError("TP layout qualification must be an object")
        expected = {
            "key",
            "shard_weights",
            "equal_control",
            "candidate",
            "exact",
            "parity_sha256",
            "promotable",
            "reason",
            "qualified_at",
        }
        _strict_keys(payload, expected, "TP layout qualification")
        weights = payload["shard_weights"]
        if not isinstance(weights, list):
            raise ValueError("qualification shard weights must be an array")
        return cls(
            key=TPQualificationKey.from_dict(payload["key"]),
            shard_weights=tuple(weights),
            equal_control=TPRateEvidence.from_dict(payload["equal_control"]),
            candidate=TPRateEvidence.from_dict(payload["candidate"]),
            exact=payload["exact"],
            parity_sha256=payload["parity_sha256"],
            promotable=payload["promotable"],
            reason=payload["reason"],
            qualified_at=payload["qualified_at"],
        )


@dataclass(frozen=True)
class TPQualificationProvenance:
    """Small immutable proof carried into the signed plan and deployment."""

    source: Literal["persistent", "environment_override"]
    qualification_id: str
    record_digest: str
    shard_weights: tuple[int, ...]
    exact: bool
    parity_sha256: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.source not in {"persistent", "environment_override"}:
            raise ValueError("qualification provenance source is invalid")
        object.__setattr__(
            self,
            "qualification_id",
            _digest(self.qualification_id, "qualification provenance ID"),
        )
        object.__setattr__(
            self,
            "record_digest",
            _digest(self.record_digest, "qualification record digest"),
        )
        if not isinstance(self.shard_weights, tuple):
            object.__setattr__(self, "shard_weights", tuple(self.shard_weights))
        if not self.shard_weights or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in self.shard_weights
        ):
            raise ValueError("qualification provenance vector is invalid")
        if not isinstance(self.exact, bool):
            raise ValueError("qualification provenance exact flag is invalid")
        if self.parity_sha256 is not None:
            object.__setattr__(
                self,
                "parity_sha256",
                _digest(self.parity_sha256, "qualification parity hash"),
            )
        if self.source == "persistent" and (
            not self.exact or self.parity_sha256 is None
        ):
            raise ValueError("persistent qualification provenance must prove parity")
        if self.source == "environment_override" and self.exact:
            raise ValueError("an environment override cannot claim qualification")
        object.__setattr__(self, "reason", _text(self.reason, "qualification reason"))

    @classmethod
    def from_record(cls, record: TPLayoutQualification) -> TPQualificationProvenance:
        return cls(
            source="persistent",
            qualification_id=record.qualification_id,
            record_digest=record.record_digest,
            shard_weights=record.shard_weights,
            exact=record.exact,
            parity_sha256=record.parity_sha256,
            reason=record.reason,
        )

    @classmethod
    def environment(cls, shard_weights: Sequence[int]) -> TPQualificationProvenance:
        weights = tuple(int(value) for value in shard_weights)
        payload = {"source": "environment_override", "shard_weights": weights}
        digest = _canonical_digest(payload)
        return cls(
            source="environment_override",
            qualification_id=digest,
            record_digest=digest,
            shard_weights=weights,
            exact=False,
            parity_sha256=None,
            reason=(
                "Experimental coordinator environment override; not a persisted "
                "hardware/runtime qualification"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "qualification_id": self.qualification_id,
            "record_digest": self.record_digest,
            "shard_weights": list(self.shard_weights),
            "exact": self.exact,
            "parity_sha256": self.parity_sha256,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TPQualificationProvenance:
        if not isinstance(payload, dict):
            raise ValueError("qualification provenance must be an object")
        expected = {
            "source",
            "qualification_id",
            "record_digest",
            "shard_weights",
            "exact",
            "parity_sha256",
            "reason",
        }
        _strict_keys(payload, expected, "qualification provenance")
        weights = payload["shard_weights"]
        if not isinstance(weights, list):
            raise ValueError("qualification provenance weights must be an array")
        return cls(
            **{key: value for key, value in payload.items() if key != "shard_weights"},
            shard_weights=tuple(weights),
        )


class TPLayoutQualificationStore:
    """Atomic, mode-0600 store indexed by the complete qualification key."""

    def __init__(self, base_path: Path) -> None:
        self.path = Path(base_path) / "cluster" / "tp-layout-qualifications.json"
        self._lock = threading.RLock()
        self._records: dict[str, TPLayoutQualification] = {}
        self.load_error: str | None = None
        try:
            self._load()
        except ValueError as exc:
            self.load_error = str(exc)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read TP layout qualifications: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("TP layout qualification store must be an object")
        _strict_keys(
            payload,
            {"schema_version", "qualifications"},
            "TP layout qualification store",
        )
        if payload["schema_version"] != _SCHEMA_VERSION:
            raise ValueError("unsupported TP layout qualification schema")
        raw = payload["qualifications"]
        if not isinstance(raw, list) or len(raw) > _MAX_RECORDS:
            raise ValueError("TP layout qualifications must be a bounded array")
        records: dict[str, TPLayoutQualification] = {}
        for item in raw:
            record = TPLayoutQualification.from_dict(item)
            if record.qualification_id in records:
                raise ValueError("duplicate TP layout qualification key")
            records[record.qualification_id] = record
        self._records = records

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".tp-layout-qualifications.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "qualifications": [
                            record.to_dict()
                            for _, record in sorted(self._records.items())
                        ],
                    },
                    stream,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def record(self, qualification: TPLayoutQualification) -> None:
        if not isinstance(qualification, TPLayoutQualification):
            raise TypeError("qualification must be a TPLayoutQualification")
        with self._lock:
            if self.load_error is not None:
                raise ValueError(
                    "refusing to overwrite an unreadable TP qualification store: "
                    + self.load_error
                )
            before = dict(self._records)
            self._records[qualification.qualification_id] = qualification
            if len(self._records) > _MAX_RECORDS:
                # Dict insertion order is deterministic; discard the oldest
                # qualification only after a replacement has been accounted for.
                oldest = next(iter(self._records))
                del self._records[oldest]
            try:
                self._save()
            except BaseException:
                self._records = before
                raise

    def lookup(self, key: TPQualificationKey) -> TPLayoutQualification | None:
        if not isinstance(key, TPQualificationKey):
            raise TypeError("qualification lookup key is invalid")
        with self._lock:
            if self.load_error is not None:
                return None
            record = self._records.get(key.qualification_id)
            if record is None or record.key != key:
                return None
            if not record.promotable or not record.exact:
                return None
            return record

    def decision(self, key: TPQualificationKey) -> dict[str, Any]:
        record = self.lookup(key)
        if record is not None:
            return {
                "matched": True,
                "source": "persistent",
                "qualification_id": record.qualification_id,
                "record_digest": record.record_digest,
                "shard_weights": list(record.shard_weights),
                "reason": record.reason,
            }
        with self._lock:
            rejected = self._records.get(key.qualification_id)
            if rejected is not None and rejected.key == key:
                return {
                    "matched": False,
                    "source": "rejected_evidence",
                    "qualification_id": rejected.qualification_id,
                    "record_digest": rejected.record_digest,
                    "shard_weights": list(rejected.shard_weights),
                    "exact": rejected.exact,
                    "promotable": rejected.promotable,
                    "reason": rejected.reason,
                }
        return {
            "matched": False,
            "source": "equal_fallback",
            "qualification_id": key.qualification_id,
            "reason": (
                f"qualification store unavailable: {self.load_error}"
                if self.load_error is not None
                else "no exact promotable TP layout qualification matched"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": _SCHEMA_VERSION,
                "qualifications": [
                    record.to_dict() for _, record in sorted(self._records.items())
                ],
                "load_error": self.load_error,
            }


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _aggregate_artifacts(paths: Sequence[Path], *, root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        relative = str(path.relative_to(root)).encode()
        digest = _sha256_file(path).encode()
        hasher.update(len(relative).to_bytes(4, "big"))
        hasher.update(relative)
        hasher.update(digest)
    return hasher.hexdigest() if paths else ""


@lru_cache(maxsize=1)
def local_runtime_artifact_identifiers() -> dict[str, str]:
    """Hash the installed JACCL library and oMLX native kernel bundle once."""

    jaccl = ""
    try:
        import mlx

        roots = [Path(item).resolve() for item in getattr(mlx, "__path__", ())]
        if getattr(mlx, "__file__", None):
            roots.append(Path(mlx.__file__).resolve().parent)
        candidates = tuple(
            candidate
            for mlx_root in roots
            for candidate in (
                mlx_root / "lib" / "libjaccl.dylib",
                mlx_root.parent / "lib" / "libjaccl.dylib",
            )
        )
        jaccl_path = next((path for path in candidates if path.is_file()), None)
        if jaccl_path is not None:
            jaccl = _sha256_file(jaccl_path)
    except (ImportError, OSError, TypeError, ValueError):
        pass

    package_root = Path(__file__).resolve().parents[1]
    kernel_root = package_root / "custom_kernels"
    kernel_paths: list[Path] = []
    if kernel_root.is_dir():
        for suffix in ("*.so", "*.dylib", "*.metallib"):
            kernel_paths.extend(path for path in kernel_root.rglob(suffix) if path.is_file())
    try:
        kernels = _aggregate_artifacts(kernel_paths, root=package_root)
    except OSError:
        kernels = ""
    return {"jaccl_identifier": jaccl, "kernel_identifier": kernels}


def node_fingerprints_from_statuses(
    node_ids: Sequence[str],
    statuses: Sequence[dict[str, Any]],
    *,
    backend: str,
) -> tuple[TPNodeFingerprint, ...]:
    """Build exact ordered fingerprints; raise when any evidence is missing."""

    if backend not in _BACKENDS:
        raise ValueError("qualification backend is invalid")
    if len(node_ids) < 2 or len(node_ids) != len(statuses):
        raise ValueError("qualification status count does not match node order")
    result: list[TPNodeFingerprint] = []
    for node_id, wrapper in zip(node_ids, statuses):
        if not isinstance(wrapper, dict):
            raise ValueError(f"qualification status is unavailable for {node_id}")
        if wrapper.get("runtime_compatible") is False:
            raise ValueError(f"qualification runtime is incompatible for {node_id}")
        status = wrapper.get("status", wrapper)
        if not isinstance(status, dict):
            raise ValueError(f"qualification status is malformed for {node_id}")
        node = status.get("node")
        runtime = status.get("runtime")
        transport = status.get("transport")
        if not all(isinstance(value, dict) for value in (node, runtime, transport)):
            raise ValueError(f"qualification status is incomplete for {node_id}")
        rdma = transport.get("rdma")
        thunderbolt = transport.get("thunderbolt")
        if not isinstance(rdma, dict) or not isinstance(thunderbolt, dict):
            raise ValueError(f"qualification fabric evidence is incomplete for {node_id}")
        if backend.startswith("jaccl") and not (
            rdma.get("enabled") is True and rdma.get("control_status") == "enabled"
        ):
            raise ValueError(f"qualification JACCL fabric is not ready for {node_id}")
        stable_fabric = {
            "backend": backend,
            "fabric_kind": node.get("fabric_kind") or (
                "thunderbolt-rdma" if backend.startswith("jaccl") else "tcp"
            ),
            "rdma_enabled": rdma.get("enabled") is True,
            "rdma_device_count": len(rdma.get("devices") or ()),
            "thunderbolt_peer_connected": thunderbolt.get("peer_connected") is True,
            "thunderbolt_speeds": sorted(
                str(item.get("speed"))
                for item in (thunderbolt.get("ports") or ())
                if isinstance(item, dict) and item.get("peer_connected") is True
                and item.get("speed")
            ),
        }
        jaccl_identifier = runtime.get("jaccl_identifier")
        if backend == "ring" and not jaccl_identifier:
            # JACCL does not execute on this topology; keep an exact canonical
            # value rather than treating an irrelevant library as evidence.
            jaccl_identifier = hashlib.sha256(b"not-applicable:ring").hexdigest()
        result.append(
            TPNodeFingerprint(
                node_id=str(node_id),
                chip_name=node.get("chip_name"),
                physical_memory_bytes=node.get("physical_memory_bytes"),
                accelerator=node.get("accelerator"),
                fabric_identifier=_canonical_digest(stable_fabric),
                omlx_version=runtime.get("omlx_version"),
                mlx_version=runtime.get("mlx_version"),
                mlx_lm_version=runtime.get("mlx_lm_version"),
                python_version=runtime.get("python_version"),
                os_name=runtime.get("os_name"),
                os_version=runtime.get("os_version"),
                jaccl_identifier=jaccl_identifier,
                kernel_identifier=runtime.get("kernel_identifier"),
            )
        )
    return tuple(result)


_configured_store: TPLayoutQualificationStore | None = None
_configured_lock = threading.Lock()


def configure_tp_layout_qualification_store(
    base_path: Path,
) -> TPLayoutQualificationStore:
    global _configured_store
    with _configured_lock:
        _configured_store = TPLayoutQualificationStore(base_path)
        return _configured_store


def get_tp_layout_qualification_store() -> TPLayoutQualificationStore:
    with _configured_lock:
        if _configured_store is None:
            raise RuntimeError("TP layout qualification store is not configured")
        return _configured_store


__all__ = [
    "TPLayoutQualification",
    "TPLayoutQualificationStore",
    "TPNodeFingerprint",
    "TPQualificationKey",
    "TPQualificationProvenance",
    "TPRateEvidence",
    "configure_tp_layout_qualification_store",
    "get_tp_layout_qualification_store",
    "local_runtime_artifact_identifiers",
    "node_fingerprints_from_statuses",
]
