# SPDX-License-Identifier: Apache-2.0
"""Validated, serializable configuration for one distributed inference job."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .performance import (
    ExecutionSettings,
    NodePerformanceProfile,
    execution_profile,
)
from .planner import (
    PipelineAssignment,
    normalize_memory_guard_tier,
    normalize_node_role,
)

DistributedBackend = Literal["ring", "jaccl", "jaccl-ring"]

_SSH_TARGET = re.compile(
    r"^(?:[A-Za-z0-9._-]+@)?(?:[A-Za-z0-9._-]+|\[[0-9A-Fa-f:]+\])$"
)
_RDMA_DEVICE = re.compile(r"^rdma_[A-Za-z0-9_.-]+$")
_NODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_MODEL_ID_BYTES = 16 * 1024
_MAX_PLAN_BYTES = 256 * 1024

RDMAPath = str | tuple[str, ...] | None


def validate_ssh_target(value: str) -> str:
    """Validate an SSH destination without accepting options or shell syntax."""

    value = value.strip()
    if (
        not value
        or len(value) > 255
        or value.startswith("-")
        or _SSH_TARGET.fullmatch(value) is None
    ):
        raise ValueError(f"invalid SSH target: {value!r}")
    return value


def _validate_ip(value: str) -> str:
    value = value.strip()
    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"invalid communication IP: {value!r}") from exc
    return value


def _validate_rdma_path(value: Any) -> RDMAPath:
    if value is None:
        return None
    if isinstance(value, str):
        if _RDMA_DEVICE.fullmatch(value) is None:
            raise ValueError(f"invalid RDMA device: {value!r}")
        return value
    if isinstance(value, (list, tuple)):
        paths = tuple(value)
        if not paths:
            raise ValueError("an RDMA path list cannot be empty")
        for path in paths:
            if not isinstance(path, str) or _RDMA_DEVICE.fullmatch(path) is None:
                raise ValueError(f"invalid RDMA device: {path!r}")
        return paths
    raise ValueError("RDMA entries must be a device, a device list, or null")


@dataclass(frozen=True)
class ClusterHost:
    """One rank in an MLX hostfile."""

    node_id: str
    ssh: str
    ips: tuple[str, ...]
    rdma: tuple[RDMAPath, ...] = ()
    python_executable: str | None = None

    def __post_init__(self) -> None:
        if _NODE_ID.fullmatch(self.node_id) is None:
            raise ValueError(f"invalid node ID: {self.node_id!r}")
        object.__setattr__(self, "ssh", validate_ssh_target(self.ssh))
        object.__setattr__(self, "ips", tuple(_validate_ip(ip) for ip in self.ips))
        object.__setattr__(
            self,
            "rdma",
            tuple(_validate_rdma_path(path) for path in self.rdma),
        )
        if self.python_executable is not None:
            executable = self.python_executable.strip()
            path = Path(executable)
            if not path.is_absolute() or "\x00" in executable or len(executable) > 4096:
                raise ValueError("cluster host Python executable must be absolute")
            object.__setattr__(self, "python_executable", executable)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "node_id": self.node_id,
            "ssh": self.ssh,
            "ips": list(self.ips),
            "rdma": [
                list(path) if isinstance(path, tuple) else path for path in self.rdma
            ],
        }
        if self.python_executable:
            payload["python_executable"] = self.python_executable
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClusterHost:
        if not isinstance(payload, dict):
            raise ValueError("cluster host must be an object")
        node_id = payload.get("node_id")
        ssh = payload.get("ssh")
        ips = payload.get("ips", [])
        rdma = payload.get("rdma", [])
        if not isinstance(node_id, str) or not isinstance(ssh, str):
            raise ValueError("cluster host requires string node_id and ssh fields")
        if not isinstance(ips, list) or not isinstance(rdma, list):
            raise ValueError("cluster host ips and rdma fields must be arrays")
        python_executable = payload.get("python_executable")
        if python_executable is not None and not isinstance(python_executable, str):
            raise ValueError("cluster host Python executable must be a string")
        return cls(
            node_id=node_id,
            ssh=ssh,
            ips=tuple(ips),
            rdma=tuple(rdma),
            python_executable=python_executable,
        )


def _assignment_from_dict(payload: dict[str, Any]) -> PipelineAssignment:
    if not isinstance(payload, dict):
        raise ValueError("pipeline assignment must be an object")
    required = (
        "node_id",
        "rank",
        "start_layer",
        "end_layer",
        "layer_weight_bytes",
        "fixed_weight_bytes",
        "reserve_bytes",
        "capacity_bytes",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"pipeline assignment is missing {', '.join(missing)}")
    # Decoded before the constructor so the reason survives: everything inside
    # the try below is reported as "contains an invalid value", and a rank that
    # refuses to launch over a role deserves to say which role.
    role = normalize_node_role(payload.get("role", ""))
    memory_guard_tier = normalize_memory_guard_tier(
        payload.get("memory_guard_tier", "balanced")
    )
    try:
        predicted = {}
        for key in (
            "predicted_compute_seconds",
            "predicted_send_seconds",
            "predicted_stage_seconds",
        ):
            value = payload.get(key)
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{key} must be finite and non-negative")
            predicted[key] = value
        assignment = PipelineAssignment(
            node_id=str(payload["node_id"]),
            rank=int(payload["rank"]),
            start_layer=int(payload["start_layer"]),
            end_layer=int(payload["end_layer"]),
            layer_weight_bytes=int(payload["layer_weight_bytes"]),
            fixed_weight_bytes=int(payload["fixed_weight_bytes"]),
            coordinator_weight_bytes=int(
                payload.get("coordinator_weight_bytes", 0)
            ),
            reserve_bytes=int(payload["reserve_bytes"]),
            capacity_bytes=int(payload["capacity_bytes"]),
            manual_memory_limit=bool(payload.get("manual_memory_limit", False)),
            role=role,
            memory_guard_tier=memory_guard_tier,
            tensor_parallel_rank=int(payload.get("tensor_parallel_rank", 0)),
            tensor_parallel_size=int(payload.get("tensor_parallel_size", 1)),
            sharded_weight_bytes=int(payload.get("sharded_weight_bytes", 0)),
            # ``to_dict`` has always emitted these three; nothing read them
            # back, so every decoded assignment claimed a 0-byte KV cache.
            # ``planned_weight_bytes`` includes the cache, and that is the
            # number the rank's memory guard is charged and the engine pool
            # reserves against — dropping it under-charged both by the whole
            # cache. A 40 GiB-weights + 20 GiB-KV stage was admitted as 40.
            kv_cache_bytes=int(payload.get("kv_cache_bytes", 0)),
            kv_bytes_per_token=int(payload.get("kv_bytes_per_token", 0)),
            max_context_tokens=int(payload.get("max_context_tokens", 0)),
            **predicted,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("pipeline assignment contains an invalid value") from exc
    if assignment.rank < 0:
        raise ValueError("pipeline assignment rank must be non-negative")
    if not 0 <= assignment.start_layer < assignment.end_layer:
        raise ValueError("pipeline assignment must contain at least one layer")
    if (
        min(
            assignment.layer_weight_bytes,
            assignment.fixed_weight_bytes,
            assignment.coordinator_weight_bytes,
            assignment.reserve_bytes,
            assignment.capacity_bytes,
            assignment.kv_cache_bytes,
            assignment.kv_bytes_per_token,
            assignment.max_context_tokens,
        )
        < 0
    ):
        raise ValueError("pipeline assignment byte counts must be non-negative")
    if assignment.capacity_bytes <= assignment.reserve_bytes:
        raise ValueError("pipeline assignment reserve must be below capacity")
    if assignment.headroom_bytes < 0:
        raise ValueError("pipeline assignment exceeds node capacity")
    return assignment


@dataclass(frozen=True)
class ClusterDeployment:
    """Immutable input used by the engine and the MLX launcher."""

    deployment_id: str
    model: str
    backend: DistributedBackend
    hosts: tuple[ClusterHost, ...]
    assignments: tuple[PipelineAssignment, ...]
    plan_hash: str
    trust_remote_code: bool = False
    execution: ExecutionSettings = field(
        default_factory=lambda: execution_profile("balanced")
    )
    performance_profiles: tuple[NodePerformanceProfile, ...] = ()
    tensor_parallel_size: int = 1
    target_context_tokens: int = 8192

    def __post_init__(self) -> None:
        if _NODE_ID.fullmatch(self.deployment_id) is None:
            raise ValueError(f"invalid deployment ID: {self.deployment_id!r}")
        if (
            not isinstance(self.model, str)
            or not self.model.strip()
            or len(self.model.encode()) > _MAX_MODEL_ID_BYTES
            or "\x00" in self.model
        ):
            raise ValueError("model must be a non-empty path or repository ID")
        if self.backend not in {"ring", "jaccl", "jaccl-ring"}:
            raise ValueError(f"unsupported distributed backend: {self.backend!r}")
        if not 2 <= len(self.hosts) <= 64:
            raise ValueError("distributed inference requires between 2 and 64 hosts")
        if not 1 <= self.tensor_parallel_size <= len(self.hosts):
            raise ValueError(
                "tensor_parallel_size must be between 1 and the host count"
            )
        if len(self.hosts) % self.tensor_parallel_size != 0:
            raise ValueError(
                "host count must be divisible by tensor_parallel_size"
            )
        if (
            not isinstance(self.target_context_tokens, int)
            or isinstance(self.target_context_tokens, bool)
            or not 1 <= self.target_context_tokens <= 1_048_576
        ):
            raise ValueError(
                "target_context_tokens must be between 1 and 1,048,576"
            )
        if len(self.assignments) != len(self.hosts):
            raise ValueError("host count must match pipeline assignment count")
        if self.hosts[0].ssh != "127.0.0.1":
            raise ValueError("rank 0 must use SSH target 127.0.0.1")
        if len({host.node_id for host in self.hosts}) != len(self.hosts):
            raise ValueError("cluster node IDs must be unique")
        if len(self.plan_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.plan_hash
        ):
            raise ValueError("plan_hash must be a lowercase SHA-256 digest")

        assignments = sorted(self.assignments, key=lambda item: item.rank)
        if [item.rank for item in assignments] != list(range(len(self.hosts))):
            raise ValueError("pipeline ranks must be contiguous from zero")
        for rank, (host, assignment) in enumerate(zip(self.hosts, assignments)):
            if host.node_id != assignment.node_id or assignment.rank != rank:
                raise ValueError("host order must match node IDs and pipeline ranks")
        if self.performance_profiles:
            if len(self.performance_profiles) != len(self.hosts):
                raise ValueError(
                    "performance profile count must match cluster host count"
                )
            for rank, (host, profile) in enumerate(
                zip(self.hosts, self.performance_profiles)
            ):
                if (
                    profile.rank != rank
                    or profile.node_id != host.node_id
                    or profile.backend != self.backend
                ):
                    raise ValueError(
                        "performance profiles must match host rank, ID, and backend"
                    )

        if self.backend == "ring":
            if any(not host.ips for host in self.hosts):
                raise ValueError("ring hosts require at least one communication IP")
        else:
            size = len(self.hosts)
            for rank, host in enumerate(self.hosts):
                if not host.ips:
                    raise ValueError("JACCL hosts require a communication IP")
                if len(host.rdma) != size:
                    raise ValueError("JACCL requires a full RDMA connectivity matrix")
                if host.rdma[rank] is not None:
                    raise ValueError("JACCL RDMA matrix diagonal must be null")
                if any(
                    path is None
                    for index, path in enumerate(host.rdma)
                    if index != rank
                ):
                    raise ValueError("JACCL RDMA matrix is missing a peer path")

    @property
    def world_size(self) -> int:
        return len(self.hosts)

    @property
    def distributed_init_backend(self) -> str:
        return "jaccl" if self.backend.startswith("jaccl") else "ring"

    def hostfile_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "envs": ["MLX_METAL_FAST_SYNCH=1"],
            "hosts": [
                {
                    "ssh": host.ssh,
                    "ips": list(host.ips),
                    "rdma": [
                        list(path) if isinstance(path, tuple) else path
                        for path in host.rdma
                    ],
                }
                for host in self.hosts
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "deployment_id": self.deployment_id,
            "model": self.model,
            "backend": self.backend,
            "hosts": [host.to_dict() for host in self.hosts],
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "plan_hash": self.plan_hash,
            "trust_remote_code": self.trust_remote_code,
            "execution": self.execution.to_dict(),
            "performance_profiles": [
                profile.to_dict() for profile in self.performance_profiles
            ],
            "tensor_parallel_size": self.tensor_parallel_size,
            "target_context_tokens": self.target_context_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClusterDeployment:
        if not isinstance(payload, dict) or payload.get("schema_version", 1) != 1:
            raise ValueError("unsupported cluster deployment schema")
        hosts = payload.get("hosts")
        assignments = payload.get("assignments")
        performance_profiles = payload.get("performance_profiles", [])
        if (
            not isinstance(hosts, list)
            or not isinstance(assignments, list)
            or not isinstance(performance_profiles, list)
        ):
            raise ValueError("deployment hosts and assignments must be arrays")
        deployment_id = payload.get("deployment_id")
        model = payload.get("model")
        backend = payload.get("backend")
        plan_hash = payload.get("plan_hash")
        if not all(
            isinstance(value, str)
            for value in (deployment_id, model, backend, plan_hash)
        ):
            raise ValueError("deployment identity fields must be strings")
        return cls(
            deployment_id=deployment_id,
            model=model,
            backend=backend,
            hosts=tuple(ClusterHost.from_dict(host) for host in hosts),
            assignments=tuple(
                _assignment_from_dict(assignment) for assignment in assignments
            ),
            plan_hash=plan_hash,
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
            execution=ExecutionSettings.from_dict(payload.get("execution")),
            performance_profiles=tuple(
                NodePerformanceProfile.from_dict(profile)
                for profile in performance_profiles
            ),
            tensor_parallel_size=int(payload.get("tensor_parallel_size", 1)),
            target_context_tokens=int(payload.get("target_context_tokens", 8192)),
        )

    def encode_worker_plan(self) -> str:
        """Encode the small trusted plan as a bounded command-line argument."""

        raw = json.dumps(
            {
                "schema_version": 1,
                "plan_hash": self.plan_hash,
                "assignments": [
                    assignment.to_dict() for assignment in self.assignments
                ],
                "performance_profiles": [
                    profile.to_dict() for profile in self.performance_profiles
                ],
                "tensor_parallel_size": self.tensor_parallel_size,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if len(raw) > _MAX_PLAN_BYTES:
            raise ValueError("pipeline plan is too large")
        return base64.urlsafe_b64encode(zlib.compress(raw, level=9)).decode()


def decode_worker_contract(
    encoded: str,
) -> tuple[
    str,
    tuple[PipelineAssignment, ...],
    tuple[NodePerformanceProfile, ...],
    int,
]:
    """Decode and validate the full worker contract without accepting code."""

    if not isinstance(encoded, str) or len(encoded) > _MAX_PLAN_BYTES * 2:
        raise ValueError("encoded pipeline plan is too large")
    try:
        compressed = base64.b64decode(encoded, altchars=b"-_", validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, _MAX_PLAN_BYTES + 1)
        if decompressor.unconsumed_tail:
            raise ValueError("decoded pipeline plan is too large")
        raw += decompressor.flush(max(1, _MAX_PLAN_BYTES + 1 - len(raw)))
    except (binascii.Error, zlib.error) as exc:
        raise ValueError("encoded pipeline plan is invalid") from exc
    if (
        len(raw) > _MAX_PLAN_BYTES
        or decompressor.unconsumed_tail
        or decompressor.unused_data
    ):
        raise ValueError("decoded pipeline plan is too large or malformed")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pipeline plan is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported pipeline plan schema")
    plan_hash = payload.get("plan_hash")
    assignments = payload.get("assignments")
    performance_profiles = payload.get("performance_profiles", [])
    if (
        not isinstance(plan_hash, str)
        or not isinstance(assignments, list)
        or not isinstance(performance_profiles, list)
    ):
        raise ValueError("pipeline plan is missing required fields")
    if len(plan_hash) != 64 or any(
        char not in "0123456789abcdef" for char in plan_hash
    ):
        raise ValueError("pipeline plan hash is invalid")
    parsed = tuple(_assignment_from_dict(item) for item in assignments)
    if [item.rank for item in sorted(parsed, key=lambda item: item.rank)] != list(
        range(len(parsed))
    ):
        raise ValueError("pipeline plan ranks must be contiguous from zero")
    profiles = tuple(
        NodePerformanceProfile.from_dict(item) for item in performance_profiles
    )
    if profiles and (
        len(profiles) != len(parsed)
        or [item.rank for item in profiles] != list(range(len(parsed)))
        or any(profile.node_id != parsed[profile.rank].node_id for profile in profiles)
    ):
        raise ValueError("worker performance profiles do not match the shard plan")
    tensor_parallel_size = int(payload.get("tensor_parallel_size", 1))
    if not 1 <= tensor_parallel_size <= len(parsed):
        raise ValueError("tensor_parallel_size must be between 1 and the assignment count")
    return plan_hash, parsed, profiles, tensor_parallel_size


def decode_worker_plan(encoded: str) -> tuple[str, tuple[PipelineAssignment, ...]]:
    """Backward-compatible assignment-only worker plan decoder."""

    plan_hash, assignments, _, _ = decode_worker_contract(encoded)
    return plan_hash, assignments
