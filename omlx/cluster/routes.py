# SPDX-License-Identifier: Apache-2.0
"""Authenticated admin API routes for local cluster diagnostics."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence

from fastapi import APIRouter, Header, HTTPException, Query, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import ModelBusyError, ModelNotFoundError
from .autoconfigure import (
    STRATEGIES,
    build_rdma_matrix,
    choose_backend,
    choose_parallelism,
    describe_preflight,
    describe_transports,
    order_hosts_for_topology,
    peer_import_issues,
    preflight_issues,
    tp_groups_spanning_slow_links,
)
from .backends import (
    MemberFabric,
    members_from_host_records,
    select_cluster_backend,
)
from .catalogue import ModelFit, assess_model, catalogue_for_cluster
from .collective import (
    CollectiveSmokeError,
    run_local_collective_smoke,
    run_local_pipeline_smoke,
)
from .deployment import (
    ClusterDeployment,
    ClusterHost,
    validate_model_path_map,
    validate_ssh_target,
)
from .discovery import (
    discover_all_peers,
    record_peer_transports,
    verify_pairing_token,
)
from .enrollment import EnrolledNode, EnrollmentError, get_cluster_enrollment
from .guidance import explain
from .incidents import Severity, get_cluster_incidents
from .launch import (
    CudaFabricProbeHost,
    DistributedLaunchError,
    DistributedTeardownError,
    preflight_remote_hosts,
    probe_remote_admission_ceiling,
    probe_remote_host,
    resolve_remote_python,
    run_cluster_performance_probe,
    run_cuda_fabric_probe,
    stop_deployment_processes,
)
from .liveness import (
    PeerLostError,
    check_peers,
    describe_failure,
    raise_if_peer_lost,
)
from .model_inventory import (
    engine_pool_model_inventory,
    merge_model_inventories,
    remote_model_inventory,
)
from .performance import (
    DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES,
    DeepseekAnePrefillSettings,
    NodePerformanceProfile,
    execution_profile,
    tune_execution_settings,
)
from .parallel_groups import hybrid_group_split_supported
from .disaggregated import build_full_replica_shard_plan
from .planner import (
    LOCAL_NODE,
    NodeBudget,
    PlanningError,
    ShardPlan,
    complete_model_layout,
    plan_hybrid,
    plan_proportional_pipeline,
    plan_unequal_pipeline,
    recommend_tensor_shard_weights,
    remote_model_layout,
    synthetic_model_layout,
)
from .probe import collect_cluster_status, detect_low_power_mode
from .registry import get_cluster_registry, get_device_registry
from .identity import get_node_identity
from .replan import (
    hosts_from_deployment,
    nodes_from_deployment,
    placement_view,
    summarize_deployment,
)
from .runtime import read_runtime_markers
from .staging import (
    InsufficientDiskError,
    home_relative_model_path,
    index_shards,
    model_identity_digest,
    model_staging_inventory,
    plan_staging,
    remote_file_sizes,
    remote_model_dir,
    remote_model_staging_inventory,
    stage_files_from_source,
    stage_manifest,
)
from .strategy_benchmarks import context_bucket, get_strategy_benchmark_store
from .tp_qualifications import (
    TPLayoutQualification,
    TPQualificationKey,
    TPQualificationProvenance,
    TPRateEvidence,
    get_tp_layout_qualification_store,
    node_fingerprints_from_statuses,
)
from .supervisor import run_worker_smoke
from .transport import (
    LinkAuthorizationCancelledError,
    LinkSetupError,
    assess_link,
    configure_link,
    detect_cluster_transports,
    probe_host_interfaces,
    resolve_link_addresses,
    verify_link_reachability,
)
from .worker_bundle import (
    build_cuda_join_command,
    cuda_bootstrap_program,
    worker_source_bundle,
    worker_source_digest,
)

router = APIRouter(prefix="/admin/api/cluster", tags=["cluster"])
join_router = APIRouter(prefix="/cluster/join", tags=["cluster-enrollment"])
logger = logging.getLogger(__name__)

_get_engine_pool: Any | None = None


class ClusterPairingTokenRequest(BaseModel):
    shared_secret: str = Field(min_length=16, max_length=256)


class ClusterPairingTokenVerificationRequest(ClusterPairingTokenRequest):
    token: str = Field(min_length=1, max_length=16 * 1024)


class ClusterKeyExchangeTokenRequest(ClusterPairingTokenRequest):
    node_id: str = Field(min_length=1, max_length=255)


class ClusterKeyExchangeRequest(ClusterPairingTokenRequest):
    exchange_token: str = Field(min_length=1, max_length=64 * 1024)


class ClusterJoinKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    controller_ip: str = Field(min_length=2, max_length=64)
    controller_port: int = Field(ge=1, le=65535)
    scheme: Literal["http", "https"] = "http"
    ttl_seconds: int = Field(default=1800, ge=30, le=1800)


class ClusterWorkerClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,255}$")
    hostname: str = Field(pattern=r"^[A-Za-z0-9._-]{1,255}$")
    ssh_user: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
    ssh_port: int = Field(default=22, ge=1, le=65535)
    addresses: list[str] = Field(min_length=1, max_length=8)
    accelerator: Literal["cuda"]
    platform: str = Field(min_length=1, max_length=255)


class ClusterWorkerCompleteRequest(ClusterWorkerClaimRequest):
    python_executable: str = Field(
        pattern=r"^/[A-Za-z0-9._/+:-]{1,1023}$",
    )
    source_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    ssh_host_public_key: str = Field(min_length=32, max_length=8192)
    ssh_host_fingerprint: str = Field(
        pattern=r"^SHA256:[A-Za-z0-9+/]{40,64}$"
    )
    runtime: dict[str, str] = Field(default_factory=dict, max_length=16)


def _join_bearer(authorization: str | None) -> str:
    prefix = "Bearer "
    if (
        not isinstance(authorization, str)
        or not authorization.startswith(prefix)
        or not 32 <= len(authorization.removeprefix(prefix)) <= 512
    ):
        raise HTTPException(status_code=401, detail="invalid enrollment credential")
    token = authorization.removeprefix(prefix)
    if token != token.strip() or any(char.isspace() for char in token):
        raise HTTPException(status_code=401, detail="invalid enrollment credential")
    return token


def _join_addresses(values: list[str]) -> tuple[str, ...]:
    addresses: list[str] = []
    for value in values:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid worker IP address") from exc
        if address.is_unspecified or address.is_multicast or address.is_loopback:
            raise HTTPException(status_code=400, detail="worker IP must be LAN-reachable")
        if address.version != 4 or not (address.is_private or address.is_link_local):
            raise HTTPException(
                status_code=400,
                detail="worker must use a private or link-local IPv4 address",
            )
        normalized = str(address)
        if normalized not in addresses:
            addresses.append(normalized)
    return tuple(addresses)


def _controller_url(request: ClusterJoinKeyRequest) -> str:
    try:
        address = ipaddress.ip_address(request.controller_ip)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="controller must be a literal IP") from exc
    if (
        address.version != 4
        or address.is_unspecified
        or address.is_multicast
        or address.is_loopback
        or not (address.is_private or address.is_link_local)
    ):
        raise HTTPException(status_code=400, detail="controller must use a local IP")
    return f"{request.scheme}://{address}:{request.controller_port}"


def _validated_ssh_targets(hosts: str) -> list[str]:
    """Parse and validate comma-separated SSH destinations from a query."""

    try:
        return [
            validate_ssh_target(item.strip())
            for item in hosts.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def inspect_safetensors_layout(model_path: str | Path):
    """Compatibility seam for route tests, backed by the complete-model check.

    Older callers patched this route-local name.  Keeping the seam avoids
    coupling them to planner internals while ensuring production planning
    refuses a directory containing only one rank's previous stage.
    """

    return complete_model_layout(model_path)


def set_cluster_getters(engine_pool_getter: Any) -> None:
    """Inject server-owned dependencies without importing ``omlx.server``."""

    global _get_engine_pool
    _get_engine_pool = engine_pool_getter


def _engine_pool() -> Any:
    if _get_engine_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Cluster activation is unavailable until the server is initialized",
        )
    return _get_engine_pool()


def _package_version_or_none(name: str) -> str:
    """Installed version of a package, or an empty string when unknown."""

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return ""


_DIAGNOSTIC_SECRET_FIELDS = {
    "api_key",
    "authorization",
    "join_key",
    "pairing_token",
    "password",
    "private_key",
    "secret",
    "token",
}


def _redact_diagnostic(value: Any, *, field: str = "", depth: int = 0) -> Any:
    """Bound and redact a diagnostic tree before it can leave the admin API."""

    if depth > 12:
        return "<maximum depth reached>"
    normalized_field = field.lower().replace("-", "_")
    if (
        normalized_field in _DIAGNOSTIC_SECRET_FIELDS
        or normalized_field.endswith("_password")
        or normalized_field.endswith("_token")
        or normalized_field.endswith("_private_key")
    ):
        return "<redacted>"
    if isinstance(value, dict):
        items = list(value.items())
        result = {
            str(key): _redact_diagnostic(item, field=str(key), depth=depth + 1)
            for key, item in items[:512]
        }
        if len(items) > 512:
            result["_truncated_fields"] = len(items) - 512
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _redact_diagnostic(item, field=field, depth=depth + 1)
            for item in value[:256]
        ]
        if len(value) > 256:
            result.append(f"<{len(value) - 256} more items>")
        return result
    if isinstance(value, str):
        text = value.replace(str(Path.home()), "~")
        # SSH targets occasionally include the local account name. The host is
        # useful evidence; the username is not.
        text = re.sub(
            r"(?<![\w.-])[\w.-]+@(?=[A-Za-z0-9_.-]+(?:\s|$|[:,/]))",
            "<user>@",
            text,
        )
        return text if len(text) <= 4096 else text[:4096] + "…<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:4096]


def _record_cluster_incident(
    severity: Severity,
    state_code: str,
    message: str,
    *,
    source: str = "coordinator",
    job_id: str | None = None,
    deployment_id: str | None = None,
) -> None:
    """Best-effort incident funnel for the routes layer.

    Failure paths call this immediately before re-raising, so nothing here may
    mask the original error: an unconfigured store (worker-only installs,
    bare test apps) or a failed save is swallowed. The message is redacted at
    record time so the stored copy is already safe to serve.
    """

    try:
        store = get_cluster_incidents()
    except RuntimeError:
        return
    redacted = str(_redact_diagnostic(str(message)))
    try:
        store.record(
            severity,
            source,
            state_code,
            redacted,
            guidance_code=explain(redacted).code,
            job_id=job_id,
            deployment_id=deployment_id,
        )
    except Exception:  # noqa: BLE001 - logging must never outrank the failure
        return


class ClusterPlanNodeRequest(BaseModel):
    """One rank-ordered memory budget supplied by the admin UI."""

    node_id: str = Field(min_length=1, max_length=255)
    capacity_bytes: int = Field(gt=0)
    reserve_bytes: int = Field(default=0, ge=0)
    # The memory slider is an explicit operator ceiling. Roles choose the
    # automatic reserve, but must not silently replace a value the page is
    # displaying as authoritative.
    manual_memory_limit: bool = False
    # The split control: cap the model weight placed here, leaving the rest of
    # this Mac for KV cache. 0 lets the planner balance.
    max_weight_bytes: int = Field(default=0, ge=0)
    # Soft per-rank preference used by the 2–N node balance controls. The
    # planner keeps layers contiguous and lands on the nearest feasible split.
    target_weight_bytes: int = Field(default=0, ge=0)
    # "headless" or "workstation" — decides how much is held back for the
    # person using this Mac. See omlx/cluster/node_role.py.
    role: str = Field(default="headless", max_length=32)
    memory_guard_tier: Literal["safe", "balanced", "aggressive", "custom"] = (
        "balanced"
    )
    performance: dict[str, Any] | None = None
    accelerator: Literal["metal", "cuda", "cpu"] | None = None
    fabric_kind: str | None = Field(default=None, max_length=64)
    fabric_group_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    )
    fabric_verified: bool = False


class ClusterPlanRequest(BaseModel):
    """Estimated or downloaded model input for unequal-memory planning."""

    model_path: str | None = Field(default=None, max_length=4096)
    # SSH target of the Mac that owns the complete model. Empty/local keeps
    # planning on the coordinator; a peer source is measured on that peer.
    model_source: str | None = Field(default=None, max_length=255)
    # Interpreter reported by the source node's live probe. macOS and Linux
    # installations do not share a filesystem layout, so the coordinator's
    # executable must never be assumed to exist on a remote model owner.
    model_source_python: str | None = Field(default=None, max_length=4096)
    model_size_bytes: int | None = Field(default=None, gt=0)
    layer_count: int = Field(default=80, gt=0, le=2048)
    nodes: list[ClusterPlanNodeRequest] = Field(min_length=1, max_length=64)
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"
    # "balanced" runs the bottleneck-minimizing DP planner; "proportional"
    # splits layers ∝ usable RAM with largest-remainder rounding (exo-style).
    allocation: Literal["balanced", "proportional"] = "balanced"
    pipeline_microbatch_size: int | None = Field(default=None, gt=0, le=256)
    tensor_parallel_size: int = Field(default=1, ge=1, le=64)
    serving_mode: Literal["sharded", "disaggregated"] = "sharded"
    prefill_rank: int | None = Field(default=None, ge=0, le=63)
    decode_rank: int | None = Field(default=None, ge=0, le=63)
    target_context_tokens: int = Field(default=8192, ge=1, le=1_048_576)
    mtp_enabled: bool = False
    mtp_num_draft_tokens: int | None = Field(default=None, ge=1, le=8)
    # Process-lifetime prefix snapshots are a reuse optimization, not a cold
    # prefill requirement. None/False keeps the bounded SSD write-behind tier
    # out of the launch contract; True enables it explicitly on every rank.
    prompt_cache_ssd: bool | None = None
    prompt_cache_ssd_max_bytes: int | None = Field(default=None, gt=0)
    # Cluster v2: optional node_id → absolute model path on that node. Empty
    # keeps the legacy same-absolute-path-on-every-node behavior.
    path_map: dict[str, str] | None = Field(default=None, max_length=64)


class ClusterHostRequest(BaseModel):
    """One rank-ordered SSH and collective transport endpoint."""

    node_id: str = Field(min_length=1, max_length=128)
    ssh: str = Field(min_length=1, max_length=255)
    ips: list[str] = Field(min_length=1, max_length=16)
    rdma: list[str | list[str] | None] = Field(default_factory=list, max_length=64)
    python_executable: str | None = Field(default=None, max_length=4096)


class DeepseekAnePrefillRequest(BaseModel):
    """Explicit per-rank DeepSeek-V4 hybrid prefill settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sequence_length: int = Field(default=4096, ge=1024, le=16384, multiple_of=64)
    tail_padding_min_tokens: int = Field(default=0, ge=0, le=16383)
    down_enabled: bool = True
    down_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    wo_a_enabled: bool = False
    wo_a_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    cpu_enabled: bool = False
    cpu_fraction: float = Field(default=0.125, ge=0.0, lt=0.5)
    cpu_threads: int = Field(default=12, ge=0, le=64)
    cpu_shared_resource: bool = True


def _validate_cluster_hosts(hosts: list[ClusterHostRequest]) -> None:
    """Validate a hostfile before any request can use its SSH destinations."""

    try:
        for host in hosts:
            ClusterHost(
                node_id=host.node_id.strip(),
                ssh=host.ssh,
                ips=tuple(host.ips),
                rdma=tuple(host.rdma),
                python_executable=host.python_executable,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ClusterDeploymentRequest(BaseModel):
    """User-approved activation request; the server recomputes the shard plan."""

    deployment_id: str | None = Field(default=None, max_length=128)
    model_path: str = Field(min_length=1, max_length=4096)
    model_source: str | None = Field(default=None, max_length=255)
    model_source_python: str | None = Field(default=None, max_length=4096)
    backend: Literal["ring", "jaccl", "jaccl-ring"]
    nodes: list[ClusterPlanNodeRequest] = Field(min_length=2, max_length=64)
    hosts: list[ClusterHostRequest] = Field(min_length=2, max_length=64)
    preflight: Literal[True] = True
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"
    allocation: Literal["balanced", "proportional"] = "balanced"
    auto_tune: bool = True
    sampling_rank_only: bool = True
    async_overlap: bool = True
    cache_affinity: bool = True
    max_kv_size: int | None = Field(default=None, gt=0)
    ring_connections_per_ip: int | None = Field(default=None, ge=1, le=32)
    tensor_parallel_size: int = Field(default=1, ge=1, le=64)
    serving_mode: Literal["sharded", "disaggregated"] = "sharded"
    prefill_rank: int | None = Field(default=None, ge=0, le=63)
    decode_rank: int | None = Field(default=None, ge=0, le=63)
    target_context_tokens: int = Field(default=8192, ge=1, le=1_048_576)
    mtp_enabled: bool = False
    mtp_num_draft_tokens: int | None = Field(default=None, ge=1, le=8)
    # Server-issued ID of the exact hardware/runtime qualification used by the
    # preview. Activation re-probes and resolves it again; arbitrary vectors
    # never cross the public request schema.
    tp_qualification_id: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    prompt_cache_ssd: bool | None = None
    prompt_cache_ssd_max_bytes: int | None = Field(default=None, gt=0)
    deepseek_ane_prefill: DeepseekAnePrefillRequest | None = None
    # ``placement_signature`` from the /plan response the user was shown. The
    # server refuses to activate anything else, which is the only
    # thing that makes "the plan you approved" a fact rather than a hope:
    # /plan and /deployments used to post different node objects and the second
    # one dropped the role and the split cap without saying so. Activation is a
    # GUI workflow: callers must preview and name the placement they approve.
    approved_placement: str = Field(min_length=16, max_length=64)
    # Cluster v2: node_id → absolute model path on that node. Nodes not listed
    # load ``model_path`` — the pre-v2 shared-path behavior.
    path_map: dict[str, str] | None = Field(default=None, max_length=64)


class ClusterTPRateEvidenceRequest(BaseModel):
    """Aggregated full-model rate evidence for one TP layout."""

    model_config = ConfigDict(extra="forbid")

    prefill_tokens_per_second: float = Field(gt=0.0, le=1e9)
    decode_tokens_per_second: float = Field(gt=0.0, le=1e9)
    samples: int = Field(ge=2, le=10_000)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ClusterTPLayoutQualificationRequest(BaseModel):
    """Persist matched, parity-proven evidence for one asymmetric TP vector."""

    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(min_length=1, max_length=4096)
    backend: Literal["ring", "jaccl", "jaccl-ring"]
    nodes: list[ClusterPlanNodeRequest] = Field(min_length=2, max_length=64)
    hosts: list[ClusterHostRequest] = Field(min_length=2, max_length=64)
    tensor_parallel_size: int = Field(ge=2, le=64)
    target_context_tokens: int = Field(ge=1, le=1_048_576)
    execution_profile: Literal["interactive", "balanced", "throughput"] = (
        "balanced"
    )
    auto_tune: bool = True
    sampling_rank_only: bool = True
    mtp_enabled: bool = False
    mtp_num_draft_tokens: int | None = Field(default=None, ge=1, le=8)
    shard_weights: list[int] = Field(min_length=2, max_length=64)
    equal_control: ClusterTPRateEvidenceRequest
    candidate: ClusterTPRateEvidenceRequest
    reason: str = Field(default="matched full-model qualification", max_length=2000)


class ClusterPeerProbeRequest(BaseModel):
    """Approved SSH peer and optional address used for its return route."""

    ssh: str = Field(min_length=1, max_length=255)
    route_to: str | None = Field(default=None, max_length=64)


class ClusterCudaFabricMemberRequest(BaseModel):
    """One dashboard-selected CUDA worker in a proposed direct-link pair."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=128)
    ssh: str = Field(min_length=1, max_length=255)


class ClusterCudaFabricVerifyRequest(BaseModel):
    """A bounded two-worker NCCL verification started from the dashboard."""

    model_config = ConfigDict(extra="forbid")

    hosts: list[ClusterCudaFabricMemberRequest] = Field(min_length=2, max_length=2)


class ClusterLinkSetupRequest(BaseModel):
    """One pair in the detected fabric; no command text crosses the API."""

    model_config = ConfigDict(extra="forbid")

    hosts: list[str] = Field(min_length=2, max_length=2)


def _reserve_bytes_for(node: ClusterPlanNodeRequest) -> int:
    """Bytes held back on this node.

    The role used to apply only ``if not reserve_bytes``, and the dashboard
    always sends a non-zero reserve — so on the one path a user could reach,
    the Workstation button changed nothing. A role is a statement about the
    machine ("someone is typing on it"), not a default for an empty field: it
    raises an automatic reserve and must never be silenced by a form default.

    Moving the memory slider is different: ``manual_memory_limit`` says the
    person deliberately replaced that automatic policy with the exact limit
    shown in the cockpit. Reapplying the role after that made the GUI promise
    90 GiB while planning with 54 GiB.
    """

    from .node_role import role_for

    if node.manual_memory_limit:
        return node.reserve_bytes
    return max(
        node.reserve_bytes,
        role_for(node.role).reserve_for(node.capacity_bytes),
    )


def _node_budgets(
    nodes: list[ClusterPlanNodeRequest],
    *,
    profiles: tuple[NodePerformanceProfile, ...] = (),
) -> list[NodeBudget]:
    """The rank-ordered budgets every plan in this module is built from.

    One construction site, because there were three and they disagreed. The
    copy the auto-tune re-plan used carried neither ``max_weight_bytes`` nor
    ``role``, so a MacBook the user had capped and marked Workstation was
    re-planned to nearly its whole ceiling after the probe — between the plan
    being approved and the ranks being launched, with nothing shown.

    ``profiles`` are the measured ones from the performance probe, which
    replace anything the caller sent: they are rank-ordered by construction and
    ``NodeBudget`` refuses one that does not match its node.
    """

    budgets: list[NodeBudget] = []
    for rank, node in enumerate(nodes):
        performance = profiles[rank] if rank < len(profiles) else None
        if performance is None and node.performance is not None:
            performance = NodePerformanceProfile.from_dict(node.performance)
        if performance is not None and not performance.promotable:
            performance = None
        reserve_bytes = _reserve_bytes_for(node)
        # A target is a preference, not an admission override. The measured
        # budget or workstation role can legitimately shrink between the
        # slider moving and this request arriving; clamp the preference to the
        # current safe ceiling rather than rejecting an otherwise valid plan.
        target_weight_bytes = min(
            node.target_weight_bytes,
            max(0, node.capacity_bytes - reserve_bytes),
        )
        budgets.append(
            NodeBudget(
                node_id=node.node_id.strip(),
                capacity_bytes=node.capacity_bytes,
                reserve_bytes=reserve_bytes,
                manual_memory_limit=node.manual_memory_limit,
                max_weight_bytes=node.max_weight_bytes,
                target_weight_bytes=target_weight_bytes,
                role=node.role,
                memory_guard_tier=node.memory_guard_tier,
                rank=rank,
                performance=performance,
            )
        )
    return budgets


def _coalesce_verified_cuda_groups(
    host_order: list[str],
    hosts: list[ClusterHostRequest],
    nodes: list[ClusterPlanNodeRequest],
) -> list[str]:
    """Keep a verified two-worker CUDA fabric adjacent in the outer Ring."""

    node_by_id = {node.node_id.strip(): node for node in nodes}
    group_by_ssh: dict[str, str] = {}
    members_by_group: dict[str, list[str]] = {}
    for host in hosts:
        node = node_by_id.get(host.node_id.strip())
        if (
            node is None
            or node.accelerator != "cuda"
            or not node.fabric_verified
            or not node.fabric_group_id
        ):
            continue
        group = node.fabric_group_id
        group_by_ssh[host.ssh] = group
        members_by_group.setdefault(group, []).append(host.ssh)
    eligible = {
        group: set(members)
        for group, members in members_by_group.items()
        if len(set(members)) == 2
    }
    ordered: list[str] = []
    emitted: set[str] = set()
    for ssh in host_order:
        group = group_by_ssh.get(ssh)
        if group not in eligible:
            ordered.append(ssh)
            continue
        if group in emitted:
            continue
        ordered.extend(item for item in host_order if item in eligible[group])
        emitted.add(group)
    return ordered


def _model_and_nodes(request: ClusterPlanRequest):
    """Resolve the request's model layout and rank-ordered node budgets."""

    model_path = request.model_path.strip() if request.model_path else None
    if (model_path is None) == (request.model_size_bytes is None):
        raise PlanningError("provide exactly one of model_path or model_size_bytes")

    if model_path is not None:
        source = (request.model_source or "").strip()
        if source and source not in {
            LOCAL_NODE,
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            model = remote_model_layout(
                validate_ssh_target(source),
                model_path,
                # None lets the peer's own interpreter be discovered over SSH
                # (launch.resolve_remote_python); it is never assumed.
                python_executable=request.model_source_python,
            )
        else:
            # A coordinator may retain only its previous pipeline stage.  Such
            # a directory is not a smaller complete model and must never be
            # used to build the next plan.
            model = inspect_safetensors_layout(model_path)
    else:
        model = synthetic_model_layout(
            total_weight_bytes=request.model_size_bytes,
            layer_count=request.layer_count,
        )
    return model, _node_budgets(request.nodes)


# Exactly the fields the activation screen shows and a person agrees to. Not
# ``plan_hash``: that also covers the model layout and the microbatch size, so
# a tuning step that moves no layer changes it, and a guard built on it would
# refuse launches for a reason the user cannot see.
_PLACEMENT_FIELDS = (
    "rank",
    "node_id",
    "start_layer",
    "end_layer",
    "planned_weight_bytes",
    "kv_cache_bytes",
    "max_context_tokens",
    "reserve_bytes",
    "capacity_bytes",
    "manual_memory_limit",
    "role",
    "memory_guard_tier",
    "tensor_parallel_rank",
    "tensor_parallel_size",
    "tensor_parallel_shard_weight",
)


def _placement_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Who holds what, under which memory constraints, rank-ordered."""

    rows = [
        {field: item.get(field) for field in _PLACEMENT_FIELDS}
        for item in plan.get("assignments", [])
    ]
    rows.sort(key=lambda row: (row.get("rank") or 0, str(row.get("node_id") or "")))
    return rows


def _placement_signature(plan: dict[str, Any]) -> str:
    """Identity of the plan a user approves, stable across cosmetic re-planning."""

    rows: Any = _placement_rows(plan)
    # A per-node path override changes what actually runs, so it is part of
    # what the user approves. Folded in only when present, keeping signatures
    # byte-identical to the legacy format for shared-path deployments.
    path_map = plan.get("path_map")
    if path_map:
        rows = {"rows": rows, "path_map": path_map}
    # Speculative execution changes both the worker graph and the weights that
    # must be resident.  It is therefore part of the launch contract, not a
    # cosmetic generation option.  Keep the legacy signature byte-identical
    # when MTP is absent/default so existing non-MTP approvals remain valid.
    mtp_enabled = bool(plan.get("mtp_enabled", False))
    mtp_num_draft_tokens = plan.get("mtp_num_draft_tokens")
    if mtp_enabled or mtp_num_draft_tokens is not None:
        if isinstance(rows, dict):
            rows = rows | {
                "mtp_enabled": mtp_enabled,
                "mtp_num_draft_tokens": mtp_num_draft_tokens,
            }
        else:
            rows = {
                "rows": rows,
                "mtp_enabled": mtp_enabled,
                "mtp_num_draft_tokens": mtp_num_draft_tokens,
            }
    # False is the latency-safe default and intentionally shares the legacy
    # signature. Enabling distributed SSD snapshot write-behind is a material
    # launch-mode change and must be explicitly approved.
    if bool(plan.get("prompt_cache_ssd", False)):
        cache_contract = {
            "prompt_cache_ssd": True,
            "prompt_cache_ssd_max_bytes": int(
                plan.get("prompt_cache_ssd_max_bytes")
                or DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES
            ),
        }
        if isinstance(rows, dict):
            rows = rows | cache_contract
        else:
            rows = {"rows": rows, **cache_contract}
    qualification = plan.get("tensor_parallel_qualification")
    if qualification is not None:
        if isinstance(rows, dict):
            rows = rows | {"tensor_parallel_qualification": qualification}
        else:
            rows = {
                "rows": rows,
                "tensor_parallel_qualification": qualification,
            }
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _plan_with_signature(plan: dict[str, Any]) -> dict[str, Any]:
    """A plan payload the caller can hand back to prove what it approved."""

    return plan | {"placement_signature": _placement_signature(plan)}


def _gib(value: Any) -> str:
    return f"{int(value or 0) / 1024**3:.1f} GiB"


def _describe_placement(plan: dict[str, Any]) -> str:
    """One sentence per rank, in the terms the activation screen uses."""

    return "; ".join(
        f"{row['node_id']} layers {row['start_layer']}-{row['end_layer']}, "
        f"{_gib(row['planned_weight_bytes'])} planned, "
        f"{_gib(row['reserve_bytes'])} held back "
        f"({row['role'] or 'headless'})"
        for row in _placement_rows(plan)
    )


def _plan_changes(approved: dict[str, Any], launched: dict[str, Any]) -> dict[str, Any]:
    """What moved between the plan that was shown and the plan that will run.

    Automatic tuning may propose a re-plan because it has measurements the
    memory planner did not. This describes that proposal; activation keeps the
    signed placement unless a later preview explicitly approves another one.
    """

    previous = {row["rank"]: row for row in _placement_rows(approved)}
    ranks: list[dict[str, Any]] = []
    for row in _placement_rows(launched):
        before = previous.get(row["rank"])
        if before is None or before == row:
            continue
        weight_delta = int(row["planned_weight_bytes"] or 0) - int(
            before["planned_weight_bytes"] or 0
        )
        ranks.append(
            {
                "rank": row["rank"],
                "node_id": row["node_id"],
                "before": before,
                "after": row,
                "layer_delta": (
                    (int(row["end_layer"] or 0) - int(row["start_layer"] or 0))
                    - (int(before["end_layer"] or 0) - int(before["start_layer"] or 0))
                ),
                "planned_weight_delta_bytes": weight_delta,
                "summary": (
                    f"{row['node_id']} would hold layers "
                    f"{row['start_layer']}-{row['end_layer']} instead of "
                    f"{before['start_layer']}-{before['end_layer']}: "
                    f"{_gib(row['planned_weight_bytes'])} planned, "
                    f"{'up' if weight_delta > 0 else 'down'} from "
                    f"{_gib(before['planned_weight_bytes'])}"
                ),
            }
        )
    settings: dict[str, dict[str, Any]] = {}
    for field, default in (
        ("path_map", {}),
        ("mtp_enabled", False),
        ("mtp_num_draft_tokens", None),
        ("prompt_cache_ssd", False),
        ("prompt_cache_ssd_max_bytes", DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES),
        ("tensor_parallel_qualification", None),
    ):
        before = approved.get(field, default)
        after = launched.get(field, default)
        if before != after:
            settings[field] = {"before": before, "after": after}
    return {
        "changed": bool(ranks or settings),
        "reason": (
            "launch settings changed"
            if settings and not ranks
            else "automatic tuning re-planned from the measured link speeds"
        ),
        "approved_signature": _placement_signature(approved),
        "launched_signature": _placement_signature(launched),
        "ranks": ranks,
        "settings": settings,
    }


_QUALIFIED_TP_SHARD_WEIGHTS_ENV = "OMLX_TP_QUALIFIED_SHARD_WEIGHTS"
_QUALIFIED_TP_MODEL_IDENTITY_ENV = "OMLX_TP_QUALIFIED_MODEL_IDENTITY"
_EXPERIMENTAL_DISTRIBUTED_DSV4_ANE_ENV = (
    "OMLX_CLUSTER_DSV4_ANE_EXPERIMENTAL"
)


class _UnpromotablePerformanceCalibration(ValueError):
    """Synthetic measurements captured under a known throttled power state."""


def _operator_qualified_tp_shard_weights(
    *,
    tensor_parallel_size: int,
    node_count: int,
    model_path: str | Path | None = None,
) -> tuple[tuple[int, ...], ...] | None:
    """Parse the coordinator-only experimental pure-TP shard override.

    The setting is deliberately absent from every request schema: only the
    operator controlling the coordinator process can qualify a vector.  It is
    ignored for pipeline-only planning and rejected for hybrid topology.  The
    planner still validates the model's exact shard-unit sum and memory fit;
    workers never read this environment variable and derive their weights from
    the signed rank assignments.
    """

    raw = os.environ.get(_QUALIFIED_TP_SHARD_WEIGHTS_ENV)
    if raw is None or tensor_parallel_size <= 1:
        return None
    scope = os.environ.get(_QUALIFIED_TP_MODEL_IDENTITY_ENV, "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", scope) is None:
        logger.warning(
            "%s is ignored because %s is missing or invalid",
            _QUALIFIED_TP_SHARD_WEIGHTS_ENV,
            _QUALIFIED_TP_MODEL_IDENTITY_ENV,
        )
        return None
    if model_path is None:
        return None
    root = Path(model_path).expanduser()
    if not root.is_dir() or model_identity_digest(root) != scope:
        return None
    if tensor_parallel_size != node_count:
        raise PlanningError(
            f"{_QUALIFIED_TP_SHARD_WEIGHTS_ENV} is supported only for pure "
            "tensor parallelism across every selected node"
        )
    value = raw.strip()
    if not value or len(value) > 4096:
        raise PlanningError(
            f"{_QUALIFIED_TP_SHARD_WEIGHTS_ENV} must be a comma-separated "
            "vector of positive integers"
        )
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != tensor_parallel_size or any(
        re.fullmatch(r"[1-9][0-9]*", part) is None for part in parts
    ):
        raise PlanningError(
            f"{_QUALIFIED_TP_SHARD_WEIGHTS_ENV} must contain exactly "
            f"{tensor_parallel_size} comma-separated positive integers in "
            "rank order"
        )
    weights = tuple(int(part) for part in parts)
    logger.warning(
        "%s is forcing experimental TP shard weights %s; this override "
        "takes precedence over persisted exact-match qualifications",
        _QUALIFIED_TP_SHARD_WEIGHTS_ENV,
        weights,
    )
    return (weights,)


def _qualification_statuses(
    hosts: Sequence[ClusterHostRequest],
) -> dict[str, dict[str, Any]]:
    """Current exact runtime evidence in the posted rank order.

    Qualification lookup is an optimization and callers decide whether an
    unavailable peer is a warning or a hard activation failure.  This helper
    itself stays strict: a partial result must never look like a match.
    """

    statuses: dict[str, dict[str, Any]] = {}
    for host in hosts:
        if host.ssh in {"127.0.0.1", "localhost", "::1"}:
            statuses[host.node_id] = {
                "runtime_compatible": True,
                "status": collect_cluster_status().to_dict(),
            }
            continue
        statuses[host.node_id] = probe_remote_host(
            host.ssh,
            python_executable=host.python_executable or sys.executable,
        )
    return statuses


def _tp_qualification_key(
    *,
    model_path: str,
    nodes: Sequence[ClusterPlanNodeRequest],
    statuses: dict[str, dict[str, Any]],
    backend: str,
    tensor_parallel_size: int,
    target_context_tokens: int,
    execution_profile_name: str,
    auto_tune: bool,
    sampling_rank_only: bool,
    mtp_enabled: bool,
    mtp_num_draft_tokens: int | None,
) -> TPQualificationKey:
    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise ValueError("the complete model is unavailable for qualification")
    ordered_statuses = []
    for node in nodes:
        status = statuses.get(node.node_id)
        if not isinstance(status, dict):
            raise ValueError(
                f"runtime qualification evidence is unavailable for {node.node_id}"
            )
        ordered_statuses.append(status)
    defaults = execution_profile(
        execution_profile_name,
        auto_tune=auto_tune,
        sampling_rank_only=sampling_rank_only,
    )
    return TPQualificationKey(
        model_identity=model_identity_digest(root),
        nodes=node_fingerprints_from_statuses(
            [node.node_id for node in nodes],
            ordered_statuses,
            backend=backend,
        ),
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        context_bucket=context_bucket(target_context_tokens),
        execution_profile=execution_profile_name,
        microbatch_size=defaults.pipeline_microbatch_size,
        decode_concurrency=defaults.decode_concurrency,
        prompt_concurrency=defaults.prompt_concurrency,
        prefill_step_size=defaults.prefill_step_size,
        auto_tune=auto_tune,
        mtp_enabled=mtp_enabled,
        mtp_depth=(mtp_num_draft_tokens or 3) if mtp_enabled else None,
    )


def _resolve_tp_layout_qualification(
    *,
    model_path: str | None,
    nodes: Sequence[ClusterPlanNodeRequest],
    statuses: dict[str, dict[str, Any]] | None,
    backend: str,
    tensor_parallel_size: int,
    target_context_tokens: int,
    execution_profile_name: str,
    auto_tune: bool,
    sampling_rank_only: bool,
    mtp_enabled: bool,
    mtp_num_draft_tokens: int | None,
    expected_qualification_id: str | None = None,
    allow_persistent: bool = True,
) -> tuple[
    tuple[tuple[int, ...], ...] | None,
    TPQualificationProvenance | None,
    dict[str, Any],
]:
    """Environment override first, then one exact persisted record."""

    override = _operator_qualified_tp_shard_weights(
        tensor_parallel_size=tensor_parallel_size,
        node_count=len(nodes),
        model_path=model_path,
    )
    if override is not None:
        provenance = TPQualificationProvenance.environment(override[0])
        return (
            override,
            provenance,
            {
                "matched": True,
                "source": "environment_override",
                "qualification_id": provenance.qualification_id,
                "shard_weights": list(override[0]),
                "reason": provenance.reason,
            },
        )
    if tensor_parallel_size <= 1:
        return None, None, {
            "matched": False,
            "source": "not_applicable",
            "reason": "tensor layout qualification requires tensor parallelism",
        }
    if tensor_parallel_size != len(nodes):
        return None, None, {
            "matched": False,
            "source": "equal_fallback",
            "reason": "persisted layout qualification currently requires pure TP",
        }
    if not allow_persistent:
        return None, None, {
            "matched": False,
            "source": "equal_fallback",
            "reason": "the approved preview did not name a TP qualification",
        }
    if not model_path or statuses is None:
        return None, None, {
            "matched": False,
            "source": "equal_fallback",
            "reason": "complete model and live rank fingerprints are required",
        }
    try:
        key = _tp_qualification_key(
            model_path=model_path,
            nodes=nodes,
            statuses=statuses,
            backend=backend,
            tensor_parallel_size=tensor_parallel_size,
            target_context_tokens=target_context_tokens,
            execution_profile_name=execution_profile_name,
            auto_tune=auto_tune,
            sampling_rank_only=sampling_rank_only,
            mtp_enabled=mtp_enabled,
            mtp_num_draft_tokens=mtp_num_draft_tokens,
        )
        if (
            expected_qualification_id is not None
            and key.qualification_id != expected_qualification_id
        ):
            return None, None, {
                "matched": False,
                "source": "equal_fallback",
                "qualification_id": key.qualification_id,
                "reason": (
                    "live model/hardware/runtime fingerprint no longer matches "
                    "the approved qualification"
                ),
            }
        store = get_tp_layout_qualification_store()
        record = store.lookup(key)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, None, {
            "matched": False,
            "source": "equal_fallback",
            "reason": f"TP qualification unavailable: {exc}",
        }
    if record is None:
        decision = store.decision(key)
        if expected_qualification_id is not None:
            decision["reason"] = (
                "the approved TP qualification is missing, corrupt, "
                "inexact, or no longer promotable"
            )
        return None, None, decision
    if (
        expected_qualification_id is not None
        and record.qualification_id != expected_qualification_id
    ):
        return None, None, {
            "matched": False,
            "source": "equal_fallback",
            "qualification_id": record.qualification_id,
            "reason": "the exact record differs from the approved qualification",
        }
    provenance = TPQualificationProvenance.from_record(record)
    return (
        (record.shard_weights,),
        provenance,
        store.decision(key),
    )


def _create_cluster_plan(
    request: ClusterPlanRequest,
    *,
    qualified_tensor_shard_weights: tuple[tuple[int, ...], ...] | None = None,
    tensor_parallel_qualification: TPQualificationProvenance | None = None,
):
    model, nodes = _model_and_nodes(request)
    if request.serving_mode == "disaggregated":
        if request.tensor_parallel_size != 1:
            raise PlanningError(
                "phase-split serving uses complete replicas, not tensor shards"
            )
        if request.mtp_enabled or request.mtp_num_draft_tokens is not None:
            raise PlanningError(
                "phase-split serving does not yet support speculative decode"
            )
        if request.prefill_rank is None or request.decode_rank is None:
            raise PlanningError(
                "phase-split serving requires selected prefill and decode Macs"
            )
        plan = build_full_replica_shard_plan(
            model,
            nodes,
            prefill_rank=request.prefill_rank,
            decode_rank=request.decode_rank,
            context_tokens=request.target_context_tokens,
            workload_profile=request.execution_profile,
        )
        path_map = validate_model_path_map(
            request.path_map,
            tuple(node.node_id.strip() for node in request.nodes),
        )
        return replace(plan, path_map=path_map) if path_map else plan
    if (
        request.tensor_parallel_size > 1
        and request.tensor_parallel_size < len(nodes)
        and len(nodes) % request.tensor_parallel_size == 0
        and model.source != "synthetic"
        and not model.supports_pipeline
    ):
        raise PlanningError(
            "hybrid TP x pipeline requires a model architecture with a "
            "validated pipeline forward path"
        )
    if (
        len(nodes) > 1
        and request.tensor_parallel_size == 1
        and model.source != "synthetic"
        and not model.supports_pipeline
    ):
        detail = (
            "pipeline parallelism is not possible for this model: the "
            "architecture does not implement the MLX-LM pipeline forward path"
        )
        if model.supports_tensor_parallel:
            detail += (
                f". Choose {len(nodes)}-way tensor parallelism to run it "
                f"across {len(nodes)} Macs."
            )
        raise PlanningError(detail)
    path_map = validate_model_path_map(
        request.path_map,
        tuple(node.node_id.strip() for node in request.nodes),
    )
    defaults = execution_profile(request.execution_profile)
    if request.tensor_parallel_size > 1:
        if request.allocation != "balanced":
            raise PlanningError(
                "RAM-proportional allocation is a pipeline-only rule; tensor "
                "parallelism uses the hybrid planner (allocation='balanced')"
            )
        # The coordinator-only environment remains an explicit experimental
        # escape hatch and takes precedence over a persisted record selected
        # by the route.  It receives unqualified provenance so the plan and
        # approval signature cannot present it as a proven layout.
        override = _operator_qualified_tp_shard_weights(
            tensor_parallel_size=request.tensor_parallel_size,
            node_count=len(nodes),
            model_path=request.model_path,
        )
        if override is not None:
            qualified_tensor_shard_weights = override
            tensor_parallel_qualification = (
                TPQualificationProvenance.environment(override[0])
            )
        plan = plan_hybrid(
            model,
            nodes,
            tensor_parallel_size=request.tensor_parallel_size,
            workload_profile=request.execution_profile,
            microbatch_size=(
                request.pipeline_microbatch_size or defaults.pipeline_microbatch_size
            ),
            context_tokens=request.target_context_tokens,
            qualified_tensor_shard_weights=qualified_tensor_shard_weights,
            tensor_parallel_qualification=tensor_parallel_qualification,
        )
    elif request.allocation == "proportional":
        plan = plan_proportional_pipeline(
            model,
            nodes,
            workload_profile=request.execution_profile,
            microbatch_size=(
                request.pipeline_microbatch_size or defaults.pipeline_microbatch_size
            ),
            context_tokens=request.target_context_tokens,
        )
    else:
        plan = plan_unequal_pipeline(
            model,
            nodes,
            workload_profile=request.execution_profile,
            microbatch_size=(
                request.pipeline_microbatch_size or defaults.pipeline_microbatch_size
            ),
            context_tokens=request.target_context_tokens,
        )
    if path_map:
        plan = replace(plan, path_map=path_map)
    return plan


class ClusterAutoconfigureRequest(BaseModel):
    """Everything one-click activation needs; the server decides the rest."""

    # Active-cluster membership changes keep the durable deployment identity
    # while recomputing every rank/host/transport field. First-time setup omits
    # this and receives the normal model+plan-derived ID.
    deployment_id: str | None = Field(default=None, max_length=128)
    model_path: str | None = Field(default=None, max_length=4096)
    model_source: str | None = Field(default=None, max_length=255)
    model_source_python: str | None = Field(default=None, max_length=4096)
    model_size_bytes: int | None = Field(default=None, gt=0)
    layer_count: int = Field(default=80, gt=0, le=2048)
    nodes: list[ClusterPlanNodeRequest] = Field(min_length=1, max_length=64)
    hosts: list[ClusterHostRequest] = Field(default_factory=list, max_length=64)
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"
    prefer: Literal["speed", "capacity"] = "speed"
    strategy: Literal["auto", "tensor", "pipeline", "disaggregated"] = "auto"
    prefill_rank: int | None = Field(default=None, ge=0, le=63)
    decode_rank: int | None = Field(default=None, ge=0, le=63)
    detect_transports: bool = True
    preflight: bool = True
    auto_tune: bool = True
    # The GUI enables this for one-click setup. The synthetic probe is run
    # before staging so its compute measurements can shape the signed layer
    # placement, instead of proposing a faster split after the wrong shards
    # have already been copied.
    measure_performance: bool = False
    sampling_rank_only: bool = True
    async_overlap: bool = True
    cache_affinity: bool = True
    # Distributed workers always retain their bounded in-memory prompt LRU.
    # Durable boundary snapshots add bounded detached state and SSD traffic,
    # so one-click setup keeps them opt-in and carries the choice through the
    # signed plan instead of silently changing memory/disk use at activation.
    prompt_cache_ssd: bool = False
    prompt_cache_ssd_max_bytes: int = Field(
        default=DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES,
        gt=0,
    )
    max_kv_size: int | None = Field(default=None, gt=0)
    ring_connections_per_ip: int | None = Field(default=None, ge=1, le=32)
    target_context_tokens: int = Field(default=8192, ge=1, le=1_048_576)
    mtp_enabled: bool = False
    mtp_num_draft_tokens: int | None = Field(default=None, ge=1, le=8)


def _measured_link_profiles(
    request: ClusterAutoconfigureRequest,
) -> tuple[SimpleNamespace, ...]:
    """Measured collective bandwidth, re-keyed to the ids placement works in.

    Profiles arrive keyed by ``node_id`` while the transport graph is keyed by
    SSH target, so the two must be joined before placement can use a
    measurement. A node with no profile is simply absent — the placement then
    falls back to the link's nominal speed and says so.
    """

    ssh_by_node = {host.node_id: host.ssh for host in request.hosts}
    profiles = []
    for node in request.nodes:
        payload = node.performance
        ssh = ssh_by_node.get(node.node_id)
        if not payload or not ssh:
            continue
        rate = payload.get("collective_bandwidth_bytes_per_second")
        if not rate or float(rate) <= 0:
            continue
        profiles.append(
            SimpleNamespace(
                node_id=ssh,
                collective_bandwidth_bytes_per_second=float(rate),
            )
        )
    return tuple(profiles)


def _staging_for(
    request: ClusterAutoconfigureRequest, choice: Any
) -> dict[str, Any] | None:
    """Per-node staging plan read from the Mac that owns the complete model."""

    if not request.model_path:
        return None
    source_host = (request.model_source or "127.0.0.1").strip()
    source = Path(request.model_path).expanduser()
    if _local_ssh_target(source_host) and not source.is_dir():
        return {
            "error": f"The coordinator cannot read the selected model: {source}",
            "ready": False,
        }
    hosts_by_node = {host.node_id: host.ssh for host in request.hosts}
    try:
        return stage_manifest(
            request.model_path,
            choice.plan.assignments,
            hosts_by_node,
            source_host=(
                "127.0.0.1"
                if _local_ssh_target(source_host)
                else validate_ssh_target(source_host)
            ),
            source_python_executable=request.model_source_python,
        )
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc), "ready": False}


def _resolve_fabric(
    hosts: list[str],
    *,
    verifier: Any | None = None,
) -> dict[str, Any]:
    """Where each Mac answers, and the backend those addresses actually allow.

    One reading of every host answers both questions. An address someone typed
    was true when they typed it — macOS renumbers a Thunderbolt port across a
    reboot — and a backend someone picked is one ``ClusterDeployment`` may
    refuse, because anything but ``ring`` needs a full RDMA matrix that only
    the live interface names can produce.
    """

    interfaces = {host: probe_host_interfaces(host) for host in hosts}
    verify = verifier or verify_link_reachability
    verified_links: dict[
        tuple[tuple[str, str, str], ...], tuple[bool, str]
    ] = {}

    def verify_once(link: Any) -> tuple[bool, str]:
        endpoints = (link.source, link.peer)
        key = tuple(
            sorted(
                (endpoint.host, endpoint.interface, endpoint.address)
                for endpoint in endpoints
                if endpoint is not None
            )
        )
        if key not in verified_links:
            verified_links[key] = verify(link)
        return verified_links[key]

    # Each host is described by its link to the first host that is not itself:
    # that is the address the rest of the cluster reaches it on. Backend
    # readiness is stricter: a collective is a full graph, so every pair must
    # also answer before any fast backend is named or activation is allowed.
    links = [
        resolve_link_addresses(
            host,
            hosts[1 if index == 0 else 0],
            probe=lambda target: interfaces[target],
            verify=verify_once,
        )
        for index, host in enumerate(hosts)
    ]
    pair_links = [
        resolve_link_addresses(
            source,
            peer,
            probe=lambda target: interfaces[target],
            verify=verify_once,
        )
        for index, source in enumerate(hosts)
        for peer in hosts[index + 1 :]
    ]
    matrix = build_rdma_matrix([interfaces[host] for host in hosts])
    rdma = matrix.to_dict()
    unverified_rdma = next(
        (link for link in pair_links if not link.ok or link.kind != "rdma"),
        None,
    )
    if rdma["ok"] and unverified_rdma is not None:
        rdma["ok"] = False
        rdma["reason"] = (
            "not every cluster pair verified over RDMA: "
            f"{unverified_rdma.reason}"
        )

    proposed, reason = choose_backend(pair_links)
    # choose_backend answers from what was detected, but no backend works
    # without an address both ends share, and every non-ring backend needs the
    # full RDMA matrix. Reconciling them here is what turns a failure inside a
    # constructor into a fallback the page can state.
    unresolved = next((link for link in pair_links if not link.ok), None)
    blocker = ""
    if unresolved is not None:
        blocker = unresolved.reason
    elif proposed != "ring" and not rdma["ok"]:
        blocker = str(rdma["reason"])
    fell_back = bool(blocker) and proposed != "ring"
    if fell_back:
        reason = f"{reason}; falling back to the TCP ring because {blocker}"
    elif blocker:
        reason = blocker
    backend = "ring" if blocker else proposed

    return {
        "ok": unresolved is None,
        "backend": backend,
        "backend_reason": reason,
        "blocker": blocker,
        "fell_back": fell_back,
        "link": (unresolved or links[0]).to_dict(),
        "rdma": rdma,
        "hosts": [
            {
                "host": host,
                "ips": [link.source.address] if link.source else [],
                "interface": link.source.interface if link.source else "",
                "rdma": list(matrix.rows[index]) if backend != "ring" else [],
            }
            for index, (host, link) in enumerate(zip(hosts, links))
        ],
    }


def _tp_layout_recommendation_payload(
    model: Any,
    nodes: list[NodeBudget],
    plan: ShardPlan,
    *,
    workload_profile: str,
    context_tokens: int,
    qualification: TPQualificationProvenance | None,
) -> dict[str, Any] | None:
    """Describe a measured candidate without silently activating it."""

    if plan.tensor_parallel_size <= 1 or plan.pipeline_stages != 1:
        return None
    assignments = sorted(plan.assignments, key=lambda item: item.rank)
    current = tuple(
        int(item.tensor_parallel_shard_weight or 1) for item in assignments
    )
    if qualification is not None:
        persistent = qualification.source == "persistent"
        return {
            "state": "qualified" if persistent else "experimental",
            "current_weights": list(current),
            "recommended_weights": list(qualification.shard_weights),
            "requires_qualification": not persistent,
            "qualification_id": qualification.qualification_id,
            "reason": qualification.reason,
        }
    try:
        candidate = recommend_tensor_shard_weights(
            model,
            nodes,
            workload_profile=workload_profile,
        )
    except (TypeError, ValueError) as exc:
        return {
            "state": "equal",
            "current_weights": list(current),
            "recommended_weights": list(current),
            "requires_qualification": False,
            "reason": f"shard recommendation unavailable: {exc}",
        }
    if candidate == current:
        return {
            "state": "equal",
            "current_weights": list(current),
            "recommended_weights": list(current),
            "requires_qualification": False,
            "reason": "measured rank rates do not justify an asymmetric split",
        }
    try:
        plan_hybrid(
            model,
            nodes,
            tensor_parallel_size=plan.tensor_parallel_size,
            workload_profile=workload_profile,
            context_tokens=context_tokens,
            qualified_tensor_shard_weights=(candidate,),
        )
    except PlanningError as exc:
        return {
            "state": "equal",
            "current_weights": list(current),
            "recommended_weights": list(current),
            "requires_qualification": False,
            "reason": f"measured asymmetric candidate does not fit: {exc}",
        }
    return {
        "state": "calibration_required",
        "current_weights": list(current),
        "recommended_weights": list(candidate),
        "requires_qualification": True,
        "reason": (
            "synthetic rank rates nominate this vector; matched full-model "
            "A/B and parity are required before activation"
        ),
    }


@router.get("/fabric")
async def cluster_fabric(hosts: str = Query(...)):
    """The addresses these Macs answer on, and the backend they allow.

    Both are read rather than asked for. The address fields were the ones that
    broke a launch, and a backend is a consequence of the cable rather than a
    preference — so neither is a control the user has to get right.
    """

    host_list = _validated_ssh_targets(hosts)
    if len(host_list) < 2:
        raise HTTPException(
            status_code=400, detail="a distributed cluster needs at least two hosts"
        )
    # A probe failure here is an unpaired or unreachable Mac, not a server
    # fault. An unhandled raise became a scrubbed 500 that swallowed the SSH
    # stderr the dashboard needs to explain pairing.
    try:
        return await asyncio.to_thread(_resolve_fabric, host_list)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/autoconfigure")
async def cluster_autoconfigure(request: ClusterAutoconfigureRequest):
    """Build a complete cluster configuration from peers and a model.

    This is the brain behind the one-click button: it probes the fabric, picks
    the parallelism split and the collective backend, and returns a proposal the
    dashboard can activate as-is. This endpoint deliberately does not start
    processes itself: the dashboard only posts its activation block after all
    preflight and staging checks report ready.
    """

    _validate_cluster_hosts(request.hosts)

    plan_request = ClusterPlanRequest(
        model_path=request.model_path,
        model_source=request.model_source,
        model_source_python=request.model_source_python,
        model_size_bytes=request.model_size_bytes,
        layer_count=request.layer_count,
        nodes=request.nodes,
        execution_profile=request.execution_profile,
        prompt_cache_ssd=request.prompt_cache_ssd,
        prompt_cache_ssd_max_bytes=request.prompt_cache_ssd_max_bytes,
        mtp_enabled=request.mtp_enabled,
        mtp_num_draft_tokens=request.mtp_num_draft_tokens,
    )
    try:
        model, nodes = _model_and_nodes(plan_request)
    except PlanningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    transports: tuple[Any, ...] = ()
    transport_error: str | None = None
    fabric: dict[str, Any] | None = None
    fabric_error: str | None = None
    requested_host_order = [host.ssh for host in request.hosts]
    if request.detect_transports and request.hosts:
        try:
            matrix = await asyncio.to_thread(
                detect_cluster_transports, requested_host_order
            )
            transports = tuple(matrix.transports)
            record_peer_transports(transports)
        except (RuntimeError, OSError) as exc:
            # A fabric we cannot probe is not a reason to refuse to configure —
            # it only means the choice is made without link information, which
            # choose_parallelism reports as a warning.
            transport_error = str(exc)

    # Raw device detection can see a Thunderbolt/RDMA port before it has a
    # routable address. Resolve the usable fabric before choosing TP: otherwise
    # Automatic selects the chatty strategy from the cable label and only later
    # falls back to a 100-MiB/s TCP ring.
    if request.detect_transports and len(request.hosts) > 1:
        try:
            fabric = await asyncio.to_thread(
                _resolve_fabric,
                requested_host_order,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            fabric_error = str(exc)
    provisional_backend = (
        str(fabric["backend"])
        if fabric is not None
        else choose_backend(transports)[0]
    )
    strategy_transports = transports
    if provisional_backend == "ring" and transports:
        strategy_transports = (SimpleNamespace(kind="ethernet"),)

    measurements = {}
    if request.strategy == "auto" and request.model_path:
        try:
            measurements = await asyncio.to_thread(
                get_strategy_benchmark_store().measurements,
                model=request.model_path,
                node_ids=tuple(node.node_id for node in request.nodes),
                backend=provisional_backend,
                target_context_tokens=request.target_context_tokens,
            )
        except (OSError, RuntimeError, ValueError):
            # A first-run cluster has no history. The safe capability/link
            # heuristic remains the fallback until both strategies have real
            # end-to-end samples.
            measurements = {}

    try:
        if request.strategy == "disaggregated":
            if request.prefill_rank is None or request.decode_rank is None:
                raise PlanningError(
                    "phase split requires selected prefill and decode Macs"
                )
            phase_plan = build_full_replica_shard_plan(
                model,
                nodes,
                prefill_rank=request.prefill_rank,
                decode_rank=request.decode_rank,
                context_tokens=request.target_context_tokens,
                workload_profile=request.execution_profile,
            )
            qualified_tensor_shard_weights = None
            qualification_provenance = None
            choice = SimpleNamespace(
                plan=phase_plan,
                tensor_parallel_size=1,
                pipeline_stages=1,
                reason=(
                    "Each Mac holds a complete model replica. "
                    f"Rank {request.prefill_rank} processes prompts and rank "
                    f"{request.decode_rank} owns generation."
                ),
                warnings=(),
            )
        else:
            qualified_tensor_shard_weights = (
                _operator_qualified_tp_shard_weights(
                    tensor_parallel_size=len(nodes),
                    node_count=len(nodes),
                    model_path=request.model_path,
                )
                if request.strategy != "pipeline"
                else None
            )
            qualification_provenance = (
                TPQualificationProvenance.environment(
                    qualified_tensor_shard_weights[0]
                )
                if qualified_tensor_shard_weights is not None
                else None
            )
            choice = choose_parallelism(
                model,
                nodes,
                transports=strategy_transports,
                prefer=request.prefer,
                strategy=request.strategy,
                measurements=measurements,
                workload_profile=request.execution_profile,
                context_tokens=request.target_context_tokens,
                qualified_tensor_shard_weights=qualified_tensor_shard_weights,
                hybrid_runtime_supported=hybrid_group_split_supported(
                    provisional_backend
                ),
            )
    except PlanningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Pre-flight: find the reasons activation would fail before proposing it,
    # rather than after every rank has loaded its weights.
    peer_statuses: dict[str, Any] = {}
    if request.preflight and request.hosts:
        for host in request.hosts:
            if host.ssh in {"127.0.0.1", "localhost", "::1"}:
                peer_statuses[host.node_id] = {
                    "runtime_compatible": True,
                    "status": collect_cluster_status().to_dict(),
                }
                continue
            try:
                peer_statuses[host.node_id] = await asyncio.to_thread(
                    probe_remote_host, host.ssh
                )
            except (DistributedLaunchError, OSError, ValueError):
                peer_statuses[host.node_id] = None
    issues = preflight_issues(
        peer_statuses,
        model_path=request.model_path,
        local_versions={"mlx_version": _package_version_or_none("mlx")},
    )
    warnings = list(choice.warnings)
    # The other half of pre-flight: a rank does not fail on the model it cannot
    # fit — that is guarded — it fails on the import it cannot perform, after
    # every other rank has already paid for its weights.
    if request.preflight and request.hosts and request.model_path:
        try:
            issues += await asyncio.to_thread(
                peer_import_issues,
                {host.node_id: host.ssh for host in request.hosts},
                model_path=request.model_path,
                python_by_node={
                    node_id: str(
                        ((status or {}).get("status") or {})
                        .get("runtime", {})
                        .get("python_executable")
                        or ""
                    )
                    for node_id, status in peer_statuses.items()
                    if status
                },
            )
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"Peer environment check failed: {exc}")

    # Issues are not repeated as warnings: they leave here whole, each with the
    # command that fixes it, and a caller that restates the detail as a loose
    # sentence has thrown that command away.
    if transport_error:
        warnings.append(f"Transport detection failed: {transport_error}")
    if fabric_error:
        warnings.append(f"Address discovery failed: {fabric_error}")

    # Place tensor-parallel groups on the fastest links: TP all-reduces once
    # per layer per token, while a pipeline boundary sends one tensor per
    # stage. Measured bandwidth is used where a node has been probed, so two
    # links of the same kind are not treated as interchangeable.
    link_profiles = _measured_link_profiles(request)
    placement = (
        SimpleNamespace(
            hosts=tuple(requested_host_order),
            warnings=(),
            reason="phase roles retain the selected Mac order",
        )
        if request.strategy == "disaggregated"
        else order_hosts_for_topology(
            [host.ssh for host in request.hosts],
            transports,
            choice.tensor_parallel_size,
            link_profiles,
        )
    )
    warnings.extend(placement.warnings)
    placed_host_order = list(placement.hosts or requested_host_order)
    grouped_host_order = (
        placed_host_order
        if request.strategy == "disaggregated"
        else _coalesce_verified_cuda_groups(
            placed_host_order,
            list(request.hosts),
            list(request.nodes),
        )
    )
    if grouped_host_order != placed_host_order:
        warnings.append(
            "Verified ConnectX CUDA workers were kept adjacent in the outer Ring."
        )
    ordered_hosts = list(request.hosts)
    ordered_request_nodes = list(request.nodes)
    ordered_budgets = list(nodes)
    if grouped_host_order:
        by_ssh = {host.ssh: host for host in request.hosts}
        ordered_hosts = [by_ssh[ssh] for ssh in grouped_host_order if ssh in by_ssh]
        request_node_by_id = {node.node_id: node for node in request.nodes}
        budget_by_id = {node.node_id: node for node in nodes}
        ordered_request_nodes = [
            request_node_by_id[host.node_id]
            for host in ordered_hosts
            if host.node_id in request_node_by_id
        ]
        ordered_budgets = [
            replace(budget_by_id[host.node_id], rank=rank)
            for rank, host in enumerate(ordered_hosts)
            if host.node_id in budget_by_id
        ]
        if len(ordered_budgets) != len(nodes):
            raise HTTPException(
                status_code=400,
                detail="topology placement did not preserve every node budget",
            )
        # Host order defines rank order. Rebuild the plan against that exact
        # order so assignments, memory budgets and transport endpoints cannot
        # describe three different rank maps.
        ordered_plan = (
            build_full_replica_shard_plan(
                model,
                ordered_budgets,
                prefill_rank=int(request.prefill_rank),
                decode_rank=int(request.decode_rank),
                context_tokens=request.target_context_tokens,
                workload_profile=request.execution_profile,
            )
            if request.strategy == "disaggregated"
            else plan_hybrid(
                model,
                ordered_budgets,
                tensor_parallel_size=choice.tensor_parallel_size,
                workload_profile=request.execution_profile,
                context_tokens=request.target_context_tokens,
                qualified_tensor_shard_weights=(
                    qualified_tensor_shard_weights
                    if choice.tensor_parallel_size == len(ordered_budgets)
                    else None
                ),
                tensor_parallel_qualification=(
                    qualification_provenance
                    if choice.tensor_parallel_size == len(ordered_budgets)
                    else None
                ),
            )
        )
        if request.strategy == "disaggregated":
            choice.plan = ordered_plan
        else:
            choice = replace(choice, plan=ordered_plan)
    for group in tp_groups_spanning_slow_links(
        [host.ssh for host in ordered_hosts],
        transports,
        choice.tensor_parallel_size,
        link_profiles,
    ):
        warnings.append(
            f"Tensor-parallel group {' + '.join(group)} spans a slow link; "
            f"every layer's all-reduce will cross it."
        )

    # Addresses last, and against the placed order. The RDMA matrix is indexed
    # by rank: read it before placement has settled the order and every rank is
    # handed the path to the wrong peer.
    backend, backend_reason = choose_backend(transports)
    if request.detect_transports and len(ordered_hosts) > 1:
        ordered_host_targets = [host.ssh for host in ordered_hosts]
        if fabric is None or ordered_host_targets != requested_host_order:
            # A fabric matrix is rank-ordered. If the placed order needs a new
            # reading and that reading fails, the old matrix is actively
            # unsafe — keeping it would hand each rank another Mac's path.
            fabric = None
            try:
                fabric = await asyncio.to_thread(
                    _resolve_fabric,
                    ordered_host_targets,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                fabric_error = str(exc)
                warnings.append(f"Address discovery failed: {fabric_error}")
    activation_hosts = [host.model_dump() for host in ordered_hosts]
    if fabric is not None:
        backend, backend_reason = fabric["backend"], fabric["backend_reason"]
        if fabric["ok"]:
            for host, discovered in zip(activation_hosts, fabric["hosts"]):
                host["ips"] = discovered["ips"]
                host["rdma"] = discovered["rdma"]
        else:
            warnings.append(fabric.get("blocker") or fabric["link"]["reason"])

    fabric_required = request.detect_transports and len(ordered_hosts) > 1
    fabric_ready = not fabric_required or bool(fabric and fabric.get("ok"))
    fabric_blocker = ""
    if not fabric_ready:
        if fabric is not None:
            fabric_blocker = str(
                fabric.get("blocker")
                or fabric.get("backend_reason")
                or fabric.get("link", {}).get("reason")
                or "the cluster route did not verify"
            )
        else:
            fabric_blocker = fabric_error or "the cluster route could not be read"

    # The final backend and rank order are now known.  Only here can an exact
    # persisted heterogeneous layout be matched safely; a vector is rank
    # ordered, so looking it up before topology placement would apply the
    # right numbers to the wrong Macs.
    (
        qualified_tensor_shard_weights,
        qualification_provenance,
        qualification_decision,
    ) = _resolve_tp_layout_qualification(
        model_path=request.model_path,
        nodes=ordered_request_nodes,
        statuses=peer_statuses if peer_statuses else None,
        backend=backend,
        tensor_parallel_size=choice.tensor_parallel_size,
        target_context_tokens=request.target_context_tokens,
        execution_profile_name=request.execution_profile,
        auto_tune=request.auto_tune,
        sampling_rank_only=request.sampling_rank_only,
        mtp_enabled=request.mtp_enabled,
        mtp_num_draft_tokens=request.mtp_num_draft_tokens,
    )
    if choice.tensor_parallel_size > 1 and qualified_tensor_shard_weights is not None:
        try:
            qualified_plan = plan_hybrid(
                model,
                ordered_budgets,
                tensor_parallel_size=choice.tensor_parallel_size,
                workload_profile=request.execution_profile,
                context_tokens=request.target_context_tokens,
                qualified_tensor_shard_weights=qualified_tensor_shard_weights,
                tensor_parallel_qualification=qualification_provenance,
            )
        except PlanningError as exc:
            if (
                qualification_provenance is not None
                and qualification_provenance.source == "persistent"
            ):
                qualification_decision = {
                    "matched": False,
                    "source": "equal_fallback",
                    "qualification_id": (
                        qualification_provenance.qualification_id
                    ),
                    "reason": f"qualified layout no longer fits: {exc}",
                }
                qualified_tensor_shard_weights = None
                qualification_provenance = None
            else:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            choice = replace(choice, plan=qualified_plan)

    performance_probe: dict[str, Any] = {
        "ok": False,
        "status": (
            "phase_probe_required"
            if request.strategy == "disaggregated"
            else "disabled"
        ),
        "reason": (
            "phase split uses full-model prefill/decode calibration, not the "
            "sharded synthetic placement probe"
            if request.strategy == "disaggregated"
            else "automatic performance measurement disabled"
        ),
    }
    profiled_request_nodes = list(ordered_request_nodes)
    existing_profiles = (
        ()
        if request.strategy == "disaggregated"
        else _request_performance_profiles(profiled_request_nodes)
    )
    if existing_profiles:
        # The first proposal carries these measurements into the post-staging
        # refresh. Reusing them avoids running the same distributed synthetic
        # probe twice when Start Cluster had to copy model shards in between.
        performance_probe = {
            "ok": True,
            "status": "reused_before_staging",
            "backend": backend,
            "profiles": [profile.to_dict() for profile in existing_profiles],
            "plan_changed": False,
        }
    elif (
        fabric_ready
        and request.measure_performance
        and request.auto_tune
        and request.model_path
        and len(activation_hosts) >= 2
        and qualification_provenance is None
        and request.strategy != "disaggregated"
    ):
        try:
            probe_execution = _execution_for_request(
                request,
                choice.plan.assignments,
                backend=backend,
            )
            probe_deployment = ClusterDeployment(
                deployment_id=f"probe-{choice.plan.plan_hash[:12]}",
                model=str(Path(request.model_path).expanduser().resolve()),
                backend=backend,
                hosts=tuple(
                    ClusterHost(
                        node_id=str(host["node_id"]),
                        ssh=str(host["ssh"]),
                        ips=tuple(host.get("ips") or ()),
                        rdma=tuple(host.get("rdma") or ()),
                        python_executable=host.get("python_executable"),
                    )
                    for host in activation_hosts
                ),
                assignments=choice.plan.assignments,
                plan_hash=choice.plan.plan_hash,
                execution=probe_execution,
                tensor_parallel_size=choice.tensor_parallel_size,
                target_context_tokens=request.target_context_tokens,
            )
            performance_probe = await asyncio.to_thread(
                run_cluster_performance_probe,
                probe_deployment,
            )
            if performance_probe.get("promotable") is False:
                raise _UnpromotablePerformanceCalibration(
                    str(
                        performance_probe.get("qualification_reason")
                        or "Low Power Mode was enabled during calibration"
                    )
                )
            profiles = tuple(
                NodePerformanceProfile.from_dict(profile)
                for profile in performance_probe.get("profiles", ())
            )
            if len(profiles) != len(profiled_request_nodes):
                raise ValueError(
                    "performance probe did not return every cluster rank"
                )
            profile_by_node = {profile.node_id: profile for profile in profiles}
            profiled_request_nodes = [
                node.model_copy(
                    update={
                        "performance": profile_by_node[node.node_id].to_dict(),
                    }
                )
                for node in profiled_request_nodes
            ]
            measured_budgets = _node_budgets(profiled_request_nodes)
            optimized_plan = _build_performance_plan(
                model,
                measured_budgets,
                model_path=request.model_path,
                tensor_parallel_size=choice.tensor_parallel_size,
                workload_profile=request.execution_profile,
                microbatch_size=probe_execution.pipeline_microbatch_size,
                context_tokens=request.target_context_tokens,
            )
            performance_changes = _plan_changes(
                choice.plan.to_dict(),
                optimized_plan.to_dict(),
            )
            choice = replace(
                choice,
                plan=optimized_plan,
                reason=(
                    f"{choice.reason} Node compute and link measurements were "
                    "applied before model staging."
                ),
            )
            performance_probe["status"] = "applied_before_staging"
            performance_probe["plan_changed"] = performance_changes["changed"]
            performance_probe["plan_changes"] = performance_changes
        except _UnpromotablePerformanceCalibration as exc:
            profiled_request_nodes = list(ordered_request_nodes)
            performance_probe = {
                "ok": False,
                "promotable": False,
                "status": "power_limited",
                "reason": str(exc)[:1000],
                "plan_changed": False,
            }
            warnings.append(
                "Low Power Mode is enabled on a calibration rank; synthetic "
                "measurements were ignored for placement. Turn it off in "
                "System Settings before recalibrating."
            )
        except (DistributedLaunchError, OSError, PlanningError, ValueError) as exc:
            # First-run performance calibration is an optimization, not a
            # prerequisite. Keep the safe memory plan and say why. Profiles
            # belong to the rejected optimized placement: carrying them into
            # the activation payload would make /deployments re-run that same
            # rejected split and turn a successful preview into a launch-time
            # memory refusal.
            profiled_request_nodes = list(ordered_request_nodes)
            performance_probe = {
                "ok": False,
                "status": "memory_fallback",
                "reason": str(exc)[:1000],
            }
            warnings.append(
                "Performance measurement was unavailable; using the safe "
                f"memory-balanced split. {exc}"
            )

    # stage_manifest probes peers with blocking SSH. Keep it off the FastAPI
    # event loop just like transport detection and preflight above.
    staging = await asyncio.to_thread(_staging_for, request, choice)
    staging_ready = staging is None or bool(staging.get("ready"))

    plan_contract = choice.plan.to_dict()
    if request.mtp_enabled or request.mtp_num_draft_tokens is not None:
        plan_contract.update(
            mtp_enabled=request.mtp_enabled,
            mtp_num_draft_tokens=request.mtp_num_draft_tokens,
        )
    # False deliberately retains the legacy placement signature. True is a
    # material launch-mode choice because every rank creates a persistent,
    # plan-scoped SSD snapshot store and must therefore be explicitly signed.
    if request.prompt_cache_ssd:
        plan_contract.update(
            prompt_cache_ssd=True,
            prompt_cache_ssd_max_bytes=request.prompt_cache_ssd_max_bytes,
        )
    plan_payload = _plan_with_signature(plan_contract)
    preflight_summary = describe_preflight(issues)
    if fabric_blocker:
        preflight_summary = f"Cluster link is not ready: {fabric_blocker}"
    tp_layout_recommendation = _tp_layout_recommendation_payload(
        model,
        _node_budgets(profiled_request_nodes),
        choice.plan,
        workload_profile=request.execution_profile,
        context_tokens=request.target_context_tokens,
        qualification=qualification_provenance,
    )
    if qualification_decision.get("source") == "rejected_evidence":
        tp_layout_recommendation = {
            "state": "rejected",
            "current_weights": [
                int(item.tensor_parallel_shard_weight or 1)
                for item in sorted(choice.plan.assignments, key=lambda row: row.rank)
            ],
            "recommended_weights": list(
                qualification_decision.get("shard_weights") or []
            ),
            "requires_qualification": False,
            "qualification_id": qualification_decision.get("qualification_id"),
            "reason": qualification_decision.get("reason"),
        }
    # Warning and blocker strings embed remote SSH stderr. Redact those and
    # only those: preflight issue commands are pasteable fixes that need their
    # user@host intact, and the activation block round-trips to /deployments.
    return {
        "backend": backend,
        "backend_reason": backend_reason,
        "fabric": fabric,
        "fabric_ready": fabric_ready,
        "fabric_blocker": _redact_diagnostic(fabric_blocker),
        "tensor_parallel_size": choice.tensor_parallel_size,
        "pipeline_stages": choice.pipeline_stages,
        "serving_mode": choice.plan.serving_mode,
        "prefill_rank": choice.plan.prefill_rank,
        "decode_rank": choice.plan.decode_rank,
        "summary": choice.reason,
        "link": describe_transports(transports),
        "placement": placement.reason,
        "strategy": request.strategy,
        "strategy_measurements": {
            str(size): outcome.to_dict()
            for size, outcome in sorted(measurements.items())
        },
        "tp_layout_qualification": qualification_decision,
        "tp_layout_recommendation": tp_layout_recommendation,
        "performance_probe": performance_probe,
        "staging": staging,
        "strategies": STRATEGIES,
        "preflight": _redact_diagnostic(preflight_summary),
        # Structured as well as summarised: an issue that carries a command is
        # a fix the user can paste, and a sentence hides it.
        "preflight_issues": [asdict(issue) for issue in issues],
        "ready_to_activate": (
            not any(issue.blocking for issue in issues)
            and staging_ready
            and fabric_ready
        ),
        "warnings": _redact_diagnostic(warnings),
        "transports": [transport.__dict__ for transport in transports],
        "plan": plan_payload,
        # Ready to POST straight to /deployments once the user approves.
        "activation": {
            "deployment_id": request.deployment_id,
            "model_path": request.model_path,
            "model_source": request.model_source,
            "model_source_python": request.model_source_python,
            "backend": backend,
            "execution_profile": request.execution_profile,
            "auto_tune": request.auto_tune,
            "sampling_rank_only": request.sampling_rank_only,
            "async_overlap": request.async_overlap,
            "cache_affinity": request.cache_affinity,
            "prompt_cache_ssd": request.prompt_cache_ssd,
            "prompt_cache_ssd_max_bytes": request.prompt_cache_ssd_max_bytes,
            "max_kv_size": request.max_kv_size,
            "target_context_tokens": request.target_context_tokens,
            "mtp_enabled": request.mtp_enabled,
            "mtp_num_draft_tokens": request.mtp_num_draft_tokens,
            "tp_qualification_id": (
                qualification_provenance.qualification_id
                if qualification_provenance is not None
                and qualification_provenance.source == "persistent"
                else None
            ),
            "ring_connections_per_ip": (
                request.ring_connections_per_ip if backend == "ring" else None
            ),
            "tensor_parallel_size": choice.tensor_parallel_size,
            "serving_mode": choice.plan.serving_mode,
            "prefill_rank": choice.plan.prefill_rank,
            "decode_rank": choice.plan.decode_rank,
            "nodes": [node.model_dump() for node in profiled_request_nodes],
            "hosts": activation_hosts,
            "preflight": True,
            "approved_placement": plan_payload["placement_signature"],
        },
    }


class ClusterGuidanceRequest(BaseModel):
    """A failure message the dashboard wants turned into next steps."""

    message: str = Field(default="", max_length=4096)


@router.post("/guidance")
async def cluster_guidance(request: ClusterGuidanceRequest):
    """Explain a cluster failure in terms a user can act on.

    Kept as its own endpoint rather than folded into every error body: the
    existing handlers return a plain string ``detail`` that callers and tests
    already depend on, and an explanation is only ever needed after a failure.
    """

    return explain(request.message).to_dict()


class ClusterStageRequest(BaseModel):
    """A signed proposal whose missing model files should be staged."""

    activation: ClusterDeploymentRequest
    parallel: int = Field(default=4, ge=1, le=16)


_STAGING_JOBS: dict[str, dict[str, Any]] = {}
_STAGING_JOBS_LOCK = threading.Lock()
_MAX_STAGING_JOBS = 32

# The coordinator narrates deaths: the first health poll that sees a
# previously-healthy deployment report unhealthy records the incident, once
# per transition rather than once per poll. In-memory is enough — after a
# server restart the next healthy poll re-arms the transition.
_PEER_HEALTH_LAST_HEALTHY: dict[str, bool] = {}
_PEER_HEALTH_LOCK = threading.Lock()


def _staging_job_snapshot(job_id: str) -> dict[str, Any] | None:
    with _STAGING_JOBS_LOCK:
        job = _STAGING_JOBS.get(job_id)
        if job is None:
            return None
        return json.loads(json.dumps(job))


def _record_staging_job(job: dict[str, Any]) -> None:
    with _STAGING_JOBS_LOCK:
        if len(_STAGING_JOBS) >= _MAX_STAGING_JOBS:
            finished = [
                key
                for key, value in _STAGING_JOBS.items()
                if value.get("status") in {"completed", "failed"}
            ]
            if finished:
                _STAGING_JOBS.pop(finished[0], None)
        _STAGING_JOBS[job["job_id"]] = job


def _update_staging_job(job_id: str, update: Any) -> None:
    with _STAGING_JOBS_LOCK:
        job = _STAGING_JOBS[job_id]
        update(job)
        job["updated_at"] = time.time()


def _run_staging_job(
    job_id: str,
    deployment: ClusterDeployment,
    *,
    source_host: str,
    parallel: int,
) -> None:
    """Copy each rank's files from whichever Mac owns the complete model."""

    model_path = Path(deployment.model).expanduser()
    try:
        source_host = (
            "127.0.0.1"
            if _local_ssh_target(source_host)
            else validate_ssh_target(source_host)
        )
        if _local_ssh_target(source_host):
            inventory = model_staging_inventory(model_path)
            shards = index_shards(model_path)
            sidecar_sizes = {
                str(name): int(size)
                for name, size in inventory["sidecars"].items()
            }
        else:
            shards, sidecar_sizes = remote_model_staging_inventory(
                source_host, home_relative_model_path(str(model_path))
            )
        shard_sizes = {item.name: item.size_bytes for item in shards}
        sidecars = tuple(sorted(sidecar_sizes))
        assignments = sorted(deployment.assignments, key=lambda item: item.rank)
        _update_staging_job(
            job_id,
            lambda job: job.update(status="running"),
        )
        # deployment.model is the coordinator's own absolute path. A peer with a
        # different macOS account has a different $HOME, so probing it at that
        # path finds nothing and re-copies the whole model (and then trips the
        # disk-space check). Resolve each peer's copy in its OWN home from the
        # portable ~-form.
        portable_model_path = home_relative_model_path(deployment.model)
        failed_nodes: list[str] = []
        for host, assignment in zip(deployment.hosts, assignments):
            # Cluster v2: files land at the node's path_map entry when one
            # exists (a peer-local absolute path); otherwise the peer's copy
            # of the shared model, resolved in its OWN home from the ~-form.
            path_override = deployment.path_map.get(host.node_id)
            if _local_ssh_target(host.ssh):
                destination_path = (
                    Path(path_override) if path_override else model_path
                )
                destination_dir = str(destination_path)
                present = (
                    {
                        path.name: path.stat().st_size
                        for path in destination_path.iterdir()
                        if path.is_file()
                    }
                    if destination_path.is_dir()
                    else {}
                )
            else:
                destination_dir = path_override or remote_model_dir(
                    host.ssh, portable_model_path
                )
                present = remote_file_sizes(host.ssh, destination_dir)
            plan = plan_staging(
                model_path,
                node_id=assignment.node_id,
                start_layer=assignment.start_layer,
                end_layer=assignment.end_layer,
                present=present,
                shards=shards,
            )
            needed = tuple(
                name
                for name in (*plan.missing, *sidecars)
                if present.get(name)
                != (shard_sizes | sidecar_sizes).get(name)
            )
            total_bytes = sum((shard_sizes | sidecar_sizes)[name] for name in needed)

            def prepare(
                job: dict[str, Any],
                *,
                node_id=assignment.node_id,
                filenames=needed,
                bytes_total=total_bytes,
            ) -> None:
                node = job["nodes"][node_id]
                node.update(
                    status="copying",
                    files_total=len(filenames),
                    files_completed=0,
                    bytes_total=bytes_total,
                    bytes_completed=0,
                    files={name: "queued" for name in filenames},
                )

            _update_staging_job(job_id, prepare)

            def progress(
                filename: str,
                status: str,
                bytes_copied: int,
                *,
                node_id=assignment.node_id,
            ) -> None:
                def apply(job: dict[str, Any]) -> None:
                    node = job["nodes"][node_id]
                    previous = node["files"].get(filename)
                    node["files"][filename] = status
                    if status == "copied" and previous != "copied":
                        node["files_completed"] += 1
                        node["bytes_completed"] += bytes_copied

                _update_staging_job(job_id, apply)

            expected_sizes = {
                name: (shard_sizes | sidecar_sizes)[name]
                for name in needed
            }
            # destination_dir is already path_map-aware (computed above), so
            # pass it unconditionally.
            result = stage_files_from_source(
                plan,
                model_path=model_path,
                source_host=source_host,
                destination_host=host.ssh,
                expected_sizes=expected_sizes,
                destination_dir=destination_dir,
                parallel=parallel,
                progress=progress,
            )

            def finish(
                job: dict[str, Any],
                *,
                node_id=assignment.node_id,
                staging_result=result,
            ) -> None:
                node = job["nodes"][node_id]
                node["status"] = "ready" if staging_result.ok else "failed"
                node["result"] = staging_result.to_dict()
                if not staging_result.ok:
                    node["error"] = (
                        "Failed to copy: " + ", ".join(staging_result.failed)
                    )

            _update_staging_job(job_id, finish)
            if not result.ok:
                failed_nodes.append(assignment.node_id)

        def complete(job: dict[str, Any]) -> None:
            job["status"] = "failed" if failed_nodes else "completed"
            job["ready"] = not failed_nodes
            if failed_nodes:
                job["error"] = "Model staging failed on " + ", ".join(failed_nodes)

        _update_staging_job(job_id, complete)
        if failed_nodes:
            _record_cluster_incident(
                Severity.ERROR,
                "staging_failed",
                "Model staging failed on " + ", ".join(failed_nodes),
                job_id=job_id,
                deployment_id=deployment.deployment_id,
            )
    except Exception as exc:  # noqa: BLE001 - background job reports the failure
        def fail(job: dict[str, Any], *, error=str(exc)) -> None:
            job["status"] = "failed"
            job["ready"] = False
            job["error"] = error

        _update_staging_job(job_id, fail)
        # Already on the staging worker thread: a blocking record is fine.
        _record_cluster_incident(
            Severity.ERROR,
            "staging_failed",
            str(exc),
            job_id=job_id,
            deployment_id=deployment.deployment_id,
        )


@router.post("/stage")
async def cluster_stage(request: ClusterStageRequest):
    """Start an observable, resumable model×node staging job."""

    try:
        deployment, plan = await asyncio.to_thread(
            _create_deployment, request.activation
        )
        approved = request.activation.approved_placement.strip()
        if approved != _placement_signature(plan):
            raise HTTPException(
                status_code=409,
                detail="The staging request no longer matches the approved plan.",
            )
        model_path = Path(deployment.model).expanduser()
        source_host = (request.activation.model_source or "127.0.0.1").strip()
        if _local_ssh_target(source_host) and not model_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail="The coordinator does not have the selected model to stage.",
            )
        job_id = secrets.token_hex(12)
        job = {
            "job_id": job_id,
            "status": "queued",
            "ready": False,
            "model_path": str(model_path),
            "created_at": time.time(),
            "updated_at": time.time(),
            "error": "",
            "nodes": {
                assignment.node_id: {
                    "node_id": assignment.node_id,
                    "rank": assignment.rank,
                    "status": "queued",
                    "files_total": 0,
                    "files_completed": 0,
                    "bytes_total": 0,
                    "bytes_completed": 0,
                    "files": {},
                    "error": "",
                }
                for assignment in deployment.assignments
            },
        }
        _record_staging_job(job)
        thread = threading.Thread(
            target=_run_staging_job,
            args=(job_id, deployment),
            kwargs={
                "source_host": source_host,
                "parallel": request.parallel,
            },
            name=f"omlx-model-staging-{job_id[:8]}",
            daemon=True,
        )
        thread.start()
        return _staging_job_snapshot(job_id)
    except InsufficientDiskError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/stage/{job_id}")
async def cluster_stage_status(job_id: str):
    """Current model×node transfer progress for one staging job."""

    if not re.fullmatch(r"[0-9a-f]{24}", job_id):
        raise HTTPException(status_code=404, detail="staging job not found")
    snapshot = _staging_job_snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="staging job not found")
    return snapshot


@router.get("/status")
async def cluster_status(route_to: str | None = None):
    """Return this node's read-only distributed capability snapshot."""

    try:
        status = await asyncio.to_thread(
            collect_cluster_status,
            route_to=route_to,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return status.to_dict() | {"runtime_jobs": await cluster_runtime()}


def _reconcile_runtime_ownership(payload: dict[str, Any], pool: Any) -> None:
    """Make the coordinator's engine registry authoritative for GUI liveness.

    Rank marker files survive crashes and are deliberately retained for
    diagnostics. They therefore cannot, by themselves, prove that oMLX still
    owns a loaded model. Keep markers live while their engine is loading or
    loaded, but demote every detached marker so a stale file (or a reused PID)
    cannot resurrect an unloaded model in the cluster tab.
    """

    loaded_deployments: set[str] = set()
    loading_deployments: set[str] = set()
    launchers: list[dict[str, Any]] = []
    live_metrics: dict[str, dict[str, Any]] = {}

    get_loaded = getattr(pool, "get_loaded_model_ids", None)
    loaded_ids = get_loaded() if callable(get_loaded) else []
    for model_id in loaded_ids:
        entry = pool.get_entry(model_id)
        status = getattr(getattr(entry, "engine", None), "cluster_status", None)
        if not callable(status):
            continue
        try:
            launcher = status() | {"model_id": model_id}
        except Exception:  # noqa: BLE001
            continue
        deployment_id = launcher.get("deployment_id")
        if isinstance(deployment_id, str) and deployment_id:
            loaded_deployments.add(deployment_id)
            get_live_metrics = getattr(entry.engine, "get_live_metrics", None)
            if callable(get_live_metrics):
                try:
                    snapshot = get_live_metrics()
                except Exception:  # noqa: BLE001
                    snapshot = None
                if isinstance(snapshot, dict) and isinstance(
                    snapshot.get("metrics"), dict
                ):
                    live_metrics[deployment_id] = snapshot["metrics"]
        launchers.append(launcher)

    get_model_ids = getattr(pool, "get_model_ids", None)
    if callable(get_model_ids):
        try:
            registry = get_cluster_registry()
            for model_id in get_model_ids():
                entry = pool.get_entry(model_id)
                if entry is None or not getattr(entry, "is_loading", False):
                    continue
                deployment = registry.get_for_model(entry.model_path)
                if deployment is not None:
                    loading_deployments.add(deployment.deployment_id)
        except (OSError, RuntimeError, ValueError):
            # Ownership remains fail-closed: an unresolvable loading entry does
            # not grant a marker permission to advertise a live model.
            pass

    for job in payload.get("jobs", []):
        deployment_id = job.get("deployment_id")
        if deployment_id in loaded_deployments:
            job["ownership"] = "loaded"
            if "metrics" not in job and deployment_id in live_metrics:
                # A reversed Phase split hosts decode telemetry on the peer.
                # Mirror the already-validated engine snapshot onto the local
                # coordinator row so the dashboard keeps end-to-end rates.
                job["metrics"] = live_metrics[deployment_id]
        elif deployment_id in loading_deployments:
            job["ownership"] = "loading"
        else:
            job["ownership"] = "detached"
            job["live"] = False
        for launcher in launchers:
            if deployment_id == launcher.get("deployment_id"):
                job["ranks"] = launcher.get("ranks", [])
                job["endpoint"] = launcher.get("endpoint")
                break
    payload["launchers"] = launchers


@router.get("/runtime")
async def cluster_runtime():
    """Return lightweight local rank markers for dashboard polling."""

    payload = await asyncio.to_thread(read_runtime_markers)
    if _get_engine_pool is None:
        return payload
    try:
        pool = _engine_pool()
    except HTTPException:
        return payload
    _reconcile_runtime_ownership(payload, pool)
    return payload


@router.get("/diagnostics")
async def cluster_diagnostics():
    """Downloadable, bounded evidence for one cluster support report.

    The bundle is deliberately read-only. It captures the same local status,
    runtime markers, launcher tails, approved plans, and staging progress that
    are otherwise spread across several panels, while removing credentials and
    shortening unbounded worker output before the browser downloads it.
    """

    errors: list[str] = []
    try:
        status = await asyncio.to_thread(collect_cluster_status)
        status_payload: dict[str, Any] | None = status.to_dict()
    except (OSError, RuntimeError, ValueError) as exc:
        status_payload = None
        errors.append(f"local status: {type(exc).__name__}: {exc}")

    try:
        runtime_payload = await cluster_runtime()
    except (OSError, RuntimeError, ValueError) as exc:
        runtime_payload = {"jobs": [], "launchers": []}
        errors.append(f"runtime: {type(exc).__name__}: {exc}")

    deployments: list[ClusterDeployment] = []
    registry_payload: dict[str, Any]
    try:
        registry = get_cluster_registry()
        deployments = list(registry.list())[:16]
        registry_payload = registry.to_dict()
    except RuntimeError as exc:
        registry_payload = {"schema_version": 1, "deployments": []}
        errors.append(f"registry: {type(exc).__name__}: {exc}")

    peer_health: list[dict[str, Any]] = []
    for deployment in deployments:
        try:
            health = await asyncio.to_thread(
                check_peers,
                {
                    rank: (host.node_id, host.ssh)
                    for rank, host in enumerate(deployment.hosts)
                },
                deployment_id=deployment.deployment_id,
                require_heartbeat=True,
            )
            peer_health.append(
                {
                    "deployment_id": deployment.deployment_id,
                    "healthy": all(item.healthy for item in health),
                    "summary": describe_failure(health),
                    "peers": [item.to_dict() for item in health],
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(
                f"peer health for {deployment.deployment_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    with _STAGING_JOBS_LOCK:
        staging_jobs = json.loads(
            json.dumps(list(_STAGING_JOBS.values())[-_MAX_STAGING_JOBS:])
        )
    incidents: list[dict[str, Any]] = []
    try:
        incidents = [
            incident.to_dict() for incident in get_cluster_incidents().list()
        ]
    except RuntimeError as exc:
        errors.append(f"incidents: {type(exc).__name__}: {exc}")
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "local-only cluster support bundle",
        "status": status_payload,
        "runtime": runtime_payload,
        "registry": registry_payload,
        "peer_health": peer_health,
        "staging_jobs": staging_jobs,
        "incidents": incidents,
        "errors": errors,
    }
    return _redact_diagnostic(payload)


@router.get("/incidents")
async def cluster_incidents(since: int = Query(default=0, ge=0)):
    """Return incidents after the caller's cursor, plus the new cursor.

    The ``since`` cursor makes monotonic merge a server-enforced property: a
    poll can only ever add records the browser has not seen, so no refresh can
    wipe error state. Dismissal (below) is the only removal path.
    """

    try:
        store = get_cluster_incidents()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    incidents = [incident.to_dict() for incident in store.list(since_seq=since)]
    for item in incidents:
        item["message"] = _redact_diagnostic(item["message"])
    return {
        "incidents": incidents,
        "latest_seq": store.latest_seq(),
        # Identity of the seq numbering. A corrupt-log reset restarts seq at
        # 1 under a new epoch; a client holding an old cursor must detect the
        # change and restart from 0 instead of going silent forever.
        "epoch": store.epoch,
    }


@router.post("/incidents/{incident_id}/dismiss")
async def dismiss_cluster_incident(incident_id: str):
    """Mark one incident dismissed — server-owned, so it survives reloads."""

    try:
        store = get_cluster_incidents()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not store.dismiss(incident_id):
        raise HTTPException(status_code=404, detail="Unknown incident.")
    return {"ok": True}


@router.get("/discover")
async def cluster_discover():
    """Return untrusted Bonjour peer suggestions without enrolling them."""

    return await asyncio.to_thread(discover_all_peers)


@router.post("/pairing-token")
async def cluster_pairing_token(request: ClusterPairingTokenRequest):
    """Generate a pairing token for QR code exchange.

    Advanced (legacy): step 1 of the 3-step copy-paste relay. The cluster v2
    wizard uses POST /api/cluster/pair/request + /pair/approve instead; this
    endpoint is kept working unchanged for the legacy flow.
    """

    from .discovery import generate_pairing_token

    try:
        token = generate_pairing_token(shared_secret=request.shared_secret)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"pairing_token": token}


@router.post("/verify-pairing-token")
async def cluster_verify_pairing_token(
    request: ClusterPairingTokenVerificationRequest,
):
    """Verify a pairing token received via QR code scan.

    Advanced (legacy): part of the 3-step copy-paste relay; superseded by the
    cluster v2 code-based pairing flow but kept working unchanged.
    """

    return {
        "valid": verify_pairing_token(
            request.token,
            shared_secret=request.shared_secret,
        )
    }


@router.post("/join-keys")
async def cluster_create_join_key(
    request: ClusterJoinKeyRequest,
    response: Response,
):
    """Create one expiring, single-use CUDA worker join command."""

    from .ssh_keys import get_or_create_ssh_key

    controller_url = _controller_url(request)
    try:
        source_digest = await asyncio.to_thread(worker_source_digest)
        key_pair = await asyncio.to_thread(get_or_create_ssh_key)
        join_key, record = get_cluster_enrollment().issue_join_key(
            controller_url=controller_url,
            source_digest=source_digest,
            ttl=request.ttl_seconds,
        )
    except (EnrollmentError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    command = build_cuda_join_command(
        controller_url=controller_url,
        join_key=join_key,
        controller_key_fingerprint=key_pair.fingerprint,
        source_digest=source_digest,
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return record | {
        "command": command,
        "controller_key_fingerprint": key_pair.fingerprint,
        "single_use": True,
    }


@router.get("/join-status")
async def cluster_join_status():
    """List pending commands and credential-free enrolled CUDA nodes."""

    try:
        return get_cluster_enrollment().to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/join-keys/{join_id}")
async def cluster_revoke_join_key(join_id: str):
    if not re.fullmatch(r"[a-f0-9]{16}", join_id):
        raise HTTPException(status_code=400, detail="invalid join-key ID")
    if not get_cluster_enrollment().revoke_join_key(join_id):
        raise HTTPException(status_code=404, detail="join key not found")
    return {"ok": True, "join_id": join_id}


@join_router.get("/bootstrap.py")
async def cluster_cuda_bootstrap_program():
    """Serve the bootstrap whose SHA-256 is pinned in the admin command."""

    return PlainTextResponse(
        cuda_bootstrap_program(),
        media_type="text/x-python; charset=utf-8",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@join_router.post("/claim")
async def cluster_claim_join_key(
    request: ClusterWorkerClaimRequest,
    response: Response,
    authorization: str | None = Header(default=None),
):
    """Consume a join key and return a short-lived source-download session."""

    from .ssh_keys import get_or_create_ssh_key

    raw_key = _join_bearer(authorization)
    addresses = _join_addresses(request.addresses)
    try:
        session_token, session = get_cluster_enrollment().claim(
            raw_key,
            node_id=request.node_id,
            hostname=request.hostname,
            ssh_user=request.ssh_user,
            ssh_port=request.ssh_port,
            addresses=addresses,
        )
        key_pair = await asyncio.to_thread(get_or_create_ssh_key)
    except (EnrollmentError, RuntimeError):
        # Bearer failures intentionally have one response so callers cannot
        # distinguish expired, revoked, and already-used credentials.
        raise HTTPException(
            status_code=401, detail="invalid enrollment credential"
        ) from None
    response.headers["Cache-Control"] = "no-store"
    return {
        "session_token": session_token,
        "session_expires_at": session.expires_at,
        "source_digest": session.source_digest,
        "controller_public_key": key_pair.public_key,
        "controller_key_fingerprint": key_pair.fingerprint,
    }


@join_router.get("/source")
async def cluster_cuda_worker_source(
    authorization: str | None = Header(default=None),
):
    """Download the exact controller source snapshot under a join session."""

    raw_session = _join_bearer(authorization)
    try:
        get_cluster_enrollment().authorize_session(raw_session)
    except (EnrollmentError, RuntimeError):
        raise HTTPException(
            status_code=401, detail="invalid enrollment credential"
        ) from None
    bundle = await asyncio.to_thread(worker_source_bundle)
    return Response(
        content=bundle,
        media_type="application/gzip",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'attachment; filename="omlx-cluster-worker.tar.gz"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@join_router.post("/complete")
async def cluster_complete_worker_join(
    request: ClusterWorkerCompleteRequest,
    authorization: str | None = Header(default=None),
):
    """Pin the worker's SSH identity and persist it without join secrets."""

    from .ssh_keys import pin_enrolled_host_key, ssh_public_key_fingerprint

    raw_session = _join_bearer(authorization)
    addresses = _join_addresses(request.addresses)
    try:
        session = get_cluster_enrollment().authorize_session(raw_session)
    except (EnrollmentError, RuntimeError):
        raise HTTPException(
            status_code=401, detail="invalid enrollment credential"
        ) from None
    try:
        observed_fingerprint = ssh_public_key_fingerprint(
            request.ssh_host_public_key
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="invalid worker SSH host key") from exc
    if not hmac.compare_digest(observed_fingerprint, request.ssh_host_fingerprint):
        raise HTTPException(status_code=400, detail="worker SSH fingerprint mismatch")
    if request.source_digest != session.source_digest:
        raise HTTPException(status_code=400, detail="worker source digest mismatch")
    if (
        request.node_id != session.node_id
        or request.hostname != session.hostname
        or request.ssh_user != session.ssh_user
        or request.ssh_port != session.ssh_port
        or addresses != session.addresses
    ):
        raise HTTPException(
            status_code=400,
            detail="worker identity changed after claiming the join key",
        )

    primary_address = addresses[0]
    ssh_target = f"{request.ssh_user}@{primary_address}"
    now = time.time()
    node = EnrolledNode(
        node_id=request.node_id,
        hostname=request.hostname,
        ssh=ssh_target,
        ssh_user=request.ssh_user,
        ssh_port=request.ssh_port,
        addresses=addresses,
        accelerator=request.accelerator,
        platform=request.platform,
        python_executable=request.python_executable,
        source_digest=request.source_digest,
        ssh_host_fingerprint=request.ssh_host_fingerprint,
        joined_at=now,
        last_seen_at=now,
    )
    try:
        await asyncio.to_thread(
            pin_enrolled_host_key,
            hostname=primary_address,
            public_key=request.ssh_host_public_key,
        )
        enrolled = get_cluster_enrollment().complete(raw_session, node)
    except (EnrollmentError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return enrolled.to_dict()


@router.get("/ssh-key")
async def cluster_ssh_key():
    """Get the current SSH key pair information."""

    from .ssh_keys import get_ssh_key_info

    return await asyncio.to_thread(get_ssh_key_info)


@router.post("/ssh-key/generate")
async def cluster_generate_ssh_key(
    overwrite: bool = Query(default=False),
):
    """Create the managed SSH key, rotating it only when explicitly requested."""

    from .ssh_keys import generate_ssh_key_pair

    try:
        key_pair = await asyncio.to_thread(
            generate_ssh_key_pair,
            overwrite=overwrite,
        )
        return {
            "success": True,
            "available": True,
            "key_type": key_pair.key_type,
            "fingerprint": key_pair.fingerprint,
            "public_key": key_pair.public_key,
            "private_key_path": str(key_pair.private_key_path),
            "public_key_path": str(key_pair.public_key_path),
            "created_at": key_pair.created_at,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ssh-key/exchange-token")
async def cluster_generate_key_exchange_token(
    request: ClusterKeyExchangeTokenRequest,
):
    """Generate a key exchange token for pairing with a peer.

    Advanced (legacy): step 3 of the copy-paste relay. Cluster v2 pairing
    (POST /api/cluster/pair/approve) drives the same SSH TOFU primitives
    programmatically; this endpoint remains for the manual flow.
    """

    from .ssh_keys import generate_key_exchange_for_peer

    try:
        node_id = validate_ssh_target(request.node_id)
        token = await asyncio.to_thread(
            generate_key_exchange_for_peer,
            node_id=node_id,
            shared_secret=request.shared_secret,
        )
        return {"exchange_token": token}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ssh-key/exchange")
async def cluster_exchange_keys(request: ClusterKeyExchangeRequest):
    """Exchange SSH keys with a peer using their exchange token.

    Advanced (legacy): step 3 of the copy-paste relay; kept working unchanged
    alongside the cluster v2 code-based pairing flow.
    """

    from .ssh_keys import exchange_keys_with_peer

    result = await asyncio.to_thread(
        exchange_keys_with_peer,
        peer_token=request.exchange_token,
        shared_secret=request.shared_secret,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/ssh-key/store-keychain")
async def cluster_store_key_in_keychain():
    """Store the SSH key fingerprint in macOS Keychain."""

    from .ssh_keys import store_key_in_keychain

    success = await asyncio.to_thread(store_key_in_keychain)
    return {"stored": success}


@router.get("/transports")
async def cluster_transports(hosts: str = Query(...)):
    """Detect available transports for the given cluster hosts.

    Returns transport info (TB4, TB5, Ethernet, RDMA) and the recommended backend.
    """

    host_list = _validated_ssh_targets(hosts)
    if not host_list:
        raise HTTPException(status_code=400, detail="at least one host is required")
    try:
        matrix = await asyncio.to_thread(detect_cluster_transports, host_list)
        # Cache the result so /discover can report transports without paying
        # for an SSH round trip on its own request path.
        record_peer_transports(matrix.transports)
        return {
            "transports": [t.__dict__ for t in matrix.transports],
            "backend": matrix.backend,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/peer-health")
async def cluster_peer_health(hosts: str = Query(...), deployment_id: str = ""):
    """Is every rank still answering?

    A collective cannot proceed without all of them, so a peer that has gone
    away should be visible as a stated failure rather than a hung request.
    """

    # Validate the complete optional ``user@host`` item before extracting the
    # hostname used in the human-readable health result. Passing an unchecked
    # leading-dash item to OpenSSH would let it be parsed as a client option.
    entries = _validated_ssh_targets(hosts)
    hosts_by_rank = {
        index: (item.split("@")[-1], item) for index, item in enumerate(entries)
    }
    health = await asyncio.to_thread(
        check_peers,
        hosts_by_rank,
        deployment_id=deployment_id,
        require_heartbeat=bool(deployment_id),
    )
    healthy = all(item.healthy for item in health)
    if deployment_id:
        with _PEER_HEALTH_LOCK:
            was_healthy = _PEER_HEALTH_LAST_HEALTHY.get(deployment_id)
            _PEER_HEALTH_LAST_HEALTHY[deployment_id] = healthy
        if was_healthy and not healthy:
            lost = next((item for item in health if not item.healthy), None)
            # to_thread: the store fsyncs while holding its lock, which must
            # not stall the event loop mid-stream for every other request.
            await asyncio.to_thread(
                _record_cluster_incident,
                Severity.ERROR,
                "peer_unhealthy",
                describe_failure(health),
                source=(
                    f"peer:{lost.node_id}"
                    if lost is not None and lost.node_id
                    else "coordinator"
                ),
                deployment_id=deployment_id,
            )
    return {
        "peers": [item.to_dict() for item in health],
        "healthy": healthy,
        "summary": describe_failure(health),
    }


@router.get("/link-status")
async def cluster_link_status(hosts: str = Query(...)):
    """Report whether the fabric is actually usable, and how to fix it if not.

    Device presence is not readiness: RDMA can be enabled but the port down, or
    active but unroutable. Each state has a different remedy and one of them
    needs administrator rights oMLX does not have, so the page has to say so
    rather than silently falling back to TCP.
    """

    host_list = _validated_ssh_targets(hosts)
    status = await asyncio.to_thread(assess_link, host_list)
    payload = status.to_dict()
    # The detail string embeds remote SSH stderr; the commands are pasteable
    # fixes whose user@host must survive, so only detail is redacted.
    payload["detail"] = _redact_diagnostic(payload["detail"])
    return payload


@router.post("/link-setup")
async def cluster_link_setup(request: ClusterLinkSetupRequest):
    """Authorize, configure, and verify the Thunderbolt fast path.

    This is intentionally narrow: the browser cannot supply shell commands or
    interface names. The server discovers the active RDMA ports and macOS owns
    the administrator credential prompt.
    """

    try:
        hosts = [validate_ssh_target(host) for host in request.hosts]
        status = await asyncio.to_thread(configure_link, hosts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LinkAuthorizationCancelledError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LinkSetupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return status.to_dict()


@router.post("/peer-probe")
async def cluster_peer_probe(request: ClusterPeerProbeRequest):
    """Probe a trusted known_hosts peer without changing either Mac."""

    try:
        return await asyncio.to_thread(
            probe_remote_host,
            request.ssh,
            route_to=request.route_to,
        )
    except DistributedLaunchError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/cuda-fabric/verify")
async def cluster_cuda_fabric_verify(request: ClusterCudaFabricVerifyRequest):
    """Prove a selected CUDA pair with an isolated NCCL direct-link test."""

    try:
        members = [
            host.model_copy(update={"ssh": validate_ssh_target(host.ssh.strip())})
            for host in request.hosts
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if len({host.ssh for host in members}) != 2:
        raise HTTPException(
            status_code=400,
            detail="CUDA fabric verification requires two distinct workers",
        )

    async def capability_for(host: ClusterCudaFabricMemberRequest) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(probe_remote_host, host.ssh)
        except (DistributedLaunchError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail=f"Could not inspect {host.node_id}: {exc}",
            ) from exc

    capabilities = await asyncio.gather(
        *(capability_for(host) for host in members)
    )
    probe_hosts: list[CudaFabricProbeHost] = []
    for host, capability in zip(members, capabilities):
        status = capability.get("status") or {}
        node = status.get("node") or {}
        runtime = status.get("runtime") or {}
        transport = status.get("transport") or {}
        rdma = transport.get("rdma") or {}
        if node.get("accelerator") != "cuda":
            raise HTTPException(
                status_code=409,
                detail=f"{host.node_id} did not report a CUDA accelerator",
            )
        if "nccl" not in set(node.get("distributed_backends") or []):
            raise HTTPException(
                status_code=409,
                detail=f"{host.node_id} did not report MLX NCCL support",
            )
        if node.get("fabric_kind") != "connectx-7":
            raise HTTPException(
                status_code=409,
                detail=f"{host.node_id} did not report a ConnectX interface",
            )
        addresses = rdma.get("addresses") or {}
        interfaces = rdma.get("network_interfaces") or {}
        endpoints = [
            (
                str(addresses[device]),
                str(interfaces[device]),
                str(device),
            )
            for device in rdma.get("devices") or []
            if addresses.get(device) and interfaces.get(device)
        ]
        if not endpoints:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{host.node_id} has ConnectX hardware but no direct-link IP. "
                    "Assign the ConnectX link in the device network settings, "
                    "then press Verify again."
                ),
            )
        try:
            probe_hosts.append(
                CudaFabricProbeHost(
                    node_id=host.node_id,
                    ssh=host.ssh,
                    ips=tuple(endpoint[0] for endpoint in endpoints),
                    interfaces=tuple(endpoint[1] for endpoint in endpoints),
                    rdma_devices=tuple(endpoint[2] for endpoint in endpoints),
                    python_executable=runtime.get("python_executable") or None,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        return await asyncio.to_thread(
            run_cuda_fabric_probe,
            (probe_hosts[0], probe_hosts[1]),
        )
    except DistributedLaunchError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/worker-smoke")
async def cluster_worker_smoke(
    timeout: float = Query(default=5.0, gt=0.0, le=30.0),
):
    """Exercise the isolated worker lifecycle without initializing MLX/JACCL."""

    try:
        return await asyncio.to_thread(run_worker_smoke, timeout=timeout)
    except (OSError, RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/collective-smoke")
async def cluster_collective_smoke(
    timeout: float = Query(default=20.0, gt=0.0, le=60.0),
):
    """Run a two-rank loopback MLX collective without loading a model."""

    try:
        return await asyncio.to_thread(run_local_collective_smoke, timeout=timeout)
    except (CollectiveSmokeError, OSError, RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/pipeline-smoke")
async def cluster_pipeline_smoke(
    timeout: float = Query(default=30.0, gt=0.0, le=90.0),
):
    """Run an unequal two-rank Nemotron-H graph without model weights."""

    try:
        return await asyncio.to_thread(run_local_pipeline_smoke, timeout=timeout)
    except (CollectiveSmokeError, OSError, RuntimeError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _tp_layout_candidate_clears_promotion(
    profile: str,
    equal: ClusterTPRateEvidenceRequest,
    candidate: ClusterTPRateEvidenceRequest,
) -> tuple[bool, str]:
    """Require a useful gain without hiding a material regression."""

    prefill_ratio = (
        candidate.prefill_tokens_per_second / equal.prefill_tokens_per_second
    )
    decode_ratio = (
        candidate.decode_tokens_per_second / equal.decode_tokens_per_second
    )
    if profile == "throughput":
        accepted = prefill_ratio >= 1.03 and decode_ratio >= 0.95
    elif profile == "interactive":
        accepted = decode_ratio >= 1.03 and prefill_ratio >= 0.95
    else:
        accepted = (
            min(prefill_ratio, decode_ratio) >= 0.98
            and math.sqrt(prefill_ratio * decode_ratio) >= 1.02
        )
    reason = (
        f"prefill {prefill_ratio:.4f}x, decode {decode_ratio:.4f}x "
        f"under the {profile} policy"
    )
    return accepted, reason


@router.get("/tp-layout-qualifications")
async def list_tp_layout_qualifications() -> dict[str, Any]:
    """Return exact-match heterogeneous TP records and store health."""

    try:
        return get_tp_layout_qualification_store().to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/tp-layout-qualifications")
async def qualify_tp_layout(
    request: ClusterTPLayoutQualificationRequest,
) -> dict[str, Any]:
    """Promote one matched full-model A/B into the signed layout store."""

    if request.tensor_parallel_size != len(request.nodes):
        raise HTTPException(
            status_code=400,
            detail="TP layout qualification currently requires pure TP on every node",
        )
    node_ids = [node.node_id.strip() for node in request.nodes]
    host_ids = [host.node_id.strip() for host in request.hosts]
    if host_ids != node_ids:
        raise HTTPException(
            status_code=400,
            detail="qualification hosts must exactly match rank-ordered nodes",
        )
    if len(request.shard_weights) != request.tensor_parallel_size:
        raise HTTPException(
            status_code=400,
            detail="qualification shard vector does not match TP size",
        )
    exact = request.equal_control.output_sha256 == request.candidate.output_sha256
    accepted, performance_reason = _tp_layout_candidate_clears_promotion(
        request.execution_profile,
        request.equal_control,
        request.candidate,
    )
    promotable = bool(exact and accepted)
    if exact:
        parity_sha256 = request.candidate.output_sha256
        evidence_reason = performance_reason
    else:
        parity_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "equal": request.equal_control.output_sha256,
                    "candidate": request.candidate.output_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        evidence_reason = (
            "rejected because the candidate output hash differs from the "
            f"matched equal control; {performance_reason}"
        )
    if exact and not accepted:
        evidence_reason = (
            f"rejected by TP promotion policy; {performance_reason}"
        )

    try:
        _validate_cluster_hosts(request.hosts)
        statuses = await asyncio.to_thread(
            _qualification_statuses,
            request.hosts,
        )
        key = _tp_qualification_key(
            model_path=request.model_path,
            nodes=request.nodes,
            statuses=statuses,
            backend=request.backend,
            tensor_parallel_size=request.tensor_parallel_size,
            target_context_tokens=request.target_context_tokens,
            execution_profile_name=request.execution_profile,
            auto_tune=request.auto_tune,
            sampling_rank_only=request.sampling_rank_only,
            mtp_enabled=request.mtp_enabled,
            mtp_num_draft_tokens=request.mtp_num_draft_tokens,
        )
        model, budgets = _model_and_nodes(
            ClusterPlanRequest(
                model_path=request.model_path,
                nodes=request.nodes,
                execution_profile=request.execution_profile,
                tensor_parallel_size=request.tensor_parallel_size,
                target_context_tokens=request.target_context_tokens,
                mtp_enabled=request.mtp_enabled,
                mtp_num_draft_tokens=request.mtp_num_draft_tokens,
            )
        )
        # Validate adapter shard units and per-node memory fit before storing
        # evidence. The exact record will be revalidated again at every use.
        plan_hybrid(
            model,
            budgets,
            tensor_parallel_size=request.tensor_parallel_size,
            workload_profile=request.execution_profile,
            context_tokens=request.target_context_tokens,
            qualified_tensor_shard_weights=(tuple(request.shard_weights),),
        )
        record = TPLayoutQualification(
            key=key,
            shard_weights=tuple(request.shard_weights),
            equal_control=TPRateEvidence(
                request.equal_control.prefill_tokens_per_second,
                request.equal_control.decode_tokens_per_second,
                request.equal_control.samples,
            ),
            candidate=TPRateEvidence(
                request.candidate.prefill_tokens_per_second,
                request.candidate.decode_tokens_per_second,
                request.candidate.samples,
            ),
            exact=exact,
            parity_sha256=parity_sha256,
            promotable=promotable,
            reason=f"{request.reason.strip()}; {evidence_reason}",
            qualified_at=datetime.now(UTC).isoformat(),
        )
        store = get_tp_layout_qualification_store()
        await asyncio.to_thread(store.record, record)
    except HTTPException:
        raise
    except (OSError, PlanningError, RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "ok": True,
        "state": "qualified" if record.promotable else "rejected",
        "qualification_id": record.qualification_id,
        "record_digest": record.record_digest,
        "shard_weights": list(record.shard_weights),
        "exact": record.exact,
        "promotable": record.promotable,
        "reason": record.reason,
    }


@router.post("/plan")
async def cluster_plan(request: ClusterPlanRequest):
    """Plan explicit contiguous layers across rank-ordered node budgets."""

    try:
        plan = await asyncio.to_thread(_create_cluster_plan, request)
    except (OSError, PlanningError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # The signature travels back with the plan so activation can prove it is
    # launching the thing that was shown here, and not a re-plan built from a
    # payload that quietly dropped the reserve, the cap or the role.
    payload = plan.to_dict()
    if request.mtp_enabled or request.mtp_num_draft_tokens is not None:
        payload.update(
            mtp_enabled=request.mtp_enabled,
            mtp_num_draft_tokens=request.mtp_num_draft_tokens,
        )
    if request.prompt_cache_ssd is not None:
        payload["prompt_cache_ssd"] = request.prompt_cache_ssd
        if request.prompt_cache_ssd:
            payload["prompt_cache_ssd_max_bytes"] = int(
                request.prompt_cache_ssd_max_bytes
                or DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES
            )
    return _plan_with_signature(payload)


class ClusterBackendMemberRequest(BaseModel):
    """One member's observed RDMA capability, from local or peer probes."""

    node_id: str = Field(min_length=1, max_length=128)
    rdma_ctl_enabled: bool = False
    rdma_devices: list[str] = Field(default_factory=list, max_length=16)


class ClusterBackendSelectionRequest(BaseModel):
    """Which collective backend should this exact member set use?"""

    model_config = ConfigDict(extra="forbid")

    members: list[ClusterBackendMemberRequest] = Field(min_length=2, max_length=64)


@router.post("/backend-selection")
async def cluster_backend_selection(
    request: ClusterBackendSelectionRequest,
) -> dict[str, Any]:
    """jaccl when rdma_ctl is enabled on ALL members, else the TCP ring.

    The plan view renders this decision (including which members block JACCL)
    before anyone approves a placement; the replan endpoint makes the same
    call when asked for ``backend="auto"``.
    """

    try:
        selection = select_cluster_backend(
            tuple(
                MemberFabric(
                    node_id=member.node_id.strip(),
                    rdma_ctl_enabled=member.rdma_ctl_enabled,
                    rdma_devices=tuple(member.rdma_devices),
                )
                for member in request.members
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return selection.to_dict()


def _deployment_id(model_path: Path, plan_hash: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_path.name).strip("-._")
    return f"{slug or 'model'}-{plan_hash[:12]}"


def _execution_for_request(
    request: Any,
    assignments: Any,
    *,
    backend: str,
):
    """Resolve the execution contract identically for preview and launch."""

    requested = execution_profile(
        request.execution_profile,
        auto_tune=request.auto_tune,
        sampling_rank_only=request.sampling_rank_only,
    )
    deepseek_ane_request = getattr(request, "deepseek_ane_prefill", None)
    deepseek_ane = DeepseekAnePrefillSettings.from_dict(
        deepseek_ane_request.model_dump()
        if deepseek_ane_request is not None
        else None
    )
    distributed = len(assignments) > 1
    experimental_distributed_ane = os.environ.get(
        _EXPERIMENTAL_DISTRIBUTED_DSV4_ANE_ENV,
        "0",
    ).strip().lower() in {"1", "true", "on", "yes"}
    distributed_ane_rejected = bool(
        deepseek_ane.enabled
        and distributed
        and not experimental_distributed_ane
    )
    if distributed_ane_rejected:
        # PR #3059 is physically qualified only on a single M3 Ultra. TP2
        # uses a lockstep 1024-token DS4 kernel for long prompts; a 4096-tile
        # provider never dispatches and adds fallback overhead, while a 1024
        # provider made the heterogeneous ranks miss a collective deadline.
        # Keep the normal cluster path exact and fast until a model/runtime/
        # topology qualification proves one fixed shape on every rank.
        deepseek_ane = DeepseekAnePrefillSettings()
    requested = replace(
        requested,
        async_overlap=request.async_overlap,
        cache_affinity=request.cache_affinity,
        prompt_cache_ssd=bool(getattr(request, "prompt_cache_ssd", False)),
        prompt_cache_ssd_max_bytes=(
            getattr(request, "prompt_cache_ssd_max_bytes", None)
            or requested.prompt_cache_ssd_max_bytes
        ),
        # The context chosen beside the model is both a reservation and a
        # runtime ceiling. Without this fallback the planner could reserve
        # 256k while the server used an unrelated advanced default (or no
        # bound at all), making the memory promise on screen untrue.
        max_kv_size=(
            request.max_kv_size
            or getattr(request, "target_context_tokens", None)
        ),
        ring_connections_per_ip=(
            request.ring_connections_per_ip or requested.ring_connections_per_ip
        ),
        prefill_step_size=(
            deepseek_ane.sequence_length
            if deepseek_ane.enabled
            else requested.prefill_step_size
        ),
        deepseek_ane_prefill=deepseek_ane,
    )
    tuned = tune_execution_settings(
        requested,
        assignments,
        backend=backend,
    )
    if distributed_ane_rejected:
        tuned = replace(
            tuned,
            tuning_reason=(
                f"{tuned.tuning_reason}; DeepSeek ANE disabled because "
                "distributed fixed-shape offload is not lockstep-qualified"
            ),
        )
    if (
        tuned.deepseek_ane_prefill.enabled
        and tuned.prefill_step_size != tuned.deepseek_ane_prefill.sequence_length
    ):
        tuned = replace(
            tuned,
            deepseek_ane_prefill=DeepseekAnePrefillSettings(),
            tuning_reason=(
                f"{tuned.tuning_reason}; DeepSeek ANE disabled because the "
                "memory tuner reduced the prefill step below its fixed tile"
            ),
        )
    return tuned


def _request_performance_profiles(
    nodes: list[ClusterPlanNodeRequest],
) -> tuple[NodePerformanceProfile, ...]:
    """Return a complete, rank-ordered measured profile set or no set."""

    if not nodes or any(node.performance is None for node in nodes):
        return ()
    profiles = tuple(
        NodePerformanceProfile.from_dict(node.performance or {}) for node in nodes
    )
    for rank, (node, profile) in enumerate(zip(nodes, profiles)):
        if profile.rank != rank or profile.node_id != node.node_id.strip():
            raise ValueError(
                "node performance profiles must match the activation rank order"
            )
    if any(not profile.promotable for profile in profiles):
        return ()
    return profiles


def _create_deployment(
    request: ClusterDeploymentRequest,
) -> tuple[ClusterDeployment, dict[str, Any]]:
    source = (request.model_source or "").strip()
    model_path = (
        Path(request.model_path)
        if source and not _local_ssh_target(source)
        else Path(request.model_path).expanduser().resolve()
    )
    requested_microbatch = execution_profile(
        request.execution_profile,
        auto_tune=request.auto_tune,
        sampling_rank_only=request.sampling_rank_only,
    ).pipeline_microbatch_size
    plan_request = ClusterPlanRequest(
        model_path=str(model_path),
        model_source=request.model_source,
        model_source_python=request.model_source_python,
        nodes=request.nodes,
        execution_profile=request.execution_profile,
        allocation=request.allocation,
        pipeline_microbatch_size=requested_microbatch,
        tensor_parallel_size=request.tensor_parallel_size,
        serving_mode=request.serving_mode,
        prefill_rank=request.prefill_rank,
        decode_rank=request.decode_rank,
        target_context_tokens=request.target_context_tokens,
        mtp_enabled=request.mtp_enabled,
        mtp_num_draft_tokens=request.mtp_num_draft_tokens,
        prompt_cache_ssd=request.prompt_cache_ssd,
        prompt_cache_ssd_max_bytes=request.prompt_cache_ssd_max_bytes,
        path_map=request.path_map,
    )
    qualified_tensor_shard_weights = None
    qualification_provenance = None
    if request.tp_qualification_id is not None:
        try:
            statuses = _qualification_statuses(request.hosts)
        except (DistributedLaunchError, OSError, RuntimeError, ValueError) as exc:
            raise PlanningError(
                f"could not revalidate the approved TP qualification: {exc}"
            ) from exc
        (
            qualified_tensor_shard_weights,
            qualification_provenance,
            qualification_decision,
        ) = _resolve_tp_layout_qualification(
            model_path=str(model_path),
            nodes=request.nodes,
            statuses=statuses,
            backend=request.backend,
            tensor_parallel_size=request.tensor_parallel_size,
            target_context_tokens=request.target_context_tokens,
            execution_profile_name=request.execution_profile,
            auto_tune=request.auto_tune,
            sampling_rank_only=request.sampling_rank_only,
            mtp_enabled=request.mtp_enabled,
            mtp_num_draft_tokens=request.mtp_num_draft_tokens,
            expected_qualification_id=request.tp_qualification_id,
        )
        if qualification_provenance is None:
            raise PlanningError(str(qualification_decision["reason"]))
    plan = (
        _create_cluster_plan(
            plan_request,
            qualified_tensor_shard_weights=qualified_tensor_shard_weights,
            tensor_parallel_qualification=qualification_provenance,
        )
        if qualification_provenance is not None
        else _create_cluster_plan(plan_request)
    )
    execution = _execution_for_request(
        request,
        plan.assignments,
        backend=request.backend,
    )
    if (
        execution.pipeline_microbatch_size != requested_microbatch
    ):
        plan_request.pipeline_microbatch_size = execution.pipeline_microbatch_size
        plan = (
            _create_cluster_plan(
                plan_request,
                qualified_tensor_shard_weights=qualified_tensor_shard_weights,
                tensor_parallel_qualification=qualification_provenance,
            )
            if qualification_provenance is not None
            else _create_cluster_plan(plan_request)
        )
    if len(request.hosts) != len(request.nodes):
        raise ValueError("host count must match node budget count")
    node_ids = [node.node_id.strip() for node in request.nodes]
    host_ids = [host.node_id.strip() for host in request.hosts]
    if host_ids != node_ids:
        raise ValueError("rank-ordered host IDs must match node budget IDs")

    host_rdma: list[Any] = [tuple(host.rdma) for host in request.hosts]
    if request.backend != "ring" and any(not rdma for rdma in host_rdma):
        # Clients (the cluster v2 wizard among them) legitimately activate
        # without RDMA rows — they know the link exists, not the interface
        # names, and only the live names survive macOS renumbering a
        # Thunderbolt port. Derive the full matrix here, the same way the
        # transports report does, or the ClusterDeployment constructor
        # rejects the activation with nothing the user can act on.
        interfaces = [probe_host_interfaces(host.ssh) for host in request.hosts]
        matrix = build_rdma_matrix(interfaces)
        if not matrix.ok:
            raise ValueError(
                f"cannot use the {request.backend} backend: {matrix.reason}"
            )
        host_rdma = [list(row) for row in matrix.rows]

    deployment = ClusterDeployment(
        deployment_id=(
            request.deployment_id.strip()
            if request.deployment_id
            else _deployment_id(model_path, plan.plan_hash)
        ),
        model=str(model_path),
        backend=request.backend,
        hosts=tuple(
            ClusterHost(
                node_id=host.node_id.strip(),
                ssh=host.ssh,
                ips=tuple(host.ips),
                rdma=tuple(host_rdma[index]),
                python_executable=host.python_executable,
            )
            for index, host in enumerate(request.hosts)
        ),
        assignments=plan.assignments,
        plan_hash=plan.plan_hash,
        execution=execution,
        performance_profiles=_request_performance_profiles(request.nodes),
        tensor_parallel_size=request.tensor_parallel_size,
        target_context_tokens=request.target_context_tokens,
        mtp_enabled=request.mtp_enabled,
        mtp_num_draft_tokens=request.mtp_num_draft_tokens,
        serving_mode=request.serving_mode,
        prefill_rank=request.prefill_rank,
        decode_rank=request.decode_rank,
        path_map=validate_model_path_map(request.path_map, tuple(host_ids)),
        tensor_parallel_qualification=getattr(
            plan, "tensor_parallel_qualification", None
        ),
    )
    plan_payload = plan.to_dict()
    if request.mtp_enabled or request.mtp_num_draft_tokens is not None:
        plan_payload.update(
            mtp_enabled=request.mtp_enabled,
            mtp_num_draft_tokens=request.mtp_num_draft_tokens,
        )
    if request.prompt_cache_ssd is not None:
        plan_payload["prompt_cache_ssd"] = request.prompt_cache_ssd
        if request.prompt_cache_ssd:
            plan_payload["prompt_cache_ssd_max_bytes"] = int(
                request.prompt_cache_ssd_max_bytes
                or DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES
            )
    return deployment, plan_payload


def _build_performance_plan(
    model: Any,
    nodes: list[NodeBudget],
    *,
    model_path: str,
    tensor_parallel_size: int,
    workload_profile: str,
    microbatch_size: int,
    context_tokens: int,
) -> ShardPlan:
    """Build a shard plan using the hybrid planner when tensor parallelism is active."""

    if tensor_parallel_size > 1:
        return plan_hybrid(
            model,
            nodes,
            tensor_parallel_size=tensor_parallel_size,
            workload_profile=workload_profile,
            microbatch_size=microbatch_size,
            context_tokens=context_tokens,
            qualified_tensor_shard_weights=(
                _operator_qualified_tp_shard_weights(
                    tensor_parallel_size=tensor_parallel_size,
                    node_count=len(nodes),
                    model_path=model_path,
                )
            ),
        )
    return plan_unequal_pipeline(
        model,
        nodes,
        workload_profile=workload_profile,
        microbatch_size=microbatch_size,
        context_tokens=context_tokens,
    )


def _performance_optimized_deployment(
    deployment: ClusterDeployment,
    request: ClusterDeploymentRequest,
    report: dict[str, Any],
) -> tuple[ClusterDeployment, dict[str, Any]]:
    profiles = tuple(
        NodePerformanceProfile.from_dict(profile)
        for profile in report.get("profiles", [])
    )
    if len(profiles) != deployment.world_size:
        raise ValueError("performance probe did not return every cluster rank")
    source = (request.model_source or "").strip()
    model = (
        remote_model_layout(
            validate_ssh_target(source),
            deployment.model,
            python_executable=request.model_source_python,
        )
        if source
        and source not in {LOCAL_NODE, "127.0.0.1", "localhost", "::1"}
        else inspect_safetensors_layout(deployment.model)
    )
    # Same budgets the approved plan was built from — reserve, split cap and
    # role included. Re-deriving them here without the cap and the role is how
    # a capped, Workstation-marked Mac came out of the probe holding almost its
    # whole ceiling.
    nodes = _node_budgets(request.nodes, profiles=profiles)
    plan = _build_performance_plan(
        model,
        nodes,
        model_path=deployment.model,
        tensor_parallel_size=deployment.tensor_parallel_size,
        workload_profile=deployment.execution.profile,
        microbatch_size=deployment.execution.pipeline_microbatch_size,
        context_tokens=request.target_context_tokens,
    )
    execution = tune_execution_settings(
        deployment.execution,
        plan.assignments,
        backend=deployment.backend,
    )
    if (
        execution.pipeline_microbatch_size
        != deployment.execution.pipeline_microbatch_size
    ):
        plan = _build_performance_plan(
            model,
            nodes,
            model_path=deployment.model,
            tensor_parallel_size=deployment.tensor_parallel_size,
            workload_profile=execution.profile,
            microbatch_size=execution.pipeline_microbatch_size,
            context_tokens=request.target_context_tokens,
        )
    deployment_id = (
        request.deployment_id.strip()
        if request.deployment_id
        else _deployment_id(Path(deployment.model), plan.plan_hash)
    )
    return (
        replace(
            deployment,
            deployment_id=deployment_id,
            assignments=plan.assignments,
            plan_hash=plan.plan_hash,
            execution=execution,
            performance_profiles=profiles,
            target_context_tokens=request.target_context_tokens,
        ),
        plan.to_dict(),
    )


class ClusterCatalogueModelRequest(BaseModel):
    """One deduplicated model and the Mac that owns its complete copy."""

    id: str = Field(min_length=1, max_length=512)
    model_path: str = Field(min_length=1, max_length=4096)
    model_source: str = Field(default="127.0.0.1", min_length=1, max_length=255)
    model_source_python: str | None = Field(default=None, max_length=4096)
    source_node_id: str = Field(default="", max_length=128)
    model_context_length: int | None = Field(default=None, gt=0)


class ClusterCatalogueRequest(BaseModel):
    """Which of these models will run on this cluster?"""

    nodes: list[ClusterPlanNodeRequest] = Field(min_length=1, max_length=64)
    model_paths: list[str] = Field(default_factory=list, max_length=256)
    models: list[ClusterCatalogueModelRequest] = Field(
        default_factory=list, max_length=256
    )
    model_dir: str | None = Field(default=None, max_length=4096)
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"


class ClusterInventoryHostRequest(BaseModel):
    """A selected worker whose local oMLX model inventory should be included."""

    node_id: str = Field(min_length=1, max_length=128)
    ssh: str = Field(min_length=1, max_length=255)
    python_executable: str | None = Field(default=None, max_length=4096)


class ClusterModelInventoryRequest(BaseModel):
    hosts: list[ClusterInventoryHostRequest] = Field(min_length=1, max_length=64)


class ClusterNodeBudgetHostRequest(BaseModel):
    """A Mac whose memory ceiling is measured over SSH.

    Collective addresses do not belong here. Requiring ``ips`` before the
    fabric was configured made the first memory measurement fail validation,
    leaving the dashboard's old 64 GiB placeholder in place.
    """

    node_id: str = Field(min_length=1, max_length=128)
    ssh: str = Field(min_length=1, max_length=255)
    python_executable: str | None = Field(default=None, max_length=4096)


class ClusterNodeBudgetRequest(BaseModel):
    """Ask each Mac what it can actually offer, rather than assuming."""

    hosts: list[ClusterNodeBudgetHostRequest] = Field(min_length=1, max_length=64)
    roles: dict[str, str] = Field(default_factory=dict)


@router.get("/node-roles")
async def cluster_node_roles() -> dict[str, Any]:
    """The roles a node can take, with the reasoning behind each."""

    from .node_role import DEFAULT_ROLE, ROLES

    return {
        "default": DEFAULT_ROLE,
        "roles": [
            {
                "key": role.key,
                "label": role.label,
                "summary": role.summary,
                "detail": role.detail,
                "reserve_bytes": role.reserve_bytes,
                # Clients rendering a usable budget (the cluster v2 wizard)
                # compute reserve_for() from this pair rather than mirroring
                # the constants by hand.
                "reserve_fraction": role.reserve_fraction,
            }
            for role in ROLES.values()
        ],
    }


@router.post("/node-budgets")
async def cluster_node_budgets(request: ClusterNodeBudgetRequest) -> dict[str, Any]:
    """What each Mac should contribute, measured on the machine itself.

    Reads the live oMLX admission ceiling per node rather than installed RAM:
    a 256 GiB Studio can have a ~223 GiB MLX working set, and current unified
    memory pressure can lower that further. A plan built on the larger number
    is refused by the memory guard at load.
    """

    from .node_role import suggest_budget

    try:
        hosts = [
            host.model_copy(
                update={"ssh": validate_ssh_target(host.ssh.strip())}
            )
            for host in request.hosts
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _for(host: Any) -> dict[str, Any]:
        capacity_bytes = 0
        capacity_source: str | None = None
        try:
            if not _local_ssh_target(host.ssh):
                capacity_bytes = await asyncio.to_thread(
                    probe_remote_admission_ceiling,
                    host.ssh,
                    # No fallback to sys.executable: inside the packaged app that
                    # is a bundled interpreter which exists on the peer but cannot
                    # import oMLX, so every poll 503'd (#2680). Unknown means the
                    # probe discovers the peer's own interpreter.
                    python_executable=host.python_executable,
                )
                capacity_source = "admission_ceiling"
            budget = await asyncio.to_thread(
                suggest_budget,
                role=request.roles.get(host.node_id, "headless"),
                ssh_target=host.ssh,
                capacity_bytes=capacity_bytes,
                capacity_source=capacity_source,
            )
        except (DistributedLaunchError, OSError, RuntimeError, ValueError) as exc:
            # One enrolled Mac that no longer runs oMLX must not fail the
            # measurement of every other node: a whole-request 503 made the
            # legacy dashboard retry on every poll. The node is reported in
            # place, marked unusable with the reason, and the caller's skip of
            # zero-capacity rows does the rest.
            return {
                "node_id": host.node_id,
                "ssh": host.ssh,
                "capacity_bytes": 0,
                "reserve_bytes": 0,
                "usable_bytes": 0,
                "role": request.roles.get(host.node_id, "headless"),
                "capacity_source": "unavailable",
                "summary": "",
                "unusable": True,
                "error": str(exc),
            }
        return {"node_id": host.node_id, "ssh": host.ssh, **budget.to_dict()}

    nodes = list(await asyncio.gather(*(_for(host) for host in hosts)))
    return {"nodes": nodes}


def _local_ssh_target(value: str) -> bool:
    return value.strip() in {LOCAL_NODE, "127.0.0.1", "localhost", "::1"}


@router.post("/models")
async def cluster_models(request: ClusterModelInventoryRequest) -> dict[str, Any]:
    """Union the downloaded models on every selected Mac.

    A shared model appears once with every location named.  The largest copy is
    retained as the planning/staging source, which makes a complete Studio copy
    win over a coordinator directory containing only an old pipeline stage.
    """

    async def read_host(
        host: ClusterInventoryHostRequest,
    ) -> tuple[str, str, list[dict[str, Any]], str]:
        target = host.ssh.strip()
        if _local_ssh_target(target):
            try:
                models = [
                    dict(model, python_executable=sys.executable)
                    for model in engine_pool_model_inventory(_engine_pool())
                ]
                return (
                    host.node_id,
                    "127.0.0.1",
                    models,
                    "",
                )
            except (HTTPException, RuntimeError, ValueError) as exc:
                return host.node_id, "127.0.0.1", [], str(exc)
        try:
            validated = validate_ssh_target(target)
            # An unknown interpreter is discovered on the peer (and cached
            # per host), never assumed from one machine's checkout layout.
            source_python = host.python_executable or await asyncio.to_thread(
                resolve_remote_python, validated
            )
            models = await asyncio.to_thread(
                remote_model_inventory,
                validated,
                python_executable=source_python,
            )
            models = [
                dict(model, python_executable=source_python) for model in models
            ]
            return host.node_id, validated, models, ""
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            return host.node_id, target, [], str(exc)

    results = await asyncio.gather(*(read_host(host) for host in request.hosts))
    inventories = [
        (node_id, ssh_target, models)
        for node_id, ssh_target, models, _ in results
    ]
    errors = [
        {"node_id": node_id, "ssh": ssh_target, "detail": error}
        for node_id, ssh_target, _, error in results
        if error
    ]
    models = merge_model_inventories(inventories)
    models.sort(
        key=lambda model: (
            -int(model.get("estimated_size") or 0),
            str(model.get("display_name") or model.get("id") or "").lower(),
        )
    )
    return {
        "models": models,
        "model_count": len(models),
        "nodes": [
            {
                "node_id": node_id,
                "ssh": ssh_target,
                "model_count": len(host_models),
            }
            for node_id, ssh_target, host_models, _ in results
        ],
        "errors": errors,
    }


def _requested_links_fast(nodes: list[ClusterPlanNodeRequest]) -> bool:
    """Whether every requested peer sits on a fast (JACCL/Thunderbolt) link.

    The catalogue's default tie-break prefers pipeline at equal width because
    it survives slow links; when the paired registry shows every requested
    peer is reachable over JACCL/Thunderbolt RDMA, tensor parallelism is the
    better recommendation — it splits per-token compute across the group.
    The coordinator's own row is not in the paired registry, so only
    non-self node_ids are consulted. Unknown or unpaired peers fail closed:
    pipeline stays the recommendation where tensor would crawl.
    """

    if len(nodes) < 2:
        return False
    try:
        identity = get_node_identity()
        registry = get_device_registry()
    except RuntimeError:
        return False
    for node in nodes:
        if node.node_id == identity.node_id:
            continue
        if not registry.is_paired(node.node_id):
            return False
        record = registry.get(node.node_id) or {}
        caps = record.get("caps")
        if not isinstance(caps, dict):
            return False
        if not (caps.get("jaccl") or caps.get("thunderbolt")):
            return False
    return True


def _catalogue_for_candidates(
    candidates: list[ClusterCatalogueModelRequest],
    nodes: list[NodeBudget],
    *,
    workload_profile: str,
    prefer_tensor: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        source = candidate.model_source.strip()
        try:
            layout = (
                complete_model_layout(candidate.model_path)
                if _local_ssh_target(source)
                else remote_model_layout(
                    validate_ssh_target(source),
                    candidate.model_path,
                    python_executable=candidate.model_source_python,
                )
            )
            fit = assess_model(
                layout,
                nodes,
                model_id=candidate.id,
                declared_context_tokens=candidate.model_context_length,
                workload_profile=workload_profile,
                prefer_tensor=prefer_tensor,
            )
        except (OSError, PlanningError, RuntimeError, ValueError) as exc:
            fit = ModelFit(
                model_id=candidate.id,
                weight_bytes=0,
                fits=False,
                reason=f"could not read the complete model: {exc}",
                model_path=candidate.model_path,
                failure_kind="model_unreadable",
            )
        rows.append(
            fit.to_dict()
            | {
                "model_path": candidate.model_path,
                "model_source": (
                    "127.0.0.1" if _local_ssh_target(source) else source
                ),
                "source_node_id": candidate.source_node_id,
            }
        )
    runnable = sorted(
        (row for row in rows if row["fits"]),
        key=lambda row: -int(row["weight_bytes"]),
    )
    rejected = sorted(
        (row for row in rows if not row["fits"]),
        key=lambda row: int(row["weight_bytes"]),
    )
    return runnable + rejected


@router.post("/catalogue")
async def cluster_catalogue(request: ClusterCatalogueRequest) -> dict[str, Any]:
    """Every model this cluster can run, largest first, with its context limit.

    The verdicts come from the same planner activation uses, so a model listed
    as fitting is one the cluster will actually load.
    """

    nodes = _node_budgets(request.nodes)
    prefer_tensor = _requested_links_fast(request.nodes)

    paths = [Path(item) for item in request.model_paths]
    if request.model_dir:
        root = Path(request.model_dir).expanduser()
        try:
            paths.extend(child for child in sorted(root.iterdir()) if child.is_dir())
        except OSError as exc:
            raise HTTPException(
                status_code=400, detail=f"could not read {root}: {exc}"
            ) from exc
    if not paths and not request.models:
        raise HTTPException(
            status_code=400, detail="provide models, model_paths or model_dir"
        )

    if request.models:
        model_rows = await asyncio.to_thread(
            _catalogue_for_candidates,
            request.models,
            nodes,
            workload_profile=request.execution_profile,
            prefer_tensor=prefer_tensor,
        )
    else:
        catalogue = await asyncio.to_thread(
            catalogue_for_cluster,
            paths,
            nodes,
            workload_profile=request.execution_profile,
            prefer_tensor=prefer_tensor,
        )
        model_rows = [fit.to_dict() for fit in catalogue]
    runnable = [row for row in model_rows if row["fits"]]
    return {
        "cluster_capacity_bytes": sum(node.capacity_bytes for node in nodes),
        "node_count": len(nodes),
        "models": model_rows,
        "runnable_count": len(runnable),
        "largest_runnable": runnable[0] if runnable else None,
    }


@router.get("/deployments")
async def cluster_deployments():
    """List approved model deployments; secrets and SSH keys are never stored."""

    try:
        return get_cluster_registry().to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _live_jaccl_deployments(
    *,
    state_dir: str | Path | None = None,
) -> tuple[str, ...]:
    """Deployment IDs currently owning this Mac's JACCL/RDMA device."""

    if state_dir is None:
        state_dir = os.environ.get(
            "OMLX_CLUSTER_STATE_DIR",
            "~/.omlx/cluster/runtime",
        )
    runtime = read_runtime_markers(state_dir)
    return tuple(
        sorted(
            {
                str(job["deployment_id"])
                for job in runtime.get("jobs", ())
                if job.get("live") is True
                and str(job.get("backend", "")).startswith("jaccl")
            }
        )
    )


async def _activate_and_report(
    request: ClusterDeploymentRequest,
) -> dict[str, Any]:
    """Recompute, preflight, eagerly load, and prove one distributed model.

    This is the single activation pipeline behind both ``POST /deployments``
    (explicit approval of a previewed plan) and the approve phase of
    ``POST /replan`` (one-action deactivate → re-plan → reload). The
    deactivate half of a replan is exactly what this pipeline already does:
    ``pool.prepare_cluster_reload`` unloads the resident engine under the
    pool lock — quiescence-gated, and for distributed engines the supervisor's
    *verified* teardown is the memory barrier — before the registry swap and
    the eager reload below.
    """

    plan_changes: dict[str, Any] = {
        "changed": False,
        "reason": "",
        "approved_signature": "",
        "launched_signature": "",
        "ranks": [],
    }
    # Bound before the try so the incident emitters below can attach the
    # deployment ID when planning got far enough to assign one.
    deployment = None
    try:
        deployment, plan = await asyncio.to_thread(_create_deployment, request)
        if (
            deployment.tensor_parallel_size > 1
            and deployment.tensor_parallel_size < deployment.world_size
            and not hybrid_group_split_supported(deployment.backend)
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"The loaded MLX {deployment.distributed_init_backend} backend "
                    "does not implement Group.split, so this TP x pipeline "
                    "placement cannot launch safely. Install the oMLX MLX/JACCL "
                    "subgroup runtime on every rank or choose pure TP/pipeline."
                ),
            )
        live_jaccl = _live_jaccl_deployments()
        other_jaccl = tuple(
            item for item in live_jaccl if item != deployment.deployment_id
        )
        if deployment.backend.startswith("jaccl") and other_jaccl:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Thunderbolt RDMA is already owned by live deployment "
                    f"{', '.join(other_jaccl)}. oMLX permits one JACCL "
                    "communicator per Mac until Apple's driver passes the "
                    "multi-communicator lost-completion soak. Deactivate or "
                    "re-plan that deployment first."
                ),
            )
        # Before anything touches another Mac: is this the plan the user was
        # shown? Preview and activation posted different node objects for
        # months, and the second one dropped the split cap and the role, so the
        # stage that launched was not the stage anybody approved. Refusing here
        # costs nothing; discovering it after every rank has staged its weights
        # costs the machine.
        approved_placement = request.approved_placement.strip()
        if approved_placement != _placement_signature(plan):
            raise HTTPException(
                status_code=409,
                detail=(
                    "This is not the plan you approved — the budgets, roles or "
                    "layer split changed since it was built. Build the plan "
                    "again and approve what it shows. As posted, this request "
                    f"would place: {_describe_placement(plan)}."
                ),
            )
        memory_plan = plan
        # Refuse to start against a Mac that is not answering. Launching into a
        # dead peer produces a collective that blocks forever rather than an
        # error, so this has to be checked before any rank starts.
        health = await asyncio.to_thread(
            check_peers,
            {
                index: (host.node_id, host.ssh)
                for index, host in enumerate(deployment.hosts)
            },
            deployment_id=deployment.deployment_id,
        )
        raise_if_peer_lost(health)
        preflight = await asyncio.to_thread(
            preflight_remote_hosts,
            deployment,
        )
        performance_probe: dict[str, Any] = {
            "ok": False,
            "status": "disabled",
            "reason": "automatic tuning disabled",
        }
        same_jaccl_is_live = (
            deployment.backend.startswith("jaccl")
            and deployment.deployment_id in live_jaccl
        )
        if deployment.execution.auto_tune and same_jaccl_is_live:
            # A re-plan may reuse the deployment ID while its old engine is
            # serving. Launching a second JACCL performance communicator before
            # prepare_cluster_reload tears the first one down physically
            # reproduces Apple's lost-completion failure. Keep the signed plan;
            # the next cold activation can measure it safely.
            performance_probe = {
                "ok": False,
                "status": "communicator_busy",
                "reason": (
                    "automatic tuning skipped because this deployment already "
                    "owns the JACCL/RDMA device"
                ),
                "plan_changed": False,
            }
        elif deployment.execution.auto_tune:
            if deployment.performance_profiles:
                # The one-click path measured these ranks before model staging
                # and signed the resulting placement. Re-running a noisy probe
                # here could only propose an unapproved shard map after copying.
                performance_probe = {
                    "ok": True,
                    "status": "precomputed_before_staging",
                    "backend": deployment.backend,
                    "world_size": deployment.world_size,
                    "profiles": [
                        profile.to_dict()
                        for profile in deployment.performance_profiles
                    ],
                    "plan_changed": False,
                }
            else:
                try:
                    performance_probe = await asyncio.to_thread(
                        run_cluster_performance_probe,
                        deployment,
                    )
                    if performance_probe.get("promotable") is False:
                        raise _UnpromotablePerformanceCalibration(
                            str(
                                performance_probe.get("qualification_reason")
                                or "Low Power Mode was enabled during calibration"
                            )
                        )
                    candidate_deployment, candidate_plan = await asyncio.to_thread(
                        _performance_optimized_deployment,
                        deployment,
                        request,
                        performance_probe,
                    )
                    plan_changes = _plan_changes(memory_plan, candidate_plan)
                    performance_probe["plan_changed"] = plan_changes["changed"]
                    if plan_changes["changed"]:
                        # Manual/legacy callers may not have measured before
                        # staging. Never switch their signed shard map here.
                        deployment = replace(
                            deployment,
                            execution=tune_execution_settings(
                                deployment.execution,
                                deployment.assignments,
                                backend=deployment.backend,
                            ),
                            performance_profiles=(
                                candidate_deployment.performance_profiles
                            ),
                        )
                        plan = memory_plan
                        plan_changes["reason"] = (
                            "automatic tuning proposed another placement; oMLX "
                            "kept the signed, already-staged rank map"
                        )
                        plan_changes["launched_signature"] = _placement_signature(plan)
                        performance_probe["status"] = "placement_locked"
                    else:
                        deployment, plan = candidate_deployment, candidate_plan
                        performance_probe["status"] = "applied"
                except _UnpromotablePerformanceCalibration as exc:
                    performance_probe = {
                        "ok": False,
                        "promotable": False,
                        "status": "power_limited",
                        "reason": str(exc)[:1000],
                        "plan_changed": False,
                    }
                except (DistributedLaunchError, OSError, ValueError) as exc:
                    # Benchmark failure must not make a memory-safe deployment
                    # unusable. Persist the exact memory-only fallback instead.
                    performance_probe = {
                        "ok": False,
                        "status": "memory_fallback",
                        "reason": str(exc)[:1000],
                    }
        pool = _engine_pool()
        try:
            model_id = pool.resolve_cluster_model_id(deployment.model)
        except ModelNotFoundError:
            register = getattr(pool, "register_cluster_model", None)
            if not callable(register):
                raise
            estimated_size = int(
                plan.get("model", {}).get("total_weight_bytes")
                or sum(
                    assignment.planned_weight_bytes
                    for assignment in deployment.assignments
                )
            )
            model_id, _ = register(
                deployment.model,
                estimated_size=estimated_size,
            )
        registry = get_cluster_registry()
        previous = await asyncio.to_thread(
            registry.get_for_model,
            deployment.model,
        )
        entry = pool.get_entry(model_id)
        loaded_deployment = getattr(
            getattr(entry, "engine", None),
            "deployment",
            None,
        )
        # Every persisted deployment field that reaches a worker is part of
        # runtime identity.  Comparing only plan/MTP let execution toggles
        # (notably SSD snapshot write-behind) reuse an engine launched with a
        # different contract.
        already_loaded = loaded_deployment == deployment
        if not already_loaded:
            await pool.prepare_cluster_reload(model_id)
        await asyncio.to_thread(registry.upsert, deployment)
        try:
            engine = await pool.get_engine(model_id)
            active_deployment = getattr(engine, "deployment", None)
            if active_deployment != deployment:
                raise DistributedLaunchError(
                    "engine pool did not activate the approved distributed plan"
                )
            canary = await engine.generate(
                "__omlx_cluster_readiness__",
                # A one-token GenerationBatch still pipelines and then
                # discards a successor graph. That terminal-only boundary can
                # leave a peer inside its final JACCL all-reduce even though
                # rank zero already produced a valid response. Four tokens
                # exercise sustained synchronized decode and make readiness a
                # stronger, representative proof.
                max_tokens=4,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                _request_id="omlx-internal-readiness",
            )
            cluster_status = engine.cluster_status()
        except BaseException as exc:
            # A deployment is not active merely because it passed planning.
            # Remove the failed engine first, then restore the exact registry
            # record clients saw before this request.
            try:
                await pool.prepare_cluster_reload(model_id)
            finally:
                if previous is None:
                    await asyncio.to_thread(
                        registry.remove,
                        deployment.deployment_id,
                    )
                    unregister = getattr(
                        pool,
                        "unregister_cluster_model",
                        None,
                    )
                    if callable(unregister):
                        unregister(model_id)
                else:
                    await asyncio.to_thread(registry.upsert, previous)
            if isinstance(exc, Exception):
                raise DistributedLaunchError(
                    f"Cluster readiness check failed: {exc}"
                ) from exc
            raise
    except PeerLostError as exc:
        # 409: the cluster is not in a state to start. Launching into a peer
        # that is not answering yields a collective that blocks forever.
        await asyncio.to_thread(
            _record_cluster_incident,
            Severity.ERROR,
            "activation_peer_lost",
            str(exc),
            deployment_id=deployment.deployment_id if deployment else None,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelBusyError as exc:
        await asyncio.to_thread(
            _record_cluster_incident,
            Severity.ERROR,
            "activation_model_busy",
            "This model is serving a request and cannot change cluster "
            "topology until that request finishes.",
            deployment_id=deployment.deployment_id if deployment else None,
        )
        raise HTTPException(
            status_code=409,
            detail=(
                "This model is serving a request and cannot change cluster "
                "topology until that request finishes."
            ),
        ) from exc
    except DistributedLaunchError as exc:
        await asyncio.to_thread(
            _record_cluster_incident,
            Severity.ERROR,
            "activation_launch_failed",
            str(exc),
            deployment_id=deployment.deployment_id if deployment else None,
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        await asyncio.to_thread(
            _record_cluster_incident,
            Severity.ERROR,
            "activation_model_not_found",
            str(exc),
            deployment_id=deployment.deployment_id if deployment else None,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (OSError, PlanningError, ValueError) as exc:
        await asyncio.to_thread(
            _record_cluster_incident,
            Severity.ERROR,
            "activation_rejected",
            str(exc),
            deployment_id=deployment.deployment_id if deployment else None,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "deployment": deployment.to_dict(),
        "plan": _plan_with_signature(plan),
        "preflight": preflight,
        "performance_probe": performance_probe,
        "plan_changes": plan_changes,
        "load_behavior": "eager",
        "readiness": {
            "state": "ready",
            "weights_resident": True,
            "all_ranks_ready": True,
            "canary_passed": True,
            "canary_completion_tokens": canary.completion_tokens,
            "ranks": cluster_status.get("ranks", []),
        },
        "api": {
            "base_url": "/v1",
            "model": model_id,
            "chat_completions": "/v1/chat/completions",
            "responses": "/v1/responses",
        },
    }


@router.post("/deployments")
async def activate_cluster_deployment(request: ClusterDeploymentRequest):
    """Recompute, preflight, eagerly load, and prove one distributed model."""

    return await _activate_and_report(request)


class ClusterReplanRequest(BaseModel):
    """One-action deactivate → re-plan → reload for a clustered model.

    With no ``approved_placement`` this is a dry run: it renders the plan the
    one-action call would launch, signed, with a diff against the running
    placement. Posting that signature back performs the whole dance at once.

    ``nodes``/``hosts``/``backend`` default to the current deployment's, so a
    budget-only or context-only change needs no host details; a membership
    change (a node joining or leaving the model) is expressed by posting the
    new explicit ``nodes`` + ``hosts`` lists. ``path_map`` likewise defaults
    to the current deployment's per-node paths so a replan does not silently
    revert a heterogeneous-path cluster to same-path resolution.
    """

    model_config = ConfigDict(extra="forbid")

    deployment_id: str | None = Field(default=None, max_length=128)
    model_path: str | None = Field(default=None, max_length=4096)
    model_source: str | None = Field(default=None, max_length=255)
    model_source_python: str | None = Field(default=None, max_length=4096)
    backend: Literal["auto", "ring", "jaccl", "jaccl-ring"] | None = None
    nodes: list[ClusterPlanNodeRequest] | None = Field(
        default=None, min_length=2, max_length=64
    )
    hosts: list[ClusterHostRequest] | None = Field(
        default=None, min_length=2, max_length=64
    )
    execution_profile: Literal["interactive", "balanced", "throughput"] = (
        "balanced"
    )
    allocation: Literal["balanced", "proportional"] = "balanced"
    auto_tune: bool = False
    sampling_rank_only: bool = True
    async_overlap: bool = True
    cache_affinity: bool = True
    max_kv_size: int | None = Field(default=None, gt=0)
    ring_connections_per_ip: int | None = Field(default=None, ge=1, le=32)
    tensor_parallel_size: int | None = Field(default=None, ge=1, le=64)
    serving_mode: Literal["sharded", "disaggregated"] | None = None
    prefill_rank: int | None = Field(default=None, ge=0, le=63)
    decode_rank: int | None = Field(default=None, ge=0, le=63)
    target_context_tokens: int | None = Field(
        default=None, ge=1, le=1_048_576
    )
    mtp_enabled: bool | None = None
    mtp_num_draft_tokens: int | None = Field(default=None, ge=1, le=8)
    prompt_cache_ssd: bool | None = None
    prompt_cache_ssd_max_bytes: int | None = Field(default=None, gt=0)
    deepseek_ane_prefill: DeepseekAnePrefillRequest | None = None
    path_map: dict[str, str] | None = Field(default=None, max_length=64)
    approved_placement: str | None = Field(default=None, min_length=16, max_length=64)


_REPLAN_STEPS = (
    "deactivate (quiescence-gated unload with verified teardown)",
    "re-plan (recomputed server-side and pinned by signature)",
    "reload (eager distributed load with readiness canary)",
)


@router.post("/replan")
async def replan_cluster_deployment(request: ClusterReplanRequest):
    """Collapse deactivate → re-plan → reload into one action.

    The engine-pool quiescence gate (``prepare_cluster_reload``) refuses to
    interrupt in-flight requests, and the distributed supervisor's verified
    teardown is the memory barrier between the old world and the new one: if
    teardown cannot be proven, the old registry record stays and the error
    says what survived.
    """

    try:
        registry = get_cluster_registry()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    current: ClusterDeployment | None = None
    if request.deployment_id:
        current = await asyncio.to_thread(registry.get, request.deployment_id)
        if current is None:
            raise HTTPException(
                status_code=404, detail="cluster deployment not found"
            )
    elif request.model_path:
        current = await asyncio.to_thread(
            registry.get_for_model, request.model_path
        )

    if current is None and not (
        request.model_path and request.nodes and request.hosts
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "replan needs an existing deployment (deployment_id or a "
                "model_path with a registered deployment) or an explicit "
                "model_path together with nodes and hosts"
            ),
        )

    derived: dict[str, bool] = {
        "nodes": False,
        "hosts": False,
        "backend": False,
        "path_map": False,
    }
    nodes = request.nodes
    if nodes is None:
        if current is None:
            raise HTTPException(status_code=400, detail="nodes are required")
        nodes = [
            ClusterPlanNodeRequest(**payload)
            for payload in nodes_from_deployment(current)
        ]
        derived["nodes"] = True
    hosts = request.hosts
    if hosts is None:
        if current is None:
            raise HTTPException(status_code=400, detail="hosts are required")
        hosts = [
            ClusterHostRequest(**payload)
            for payload in hosts_from_deployment(current)
        ]
        derived["hosts"] = True
    backend = request.backend
    if backend is None:
        if current is None:
            raise HTTPException(status_code=400, detail="backend is required")
        backend = current.backend
        derived["backend"] = True
    path_map = request.path_map
    if path_map is None and current is not None and current.path_map:
        # Per-node paths are part of the deployment contract: a replan that
        # dropped them would silently revert workers to the coordinator's
        # shared path. Carry them forward unless the caller overrides.
        path_map = dict(current.path_map)
        derived["path_map"] = True
    if backend == "auto":
        # rdma_ctl on every member → jaccl; any member without it pulls the
        # whole cluster onto the TCP ring. Derived from the posted host
        # records; launch preflight re-verifies the live state.
        try:
            selection = select_cluster_backend(members_from_host_records(hosts))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        backend_decision: dict[str, Any] = selection.to_dict()
        backend = selection.backend
    else:
        backend_decision = {
            "backend": backend,
            "reason": "operator-selected backend",
            "blockers": [],
            "members": [
                member.to_dict() for member in members_from_host_records(hosts)
            ],
        }
    try:
        _validate_cluster_hosts(hosts)
        effective = ClusterDeploymentRequest(
            deployment_id=request.deployment_id,
            model_path=(request.model_path or (current.model if current else "")),
            model_source=request.model_source,
            model_source_python=request.model_source_python,
            backend=backend,
            nodes=nodes,
            hosts=hosts,
            execution_profile=request.execution_profile,
            allocation=request.allocation,
            auto_tune=request.auto_tune,
            sampling_rank_only=request.sampling_rank_only,
            async_overlap=request.async_overlap,
            cache_affinity=request.cache_affinity,
            max_kv_size=request.max_kv_size,
            ring_connections_per_ip=request.ring_connections_per_ip,
            tensor_parallel_size=(
                request.tensor_parallel_size
                if request.tensor_parallel_size is not None
                else current.tensor_parallel_size
                if current is not None
                else 1
            ),
            serving_mode=(
                request.serving_mode
                if request.serving_mode is not None
                else current.serving_mode
                if current is not None
                else "sharded"
            ),
            prefill_rank=(
                request.prefill_rank
                if "prefill_rank" in request.model_fields_set
                else current.prefill_rank
                if current is not None
                else None
            ),
            decode_rank=(
                request.decode_rank
                if "decode_rank" in request.model_fields_set
                else current.decode_rank
                if current is not None
                else None
            ),
            target_context_tokens=(
                request.target_context_tokens
                if request.target_context_tokens is not None
                else current.target_context_tokens
                if current is not None
                else 8192
            ),
            mtp_enabled=(
                request.mtp_enabled
                if request.mtp_enabled is not None
                else current.mtp_enabled
                if current is not None
                else False
            ),
            mtp_num_draft_tokens=(
                request.mtp_num_draft_tokens
                if "mtp_num_draft_tokens" in request.model_fields_set
                else current.mtp_num_draft_tokens
                if current is not None
                else None
            ),
            prompt_cache_ssd=(
                request.prompt_cache_ssd
                if "prompt_cache_ssd" in request.model_fields_set
                else current.execution.prompt_cache_ssd
                if current is not None
                else False
            ),
            prompt_cache_ssd_max_bytes=(
                request.prompt_cache_ssd_max_bytes
                if "prompt_cache_ssd_max_bytes" in request.model_fields_set
                else current.execution.prompt_cache_ssd_max_bytes
                if current is not None
                else DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES
            ),
            deepseek_ane_prefill=(
                request.deepseek_ane_prefill
                if "deepseek_ane_prefill" in request.model_fields_set
                else DeepseekAnePrefillRequest(
                    **current.execution.deepseek_ane_prefill.to_dict()
                )
                if current is not None
                and current.execution.deepseek_ane_prefill.enabled
                else None
            ),
            path_map=path_map,
            # Not consulted by planning; activation re-checks the real one
            # below. Preview callers do not have a signature yet.
            approved_placement=request.approved_placement or ("0" * 16),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        deployment, plan = await asyncio.to_thread(_create_deployment, effective)
    except (OSError, PlanningError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    signed_plan = _plan_with_signature(plan)
    changes = (
        _plan_changes(placement_view(current), plan)
        if current is not None
        else None
    )

    if request.approved_placement is None:
        return {
            "ok": True,
            "mode": "preview",
            "steps": list(_REPLAN_STEPS),
            "derived": derived,
            "current": (
                summarize_deployment(current) if current is not None else None
            ),
            "changes": changes,
            "deployment_id": deployment.deployment_id,
            "backend": deployment.backend,
            "backend_decision": backend_decision,
            "plan": signed_plan,
        }

    if request.approved_placement.strip() != signed_plan["placement_signature"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "This is not the plan the replan preview showed — the budgets, "
                "roles or layer split changed in between. Preview again and "
                f"approve what it shows. As posted, this request would place: "
                f"{_describe_placement(plan)}."
            ),
        )

    result = await _activate_and_report(effective)
    return result | {
        "mode": "applied",
        "replan": {
            "steps": list(_REPLAN_STEPS),
            "derived": derived,
            "backend_decision": backend_decision,
            "previous": (
                summarize_deployment(current) if current is not None else None
            ),
            "changes": changes,
        },
    }


@router.delete("/deployments/{deployment_id}")
async def deactivate_cluster_deployment(deployment_id: str):
    """Stop the resident cluster, then disable future distributed loads."""

    registry = get_cluster_registry()
    deployment = await asyncio.to_thread(registry.get, deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="cluster deployment not found")
    try:
        pool = _engine_pool()
        try:
            model_id = pool.resolve_cluster_model_id(deployment.model)
        except ModelNotFoundError:
            model_id = None
        if model_id is not None:
            await pool.prepare_cluster_reload(model_id)
        await asyncio.to_thread(stop_deployment_processes, deployment)
        removed = await asyncio.to_thread(registry.remove, deployment_id)
        unregister = getattr(pool, "unregister_cluster_model", None)
        if model_id is not None and callable(unregister):
            unregister(model_id)
    except ModelBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The cluster is serving a request. It will not be interrupted; "
                "stop it again after the request finishes."
            ),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="cluster deployment not found")
    return {
        "ok": True,
        "deployment_id": deployment_id,
        "stopped": True,
    }


@router.post("/deployments/{deployment_id}/unload")
async def unload_cluster_deployment(deployment_id: str):
    """Release resident ranks while keeping the signed deployment configured."""

    registry = get_cluster_registry()
    deployment = await asyncio.to_thread(registry.get, deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="cluster deployment not found")
    model_id = None
    try:
        pool = _engine_pool()
        model_id = pool.resolve_cluster_model_id(deployment.model)
        entry = pool.get_entry(model_id)
        if entry is not None and entry.engine is not None:
            await pool.prepare_cluster_reload(model_id)
    except ModelNotFoundError:
        # A configured deployment may legitimately be cold after restart.
        # Keeping unload idempotent lets the dashboard recover without
        # reconstructing EnginePool's private registration state.
        model_id = None
    except ModelBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The cluster is serving a request. It will not be interrupted; "
                "unload it after the request finishes."
            ),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        teardown = await asyncio.to_thread(stop_deployment_processes, deployment)
    except (DistributedTeardownError, DistributedLaunchError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "ok": True,
        "deployment_id": deployment_id,
        "model_id": model_id,
        "stopped": True,
        "configured": True,
        "teardown": teardown,
    }


@router.post("/deployments/{deployment_id}/load")
async def load_cluster_deployment(deployment_id: str):
    """Load a configured deployment and prove all ranks with a short canary."""

    registry = get_cluster_registry()
    deployment = await asyncio.to_thread(registry.get, deployment_id)
    if deployment is None:
        raise HTTPException(status_code=404, detail="cluster deployment not found")
    try:
        pool = _engine_pool()
        try:
            model_id = pool.resolve_cluster_model_id(deployment.model)
        except ModelNotFoundError:
            register = getattr(pool, "register_cluster_model", None)
            if not callable(register):
                raise
            estimated_size = sum(
                assignment.planned_weight_bytes
                for assignment in deployment.assignments
            )
            model_id, _ = register(
                deployment.model,
                estimated_size=estimated_size,
            )
        entry = pool.get_entry(model_id)
        resident = getattr(entry, "engine", None) if entry is not None else None
        if resident is not None and getattr(
            resident, "runtime_failed_reason", None
        ):
            await pool.prepare_cluster_reload(model_id)
        engine = await pool.get_engine(model_id)
        if getattr(engine, "deployment", None) != deployment:
            raise DistributedLaunchError(
                "engine pool did not load the configured distributed plan"
            )
        canary = await engine.generate(
            "__omlx_cluster_readiness__",
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            _request_id="omlx-internal-readiness",
        )
        status = engine.cluster_status()
    except ModelBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail="The model is already changing state; wait and try again.",
        ) from exc
    except (DistributedLaunchError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "deployment_id": deployment_id,
        "model_id": model_id,
        "loaded": True,
        "canary_completion_tokens": canary.completion_tokens,
        "ranks": status.get("ranks", []),
    }
