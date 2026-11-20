# SPDX-License-Identifier: Apache-2.0
"""Authenticated admin API routes for local cluster diagnostics."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
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
    members_from_host_records,
    select_cluster_backend,
)
from .catalogue import ModelFit, assess_model, catalogue_for_cluster
from .deployment import (
    ClusterDeployment,
    ClusterHost,
    validate_model_path_map,
    validate_ssh_target,
)
from .launch import (
    DistributedLaunchError,
    DistributedTeardownError,
    preflight_remote_hosts,
    probe_remote_admission_ceiling,
    probe_remote_host,
    run_cluster_performance_probe,
    stop_deployment_processes,
)
from .liveness import (
    PeerLostError,
    check_peers,
    raise_if_peer_lost,
)
from .model_inventory import (
    engine_pool_model_inventory,
    merge_model_inventories,
    remote_model_inventory,
)
from .performance import (
    NodePerformanceProfile,
    execution_profile,
    tune_execution_settings,
)
from .planner import (
    LOCAL_NODE,
    ModelLayout,
    NodeBudget,
    PlanningError,
    ShardPlan,
    complete_model_layout,
    plan_hybrid,
    plan_proportional_pipeline,
    plan_unequal_pipeline,
    remote_model_layout,
    synthetic_model_layout,
    tensor_split_from_dict,
    tensor_split_to_dict,
    with_mtp_head,
)
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
    DEFAULT_REMOTE_PYTHON,
    InsufficientDiskError,
    fabric_transfer_address,
    home_relative_model_path,
    index_shards,
    model_staging_inventory,
    plan_staging,
    remote_file_sizes,
    remote_model_dir,
    remote_model_staging_inventory,
    stage_files_from_source,
    stage_manifest,
)
from .strategy_benchmarks import get_strategy_benchmark_store
from .transport import (
    detect_cluster_transports,
    link_warnings,
    probe_host_interfaces,
    resolve_link_addresses,
    verify_link_reachability,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/api/cluster", tags=["cluster"])

_get_engine_pool: Any | None = None
_get_settings_manager: Any | None = None


def inspect_safetensors_layout(model_path: str | Path):
    """Compatibility seam for route tests, backed by the complete-model check.

    Older callers patched this route-local name.  Keeping the seam avoids
    coupling them to planner internals while ensuring production planning
    refuses a directory containing only one rank's previous stage.
    """

    return complete_model_layout(model_path)


def set_cluster_getters(
    engine_pool_getter: Any, settings_manager_getter: Any | None = None
) -> None:
    """Inject server-owned dependencies without importing ``omlx.server``."""

    global _get_engine_pool, _get_settings_manager
    _get_engine_pool = engine_pool_getter
    _get_settings_manager = settings_manager_getter


def _saved_model_settings(model_path: str) -> Any:
    """The coordinator's saved settings for this model, if any."""

    manager = _get_settings_manager() if _get_settings_manager else None
    if manager is None or _get_engine_pool is None:
        return None
    target = Path(model_path).expanduser().resolve()
    for item in engine_pool_model_inventory(_engine_pool()):
        if Path(item["model_path"]).expanduser().resolve() != target:
            continue
        if not manager.has_settings(item["id"]):
            return None
        return manager.get_settings(item["id"])
    return None


def _saved_lightning_mtp(model_path: str) -> bool | None:
    """The coordinator's saved Lightning MTP setting for this model, if any."""

    settings = _saved_model_settings(model_path)
    return None if settings is None else bool(settings.mtp_enabled)


def _saved_engram_ssd_offload(model_path: str | None) -> bool:
    """Whether the saved settings keep DeepSeek V4.1 Engram tables on SSD."""

    settings = _saved_model_settings(model_path) if model_path else None
    return bool(getattr(settings, "deepseek_v41_engram_ssd_offload", False))


def _cluster_mtp(
    requested: bool | None, model: ModelLayout, model_path: str | None
) -> bool:
    """Whether a new plan budgets the Lightning MTP head on rank zero.

    Unset follows the model like a single Mac does: a checkpoint with a head
    starts with MTP on unless its saved settings turned it off.
    """

    if requested is not None:
        if requested:
            with_mtp_head(model)
        return requested
    if model.mtp_weight_bytes <= 0:
        return False
    saved = _saved_lightning_mtp(model_path) if model_path else None
    return saved is not False


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
    memory_guard_tier: Literal["safe", "balanced", "aggressive", "custom"] = "balanced"
    performance: dict[str, Any] | None = None


class ClusterPlanRequest(BaseModel):
    """Estimated or downloaded model input for unequal-memory planning."""

    model_path: str | None = Field(default=None, max_length=4096)
    # SSH target of the Mac that owns the complete model. Empty/local keeps
    # planning on the coordinator; a peer source is measured on that peer.
    model_source: str | None = Field(default=None, max_length=255)
    # Interpreter reported by the source node's live probe. App and source
    # installs do not share a filesystem layout, so the coordinator's
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
    target_context_tokens: int = Field(default=8192, ge=1, le=1_048_576)
    # Optional node_id → absolute model path on that node. Empty
    # keeps the legacy same-absolute-path-on-every-node behavior.
    path_map: dict[str, str] | None = Field(default=None, max_length=64)
    # Lightning MTP: reserve the draft head on rank zero.
    mtp: bool = False
    # DeepSeek V4.1 shares per rank, {"attention": [...], "experts": [...]}.
    # None splits evenly when that fits, otherwise by each node's memory.
    tensor_split: dict[str, list[int]] | None = Field(default=None, max_length=2)


class ClusterHostRequest(BaseModel):
    """One rank-ordered SSH and collective transport endpoint."""

    node_id: str = Field(min_length=1, max_length=128)
    ssh: str = Field(min_length=1, max_length=255)
    ips: list[str] = Field(min_length=1, max_length=16)
    rdma: list[str | list[str] | None] = Field(default_factory=list, max_length=64)
    python_executable: str | None = Field(default=None, max_length=4096)


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
    prompt_cache_ssd: bool = True
    prompt_cache_ssd_max_bytes: int = Field(default=20 * 1024**3, gt=0)
    max_kv_size: int | None = Field(default=None, gt=0)
    ring_connections_per_ip: int | None = Field(default=None, ge=1, le=32)
    tensor_parallel_size: int = Field(default=1, ge=1, le=64)
    # DeepSeek V4.1 shares per rank, {"attention": [...], "experts": [...]}.
    # None splits evenly when that fits, otherwise by each node's memory.
    tensor_split: dict[str, list[int]] | None = Field(default=None, max_length=2)
    target_context_tokens: int = Field(default=8192, ge=1, le=1_048_576)
    # ``placement_signature`` from the /plan response the user was shown. The
    # server refuses to activate anything else, which is the only
    # thing that makes "the plan you approved" a fact rather than a hope:
    # /plan and /deployments used to post different node objects and the second
    # one dropped the role and the split cap without saying so. Activation is a
    # GUI workflow: callers must preview and name the placement they approve.
    approved_placement: str = Field(min_length=16, max_length=64)
    # node_id → absolute model path on that node. Nodes not listed load
    # ``model_path``, the shared path.
    path_map: dict[str, str] | None = Field(default=None, max_length=64)
    mtp: bool = False


class ClusterPeerProbeRequest(BaseModel):
    """Approved SSH peer and optional address used for its return route."""

    ssh: str = Field(min_length=1, max_length=255)
    route_to: str | None = Field(default=None, max_length=64)


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
                python_executable=(
                    request.model_source_python or DEFAULT_REMOTE_PYTHON
                ),
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
    "tensor_parallel_split",
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
    return {
        "changed": bool(ranks),
        "reason": "automatic tuning re-planned from the measured link speeds",
        "approved_signature": _placement_signature(approved),
        "launched_signature": _placement_signature(launched),
        "ranks": ranks,
    }


def _tensor_split_arg(value: dict[str, list[int]] | None) -> Any:
    """``plan_hybrid``'s ``tensor_split`` for a request's optional override."""

    if not value:
        return "auto"
    try:
        tensor_split_from_dict(value)
    except ValueError as exc:
        raise PlanningError(str(exc)) from exc
    return value


def _create_cluster_plan(request: ClusterPlanRequest):
    if (
        request.tensor_parallel_size > 1
        and request.tensor_parallel_size != len(request.nodes)
    ):
        raise PlanningError(
            "Tensor parallelism must use every detected node. Combining tensor "
            "parallelism with multiple pipeline stages is not supported by the "
            "pinned MLX model sharding path; choose 1 for pipeline-only or "
            f"{len(request.nodes)} for one tensor-parallel stage."
        )
    model, nodes = _model_and_nodes(request)
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
    pipeline_model = with_mtp_head(model) if request.mtp else model
    head_bytes = model.mtp_weight_bytes if request.mtp else 0
    engram_ssd_offload = _saved_engram_ssd_offload(request.model_path)
    defaults = execution_profile(request.execution_profile)
    if request.tensor_parallel_size > 1:
        if request.allocation != "balanced":
            raise PlanningError(
                "RAM-proportional allocation is a pipeline-only rule; tensor "
                "parallelism uses the hybrid planner (allocation='balanced')"
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
            head_rank_extra_bytes=head_bytes,
            tensor_split=_tensor_split_arg(request.tensor_split),
        )
    elif request.allocation == "proportional":
        plan = plan_proportional_pipeline(
            pipeline_model,
            nodes,
            workload_profile=request.execution_profile,
            microbatch_size=(
                request.pipeline_microbatch_size or defaults.pipeline_microbatch_size
            ),
            context_tokens=request.target_context_tokens,
            engram_ssd_offload=engram_ssd_offload,
        )
    else:
        plan = plan_unequal_pipeline(
            pipeline_model,
            nodes,
            workload_profile=request.execution_profile,
            microbatch_size=(
                request.pipeline_microbatch_size or defaults.pipeline_microbatch_size
            ),
            context_tokens=request.target_context_tokens,
            engram_ssd_offload=engram_ssd_offload,
        )
    if path_map:
        plan = replace(plan, path_map=path_map)
    return plan


class ClusterAutoconfigureRequest(BaseModel):
    """Everything one-click activation needs; the server decides the rest."""

    model_config = ConfigDict(extra="forbid")

    deployment_id: str | None = Field(default=None, max_length=128)
    path_map: dict[str, str] | None = Field(default=None, max_length=64)
    model_path: str | None = Field(default=None, max_length=4096)
    model_source: str | None = Field(default=None, max_length=255)
    model_source_python: str | None = Field(default=None, max_length=4096)
    model_size_bytes: int | None = Field(default=None, gt=0)
    layer_count: int = Field(default=80, gt=0, le=2048)
    nodes: list[ClusterPlanNodeRequest] = Field(min_length=1, max_length=64)
    hosts: list[ClusterHostRequest] = Field(default_factory=list, max_length=64)
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"
    prefer: Literal["speed", "capacity"] = "speed"
    strategy: Literal["auto", "tensor", "pipeline"] = "auto"
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
    prompt_cache_ssd: bool = True
    prompt_cache_ssd_max_bytes: int = Field(default=20 * 1024**3, gt=0)
    max_kv_size: int | None = Field(default=None, gt=0)
    ring_connections_per_ip: int | None = Field(default=None, ge=1, le=32)
    target_context_tokens: int = Field(default=8192, ge=1, le=1_048_576)
    # None follows the checkpoint and the model's saved Lightning MTP setting.
    mtp: bool | None = None
    # DeepSeek V4.1 shares per rank, {"attention": [...], "experts": [...]}.
    # None splits evenly when that fits, otherwise by each node's memory.
    tensor_split: dict[str, list[int]] | None = Field(default=None, max_length=2)


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
            source_python_executable=(
                request.model_source_python or DEFAULT_REMOTE_PYTHON
            ),
            **({"path_map": request.path_map} if request.path_map else {}),
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
    pairs = [
        (source, peer)
        for index, source in enumerate(hosts)
        for peer in hosts[index + 1 :]
    ]
    pair_links = [
        resolve_link_addresses(
            source,
            peer,
            probe=lambda target: interfaces[target],
            verify=verify_once,
        )
        for source, peer in pairs
    ]
    warnings = list(
        dict.fromkeys(
            warning
            for (source, peer), link in zip(pairs, pair_links)
            for warning in link_warnings(interfaces[source], interfaces[peer], link)
        )
    )
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
        warnings.append(
            f"The cluster uses the TCP ring instead of JACCL, because {blocker}"
        )
    elif blocker:
        reason = blocker
    backend = "ring" if blocker else proposed

    return {
        "ok": unresolved is None,
        "backend": backend,
        "backend_reason": reason,
        "blocker": blocker,
        "fell_back": fell_back,
        "warnings": warnings,
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
    try:
        path_map = validate_model_path_map(
            request.path_map, tuple(node.node_id for node in request.nodes)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request = request.model_copy(update={"path_map": path_map})

    plan_request = ClusterPlanRequest(
        model_path=request.model_path,
        model_source=request.model_source,
        model_source_python=request.model_source_python,
        model_size_bytes=request.model_size_bytes,
        layer_count=request.layer_count,
        nodes=request.nodes,
        execution_profile=request.execution_profile,
    )
    try:
        model, nodes = _model_and_nodes(plan_request)
        mtp = _cluster_mtp(request.mtp, model, request.model_path)
    except PlanningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    engram_ssd_offload = _saved_engram_ssd_offload(request.model_path)

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
        str(fabric["backend"]) if fabric is not None else choose_backend(transports)[0]
    )
    strategy_transports = transports
    # A resolved ring fabric is slow even when the transport probe failed.
    if provisional_backend == "ring" and (transports or fabric is not None):
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
        choice = choose_parallelism(
            model,
            nodes,
            transports=strategy_transports,
            prefer=request.prefer,
            strategy=request.strategy,
            measurements=measurements,
            workload_profile=request.execution_profile,
            context_tokens=request.target_context_tokens,
            head_rank_extra_bytes=model.mtp_weight_bytes if mtp else 0,
            engram_ssd_offload=engram_ssd_offload,
            tensor_split=_tensor_split_arg(request.tensor_split),
        )
    except PlanningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Pre-flight: find the reasons activation would fail before proposing it,
    # rather than after every rank has loaded its weights.
    peer_statuses: dict[str, Any] = {}
    if request.preflight and request.hosts:
        for host in request.hosts:
            if host.ssh in {"127.0.0.1", "localhost"}:
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
    placement = order_hosts_for_topology(
        [host.ssh for host in request.hosts],
        transports,
        choice.tensor_parallel_size,
        link_profiles,
    )
    warnings.extend(placement.warnings)
    placed_host_order = list(placement.hosts or requested_host_order)
    ordered_hosts = list(request.hosts)
    ordered_request_nodes = list(request.nodes)
    if placed_host_order:
        by_ssh = {host.ssh: host for host in request.hosts}
        ordered_hosts = [by_ssh[ssh] for ssh in placed_host_order if ssh in by_ssh]
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
        ordered_plan = plan_hybrid(
            model,
            ordered_budgets,
            tensor_parallel_size=choice.tensor_parallel_size,
            workload_profile=request.execution_profile,
            context_tokens=request.target_context_tokens,
            head_rank_extra_bytes=model.mtp_weight_bytes if mtp else 0,
            engram_ssd_offload=engram_ssd_offload,
            tensor_split=_tensor_split_arg(request.tensor_split),
        )
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
        warnings.extend(fabric.get("warnings", ()))
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

    performance_probe: dict[str, Any] = {
        "ok": False,
        "status": "disabled",
        "reason": "automatic performance measurement disabled",
    }
    profiled_request_nodes = list(ordered_request_nodes)
    existing_profiles = _request_performance_profiles(profiled_request_nodes)
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
            profiles = tuple(
                NodePerformanceProfile.from_dict(profile)
                for profile in performance_probe.get("profiles", ())
            )
            if len(profiles) != len(profiled_request_nodes):
                raise ValueError("performance probe did not return every cluster rank")
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
                tensor_parallel_size=choice.tensor_parallel_size,
                workload_profile=request.execution_profile,
                microbatch_size=probe_execution.pipeline_microbatch_size,
                context_tokens=request.target_context_tokens,
                mtp=mtp,
                engram_ssd_offload=engram_ssd_offload,
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

    # Destination paths participate in the same placement signature checked
    # by staging and activation; adding them only to activation yields 409.
    if path_map:
        choice = replace(choice, plan=replace(choice.plan, path_map=path_map))

    # stage_manifest probes peers with blocking SSH. Keep it off the FastAPI
    # event loop just like transport detection and preflight above.
    staging = await asyncio.to_thread(_staging_for, request, choice)
    staging_ready = staging is None or bool(staging.get("ready"))

    plan_payload = _plan_with_signature(choice.plan.to_dict())
    preflight_summary = describe_preflight(issues)
    if fabric_blocker:
        preflight_summary = f"Cluster link is not ready: {fabric_blocker}"
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
        "summary": choice.reason,
        "link": describe_transports(transports),
        "placement": placement.reason,
        "strategy": request.strategy,
        "strategy_measurements": {
            str(size): outcome.to_dict()
            for size, outcome in sorted(measurements.items())
        },
        "performance_probe": performance_probe,
        "staging": staging,
        "strategies": STRATEGIES,
        "preflight": _redact_diagnostic(preflight_summary),
        # Structured as well as summarised: an issue that carries a command is
        # a fix the user can paste, and a sentence hides it.
        "preflight_issues": [asdict(issue) for issue in issues],
        "ready_to_activate": not issues and staging_ready and fabric_ready,
        "ready_to_stage": not issues
        and fabric_ready
        and not bool((staging or {}).get("error")),
        "warnings": _redact_diagnostic(warnings),
        "transports": [transport.__dict__ for transport in transports],
        "plan": plan_payload,
        # Ready to POST straight to /deployments once the user approves.
        "activation": {
            "deployment_id": request.deployment_id,
            "path_map": request.path_map,
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
            "ring_connections_per_ip": (
                request.ring_connections_per_ip if backend == "ring" else None
            ),
            "mtp": mtp,
            "tensor_parallel_size": choice.tensor_parallel_size,
            "tensor_split": request.tensor_split,
            "nodes": [node.model_dump() for node in profiled_request_nodes],
            "hosts": activation_hosts,
            "preflight": True,
            "approved_placement": plan_payload["placement_signature"],
        },
    }


class ClusterStageRequest(BaseModel):
    """A signed proposal whose missing model files should be staged."""

    activation: ClusterDeploymentRequest
    parallel: int = Field(default=4, ge=1, le=16)


_STAGING_JOBS: dict[str, dict[str, Any]] = {}
_STAGING_JOBS_LOCK = threading.Lock()
_MAX_STAGING_JOBS = 32


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
        source_entry = next(
            (item for item in deployment.hosts if item.ssh == source_host), None
        )
        failed_nodes: list[str] = []
        for host, assignment in zip(deployment.hosts, assignments):
            # Files land at the node's path_map entry when one
            # exists. Otherwise preserve the cross-account behavior by
            # resolving the portable model path in the remote Mac's home.
            explicit_destination = deployment.path_map.get(host.node_id)
            if _local_ssh_target(host.ssh):
                destination_dir = explicit_destination or str(model_path)
                destination_path = Path(destination_dir)
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
                destination_dir = explicit_destination or remote_model_dir(
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
            # When this Mac is one end, the data takes the address the
            # collectives verified (Thunderbolt first) if SSH answers there.
            source_address = destination_address = None
            if needed and _local_ssh_target(source_host) != _local_ssh_target(host.ssh):
                if not _local_ssh_target(host.ssh):
                    destination_address = fabric_transfer_address(host.ssh, host.ips)
                elif source_entry is not None:
                    source_address = fabric_transfer_address(
                        source_host, source_entry.ips
                    )
            result = stage_files_from_source(
                plan,
                model_path=model_path,
                source_host=source_host,
                destination_host=host.ssh,
                expected_sizes=expected_sizes,
                destination_dir=destination_dir,
                parallel=parallel,
                progress=progress,
                source_address=source_address,
                destination_address=destination_address,
            )
            if result.bytes_copied:
                logger.info(
                    "Staged %.1f GiB to %s over %s in %.0f s (%.2f GB/s)",
                    result.bytes_copied / 1024**3,
                    assignment.node_id,
                    destination_address or source_address or "the paired address",
                    result.seconds,
                    result.gigabytes_per_second,
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
    except Exception as exc:  # noqa: BLE001 - background job reports the failure
        def fail(job: dict[str, Any], *, error=str(exc)) -> None:
            job["status"] = "failed"
            job["ready"] = False
            job["error"] = error

        _update_staging_job(job_id, fail)


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


def _reconcile_runtime_ownership(payload: dict[str, Any], pool: Any) -> None:
    """Retained marker files do not establish engine ownership."""
    loaded: set[str] = set()
    loading: set[str] = set()
    launchers = []
    for model_id in getattr(pool, "get_loaded_model_ids", lambda: [])():
        entry = pool.get_entry(model_id)
        status = getattr(getattr(entry, "engine", None), "cluster_status", None)
        if not callable(status):
            continue
        try:
            launcher = status() | {"model_id": model_id}
        except Exception:
            continue
        loaded.add(launcher.get("deployment_id"))
        launchers.append(launcher)
    try:
        registry = get_cluster_registry()
        for model_id in pool.get_model_ids():
            entry = pool.get_entry(model_id)
            if entry is not None and getattr(entry, "is_loading", False):
                deployment = registry.get_for_model(entry.model_path)
                if deployment is not None:
                    loading.add(deployment.deployment_id)
                    if deployment.deployment_id not in loaded:
                        launchers.append(
                            {
                                "deployment_id": deployment.deployment_id,
                                "model_id": model_id,
                                "phase": "loading",
                            }
                        )
    except (AttributeError, OSError, RuntimeError, ValueError):
        pass
    for job in payload.get("jobs", []):
        deployment_id = job.get("deployment_id")
        job["ownership"] = (
            "loaded"
            if deployment_id in loaded
            else "loading" if deployment_id in loading else "detached"
        )
        if job["ownership"] == "detached":
            job["live"] = False
        for launcher in launchers:
            if deployment_id == launcher.get("deployment_id"):
                job["ranks"] = launcher.get("ranks", [])
                job["endpoint"] = launcher.get("endpoint")
    payload["launchers"] = launchers


@router.get("/runtime")
async def cluster_runtime():
    """Return lightweight local rank markers for dashboard polling."""

    payload = await asyncio.to_thread(read_runtime_markers)
    if _get_engine_pool is None:
        _reconcile_runtime_ownership(payload, None)
        return payload
    try:
        pool = _engine_pool()
    except HTTPException:
        _reconcile_runtime_ownership(payload, None)
        return payload
    _reconcile_runtime_ownership(payload, pool)
    return payload


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
    requested = replace(
        requested,
        async_overlap=request.async_overlap,
        cache_affinity=request.cache_affinity,
        prompt_cache_ssd=getattr(request, "prompt_cache_ssd", requested.prompt_cache_ssd),
        prompt_cache_ssd_max_bytes=getattr(request, "prompt_cache_ssd_max_bytes", requested.prompt_cache_ssd_max_bytes),
        # The context chosen beside the model is both a reservation and a
        # runtime ceiling. Without this fallback the planner could reserve
        # 256k while the server used an unrelated advanced default (or no
        # bound at all), making the memory promise on screen untrue.
        max_kv_size=(
            request.max_kv_size or getattr(request, "target_context_tokens", None)
        ),
        ring_connections_per_ip=(
            request.ring_connections_per_ip or requested.ring_connections_per_ip
        ),
        mtp=bool(getattr(request, "mtp", False)),
    )
    return tune_execution_settings(
        requested,
        assignments,
        backend=backend,
    )


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
        target_context_tokens=request.target_context_tokens,
        path_map=request.path_map,
        mtp=request.mtp,
        tensor_split=request.tensor_split,
    )
    plan = _create_cluster_plan(plan_request)
    execution = _execution_for_request(
        request,
        plan.assignments,
        backend=request.backend,
    )
    if (
        execution.pipeline_microbatch_size != requested_microbatch
    ):
        plan_request.pipeline_microbatch_size = execution.pipeline_microbatch_size
        plan = _create_cluster_plan(plan_request)
    if len(request.hosts) != len(request.nodes):
        raise ValueError("host count must match node budget count")
    node_ids = [node.node_id.strip() for node in request.nodes]
    host_ids = [host.node_id.strip() for host in request.hosts]
    if host_ids != node_ids:
        raise ValueError("rank-ordered host IDs must match node budget IDs")

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
                rdma=tuple(host.rdma),
                python_executable=host.python_executable,
            )
            for host in request.hosts
        ),
        assignments=plan.assignments,
        plan_hash=plan.plan_hash,
        execution=execution,
        performance_profiles=_request_performance_profiles(request.nodes),
        tensor_parallel_size=request.tensor_parallel_size,
        target_context_tokens=request.target_context_tokens,
        path_map=validate_model_path_map(request.path_map, tuple(host_ids)),
    )
    return deployment, plan.to_dict()


def _build_performance_plan(
    model: Any,
    nodes: list[NodeBudget],
    *,
    tensor_parallel_size: int,
    workload_profile: str,
    microbatch_size: int,
    context_tokens: int,
    mtp: bool = False,
    engram_ssd_offload: bool = False,
    tensor_split: Any = "auto",
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
            head_rank_extra_bytes=model.mtp_weight_bytes if mtp else 0,
            tensor_split=tensor_split,
        )
    return plan_unequal_pipeline(
        with_mtp_head(model) if mtp else model,
        nodes,
        workload_profile=workload_profile,
        microbatch_size=microbatch_size,
        context_tokens=context_tokens,
        engram_ssd_offload=engram_ssd_offload,
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
            python_executable=(
                request.model_source_python or DEFAULT_REMOTE_PYTHON
            ),
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
    # Keep the approved plan's Engram decision instead of rereading settings.
    engram_ssd_offload = any(item.ssd_weight_bytes for item in deployment.assignments)
    # And its tensor split, so tuning re-measures the shares the user approved.
    approved_split = next(
        (
            tensor_split_to_dict(item.tensor_parallel_split)
            for item in deployment.assignments
            if item.tensor_parallel_split
        ),
        "auto",
    )
    plan = _build_performance_plan(
        model,
        nodes,
        tensor_parallel_size=deployment.tensor_parallel_size,
        workload_profile=deployment.execution.profile,
        microbatch_size=deployment.execution.pipeline_microbatch_size,
        context_tokens=request.target_context_tokens,
        mtp=deployment.execution.mtp,
        engram_ssd_offload=engram_ssd_offload,
        tensor_split=approved_split,
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
            tensor_parallel_size=deployment.tensor_parallel_size,
            workload_profile=execution.profile,
            microbatch_size=execution.pipeline_microbatch_size,
            context_tokens=request.target_context_tokens,
            mtp=deployment.execution.mtp,
            engram_ssd_offload=engram_ssd_offload,
            tensor_split=approved_split,
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
        return {"node_id": host.node_id, "ssh": host.ssh, **budget.to_dict()}

    try:
        nodes = list(await asyncio.gather(*(_for(host) for host in hosts)))
    except (DistributedLaunchError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not measure every Mac's usable model memory: {exc}",
        ) from exc
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
            models = await asyncio.to_thread(
                remote_model_inventory,
                validated,
                python_executable=host.python_executable or DEFAULT_REMOTE_PYTHON,
            )
            source_python = host.python_executable or DEFAULT_REMOTE_PYTHON
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
                    python_executable=(
                        candidate.model_source_python or DEFAULT_REMOTE_PYTHON
                    ),
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
        if deployment.execution.auto_tune:
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
        already_loaded = bool(
            loaded_deployment is not None
            and loaded_deployment.deployment_id == deployment.deployment_id
            and loaded_deployment.plan_hash == deployment.plan_hash
            and loaded_deployment.path_map == deployment.path_map
        )
        if not already_loaded:
            await pool.prepare_cluster_reload(model_id)
        await asyncio.to_thread(registry.upsert, deployment)
        try:
            engine = await pool.get_engine(model_id)
            active_deployment = getattr(engine, "deployment", None)
            if (
                active_deployment is None
                or active_deployment.deployment_id != deployment.deployment_id
                or active_deployment.plan_hash != deployment.plan_hash
            ):
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
            )
            cluster_status = engine.cluster_status()
        except BaseException as exc:
            # A deployment is not active merely because it passed planning.
            # Remove the failed engine first, then restore the exact registry
            # record clients saw before this request. Rollback errors are only
            # logged, so the caller still gets the readiness failure.
            try:
                await pool.prepare_cluster_reload(model_id)
            except Exception:
                logger.exception("Could not unload failed cluster model %s", model_id)
            try:
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
            except Exception:
                logger.exception(
                    "Could not restore the cluster registry for %s", model_id
                )
            if isinstance(exc, Exception):
                raise DistributedLaunchError(
                    f"Cluster readiness check failed: {exc}"
                ) from exc
            raise
    except PeerLostError as exc:
        # 409: the cluster is not in a state to start. Launching into a peer
        # that is not answering yields a collective that blocks forever.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ModelBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "This model is serving a request and cannot change cluster "
                "topology until that request finishes."
            ),
        ) from exc
    except DistributedLaunchError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (OSError, PlanningError, ValueError) as exc:
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
    execution_profile: Literal["interactive", "balanced", "throughput"] = "balanced"
    auto_tune: bool = False
    sampling_rank_only: bool = True
    async_overlap: bool = True
    cache_affinity: bool = True
    prompt_cache_ssd: bool = True
    prompt_cache_ssd_max_bytes: int = Field(default=20 * 1024**3, gt=0)
    max_kv_size: int | None = Field(default=None, gt=0)
    ring_connections_per_ip: int | None = Field(default=None, ge=1, le=32)
    tensor_parallel_size: int | None = Field(default=None, ge=1, le=64)
    target_context_tokens: int | None = Field(default=None, ge=1, le=1_048_576)
    path_map: dict[str, str] | None = Field(default=None, max_length=64)
    # None keeps the current deployment's Lightning MTP choice.
    mtp: bool | None = None
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
            raise HTTPException(status_code=404, detail="cluster deployment not found")
    elif request.model_path:
        current = await asyncio.to_thread(registry.get_for_model, request.model_path)

    if current is None and not (request.model_path and request.nodes and request.hosts):
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
            ClusterHostRequest(**payload) for payload in hosts_from_deployment(current)
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
            deployment_id=request.deployment_id
            or (current.deployment_id if current else None),
            model_path=(request.model_path or (current.model if current else "")),
            model_source=request.model_source,
            model_source_python=request.model_source_python,
            backend=backend,
            nodes=nodes,
            hosts=hosts,
            execution_profile=request.execution_profile,
            auto_tune=request.auto_tune,
            sampling_rank_only=request.sampling_rank_only,
            async_overlap=request.async_overlap,
            cache_affinity=request.cache_affinity,
            prompt_cache_ssd=request.prompt_cache_ssd,
            prompt_cache_ssd_max_bytes=request.prompt_cache_ssd_max_bytes,
            max_kv_size=request.max_kv_size,
            ring_connections_per_ip=request.ring_connections_per_ip,
            tensor_parallel_size=(
                request.tensor_parallel_size
                if request.tensor_parallel_size is not None
                else current.tensor_parallel_size if current is not None else 1
            ),
            target_context_tokens=(
                request.target_context_tokens
                if request.target_context_tokens is not None
                else current.target_context_tokens if current is not None else 8192
            ),
            path_map=path_map,
            mtp=(
                request.mtp
                if request.mtp is not None
                else current is not None and current.execution.mtp
            ),
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
        _plan_changes(placement_view(current), plan) if current is not None else None
    )

    if request.approved_placement is None:
        return {
            "ok": True,
            "mode": "preview",
            "steps": list(_REPLAN_STEPS),
            "derived": derived,
            "current": (summarize_deployment(current) if current is not None else None),
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


@router.delete("/forget")
async def forget_cluster(node_id: str | None = None):
    """Forget one peer, or leave the entire cluster, without remote contact.

    A signed model placement cannot survive losing a rank. Remove affected
    deployments after verified local teardown; retain all other peer trust.
    Omitting node_id, or selecting this Mac, leaves the whole local cluster.
    """
    from .pairing_routes import PairingError, _manager, _pairing_http_error

    manager = _manager()
    peers = await asyncio.to_thread(manager.paired_devices)
    deployments = await asyncio.to_thread(get_cluster_registry().list)
    forget_all = node_id is None or node_id == manager.node_id
    affected = [
        deployment
        for deployment in deployments
        if forget_all or any(host.node_id == node_id for host in deployment.hosts)
    ]
    targets = [
        peer["node_id"] for peer in peers if forget_all or peer["node_id"] == node_id
    ]
    if not forget_all and not targets and not affected:
        raise HTTPException(status_code=404, detail="cluster member not found")
    # Do not revoke peer trust if local teardown fails or a request is active.
    for deployment in affected:
        await deactivate_cluster_deployment(deployment.deployment_id, local_only=True)
    results = []
    try:
        for target in targets:
            results.append(await asyncio.to_thread(manager.unpair, target))
    except PairingError as exc:
        raise _pairing_http_error(exc) from exc
    return {
        "ok": True,
        "local_only": True,
        "stopped": False,
        "forgotten_node_ids": targets,
        "removed_deployment_ids": [item.deployment_id for item in affected],
        "revocations": results,
    }


@router.delete("/deployments/{deployment_id}")
async def deactivate_cluster_deployment(deployment_id: str, local_only: bool = False):
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
            await pool.prepare_cluster_reload(
                model_id, **({"local_only": True} if local_only else {})
            )
        await asyncio.to_thread(
            stop_deployment_processes,
            deployment,
            **({"local_only": True} if local_only else {}),
        )
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
        "stopped": not local_only,
        "local_only": local_only,
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
        if resident is not None and (
            getattr(resident, "runtime_failed_reason", None)
            or getattr(entry, "pending_unload_reason", None)
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
