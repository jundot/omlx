# SPDX-License-Identifier: Apache-2.0
"""One-action cluster re-plan: deactivate → re-plan → reload.

Today a membership or budget change is a manual four-step dance: deactivate
the deployment, rebuild a plan, approve it, then reload the model. This module
is the pure/data half of collapsing that into a single ``POST
/admin/api/cluster/replan`` call (the route handler in
``omlx/cluster/routes.py`` owns the async orchestration):

- :func:`nodes_from_deployment` / :func:`hosts_from_deployment` derive the
  plan inputs for the *next* plan from the *current* approved deployment, so
  a replan with no explicit overrides re-plans the same cluster the operator
  already trusts. A pairing/membership inventory (the v2 discovery/pairing
  modules) can later supply overrides for added or removed nodes; until then
  the explicit ``nodes``/``hosts`` request fields are how membership changes
  are expressed.
- :func:`summarize_deployment` produces the compact, secret-free description
  of what a replan replaced.

Derivation caveats (documented contract, not bugs):

- ``max_weight_bytes`` / ``target_weight_bytes`` are operator preferences that
  live on the plan *request*, not on the signed assignment, so a derived
  replan cannot recover them and resets both to "automatic". Operators moving
  the split control must pass explicit ``nodes``.
- KV reservations and per-rank context limits are recomputed by the planner
  from ``target_context_tokens``, never copied forward.
- Signed performance profiles *are* copied forward.  Dropping them silently
  turns a heterogeneous tensor deployment back into an equal split on every
  replan, even when the previous activation already measured both ranks.
"""

from __future__ import annotations

from typing import Any

from .deployment import ClusterDeployment


def nodes_from_deployment(deployment: ClusterDeployment) -> list[dict[str, Any]]:
    """Rank-ordered node-budget payloads equivalent to this deployment.

    The returned dicts match ``ClusterPlanNodeRequest`` fields. Split-control
    preferences (``max_weight_bytes`` / ``target_weight_bytes``) are not
    recoverable from a signed plan and are reset to automatic.
    """

    profiles = {
        (profile.rank, profile.node_id): profile
        for profile in deployment.performance_profiles
    }
    nodes: list[dict[str, Any]] = []
    for assignment in sorted(deployment.assignments, key=lambda item: item.rank):
        payload: dict[str, Any] = {
            "node_id": assignment.node_id,
            "capacity_bytes": assignment.capacity_bytes,
            "reserve_bytes": assignment.reserve_bytes,
            "manual_memory_limit": assignment.manual_memory_limit,
            "role": assignment.role or "headless",
            "memory_guard_tier": assignment.memory_guard_tier,
        }
        profile = profiles.get((assignment.rank, assignment.node_id))
        if profile is not None:
            payload["performance"] = profile.to_dict()
        nodes.append(payload)
    return nodes


def hosts_from_deployment(deployment: ClusterDeployment) -> list[dict[str, Any]]:
    """Rank-ordered host payloads equivalent to this deployment's hostfile."""

    hosts: list[dict[str, Any]] = []
    for host in deployment.hosts:
        payload: dict[str, Any] = {
            "node_id": host.node_id,
            "ssh": host.ssh,
            "ips": list(host.ips),
            "rdma": [
                list(path) if isinstance(path, tuple) else path
                for path in host.rdma
            ],
        }
        if host.python_executable:
            payload["python_executable"] = host.python_executable
        hosts.append(payload)
    return hosts


def summarize_deployment(deployment: ClusterDeployment) -> dict[str, Any]:
    """What a replan is replacing, in the terms the plan view displays."""

    return {
        "deployment_id": deployment.deployment_id,
        "model": deployment.model,
        "backend": deployment.backend,
        "world_size": deployment.world_size,
        "plan_hash": deployment.plan_hash,
        "target_context_tokens": deployment.target_context_tokens,
        "mtp_enabled": deployment.mtp_enabled,
        "mtp_num_draft_tokens": deployment.mtp_num_draft_tokens,
        "prompt_cache_ssd": deployment.execution.prompt_cache_ssd,
        "prompt_cache_ssd_max_bytes": (
            deployment.execution.prompt_cache_ssd_max_bytes
        ),
        "path_map": dict(deployment.path_map),
        "assignments": [
            {
                "rank": assignment.rank,
                "node_id": assignment.node_id,
                "start_layer": assignment.start_layer,
                "end_layer": assignment.end_layer,
                "planned_weight_bytes": assignment.planned_weight_bytes,
                "kv_cache_bytes": assignment.kv_cache_bytes,
            }
            for assignment in sorted(
                deployment.assignments, key=lambda item: item.rank
            )
        ],
    }


def placement_view(deployment: ClusterDeployment) -> dict[str, Any]:
    """The ``plan``-shaped assignment list used for placement diffs.

    ``_placement_rows`` in the routes module reads ``plan["assignments"]``;
    this rebuilds that shape from a persisted deployment so a replan can diff
    the running placement against the proposed one with the exact same code.
    """

    return {
        "path_map": dict(deployment.path_map),
        "mtp_enabled": deployment.mtp_enabled,
        "mtp_num_draft_tokens": deployment.mtp_num_draft_tokens,
        "prompt_cache_ssd": deployment.execution.prompt_cache_ssd,
        "prompt_cache_ssd_max_bytes": (
            deployment.execution.prompt_cache_ssd_max_bytes
        ),
        "assignments": [
            assignment.to_dict() for assignment in deployment.assignments
        ]
    }
