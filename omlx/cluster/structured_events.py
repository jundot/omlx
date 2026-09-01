# SPDX-License-Identifier: Apache-2.0
"""
Cluster-specific structured event emitters for oMLX.

Integrates with the cluster module to emit structured JSON events for
cluster lifecycle changes, worker status transitions, and collective
operations. All emitters delegate to the core ``structured_logging``
module for formatting and secret redaction.
"""

from __future__ import annotations

import logging
from typing import Any

from omlx.structured_logging import _emit, sanitize_for_log

logger = logging.getLogger("omlx.cluster")


def log_cluster_deployment_started(
    *,
    deployment_id: str,
    model: str,
    ranks: list[str],
    coordinator: str,
    trace_id: str | None = None,
) -> None:
    """Emit when a distributed deployment begins."""
    _emit(
        "omlx.cluster",
        logging.INFO,
        f"Deployment {deployment_id} started",
        event="cluster.deployment_started",
        deployment_id=deployment_id,
        model=model,
        rank_count=len(ranks),
        coordinator=coordinator,
        trace_id=trace_id,
    )


def log_cluster_deployment_completed(
    *,
    deployment_id: str,
    model: str,
    ranks: list[str],
    duration_s: float,
    trace_id: str | None = None,
) -> None:
    """Emit when all ranks finish loading and the deployment is live."""
    _emit(
        "omlx.cluster",
        logging.INFO,
        f"Deployment {deployment_id} ready",
        event="cluster.deployment_completed",
        deployment_id=deployment_id,
        model=model,
        rank_count=len(ranks),
        duration_s=round(duration_s, 3),
        trace_id=trace_id,
    )


def log_worker_joined(
    *,
    node_id: str,
    rank: int,
    deployment_id: str,
    host: str = "",
    trace_id: str | None = None,
) -> None:
    """Emit when a worker successfully joins a deployment."""
    _emit(
        "omlx.cluster",
        logging.INFO,
        f"Worker {node_id} joined deployment {deployment_id} as rank {rank}",
        event="cluster.worker_joined",
        node_id=node_id,
        rank=rank,
        deployment_id=deployment_id,
        host=sanitize_for_log(host),
        trace_id=trace_id,
    )


def log_worker_left(
    *,
    node_id: str,
    rank: int,
    deployment_id: str,
    reason: str = "",
    trace_id: str | None = None,
) -> None:
    """Emit when a worker leaves or is removed from a deployment."""
    _emit(
        "omlx.cluster",
        logging.WARNING,
        f"Worker {node_id} left deployment {deployment_id}",
        event="cluster.worker_left",
        node_id=node_id,
        rank=rank,
        deployment_id=deployment_id,
        reason=reason,
        trace_id=trace_id,
    )


def log_collective_operation(
    *,
    operation: str,
    deployment_id: str,
    rank_count: int,
    duration_s: float,
    success: bool,
    error: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Emit for collective operations (all-reduce, broadcast, etc.)."""
    _emit(
        "omlx.cluster",
        logging.INFO if success else logging.ERROR,
        f"Collective {operation} {'succeeded' if success else 'failed'}",
        event="cluster.collective_operation",
        operation=operation,
        deployment_id=deployment_id,
        rank_count=rank_count,
        duration_s=round(duration_s, 3),
        success=success,
        error=sanitize_for_log(error) if error else None,
        trace_id=trace_id,
    )


def log_inference_routed(
    *,
    request_id: str,
    deployment_id: str,
    target_node: str,
    target_rank: int,
    model: str,
    prompt_tokens: int,
    trace_id: str | None = None,
) -> None:
    """Emit when an inference request is routed to a specific cluster node."""
    _emit(
        "omlx.cluster",
        logging.INFO,
        f"Inference request routed to node {target_node}",
        event="cluster.inference_routed",
        request_id=request_id,
        deployment_id=deployment_id,
        target_node=target_node,
        target_rank=target_rank,
        model=model,
        prompt_tokens=prompt_tokens,
        trace_id=trace_id,
    )
