# SPDX-License-Identifier: Apache-2.0
"""Redacted support bundle for the Cluster Assistant.

A DAU support bundle has to answer "what did the assistant measure, decide and
start?" without containing a secret or a prompt. The rule is structural: only
allow-listed fields are copied, each model path is reduced to its folder name,
and free-text warnings are scrubbed of SSH users, addresses, absolute paths and
credential-shaped strings.

The bundle is read-only. It never re-elects, probes, stages or launches; a
missing record is an app-only bundle, not an error.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .operations import OperationRecord, OperationWarning
from .state import MeasurementRecord, MeasurementWarning
from .topology import missing_line_links

GIB = 1024**3

REDACTIONS = (
    "no API keys, tokens or passwords",
    "no SSH users, host keys or private keys",
    "model paths are reduced to their folder name",
    "free-text warnings have addresses and absolute paths removed",
    "no prompts, completions or model content",
)


def build_diagnostics(
    measurement: MeasurementRecord | None,
    operation: OperationRecord | None,
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """One allow-listed, redacted support bundle."""

    paths = _model_paths(measurement)
    return {
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "app": _app_versions(),
        "measurement": _measurement_block(measurement, paths),
        "operation": _operation_block(operation, paths),
        "plan": _plan_block(operation),
        "topology": _topology_block(measurement, operation),
        "health": _health_block(operation),
        "events": _events(measurement, operation, paths),
        "redactions": list(REDACTIONS),
    }


def _app_versions() -> dict[str, str]:
    from ...utils.hardware import get_mlx_version

    return {
        "omlx": _package_version("omlx"),
        "mlx": get_mlx_version(),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0] or "unknown",
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _model_paths(record: MeasurementRecord | None) -> dict[str, str]:
    """Every known checkpoint path -> the label that replaces it."""

    paths: dict[str, str] = {}

    def remember(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            paths[text] = _label(text)

    if record is None:
        return paths
    remember(record.model.model_path)
    for entries in record.inventories.values():
        for item in entries:
            remember(item.get("model_path"))
    return paths


def _label(path: Any) -> str:
    text = str(path or "").strip()
    if not text:
        return "unknown"
    return Path(text.rstrip("/")).name or "unknown"


def _redact_text(text: Any, paths: dict[str, str]) -> str:
    """Scrub a free-text message; the caller only ever gets recorded strings."""

    value = str(text or "")
    for path, label in sorted(paths.items(), key=lambda item: -len(item[0])):
        value = value.replace(path, f"[{label}]")
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\b", "[ssh-target]", value)
    value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[address]", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted-key]", value)
    value = re.sub(
        r"(?i)\b(api[_-]?key|token|password|authorization)\b\s*[:=]\s*\S+",
        r"\1=[redacted]",
        value,
    )
    value = re.sub(r"(?<![\w.])/(?:[\w.@+-]+/)+[\w.@+-]+", "[path]", value)
    return value


def _warning_block(
    warning: MeasurementWarning | OperationWarning, paths: dict[str, str]
) -> dict[str, Any]:
    return {
        "node_id": getattr(warning, "node_id", ""),
        "code": warning.code,
        "message": _redact_text(warning.message, paths),
    }


def _measurement_block(
    record: MeasurementRecord | None, paths: dict[str, str]
) -> dict[str, Any] | None:
    if record is None:
        return None
    nodes: list[dict[str, Any]] = []
    for facts in record.measurement_set().nodes:
        endpoint = record.endpoint_for(facts.node_id)
        inventory = record.inventories.get(facts.node_id) or ()
        nodes.append(
            {
                "node_id": facts.node_id,
                "capacity_gib": round(facts.capacity_gib, 1),
                "admission_p05_gib": round(facts.admission_p05_gib, 1),
                "admission_p50_gib": facts.admission_p50_gib,
                "admission_p95_gib": facts.admission_p95_gib,
                "memory_guard_tier": facts.memory_guard_tier,
                "wired_limit_mb": facts.wired_limit_mb,
                "link": facts.link.to_dict(),
                "disk_free_gib": facts.disk_free_gib,
                "model_present": facts.model_present,
                "healthy": facts.healthy,
                "endpoint": (
                    None
                    if endpoint is None
                    else {"host": endpoint.host, "port": endpoint.port}
                ),
                "models": [
                    {
                        "label": _label(item.get("model_path")),
                        "size_gib": round(
                            (item.get("size_bytes") or 0) / GIB, 1
                        ),
                    }
                    for item in inventory[:32]
                ],
            }
        )
    layout = record.model_layout()
    return {
        "measurement_id": record.measurement_id,
        "state": record.state,
        "measured_at": record.measured_at,
        "model": _label(record.model.model_path),
        "layer_count": None if layout is None else layout.layer_count,
        "total_weight_gib": (
            None
            if layout is None
            else round(layout.total_weight_bytes / GIB, 1)
        ),
        "fabric_measured": record.fabric_measured,
        "fabric_links": [list(pair) for pair in record.fabric_links],
        "warnings": [
            _warning_block(warning, paths) for warning in record.warnings
        ],
        "nodes": nodes,
    }


def _operation_block(
    operation: OperationRecord | None, paths: dict[str, str]
) -> dict[str, Any] | None:
    if operation is None:
        return None
    return {
        "operation_id": operation.operation_id,
        "kind": operation.kind,
        "state": operation.state,
        "progress": operation.progress.to_dict(),
        "coordinator_node": operation.coordinator_node,
        "coordinator_local": operation.coordinator_local,
        "rank_order": list(operation.rank_order),
        "context_tokens": operation.context_tokens,
        "target_context_tokens": operation.target_context_tokens,
        "model": _label(operation.model_path),
        "plan_signature": operation.plan_signature,
        "placement_signature": operation.placement_signature,
        "deployment_id": operation.deployment_id,
        "stopped": operation.stopped,
        "two_mac_fallback": operation.two_mac_fallback,
        "endpoint": (
            None
            if operation.endpoint is None
            else operation.endpoint.to_dict()
        ),
        "observed": (
            None if operation.observed is None else operation.observed.to_dict()
        ),
        "warnings": [
            _warning_block(warning, paths) for warning in operation.warnings
        ],
        "error": (
            None
            if operation.error is None
            else {
                "code": operation.error.code,
                "message": _redact_text(operation.error.message, paths),
                "action": operation.error.action,
            }
        ),
    }


def _plan_block(operation: OperationRecord | None) -> dict[str, Any] | None:
    if operation is None:
        return None
    return {
        "plan_signature": operation.plan_signature,
        "coordinator_node": operation.coordinator_node,
        "rank_order": list(operation.rank_order),
        "context_tokens": operation.context_tokens,
        "target_context_tokens": operation.target_context_tokens,
        "mtp": operation.mtp,
        "two_mac_fallback": operation.two_mac_fallback,
        "policy": dict(operation.policy),
    }


def _topology_block(
    measurement: MeasurementRecord | None, operation: OperationRecord | None
) -> dict[str, Any] | None:
    if measurement is None:
        return None
    links = [list(pair) for pair in measurement.fabric_links]
    order = [] if operation is None else list(operation.rank_order)
    if not measurement.fabric_measured:
        return {
            "kind": "line",
            "measured": False,
            "complete": None,
            "rank_order": order,
            "links": links,
            "missing_links": [],
        }
    missing = missing_line_links(order, measurement.fabric_links) if order else ()
    return {
        "kind": "line",
        "measured": True,
        "complete": None if not order else not missing,
        "rank_order": order,
        "links": links,
        "missing_links": [[first, second] for first, second in missing],
    }


def _health_block(operation: OperationRecord | None) -> dict[str, Any] | None:
    if operation is None:
        return None
    return {
        "state": operation.state,
        "endpoint_ready": bool(
            operation.endpoint is not None and operation.endpoint.ready
        ),
        "deployment_id": operation.deployment_id,
        "stopped": operation.stopped,
    }


def _events(
    measurement: MeasurementRecord | None,
    operation: OperationRecord | None,
    paths: dict[str, str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if measurement is not None:
        for warning in measurement.warnings:
            events.append(
                {"source": "measurement", **_warning_block(warning, paths)}
            )
    if operation is not None:
        for warning in operation.warnings:
            events.append(
                {"source": "operation", **_warning_block(warning, paths)}
            )
        if operation.error is not None:
            events.append(
                {
                    "source": "operation",
                    "code": operation.error.code,
                    "node_id": "",
                    "message": _redact_text(operation.error.message, paths),
                    "action": operation.error.action,
                }
            )
    return events
