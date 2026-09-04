# SPDX-License-Identifier: Apache-2.0
"""Bounded local visibility into distributed rank processes."""

from __future__ import annotations

import json
import math
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .performance import ExecutionSettings, NodePerformanceProfile

_MAX_MARKERS = 64
_MAX_MARKER_BYTES = 64 * 1024
_PHASES = {"loading", "ready", "peer_lost", "launcher_lost", "failed"}
_LOAD_STAGES = {"initializing", "loading_weights", "validating", "ready"}
_BACKENDS = {"ring", "jaccl", "jaccl-ring"}
_REQUEST_STATES = {"running", "completed", "failed", "cancelled"}
_MAX_COUNTER = 2**63 - 1
_RUNTIME_STALE_AFTER_SECONDS = 45.0


def _process_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _marker_is_fresh(updated_at: str) -> bool:
    """A reused PID must not resurrect an old runtime marker in the GUI."""

    try:
        stamp = datetime.fromisoformat(updated_at)
    except (TypeError, ValueError):
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    age = max(0.0, datetime.now(UTC).timestamp() - stamp.timestamp())
    return age <= _RUNTIME_STALE_AFTER_SECONDS


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not 0 <= parsed <= _MAX_COUNTER:
        raise ValueError(f"{label} is out of range")
    return parsed


def _nonnegative_float(
    value: Any,
    label: str,
    *,
    optional: bool = False,
) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} is out of range")
    return parsed


def _validated_assignments(
    value: Any,
    *,
    world_size: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != world_size:
        raise ValueError("runtime shard map must contain every rank")
    assignments: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("runtime shard assignment must be an object")
        node_id = raw.get("node_id")
        if (
            not isinstance(node_id, str)
            or not node_id
            or len(node_id.encode()) > 128
            or any(ord(char) < 32 for char in node_id)
        ):
            raise ValueError("runtime shard assignment has an invalid node ID")
        rank = _nonnegative_int(raw.get("rank"), "assignment rank")
        start = _nonnegative_int(raw.get("start_layer"), "assignment start")
        end = _nonnegative_int(raw.get("end_layer"), "assignment end")
        capacity = _nonnegative_int(
            raw.get("capacity_bytes"),
            "assignment capacity",
        )
        reserve = _nonnegative_int(
            raw.get("reserve_bytes"),
            "assignment reserve",
        )
        planned = _nonnegative_int(
            raw.get("planned_weight_bytes"),
            "assignment planned weight",
        )
        tensor_size = _nonnegative_int(
            raw.get("tensor_parallel_size", 1),
            "assignment tensor parallel size",
        )
        tensor_rank = _nonnegative_int(
            raw.get("tensor_parallel_rank", 0),
            "assignment tensor parallel rank",
        )
        if start >= end or reserve >= capacity or planned > capacity - reserve:
            raise ValueError("runtime shard assignment exceeds its boundaries")
        if tensor_size < 1 or tensor_rank >= tensor_size:
            raise ValueError("runtime tensor parallel assignment is invalid")
        assignment = {
            "node_id": node_id,
            "rank": rank,
            "start_layer": start,
            "end_layer": end,
            "layer_count": end - start,
            "planned_weight_bytes": planned,
            "reserve_bytes": reserve,
            "capacity_bytes": capacity,
            "headroom_bytes": capacity - reserve - planned,
            "tensor_parallel_size": tensor_size,
            "tensor_parallel_rank": tensor_rank,
            "sharded_weight_bytes": _nonnegative_int(
                raw.get("sharded_weight_bytes", 0),
                "assignment sharded weight",
            ),
        }
        predicted_keys = (
            "predicted_compute_seconds",
            "predicted_send_seconds",
            "predicted_stage_seconds",
        )
        if any(raw.get(key) is not None for key in predicted_keys):
            for key in predicted_keys:
                assignment[key] = _nonnegative_float(
                    raw.get(key),
                    f"assignment {key}",
                )
        assignments.append(assignment)
    if sorted(item["rank"] for item in assignments) != list(range(world_size)):
        raise ValueError("runtime shard ranks must be contiguous")
    if len({item["node_id"] for item in assignments}) != world_size:
        raise ValueError("runtime shard node IDs must be unique")
    tensor_sizes = {item["tensor_parallel_size"] for item in assignments}
    if len(tensor_sizes) != 1:
        raise ValueError("runtime tensor parallel sizes disagree")
    tensor_size = tensor_sizes.pop()
    if world_size % tensor_size:
        raise ValueError("runtime tensor parallel size does not divide world size")
    stage_ranges: list[tuple[int, int]] = []
    by_range: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in assignments:
        key = (item["start_layer"], item["end_layer"])
        by_range.setdefault(key, []).append(item)
    for key, group in by_range.items():
        if len(group) != tensor_size or sorted(
            item["tensor_parallel_rank"] for item in group
        ) != list(range(tensor_size)):
            raise ValueError("runtime tensor parallel group is incomplete")
        stage_ranges.append(key)
    ordered_layers = sorted(stage_ranges)
    cursor = 0
    for start_layer, end_layer in ordered_layers:
        if start_layer != cursor:
            raise ValueError("runtime shard layer ranges must be contiguous")
        cursor = end_layer
    return sorted(assignments, key=lambda item: item["rank"])


def _validated_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("scope") != "end_to_end_pipeline":
        raise ValueError("runtime metrics have an invalid scope")
    result: dict[str, Any] = {
        "scope": "end_to_end_pipeline",
    }
    for key in (
        "active_requests",
        "requests_completed",
        "requests_failed",
        "prompt_tokens_total",
        "completion_tokens_total",
        "cached_tokens_total",
    ):
        result[key] = _nonnegative_int(value.get(key), f"metrics {key}")
    result["requests_cancelled"] = _nonnegative_int(
        value.get("requests_cancelled", 0),
        "metrics requests_cancelled",
    )
    result["aggregate_decode_tps"] = _nonnegative_float(
        value.get("aggregate_decode_tps", 0.0),
        "metrics aggregate_decode_tps",
    )

    cache = value.get("cache")
    if cache is not None:
        if not isinstance(cache, dict) or cache.get("affinity") not in {
            "none",
            "deployment",
        }:
            raise ValueError("runtime cache metrics are invalid")
        validated_cache: dict[str, Any] = {"affinity": cache["affinity"]}
        for key in (
            "lookups",
            "hits",
            "misses",
            "tokens_reused",
            "entries",
            "bytes",
        ):
            validated_cache[key] = _nonnegative_int(
                cache.get(key),
                f"cache metrics {key}",
            )
        if (
            validated_cache["hits"] > validated_cache["lookups"]
            or validated_cache["hits"] + validated_cache["misses"]
            != validated_cache["lookups"]
        ):
            raise ValueError("runtime cache counters are inconsistent")
        hit_rate = _nonnegative_float(
            cache.get("hit_rate"),
            "cache metrics hit_rate",
        )
        if hit_rate is None or hit_rate > 1:
            raise ValueError("runtime cache hit rate is out of range")
        validated_cache["hit_rate"] = hit_rate
        result["cache"] = validated_cache

    pipeline = value.get("pipeline")
    if pipeline is not None:
        if not isinstance(pipeline, dict):
            raise ValueError("runtime pipeline metrics are invalid")
        validated_pipeline: dict[str, Any] = {}
        for key in ("batch_steps", "microbatch_target"):
            validated_pipeline[key] = _nonnegative_int(
                pipeline.get(key),
                f"pipeline metrics {key}",
            )
        for key in ("busy_seconds", "idle_seconds", "utilization"):
            validated_pipeline[key] = _nonnegative_float(
                pipeline.get(key),
                f"pipeline metrics {key}",
            )
        if validated_pipeline["utilization"] > 1:
            raise ValueError("runtime pipeline utilization is out of range")
        async_overlap = pipeline.get("async_overlap")
        if not isinstance(async_overlap, bool):
            raise ValueError("runtime async overlap flag is invalid")
        validated_pipeline["async_overlap"] = async_overlap
        last_batch = pipeline.get("last_batch")
        if last_batch is not None:
            if not isinstance(last_batch, dict):
                raise ValueError("runtime last batch metrics are invalid")
            validated_pipeline["last_batch"] = {
                "step_seconds": _nonnegative_float(
                    last_batch.get("step_seconds"),
                    "last batch step_seconds",
                ),
                "prompt_responses": _nonnegative_int(
                    last_batch.get("prompt_responses"),
                    "last batch prompt_responses",
                ),
                "generation_responses": _nonnegative_int(
                    last_batch.get("generation_responses"),
                    "last batch generation_responses",
                ),
                "coalesced_batch_size": _nonnegative_int(
                    last_batch.get("coalesced_batch_size"),
                    "last batch coalesced_batch_size",
                ),
            }
        else:
            validated_pipeline["last_batch"] = None
        result["pipeline"] = validated_pipeline

    execution = value.get("execution")
    if execution is not None:
        result["execution"] = ExecutionSettings.from_dict(execution).to_dict()

    stage = value.get("stage")
    if stage is not None:
        if not isinstance(stage, dict):
            raise ValueError("runtime stage metrics are invalid")
        validated_stage: dict[str, Any] = {
            "rank": _nonnegative_int(stage.get("rank"), "stage rank"),
        }
        for key in (
            "predicted_compute_seconds",
            "predicted_send_seconds",
            "predicted_stage_seconds",
            "observed_step_seconds",
        ):
            validated_stage[key] = _nonnegative_float(
                stage.get(key),
                f"stage {key}",
                optional=True,
            )
        result["stage"] = validated_stage

    current = value.get("last_request")
    if current is None:
        result["last_request"] = None
        return result
    if not isinstance(current, dict) or current.get("status") not in _REQUEST_STATES:
        raise ValueError("runtime last-request metrics are invalid")
    request = {"status": current["status"]}
    for key in ("prompt_tokens", "cached_tokens", "completion_tokens"):
        request[key] = _nonnegative_int(current.get(key), f"metrics {key}")
    if request["cached_tokens"] > request["prompt_tokens"]:
        raise ValueError("runtime cached token count exceeds prompt tokens")
    for key in (
        "elapsed_seconds",
        "prefill_tps",
        "decode_tps",
        "end_to_end_tps",
    ):
        request[key] = _nonnegative_float(current.get(key), f"metrics {key}")
    request["ttft_seconds"] = _nonnegative_float(
        current.get("ttft_seconds"),
        "metrics ttft_seconds",
        optional=True,
    )
    progress = current.get("prefill_progress")
    if progress is not None:
        if not isinstance(progress, dict):
            raise ValueError("runtime prefill progress is invalid")
        active = progress.get("active")
        if not isinstance(active, bool):
            raise ValueError("runtime prefill progress active flag is invalid")
        validated_progress = {
            "active": active,
            "processed": _nonnegative_int(
                progress.get("processed"),
                "metrics prefill processed",
            ),
            "total": _nonnegative_int(
                progress.get("total"),
                "metrics prefill total",
            ),
            "speed": _nonnegative_float(
                progress.get("speed"),
                "metrics prefill speed",
            ),
            # Added after the first live cluster release. Old rank markers only
            # have ``speed`` (the latest chunk), so retain compatibility while
            # giving new dashboards a stable end-to-end running average.
            "average_speed": _nonnegative_float(
                progress.get("average_speed", progress.get("speed")),
                "metrics prefill average speed",
            ),
            "eta": _nonnegative_float(
                progress.get("eta"),
                "metrics prefill eta",
                optional=True,
            ),
            "elapsed": _nonnegative_float(
                progress.get("elapsed"),
                "metrics prefill elapsed",
            ),
        }
        if validated_progress["processed"] > validated_progress["total"]:
            raise ValueError("runtime prefill progress exceeds its total")
        request["prefill_progress"] = validated_progress
    result["last_request"] = request
    return result


def _validated_marker(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported runtime marker schema")
    required_strings = (
        "deployment_id",
        "model",
        "backend",
        "plan_hash",
        "phase",
        "updated_at",
    )
    if any(not isinstance(payload.get(key), str) for key in required_strings):
        raise ValueError("runtime marker has invalid string fields")
    try:
        pid = int(payload["pid"])
        rank = int(payload["rank"])
        world_size = int(payload["world_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("runtime marker has invalid rank fields") from exc
    if pid <= 0 or pid > 2**31 - 1 or not 0 <= rank < world_size <= 64:
        raise ValueError("runtime marker rank metadata is out of range")
    phase = payload["phase"]
    backend = payload["backend"]
    plan_hash = payload["plan_hash"]
    if phase not in _PHASES or backend not in _BACKENDS:
        raise ValueError("runtime marker has an invalid phase or backend")
    if len(plan_hash) != 64 or any(
        char not in "0123456789abcdef" for char in plan_hash
    ):
        raise ValueError("runtime marker plan hash is invalid")
    marker = {
        "deployment_id": payload["deployment_id"][:128],
        "pid": pid,
        "rank": rank,
        "world_size": world_size,
        "model": payload["model"][:4096],
        "backend": backend,
        "plan_hash": plan_hash,
        "phase": phase,
        "updated_at": payload["updated_at"][:64],
        "live": (
            phase in {"loading", "ready"}
            and _process_is_live(pid)
            and _marker_is_fresh(payload["updated_at"])
        ),
    }
    error = payload.get("error")
    if error is not None:
        if not isinstance(error, str):
            raise ValueError("runtime marker error must be a string")
        marker["error"] = error[:1000]
    load_stage = payload.get("load_stage")
    if load_stage is not None:
        if load_stage not in _LOAD_STAGES:
            raise ValueError("runtime marker has an invalid load stage")
        if phase == "ready" and load_stage != "ready":
            raise ValueError("ready runtime marker has an incomplete load stage")
        marker["load_stage"] = load_stage
    has_start = "start_layer" in payload
    has_end = "end_layer" in payload
    if has_start != has_end:
        raise ValueError("runtime marker has an incomplete local layer range")
    if has_start:
        marker["start_layer"] = _nonnegative_int(
            payload["start_layer"],
            "runtime start_layer",
        )
        marker["end_layer"] = _nonnegative_int(
            payload["end_layer"],
            "runtime end_layer",
        )
        if marker["start_layer"] >= marker["end_layer"]:
            raise ValueError("runtime marker has an invalid local layer range")
    assignments = payload.get("assignments")
    if assignments is not None:
        marker["assignments"] = _validated_assignments(
            assignments,
            world_size=world_size,
        )
        local = marker["assignments"][rank]
        if (
            marker.get("start_layer", local["start_layer"]) != local["start_layer"]
            or marker.get("end_layer", local["end_layer"]) != local["end_layer"]
        ):
            raise ValueError("runtime local layer range disagrees with shard map")
        marker.update(
            {
                "planned_weight_bytes": local["planned_weight_bytes"],
                "capacity_bytes": local["capacity_bytes"],
                "reserve_bytes": local["reserve_bytes"],
                "headroom_bytes": local["headroom_bytes"],
                "tensor_parallel_size": local["tensor_parallel_size"],
                "tensor_parallel_rank": local["tensor_parallel_rank"],
                "kv_cache_scope": (
                    "rank_local"
                    if payload.get("kv_cache_scope") == "rank_local"
                    else "unknown"
                ),
            }
        )
    for key in (
        "measured_weight_bytes",
        "admission_ceiling_bytes",
        "admission_budget_bytes",
        "wired_limit_bytes",
        "load_memory_bytes",
    ):
        if payload.get(key) is not None:
            marker[key] = _nonnegative_int(
                payload[key],
                f"runtime {key}",
            )
    measured = marker.get("measured_weight_bytes")
    capacity = marker.get("capacity_bytes")
    if measured is not None and capacity is not None and measured > capacity:
        raise ValueError("runtime measured weights exceed node capacity")
    metrics = payload.get("metrics")
    if metrics is not None:
        marker["metrics"] = _validated_metrics(metrics)
    execution = payload.get("execution")
    if execution is not None:
        marker["execution"] = ExecutionSettings.from_dict(execution).to_dict()
    profiles = payload.get("performance_profiles")
    if profiles is not None:
        if not isinstance(profiles, list) or len(profiles) not in {
            0,
            world_size,
        }:
            raise ValueError("runtime performance profiles are invalid")
        marker["performance_profiles"] = [
            NodePerformanceProfile.from_dict(profile).to_dict() for profile in profiles
        ]
    capabilities = payload.get("optimizations")
    if capabilities is not None:
        if not isinstance(capabilities, dict):
            raise ValueError("runtime optimization capabilities are invalid")
        safe_capabilities: dict[str, Any] = {}
        for key in (
            "coalesced_batching",
            "sampling_rank_only",
            "async_overlap",
            "cache_affinity",
        ):
            item = capabilities.get(key)
            if not isinstance(item, dict):
                raise ValueError("runtime optimization capability is invalid")
            enabled = item.get("enabled")
            active = item.get("active")
            reason = item.get("reason")
            if (
                not isinstance(enabled, bool)
                or not isinstance(active, bool)
                or not isinstance(reason, str)
            ):
                raise ValueError("runtime optimization capability is invalid")
            safe_capabilities[key] = {
                "enabled": enabled,
                "active": active,
                "reason": reason[:500],
            }
        optional = capabilities.get("pipeline_prefill_overlap")
        if optional is not None:
            if not isinstance(optional, dict):
                raise ValueError("runtime optimization capability is invalid")
            enabled = optional.get("enabled")
            active = optional.get("active")
            reason = optional.get("reason")
            if (
                not isinstance(enabled, bool)
                or not isinstance(active, bool)
                or not isinstance(reason, str)
            ):
                raise ValueError("runtime optimization capability is invalid")
            safe_capabilities["pipeline_prefill_overlap"] = {
                "enabled": enabled,
                "active": active,
                "reason": reason[:500],
            }
        marker["optimizations"] = safe_capabilities
    return marker


def read_runtime_markers(
    state_dir: str | Path = "~/.omlx/cluster/runtime",
) -> dict[str, Any]:
    """Read regular, size-bounded marker files without mutating local state."""

    root = Path(state_dir).expanduser()
    if not root.exists():
        return {"jobs": [], "warnings": []}
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        return {"jobs": [], "warnings": [f"runtime state unavailable: {exc}"]}

    jobs: list[dict[str, Any]] = []
    warnings: list[str] = []
    candidates = [path for path in entries if path.name.endswith(".json")]
    if len(candidates) > _MAX_MARKERS:
        warnings.append(f"runtime marker limit exceeded; showing first {_MAX_MARKERS}")
    for path in candidates[:_MAX_MARKERS]:
        descriptor = -1
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > _MAX_MARKER_BYTES
            ):
                raise ValueError("marker is not a bounded regular file")
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                raw = stream.read(_MAX_MARKER_BYTES + 1)
            if len(raw.encode()) > _MAX_MARKER_BYTES:
                raise ValueError("runtime marker is too large")
            payload = json.loads(raw)
            jobs.append(_validated_marker(payload))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"{path.name}: {exc}")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    return {
        "jobs": sorted(
            jobs,
            key=lambda job: (job["deployment_id"], job["rank"]),
        ),
        "warnings": warnings,
    }
