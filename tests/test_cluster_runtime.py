# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import UTC, datetime, timedelta

from omlx.cluster.performance import NodePerformanceProfile, execution_profile
from omlx.cluster.runtime import read_runtime_markers


def _marker(**overrides):
    return {
        "schema_version": 1,
        "deployment_id": "nemotron-pool",
        "pid": os.getpid(),
        "rank": 1,
        "world_size": 2,
        "model": "/models/nemotron",
        "backend": "jaccl",
        "plan_hash": "a" * 64,
        "phase": "ready",
        "updated_at": datetime.now(UTC).isoformat(),
        "start_layer": 0,
        "end_layer": 26,
    } | overrides


def _assignments():
    gib = 1024**3
    return [
        {
            "node_id": "studio",
            "rank": 0,
            "start_layer": 26,
            "end_layer": 80,
            "layer_count": 54,
            "planned_weight_bytes": 204 * gib,
            "reserve_bytes": 8 * gib,
            "capacity_bytes": 256 * gib,
            "headroom_bytes": 44 * gib,
        },
        {
            "node_id": "mobile",
            "rank": 1,
            "start_layer": 0,
            "end_layer": 26,
            "layer_count": 26,
            "planned_weight_bytes": 96 * gib,
            "reserve_bytes": 8 * gib,
            "capacity_bytes": 128 * gib,
            "headroom_bytes": 24 * gib,
        },
    ]


def _metrics():
    return {
        "scope": "end_to_end_pipeline",
        "active_requests": 0,
        "requests_completed": 3,
        "requests_failed": 0,
        "requests_cancelled": 1,
        "prompt_tokens_total": 1_024,
        "completion_tokens_total": 384,
        "cached_tokens_total": 256,
        "last_request": {
            "status": "completed",
            "prompt_tokens": 512,
            "cached_tokens": 128,
            "completion_tokens": 128,
            "elapsed_seconds": 8.0,
            "ttft_seconds": 2.0,
            "prefill_tps": 192.0,
            "decode_tps": 21.2,
            "end_to_end_tps": 16.0,
            "prefill_progress": {
                "active": False,
                "processed": 384,
                "total": 384,
                "speed": 192.0,
                "average_speed": 192.0,
                "eta": None,
                "elapsed": 2.0,
            },
        },
    }


def test_runtime_markers_report_this_macs_live_rank(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps(_marker()))

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    assert result["jobs"][0]["live"] is True
    assert result["jobs"][0]["rank"] == 1
    assert result["jobs"][0]["start_layer"] == 0
    assert result["jobs"][0]["end_layer"] == 26


def test_runtime_marker_with_reused_live_pid_is_not_reported_as_running(tmp_path):
    payload = _marker(
        updated_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
    )
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    assert result["jobs"][0]["live"] is False


def test_failed_runtime_phase_never_looks_live_while_process_exits(tmp_path):
    payload = _marker(
        phase="launcher_lost",
        error="rank launcher parent changed",
    )
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    assert result["jobs"][0]["phase"] == "launcher_lost"
    assert result["jobs"][0]["live"] is False
    assert result["jobs"][0]["error"] == "rank launcher parent changed"


def test_runtime_marker_rejects_non_string_failure_evidence(tmp_path):
    payload = _marker(phase="failed", error={"unsafe": "shape"})
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert "error must be a string" in result["warnings"][0]


def test_runtime_markers_expose_full_unequal_shard_map_and_pipeline_rates(
    tmp_path,
):
    payload = _marker(
        assignments=_assignments(),
        metrics=_metrics(),
        kv_cache_scope="rank_local",
        load_stage="ready",
        measured_weight_bytes=91 * 1024**3,
    )
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    job = result["jobs"][0]
    assert [item["layer_count"] for item in job["assignments"]] == [54, 26]
    assert job["planned_weight_bytes"] == 96 * 1024**3
    assert job["measured_weight_bytes"] == 91 * 1024**3
    assert job["load_stage"] == "ready"
    assert job["headroom_bytes"] == 24 * 1024**3
    assert job["kv_cache_scope"] == "rank_local"
    assert job["metrics"]["last_request"]["prefill_tps"] == 192.0
    assert job["metrics"]["last_request"]["decode_tps"] == 21.2
    assert job["metrics"]["last_request"]["prefill_progress"] == {
        "active": False,
        "processed": 384,
        "total": 384,
        "speed": 192.0,
        "average_speed": 192.0,
        "eta": None,
        "elapsed": 2.0,
    }
    assert job["metrics"]["requests_cancelled"] == 1


def test_runtime_markers_reject_inconsistent_shard_map(tmp_path):
    assignments = _assignments()
    assignments[1]["end_layer"] = 25
    payload = _marker(assignments=assignments)
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert "contiguous" in result["warnings"][0]


def test_runtime_markers_accept_tensor_parallel_stage_groups(tmp_path):
    gib = 1024**3
    assignments = []
    for rank in range(4):
        stage = rank // 2
        assignments.append(
            {
                "node_id": f"node-{rank}",
                "rank": rank,
                "start_layer": stage * 20,
                "end_layer": (stage + 1) * 20,
                "planned_weight_bytes": 20 * gib,
                "reserve_bytes": 8 * gib,
                "capacity_bytes": 64 * gib,
                "tensor_parallel_size": 2,
                "tensor_parallel_rank": rank % 2,
                "sharded_weight_bytes": 16 * gib,
            }
        )
    payload = _marker(
        rank=3,
        world_size=4,
        start_layer=20,
        end_layer=40,
        assignments=assignments,
        load_stage="ready",
    )
    (tmp_path / "tp.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    job = result["jobs"][0]
    assert job["tensor_parallel_size"] == 2
    assert [item["tensor_parallel_rank"] for item in job["assignments"]] == [
        0,
        1,
        0,
        1,
    ]


def test_runtime_markers_reject_nonfinite_rates(tmp_path):
    metrics = _metrics()
    metrics["last_request"]["decode_tps"] = float("nan")
    payload = _marker(assignments=_assignments(), metrics=metrics)
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert "out of range" in result["warnings"][0]


def test_runtime_markers_reject_impossible_prefill_progress(tmp_path):
    metrics = _metrics()
    metrics["last_request"]["prefill_progress"]["processed"] = 385
    payload = _marker(assignments=_assignments(), metrics=metrics)
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert "prefill progress exceeds" in result["warnings"][0]


def test_runtime_markers_validate_performance_controls_and_live_pipeline_metrics(
    tmp_path,
):
    metrics = _metrics() | {
        "aggregate_decode_tps": 31.5,
        "cache": {
            "affinity": "deployment",
            "lookups": 4,
            "hits": 3,
            "misses": 1,
            "hit_rate": 0.75,
            "tokens_reused": 512,
            "entries": 3,
            "bytes": 4096,
        },
        "pipeline": {
            "batch_steps": 9,
            "busy_seconds": 4.0,
            "idle_seconds": 1.0,
            "utilization": 0.8,
            "microbatch_target": 4,
            "async_overlap": True,
            "last_batch": {
                "step_seconds": 0.2,
                "prompt_responses": 0,
                "generation_responses": 4,
                "coalesced_batch_size": 4,
            },
        },
        "execution": execution_profile("balanced").to_dict(),
        "stage": {
            "rank": 1,
            "predicted_compute_seconds": 0.15,
            "predicted_send_seconds": 0.01,
            "predicted_stage_seconds": 0.16,
            "observed_step_seconds": 0.2,
        },
    }
    profiles = [
        NodePerformanceProfile(
            node_id=item["node_id"],
            rank=item["rank"],
            decode_weight_bytes_per_second=100 + item["rank"],
            prefill_weight_bytes_per_second=200 + item["rank"],
            collective_latency_seconds=0.001,
            collective_bandwidth_bytes_per_second=10_000,
            backend="jaccl",
            measured_at="2026-07-26T12:00:00+00:00",
            samples=5,
        ).to_dict()
        for item in _assignments()
    ]
    optimizations = {
        name: {
            "enabled": True,
            "active": name != "sampling_rank_only",
            "reason": "tested",
        }
        for name in (
            "coalesced_batching",
            "sampling_rank_only",
            "async_overlap",
            "cache_affinity",
            "pipeline_prefill_overlap",
        )
    }
    payload = _marker(
        assignments=_assignments(),
        metrics=metrics,
        execution=execution_profile("balanced").to_dict(),
        performance_profiles=profiles,
        optimizations=optimizations,
    )
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    job = result["jobs"][0]
    assert job["metrics"]["aggregate_decode_tps"] == 31.5
    assert job["metrics"]["cache"]["hit_rate"] == 0.75
    assert job["metrics"]["pipeline"]["utilization"] == 0.8
    assert job["performance_profiles"][1]["node_id"] == "mobile"
    assert job["optimizations"]["sampling_rank_only"]["active"] is False
    assert job["optimizations"]["pipeline_prefill_overlap"]["active"] is True


def test_runtime_markers_ignore_symlinks_and_invalid_json(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("{}")
    (tmp_path / "linked.json").symlink_to(target)
    (tmp_path / "bad.json").write_text("{")

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert len(result["warnings"]) == 2
