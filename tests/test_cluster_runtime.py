# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

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


def test_runtime_reader_ignores_launch_and_control_json_files(tmp_path):
    (tmp_path / "job.json").write_text(json.dumps(_marker()))
    for name in (
        "launch-deployment.json",
        "deployment-cancel.json",
        "deployment-cancel-ack.json",
        "deployment-serve.json",
    ):
        (tmp_path / name).write_text("not a rank marker", encoding="utf-8")

    result = read_runtime_markers(tmp_path)

    assert len(result["jobs"]) == 1
    assert result["warnings"] == []


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


@pytest.mark.parametrize(
    "load_stage",
    (
        "initializing",
        "initializing_full_replica",
        "loading_weights",
        "materializing_fixed",
        "materializing_layers",
        "tensor_ready",
        "weights_resident",
        "validating",
        "warming_prefill_shape",
    ),
)
def test_runtime_preserves_real_intermediate_load_stages(tmp_path, load_stage):
    (tmp_path / "job.json").write_text(
        json.dumps(_marker(phase="loading", load_stage=load_stage))
    )

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    assert result["jobs"][0]["load_stage"] == load_stage


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


def test_runtime_markers_admit_full_replica_phase_ownership_and_handoff(tmp_path):
    assignments = _assignments()
    for assignment in assignments:
        assignment["start_layer"] = 0
        assignment["end_layer"] = 80
        assignment["layer_count"] = 80
    phase = {
        "handoffs_completed": 3,
        "last_handoff_bytes": 2_000_000_000,
        "last_handoff_arrays": 128,
        "last_handoff_seconds": 0.25,
        "last_handoff_bytes_per_second": 8_000_000_000.0,
        "queue_depth": 1,
    }
    payload = _marker(
        start_layer=0,
        end_layer=80,
        assignments=assignments,
        serving_mode="disaggregated",
        prefill_rank=1,
        decode_rank=0,
        metrics=_metrics() | {"phase_split": phase},
    )
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    job = result["jobs"][0]
    assert job["serving_mode"] == "disaggregated"
    assert job["prefill_rank"] == 1
    assert job["decode_rank"] == 0
    assert job["metrics"]["phase_split"] == phase
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


def _tiered_cache_metrics():
    return {
        "affinity": "deployment",
        "lookups": 5,
        "hits": 3,
        "misses": 2,
        "hit_rate": 0.6,
        "tokens_reused": 6_144,
        "entries": 9,
        "bytes": 69_632,
        "ssd_enabled": True,
        "memory": {"entries": 2, "bytes": 4_096, "hits": 1},
        "ssd": {
            "entries": 7,
            "bytes": 65_536,
            "hits": 2,
            "max_bytes": 20 * 1024**3,
            "capacity_bytes": 70_000,
            "evictions": 3,
            "capacity_drops": 1,
            "pending_bytes": 0,
            "pending_max_bytes": 512 * 1024**2,
            "write_failures": 0,
        },
    }


def test_runtime_markers_preserve_validated_cache_tier_metrics(tmp_path):
    metrics = _metrics() | {"cache": _tiered_cache_metrics()}
    payload = _marker(assignments=_assignments(), metrics=metrics)
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    assert result["jobs"][0]["metrics"]["cache"] == metrics["cache"]


def test_runtime_markers_reject_inconsistent_cache_tier_totals(tmp_path):
    cache = _tiered_cache_metrics()
    cache["ssd"]["bytes"] += 1
    metrics = _metrics() | {"cache": cache}
    (tmp_path / "job.json").write_text(
        json.dumps(_marker(assignments=_assignments(), metrics=metrics))
    )

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert "cache tier totals are inconsistent" in result["warnings"][0]


def test_runtime_markers_reject_partial_or_disabled_ssd_tier_metrics(tmp_path):
    partial = _tiered_cache_metrics()
    partial.pop("memory")
    (tmp_path / "partial.json").write_text(
        json.dumps(
            _marker(assignments=_assignments(), metrics=_metrics() | {"cache": partial})
        )
    )
    disabled = _tiered_cache_metrics()
    disabled["ssd_enabled"] = False
    (tmp_path / "disabled.json").write_text(
        json.dumps(
            _marker(
                assignments=_assignments(), metrics=_metrics() | {"cache": disabled}
            )
        )
    )

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert any(
        "cache tier metrics are incomplete" in item for item in result["warnings"]
    )
    assert any(
        "SSD tier is populated while disabled" in item for item in result["warnings"]
    )


def test_runtime_markers_preserve_distinct_active_request_rates(tmp_path):
    metrics = _metrics()
    first = metrics["last_request"] | {
        "request_id": 7,
        "status": "running",
        "prefill_tps": 410.0,
        "decode_tps": 0.0,
    }
    second = metrics["last_request"] | {
        "request_id": 8,
        "status": "running",
        "prefill_tps": 205.0,
        "decode_tps": 37.5,
    }
    metrics |= {
        "active_requests": 2,
        "active_request_metrics": [first, second],
        "active_request_metrics_truncated": 0,
        "last_request": second,
    }
    payload = _marker(assignments=_assignments(), metrics=metrics)
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    requests = result["jobs"][0]["metrics"]["active_request_metrics"]
    assert [
        (item["request_id"], item["prefill_tps"], item["decode_tps"])
        for item in requests
    ] == [
        (7, 410.0, 0.0),
        (8, 205.0, 37.5),
    ]


def test_runtime_markers_reject_duplicate_active_request_ids(tmp_path):
    metrics = _metrics()
    request = metrics["last_request"] | {
        "request_id": 7,
        "status": "running",
    }
    metrics |= {
        "active_requests": 2,
        "active_request_metrics": [request, dict(request)],
        "active_request_metrics_truncated": 0,
    }
    (tmp_path / "job.json").write_text(json.dumps(_marker(metrics=metrics)))

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert "IDs are not unique" in result["warnings"][0]


def test_runtime_markers_reject_impossible_prefill_progress(tmp_path):
    metrics = _metrics()
    metrics["last_request"]["prefill_progress"]["processed"] = 385
    payload = _marker(assignments=_assignments(), metrics=metrics)
    (tmp_path / "job.json").write_text(json.dumps(payload))

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert "prefill progress exceeds" in result["warnings"][0]


def test_runtime_markers_validate_mtp_cycle_economics(tmp_path):
    metrics = _metrics()
    metrics["mtp"] = {
        "sequences": 3,
        "tokens": 360,
        "cycles": 120,
        "accepted_draft_tokens": 240,
        "drafted_tokens": 300,
        "zero_depth_cycles": 2,
        "acceptance_ratio": 0.8,
        "tokens_per_cycle": 3.0,
        "depth_drafted": [120, 100, 80],
        "depth_accepted": [115, 85, 40],
        "timing_ms": {
            "backbone": 2_400.0,
            "mtp_head": 600.0,
            "sampling": 60.0,
            "cache_ops": 30.0,
        },
        "last_finish_reason": "length",
    }
    (tmp_path / "job.json").write_text(
        json.dumps(_marker(assignments=_assignments(), metrics=metrics))
    )

    result = read_runtime_markers(tmp_path)

    assert result["warnings"] == []
    assert result["jobs"][0]["metrics"]["mtp"] == metrics["mtp"]


def test_runtime_markers_reject_impossible_mtp_acceptance(tmp_path):
    metrics = _metrics()
    metrics["mtp"] = {
        "sequences": 1,
        "tokens": 20,
        "cycles": 10,
        "accepted_draft_tokens": 11,
        "drafted_tokens": 10,
        "zero_depth_cycles": 0,
        "acceptance_ratio": 1.1,
        "tokens_per_cycle": 2.0,
        "depth_drafted": [10],
        "depth_accepted": [11],
        "timing_ms": {
            "backbone": 1.0,
            "mtp_head": 1.0,
            "sampling": 1.0,
            "cache_ops": 1.0,
        },
        "last_finish_reason": "length",
    }
    (tmp_path / "job.json").write_text(json.dumps(_marker(metrics=metrics)))

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert "accepted count exceeds drafted" in result["warnings"][0]


def test_runtime_markers_validate_performance_controls_and_live_pipeline_metrics(
    tmp_path,
):
    metrics = _metrics() | {
        "aggregate_decode_tps": 31.5,
        "average_request_decode_tps": 9.25,
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
            "prefill_logits_skip",
            "prefill_allocator_reuse",
            "sparse_indexer_row_parallel",
            "deepseek_v4_fused_decode_attention",
            "deepseek_v4_adaptive_prefill",
            "deepseek_v4_prefill_yield",
            "deepseek_v4_prefill_async",
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
    assert job["metrics"]["average_request_decode_tps"] == 9.25
    assert job["metrics"]["cache"]["hit_rate"] == 0.75
    assert job["metrics"]["pipeline"]["utilization"] == 0.8
    assert job["performance_profiles"][1]["node_id"] == "mobile"
    assert job["optimizations"]["sampling_rank_only"]["active"] is False
    assert job["optimizations"]["pipeline_prefill_overlap"]["active"] is True
    assert job["optimizations"]["prefill_logits_skip"]["active"] is True
    assert job["optimizations"]["prefill_allocator_reuse"]["active"] is True
    assert job["optimizations"]["sparse_indexer_row_parallel"]["active"] is True
    assert job["optimizations"]["deepseek_v4_fused_decode_attention"]["active"] is True
    assert job["optimizations"]["deepseek_v4_adaptive_prefill"]["active"] is True
    assert job["optimizations"]["deepseek_v4_prefill_yield"]["active"] is True
    assert job["optimizations"]["deepseek_v4_prefill_async"]["active"] is True


def test_runtime_markers_ignore_symlinks_and_invalid_json(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("{}")
    (tmp_path / "linked.json").symlink_to(target)
    (tmp_path / "bad.json").write_text("{")

    result = read_runtime_markers(tmp_path)

    assert result["jobs"] == []
    assert len(result["warnings"]) == 2
