# SPDX-License-Identifier: Apache-2.0
"""Tests for rank-local, end-to-end distributed inference telemetry."""

import json
import threading
import time
from io import BytesIO
from types import SimpleNamespace

import pytest

from omlx.cluster.performance import execution_profile
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.telemetry import (
    RuntimeTelemetry,
    _agreed_snapshot_capacity_charge,
    _capture_prompt_boundary_cache,
    _python_token_id,
    _TelemetryQueue,
    install_server_telemetry,
)


def test_snapshot_capacity_agreement_never_falls_back_to_tiny_collective():
    assert _agreed_snapshot_capacity_charge(
        123,
        world_size=1,
        rank=0,
        control_plane=None,
    ) == 123
    assert _agreed_snapshot_capacity_charge(
        123,
        world_size=2,
        rank=0,
        control_plane=None,
    ) is None


def test_snapshot_capacity_agreement_uses_largest_reliable_rank_charge():
    class ControlPlane:
        def __init__(self):
            self.calls = []

        def broadcast_owned_bytes(self, payload, *, source_rank, expected_size):
            self.calls.append((payload, source_rank, expected_size))
            if source_rank == 0:
                return payload
            return b"\x01" + (456).to_bytes(8, "big")

    control_plane = ControlPlane()

    assert _agreed_snapshot_capacity_charge(
        123,
        world_size=2,
        rank=0,
        control_plane=control_plane,
    ) == 456
    assert control_plane.calls == [
        (b"\x01" + (123).to_bytes(8, "big"), 0, 9),
        (None, 1, 9),
    ]


def test_phase_handoff_metrics_publish_real_bytes_and_bandwidth():
    telemetry = RuntimeTelemetry(_Marker(), clock=_Clock(), publish_interval=0)

    telemetry.observe_phase_handoff(
        tensor_bytes=2_000_000_000,
        array_count=128,
        elapsed_seconds=0.25,
        queue_depth=3,
    )

    phase = telemetry.snapshot()["phase_split"]
    assert phase == {
        "handoffs_completed": 1,
        "last_handoff_bytes": 2_000_000_000,
        "last_handoff_arrays": 128,
        "last_handoff_seconds": 0.25,
        "last_handoff_bytes_per_second": 8_000_000_000.0,
        "queue_depth": 3,
    }


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _Marker:
    def __init__(self) -> None:
        self.updates = []

    def update(self, phase, **extra):
        self.updates.append((phase, extra))


class _Queue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item, *args, **kwargs):
        self.items.append((item, args, kwargs))
        return "queued"


def test_private_rank_cache_clear_endpoint_is_plan_authenticated(tmp_path):
    import mlx_lm.server as mlx_server

    class Marker(_Marker):
        payload = {"deployment_id": "dep", "plan_hash": "p" * 64}
        path = None

    class Handler:
        path = "/omlx/internal/cache/ssd/clear"

        def __init__(self, token):
            self.headers = {"X-oMLX-Plan-Hash": token}
            self.wfile = BytesIO()
            self.status = None

        def _set_completion_headers(self, status):
            self.status = status

        def end_headers(self):
            return None

    with install_server_telemetry(
        Marker(),
        heartbeat_interval=0,
        ssd_cache_dir=str(tmp_path),
        ssd_cache_persistent=True,
        ssd_write_behind=True,
    ):
        allowed = Handler("p" * 64)
        mlx_server.APIHandler.do_POST(allowed)
        denied = Handler("wrong")
        mlx_server.APIHandler.do_POST(denied)

    assert allowed.status == 200
    assert json.loads(allowed.wfile.getvalue()) == {
        "status": "ok",
        "rank": 0,
        "ssd_deleted": 0,
        "hot_cleared": 0,
    }
    assert denied.status == 403


def test_peer_cache_clear_marker_is_heartbeat_acknowledged(tmp_path, monkeypatch):
    import mlx.core as mx

    class Group:
        rank = staticmethod(lambda: 1)
        size = staticmethod(lambda: 2)

    class Marker(_Marker):
        payload = {"deployment_id": "dep", "plan_hash": "p" * 64}
        path = tmp_path / "dep-rank-1.json"

    monkeypatch.setattr(mx.distributed, "init", lambda: Group())
    request_path = tmp_path / "dep-cache-clear.json"
    ack_path = tmp_path / "dep-cache-clear-rank-1.json"
    with install_server_telemetry(Marker(), heartbeat_interval=0) as telemetry:
        epoch = time.time_ns()
        request_path.write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "deployment_id": "dep",
                    "plan_hash": "p" * 64,
                    "ssd": True,
                    "hot": False,
                }
            )
        )
        telemetry.heartbeat()

    ack = json.loads(ack_path.read_text())
    assert ack == {
        "status": "ok",
        "rank": 1,
        "epoch": epoch,
        "ssd_deleted": 0,
        "hot_cleared": 0,
    }
    assert not request_path.exists()


def test_generated_token_is_normalized_for_logprob_indexing():
    class Scalar:
        def item(self):
            return 129_279

    assert _python_token_id(Scalar()) == 129_279
    assert _python_token_id([[42]]) == 42


def test_generated_token_normalization_rejects_non_scalar_and_high_bit():
    import pytest

    with pytest.raises(ValueError, match="scalar"):
        _python_token_id([1, 2])
    with pytest.raises(ValueError, match="signed int32"):
        _python_token_id(2**31)


def test_prompt_boundary_capture_uses_the_pre_decode_cache_key():
    class PromptCache:
        def __init__(self):
            self.insertions = []

        def insert_cache(self, model, tokens, cache, *, cache_type):
            self.insertions.append((model, tokens, cache, cache_type))

    class BatchGenerator:
        def extract_cache(self, uids):
            assert uids == [17]
            return {17: (["prompt-cache"], [1, 2, 3])}

    prompt_cache = PromptCache()
    response = SimpleNamespace(uid=17, end_of_prompt=True)

    assert _capture_prompt_boundary_cache(
        prompt_cache,
        "model-key",
        BatchGenerator(),
        response.uid,
    ) == 3
    assert prompt_cache.insertions == [
        ("model-key", [1, 2, 3], ["prompt-cache"], "user")
    ]


def test_prompt_boundary_capture_ignores_non_boundary_responses():
    assert _capture_prompt_boundary_cache(
        object(),
        "model-key",
        object(),
        None,
    ) == 0


def test_telemetry_calculates_ttft_prefill_and_decode_rates():
    clock = _Clock()
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()

    clock.value = 0.5
    telemetry.observe_context(
        request_id,
        prompt_tokens=10,
        cached_tokens=2,
    )
    clock.value = 1.0
    telemetry.observe_token(request_id)
    clock.value = 2.0
    telemetry.observe_token(request_id)
    clock.value = 3.0
    telemetry.finish_request(request_id)

    snapshot = telemetry.snapshot()
    request = snapshot["last_request"]
    assert snapshot["scope"] == "end_to_end_pipeline"
    assert snapshot["active_requests"] == 0
    assert snapshot["requests_completed"] == 1
    assert snapshot["requests_cancelled"] == 0
    assert snapshot["prompt_tokens_total"] == 10
    assert snapshot["completion_tokens_total"] == 2
    assert request["ttft_seconds"] == 1.0
    assert request["prefill_tps"] == 8.0
    assert request["decode_tps"] == 0.5
    assert request["end_to_end_tps"] == 2 / 3
    assert marker.updates[-1][0] == "ready"


def test_telemetry_publishes_structured_mtp_economics(monkeypatch):
    from omlx.patches.mlx_lm_mtp import batch_generator

    expected = {
        "sequences": 1,
        "tokens": 120,
        "cycles": 40,
        "accepted_draft_tokens": 80,
        "drafted_tokens": 100,
        "zero_depth_cycles": 0,
        "acceptance_ratio": 0.8,
        "tokens_per_cycle": 3.0,
        "depth_drafted": [40, 35, 25],
        "depth_accepted": [38, 30, 12],
        "timing_ms": {
            "backbone": 800.0,
            "mtp_head": 200.0,
            "sampling": 20.0,
            "cache_ops": 10.0,
        },
        "last_finish_reason": "length",
    }
    monkeypatch.setattr(batch_generator, "mtp_runtime_stats_snapshot", lambda: expected)

    snapshot = RuntimeTelemetry(_Marker(), clock=_Clock()).snapshot()

    assert snapshot["mtp"] == expected


def test_average_request_decode_tps_uses_decode_time_not_uptime():
    """Idle uptime must not dilute the average per-request decode rate.

    The old divisor (process uptime) reported ~2 tok/s on a deployment whose
    requests decoded at ~23 tok/s, because a serving rank is mostly idle
    between requests.
    """
    clock = _Clock()
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()
    telemetry.observe_context(request_id, prompt_tokens=10, cached_tokens=0)

    clock.value = 10.0  # queue + prefill before the first decoded token
    telemetry.observe_token(request_id)
    clock.value = 11.0
    telemetry.observe_token(request_id)
    clock.value = 12.0
    telemetry.finish_request(request_id)

    # 90 s of idle uptime before anyone reads the marker.
    clock.value = 102.0
    snapshot = telemetry.snapshot()
    # 2 tokens over the 2 s decode window (first token -> finish).
    assert snapshot["average_request_decode_tps"] == 1.0
    assert snapshot["aggregate_decode_tps"] == 0.0
    # The old semantic survives under an honest name.
    assert snapshot["aggregate_wall_tps"] == 2 / 102.0

    # An in-flight request contributes its decode window while active.
    request_id = telemetry.begin_request()
    telemetry.observe_context(request_id, prompt_tokens=4, cached_tokens=0)
    clock.value = 202.0
    telemetry.observe_token(request_id)
    clock.value = 204.0
    telemetry.observe_token(request_id)
    snapshot = telemetry.snapshot()
    # (2 finished tokens + 2 active) / (2 s finished + 2 s active).
    assert snapshot["average_request_decode_tps"] == 1.0


@pytest.mark.parametrize(
    ("generation_rows", "elapsed", "expected"),
    ((1, 0.05, 20.0), (2, 0.05, 40.0), (4, 0.05, 80.0)),
)
def test_aggregate_decode_tps_counts_overlapping_b1_b2_b4_steps(
    generation_rows, elapsed, expected
):
    telemetry = RuntimeTelemetry(_Marker(), clock=_Clock(), publish_interval=0)

    telemetry.observe_batch_step(
        prompt_responses=3,
        generation_responses=generation_rows,
        elapsed_seconds=elapsed,
    )

    snapshot = telemetry.snapshot()
    assert snapshot["aggregate_decode_tps"] == pytest.approx(expected)
    assert snapshot["average_request_decode_tps"] == 0.0


def test_aggregate_decode_tps_ignores_prompt_only_step_time():
    telemetry = RuntimeTelemetry(_Marker(), clock=_Clock(), publish_interval=0)
    telemetry.observe_batch_step(
        prompt_responses=4,
        generation_responses=0,
        elapsed_seconds=1.0,
    )
    telemetry.observe_batch_step(
        prompt_responses=0,
        generation_responses=4,
        elapsed_seconds=0.1,
    )

    assert telemetry.snapshot()["aggregate_decode_tps"] == pytest.approx(40.0)


def test_aggregate_decode_tps_accumulates_b1_b2_b4_step_math():
    telemetry = RuntimeTelemetry(_Marker(), clock=_Clock(), publish_interval=0)
    for rows, elapsed in ((1, 0.02), (2, 0.04), (4, 0.08)):
        telemetry.observe_batch_step(
            prompt_responses=0,
            generation_responses=rows,
            elapsed_seconds=elapsed,
        )

    # Seven generated row-tokens over 140 ms of actual generation-step wall.
    assert telemetry.snapshot()["aggregate_decode_tps"] == pytest.approx(50.0)


def test_internal_readiness_request_is_not_published_as_user_traffic():
    clock = _Clock()
    telemetry = RuntimeTelemetry(_Marker(), clock=clock, publish_interval=0)
    request_id = telemetry.begin_request("omlx-internal-readiness")
    telemetry.observe_context(request_id, prompt_tokens=8, cached_tokens=0)
    clock.value = 0.5
    telemetry.observe_token(request_id)
    telemetry.observe_batch_step(
        prompt_responses=0,
        generation_responses=1,
        elapsed_seconds=0.05,
    )

    running = telemetry.snapshot()
    assert running["active_requests"] == 0
    assert running["active_request_metrics"] == []
    assert running["aggregate_decode_tps"] == 0.0

    telemetry.finish_request(request_id)
    finished = telemetry.snapshot()
    assert finished["requests_completed"] == 0
    assert finished["prompt_tokens_total"] == 0
    assert finished["completion_tokens_total"] == 0
    assert finished["last_request"] is None


def test_telemetry_publishes_live_mlx_lm_prefill_progress():
    clock = _Clock()
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()

    clock.value = 0.25
    telemetry.observe_context(
        request_id,
        prompt_tokens=12_000,
        cached_tokens=4_000,
    )
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((73,))

    clock.value = 2.25
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=2_000,
        total_tokens=8_000,
    )

    request = telemetry.snapshot()["last_request"]
    progress = request["prefill_progress"]
    assert request["status"] == "running"
    assert request["ttft_seconds"] is None
    assert request["decode_tps"] == 0.0
    assert request["prefill_tps"] == 1_000.0
    assert progress == {
        "active": True,
        "processed": 2_000,
        "total": 8_000,
        "speed": 1_000.0,
        "average_speed": 1_000.0,
        "eta": 6.0,
        "elapsed": 2.0,
    }

    clock.value = 4.25
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=4_000,
        total_tokens=8_000,
    )
    progress = telemetry.snapshot()["last_request"]["prefill_progress"]
    assert progress["processed"] == 4_000
    assert progress["speed"] == 1_000.0
    assert progress["average_speed"] == 1_000.0
    assert progress["eta"] == 4.0

    clock.value = 8.25
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=8_000,
        total_tokens=8_000,
    )
    telemetry.observe_token(request_id)
    request = telemetry.snapshot()["last_request"]
    assert request["prefill_progress"]["active"] is False
    assert request["prefill_progress"]["processed"] == 8_000
    assert request["ttft_seconds"] == 8.25


def test_completed_prefill_progress_clock_stops_at_prefill_end():
    """Decode time must not leak into prefill_progress elapsed/average.

    Observed on the live TP=2 deployment: a request with ttft 2.49 s reported
    prefill_progress.elapsed 25.3 s and average_speed 16 tok/s because the
    snapshot divided processed tokens by (finish_time - prefill_start).
    """

    clock = _Clock()
    telemetry = RuntimeTelemetry(_Marker(), clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()

    clock.value = 0.5
    telemetry.observe_context(request_id, prompt_tokens=407, cached_tokens=0)
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((73,))

    # Single 407-token chunk finishes prefill at t=2.99 (≈163 tok/s).
    clock.value = 2.99
    telemetry.observe_prefill_progress(73, processed_tokens=407, total_tokens=407)
    telemetry.observe_token(request_id)

    # Decode runs for ~23 s, then the request finishes.
    clock.value = 26.0
    for _ in range(545):
        telemetry.observe_token(request_id)
    telemetry.finish_request(request_id)

    request = telemetry.snapshot()["last_request"]
    progress = request["prefill_progress"]
    assert request["ttft_seconds"] == 2.99
    assert request["elapsed_seconds"] == 26.0
    assert progress["active"] is False
    # Frozen at the final chunk callback, not at finish time.
    assert progress["elapsed"] == 2.49
    assert progress["average_speed"] == 407 / 2.49
    assert request["prefill_tps"] == 407 / 2.99

    # A later heartbeat re-snapshot must not age the completed prefill either.
    clock.value = 60.0
    progress = telemetry.snapshot()["last_request"]["prefill_progress"]
    assert progress["elapsed"] == 2.49
    assert progress["average_speed"] == 407 / 2.49


def test_live_prefill_separates_recent_chunk_rate_from_sustained_average():
    """A slow later chunk must not relabel the whole request as 200 tok/s."""

    clock = _Clock()
    telemetry = RuntimeTelemetry(_Marker(), clock=clock, publish_interval=0)
    request_id = telemetry.begin_request()
    telemetry.observe_context(
        request_id,
        prompt_tokens=8_000,
        cached_tokens=0,
    )
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((73,))

    clock.value = 2.0
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=2_000,
        total_tokens=8_000,
    )
    clock.value = 12.0
    telemetry.observe_prefill_progress(
        73,
        processed_tokens=4_000,
        total_tokens=8_000,
    )

    request = telemetry.snapshot()["last_request"]
    progress = request["prefill_progress"]
    assert progress["speed"] == 200.0
    assert progress["average_speed"] == 4_000 / 12
    assert request["prefill_tps"] == 4_000 / 12
    assert progress["eta"] == 20.0


def test_concurrent_requests_keep_separate_live_prefill_and_decode_rates():
    """A newer request must not overwrite another request's live rate."""

    clock = _Clock()
    telemetry = RuntimeTelemetry(_Marker(), clock=clock, publish_interval=0)

    prefill_id = telemetry.begin_request()
    telemetry.observe_context(prefill_id, prompt_tokens=100, cached_tokens=0)
    telemetry.mark_pending_uid(prefill_id)
    telemetry.bind_pending_uid((71,))

    clock.value = 1.0
    decode_id = telemetry.begin_request()
    telemetry.observe_context(decode_id, prompt_tokens=20, cached_tokens=0)
    clock.value = 2.0
    telemetry.observe_token(decode_id)
    clock.value = 3.0
    telemetry.observe_token(decode_id)
    clock.value = 4.0
    telemetry.observe_prefill_progress(
        71,
        processed_tokens=40,
        total_tokens=100,
    )

    snapshot = telemetry.snapshot()
    requests = snapshot["active_request_metrics"]

    assert snapshot["active_requests"] == 2
    assert snapshot["active_request_metrics_truncated"] == 0
    assert [request["request_id"] for request in requests] == [
        prefill_id,
        decode_id,
    ]
    assert requests[0]["prefill_progress"]["active"] is True
    assert requests[0]["prefill_tps"] == 10.0
    assert requests[0]["decode_tps"] == 0.0
    assert requests[1]["prefill_progress"]["active"] is False
    assert requests[1]["prefill_tps"] == 20.0
    assert requests[1]["decode_tps"] == 0.5

    telemetry.finish_request(decode_id)
    snapshot = telemetry.snapshot()
    assert [
        request["request_id"] for request in snapshot["active_request_metrics"]
    ] == [prefill_id]


def test_active_request_detail_is_bounded_without_losing_total_count():
    telemetry = RuntimeTelemetry(_Marker(), clock=_Clock(), publish_interval=0)

    for _ in range(66):
        telemetry.begin_request()

    snapshot = telemetry.snapshot()

    assert snapshot["active_requests"] == 66
    assert len(snapshot["active_request_metrics"]) == 64
    assert snapshot["active_request_metrics_truncated"] == 2
    assert snapshot["active_request_metrics"][0]["request_id"] == 1
    assert snapshot["active_request_metrics"][-1]["request_id"] == 64


def test_queue_observer_preserves_mlx_lm_queue_contract():
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, publish_interval=0)
    target = _Queue()
    queue = _TelemetryQueue(target, telemetry)
    context = SimpleNamespace(prompt=[1, 2, 3, 4], prompt_cache_count=1)
    token = SimpleNamespace(token=7, finish_reason=None)

    assert queue.put(context, False) == "queued"
    assert queue.put(token) == "queued"
    assert queue.put(None) == "queued"

    snapshot = telemetry.snapshot()
    assert [item[0] for item in target.items] == [context, token, None]
    assert target.items[0][1] == (False,)
    assert snapshot["active_requests"] == 0
    assert snapshot["requests_completed"] == 1
    assert snapshot["prompt_tokens_total"] == 4
    assert snapshot["cached_tokens_total"] == 1
    assert snapshot["completion_tokens_total"] == 1


def test_shared_uid_removal_terminates_the_waiting_response_queue():
    telemetry = RuntimeTelemetry(_Marker(), publish_interval=0)
    target = _Queue()
    queue = _TelemetryQueue(target, telemetry)
    context = SimpleNamespace(
        prompt=[1, 2, 3],
        prompt_cache_count=0,
        stop=lambda: None,
    )
    queue.put(context)
    telemetry.bind_pending_uid((91,))

    telemetry.cancel_uids([91])

    assert [item[0] for item in target.items] == [context, None]
    snapshot = telemetry.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["requests_cancelled"] == 1
    assert snapshot["last_request"]["status"] == "cancelled"


def test_telemetry_marker_failure_never_interrupts_inference():
    class BrokenMarker:
        def update(self, phase, **extra):
            raise OSError("disk unavailable")

    telemetry = RuntimeTelemetry(BrokenMarker(), publish_interval=0)

    request_id = telemetry.begin_request()
    telemetry.observe_context(request_id, prompt_tokens=2, cached_tokens=0)
    telemetry.observe_token(request_id)
    telemetry.finish_request(request_id)

    assert telemetry.snapshot()["requests_completed"] == 1


def test_telemetry_reports_coalescing_cache_affinity_and_stage_prediction():
    clock = _Clock()
    marker = _Marker()
    assignment = PipelineAssignment(
        "local",
        0,
        2,
        6,
        40,
        5,
        10,
        100,
        predicted_compute_seconds=0.2,
        predicted_send_seconds=0.01,
        predicted_stage_seconds=0.21,
    )
    telemetry = RuntimeTelemetry(
        marker,
        clock=clock,
        publish_interval=0,
        execution=execution_profile("balanced"),
        assignment=assignment,
        prompt_cache_ssd_enabled=True,
    )
    clock.value = 1.0
    telemetry.observe_batch_step(
        prompt_responses=2,
        generation_responses=4,
        elapsed_seconds=0.25,
    )
    telemetry.observe_cache_lookup(
        prompt_tokens=100,
        remaining_tokens=25,
        entries=3,
        nbytes=4096,
        memory_entries=2,
        memory_bytes=1024,
        ssd_entries=1,
        ssd_bytes=3072,
        ssd_max_bytes=20 * 1024**3,
        ssd_capacity_bytes=4096,
        ssd_evictions=2,
        ssd_capacity_drops=1,
        ssd_pending_bytes=512,
        ssd_pending_max_bytes=512 * 1024**2,
        ssd_write_failures=3,
        hit_tier="ssd",
    )

    snapshot = telemetry.snapshot()

    assert snapshot["pipeline"]["last_batch"]["coalesced_batch_size"] == 4
    assert snapshot["pipeline"]["microbatch_target"] == 4
    assert snapshot["pipeline"]["utilization"] == 0.25
    assert snapshot["aggregate_decode_tps"] == 16.0
    assert snapshot["cache"]["affinity"] == "deployment"
    assert snapshot["cache"]["hit_rate"] == 1.0
    assert snapshot["cache"]["tokens_reused"] == 75
    assert snapshot["cache"]["ssd_enabled"] is True
    assert snapshot["cache"]["memory"] == {
        "entries": 2,
        "bytes": 1024,
        "hits": 0,
    }
    assert snapshot["cache"]["ssd"] == {
        "entries": 1,
        "bytes": 3072,
        "hits": 1,
        "max_bytes": 20 * 1024**3,
        "capacity_bytes": 4096,
        "evictions": 2,
        "capacity_drops": 1,
        "pending_bytes": 512,
        "pending_max_bytes": 512 * 1024**2,
        "write_failures": 3,
    }
    assert snapshot["stage"]["predicted_stage_seconds"] == 0.21
    assert snapshot["stage"]["observed_step_seconds"] == 0.25


def test_batch_uid_cancellation_closes_request_on_every_rank():
    marker = _Marker()
    telemetry = RuntimeTelemetry(marker, publish_interval=0)
    request_id = telemetry.begin_request()
    telemetry.observe_context(request_id, prompt_tokens=8, cached_tokens=2)
    telemetry.mark_pending_uid(request_id)
    telemetry.bind_pending_uid((42,))

    telemetry.cancel_uids([42])

    snapshot = telemetry.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["requests_completed"] == 0
    assert snapshot["requests_cancelled"] == 1
    assert snapshot["last_request"]["status"] == "cancelled"


def test_server_patch_binds_batch_uid_and_restores_mlx_lm_classes(monkeypatch):
    import mlx_lm.server as mlx_server

    class FakeResponseGenerator:
        def __init__(self):
            self.model_provider = SimpleNamespace(model_key="model")
            self.prompt_cache = mlx_server.LRUPromptCache()

        def _share_request(self, request):
            return request

        def _tokenize(self, _tokenizer, _request, _args):
            prompt = [1, 2, 3, 4]
            return prompt, [prompt], ["assistant"], "normal"

    class FakeBatchGenerator:
        def __init__(self):
            self.removed = []
            self._prompt_batch = SimpleNamespace()

        def insert_segments(self, *args, **kwargs):
            return (73,)

        def _make_batch(self, _n):
            return SimpleNamespace(uids=[73])

        def next(self):
            return (
                [SimpleNamespace(uid=73, progress=(2, 3))],
                [],
            )

        def remove(self, uids):
            self.removed.extend(uids)
            return "removed"

    class FakePromptCache:
        def fetch_nearest_cache(self, _model, tokens):
            return "cache", tokens[2:]

        def insert_cache(self, *args, **kwargs):
            return None

        def __len__(self):
            return 1

        @property
        def nbytes(self):
            return 64

    monkeypatch.setattr(
        mlx_server,
        "ResponseGenerator",
        FakeResponseGenerator,
    )
    monkeypatch.setattr(mlx_server, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(mlx_server, "LRUPromptCache", FakePromptCache)
    marker = _Marker()
    target = _Queue()
    guard_calls = []
    guard = SimpleNamespace(
        check_collective=lambda *args, **kwargs: guard_calls.append((args, kwargs))
    )

    with install_server_telemetry(marker, prefill_guard=guard) as telemetry:
        generator = mlx_server.ResponseGenerator()
        queue, request, args = generator._share_request((target, "request", "args"))
        queue.put(
            SimpleNamespace(
                prompt=[1, 2, 3],
                prompt_cache_count=1,
            )
        )
        batch = mlx_server.BatchGenerator()
        assert batch.insert_segments(
            segments=[[[1, 2, 3, 4]]],
            all_tokens=[list(range(6000))],
        ) == (73,)
        batch._make_batch(1)
        assert batch._prompt_batch._omlx_total_prompt_lengths == {73: 6004}
        batch.next()
        progress = telemetry.snapshot()["last_request"]["prefill_progress"]
        assert progress["processed"] == 2
        assert progress["total"] == 3
        assert progress["active"] is True
        assert batch.remove([73]) == "removed"
        assert generator._tokenize(None, None, None)[0] == [1, 2, 3, 4]
        assert generator.prompt_cache.fetch_nearest_cache(
            "model", [1, 2, 3, 4]
        ) == (
            "cache",
            [3, 4],
        )
        assert guard_calls[0][0] == (4,)
        assert guard_calls[0][1]["cached_tokens"] == 2
        assert guard_calls[0][1]["mx_module"] is not None
        assert request == "request"
        assert args == "args"
        assert telemetry.snapshot()["requests_cancelled"] == 1

    assert mlx_server.ResponseGenerator is FakeResponseGenerator
    assert mlx_server.BatchGenerator is FakeBatchGenerator


def test_pipeline_cache_plan_requires_rank_and_physical_coherence(monkeypatch):
    import struct

    import mlx.core as mx
    import mlx_lm.server as mlx_server

    class Group:
        rank = staticmethod(lambda: 0)
        size = staticmethod(lambda: 2)

    class FakePromptCache:
        value = "rank-zero-cache"

        def fetch_nearest_cache(self, _model, tokens):
            return self.value, tokens[2:]

        def __len__(self):
            return 1

        nbytes = 64

    class ControlPlane:
        peer_plan = (2, 2, 0)

        def __init__(self):
            self.calls = []

        def broadcast_owned_bytes(self, payload, *, source_rank, expected_size):
            self.calls.append((source_rank, payload))
            assert expected_size == 24
            return (
                payload
                if source_rank == 0
                else struct.pack("!QQQ", *self.peer_plan)
            )

    monkeypatch.setattr(mx.distributed, "init", lambda: Group())
    monkeypatch.setattr(mlx_server, "LRUPromptCache", FakePromptCache)
    control = ControlPlane()

    with install_server_telemetry(
        _Marker(),
        prefill_guard=None,
        control_plane=control,
    ):
        cache = mlx_server.LRUPromptCache()
        tokens = [1, 2, 3, 4]

        # Identical rank plans preserve each stage's local cache object.
        assert cache.fetch_nearest_cache("model", tokens) == (
            "rank-zero-cache",
            [3, 4],
        )

        # A peer miss would post a four-row receive for this rank's two-row
        # activation. Both ranks instead discard their cache and prefill four.
        control.peer_plan = (0, 4, 0)
        assert cache.fetch_nearest_cache("model", tokens) == (None, tokens)

        # Equal advertised suffixes are also rejected when the copied cache is
        # physically still at four tokens after an unsupported arbitrary trim.
        FakePromptCache.value = [
            SimpleNamespace(caches=(SimpleNamespace(offset=4),))
        ]
        control.peer_plan = (2, 2, 1)
        assert cache.fetch_nearest_cache("model", tokens) == (None, tokens)

    assert [source for source, _payload in control.calls] == [0, 1] * 3


def test_server_patch_broadcasts_distributed_request_with_checked_gather(monkeypatch):
    import mlx.core as mx
    import mlx_lm.server as mlx_server

    class Group:
        @staticmethod
        def rank():
            return 0

        @staticmethod
        def size():
            return 2

    class FakeResponseGenerator:
        def __init__(self):
            self._is_distributed = True
            self._rank = 0

        def _share_object(self, _obj):
            raise AssertionError("distributed path must not call MLX-LM all-sum share")

        def _share_request(self, request):
            shareable = self._share_object(request[1:] if request else None)
            return None if shareable is None else (request[0], *shareable)

    gathered = []
    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mx.distributed, "init", lambda: Group())
    monkeypatch.setattr(
        mx.distributed,
        "all_gather",
        lambda value, group=None: gathered.append((value.dtype, value.tolist()))
        or mx.concatenate([value, mx.zeros_like(value)]),
    )
    monkeypatch.setattr(mx, "eval", lambda *_values: None)
    target = _Queue()

    with install_server_telemetry(_Marker(), prefill_guard=None):
        generator = mlx_server.ResponseGenerator()
        queue, request, args = generator._share_request(
            (target, "request", {"max_tokens": 7})
        )

    assert queue._queue is target
    assert request == "request"
    assert args == {"max_tokens": 7}
    assert len(gathered) == 2
    assert gathered[0][0] == mx.uint8
    assert len(gathered[0][1]) == 64
    assert gathered[1][0] == mx.uint8
    assert len(gathered[1][1]) > 0


def test_server_patch_receives_distributed_request_with_checked_gather(monkeypatch):
    import pickle

    import mlx.core as mx
    import mlx_lm.server as mlx_server

    payload = pickle.dumps(("request", {"max_tokens": 7}))
    import struct
    import zlib

    header = struct.pack(">III", 0x4F4D4C58, len(payload), zlib.crc32(payload)).ljust(
        64, b"\0"
    )
    calls = []

    class Group:
        @staticmethod
        def rank():
            return 1

        @staticmethod
        def size():
            return 2

    class FakeResponseGenerator:
        def __init__(self):
            self._is_distributed = True
            self._rank = 1

        def _share_object(self, _obj):
            raise AssertionError("distributed path must not call MLX-LM all-sum share")

        def _share_request(self, request):
            from queue import Queue

            shareable = self._share_object(request[1:] if request else None)
            return None if shareable is None else (Queue(), *shareable)

    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mx.distributed, "init", lambda: Group())
    monkeypatch.setattr(
        mx.distributed,
        "all_gather",
        lambda value, group=None: calls.append(value.tolist())
        or (
            mx.concatenate([mx.array(header, dtype=mx.uint8), value])
            if len(calls) == 1
            else mx.concatenate([mx.array(payload, dtype=mx.uint8), value])
        ),
    )

    with install_server_telemetry(_Marker(), prefill_guard=None):
        generator = mlx_server.ResponseGenerator()
        queue, request, args = generator._share_request(None)

    assert isinstance(queue, _TelemetryQueue)
    assert request == "request"
    assert args == {"max_tokens": 7}
    assert len(calls) == 2


def test_sequential_distributed_cancellation_exits_all_ranks_without_upstream_error(
    monkeypatch,
):
    """The pinned server raises NotImplementedError here without our patch."""

    import mlx_lm.server as mlx_server

    observed = []

    class FakeResponseGenerator:
        def __init__(self):
            self._is_distributed = True

        def _serve_single(self, _request):
            ctx = mlx_server.GenerationContext(
                has_tool_calling=False,
                has_thinking=False,
                tool_parser=lambda *_args: {},
                sequences={},
                prompt=[],
            )
            ctx.stop()
            if ctx._should_stop:
                if self._is_distributed:
                    raise NotImplementedError()
                observed.append("cancelled")

    class FakeBatchGenerator:
        pass

    original_context = mlx_server.GenerationContext
    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mlx_server, "BatchGenerator", FakeBatchGenerator)

    with install_server_telemetry(_Marker()):
        generator = mlx_server.ResponseGenerator()
        generator._serve_single(("queue", "request", "args"))
        assert generator._is_distributed is True
        assert mlx_server.GenerationContext is not original_context

    assert observed == ["cancelled"]
    assert mlx_server.GenerationContext is original_context


# ---------------------------------------------------------------------------
# The idle heartbeat.
#
# Every publish here used to be request-driven, so an idle rank's marker simply
# stopped ageing. The peer watchdog reads that timestamp and calls anything
# older than 45 s stale, so a healthy, loaded, serving cluster killed itself
# 60 s after the last token — and in conversational use that is between every
# turn, each one paying for a full model reload.
# ---------------------------------------------------------------------------


class _CountingMarker:
    def __init__(self) -> None:
        self.updates = []
        self._event = threading.Event()
        self._lock = threading.Lock()

    def update(self, phase, **extra):
        with self._lock:
            self.updates.append((phase, extra))
        self._event.set()

    def wait_for_update(self, timeout=5.0) -> bool:
        return self._event.wait(timeout)

    def count(self) -> int:
        with self._lock:
            return len(self.updates)


def test_an_idle_rank_still_refreshes_its_marker():
    """No requests, no tokens, nothing to report — and the marker still ages."""

    marker = _CountingMarker()
    telemetry = RuntimeTelemetry(
        marker, publish_interval=0, heartbeat_interval=0.01
    )

    telemetry.start_heartbeat()
    try:
        assert marker.wait_for_update(timeout=5.0), (
            "an idle rank published nothing; the peer watchdog will call it stale"
        )
    finally:
        telemetry.stop_heartbeat()

    assert marker.updates[0][0] == "ready"
    assert marker.count() >= 1


def test_stopping_the_heartbeat_ends_the_thread():
    marker = _CountingMarker()
    telemetry = RuntimeTelemetry(
        marker, publish_interval=0, heartbeat_interval=0.01
    )
    before = set(threading.enumerate())

    telemetry.start_heartbeat()
    telemetry.start_heartbeat()  # idempotent
    assert marker.wait_for_update(timeout=5.0)
    telemetry.stop_heartbeat()
    settled = marker.count()
    time.sleep(0.1)

    assert marker.count() == settled, "the heartbeat outlived stop_heartbeat"
    leaked = {
        thread
        for thread in threading.enumerate()
        if thread not in before and thread.is_alive()
        and thread.name == "omlx-cluster-telemetry-heartbeat"
    }
    assert not leaked


def test_the_heartbeat_advances_the_timestamp_a_peer_watchdog_reads(tmp_path):
    """The writer and the reader, not two hand-typed dicts.

    ``marker_age_seconds`` is what decides "stale"; a heartbeat that refreshed
    some other field would look identical in a mock and change nothing.
    """

    from omlx.cluster.inference_worker import RuntimeMarker
    from omlx.cluster.liveness import marker_age_seconds, read_marker

    marker = RuntimeMarker(
        state_dir=str(tmp_path),
        deployment_id="d",
        rank=0,
        world_size=2,
        model="org/model",
        backend="ring",
        plan_hash="a" * 64,
    )
    marker.update("ready", start_layer=0, end_layer=4)
    first = read_marker(marker.path)["updated_at"]

    telemetry = RuntimeTelemetry(marker, publish_interval=0, heartbeat_interval=0.01)
    telemetry.start_heartbeat()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if read_marker(marker.path)["updated_at"] != first:
                break
            time.sleep(0.01)
        else:  # pragma: no cover - only on a wedged heartbeat
            raise AssertionError("the marker's updated_at never advanced")
    finally:
        telemetry.stop_heartbeat()

    payload = read_marker(marker.path)
    assert payload["phase"] == "ready"
    assert marker_age_seconds(payload) < 45.0, "still inside the staleness window"


def test_serving_starts_the_heartbeat_without_the_caller_asking(monkeypatch):
    """The seam: install_server_telemetry owns the span a rank is alive for.

    A heartbeat the worker has to remember to start is a heartbeat a refactor
    will drop, and dropping it restores the 60-second self-kill silently.
    """

    import mlx_lm.server as mlx_server

    class FakeResponseGenerator:
        pass

    class FakeBatchGenerator:
        pass

    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mlx_server, "BatchGenerator", FakeBatchGenerator)
    marker = _CountingMarker()

    with install_server_telemetry(marker, heartbeat_interval=0.01) as telemetry:
        assert marker.wait_for_update(timeout=5.0), (
            "serving did not refresh the marker while idle"
        )
        assert telemetry._heartbeat_thread is not None

    settled = marker.count()
    time.sleep(0.1)
    assert marker.count() == settled, "the heartbeat outlived the serving block"


class _BatchGenerator:
    def __init__(self) -> None:
        self.removed = []

    def remove(self, uids):
        self.removed.append(list(uids))


class _GenerationContext:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _cancel_telemetry(
    tmp_path,
    clock=None,
    *,
    plan_hash="",
    epoch_floor=0,
):
    marker = _Marker()
    telemetry = RuntimeTelemetry(
        marker,
        clock=clock or _Clock(),
        publish_interval=0,
        cancel_path=tmp_path / "dep-1-cancel.json",
        cancel_deployment_id="dep-1",
        cancel_plan_hash=plan_hash,
        cancel_epoch_floor=epoch_floor,
    )
    return telemetry


def test_force_cancel_all_marks_context_for_the_shared_batch_loop(tmp_path):
    telemetry = _cancel_telemetry(tmp_path)
    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    request_id = telemetry.begin_request()
    telemetry.mark_pending_uid(request_id)
    context = _GenerationContext()
    telemetry.register_context(request_id, context)
    telemetry.bind_pending_uid((73,))

    cancelled = telemetry.force_cancel_all(reason="test")

    assert cancelled == 1
    assert context.stopped is True
    assert generator.removed == [], "telemetry must never mutate one rank directly"
    # The pinned server now broadcasts UID 73, then every rank removes it.
    generator.remove([73])
    telemetry.cancel_uids([73])
    assert generator.removed == [[73]]
    assert telemetry._requests == {}
    assert telemetry._requests_cancelled == 1


def test_targeted_cancel_stops_only_the_matching_transport_request(tmp_path):
    import json

    telemetry = _cancel_telemetry(tmp_path)
    first_queue = _TelemetryQueue(
        _Queue(), telemetry, transport_request_id="transport-first"
    )
    first_context = _GenerationContext()
    telemetry.mark_pending_uid(first_queue._request_id)
    telemetry.register_context(first_queue._request_id, first_context)
    telemetry.bind_pending_uid((71,))

    second_queue = _TelemetryQueue(
        _Queue(), telemetry, transport_request_id="transport-second"
    )
    second_context = _GenerationContext()
    telemetry.mark_pending_uid(second_queue._request_id)
    telemetry.register_context(second_queue._request_id, second_context)
    telemetry.bind_pending_uid((72,))

    (tmp_path / "dep-1-cancel.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "dep-1",
                "epoch": 42,
                "scope": "requests",
                "request_ids": ["transport-first"],
                "reason": "client disconnected",
            }
        ),
        encoding="utf-8",
    )

    assert telemetry.poll_cancel_requests(min_interval=0.0) == 1
    assert first_context.stopped is True
    assert second_context.stopped is False
    assert telemetry.merge_requested_cancel_uids([]) == [71]


def test_private_http_header_is_attached_before_request_sharing(monkeypatch):
    import mlx_lm.server as mlx_server

    observed = []

    def original_handle(handler, request, stop_words):
        observed.append((request, stop_words))

    monkeypatch.setattr(mlx_server.APIHandler, "handle_completion", original_handle)
    with install_server_telemetry(_Marker(), heartbeat_interval=0):
        handler = object.__new__(mlx_server.APIHandler)
        handler.headers = {"X-oMLX-Request-ID": "transport-header-owner"}
        request = SimpleNamespace()
        handler.handle_completion(request, ["stop"])

    assert observed == [(request, ["stop"])]
    assert request._omlx_transport_request_id == "transport-header-owner"


def test_targeted_cancel_arriving_before_context_registration_is_not_lost(tmp_path):
    telemetry = _cancel_telemetry(tmp_path)

    assert telemetry.force_cancel_request("transport-early") == 0
    queue = _TelemetryQueue(
        _Queue(), telemetry, transport_request_id="transport-early"
    )
    context = _GenerationContext()
    telemetry.register_context(queue._request_id, context)

    assert context.stopped is True


def test_long_prefill_exposes_cancel_vote_within_one_chunk(tmp_path):
    """A >25-chunk prompt cannot sit inside MLX-LM's 25-step budget slice."""

    import mlx_lm.server as mlx_server

    with install_server_telemetry(_Marker(), heartbeat_interval=0) as telemetry:
        request_id = telemetry.begin_request("transport-long-prefill")
        telemetry.observe_context(
            request_id,
            prompt_tokens=59_104,
            cached_tokens=0,
        )
        telemetry.mark_pending_uid(request_id)
        context = _GenerationContext()
        telemetry.register_context(request_id, context)
        telemetry.bind_pending_uid((73,))
        assert telemetry.force_cancel_request("transport-long-prefill") == 1

        budget = mlx_server.TimeBudget(budget=999, iterations=25)
        budget._is_distributed = True
        assert sum(1 for _ in budget) == 1
        assert telemetry.merge_requested_cancel_uids([]) == [73]

        telemetry.observe_prefill_progress(
            73,
            processed_tokens=59_104,
            total_tokens=59_104,
        )
        # The cap is lifecycle-based, not a permanent mutation of the budget:
        # decode/subsequent non-prefill slices retain MLX-LM's 25-step batch.
        assert sum(1 for _ in budget) == 25


def test_distributed_cancel_epoch_drains_and_rendezvous_before_removal(
    monkeypatch,
):
    """A cancel cannot filter rank-local caches before every peer is drained."""

    import mlx.core as mx
    import mlx_lm.server as mlx_server

    events = []

    class Group:
        rank = staticmethod(lambda: 0)
        size = staticmethod(lambda: 2)

    class ControlPlane:
        def broadcast_object(self, obj):
            events.append(("broadcast", obj))
            return obj

        def barrier(self):
            events.append(("barrier", None))

    class FakeResponseGenerator:
        def __init__(self):
            self._is_distributed = True
            self._rank = 0

        def _share_object(self, _obj):
            raise AssertionError("patched distributed sharing must own cancellation")

    class FakeBatchGenerator:
        def __init__(self):
            self._prompt_batch = SimpleNamespace()

        def remove(self, uids):
            events.append(("remove", list(uids)))
            return "removed"

    monkeypatch.setattr(mx.distributed, "init", lambda: Group())
    monkeypatch.setattr(mx, "synchronize", lambda *args: events.append(("drain", None)))
    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mlx_server, "BatchGenerator", FakeBatchGenerator)
    control = ControlPlane()

    with install_server_telemetry(
        _Marker(),
        heartbeat_interval=0,
        control_plane=control,
    ):
        import pytest

        generator = mlx_server.ResponseGenerator()
        batch = mlx_server.BatchGenerator()
        with pytest.raises(RuntimeError, match="not armed"):
            batch.remove([73])
        assert events == []
        assert generator._share_object([]) == []
        assert events == [("broadcast", [])]
        events.clear()
        assert generator._share_object([73]) == [73]
        assert batch.remove([73]) == "removed"

    assert [event[0] for event in events] == [
        "broadcast",
        "drain",
        "barrier",
        "remove",
    ]
    envelope = events[0][1]
    assert envelope["kind"] == "omlx.cancel_vote"
    assert envelope["schema_version"] == 1
    assert envelope["epoch"] > 0
    assert envelope["uids"] == [73]


def test_distributed_cancel_epoch_replay_is_rejected_before_removal(tmp_path):
    telemetry = _cancel_telemetry(tmp_path)
    vote = telemetry.make_cancel_vote([7])

    assert telemetry.accept_cancel_vote(vote) == (vote["epoch"], [7])
    import pytest

    with pytest.raises(RuntimeError, match="did not advance"):
        telemetry.accept_cancel_vote(vote)


def test_force_cancel_all_without_a_generation_context_is_a_noop(tmp_path):
    telemetry = _cancel_telemetry(tmp_path)

    assert telemetry.force_cancel_all(reason="test") == 0

    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    assert telemetry.force_cancel_all(reason="test") == 0
    assert generator.removed == []


def test_force_cancel_all_survives_a_failing_generation_context(tmp_path):
    telemetry = _cancel_telemetry(tmp_path)

    class BrokenContext:
        def stop(self):
            raise RuntimeError("wedged")

    request_id = telemetry.begin_request()
    telemetry.mark_pending_uid(request_id)
    telemetry.register_context(request_id, BrokenContext())
    telemetry.bind_pending_uid((5,))

    assert telemetry.force_cancel_all(reason="test") == 0
    # The request stays tracked; the coordinator's process teardown is the
    # fallback for this failure mode.
    assert request_id in telemetry._requests


def test_cancel_file_is_consumed_once_and_acked(tmp_path):
    import json

    telemetry = _cancel_telemetry(tmp_path)
    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    request_id = telemetry.begin_request()
    telemetry.mark_pending_uid(request_id)
    context = _GenerationContext()
    telemetry.register_context(request_id, context)
    telemetry.bind_pending_uid((9,))

    cancel_path = tmp_path / "dep-1-cancel.json"
    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "dep-1",
                "epoch": 42,
                "scope": "all",
                "reason": "memory pressure",
            }
        ),
        encoding="utf-8",
    )

    assert telemetry.poll_cancel_requests(min_interval=0.0) == 1
    assert context.stopped is True
    assert generator.removed == []
    ack = json.loads(
        (tmp_path / "dep-1-cancel-ack.json").read_text(encoding="utf-8")
    )
    assert ack["epoch"] == 42
    assert ack["cancelled"] == 1

    # Same epoch is not consumed twice.
    assert telemetry.poll_cancel_requests(min_interval=0.0) == 0
    assert generator.removed == []


def test_cancel_file_from_a_foreign_deployment_is_ignored(tmp_path):
    import json

    telemetry = _cancel_telemetry(tmp_path)
    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    (tmp_path / "dep-1-cancel.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "somebody-else",
                "epoch": 7,
                "scope": "all",
            }
        ),
        encoding="utf-8",
    )

    assert telemetry.poll_cancel_requests(min_interval=0.0) == 0
    assert generator.removed == []


def test_cancel_file_from_an_old_plan_or_worker_lifetime_is_ignored(tmp_path):
    import json

    telemetry = _cancel_telemetry(
        tmp_path,
        plan_hash="current-plan",
        epoch_floor=1000,
    )
    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    request_id = telemetry.begin_request()
    telemetry.mark_pending_uid(request_id)
    context = _GenerationContext()
    telemetry.register_context(request_id, context)
    telemetry.bind_pending_uid((11,))
    cancel_path = tmp_path / "dep-1-cancel.json"

    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "dep-1",
                "plan_hash": "old-plan",
                "epoch": 2000,
                "scope": "all",
            }
        ),
        encoding="utf-8",
    )
    assert telemetry.poll_cancel_requests(min_interval=0.0) == 0

    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "dep-1",
                "plan_hash": "current-plan",
                "epoch": 999,
                "scope": "all",
            }
        ),
        encoding="utf-8",
    )
    assert telemetry.poll_cancel_requests(min_interval=0.0) == 0
    assert generator.removed == []

    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "dep-1",
                "plan_hash": "current-plan",
                "epoch": 1001,
                "scope": "all",
            }
        ),
        encoding="utf-8",
    )
    assert telemetry.poll_cancel_requests(min_interval=0.0) == 1
    assert context.stopped is True
    assert generator.removed == []


def test_existing_matching_cancel_is_a_startup_watermark_not_durable_state(tmp_path):
    import json

    cancel_path = tmp_path / "dep-1-cancel.json"
    cancel_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "deployment_id": "dep-1",
                "plan_hash": "same-plan",
                "epoch": 4242,
                "scope": "all",
            }
        ),
        encoding="utf-8",
    )
    telemetry = _cancel_telemetry(tmp_path, plan_hash="same-plan")
    generator = _BatchGenerator()
    telemetry.register_batch_generator(generator)
    request_id = telemetry.begin_request()
    telemetry.mark_pending_uid(request_id)
    context = _GenerationContext()
    telemetry.register_context(request_id, context)
    telemetry.bind_pending_uid((12,))

    assert telemetry.poll_cancel_requests(min_interval=0.0) == 0
    assert generator.removed == []

    payload = json.loads(cancel_path.read_text(encoding="utf-8"))
    payload["epoch"] = 4243
    cancel_path.write_text(json.dumps(payload), encoding="utf-8")
    assert telemetry.poll_cancel_requests(min_interval=0.0) == 1
    assert context.stopped is True
    assert generator.removed == []
