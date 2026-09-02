# SPDX-License-Identifier: Apache-2.0
"""Tests for rank-local, end-to-end distributed inference telemetry."""

import threading
import time
from types import SimpleNamespace

from omlx.cluster.performance import execution_profile
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.telemetry import (
    RuntimeTelemetry,
    _TelemetryQueue,
    install_server_telemetry,
)


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
    )

    snapshot = telemetry.snapshot()

    assert snapshot["pipeline"]["last_batch"]["coalesced_batch_size"] == 4
    assert snapshot["pipeline"]["microbatch_target"] == 4
    assert snapshot["pipeline"]["utilization"] == 0.25
    assert snapshot["cache"]["affinity"] == "deployment"
    assert snapshot["cache"]["hit_rate"] == 1.0
    assert snapshot["cache"]["tokens_reused"] == 75
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

        def insert_segments(self, *args, **kwargs):
            return (73,)

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
        assert batch.insert_segments() == (73,)
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


def test_vision_sharing_precedes_telemetry_collectives_and_guards_actual_suffix(
    monkeypatch,
):
    from dataclasses import dataclass

    import mlx_lm.server as mlx_server

    from omlx.cluster import deepseek_v4_vision_runtime
    from omlx.cluster.deepseek_v4_vision_runtime import (
        install_deepseek_v4_vision_runtime,
    )

    events = []
    prepared = SimpleNamespace(
        patches=b"pixels",
        n_vit_h=2,
        n_vit_w=2,
        types=(0, 1, 2),
    )
    monkeypatch.setattr(
        deepseek_v4_vision_runtime,
        "prepare_token_ids",
        lambda *_args, **_kwargs: (list(range(10)), (prepared,)),
    )

    class Model:
        def set_vision_inputs(self, images):
            self.images = images

    class FakeResponseGenerator:
        def __init__(self):
            self.model_provider = provider
            self.prompt_cache = mlx_server.LRUPromptCache()

        def _share_request(self, request):
            events.append("share_request")
            return request

        def _tokenize(self, _tokenizer, _request, _args):
            events.append("rank_zero_tokenize")
            return [1, 99, 2], [[1, 99, 2]], ["assistant"], "normal"

    class FakeBatchGenerator:
        pass

    class FakePromptCache:
        def fetch_nearest_cache(self, _model, tokens):
            events.append("cache_lookup")
            return "cache", tokens[3:]

        def __len__(self):
            return 1

        @property
        def nbytes(self):
            return 64

    class Guard:
        def check_collective(self, *args, **kwargs):
            events.append("guard_collective")
            self.call = args, kwargs

    guard = Guard()
    monkeypatch.setattr(mlx_server, "ResponseGenerator", FakeResponseGenerator)
    monkeypatch.setattr(mlx_server, "BatchGenerator", FakeBatchGenerator)
    monkeypatch.setattr(mlx_server, "LRUPromptCache", FakePromptCache)

    @dataclass
    class Request:
        messages: list

    request = Request(
        [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "image"}}],
            }
        ]
    )
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda _token: 99,
        unk_token_id=-1,
    )
    provider = SimpleNamespace(
        model=Model(),
        model_key=("model", None, None),
        tokenizer=tokenizer,
        is_batchable=True,
    )
    args = SimpleNamespace(prefill_step_size=4)

    with (
        install_deepseek_v4_vision_runtime(mlx_server, provider, config={}, rank=0),
        install_server_telemetry(_Marker(), prefill_guard=guard, prefill_step_size=4),
    ):
        generator = mlx_server.ResponseGenerator()
        _queue, shared_request, shared_args = generator._share_request(
            (object(), request, args)
        )
        assert generator._tokenize(tokenizer, shared_request, shared_args)[0] == list(
            range(10)
        )
        assert len(provider.model_key) == 5

    assert events == [
        "rank_zero_tokenize",
        "share_request",
        "cache_lookup",
        "guard_collective",
    ]
    assert guard.call[0] == (10,)
    assert guard.call[1]["cached_tokens"] == 3
    assert guard.call[1]["prefill_step_size"] == 7
    assert args.prefill_step_size == 4
    assert provider.model_key == ("model", None, None)
    assert provider.model.images is None


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
