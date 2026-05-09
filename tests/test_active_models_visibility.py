from types import SimpleNamespace
from unittest.mock import patch

from omlx.admin import routes as admin_routes


class FakePool:
    def __init__(self, scheduler, loading_started_at=None):
        core = SimpleNamespace(
            _output_collectors={"gen-1": object(), "prefill-1": object(), "wait-1": object()},
            scheduler=scheduler,
        )
        engine = SimpleNamespace(_engine=SimpleNamespace(engine=core))
        self._entries = {"model-a": SimpleNamespace(engine=engine)}
        self.loading_started_at = loading_started_at

    def get_status(self):
        return {
            "current_model_memory": 1024,
            "max_model_memory": 2048,
            "models": [
                {
                    "id": "model-a",
                    "loaded": True,
                    "is_loading": self.loading_started_at is not None,
                    "loading_started_at": self.loading_started_at,
                    "estimated_size": 1024,
                    "pinned": False,
                }
            ],
        }


class FakePrefillTracker:
    def get_model_progress(self, model_id):
        assert model_id == "model-a"
        return [{"request_id": "prefill-1", "processed": 10, "total": 20}]


def test_active_models_generation_includes_activity_and_waiting_rows():
    running_request = SimpleNamespace(
        request_id="gen-1",
        generation_started_at=100.0,
        last_activity_at=109.5,
        num_output_tokens=20,
        num_prompt_tokens=12,
        max_tokens=64,
    )
    waiting_request = SimpleNamespace(
        request_id="wait-1",
        arrival_time=105.0,
        num_prompt_tokens=30,
    )
    scheduler = SimpleNamespace(
        snapshot_for_admin=lambda: {
            "running_by_id": {"gen-1": running_request},
            "waiting": [waiting_request],
        },
    )

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=FakePool(scheduler)),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=FakePrefillTracker()),
        patch("time.monotonic", return_value=110.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    assert data["total_active_requests"] == 3
    assert model["waiting_requests"] == 1
    assert model["prefilling"] == [
        {"request_id": "prefill-1", "processed": 10, "total": 20}
    ]
    assert model["waiting"] == [
        {
            "request_id": "wait-1",
            "queue_position": 1,
            "elapsed_seconds": 5.0,
            "prompt_tokens": 30,
        }
    ]
    assert model["generating"] == [
        {
            "request_id": "gen-1",
            "elapsed_seconds": 10.0,
            "generated_tokens": 20,
            "tokens_per_second": 2.0,
            "last_activity_age_seconds": 0.5,
            "prompt_tokens": 12,
            "max_tokens": 64,
        }
    ]


def test_active_models_loading_includes_elapsed_and_percent_estimate():
    scheduler = SimpleNamespace(
        snapshot_for_admin=lambda: {"running_by_id": {}, "waiting": []},
    )

    with (
        patch.object(
            admin_routes,
            "_get_engine_pool",
            return_value=FakePool(scheduler, loading_started_at=102.0),
        ),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=FakePrefillTracker()),
        patch("time.monotonic", return_value=110.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    assert model["is_loading"] is True
    assert model["loading_elapsed_seconds"] == 8.0
    assert model["loading_estimated_seconds"] is None
    assert model["loading_remaining_seconds_estimate"] is None


class FakeNonStreamingEngine:
    def get_activity_snapshot(self):
        return {
            "active_requests": 1,
            "activities": [
                {
                    "request_id": "embed-1",
                    "kind": "embedding",
                    "detail": "Embedding",
                    "elapsed_seconds": 12.0,
                    "input_count": 200,
                    "token_count": 120200,
                }
            ],
        }


class FakeNonStreamingPool:
    def __init__(self):
        self._entries = {"embed-model": SimpleNamespace(engine=FakeNonStreamingEngine())}

    def get_status(self):
        return {
            "current_model_memory": 1024,
            "max_model_memory": 2048,
            "models": [
                {
                    "id": "embed-model",
                    "loaded": True,
                    "is_loading": False,
                    "estimated_size": 1024,
                    "pinned": False,
                }
            ],
        }


def test_active_models_includes_non_streaming_activity_rows():
    class EmptyPrefillTracker:
        def get_model_progress(self, model_id):
            return []

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=FakeNonStreamingPool()),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.monotonic", return_value=110.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    assert data["total_active_requests"] == 1
    assert model["active_requests"] == 1
    assert model["activities"] == [
        {
            "request_id": "embed-1",
            "kind": "embedding",
            "detail": "Embedding",
            "elapsed_seconds": 12.0,
            "input_count": 200,
            "token_count": 120200,
        }
    ]
