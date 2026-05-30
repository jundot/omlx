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
            "final_ceiling": 2048,
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
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=FakePrefillTracker()),
        patch("time.monotonic", return_value=110.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    assert data["total_active_requests"] == 2
    assert model["active_requests"] == 2
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


def test_active_models_does_not_count_waiting_collectors_as_active():
    waiting_request = SimpleNamespace(
        request_id="wait-1",
        arrival_time=105.0,
        num_prompt_tokens=30,
    )
    scheduler = SimpleNamespace(
        snapshot_for_admin=lambda: {
            "running_by_id": {},
            "waiting": [waiting_request],
        },
    )

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=FakePool(scheduler)),
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.monotonic", return_value=110.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    assert data["total_active_requests"] == 0
    assert data["total_waiting_requests"] == 1
    assert model["active_requests"] == 0
    assert model["waiting_requests"] == 1
    assert model["generating"] == []


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
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
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
            "final_ceiling": 2048,
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
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
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


# ── DFlash boolean activity fallback (#1477) ──────────────────────────────


class FakeDFlashEngine:
    """DFlash exposes a boolean in-flight signal but neither an
    AsyncEngineCore (_engine) nor get_activity_snapshot."""

    def __init__(self, active):
        self._active = active

    def has_active_requests(self):
        return self._active


class FakeDFlashPool:
    def __init__(self, active):
        self._entries = {
            "dflash-model": SimpleNamespace(engine=FakeDFlashEngine(active))
        }

    def get_status(self):
        return {
            "current_model_memory": 1024,
            "final_ceiling": 2048,
            "models": [
                {
                    "id": "dflash-model",
                    "loaded": True,
                    "is_loading": False,
                    "estimated_size": 1024,
                    "pinned": False,
                }
            ],
        }


def _build_dflash_data(active):
    class EmptyPrefillTracker:
        def get_model_progress(self, model_id):
            return []

    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=FakeDFlashPool(active)),
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.monotonic", return_value=110.0),
    ):
        return admin_routes._build_active_models_data()


def test_dflash_reports_active_via_boolean_signal():
    data = _build_dflash_data(active=True)
    model = data["models"][0]
    assert model["active_requests"] == 1
    assert data["total_active_requests"] == 1


def test_dflash_reports_idle_when_no_active_request():
    data = _build_dflash_data(active=False)
    model = data["models"][0]
    assert model["active_requests"] == 0
    assert data["total_active_requests"] == 0


# ── idle / TTL countdown (#1307) ──────────────────────────────────────────


def _make_global_settings(idle_timeout_seconds=None):
    """Build a fake global_settings SimpleNamespace with optional idle_timeout."""
    idle_timeout = SimpleNamespace(idle_timeout_seconds=idle_timeout_seconds)
    return SimpleNamespace(idle_timeout=idle_timeout)


def _make_settings_manager(ttl_seconds=None):
    """Build a fake settings_manager whose get_settings() returns ttl_seconds."""
    settings = SimpleNamespace(ttl_seconds=ttl_seconds)
    return SimpleNamespace(get_settings=lambda _mid: settings)


class FakeIdlePool:
    """Pool with a single loaded idle model that has last_access set."""

    def __init__(self, last_access=100.0):
        self._entries = {
            "model-a": SimpleNamespace(engine=object()),  # engine is not None → loaded
        }
        self._last_access = last_access

    def get_status(self):
        return {
            "current_model_memory": 1024,
            "final_ceiling": 2048,
            "models": [
                {
                    "id": "model-a",
                    "loaded": True,
                    "is_loading": False,
                    "loading_started_at": None,
                    "estimated_size": 1024,
                    "pinned": False,
                    "last_access": self._last_access,
                }
            ],
        }


class EmptyPrefillTracker:
    def get_model_progress(self, model_id):
        return []


def test_idle_seconds_computed_from_last_access():
    """idle_seconds = time.time() - last_access for a loaded model."""
    with (
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_engine_pool", return_value=FakeIdlePool(last_access=100.0)),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.time", return_value=115.0),
        patch("time.monotonic", return_value=115.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    assert model["idle_seconds"] == 15.0
    assert model["ttl_remaining_seconds"] is None  # no TTL configured


def test_idle_seconds_none_when_no_last_access():
    """idle_seconds is None when last_access is missing or 0."""
    with (
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_engine_pool", return_value=FakeIdlePool(last_access=0)),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.time", return_value=115.0),
        patch("time.monotonic", return_value=115.0),
    ):
        data = admin_routes._build_active_models_data()

    assert data["models"][0]["idle_seconds"] is None


def test_ttl_remaining_from_per_model_setting():
    """TTL countdown uses per-model ttl_seconds when available."""
    with (
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_engine_pool", return_value=FakeIdlePool(last_access=100.0)),
        patch.object(
            admin_routes,
            "_get_settings_manager",
            return_value=_make_settings_manager(ttl_seconds=30),
        ),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.time", return_value=115.0),
        patch("time.monotonic", return_value=115.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    assert model["idle_seconds"] == 15.0
    # 30s TTL - 15s idle = 15s remaining
    assert model["ttl_remaining_seconds"] == 15.0


def test_ttl_remaining_falls_back_to_global_idle_timeout():
    """When per-model ttl_seconds is None, global idle_timeout is used."""
    with (
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_engine_pool", return_value=FakeIdlePool(last_access=100.0)),
        patch.object(
            admin_routes,
            "_get_settings_manager",
            return_value=_make_settings_manager(ttl_seconds=None),
        ),
        patch.object(
            admin_routes,
            "_get_global_settings",
            return_value=_make_global_settings(idle_timeout_seconds=60),
        ),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.time", return_value=115.0),
        patch("time.monotonic", return_value=115.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    assert model["idle_seconds"] == 15.0
    # 60s global TTL - 15s idle = 45s remaining
    assert model["ttl_remaining_seconds"] == 45.0


def test_ttl_remaining_per_model_takes_priority():
    """Per-model ttl_seconds overrides global idle_timeout."""
    with (
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_engine_pool", return_value=FakeIdlePool(last_access=100.0)),
        patch.object(
            admin_routes,
            "_get_settings_manager",
            return_value=_make_settings_manager(ttl_seconds=20),
        ),
        patch.object(
            admin_routes,
            "_get_global_settings",
            return_value=_make_global_settings(idle_timeout_seconds=300),
        ),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.time", return_value=115.0),
        patch("time.monotonic", return_value=115.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    # per-model 20s wins over global 300s
    assert model["ttl_remaining_seconds"] == 5.0  # 20 - 15


def test_ttl_remaining_clamped_to_zero():
    """ttl_remaining_seconds floors at 0 when TTL has expired."""
    with (
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_engine_pool", return_value=FakeIdlePool(last_access=100.0)),
        patch.object(
            admin_routes,
            "_get_settings_manager",
            return_value=_make_settings_manager(ttl_seconds=10),
        ),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.time", return_value=130.0),  # 30s idle, 10s TTL → expired
        patch("time.monotonic", return_value=130.0),
    ):
        data = admin_routes._build_active_models_data()

    assert data["models"][0]["ttl_remaining_seconds"] == 0.0


def test_idle_and_ttl_not_computed_for_loading_model():
    """Loading models have no idle/ttl fields set (both None)."""
    class LoadingPool:
        def __init__(self):
            self._entries = {"model-a": SimpleNamespace(engine=None)}  # not loaded yet

        def get_status(self):
            return {
                "current_model_memory": 0,
                "final_ceiling": 0,
                "models": [
                    {
                        "id": "model-a",
                        "loaded": False,
                        "is_loading": True,
                        "loading_started_at": 100.0,
                        "estimated_size": 1024,
                        "pinned": False,
                        "last_access": 0,
                    }
                ],
            }

    with (
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_engine_pool", return_value=LoadingPool()),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch("omlx.prefill_progress.get_prefill_tracker", return_value=EmptyPrefillTracker()),
        patch("time.time", return_value=115.0),
        patch("time.monotonic", return_value=115.0),
    ):
        data = admin_routes._build_active_models_data()

    model = data["models"][0]
    assert model["is_loading"] is True
    assert model["idle_seconds"] is None
    assert model["ttl_remaining_seconds"] is None
