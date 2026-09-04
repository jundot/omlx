import json
from pathlib import Path
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


# ── DFlash engine visibility (#2396) ──────────────────────────────────────


class FakeDFlashPool:
    """Pool whose engine has no AsyncEngineCore, mirroring DFlashEngine.

    ``scheduler`` stands in for DFlashEngine's property that surfaces the
    fallback engine's scheduler (None while in primary mode).
    """

    def __init__(self, engine):
        self._entries = {"dflash-model": SimpleNamespace(engine=engine)}

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


def _build_with_pool(pool):
    with (
        patch.object(admin_routes, "_get_engine_pool", return_value=pool),
        patch("omlx.admin.routes._get_server_state", return_value=None),
        patch.object(admin_routes, "_get_settings_manager", return_value=None),
        patch.object(admin_routes, "_get_global_settings", return_value=None),
        patch(
            "omlx.prefill_progress.get_prefill_tracker",
            return_value=EmptyPrefillTracker(),
        ),
        patch("time.monotonic", return_value=110.0),
    ):
        return admin_routes._build_active_models_data()


def test_active_models_dflash_primary_counts_activity():
    """Primary mode: no scheduler anywhere, activity snapshot drives the count."""

    class Engine:
        scheduler = None

        def get_activity_snapshot(self):
            return {
                "active_requests": 1,
                "activities": [
                    {
                        "request_id": "req-1",
                        "kind": "generate",
                        "detail": "generating",
                        "elapsed_seconds": 3.0,
                        "token_count": 42,
                    }
                ],
            }

    data = _build_with_pool(FakeDFlashPool(Engine()))

    model = data["models"][0]
    assert data["total_active_requests"] == 1
    assert model["active_requests"] == 1
    assert model["activities"][0]["detail"] == "generating"


def test_active_models_resolves_scheduler_property_without_async_core():
    """Fallback mode: the engine exposes a scheduler property instead of an
    AsyncEngineCore, and the snapshot path must still be taken."""
    running_request = SimpleNamespace(
        request_id="gen-1",
        generation_started_at=100.0,
        last_activity_at=109.5,
        num_output_tokens=20,
        num_prompt_tokens=12,
        max_tokens=64,
    )

    class Engine:
        scheduler = SimpleNamespace(
            snapshot_for_admin=lambda: {
                "running_by_id": {"gen-1": running_request},
                "waiting": [],
            }
        )

        def get_activity_snapshot(self):
            return {"active_requests": 0, "activities": []}

    data = _build_with_pool(FakeDFlashPool(Engine()))

    model = data["models"][0]
    assert data["total_active_requests"] == 1
    assert model["active_requests"] == 1
    assert model["generating"][0]["request_id"] == "gen-1"
    assert model["activities"] == []


def test_active_models_surfaces_dflash_guardrail_stats():
    """DFlash warning and exact speculation counters reach the dashboard API."""

    speculation = {
        "last": {
            "generation_tokens": 768,
            "cycles": 377,
            "acceptance_ratio": 391 / 768,
            "accepted_draft_tokens": 391,
            "tokens_per_cycle": 768 / 377,
            "accepted_draft_tokens_per_cycle": 391 / 377,
            "fallback_ar": False,
            "fallback_reason": None,
        },
        "totals": {
            "requests": 1,
            "speculative_requests": 1,
            "fallback_requests": 0,
            "generation_tokens": 768,
            "accepted_draft_tokens": 391,
            "cycles": 377,
            "acceptance_ratio": 391 / 768,
            "tokens_per_cycle": 768 / 377,
            "accepted_draft_tokens_per_cycle": 391 / 377,
        },
    }

    class Engine:
        scheduler = None
        pairing_warning = "Possible DFlash precision mismatch"

        def get_activity_snapshot(self):
            return {"active_requests": 0, "activities": []}

        def get_speculation_stats(self):
            return speculation

    data = _build_with_pool(FakeDFlashPool(Engine()))

    assert data["models"][0]["dflash"] == {
        "speculation": speculation,
        "pairing_warning": "Possible DFlash precision mismatch",
    }


def test_dflash_dashboard_localizes_metrics_and_shows_session_fallbacks():
    root = Path(__file__).resolve().parents[1]
    template = (
        root / "omlx/admin/templates/dashboard/_status.html"
    ).read_text(encoding="utf-8")
    dashboard_js = (root / "omlx/admin/static/js/dashboard.js").read_text(
        encoding="utf-8"
    )
    i18n_dir = root / "omlx/admin/i18n"
    keys = {
        "status.active_models.dflash_fallback_ar",
        "status.active_models.dflash_draft_share",
        "status.active_models.dflash_accepted_draft_per_cycle",
        "status.active_models.dflash_output_per_cycle",
        "status.active_models.dflash_draft_tokens_last_request",
        "status.active_models.dflash_session",
        "status.active_models.dflash_speculative_requests",
        "status.active_models.dflash_fallback_requests",
    }

    assert "m.dflash.speculation.totals.requests > 1" in template
    assert "formatDFlashSessionStats(m.dflash.speculation.totals)" in template
    assert "totals.fallback_requests > 0" in dashboard_js
    for key in keys:
        assert key in template + dashboard_js
    for locale_path in i18n_dir.glob("*.json"):
        locale = json.loads(locale_path.read_text(encoding="utf-8"))
        assert not keys - locale.keys(), f"{locale_path.name} is missing DFlash keys"
