# SPDX-License-Identifier: Apache-2.0
"""Tests for GET /api/status endpoint."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from omlx.server import ServerState, app


@pytest.fixture
def client():
    return TestClient(app)


class TestStatusEndpoint:
    """Tests for /api/status lightweight status endpoint."""

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        """Set up a clean server state for each test."""
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def test_returns_ok_when_pool_is_none(self, client):
        """When engine pool is not initialized, return basic status."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_discovered"] == 0
        assert data["models_loaded"] == 0
        assert data["models_loading"] == 0
        assert data["loaded_models"] == []
        assert data["active_requests"] == 0
        assert data["waiting_requests"] == 0
        assert "version" in data
        assert "uptime_seconds" in data

    def test_returns_pool_info(self, client):
        """When engine pool exists, return model and memory stats."""
        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "get_loaded_model_ids",
            "current_model_memory", "_entries",
        ])
        pool.model_count = 5
        pool.loaded_model_count = 2
        pool.get_loaded_model_ids.return_value = ["model-a", "model-b"]
        pool.current_model_memory = 16 * 1024**3
        enforcer = MagicMock(spec=["get_final_ceiling"])
        enforcer.get_final_ceiling.return_value = 32 * 1024**3
        self._state.process_memory_enforcer = enforcer

        entry_a = MagicMock(spec=["is_loading", "engine"])
        entry_a.is_loading = False
        entry_a.engine = None
        entry_b = MagicMock(spec=["is_loading", "engine"])
        entry_b.is_loading = True
        entry_b.engine = None
        pool._entries = {"model-a": entry_a, "model-b": entry_b}

        self._state.engine_pool = pool

        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models_discovered"] == 5
        assert data["models_loaded"] == 2
        assert data["models_loading"] == 1
        assert data["loaded_models"] == ["model-a", "model-b"]
        assert data["model_memory_used"] == 16 * 1024**3
        assert data["model_memory_max"] == 32 * 1024**3
        assert "GB" in data["model_memory_used_formatted"]
        assert "GB" in data["model_memory_max_formatted"]

    def test_status_ignores_memory_ceiling_error(self, client):
        """Memory telemetry failures should not break status polling."""
        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "get_loaded_model_ids",
            "current_model_memory", "_entries",
        ])
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.get_loaded_model_ids.return_value = ["model-a"]
        pool.current_model_memory = 16 * 1024**3
        pool._entries = {}
        enforcer = MagicMock(spec=["get_final_ceiling"])
        enforcer.get_final_ceiling.side_effect = RuntimeError(
            "host_statistics64 failed"
        )
        self._state.engine_pool = pool
        self._state.process_memory_enforcer = enforcer

        resp = client.get("/api/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_memory_max"] is None
        assert data["model_memory_max_formatted"] == "unlimited"

    def test_health_ignores_memory_ceiling_error(self, client):
        """Health should stay healthy when optional memory telemetry fails."""
        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "current_model_memory",
        ])
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.current_model_memory = 16 * 1024**3
        enforcer = MagicMock(spec=["get_final_ceiling"])
        enforcer.get_final_ceiling.side_effect = RuntimeError(
            "host_statistics64 failed"
        )
        self._state.engine_pool = pool
        self._state.process_memory_enforcer = enforcer

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["engine_pool"]["final_ceiling"] == 0

    def test_aggregates_active_waiting_requests(self, client):
        """Active/waiting request counts are summed across loaded engines."""
        # Build a mock engine with scheduler
        scheduler = MagicMock(spec=["waiting"])
        scheduler.waiting = [1, 2]  # 2 waiting

        core = MagicMock(spec=["_output_collectors", "scheduler"])
        core._output_collectors = {"req-1": None, "req-2": None, "req-3": None}
        core.scheduler = scheduler

        async_core = MagicMock(spec=["engine"])
        async_core.engine = core

        engine = MagicMock(spec=["_engine"])
        engine._engine = async_core

        entry = MagicMock(spec=["is_loading", "engine"])
        entry.is_loading = False
        entry.engine = engine

        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "get_loaded_model_ids",
            "current_model_memory", "_entries",
        ])
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.get_loaded_model_ids.return_value = ["model-a"]
        pool.current_model_memory = 0
        pool._entries = {"model-a": entry}

        self._state.engine_pool = pool

        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_requests"] == 3
        assert data["waiting_requests"] == 2

    def test_requires_auth_when_api_key_set(self, client):
        """The endpoint should require an API key when one is configured."""
        self._state.api_key = "test-secret-key"
        resp = client.get("/api/status")
        assert resp.status_code == 401

        resp = client.get(
            "/api/status",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 200

    def test_serving_metrics_included(self, client):
        """Check that serving metrics from ServerMetrics are present."""
        resp = client.get("/api/status")
        data = resp.json()
        expected_keys = [
            "total_requests", "total_prompt_tokens", "total_completion_tokens",
            "total_cached_tokens", "cache_efficiency",
            "avg_prefill_tps", "avg_generation_tps",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_unlimited_memory_max(self, client):
        """When no enforcer is present, formatted shows 'unlimited'."""
        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "get_loaded_model_ids",
            "current_model_memory", "_entries",
        ])
        pool.model_count = 0
        pool.loaded_model_count = 0
        pool.get_loaded_model_ids.return_value = []
        pool.current_model_memory = 0
        pool._entries = {}
        self._state.process_memory_enforcer = None

        self._state.engine_pool = pool

        resp = client.get("/api/status")
        data = resp.json()
        assert data["model_memory_max"] is None
        assert data["model_memory_max_formatted"] == "unlimited"


class TestStatusCustomKernels:
    """/api/status reports native custom kernel availability.

    Diagnosability for silently-degraded source installs: without the
    native extensions the affected model families fall back to much slower
    generic paths (issue #2137), and this block makes that detectable by
    external polling instead of only a log line.
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            yield

    def test_custom_kernels_block_lists_every_package(self, client):
        from omlx.custom_kernels import NATIVE_KERNEL_PACKAGES

        resp = client.get("/api/status")
        assert resp.status_code == 200
        kernels = resp.json()["custom_kernels"]
        assert set(kernels) == set(NATIVE_KERNEL_PACKAGES)
        for report in kernels.values():
            assert set(report) == {"available", "import_error"}
            assert isinstance(report["available"], bool)
            assert report["import_error"] is None or isinstance(
                report["import_error"], str
            )

    def test_available_packages_report_no_import_error(self, client):
        resp = client.get("/api/status")
        for name, report in resp.json()["custom_kernels"].items():
            if report["available"]:
                assert report["import_error"] is None, name
            else:
                assert report["import_error"], name

    def test_native_kernel_status_never_raises_on_broken_package(self):
        from omlx import custom_kernels

        real_import = custom_kernels.importlib.import_module

        def broken_import(name, *args, **kwargs):
            if name.endswith(".fast"):
                raise RuntimeError("simulated native import explosion")
            return real_import(name, *args, **kwargs)

        with patch.object(
            custom_kernels.importlib, "import_module", side_effect=broken_import
        ):
            status = custom_kernels.native_kernel_status()
        for name, report in status.items():
            assert report["available"] is False, name
            assert "simulated native import explosion" in report["import_error"]


def _engine_with_scheduler(scheduler, settings=None):
    """A pool-entry engine shaped like the loaded engines (#2859).

    Real engines expose the request core as ``_engine.engine`` and keep the
    per-model ``ModelSettings`` on ``_model_settings``.
    """
    core = SimpleNamespace(scheduler=scheduler)
    return SimpleNamespace(
        _engine=SimpleNamespace(engine=core),
        _model_settings=settings,
    )


def _pool_with_entries(entries):
    pool = MagicMock(
        spec=[
            "model_count",
            "loaded_model_count",
            "get_loaded_model_ids",
            "current_model_memory",
            "_entries",
        ]
    )
    pool.model_count = len(entries)
    pool.loaded_model_count = sum(1 for e in entries.values() if e.engine)
    pool.get_loaded_model_ids.return_value = [
        mid for mid, e in entries.items() if e.engine
    ]
    pool.current_model_memory = 0
    pool._entries = entries
    return pool


def _entry(engine):
    return SimpleNamespace(is_loading=False, engine=engine)


class _ExplodingEngine:
    """Engine that fails every attribute probe (unload racing a status poll)."""

    def __getattr__(self, name):
        raise RuntimeError("engine went away mid-poll")


class TestStatusTurboQuant:
    """/api/status exposes the effective per-model TurboQuant decision (#2859).

    Whether TQ engaged used to be discoverable only by grepping server logs,
    which contradict themselves inside a single load: "TurboQuant KV cache
    enabled for VLM: 8.0 bits" followed a second later by "TurboQuant
    disabled: model uses Multi-head Latent Attention". The engine-level bit
    depth is still reported as armed, so the status field is the only place
    a user can see the downgrade.
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def test_armed_eligible_model_reports_active(self, client):
        settings = SimpleNamespace(turboquant_kv_enabled=True, turboquant_kv_bits=8)
        scheduler = SimpleNamespace(
            _turboquant_kv_bits=8.0,
            _model_uses_mla=lambda: False,
            _model_uses_attention_sinks=lambda: False,
        )
        self._state.engine_pool = _pool_with_entries(
            {
                "qwen3.6-35b-a3b-4bit": _entry(
                    _engine_with_scheduler(scheduler, settings)
                ),
            }
        )

        data = client.get("/api/status").json()

        assert data["turboquant"] == {
            "requested_models": 1,
            "active_models": 1,
            "models": [
                {
                    "model_id": "qwen3.6-35b-a3b-4bit",
                    "requested": True,
                    "bits": 8.0,
                    "active": True,
                }
            ],
        }

    @pytest.mark.parametrize(
        ("armed_bits", "mla", "sinks", "expected_reason"),
        [
            (8.0, True, False, "Multi-head Latent Attention"),
            (8.0, False, True, "attention sinks"),
            # Requested in settings but the engine never armed it (GLM-5.3
            # composite latent/indexer cache bails out before setting bits).
            (None, False, False, "engine did not arm TurboQuant"),
        ],
    )
    def test_ineligible_model_reports_reason(
        self, client, armed_bits, mla, sinks, expected_reason
    ):
        settings = SimpleNamespace(turboquant_kv_enabled=True, turboquant_kv_bits=8)
        scheduler = SimpleNamespace(
            _turboquant_kv_bits=armed_bits,
            _model_uses_mla=lambda: mla,
            _model_uses_attention_sinks=lambda: sinks,
        )
        self._state.engine_pool = _pool_with_entries(
            {
                "deepseek-v4.1-flash-oq4e": _entry(
                    _engine_with_scheduler(scheduler, settings)
                ),
            }
        )

        data = client.get("/api/status").json()

        assert data["turboquant"]["requested_models"] == 1
        assert data["turboquant"]["active_models"] == 0
        report = data["turboquant"]["models"][0]
        assert report["requested"] is True
        assert report["bits"] == 8.0
        assert report["active"] is False
        assert expected_reason in report["reason"]

    def test_unrequested_and_unloaded_models_are_omitted(self, client):
        scheduler = SimpleNamespace(
            _turboquant_kv_bits=None,
            _model_uses_mla=lambda: False,
            _model_uses_attention_sinks=lambda: False,
        )
        settings = SimpleNamespace(turboquant_kv_enabled=False, turboquant_kv_bits=4)
        self._state.engine_pool = _pool_with_entries(
            {
                "fp16-model": _entry(_engine_with_scheduler(scheduler, settings)),
                "not-loaded-model": _entry(None),
            }
        )

        data = client.get("/api/status").json()

        assert data["turboquant"] == {
            "requested_models": 0,
            "active_models": 0,
            "models": [],
        }

    def test_engine_that_fails_attribute_probe_does_not_break_status(self, client):
        self._state.engine_pool = _pool_with_entries(
            {
                "racing-model": _entry(_ExplodingEngine()),
            }
        )

        resp = client.get("/api/status")

        assert resp.status_code == 200
        assert resp.json()["turboquant"]["models"] == []


class TestStatusCacheMemory:
    """`/api/status` reports per-model cache-tier occupancy (#2859).

    ``model_memory_used`` is the settled load footprint, not a KV gauge, and
    omlx keeps no resident-KV byte counter. Rather than inventing an estimate
    this block surfaces the counter the reporter used to measure KV growth
    indirectly: bytes written to the paged SSD tier, plus the in-memory hot
    cache, scoped to the owning model.
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def test_reports_ssd_tier_bytes_for_models_that_own_a_manager(self, client):
        manager = SimpleNamespace(
            get_stats_for_model=lambda name: SimpleNamespace(
                total_size_bytes=1_978_000_000,
                hot_cache_size_bytes=52_428_800,
                hot_cache_entries=12,
                num_files=640,
            )
        )
        ssd_scheduler = SimpleNamespace(
            _turboquant_kv_bits=None,
            config=SimpleNamespace(model_name="qwen3.6-35b-a3b-4bit"),
            paged_ssd_cache_manager=manager,
        )
        # No SSD cache configured for this one: it has no cache tier to report
        # and must be left out rather than rendered as zeros.
        memory_only_scheduler = SimpleNamespace(
            _turboquant_kv_bits=None,
            config=SimpleNamespace(model_name="memory-only-model"),
            paged_ssd_cache_manager=None,
        )
        self._state.engine_pool = _pool_with_entries(
            {
                "qwen3.6-35b-a3b-4bit": _entry(_engine_with_scheduler(ssd_scheduler)),
                "memory-only-model": _entry(
                    _engine_with_scheduler(memory_only_scheduler)
                ),
            }
        )

        data = client.get("/api/status").json()

        assert data["cache_memory"]["models"] == [
            {
                "model_id": "qwen3.6-35b-a3b-4bit",
                "ssd_cache_bytes": 1_978_000_000,
                "hot_cache_bytes": 52_428_800,
                "hot_cache_entries": 12,
                "num_files": 640,
            }
        ]


class TestStatusKvMemory:
    """`/api/status` measures the KV in-flight sequences hold right now (#2859).

    ``model_memory_used`` is the settled load footprint and ``cache_memory``
    is the paged SSD/hot tier, so neither moves while a prefill or a decode
    grows the live KV — the reporter sampled /api/status through a 40k-token
    cold prefill and saw a flat line. This block walks the cache tensors the
    request path is holding at poll time instead of estimating them.
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def _load(self, scheduler):
        self._state.engine_pool = _pool_with_entries(
            {"qwen3.5-0.8b-8bit": _entry(_engine_with_scheduler(scheduler))}
        )

    @pytest.mark.parametrize(
        ("holder", "expected_bytes"),
        [
            # Idle engine: nothing in flight, so the gauge is 0 — not null,
            # and the model is not dropped from the report.
            ("idle", 0),
            ("batch_generator", 4096),
            ("chunked_prefill", 2048),
            ("waiting_request", 1024),
        ],
    )
    def test_reports_the_kv_each_holder_is_holding(
        self, client, holder, expected_bytes
    ):
        scheduler = SimpleNamespace(batch_generator=None)
        if holder == "batch_generator":
            # mlx-lm's own accessor; absent (None) on generators that never
            # queued a sequence.
            scheduler.batch_generator = SimpleNamespace(prompt_cache_nbytes=4096)
        elif holder == "chunked_prefill":
            scheduler._prefill_states = {
                "req-1": SimpleNamespace(cache=[SimpleNamespace(nbytes=2048)])
            }
        elif holder == "waiting_request":
            scheduler.waiting = [
                SimpleNamespace(prompt_cache=[SimpleNamespace(nbytes=1024)])
            ]
        self._load(scheduler)

        data = client.get("/api/status").json()

        assert data["kv_memory"]["models"] == [
            {"model_id": "qwen3.5-0.8b-8bit", "resident_bytes": expected_bytes}
        ]

    def test_sums_all_holders_and_counts_an_aliased_cache_once(self, client):
        # A prefill paused for LRU eviction keeps its cache on the request
        # while a resumed chunked prefill holds the same list object, so the
        # same tensors are reachable twice.
        shared = [SimpleNamespace(nbytes=8192)]
        scheduler = SimpleNamespace(
            batch_generator=SimpleNamespace(prompt_cache_nbytes=4096),
            _prefill_states={"req-1": SimpleNamespace(cache=shared)},
            _vlm_mtp_active={
                7: SimpleNamespace(prompt_cache=[SimpleNamespace(nbytes=512)])
            },
            waiting=[SimpleNamespace(prompt_cache=shared)],
            running={},
        )
        self._load(scheduler)

        data = client.get("/api/status").json()

        assert data["kv_memory"]["models"] == [
            {"model_id": "qwen3.5-0.8b-8bit", "resident_bytes": 4096 + 8192 + 512}
        ]

    def test_a_holder_mutating_mid_poll_is_dropped_not_fatal(self, client):
        class _RacingBatchGenerator:
            """BatchGenerator whose queues a step grows while status reads them."""

            @property
            def prompt_cache_nbytes(self):
                raise RuntimeError("deque mutated during iteration")

        scheduler = SimpleNamespace(
            batch_generator=_RacingBatchGenerator(),
            waiting=[SimpleNamespace(prompt_cache=[SimpleNamespace(nbytes=1024)])],
        )
        self._load(scheduler)

        resp = client.get("/api/status")

        assert resp.status_code == 200
        assert resp.json()["kv_memory"]["models"] == [
            {"model_id": "qwen3.5-0.8b-8bit", "resident_bytes": 1024}
        ]


class TestStatusTurboQuantConversionEvidence:
    """`/api/status` shows whether a TurboQuant conversion actually ran (#2859).

    ``active`` only means "armed, with no model-level veto", which is already
    true before the first request ever arrives, so it cannot tell a user
    whether a served request's fp16 KV was ever quantized. The evidence is
    the layer count the conversion itself reports; like ``reason``, the key
    is omitted while it would say nothing.
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    @pytest.mark.parametrize(
        ("mla", "converted_layers", "expected_layers", "expected_active"),
        [
            # Armed and eligible, but no request has converted anything yet.
            (False, 0, None, True),
            (False, 24, 24, True),
            # Vetoed model: arming is not evidence, so there is none to show.
            (True, 0, None, False),
        ],
    )
    def test_conversion_evidence_tracks_real_conversions(
        self, client, mla, converted_layers, expected_layers, expected_active
    ):
        settings = SimpleNamespace(turboquant_kv_enabled=True, turboquant_kv_bits=8)
        scheduler = SimpleNamespace(
            _turboquant_kv_bits=8.0,
            _turboquant_kv_converted_layers=converted_layers,
            _model_uses_mla=lambda: mla,
            _model_uses_attention_sinks=lambda: False,
        )
        self._state.engine_pool = _pool_with_entries(
            {
                "qwen3.6-35b-a3b-4bit": _entry(
                    _engine_with_scheduler(scheduler, settings)
                ),
            }
        )

        data = client.get("/api/status").json()

        report = data["turboquant"]["models"][0]
        assert report["active"] is expected_active
        assert report.get("converted_layers") == expected_layers
        assert ("converted_layers" in report) is (expected_layers is not None)
