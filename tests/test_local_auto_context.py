# SPDX-License-Identifier: Apache-2.0
"""Auto context for local (single-node) serving — function and endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import omlx.local_context as local_context
from omlx.admin import routes as admin_routes
from omlx.cluster.planner import ModelLayout, synthetic_model_layout
from omlx.local_context import auto_context_for_layout, local_auto_context
from omlx.model_settings import ModelSettingsManager

GiB = 1024**3

# Qwen3-32B shaped: 8 KV heads x 128 dims x 2 bytes x (K and V).
KV_BYTES_PER_TOKEN_PER_LAYER = 8 * 128 * 2 * 2


def _model(size_gib, layers=48, kv=KV_BYTES_PER_TOKEN_PER_LAYER):
    """A layout with a real KV rate — the thing that makes context cost."""

    total = int(size_gib * GiB)
    base, remainder = divmod(total, layers)
    return ModelLayout(
        source="synthetic",
        fixed_weight_bytes=0,
        layer_weight_bytes=tuple(
            base + (1 if index < remainder else 0) for index in range(layers)
        ),
        kv_bytes_per_token_by_layer=(kv,) * layers,
        supports_tensor_parallel=True,
        supports_pipeline=True,
    )


# --- The layout-level arithmetic --------------------------------------------


def test_a_model_with_room_to_spare_gets_a_real_context():
    tokens = auto_context_for_layout(_model(20), capacity_bytes=128 * GiB)
    assert tokens >= 8192


def test_auto_never_exceeds_the_declared_context_length():
    tokens = auto_context_for_layout(
        _model(20),
        capacity_bytes=128 * GiB,
        declared_context_tokens=40_960,
    )
    assert 0 < tokens <= 40_960


def test_a_layout_with_no_kv_information_says_zero_not_a_guess():
    layout = synthetic_model_layout(total_weight_bytes=20 * GiB, layer_count=48)
    assert layout.kv_bytes_per_token_by_layer == ()
    assert auto_context_for_layout(layout, capacity_bytes=128 * GiB) == 0


def test_a_model_bigger_than_the_machine_says_zero():
    assert auto_context_for_layout(_model(200), capacity_bytes=64 * GiB) == 0


def test_zero_capacity_says_zero():
    assert auto_context_for_layout(_model(20), capacity_bytes=0) == 0


# --- The path-level wrapper: advisory, never raises --------------------------


def test_an_unreadable_model_path_returns_zeros_without_raising(tmp_path):
    result = local_auto_context(tmp_path / "no-such-model", capacity_bytes=64 * GiB)
    assert result == {"max_context_tokens": 0, "declared_context_tokens": 0}


def test_a_readable_model_reports_both_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "omlx.cluster.planner.complete_model_layout", lambda path: _model(20)
    )
    monkeypatch.setattr(
        "omlx.model_discovery._read_model_context_length", lambda path: 40_960
    )
    result = local_auto_context(tmp_path, capacity_bytes=128 * GiB)
    assert result["declared_context_tokens"] == 40_960
    assert 0 < result["max_context_tokens"] <= 40_960


def test_default_capacity_comes_from_the_live_memory_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "omlx.cluster.planner.complete_model_layout", lambda path: _model(20)
    )
    monkeypatch.setattr(
        "omlx.model_discovery._read_model_context_length", lambda path: None
    )
    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown",
        lambda *args, **kwargs: {"hard_limit": 128 * GiB},
    )
    result = local_auto_context(tmp_path)
    assert result["max_context_tokens"] >= 8192
    assert result["declared_context_tokens"] == 0


# --- The endpoint -------------------------------------------------------------


class _FakeEntry:
    def __init__(self, model_path: str = "/fake"):
        self.model_path = model_path


class _FakePool:
    def __init__(self):
        self._entries = {"model-a": _FakeEntry()}

    def get_entry(self, model_id):
        return self._entries.get(model_id)


@pytest.fixture
def client(tmp_path, monkeypatch):
    mgr = ModelSettingsManager(tmp_path)
    pool = _FakePool()

    admin_routes._get_settings_manager = lambda: mgr
    admin_routes._get_engine_pool = lambda: pool

    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app)


class TestAutoContextEndpoint:
    def test_unknown_model_is_404(self, client):
        r = client.get("/admin/api/models/nope/auto_context")
        assert r.status_code == 404

    def test_known_model_returns_the_assessment(self, client, monkeypatch):
        monkeypatch.setattr(
            local_context,
            "local_auto_context",
            lambda path, **kwargs: {
                "max_context_tokens": 131_072,
                "declared_context_tokens": 262_144,
            },
        )
        r = client.get("/admin/api/models/model-a/auto_context")
        assert r.status_code == 200
        assert r.json() == {
            "max_context_tokens": 131_072,
            "declared_context_tokens": 262_144,
        }

    def test_an_unreadable_model_yields_zeros_not_an_error(self, client):
        # model_path is "/fake" — no config, no safetensors. Advisory
        # endpoints answer "unknown", they do not fail the modal.
        r = client.get("/admin/api/models/model-a/auto_context")
        assert r.status_code == 200
        assert r.json() == {"max_context_tokens": 0, "declared_context_tokens": 0}
