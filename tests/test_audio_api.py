# SPDX-License-Identifier: Apache-2.0
"""Tests for GET /v1/models listing audio models (INV-02).

Verifies that audio_stt and audio_tts models appear in the /v1/models
response with correct fields, and that they coexist with other engine types.

All tests use FastAPI TestClient with a mocked EnginePool — no mlx-audio
or real model loading required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_entry(
    model_id: str,
    model_type: str,
    engine_type: str,
    engine=None,
    is_pinned: bool = False,
    is_loading: bool = False,
) -> MagicMock:
    """Build a minimal mock EngineEntry."""
    entry = MagicMock()
    entry.model_id = model_id
    entry.model_type = model_type
    entry.engine_type = engine_type
    entry.engine = engine
    entry.is_pinned = is_pinned
    entry.is_loading = is_loading
    entry.estimated_size = 1024 * 1024 * 500  # 500 MB
    entry.last_access = 0.0
    return entry


def _make_pool(entries: list) -> MagicMock:
    """Build a mock EnginePool with the given entries."""
    pool = MagicMock()
    pool.preload_pinned_models = AsyncMock()
    pool.check_ttl_expirations = AsyncMock()
    pool.shutdown = AsyncMock()
    pool.get_model_ids.return_value = [e.model_id for e in entries]
    pool.get_entry.side_effect = lambda mid: next(
        (e for e in entries if e.model_id == mid), None
    )
    pool.get_status.return_value = {
        "max_model_memory": 32 * 1024**3,
        "current_model_memory": 0,
        "model_count": len(entries),
        "loaded_count": sum(1 for e in entries if e.engine is not None),
        "models": [
            {
                "id": e.model_id,
                "model_type": e.model_type,
                "engine_type": e.engine_type,
                "loaded": e.engine is not None,
                "pinned": e.is_pinned,
                "is_loading": e.is_loading,
                "estimated_size": e.estimated_size,
                "last_access": e.last_access,
            }
            for e in entries
        ],
    }
    return pool


# ---------------------------------------------------------------------------
# TestModelsListAudio
# ---------------------------------------------------------------------------


class TestModelsListAudio:
    """GET /v1/models must include audio models with correct fields."""

    @pytest.fixture
    def stt_entry(self):
        return _make_engine_entry(
            "whisper-large-v3", "audio_stt", "stt", engine=MagicMock()
        )

    @pytest.fixture
    def tts_entry(self):
        return _make_engine_entry(
            "qwen3-tts", "audio_tts", "tts", engine=None
        )

    @pytest.fixture
    def llm_entry(self):
        return _make_engine_entry(
            "llama-3b", "llm", "batched", engine=None
        )

    @pytest.fixture
    def client_with_stt(self, stt_entry):
        """TestClient with a pool containing only an STT model."""
        from omlx.server import app

        mock_pool = _make_pool([stt_entry])
        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = MagicMock()
            mock_state.settings_manager.get_settings.return_value = MagicMock(
                model_alias=None
            )
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, mock_pool

    @pytest.fixture
    def client_with_mixed(self, stt_entry, tts_entry, llm_entry):
        """TestClient with a pool containing STT + TTS + LLM models."""
        from omlx.server import app

        mock_pool = _make_pool([stt_entry, tts_entry, llm_entry])
        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = MagicMock()
            mock_state.settings_manager.get_settings.return_value = MagicMock(
                model_alias=None
            )
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, mock_pool

    def test_models_list_returns_200(self, client_with_stt):
        client, _ = client_with_stt
        response = client.get("/v1/models")
        assert response.status_code == 200

    def test_models_list_includes_stt_model(self, client_with_stt):
        """audio_stt model appears in /v1/models response."""
        client, _ = client_with_stt
        response = client.get("/v1/models")
        assert response.status_code == 200

        body = response.json()
        assert "data" in body
        model_ids = [m["id"] for m in body["data"]]
        assert "whisper-large-v3" in model_ids

    def test_stt_model_has_required_openai_fields(self, client_with_stt):
        """Each model entry has id, object, owned_by per OpenAI spec."""
        client, _ = client_with_stt
        response = client.get("/v1/models")
        body = response.json()

        stt_model = next(
            (m for m in body["data"] if m["id"] == "whisper-large-v3"), None
        )
        assert stt_model is not None, "whisper-large-v3 not found in /v1/models"
        assert "id" in stt_model
        assert "object" in stt_model
        assert "owned_by" in stt_model

    def test_stt_model_object_field_value(self, client_with_stt):
        """Model object field is 'model'."""
        client, _ = client_with_stt
        response = client.get("/v1/models")
        body = response.json()

        stt_model = next(
            (m for m in body["data"] if m["id"] == "whisper-large-v3"), None
        )
        assert stt_model is not None
        assert stt_model["object"] == "model"

    def test_models_list_includes_tts_model(self, client_with_mixed):
        """audio_tts model appears in /v1/models response."""
        client, _ = client_with_mixed
        response = client.get("/v1/models")
        assert response.status_code == 200

        body = response.json()
        model_ids = [m["id"] for m in body["data"]]
        assert "qwen3-tts" in model_ids

    def test_audio_models_coexist_with_llm(self, client_with_mixed):
        """Audio models and LLM appear together in the same /v1/models response."""
        client, _ = client_with_mixed
        response = client.get("/v1/models")
        body = response.json()
        model_ids = {m["id"] for m in body["data"]}

        assert "whisper-large-v3" in model_ids
        assert "qwen3-tts" in model_ids
        assert "llama-3b" in model_ids

    def test_models_list_response_top_level_fields(self, client_with_stt):
        """Response top-level has 'object' and 'data' fields."""
        client, _ = client_with_stt
        body = client.get("/v1/models").json()
        assert body.get("object") == "list"
        assert isinstance(body.get("data"), list)
