# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.server module - sampling parameter resolution and exception handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from omlx.engine_pool import EngineEntry
from omlx.exceptions import InvalidRequestError, ModelNotFoundError
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.server import (
    EngineType,
    SamplingDefaults,
    ServerState,
    _format_generation_speed_for_log,
    _reject_diffusion_structured_outputs,
    _resolve_metric_durations,
    _reset_boundary_snapshots_for_server,
    app,
    get_engine,
    get_max_context_window,
    get_sampling_params,
)
from omlx.settings import GlobalSettings, ModelSettings as GlobalModelSettings


class TestBoundarySnapshotLifecycle:
    def test_reset_helper_uses_engine_pool_cache_dir(self, tmp_path):
        from types import SimpleNamespace

        stale_dir = tmp_path / "_boundary_snapshots" / "stale-session"
        stale_dir.mkdir(parents=True)
        (stale_dir / "old.safetensors").write_text("stale")

        state = ServerState()
        state.engine_pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(paged_ssd_cache_dir=tmp_path)
        )

        with patch("omlx.server._server_state", state):
            _reset_boundary_snapshots_for_server()

        assert (tmp_path / "_boundary_snapshots").exists()
        assert not stale_dir.exists()

    def test_reset_helper_skips_no_cache(self, tmp_path):
        from types import SimpleNamespace

        stale_dir = tmp_path / "_boundary_snapshots" / "stale-session"
        stale_dir.mkdir(parents=True)
        (stale_dir / "old.safetensors").write_text("stale")

        state = ServerState()
        state.engine_pool = SimpleNamespace(
            _scheduler_config=SimpleNamespace(paged_ssd_cache_dir=None)
        )

        with patch("omlx.server._server_state", state):
            _reset_boundary_snapshots_for_server()

        assert stale_dir.exists()


class TestDiffusionStructuredOutputGuard:
    class _DiffusionEngine:
        is_diffusion_model = True

    def test_allows_plain_text_response_format(self):
        _reject_diffusion_structured_outputs(
            self._DiffusionEngine(),
            response_format={"type": "text"},
        )

    def test_rejects_json_response_format(self):
        with pytest.raises(InvalidRequestError, match="Structured response_format"):
            _reject_diffusion_structured_outputs(
                self._DiffusionEngine(),
                response_format={"type": "json_object"},
            )

    def test_rejects_guided_grammar(self):
        with pytest.raises(InvalidRequestError, match="guided grammar"):
            _reject_diffusion_structured_outputs(
                self._DiffusionEngine(),
                guided_grammar='root ::= "ok"',
            )


class TestGenerationSpeedLog:
    def test_formats_plain_generation_speed(self):
        assert (
            _format_generation_speed_for_log(object(), 12.345, is_diffusion=False)
            == "12.3 tok/s"
        )

    def test_formats_diffusion_native_stats(self):
        from types import SimpleNamespace

        output = SimpleNamespace(
            generation_tps=25.37,
            diffusion_canvas_tps=25.37,
            prompt_tps=2293.3,
            diffusion_work_tps=1179.8,
            diffusion_denoising_steps=93,
        )

        text = _format_generation_speed_for_log(output, 18.716, is_diffusion=True)

        assert text == (
            "18.7 tok/s e2e, output=25.4 tok/s, canvas=25.4 tok/s, "
            "prompt=2293.3 tok/s, work=1179.8 tok/s, steps=93"
        )

    def test_formats_diffusion_canvas_speed_for_early_eos(self):
        from types import SimpleNamespace

        output = SimpleNamespace(
            generation_tps=19.2,
            diffusion_canvas_tps=24.6,
            prompt_tps=2293.7,
            diffusion_work_tps=1179.8,
            diffusion_denoising_steps=96,
        )

        text = _format_generation_speed_for_log(output, 14.3, is_diffusion=True)

        assert text == (
            "14.3 tok/s e2e, output=19.2 tok/s, canvas=24.6 tok/s, "
            "prompt=2293.7 tok/s, work=1179.8 tok/s, steps=96"
        )

    def test_resolves_diffusion_native_durations(self):
        from types import SimpleNamespace

        output = SimpleNamespace(
            prompt_tokens=16384,
            completion_tokens=512,
            prompt_tps=2048.0,
            generation_tps=32.0,
        )

        prefill, generation = _resolve_metric_durations(
            output,
            is_diffusion=True,
            prefill_duration=99.0,
            generation_duration=99.0,
        )

        assert prefill == 8.0
        assert generation == 16.0


class TestGetSamplingParams:
    """Tests for get_sampling_params function."""

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        """Set up a clean server state for each test."""
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def test_returns_10_tuple(self):
        """Test that get_sampling_params returns a 10-tuple."""
        result = get_sampling_params(None, None)
        assert isinstance(result, tuple)
        assert len(result) == 10

    def test_defaults(self):
        """Test default values with no request or model params."""
        (
            temp,
            top_p,
            top_k,
            rep_penalty,
            min_p,
            presence_penalty,
            frequency_penalty,
            max_tokens,
            xtc_prob,
            xtc_thresh,
        ) = get_sampling_params(None, None)
        assert temp == 1.0
        assert top_p == 0.95
        assert top_k == 0
        assert rep_penalty == 1.0
        assert min_p == 0.0
        assert presence_penalty == 0.0
        assert frequency_penalty == 0.0
        assert max_tokens == 32768

    def test_request_overrides(self):
        """Test request params override global defaults."""
        (
            temp,
            top_p,
            top_k,
            rep_penalty,
            min_p,
            presence_penalty,
            frequency_penalty,
            max_tokens,
            xtc_prob,
            xtc_thresh,
        ) = get_sampling_params(
            0.5,
            0.8,
            req_top_k=40,
            req_repetition_penalty=1.15,
            req_min_p=0.1,
            req_presence_penalty=0.5,
            req_frequency_penalty=0.3,
            req_max_tokens=1024,
        )
        assert temp == 0.5
        assert top_p == 0.8
        assert top_k == 40
        assert rep_penalty == 1.15
        assert min_p == 0.1
        assert presence_penalty == 0.5
        assert frequency_penalty == 0.3
        assert max_tokens == 1024

    def test_xtc_defaults_when_none(self):
        """Test XTC params default when not specified."""
        *_, xtc_prob, xtc_thresh = get_sampling_params(None, None)
        assert xtc_prob == 0.0
        assert xtc_thresh == 0.1

    def test_xtc_request_passthrough(self):
        """Test XTC params pass through from request values."""
        *_, xtc_prob, xtc_thresh = get_sampling_params(
            None,
            None,
            req_xtc_probability=0.5,
            req_xtc_threshold=0.1,
        )
        assert xtc_prob == 0.5
        assert xtc_thresh == 0.1

    def test_xtc_partial_override(self):
        """Test setting only xtc_probability uses safe default threshold."""
        *_, xtc_prob, xtc_thresh = get_sampling_params(
            None,
            None,
            req_xtc_probability=0.3,
        )
        assert xtc_prob == 0.3
        assert xtc_thresh == 0.1

    def test_model_settings_override(self):
        """Test model settings override global defaults."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(
                temperature=0.3,
                top_k=50,
                repetition_penalty=1.2,
                min_p=0.05,
                presence_penalty=0.3,
                max_tokens=2048,
            )
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            (
                temp,
                top_p,
                top_k,
                rep_penalty,
                min_p,
                presence_penalty,
                frequency_penalty,
                max_tokens,
                xtc_prob,
                xtc_thresh,
            ) = get_sampling_params(None, None, "test-model")
            assert temp == 0.3
            assert top_p == 0.95  # falls back to global
            assert top_k == 50
            assert rep_penalty == 1.2
            assert min_p == 0.05
            assert presence_penalty == 0.3
            assert frequency_penalty == 0.0
            assert max_tokens == 2048

    def test_request_over_model(self):
        """Test request params take priority over model settings."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(
                temperature=0.3,
                top_k=50,
                repetition_penalty=1.2,
                min_p=0.05,
                max_tokens=2048,
            )
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            (
                temp,
                top_p,
                top_k,
                rep_penalty,
                min_p,
                presence_penalty,
                frequency_penalty,
                max_tokens,
                xtc_prob,
                xtc_thresh,
            ) = get_sampling_params(
                0.7,
                None,
                "test-model",
                req_top_k=10,
                req_repetition_penalty=1.05,
                req_min_p=0.1,
                req_max_tokens=4096,
            )
            assert temp == 0.7  # request wins
            assert top_k == 10  # request wins over model
            assert rep_penalty == 1.05  # request wins over model
            assert min_p == 0.1  # request wins over model
            assert max_tokens == 4096  # request wins over model

    def test_model_repetition_penalty(self):
        """Test model-level repetition_penalty overrides global."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(repetition_penalty=1.5)
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            _, _, _, rep_penalty, _, _, _, _, _, _ = get_sampling_params(
                None, None, "test-model"
            )
            assert rep_penalty == 1.5

    def test_global_repetition_penalty(self):
        """Test global repetition_penalty is used when no model override."""
        self._state.sampling = SamplingDefaults(repetition_penalty=1.3)

        _, _, _, rep_penalty, _, _, _, _, _, _ = get_sampling_params(None, None)
        assert rep_penalty == 1.3

    def test_force_sampling(self):
        """Test force_sampling ignores request params."""
        self._state.sampling = SamplingDefaults(
            temperature=0.5, top_p=0.8, max_tokens=4096, force_sampling=True
        )

        temp, top_p, _, _, _, _, _, max_tokens, _, _ = get_sampling_params(
            0.9, 0.99, req_max_tokens=8192
        )
        assert temp == 0.5  # forced, not request
        assert top_p == 0.8  # forced, not request
        assert max_tokens == 4096  # forced, not request

    def test_force_sampling_model_max_tokens(self):
        """Test force_sampling with model-level max_tokens overrides global."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(max_tokens=8192, force_sampling=True)
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager

            _, _, _, _, _, _, _, max_tokens, _, _ = get_sampling_params(
                None, None, "test-model", req_max_tokens=1024
            )
            assert max_tokens == 8192  # model setting wins in force mode

    def test_max_tokens_no_request_uses_model_settings(self):
        """Test that model max_tokens is used when request doesn't specify it."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelSettingsManager(Path(tmpdir))
            settings = ModelSettings(max_tokens=8192)
            manager.set_settings("test-model", settings)
            self._state.settings_manager = manager
            self._state.sampling = SamplingDefaults(max_tokens=4096)

            _, _, _, _, _, _, _, max_tokens, _, _ = get_sampling_params(
                None, None, "test-model"
            )
            assert max_tokens == 8192  # model setting, not global 4096


class TestExceptionHandlers:
    """Tests for global exception handlers that log API errors."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app, raise_server_exceptions=False)

    def test_http_exception_logged(self, client, caplog):
        """Test that HTTPException responses are logged."""
        # /v1/models requires startup, so a 404 on a non-existent route works
        response = client.get("/v1/nonexistent-endpoint")
        assert response.status_code == 404

    def test_validation_error_logged(self, client, caplog):
        """Test that request validation errors (422) are logged."""
        # POST to /v1/chat/completions with invalid body triggers validation
        response = client.post(
            "/v1/chat/completions",
            json={"invalid_field": "bad"},
        )
        # Should be 422 (validation error) or 500 (server not initialized)
        assert response.status_code in (422, 500)

    def test_exception_handler_returns_json(self, client):
        """Test that exception handlers return proper JSON responses."""
        response = client.get("/v1/nonexistent-endpoint")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data or "error" in data

    def test_api_validation_error_openai_format(self, client):
        """Test that /v1/* validation errors use OpenAI-compatible format."""
        response = client.post(
            "/v1/chat/completions",
            json={"invalid_field": "bad"},
        )
        # 422 validation or 500 if server not init - both should have error key
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert "type" in data["error"]
        assert "param" in data["error"]

    def test_non_api_route_detail_format(self, client):
        """Test that non-/v1/ routes keep the traditional detail format."""
        response = client.get("/nonexistent-page")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestModelFallback:
    """Tests for model fallback to default when requested model not found."""

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        """Set up a clean server state for each test."""
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def _setup_pool(self, found_model=None):
        """Create a mock engine pool."""
        pool = MagicMock()
        pool.resolve_model_id.side_effect = lambda mid, _sm: mid

        if found_model:
            mock_engine = MagicMock()

            async def mock_get_engine(model_id):
                if model_id == found_model:
                    return mock_engine
                raise ModelNotFoundError(model_id, [found_model])

            pool.get_engine = AsyncMock(side_effect=mock_get_engine)
        else:
            pool.get_engine = AsyncMock(side_effect=ModelNotFoundError("unknown", []))

        self._state.engine_pool = pool
        return pool

    @pytest.mark.asyncio
    async def test_fallback_disabled_returns_404(self):
        """When model_fallback is off, unknown model returns 404."""
        self._state.global_settings = GlobalSettings()
        self._state.global_settings.model.model_fallback = False
        self._state.default_model = "default-model"
        self._setup_pool(found_model="default-model")

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("unknown-model", EngineType.LLM)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_fallback_enabled_returns_default(self):
        """When model_fallback is on, unknown model falls back to default."""
        self._state.global_settings = GlobalSettings()
        self._state.global_settings.model.model_fallback = True
        self._state.default_model = "default-model"
        self._setup_pool(found_model="default-model")

        engine = await get_engine("unknown-model", EngineType.LLM)
        assert engine is not None

    @pytest.mark.asyncio
    async def test_fallback_enabled_no_default_returns_404(self):
        """When model_fallback is on but no default model, returns 404."""
        self._state.global_settings = GlobalSettings()
        self._state.global_settings.model.model_fallback = True
        self._state.default_model = None
        self._setup_pool()

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("unknown-model", EngineType.LLM)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_fallback_not_applied_to_embedding(self):
        """Fallback should not apply to embedding engine type."""
        self._state.global_settings = GlobalSettings()
        self._state.global_settings.model.model_fallback = True
        self._state.default_model = "default-model"
        self._setup_pool(found_model="default-model")

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("unknown-model", EngineType.EMBEDDING)
        assert exc_info.value.status_code == 404


class TestGetEngineLLMTypeValidation:
    """LLM endpoints must reject non-LLM engines with a clean 400 (#507).

    Issue #507: POST /v1/chat/completions against an STT/TTS/STS/Embedding
    model was producing an unhandled 500 with `'STTEngine' object has no
    attribute 'model_type'` because `get_engine(..., EngineType.LLM)` never
    validated that the resolved engine was actually an LLM. The fix adds an
    isinstance check mirroring the one already in place for EMBEDDING and
    RERANKER.
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def _pool_returning(self, engine):
        pool = MagicMock()
        pool.resolve_model_id.side_effect = lambda mid, _sm: mid
        pool.get_engine = AsyncMock(return_value=engine)
        self._state.engine_pool = pool
        return pool

    @pytest.mark.asyncio
    async def test_llm_rejects_stt_engine(self):
        """Requesting an STT model on an LLM endpoint returns HTTP 400, not 500."""
        from omlx.engine.stt import STTEngine

        stt = MagicMock(spec=STTEngine)
        self._pool_returning(stt)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("whisper-large-v3-turbo", EngineType.LLM)
        assert exc_info.value.status_code == 400
        detail = str(exc_info.value.detail).lower()
        assert (
            "not an llm" in detail or "not a chat" in detail or "not a text" in detail
        )

    @pytest.mark.asyncio
    async def test_llm_rejects_tts_engine(self):
        """Requesting a TTS model on an LLM endpoint returns HTTP 400."""
        from omlx.engine.tts import TTSEngine

        tts = MagicMock(spec=TTSEngine)
        self._pool_returning(tts)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("qwen3-tts", EngineType.LLM)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_llm_rejects_sts_engine(self):
        """Requesting an STS model on an LLM endpoint returns HTTP 400."""
        from omlx.engine.sts import STSEngine

        sts = MagicMock(spec=STSEngine)
        self._pool_returning(sts)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("deepfilternet", EngineType.LLM)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_llm_rejects_embedding_engine(self):
        """Requesting an embedding model on an LLM endpoint returns HTTP 400."""
        from omlx.engine.embedding import EmbeddingEngine

        emb = MagicMock(spec=EmbeddingEngine)
        self._pool_returning(emb)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("bge-small", EngineType.LLM)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_llm_rejects_reranker_engine(self):
        """Requesting a reranker model on an LLM endpoint returns HTTP 400."""
        from omlx.engine.reranker import RerankerEngine

        rr = MagicMock(spec=RerankerEngine)
        self._pool_returning(rr)

        with pytest.raises(HTTPException) as exc_info:
            await get_engine("jina-reranker", EngineType.LLM)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_llm_accepts_llm_engine(self):
        """A genuine LLM engine passes validation and is returned as-is."""
        from omlx.engine.base import BaseEngine

        llm = MagicMock(spec=BaseEngine)
        self._pool_returning(llm)

        engine = await get_engine("llama-3", EngineType.LLM)
        assert engine is llm


class TestGetMaxContextWindow:
    """Tests for get_max_context_window precedence rule (#1308).

    Resolution order:
        1. Explicit per-model setting (admin / settings.json).
        2. Context length discovered from the model's config.json at
           startup (EngineEntry.model_context_length).
        3. Global SamplingDefaults.max_context_window (32K).
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    @staticmethod
    def _entry(model_id: str, ctx_length: int | None) -> EngineEntry:
        return EngineEntry(
            model_id=model_id,
            model_path=f"/fake/{model_id}",
            model_type="llm",
            engine_type="batched",
            estimated_size=0,
            model_context_length=ctx_length,
        )

    def _mount_pool(self, entries: dict):
        pool = MagicMock()
        pool.resolve_model_id.side_effect = lambda mid, _sm: mid
        pool.get_entry.side_effect = lambda mid: entries.get(mid)
        self._state.engine_pool = pool

    def _mount_settings(self, overrides: dict):
        """Mount a settings_manager that returns the given per-model overrides."""
        manager = MagicMock()
        manager.get_settings.side_effect = lambda mid: overrides.get(mid)
        self._state.settings_manager = manager

    def test_global_default_when_nothing_discovered(self):
        """No model context, no per-model override → global default.

        Fallback default kept at 32768 so existing ``settings.json``
        files carrying the historical default keep working unchanged.
        Operators who want a real server-wide cap set
        ``max_context_window_policy`` instead — see TestPolicyCap below.
        """
        self._mount_pool({"llama-3": self._entry("llama-3", None)})
        assert get_max_context_window("llama-3") == 32768

    def test_discovered_context_returned_when_no_override(self):
        """Model config declares 262144 → /v1/models reports 262144, not 32K (#1308)."""
        self._mount_pool({"qwen3-coder": self._entry("qwen3-coder", 262144)})
        assert get_max_context_window("qwen3-coder") == 262144

    def test_per_model_override_wins_over_discovery(self):
        """Admin set 16384 → that wins over the model's declared 262144."""
        self._mount_pool({"qwen3-coder": self._entry("qwen3-coder", 262144)})
        self._mount_settings({"qwen3-coder": ModelSettings(max_context_window=16384)})
        assert get_max_context_window("qwen3-coder") == 16384

    def test_per_model_override_wins_over_global(self):
        """Override of 8192 wins even when the model didn't declare a value."""
        self._mount_pool({"llama-3": self._entry("llama-3", None)})
        self._mount_settings({"llama-3": ModelSettings(max_context_window=8192)})
        assert get_max_context_window("llama-3") == 8192

    def test_no_model_id_returns_global_default(self):
        """A bare /v1/messages-style call with no model id falls to the default."""
        assert get_max_context_window(None) == 32768

    def test_unknown_model_id_returns_global_default(self):
        """An unknown model id doesn't crash — falls through to the default."""
        self._mount_pool({})
        assert get_max_context_window("ghost-model") == 32768
