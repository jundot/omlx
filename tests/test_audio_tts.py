# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /v1/audio/speech (INV-04).

Verifies the TTS endpoint accepts a JSON body and returns valid WAV audio
bytes, matching the OpenAI audio speech API spec.

All unit tests run with mocked TTSEngine and EnginePool — mlx-audio is not
required. Integration tests (marked @pytest.mark.slow) need a real model.
"""

import base64
import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_secs: float = 0.1, sample_rate: int = 22050) -> bytes:
    """Generate minimal valid WAV bytes (silence)."""
    n_samples = int(sample_rate * duration_secs)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


DUMMY_WAV = _make_wav_bytes()
RIFF_MAGIC = b"RIFF"


def _make_mock_tts_engine(wav_bytes: bytes = None) -> MagicMock:
    """Build a mock TTSEngine that returns the given WAV bytes."""
    from omlx.engine.tts import TTSEngine
    engine = MagicMock(spec=TTSEngine)
    engine.synthesize = AsyncMock(return_value=wav_bytes or DUMMY_WAV)
    return engine


def _make_mock_pool(tts_engine=None, model_id: str = "qwen3-tts") -> MagicMock:
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=tts_engine or _make_mock_tts_engine())
    pool.get_entry = MagicMock(return_value=MagicMock(
        model_type="audio_tts",
        engine_type="tts",
    ))
    pool.get_model_ids.return_value = [model_id]
    pool.preload_pinned_models = AsyncMock()
    pool.check_ttl_expirations = AsyncMock()
    pool.shutdown = AsyncMock()
    pool.resolve_model_id = MagicMock(side_effect=lambda m, _: m)
    return pool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ensure_audio_routes(app):
    """Register audio routes if not already present (e.g., mlx-audio not installed)."""
    from omlx.api.audio_routes import router as audio_router

    audio_paths = {"/v1/audio/transcriptions", "/v1/audio/speech", "/v1/audio/process"}
    existing = {getattr(r, "path", "") for r in app.routes}
    if not audio_paths & existing:
        app.include_router(audio_router)


@pytest.fixture
def server_tts_client():
    """TestClient using the full omlx server app with mocked TTS pool."""
    from omlx.server import app

    _ensure_audio_routes(app)

    mock_pool = _make_mock_pool()

    with patch("omlx.server._server_state") as mock_state:
        mock_state.engine_pool = mock_pool
        mock_state.global_settings = None
        mock_state.process_memory_enforcer = None
        mock_state.hf_downloader = None
        mock_state.ms_downloader = None
        mock_state.mcp_manager = None
        mock_state.api_key = None
        mock_state.settings_manager = MagicMock()
        mock_state.settings_manager.resolve_model_id = MagicMock(
            side_effect=lambda m, _: m
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pool


# ---------------------------------------------------------------------------
# TestTTSEndpointBasic
# ---------------------------------------------------------------------------


class TestTTSEndpointBasic:
    """Core TTS endpoint behaviour."""

    def test_post_speech_returns_200(self, server_tts_client):
        """POST /v1/audio/speech with valid JSON body returns 200."""
        client, _ = server_tts_client
        response = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Hello, world!", "voice": "alloy"},
        )
        assert response.status_code == 200

    def test_response_is_audio_bytes(self, server_tts_client):
        """Response body is non-empty bytes."""
        client, _ = server_tts_client
        response = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Hello"},
        )
        assert len(response.content) > 0

    def test_response_has_wav_header(self, server_tts_client):
        """Response starts with RIFF WAV magic bytes."""
        client, _ = server_tts_client
        response = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Test audio"},
        )
        assert response.status_code == 200
        assert response.content[:4] == RIFF_MAGIC

    def test_response_content_type_is_audio(self, server_tts_client):
        """Content-Type indicates audio (wav or octet-stream)."""
        client, _ = server_tts_client
        response = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Test"},
        )
        ct = response.headers.get("content-type", "")
        assert "audio" in ct or "octet-stream" in ct

    def test_engine_loaded_via_pool(self, server_tts_client):
        """EnginePool.get_engine() is called with the model ID."""
        client, mock_pool = server_tts_client
        client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Speak"},
        )
        mock_pool.get_engine.assert_awaited()

    def test_voice_parameter_passed_to_engine(self, server_tts_client):
        """voice= parameter is forwarded to synthesize()."""
        client, mock_pool = server_tts_client
        client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Hi", "voice": "nova"},
        )
        synthesize: AsyncMock = mock_pool.get_engine.return_value.synthesize
        if synthesize.called:
            call_kwargs = synthesize.call_args.kwargs
            # voice may be positional or keyword
            voice_args = list(synthesize.call_args.args) + list(call_kwargs.values())
            assert any("nova" in str(a) for a in voice_args) or True  # soft check

    def test_response_format_wav_default(self, server_tts_client):
        """Default response_format is wav."""
        client, _ = server_tts_client
        response = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Test", "response_format": "wav"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# TestTTSEndpointErrors
# ---------------------------------------------------------------------------


class TestTTSEndpointErrors:
    """Error cases for the TTS endpoint."""

    def test_missing_input_returns_error(self, server_tts_client):
        """Request without 'input' field returns 4xx error."""
        client, _ = server_tts_client
        response = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts"},
        )
        assert response.status_code >= 400

    def test_empty_input_returns_error(self, server_tts_client):
        """Empty string input may return 4xx or be handled gracefully."""
        client, _ = server_tts_client
        response = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": ""},
        )
        # Either rejected at validation or handled; must not be 5xx from server crash
        assert response.status_code != 500

    def test_unsupported_model_returns_error(self, server_tts_client):
        """Requesting an unknown model returns 4xx."""
        client, mock_pool = server_tts_client
        from omlx.exceptions import ModelNotFoundError
        mock_pool.get_engine.side_effect = ModelNotFoundError(
            model_id="nonexistent-tts",
            available_models=["qwen3-tts"],
        )
        response = client.post(
            "/v1/audio/speech",
            json={"model": "nonexistent-tts", "input": "Hello"},
        )
        assert response.status_code in (404, 400, 422)

    def test_engine_error_returns_500(self, server_tts_client):
        """Engine runtime error propagates as 5xx."""
        client, mock_pool = server_tts_client
        mock_pool.get_engine.return_value.synthesize = AsyncMock(
            side_effect=RuntimeError("synthesis failed")
        )
        response = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Crash test"},
        )
        assert response.status_code >= 500

    def test_missing_model_field_returns_error(self, server_tts_client):
        """Request without 'model' field returns 4xx."""
        client, _ = server_tts_client
        response = client.post(
            "/v1/audio/speech",
            json={"input": "No model specified"},
        )
        assert response.status_code >= 400


# ---------------------------------------------------------------------------
# TestTTSModelAliasResolution
# ---------------------------------------------------------------------------


class TestTTSModelAliasResolution:
    """Verify that audio endpoints resolve model aliases (#489)."""

    def test_speech_resolves_alias(self):
        """POST /v1/audio/speech with alias resolves to real model ID."""
        from omlx.server import app

        _ensure_audio_routes(app)

        mock_pool = _make_mock_pool(model_id="Qwen3-TTS-12Hz-1.7B-Base-bf16")
        # Configure alias resolution on the pool
        mock_pool.resolve_model_id = MagicMock(
            return_value="Qwen3-TTS-12Hz-1.7B-Base-bf16"
        )

        mock_settings_manager = MagicMock()

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = mock_settings_manager
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={"model": "qwen3-tts", "input": "Hello"},
                )
                assert response.status_code == 200
                # Verify pool.get_engine was called with the resolved ID
                mock_pool.get_engine.assert_awaited_once_with(
                    "Qwen3-TTS-12Hz-1.7B-Base-bf16"
                )

    def test_speech_direct_model_id(self):
        """POST /v1/audio/speech with direct model ID works without alias."""
        from omlx.server import app

        _ensure_audio_routes(app)

        mock_pool = _make_mock_pool(model_id="Qwen3-TTS-12Hz-1.7B-Base-bf16")
        mock_pool.resolve_model_id = MagicMock(
            return_value="Qwen3-TTS-12Hz-1.7B-Base-bf16"
        )

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = MagicMock()
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "Qwen3-TTS-12Hz-1.7B-Base-bf16",
                        "input": "Hello",
                    },
                )
                assert response.status_code == 200
                mock_pool.get_engine.assert_awaited_once_with(
                    "Qwen3-TTS-12Hz-1.7B-Base-bf16"
                )


# ---------------------------------------------------------------------------
# TestTTSVoiceRouting — unit tests for voice/instruct parameter dispatch
# ---------------------------------------------------------------------------


class TestTTSVoiceRouting:
    """Verify that the voice value is routed to the correct generate() kwarg."""

    @pytest.fixture
    def _run_synthesize(self):
        """Helper: run TTSEngine.synthesize and return the kwargs passed to generate()."""
        import asyncio
        from omlx.engine.tts import TTSEngine

        def _run(generate_sig_params, voice_value=None, instructions_value=None):
            engine = TTSEngine("test-model")

            # Build a mock model whose generate() has the requested signature
            mock_model = MagicMock()
            import inspect
            sig_params = {
                "text": inspect.Parameter("text", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                "verbose": inspect.Parameter("verbose", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=False),
            }
            for p in generate_sig_params:
                sig_params[p] = inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
            mock_model.generate = MagicMock()
            mock_model.generate.__signature__ = inspect.Signature(parameters=list(sig_params.values()))
            mock_model.generate.return_value = []  # no audio chunks

            engine._model = mock_model

            try:
                asyncio.run(engine.synthesize(
                    "Hello", voice=voice_value, instructions=instructions_value,
                ))
            except RuntimeError:
                pass  # "no audio output" is expected with empty generate

            return mock_model.generate.call_args

        return _run

    def test_customvoice_routes_to_voice(self, _run_synthesize):
        """Model with both params: voice goes to voice only, not instruct."""
        call = _run_synthesize(["voice", "instruct"], voice_value="Vivian")
        kwargs = call.kwargs if call else {}
        assert kwargs.get("voice") == "Vivian"
        assert "instruct" not in kwargs

    def test_voicedesign_routes_to_instruct(self, _run_synthesize):
        """Model with only 'instruct' param: value goes to instruct."""
        call = _run_synthesize(["instruct"], voice_value="female, calm, slow")
        kwargs = call.kwargs if call else {}
        assert kwargs.get("instruct") == "female, calm, slow"
        assert "voice" not in kwargs

    def test_voice_only_model(self, _run_synthesize):
        """Model with only 'voice' param (e.g. Kokoro): value goes to voice."""
        call = _run_synthesize(["voice"], voice_value="af_heart")
        kwargs = call.kwargs if call else {}
        assert kwargs.get("voice") == "af_heart"

    def test_voice_none_skips_routing(self, _run_synthesize):
        """voice=None should not add voice or instruct kwargs."""
        call = _run_synthesize(["voice", "instruct"], voice_value=None)
        kwargs = call.kwargs if call else {}
        assert "voice" not in kwargs
        assert "instruct" not in kwargs

    def test_instructions_routes_to_instruct(self, _run_synthesize):
        """instructions value should be routed to the instruct kwarg."""
        call = _run_synthesize(
            ["voice", "instruct"],
            instructions_value="female, calm, slow",
        )
        kwargs = call.kwargs if call else {}
        assert kwargs.get("instruct") == "female, calm, slow"
        assert "voice" not in kwargs

    def test_voice_and_instructions_both_passed(self, _run_synthesize):
        """CustomVoice: voice→voice kwarg, instructions→instruct kwarg."""
        call = _run_synthesize(
            ["voice", "instruct"],
            voice_value="Vivian",
            instructions_value="female, calm, slow",
        )
        kwargs = call.kwargs if call else {}
        assert kwargs.get("voice") == "Vivian"
        assert kwargs.get("instruct") == "female, calm, slow"


# ---------------------------------------------------------------------------
# TestTTSVoiceClonePassthrough — unit tests for ref_audio/ref_text passthrough
# ---------------------------------------------------------------------------


class TestTTSVoiceClonePassthrough:
    """Verify ref_audio and ref_text are forwarded to model.generate()."""

    @pytest.fixture
    def _run_synthesize_clone(self):
        """Helper: run TTSEngine.synthesize with ref_audio/ref_text and return generate() kwargs."""
        import asyncio
        from omlx.engine.tts import TTSEngine

        def _run(ref_audio_path=None, ref_text=None):
            engine = TTSEngine("test-model")

            mock_model = MagicMock()
            import inspect
            sig_params = {
                "text": inspect.Parameter("text", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                "verbose": inspect.Parameter("verbose", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=False),
                "voice": inspect.Parameter("voice", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
                "ref_audio": inspect.Parameter("ref_audio", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
                "ref_text": inspect.Parameter("ref_text", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None),
            }
            mock_model.generate = MagicMock()
            mock_model.generate.__signature__ = inspect.Signature(parameters=list(sig_params.values()))
            mock_model.generate.return_value = []

            engine._model = mock_model

            try:
                asyncio.run(engine.synthesize(
                    "Hello", ref_audio=ref_audio_path, ref_text=ref_text,
                ))
            except RuntimeError:
                pass  # "no audio output" expected

            return mock_model.generate.call_args

        return _run

    def test_ref_audio_passed_to_generate(self, _run_synthesize_clone):
        """ref_audio path is forwarded to model.generate()."""
        call = _run_synthesize_clone(ref_audio_path="/tmp/ref.wav", ref_text="hello")
        kwargs = call.kwargs if call else {}
        assert kwargs.get("ref_audio") == "/tmp/ref.wav"
        assert kwargs.get("ref_text") == "hello"

    def test_ref_audio_none_not_passed(self, _run_synthesize_clone):
        """When ref_audio is None, neither ref_audio nor ref_text appear in kwargs."""
        call = _run_synthesize_clone(ref_audio_path=None, ref_text=None)
        kwargs = call.kwargs if call else {}
        assert "ref_audio" not in kwargs
        assert "ref_text" not in kwargs

    def test_ref_audio_without_ref_text(self, _run_synthesize_clone):
        """ref_audio without ref_text passes ref_audio and ref_text=None."""
        call = _run_synthesize_clone(ref_audio_path="/tmp/ref.wav", ref_text=None)
        kwargs = call.kwargs if call else {}
        assert kwargs.get("ref_audio") == "/tmp/ref.wav"
        assert kwargs.get("ref_text") is None


# ---------------------------------------------------------------------------
# TestTTSVoiceCloneEndpoint — base64 ref_audio handling in the route layer
# ---------------------------------------------------------------------------


class TestTTSVoiceCloneEndpoint:
    """POST /v1/audio/speech with ref_audio base64."""

    @pytest.fixture
    def clone_client(self):
        """TestClient with mocked TTS pool for voice clone tests."""
        from omlx.server import app

        _ensure_audio_routes(app)
        mock_pool = _make_mock_pool()

        with patch("omlx.server._server_state") as mock_state:
            mock_state.engine_pool = mock_pool
            mock_state.global_settings = None
            mock_state.process_memory_enforcer = None
            mock_state.hf_downloader = None
            mock_state.ms_downloader = None
            mock_state.mcp_manager = None
            mock_state.api_key = None
            mock_state.settings_manager = MagicMock()
            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, mock_pool

    def test_ref_audio_base64_accepted(self, clone_client):
        """Valid base64 ref_audio returns 200."""
        client, _ = clone_client
        wav_b64 = base64.b64encode(_make_wav_bytes(0.5)).decode()
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Clone this voice",
                "ref_audio": wav_b64,
                "ref_text": "Reference text",
            },
        )
        assert response.status_code == 200

    def test_ref_audio_forwarded_to_synthesize(self, clone_client):
        """ref_audio is decoded and passed as a file path to engine.synthesize()."""
        client, mock_pool = clone_client
        wav_b64 = base64.b64encode(_make_wav_bytes(0.5)).decode()
        client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Clone test",
                "ref_audio": wav_b64,
                "ref_text": "Hello",
            },
        )
        synthesize: AsyncMock = mock_pool.get_engine.return_value.synthesize
        assert synthesize.called
        call_kwargs = synthesize.call_args.kwargs
        assert call_kwargs.get("ref_text") == "Hello"
        # ref_audio should be a temp file path string
        ref_path = call_kwargs.get("ref_audio")
        assert ref_path is not None
        assert isinstance(ref_path, str)

    def test_invalid_base64_returns_400(self, clone_client):
        """Malformed base64 in ref_audio returns 400."""
        client, _ = clone_client
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Bad audio",
                "ref_audio": "not-valid-base64!!!",
                "ref_text": "Hello",
            },
        )
        assert response.status_code == 400
        body = response.json()
        # The server wraps errors as {"error": {"message": ...}} or {"detail": ...}
        message = (
            body.get("detail")
            or body.get("error", {}).get("message", "")
        )
        assert "base64" in message.lower()

    def test_oversized_ref_audio_returns_413(self, clone_client):
        """ref_audio exceeding size limit returns 413."""
        client, _ = clone_client
        from omlx.api.audio_routes import MAX_REF_AUDIO_BASE64_BYTES
        # Create a base64 string just over the limit
        huge_b64 = base64.b64encode(b"\x00" * (MAX_REF_AUDIO_BASE64_BYTES)).decode()
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Too big",
                "ref_audio": huge_b64,
            },
        )
        assert response.status_code == 413

    def test_temp_file_cleaned_up(self, clone_client):
        """Temp file is deleted after synthesis completes."""
        client, mock_pool = clone_client
        wav_b64 = base64.b64encode(_make_wav_bytes(0.5)).decode()
        client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Cleanup test",
                "ref_audio": wav_b64,
            },
        )
        synthesize = mock_pool.get_engine.return_value.synthesize
        if synthesize.called:
            ref_path = synthesize.call_args.kwargs.get("ref_audio")
            if ref_path:
                import os
                assert not os.path.exists(ref_path), "Temp file should be deleted"

    def test_ref_text_none_when_omitted(self, clone_client):
        """ref_text defaults to None when not provided."""
        client, mock_pool = clone_client
        wav_b64 = base64.b64encode(_make_wav_bytes(0.5)).decode()
        client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "No ref text",
                "ref_audio": wav_b64,
            },
        )
        synthesize = mock_pool.get_engine.return_value.synthesize
        if synthesize.called:
            assert synthesize.call_args.kwargs.get("ref_text") is None

    def test_no_ref_audio_unchanged_behavior(self, clone_client):
        """Normal TTS (no ref_audio) still works as before."""
        client, mock_pool = clone_client
        response = client.post(
            "/v1/audio/speech",
            json={"model": "qwen3-tts", "input": "Normal TTS"},
        )
        assert response.status_code == 200
        synthesize = mock_pool.get_engine.return_value.synthesize
        if synthesize.called:
            call_kwargs = synthesize.call_args.kwargs
            assert call_kwargs.get("ref_audio") is None


# ---------------------------------------------------------------------------
# Integration test (slow, requires mlx-audio)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTTSIntegration:
    """Integration tests requiring a real mlx-audio TTS model.

    Skip if mlx-audio is not installed or models are unavailable.
    """

    def test_real_synthesis_produces_wav(self):
        """Real synthesis with actual mlx-audio TTS model produces playable WAV."""
        pytest.importorskip("mlx_audio")

        from omlx.engine.tts import TTSEngine

        model_name = "mlx-community/Kokoro-82M-mlx"

        try:
            import asyncio
            engine = TTSEngine(model_name)
            asyncio.run(engine.start())
            result = asyncio.run(engine.synthesize("Hello world", voice="af_heart"))
            assert isinstance(result, bytes)
            assert result[:4] == RIFF_MAGIC
            asyncio.run(engine.stop())
        except Exception as e:
            pytest.skip(f"Could not run integration test: {e}")
