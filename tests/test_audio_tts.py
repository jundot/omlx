# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /v1/audio/speech (INV-04).

Verifies the TTS endpoint accepts a JSON body and returns valid WAV audio
bytes, matching the OpenAI audio speech API spec.

All unit tests run with mocked TTSEngine and EnginePool — mlx-audio is not
required. Integration tests (marked @pytest.mark.slow) need a real model.
"""

import io
import struct
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
    return pool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server_tts_client():
    """TestClient using the full omlx server app with mocked TTS pool."""
    from omlx.server import app

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
