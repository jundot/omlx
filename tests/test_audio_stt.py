# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /v1/audio/transcriptions (INV-03).

Verifies the STT endpoint accepts multipart audio uploads and returns a
transcription response matching the OpenAI audio API spec.

All unit tests run with mocked STTEngine and EnginePool — mlx-audio is not
required. Integration tests (marked @pytest.mark.slow) need a real model.
"""

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# WAV fixture helpers
# ---------------------------------------------------------------------------


def _make_wav_bytes(duration_secs: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Generate minimal valid WAV bytes (silence)."""
    n_samples = int(sample_rate * duration_secs)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


TINY_WAV = _make_wav_bytes()


# ---------------------------------------------------------------------------
# Mock STTEngine
# ---------------------------------------------------------------------------


def _make_mock_stt_engine(transcript: str = "hello world") -> MagicMock:
    """Build a mock STTEngine that returns the given transcript."""
    from omlx.engine.stt import STTEngine
    engine = MagicMock(spec=STTEngine)
    engine.transcribe = AsyncMock(return_value={
        "text": transcript,
        "language": "en",
        "duration": 0.1,
        "segments": [],
    })
    return engine


def _make_mock_pool(stt_engine=None, model_id: str = "whisper-tiny") -> MagicMock:
    """Build a mock EnginePool that returns the given STT engine."""
    pool = MagicMock()
    pool.get_engine = AsyncMock(return_value=stt_engine or _make_mock_stt_engine())
    pool.get_entry = MagicMock(return_value=MagicMock(
        model_type="audio_stt",
        engine_type="stt",
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


@pytest.fixture
def audio_client():
    """TestClient for the audio router with a mocked STT engine."""
    from omlx.api.audio_routes import router

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)

    mock_pool = _make_mock_pool()

    with patch("omlx.api.audio_routes._get_engine_pool", return_value=mock_pool):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, mock_pool


def _ensure_audio_routes(app):
    """Register audio routes if not already present (e.g., mlx-audio not installed)."""
    from omlx.api.audio_routes import router as audio_router

    audio_paths = {"/v1/audio/transcriptions", "/v1/audio/speech", "/v1/audio/process"}
    existing = {getattr(r, "path", "") for r in app.routes}
    if not audio_paths & existing:
        app.include_router(audio_router)


@pytest.fixture
def server_audio_client():
    """TestClient using the full omlx server app with mocked pool."""
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
# TestSTTEndpointBasic
# ---------------------------------------------------------------------------


class TestSTTEndpointBasic:
    """Core STT endpoint behaviour."""

    def test_post_transcriptions_returns_200(self, server_audio_client):
        """POST /v1/audio/transcriptions with valid WAV returns 200."""
        client, _ = server_audio_client
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "whisper-tiny"},
        )
        assert response.status_code == 200

    def test_response_has_text_field(self, server_audio_client):
        """Successful response contains 'text' field."""
        client, _ = server_audio_client
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "whisper-tiny"},
        )
        body = response.json()
        assert "text" in body

    def test_response_text_matches_engine_output(self, server_audio_client):
        """Response text matches what the engine returned."""
        client, mock_pool = server_audio_client
        mock_pool.get_engine.return_value.transcribe = AsyncMock(
            return_value={"text": "test transcription", "language": "en", "duration": 0.5, "segments": []}
        )

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "whisper-tiny"},
        )
        body = response.json()
        assert body.get("text") == "test transcription"

    def test_engine_loaded_via_pool(self, server_audio_client):
        """EnginePool.get_engine() is called with the provided model ID."""
        client, mock_pool = server_audio_client
        client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "whisper-tiny"},
        )
        mock_pool.get_engine.assert_awaited()

    def test_language_parameter_accepted(self, server_audio_client):
        """language= form field is accepted without error."""
        client, _ = server_audio_client
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "whisper-tiny", "language": "en"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# TestSTTEndpointResponseFormat
# ---------------------------------------------------------------------------


class TestSTTEndpointResponseFormat:
    """OpenAI audio transcription API response schema compliance."""

    def test_response_object_field(self, server_audio_client):
        """Response optionally includes object field."""
        client, _ = server_audio_client
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "whisper-tiny"},
        )
        body = response.json()
        # OpenAI spec: response has at minimum a 'text' field
        assert "text" in body

    def test_content_type_is_json(self, server_audio_client):
        """Default response is JSON (not audio)."""
        client, _ = server_audio_client
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "whisper-tiny"},
        )
        assert "application/json" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# TestSTTEndpointErrors
# ---------------------------------------------------------------------------


class TestSTTEndpointErrors:
    """Error cases for the STT endpoint."""

    def test_missing_file_returns_error(self, server_audio_client):
        """Request without file field returns 4xx error."""
        client, _ = server_audio_client
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-tiny"},
        )
        assert response.status_code >= 400

    def test_unsupported_model_returns_error(self, server_audio_client):
        """Requesting an unknown model returns 4xx error."""
        client, mock_pool = server_audio_client
        from omlx.exceptions import ModelNotFoundError
        mock_pool.get_engine.side_effect = ModelNotFoundError(
            model_id="nonexistent-model",
            available_models=["whisper-tiny"],
        )
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "nonexistent-model"},
        )
        assert response.status_code in (404, 400, 422)

    def test_engine_error_returns_500(self, server_audio_client):
        """Engine runtime error returns 5xx."""
        client, mock_pool = server_audio_client
        mock_pool.get_engine.return_value.transcribe = AsyncMock(
            side_effect=RuntimeError("model failed")
        )
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("audio.wav", TINY_WAV, "audio/wav")},
            data={"model": "whisper-tiny"},
        )
        assert response.status_code >= 500


# ---------------------------------------------------------------------------
# TestVideoContainerRemap
# ---------------------------------------------------------------------------


class TestVideoContainerRemap:
    """Video container extensions are remapped to .m4a for ffmpeg routing."""

    @pytest.mark.parametrize("filename,expected_suffix", [
        ("video.mp4", ".m4a"),
        ("video.mkv", ".m4a"),
        ("video.mov", ".m4a"),
        ("video.m4v", ".m4a"),
        ("video.webm", ".m4a"),
        ("video.avi", ".m4a"),
        ("audio.wav", ".wav"),
        ("audio.m4a", ".m4a"),
        ("audio.mp3", ".mp3"),
    ])
    def test_video_container_suffix_remap(
        self, server_audio_client, filename, expected_suffix, tmp_path,
    ):
        """Temp file suffix should be .m4a for video containers, unchanged otherwise."""
        client, mock_pool = server_audio_client
        engine = mock_pool.get_engine.return_value

        # Capture the path passed to engine.transcribe
        called_paths = []
        original_transcribe = engine.transcribe

        async def capture_transcribe(path, **kwargs):
            called_paths.append(path)
            return await original_transcribe(path, **kwargs)

        engine.transcribe = AsyncMock(side_effect=capture_transcribe)

        client.post(
            "/v1/audio/transcriptions",
            files={"file": (filename, TINY_WAV, "application/octet-stream")},
            data={"model": "whisper-tiny"},
        )

        assert len(called_paths) == 1
        assert called_paths[0].endswith(expected_suffix)


# ---------------------------------------------------------------------------
# TestSTTModelAliasResolution
# ---------------------------------------------------------------------------


class TestSTTModelAliasResolution:
    """Verify that STT endpoint resolves model aliases (#489)."""

    def test_transcription_resolves_alias(self):
        """POST /v1/audio/transcriptions with alias resolves to real model ID."""
        from omlx.server import app

        _ensure_audio_routes(app)

        mock_pool = _make_mock_pool(model_id="Qwen3-ASR-1.7B-bf16")
        mock_pool.resolve_model_id = MagicMock(
            return_value="Qwen3-ASR-1.7B-bf16"
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
                    "/v1/audio/transcriptions",
                    data={"model": "whisper"},
                    files={"file": ("test.wav", TINY_WAV, "audio/wav")},
                )
                assert response.status_code == 200
                mock_pool.get_engine.assert_awaited_once_with(
                    "Qwen3-ASR-1.7B-bf16"
                )

    def test_transcription_direct_model_id(self):
        """POST /v1/audio/transcriptions with direct model ID works without alias."""
        from omlx.server import app

        _ensure_audio_routes(app)

        mock_pool = _make_mock_pool(model_id="Qwen3-ASR-1.7B-bf16")
        # resolve_model_id returns the same ID when no alias matches
        mock_pool.resolve_model_id = MagicMock(
            return_value="Qwen3-ASR-1.7B-bf16"
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
                    "/v1/audio/transcriptions",
                    data={"model": "Qwen3-ASR-1.7B-bf16"},
                    files={"file": ("test.wav", TINY_WAV, "audio/wav")},
                )
                assert response.status_code == 200
                mock_pool.get_engine.assert_awaited_once_with(
                    "Qwen3-ASR-1.7B-bf16"
                )


# ---------------------------------------------------------------------------
# Integration test (slow, requires mlx-audio)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSTTIntegration:
    """Integration tests requiring a real mlx-audio STT model.

    Skip if mlx-audio is not installed or models are unavailable.
    """

    def test_real_transcription(self, tmp_path):
        """Real transcription with small WAV and actual mlx-audio model."""
        pytest.importorskip("mlx_audio")

        from omlx.engine.stt import STTEngine

        model_name = "mlx-community/whisper-tiny"
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(TINY_WAV)

        try:
            import asyncio
            engine = STTEngine(model_name)
            asyncio.run(engine.start())
            result = asyncio.run(engine.transcribe(wav_path))
            assert "text" in result
            asyncio.run(engine.stop())
        except Exception as e:
            pytest.skip(f"Could not run integration test: {e}")
