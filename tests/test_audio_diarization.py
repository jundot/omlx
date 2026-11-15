# SPDX-License-Identifier: Apache-2.0
"""Contract tests for timed speaker attribution on the transcription route."""

import inspect
import io
import json
import wave
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.api import audio_routes
from omlx.engine.stt import STTEngine


@pytest.fixture
def transcribe(monkeypatch):
    pytest.importorskip("mlx_audio")
    app = FastAPI()
    app.include_router(audio_routes.router)
    engine = object.__new__(STTEngine)
    inference = AsyncMock()
    engine.transcribe = inference
    pool = type("Pool", (), {"get_engine": AsyncMock(return_value=engine)})()
    monkeypatch.setattr(audio_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(audio_routes, "_resolve_model", lambda model: model)
    monkeypatch.setattr(audio_routes, "_get_settings_manager", lambda: None)
    monkeypatch.setattr(audio_routes, "_record_audio_request", lambda model: None)

    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(16_000 * 2))

    with TestClient(app) as client:

        def send(payload, backend=None, **options):
            inference.reset_mock()
            inference.return_value = payload
            engine._model = backend
            return client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "fixture",
                    "response_format": "diarized_json",
                    "chunking_strategy": "auto",
                    **options,
                },
                files={"file": ("test.wav", audio.getvalue(), "audio/wav")},
            )

        yield send, inference


def test_vibevoice_returns_plain_text_and_timed_speaker_labels(transcribe):
    from mlx_audio.stt.models.vibevoice_asr.vibevoice_asr import Model

    send, inference = transcribe
    backend = type("VibeVoice", (), {"__module__": Model.__module__})()
    raw = json.dumps(
        [
            {"Start": 0, "End": 0.4, "Speaker": 0, "Content": "Bonjour."},
            {"Start": 0.4, "End": 0.8, "Speaker": 1, "Content": "Salut."},
        ]
    )
    segments = Model.parse_transcription(backend, raw)
    response = send({"text": raw, "segments": segments, "duration": 8}, backend)

    assert response.status_code == 200
    assert response.json() == {
        "task": "transcribe",
        "text": "Bonjour. Salut.",
        "duration": 1.0,
        "usage": {"type": "duration", "seconds": 1},
        "segments": [
            {
                "type": "transcript.text.segment",
                "id": "seg_0",
                "speaker": "0",
                "start": 0,
                "end": 0.4,
                "text": "Bonjour.",
            },
            {
                "type": "transcript.text.segment",
                "id": "seg_1",
                "speaker": "1",
                "start": 0.4,
                "end": 0.8,
                "text": "Salut.",
            },
        ],
    }
    assert inference.call_args.kwargs == {"language": None}


def test_nemotron_requests_native_alignment_only_for_diarized_json(transcribe):
    from mlx_audio.stt.models.nemotron_transcribe import Model

    assert "word_timestamps" in inspect.signature(Model.generate).parameters
    send, inference = transcribe
    backend = type("Nemotron", (), {"__module__": Model.__module__})()
    payload = {
        "text": "<spk:0> Bonjour.",
        "segments": [
            {"speaker": 0, "start": 0, "end": 0.5, "text": "Bonjour."},
        ],
    }
    response = send(payload, backend, language="fr")
    assert response.status_code == 200
    assert response.json()["text"] == "Bonjour."
    assert inference.call_args.kwargs == {
        "language": "fr",
        "task": "saa",
        "word_timestamps": True,
    }

    assert (
        send(payload, backend, response_format="verbose_json").json()["segments"]
        == payload["segments"]
    )
    assert inference.call_args.kwargs == {"language": None}


@pytest.mark.parametrize(
    ("segments", "status"),
    [
        ([{"start": 0, "end": 0.5, "text": "Bonjour."}], 422),
        ([{"speaker": 0, "start": 0.5, "end": 0, "text": "Bonjour."}], 502),
    ],
)
def test_diarization_rejects_incomplete_segments(transcribe, segments, status):
    send, _ = transcribe
    response = send({"text": "Bonjour.", "segments": segments})
    assert response.status_code == status


@pytest.mark.parametrize("options", [{"stream": "true"}, {"chunking_strategy": "vad"}])
def test_diarization_rejects_unsupported_stream_or_chunking(transcribe, options):
    send, inference = transcribe
    assert send({}, **options).status_code == 400
    inference.assert_not_awaited()
