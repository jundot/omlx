# SPDX-License-Identifier: Apache-2.0
"""Tests for GET /v1/audio/voices — list built-in TTS speaker names.

Reads static model metadata only (a Kokoro-style ``voices/`` directory or
the Qwen3-TTS CustomVoice speaker table in config.json), so no engine is
loaded and mlx-audio is not required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def voices_client(tmp_path):
    """TestClient for the audio router with a pool entry backed by tmp_path."""
    from fastapi import FastAPI

    from omlx.api.audio_routes import router

    app = FastAPI()
    app.include_router(router)

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    pool = MagicMock()
    pool.get_entry = MagicMock(
        return_value=MagicMock(model_path=str(model_dir))
    )

    with (
        patch("omlx.api.audio_routes._get_engine_pool", return_value=pool),
        patch(
            "omlx.api.audio_routes._resolve_model",
            side_effect=lambda m: m,
        ),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        yield client, pool, model_dir


class TestVoicesEndpoint:
    def test_missing_model_param_returns_400(self, voices_client):
        client, _, _ = voices_client
        response = client.get("/v1/audio/voices")
        assert response.status_code == 400

    def test_unknown_model_returns_404(self, voices_client):
        client, pool, _ = voices_client
        pool.get_entry.return_value = None
        response = client.get("/v1/audio/voices", params={"model": "nope"})
        assert response.status_code == 404

    def test_kokoro_style_voices_dir(self, voices_client):
        """Voice names come from the voices/ dir, deduped across formats."""
        client, _, model_dir = voices_client
        voices_dir = model_dir / "voices"
        voices_dir.mkdir()
        # Kokoro repos ship each voice in both formats; stems must dedupe.
        for name in ("af_heart", "af_alloy", "zm_yunxi"):
            (voices_dir / f"{name}.safetensors").write_bytes(b"0")
            (voices_dir / f"{name}.pt").write_bytes(b"0")
        (voices_dir / "README.md").write_text("not a voice")

        response = client.get("/v1/audio/voices", params={"model": "kokoro"})

        assert response.status_code == 200
        assert response.json() == {
            "model": "kokoro",
            "voices": ["af_alloy", "af_heart", "zm_yunxi"],
        }

    def test_customvoice_spk_id_from_config(self, voices_client):
        """Qwen3-TTS CustomVoice speakers come from talker_config.spk_id."""
        client, _, model_dir = voices_client
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "qwen3_tts",
            "talker_config": {
                "spk_id": {"vivian": 0, "aiden": 1, "serena": 2},
            },
        }))

        response = client.get(
            "/v1/audio/voices", params={"model": "qwen3-tts-customvoice"}
        )

        assert response.status_code == 200
        assert response.json()["voices"] == ["aiden", "serena", "vivian"]

    def test_model_without_named_voices_returns_empty_list(self, voices_client):
        """Voice-cloning base models legitimately have no named speakers."""
        client, _, model_dir = voices_client
        (model_dir / "config.json").write_text(json.dumps({
            "model_type": "vibevoice",
        }))

        response = client.get("/v1/audio/voices", params={"model": "vibevoice"})

        assert response.status_code == 200
        assert response.json()["voices"] == []

    def test_missing_config_returns_empty_list(self, voices_client):
        """No voices/ dir and no config.json still answers 200 with []."""
        client, _, _ = voices_client
        response = client.get("/v1/audio/voices", params={"model": "bare"})
        assert response.status_code == 200
        assert response.json()["voices"] == []
