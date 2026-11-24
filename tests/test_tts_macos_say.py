# SPDX-License-Identifier: Apache-2.0
"""macos-say TTS. Every system call is faked, so this runs on any OS."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from omlx.exceptions import ModelNotFoundError
from omlx.settings import GlobalSettings
from omlx.utils import macos_say

_ROSTER = """Daniel (English (UK)) en_GB    # Hello! My name is Daniel.
Zuzana              cs_CZ    # Ahoj! Já jsem Zuzana.
Zuzana (Premium)    cs_CZ    # Ahoj! Já jsem Zuzana.
"""
_PICKED = "com.apple.voice.premium.cs-CZ.Zuzana"


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(macos_say.SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * 1200)
    return buf.getvalue()


@pytest.fixture
def say(monkeypatch):
    from omlx.api.audio_routes import router as audio_router
    from omlx.server import app

    if "/v1/audio/speech" not in {getattr(r, "path", "") for r in app.routes}:
        app.include_router(audio_router)

    calls: list = []
    osascript = SimpleNamespace(returncode=0, stdout=_PICKED + "\n", stderr="")

    def run(args, **kwargs):
        calls.append((args, kwargs.get("input")))
        if args[0] == "osascript":
            return osascript
        if args[1:] == ["-v", "?"]:
            return SimpleNamespace(returncode=0, stdout=_ROSTER, stderr="")
        Path(args[args.index("-o") + 1]).write_bytes(_wav())
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(macos_say.subprocess, "run", run)
    gs = GlobalSettings()
    gs.integrations.macos_say_enabled = True
    pool = MagicMock()
    pool.get_engine = AsyncMock(side_effect=ModelNotFoundError("macos-say", []))
    pool.get_status.return_value = {"models": []}
    with patch("omlx.server._server_state") as state:
        state.engine_pool, state.global_settings = pool, gs
        state.api_key, state.settings_manager = None, None
        yield SimpleNamespace(
            client=TestClient(app, raise_server_exceptions=False),
            settings=gs,
            calls=calls,
            osascript=osascript,
        )


def _speak(s, **body):
    body = {"model": "macos-say", "input": "Dobrý [[den]]", **body}
    return s.client.post("/v1/audio/speech", json=body)


def test_off_by_default(say):
    say.settings.integrations.macos_say_enabled = False
    assert GlobalSettings().integrations.macos_say_enabled is False
    assert _speak(say).status_code == 404
    with patch("omlx.server.get_max_context_window", return_value=None):
        ids = [m["id"] for m in say.client.get("/v1/models").json()["data"]]
        assert "macos-say" not in ids
        say.settings.integrations.macos_say_enabled = True
        data = say.client.get("/v1/models").json()["data"]
    assert {"id": "macos-say", "model_type": "audio_tts"}.items() <= next(
        m for m in data if m["id"] == "macos-say"
    ).items()


@pytest.mark.parametrize(
    "body,argv_tail", [({"voice": "ash"}, []), ({"language": "-ebeep"}, ["-ebeep"])]
)
def test_voice_pick_reads_text_from_stdin(say, body, argv_tail):
    r = _speak(say, **body)
    assert r.status_code == 200, r.text
    assert r.content[:4] == b"RIFF"
    (pick_argv, pick_in), (say_argv, say_in) = say.calls
    assert pick_argv[:2] == ["osascript", "-e"]
    assert pick_argv[3:] == ["--", *argv_tail]
    assert (pick_in, say_in) == ("Dobrý [[den]]", "Dobrý [ [den]]")
    assert not any("Dobrý" in a for a in pick_argv + say_argv)
    assert say_argv[:3] == ["say", "-v", _PICKED] and "-f" in say_argv


def test_voice_pick_failure_is_500(say):
    say.osascript.returncode, say.osascript.stderr = 1, "boom"
    assert _speak(say).status_code == 500
    assert len(say.calls) == 1


@pytest.mark.parametrize("voice", ["Daniel", "zuzana (premium)", _PICKED])
def test_named_voices_skip_the_pick(say, voice):
    assert _speak(say, voice=voice).status_code == 200
    assert all(args[0] == "say" for args, _ in say.calls)
    assert say.calls[-1][0][:3] == ["say", "-v", voice]


@pytest.mark.parametrize(
    "body", [{"voice": "Bogus"}, {"input": "a" * (macos_say.MAX_INPUT_CHARS + 1)}]
)
def test_bad_requests_are_400(say, body):
    assert _speak(say, **body).status_code == 400
    assert not any("-o" in args for args, _ in say.calls)
