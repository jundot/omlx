# SPDX-License-Identifier: Apache-2.0
"""Tests for the macos-say TTS model (omlx/utils/macos_say.py).

Every system call is faked, so the suite runs on any OS.
"""

from __future__ import annotations

import io
import json
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from omlx.settings import GlobalSettings, IntegrationSettings
from omlx.utils import macos_say


def _wav(seconds: float = 0.05, rate: int = macos_say.SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("cs_CZ", "cs-CZ"),
        (" EN-gb ", "en-GB"),
        ("zh-Hans-CN", "zh-Hans-CN"),
        ("de", "de"),
        ("", ""),
        (None, ""),
        ("not a tag", ""),
        ("e", ""),
    ],
)
def test_normalize_language(raw, expected):
    assert macos_say.normalize_language(raw) == expected


@pytest.mark.parametrize("voice", [None, "", "default", "ash", "Alloy", "marin"])
def test_openai_and_default_voices_mean_auto(voice):
    assert macos_say.explicit_voice(voice) == ""


def test_named_voice_is_kept():
    assert macos_say.explicit_voice(" Zuzana (Premium) ") == "Zuzana (Premium)"


_ROSTER = """Albert              en_US    # Hello! My name is Albert.
Good News           en_US    # Hello! My name is Good News.
Zuzana              cs_CZ    # Ahoj! Já jsem Zuzana.
Zuzana (Premium)    cs_CZ    # Ahoj! Já jsem Zuzana.
Shelley (Czech (Czechia)) cs_CZ    # Ahoj! Já jsem Shelley.
"""


def test_voice_check_reads_the_roster(monkeypatch):
    run = MagicMock(return_value=SimpleNamespace(stdout=_ROSTER, returncode=0))
    monkeypatch.setattr(macos_say.subprocess, "run", run)
    assert macos_say.voice_is_installed("zuzana (premium)")
    assert macos_say.voice_is_installed("Good News")
    assert macos_say.voice_is_installed("Shelley (Czech (Czechia))")
    assert not macos_say.voice_is_installed("Bogus")
    assert run.call_args.args[0] == ["say", "-v", "?"]


def test_identifiers_skip_the_roster(monkeypatch):
    run = MagicMock(side_effect=AssertionError("no roster call expected"))
    monkeypatch.setattr(macos_say.subprocess, "run", run)
    assert macos_say.voice_is_installed("com.apple.voice.premium.cs-CZ.Zuzana")


def test_pick_voice_does_nothing_without_a_reason(monkeypatch):
    run = MagicMock(side_effect=AssertionError("no osascript call expected"))
    monkeypatch.setattr(macos_say.subprocess, "run", run)
    assert macos_say.pick_voice("Hello", "", "", detect=False) == ""


def test_pick_voice_passes_arguments_and_reads_identifier(monkeypatch):
    monkeypatch.setattr(macos_say.shutil, "which", lambda _: "/usr/bin/osascript")
    run = MagicMock(
        return_value=SimpleNamespace(
            returncode=0, stdout="cs\tcom.apple.voice.premium.cs-CZ.Zuzana\n", stderr=""
        )
    )
    monkeypatch.setattr(macos_say.subprocess, "run", run)
    got = macos_say.pick_voice("Dobrý den", "", "en-GB", detect=True)
    assert got == "com.apple.voice.premium.cs-CZ.Zuzana"
    argv = run.call_args.args[0]
    assert argv == ["osascript", "-", "Dobrý den", "", "en-GB", "1"]
    assert "NLLanguageRecognizer" in run.call_args.kwargs["input"]
    assert "voiceWithLanguage" in run.call_args.kwargs["input"]


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        SimpleNamespace(returncode=0, stdout="xx\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="", stderr=""),
    ],
)
def test_pick_voice_failure_falls_back_to_plain_say(monkeypatch, result):
    monkeypatch.setattr(macos_say.shutil, "which", lambda _: "/usr/bin/osascript")
    monkeypatch.setattr(macos_say.subprocess, "run", MagicMock(return_value=result))
    assert macos_say.pick_voice("Hello there", "", "", detect=True) == ""


def test_pick_voice_survives_timeout(monkeypatch):
    monkeypatch.setattr(macos_say.shutil, "which", lambda _: "/usr/bin/osascript")
    monkeypatch.setattr(
        macos_say.subprocess,
        "run",
        MagicMock(side_effect=subprocess.TimeoutExpired("osascript", 15)),
    )
    assert macos_say.pick_voice("Hello there", "de", "", detect=False) == ""


def _fake_say(captured: list):
    def run(args, **kwargs):
        captured.append(args)
        out = Path(args[args.index("-o") + 1])
        out.write_bytes(_wav())
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def test_synthesize_argv(monkeypatch):
    calls: list = []
    monkeypatch.setattr(macos_say.subprocess, "run", _fake_say(calls))
    audio = macos_say.synthesize("Ahoj", "Zuzana", speed=2.0)
    assert audio[:4] == b"RIFF"
    args = calls[0]
    assert args[:3] == ["say", "-v", "Zuzana"]
    assert args[args.index("-r") + 1] == "350"
    assert "--data-format=LEI16@24000" in args


def test_synthesize_plain_say_and_speed_clamp(monkeypatch):
    calls: list = []
    monkeypatch.setattr(macos_say.subprocess, "run", _fake_say(calls))
    macos_say.synthesize("Hi", "", speed=1.0)
    macos_say.synthesize("Hi", "", speed=40)
    assert "-v" not in calls[0] and "-r" not in calls[0]
    assert calls[1][calls[1].index("-r") + 1] == "700"


def test_synthesize_reports_say_errors(monkeypatch):
    monkeypatch.setattr(
        macos_say.subprocess,
        "run",
        MagicMock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="no")),
    )
    with pytest.raises(RuntimeError, match="say failed"):
        macos_say.synthesize("Hi")


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------


def test_settings_defaults_and_round_trip():
    s = IntegrationSettings()
    assert (s.macos_say_enabled, s.macos_say_autodetect, s.macos_say_language) == (
        True,
        True,
        "",
    )
    s.macos_say_enabled = False
    s.macos_say_language = "cs-CZ"
    back = IntegrationSettings.from_dict(s.to_dict())
    assert back.macos_say_enabled is False
    assert back.macos_say_language == "cs-CZ"
    assert IntegrationSettings.from_dict({}).macos_say_autodetect is True


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


@pytest.fixture
def say_client(monkeypatch):
    from omlx.api.audio_routes import router as audio_router
    from omlx.server import app

    if "/v1/audio/speech" not in {getattr(r, "path", "") for r in app.routes}:
        app.include_router(audio_router)

    gs = GlobalSettings()
    picks: list = []
    renders: list = []

    def pick(text, language, default_language, detect):
        picks.append((text, language, default_language, detect))
        return "com.apple.voice.compact.cs-CZ.Zuzana"

    def synth(text, voice="", speed=None):
        renders.append((text, voice, speed))
        return _wav()

    monkeypatch.setattr(macos_say, "available", lambda: True)
    monkeypatch.setattr(macos_say, "pick_voice", pick)
    monkeypatch.setattr(macos_say, "synthesize", synth)
    monkeypatch.setattr(
        macos_say, "voice_is_installed", lambda v: v.lower() == "zuzana"
    )
    pool = MagicMock()
    pool.get_status.return_value = {"models": []}
    with patch("omlx.server._server_state") as state:
        state.engine_pool = pool
        state.global_settings = gs
        state.api_key = None
        state.settings_manager = None
        client = TestClient(app, raise_server_exceptions=False)
        yield SimpleNamespace(
            client=client, settings=gs, picks=picks, renders=renders, pool=pool
        )


def _speak(c, **body):
    return c.client.post(
        "/v1/audio/speech", json={"model": "macos-say", "input": "Ahoj", **body}
    )


def test_speech_autodetects_with_openai_voice(say_client):
    r = _speak(say_client, voice="ash")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"
    assert say_client.picks == [("Ahoj", "", "", True)]
    assert say_client.renders[0][1] == "com.apple.voice.compact.cs-CZ.Zuzana"
    say_client.pool.get_engine.assert_not_called()


def test_speech_request_language_and_settings_reach_the_pick(say_client):
    say_client.settings.integrations.macos_say_autodetect = False
    say_client.settings.integrations.macos_say_language = "en_GB"
    assert _speak(say_client, language="de_de").status_code == 200
    assert say_client.picks == [("Ahoj", "de-DE", "en-GB", False)]


def test_speech_named_voice_skips_the_pick(say_client):
    assert _speak(say_client, voice="Zuzana", speed=1.5).status_code == 200
    assert say_client.picks == []
    assert say_client.renders == [("Ahoj", "Zuzana", 1.5)]


def test_speech_unknown_voice_is_400(say_client):
    r = _speak(say_client, voice="Bogus")
    assert r.status_code == 400
    assert "Bogus" in r.json()["error"]["message"]


def test_speech_mp3_is_transcoded(say_client):
    r = _speak(say_client, response_format="mp3")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.content[:4] != b"RIFF"


def test_speech_input_limit(say_client):
    r = _speak(say_client, input="a" * (macos_say.MAX_INPUT_CHARS + 1))
    assert r.status_code == 400
    assert say_client.renders == []


def test_speech_disabled_is_400(say_client):
    say_client.settings.integrations.macos_say_enabled = False
    assert _speak(say_client).status_code == 400


def test_speech_off_macos_is_400(say_client, monkeypatch):
    monkeypatch.setattr(macos_say, "available", lambda: False)
    r = _speak(say_client)
    assert r.status_code == 400
    assert "macOS" in r.json()["error"]["message"]


def test_models_list_carries_macos_say(say_client):
    with patch("omlx.server.get_max_context_window", return_value=None):
        data = say_client.client.get("/v1/models").json()["data"]
        assert {"id": "macos-say", "model_type": "audio_tts"}.items() <= next(
            m for m in data if m["id"] == "macos-say"
        ).items()
        say_client.settings.integrations.macos_say_enabled = False
        data = say_client.client.get("/v1/models").json()["data"]
        assert all(m["id"] != "macos-say" for m in data)


# --------------------------------------------------------------------------
# admin settings API
# --------------------------------------------------------------------------


def _post_settings(monkeypatch, gs, **fields):
    import asyncio

    from omlx.admin import routes

    monkeypatch.setattr(routes, "_get_global_settings", lambda: gs)
    monkeypatch.setattr(gs, "save", lambda *a, **k: None)
    return asyncio.run(
        routes.update_global_settings(
            request=routes.GlobalSettingsRequest(**fields), is_admin=True
        )
    )


def test_admin_post_normalizes_and_keeps_other_fields(monkeypatch):
    from fastapi import HTTPException

    gs = GlobalSettings()
    _post_settings(
        monkeypatch, gs, macos_say_language="cs_cz", macos_say_autodetect=False
    )
    assert gs.integrations.macos_say_language == "cs-CZ"
    assert gs.integrations.macos_say_autodetect is False
    # Saving an unrelated integration setting leaves ours alone.
    _post_settings(monkeypatch, gs, integrations_openclaw_tools_profile="full")
    assert gs.integrations.macos_say_language == "cs-CZ"
    assert gs.integrations.macos_say_autodetect is False
    _post_settings(monkeypatch, gs, macos_say_language="")
    assert gs.integrations.macos_say_language == ""
    with pytest.raises(HTTPException) as err:
        _post_settings(monkeypatch, gs, macos_say_language="not a tag")
    assert err.value.status_code == 400


# --------------------------------------------------------------------------
# dashboard card
# --------------------------------------------------------------------------

_WEB = Path(__file__).resolve().parents[1] / "apps" / "omlx-web" / "omlx_web"


def test_card_keys_exist_in_every_locale():
    html = (_WEB / "templates" / "dashboard" / "_settings.html").read_text()
    keys = {
        k
        for k in json.loads((_WEB / "i18n" / "en.json").read_text())
        if k.startswith("settings.integrations.macos_say.")
    }
    assert len(keys) == 7
    for key in keys:
        assert f"t('{key}')" in html
    for path in (_WEB / "i18n").glob("*.json"):
        assert keys <= set(json.loads(path.read_text())), path.name


def test_dashboard_saves_every_field():
    js = (_WEB / "static" / "js" / "dashboard.js").read_text()
    for field in ("macos_say_enabled", "macos_say_autodetect", "macos_say_language"):
        assert f"{field}: this.globalSettings.integrations.{field}" in js
