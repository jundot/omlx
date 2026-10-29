# SPDX-License-Identifier: Apache-2.0
"""macOS `say` as a TTS engine (omlx/utils/macos_say.py) and its settings.

The probes shell out to macOS-only tooling (`say -v ?`, osascript,
`defaults read`), so every test here feeds canned probe output instead: the
suite runs on any machine, and a Linux CI still proves the parsing, the
routing chain and the endpoint contract. Nothing here calls a system binary.
"""

import asyncio
import json

import pytest
from fastapi import HTTPException

from omlx.utils import macos_say as ms

SAY_ROSTER = """\
Zuzana               cs_CZ    # Ahoj! Já jsem Zuzana.
Iveta                cs_CZ    # Dobrý den, já jsem Iveta.
Samantha (English (US)) en_US # Hello! My name is Samantha.
Daniel (English (UK)) en_GB   # Hello, my name is Daniel.
junk line without a locale
"""

AV_ROWS = """\
Zuzana\tcs-CZ\tcom.apple.voice.premium.cs-CZ.Zuzana\tAVSpeechSynthesisVoice Quality: Premium\t0
Zuzana\tcs-CZ\tcom.apple.voice.super-compact.cs-CZ.Zuzana\tAVSpeechSynthesisVoice Quality: Default\t1
Laura\tsk-SK\tcom.apple.voice.compact.sk-SK.Laura\tAVSpeechSynthesisVoice Quality: Default\t1
Samantha\ten-US\tcom.apple.voice.compact.en-US.Samantha\tAVSpeechSynthesisVoice Quality: Default\t1
"""

# `defaults read com.apple.Accessibility SpokenContentDefaultVoiceSelectionsByLanguage`
SPKEN = """(\n    cs,\n    {\n        boundLanguage = cs;\n        voiceId = "com.apple.voice.premium.cs-CZ.Zuzana";\n    }\n)"""

INSTALLED = [
    {
        "name": "Zuzana (Premium)",
        "locale": "cs-CZ",
        "quality": "premium",
        "sample": "Ahoj",
        "identifier": "com.apple.voice.premium.cs-CZ.Zuzana",
        "installed": True,
        "default": False,
        "downloaded": True,
    },
    {
        "name": "Zuzana",
        "locale": "cs-CZ",
        "quality": "standard",
        "sample": "Ahoj",
        "identifier": "com.apple.voice.super-compact.cs-CZ.Zuzana",
        "installed": True,
        "default": True,
        "downloaded": False,
    },
    {
        "name": "Laura",
        "locale": "sk-SK",
        "quality": "standard",
        "sample": "Ahoj",
        "identifier": "com.apple.voice.compact.sk-SK.Laura",
        "installed": True,
        "default": True,
        "downloaded": False,
    },
    {
        "name": "Samantha",
        "locale": "en-US",
        "quality": "standard",
        "sample": "Hello",
        "identifier": "com.apple.voice.compact.en-US.Samantha",
        "installed": True,
        "default": True,
        "downloaded": False,
    },
    {
        "name": "Not On This Mac",
        "locale": "de-DE",
        "quality": "premium",
        "sample": "",
        "identifier": "",
        "installed": False,
        "default": False,
        "downloaded": False,
    },
]


@pytest.fixture
def mac(monkeypatch):
    """A Mac with INSTALLED as its roster and nothing selected in System
    Settings - tests add picks from here."""
    monkeypatch.setattr(ms, "available", lambda: True)
    monkeypatch.setattr(ms, "list_voices", lambda force=False: list(INSTALLED))
    monkeypatch.setattr(ms, "spoken_selections", lambda force=False: {})
    monkeypatch.setattr(ms, "selected_voice_identifier", lambda force=False: "")
    monkeypatch.setattr(ms, "system_language", lambda force=False: "en-US")
    monkeypatch.setattr(ms, "detect_language", lambda text: "")
    return ms


# ---------------------------------------------------------------- parsing --


def test_parse_voices_keeps_names_with_spaces_and_parentheses():
    rows = ms.parse_voices(SAY_ROSTER)
    assert [r["name"] for r in rows] == [
        "Zuzana",
        "Iveta",
        "Samantha (English (US))",
        "Daniel (English (UK))",
    ]
    assert rows[0]["locale"] == "cs-CZ"  # say writes cs_CZ; one spelling out
    assert rows[0]["sample"].startswith("Ahoj")
    assert rows[2]["quality"] == "standard"


def test_parse_voices_reads_quality_tiers_from_the_name():
    rows = ms.parse_voices("Ava (Premium) en_US # Hi\nAva (Enhanced) en-US # Hi\n")
    assert [r["quality"] for r in rows] == ["premium", "enhanced"]


def test_parse_av_reads_identifier_and_flag():
    rows = ms.parse_av(AV_ROWS)
    assert rows[0]["identifier"] == "com.apple.voice.premium.cs-CZ.Zuzana"
    assert rows[0]["quality"] == "premium"
    assert rows[1]["quality"] == "standard"  # 'Default' is the plain tier
    assert rows[1]["default"] is True
    assert ms.parse_av("garbage") == []


def test_parse_av_accepts_probe_output_without_the_default_column():
    row = ms.parse_av("X\ten-US\tid.x\tAVSpeechSynthesisVoice Quality: Enhanced")[0]
    assert row["quality"] == "enhanced" and row["default"] is False


def test_parse_catalog_reads_footprints_and_multiple_languages():
    doc = {
        "Assets": [
            {"Name": "Zuzana", "Footprint": "premium", "Languages": ["cs-CZ"]},
            {"Name": "Iveta", "Footprint": "compact", "Language": "cs-CZ"},
            {"Name": "Junk"},
        ]
    }
    rows = ms.parse_catalog(doc, "footprint")
    assert {(r["name"], r["quality"]) for r in rows} == {
        ("Zuzana", "premium"),
        ("Iveta", "standard"),
    }


def test_parse_spoken_selections_reads_the_alternating_array():
    assert ms._parse_spoken_selections(SPKEN) == {
        "cs": "com.apple.voice.premium.cs-CZ.Zuzana"
    }
    assert ms._parse_spoken_selections("") == {}
    assert ms._parse_spoken_selections("{ nope = 1; }") == {}


# ------------------------------------------------------------- locale math --


@pytest.mark.parametrize(
    "requested,have,expected",
    [
        ("cs-CZ", "cs-CZ", 2),
        ("cs", "cs-CZ", 1),
        ("cs-CZ", "cs", 1),
        ("cs-CZ", "cs-SK", 0),
        ("en-US", "en-GB", 0),
        ("", "cs-CZ", 0),
    ],
)
def test_locale_score(requested, have, expected):
    assert ms.locale_score(requested, have) == expected


def test_split_model_token_and_locale_test():
    assert ms.split_model_token("say") == (True, "")
    assert ms.split_model_token("SAY:Zuzana") == (True, "Zuzana")
    assert ms.split_model_token("say:cs-CZ") == (True, "cs-CZ")
    assert ms.split_model_token("say:") == (True, "")
    assert ms.split_model_token("kokoro") == (False, "")
    assert ms.split_model_token("") == (False, "")
    assert ms.looks_like_locale("zh-Hans") and not ms.looks_like_locale("Zuzana")


# ---------------------------------------------------------------- routing --


def test_language_prefers_the_configured_voice(mac):
    picked = ms.language_voice(
        "cs-CZ", {"cs-CZ": "com.apple.voice.super-compact.cs-CZ.Zuzana"}
    )
    assert picked["identifier"] == "com.apple.voice.super-compact.cs-CZ.Zuzana"


def test_configuration_for_another_language_never_answers(mac):
    assert ms.route(language="sk-SK", voices={"cs-CZ": "Laura"}) == (
        "com.apple.voice.compact.sk-SK.Laura"
    )


def test_stale_configuration_falls_back_to_the_system_chain(mac, monkeypatch):
    monkeypatch.setattr(
        ms,
        "system_selected_voice",
        lambda lang: next(v for v in INSTALLED if v["locale"] == "cs-CZ"),
    )
    got = ms.route(language="cs-CZ", voices={"cs-CZ": "Voice Deleted Yesterday"})
    assert got == "com.apple.voice.premium.cs-CZ.Zuzana"


def test_system_settings_pick_beats_the_factory_flag(mac, monkeypatch):
    # cs has a real Spoken Content selection (a compact voice, i.e. a LOWER
    # tier than the installed Premium row); sk only has the AV default flag.
    monkeypatch.setattr(
        ms,
        "spoken_selections",
        lambda force=False: {"cs": "com.apple.voice.super-compact.cs-CZ.Zuzana"},
    )
    assert ms.system_selected_voice("cs-CZ")["identifier"] == (
        "com.apple.voice.super-compact.cs-CZ.Zuzana"
    )
    assert ms.route(language="cs-CZ") == (
        "com.apple.voice.super-compact.cs-CZ.Zuzana"
    )
    # 'sk' was never picked: nothing answers it, so its own chain goes on.
    assert ms.system_selected_voice("sk-SK") is None
    assert ms.route(language="sk-SK") == "com.apple.voice.compact.sk-SK.Laura"


def test_region_specific_spoken_pick_does_not_claim_the_whole_family(mac, monkeypatch):
    monkeypatch.setattr(
        ms, "spoken_selections", lambda force=False: {"en-US": "com.apple.voice.compact.en-US.Samantha"}
    )
    assert ms.system_selected_voice("en-US") is not None
    # An 'en-US' selection never speaks for en-GB: 'en-GB' has nothing here.
    assert ms.system_selected_voice("en-GB") is None


def test_selected_system_voice_answers_only_its_own_language(mac, monkeypatch):
    monkeypatch.setattr(
        ms,
        "selected_row",
        lambda: next(v for v in INSTALLED if v["locale"] == "en-US"),
    )
    assert ms.route(language="en-US") == "com.apple.voice.compact.en-US.Samantha"
    # cs-CZ has voices installed but no selection of its own: the language's
    # own flagged default answers (step 4), before its best tier - and it is
    # never the English voice of a different language.
    assert ms.route(language="cs-CZ") == (
        "com.apple.voice.super-compact.cs-CZ.Zuzana"
    )


def test_best_tier_answers_only_when_the_language_has_no_flagged_row(mac, monkeypatch):
    rows = [dict(v, default=False) for v in INSTALLED if v["locale"] == "cs-CZ"]
    rest = [v for v in INSTALLED if v["locale"] != "cs-CZ"]
    monkeypatch.setattr(ms, "list_voices", lambda force=False: rows + rest)
    assert ms.route(language="cs-CZ") == "com.apple.voice.premium.cs-CZ.Zuzana"


def test_language_with_nothing_installed_fails_instead_of_guessing(mac):
    with pytest.raises(LookupError):
        ms.route(language="de-DE")
    with pytest.raises(LookupError):
        ms.route(voice="Definitely Not Here")


def test_language_less_request_asks_macos_and_then_routes(mac, monkeypatch):
    monkeypatch.setattr(ms, "detect_language", lambda text: "cs")
    assert ms.route(text="Ahoj, jak se máš?") == (
        "com.apple.voice.super-compact.cs-CZ.Zuzana"
    )


def test_unanswered_detection_falls_back_to_the_macs_default(mac, monkeypatch):
    # Detection says 'de' and nothing German is installed: a guess must never
    # be WORSE than no guess, so routing keeps the Mac's own default voice
    # instead of failing or speaking German with a Czech voice.
    monkeypatch.setattr(ms, "detect_language", lambda text: "de")
    assert ms.route(text="Hallo") == "com.apple.voice.compact.en-US.Samantha"


def test_route_returns_empty_when_nothing_can_be_named(mac, monkeypatch):
    monkeypatch.setattr(ms, "default_voice", lambda: None)
    assert ms.route(text="hello") == ""


# --- names that are not unique: macOS reuses them across languages -------

SHARED = list(INSTALLED) + [
    {
        "name": "Rico",
        "locale": "it-IT",
        "quality": "standard",
        "sample": "Ciao",
        "identifier": "com.apple.eloquence.it-IT.Rico",
        "installed": True,
        "default": False,
        "downloaded": False,
    },
    {
        "name": "Rico",
        "locale": "pt-BR",
        "quality": "standard",
        "sample": "Ola",
        "identifier": "com.apple.eloquence.pt-BR.Rico",
        "installed": True,
        "default": False,
        "downloaded": False,
    },
    {
        "name": "Zuzana (Premium)",
        "locale": "cs-CZ",
        "quality": "premium",
        "sample": "Ahoj",
        "identifier": "com.apple.voice.premium.cs-CZ.Zuzana",
        "installed": True,
        "default": False,
        "downloaded": True,
    },
]


@pytest.fixture
def shared_names(monkeypatch, mac):
    """The same Mac, but with a name that exists in two languages and a
    display name carrying a space and parentheses - both real on macOS."""
    monkeypatch.setattr(mac, "list_voices", lambda force=False: list(SHARED))
    return mac


def test_address_parser_accepts_the_language_on_either_side(shared_names):
    assert ms.parse_voice_language("Rico") == ("Rico", "")
    assert ms.parse_voice_language("Rico:pt-BR") == ("Rico", "pt-BR")
    assert ms.parse_voice_language("pt-BR:Rico") == ("Rico", "pt-BR")
    # identifiers use dots, so they must never be torn apart
    assert ms.parse_voice_language("com.apple.eloquence.pt-BR.Rico") == (
        "com.apple.eloquence.pt-BR.Rico",
        "",
    )


def test_a_shared_name_without_a_language_still_answers(shared_names):
    """Naming a voice alone is a real request: it speaks, in the language the
    roster lists first. Nothing is invented here, which is exactly why the
    card and the example tell the operator to pin the language."""
    assert ms.resolve("Rico")["locale"] == "it-IT"


@pytest.mark.parametrize(
    "address,locale",
    [
        ("Rico:pt-BR", "pt-BR"),
        ("pt-BR:Rico", "pt-BR"),
        ("Rico:it-IT", "it-IT"),
    ],
)
def test_a_shared_name_pinned_to_a_language_answers_that_row(
    shared_names, address, locale
):
    assert ms.resolve(address)["locale"] == locale


def test_a_shared_name_pinned_to_a_language_it_lacks_resolves_to_nothing(
    shared_names,
):
    assert ms.resolve("Rico:de-DE") is None
    assert ms.name_languages("Rico") == ["it-IT", "pt-BR"]


def test_route_names_the_languages_a_pinned_one_does_not_cover(shared_names):
    with pytest.raises(LookupError) as exc:
        ms.route("Rico:de-DE", None, "Hallo", {})
    text = str(exc.value)
    assert "it-IT" in text and "pt-BR" in text
    assert "de-DE" in text and "identifier" in text


def test_a_named_voice_outlives_a_language_field_but_not_its_own(shared_names):
    # Pinned in the address: both halves are honoured. Named beside a language:
    # the voice wins, which is the long-standing contract and what `say -v`
    # does, so a Slovak voice asked for with Czech reads the Czech text.
    assert ms.route("Rico:pt-BR", None, "Ola", {}) == "com.apple.eloquence.pt-BR.Rico"
    assert ms.route("Rico", "pt-BR", "Ola", {}) == "com.apple.eloquence.it-IT.Rico"


NAMED_ROSTER = """Albert              en_US    # Hello! My name is Albert.
Rico                 it_IT    # Ciao! Mi chiamo Rico.
Rico                 pt_BR    # Ola! Meu nome e Rico.
Zuzana               cs_CZ    # Ahoj! Jsem Zuzana.
"""


def test_a_pinned_language_survives_a_row_whose_identifier_is_just_its_name(
    monkeypatch,
):
    """`say -v ?` alone yields rows whose identifier IS the display name - the
    non-unique string. Without care a pinned language then matched that name
    first and returned whichever row happened to be listed, quietly answering
    'Rico:pt-BR' with the Italian voice."""
    rows = ms.parse_voices(NAMED_ROSTER)   # identifiers are the bare names
    monkeypatch.setattr(ms, "available", lambda: True)
    monkeypatch.setattr(ms, "list_voices", lambda force=False: list(rows))
    assert [r["identifier"] for r in rows if r["name"] == "Rico"] == ["Rico", "Rico"]
    assert ms.resolve("Rico")["locale"] == "it-IT", "no language: first row, as ever"
    assert ms.resolve("Rico:pt-BR")["locale"] == "pt-BR"
    assert ms.resolve("Rico:it-IT")["locale"] == "it-IT"
    assert ms.resolve("Rico:de-DE") is None
    with pytest.raises(LookupError, match="pt-BR"):
        ms.route("Rico:de-DE", None, "Hallo", {})


def test_display_names_with_spaces_and_parentheses_stay_addressable(shared_names):
    """'(Premium)' is part of how macOS lists the voice, and the model field is
    JSON, so a space needs no encoding - pinned so nobody 'cleans' it away."""
    assert ms.resolve("Zuzana (Premium)")["quality"] == "premium"
    assert ms.route("Zuzana (Premium)", None, "Ahoj", {}) == (
        "com.apple.voice.premium.cs-CZ.Zuzana"
    )


def test_explicit_voice_wins_over_language_and_configuration(mac):
    got = ms.route(
        voice="Laura",
        language="cs-CZ",
        voices={"cs-CZ": "com.apple.voice.premium.cs-CZ.Zuzana"},
    )
    assert got == "com.apple.voice.compact.sk-SK.Laura"


def test_locale_named_as_voice_routes_like_a_language(mac):
    assert ms.route(voice="sk") == "com.apple.voice.compact.sk-SK.Laura"


def test_language_options_labels_the_detectable_default_only(mac):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ms,
        "system_selected_voice",
        lambda lang: next(
            (v for v in INSTALLED if v["locale"] == lang), None
        )
        if lang == "cs-CZ"
        else None,
    )
    options = ms.language_options({"en-US": "com.apple.voice.compact.en-US.Samantha"})
    by_locale = {o["locale"]: o for o in options}
    cs = by_locale["cs-CZ"]
    assert cs["default"] == {
        "name": "Zuzana (Premium)",
        "locale": "cs-CZ",
        "preference": True,
    }
    # sk-SK has no user pick: the label must stay unnamed, not claim a voice.
    assert by_locale["sk-SK"]["default"] == {
        "name": "",
        "locale": "",
        "preference": False,
    }
    # a downloadable-but-absent voice is not offered
    assert "de-DE" not in by_locale
    assert by_locale["en-US"]["configured"] == (
        "com.apple.voice.compact.en-US.Samantha"
    )
    assert [v["quality"] for v in cs["voices"]] == ["premium", "standard"]
    monkeypatch.undo()


def test_effective_rows_exclude_downloadable_voices(mac):
    assert all(v["installed"] for v in ms.installed_rows())


# ----------------------------------------------------------- persistence --


def test_integration_settings_round_trip_the_speech_fields():
    from omlx.settings import IntegrationSettings

    s = IntegrationSettings(
        say_enabled=False,
        say_voices={"cs-CZ": "com.apple.voice.premium.cs-CZ.Zuzana"},
        say_long_input="reject",
    )
    stored = json.loads(json.dumps(s.to_dict()))
    back = IntegrationSettings.from_dict(stored)
    assert back.say_enabled is False
    assert back.say_voices == {"cs-CZ": "com.apple.voice.premium.cs-CZ.Zuzana"}
    assert back.say_long_input == "reject"


def test_speech_settings_default_to_enabled_with_no_overrides():
    from omlx.settings import IntegrationSettings

    s = IntegrationSettings.from_dict({})
    assert s.say_enabled is True
    assert s.say_voices == {}
    assert s.say_long_input == "speak"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "speak"),
        ("", "speak"),
        ("speak", "speak"),
        ("REJECT", "reject"),
        ("  reject ", "reject"),
        ("split", "speak"),  # not a policy this engine offers
        (7, "speak"),
    ],
)
def test_clean_say_long_input_falls_back_to_speaking(raw, expected):
    from omlx.settings import _clean_say_long_input

    assert _clean_say_long_input(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, {}),
        ("nonsense", {}),
        ([{"cs": "x"}], {}),
        ({"cs_CZ": "Voice"}, {"cs-CZ": "Voice"}),
        ({"cs-CZ": "", "en-US": "Samantha"}, {"en-US": "Samantha"}),
        ({"cs-CZ": 7}, {}),
    ],
)
def test_clean_say_voices_rejects_junk(raw, expected):
    from omlx.settings import _clean_say_voices

    assert _clean_say_voices(raw) == expected


def test_clean_voices_drops_addresses_this_mac_cannot_speak(mac):
    assert ms.clean_voices(
        {
            "cs-CZ": "com.apple.voice.premium.cs-CZ.Zuzana",
            "sk-SK": "Voice That Is Gone",
            "": "whatever",
        }
    ) == {"cs-CZ": "com.apple.voice.premium.cs-CZ.Zuzana"}


# ---------------------------------------------------------- synthesis argv --


class FakeProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_synth_addresses_say_through_argv_with_a_temp_file(mac, monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(list(cmd))
        # `say` writes the file named by -o
        out = cmd[cmd.index("-o") + 1]
        with open(out, "wb") as fh:
            fh.write(b"RIFF" + b"\0" * 500)
        return FakeProcess(stdout="ok")

    monkeypatch.setattr(ms.subprocess, "run", fake_run)
    monkeypatch.setattr(ms.tempfile, "TemporaryDirectory", _dir_context(tmp_path))

    wav = ms.synth_bytes(
        "Ahoj, jak se máš?",
        language="cs-CZ",
        voices={"cs-CZ": "com.apple.voice.premium.cs-CZ.Zuzana"},
        speed=1.5,
    )
    assert wav.startswith(b"RIFF")
    cmd = calls[0]
    assert cmd[0] == "say"
    assert cmd[cmd.index("-v") + 1] == "com.apple.voice.premium.cs-CZ.Zuzana"
    assert cmd[cmd.index("-r") + 1] == str(round(175 * 1.5))
    assert cmd[cmd.index("-f") + 1].endswith("in.txt")
    assert "--data-format=LEI16@24000" in cmd
    assert not any("&&" in part or ";" in part or " " in part for part in cmd[1:2])


def test_synth_drops_the_voice_flag_when_macos_should_choose(mac, monkeypatch, tmp_path):
    # Nothing nameable: no selection, no system language, no flagged row. The
    # utterance still speaks - with the voice `say` itself picks.
    monkeypatch.setattr(ms, "default_voice", lambda: None)
    monkeypatch.setattr(ms, "system_language", lambda force=False: "")
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(list(cmd))
        with open(cmd[cmd.index("-o") + 1], "wb") as fh:
            fh.write(b"RIFF" + b"\0" * 500)
        return FakeProcess()

    monkeypatch.setattr(ms.subprocess, "run", fake_run)
    monkeypatch.setattr(ms.tempfile, "TemporaryDirectory", _dir_context(tmp_path))

    ms.synth_bytes("Hello there, friend of mine")
    assert "-v" not in calls[0]


def test_synth_reports_a_failure_instead_of_returning_silence(mac, monkeypatch, tmp_path):
    def fake_run(cmd, capture_output, text, timeout):
        return FakeProcess(returncode=1, stderr="unknown voice")

    monkeypatch.setattr(ms.subprocess, "run", fake_run)
    monkeypatch.setattr(ms.tempfile, "TemporaryDirectory", _dir_context(tmp_path))
    with pytest.raises(RuntimeError, match="unknown voice"):
        ms.synth_bytes("Hello", voice="Laura")


def _capture_say_run(ms, monkeypatch, tmp_path, calls):
    def fake_run(cmd, capture_output, text, timeout):
        with open(cmd[cmd.index("-f") + 1], encoding="utf-8") as fh:
            calls.append(len(fh.read()))
        with open(cmd[cmd.index("-o") + 1], "wb") as fh:
            fh.write(b"RIFF" + b"\0" * 500)
        return FakeProcess()

    monkeypatch.setattr(ms.subprocess, "run", fake_run)
    monkeypatch.setattr(ms.tempfile, "TemporaryDirectory", _dir_context(tmp_path))


def test_long_text_is_said_whole_and_in_one_call(mac, monkeypatch, tmp_path):
    # Past the OpenAI limit the operator's policy decides, and the default
    # says everything. One `say` call renders the whole utterance: measured on
    # a Mac, 8.3k characters (7.5 minutes of audio) take 7 s of wall time,
    # while splitting the same text costs ~0.9 s of CPU per extra call and
    # returns the identical bytes. Nothing is dropped on the floor either -
    # silently truncating a reply is worse than either policy.
    calls = []
    _capture_say_run(ms, monkeypatch, tmp_path, calls)
    # 200k characters cannot fit the DEFAULT 256 MB ceiling even at the fast
    # bound, so this raises the ceiling to test the character gate itself:
    # under a generous ceiling the whole utterance still speaks in one call.
    monkeypatch.setenv(ms.MAX_AUDIO_ENV, "512")
    ms.synth_bytes("a" * ms.MAX_SAY_CHARS)
    assert calls == [ms.MAX_SAY_CHARS]


def test_reject_policy_refuses_before_spawning_anything(mac, monkeypatch, tmp_path):
    calls = []
    _capture_say_run(ms, monkeypatch, tmp_path, calls)
    with pytest.raises(ValueError, match="4096"):
        ms.synth_bytes("a" * (ms.OPENAI_INPUT_CHARS + 1), long_input="reject")
    assert calls == []
    # Text inside the limit speaks normally under the same policy.
    ms.synth_bytes("a" * ms.OPENAI_INPUT_CHARS, long_input="reject")
    assert calls == [ms.OPENAI_INPUT_CHARS]


@pytest.mark.parametrize("policy", ms.LONG_INPUT_POLICIES)
def test_both_policies_refuse_past_the_ceiling(mac, policy):
    with pytest.raises(ValueError, match=str(ms.MAX_SAY_CHARS)):
        ms.synth_bytes("a" * (ms.MAX_SAY_CHARS + 1), long_input=policy)


@pytest.mark.parametrize(
    "length,policy,expected",
    [
        (100, "speak", None),
        (ms.OPENAI_INPUT_CHARS, "reject", None),
        (ms.OPENAI_INPUT_CHARS + 1, "speak", None),
        (ms.OPENAI_INPUT_CHARS + 1, "junk", None),  # junk reads as the default
        (ms.OPENAI_INPUT_CHARS + 1, "reject", "4096"),
        (ms.MAX_SAY_CHARS + 1, "speak", str(ms.MAX_SAY_CHARS)),
    ],
)
def test_long_input_policy_matrix(length, policy, expected):
    refused = ms.long_input_error(length, policy)
    assert (expected in refused) if expected else refused is None


def test_empty_text_is_a_value_error_not_silence(mac):
    with pytest.raises(ValueError):
        ms.synth_bytes("   ")


def test_speed_junk_keeps_the_engine_default(mac):
    assert ms._speed_wpm("nonsense") is None
    assert ms._speed_wpm(0) is None
    assert ms._speed_wpm(None) is None
    assert ms._speed_wpm(1) == "175"
    assert ms._speed_wpm(99) == "450"  # clamped: no chipmunk requests


def _dir_context(tmp_path):
    import contextlib

    @contextlib.contextmanager
    def _ctx(*args, **kwargs):
        yield str(tmp_path)

    return _ctx


# ------------------------------------------------------------------ probes --


def test_probes_degrade_to_empty_answers_off_macos(monkeypatch):
    monkeypatch.setattr(ms, "available", lambda: False)
    monkeypatch.setattr(ms, "_run_osascript", lambda *a, **k: "")
    assert ms.list_voices(force=True) == []
    assert ms.spoken_selections(force=True) == {}
    assert ms.selected_voice_identifier(force=True) == ""
    assert ms.route(text="hello") == ""


def test_run_osascript_swallows_a_failing_probe(monkeypatch):
    monkeypatch.setattr(ms.shutil, "which", lambda name: "/usr/bin/osascript")

    def boom(*args, **kwargs):
        raise OSError("no osascript")

    monkeypatch.setattr(ms.subprocess, "run", boom)
    assert ms._run_osascript("anything") == ""


def test_detect_language_rejects_low_confidence_and_und(monkeypatch):
    monkeypatch.setattr(ms, "available", lambda: True)
    monkeypatch.setattr(ms, "_run_osascript", lambda script, arg=None, timeout=30: "pl\t29")
    assert ms.detect_language("ok") == ""
    monkeypatch.setattr(ms, "_run_osascript", lambda script, arg=None, timeout=30: "und\t55")
    assert ms.detect_language("asdf") == ""
    monkeypatch.setattr(ms, "_run_osascript", lambda script, arg=None, timeout=30: "zh-Hans\t99")
    assert ms.detect_language("你好，今天过得怎么样？") == "zh"
    monkeypatch.setattr(ms, "_run_osascript", lambda script, arg=None, timeout=30: "cs\t99")
    assert ms.detect_language("Ahoj") == "cs"
    assert ms.detect_language("no") == ""  # below the minimum length


# ---------------------------------------------------------------- endpoint --


def test_say_model_token_never_reaches_the_engine_pool():
    from omlx.api import audio_routes as ar

    assert ar._say_engine_request("say") == (True, "")
    assert ar._say_engine_request("say:Zuzana") == (True, "Zuzana")
    assert ar._say_engine_request("Kokoro") == (False, "")


def test_registered_model_named_say_keeps_its_name(monkeypatch):
    from omlx.api import audio_routes as ar
    from omlx import server as server_mod

    class Pool:
        def get_model_ids(self):
            return ["Kokoro", "say"]

    monkeypatch.setattr(server_mod._server_state, "engine_pool", Pool(), raising=False)
    assert ar._say_engine_request("say") == (False, "")
    assert ar._say_engine_request("say:Zuzana") == (True, "Zuzana")


def _speech_request(**kwargs):
    from omlx.api.audio_models import AudioSpeechRequest

    base = {"model": "say", "input": "Hello there, this is a test."}
    base.update(kwargs)
    return AudioSpeechRequest(**base)


def _mini_wav(text, frames_per_char=1, rate=24000):
    """A real WAV the size of its text, so a test can measure the stitch."""
    import io as _io
    import wave as _wave

    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x01\x02" * max(1, len(text) * frames_per_char))
    return buf.getvalue()


@pytest.fixture
def say_endpoint(monkeypatch):
    """The endpoint with `say` replaced by a deterministic WAV maker that
    records every segment, so the split and the stitch are observed rather
    than trusted. ``frames_per_char`` scales the audio without scaling the
    text, which is how the ceiling gets tested without rendering an hour."""
    from omlx.api import audio_routes as ar
    from omlx.utils import macos_say as engine

    monkeypatch.setattr(engine, "available", lambda: True)
    monkeypatch.setattr(engine, "resolve", lambda name: None)
    monkeypatch.setattr(
        ar,
        "_macos_say_settings",
        lambda: (True, {"cs-CZ": "Zuzana"}, "speak", 1200),
    )
    calls = {"segments": [], "frames_per_char": 1}

    def fake_route(voice, language, text, voices):
        calls["voice"] = voice
        calls["language"] = language
        calls["voices"] = voices
        # routing must see the WHOLE utterance: sentence-by-sentence detection
        # is how one Czech answer ends up in three different voices
        calls["routed_for"] = len(text)
        return voice or ""

    def fake_synth(text, address, speed=None):
        calls["segments"].append(text)
        calls["address"] = address
        calls["speed"] = speed
        return _mini_wav(text, calls["frames_per_char"])

    monkeypatch.setattr(engine, "route", fake_route)
    monkeypatch.setattr(engine, "synth_wav", fake_synth)

    # A spool is only deleted by the response's background task, and a
    # hand-called endpoint never runs one: without this check the suite would
    # litter the temp directory a few MB per test and nobody would notice for
    # a hundred commits. Anything left behind is reported and removed.
    import glob
    import os
    import tempfile

    spools = os.path.join(tempfile.gettempdir(), "omlx-say-*")
    before = set(glob.glob(spools))
    yield ar, calls
    leaked = sorted(set(glob.glob(spools)) - before)
    for path in leaked:
        try:
            os.unlink(path)
        except OSError:
            pass
    assert not leaked, f"{len(leaked)} spool file(s) never cleaned up: " \
        f"{[os.path.basename(p) for p in leaked]}"


def _drain_file(response):
    """Bytes a FileResponse would stream, then its own cleanup. The background
    task only runs inside ASGI, so a direct call must delete the spool and the
    test must prove it gets deleted - a leaked temp file per request is how a
    Mac quietly runs out of space."""
    import os as _os

    task = response.background
    assert task is not None, "a spooled reply must clean itself up after sending"
    with open(response.path, "rb") as fh:
        data = fh.read()
    task.func(*task.args, **task.kwargs)
    assert not _os.path.exists(response.path), "spool left behind after sending"
    return data


def _use(monkeypatch, ar, chunk=1200, policy="speak"):
    """Point the endpoint at one set of operator settings. The audio ceiling
    is not among them any more - it is the engine's fixed guardrail, and a
    test that wants a small one patches macos_say.DEFAULT_MAX_AUDIO_MB."""
    monkeypatch.setattr(
        ar, "_macos_say_settings", lambda: (True, {}, policy, chunk)
    )


def _off(monkeypatch, ar):
    """Same, with the engine switched off."""
    monkeypatch.setattr(
        ar, "_macos_say_settings", lambda: (False, {}, "speak", 1200)
    )


def _ask(ar, fmt="wav", **kwargs):
    """One request through the macOS speech path."""
    return asyncio.run(ar._speech_via_macos_say(_speech_request(**kwargs), "", fmt))


def _wav_frames(payload):
    import io as _io
    import wave as _wave

    with _wave.open(_io.BytesIO(payload), "rb") as wf:
        return wf.getnframes(), wf.getframerate(), wf.getnchannels()


def test_speech_serves_the_macos_engine(monkeypatch, say_endpoint):
    ar, calls = say_endpoint
    response = asyncio.run(ar._speech_via_macos_say(_speech_request(), "", "wav"))
    data = _drain_file(response)
    assert data.startswith(b"RIFF")
    assert response.media_type == "audio/wav"
    assert calls["voices"] == {"cs-CZ": "Zuzana"}
    assert len(calls["segments"]) == 1, "a short reply is one call, as it was"


def test_the_long_input_policy_is_applied_before_rendering(
    monkeypatch, say_endpoint
):
    """The policy is checked once, on the whole utterance, in the router: it
    cannot travel to the engine any more because the engine now sees segments,
    and a segment is never the thing the operator meant to allow or refuse."""
    ar, calls = say_endpoint
    long_text = "word " * 1000  # 5000 characters, past OpenAI's limit
    _use(monkeypatch, ar, policy="speak")
    _drain_file(_ask(ar, input=long_text))
    assert len(calls["segments"]) > 1, "speak means say it all, in pieces"

    _use(monkeypatch, ar, policy="reject")
    with pytest.raises(HTTPException) as exc:
        _ask(ar, input=long_text)
    assert exc.value.status_code == 400
    assert "4096" in str(exc.value.detail)
    calls["segments"].clear()
    with pytest.raises(HTTPException):
        _ask(ar, input=long_text)
    assert calls["segments"] == [], "a refusal must not render anything first"


def test_the_engine_is_advertised_only_when_it_can_answer(monkeypatch):
    """`say` never occupies a pool slot, so the pool cannot name it. Strict
    clients validate the model id against GET /v1/models, which means the list
    has to carry it - and carry it only when a request would actually work."""
    from omlx.api import audio_routes as ar
    from omlx.utils import macos_say as engine

    monkeypatch.setattr(engine, "available", lambda: True)
    _use(monkeypatch, ar)
    assert ar.say_model_visible() is True
    _off(monkeypatch, ar)
    assert ar.say_model_visible() is False, "a switched-off engine is not offered"
    monkeypatch.setattr(engine, "available", lambda: False)
    _use(monkeypatch, ar)
    assert ar.say_model_visible() is False, "Linux has no say to offer"


def test_models_endpoint_lists_the_say_row(monkeypatch):
    import json as _json

    from omlx import server
    from omlx.api import audio_routes as ar

    async def ask():
        return await server.list_models(_=True)   # the auth dependency is moot here

    def listed():
        body = _json.loads(asyncio.run(ask()).body)
        return [m["id"] for m in body["data"]]

    monkeypatch.setattr(ar, "say_model_visible", lambda: True)
    ids = listed()
    assert "say" in ids

    monkeypatch.setattr(ar, "say_model_visible", lambda: False)
    ids = listed()
    assert "say" not in ids
    # the row is addressable as-is: the endpoint routes exactly this token
    assert ar._say_engine_request("say") == (True, "")


def test_speech_answers_400_for_a_language_the_mac_cannot_speak(say_endpoint):
    ar, calls = say_endpoint

    def raise_lookup(*args, **kwargs):
        raise LookupError("no installed macOS voice for language 'de-DE'")

    from omlx.utils import macos_say as engine

    engine.route = raise_lookup
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ar._speech_via_macos_say(_speech_request(language="de-DE"), "", "wav"))
    assert exc.value.status_code == 400


def test_speech_answers_500_when_say_itself_fails(say_endpoint):
    ar, _ = say_endpoint
    from omlx.utils import macos_say as engine

    def explode(*args, **kwargs):
        raise RuntimeError("macOS say: no audio produced")

    engine.synth_wav = explode
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ar._speech_via_macos_say(_speech_request(), "", "wav"))
    assert exc.value.status_code == 500


def test_speech_refuses_a_switched_off_engine(monkeypatch, say_endpoint):
    ar, calls = say_endpoint
    _off(monkeypatch, ar)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ar._speech_via_macos_say(_speech_request(), "", "wav"))
    assert exc.value.status_code == 400
    assert "switched off" in str(exc.value.detail)
    assert calls["segments"] == [], "an off engine must not render anything"


def test_speech_refuses_a_non_macos_server(monkeypatch, say_endpoint):
    ar, _ = say_endpoint
    from omlx.utils import macos_say as engine

    monkeypatch.setattr(engine, "available", lambda: False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ar._speech_via_macos_say(_speech_request(), "", "wav"))
    assert exc.value.status_code == 400
    assert "macOS" in str(exc.value.detail)


def test_speech_refuses_voice_cloning(monkeypatch, say_endpoint):
    ar, _ = say_endpoint
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            ar._speech_via_macos_say(_speech_request(ref_audio="AAAA"), "", "wav")
        )
    assert exc.value.status_code == 400
    assert "ref_audio" in str(exc.value.detail)


def test_openai_voice_field_is_ignored_when_it_is_not_a_macos_voice(say_endpoint):
    ar, calls = say_endpoint
    _drain_file(ar_ := asyncio.run(
        ar._speech_via_macos_say(_speech_request(voice="alloy"), "", "wav")
    ))
    assert calls["voice"] == ""
    assert ar_.status_code == 200


def test_macos_voice_from_the_voice_field_is_honoured(say_endpoint):
    ar, calls = say_endpoint
    from omlx.utils import macos_say as engine

    def fake_resolve(name):
        return {"name": name, "identifier": "id-" + name} if name == "Zuzana" else None

    engine.resolve = fake_resolve
    _drain_file(asyncio.run(
        ar._speech_via_macos_say(_speech_request(voice="Zuzana"), "", "wav")
    ))
    assert calls["voice"] == "Zuzana"


def test_voice_named_in_the_model_token_is_held_to(say_endpoint):
    ar, _ = say_endpoint

    def explode(voice, language, text, voices):
        raise LookupError("voice 'Ghost' is not installed on this Mac")

    from omlx.utils import macos_say as engine

    engine.route = explode
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ar._speech_via_macos_say(_speech_request(), "Ghost", "wav"))
    assert exc.value.status_code == 400


def _collect(response):
    async def run():
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(run())


def test_a_streamed_reply_sends_the_first_sentence_before_the_last(say_endpoint):
    """`say` renders one piece at a time, so a streamed reply can carry piece
    one while piece two is still being spoken: header, then one block per
    segment. The header is the house unknown-length one (audio_utils), the same
    promise the model streaming path makes; a client that needs the true
    length asks without `stream` and gets the stitched file instead."""
    ar, calls = say_endpoint
    text = "Alpha beta gamma. Delta epsilon zeta. " * 80
    response = asyncio.run(
        ar._speech_via_macos_say(_speech_request(input=text, stream=True), "", "wav")
    )
    from fastapi.responses import StreamingResponse

    assert isinstance(response, StreamingResponse)
    pieces = _collect(response)
    assert len(calls["segments"]) > 1
    # header first, then one block per segment
    assert len(pieces) == len(calls["segments"]) + 1
    assert pieces[0][:4] == b"RIFF" and len(pieces[0]) == 44
    first = 2 * len(calls["segments"][0])
    assert len(pieces[1]) == first, "the first audio IS the first sentence"
    assert b"".join(pieces[1:]).count(b"\x01\x02") >= sum(
        len(s) for s in calls["segments"]
    ), "every segment reached the wire"


def test_a_long_reply_is_split_at_sentences_and_stitched(say_endpoint):
    """The whole point of segmentation: many `say` calls, one honest WAV.
    Asserted on the bytes, because a stitched file that lies about its length
    is the failure mode nobody notices until a player shows the wrong duration."""
    ar, calls = say_endpoint
    text = "One two three four five six. " * 120  # 3000 characters
    data = _drain_file(_ask(ar, input=text))
    segments = calls["segments"]
    assert len(segments) > 1, "a 3000 character reply must not be one call"
    joined = "".join(segments).replace(" ", "")
    assert joined == text.replace(" ", ""), "the text survives intact"
    assert calls["routed_for"] == len(text.strip()), "routing saw the whole utterance"
    frames, rate, channels = _wav_frames(data)
    assert frames == sum(len(s) for s in segments), "every segment is in the file"
    assert len(data) == frames * 2 + 44, "the header matches the payload exactly"
    assert (rate, channels) == (24000, 1)


def test_segmentation_can_be_switched_off(monkeypatch, say_endpoint):
    ar, calls = say_endpoint
    _use(monkeypatch, ar, chunk=0)
    _drain_file(_ask(ar, input="One two three four five six. " * 120))
    assert len(calls["segments"]) == 1, "chunk size 0 means one `say` call"


def test_a_reply_that_cannot_fit_is_refused_before_the_first_render(
    monkeypatch, say_endpoint
):
    """Fail or pass up front. The ceiling is the environment's
    (OMLX_SAY_MAX_AUDIO_MB), and at the fast bound of speech one MB is about
    416 characters - so 2000 characters is arithmetic, not a guess: the answer
    must arrive with NOTHING rendered, and it must name the env to raise."""
    from omlx.utils import macos_say as engine

    ar, calls = say_endpoint
    monkeypatch.setenv(engine.MAX_AUDIO_ENV, "1")
    _use(monkeypatch, ar)
    with pytest.raises(HTTPException) as exc:
        _ask(ar, input="word " * 400)
    assert exc.value.status_code == 400
    assert "ceiling" in str(exc.value.detail)
    assert engine.MAX_AUDIO_ENV in str(exc.value.detail), "the way out is named"
    assert calls["segments"] == [], "refused up front: not one `say` call"


def test_the_ceiling_moves_with_the_environment_and_junk_falls_back(
    monkeypatch,
):
    """The knob left the card but not the operator: the env is read per
    request - no settings file, no restart. Junk, blank and 0 mean the
    default: the guardrail can be moved, never removed."""
    from omlx.utils import macos_say as engine

    monkeypatch.delenv(engine.MAX_AUDIO_ENV, raising=False)
    assert engine.max_audio_mb() == engine.DEFAULT_MAX_AUDIO_MB
    monkeypatch.setenv(engine.MAX_AUDIO_ENV, "64")
    assert engine.max_audio_mb() == 64
    monkeypatch.setenv(engine.MAX_AUDIO_ENV, "junk")
    assert engine.max_audio_mb() == engine.DEFAULT_MAX_AUDIO_MB
    monkeypatch.setenv(engine.MAX_AUDIO_ENV, "0")
    assert engine.max_audio_mb() == engine.DEFAULT_MAX_AUDIO_MB


def test_a_slow_voice_crossing_the_ceiling_mid_render_stops_early(
    monkeypatch, say_endpoint
):
    """Under the fast bound the pre-check passes, and a voice slower than the
    bound can still walk past the ceiling while rendering. That exception to
    fail-up-front must stop at the crossing, name the env, and leak no spool."""
    import glob
    import os
    import tempfile

    from omlx.utils import macos_say as engine

    ar, calls = say_endpoint
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "omlx-say-*")))
    monkeypatch.setenv(engine.MAX_AUDIO_ENV, "1")
    _use(monkeypatch, ar, chunk=300)
    calls["frames_per_char"] = 1500      # 3 KB of audio per character
    with pytest.raises(HTTPException) as exc:
        _ask(ar, input="word " * 70)     # 350 chars: under the fast bound
    assert exc.value.status_code == 400
    assert engine.MAX_AUDIO_ENV in str(exc.value.detail)
    assert 0 < len(calls["segments"]) < 7, "stops at the crossing, not after"
    left = set(glob.glob(os.path.join(tempfile.gettempdir(), "omlx-say-*")))
    assert left == before, "a refused request left spool files behind"


def test_an_encoded_reply_is_written_from_the_spool_file(
    monkeypatch, say_endpoint
):
    """mp3 and friends must be encoded by block copy from the stitched file:
    loading the reply to transcode it is the memory this path exists to avoid."""
    import os

    ar, calls = say_endpoint
    seen = {}

    def fake_encode(src, dst, response_format):
        seen["src"] = src
        seen["format"] = response_format
        assert os.path.getsize(src) > 44, "the spool must be a real WAV on disk"
        with open(dst, "wb") as fh:
            fh.write(b"ID3fake")

    monkeypatch.setattr(ar, "_transcode_speech_file", fake_encode)
    data = _drain_file(
        _ask(ar, fmt="mp3", input="One two three four five six. " * 120)
    )
    assert seen["format"] == "mp3"
    assert data == b"ID3fake"
    assert not os.path.exists(seen["src"]), "the WAV spool must not outlive the encode"


def test_the_segment_size_is_a_setting_and_the_ceiling_is_not():
    """The card writes the segment size and the loader clamps it on the way in.
    The audio ceiling is no longer a CARD setting (user 2026-10-29: 256 MB is
    about ninety minutes of speech, no reply needs a knob for it): the one
    way to move it is the OMLX_SAY_MAX_AUDIO_MB environment. The settings
    schema must not carry the field, and a settings.json left over from the
    version that did must simply drop the key."""
    from omlx.settings import IntegrationSettings

    loaded = IntegrationSettings.from_dict({"say_chunk_chars": 50})
    assert loaded.say_chunk_chars == 200, "raised to the floor, not taken literally"
    junk = IntegrationSettings.from_dict({"say_chunk_chars": "off"})
    blank = IntegrationSettings.from_dict({"say_chunk_chars": None})
    assert junk.say_chunk_chars == 1200, "junk reads as the default"
    assert blank.say_chunk_chars == 1200, "absent reads as the default too"
    assert not hasattr(IntegrationSettings(), "say_max_audio_mb")
    stale = IntegrationSettings.from_dict({"say_max_audio_mb": 1})
    assert "say_max_audio_mb" not in stale.to_dict(), \
        "an old settings file must not resurrect the ceiling"


# --------------------------------------------------------- settings schema --


def test_admin_settings_expose_the_speech_fields():
    """The card reads them straight off /admin/api/global-settings."""
    from omlx.settings import IntegrationSettings

    payload = IntegrationSettings().to_dict()
    assert payload["say_enabled"] is True
    assert payload["say_voices"] == {}


def test_update_model_accepts_the_speech_fields():
    from omlx.admin.routes import GlobalSettingsRequest

    req = GlobalSettingsRequest(
        say_enabled=False, say_voices={"cs-CZ": "com.apple.voice.premium.cs-CZ.Zuzana"}
    )
    assert "say_enabled" in req.model_fields_set
    assert req.say_voices == {"cs-CZ": "com.apple.voice.premium.cs-CZ.Zuzana"}
