# SPDX-License-Identifier: Apache-2.0
"""Text-to-speech through the macOS ``say`` command.

Serves ``model: "macos-say"`` on ``/v1/audio/speech``. There is no model to
load: the audio comes from the voices macOS already has installed.

Voice choice, first match wins:

1. ``voice`` naming a macOS voice (``"Zuzana"``, ``"Zuzana (Premium)"`` or a
   ``com.apple...`` identifier) is used as given. OpenAI's stock voice names
   (``alloy``, ``ash``...) and ``"default"`` mean "pick for me".
2. ``language`` from the request.
3. The language of the text, detected by macOS (when autodetect is on).
4. The default language from Settings.
5. Nothing: plain ``say`` speaks with the system voice.

A language becomes a voice through ``AVSpeechSynthesisVoice
voiceWithLanguage:``, which answers with the voice macOS itself uses for
that language (the System Settings choice when the owner made one).
Detection and that lookup are public APIs (macOS 10.14+) reached through one
short ``osascript`` call; when it fails, plain ``say`` still speaks.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_ID = "macos-say"
MAX_INPUT_CHARS = 4096  # same limit as OpenAI's /v1/audio/speech
SAMPLE_RATE = 24000  # an Opus-encodable rate, same as most TTS models
_BASE_WPM = 175  # about the normal pace of the stock voices
_TIMEOUT_S = 120

OPENAI_VOICES = frozenset(
    "alloy ash ballad coral echo fable onyx nova sage shimmer verse marin cedar".split()
)

# argv: text, explicit language, default language, "1" to detect.
# Prints "<language>\t<voice identifier>"; either part may be empty.
_PICK_SCRIPT = """use framework "Foundation"
use framework "NaturalLanguage"
use framework "AVFoundation"
on primaryOf(tag)
\treturn ((current application's NSString's stringWithString:tag)'s componentsSeparatedByString:"-")'s firstObject() as text
end primaryOf
on run argv
\tset lang to item 2 of argv
\tset dflt to item 3 of argv
\tif lang is "" and item 4 of argv is "1" then
\t\tset rec to current application's NLLanguageRecognizer's alloc()'s init()
\t\trec's processString:(item 1 of argv)
\t\tset hyp to rec's languageHypothesesWithMaximum:1
\t\tset ks to hyp's allKeys()
\t\tif (count of ks) > 0 then
\t\t\tset k to (item 1 of ks) as text
\t\t\tif ((hyp's objectForKey:k)'s doubleValue()) as real >= 0.5 then
\t\t\t\tset lang to k
\t\t\t\tif dflt is not "" and my primaryOf(dflt) is my primaryOf(k) then set lang to dflt
\t\t\tend if
\t\tend if
\tend if
\tif lang is "" then set lang to dflt
\tif lang is "" then return ""
\tset v to current application's AVSpeechSynthesisVoice's voiceWithLanguage:lang
\tif v is missing value then return lang
\treturn lang & tab & ((v's identifier()) as text)
end run
"""

_ROSTER_LINE = re.compile(r"^(?P<name>.+?)\s+[a-z]{2,3}(?:_[A-Za-z0-9]+)*\s+#")


def available() -> bool:
    """True on macOS with the ``say`` command present."""
    return sys.platform == "darwin" and shutil.which("say") is not None


def enabled(global_settings) -> bool:
    """The Settings > Integrations switch (on when settings are absent)."""
    integrations = getattr(global_settings, "integrations", None)
    return bool(getattr(integrations, "macos_say_enabled", True))


def model_visible(global_settings) -> bool:
    """List ``macos-say`` in ``/v1/models`` when it can actually answer."""
    return enabled(global_settings) and available()


def normalize_language(value: str | None) -> str:
    """``"cs_CZ"`` / ``" cs-cz "`` -> ``"cs-CZ"``; junk -> ``""``."""
    tag = (value or "").strip().replace("_", "-")
    if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", tag):
        return ""
    head, *rest = tag.split("-")
    return "-".join(
        [head.lower()] + [p.upper() if len(p) == 2 else p for p in rest]
    )


def explicit_voice(voice: str | None) -> str:
    """The macOS voice a request names, or ``""`` for "pick for me"."""
    v = (voice or "").strip()
    if not v or v.lower() == "default" or v.lower() in OPENAI_VOICES:
        return ""
    return v


def installed_voice_names() -> set[str]:
    """Lower-cased names from ``say -v ?`` (empty set when it fails)."""
    try:
        out = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=15
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("macos-say: voice list failed: %s", exc)
        return set()
    names = set()
    for line in out.splitlines():
        m = _ROSTER_LINE.match(line)
        if m:
            names.add(m.group("name").strip().lower())
    return names


def voice_is_installed(voice: str) -> bool:
    """``say`` falls back to the system voice without an error when a name is
    unknown, so names are checked first. Identifiers are passed through."""
    if voice.startswith("com.apple."):
        return True
    return voice.lower() in installed_voice_names()


def pick_voice(text: str, language: str, default_language: str, detect: bool) -> str:
    """Voice identifier for this text, or ``""`` to let plain ``say`` speak."""
    if not language and not default_language and not detect:
        return ""
    if shutil.which("osascript") is None:
        return ""
    argv = ["osascript", "-", text[:1000], language, default_language]
    argv.append("1" if detect else "0")
    try:
        p = subprocess.run(
            argv, input=_PICK_SCRIPT, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("macos-say: voice pick failed: %s", exc)
        return ""
    if p.returncode != 0:
        logger.warning("macos-say: voice pick failed: %s", p.stderr.strip())
        return ""
    lang, _, ident = p.stdout.strip().partition("\t")
    logger.debug("macos-say: language %r -> voice %r", lang, ident)
    return ident.strip()


def synthesize(text: str, voice: str = "", speed: float | None = None) -> bytes:
    """Render ``text`` to 24 kHz mono 16-bit WAV bytes."""
    args = ["say"]
    if voice:
        args += ["-v", voice]
    if speed and speed != 1.0:
        speed = min(max(float(speed), 0.25), 4.0)
        args += ["-r", str(round(_BASE_WPM * speed))]
    with tempfile.TemporaryDirectory(prefix="omlx-say-") as td:
        src = Path(td) / "in.txt"
        dst = Path(td) / "out.wav"
        src.write_text(text, encoding="utf-8")
        args += ["-f", str(src), "-o", str(dst), f"--data-format=LEI16@{SAMPLE_RATE}"]
        p = subprocess.run(args, capture_output=True, text=True, timeout=_TIMEOUT_S)
        if p.returncode != 0 or not dst.exists():
            raise RuntimeError(f"say failed: {p.stderr.strip() or p.returncode}")
        return dst.read_bytes()
