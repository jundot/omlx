# SPDX-License-Identifier: Apache-2.0
"""Text-to-speech through the macOS ``say`` command (``model: "macos-say"``).

A request that names no voice gets the voice macOS uses for the request
``language``, or else for the detected language of the text
(``NLLanguageRecognizer`` and ``AVSpeechSynthesisVoice voiceWithLanguage:``,
reached through ``osascript`` because PyObjC is not a dependency).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from ..exceptions import InvalidRequestError

MODEL_ID = "macos-say"
MAX_INPUT_CHARS = 4096  # OpenAI's limit for /v1/audio/speech
SAMPLE_RATE = 24000  # Opus needs one of its fixed rates
_BASE_WPM = 175
_TIMEOUT_S = 120

OPENAI_VOICES = frozenset(
    "alloy ash ballad coral echo fable onyx nova sage shimmer verse marin cedar".split()
)

# argv: optional language. The text comes on stdin, never on the command line.
_PICK_SCRIPT = """use framework "Foundation"
use framework "NaturalLanguage"
use framework "AVFoundation"
on run argv
\tif (count of argv) > 0 then
\t\tset v to current application's AVSpeechSynthesisVoice's voiceWithLanguage:(item 1 of argv)
\t\tif v is not missing value then return (v's identifier()) as text
\tend if
\tset d to current application's NSFileHandle's fileHandleWithStandardInput()'s readDataToEndOfFile()
\tset txt to current application's NSString's alloc()'s initWithData:d encoding:4
\tset rec to current application's NLLanguageRecognizer's alloc()'s init()
\trec's processString:txt
\tset hyp to rec's languageHypothesesWithMaximum:1
\tset ks to hyp's allKeys()
\tif (count of ks) is 0 then return ""
\tset k to (item 1 of ks) as text
\tif ((hyp's objectForKey:k)'s doubleValue()) as real < 0.5 then return ""
\tset v to current application's AVSpeechSynthesisVoice's voiceWithLanguage:k
\tif v is missing value then return ""
\treturn (v's identifier()) as text
end run
"""

_ROSTER_LINE = re.compile(r"^(?P<name>.+?)\s+[a-z]{2,3}(?:_[A-Za-z0-9]+)*\s+#")


def enabled(global_settings) -> bool:
    integrations = getattr(global_settings, "integrations", None)
    return bool(getattr(integrations, "macos_say_enabled", False))


def explicit_voice(voice: str | None) -> str:
    """``""`` for OpenAI stock voices and ``default``: pick automatically."""
    v = (voice or "").strip()
    if not v or v.lower() == "default" or v.lower() in OPENAI_VOICES:
        return ""
    return v


def installed_voice_names() -> set[str]:
    """Lower-cased names from ``say -v ?``, with and without the
    ``(English (UK))`` style suffix that ``say -v`` also accepts."""
    out = subprocess.run(
        ["say", "-v", "?"], capture_output=True, text=True, timeout=15
    ).stdout
    names = set()
    for line in out.splitlines():
        m = _ROSTER_LINE.match(line)
        if m:
            name = m.group("name").strip().lower()
            names.add(name)
            names.add(name.split(" (", 1)[0])
    return names


def voice_is_installed(voice: str) -> bool:
    # say uses the system voice without an error when a name is unknown.
    if voice.startswith("com.apple."):
        return True
    return voice.lower() in installed_voice_names()


def pick_voice(text: str, language: str | None) -> str:
    """Voice identifier for ``text``; ``""`` when detection is not sure."""
    # Without "--", osascript reads a language that starts with "-e" as script code.
    argv = ["osascript", "-e", _PICK_SCRIPT, "--"]
    if language and language.strip():
        argv.append(language.strip())
    try:
        p = subprocess.run(
            argv, input=text, capture_output=True, text=True, timeout=15
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("macos-say: voice pick timed out") from exc
    if p.returncode != 0:
        raise RuntimeError(f"macos-say: voice pick failed: {p.stderr.strip()}")
    return p.stdout.strip()


def synthesize(
    text: str,
    voice: str | None = None,
    language: str | None = None,
    speed: float | None = None,
) -> bytes:
    if len(text) > MAX_INPUT_CHARS:
        raise InvalidRequestError(
            f"'input' is longer than {MAX_INPUT_CHARS} characters", field="input"
        )
    name = explicit_voice(voice)
    if name and not voice_is_installed(name):
        raise InvalidRequestError(
            f"macOS voice '{name}' is not installed (see `say -v ?`)", field="voice"
        )
    if not name:
        name = pick_voice(text, language)
    # say runs "[[...]]" in the text as a command, such as a pause of any length.
    text = re.sub(r"\[(?=\[)", "[ ", text)
    args = ["say"]
    if name:
        args += ["-v", name]
    if speed and speed != 1.0:
        args += ["-r", str(round(_BASE_WPM * speed))]
    with tempfile.TemporaryDirectory(prefix="omlx-say-") as td:
        dst = Path(td) / "out.wav"
        args += ["-f", "-", "-o", str(dst), f"--data-format=LEI16@{SAMPLE_RATE}"]
        p = subprocess.run(
            args, input=text, capture_output=True, text=True, timeout=_TIMEOUT_S
        )
        if p.returncode != 0 or not dst.exists():
            raise RuntimeError(f"say failed: {p.stderr.strip() or p.returncode}")
        return dst.read_bytes()
