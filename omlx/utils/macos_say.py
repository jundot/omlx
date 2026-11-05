# SPDX-License-Identifier: Apache-2.0
"""macOS ``say`` as a text-to-speech engine: no model, no GPU, instant.

Every Mac ships the speech synthesizer behind the ``say`` command with a
roster of system voices (one set per locale, plus whatever the owner
downloaded in System Settings > Spoken Content > Voices). For reading a
reply aloud that is often exactly enough: no RAM, no load time, and the
voice is the one the user's own accessibility setup already sounds like.

This module never assumes the platform supports it: :func:`available`
probes, and every entry point degrades to an honest empty answer off
macOS. ``model='say'`` on ``POST /v1/audio/speech`` reaches this engine.

Contract:

- the voice list comes from ``say -v ?`` - the ground truth of what THIS
  machine can speak. Lines look like::

      Zuzana                cs_CZ    # Ahoj! Já jsem Zuzana.
      Samantha (English (US)) en_US  # Hello! My name is Samantha.

  so the name is everything before the locale token (it may contain
  spaces and parentheses) and the greeting doubles as the audition text;
- quality tiers ride IN the name on modern macOS ('Foo (Enhanced)',
  'Foo (Premium)', 'Foo (Personal)'); a plain name is Standard;
- synthesis writes the text to a temp file and runs
  ``say -v NAME -f FILE -o OUT --data-format=LEI16@22050``, which yields a
  plain 22.05 kHz mono RIFF WAV. argv is a list, the voice must be an
  installed one, and the text is capped: no shell, no injection surface,
  no unbounded work;
- formats other than WAV are the endpoint's business: it transcodes the
  WAV this module returns with the same in-memory path every other TTS
  model uses (``_transcode_speech_output`` in omlx/api/audio_routes.py);
- ``say`` also works bare, and with a locale tail ('say:cs-CZ'). With no
  explicit voice, :func:`route` picks one: the per-language choice from
  the settings (:attr:`IntegrationSettings.say_voices`), the owner's
  per-language System Voice, the Mac's selected voice, the language's
  default row, its best installed tier. With no language at all the text
  decides - Apple's own language recognizer names it (:func:`detect_language`)
  and the same chain answers; when even that fails, ``say`` runs without
  ``-v`` and macOS speaks its own default. A request for a language with
  nothing installed raises instead of speaking a foreign voice.
"""

from __future__ import annotations

import logging
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SAY_ENGINE = "say"              # model token that selects this engine
SAY_PREFIX = "say:"             # model token carrying an explicit voice
MAX_SAY_CHARS = 200000          # a text this long is a mistake, not a reply
OPENAI_INPUT_CHARS = 4096       # OpenAI's documented input limit for this call
# Long replies are rendered in sentence-sized calls and stitched, so the
# engine's own bound is bytes of audio, not characters: 24 kHz mono 16-bit is
# 48 KB per second of speech whatever the text, while characters per second is
# a property of the voice (measured here: 18 chars/s for Samantha, 5 chars/s
# for the novelty voice this Mac ships as its English default).
DEFAULT_CHUNK_CHARS = 1200      # ~1-2 sentences; one call per chunk
MIN_CHUNK_CHARS = 200           # below this the per-call cost dominates
MAX_CHUNK_CHARS = 20000         # `say` reads a file, so argv is not the limit
# The ceiling is a guardrail rather than a card setting (decided 2026-10-29):
# the default is 256 MB, ~89 minutes of speech, deliberately generous because
# guessing what an operator needs is a losing game. It stays configurable
# through the environment - the one place a number this size belongs - and a
# reply past it is refused, never cut short.
MAX_AUDIO_ENV = "OMLX_SAY_MAX_AUDIO_MB"
DEFAULT_MAX_AUDIO_MB = 256
SPEECH_BYTES_PER_SEC = 48000     # 24 kHz mono 16-bit, fixed whatever the voice
# Measured chars/s: 18 for Samantha, 5 for the novelty default voice on this
# Mac. Rounding UP to 20 gives the fast bound of speech: the minimum bytes
# this text can produce. Past the ceiling even at that pace, the answer is
# refuse NOW - before the first `say` call - rather than mid-render.
FAST_SPEECH_CHARS_PER_SEC = 20
LONG_INPUT_SPEAK = "speak"      # operator policy: say the whole thing
LONG_INPUT_REJECT = "reject"    # operator policy: refuse like OpenAI does
LONG_INPUT_POLICIES = (LONG_INPUT_SPEAK, LONG_INPUT_REJECT)
_MIN_DETECT_CHARS = 3           # below this there is nothing to recognise
_NL_CONFIDENT = 50              # percent; 'ok' scores pl@29 - garbage stops here
_DETECT_CHARS = 1000            # the first lines of a reply decide the language
_TIMEOUT_S = 180
# 24 kHz, not 22.05: libsndfile encodes Opus only at 8/12/16/24/48 kHz, so a
# 22.05 kHz render breaks response_format=opus for this engine alone.
_SAMPLE_RATE = 24000
BYTES_PER_AUDIO_SECOND = _SAMPLE_RATE * 2  # mono 16-bit, 48 KB per second
_BASE_WPM = 175                 # say's own pace is ~175-190; multipliers ride on it
_CACHE_TTL_S = 300
_PROBE_TTL_S = 300              # system settings rarely change, but they can

_LINE_RE = re.compile(
    r"^(?P<name>.+?)\s+(?P<locale>[a-z]{2}(?:[-_][A-Za-z0-9]{2,6})*(?:-x)?)"
    r"\s+#\s*(?P<sample>.*)$"
)

_TIER_RANK = {"premium": 0, "enhanced": 1, "personal": 2}

# The installed-voice catalog System Settings itself uses: AVFoundation sees
# voices the CLI roster omits (downloaded Siri/neural sets) and carries the
# true audio quality (Default/Enhanced/Premium) in the object description.
# Probed headless through AppleScriptObjC - works over ssh and in launchd
# sessions; a failure degrades to the plain `say -v ?` list, never to nothing.
_AV_PROBE = '''use framework "AVFoundation"
use scripting additions
set vs to current application's AVSpeechSynthesisVoice's speechVoices()
set n to (vs's |count|()) as integer
set out to ""
repeat with i from 0 to (n - 1)
\tset v to vs's objectAtIndex:i
\tset nm to (v's |name|()) as text
\tset lg to (v's language()) as text
\tset idr to (v's identifier()) as text
\tset dsc to (v's |description|()) as text
\tset dfl to ((v's isDefault()) as integer)
set out to out & nm & "\\t" & lg & "\\t" & idr & "\\t" & dsc & "\\t" & dfl & "
"
end repeat
return out
'''

_QUALITY_RE = re.compile(r"Quality:\s*([A-Za-z-]+)")

# Apple's own text-language identifier (the NaturalLanguage framework that
# Spotlight and TextEdit use): offline, no model download. Czech scores
# cs@99 with and without diacritics, Slovak stays sk, and garbage like 'ok'
# only reaches pl@29 - under the confidence gate below. The candidate rides
# in argv (no shell), the script arrives on stdin. Output: 'cs\t99'.
_NL_PROBE = '''use framework "NaturalLanguage"
use scripting additions
on run argv
\tset rec to current application's NLLanguageRecognizer's alloc()'s init()
\t(rec's processString:(item 1 of argv))
\tset d to rec's languageHypothesesWithMaximum:1
\tset ks to d's allKeys()
\tif (count of ks) is 0 then return ""
\tset k to item 1 of ks
\tset conf to ((d's objectForKey:k)'s doubleValue()) as real
\treturn (k as text) & "\\t" & ((conf * 100) div 1 as text)
end run
'''

# NSSpeechSynthesizer answers with the identifier of the voice System
# Settings > Accessibility > Read & Speak has selected - the one bare `say`
# speaks with. AVFoundation's per-voice isDefault is NOT that: it marks the
# factory default-QUALITY row of every language.
_DV_PROBE = '''use framework "AppKit"
use scripting additions
set dv to current application's NSSpeechSynthesizer's defaultVoice()
return dv as text
'''

# System Settings > Accessibility > Spoken Content > Voices lets the owner
# pick a System Voice PER LANGUAGE (the list groups them by primary
# language, so an 'en' pick answers every en-* variant the way macOS itself
# answers). macOS 26+ stores those picks in the com.apple.Accessibility
# domain; older macOS has no such key and an untouched Spoken Content never
# writes one. Absence is honest silence: every language then keeps its
# factory behaviour.
_OS_SEL_DOMAIN = "com.apple.Accessibility"
_OS_SEL_KEY = "SpokenContentDefaultVoiceSelectionsByLanguage"

# What System Settings' own downloadable roster ships (readable, no root).
# Three generations live side by side:
#   MacinTalk         - the classic compact voices (all standard quality)
#   CombinedVocalizer - the legacy enhanced-quality downloads
#   TTSAXResourceModel- the modern neural set; ONE file carries every tier
#                       per voice, and System Settings builds its picker
#                       from it
# What is INSTALLED still comes from AV/say only: these files say what
# Apple offers here, never what this Mac currently holds.
_CATALOG_PATHS = (
    (
        "/System/Library/AssetsV2/com_apple_MobileAsset_MacinTalkVoiceAssets/"
        "com_apple_MobileAsset_MacinTalkVoiceAssets.xml",
        "standard",
    ),
    (
        "/System/Library/AssetsV2/"
        "com_apple_MobileAsset_VoiceServices_CombinedVocalizerVoices/"
        "com_apple_MobileAsset_VoiceServices_CombinedVocalizerVoices.xml",
        "enhanced",
    ),
    (
        "/System/Library/AssetsV2/com_apple_MobileAsset_TTSAXResourceModelAssets/"
        "com_apple_MobileAsset_TTSAXResourceModelAssets.xml",
        "footprint",
    ),
)

_TIER_BY_FOOTPRINT = {
    "premium": "premium",
    "enhanced": "enhanced",
    "personal": "personal",
    "compact": "standard",
    "default": "standard",
}

def available() -> bool:
    """True when ``say`` exists on this platform. Probed per call (a cheap
    PATH lookup) so a Linux box or a stripped system behaves as unsupported."""
    return sys.platform == "darwin" and shutil.which("say") is not None


def _norm_locale(loc: Any) -> str:
    """One spelling for one language: say writes cs_CZ, AVFoundation and the
    system catalogs write cs-CZ."""
    return str(loc or "").strip().replace("_", "-")


def _root_name(name: Any) -> str:
    """Display identity across rosters: 'Samantha (English (US))' and
    'Samantha' are the same voice; 'Ava (Premium)' and 'Ava (Enhanced)' are
    reached by the row's quality instead of its suffix text."""
    return str(name or "").split(" (")[0].strip().lower()


def parse_voices(text: str) -> list[dict[str, Any]]:
    """``say -v ?`` output -> [{name, locale, quality, sample, identifier}].

    Junk lines are skipped, not fatal: the roster is a system file we do not
    control. CLI rows have no separate identifier, so the name addresses it.
    """
    out: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name").strip()
        if not name:
            continue
        quality = "standard"
        lowered = name.lower()
        for tier in ("premium", "enhanced", "personal"):
            if "(%s)" % tier in lowered:
                quality = tier
                break
        out.append(
            {
                "name": name,
                "locale": _norm_locale(m.group("locale")),
                "quality": quality,
                "sample": m.group("sample").strip(),
                "identifier": name,
            }
        )
    return out


def parse_av(text: str) -> list[dict[str, Any]]:
    """AVFoundation probe output -> rows. The 5th column (isDefault) is
    optional: output from an older probe parses as non-default, never as
    junk."""
    out: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name, lang, ident, desc = (p.strip() for p in parts[:4])
        if not name or not ident:
            continue
        m = _QUALITY_RE.search(desc)
        quality = (m.group(1).lower() if m else "default")
        if quality == "default":
            quality = "standard"  # same vocabulary the tiers use everywhere
        out.append(
            {
                "name": name,
                "locale": _norm_locale(lang),
                "quality": quality,
                "identifier": ident,
                "desc": desc,
                "default": len(parts) > 4 and parts[4].strip() == "1",
            }
        )
    return out


def parse_catalog(doc: Any, quality: str) -> list[dict[str, Any]]:
    """One AssetsV2 plist -> [{name, locale, quality}]. ``quality`` is a
    fixed tier word, or 'footprint' to read each entry's own Footprint field
    (the modern catalog holds several tiers per voice)."""
    out: list[dict[str, Any]] = []
    for e in ((doc or {}).get("Assets") or []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("Name") or "").strip()
        tier = quality
        if quality == "footprint":
            tier = _TIER_BY_FOOTPRINT.get(
                str(e.get("Footprint") or "").strip().lower(), "standard"
            )
        locs = e.get("Languages") or ([e.get("Language")] if e.get("Language") else [])
        for loc in locs:
            if name and loc:
                out.append({"name": name, "locale": _norm_locale(loc), "quality": tier})
    return out


def _downloaded_names() -> set[str]:
    """Names backed by a *.SpeechVoice bundle in the two locations readable
    without root. A voice outside those bundles is simply NOT badged: we
    never claim 'System' for something we cannot prove. (Legacy macOS 11-era
    path; modern downloads surface through the AV probe instead.)"""
    found: set[str] = set()
    for base in (
        Path.home() / "Library" / "Speech" / "Voices",
        Path("/Library/Speech/Voices"),
    ):
        try:
            bundles = list(base.glob("*.SpeechVoice"))
        except OSError:
            continue
        for bundle in bundles:
            name = ""
            try:
                with open(bundle / "Contents" / "Info.plist", "rb") as fh:
                    name = str(plistlib.load(fh).get("VoiceName") or "")
            except (OSError, ValueError):
                pass
            found.add((name or bundle.stem).strip())
    return found


def _run_osascript(script: str, arg: str | None = None, timeout: int = 30) -> str:
    """Run an AppleScriptObjC probe through argv/stdin (never a shell) and
    return stdout; '' when osascript is missing or the probe fails."""
    if shutil.which("osascript") is None:
        return ""
    cmd = ["osascript", "-"] + ([arg] if arg is not None else [])
    try:
        p = subprocess.run(
            cmd, input=script, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.info("macOS speech probe failed to start: %s", e)
        return ""
    if p.returncode != 0:
        logger.info(
            "macOS speech probe rc=%d: %s", p.returncode, (p.stderr or "").strip()[:200]
        )
        return ""
    return p.stdout or ""


def _av_rows() -> list[dict[str, Any]]:
    return parse_av(_run_osascript(_AV_PROBE))


def _say_rows() -> list[dict[str, Any]]:
    try:
        p = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.info("say voice probe failed: %s", e)
        return []
    return parse_voices(p.stdout or "")


def _catalog_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, quality in _CATALOG_PATHS:
        try:
            with open(path, "rb") as fh:
                rows += parse_catalog(plistlib.load(fh), quality)
        except (OSError, ValueError) as e:
            logger.info(
                "voice catalog %s unavailable: %s", path.rsplit("/", 1)[-1], e
            )
    return rows


def _display_name(root_name: str, quality: str, os_name: str) -> str:
    """Plain names stay plain; above-Default rows show their tier. The OS
    sometimes bakes the suffix into the display name already - never append
    it twice."""
    if quality == "standard":
        return os_name or root_name.title()
    suffix = "(%s)" % quality.title()
    if suffix.lower() in str(os_name or "").lower():
        return os_name
    return "%s %s" % (os_name or root_name.title(), suffix)


_voice_cache: dict[str, Any] = {"ts": 0.0, "voices": []}
# Probe caches: system state that rarely changes but can (the TTL below is
# what makes a System Settings edit land without a restart).
_sel_cache: dict[str, Any] = {"ts": 0.0, "id": None}
_os_sel_cache: dict[str, Any] = {"ts": 0.0, "map": None}
_sys_lang_cache: dict[str, Any] = {"ts": 0.0, "lang": None}


def list_voices(force: bool = False) -> list[dict[str, Any]]:
    """The full picture of this Mac's speech voices, cached.

    INSTALLED rows - every voice the machine actually holds: AVFoundation
    (System Settings' own source; it sees downloaded neural/Siri voices the
    CLI roster omits) UNION ``say -v ?`` (adds anything AV misses, and
    supplies each voice's demo greeting). Each row carries its OS
    identifier, the exact ``say -v`` address.

    DOWNLOADABLE rows - Apple's local catalogs: offered here, not on this
    Mac yet (``installed`` False; the engine cannot speak them).
    """
    if not available():
        return []
    now = time.time()
    if not force and now - _voice_cache["ts"] < _CACHE_TTL_S:
        return _voice_cache["voices"]

    av = _av_rows()
    cli = _say_rows()
    bundles = _downloaded_names()
    cli_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    cli_flat: dict[tuple[str, str], dict[str, Any]] = {}
    for s in cli:
        # setdefault, not a comprehension: the FIRST row for a key answers
        # it, the way the system roster is read top-down
        cli_by_key.setdefault(
            (_root_name(s["name"]), s["locale"].lower(), s["quality"]), s
        )
        cli_flat.setdefault((_root_name(s["name"]), s["locale"].lower()), s)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(row: dict[str, Any]) -> None:
        key = (_root_name(row["name"]), row["locale"].lower(), row["quality"])
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    for a in av:
        s = cli_by_key.get((_root_name(a["name"]), a["locale"].lower(), a["quality"]))
        if s is None and a["quality"] == "standard":
            # AV 'Samantha' is the CLI row 'Samantha (English (US))'
            s = cli_flat.get((_root_name(a["name"]), a["locale"].lower()))
        quality = a["quality"]
        if quality == "standard" and s and s["quality"] != "standard":
            quality = s["quality"]
        add(
            {
                "name": _display_name(_root_name(a["name"]), quality, a["name"]),
                "locale": a["locale"],
                "quality": quality,
                "sample": (s or {}).get("sample", ""),
                "identifier": a["identifier"],
                "installed": True,
                "default": bool(a.get("default")),
                "downloaded": (quality != "standard")
                or (s is None)
                or (a["name"] in bundles),
            }
        )
    for s in cli:
        add(
            {
                **s,
                "installed": True,
                "default": False,
                "downloaded": s["name"] in bundles or s["quality"] != "standard",
            }
        )
    for c in _catalog_rows():
        if (_root_name(c["name"]), c["locale"].lower(), c["quality"]) in seen:
            continue
        add(
            {
                "name": _display_name(_root_name(c["name"]), c["quality"], c["name"]),
                "locale": c["locale"],
                "quality": c["quality"],
                "sample": "",
                "identifier": "",
                "installed": False,
                "default": False,
                "downloaded": False,
            }
        )
    if rows:
        _voice_cache.update(ts=now, voices=rows)
    return rows


def installed_rows() -> list[dict[str, Any]]:
    """Voices this Mac can actually speak right now."""
    return [v for v in list_voices() if v.get("installed") is not False]


def address_of(row: dict[str, Any]) -> str:
    """What ``say -v`` must be given: the identifier when the row has one.
    Display names collide (the system ships two Samanthas); identifiers do
    not."""
    return str(row.get("identifier") or row.get("name") or "")


def parse_voice_language(token: Any) -> tuple[str, str]:
    """Split a voice address into ``(name, language)``, either order.

    Display names are NOT unique on macOS: on this Mac eight names (the
    Eloquence family - Shelley, Eddy, Sandy, Grandma...) exist once per
    language, fourteen of them, so "say:Shelley" cannot say which Shelley is
    meant. Writing the language next to the name does: `say:Shelley:cs-CZ`
    and `say:cs-CZ:Shelley` both address one row. Identifiers carry dots, not
    colons, so they come back unchanged."""
    raw = str(token or "").strip()
    if ":" not in raw:
        return raw, ""
    head, _, tail = raw.partition(":")
    head, tail = head.strip(), tail.strip()
    if not head or not tail:
        return raw, ""
    if looks_like_locale(tail) and not looks_like_locale(head):
        return head, tail
    if looks_like_locale(head) and not looks_like_locale(tail):
        return tail, head
    return raw, ""


def name_languages(name: str) -> list[str]:
    """Every language this Mac carries a display name in, for honest errors."""
    return sorted(
        {
            _norm_locale(v.get("locale"))
            for v in list_voices()
            if v.get("installed") is not False and v["name"] == name
            and v.get("locale")
        },
        key=str.lower,
    )


def resolve(
    name: str, language: Any = None
) -> Optional[dict[str, Any]]:
    """Match a requested voice by IDENTIFIER first, then by display name.
    ``language`` picks between the rows that share a name: a caller who said
    Shelley plus pt-BR gets the Brazilian Shelley, and one who asked for a
    language that name does not cover gets None rather than a surprise. With
    no language the first roster row answers, as before. A miss refreshes the
    roster once: the owner may have downloaded the voice seconds ago, and
    waiting out a cache TTL would read as a bug."""
    if not name:
        return None
    want, named = parse_voice_language(name)
    language = language or named
    for fresh in (False, True):
        rows = [
            v
            for v in list_voices(force=fresh)
            if v.get("installed") is not False  # downloadable-but-absent
        ]
        for v in rows:
            # A voice only `say -v ?` knows carries its DISPLAY NAME as its
            # identifier - the very string that is not unique. Under a pinned
            # language such a row only answers if it speaks that language; a
            # real identifier (dotted, locale inside) passes the same test.
            if v["identifier"] == want and (
                not language or locale_score(language, v.get("locale")) > 0
            ):
                return v
        shares = [v for v in rows if v["name"] == want]
        if not shares:
            continue
        if not language:
            return shares[0]
        ranked = sorted(
            shares,
            key=lambda v: (
                -locale_score(language, v.get("locale")),
                _quality_rank(v.get("quality")),
            ),
        )
        if locale_score(language, ranked[0].get("locale")) > 0:
            return ranked[0]
        return None
    return None


def locale_score(requested: Any, have: Any) -> int:
    """How well one locale answers another: 2 = the same full locale,
    1 = the same primary language when ONE side names no region (a request
    for 'cs' is answered by 'cs-CZ' and the other way round), 0 = different.
    'cs-CZ' never answers 'cs-SK'."""
    r = _norm_locale(requested).lower()
    k = _norm_locale(have).lower()
    if not r or not k:
        return 0
    if r == k:
        return 2
    rp, _, rs = r.partition("-")
    kp, _, ks = k.partition("-")
    if rp == kp and not (rs and ks):
        return 1
    return 0


def _quality_rank(quality: Any) -> int:
    return _TIER_RANK.get(str(quality or ""), 3)


def _row_by_address(address: Any) -> Optional[dict[str, Any]]:
    if not address:
        return None
    for v in installed_rows():
        if address_of(v) == str(address):
            return v
    return None


def configured_voice(lang: Any, voices: dict | None) -> Optional[dict[str, Any]]:
    """The installed row the user's setting points at for this language: an
    exact locale beats a primary-language match, and a setting naming a
    voice this Mac no longer holds is skipped rather than honoured. None
    when nothing configured answers the language."""
    best, best_score = None, 0
    for loc, addr in (voices or {}).items():
        row = _row_by_address(addr)
        if row is None:
            continue
        score = locale_score(lang, loc)
        if score > best_score:
            best, best_score = row, score
    return best


# --- what macOS itself says ------------------------------------------------


def spoken_selections(force: bool = False) -> dict[str, str]:
    """{primary language: voice identifier} as picked in System Settings >
    Spoken Content > Voices - user choices only, never a factory claim.
    ``{}`` off macOS, on older macOS, or when the probe fails."""
    global _os_sel_cache
    if not available():
        return {}
    now = time.time()
    cached = _os_sel_cache
    if not force and cached["map"] is not None and now - cached["ts"] < _PROBE_TTL_S:
        return cached["map"]
    mapping: dict[str, str] = {}
    if shutil.which("defaults") is not None:
        try:
            p = subprocess.run(
                ["defaults", "read", _OS_SEL_DOMAIN, _OS_SEL_KEY],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if p.returncode == 0:
                mapping = _parse_spoken_selections(p.stdout)
        except (OSError, subprocess.SubprocessError) as e:
            logger.info("spoken-selection probe failed: %s", e)
    _os_sel_cache = {"ts": now, "map": mapping}
    return mapping


def _parse_spoken_selections(text: Any) -> dict[str, str]:
    """``defaults read`` output for the selection array -> {language: id}.
    Each record is a flat {...} block carrying boundLanguage and voiceId in
    whatever order the OS wrote them; a record without boundLanguage pairs
    with the scalar that precedes its block (the array's alternating
    spelling). Junk blocks are skipped, never fatal."""
    out: dict[str, str] = {}
    frags = re.split(r"\{", text or "")
    for i, block in enumerate(frags[1:], 1):
        block = block.split("}", 1)[0]
        m_vid = re.search(r'voiceId\s*=\s*"([^"]+)"', block)
        if not m_vid:
            continue
        addr = m_vid.group(1)
        m_lang = re.search(r'boundLanguage\s*=\s*"?([^";]+?)"?\s*;', block)
        lang = ""
        if m_lang:
            lang = m_lang.group(1)
        else:
            m_prev = re.search(
                r'"?([A-Za-z][A-Za-z0-9-]*)"?\s*,\s*$', frags[i - 1].rstrip()[-200:]
            )
            if m_prev:
                lang = m_prev.group(1)
        lang = _norm_locale(lang).lower()
        if lang and addr:
            out[lang] = addr
    return out


def system_selected_voice(lang: Any) -> Optional[dict[str, Any]]:
    """The installed row the owner's per-language System Voice points at.
    None when no selection answers the language or the chosen voice is not
    on this Mac anymore - macOS falls back to its own default there and so
    do we: a voice that is not installed is never addressed."""
    if not lang:
        return None
    sel = spoken_selections()
    if not sel:
        return None
    want = _norm_locale(lang).lower()
    # Most specific answering key first: an 'en-US' selection beats an 'en'
    # one if both exist. macOS writes primaries today; this stays correct
    # either way and is deterministic.
    for key, addr in sorted(
        sel.items(), key=lambda kv: -locale_score(lang, str(kv[0] or ""))
    ):
        k = str(key or "").lower()
        score = locale_score(lang, k)
        if score <= 0:
            continue
        # A key WITH a region answers only its exact locale: 'en-US' never
        # claims en-GB. A region-less key answers the whole family.
        if "-" in k and k != want:
            continue
        row = _row_by_address(addr)
        if row is not None:
            return row
    return None


def selected_voice_identifier(force: bool = False) -> str:
    """Identifier of the selected system voice ('' when unknown: non-Mac,
    probe failure, or a name the responder cannot print)."""
    global _sel_cache
    if not available():
        return ""
    now = time.time()
    if (
        not force
        and _sel_cache["id"] is not None
        and now - _sel_cache["ts"] < _PROBE_TTL_S
    ):
        return _sel_cache["id"]
    ident = _run_osascript(_DV_PROBE, timeout=15).strip()
    _sel_cache = {"ts": now, "id": ident}
    return ident


def selected_row() -> Optional[dict[str, Any]]:
    """The catalog row behind the selected system voice. Identifier first;
    legacy CLI rows address themselves by name, so fall back to the
    identifier's last segment. None when the selection names a voice this
    Mac no longer holds or the probe failed."""
    sel = selected_voice_identifier()
    if not sel:
        return None
    rows = installed_rows()
    for v in rows:
        if address_of(v) == sel:
            return v
    tail = sel.rsplit(".", 1)[-1].lower()
    for v in rows:
        if _root_name(v["name"]) == tail:
            return v
    return None


def _read_global_default(key: str) -> str:
    """One NSGlobalDomain value through the ``defaults`` CLI ('' on any
    failure). The global domain is user-readable without elevation."""
    if shutil.which("defaults") is None:
        return ""
    try:
        p = subprocess.run(
            ["defaults", "read", "-g", key], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.info("system language probe failed (%s): %s", key, e)
        return ""
    return p.stdout if p.returncode == 0 else ""


_TAGGED_RE = re.compile(r'"([A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*)"')


def system_language(force: bool = False) -> str:
    """The Mac's own language as a BCP-47 tag ('en-US'): the first
    AppleLanguages entry, with AppleLocale ('en_US@rg=czzzzz') as the
    fallback spelling. '' when unknown - every caller then keeps the
    language-less behaviour."""
    global _sys_lang_cache
    now = time.time()
    if (
        not force
        and _sys_lang_cache["lang"] is not None
        and now - _sys_lang_cache["ts"] < _PROBE_TTL_S
    ):
        return _sys_lang_cache["lang"]
    out = _read_global_default("AppleLanguages")
    m = _TAGGED_RE.search(out)
    if m:
        lang = m.group(1)
    else:
        lang = _read_global_default("AppleLocale").split("@")[0].strip().strip('"')
    lang = _norm_locale(lang)
    _sys_lang_cache = {"ts": now, "lang": lang}
    return lang


def detect_language(text: Any) -> str:
    """Dominant written language of one utterance, as macOS sees it,
    normalised to what the voice chain understands. '' - meaning 'do not
    route' - for too little text, a missing osascript, any probe failure,
    'und', or a top hypothesis under the confidence gate. Script-only
    subtags fold to the primary language ('zh-Hans' -> 'zh'), because voice
    locales carry a REGION ('zh-CN') and the chain answers a region-less
    request by primary language."""
    t = " ".join(str(text or "").split())
    if len(t) < _MIN_DETECT_CHARS:
        return ""
    out = _run_osascript(_NL_PROBE, arg=t[:_DETECT_CHARS], timeout=15)
    lang, _, conf = (out or "").strip().partition("\t")
    try:
        score = int(conf)
    except ValueError:
        return ""
    lang = lang.strip()
    if not lang or lang.lower() == "und" or score < _NL_CONFIDENT:
        return ""
    parts = [x for x in lang.split("-") if not (len(x) == 4 and x.isalpha())]
    return "-".join(parts)


# --- routing ---------------------------------------------------------------


def language_voice(
    lang: Any, voices: dict | None = None
) -> Optional[dict[str, Any]]:
    """The row that answers a language-carrying request, in order:

    1. the voice configured for that language (Settings > Integrations >
       macOS Speech); a setting naming a voice this Mac no longer holds is
       skipped;
    2. the voice the owner picked for that language in System Settings >
       Spoken Content > Voices - a real per-language user choice, and it
       outranks a global pick that belongs to a different language;
    3. the Mac's selected system voice, when its own language answers this
       request: that is what bare ``say`` and macOS itself speak with;
    4. the language's AVFoundation-flagged default row;
    5. its best installed tier (Premium > Enhanced > Personal > standard);
    6. None when nothing is installed for the language - callers then fail
       instead of speaking the text in a foreign voice.
    """
    if not lang:
        return None
    configured = configured_voice(lang, voices)
    if configured is not None:
        return configured
    in_lang = [v for v in installed_rows() if locale_score(lang, v.get("locale"))]
    if not in_lang:
        return None
    picked = system_selected_voice(lang)
    if picked is not None:
        return picked
    sel = selected_row()
    if sel is not None and locale_score(lang, sel.get("locale")):
        return sel
    in_lang.sort(
        key=lambda v: (
            not v.get("default"),
            _quality_rank(v.get("quality")),
            str(v.get("name") or ""),
        )
    )
    return in_lang[0]


def default_voice() -> Optional[dict[str, Any]]:
    """What this Mac speaks with when nothing asks for a language: the
    selected system voice when we can see it, else the system language's
    own answer, else any factory-flagged row. None when nothing can be
    named - the caller then addresses ``say`` with no voice at all."""
    sel = selected_row()
    if sel is not None:
        return sel
    sysl = system_language()
    if sysl:
        row = language_voice(sysl)
        if row is not None:
            return row
    for v in installed_rows():
        if v.get("default"):
            return v
    return None


_LOCALEISH_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


def looks_like_locale(value: Any) -> bool:
    """True when a voice field is really a language ('cs', 'cs-CZ', 'zh-Hans')
    and can therefore be routed instead of addressed."""
    return bool(_LOCALEISH_RE.match(str(value or "").strip()))


def split_model_token(model: Any) -> tuple[bool, str]:
    """(is_say, tail) for every spelling of the engine token: 'say' (bare -
    route by language), 'say:' and 'say:<tail>', where the tail is either an
    explicit voice (identifier or display name) or a locale ('say:cs-CZ'
    routes exactly like a request carrying language cs-CZ). The token is
    matched case-insensitively; the tail keeps its spelling because that is
    what :func:`resolve` must match. A tail like 'cs-CZ' is a language, not
    a voice name; a genuine three-letter voice name still wins because
    :func:`resolve` runs first."""
    s = str(model or "").strip()
    low = s.lower()
    if low == SAY_ENGINE:
        return True, ""
    if low.startswith(SAY_PREFIX):
        return True, s[len(SAY_PREFIX) :].strip()
    return False, ""


def route(
    voice: Any = None,
    language: Any = None,
    text: Any = None,
    voices: dict | None = None,
) -> str:
    """The address to speak with, or '' meaning 'run ``say`` with no -v and
    let macOS choose'. An explicit voice wins when it resolves to an
    installed voice; a locale-shaped voice routes like a language. With no
    voice, the language decides; with no language either, the TEXT does -
    macOS names the language of the text and the chain answers, and a shaky
    or unanswerable guess leaves the system default in place. A setting for
    the Mac's own language overrides the system default even for
    language-less requests: that is what the per-language dropdown promises.
    Raises LookupError with a user-safe reason when an EXPLICIT language has
    nothing installed."""
    voice = str(voice or "").strip()
    if voice:
        # A voice the caller NAMED wins over a language field beside it: they
        # asked for that voice, and a Slovak voice reading Czech text is what
        # `say -v Laura` has always done. A language written INTO the address
        # ('say:Laura:sk-SK') is a different promise - both halves were meant,
        # so a half that does not exist is refused instead of silently dropped.
        name, pinned = parse_voice_language(voice)
        row = resolve(voice, pinned)
        if row is not None:
            return address_of(row)
        if looks_like_locale(voice):
            return _by_language(voice, voices)
        if pinned and name and not looks_like_locale(name):
            langs = name_languages(name)
            if langs:
                shown = ", ".join(langs[:6]) + (", ..." if len(langs) > 6 else "")
                raise LookupError(
                    "voice '%s' is installed for %s, not for %s. Name one of "
                    "those languages, or address the voice by its full "
                    "identifier" % (name[:40], shown, pinned[:16])
                )
        raise LookupError("voice '%s' is not installed on this Mac" % voice[:64])
    if language:
        return _by_language(language, voices)
    detected = detect_language(text)
    if detected:
        row = language_voice(detected, voices)
        if row is not None:
            # A guess must never be WORSE than no guess: with nothing
            # installed for the detected language we fall through.
            return address_of(row)
    sysl = system_language()
    row = (configured_voice(sysl, voices) if sysl else None) or default_voice()
    return address_of(row) if row else ""


def _by_language(lang: Any, voices: dict | None) -> str:
    row = language_voice(lang, voices)
    if row is None:
        raise LookupError(
            "no installed macOS voice for language '%s' - choose one under "
            "Settings > Integrations > macOS Speech, or download a voice in "
            "System Settings" % str(lang)[:32]
        )
    return address_of(row)


# --- settings catalog ------------------------------------------------------


def clean_voices(mapping: Any) -> dict[str, str]:
    """The pairs a settings save may keep: a locale naming a voice this Mac
    cannot speak is dropped rather than stored, because routing ignores it
    anyway and a dead entry in the map only confuses the card."""
    out: dict[str, str] = {}
    if not isinstance(mapping, dict):
        return out
    for locale, voice in mapping.items():
        key = _norm_locale(locale)
        if not key or not isinstance(voice, str) or not voice.strip():
            continue
        if resolve(voice.strip()) is None:
            logger.info(
                "macOS speech: dropping voice '%s' for %s (not installed)",
                voice.strip()[:64],
                key,
            )
            continue
        out[key] = voice.strip()
    return out


def language_options(voices: dict | None = None) -> list[dict[str, Any]]:
    """One entry per locale this Mac has an installed voice for, for the
    settings card: the configured override, every voice to choose from, and
    what macOS itself answers with when nothing is configured.

    ``default.name`` is empty unless System Settings carries a real
    per-language pick for it - the true system default is not observable
    per language, so the UI says 'macOS default' rather than naming a voice
    it cannot prove. ``default.preference`` marks the names it CAN prove."""
    rows = installed_rows()
    locales = sorted(
        {_norm_locale(v.get("locale")) for v in rows if v.get("locale")},
        key=lambda s: s.lower(),
    )
    out: list[dict[str, Any]] = []
    for loc in locales:
        choices = [v for v in rows if locale_score(loc, v.get("locale"))]
        if not choices:
            continue
        choices.sort(
            key=lambda v: (_quality_rank(v.get("quality")), str(v.get("name")))
        )
        picked = system_selected_voice(loc)
        configured = configured_voice(loc, voices)
        out.append(
            {
                "locale": loc,
                "configured": address_of(configured) if configured else "",
                "default": {
                    "name": str(picked["name"]) if picked else "",
                    "locale": _norm_locale(picked["locale"]) if picked else "",
                    "preference": picked is not None,
                },
                "voices": [
                    {
                        "value": address_of(v),
                        "name": str(v["name"]),
                        "locale": _norm_locale(v.get("locale")),
                        "quality": v.get("quality") or "standard",
                    }
                    for v in choices
                ],
            }
        )
    return out


# --- synthesis -------------------------------------------------------------


def _speed_wpm(speed: Any) -> Optional[str]:
    try:
        mult = float(str(speed).strip() or "nan")
    except (TypeError, ValueError):
        return None
    if mult != mult or mult <= 0:  # NaN guard; junk means the engine default
        return None
    return str(max(80, min(450, round(_BASE_WPM * mult))))


def clean_chunk_chars(raw: Any) -> int:
    """Chunk size in characters: 0 switches segmentation off (one `say` call
    whatever the length), anything else lands in the useful range. Junk means
    the default, because a settings file with `null` in it is not a request to
    stop speaking."""
    if raw is None:
        return DEFAULT_CHUNK_CHARS
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_CHARS
    if value <= 0:
        return 0
    return max(MIN_CHUNK_CHARS, min(MAX_CHUNK_CHARS, value))


def max_audio_mb() -> int:
    """Ceiling for one reply, in MB of audio, from :data:`MAX_AUDIO_ENV`.
    Unset, blank, junk or 0 means the default - the guardrail can never be
    switched off, only moved. The environment is read per request, so an
    operator can raise it without touching a settings file or a restart."""
    raw = str(os.environ.get(MAX_AUDIO_ENV) or "").strip()
    try:
        value = int(float(raw))
    except ValueError:
        return DEFAULT_MAX_AUDIO_MB
    return value if value >= 1 else DEFAULT_MAX_AUDIO_MB


def min_audio_bytes(text_len: int) -> int:
    """The SMALLEST this much text can ever render to: the fastest speaking
    pace times the fixed byte rate. Slow voices only add bytes."""
    return text_len * SPEECH_BYTES_PER_SEC // FAST_SPEECH_CHARS_PER_SEC


def ceiling_refusal(text_len: int, ceiling_mb: int | None = None,
                    env: str = MAX_AUDIO_ENV) -> str | None:
    """Refuse the reply BEFORE rendering if it cannot possibly fit.

    Fail-or-pass up front is the contract: past the fast bound below, no
    voice on this Mac could keep the audio under the ceiling, so the answer
    comes immediately instead of an hour into a render. A reply under the
    bound may still cross it mid-render on a slow voice - that abort exists
    and is honest, but it is the exception the pre-check cannot see, not the
    design."""
    mb = max_audio_mb() if ceiling_mb is None else ceiling_mb
    if min_audio_bytes(text_len) > mb * 1_000_000:
        return (
            "input is %d characters: that is at least %d MB of audio even at "
            "the fastest speaking pace, over the %d MB ceiling of one spoken "
            "answer. Ask for a shorter reply, or set %s to raise the ceiling"
            % (text_len, min_audio_bytes(text_len) // 1_000_000 + 1,
               mb, env)
        )
    return None


def long_input_error(text_len: int, policy: Any = LONG_INPUT_SPEAK,
                     max_audio_mb_: int | None = None) -> str | None:
    """Why this utterance must not be rendered, or ``None`` to render it.

    Three limits, on purpose. OpenAI's ``/v1/audio/speech`` documents 4096
    characters and answers 400 past it; an operator who wants that exact
    behaviour picks ``reject``. Everyone else gets the whole utterance said:
    the router splits it at sentence boundaries and stitches the pieces, which
    costs about 0.9 s of setup per extra call (measured) and buys a bounded
    render and a first byte that does not wait for the last sentence. The
    absurd is refused here, and so is text that cannot fit the MB ceiling at
    ANY pace (:func:`ceiling_refusal`); the byte ceiling still does the real
    limiting, because characters say nothing about the size of the audio."""
    if text_len > MAX_SAY_CHARS:
        return (
            "input is %d characters; this engine speaks at most %d"
            % (text_len, MAX_SAY_CHARS)
        )
    if text_len > OPENAI_INPUT_CHARS and str(policy) == LONG_INPUT_REJECT:
        return (
            "input is %d characters; the OpenAI-compatible limit is %d. "
            "Shorten it, or set long text handling to 'speak it all' under "
            "Settings > Integrations > macOS Speech"
            % (text_len, OPENAI_INPUT_CHARS)
        )
    return ceiling_refusal(text_len, max_audio_mb_)


def synth_bytes(
    text: str,
    voice: Any = None,
    language: Any = None,
    voices: dict | None = None,
    speed: Any = None,
    long_input: Any = LONG_INPUT_SPEAK,
) -> bytes:
    """Render one utterance to WAV bytes. The text never reaches a shell.
    Addressing goes through :func:`route`, so an empty voice means 'macOS
    decides'. Raises LookupError with a user-safe reason when the caller
    asked for a voice or language this Mac cannot speak, ValueError for an
    empty or refused utterance, and RuntimeError when the synthesis itself
    fails. ``long_input`` is the operator's policy for text past OpenAI's
    character limit."""
    if not available():
        raise RuntimeError("macOS speech is not available on this server")
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to say")
    refused = long_input_error(len(text), long_input)
    if refused:
        raise ValueError(refused)
    # An addressing failure is the caller's fault (LookupError), a synthesis
    # failure is the machine's (RuntimeError): endpoints map them apart.
    return synth_wav(text, route(voice, language, text, voices), speed)


def synth_wav(text: str, address: Any = None, speed: Any = None) -> bytes:
    """Render one string with an ALREADY resolved voice address.

    :func:`synth_bytes` routes, which is right for a single utterance and
    wrong per segment: re-detecting the language sentence by sentence can hand
    one Czech answer to three different voices. The segmentation path routes
    once, then calls this for each piece. ``address`` empty means 'let macOS
    decide', exactly as :func:`route` returns when nothing answers.'"""
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to say")
    args = ["say"] + (["-v", str(address)] if address else [])
    wpm = _speed_wpm(speed)
    if wpm:
        args += ["-r", wpm]
    with tempfile.TemporaryDirectory(prefix="omlx-say-") as td:
        src = Path(td) / "in.txt"
        dst = Path(td) / "out.wav"
        src.write_text(text, encoding="utf-8")
        cmd = args + [
            "-f",
            str(src),
            "-o",
            str(dst),
            "--data-format=LEI16@%d" % _SAMPLE_RATE,
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise RuntimeError("macOS say timed out after %ds" % _TIMEOUT_S)
        except OSError as e:
            raise RuntimeError("macOS say failed to start: %s" % e.strerror)
        if p.returncode != 0 or not dst.is_file() or dst.stat().st_size < 100:
            reason = (p.stderr or "").strip()[:200] or "no audio produced"
            raise RuntimeError("macOS say: %s" % reason)
        return dst.read_bytes()
