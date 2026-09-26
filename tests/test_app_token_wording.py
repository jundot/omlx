"""The macOS app's token unit spells its T wherever a user can read it.

``apps/omlx-mac/Resources/Localizable.xcstrings`` is the string that ships,
while the ``defaultValue:`` beside each ``String(localized:)`` key is what
Xcode re-extracts on the next build — a lower-case unit there would overwrite
the translation. The ``suffix:`` on a numeric field is user-visible copy that
never reaches the catalogue at all, so it is swept alongside. A lower-case
``tok/s``, ``t/s``, ``tok``, ``token``, ``tokens`` or ``tk`` fails the rule.

The deliberate exceptions are code, not copy, and stay out of the pattern
because a token word only matches when it is delimited by non-word characters:

* API / field names — ``max_tokens``, ``hf_token`` — the leading underscore
  means there is no word boundary before ``token``;
* the ``tokenizer`` component — the trailing ``i`` means ``token`` is not
  delimited either;
* ``hf_…`` placeholders — underscores again.

This mirrors the Swift guards in
``apps/omlx-mac/Tests/oMLXTests/LocalizationSmokeTests.swift``. CI only runs
pytest, never xcodebuild, so the mechanical rule lives here as well and the
Swift test keeps the same wording for the call sites it can see.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "apps" / "omlx-mac" / "Resources" / "Localizable.xcstrings"
SOURCES = ROOT / "apps" / "omlx-mac" / "Sources"

# Word-delimited, case-sensitive: `max_tokens`, `hf_token` and `tokenizer`
# never match, while a bare unit or `token`/`tokens`/`tok`/`tk` does.
LOWERCASE_TOKEN_WORD = re.compile(r"\b(tok/s|t/s|tok|token|tokens|tk)\b")

# The only two literal positions that reach the user: the re-extraction
# default beside a catalogue key, and the unit suffix on a numeric field.
SOURCE_LITERAL = re.compile(r'(?:defaultValue|suffix):\s*"([^"]*)"')


def _catalog_offenders():
    """Every locale value in the catalogue with a lower-case token word."""

    strings = json.loads(CATALOG.read_text(encoding="utf-8"))["strings"]
    for key, entry in strings.items():
        for locale, raw in (entry.get("localizations") or {}).items():
            value = (raw.get("stringUnit") or {}).get("value")
            match = LOWERCASE_TOKEN_WORD.search(value or "")
            if match:
                yield f"{locale} {key}: {match.group(0)!r} in {value!r}"


def _source_offenders():
    """Every `defaultValue:`/`suffix:` in the app sources with a lower-case unit."""

    for path in sorted(SOURCES.rglob("*.swift")):
        text = path.read_text(encoding="utf-8")
        for value in SOURCE_LITERAL.findall(text):
            match = LOWERCASE_TOKEN_WORD.search(value)
            if match:
                yield f"{path.relative_to(ROOT)}: {match.group(0)!r} in {value!r}"


def test_catalog_token_words_are_capitalised():
    offenders = list(_catalog_offenders())
    assert not offenders, "lower-case token words in the catalogue:\n" + "\n".join(
        offenders
    )


def test_source_token_words_are_capitalised():
    offenders = list(_source_offenders())
    assert not offenders, (
        "lower-case token words in a defaultValue:/suffix: in the app sources:\n"
        + "\n".join(offenders)
    )


def test_the_token_word_detector_bites():
    # The rule is only worth anything while it fails every spelling it replaced.
    for sample in ["12 tok/s", "8192 tk", "HF token", "50 t/s", "Max tokens"]:
        assert LOWERCASE_TOKEN_WORD.search(sample), sample
    # ...and while the deliberate code-only exceptions stay out of the pattern.
    for sample in ["max_tokens", "hf_token", "tokenizer", "max_output_tokens"]:
        assert not LOWERCASE_TOKEN_WORD.search(sample), sample
