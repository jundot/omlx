"""The macOS app spells its token unit with a capital T — on a unit, not in prose.

``apps/omlx-mac/Resources/Localizable.xcstrings`` is the string that ships,
while the ``defaultValue:`` beside each ``String(localized:)`` key is what
Xcode re-extracts on the next build — a differently-cased unit there would
overwrite the translation. The ``suffix:`` on a numeric field is user-visible
copy that never reaches the catalogue at all, so it is swept alongside.

The capital belongs where the token is a **unit**: counted beside a number or
a placeholder, or standing on a label of its own — ``8192 Tok``,
``3 Tokens (Default)``, ``Output Tok/s``, ``Tokens applied to Context
Window``.

The lowercase form is kept everywhere the word is an ordinary noun in a
sentence — ``Penalize repeated tokens.``, ``Limit thinking tokens for
reasoning models.`` — and everywhere it names a credential or an
architecture term: the Hugging Face token, its validation subtitle, MTP's
multi-token prediction. Capitalising those reads as a different product
name; capitalising a mid-sentence common noun just reads as a mistake.

A case pass is easy to write and easy to overrun in either direction, so the
rules below are pinned against the catalogue and the sources.

The deliberate exceptions are code, not copy, and stay out of the pattern
because a token word only matches when it is delimited by non-word
characters:

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

# A capitalised token word, whole-word on purpose: "Tokens" counts, and so
# does the "Tok" of "Tok/s" and the "Tok" of "%lld Tok".
CAPITAL_TOKEN_WORD = re.compile(r"\b(Tok|Token|Tokens)\b")

# The only two literal positions that reach the user: the re-extraction
# default beside a catalogue key, and the unit suffix on a numeric field.
SOURCE_LITERAL = re.compile(r'(?:defaultValue|suffix):\s*"([^"]*)"')

# A subtitle or a note explains something in sentences. Past its first
# sentence opener the word is a common noun again and takes the lowercase
# form — this is the rule the case pass in 7da1a05 was missing.
PROSE_KEY = re.compile(r"(\.sub|\.note|_note)$")
SENTENCE_OPENERS = (" — ", ". ")

# The keys whose unit this rule pins. Each one is a count beside a number or
# a placeholder, or a label that names the unit on its own.
UNIT_KEYS = (
    "bench.context.result.tokens_label",
    "profile.detail.acceleration.specprefill.threshold",
    "profile.detail.behavior.thinking_budget.on",
    "profile.detail.capacity.tokens",
    "profile.detail.capacity.tokens.raw",
    "settings.acceleration.mtp.depth.adaptive",
    "settings.acceleration.mtp.depth.option",
    "status.usage.heatmap.cell",
    "status.usage.row.requests_speed",
    "status.usage.row.tokens",
)

# Russian transliterates the unit into Cyrillic — "тк", "токенов", "ток/с" —
# which is upstream's translation and not this rule's business. The capital is
# only asserted where the value still carries the Latin unit.
LATIN_UNIT = re.compile(r"\b(tok/s|t/s|tok|token|tokens|tk)\b", re.IGNORECASE)

# A credential is not a token count, and neither is an architecture term. The
# Hugging Face token, its validation subtitle and MTP's multi-token
# prediction keep the spelling upstream uses for all nine locales — the same
# one #4364 kept in the console's catalogues when it refreshed them. The
# catalogue sweep sees keys, so these are keyed; the source sweep sees only a
# `defaultValue:` literal and not the key it belongs to, so it matches on text.
LOWERCASE_CATALOG_KEYS = {
    "quant.upload_modal.token.label",
    "quant.upload_modal.credentials.subtitle.needs_validate",
    "quant.advanced.preserve_mtp.sub.available",
    # A group heading, not a count: "Best token generation (TG)".
    "settings.apply.choose.group_tg",
}
LOWERCASE_SOURCE_VALUES = {
    "HF token",
    "Validate a token to enable upload",
    "Keep multi-token prediction heads in the quantized output",
    "Best token generation (TG)",
}


def _locales_of(entry: dict) -> dict:
    return {
        locale: (raw.get("stringUnit") or {}).get("value")
        for locale, raw in (entry.get("localizations") or {}).items()
    }


def _past_the_opening(value: str) -> str:
    """The part of a prose value that follows its first sentence opener."""

    cuts = [
        value.index(opener) + len(opener)
        for opener in SENTENCE_OPENERS
        if opener in value
    ]
    return value[min(cuts) :] if cuts else ""


def test_the_unit_keys_carry_the_capital():
    strings = json.loads(CATALOG.read_text(encoding="utf-8"))["strings"]

    for key in UNIT_KEYS:
        assert key in strings, f"the catalogue is missing {key}"
        for locale, value in _locales_of(strings[key]).items():
            if not LATIN_UNIT.search(value or ""):
                continue  # translated into another script, or no unit at all
            message = f"{locale} {key} = {value!r} keeps the lowercase unit"
            assert CAPITAL_TOKEN_WORD.search(value), message


def test_english_prose_keeps_the_common_noun_lowercase():
    """A sentence says "tokens"; only a label says "Tokens"."""

    strings = json.loads(CATALOG.read_text(encoding="utf-8"))["strings"]
    offenders = []

    for key, entry in strings.items():
        if not PROSE_KEY.search(key) or key in LOWERCASE_CATALOG_KEYS:
            continue
        value = _locales_of(entry).get("en") or ""
        found = CAPITAL_TOKEN_WORD.search(_past_the_opening(value))
        if found:
            offenders.append(f"{key}: {found.group(0)!r} in {value!r}")

    report = "a prose value capitalises the common noun:\n" + "\n".join(offenders)
    assert not offenders, report


def test_the_credentials_and_architecture_terms_stay_lowercase():
    strings = json.loads(CATALOG.read_text(encoding="utf-8"))["strings"]

    for key in LOWERCASE_CATALOG_KEYS:
        assert key in strings, f"the catalogue is missing {key}"
        value = _locales_of(strings[key]).get("en")
        message = f"{key} = {value!r} capitalises a credential or an architecture term"
        assert not CAPITAL_TOKEN_WORD.search(value or ""), message


def test_the_source_literals_agree_with_the_catalogue():
    """A `defaultValue:` a build would re-extract must not disagree with what ships.

    The source sweep cannot see the key a literal belongs to, so it does not
    guess whether the word is a unit or a noun. It checks the one thing it can
    check: where a literal spells the same text as a value in the catalogue,
    it must spell it the same way, casing included. A build re-extracts the
    source, so a disagreement here would silently undo the translation.
    """

    strings = json.loads(CATALOG.read_text(encoding="utf-8"))["strings"]
    shipped = {}
    for entry in strings.values():
        for value in _locales_of(entry).values():
            if value:
                shipped.setdefault(value.lower(), set()).add(value)

    offenders = []
    for path in sorted(SOURCES.rglob("*.swift")):
        text = path.read_text(encoding="utf-8")
        for value in SOURCE_LITERAL.findall(text):
            forms = shipped.get(value.lower())
            if not forms or value in forms:
                continue
            where = f"{path.relative_to(ROOT)}"
            spelled = " or ".join(sorted(forms))
            offenders.append(
                f"{where}: {value!r} disagrees with the catalogue's {spelled!r}"
            )

    report = "a defaultValue:/suffix: spells a shipped value differently:\n"
    assert not offenders, report + "\n".join(offenders)


def test_the_token_word_detector_bites():
    # The rule is only worth anything while it fails every spelling it replaced.
    for sample in ["12 tok/s", "8192 tk", "HF token", "50 t/s", "Max tokens"]:
        assert LOWERCASE_TOKEN_WORD.search(sample), sample
    # ...and while the deliberate code-only exceptions stay out of the pattern.
    for sample in ["max_tokens", "hf_token", "tokenizer", "max_output_tokens"]:
        assert not LOWERCASE_TOKEN_WORD.search(sample), sample


def test_the_exemptions_are_still_needed():
    # The sweep is only honest while it could catch the values it skips: each
    # of them has to fail the detector, or the exemption is swallowing a
    # spelling this rule would otherwise accept by accident.
    strings = json.loads(CATALOG.read_text(encoding="utf-8"))["strings"]
    for key in LOWERCASE_CATALOG_KEYS:
        value = _locales_of(strings[key])["en"]
        assert LOWERCASE_TOKEN_WORD.search(value), key
        assert value in LOWERCASE_SOURCE_VALUES, key
