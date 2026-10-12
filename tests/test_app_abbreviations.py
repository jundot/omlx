"""PP, TG and TTFT are abbreviations, so the app spells them in capitals.

The same rule landed for the web console in #4451; this is the macOS app's
half. The catalogue is what ships, while the ``defaultValue:`` beside each
``String(localized:)`` key is what Xcode re-extracts on the next build — a
lowercase abbreviation there would overwrite the translation — so both the
catalogue and the app's string literals are swept.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "apps" / "omlx-mac" / "Resources" / "Localizable.xcstrings"
SOURCES = ROOT / "apps" / "omlx-mac" / "Sources"

LABEL_KEYS = (
    "bench.throughput.batch.col.pp_tps",
    "bench.throughput.batch.col.tg_tps",
    "bench.throughput.single.col.pp_tps",
    "bench.throughput.single.col.tg_tps",
    "bench.throughput.single.test_label",
    "bench.throughput.upload.row.context_length",
    "menubar.stats.avg_pp_speed",
    "menubar.stats.avg_tg_speed",
    "settings.apply.choose.group_pp",
    "settings.apply.choose.group_tg",
)

LOWERCASE = re.compile(r"\b(pp|tg|ttft)\b")
LITERAL = re.compile(r'(?:defaultValue|suffix):\s*"([^"]*)"')
COMMENT_START = "//"
# `\(pp)` is a Swift interpolation naming a local, not copy: the abbreviation
# being swept is the literal text around it.
INTERPOLATION = re.compile(r"\\\([^)]*\)")


def _catalog():
    return json.loads(CATALOG.read_text(encoding="utf-8"))["strings"]


def _values(entry):
    for locale, lo in (entry.get("localizations") or {}).items():
        value = (lo.get("stringUnit") or {}).get("value")
        if value:
            yield locale, value


def test_the_label_keys_are_capitalised():
    strings = _catalog()
    for key in LABEL_KEYS:
        entry = strings.get(key)
        assert entry is not None, f"the catalogue lost {key}"
        for locale, value in _values(entry):
            found = LOWERCASE.findall(value)
            assert not found, f"{key}/{locale} still spells {found} lowercase: {value!r}"


def test_the_ttft_columns_keep_the_abbreviation():
    strings = _catalog()
    for key in ("bench.throughput.single.col.ttft", "bench.throughput.batch.col.avg_ttft"):
        entry = strings.get(key)
        assert entry is not None, f"the catalogue lost {key}"
        for locale, value in _values(entry):
            assert "TTFT" in value, f"{key}/{locale} lost the abbreviation: {value!r}"


def test_the_app_sources_do_not_spell_them_lowercase():
    offenders = []
    for path in sorted(SOURCES.rglob("*.swift")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split(COMMENT_START, 1)[0]
            for value in LITERAL.findall(code):
                text = INTERPOLATION.sub("", value)
                found = LOWERCASE.search(text)
                if found:
                    offenders.append(f"{path.relative_to(ROOT)}:{number}: {value!r}")
    assert not offenders, "a source literal still spells an abbreviation lowercase:\n" + "\n".join(offenders)
