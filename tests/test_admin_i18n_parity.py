"""Parity checks for admin dashboard locale catalogs.

Every locale file must stay in sync with ``en.json``: identical key sets,
matching ``{placeholder}`` sets, and non-empty values, so the English
fallback never silently masks catalog drift.
"""

import json
import re
from pathlib import Path

_I18N_DIR = Path(__file__).resolve().parents[1] / "omlx" / "admin" / "i18n"
_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")


def _load_catalogs():
    catalogs = {}
    for path in sorted(_I18N_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            catalogs[path.stem] = json.load(f)
    return catalogs


def test_locale_catalogs_exist():
    catalogs = _load_catalogs()
    assert "en" in catalogs, "en.json baseline catalog is missing"
    assert len(catalogs) >= 2, "expected at least one translation catalog"


def test_locale_keys_match_english_baseline():
    en_keys = set(_load_catalogs()["en"])
    for locale, catalog in _load_catalogs().items():
        if locale == "en":
            continue
        missing = en_keys - set(catalog)
        extra = set(catalog) - en_keys
        assert not missing, f"{locale}.json missing keys: {sorted(missing)}"
        assert not extra, f"{locale}.json has extra keys: {sorted(extra)}"


def test_locale_placeholders_match_english_baseline():
    en = _load_catalogs()["en"]
    for locale, catalog in _load_catalogs().items():
        if locale == "en":
            continue
        for key, en_value in en.items():
            expected = set(_PLACEHOLDER_RE.findall(en_value))
            actual = set(_PLACEHOLDER_RE.findall(catalog[key]))
            assert actual == expected, (
                f"{locale}.json placeholder mismatch for {key!r}: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )


def test_locale_values_are_non_empty():
    for locale, catalog in _load_catalogs().items():
        for key, value in catalog.items():
            assert isinstance(value, str) and value.strip(), (
                f"{locale}.json has empty value for {key!r}"
            )
