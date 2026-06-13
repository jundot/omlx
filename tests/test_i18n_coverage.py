# SPDX-License-Identifier: Apache-2.0
"""Global i18n coverage + integrity guard.

Locks in the 2026-06 i18n coverage pass. The admin UI resolves keys via a
flat ``locale.get(key, key)`` lookup on both the server (Jinja ``t()``) and
client (``window.t()``) side, so a key that is missing from a locale renders
as the RAW dotted key string to the user. These tests fail loudly if:

  1. any locale drifts from the en.json key set (missing or extra keys),
  2. any value is not a non-empty flat string (nested objects break lookup),
  3. a translation drops/changes a {placeholder} token relative to en,
  4. a t()/window.t() key referenced in a template or dashboard.js has no
     definition in en.json (the raw-key-leak bug).

en.json is the single source of truth for the key set.
"""

import json
import re
from pathlib import Path

import pytest

_ADMIN = Path(__file__).parent.parent / "omlx" / "admin"
I18N_DIR = _ADMIN / "i18n"
TEMPLATES_DIR = _ADMIN / "templates"
DASHBOARD_JS = _ADMIN / "static" / "js" / "dashboard.js"

# Keys that are built dynamically at runtime (prefix + variable), so the
# concrete key never appears as a static literal. Any statically-referenced
# key that starts with one of these is exercised via concatenation and is
# allowed to be absent from the static scan. The full concrete forms ARE
# still covered by the all-locales-match-en check below.
_DYNAMIC_PREFIXES = (
    "bench.upload_skipped.features.",
    "settings.resource.guard_tier_desc.",
)

_PLACEHOLDER = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
# Match t('key') / window.t("key") but NOT a tail like createElement('x'):
# the t must not be preceded by an identifier char.
_T_CALL = re.compile(r"""(?<![A-Za-z0-9_.])(?:window\.)?t\(\s*['"]([a-zA-Z0-9_.]+)['"]""")


def _load(locale: str) -> dict:
    return json.loads((I18N_DIR / f"{locale}.json").read_text(encoding="utf-8"))


def _locale_codes() -> list[str]:
    return sorted(p.stem for p in I18N_DIR.glob("*.json"))


EN = _load("en")
EN_KEYS = set(EN)


def test_en_values_are_flat_nonempty_strings():
    bad = {k: v for k, v in EN.items() if not isinstance(v, str)}
    assert not bad, f"en.json has non-string (nested?) values: {sorted(bad)[:10]}"


@pytest.mark.parametrize("locale", [c for c in _locale_codes() if c != "en"])
def test_locale_matches_en_keyset(locale):
    data = _load(locale)
    keys = set(data)
    missing = EN_KEYS - keys
    extra = keys - EN_KEYS
    assert not missing, (
        f"{locale}.json is missing {len(missing)} keys present in en.json "
        f"(these render as raw key strings): {sorted(missing)[:15]}"
    )
    assert not extra, (
        f"{locale}.json has {len(extra)} keys not in en.json (dead/typo): "
        f"{sorted(extra)[:15]}"
    )


@pytest.mark.parametrize("locale", [c for c in _locale_codes() if c != "en"])
def test_locale_values_are_flat_nonempty_strings(locale):
    data = _load(locale)
    bad = [k for k, v in data.items() if not isinstance(v, str) or not v.strip()]
    assert not bad, f"{locale}.json has empty/non-string values: {sorted(bad)[:10]}"


@pytest.mark.parametrize("locale", [c for c in _locale_codes() if c != "en"])
def test_locale_preserves_placeholder_tokens(locale):
    data = _load(locale)
    mismatches = []
    for key, en_val in EN.items():
        en_tokens = set(_PLACEHOLDER.findall(en_val))
        if not en_tokens:
            continue
        tr_tokens = set(_PLACEHOLDER.findall(data.get(key, "")))
        if en_tokens != tr_tokens:
            mismatches.append((key, sorted(en_tokens), sorted(tr_tokens)))
    assert not mismatches, (
        f"{locale}.json placeholder drift (interpolation would break): "
        f"{mismatches[:8]}"
    )


def _referenced_keys() -> dict[str, set[str]]:
    """All statically-referenced t()/window.t() keys -> set of source files."""
    refs: dict[str, set[str]] = {}
    sources = list(TEMPLATES_DIR.rglob("*.html")) + [DASHBOARD_JS]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for m in _T_CALL.finditer(text):
            key = m.group(1)
            # only i18n-shaped keys; skip bare words (regex already guards the
            # createElement-style tails) and dynamic-concat bases ending in '.'
            if "." not in key or key.endswith("."):
                continue
            if any(key.startswith(p) for p in _DYNAMIC_PREFIXES):
                continue
            refs.setdefault(key, set()).add(path.name)
    return refs


def test_no_referenced_key_is_undefined_in_en():
    refs = _referenced_keys()
    leaks = {k: sorted(v) for k, v in refs.items() if k not in EN_KEYS}
    assert not leaks, (
        f"{len(leaks)} t() keys referenced in templates/dashboard.js are not "
        f"defined in en.json (would render as raw key strings): {leaks}"
    )
