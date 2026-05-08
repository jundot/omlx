#!/usr/bin/env python3
"""
i18n_sync.py — generate / verify apps/omlx-mac/Resources/Localizable.xcstrings
from omlx/admin/i18n/*.json.

The macOS app uses its own (smaller) set of keys. This script defines the
mapping from native-app keys → admin-i18n keys; for each locale it writes
the translated string into the xcstrings localization. Keys without a
mapping fall back to the en source string in non-en locales — Xcode shows
those as "Missing" in the Strings catalog editor, which is the cue to add
them when copy lands.

Usage
    # Regenerate the catalog in place.
    apps/omlx-mac/Scripts/i18n_sync.py write

    # CI: exit non-zero if the on-disk catalog drifts from a fresh build.
    apps/omlx-mac/Scripts/i18n_sync.py check

The catalog format is xcstrings v1 (Xcode 15+). Schema:
https://developer.apple.com/documentation/xcode/localizing-and-varying-text-with-a-string-catalog
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- Paths -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
ADMIN_I18N_DIR = REPO_ROOT / "omlx" / "admin" / "i18n"
XCSTRINGS_PATH = (
    REPO_ROOT
    / "apps"
    / "omlx-mac"
    / "Resources"
    / "Localizable.xcstrings"
)

# --- Locale set ------------------------------------------------------------

# Map native-app locale identifier → admin-i18n filename (without `.json`).
# CFBundleLocalizations matches these identifiers.
LOCALES: dict[str, str] = {
    "en":      "en",
    "zh-Hans": "zh",
    "zh-Hant": "zh-TW",
    "ja":      "ja",
    "ko":      "ko",
}
SOURCE_LANGUAGE = "en"

# --- Native-key → admin-key mapping ---------------------------------------

# Native keys live in Swift code (Text("…"), LocalizedStringResource, etc).
# When a native key has a clean admin-i18n equivalent, list it on the right.
# Otherwise leave the right side as None and the script will use the literal
# English string from NATIVE_DEFAULTS as the source.
KEY_MAP: dict[str, str | None] = {
    # Sidebar
    "sidebar.server":           "navbar.tab.status",
    "sidebar.status":           "status.heading",
    "sidebar.logs":             "navbar.tab.logs",
    "sidebar.models":           "navbar.tab.models",
    "sidebar.downloads":        "navbar.dropdown.downloader",
    "sidebar.integrations":     None,
    "sidebar.security":         None,
    "sidebar.about":            None,
    "sidebar.group.server":     None,
    "sidebar.group.ai":         None,
    "sidebar.group.general":    None,

    # Server screen
    "server.hero.start":           None,
    "server.hero.stop":            None,
    "server.hero.restart":         None,
    "server.section.network":      None,
    "server.section.endpoints":    None,
    "server.section.logging":      None,
    "server.row.listen_address":   None,
    "server.row.port":             None,
    "server.row.max_concurrent":   None,
    "server.row.log_level":        None,

    # Status screen
    "status.session_stats":     "status.heading",
    "status.scope.session":     "status.scope_session",
    "status.scope.alltime":     "status.scope_alltime",
    "status.tile.total":        None,
    "status.tile.cached":       None,
    "status.tile.generation":   None,
    "status.tile.requests":     None,
    "status.updates.title":     None,
    "status.updates.check":     None,
    "status.updates.install":   None,

    # Common
    "common.cancel":            "status.clear_cancel",
    "common.save":              None,
    "common.copy":              None,
    "common.open":              None,
    "common.back":              None,
    "common.continue":          None,
    "common.delete":            None,
    "common.create":            None,
}

# Literal English copy for native keys. Source-of-truth in `en`. Keys missing
# from KEY_MAP also need an entry here so the en localization is non-empty.
NATIVE_DEFAULTS: dict[str, str] = {
    # Sidebar
    "sidebar.server":            "Server",
    "sidebar.status":            "Status",
    "sidebar.logs":              "Logs",
    "sidebar.models":            "Models",
    "sidebar.downloads":         "Downloads",
    "sidebar.integrations":      "Integrations",
    "sidebar.security":          "Security",
    "sidebar.about":             "About",
    "sidebar.group.server":      "Server",
    "sidebar.group.ai":          "AI",
    "sidebar.group.general":     "General",

    # Server screen
    "server.hero.start":         "Start Server",
    "server.hero.stop":          "Stop",
    "server.hero.restart":       "Restart",
    "server.section.network":    "Network",
    "server.section.endpoints":  "API Endpoints",
    "server.section.logging":    "Logging",
    "server.row.listen_address": "Listen Address",
    "server.row.port":           "Port",
    "server.row.max_concurrent": "Max Concurrent Requests",
    "server.row.log_level":      "Log Level",

    # Status screen
    "status.session_stats":      "Session Stats",
    "status.scope.session":      "Session",
    "status.scope.alltime":      "All Time",
    "status.tile.total":         "Total Tokens",
    "status.tile.cached":        "Cached",
    "status.tile.generation":    "Generation",
    "status.tile.requests":      "Requests",
    "status.updates.title":      "Updates",
    "status.updates.check":      "Check Now",
    "status.updates.install":    "Install & Restart",

    # Common
    "common.cancel":             "Cancel",
    "common.save":               "Save",
    "common.copy":               "Copy",
    "common.open":               "Open",
    "common.back":               "Back",
    "common.continue":           "Continue",
    "common.delete":             "Delete",
    "common.create":             "Create",
}


def load_admin(locale_filename: str) -> dict[str, str]:
    path = ADMIN_I18N_DIR / f"{locale_filename}.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_catalog() -> dict:
    """Return an xcstrings v1 dict. State-of-translation rules:

    * Source language entries always have `state: "translated"`.
    * Non-source locales get `state: "translated"` if the admin-i18n had a
      mapped value; otherwise the entry is omitted (Xcode treats absence
      as "Missing" — visible cue to the translator).
    """
    sources_by_locale = {
        loc: load_admin(filename) for loc, filename in LOCALES.items()
    }
    strings: dict[str, dict] = {}

    for native_key, admin_key in KEY_MAP.items():
        en_source = NATIVE_DEFAULTS.get(native_key, native_key)
        localizations: dict[str, dict] = {}

        for loc in LOCALES:
            if loc == SOURCE_LANGUAGE:
                value = en_source
            elif admin_key is None:
                value = None
            else:
                value = sources_by_locale.get(loc, {}).get(admin_key)

            if value is None:
                continue
            localizations[loc] = {
                "stringUnit": {
                    "state": "translated",
                    "value": value,
                }
            }

        strings[native_key] = {
            "extractionState": "manual",
            "localizations": localizations,
        }

    return {
        "sourceLanguage": SOURCE_LANGUAGE,
        "strings": strings,
        "version": "1.0",
    }


def write_catalog() -> None:
    catalog = build_catalog()
    XCSTRINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with XCSTRINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print(
        f"wrote {len(catalog['strings'])} keys × {len(LOCALES)} locales "
        f"→ {XCSTRINGS_PATH.relative_to(REPO_ROOT)}"
    )


def check_catalog() -> int:
    if not XCSTRINGS_PATH.exists():
        print(f"missing: {XCSTRINGS_PATH}", file=sys.stderr)
        return 1
    expected = build_catalog()
    actual = json.loads(XCSTRINGS_PATH.read_text(encoding="utf-8"))
    if expected == actual:
        print("ok: Localizable.xcstrings is up to date")
        return 0
    print(
        "drift: Localizable.xcstrings is out of date. "
        "Run `apps/omlx-mac/Scripts/i18n_sync.py write`.",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("write", "check"):
        print(__doc__, file=sys.stderr)
        return 1
    if sys.argv[1] == "write":
        write_catalog()
        return 0
    return check_catalog()


if __name__ == "__main__":
    sys.exit(main())
