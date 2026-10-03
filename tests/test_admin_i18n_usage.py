# SPDX-License-Identifier: Apache-2.0
"""Every catalogue key the console asks for must exist.

``t()`` falls back to the key itself, so a missing entry reaches the screen as
``palette.group_pages`` instead of as text. The shared layer added in this
change is the first console code to ask for these strings, so the guard keeps a
later reference from shipping without its entry.
"""

import json
import re
from pathlib import Path

ADMIN = Path(__file__).resolve().parents[1] / "omlx" / "admin"
I18N = ADMIN / "i18n"
EN = json.loads((I18N / "en.json").read_text(encoding="utf-8"))

# The console's own sources: its templates and the scripts it ships.
TRANSLATED_SOURCES = sorted((ADMIN / "templates").rglob("*.html")) + sorted(
    (ADMIN / "static" / "js").glob("*.js")
)
KEY_CALL = re.compile(r"""(?:window\.)?\bt\(\s*'([^']*)'""")
# A key built at the call site ("status.layout.width_" + id) is invisible to the
# extractor above, which is exactly how a renamed key ships unnoticed: the
# template literal's static prefix has to resolve to at least one catalogue key.
KEY_TEMPLATE = re.compile(r"""(?:window\.)?\bt\(\s*`([^`$]*)""")


def _without_comments(text: str) -> str:
    text = re.sub(r"\{#.*?#\}", "", text, flags=re.S)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", text)


def _keys_used() -> set:
    keys = set()
    for path in TRANSLATED_SOURCES:
        keys.update(
            KEY_CALL.findall(_without_comments(path.read_text(encoding="utf-8")))
        )
    return keys


def _template_prefixes() -> set:
    prefixes = set()
    for path in TRANSLATED_SOURCES:
        text = _without_comments(path.read_text(encoding="utf-8"))
        prefixes.update(prefix for prefix in KEY_TEMPLATE.findall(text) if prefix)
    return prefixes


def test_every_template_literal_key_prefix_resolves():
    unresolved = sorted(
        prefix for prefix in _template_prefixes()
        if not any(key.startswith(prefix) for key in EN)
    )
    assert not unresolved, (
        "a key built from a template literal matches no catalogue key: "
        f"{unresolved}"
    )


def test_sources_actually_ask_for_keys():
    assert TRANSLATED_SOURCES, "the console's sources moved; this guard watches nothing"
    assert len(_keys_used()) > 100, "the key extraction found almost nothing"


def test_every_key_the_console_asks_for_exists():
    missing = sorted(key for key in _keys_used() if key not in EN)
    # A key built by concatenation ("usage." + label) shows up as its prefix.
    dynamic = [
        key for key in missing if any(existing.startswith(key) for existing in EN)
    ]
    leftovers = [key for key in missing if key not in dynamic]
    assert (
        not leftovers
    ), f"the console asks for keys English does not have: {leftovers}"
