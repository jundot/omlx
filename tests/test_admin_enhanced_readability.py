# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Enhanced Readability accessibility toggle.

These are static-template assertions (no browser render, no server). They pin
the three behaviors the upstream reviewer asked for:

1. Cascade order  - the readability <style> block loads AFTER ``{% block head %}``
                    so it wins over theme variable declarations defined inside it.
2. KaTeX excluded - no rule sets ``.katex { font-size: ... }`` (formula scaling
                    (1.21em) must stay untouched).
3. Font-size floor - the floor is the scale's, so it is generated into
                    ``static/css/tokens.css`` (escaped selectors, applied to
                    everyone); what this file pins in the readability block is
                    the two relative-sized icon buttons it still lifts.

Plus invariants: gray helper text -> primary, red kept, model-card gray text is
also lifted to primary, no blanket disabled-text override, and the i18n key
exists in every locale (English fallback, not translated here).
"""

import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "omlx/admin/templates"
I18N = ROOT / "omlx/admin/i18n"

BASE = (TEMPLATES / "base.html").read_text(encoding="utf-8")
TOKENS_CSS = (ROOT / "omlx/admin/static/css/tokens.css").read_text(encoding="utf-8")
CHAT = (TEMPLATES / "chat.html").read_text(encoding="utf-8")
DASHBOARD_NAV = (TEMPLATES / "dashboard/_navbar.html").read_text(encoding="utf-8")
DASHBOARD_JS = (ROOT / "omlx/admin/static/js/dashboard.js").read_text(encoding="utf-8")


def _env():
    return Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)


def test_templates_compile():
    env = _env()
    env.get_template("base.html")
    env.get_template("chat.html")
    env.get_template("dashboard.html")


def test_cascade_after_block_head():
    idx_block = BASE.index("{% block head %}")
    idx_read = BASE.index("Enhanced Readability: global overrides")
    assert idx_read > idx_block, "readability block must load after {% block head %}"


def test_no_katex_font_size_override():
    bad = re.search(r"\.katex[^{]*\{[^}]*font-size", BASE)
    assert bad is None, f"KaTeX font-size override present: {bad.group(0)}"


def test_font_size_floor_lives_in_the_token_layer():
    """The floor is the scale's, not a readability switch's: it applies to
    everyone, so it is generated from tokens.json into tokens.css and base.html
    no longer repeats it. What stays here is only the pair of relative-sized
    icon buttons the toggle lifts."""
    for selector in (".text-\\[9px\\]", ".text-\\[10px\\]", ".text-\\[11px\\]"):
        assert selector in TOKENS_CSS, f"{selector} missing from the generated floor"
    assert '[style*="font-size: 10px"]' in TOKENS_CSS
    assert '[style*="font-size: 11px"]' in TOKENS_CSS
    assert "font-size: var(--fs-aux) !important" in TOKENS_CSS
    assert "[data-enhanced-readability] .text-\\[10px\\]" not in BASE, (
        "the floor is duplicated in base.html again"
    )
    assert ".code-copy-btn" in BASE
    assert ".svg-render-btn" in BASE
    assert "font-size: 12px !important" in BASE


def test_gray_text_mapped_to_primary():
    assert "--text-secondary: var(--text-primary) !important" in BASE
    assert ".text-neutral-500" in BASE
    assert ".text-fg-tertiary" in BASE


def test_red_kept_as_functional_red():
    assert "#d92d20 !important" in BASE
    assert "#ef5b54 !important" in BASE


def test_no_blanket_disabled_text_color_override():
    assert "[data-enhanced-readability] :disabled" not in BASE
    assert "[data-enhanced-readability] [disabled]" not in BASE


def test_model_card_gray_text_maps_to_primary():
    # The model-card markup container is no longer exempt; its gray (incl.
    # #475569, a common model-card gray) is lifted to primary via the gray-hex
    # mapping. Explicitly-colored markdown (links/code) is untouched because
    # those colors are not in the gray-hex list.
    assert "[data-enhanced-readability] .model-card-content" not in BASE, (
        "the model card is exempt from the readability mapping again"
    )
    assert '[style*="color: #475569"]' in BASE  # gray now mapped -> primary
    assert "Model card markdown keeps ORIGINAL" not in BASE


def test_no_jinja_block_in_comment():
    # The readability comment must NOT contain a literal {% block head %} that
    # Jinja would parse as a real block (would break the whole template -> 500).
    # Official base.html has exactly one `{% block head %}`; our comment says
    # "the head block" in plain words, so count stays 1.
    assert BASE.count("{% block head %}") == 1


def test_switch_below_theme_and_wired():
    # Switch sits BELOW the Theme section (user requirement).
    assert CHAT.index("chat.theme_label") < CHAT.index("chat.enhanced_readability")
    assert "enhancedReadability" in CHAT
    assert "setEnhancedReadability" in CHAT
    assert "setEnhancedReadability(!enhancedReadability)" in CHAT


def test_dashboard_theme_menu_controls_same_readability_setting():
    assert DASHBOARD_NAV.count("setEnhancedReadability(!enhancedReadability)") == 2
    assert "chat.enhanced_readability" in DASHBOARD_NAV
    assert '@focusin="themeDropdown = true"' in DASHBOARD_NAV
    assert ":aria-expanded=\"themeDropdown ? 'true' : 'false'\"" in DASHBOARD_NAV
    assert "enhancedReadability:" in DASHBOARD_JS
    assert "localStorage.getItem(ENHANCED_READABILITY_KEY) === 'on'" in DASHBOARD_JS
    assert (
        "localStorage.setItem(ENHANCED_READABILITY_KEY, enabled ? 'on' : 'off')"
        in DASHBOARD_JS
    )
    assert (
        "document.documentElement.setAttribute('data-enhanced-readability', '')"
        in DASHBOARD_JS
    )
    assert (
        "document.documentElement.removeAttribute('data-enhanced-readability')"
        in DASHBOARD_JS
    )


def test_shortcuts_at_top_of_chat_settings():
    # Keyboard-shortcuts button is the first item in the chat-settings panel.
    assert CHAT.index("chat.shortcuts") < CHAT.index("chat.attachments")


def test_i18n_key_present_in_all_locales():
    locales = ["en", "zh", "es", "fr", "ja", "ko", "pt-BR", "ru", "zh-TW"]
    for loc in locales:
        data = json.loads((I18N / f"{loc}.json").read_text(encoding="utf-8"))
        assert "chat.enhanced_readability" in data, f"{loc} missing key"
        label = data["chat.enhanced_readability"]
        assert label.strip(), f"{loc} empty label"
        # zh translates the label; every other locale keeps the English fallback
        # until its own translation lands.
        if loc != "zh":
            assert label == "Enhanced Readability", f"{loc} not English fallback"
        assert "chat.enhanced_readability_desc" in data
