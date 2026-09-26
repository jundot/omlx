# SPDX-License-Identifier: Apache-2.0
"""Structure of the Settings tab after it was rebuilt around the anchor rail.

The tab is one long scroll with three sub-tabs. This PR gives it an in-page
rail (section list, copy-anchor, restart/live badges) and an Appearance section
that drives the same theme state the navbar does.

These are static assertions over the template and the component stylesheet; the
rail's arithmetic is covered by tests/admin_settings_nav.test.cjs.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import jinja2
import pytest

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "omlx" / "admin"
SETTINGS_PATH = ADMIN / "templates" / "dashboard" / "_settings.html"
SETTINGS = SETTINGS_PATH.read_text(encoding="utf-8")
UI = (ADMIN / "templates" / "components" / "ui.html").read_text(encoding="utf-8")
NAVBAR = (ADMIN / "templates" / "dashboard" / "_navbar.html").read_text(encoding="utf-8")
DASHBOARD_JS = (ADMIN / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
NAV_JS = (ADMIN / "static" / "js" / "settings_nav.js").read_text(encoding="utf-8")
COMPONENTS_CSS = (ADMIN / "static" / "css" / "components.css").read_text(encoding="utf-8")
TOKENS_CSS = (ADMIN / "static" / "css" / "tokens.css").read_text(encoding="utf-8")
TOKENS = json.loads((ADMIN / "tokens.json").read_text(encoding="utf-8"))
EN = json.loads((ADMIN / "i18n" / "en.json").read_text(encoding="utf-8"))
LOCALES = sorted(path.stem for path in (ADMIN / "i18n").glob("*.json"))

# The sections the global sub-tab must expose, in rail order.
GLOBAL_SECTIONS = [
    "settings-language",
    "settings-appearance",
    "settings-auth",
    "settings-server",
    "settings-model",
    "settings-resource",
    "settings-cache",
    "settings-generation",
    "settings-mcp",
    "settings-usage",
    "settings-network",
    "settings-advanced",
]
SUBTABS = {
    "global": GLOBAL_SECTIONS,
    "integrations": ["settings-int-markitdown", "settings-int-websearch"],
    "models": ["settings-models"],
}
# Sections whose every value is applied by the save request itself.
LIVE_SECTIONS = [
    "language", "appearance", "auth", "model", "resource", "cache",
    "generation", "usage", "network",
]
# Rows the backend only picks up on the next start, with the evidence.
RESTART_ROWS = {
    "settings.resource.max_concurrent_requests": "# Apply scheduler settings (restart required)",
    "settings.resource.embedding_batch_size": "# Apply scheduler settings (restart required)",
    "settings.cache.hot_cache_only": "# MCP config path changes require restart",
    "settings.advanced.initial_cache_blocks": "requires restart)",
    "settings.advanced.distributed_inference_enabled": (
        "# Route exposure and Bonjour publication are fixed at process startup"
    ),
}


def _render() -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(ADMIN / "templates")), autoescape=True
    )
    env.globals.update(
        t=lambda key: EN.get(key, key), static=lambda path: path, version="0.0.0"
    )
    return env.get_template("dashboard/_settings.html").render()


def _registry(html: str) -> dict:
    match = re.search(
        r'<script type="application/json" id="settings-sections">(.*?)</script>',
        html,
        re.S,
    )
    assert match, "the rail registry is not in the rendered page"
    return json.loads(match.group(1))


# === The rail, its sections and their anchors ===


def test_settings_tab_uses_the_shared_measure():
    # Every page is measured by the wrapper in dashboard.html, which reads the
    # dashboard layout's width, so the settings tab carries no measure of its
    # own -- not the console's and not the form's.
    assert "page-narrow" not in SETTINGS
    assert "page-wide" not in SETTINGS
    assert "page-gutter" not in SETTINGS


def test_the_rail_lists_the_active_sub_tab():
    assert 'x-for="section in settingsSections"' in SETTINGS
    assert "settings-rail__item" in SETTINGS
    assert "settingsActiveSection === section.id" in SETTINGS, "the rail marks the current section"
    assert 'aria-current=' in SETTINGS
    assert "settingsGoToSection(section.id)" in SETTINGS


def test_every_registry_section_has_an_id_in_the_document():
    """A rail entry that does not resolve to an element scrolls nowhere."""
    html = _render()
    registry = _registry(html)
    documented = set(re.findall(r'id="(settings-[a-z-]+)"', html))
    documented.discard("settings-sections")
    missing = {section["id"] for sections in registry.values() for section in sections} - documented
    assert not missing, f"rail entries without a section in the document: {sorted(missing)}"


def test_the_registry_matches_the_expected_sub_tabs():
    registry = _registry(_render())
    assert list(registry) == ["global", "integrations", "models"]
    assert [section["id"] for section in registry["global"]] == GLOBAL_SECTIONS
    for tab, expected in SUBTABS.items():
        assert [section["id"] for section in registry[tab]] == expected


def test_every_section_card_is_a_ui_card():
    """Section cards reuse the shared spec instead of a second card markup."""
    # Twelve global sections plus the two integrations sections; the models
    # sub-tab is a table, not a card. 11 global sections use ui.card (the
    # Advanced one is collapsible and keeps its own section element).
    assert SETTINGS.count("{% call ui.card(") == 13
    for section in GLOBAL_SECTIONS:
        if section == "settings-advanced":
            continue
        assert f'id="{section}"' in SETTINGS
    assert 'class="settings-section" id="settings-advanced"' in SETTINGS
    assert "components/ui.html" in SETTINGS, "a partial that renders alone imports its macros"


def test_ui_card_accepts_a_stable_id():
    assert "{% macro card(" in UI and "id=None" in UI
    assert '{% if id %} id="{{ id }}"{% endif %}' in UI


def test_anchors_are_deep_linkable_from_a_hash():
    assert "settingsScrollToHash" in DASHBOARD_JS
    assert "SETTINGS_SECTION_ID" in DASHBOARD_JS
    # The hash decides the tab pair before the query parameters do.
    body = DASHBOARD_JS[DASHBOARD_JS.index("applyTabStateFromUrl() {"):][:320]
    assert "settingsScrollToHash(window.location.hash)" in body
    assert body.index("settingsScrollToHash") < body.index("URLSearchParams"), (
        "an explicit section anchor outranks the restored tab"
    )


def test_rail_scrolls_and_reflects_the_active_section():
    assert "@click.prevent=\"settingsGoToSection(section.id)\"" in SETTINGS
    assert "settingsWatchScroll()" in SETTINGS, "the rail follows the page"
    for helper in ("settingsScrollToSection", "settingsWatchScroll", "activeSection"):
        assert helper in DASHBOARD_JS or helper in NAV_JS
    assert "scrollIntoView" in DASHBOARD_JS
    assert "smooth" in DASHBOARD_JS
    # Reduced motion keeps the jump, without the animation.
    assert "prefers-reduced-motion" in DASHBOARD_JS


# === The rail is the list, nothing else ===


def test_the_rail_carries_no_heading_and_no_search_box():
    """A heading over the list and a box to filter it both competed with the
    list; the browser's own find already searches the page."""
    rail = SETTINGS[SETTINGS.index('<nav class="settings-rail"'):]
    rail = rail[:rail.index("</nav>")]
    assert "settings-rail__list" in rail
    assert "<input" not in rail, "the rail has no filter box"
    assert "settings-rail__title" not in rail and "settings-rail__title" not in COMPONENTS_CSS
    assert "settings-rail__empty" not in COMPONENTS_CSS
    for helper in ("settingsSearch", "settingsApplySearch", "settingsFirstMatch", "filterSections"):
        assert helper not in DASHBOARD_JS, helper
        assert helper not in NAV_JS, helper
    # The registry only carries what the rail renders.
    for sections in _registry(_render()).values():
        for section in sections:
            assert set(section) == {"id", "title", "badge"}, section


# === Copy anchor ===


def test_each_section_offers_a_copy_link_control():
    # Every card section carries one; the Advanced card is the collapsible
    # exception (no header) and the models sub-tab is a table, so neither has a
    # header to put the control in.
    expected = [s for s in GLOBAL_SECTIONS if s != "settings-advanced"] + SUBTABS["integrations"]
    for section in expected:
        assert f"settingsCopyAnchor('{section}')" in SETTINGS, section
    assert SETTINGS.count("settingsCopyAnchor(") == len(expected)
    assert "settings.sections.copy_link" in SETTINGS
    # The feedback reuses the existing catalogue keys.
    assert "window.t('app.copy')" in SETTINGS
    assert "window.t('app.copied')" in SETTINGS
    body = DASHBOARD_JS[DASHBOARD_JS.index("settingsCopyAnchor(sectionId) {"):][:400]
    assert "sectionAnchor" in body
    assert "copyToClipboard" in body


def test_the_copied_link_is_the_page_plus_anchor():
    assert "origin" in NAV_JS and "pathname" in NAV_JS
    assert "'#'" in NAV_JS or '"#"' in NAV_JS


# === Restart / live badges ===


def test_badges_come_from_the_shared_macros():
    assert "ui.badge(t('settings.global.restart_badge'), tone='orange')" in SETTINGS
    assert "ui.badge(t('settings.resource.live_badge'), tone='green', dot=True)" in SETTINGS


def test_live_sections_are_marked_live():
    registry = _registry(_render())
    badges = {section["id"]: section["badge"] for section in registry["global"]}
    for name in LIVE_SECTIONS:
        assert badges[f"settings-{name}"] == "live", f"{name} applies immediately"


def test_sections_that_mix_both_carry_no_section_badge():
    """Marking a mixed section either way would be a guess, so it carries none."""
    registry = _registry(_render())
    badges = {section["id"]: section["badge"] for section in registry["global"]}
    for name in ("server", "mcp", "advanced"):
        assert badges[f"settings-{name}"] == "none", name


def test_restart_only_rows_keep_their_own_badge():
    # One badge component and one wording everywhere: the review found "重启"
    # next to "需要重启" on neighbouring rows, and two rows wearing both.
    badge = "ui.badge(t('settings.global.restart_badge'), tone='orange')"
    assert badge in SETTINGS
    assert "restart_badge') }}</span>" not in SETTINGS, "no hand-written badge span left"
    for key in ("settings.resource.restart_badge", "settings.mcp.restart_badge",
                "settings.advanced.restart_badge"):
        assert f"{{{{ t('{key}') }}}}" not in SETTINGS, f"{key} is the same wording as the shared key"
    # One badge per restart-only row (the five the backend marks, plus the two
    # server rows and the memory guard), the header sentence, and the band the
    # section-badge macro renders. Two rows used to wear it twice.
    assert SETTINGS.count(badge) == 10
    for line in SETTINGS.splitlines():
        assert line.count(badge) <= 1


def test_the_restart_rows_are_the_ones_the_routes_call_restart_required():
    """The badge is derived from the backend, not from the section's name."""
    routes = (ROOT / "omlx" / "admin" / "routes.py").read_text(encoding="utf-8")
    assert "# Apply scheduler settings (restart required)" in routes
    assert "# MCP config path changes require restart; exposure changes are live." in routes
    assert "initial_cache_blocks: int | None = None  # Starting blocks (requires restart)" in routes
    assert (
        "# Route exposure and Bonjour publication are fixed at process startup," in routes
    )
    # And the values the sections are marked live for really do apply at runtime.
    for applied in (
        'runtime_applied.append("log_level")',
        'runtime_applied.append("chunked_prefill")',
        'runtime_applied.append("prefill_priority")',
        'runtime_applied.append("decode_fairness")',
        'runtime_applied.append("usage_history")',
        'runtime_applied.append("ui_language")',
        'runtime_applied.append("network")',
    ):
        assert applied in routes, applied


# === Appearance ===


def test_appearance_section_reuses_the_navbar_state():
    assert 'id="settings-appearance"' in SETTINGS
    assert "setTheme('{value}')" in SETTINGS
    assert "setEnhancedReadability(!enhancedReadability)" in SETTINGS
    assert 'ui.segmented(' in SETTINGS, "the theme is a segmented control"
    for value in ("auto", "light", "dark"):
        assert f"('{value}'," in SETTINGS, value
    # Same state and same storage as the navbar's theme menu.
    assert "setTheme(theme) {" in DASHBOARD_JS
    assert "localStorage.setItem(THEME_STORAGE_KEY, this.theme)" in DASHBOARD_JS
    assert "setEnhancedReadability(enabled) {" in DASHBOARD_JS
    assert "ENHANCED_READABILITY_KEY" in DASHBOARD_JS
    assert "setEnhancedReadability(!enhancedReadability)" in NAVBAR


def test_appearance_does_not_reimplement_the_readability_rules():
    """The switch drives base.html's existing [data-enhanced-readability] rules."""
    base = (ADMIN / "templates" / "base.html").read_text(encoding="utf-8")
    assert "[data-enhanced-readability]" in base
    assert "data-enhanced-readability" not in SETTINGS, (
        "the settings page must not add a second readability rule set"
    )
    assert "settings.sections.theme" in SETTINGS


# === The rail fits the form measure ===


def test_rail_and_content_fit_the_form_measure():
    layout = TOKENS["layout"]
    assert re.search(rf"--settings-rail-width:\s+{layout['railWidth']}px;", TOKENS_CSS)
    assert ".settings-layout {" in COMPONENTS_CSS
    body = COMPONENTS_CSS[COMPONENTS_CSS.index(".settings-layout {"):][:220]
    assert "var(--settings-rail-width)" in body
    assert "minmax(0, 1fr)" in body
    # 160pt rail + 24pt gap leaves most of the 760pt measure for the form.
    assert layout["railWidth"] + TOKENS["space"]["5"] < layout["formMaxWidth"] // 3


def test_the_stacking_query_matches_the_token():
    """A media condition cannot read a token, so the two are pinned together."""
    query = re.search(r"@media \(max-width: (\d+)px\)", COMPONENTS_CSS)
    assert query, "the rail needs a stacking rule for narrow windows"
    assert int(query.group(1)) == TOKENS["layout"]["railStackBelow"]
    assert re.search(
        rf"--settings-rail-stack-below:\s+{TOKENS['layout']['railStackBelow']}px;", TOKENS_CSS
    )


def test_rail_styling_is_token_only():
    block = COMPONENTS_CSS[COMPONENTS_CSS.index("=== Settings anchor rail ==="):]
    assert "var(--space-" in block and "var(--radius-" in block
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block), "literal colour in the rail styles"


# === Copy ===


def test_section_copy_is_translated_everywhere():
    for locale in LOCALES:
        catalogue = json.loads((ADMIN / "i18n" / f"{locale}.json").read_text(encoding="utf-8"))
        for key in ("settings.sections.label", "settings.sections.copy_link",
                    "settings.sections.appearance", "settings.sections.theme"):
            assert key in catalogue, f"{locale}.json is missing {key}"
            assert str(catalogue[key]).strip(), f"{locale}.json has an empty {key}"


# === The rail's runtime behaviour ===


def test_node_rail_contracts_pass():
    """The pure rail rules and the dashboard wiring, under node --test."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the settings rail tests")
    for name in (
        "tests/admin_settings_nav.test.cjs",
        "tests/admin_settings_rail.test.cjs",
    ):
        result = subprocess.run(
            [node, "--test", str(ROOT / name)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"{name}\n{result.stdout}\n{result.stderr}"


def test_the_model_table_status_chip_is_the_switch_s_own_box():
    """Green means running here as it does in the manager table's STATUS_TONES,
    and the chip in the Status column stands beside the pin switch, so it is the
    same control box: the shared `.badge` (the switch's height, and a
    two-character label makes it the switch's width) inside a flex cell. A
    hand-rolled chip was 52x23.5 next to a 40x20 switch, and an inline chip in a
    block cell sits ~1.8px below it — the flex cell is what puts both on one
    centre line. The box itself is pinned in tests/test_admin_components.py."""
    assert 'class="w-20 flex justify-center flex-shrink-0"' in SETTINGS
    assert 'class="badge badge--orange"' in SETTINGS          # loading
    assert 'class="badge badge--green"' in SETTINGS           # loaded (virtual)
    assert ':class="hover ? \'badge--red cursor-pointer\' : \'badge--green\'"' in SETTINGS
    assert ':class="hover ? \'badge--green cursor-pointer\' : \'\'"' in SETTINGS
    # Nothing in the Status column hand-rolls its chip any more.
    status_cell = SETTINGS[SETTINGS.index("<!-- Status"):SETTINGS.index("<!-- Pin Toggle -->")]
    assert "px-2 py-0.5 text-xs font-medium rounded" not in status_cell
