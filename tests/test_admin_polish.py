# SPDX-License-Identifier: Apache-2.0
"""The console-wide finish: the ⌘K palette, the one toast, the skeletons, the
shortcut overview and the focus/keyboard wiring behind them.

Static assertions only — they read the templates, the stylesheet and the two
scripts. The palette's search ranking and the number transition are covered by
``tests/admin_polish.test.cjs``; the contrast of the colours these pieces use is
covered by ``tests/test_admin_contrast.py``.

The rule the shortcut overview is held to is "only what exists": every key the
modal lists has to have a handler in the console's own scripts, and the modal
must not list a key the console does not handle.
"""

import json
import re
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "omlx" / "admin"
TEMPLATES = ADMIN / "templates"
STATIC = ADMIN / "static"
I18N = ADMIN / "i18n"

BASE = (TEMPLATES / "base.html").read_text(encoding="utf-8")
DASHBOARD_TEMPLATE = (TEMPLATES / "dashboard.html").read_text(encoding="utf-8")
OVERLAYS = (TEMPLATES / "components" / "overlays.html").read_text(encoding="utf-8")
UI_MACROS = (TEMPLATES / "components" / "ui.html").read_text(encoding="utf-8")
STATUS = (TEMPLATES / "dashboard" / "_status.html").read_text(encoding="utf-8")
SETTINGS = (TEMPLATES / "dashboard" / "_settings.html").read_text(encoding="utf-8")
LOGS = (TEMPLATES / "dashboard" / "_logs.html").read_text(encoding="utf-8")
USAGE = (TEMPLATES / "dashboard" / "_usage.html").read_text(encoding="utf-8")
SERVING = (TEMPLATES / "dashboard" / "blocks" / "_serving_stats.html").read_text(
    encoding="utf-8"
)
ACTIVE = (TEMPLATES / "dashboard" / "blocks" / "_active_models.html").read_text(
    encoding="utf-8"
)
MODELS = (TEMPLATES / "dashboard" / "_models.html").read_text(encoding="utf-8")
COMPONENTS_CSS = (STATIC / "css" / "components.css").read_text(encoding="utf-8")
UI_JS = (STATIC / "js" / "ui.js").read_text(encoding="utf-8")
DASHBOARD_JS = (STATIC / "js" / "dashboard.js").read_text(encoding="utf-8")
USAGE_JS = (STATIC / "js" / "usage.js").read_text(encoding="utf-8")
EN = json.loads((I18N / "en.json").read_text(encoding="utf-8"))
LOCALES = sorted(path.stem for path in I18N.glob("*.json"))

KEY_CALL = re.compile(r"""(?:window\.)?\bt\(\s*'([^']*)'""")
DECLARED_CLASSES = set(re.findall(r"\.([a-z][a-z0-9_-]*)", COMPONENTS_CSS))
UTILITY_PREFIXES = ("w-", "h-", "flex", "items-", "gap-", "justify-")


def _render(page: str = "dashboard.html") -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    env.globals.update(
        t=lambda key: EN.get(key, key),
        static=lambda path: f"/admin/static/{path}",
        locale_json=json.dumps(EN),
        current_lang="en",
        version="0.0.0",
    )
    return env.get_template(page).render()


@pytest.fixture(scope="module")
def page() -> str:
    return _render()


# === the palette ===


def test_the_palette_is_global_and_the_dashboard_does_not_ship_a_second_one(page):
    assert 'include "components/overlays.html"' in BASE
    assert "js/ui.js" in BASE
    assert "ui.command_palette()" in OVERLAYS
    assert page.count('id="omlx-palette"') == 1


def test_the_palette_is_a_modal_dialog_with_a_combobox(page):
    assert 'id="omlx-palette" class="overlay-dialog"' in page
    assert 'role="dialog"' in page and 'aria-modal="true"' in page
    assert 'id="omlx-palette-input"' in page
    assert 'role="combobox"' in page
    assert 'aria-controls="omlx-palette-list"' in page
    assert 'aria-expanded="true"' in page
    assert 'id="omlx-palette-list"' in page and 'role="listbox"' in page
    assert 'id="omlx-palette-empty"' in page
    assert "palette__hint" in page


def test_the_palette_searches_the_tabs_blocks_and_settings_sections():
    for needle in (
        "paletteCommands",
        "window.omlxPalette.register",
        "DASHBOARD_TAB_ORDER",
        "this.setMainTab(tab)",
        "BLOCK_IDS",
        "this.dashPlaced(id)",
        "window.t(`status.layout.block.${id}`)",
        "DASHBOARD_SETTINGS_ORDER",
        "this.setSettingsTab(tab)",
        "paletteJumpToBlock",
    ):
        assert needle in DASHBOARD_JS, f"the palette does not reach {needle}"
    assert "navbar.tab.chat" in UI_JS, "the palette can jump to chat from any page"


def test_the_palette_keys_are_handled():
    for needle in (
        "key === 'k' || key === 'K'",
        "event.key === 'ArrowDown'",
        "event.key === 'ArrowUp'",
        "event.key === 'Enter'",
        "togglePalette",
    ):
        assert needle in UI_JS, needle


def test_a_slash_only_opens_the_palette_outside_a_text_field():
    assert "function isTypingTarget(target)" in UI_JS
    assert "key === '/' && !mod" in UI_JS
    assert "if (isTypingTarget(event.target)) return;" in UI_JS


# === the shortcut overview ===


def _overview_keys(page: str) -> list:
    overview = page[page.index('id="omlx-shortcuts"') :]
    overview = overview[: overview.index("</dialog>")]
    return re.findall(r'<kbd class="kbd">([^<]*)</kbd>', overview)


# Every key the overview lists, and where the console handles it.
HANDLED_KEYS = {
    "⌘K": ("ui.js", "key === 'k' || key === 'K'"),
    "/": ("ui.js", "key === '/' && !mod"),
    "⌘/": ("ui.js", "mod && key === '/'"),
    "?": ("ui.js", "key === '?'"),
    "↑": ("ui.js", "'ArrowUp'"),
    "↓": ("ui.js", "'ArrowDown'"),
    "↵": ("ui.js", "'Enter'"),
    "Esc": ("ui.js", "addEventListener('cancel'"),
    "←": ("dashboard.js", "'ArrowLeft'"),
    "→": ("dashboard.js", "'ArrowRight'"),
}


def test_the_shortcut_overview_lists_exactly_the_handled_keys(page):
    keys = _overview_keys(page)
    assert keys, "the console has no shortcut overview"
    unknown = [key for key in keys if key not in HANDLED_KEYS]
    assert not unknown, f"the overview lists keys nothing handles: {unknown}"
    missing = [key for key in HANDLED_KEYS if key not in keys]
    assert not missing, f"the overview hides keys the console handles: {missing}"


@pytest.mark.parametrize("key", sorted(HANDLED_KEYS))
def test_every_listed_key_has_a_handler(key):
    source, needle = HANDLED_KEYS[key]
    text = UI_JS if source == "ui.js" else DASHBOARD_JS
    assert needle in text, f"{key} is listed as {source}: {needle}"


def test_the_overview_is_a_modal_dialog_with_a_close_button(page):
    assert 'id="omlx-shortcuts" class="overlay-dialog"' in page
    assert 'aria-labelledby="omlx-shortcuts-title"' in page
    assert "shortcut-list__row" in page
    assert "data-shortcuts-close" in OVERLAYS or "data-shortcuts-close" in UI_MACROS
    # Chat ships its own list, so there is exactly one overview on this page.
    assert page.count('id="omlx-shortcuts"') == 1


# === toasts ===


def test_there_is_one_toast_stack_for_the_console(page):
    assert "ui.toast_host()" in OVERLAYS
    assert 'id="omlx-toast-stack"' in page
    assert 'class="toast-stack"' in page
    assert "global.omlxToast = omlxToast" in UI_JS


def test_the_toast_host_is_not_itself_a_live_region(page):
    host = page[page.index('id="omlx-toast-stack"') - 40 :]
    host = host[: host.index(">")]
    assert "aria-live" not in host, (
        "each toast carries its own role, so a live container would announce "
        "every message twice"
    )
    assert "setAttribute('aria-live', semantics.ariaLive)" in UI_JS


def test_toast_tones_match_the_badge_tones():
    for tone in ("green", "orange", "red", "blue", "neutral"):
        assert f".toast--{tone}" in COMPONENTS_CSS, tone
        if tone != "neutral":
            assert f"badge--{tone}" in COMPONENTS_CSS
    for tone in ("green", "orange", "red", "blue", "neutral"):
        assert f"'{tone}'" in UI_JS


def test_toasts_dismiss_themselves_and_can_be_closed_by_hand():
    assert (
        "TOAST_DURATION" in UI_JS
        and "setTimeout(function () { closeToast(entry); }" in UI_JS
    )
    assert (
        "pauseToast(entry)" in UI_JS
        and "'mouseenter'" in UI_JS
        and "'mouseleave'" in UI_JS
    )
    assert "toast__close" in UI_JS
    assert "element.setAttribute('role', semantics.role)" in UI_JS


def test_the_four_page_local_notices_are_gone():
    # The state stays in the scripts; only the rendering moved to the toast.
    assert 'x-text="restartServer.message"' not in STATUS
    assert 'restartServer.status !== "idle"' not in STATUS.replace("'", '"')
    assert 'x-text="saveError"' not in SETTINGS
    assert 'x-text="restartServer.message"' not in SETTINGS
    assert 'x-text="logError"' not in LOGS
    assert 'x-show="error"' not in USAGE
    for toast_id in ("'server-restart'", "'settings-save'", "'log-load'"):
        assert toast_id in DASHBOARD_JS, toast_id
    assert "'usage-error'" in USAGE_JS


def test_the_notice_state_still_drives_the_toasts():
    for watcher in (
        "this.$watch('restartServer.status'",
        "this.$watch('saveError'",
        "this.$watch('logError'",
    ):
        assert watcher in DASHBOARD_JS, watcher
    assert "this.$watch('error'" in USAGE_JS
    # the toast tone follows the restart state machine
    assert "RESTART_TONES[status] || 'neutral'" in DASHBOARD_JS


# === skeletons ===


def test_the_skeleton_macros_exist_and_carry_no_copy():
    for macro in (
        "skeleton_kpis",
        "skeleton_rows",
        "command_palette",
        "toast_host",
        "shortcut_overview",
    ):
        assert f"{{% macro {macro}(" in UI_MACROS, macro
    body = UI_MACROS[
        UI_MACROS.index("{% macro skeleton_kpis(") : UI_MACROS.index(
            "{% macro command_palette("
        )
    ]
    assert 'aria-hidden="true"' in body
    assert "{{ t(" not in body, "a skeleton must not invent text either"


def test_the_first_paint_of_the_status_tab_uses_skeletons():
    assert 'x-show="!statsLoaded"' in SERVING
    assert "ui.skeleton_kpis(4)" in SERVING
    assert 'x-show="statsLoaded" x-cloak' in SERVING
    assert 'x-show="!statsLoaded"' in ACTIVE
    assert "ui.skeleton_rows(3)" in ACTIVE
    assert "ui.skeleton_rows(5, variant='table')" in USAGE
    assert "ui.skeleton_rows(5, variant='table')" in MODELS
    assert 'x-show="loadingModels && hfModels.length === 0"' in MODELS


def test_real_content_replaces_the_skeleton_and_nothing_is_faked():
    assert "this.statsLoaded = true;" in DASHBOARD_JS
    assert DASHBOARD_JS.count("this.statsLoaded = true;") == 2, (
        "the flag is set on the payload and on a failed attempt, so the shimmer "
        "can never outlive the request"
    )
    # The empty states wait for the payload instead of claiming "nothing here".
    assert 'x-show="statsLoaded && (!stats.active_models.models' in ACTIVE
    assert 'x-if="hfModels.length === 0 && !loadingModels"' in MODELS


def test_the_skeleton_shimmer_is_reduced_motion_aware():
    block = COMPONENTS_CSS[
        COMPONENTS_CSS.index("@media (prefers-reduced-motion: reduce)") :
    ]
    assert ".skeleton::after" in block and "animation: none;" in block
    assert "--duration-shimmer" in COMPONENTS_CSS
    assert "@keyframes skeleton-sheen" in COMPONENTS_CSS


# === the number transition ===


def test_the_kpi_figures_animate_through_the_shared_helper():
    assert SERVING.count('x-effect="window.omlxCountUp($el, kpiValue(') == 6
    assert "toFixed(1)" not in SERVING, "the figure is animated, not re-printed"
    assert "kpiValue(field)" in DASHBOARD_JS
    assert "global.omlxCountUp = omlxCountUp" in UI_JS
    assert "requestAnimationFrame" in UI_JS
    assert "function prefersReducedMotion()" in UI_JS
    assert "prefersReducedMotion()" in UI_JS[UI_JS.index("function omlxCountUp(") :]


# === focus, keyboard and reduced motion ===


def test_both_overlays_trap_focus_and_restore_it():
    assert "function trapFocus(event)" in UI_JS
    assert "if (event.key !== 'Tab') return;" in UI_JS
    assert "previousFocus = global.document.activeElement" in UI_JS
    assert "addEventListener('close', restoreFocus)" in UI_JS
    assert "showModal()" in UI_JS
    assert "wireDialog" in UI_JS


def test_the_new_controls_are_buttons_and_keep_the_focus_ring():
    assert "createElement('button')" in UI_JS, "palette items are real buttons"
    assert 'class="overlay-close"' in UI_MACROS
    assert "outline: none" not in COMPONENTS_CSS, "the token layer owns the focus ring"
    assert ":focus-visible" in BASE


def test_the_toast_transition_is_reduced_motion_aware():
    block = COMPONENTS_CSS[
        COMPONENTS_CSS.index("@media (prefers-reduced-motion: reduce)") :
    ]
    assert ".toast {" in block and "transition: none;" in block
    assert "prefersReducedMotion()" in UI_JS[UI_JS.index("function closeToast(") :]


def test_the_new_classes_are_declared_in_the_shared_stylesheet():
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    env.globals.update(t=lambda key: EN.get(key, key))
    ui = env.get_template("components/ui.html").module
    markup = "".join(
        [
            str(ui.skeleton_kpis(2)),
            str(ui.skeleton_rows(2)),
            str(ui.skeleton_rows(2, variant="table")),
            str(ui.command_palette()),
            str(ui.toast_host()),
            str(ui.shortcut_overview([("A shortcut", ["⌘K"])])),
        ]
    )
    unknown = set()
    for value in re.findall(r'(?<![:\w-])class="([^"]*)"', markup):
        for name in value.split():
            if name not in DECLARED_CLASSES and not name.startswith(UTILITY_PREFIXES):
                unknown.add(name)
    assert not unknown, f"classes without a spec in components.css: {sorted(unknown)}"


# === copy ===


def test_every_key_the_console_scripts_ask_for_exists():
    missing = sorted(key for key in set(KEY_CALL.findall(UI_JS)) if key not in EN)
    assert (
        not missing
    ), f"static/js/ui.js asks for keys English does not have: {missing}"


def test_the_new_keys_are_translated_in_every_locale():
    new_keys = sorted(
        key for key in EN if key.startswith(("palette.", "shortcuts.", "toast."))
    )
    assert len(new_keys) == 25, new_keys
    for locale in LOCALES:
        catalogue = json.loads((I18N / f"{locale}.json").read_text(encoding="utf-8"))
        for key in new_keys:
            assert str(catalogue.get(key, "")).strip(), f"{locale}.json: {key}"
        if locale not in ("en",):
            assert all(
                catalogue[key] != EN[key] for key in new_keys
            ), f"{locale}.json still carries English for a new key"


# === every dialog traps focus ===

DIALOG = re.compile(r"<dialog\b[^>]*>", re.S)


def test_every_dialog_is_modal_and_traps_focus():
    """The three page dialogs use the dashboard's own Tab handler; the two new
    overlays are wired to the same behaviour from static/js/ui.js."""
    tags = []
    for path in sorted(TEMPLATES.rglob("*.html")):
        # Comments mention <dialog> when they explain the pattern they follow.
        text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
        for tag in DIALOG.findall(text):
            tags.append((path.relative_to(TEMPLATES), tag))
    assert len(tags) >= 5, [str(path) for path, _ in tags]
    for path, tag in tags:
        assert 'aria-modal="true"' in tag, f"{path}: a dialog without aria-modal"
        assert (
            "trapDialogFocus" in tag or "overlay-dialog" in tag
        ), f"{path}: a dialog that does not trap focus"
    assert "function trapFocus(event)" in UI_JS
    assert (
        "keydown', trapFocus" in UI_JS
        or "addEventListener('keydown', trapFocus)" in UI_JS
    )


def test_the_overlays_restore_focus_on_close():
    assert (
        UI_JS.count("previousFocus = global.document.activeElement") == 2
    ), "both overlays remember where focus came from"
    assert "addEventListener('close', restoreFocus)" in UI_JS
    assert "if (target && global.document.contains(target)" in UI_JS


def test_the_palette_uses_the_same_block_registry_as_the_grid():
    """dashboard.js takes its block commands from DashboardLayout.BLOCK_IDS while
    the grid renders the list from _status.html; an id in one and not the other
    would be a palette entry that jumps nowhere."""
    layout = (STATIC / "js" / "dashboard_layout.js").read_text(encoding="utf-8")
    block_list = layout[layout.index("const BLOCK_IDS = [") :]
    layout_ids = re.findall(r"'([a-z_]+)'", block_list[: block_list.index("]")])
    grid_ids = re.findall(r"\('([a-z_]+)', '_[a-z_]+\.html'\)", STATUS)
    assert grid_ids, "no block registry found in _status.html"
    assert layout_ids == grid_ids, (layout_ids, grid_ids)
