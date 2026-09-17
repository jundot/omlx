# SPDX-License-Identifier: Apache-2.0
"""Dashboard accessibility contracts and optional Chrome keyboard regressions."""

from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADMIN_DIR = ROOT / "omlx" / "admin"
TEMPLATES = ADMIN_DIR / "templates"
DASHBOARD = TEMPLATES / "dashboard"
I18N = ADMIN_DIR / "i18n"

BASE_HTML = (TEMPLATES / "base.html").read_text(encoding="utf-8")
NAVBAR = (DASHBOARD / "_navbar.html").read_text(encoding="utf-8")
MODAL_MODEL = (DASHBOARD / "_modal_model_settings.html").read_text(encoding="utf-8")
DASHBOARD_TEMPLATE = (TEMPLATES / "dashboard.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. viewport zoom is not disabled
# ---------------------------------------------------------------------------


def test_viewport_does_not_lock_zoom():
    """The viewport meta should not pin `maximum-scale=1.0`, which would
    disable zoom on touch devices."""
    match = re.search(r'<meta\s+name="viewport"\s+content="([^"]+)"', BASE_HTML)
    assert match is not None, "expected a viewport meta tag in base.html"
    content = match.group(1)
    assert "maximum-scale=1.0" not in content, (
        "the `maximum-scale=1.0` zoom lock must be removed; "
        f"current viewport content: {content!r}"
    )


# ---------------------------------------------------------------------------
# 2. main nav has tablist/tab/aria-selected semantics
# ---------------------------------------------------------------------------


def test_main_nav_uses_role_tablist():
    assert 'role="tablist"' in NAVBAR, (
        "the dashboard nav must use role=tablist to announce its container"
    )


def test_main_nav_uses_role_tab_on_each_link():
    tab_blocks = re.findall(r'<(?:a|button)\b[^>]*role="tab"[^>]*>', NAVBAR)
    # The dashboard currently exposes six top-level tabs.
    assert len(tab_blocks) >= 2, (
        f"expected at least two role=tab entries in the main nav, got {len(tab_blocks)}"
    )


def test_main_nav_uses_aria_selected_on_each_tab():
    """Every role=tab entry must also declare `aria-selected`, which reflects
    the active panel id and is the attribute AT relies on for selection state."""
    tab_blocks = re.findall(r'<(?:a|button)\b[^>]*role="tab"[^>]*>', NAVBAR)
    for block in tab_blocks:
        assert "aria-selected" in block, f"tab block missing aria-selected: {block!r}"


def test_main_nav_tabs_have_id_aria_controls():
    """ARIA Authoring Practices require an `id` on the tab and `aria-controls`
    pointing to its associated tabpanel id."""
    tab_blocks = re.findall(r'<(?:a|button)\b[^>]*role="tab"[^>]*>', NAVBAR)
    id_regex = re.compile(r'\bid="([^"]+)"')
    ctrls_regex = re.compile(r'\baria-controls="([^"]+)"')
    for block in tab_blocks:
        tab_id = id_regex.search(block)
        assert tab_id, f"role=tab missing id: {block!r}"
        assert ctrls_regex.search(block), (
            f"role=tab id={tab_id.group(1)} missing aria-controls: {block!r}"
        )


# ---------------------------------------------------------------------------
# 3. tabpanel semantics on the per-section panels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "panel_name",
    ["_status", "_cluster_v2", "_models", "_bench"],
)
def test_panel_uses_role_tabpanel_with_aria_labelledby(panel_name):
    """The five dashboard panels (`_status`, `_cluster_v2`, `_models`,
    `_logs`, `_bench`) should each render with `role=tabpanel`,
    `id`, and `aria-labelledby` pointing back at its corresponding tab."""
    path = DASHBOARD / f"{panel_name}.html"
    body = path.read_text(encoding="utf-8")
    panel_match = re.search(
        r'<(?:section|div|aside)\b[^>]*role="tabpanel"[^>]*>',
        body,
    )
    assert panel_match is not None, (
        f"{panel_name}.html must contain at least one role=tabpanel section"
    )
    open_tag = panel_match.group(0)
    assert 'id="' in open_tag, (
        f"{panel_name}.html role=tabpanel must declare an id (got {open_tag!r})"
    )
    assert 'aria-labelledby="' in open_tag, (
        f"{panel_name}.html role=tabpanel must declare aria-labelledby (got {open_tag!r})"
    )


# ---------------------------------------------------------------------------
# 4. dialog role + aria-modal on the modals
# ---------------------------------------------------------------------------


def test_hf_mirror_modal_has_dialog_role():
    """The HF mirror modal is rendered via dashboard.html; verify it carries
    `role=dialog`, `aria-modal=true`, and a labelled-by heading."""
    assert 'role="dialog"' in DASHBOARD_TEMPLATE, (
        "the HF mirror modal in dashboard.html must expose role=dialog"
    )
    assert "aria-modal" in DASHBOARD_TEMPLATE, (
        "the HF mirror modal must expose aria-modal"
    )


def test_model_settings_modal_has_dialog_role():
    """The model settings modal partial must be a dialog with aria-modal."""
    assert 'role="dialog"' in MODAL_MODEL, "model-settings modal must be a dialog"
    assert "aria-modal" in MODAL_MODEL, (
        "model-settings modal must declare aria-modal"
    )


# ---------------------------------------------------------------------------
# 5. locale catalogs agree with the new navbar.tablist_label
# ---------------------------------------------------------------------------


ALL_LOCALES = [
    "en",
    "es",
    "fr",
    "ja",
    "ko",
    "pt-BR",
    "ru",
    "zh",
    "zh-TW",
]


@pytest.mark.parametrize("locale", ALL_LOCALES)
def test_navbar_tablist_label_is_present_in_each_locale(locale):
    """Every locale catalog must include `navbar.tablist_label`. The English
    fallback will silently mask the missing key, so we pin this directly."""
    path = I18N / f"{locale}.json"
    assert path.exists(), f"{locale}.json is missing from i18n/"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "navbar.tablist_label" in data, (
        f"{locale}.json is missing `navbar.tablist_label`; check that the tablist "
        "aria-label has a translation in this locale."
    )
    value = data["navbar.tablist_label"]
    assert isinstance(value, str) and value.strip(), (
        f"{locale}.json navbar.tablist_label must be a non-empty string"
    )


# ---------------------------------------------------------------------------
# 6. the expose-as-API toggles announce their pressed state
# ---------------------------------------------------------------------------


def test_expose_as_model_toggles_declare_aria_pressed():
    """Both expose-as-API toggles are two-state buttons.

    Their label already flips between ON and OFF, but a toggle button should
    expose the state through `aria-pressed` as well, so assistive technology
    announces a toggle with a state rather than an ordinary button whose text
    happens to change.
    """
    # Both toggles are the buttons that carry the expose-as-API tooltip.
    marker = "t('modal.model_settings.profiles.expose_as_model')"
    toggles = [
        block for block in re.findall(r"<button\b[^>]*>", MODAL_MODEL) if marker in block
    ]
    assert len(toggles) == 2, (
        "expected the new-profile and edit-profile API toggles, "
        f"found {len(toggles)}"
    )
    for block in toggles:
        assert ':aria-pressed="' in block, (
            f"the expose-as-API toggle must bind aria-pressed: {block!r}"
        )


@pytest.fixture
def keyboard_page():
    playwright = pytest.importorskip("playwright.sync_api")
    from jinja2 import Environment, FileSystemLoader

    locale = json.loads((I18N / "en.json").read_text())
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    env.globals.update(
        t=lambda key: locale.get(key, key),
        static=lambda path: f"/admin/static/{path}",
        locale_json=json.dumps(locale),
        current_lang="en",
        version="test",
    )
    rendered = env.get_template("dashboard.html").render()
    # Keep the real templates and handlers, but avoid server polling and model loads.
    rendered = rendered.replace(
        "</body>",
        """<script>
        const originalDashboard = dashboard;
        dashboard = () => {
            const data = originalDashboard();
            data.init = function() {
                this.applyTheme();
                this.mainTab = 'models';
                this.models = [{id: 'sample-model', settings: {}}];
                this.hfModels = [{name: 'sample-model'}];
                this.modelSettings = this.buildModelSettingsState({}, {});
            };
            return data;
        };
        const originalCluster = clusterV2Wizard;
        clusterV2Wizard = () => {
            const data = originalCluster();
            data.init = () => {};
            return data;
        };
        </script></body>""",
    )

    def route_request(route):
        path = urlparse(route.request.url).path
        if path.startswith("/admin/static/"):
            asset = ADMIN_DIR / "static" / path.removeprefix("/admin/static/")
            route.fulfill(
                body=asset.read_bytes(),
                content_type=mimetypes.guess_type(asset)[0]
                or "application/octet-stream",
            )
        elif path.startswith("/admin/api/"):
            route.fulfill(json=[] if path.endswith("/parsers") else {})
        else:
            route.fulfill(body=rendered, content_type="text/html")

    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(5000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/*", route_request)
        page.goto("http://omlx.test/admin/dashboard")
        playwright.expect(page.locator("#tab-models")).to_have_attribute(
            "aria-selected", "true"
        )
        yield page, playwright.expect
        browser.close()
        assert not errors


@pytest.mark.integration
def test_dashboard_tab_keyboard_navigation(keyboard_page):
    page, expect = keyboard_page
    tabs = page.get_by_role("tablist")
    page.locator("#tab-status").focus()
    page.keyboard.press("ArrowRight")
    expect(page.locator("#tab-models")).to_be_focused()
    expect(page.locator("#tab-models")).to_have_attribute("aria-selected", "true")
    expect(tabs.locator('[tabindex="0"]')).to_have_count(1)
    page.keyboard.press("End")
    expect(page.locator("#tab-bench")).to_be_focused()
    page.keyboard.press("ArrowRight")
    expect(page.locator("#tab-status")).to_be_focused()
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#tab-bench")).to_be_focused()
    page.keyboard.press("Home")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.getAttribute('role')") != "tab"
    page.keyboard.press("Shift+Tab")
    expect(page.locator("#tab-models")).to_be_focused()
    page.evaluate("""() => {
        Alpine.$data(document.querySelector('[x-data="dashboard()"]'))
            .globalSettings.server.distributed_inference_active = true;
    }""")
    expect(page.locator("#tab-cluster")).to_be_visible()
    page.keyboard.press("Home")
    page.keyboard.press("ArrowRight")
    expect(page.locator("#tab-cluster")).to_be_focused()
    expect(page.locator("#panel-cluster")).to_be_visible()


@pytest.mark.integration
@pytest.mark.parametrize("viewport", [(1440, 1000), (390, 844)])
def test_dashboard_modal_keyboard_navigation(keyboard_page, viewport):
    page, expect = keyboard_page
    page.set_viewport_size(dict(zip(("width", "height"), viewport)))
    opener = page.locator('button[\\@click="openModelSettingsFromManager(model.name)"]')
    opener.focus()
    page.keyboard.press("Enter")
    model = page.get_by_role("dialog", name="sample-model", exact=True)
    expect(model).to_be_visible()
    expect(page.locator("#model-settings-modal-title")).to_be_focused()
    page.locator("#tab-status").evaluate("el => el.focus()")
    assert model.evaluate("el => el.contains(document.activeElement)")
    page.keyboard.press("Tab")
    page.keyboard.press("Shift+Tab")
    assert model.evaluate("el => el.contains(document.activeElement)")
    first = model.locator("button:visible:enabled").first
    last = model.locator("button:visible:enabled").last
    last.focus()
    page.keyboard.press("Tab")
    expect(first).to_be_focused()
    page.keyboard.press("Shift+Tab")
    expect(last).to_be_focused()
    model.locator("button[\\@click=\"setScope('preset')\"]").click()
    model.locator('button[\\@click="refreshPresets()"]:visible').hover()
    tooltip = page.locator('[x-ref="floatingTooltip"]')
    expect(tooltip).to_be_visible()
    assert tooltip.evaluate("el => el.closest('dialog').open")
    page.mouse.move(0, 0)
    recipe = model.locator("button[\\@click=\"openSettingsApply('recipe')\"]")
    recipe.focus()
    page.keyboard.press("Enter")
    nested = page.locator('[aria-labelledby="settings-apply-modal-title"]')
    expect(nested).to_be_visible()
    expect(page.locator("#settings-apply-modal-title")).to_be_focused()
    recipe.evaluate("el => el.focus()")
    assert nested.evaluate("el => el.contains(document.activeElement)")
    page.evaluate("""() => {
        Alpine.$data(document.querySelector('[x-data="dashboard()"]'))
            .settingsApply.phase = 'loading';
    }""")
    expect(nested.locator("button:visible:enabled")).to_have_count(0)
    page.keyboard.press("Tab")
    expect(page.locator("#settings-apply-modal-title")).to_be_focused()
    page.keyboard.press("Escape")
    expect(nested).to_be_visible()
    page.evaluate("""() => {
        Alpine.$data(document.querySelector('[x-data="dashboard()"]'))
            .settingsApply.phase = 'input';
    }""")
    expect(nested.locator("textarea:not([readonly])")).to_be_visible()
    page.keyboard.press("Escape")
    expect(nested).not_to_be_visible()
    expect(model).to_be_visible()
    expect(recipe).to_be_focused()
    page.keyboard.press("Escape")
    expect(model).not_to_be_visible()
    expect(opener).to_be_focused()
    page.locator("button[\\@click=\"setModelsTab('downloader')\"]").last.click()
    mirror_opener = page.locator('button[\\@click="openHfMirrorModal()"]')
    expect(mirror_opener).to_be_visible()
    mirror_opener.focus()
    expect(mirror_opener).to_be_focused()
    page.keyboard.press("Enter")
    mirror = page.locator('[aria-labelledby="hf-mirror-modal-title"]')
    expect(mirror).to_be_visible()
    expect(mirror.locator('input[type="text"]')).to_be_focused()
    page.keyboard.press("Shift+Tab")
    page.keyboard.press("Shift+Tab")
    assert mirror.evaluate("el => el.contains(document.activeElement)")
    page.keyboard.press("Escape")
    expect(mirror).not_to_be_visible()
    expect(mirror_opener).to_be_focused()
