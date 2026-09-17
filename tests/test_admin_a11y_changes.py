# SPDX-License-Identifier: Apache-2.0
"""Dashboard zoom and optional Chrome keyboard regression tests."""

import json
import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADMIN_DIR = ROOT / "omlx" / "admin"
TEMPLATES = ADMIN_DIR / "templates"
I18N = ADMIN_DIR / "i18n"


def test_viewport_does_not_lock_zoom():
    base = (TEMPLATES / "base.html").read_text(encoding="utf-8")
    viewport = re.search(r'<meta\s+name="viewport"\s+content="([^"]+)"', base)
    assert viewport is not None
    assert "maximum-scale" not in viewport.group(1)
    assert "user-scalable=no" not in viewport.group(1)


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
