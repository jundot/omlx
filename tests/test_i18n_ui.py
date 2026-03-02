# SPDX-License-Identifier: Apache-2.0
"""
Playwright E2E tests for Admin Dashboard i18n (language switching).

RED phase: these tests define expected behavior.
Run with: pytest tests/test_i18n_ui.py -v
Requires: omlx running on localhost:8079, OMLX_API_KEY env var set.
"""

import os
import pytest
from playwright.sync_api import sync_playwright, Page, expect


BASE_URL = "http://localhost:8079"
API_KEY = os.environ.get("OMLX_API_KEY", "test")


def login(page: Page) -> None:
    """Log in to the admin dashboard."""
    page.goto(f"{BASE_URL}/admin")
    page.wait_for_selector("input[type='password']", timeout=10000)
    page.fill("input[type='password']", API_KEY)
    page.click("button[type='submit']")
    page.wait_for_url(f"{BASE_URL}/admin/dashboard", timeout=10000)


def test_default_language_is_english():
    """Admin dashboard loads in English by default."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        login(page)
        go_to_settings(page)

        # Language section heading should appear in English
        expect(page.get_by_text("Language", exact=True).first).to_be_visible()

        browser.close()


LANG_SELECT = "select[x-model='globalSettings.ui.language']"
# Settings tab button has @click="mainTab = 'settings'" and a lucide settings icon
SETTINGS_TAB = "button[\\@click*=\"mainTab = 'settings'\"]"


def go_to_settings(page: Page) -> None:
    """Navigate to the Settings tab via the navbar button."""
    page.locator(SETTINGS_TAB).first.click()
    page.wait_for_selector(LANG_SELECT, timeout=8000)


def switch_language(page: Page, lang: str) -> None:
    """Select a language and wait for page reload."""
    page.select_option(LANG_SELECT, value=lang)
    page.wait_for_load_state("networkidle", timeout=10000)


def test_language_switcher_exists_in_settings():
    """Settings tab contains a language dropdown with en and zh options."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        login(page)
        go_to_settings(page)

        lang_select = page.locator(LANG_SELECT)
        expect(lang_select).to_be_visible()
        expect(page.locator(f"{LANG_SELECT} option[value='zh']")).to_have_count(1)

        browser.close()


def test_switch_to_chinese_reloads_with_chinese_ui():
    """Selecting Chinese in the language dropdown reloads page in Chinese."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        login(page)
        go_to_settings(page)
        switch_language(page, "zh")

        # After reload, navbar Status tab should show in Chinese
        expect(page.get_by_text("状态", exact=True).first).to_be_visible()

        # Cleanup
        go_to_settings(page)
        switch_language(page, "en")
        browser.close()


def test_switch_back_to_english():
    """After switching to Chinese, switching back to English restores English UI."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        login(page)
        go_to_settings(page)
        switch_language(page, "zh")

        go_to_settings(page)
        switch_language(page, "en")

        expect(page.get_by_text("Status", exact=True).first).to_be_visible()
        browser.close()


def test_language_preference_persists_after_reload():
    """Selected language persists after a manual page reload."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        login(page)
        go_to_settings(page)
        switch_language(page, "zh")

        page.reload()
        page.wait_for_load_state("networkidle", timeout=10000)

        # Chinese should still be active
        expect(page.get_by_text("状态", exact=True).first).to_be_visible()

        # Cleanup
        go_to_settings(page)
        switch_language(page, "en")
        browser.close()
