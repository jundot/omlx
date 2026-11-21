# SPDX-License-Identifier: Apache-2.0
"""Dashboard request built by the web search test button."""

from pathlib import Path


def test_dashboard_posts_pending_max_results():
    root = Path(__file__).resolve().parents[1]
    javascript = (root / "omlx_web/static/js/dashboard.js").read_text()
    test_method = javascript.split("async testWebSearch()", 1)[1].split(
        "async saveLanguage", 1
    )[0]
    assert (
        "max_results: this.globalSettings.integrations.web_search_max_results"
        in test_method
    )


def test_dashboard_sends_keenable_key():
    root = Path(__file__).resolve().parents[1]
    javascript = (root / "omlx_web/static/js/dashboard.js").read_text()
    test_method = javascript.split("async testWebSearch()", 1)[1].split(
        "async saveLanguage", 1
    )[0]
    assert (
        "keenable_api_key: this.globalSettings.integrations.web_search_keenable_api_key"
        in test_method
    )
    assert (
        "web_search_keenable_api_key: this.globalSettings.integrations"
        ".web_search_keenable_api_key" in javascript
    )
    template = (root / "omlx_web/templates/dashboard/_settings.html").read_text()
    assert '<option value="keenable">' in template
