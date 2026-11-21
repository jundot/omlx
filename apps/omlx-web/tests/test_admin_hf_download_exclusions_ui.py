# SPDX-License-Identifier: Apache-2.0
"""UI contract for per-download HuggingFace exclusion patterns."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUSION_I18N_KEYS = {
    "models.downloader.exclude_patterns",
    "models.downloader.exclude_patterns_hint",
}


def test_download_form_exposes_exclusion_patterns_input():
    template = (ROOT / "omlx_web/templates/dashboard/_models.html").read_text()
    assert 'x-model="hfIgnorePatterns"' in template

    script = (ROOT / "omlx_web/static/js/dashboard.js").read_text()
    start = script.index("async submitHFDownload(files)")
    body = script[start : script.index("async loadHFTasks()", start)]
    assert "this.hfIgnorePatterns" in body
    assert "ignore_patterns: ignorePatterns," in body


def test_exclusion_patterns_i18n_keys_exist_in_every_locale():
    i18n_dir = ROOT / "omlx_web/i18n"

    for locale_path in sorted(i18n_dir.glob("*.json")):
        translations = json.loads(locale_path.read_text())
        missing_keys = EXCLUSION_I18N_KEYS - translations.keys()
        assert not missing_keys, f"{locale_path.name} is missing {sorted(missing_keys)}"
