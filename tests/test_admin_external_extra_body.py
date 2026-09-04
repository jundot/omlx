# SPDX-License-Identifier: Apache-2.0
"""Regression tests for external accuracy-benchmark Extra Body controls."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "omlx" / "admin" / "i18n"


def test_accuracy_extra_body_is_wired_through_dashboard():
    js = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()
    template = (
        ROOT / "omlx/admin/templates/dashboard/_bench_accuracy.html"
    ).read_text()

    assert "accExternalExtraBody: ''" in js
    assert "parseAccuracyExtraBody()" in js
    assert "external: externalRequest" in js
    assert 'x-model="accExternalExtraBody"' in template
    assert '{"thinking":{"type":"disabled"}}' in template


def test_accuracy_extra_body_i18n_keys_exist_in_every_locale():
    keys = {
        "acc_bench.config.external_extra_body",
        "acc_bench.config.external_extra_body_hint",
        "js.error.external_extra_body_invalid_json",
        "js.error.external_extra_body_object_required",
        "js.error.external_extra_body_protected",
    }
    for locale_path in I18N_DIR.glob("*.json"):
        translations = json.loads(locale_path.read_text())
        missing = keys - translations.keys()
        assert not missing, f"{locale_path.name} is missing {sorted(missing)}"
