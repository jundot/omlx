# SPDX-License-Identifier: Apache-2.0
"""Regression tests for external accuracy diagnostic UI and exports."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I18N_DIR = ROOT / "omlx" / "admin" / "i18n"


def test_external_accuracy_diagnostics_are_wired_to_dashboard():
    js = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()
    template = (
        ROOT / "omlx/admin/templates/dashboard/_bench_accuracy.html"
    ).read_text()

    assert "valid_response_count" in js
    assert "valid_answer_accuracy" in js
    assert "reasoning_fields_nonempty" in js
    assert "r.reliability_warning" in template
    assert "r.valid_response_rate" in template


def test_external_accuracy_diagnostic_i18n_keys_exist_in_every_locale():
    keys = {
        "acc_bench.results.total_accuracy",
        "acc_bench.results.valid_responses",
        "acc_bench.results.valid_response_rate",
        "acc_bench.results.valid_answer_accuracy",
        "acc_bench.results.empty_content",
        "acc_bench.results.truncated",
        "acc_bench.results.timeout",
        "acc_bench.results.http_errors",
        "acc_bench.results.connection_errors",
        "acc_bench.results.invalid_responses",
        "acc_bench.results.parse_errors",
        "acc_bench.results.reliability_warning",
    }
    for locale_path in I18N_DIR.glob("*.json"):
        translations = json.loads(locale_path.read_text())
        missing = keys - translations.keys()
        assert not missing, f"{locale_path.name} is missing {sorted(missing)}"
