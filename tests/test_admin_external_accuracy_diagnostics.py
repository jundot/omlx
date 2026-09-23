# SPDX-License-Identifier: Apache-2.0
"""Regression tests for external accuracy diagnostic UI and exports."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

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


def test_benchmark_text_exports_preserve_literal_values():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for dashboard behavior tests")
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const catalog = JSON.parse(fs.readFileSync('omlx/admin/i18n/en.json', 'utf8'));
let download;
const context = {
    window: {t: key => catalog[key] ?? key},
    localStorage: {getItem: () => null},
    document: {createElement: () => ({click() {}})},
    Blob,
    URL: {createObjectURL: blob => {download = blob; return 'blob:test';},
          revokeObjectURL() {}},
};
const source = fs.readFileSync('omlx/admin/static/js/dashboard.js', 'utf8');
const state = vm.runInNewContext(source + '\n dashboard;', context)();

(async () => {
    for (const value of ['ordinary answer', "$$ $& $` $'", '한글\n日本語']) {
        for (const external of [false, true]) {
            const question = {
                id: 1, correct: true, category: value, finish_reason: value,
                reasoning_fields_nonempty: [value], error_message: value,
                question: value, expected: value, predicted: value,
                raw_response: value, time_s: 1,
            };
            const result = {
                model_id: value, benchmark: 'humaneval', accuracy: 1,
                correct: 1, total: 1, time_s: 1, external,
                valid_response_count: 1, valid_response_rate: 1,
                valid_answer_accuracy: 1, question_results: [question],
            };
            state.accDownloadResult(result, 'txt');
            const text = await download.text();
            const labels = ['Model', 'Category', 'Question', 'Expected',
                            'Predicted', 'Raw response'];
            if (external) labels.push('Finish reason', 'Reasoning fields', 'Error');
            for (const label of labels) {
                assert.ok(text.includes(`${label}: ${value}\n`),
                          `${label} changed in TXT export: ${JSON.stringify(text)}`);
            }
            state.accDownloadResult(result, 'json');
            assert.deepEqual(JSON.parse(await download.text()).questions, [question]);
            state.accAllResults = [result];
            assert.ok(state.accBuildText().includes(`Model: ${value}\n`));
        }
        state.benchModelId = value;
        state.benchRunExternal = null;
        assert.ok(state.benchBuildText().includes(`Benchmark Model: ${value}\n`));
        state.benchRunExternal = {model: value, base_url: value};
        assert.ok(state.benchBuildText().includes(`Benchmark Model: ${value} @ ${value}\n`));
    }
    state.accDownloadResult({model_id: 'demo', benchmark: 'humaneval',
        accuracy: 0, correct: 0, total: 1, time_s: 1,
        question_results: [{id: 1, expected: 'answer', predicted: '', time_s: 1}]}, 'txt');
    assert.ok((await download.text()).includes('Raw response: (empty)\n'));
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    result = subprocess.run(
        [node, "-e", script], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
