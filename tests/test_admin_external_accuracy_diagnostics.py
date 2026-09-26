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
            labels.push('Finish reason');
            if (external) labels.push('Reasoning fields', 'Error');
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


def test_local_truncation_is_wired_to_dashboard():
    js = (ROOT / "omlx/admin/static/js/dashboard.js").read_text()
    template = (
        ROOT / "omlx/admin/templates/dashboard/_bench_accuracy.html"
    ).read_text()

    assert "!r.external && r.truncated_count > 0" in template
    assert "r.finished_accuracy" in template
    assert "accLocalTruncationLine(r)" in js


def test_local_truncation_i18n_keys_exist_in_every_locale():
    keys = {
        "acc_bench.results.local_truncation_warning",
        "acc_bench.results.hit_token_limit",
        "acc_bench.results.finished_within_limit",
        "acc_bench.results.finished_accuracy",
        "acc_bench.results.text_export.local_truncation_line",
    }
    for locale_path in I18N_DIR.glob("*.json"):
        translations = json.loads(locale_path.read_text())
        missing = keys - translations.keys()
        assert not missing, f"{locale_path.name} is missing {sorted(missing)}"


def test_local_truncation_exports():
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
const summary = 'Hit token limit: 2/5 (1 of them scored correct) · '
    + 'Accuracy on finished answers: 66.7% (n = 3)';

(async () => {
    const question = {
        id: 7, correct: false, category: 'math', expected: 'A', predicted: '',
        question: 'q', raw_response: '<think>unfinished', time_s: 1,
        finish_reason: 'length', completion_tokens: 8192,
    };
    const result = {
        model_id: 'demo', benchmark: 'mmlu', accuracy: 0.6, correct: 3,
        total: 5, time_s: 1, external: false, truncated_count: 2,
        truncated_correct_count: 1, finished_count: 3,
        finished_accuracy: 0.6667, question_results: [question],
    };
    state.accDownloadResult(result, 'txt');
    const text = await download.text();
    assert.ok(text.includes(summary + '\n'), JSON.stringify(text));
    assert.ok(text.includes('Finish reason: length\n'));

    state.accDownloadResult(result, 'json');
    const data = JSON.parse(await download.text());
    assert.equal(data.truncated_count, 2);
    assert.equal(data.truncated_correct_count, 1);
    assert.equal(data.finished_count, 3);
    assert.equal(data.finished_accuracy, 0.6667);

    state.accDownloadResult(result, 'csv');
    const [header, row] = (await download.text()).split('\n');
    assert.ok(header.endsWith(',time_s,finish_reason,completion_tokens'), header);
    assert.ok(row.endsWith(',1,"length",8192'), row);

    state.accAllResults = [result];
    assert.ok(state.accBuildText().includes('  ' + summary));

    const allCut = {...result, truncated_count: 5, truncated_correct_count: 0,
                    finished_count: 0, finished_accuracy: null};
    state.accDownloadResult(allCut, 'txt');
    assert.ok((await download.text()).includes(
        'Accuracy on finished answers: — (n = 0)'));

    const clean = {...result, truncated_count: 0, truncated_correct_count: 0};
    state.accDownloadResult(clean, 'txt');
    assert.ok(!(await download.text()).includes('Hit token limit'));
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
