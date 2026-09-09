# SPDX-License-Identifier: Apache-2.0
"""Behavioural tests for the admin log viewer's level filter.

The repo has no JS test runner, so the filter methods are sliced out of
dashboard.js and executed under node. That runs the shipped code rather
than asserting on its source text, which matters here: the exact-match
and minimum-level paths differ only in a comparison operator.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_JS = ROOT / "omlx/admin/static/js/dashboard.js"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available"
)

SAMPLE_LOG = "\n".join(
    [
        "2026-08-18 13:27:14,676 - omlx.scheduler - INFO - engine ready",
        "2026-08-18 13:27:14,700 - omlx.server - TRACE - Incoming POST /v1/chat/completions",
        "    continuation line carrying the request body",
        "2026-08-18 13:27:14,750 - httpx - DEBUG - connect tcp",
        "2026-08-18 13:27:14,800 - omlx.server - ERROR - boom",
    ]
)


def _methods_block() -> str:
    js = DASHBOARD_JS.read_text()
    start = js.index("filteredLogContent() {")
    end = js.index("async loadLogs() {")
    return js[start:end].rstrip().rstrip(",")


def _run_node(assertions: str) -> subprocess.CompletedProcess:
    harness = f"""
const assert = require('assert');
const make = (state) => Object.assign({{
    logContent: {SAMPLE_LOG!r}.replace(/\\\\n/g, '\\n'),
    logMinLevel: 'TRACE',
    logExactLevel: null,
}}, state, {{
    {_methods_block()}
}});
{assertions}
console.log('OK');
"""
    return subprocess.run(
        ["node", "-e", harness], capture_output=True, text=True, timeout=60
    )


def _assert_ok(assertions: str) -> None:
    proc = _run_node(assertions)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "OK" in proc.stdout


def test_exact_mode_shows_only_that_level():
    """The whole point: TRACE-only, without the DEBUG/INFO/ERROR noise."""
    _assert_ok(
        """
        const o = make({logExactLevel: 'TRACE', logMinLevel: 'TRACE'});
        const out = o.filteredLogContent();
        assert(out.includes('Incoming POST'), 'trace line missing');
        assert(!out.includes('engine ready'), 'INFO leaked into exact TRACE');
        assert(!out.includes('connect tcp'), 'DEBUG leaked into exact TRACE');
        assert(!out.includes('boom'), 'ERROR leaked into exact TRACE');
        """
    )


def test_exact_mode_keeps_continuation_lines():
    """Request bodies span lines that carry no level prefix of their own."""
    _assert_ok(
        """
        const o = make({logExactLevel: 'TRACE', logMinLevel: 'TRACE'});
        const out = o.filteredLogContent();
        assert(out.includes('continuation line carrying the request body'),
               'continuation line dropped');
        """
    )


def test_exact_mode_drops_continuation_of_filtered_lines():
    _assert_ok(
        """
        const o = make({logExactLevel: 'ERROR', logMinLevel: 'ERROR'});
        const out = o.filteredLogContent();
        assert(out.includes('boom'), 'ERROR line missing');
        assert(!out.includes('continuation line'),
               'kept continuation belonging to a filtered TRACE line');
        """
    )


def test_minimum_level_path_unchanged():
    """Default behaviour must still be "this level and above"."""
    _assert_ok(
        """
        const o = make({logMinLevel: 'INFO'});
        const out = o.filteredLogContent();
        assert(out.includes('engine ready'), 'INFO missing');
        assert(out.includes('boom'), 'ERROR missing');
        assert(!out.includes('Incoming POST'), 'TRACE leaked above min INFO');
        assert(!out.includes('connect tcp'), 'DEBUG leaked above min INFO');
        """
    )


def test_modifier_click_toggles_exact_mode_off():
    _assert_ok(
        """
        const o = make({});
        o.selectLogLevel('TRACE', {metaKey: true});
        assert.strictEqual(o.logExactLevel, 'TRACE');
        o.selectLogLevel('TRACE', {metaKey: true});
        assert.strictEqual(o.logExactLevel, null, 'second click did not toggle off');
        """
    )


def test_ctrl_click_works_for_non_mac():
    _assert_ok(
        """
        const o = make({});
        o.selectLogLevel('DEBUG', {ctrlKey: true});
        assert.strictEqual(o.logExactLevel, 'DEBUG');
        """
    )


def test_plain_click_clears_exact_mode():
    _assert_ok(
        """
        const o = make({logExactLevel: 'TRACE'});
        o.selectLogLevel('WARNING', {});
        assert.strictEqual(o.logExactLevel, null, 'plain click left exact mode on');
        assert.strictEqual(o.logMinLevel, 'WARNING');
        """
    )


def test_button_styling_marks_only_pinned_level_in_exact_mode():
    """Range styling would otherwise imply levels that are filtered out."""
    _assert_ok(
        """
        const o = make({logExactLevel: 'TRACE', logMinLevel: 'TRACE'});
        const pinned = o.levelButtonClass('TRACE');
        const other = o.levelButtonClass('ERROR');
        assert(pinned !== other, 'pinned level not visually distinct');
        assert(other.includes('neutral-300'), 'non-pinned level not dimmed');
        """
    )


def test_exact_mode_classes_exist_in_compiled_css():
    """Tailwind is precompiled; an uncompiled class silently renders unstyled.

    bg-emerald-700 shipped once and left the pinned button white-on-white
    because only the background rule was missing, not text-white.
    """
    css = "\n".join(
        p.read_text() for p in (ROOT / "omlx/admin/static/css").glob("*.css")
    )
    proc = _run_node(
        """
        const o = make({logExactLevel: 'TRACE', logMinLevel: 'TRACE'});
        console.log(JSON.stringify([
            o.levelButtonClass('TRACE'),
            o.levelButtonClass('ERROR'),
        ]));
        """
    )
    assert proc.returncode == 0, proc.stderr

    import json as _json

    emitted = _json.loads(proc.stdout.splitlines()[0])
    for class_list in emitted:
        for cls in class_list.split():
            assert f".{cls}" in css, f"{cls} is not in the compiled CSS"


def test_exact_filter_strings_present_in_every_locale():
    import json

    keys = (
        "logs.level_exact_hint",
        "logs.level_exact_badge",
        "logs.level_exact_clear",
    )
    for path in sorted((ROOT / "omlx/admin/i18n").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            assert key in data, f"{path.name} missing {key}"
            assert data[key].strip(), f"{path.name} has empty {key}"
        assert "{level}" in data["logs.level_exact_badge"], (
            f"{path.name} badge lost the {{level}} placeholder"
        )
