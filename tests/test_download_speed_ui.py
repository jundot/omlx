# SPDX-License-Identifier: Apache-2.0
"""The console's download queue shows a live transfer rate.

The dashboard's own formatters are exercised through the real module in a
node sandbox — the same seam ``tests/network_auth_ui.test.cjs`` uses — so a
regression in ``formatSpeed`` or in the progress line fails here without
pinning the source text. The queue templates are only checked for the
wiring that hands the rate to those formatters.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_JS = ROOT / "omlx/admin/static/js/dashboard.js"
MODELS_TEMPLATE = ROOT / "omlx/admin/templates/dashboard/_models.html"

# Loads the dashboard the way the browser does, then applies each check to
# the live Alpine state: {"method", "args", "exact" | "contains"}.
_NODE_HARNESS = """
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8') + '\\ndashboard;';
const state = vm.runInNewContext(source, {
    URL, console,
    localStorage: {getItem: () => null},
    THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
    window: {t: key => key, location: {reload() {}}},
    navigator: {language: 'en'}, document: {}, setTimeout: () => {},
    fetch: async () => ({ok: true, json: async () => ({})}),
})();
for (const check of JSON.parse(process.argv[2])) {
    const got = state[check.method](...check.args);
    const ok = 'exact' in check
        ? got === check.exact
        : String(got).includes(check.contains);
    if (!ok) {
        console.error(`${check.method}(${JSON.stringify(check.args)}) = ${got}`);
        process.exit(1);
    }
}
"""


def _run_dashboard(checks: list[dict]) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the console's formatter checks")
    result = subprocess.run(
        [node, "-e", _NODE_HARNESS, str(DASHBOARD_JS), json.dumps(checks)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_speed_reads_bytes_per_second_across_units():
    """A stopped transfer must read an explicit 0 B/s (hiding the readout
    made an interrupted row look like a UI bug), and a live one scales."""
    _run_dashboard(
        [
            {"method": "formatSpeed", "args": [{"speed_bps": 0}], "exact": "0 B/s"},
            {"method": "formatSpeed", "args": [{"speed_bps": 512}], "exact": "512 B/s"},
            {
                "method": "formatSpeed",
                "args": [{"speed_bps": 1024}],
                "exact": "1.0 KB/s",
            },
            {
                "method": "formatSpeed",
                "args": [{"speed_bps": 45298483.2}],
                "exact": "43.2 MB/s",
            },
            {
                "method": "formatSpeed",
                "args": [{"speed_bps": 3.5 * 1024**3}],
                "exact": "3.5 GB/s",
            },
        ]
    )


def test_progress_line_carries_the_rate():
    _run_dashboard(
        [
            {
                "method": "formatProgress",
                "args": [
                    {
                        "progress": 45.5,
                        "downloaded_size": 536870912,
                        "total_size": 1073741824,
                        "speed_bps": 45298483.2,
                    }
                ],
                "contains": "43.2 MB/s",
            }
        ]
    )


def test_both_queues_render_the_rate():
    """Each queue (HuggingFace and ModelScope) hands the task to the shared
    formatters: one progress line and one queued-row rate per queue."""
    template = MODELS_TEMPLATE.read_text()
    assert template.count('x-text="formatProgress(task)"') >= 2
    assert template.count("formatSpeed(task)") >= 2
