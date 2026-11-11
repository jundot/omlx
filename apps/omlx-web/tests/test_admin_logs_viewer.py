# SPDX-License-Identifier: Apache-2.0
"""Record view of the admin Logs tab (omlx_web/static/js/logs.js)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOGS_JS = ROOT / "omlx_web/static/js/logs.js"


def _run(script: str):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the log viewer tests")
    prelude = f"const L = require({json.dumps(str(LOGS_JS))});\n"
    result = subprocess.run(
        [node, "-e", prelude + script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _line(second: int, level: str, message: str) -> str:
    stamp = f"2026-10-08 10:00:{second:02d},000"
    return f"{stamp} - omlx.server - {level} - [-] - {message}"


def test_polls_over_a_growing_file_keep_every_record_once():
    # A file longer than the window, polled after each append: every poll must
    # show exactly the records of the current window, each under its own key.
    lines = []
    for i in range(60):
        lines.append(_line(i, "INFO", f"event {i}"))
        if i % 7 == 0:
            lines.append("Traceback (most recent call last):")
            lines.append(f'  File "x.py", line {i}')
    script = f"""
const lines = {json.dumps(lines)};
const size = 20;
let rows = [];
const failures = [];
for (let end = size; end <= lines.length; end += 3) {{
    const tail = lines.slice(end - size, end);
    const records = L.parseLogText(tail.join('\\n') + '\\n', end);
    rows = L.aggregateLogRows(records, 'TRACE', rows);
    const keys = rows.map(r => r.key);
    if (new Set(keys).size !== keys.length) failures.push(['duplicate key', end]);
    if (rows.length !== records.length) failures.push(['row count', end, rows.length, records.length]);
    const shown = rows.reduce((n, r) => n + r.lines, 0);
    if (shown !== tail.length) failures.push(['lines', end, shown, tail.length]);
    rows.forEach((r, i) => {{
        if (r.message !== records[i].message) failures.push(['message', end, r.key]);
    }});
}}
console.log(JSON.stringify(failures));
"""
    assert _run(script) == []


def test_sliding_window_does_not_reuse_a_key_for_another_record():
    # a,b,c -> b,c,d -> c,d,e: the shape that duplicated and dropped rows when
    # keys were numbered by array position.
    script = f"""
const all = {json.dumps([_line(i, "INFO", m) for i, m in enumerate("abcde")])};
let rows = [];
const seen = [];
for (let end = 3; end <= 5; end++) {{
    const tail = all.slice(end - 3, end).join('\\n') + '\\n';
    rows = L.aggregateLogRows(L.parseLogText(tail, end), 'TRACE', rows);
    seen.push(rows.map(r => r.key + '=' + r.message));
}}
console.log(JSON.stringify(seen));
"""
    assert _run(script) == [
        ["L0=a", "L1=b", "L2=c"],
        ["L1=b", "L2=c", "L3=d"],
        ["L2=c", "L3=d", "L4=e"],
    ]


def test_continuation_lines_and_a_cut_record():
    text = "\n".join(
        [
            '  File "tail.py", line 3',
            "ValueError: cut by the window",
            _line(1, "ERROR", "boom"),
            "Traceback (most recent call last):",
            '  File "a.py", line 1',
            _line(2, "INFO", "next"),
        ]
    )
    script = f"""
const records = L.parseLogText({json.dumps(text + chr(10))}, 106);
console.log(JSON.stringify(records.map(r => [r.key, r.level, r.lines, r.continuation])));
"""
    assert _run(script) == [
        ["L100", "", 2, True],
        ["L102", "ERROR", 3, False],
        ["L105", "INFO", 1, False],
    ]


def test_rotation_starts_keys_from_the_new_file():
    script = f"""
const before = L.parseLogText({json.dumps(_line(1, "INFO", "old") + chr(10))}, 900);
const after = L.parseLogText({json.dumps(_line(2, "INFO", "new") + chr(10))}, 1);
console.log(JSON.stringify([before[0].key, after[0].key]));
"""
    assert _run(script) == ["L899", "L0"]


def test_repeated_warnings_collapse_and_info_stays_one_row_each():
    text = "\n".join(
        [_line(i, "WARNING", "pressure") for i in range(4)]
        + [_line(10 + i, "INFO", "tick") for i in range(3)]
    )
    script = f"""
const rows = L.aggregateLogRows(L.parseLogText({json.dumps(text)}, 7), 'TRACE', []);
const hidden = L.aggregateLogRows(L.parseLogText({json.dumps(text)}, 7), 'WARNING', []);
console.log(JSON.stringify([rows.map(r => [r.level, r.count]), hidden.length]));
"""
    rows, hidden = _run(script)
    assert rows == [["WARNING", 4], ["INFO", 1], ["INFO", 1], ["INFO", 1]]
    assert hidden == 1


def test_memory_guard_lines_become_chips():
    abort = (
        "Request aborted: process memory limit exceeded (usage 91.2 GB, abort "
        "threshold (hard watermark) 88.0 GB, dynamic ceiling 96.0 GB). Retry later."
    )
    prefill = (
        "Prefill would require ~41.20 GB peak (current 30.00 GB + KV+SDPA "
        "11.20 GB) but dynamic ceiling is 40.00 GB (usage 29.1 GB, ceiling "
        "40.0 GB). Lower max context."
    )
    script = f"""
const labels = {{usage: 'U', watermark: 'W', ceiling: 'C', peak: 'P'}};
const chips = (m) => L.memoryGuardChips(L.memoryGuardFor(m), labels).map(c => c.label + ' ' + c.value);
console.log(JSON.stringify([chips({json.dumps(abort)}), chips({json.dumps(prefill)}), chips('model loaded')]));
"""
    assert _run(script) == [
        ["U 91.2 GB", "W 88.0 GB", "C 96.0 GB"],
        ["U 29.1 GB", "C 40.00 GB", "P 41.20 GB"],
        [],
    ]
