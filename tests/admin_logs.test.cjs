// SPDX-License-Identifier: Apache-2.0
// The Logs tab derives its structure from the raw text the API returns. These
// vectors pin that derivation: the record parser (multi-line messages, a window
// that starts mid-record), the poll-to-poll delta (append, slide, rotation),
// the aggregation of repeated warnings, the memory-guard numbers and the
// windowing math. Run with: node --test tests/admin_logs.test.cjs
const assert = require('assert/strict');
const path = require('path');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const logs = require(path.join(root, 'omlx/admin/static/js/logs.js'));

const ABORT_LINE =
    '2026-09-21 07:11:02,118 - omlx.engine_core - WARNING - [req-1] - ' +
    'Request aborted: process memory limit exceeded (usage 109.0 GB, abort ' +
    'threshold (hard watermark) 107.3 GB, dynamic ceiling 113.0 GB). Close ' +
    'other apps to free RAM (static cap is 124.00 GB but only 3.93 GB is ' +
    'reclaimable right now), raise memory_guard_tier (safe \u2192 balanced \u2192 ' +
    'aggressive), or reduce context length.';

const PREFILL_LINE =
    '2026-09-21 07:12:44,002 - omlx.scheduler - WARNING - [-] - ' +
    'Prefill would require ~82.93 GB peak (current 77.81 GB + KV+SDPA 5.11 GB) ' +
    'but dynamic ceiling is 80.37 GB. Close other apps to free RAM (static cap ' +
    'is 124.00 GB but only 6.78 GB is reclaimable right now), raise ' +
    'memory_guard_tier (safe \u2192 balanced \u2192 aggressive), or reduce context length.';

test('a record is its header line plus every continuation line', () => {
    const text =
        '2026-09-21 00:56:04,543 - omlx.server - WARNING - [-] - Responses API ' +
        'streaming prefill rejected: Missing 210 parameters: \n' +
        'model.encoder.vision_tower.encoder.layers.0.input_layernorm.weight,\n' +
        'model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.weight,\n' +
        '2026-09-21 00:56:05,001 - omlx.engine_pool - INFO - [abc-123] - Loaded model\n';
    const records = logs.parseLogText(text, 0);
    assert.equal(records.length, 2);
    const [warning, loaded] = records;
    assert.equal(warning.time, '2026-09-21 00:56:04,543');
    assert.equal(warning.module, 'omlx.server');
    assert.equal(warning.level, 'WARNING');
    assert.equal(warning.requestId, '-');
    assert.equal(warning.lines, 3);
    assert.equal(warning.continuation, false);
    assert.ok(warning.message.endsWith('self_attn.q_proj.weight,'));
    assert.equal(warning.message.split('\n').length, 3);
    assert.equal(loaded.level, 'INFO');
    assert.equal(loaded.requestId, 'abc-123');
    assert.equal(loaded.index, 1, 'indices continue from baseIndex');
});

test('a window that starts mid-record keeps the fragment and marks it', () => {
    const text =
        'model.encoder.vision_tower.encoder.layers.1.mlp.down_proj.weight,\n' +
        '2026-09-21 00:56:06,000 - omlx.server - ERROR - [-] - boom\n';
    const records = logs.parseLogText(text, 4);
    assert.equal(records.length, 2);
    assert.equal(records[0].continuation, true);
    assert.equal(records[0].level, '', 'a fragment has no level of its own');
    assert.equal(records[0].index, 4);
    assert.equal(records[1].level, 'ERROR');
    assert.equal(records[1].index, 5);
});

test('all six levels parse, with or without the request-id field', () => {
    const levels = ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];
    const withId = levels
        .map((level, i) => `2026-09-21 00:00:0${i},000 - omlx.a - ${level} - [rid-${i}] - m${i}`)
        .join('\n');
    const plain = '2026-09-21 00:00:09,000 - omlx.a - INFO - no request id here';
    assert.deepEqual(logs.parseLogText(withId, 0).map(record => record.level), levels);
    const withoutId = logs.parseLogText(plain, 0)[0];
    assert.equal(withoutId.level, 'INFO');
    assert.equal(withoutId.requestId, '-');
    assert.equal(withoutId.message, 'no request id here');
    assert.equal(logs.levelRank('critical'), 5, 'ranking is case-insensitive');
    assert.equal(logs.levelRank(''), -1);
});

test('a poll appends what the previous window did not have', () => {
    const merged = logs.mergeLogText('a\nb\nc\n', 'a\nb\nc\nd\n');
    assert.equal(merged.reset, false);
    assert.equal(merged.dropped, 0);
    assert.equal(merged.overlap, 3);
    assert.equal(merged.appended, 'd\n');
});

test('a poll that slid the window reports the lines that fell off', () => {
    const merged = logs.mergeLogText('a\nb\nc\nd\n', 'c\nd\ne\n');
    assert.equal(merged.reset, false);
    assert.equal(merged.dropped, 2);
    assert.equal(merged.appended, 'e\n');
});

test('an unchanged poll appends nothing', () => {
    const merged = logs.mergeLogText('a\nb\n', 'a\nb\n');
    assert.equal(merged.reset, false);
    assert.equal(merged.dropped, 0);
    assert.equal(merged.appended, '');
});

test('a rotated or truncated file resets instead of appending', () => {
    for (const [previous, incoming] of [['a\nb\n', 'x\ny\n'], ['a\nb\n', ''], ['', 'a\n']]) {
        const merged = logs.mergeLogText(previous, incoming);
        assert.equal(merged.reset, true, `${JSON.stringify(previous)} -> ${JSON.stringify(incoming)}`);
        assert.equal(merged.appended, incoming);
    }
});

test('a record that was still being written absorbs its continuation', () => {
    const first = logs.parseLogText('2026-09-21 00:00:01,000 - omlx.a - WARNING - [-] - head\n', 0)[0];
    const appended = logs.parseLogText('frame 1,\nframe 2,\n', 1)[0];
    assert.equal(appended.continuation, true);
    assert.equal(logs.absorbContinuation(first, appended), true);
    assert.equal(first.lines, 3);
    assert.equal(first.message, 'head\nframe 1,\nframe 2,');
    assert.equal(logs.absorbContinuation(first, null), false);
});

const repeatedWarnings = [
    { index: 0, time: 't0', module: 'omlx.a', level: 'WARNING', requestId: '-', message: 'same', lines: 1 },
    { index: 1, time: 't1', module: 'omlx.a', level: 'WARNING', requestId: 'r1', message: 'same', lines: 1 },
    { index: 2, time: 't2', module: 'omlx.a', level: 'WARNING', requestId: 'r2', message: 'same', lines: 1 },
];

test('consecutive identical warnings collapse into one row with every occurrence', () => {
    const rows = logs.aggregateLogRows(repeatedWarnings, 'TRACE', []);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].count, 3);
    assert.equal(rows[0].key, 'r0');
    assert.deepEqual(rows[0].occurrences.map(occurrence => occurrence.time), ['t0', 't1', 't2']);
    assert.deepEqual(rows[0].occurrences.map(occurrence => occurrence.requestId), ['-', 'r1', 'r2']);
});

test('a different message, level or module starts a new row', () => {
    const rows = logs.aggregateLogRows([
        ...repeatedWarnings,
        { index: 3, time: 't3', module: 'omlx.a', level: 'WARNING', requestId: '-', message: 'other', lines: 1 },
        { index: 4, time: 't4', module: 'omlx.b', level: 'WARNING', requestId: '-', message: 'same', lines: 1 },
        { index: 5, time: 't5', module: 'omlx.a', level: 'ERROR', requestId: '-', message: 'same', lines: 1 },
    ], 'TRACE', []);
    assert.deepEqual(rows.map(row => [row.level, row.message, row.count]), [
        ['WARNING', 'same', 3],
        ['WARNING', 'other', 1],
        ['WARNING', 'same', 1],
        ['ERROR', 'same', 1],
    ]);
});

test('INFO does not collapse, and a line between two warnings breaks the run', () => {
    const info = index => ({ index, time: 't', module: 'm', level: 'INFO', requestId: '-', message: 'same', lines: 1 });
    const warning = index => ({ index, time: 't', module: 'm', level: 'WARNING', requestId: '-', message: 'same', lines: 1 });
    const rows = logs.aggregateLogRows(
        [info(0), info(1), warning(2), info(3), warning(4)],
        'TRACE',
        []
    );
    assert.equal(rows.length, 5, 'only adjacent warnings collapse');
    assert.deepEqual(rows.map(row => row.count), [1, 1, 1, 1, 1]);
});

test('the minimum level hides lower levels and closes a run', () => {
    const record = (index, level) => ({ index, time: 't', module: 'm', level, requestId: '-', message: 'same', lines: 1 });
    const records = [record(0, 'WARNING'), record(1, 'INFO'), record(2, 'WARNING'), record(3, 'ERROR')];
    const rows = logs.aggregateLogRows(records, 'WARNING', []);
    assert.deepEqual(rows.map(row => row.level), ['WARNING', 'WARNING', 'ERROR']);
    assert.deepEqual(rows.map(row => row.count), [1, 1, 1]);
    assert.equal(logs.aggregateLogRows(records, 'TRACE', []).length, 4);
});

test('an append reuses the rows the viewer already has in the DOM', () => {
    const first = logs.aggregateLogRows(repeatedWarnings, 'TRACE', []);
    const second = logs.aggregateLogRows(
        repeatedWarnings.concat([
            { index: 3, time: 't3', module: 'omlx.a', level: 'WARNING', requestId: '-', message: 'other', lines: 1 },
        ]),
        'TRACE',
        first
    );
    assert.equal(second[0], first[0], 'the unchanged row keeps its identity');
    assert.equal(second[0].count, 3);
    assert.equal(second[1].key, 'r3');
});

test('an unchanged poll writes nothing back into the rows', () => {
    // A write is what makes Alpine re-render, so counting them is how the
    // "a refresh must not rebuild the list" guarantee is pinned.
    const rows = logs.aggregateLogRows(repeatedWarnings, 'TRACE', []);
    const writes = new Map();
    const watched = rows.map((row, index) => new Proxy(row, {
        set(target, name, value) {
            writes.set(index, (writes.get(index) || 0) + 1);
            target[name] = value;
            return true;
        },
    }));

    const unchanged = logs.aggregateLogRows(repeatedWarnings, 'TRACE', watched);
    assert.equal(writes.size, 0, 'no new lines, no writes, no DOM work');
    assert.equal(unchanged[0].count, 3);

    logs.aggregateLogRows(
        repeatedWarnings.concat([
            { index: 3, time: 't3', module: 'omlx.a', level: 'WARNING', requestId: '-', message: 'same', lines: 1 },
            { index: 4, time: 't4', module: 'omlx.a', level: 'WARNING', requestId: '-', message: 'other', lines: 1 },
        ]),
        'TRACE',
        watched
    );
    assert.deepEqual([...writes.keys()], [0], 'only the row that grew is touched');
    assert.ok(writes.get(0) <= 3, 'and only its counter and occurrence list');
});

test('chrome is not mistaken for a guard line', () => {
    assert.equal(logs.memoryGuardFor('Loading model weights'), null);
    assert.equal(logs.memoryGuardFor(''), null);
});

test('the abort line yields usage, hard watermark and ceiling', () => {
    const guard = logs.memoryGuardFor(ABORT_LINE);
    assert.equal(guard.kind, 'abort');
    assert.equal(guard.usage.text, '109.0');
    assert.equal(guard.watermark.text, '107.3');
    assert.equal(guard.ceiling.text, '113.0');
    assert.equal(guard.ceilingBinding, 'dynamic');
    assert.equal(guard.staticCap.text, '124.00');
    assert.equal(guard.reclaimable.text, '3.93');
    assert.equal(guard.peak, null);
    assert.deepEqual(logs.memoryGuardChips(guard, { usage: 'u', watermark: 'w', ceiling: 'c', peak: 'p' }), [
        { key: 'usage', label: 'u', value: '109.0 GB' },
        { key: 'watermark', label: 'w', value: '107.3 GB' },
        { key: 'ceiling', label: 'c', value: '113.0 GB' },
    ]);
});

test('the prefill rejection yields the predicted peak against the ceiling', () => {
    const guard = logs.memoryGuardFor(PREFILL_LINE);
    assert.equal(guard.kind, 'prefill');
    assert.equal(guard.peak.text, '82.93');
    assert.equal(guard.current.text, '77.81');
    assert.equal(guard.kv.text, '5.11');
    assert.equal(guard.ceiling.text, '80.37');
    assert.equal(guard.usage.text, '77.81', 'usage falls back to the measured current');
    assert.equal(guard.watermark, null, 'this shape names no hard watermark');
    assert.deepEqual(logs.memoryGuardChips(guard, { usage: 'u', ceiling: 'c', peak: 'p' }).map(chip => chip.key), [
        'usage',
        'ceiling',
        'peak',
    ]);
});

test('the pressure advisories carry a count and a model', () => {
    const plain = logs.memoryGuardFor('Aborted 3 requests due to memory pressure');
    assert.equal(plain.kind, 'pressure');
    assert.equal(plain.aborted, 3);
    const kept = logs.memoryGuardFor(
        "Hard memory pressure: aborted 3 request(s) on 'Qwen3-VL-30B' and kept model loaded"
    );
    assert.equal(kept.kind, 'pressure');
    assert.equal(kept.aborted, 3);
    assert.equal(kept.model, 'Qwen3-VL-30B');
    assert.deepEqual(logs.memoryGuardChips(kept, { usage: 'u' }), []);
});

test('windowing mounts the visible slice plus overscan', () => {
    assert.deepEqual(logs.visibleRange(10000, 0, 600, 30, 8), { start: 0, end: 36 });
    assert.deepEqual(logs.visibleRange(10000, 3000, 600, 30, 8), { start: 92, end: 128 });
    assert.deepEqual(logs.visibleRange(10000, 3000, 600, 30, 0), { start: 100, end: 120 });
    assert.deepEqual(logs.visibleRange(10, 3000, 600, 30, 8), { start: 0, end: 10 }, 'never past the end');
    assert.deepEqual(logs.visibleRange(0, 0, 0, 30, 8), { start: 0, end: 0 });
    assert.deepEqual(logs.visibleRange(100, 300, 600, 0, 0), { start: 0, end: 100 }, 'a lost row height shows everything');
});

test('a long run of occurrences is windowed, and the rest counted', () => {
    const occurrences = Array.from({ length: 20000 }, (_, i) => ({ time: 't' + i, requestId: '-' }));
    const windowed = logs.occurrenceWindow(occurrences, logs.OCCURRENCE_WINDOW);
    assert.equal(windowed.shown.length, logs.OCCURRENCE_WINDOW);
    assert.equal(windowed.hidden, 20000 - logs.OCCURRENCE_WINDOW);
    assert.equal(windowed.shown[0].time, 't0');
    assert.equal(windowed.shown[windowed.shown.length - 1].time, 't' + (logs.OCCURRENCE_WINDOW - 1));
    // A run shorter than the window hides nothing, and an empty one is safe.
    assert.deepEqual(logs.occurrenceWindow([{ time: 'a', requestId: '-' }], 200), {
        shown: [{ time: 'a', requestId: '-' }],
        hidden: 0,
    });
    assert.deepEqual(logs.occurrenceWindow(null, 200), { shown: [], hidden: 0 });
    assert.deepEqual(logs.occurrenceWindow([], 200), { shown: [], hidden: 0 });
    // The list is never mutated.
    const copy = JSON.parse(JSON.stringify(occurrences));
    logs.occurrenceWindow(occurrences, 5);
    assert.deepEqual(occurrences, copy);
});

// === The viewer itself ===
// dashboard.js holds the Alpine state over these functions; driving it here is
// what proves a poll patches the list rather than rebuilding it. Each test
// builds its own viewer, so they do not share records.

const fs = require('fs');
const vm = require('vm');

const context = {
    localStorage: { getItem: () => null },
    document: { addEventListener: () => {}, querySelector: () => null },
    window: {
        t: key => key,
        addEventListener: () => {},
        location: { href: 'http://127.0.0.1:8000/admin/dashboard', search: '' },
        history: { replaceState: () => {} },
    },
    URL,
    setTimeout: () => 0,
    console,
    alert: message => { throw Error(message); },
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'omlx/admin/static/js/logs.js'), 'utf8'), context);
vm.runInContext(fs.readFileSync(path.join(root, 'omlx/admin/static/js/dashboard.js'), 'utf8'), context);

function freshViewer() {
    const viewer = context.dashboard();
    viewer.$nextTick = fn => fn();
    viewer.$refs = { logViewport: { scrollTop: 0, scrollHeight: 0, clientHeight: 600 } };
    return viewer;
}

const line = (second, level, message) =>
    `2026-09-21 09:00:0${second},000 - omlx.server - ${level} - [-] - ${message}\n`;
const aborted = second =>
    `2026-09-21 09:00:0${second},000 - omlx.engine_core - WARNING - [-] - Request aborted: ` +
    'process memory limit exceeded (usage 109.0 GB, abort threshold (hard watermark) 107.3 GB, ' +
    'dynamic ceiling 113.0 GB). raise memory_guard_tier (safe \u2192 balanced \u2192 aggressive)\n';

test('a poll appends rows and keeps the ones already rendered', () => {
    const viewer = freshViewer();
    viewer.ingestLogText(line(0, 'INFO', 'listening') + aborted(2));
    const rows = viewer.logRows;
    assert.deepEqual([...rows.map(row => row.level)], ['INFO', 'WARNING']);
    assert.equal(rows[1].memory.kind, 'abort');
    assert.deepEqual([...viewer.logMemoryChips(rows[1]).map(chip => chip.value)],
        ['109.0 GB', '107.3 GB', '113.0 GB']);

    viewer.ingestLogText(line(0, 'INFO', 'listening') + aborted(2) + aborted(5));
    assert.equal(viewer.logRows.length, 2, 'the repeated warning folds into the same row');
    assert.equal(viewer.logRows[0], rows[0], 'the untouched row keeps its identity');
    assert.equal(viewer.logRows[1], rows[1]);
    assert.equal(viewer.logRows[1].count, 2);
    assert.deepEqual([...viewer.logRows[1].occurrences.map(occurrence => occurrence.time)],
        ['2026-09-21 09:00:02,000', '2026-09-21 09:00:05,000']);
});

test('a poll that slid the window drops the rows that fell off', () => {
    const viewer = freshViewer();
    viewer.ingestLogText(line(0, 'INFO', 'a') + line(1, 'INFO', 'b') + line(2, 'INFO', 'c'));
    assert.deepEqual([...viewer.logRows.map(row => row.message)], ['a', 'b', 'c']);
    // Two lines fell out of the window and one arrived.
    viewer.ingestLogText(line(2, 'INFO', 'c') + line(3, 'INFO', 'd'));
    assert.deepEqual([...viewer.logRows.map(row => row.message)], ['c', 'd']);
});

test('a rotated file resets instead of merging', () => {
    const viewer = freshViewer();
    viewer.ingestLogText(line(0, 'INFO', 'a') + line(1, 'INFO', 'b'));
    viewer.ingestLogText('2026-09-22 00:00:00,000 - omlx.server - INFO - [-] - fresh\n');
    assert.deepEqual([...viewer.logRows.map(row => row.message)], ['fresh']);
});

test('the level filter re-derives the rows', () => {
    const viewer = freshViewer();
    viewer.ingestLogText(line(0, 'INFO', 'quiet') + line(1, 'ERROR', 'loud'));
    viewer.setLogMinLevel('WARNING');
    assert.deepEqual([...viewer.logRows.map(row => row.level)], ['ERROR']);
    viewer.setLogMinLevel('TRACE');
    assert.deepEqual([...viewer.logRows.map(row => row.level)], ['INFO', 'ERROR']);
});

test('only the rows near the viewport are mounted', () => {
    const viewer = freshViewer();
    viewer.ingestLogText(
        Array.from({ length: 5000 }, (unused, i) =>
            `2026-09-21 08:00:00,${String(i % 1000).padStart(3, '0')} - omlx.server - INFO - [-] - line ${i}\n`
        ).join('')
    );
    assert.equal(viewer.logRows.length, 5000);
    assert.equal(viewer.visibleLogRows.length, 36, 'the window plus overscan');
    viewer.logScrollTop = 3000;
    assert.equal(viewer.visibleLogRows.length, 36);
    assert.equal(viewer.visibleLogRows[0].message, viewer.logRows[viewer.logWindow.start].message);
    assert.equal(viewer.logWindow.start, 92);
});

test('the inline remedies switch to Settings and scroll to the anchor', () => {
    const viewer = freshViewer();
    const targets = [];
    context.document.querySelector = selector => ({
        scrollIntoView: () => targets.push(selector),
        classList: { add: () => {}, remove: () => {} },
    });
    viewer.setSettingsTab = tab => { viewer.activeTab = tab; viewer.mainTab = 'settings'; };
    const actions = viewer.logMemoryActions;
    assert.deepEqual([...actions.map(action => action.anchor)], ['memory-guard', 'context-window']);
    viewer.openLogAction(actions[0].anchor);
    assert.equal(viewer.mainTab, 'settings');
    assert.equal(viewer.activeTab, 'global');
    assert.deepEqual(targets, ['[data-anchor="memory-guard"]']);
});

test('an anchor that is not in the document still lands on Settings', () => {
    const viewer = freshViewer();
    context.document.querySelector = () => null;
    viewer.mainTab = 'logs';
    viewer.openLogAction('no-such-anchor');
    assert.equal(viewer.mainTab, 'settings');
    assert.equal(viewer.activeTab, 'global');
});

test('a sliding window keeps the row the reader is looking at', () => {
    const viewer = freshViewer();
    // Distinct messages, real timestamps: the window slides over these.
    const page = (start, count) =>
        Array.from({ length: count }, (unused, i) =>
            `2026-09-21 10:${String((start + i) % 60).padStart(2, '0')}:00,000 - omlx.server - INFO - [-] - line ${start + i}\n`
        ).join('');

    viewer.logAutoScroll = false;
    viewer.ingestLogText(page(0, 100));
    assert.equal(viewer.logRows.length, 100);
    viewer.logScrollTop = 30 * 40;
    viewer.$refs.logViewport.scrollTop = 30 * 40;
    const anchored = viewer.logRows[viewer.logWindow.start].message;

    // Ten lines fell out of the front and one arrived.
    viewer.ingestLogText(page(10, 91));
    assert.deepEqual([...viewer.logRows.map(row => row.message)].slice(0, 2), ['line 10', 'line 11']);
    assert.equal(viewer.logRows[viewer.logWindow.start].message, anchored);
    // Ten rows left the top, so the offset gives ten rows back: without that
    // the reader's line would slide away under a fixed scroll position.
    assert.equal(viewer.$refs.logViewport.scrollTop, 30 * 30);
});

test('a rotated file copied off another platform still parses', () => {
    const records = logs.parseLogText(
        '2026-09-21 00:00:01,000 - omlx.a - INFO - [-] - crlf line\r\n' +
        '2026-09-21 00:00:02,000 - omlx.a - INFO - [-] - plain\n',
        0
    );
    assert.equal(records[0].message, 'crlf line');
    assert.equal(records[1].message, 'plain');
});
