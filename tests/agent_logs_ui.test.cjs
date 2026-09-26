const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const source = fs.readFileSync(
    path.join(__dirname, '../omlx/admin/static/js/dashboard.js'), 'utf8'
);

function dashboardState() {
    const create = vm.runInNewContext(source + '\n dashboard;', {
        URL, console,
        localStorage: {getItem: () => null},
        THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
        window: {t: key => key === 'acc_bench.agent_logs.more_lines' ? '… {count} more lines' : key},
        navigator: {language: 'en'}, document: {},
        setTimeout: () => {},
        fetch: async () => ({ok: true, json: async () => ({})}),
    });
    return create();
}

// vm objects come from another realm; compare through JSON.
const plain = value => JSON.parse(JSON.stringify(value));

test('line diff keeps common lines and orders removals before additions', () => {
    const state = dashboardState();
    assert.deepEqual(plain(state.agentLogDiff('a\nb\nc', 'a\nB\nc')), [
        {op: ' ', text: 'a'},
        {op: '-', text: 'b'},
        {op: '+', text: 'B'},
        {op: ' ', text: 'c'},
    ]);
});

test('edit calls render every edit, in both pi argument shapes', () => {
    const state = dashboardState();
    const multi = state.agentToolCallBody({name: 'edit', args: {
        path: '/app/x.py',
        edits: [{oldText: 'a', newText: 'b'}, {oldText: 'c', newText: 'c'}],
    }});
    assert.equal(multi.kind, 'diff');
    assert.equal(multi.path, '/app/x.py');
    assert.deepEqual(plain(multi.lines), [
        {op: '@', text: '@@ edit 1/2'},
        {op: '-', text: 'a'},
        {op: '+', text: 'b'},
        {op: '@', text: '@@ edit 2/2'},
        {op: ' ', text: 'c'},
    ]);
    const flat = state.agentToolCallBody({name: 'edit', args: {path: 'p', oldText: 'x', newText: 'y'}});
    assert.deepEqual(plain(flat.lines), [{op: '-', text: 'x'}, {op: '+', text: 'y'}]);
});

test('write calls are capped at 300 added rows', () => {
    const state = dashboardState();
    const content = Array.from({length: 305}, (_, i) => `line ${i}`).join('\n');
    const body = state.agentToolCallBody({name: 'write', args: {path: 'f', content}});
    assert.equal(body.lines.length, 301);
    assert.ok(body.lines.slice(0, 300).every(row => row.op === '+'));
    assert.deepEqual(plain(body.lines[300]), {op: '@', text: '… 5 more lines'});
});

test('stream payloads build trials, follow running ones, and trim history', () => {
    const state = dashboardState();
    const job = 'terminalbench_4-deadbeef';
    state._applyAgentLogPayload(job, {type: 'reset'});
    state._applyAgentLogPayload(job, {type: 'trial', trial: 'a__1', task: 'a', status: 'running'});
    state._applyAgentLogPayload(job, {type: 'trial', trial: 'b__2', task: 'b', status: 'running'});
    const entry = state.agentLogs[job];
    assert.deepEqual(plain(entry.order), ['a__1', 'b__2']);
    assert.equal(entry.active, 'b__2');

    entry.userPicked = true;
    entry.active = 'a__1';
    state._applyAgentLogPayload(job, {type: 'trial', trial: 'c__3', task: 'c', status: 'running'});
    assert.equal(entry.active, 'a__1');

    const events = Array.from({length: 2005}, (_, i) => ({kind: 'stderr', text: String(i)}));
    state._applyAgentLogPayload(job, {type: 'events', trial: 'a__1', events});
    assert.equal(entry.trials.a__1.events.length, 2000);
    assert.equal(entry.trials.a__1.trimmed, 5);
    assert.equal(entry.trials.a__1.events[0].text, '5');

    state._applyAgentLogPayload(job, {type: 'trial', trial: 'a__1', task: 'a', status: 'pass'});
    assert.equal(state.agentLogCounts(job), 'acc_bench.agent_logs.counts');
    assert.equal(entry.trials.a__1.status, 'pass');

    // A reconnect replays from scratch without duplicating tabs.
    state._applyAgentLogPayload(job, {type: 'reset'});
    state._applyAgentLogPayload(job, {type: 'trial', trial: 'a__1', task: 'a', status: 'pass'});
    assert.deepEqual(plain(entry.order), ['a__1']);
});
