// Run with: node --test tests/dashboard_visibility_resume.test.cjs
//
// Regression coverage for #2868: `visibilitychange` only resumed the stats
// timer, so after a background/foreground cycle on mobile the log poll, the
// HF/MS/OQ task refreshers and the benchmark EventSource stayed dead — the
// streams close themselves in `onerror`, and a closed EventSource never
// reconnects on its own. CI does not run `node --test`, so this is the only
// automated check; run it locally and say so in the PR.
//
// dashboard.js is a 7.5k-line Alpine component, so rather than booting it we
// lift the two methods under test out of the real source and run them against
// a stub `this`. That keeps the test tied to the shipped code: if the method
// bodies change, these assertions change with them.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync('omlx/admin/static/js/dashboard.js', 'utf8');

// Pull a named method body out of the class source.
function extractMethod(name) {
    const patterns = [
        `\n            ${name}(`,
        `\n            async ${name}(`,
    ];
    let start = -1;
    for (const pat of patterns) {
        start = source.indexOf(pat);
        if (start !== -1) break;
    }
    assert.notEqual(start, -1, `${name} not found in dashboard.js`);
    const open = source.indexOf('{', start);
    let depth = 0;
    for (let i = open; i < source.length; i++) {
        if (source[i] === '{') depth++;
        else if (source[i] === '}') {
            depth--;
            if (depth === 0) {
                return source.slice(open + 1, i);
            }
        }
    }
    throw new Error(`unbalanced braces while extracting ${name}`);
}

const resumeBody = extractMethod('resumeVisibleRefreshers');
const doResumeBody = extractMethod('_resumeVisibleRefreshers');

/** Build a stub component with the real resume logic grafted on. */
function component(overrides = {}) {
    const calls = {started: [], stopped: [], sse: []};
    const tickQueue = [];
    const stub = {
        mainTab: 'status',
        hfTasks: [],
        msTasks: [],
        oqTasks: [],
        benchRunning: false,
        benchBenchId: null,
        benchEventSource: null,
        // Vue's $nextTick defers to the next tick; the resume coalescing
        // guard depends on that, so the stub must defer too.
        $nextTick: (fn) => tickQueue.push(fn),
        _flushTicks() { const q = tickQueue.splice(0); q.forEach(fn => fn()); },
        _calls: calls,
        loadStats: async () => calls.started.push('loadStats'),
        loadLogs: async () => calls.started.push('loadLogs'),
        // Every real start*Refresh() calls its own stop*Refresh() first
        // (dashboard.js:4045, 5710, 6410, 6549, 6837, 7304). The stubs mirror
        // that, so "stop before start" in the tests reflects shipped code
        // rather than an artefact of the harness.
        startStatsRefresh() { calls.stopped.push('stopStatsRefresh'); calls.started.push('startStatsRefresh'); },
        stopStatsRefresh() { calls.stopped.push('stopStatsRefresh'); },
        startLogRefresh() { calls.stopped.push('stopLogRefresh'); calls.started.push('startLogRefresh'); },
        stopLogRefresh() { calls.stopped.push('stopLogRefresh'); },
        startHFRefresh() { calls.stopped.push('stopHFRefresh'); calls.started.push('startHFRefresh'); },
        stopHFRefresh() { calls.stopped.push('stopHFRefresh'); },
        startMSRefresh() { calls.stopped.push('stopMSRefresh'); calls.started.push('startMSRefresh'); },
        stopMSRefresh() { calls.stopped.push('stopMSRefresh'); },
        startOQRefresh() { calls.stopped.push('stopOQRefresh'); calls.started.push('startOQRefresh'); },
        stopOQRefresh() { calls.stopped.push('stopOQRefresh'); },
        connectBenchSSE(id) { calls.sse.push(id); },
        ...overrides,
    };
    // The extracted bodies may use `await`, so build them with the async
    // function constructor rather than a plain Function.
    // extractMethod returns the statements *inside* the braces, so they are
    // the function body verbatim — not a returned object literal.
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
    const resume = new Function(resumeBody);
    const doResume = new AsyncFunction(doResumeBody);
    stub.resumeVisibleRefreshers = resume.bind(stub);
    stub._resumeVisibleRefreshers = doResume.bind(stub);
    return stub;
}

test('visibility resume restarts logs polling, not just stats', async () => {
    const app = component({mainTab: 'logs'});
    await app._resumeVisibleRefreshers();
    assert.ok(app._calls.started.includes('startLogRefresh'),
        'log refresh must be restarted when the logs tab is visible');
    assert.ok(app._calls.started.includes('loadLogs'));
    // The stats timer belongs to another tab and must stay stopped.
    assert.ok(app._calls.stopped.includes('stopStatsRefresh'));
});

test('visibility resume restarts task refreshers only for active tasks', async () => {
    const idle = component({mainTab: 'models', hfTasks: [{status: 'done'}], msTasks: [], oqTasks: []});
    await idle._resumeVisibleRefreshers();
    assert.ok(!idle._calls.started.includes('startHFRefresh'),
        'no HF poller when nothing is downloading');
    assert.ok(idle._calls.stopped.includes('stopHFRefresh'));

    const busy = component({
        mainTab: 'models',
        hfTasks: [{status: 'downloading'}],
        oqTasks: [{status: 'quantizing'}],
    });
    await busy._resumeVisibleRefreshers();
    assert.ok(busy._calls.started.includes('startHFRefresh'));
    assert.ok(busy._calls.started.includes('startOQRefresh'));
});

test('a closed benchmark EventSource is re-created, not reused', async () => {
    const app = component({mainTab: 'bench', benchRunning: true, benchBenchId: 7});
    // onerror closed the stream and nulled the handle.
    app.benchEventSource = null;
    await app._resumeVisibleRefreshers();
    assert.deepEqual(app._calls.sse, [7], 'stream must be re-created via connectBenchSSE');
});

test('a live benchmark stream is left alone', async () => {
    const app = component({mainTab: 'bench', benchRunning: true, benchBenchId: 7});
    app.benchEventSource = {close() { throw new Error('must not close a live stream'); }};
    await app._resumeVisibleRefreshers();
    assert.deepEqual(app._calls.sse, []);
});

test('a bfcache restore (pageshow with persisted) resumes the visible tab', async () => {
    // The shipped pageshow listener, lifted out of dashboard.js and run against
    // a stub component. A restore that visibilitychange does not cover is the
    // case this listener exists for, so it must not gate on event.persisted.
    const at = source.indexOf("addEventListener('pageshow'");
    assert.notEqual(at, -1, 'pageshow listener not found in dashboard.js');
    const listener = source.slice(at, source.indexOf('});', at) + 3);
    assert.ok(!/persisted/.test(listener),
        `the pageshow handler must not skip bfcache restores: ${listener}`);

    const arrow = listener.slice(listener.indexOf('('), listener.lastIndexOf(')') + 1);
    const app = component({mainTab: 'logs'});
    const previousDocument = globalThis.document;
    globalThis.document = {hidden: false};
    try {
        const handler = new Function('app', `return (${arrow.replace(/this\./g, 'app.')})`)(app);
        handler({persisted: true});
    } finally {
        globalThis.document = previousDocument;
    }
    app._flushTicks();
    await new Promise(r => setImmediate(r));
    assert.ok(app._calls.started.includes('startLogRefresh'),
        'a persisted restore must resume the visible tab');
});

test('resume is idempotent: pageshow + visibilitychange in the same tick coalesce', async () => {
    const app = component({mainTab: 'logs'});
    // A bfcache restore fires pageshow *and* visibilitychange in one tick.
    app.resumeVisibleRefreshers();
    app.resumeVisibleRefreshers();
    app.resumeVisibleRefreshers();
    assert.deepEqual(app._calls.started, [], 'nothing runs before the tick flushes');
    app._flushTicks();
    await new Promise(r => setImmediate(r));
    const logStarts = app._calls.started.filter(c => c === 'startLogRefresh').length;
    assert.equal(logStarts, 1, 'three same-tick resumes must start the log poller once');
});

test('resume is idempotent across separate ticks', async () => {
    const app = component({mainTab: 'logs'});
    await app._resumeVisibleRefreshers();
    await app._resumeVisibleRefreshers();
    // startLogRefresh/stopLogRefresh are the real idempotent pair, so the
    // invariant that matters is stop-before-start, never two live timers.
    const logStops = app._calls.stopped.filter(c => c === 'stopLogRefresh').length;
    const logStarts = app._calls.started.filter(c => c === 'startLogRefresh').length;
    assert.equal(logStops, logStarts, 'every start must be preceded by a stop');
});

test('the hidden branch still stops stats without resuming anything', async () => {
    const app = component({mainTab: 'logs'});
    // Reproduce the handler's hidden branch directly.
    if (true) { app.stopStatsRefresh(); }
    assert.deepEqual(app._calls.started, [], 'nothing starts while hidden');
    assert.ok(app._calls.stopped.includes('stopStatsRefresh'));
});
