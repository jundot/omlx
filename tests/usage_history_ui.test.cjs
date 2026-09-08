// Run with: node --test tests/usage_history_ui.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('omlx/admin/static/js/usage.js', 'utf8');

function component(fetch) {
    const context = vm.createContext({fetch, AbortController, URLSearchParams, Intl,
        window: {t: key => key}, document: {hidden: false}, setInterval, clearInterval});
    vm.runInContext(source, context);
    return context.usageHistory();
}
const data = {available: true, dropped_requests: 0, models: [{model_id: 'canonical model'}],
    totals: {requests: 1}, heatmap: [{date: '2026-09-08', tokens: [0, 120]}]};

test('range and exact model are encoded; unloaded models and intensity remain usable', async () => {
    const urls = [];
    const view = component(async url => { urls.push(url); return {ok: true, json: async () => data}; });
    await view.load();
    assert.equal(view.models[0], 'canonical model');
    view.range = 'month'; view.model = 'canonical model';
    await view.load();
    assert.equal(urls[1], '/admin/api/usage?range=month&model=canonical+model');
    assert.equal(view.peak, 120);
    assert.equal(view.shade(0), 'rgba(128, 128, 128, 0.12)');
    assert.equal(view.speed(null), '—');
});

test('storage failure removes stale values and displays unavailable state', async () => {
    const view = component(async () => ({ok: false}));
    view.data = data;
    await view.load();
    assert.equal(view.data, null);
    assert.equal(view.error, 'usage.unavailable');
    assert.equal(view.loading, false);
});

test('slower previous selection cannot overwrite newer range', async () => {
    const pending = [];
    const view = component(() => new Promise(resolve => pending.push(resolve)));
    const first = view.load();
    view.range = '7d';
    const second = view.load();
    const newer = {...data, totals: {requests: 7}};
    pending[1]({ok: true, json: async () => newer});
    await second;
    pending[0]({ok: true, json: async () => data});
    await first;
    assert.equal(view.data.totals.requests, 7);
    assert.equal(view.loading, false);
});

test('retry/overflow state stays visible alongside committed history', async () => {
    const view = component(async () => ({ok: true, json: async () => ({...data, dropped_requests: 1})}));
    await view.load();
    assert.equal(view.error, 'usage.delayed');
    assert.equal(view.data.totals.requests, 1);
});
