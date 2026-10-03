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

test('recording switched off shows the settings pointer, not the unavailable warning', async () => {
    let payload = {...data, enabled: false, models: [], totals: {requests: 0}, heatmap: []};
    const view = component(async () => ({ok: true, json: async () => payload}));
    view.data = data; view.models = ['canonical model']; view.error = 'usage.delayed';
    await view.load();
    assert.equal(view.disabled, true);
    assert.equal(view.data, null);
    assert.equal(view.error, '');
    assert.equal(view.models.length, 0);
    assert.equal(view.loading, false);
    payload = {...data, enabled: true};
    await view.load();
    assert.equal(view.disabled, false);
    assert.equal(view.data.totals.requests, 1);
    assert.equal(view.models[0], 'canonical model');
});

// Sandbox arrays have their own prototype; compare structure only.
const plain = value => JSON.parse(JSON.stringify(value));

test('client tabs: each key+IP pair, per key, and per IP', async () => {
    const usage = {...data, by_client: true,
        clients: [{key_kind: 'sub_key', key_id: 'Editor', client_ip: '10.0.0.7', requests: 2},
                  {key_kind: 'none', key_id: '', client_ip: '127.0.0.1', requests: 1}],
        clients_by_key: [{key_kind: 'sub_key', key_id: 'Editor', requests: 2},
                         {key_kind: 'main_key', key_id: '', requests: 1},
                         {key_kind: 'none', key_id: '', requests: 1}],
        clients_by_ip: [{client_ip: '10.0.0.7', requests: 2}]};
    const view = component(async () => ({ok: true, json: async () => usage}));
    await view.load();
    assert.equal(view.clientView, 'all');
    assert.deepEqual(plain(view.clientRows().map(r => [r.label, r.detail])),
        [['Editor', '· 10.0.0.7'], ['usage.client_no_key', '· 127.0.0.1']]);
    view.clientView = 'key';
    assert.deepEqual(plain(view.clientRows().map(r => [r.label, r.detail])),
        [['Editor', 'usage.client_sub_key'], ['usage.client_main_key', ''], ['usage.client_no_key', '']]);
    view.clientView = 'ip';
    assert.deepEqual(plain(view.clientRows().map(r => [r.id, r.label, r.requests])), [['10.0.0.7', '10.0.0.7', 2]]);
    // Older servers omit the grouped views; tabs degrade to empty, not errors.
    view.data = {...data};
    assert.deepEqual(plain(view.clientRows()), []);
});
