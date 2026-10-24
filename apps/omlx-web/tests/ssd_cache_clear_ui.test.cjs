// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8');
function setup(fetch) {
    const alerts = [];
    const app = vm.runInNewContext(source + '\n dashboard;', {
        localStorage: { getItem: () => null }, window: { t: key => key },
        console: { error() {} }, URL, fetch, alert: message => alerts.push(message),
    })();
    let refreshes = 0;
    app.loadStats = async () => { refreshes++; };
    app.showClearSsdCacheConfirm = true;
    return { app, alerts, refreshes: () => refreshes };
}

test('SSD deletion refreshes usage after success', async () => {
    const state = setup(async (url, options) => {
        assert.equal(url, '/admin/api/ssd-cache/clear');
        assert.equal(options.method, 'POST');
        return { ok: true };
    });
    await state.app.clearSsdCache();
    assert.equal(state.refreshes(), 1);
    assert.equal(state.app.showClearSsdCacheConfirm, false);
    assert.deepEqual(state.alerts, []);
});

test('SSD deletion shows the busy or failed-cache message from the API', async () => {
    const detail = 'Model is busy; wait for active requests to finish';
    const state = setup(async () => ({ ok: false, json: async () => ({ detail }) }));
    await state.app.clearSsdCache();
    assert.deepEqual(state.alerts, [detail]);
    assert.equal(state.refreshes(), 0);
    assert.equal(state.app.showClearSsdCacheConfirm, false);
});
