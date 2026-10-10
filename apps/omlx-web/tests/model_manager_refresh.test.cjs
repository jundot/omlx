// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const source = fs.readFileSync(path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8');

test('manager refresh updates catalog rows and live status without dropping exposed profiles', async () => {
    let tick;
    let loaded = false;
    const document = {hidden: false};
    const create = vm.runInNewContext(source + '\n dashboard;', {
        console, localStorage: {getItem: () => null},
        THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
        window: {t: key => key}, navigator: {language: 'en'}, document,
        setInterval: callback => {tick = callback; return 1;}, clearInterval: () => {},
        fetch: async () => ({ok: true, json: async () => ({models: [{
            id: 'cached-model', loaded, is_loading: false,
            removal_kind: 'local_cache', actual_size: 42,
            exposed_profiles: [{model_id: 'profile-model', name: 'profile'}],
        }]})}),
    });
    const app = create();
    app.mainTab = 'models';
    app.modelsTab = 'manager';
    await app.loadModels();
    assert.equal(app.managerModelStatus('cached-model'), 'unloaded');
    assert.equal(app.managerModels.length, 2);
    assert.equal(app.managerModels[1].removal_kind, 'profile');
    app.startManagerStatusRefresh();
    loaded = true;
    tick();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(app.managerModelStatus('cached-model'), 'loaded');
    assert.equal(app.managerModels[0].loaded, true);
    assert.equal(app.managerModels[0].size, 42);
    assert.equal(app.managerModels.length, 2);
    document.hidden = true;
    loaded = false;
    tick();
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(app.managerModelStatus('cached-model'), 'loaded');
});
