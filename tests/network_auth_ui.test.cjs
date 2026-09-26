const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const source = fs.readFileSync(
    path.join(__dirname, '../omlx/admin/static/js/dashboard.js'), 'utf8'
);
function fixture(defaults, defaultsOK = true) {
    const requests = [];
    const create = vm.runInNewContext(source + '\n dashboard;', {
        URL, console,
        localStorage: {getItem: () => null},
        THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
        window: {t: key => key, location: {reload() {}}}, navigator: {language: 'en'}, document: {},
        setTimeout: () => {},
        fetch: async (url, options) => {
            if (url.endsWith('/defaults')) {
                return {ok: defaultsOK, json: async () => defaults};
            }
            requests.push(JSON.parse(options.body));
            return {ok: true, json: async () => ({success: true})};
        },
    });
    const state = create();
    state.globalSettings.model.model_dirs = ['/tmp/models'];
    state.loadStats = async () => {};
    state.loadModels = async () => {};
    return {state, requests};
}

test('loopback classification matches the backend fixtures', () => {
    const {state} = fixture();
    const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
    for (const [host, expected] of cases) {
        assert.equal(state.isLoopbackBindHost(host), expected, host);
    }
});

test('stored key survives an empty input on a network bind', async () => {
    const {state, requests} = fixture();
    Object.assign(state.globalSettings.auth, {api_key_set: true, api_key: ''});
    state.globalSettings.server.host = '0.0.0.0';
    await state.saveGlobalSettings();
    assert.equal(state.saveSuccess, true);
    assert.equal(requests.length, 1);
    assert.equal(Object.hasOwn(requests[0], 'api_key'), false);
    assert.equal(requests[0].skip_api_key_verification, false);
});

test('network bind without any key is blocked before sending', async () => {
    const {state, requests} = fixture();
    state.globalSettings.server.host = '0.0.0.0';
    await state.saveGlobalSettings();
    assert.equal(state.saveError, 'js.error.api_key_required_network');
    assert.equal(requests.length, 0);
});

test('new key and network host can be saved together', async () => {
    const {state, requests} = fixture();
    state.globalSettings.server.host = '0.0.0.0';
    state.globalSettings.auth.api_key = 'test-key';
    state.globalSettings.auth.skip_api_key_verification = true;
    await state.saveGlobalSettings();
    assert.equal(state.saveSuccess, true);
    assert.equal(requests[0].api_key, 'test-key');
    assert.equal(requests[0].skip_api_key_verification, false);
});

test('expanded and mapped IPv6 loopback preserve local no-auth mode', async () => {
    for (const host of ['0:0:0:0:0:0:0:1', '::ffff:127.0.0.1']) {
        const {state, requests} = fixture();
        state.globalSettings.server.host = host;
        state.globalSettings.auth.skip_api_key_verification = true;
        await state.saveGlobalSettings();
        assert.equal(state.saveSuccess, true, host);
        assert.equal(requests[0].skip_api_key_verification, true);
    }
});


test('reset stages defaults and preserves paths and credentials until Save', async () => {
    const defaults = JSON.parse(JSON.stringify(fixture().state.globalSettings));
    defaults.ui = {language: 'en'};
    defaults.memory.prefill_memory_guard = true;
    defaults.cache.ssd_cache_max_size = 'auto';
    const {state, requests} = fixture(defaults);
    const s = state.globalSettings;
    s.server.port = 9000;
    s.memory.prefill_memory_guard = false;
    s.sampling.temperature = 0.2;
    s.cache.ssd_cache_dir = '/custom/cache';
    s.cache.hot_cache_max_size = '8GB';
    s.network.ca_bundle = '/custom/ca.pem';
    s.network.http_proxy = 'http://localhost:9999';
    s.mcp.config_path = '/custom/mcp.json';
    s.auth.api_key = 'keep-this-key';
    s.auth.sub_keys = [{key: 'keep-sub-key'}];
    s.server.distributed_inference_active = true;
    s.ui.language = 'ko';
    await state.resetGlobalSettingsDefaults();
    assert.equal(requests.length, 0);
    assert.equal(state.showGlobalResetNotice, true);
    assert.equal(state.globalDefaultsPending, true);
    assert.equal(s.server.port, defaults.server.port);
    assert.equal(s.memory.prefill_memory_guard, true);
    assert.equal(s.sampling.temperature, defaults.sampling.temperature);
    assert.equal(s.cache.ssd_cache_max_size, 'auto');
    assert.equal(s.cache.hot_cache_max_size, '0');
    assert.equal(s.cache.ssd_cache_dir, '/custom/cache');
    assert.deepEqual(Array.from(s.model.model_dirs), ['/tmp/models']);
    assert.equal(s.network.ca_bundle, '/custom/ca.pem');
    assert.equal(s.network.http_proxy, '');
    assert.equal(s.mcp.config_path, '/custom/mcp.json');
    assert.equal(s.auth.api_key, 'keep-this-key');
    assert.equal(s.auth.sub_keys[0].key, 'keep-sub-key');
    assert.equal(s.server.distributed_inference_active, true);
    assert.equal(s.ui.language, 'en');
    state.confirmGlobalSettingsReset();
    assert.equal(state.globalResetSnapshot, null);
    await state.saveGlobalSettings();
    assert.equal(requests.length, 1);
    assert.equal(requests[0].port, defaults.server.port);
    assert.equal(requests[0].ssd_cache_max_size, 'auto');
    assert.equal(requests[0].api_key, 'keep-this-key');
    assert.equal(requests[0].ui_language, 'en');
    assert.equal(state.globalDefaultsPending, false);
});

test('failed defaults request leaves drafts untouched without a success modal', async () => {
    const {state, requests} = fixture(null, false);
    state.globalSettings.server.port = 9123;
    const before = JSON.stringify(state.globalSettings);
    await state.resetGlobalSettingsDefaults();
    assert.equal(JSON.stringify(state.globalSettings), before);
    assert.equal(state.showGlobalResetNotice, false);
    assert.equal(state.resettingGlobalSettings, false);
    assert.equal(state.saveError, 'settings.global.reset_failed');
    assert.equal(requests.length, 0);
});


test('Cancel restores unsaved edits, controls, and the previous pending state', async () => {
    const defaults = JSON.parse(JSON.stringify(fixture().state.globalSettings));
    const {state, requests} = fixture(defaults);
    for (const pending of [false, true]) {
        state.globalSettings.server.port = 9456;
        state.globalSettings.sampling.temperature = 0.23;
        state.globalSettings.ui.language = 'ko';
        state.globalSettings.idle_timeout.idle_timeout_seconds = 900;
        state.idleTimeoutValue = '900';
        state.cachePercent = 27;
        state.hotCachePercent = 12;
        state.globalDefaultsPending = pending;
        const before = JSON.stringify(state.globalSettings);
        await state.resetGlobalSettingsDefaults();
        assert.equal(state.globalSettings.sampling.temperature, 1);
        state.cancelGlobalSettingsReset();
        assert.equal(JSON.stringify(state.globalSettings), before);
        assert.equal(state.globalDefaultsPending, pending);
        assert.equal(state.idleTimeoutValue, '900');
        assert.equal(state.cachePercent, 27);
        assert.equal(state.hotCachePercent, 12);
        assert.equal(state.showGlobalResetNotice, false);
        assert.equal(state.globalResetSnapshot, null);
    }
    assert.equal(requests.length, 0);
});
