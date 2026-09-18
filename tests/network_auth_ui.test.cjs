const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const source = fs.readFileSync(
    path.join(__dirname, '../omlx/admin/static/js/dashboard.js'), 'utf8'
);
function fixture() {
    const requests = [];
    const create = vm.runInNewContext(source + '\n dashboard;', {
        URL, console,
        localStorage: {getItem: () => null},
        THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
        window: {t: key => key}, navigator: {language: 'en'}, document: {},
        setTimeout: () => {},
        fetch: async (url, options) => {
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
