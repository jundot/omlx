// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8');
function setup() {
    return vm.runInNewContext(source + '\n dashboard;', {
        localStorage: { getItem: () => null }, window: { t: key => key },
        console: { error() {} }, URL, fetch: async () => ({ ok: true, json: async () => ({ api_key: 'sk-live-secret-token' }) }), alert() {},
    })();
}

test('displayed Claude Code command masks the API key; copy fetches the real one', async () => {
    const app = setup();
    app.globalSettings.claude_code.mode = 'local';
    app.stats.api_key_set = true;
    const display = app.claudeCodeCommandDisplay;
    assert.equal(display.includes("ANTHROPIC_AUTH_TOKEN='••••••••'"), true);
    assert.equal((await app.claudeCodeCommand()).includes("ANTHROPIC_AUTH_TOKEN='sk-live-secret-token'"), true);
});

test('without an API key the display matches the real command', async () => {
    const app = setup();
    app.globalSettings.claude_code.mode = 'local';
    assert.equal(app.claudeCodeCommandDisplay, await app.claudeCodeCommand());
    assert.equal(app.claudeCodeCommandDisplay.includes('ANTHROPIC_AUTH_TOKEN='), false);
});
