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
        console: { error() {} }, URL, fetch: async () => ({ ok: true }), alert() {},
    })();
}

test('displayed Claude Code command masks the API key; copy keeps the real one', () => {
    const app = setup();
    app.globalSettings.claude_code.mode = 'local';
    app.stats.api_key = 'sk-live-secret-token';
    const display = app.claudeCodeCommandDisplay;
    assert.equal(display.includes('sk-live-secret-token'), false);
    assert.equal(display.includes("ANTHROPIC_AUTH_TOKEN='••••••••'"), true);
    assert.equal(app.claudeCodeCommand.includes("ANTHROPIC_AUTH_TOKEN='sk-live-secret-token'"), true);
});

test('without an API key the display matches the real command', () => {
    const app = setup();
    app.globalSettings.claude_code.mode = 'local';
    app.stats.api_key = '';
    assert.equal(app.claudeCodeCommandDisplay, app.claudeCodeCommand);
    assert.equal(app.claudeCodeCommandDisplay.includes('ANTHROPIC_AUTH_TOKEN='), false);
});
