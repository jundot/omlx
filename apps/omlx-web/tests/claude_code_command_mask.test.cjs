// SPDX-License-Identifier: Apache-2.0
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../omlx_web/static/js/dashboard.js'), 'utf8');
function setup(fetchImpl) {
    return vm.runInNewContext(source + '\n dashboard;', {
        localStorage: { getItem: () => null }, window: { t: key => key },
        console: { error() {} }, URL,
        fetch: fetchImpl || (async () => ({ ok: true })),
        alert() {},
        navigator: { clipboard: null },
        document: {
            createElement() {
                return { style: {}, value: '', select() {}, setSelectionRange() {} };
            },
            body: { appendChild() {}, removeChild() {} },
            execCommand() { return true; },
        },
    })();
}

test('displayed Claude Code command masks the API key from api_key_set', () => {
    const app = setup();
    app.globalSettings.claude_code.mode = 'local';
    app.stats.api_key_set = true;
    const display = app.claudeCodeCommandDisplay;
    assert.equal(display.includes('sk-live-secret-token'), false);
    assert.equal(display.includes("ANTHROPIC_AUTH_TOKEN='••••••••'"), true);
});

test('without an API key the display has no auth token', () => {
    const app = setup();
    app.globalSettings.claude_code.mode = 'local';
    app.stats.api_key_set = false;
    assert.equal(app.claudeCodeCommandDisplay.includes('ANTHROPIC_AUTH_TOKEN='), false);
});

test('copyClaudeCodeCommand fetches the key once and copies the real command', async () => {
    let revealCalls = 0;
    const copied = [];
    const app = setup(async (url, opts) => {
        if (url === '/admin/api/auth/api-key' && opts && opts.method === 'POST') {
            revealCalls += 1;
            return {
                ok: true,
                status: 200,
                async json() { return { api_key: 'sk-live-secret-token' }; },
            };
        }
        return { ok: true, status: 200, async json() { return {}; } };
    });
    app.globalSettings.claude_code.mode = 'local';
    app.stats.api_key_set = true;
    app.stats.port = 8000;
    app.copyToClipboard = (text) => { copied.push(text); };

    await app.copyClaudeCodeCommand();

    assert.equal(revealCalls, 1);
    assert.equal(copied.length, 1);
    assert.equal(copied[0].includes("ANTHROPIC_AUTH_TOKEN='sk-live-secret-token'"), true);
    assert.equal('api_key' in app.stats, false);
});
