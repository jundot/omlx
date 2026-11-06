// SPDX-License-Identifier: Apache-2.0
// Run with: node --test apps/omlx-web/tests/custom_theme_ui.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WEB = path.join(__dirname, '..', 'omlx_web');
const theme = require(path.join(WEB, 'static/js/custom_theme.js'));
const themeCss = fs.readFileSync(path.join(WEB, 'static/css/theme.css'), 'utf8');

function declared(selector) {
    const start = themeCss.indexOf(`${selector} {`);
    const block = themeCss.slice(start, themeCss.indexOf('}', start));
    return new Set([...block.matchAll(/--([a-z-]+):/g)].map(m => m[1]));
}

test('every editable token is a theme.css token in both themes', () => {
    const light = declared(':root');
    const dark = declared('[data-theme="dark"]');
    for (const token of theme.TOKENS) assert.ok(light.has(token) && dark.has(token), token);
});

test('only known tokens with opaque colors reach the stylesheet', () => {
    const clean = theme.sanitize({
        light: {'bg-primary': 'FFF', 'not-a-token': '#000000', 'text-primary': 'oops'},
        dark: {'link-color': '#ef44441a'},
    });
    assert.deepEqual(JSON.parse(JSON.stringify(clean)), {light: {'bg-primary': '#ffffff'}, dark: {}});
    const css = theme.css(clean);
    assert.match(css, /:root\[data-theme="light"\] \{\n {4}--bg-primary: #ffffff;/);
    assert.doesNotMatch(css, /dark/);
});

test('muted leaves text-neutral-400 on the tertiary channel', () => {
    assert.doesNotMatch(theme.css({light: {'text-muted': '#94a3b9'}, dark: {}}), /--text-scale-400/);
    assert.match(theme.css({light: {'text-tertiary': '#555555'}, dark: {}}), /--text-scale-400: 85 85 85;/);
});

test('contrast checks flag every pair below the floor', () => {
    assert.equal(theme.contrast('#000000', '#ffffff').toFixed(1), '21.0');
    const checks = theme.contrastChecks({
        'bg-primary': '#ffffff', 'bg-secondary': '#ffffff', 'bg-tertiary': '#ffffff',
        'text-primary': '#000000', 'text-secondary': '#000000', 'link-color': '#777777',
        'btn-primary': '#000000', 'btn-primary-text': '#ffffff',
    });
    assert.deepEqual(checks.filter(c => c.ratio < theme.CONTRAST_FLOOR).map(c => c.foreground), Array(3).fill('link-color'));
});

function editor() {
    const source = fs.readFileSync(path.join(WEB, 'static/js/dashboard.js'), 'utf8');
    const clipboard = {text: null, writeText: async text => { clipboard.text = text; }};
    const create = vm.runInNewContext(source + '\n dashboard;', {
        URL, console, localStorage: {getItem: () => null},
        THEME_STORAGE_KEY: 'theme', ENHANCED_READABILITY_KEY: 'readability',
        window: {t: key => key, OmlxCustomTheme: theme}, navigator: {language: 'en', clipboard},
        document: {}, setTimeout: () => {}, clearTimeout: () => {},
    });
    const state = create();
    const stock = mode => Object.fromEntries(theme.TOKENS.map(token => [token, mode === 'light' ? '#eeeeee' : '#111111']));
    state.customThemeDefaults = {light: stock('light'), dark: stock('dark')};
    state.fillCustomThemeDraft();
    return {state, clipboard};
}

function assertComplete(json) {
    const copied = JSON.parse(json);
    for (const mode of ['light', 'dark']) {
        assert.deepEqual(Object.keys(copied[mode]).sort(), [...theme.TOKENS].sort(), mode);
        for (const token of theme.TOKENS) assert.match(copied[mode][token], /^#[0-9a-f]{6}$/, `${mode} ${token}`);
    }
    return copied;
}

test('loading partial JSON keeps the other draft colors', () => {
    const {state} = editor();
    state.customThemeDraft.light['bg-primary'] = '#fafafa';
    state.customThemeDraft.dark['link-color'] = '#88aaff';
    state.customThemeJson = JSON.stringify({light: {'text-primary': '#222222'}});
    state.importCustomThemeJson();
    assert.equal(state.customThemeJsonError, '');
    assert.equal(state.customThemeDraft.light['text-primary'], '#222222');
    assert.equal(state.customThemeDraft.light['bg-primary'], '#fafafa');
    assert.equal(state.customThemeDraft.dark['link-color'], '#88aaff');
    assert.equal(state.customThemeDraft.dark['text-primary'], '#111111');

    for (const bad of ['', '{"light": {"nope": "#000000"}}', 'not json']) {
        state.customThemeJson = bad;
        state.importCustomThemeJson();
        assert.equal(state.customThemeJsonError, 'settings.theme.json.invalid', bad);
    }
    assert.equal(state.customThemeDraft.light['text-primary'], '#222222');
});

test('copy writes every token in both modes, also after a partial load', async () => {
    const {state, clipboard} = editor();
    state.customThemeDraft.light['bg-primary'] = '#fafafa';
    state.customThemeDraft.dark['btn-primary'] = 'oops';
    await state.copyCustomThemeJson();
    assert.equal(state.customThemeJsonCopied, true);
    let copied = assertComplete(clipboard.text);
    assert.equal(copied.light['bg-primary'], '#fafafa');
    assert.equal(copied.dark['btn-primary'], '#111111');

    state.customThemeJson = JSON.stringify({dark: {'text-muted': '#999999'}});
    state.importCustomThemeJson();
    await state.copyCustomThemeJson();
    copied = assertComplete(clipboard.text);
    assert.equal(copied.dark['text-muted'], '#999999');
    assert.equal(copied.light['bg-primary'], '#fafafa');
});

test('a blocked clipboard leaves the JSON selected with a hint', async () => {
    const {state, clipboard} = editor();
    clipboard.writeText = async () => { throw new Error('NotAllowedError'); };
    const calls = [];
    state.$nextTick = fn => fn();
    state.$refs = {customThemeJsonField: {focus: () => calls.push('focus'), select: () => calls.push('select')}};
    await state.copyCustomThemeJson();
    assert.equal(state.customThemeJsonCopied, false);
    assert.equal(state.customThemeJsonCopyBlocked, true);
    assert.deepEqual(calls, ['focus', 'select']);
    assertComplete(state.customThemeJson);
});
