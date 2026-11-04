// SPDX-License-Identifier: Apache-2.0
// Run with: node --test apps/omlx-web/tests/custom_theme_ui.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.join(__dirname, '..', 'omlx_web');
const theme = require(path.join(WEB, 'static/js/custom_theme.js'));
const themeCss = fs.readFileSync(path.join(WEB, 'static/css/theme.css'), 'utf8');
const contract = fs.readFileSync(path.join(__dirname, '../../../omlx/theme_contract.py'), 'utf8');

// The Python literal between `start` and the first line that is just `end`.
function pyBlock(start, end) {
    const from = contract.indexOf(start);
    return contract.slice(from, contract.indexOf(`\n${end}\n`, from));
}

function declared(selector) {
    const start = themeCss.indexOf(`${selector} {`);
    const block = themeCss.slice(start, themeCss.indexOf('}', start));
    return new Set([...block.matchAll(/--([a-z-]+):/g)].map(m => m[1]).filter(name => !name.startsWith('palette-')));
}

test('every editable token is a theme.css token in both themes', () => {
    const light = declared(':root');
    const dark = declared('[data-theme="dark"]');
    for (const token of theme.TOKENS) assert.ok(light.has(token) && dark.has(token), token);
});

test('the groups and contrast rules match the server contract', () => {
    const groups = [...pyBlock('TOKEN_GROUPS = {', '}').matchAll(/^ {4}"([a-z]+)": \(([^)]*)\)/gm)]
        .map(m => ({ id: m[1], tokens: [...m[2].matchAll(/"([a-z-]+)"/g)].map(t => t[1]) }));
    assert.deepEqual(groups, theme.GROUPS);
    const surfaces = theme.GROUPS.find(group => group.id === 'surfaces').tokens;
    const rules = [...pyBlock('CONTRAST_RULES = (', ')').matchAll(/^ {4}\("([a-z-]+)", (SURFACES|\(([^)]*)\))\)/gm)]
        .map(m => [m[1], m[2] === 'SURFACES' ? surfaces : [...m[3].matchAll(/"([a-z-]+)"/g)].map(t => t[1])]);
    assert.deepEqual(rules, theme.CONTRAST_RULES);
    assert.match(contract, new RegExp(`^CONTRAST_FLOOR = ${theme.CONTRAST_FLOOR}$`, 'm'));
});

test('normalizeHex accepts opaque 3/6 digits with or without #', () => {
    assert.equal(theme.normalizeHex('#ABC'), '#aabbcc');
    assert.equal(theme.normalizeHex(' 1a1a1d '), '#1a1a1d');
    for (const bad of ['', 'zz', '#12345', '#ef44441a', '#0000', 'red', 'rgb(0 0 0)', null]) {
        assert.equal(theme.normalizeHex(bad), null, bad);
    }
});

test('sanitize keeps known tokens with valid colors only', () => {
    const clean = theme.sanitize({
        light: {'bg-primary': 'FFF', 'not-a-token': '#000', 'text-primary': 'oops'},
        dark: {'--bg-primary': '#000', 'link-color': '#60a5fa', 'palette-white': '#000'},
        extra: {'bg-primary': '#000'},
    });
    assert.deepEqual(JSON.parse(JSON.stringify(clean)), {light: {'bg-primary': '#ffffff'}, dark: {'link-color': '#60a5fa'}});
    assert.deepEqual(JSON.parse(JSON.stringify(theme.sanitize('garbage'))), {light: {}, dark: {}});
});

test('css scopes each theme to its data-theme value', () => {
    const css = theme.css({light: {'bg-primary': '#fdf6e3'}, dark: {}});
    assert.match(css, /:root\[data-theme="light"\]/);
    assert.match(css, /--bg-primary: #fdf6e3;/);
    assert.match(css, /--surface-scale-white: 253 246 227;/);
    assert.doesNotMatch(css, /dark/);
});

test('muted leaves text-neutral-400 on the tertiary channel', () => {
    assert.doesNotMatch(theme.css({light: {'text-muted': '#94a3b9'}, dark: {}}), /--text-scale-400/);
    assert.match(theme.css({light: {'text-tertiary': '#555555'}, dark: {}}), /--text-scale-400: 85 85 85;/);
});

test('contrast follows WCAG and checks every rule pair', () => {
    assert.equal(theme.contrast('#000000', '#ffffff').toFixed(1), '21.0');
    assert.equal(theme.contrast('#ffffff', '#ffffff'), 1);
    assert.equal(theme.contrast('#00000000', '#ffffff'), null);
    const checks = theme.contrastChecks({
        'bg-primary': '#ffffff', 'bg-secondary': '#ffffff', 'bg-tertiary': '#ffffff',
        'text-primary': '#000000', 'text-secondary': '#000000', 'link-color': '#777777',
        'btn-primary': '#000000', 'btn-primary-text': '#ffffff',
    });
    assert.equal(checks.length, 10);
    assert.deepEqual(checks.filter(c => c.ratio < theme.CONTRAST_FLOOR).map(c => c.foreground), Array(3).fill('link-color'));
});

test('a custom status role also sets the pressure meter blends', () => {
    assert.match(theme.css({light: {danger: '#b91c1c'}, dark: {}}), /--meter-high: color-mix/);
    assert.doesNotMatch(theme.css({light: {'bg-primary': '#fdf6e3'}, dark: {}}), /--meter-/);
});

test('every token and group has an English label', () => {
    const en = JSON.parse(fs.readFileSync(path.join(WEB, 'i18n/en.json'), 'utf8'));
    for (const token of theme.TOKENS) assert.ok(en[`settings.theme.token.${token}`], token);
    for (const group of theme.GROUPS) assert.ok(en[`settings.theme.group.${group.id}`], group.id);
});
