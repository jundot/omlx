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

// The stock tokens as opaque hex, flattened onto the page like the editor does.
function stockColors(mode) {
    const palette = Object.fromEntries([...themeCss.matchAll(/--(palette-[\w-]+):\s*([\d ]+);/g)]
        .map(m => [m[1], m[2].trim().split(/\s+/).map(Number)]));
    const block = selector => {
        const start = themeCss.indexOf(`${selector} {`);
        return Object.fromEntries([...themeCss.slice(start, themeCss.indexOf('\n}', start)).matchAll(/--([\w-]+):\s*([^;]+);/g)]
            .map(m => [m[1], m[2].trim()]));
    };
    const values = { ...block(':root'), ...(mode === 'dark' ? block('[data-theme="dark"]') : {}) };
    const css = token => {
        const m = /^rgb\(var\(--(palette-[\w-]+)\)(?:\s*\/\s*([\d.]+))?\)$/.exec(values[token]);
        return `rgba(${palette[m[1]].join(', ')}, ${m[2] ?? 1})`;
    };
    const colors = Object.fromEntries(theme.TOKENS.map(token => [token, theme.flatten(css(token), '#000000')]));
    for (const token of theme.DERIVED) colors[token] = theme.flatten(css(token), colors['bg-primary']);
    return colors;
}

test('the stock theme meets every contrast floor', () => {
    for (const mode of theme.MODES) {
        assert.deepEqual(theme.contrastChecks(stockColors(mode)).filter(check => !check.ok), [], mode);
    }
});

test('contrast checks flag every pair below its floor', () => {
    const colors = {...stockColors('light'), 'link-color': '#777777', 'focus-ring-color': '#bbbbbb'};
    const failed = theme.contrastChecks(colors).filter(check => !check.ok);
    assert.deepEqual(failed.map(check => [check.foreground, check.floor]),
        [...Array(3).fill(['link-color', 4.5]), ...Array(3).fill(['focus-ring-color', 3])]);
});

test('badge colors flatten the tints a custom status paints', () => {
    assert.deepEqual(theme.badgeColors('light', '#ff0000', '#ffffff'), {soft: '#ffebeb', line: '#ffb3b3', text: '#a60000'});
    assert.deepEqual(theme.badgeColors('dark', '#ff0000', '#000000'), {soft: '#260000', line: '#4d0000', text: '#ff5959'});
    assert.equal(theme.flatten('rgba(255, 0, 0, 0.5)', '#000000'), '#800000');
});

const lightness = hex => theme.oklch(hex)[0];
// Checks a generated dark palette the way the editor does: an unchanged status
// keeps its stock badge, a changed one gets the tints it paints.
function passes(colors) {
    const stock = stockColors('dark');
    const all = {...colors};
    for (const status of theme.STATUS) {
        const badge = colors[status] === stock[status]
            ? {soft: theme.flatten(stock[`${status}-soft`], colors['bg-primary']), text: stock[`${status}-text`]}
            : theme.badgeColors('dark', colors[status], colors['bg-primary']);
        all[`${status}-soft`] = badge.soft;
        all[`${status}-text`] = badge.text;
    }
    return theme.contrastChecks(all).every(check => check.ok);
}

test('OKLCH round-trips sRGB within one step per channel', () => {
    const steps = Array.from({length: 16}, (_, i) => i * 17);
    for (const r of steps) for (const g of steps) for (const b of steps) {
        const hex = '#' + [r, g, b].map(n => n.toString(16).padStart(2, '0')).join('');
        const back = theme.fromOklch(theme.oklch(hex));
        for (const i of [1, 3, 5]) {
            assert.ok(Math.abs(parseInt(back.slice(i, i + 2), 16) - parseInt(hex.slice(i, i + 2), 16)) <= 1, `${hex} -> ${back}`);
        }
    }
});

test('the stock light palette generates an Apply-able dark close to stock dark', () => {
    const light = stockColors('light');
    const dark = stockColors('dark');
    const generated = theme.darkFromLight(light);
    assert.deepEqual(Object.keys(generated), theme.TOKENS);
    for (const token of theme.TOKENS) {
        assert.equal(theme.normalizeHex(generated[token]), generated[token], token);
        const [a, b] = [generated[token], dark[token]].map(hex => {
            const [L, C, h] = theme.oklch(hex);
            return [L, C * Math.cos(h), C * Math.sin(h)];
        });
        assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 0.07, `${token}: ${generated[token]} vs ${dark[token]}`);
    }
    assert.ok(passes(generated));
});

test('generated neutrals reverse the light lightness order', () => {
    const custom = {
        ...stockColors('light'),
        'bg-primary': '#fdf6e3', 'bg-secondary': '#eee8d5', 'bg-tertiary': '#e4ddc8',
        'text-primary': '#073642', 'text-secondary': '#586e75', 'text-tertiary': '#657b83', 'text-muted': '#93a1a1',
    };
    for (const light of [stockColors('light'), custom]) {
        const dark = theme.darkFromLight(light);
        for (const group of ['surfaces', 'text']) {
            const tokens = theme.GROUPS.find(g => g.id === group).tokens;
            for (let i = 1; i < tokens.length; i++) {
                const before = Math.sign(lightness(light[tokens[i]]) - lightness(light[tokens[i - 1]]));
                const after = Math.sign(lightness(dark[tokens[i]]) - lightness(dark[tokens[i - 1]]));
                assert.equal(after, -before, `${group}: ${tokens[i - 1]} -> ${tokens[i]}`);
            }
        }
    }
});

test('a low-contrast generated foreground is nudged until it passes', () => {
    const light = {
        ...stockColors('light'),
        'bg-primary': '#bbbbbb', 'bg-secondary': '#b5b5b5', 'bg-tertiary': '#afafaf',
        'text-secondary': '#666666', 'link-color': '#3366aa',
    };
    assert.ok(passes(theme.darkFromLight(light)));
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
