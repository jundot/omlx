// SPDX-License-Identifier: Apache-2.0
// Run with: node --test apps/omlx-web/tests/custom_theme_ui.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

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
