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

test('contrast checks flag every pair below the floor', () => {
    assert.equal(theme.contrast('#000000', '#ffffff').toFixed(1), '21.0');
    const checks = theme.contrastChecks({
        'bg-primary': '#ffffff', 'bg-secondary': '#ffffff', 'bg-tertiary': '#ffffff',
        'text-primary': '#000000', 'text-secondary': '#000000', 'link-color': '#777777',
        'btn-primary': '#000000', 'btn-primary-text': '#ffffff',
    });
    assert.deepEqual(checks.filter(c => c.ratio < theme.CONTRAST_FLOOR).map(c => c.foreground), Array(3).fill('link-color'));
});
