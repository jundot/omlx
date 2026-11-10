// SPDX-License-Identifier: Apache-2.0
// Run with: node --test apps/omlx-web/tests/theme_preview.test.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const preview = require(path.join(__dirname, '..', 'omlx_web', 'static/js/theme_preview.js'));
const theme = require(path.join(__dirname, '..', 'omlx_web', 'static/js/custom_theme.js'));
const en = require(path.join(__dirname, '..', 'omlx_web', 'i18n/en.json'));

function chunk(type, data) {
    const out = new Uint8Array(12 + data.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, data.length);
    out.set(Buffer.from(type), 4);
    out.set(data, 8);
    view.setUint32(8 + data.length, preview.crc32(out.subarray(4, 8 + data.length)));
    return out;
}

test('Load image reads back the theme JSON that Save as image embeds', () => {
    assert.equal(preview.crc32(Buffer.from('IEND')), 0xae426082);
    const png = new Uint8Array(Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        chunk('IHDR', new Uint8Array(13)),
        chunk('tEXt', Buffer.from('Comment\0not this one')),
        chunk('IEND', new Uint8Array(0)),
    ]));
    const json = JSON.stringify({light: {'bg-primary': '#fdf6e3'}, dark: {'link-color': '#ff8800'}});
    const out = preview.withTextChunk(png, preview.KEYWORD, json);
    const length = new DataView(out.buffer).getUint32(33);
    assert.equal(Buffer.from(out.subarray(37, 41)).toString(), 'tEXt');
    assert.equal(new DataView(out.buffer).getUint32(41 + length), preview.crc32(out.subarray(37, 41 + length)));
    assert.equal(preview.readTextChunk(out, preview.KEYWORD), json);
    assert.equal(preview.readTextChunk(png, preview.KEYWORD), null);
    assert.equal(preview.readTextChunk(new Uint8Array(Buffer.from('{"light": {}}')), preview.KEYWORD), null);
});

test('the fixed English labels stay in sync with en.json', () => {
    for (const group of theme.GROUPS) {
        assert.equal(preview.GROUP_LABELS[group.id], en[`settings.theme.group.${group.id}`], group.id);
        for (const token of group.tokens) {
            assert.equal(preview.TOKEN_LABELS[token], en[`settings.theme.token.${token}`], token);
        }
    }
    const keys = {
        heading: 'settings.theme.heading', subtitle: 'settings.theme.preview_subtitle', port: 'settings.theme.preview_port',
        link: 'settings.theme.preview_link', danger: 'settings.theme.preview_danger', hover: 'settings.theme.preview_hover',
        save: 'settings.save.button', tabStatus: 'navbar.tab.status', tabSettings: 'navbar.tab.settings',
    };
    for (const [label, key] of Object.entries(keys)) assert.equal(preview.LABELS[label], en[key], label);
    for (const status of theme.STATUS) assert.equal(preview.LABELS.status[status], en[`settings.theme.token.${status}`], status);
});

// A 2D context that draws nothing; text is 6px per character.
const blankContext = () => new Proxy({}, {
    get: (target, key) => key in target ? target[key] : key === 'measureText' ? value => ({width: [...value].length * 6}) : () => {},
    set: (target, key, value) => { target[key] = value; return true; },
});

test('every editable token is reachable by clicking the preview card', () => {
    const colors = Object.fromEntries([...theme.TOKENS, ...theme.DERIVED].map(token => [token, '#808080']));
    const regions = [];
    preview.drawCard(blankContext(), colors, {mode: 'light', sans: 'sans-serif', labels: {...preview.LABELS, version: '1'}, regions});
    assert.deepEqual([...new Set(regions.map(region => region[4]))].sort(), [...theme.TOKENS].sort());
    // The Save label sits on the button, which sits on the focus ring.
    const label = regions.find(region => region[4] === 'btn-primary-text');
    assert.equal(preview.regionAt(regions, label[0] + 1, label[1] + 1)[4], 'btn-primary-text');
});
