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

// Stands in for CanvasRenderingContext2D.measureText: 6px per character.
const measure = {measureText: value => ({width: [...value].length * 6})};

test('the theme JSON rides in a valid tEXt chunk after IHDR', () => {
    assert.equal(preview.crc32(Buffer.from('IEND')), 0xae426082);
    const png = Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        chunk('IHDR', new Uint8Array(13)),
        chunk('IEND', new Uint8Array(0)),
    ]);
    const json = JSON.stringify({dark: {'link-color': '#ff8800'}});
    const out = preview.withTextChunk(new Uint8Array(png), preview.KEYWORD, json);

    const view = new DataView(out.buffer);
    const types = [];
    for (let at = 8; at < out.length; at += 12 + view.getUint32(at)) {
        const length = view.getUint32(at);
        types.push(Buffer.from(out.subarray(at + 4, at + 8)).toString());
        assert.equal(view.getUint32(at + 8 + length), preview.crc32(out.subarray(at + 4, at + 8 + length)));
    }
    assert.deepEqual(types, ['IHDR', 'tEXt', 'IEND']);
    const length = view.getUint32(33);
    assert.equal(Buffer.from(out.subarray(41, 41 + length)).toString(), `${preview.KEYWORD}\0${json}`);
});

test('the fixed English labels stay in sync with en.json', () => {
    for (const group of theme.GROUPS) {
        assert.equal(preview.GROUP_LABELS[group.id], en[`settings.theme.group.${group.id}`], group.id);
        for (const token of group.tokens) {
            assert.equal(preview.TOKEN_LABELS[token], en[`settings.theme.token.${token}`], token);
        }
    }
    assert.equal(preview.LABELS.heading, en['settings.theme.heading']);
    assert.equal(preview.LABELS.subtitle, en['settings.theme.preview_subtitle']);
    assert.equal(preview.LABELS.port, en['settings.theme.preview_port']);
    assert.equal(preview.LABELS.link, en['settings.theme.preview_link']);
    assert.equal(preview.LABELS.danger, en['settings.theme.preview_danger']);
    assert.equal(preview.LABELS.hover, en['settings.theme.preview_hover']);
    assert.equal(preview.LABELS.save, en['settings.save.button']);
    assert.equal(preview.LABELS.tabStatus, en['navbar.tab.status']);
    assert.equal(preview.LABELS.tabSettings, en['navbar.tab.settings']);
    for (const status of ['success', 'warning', 'danger', 'info']) {
        assert.equal(preview.LABELS.status[status], en[`settings.theme.token.${status}`], status);
    }
});

test('the table height counts the header, group headings and rows', () => {
    assert.equal(preview.tableHeight([{id: 'x', tokens: ['a']}]), 24 + 18 + 22);
    const tokens = theme.GROUPS.reduce((count, group) => count + group.tokens.length, 0);
    assert.equal(preview.tableHeight(theme.GROUPS), 24 + theme.GROUPS.length * 18 + tokens * 22);
});

test('wrap breaks at spaces', () => {
    assert.deepEqual(preview.wrap(measure, 'aa bb cc', 30), ['aa bb', 'cc']);
    assert.deepEqual(preview.wrap(measure, '', 30), []);
});

test('ellipsize keeps text that fits and cuts with an ellipsis', () => {
    assert.equal(preview.ellipsize(measure, 'abc', 18), 'abc');
    assert.equal(preview.ellipsize(measure, 'abcdef', 20), 'ab…');
    assert.equal(preview.ellipsize(measure, 'abcdef', 0), '…');
});

// A 2D context that draws nothing; text is 6px per character.
const blankContext = () => new Proxy({}, {
    get: (target, key) => key in target ? target[key] : key === 'measureText' ? measure.measureText : () => {},
    set: (target, key, value) => { target[key] = value; return true; },
});

test('every part the preview card draws maps to an editable token', () => {
    const colors = Object.fromEntries(theme.TOKENS.map(token => [token, '#808080']));
    for (const status of theme.STATUS) for (const part of ['soft', 'line', 'text']) colors[`${status}-${part}`] = '#808080';
    const regions = [];
    preview.drawCard(blankContext(), colors, {mode: 'light', sans: 'sans-serif', labels: {...preview.LABELS, version: '1'}, regions});
    const drawn = new Set(regions.map(region => region[4]));
    for (const token of drawn) assert.ok(theme.TOKENS.includes(token), token);
    // Every editable token can be reached by clicking the card.
    assert.deepEqual([...drawn].sort(), [...theme.TOKENS].sort());
    // The topmost part wins: the Save label sits on the button, which sits on the focus ring.
    const label = regions.find(region => region[4] === 'btn-primary-text');
    assert.equal(preview.regionAt(regions, label[0] + 1, label[1] + 1)[4], 'btn-primary-text');
    assert.equal(preview.regionAt(regions, 1, 120)[4], 'bg-primary');
});

test('Load image reads back the theme JSON that Save as image embeds', () => {
    const png = new Uint8Array(Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        chunk('IHDR', new Uint8Array(13)),
        chunk('tEXt', Buffer.from('Comment\0not this one')),
        chunk('IEND', new Uint8Array(0)),
    ]));
    const json = JSON.stringify({light: {'bg-primary': '#fdf6e3'}, dark: {'link-color': '#ff8800'}});
    assert.equal(preview.readTextChunk(preview.withTextChunk(png, preview.KEYWORD, json), preview.KEYWORD), json);
    assert.equal(preview.readTextChunk(png, preview.KEYWORD), null);
    assert.equal(preview.readTextChunk(new Uint8Array(Buffer.from('{"light": {}}')), preview.KEYWORD), null);
});
