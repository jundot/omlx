// SPDX-License-Identifier: Apache-2.0
// The Settings rail's pure rules: deep-link anchors and the scroll position
// that decides which section is current. Run with:
// node --test tests/admin_settings_nav.test.cjs
const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const context = { console };
context.window = context;
// The rail reads a section's own scroll-margin-top (the line the browser places
// a scrolled-to section on); the fake DOM answers with whatever the test set.
context.getComputedStyle = element => ({
    scrollMarginTop: element && element.scrollMarginTop ? element.scrollMarginTop : 'auto',
});
vm.createContext(context);
vm.runInContext(
    fs.readFileSync(path.join(root, 'omlx/admin/static/js/settings_nav.js'), 'utf8'),
    context
);
const nav = context.OMLXSettingsNav;

const SECTIONS = [
    { id: 'settings-language', title: 'Language' },
    { id: 'settings-appearance', title: 'Appearance' },
    { id: 'settings-server', title: 'Server' },
];

test('a section anchor is origin + path + hash', () => {
    assert.equal(
        nav.sectionAnchor('https://box:8000', '/admin/dashboard', 'settings-server'),
        'https://box:8000/admin/dashboard#settings-server'
    );
});

test('the current section is the last one whose top has passed the offset', () => {
    const offsets = [0, 400, 900];
    assert.equal(nav.activeSection(offsets, 0, SECTIONS), 'settings-language');
    assert.equal(nav.activeSection(offsets, 100, SECTIONS), 'settings-language');
    assert.equal(nav.activeSection(offsets, 300, SECTIONS), 'settings-appearance');
    assert.equal(nav.activeSection(offsets, 800, SECTIONS), 'settings-server');
    assert.equal(nav.activeSection(offsets, 10_000, SECTIONS), 'settings-server');
});

test('a section counts as current only once the sticky bar clears it', () => {
    // A section becomes current when its top is within the clearance of the
    // viewport top — that is, when the sticky bar no longer covers it. The
    // comparison allows the last pixel (a scrolled-to section sits on the line
    // sub-pixel and all), so the flip happens one pixel before the exact line.
    const offsets = [0, 400, 900];
    assert.ok(nav.ACTIVE_OFFSET > 0);
    const boundary = 400 - nav.ACTIVE_OFFSET;
    assert.equal(nav.activeSection(offsets, boundary, SECTIONS), 'settings-appearance');
    assert.equal(nav.activeSection(offsets, boundary - 1, SECTIONS), 'settings-appearance');
    assert.equal(nav.activeSection(offsets, boundary - 2, SECTIONS), 'settings-language');
});

test('an empty registry has no current section', () => {
    assert.equal(nav.activeSection([], 0, []), null);
});

test('an explicit clearance decides the current section', () => {
    // The rail passes the sticky row's bottom edge. A section clicked in the
    // rail is scrolled to that line, and with the old constant (120) the
    // section above it stayed current -- the selection looked stuck.
    const offsets = [143, 900, 1700];
    assert.equal(nav.activeSection(offsets, 0, SECTIONS), 'settings-language');
    assert.equal(nav.activeSection(offsets, 0, SECTIONS, 143), 'settings-language');
    // Clicking the second section leaves its top exactly on the line.
    assert.equal(nav.activeSection([-757, 143, 943], 0, SECTIONS, 143), 'settings-appearance');
    // ...while a line below it (the old constant) still says the first one.
    assert.equal(nav.activeSection([-757, 143, 943], 0, SECTIONS, 120), 'settings-language');
});

test('the clearance is where a scrolled-to section lands, then the sticky row, then the constant', () => {
    // A section the rail scrolls to sits at its own scroll-margin-top, which is
    // the line that decides the current section -- reading the row's bottom
    // instead was 8px short, so the clicked section never became current.
    const section = { scrollMarginTop: '143px' };
    assert.equal(nav.activeClearance({ getBoundingClientRect: () => ({ height: 57, bottom: 135 }) }, section), 143);
    assert.equal(nav.activeClearance({ getBoundingClientRect: () => ({ height: 57, bottom: 135 }) }), 135);
    assert.equal(nav.activeClearance({ getBoundingClientRect: () => ({ height: 0, bottom: 0 }) }), nav.ACTIVE_OFFSET);
    assert.equal(nav.activeClearance(null), nav.ACTIVE_OFFSET);
    assert.equal(nav.activeClearance(null, { scrollMarginTop: 'auto' }), nav.ACTIVE_OFFSET);
});
