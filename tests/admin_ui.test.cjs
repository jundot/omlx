// SPDX-License-Identifier: Apache-2.0
// The shared console layer's pure helpers (omlx/admin/static/js/ui.js): the
// command-palette matcher and grouping, the toast semantics, and the number
// transition. The DOM half of ui.js needs a browser; these are the parts that
// decide what the user sees, and they are pure on purpose.
// Run with: node --test tests/admin_ui.test.cjs
const assert = require('assert/strict');
const path = require('path');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const ui = require(path.join(root, 'omlx/admin/static/js/ui.js'));

test('the matcher ranks exact over prefix over word start over substring', () => {
    assert.equal(ui.matchScore('Status', 'status'), 0, 'exact, case-insensitive');
    assert.equal(ui.matchScore('Status page', 'status'), 1, 'prefix');
    assert.equal(ui.matchScore('Go to status', 'status'), 2, 'word start');
    assert.equal(ui.matchScore('Superstatus', 'status'), 3, 'substring in a word');
    assert.equal(ui.matchScore('Settings', 'stg'), 4, 'subsequence');
    assert.equal(ui.matchScore('Settings', 'zzz'), -1, 'no match at all');
    assert.equal(ui.matchScore('', 'x'), -1, 'nothing to match against');
    assert.equal(ui.matchScore('anything', ''), 0, 'an empty query matches');
});

test('the palette ranks a prefix match above a substring one, and drops the rest', () => {
    const commands = [
        { label: 'Open logs', group: 'Pages' },
        { label: 'Status', group: 'Pages' },
        { label: 'Downloads', group: 'Models' },
    ];
    const all = ui.filterCommands(commands, '');
    assert.deepEqual(all.map((command) => command.label), [
        'Open logs', 'Status', 'Downloads',
    ], 'an empty query keeps the caller\'s order');

    assert.deepEqual(ui.filterCommands(commands, 'sta').map((c) => c.label), ['Status']);

    // A prefix match outranks a substring one; the caller's order only breaks
    // ties.
    const ranked = ui.filterCommands(
        [{ label: 'Open logs' }, { label: 'Superstatus' }, { label: 'Status' }], 'status'
    );
    assert.deepEqual(ranked.map((command) => command.label), ['Status', 'Superstatus']);

    // Keywords are searched too, so a command is reachable by a word that is
    // not in its label.
    const byKeyword = ui.filterCommands(
        [{ label: 'Open logs', group: 'Pages', keywords: ['tail'] }], 'tail'
    );
    assert.equal(byKeyword.length, 1);
});

test('groups keep the first-seen order and honour the result limit', () => {
    const commands = [
        { label: 'a', group: 'Pages' },
        { label: 'b', group: 'Models' },
        { label: 'c', group: 'Pages' },
    ];
    const groups = ui.groupCommands(commands, 10);
    assert.deepEqual(groups.map((group) => group.name), ['Pages', 'Models']);
    assert.deepEqual(groups[0].commands.map((command) => command.label), ['a', 'c']);

    const limited = ui.groupCommands(commands, 2);
    assert.equal(limited.reduce((total, group) => total + group.commands.length, 0), 2);
    assert.deepEqual(limited.map((group) => group.name), ['Pages', 'Models']);
});

test('only the error tone interrupts a screen reader', () => {
    assert.deepEqual(ui.toastSemantics('red'), { role: 'alert', ariaLive: 'assertive' });
    for (const tone of ui.TOAST_TONES.filter((value) => !ui.ERROR_TONES.includes(value))) {
        assert.deepEqual(ui.toastSemantics(tone), { role: 'status', ariaLive: 'polite' }, tone);
    }
});

test('the number transition eases to its endpoints and keeps the decimals asked for', () => {
    assert.equal(ui.easeOutCubic(0), 0);
    assert.equal(ui.easeOutCubic(1), 1);
    assert.ok(ui.easeOutCubic(0.5) > 0.5, 'the curve is eased out, not linear');
    assert.equal(ui.easeOutCubic(-1), 0, 'clamped below');
    assert.equal(ui.easeOutCubic(2), 1, 'clamped above');

    assert.equal(ui.transitionValue(0, 10, 0, 0), '0');
    assert.equal(ui.transitionValue(0, 10, 1, 0), '10');
    assert.equal(ui.transitionValue(0, 10, 0.5, 1), '8.8');
});
