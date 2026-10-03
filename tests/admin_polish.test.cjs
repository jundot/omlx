// SPDX-License-Identifier: Apache-2.0
// The console-wide helpers in omlx/admin/static/js/ui.js: the number
// transition, the palette's search and the toast semantics. The DOM wiring
// (focus trap, dialog lifecycle) is pinned statically by
// tests/test_admin_polish.py. Run with: node --test tests/admin_polish.test.cjs
const assert = require('assert/strict');
const path = require('path');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const ui = require(path.join(root, 'omlx/admin/static/js/ui.js'));
// The DOM-touching entry points hang off the global, exactly as the console
// uses them; module.exports carries the pure helpers.
const countUp = globalThis.omlxCountUp;

// --- a clock and a frame queue, so the animation is deterministic -----------

let clock = 0;
let frames = new Map();
let frameId = 0;

globalThis.performance = { now: () => clock };
globalThis.requestAnimationFrame = callback => {
    frameId += 1;
    frames.set(frameId, callback);
    return frameId;
};
globalThis.cancelAnimationFrame = id => {
    frames.delete(id);
};

function queued() {
    return frames.size;
}

function runFrame() {
    const pending = Array.from(frames.values());
    frames.clear();
    pending.forEach(callback => callback(clock));
}

function element() {
    return { textContent: '', __omlxValue: undefined, __omlxCurrent: undefined, __omlxFrame: null };
}

function reset() {
    clock = 0;
    frames = new Map();
}

// --- toast semantics -------------------------------------------------------

test('only the error tone interrupts', () => {
    assert.deepEqual(ui.toastSemantics('red'), { role: 'alert', ariaLive: 'assertive' });
    for (const tone of ['green', 'orange', 'blue', 'neutral']) {
        assert.deepEqual(ui.toastSemantics(tone), { role: 'status', ariaLive: 'polite' }, tone);
    }
    assert.deepEqual(ui.TOAST_TONES, ['green', 'orange', 'red', 'blue', 'neutral']);
    assert.deepEqual(ui.ERROR_TONES, ['red']);
});

// --- number transition -----------------------------------------------------

test('counts round to whole units and keep one decimal where asked', () => {
    assert.equal(ui.transitionValue(0, 100, 1, 0), '100');
    assert.equal(ui.transitionValue(0, 100, 0.5, 0), '88');
    assert.equal(ui.transitionValue(0, 100, 0.5, 0).includes('.'), false);
    assert.equal(ui.transitionValue(0, 10, 0.5, 1), '8.8');
    assert.equal(ui.transitionValue(0, 10, 0.5, 1).split('.')[1].length, 1);
});

test('the eased curve starts at the old value and ends at the new one', () => {
    assert.equal(ui.easeOutCubic(0), 0);
    assert.equal(ui.easeOutCubic(1), 1);
    assert.equal(ui.easeOutCubic(2), 1, 'clamped above one');
    assert.equal(ui.easeOutCubic(-1), 0, 'clamped below zero');
    assert.ok(ui.easeOutCubic(0.5) > 0.5, 'ease-out front-loads the movement');
});

test('the animated element settles on the exact target', () => {
    reset();
    const node = element();
    countUp(node, 1250, { format: value => String(value) });
    assert.equal(queued(), 1, 'the first frame is scheduled');
    clock = 150;
    runFrame();
    assert.equal(node.textContent, '1094', 'half way through the ease-out');
    clock = 300;
    runFrame();
    assert.equal(node.textContent, '1250');
    assert.equal(node.__omlxValue, 1250);
    assert.equal(queued(), 0, 'no frame is left running');
});

test('a second update animates from the value on screen', () => {
    reset();
    const node = element();
    countUp(node, 100, {});
    clock = 300;
    runFrame();
    assert.equal(node.textContent, '100');
    countUp(node, 200, {});
    clock = 316;
    runFrame();
    const mid = Number(node.textContent);
    assert.ok(mid > 100 && mid < 200, `expected a value between 100 and 200, saw ${mid}`);
    clock = 616;
    runFrame();
    assert.equal(node.textContent, '200');
});

test('a one-decimal figure keeps its decimal while it animates', () => {
    reset();
    const node = element();
    countUp(node, 42.5, { decimals: 1 });
    clock = 150;
    runFrame();
    assert.match(node.textContent, /^\d+\.\d$/, node.textContent);
    clock = 300;
    runFrame();
    assert.equal(node.textContent, '42.5');
});

test('reduced motion jumps straight to the value', () => {
    reset();
    globalThis.matchMedia = () => ({ matches: true });
    const node = element();
    countUp(node, 999, { format: value => String(value) });
    assert.equal(node.textContent, '999');
    assert.equal(queued(), 0, 'nothing was scheduled');
    globalThis.matchMedia = () => ({ matches: false });
});

test('a missing value prints the dash instead of a zero', () => {
    reset();
    const node = element();
    countUp(node, undefined, {});
    assert.equal(node.textContent, ui.DASH);
    countUp(node, null, {});
    assert.equal(node.textContent, ui.DASH);
    countUp(node, 'not a number', {});
    assert.equal(node.textContent, ui.DASH);
});

test('an unchanged value is written exactly, without animating', () => {
    reset();
    const node = element();
    countUp(node, 7, {});
    clock = 300;
    runFrame();
    node.textContent = 'stale';
    countUp(node, 7, {});
    assert.equal(node.textContent, '7');
    assert.equal(queued(), 0);
});

// --- palette search --------------------------------------------------------

const COMMANDS = [
    { id: 'status', label: 'Dashboard', group: 'Go to' },
    { id: 'models', label: 'Models', group: 'Dashboard tabs' },
    { id: 'blocks', label: 'Serving Stats', group: 'Dashboard blocks' },
    { id: 'logs', label: 'Logs', group: 'Dashboard tabs', keywords: ['journal'] },
];

test('an empty query keeps every command in its published order', () => {
    assert.deepEqual(ui.filterCommands(COMMANDS, '').map(c => c.id),
        ['status', 'models', 'blocks', 'logs']);
    assert.deepEqual(ui.filterCommands(COMMANDS, '   ').map(c => c.id),
        ['status', 'models', 'blocks', 'logs']);
});

test('ranking prefers an exact label, then a prefix, then a word start', () => {
    assert.equal(ui.matchScore('Logs', 'logs'), 0);
    assert.equal(ui.matchScore('Logs', 'log'), 1);
    assert.equal(ui.matchScore('Dashboard logs', 'logs'), 2);
    assert.equal(ui.matchScore('Dialog logs', 'logs'), 2, 'a word start beats a substring');
    assert.equal(ui.matchScore('Catalog', 'alog'), 3);
    assert.equal(ui.matchScore('Long odds', 'logs'), 4, 'subsequence');
    assert.equal(ui.matchScore('Models', 'xyz'), -1);
});

test('search reaches labels, groups and keywords and stays stable', () => {
    assert.deepEqual(ui.filterCommands(COMMANDS, 'log').map(c => c.id), ['logs']);
    assert.deepEqual(ui.filterCommands(COMMANDS, 'journal').map(c => c.id), ['logs']);
    assert.deepEqual(ui.filterCommands(COMMANDS, 'tabs').map(c => c.id),
        ['models', 'logs'], 'the group name is searchable too');
    assert.deepEqual(ui.filterCommands(COMMANDS, 'dashboard').map(c => c.id),
        ['status', 'models', 'blocks', 'logs'], 'a whole group matches together');
    assert.deepEqual(ui.filterCommands(COMMANDS, 'zzz'), []);
});

test('groups keep the order the commands were published in', () => {
    const groups = ui.groupCommands(COMMANDS, 12);
    assert.deepEqual(groups.map(group => group.name), ['Go to', 'Dashboard tabs', 'Dashboard blocks']);
    assert.deepEqual(groups[1].commands.map(command => command.id), ['models', 'logs']);
    assert.deepEqual(ui.groupCommands(COMMANDS, 2).map(group => group.name), ['Go to', 'Dashboard tabs']);
    assert.equal(ui.groupCommands(COMMANDS, 2)[1].commands.length, 1, 'the limit is honoured');
});

test('the palette caps how many commands it shows', () => {
    const many = Array.from({ length: 30 }, (_, index) => ({ id: `c${index}`, label: `C${index}` }));
    assert.equal(ui.groupCommands(many).flatMap(group => group.commands).length, ui.MAX_RESULTS);
    assert.equal(ui.MAX_RESULTS, 12);
});

// --- what the dashboard registers with the palette --------------------------

const fs = require('node:fs');
const vm = require('node:vm');

function dashboardApp(windowExtras = {}) {
    const context = {
        localStorage: { getItem: () => null },
        window: Object.assign(
            { t: key => key, location: { href: '/admin/dashboard' } },
            windowExtras
        ),
        console,
        alert: message => { throw Error(message); },
        URL,
        setTimeout,
    };
    vm.createContext(context);
    vm.runInContext(
        fs.readFileSync(path.join(root, 'omlx/admin/static/js/dashboard_layout.js'), 'utf8'),
        context
    );
    // dashboard.js resolves the layout registry from the global scope.
    context.DashboardLayout = context.window.DashboardLayout;
    vm.runInContext(
        fs.readFileSync(path.join(root, 'omlx/admin/static/js/dashboard.js'), 'utf8'),
        context
    );
    return context.dashboard();
}

test('the dashboard registers its tabs, blocks and settings sections', () => {
    const app = dashboardApp();
    app.globalSettings.server.distributed_inference_active = false;
    app.dashPlacedIds = ['serving_stats'];
    const commands = app.paletteCommands();
    const ids = commands.map(command => command.id);

    // Arrays built inside the vm carry the vm's Array.prototype, which
    // deepStrictEqual compares by identity.
    const tabs = Array.from(commands).filter(command => command.group === 'palette.group_tabs');
    assert.deepEqual(Array.from(tabs, command => command.label), [
        'navbar.tab.status', 'navbar.tab.models', 'navbar.tab.settings',
        'navbar.tab.logs', 'navbar.tab.bench',
    ], 'cluster is hidden while distributed inference is off');
    assert.deepEqual(
        Array.from(commands).filter(command => command.group === 'palette.group_blocks').length,
        contextBlocks(app),
        'every registered block is a command'
    );
    assert.deepEqual(
        Array.from(commands).filter(command => command.group === 'palette.group_settings').map(c => c.label),
        ['settings.tab.global', 'settings.tab.models', 'settings.tab.integrations']
    );

    const cluster = commands.find(command => command.id === 'tab-cluster');
    assert.equal(cluster, undefined);
    app.globalSettings.server.distributed_inference_active = true;
    assert.ok(ids.length < app.paletteCommands().length, 'cluster appears when it is available');

    const placed = app.paletteCommands().find(command => command.id === 'block-serving_stats');
    assert.equal(placed.hint, '', 'a placed block needs no hint');
    const parked = app.paletteCommands().find(command => command.id === 'block-api_endpoints');
    assert.equal(parked.hint, 'palette.block_not_placed');
});

function contextBlocks(app) {
    return app._dashLayoutLib().BLOCK_IDS.length;
}

test('a tab command really switches tabs', () => {
    const app = dashboardApp();
    app.syncTabStateToUrl = () => {};
    const logs = app.paletteCommands().find(command => command.id === 'tab-logs');
    logs.run();
    assert.equal(app.mainTab, 'logs');
    const settings = app.paletteCommands().find(command => command.id === 'settings-integrations');
    settings.run();
    assert.equal(app.mainTab, 'settings');
    assert.equal(app.activeTab, 'integrations');
});

test('the palette asks for the commands when it opens, not when it loads', () => {
    const registered = [];
    const app = dashboardApp({ omlxPalette: { register: build => registered.push(build) } });
    app.registerPaletteCommands();
    assert.equal(registered.length, 1);
    assert.equal(typeof registered[0](), 'object', 'the builder returns commands');
    app.registerPaletteCommands();
    assert.equal(registered.length, 1, 'registering twice must not duplicate the source');
});
