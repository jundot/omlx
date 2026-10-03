// SPDX-License-Identifier: Apache-2.0
// The Settings rail as it behaves in the dashboard: the section registry is
// read from the document, a hash selects the right sub-tab, and the copy
// control produces the deep link. Run with:
// node --test tests/admin_settings_rail.test.cjs
const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const scrollState = { y: 0 };

const REGISTRY = {
    global: [
        { id: 'settings-language', title: 'Language', badge: 'live' },
        { id: 'settings-appearance', title: 'Appearance', badge: 'live' },
        { id: 'settings-server', title: 'Server', badge: 'none' },
    ],
    integrations: [
        { id: 'settings-int-websearch', title: 'Web Search', badge: 'live' },
    ],
    models: [
        { id: 'settings-models', title: 'Model Settings', badge: 'live' },
    ],
};

function makeElement(id, top, height = 400) {
    return {
        id,
        top,
        height,
        scrolls: [],
        getClientRects: () => [{ top, height }],
        getBoundingClientRect: () => ({ top, height }),
        scrollIntoView(options) {
            this.scrolls.push(options);
        },
    };
}

function makeApp({ elements = {}, hash = '', clipboard = null } = {}) {
    const listeners = {};
    const context = {
        console,
        navigator: {
            clipboard: {
                writeText: text => {
                    clipboard.value = text;
                    return Promise.resolve();
                },
            },
        },
        setTimeout,
        requestAnimationFrame: fn => fn(),
        URL,
        localStorage: { getItem: () => null, setItem: () => {} },
    };
    context.window = context;
    context.document = {
        getElementById: id => (id === 'settings-sections'
            ? { textContent: JSON.stringify(REGISTRY) }
            : elements[id] || null),
        // The rail measures the sticky sub-tab row's bottom edge to know where a
        // section has to reach to count as current; the fake DOM has no such
        // element, so the nav falls back to its constant.
        querySelector: () => null,
        addEventListener: (name, fn) => {
            (listeners[name] = listeners[name] || []).push(fn);
        },
        removeEventListener: (name, fn) => {
            listeners[name] = (listeners[name] || []).filter(item => item !== fn);
        },
        createElement: () => ({ style: {}, focus() {}, select() {}, remove() {} }),
        body: { appendChild() {} },
        documentElement: { setAttribute() {}, removeAttribute() {} },
    };
    context.window.location = {
        href: 'https://box:8000/admin/dashboard?tab=status',
        origin: 'https://box:8000',
        pathname: '/admin/dashboard',
        search: '?tab=status',
        hash,
    };
    context.window.history = { replaceState: (state, title, url) => { context.window.location.hash = url; } };
    context.window.scrollY = 0;
    context.window.isSecureContext = true;
    context.window.t = key => key;
    context.window.__omlxLang = 'en';
    context.window.confirm = () => true;
    context.window.addEventListener = context.document.addEventListener;
    context.window.removeEventListener = context.document.removeEventListener;
    context.window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
    vm.createContext(context);
    for (const name of ['settings_nav.js', 'dashboard.js']) {
        vm.runInContext(
            fs.readFileSync(path.join(root, 'omlx/admin/static/js', name), 'utf8'),
            context
        );
    }
    const app = context.dashboard();
    app.$nextTick = fn => fn();
    return { app, context, listeners };
}

test('the rail starts on the active sub-tab with every section', () => {
    const { app } = makeApp();
    app.settingsInitSections();
    assert.equal(app.settingsSections.map(s => s.id).join(), [
        'settings-language',
        'settings-appearance',
        'settings-server',
    ].join());
    assert.equal(app.settingsActiveSection, 'settings-language');
});

test('switching sub-tab swaps the rail to that tab\'s sections', () => {
    const { app } = makeApp();
    app.settingsInitSections();
    app.activeTab = 'integrations';
    app.settingsInitSections();
    assert.equal(app.settingsSections.map(s => s.id).join(), 'settings-int-websearch');
    app.activeTab = 'models';
    app.settingsInitSections();
    assert.equal(app.settingsSections.map(s => s.id).join(), 'settings-models');
});

test('a hash deep-links to the section on its own sub-tab', () => {
    const { app } = makeApp({ hash: '#settings-int-websearch' });
    assert.equal(app.settingsScrollToHash('#settings-int-websearch'), true);
    assert.equal(app.mainTab, 'settings');
    assert.equal(app.activeTab, 'integrations');
    assert.equal(app.settingsActiveSection, 'settings-int-websearch');
});

test('a hash that is not a settings section is ignored', () => {
    const { app } = makeApp();
    app.mainTab = 'status';
    assert.equal(app.settingsScrollToHash('#bench'), false);
    assert.equal(app.mainTab, 'status', 'an unrelated anchor does not move the tab');
});

test('clicking a rail item marks it, scrolls to it and keeps the link', () => {
    const target = makeElement('settings-server', 900);
    const { app, context } = makeApp({ elements: { 'settings-server': target } });
    app.settingsInitSections();
    app.settingsGoToSection('settings-server');
    assert.equal(app.settingsActiveSection, 'settings-server');
    assert.equal(target.scrolls.length, 1);
    assert.equal(target.scrolls[0].behavior, 'smooth');
    assert.equal(target.scrolls[0].block, 'start');
    assert.equal(context.window.location.hash, '#settings-server');
});

test('reduced motion jumps instead of animating', () => {
    const target = makeElement('settings-server', 900);
    const { app, context } = makeApp({ elements: { 'settings-server': target } });
    context.window.matchMedia = () => ({ matches: true });
    app.settingsScrollToSection('settings-server');
    assert.equal(target.scrolls.length, 1);
    assert.equal(target.scrolls[0].block, 'start');
    assert.equal(target.scrolls[0].behavior, undefined, 'reduced motion does not animate');
});

test('the copy control copies the deep link and reports it', () => {
    const copied = { value: null };
    const { app } = makeApp({ clipboard: copied });
    app.settingsCopyAnchor('settings-server');
    assert.equal(copied.value, 'https://box:8000/admin/dashboard#settings-server');
    assert.equal(app.settingsCopiedAnchor, 'settings-server');
});

test('the scroll watcher follows the section the reader is in', () => {
    // Document tops; the elements report viewport-relative rects, so each one
    // is measured as `top - scrollY` like a real browser would.
    const documentTops = {
        'settings-language': 0,
        'settings-appearance': 400,
        'settings-server': 900,
    };
    const elements = {};
    for (const [id, top] of Object.entries(documentTops)) {
        const element = makeElement(id, top);
        element.getClientRects = () => [{ top: top - scrollState.y }];
        element.getBoundingClientRect = () => ({ top: top - scrollState.y });
        elements[id] = element;
    }
    const panel = makeElement('panel-settings', 0);
    const { app, context, listeners } = makeApp({ elements });
    app.settingsInitSections();
    context.document.getElementById = id => {
        if (id === 'settings-sections') return { textContent: JSON.stringify(REGISTRY) };
        if (id === 'panel-settings') return panel;
        return elements[id] || { getClientRects: () => [] };
    };
    app.settingsWatchScroll();
    assert.ok(listeners.scroll, 'the page scroll is watched');
    for (const [scrollY, expected] of [[0, 'settings-language'], [500, 'settings-appearance'], [900, 'settings-server'], [4000, 'settings-server']]) {
        scrollState.y = scrollY;
        context.window.scrollY = scrollY;
        listeners.scroll.forEach(fn => fn());
        assert.equal(app.settingsActiveSection, expected, `at scrollY ${scrollY}`);
    }
});

test('scrolling is skipped while the settings tab is hidden', () => {
    const target = makeElement('settings-server', 900);
    const hidden = { getClientRects: () => [] };
    const { app, context } = makeApp({ elements: { 'settings-server': target } });
    context.document.getElementById = id => {
        if (id === 'panel-settings') return hidden;
        return id === 'settings-server' ? target : null;
    };
    app.settingsScrollToSection('settings-server');
    assert.equal(target.scrolls.length, 0, 'a hidden tab does not scroll the page');
});

test('a hidden settings panel keeps the rail out of the sync', () => {
    const hidden = { getClientRects: () => [] };
    const { app, context } = makeApp();
    context.document.getElementById = id => {
        if (id === 'settings-sections') return { textContent: JSON.stringify(REGISTRY) };
        if (id === 'panel-settings') return hidden;
        return null;
    };
    app.settingsInitSections();
    app.settingsActiveSection = 'settings-language';
    app.settingsWatchScroll();
    assert.equal(app.settingsActiveSection, 'settings-language',
        'a panel that is not shown leaves the rail where it was');
});
