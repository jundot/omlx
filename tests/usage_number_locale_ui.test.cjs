// Run with: node --test tests/usage_number_locale_ui.test.cjs
// Regression test for #4227: the Usage panel's compact token counters must
// follow the oMLX UI language (<html lang>, rendered from ui.language), not the
// runtime locale. Before the fix, Intl.NumberFormat(undefined, ...) picked up
// the browser/OS locale, so an English UI on a zh-TW machine showed 3332萬.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {execFileSync} = require('node:child_process');

// OMLX_USAGE_JS lets the same assertions be pointed at origin/main to prove red.
const source = fs.readFileSync(process.env.OMLX_USAGE_JS || 'omlx/admin/static/js/usage.js', 'utf8');

const TOKENS = 33320000;

// The locales omlx/admin/i18n ships; each must resolve to itself, never to the
// runtime locale.
const SHIPPED = ['en', 'zh', 'zh-TW', 'ja', 'ko', 'fr', 'cs', 'ru', 'es', 'pt-BR'];

// CLDR uses U+00A0 (and sometimes U+202F) as the fr/cs/ru group separator; fold
// them to a plain space so the assertion survives CLDR revisions.
const fold = text => text.replace(/[\u00A0\u202F]/g, ' ');

// Load usage.js and return the Alpine component bound to a UI language. An
// undefined uiLang models a missing documentElement entirely.
function componentIn(uiLang, extra = {}) {
    const document = {hidden: false};
    if (uiLang !== undefined) document.documentElement = {lang: uiLang};
    const context = vm.createContext({fetch: () => {}, AbortController, URLSearchParams, Intl,
        window: {t: key => key}, document, setInterval, clearInterval, ...extra});
    vm.runInContext(source, context);
    return context.usageHistory();
}

function numberIn(uiLang) { return componentIn(uiLang).number; }

test('UI language, not runtime locale, selects the compact unit', () => {
    // 'en' must give SI units even when the runtime locale is CJK; 'zh-TW' must
    // keep its own units — the CJK UI is the one place where 萬 is correct.
    assert.equal(numberIn('en')(TOKENS), '33.3M');
    assert.equal(numberIn('zh-TW')(TOKENS), '3332萬');
    assert.equal(numberIn('ja')(TOKENS), '3332万');
    assert.equal(numberIn('ko')(TOKENS), '3332만');
    assert.equal(fold(numberIn('fr')(TOKENS)), '33,3 M');
});

test('every shipped UI language resolves to itself', () => {
    for (const code of SHIPPED) {
        assert.equal(componentIn(code).locale(), code);
    }
    // A compact counter for each shipped code renders without throwing.
    for (const code of SHIPPED) {
        assert.equal(typeof componentIn(code).number(TOKENS), 'string');
    }
});

test('an English UI on a zh-TW machine shows 33.3M, not 3332萬', () => {
    // Drive the real Intl in a child process whose runtime locale is zh-TW, the
    // reporter's environment. Without the fix this returns 3332萬.
    const harness = `
        const fs = require('node:fs'), vm = require('node:vm');
        const source = fs.readFileSync(${JSON.stringify(process.env.OMLX_USAGE_JS || 'omlx/admin/static/js/usage.js')}, 'utf8');
        const ctx = vm.createContext({fetch: () => {}, AbortController, URLSearchParams, Intl,
            window: {t: k => k}, document: {hidden: false, documentElement: {lang: 'en'}},
            setInterval, clearInterval});
        vm.runInContext(source, ctx);
        console.log(JSON.stringify({runtime: Intl.NumberFormat().resolvedOptions().locale,
                                     rendered: ctx.usageHistory().number(${TOKENS})}));`;
    const out = JSON.parse(execFileSync(process.execPath, ['-e', harness],
        {env: {...process.env, LANG: 'zh_TW.UTF-8', LC_ALL: 'zh_TW.UTF-8'}, encoding: 'utf8'}));
    // Guard: if the host cannot produce a CJK runtime locale this test would
    // pass vacuously, so fail loudly rather than report a false green.
    assert.match(out.runtime, /^zh/i, `expected a CJK runtime locale, got ${out.runtime}`);
    assert.equal(out.rendered, '33.3M');
});

test('unusable ui.language tags never fall back to the runtime locale', () => {
    // ui.language is an unvalidated string (omlx/admin/routes.py:704). Two kinds
    // of bad tag reach <html lang>: structurally invalid ones Intl throws on
    // ('en_US', 'None', '') and structurally valid but unresolvable ones it
    // silently maps to the runtime locale ('bogus'). On this zh-TW child both
    // must render English, and an unshipped-but-valid tag ('de') must render its
    // own locale — anything but the runtime CJK units.
    const harness = `
        const fs = require('node:fs'), vm = require('node:vm');
        const source = fs.readFileSync(${JSON.stringify(process.env.OMLX_USAGE_JS || 'omlx/admin/static/js/usage.js')}, 'utf8');
        const cases = [['bogus', 'bogus'], ['empty', ''], ['missing', null], ['unshipped', 'de'], ['malformed', 'None']];
        const out = {runtime: Intl.NumberFormat().resolvedOptions().locale, cases: {}};
        for (const [name, lang] of cases) {
            const document = {hidden: false};
            if (lang !== null) document.documentElement = {lang};
            const ctx = vm.createContext({fetch: () => {}, AbortController, URLSearchParams, Intl,
                window: {t: k => k}, document, setInterval, clearInterval});
            vm.runInContext(source, ctx);
            const view = ctx.usageHistory();
            out.cases[name] = {locale: view.locale(), rendered: view.number(${TOKENS})};
        }
        console.log(JSON.stringify(out));`;
    const out = JSON.parse(execFileSync(process.execPath, ['-e', harness],
        {env: {...process.env, LANG: 'zh_TW.UTF-8', LC_ALL: 'zh_TW.UTF-8'}, encoding: 'utf8'}));
    assert.match(out.runtime, /^zh/i, `expected a CJK runtime locale, got ${out.runtime}`);
    for (const name of ['bogus', 'empty', 'missing', 'malformed']) {
        assert.equal(out.cases[name].locale, 'en', `${name}: expected the English fallback`);
        assert.equal(out.cases[name].rendered, '33.3M', `${name}: leaked the runtime locale`);
    }
    // 'de' is a valid tag omlx does not ship: ICU resolves it (or, on a minimal
    // ICU build, it falls back to English) — never to the zh-TW runtime units.
    assert.doesNotMatch(out.cases.unshipped.rendered, /[萬万만]/,
        `unshipped tag leaked the runtime locale: ${out.cases.unshipped.rendered}`);
});

test('number() and exact() pass the resolved locale, never undefined', () => {
    // A recording Intl makes the locale argument observable on any host, so the
    // tooltip/aria-label formatting is pinned to the same locale as the counter
    // without depending on the machine's ICU data.
    const seen = [];
    const spy = {
        NumberFormat: function (locale, options) {
            seen.push(locale);
            return new Intl.NumberFormat(locale, options);
        },
    };
    spy.NumberFormat.supportedLocalesOf = Intl.NumberFormat.supportedLocalesOf.bind(Intl.NumberFormat);
    const view = componentIn('en', {Intl: spy});
    assert.equal(view.number(TOKENS), '33.3M');
    assert.equal(view.locale(), 'en');
    assert.equal(view.number(1500), '1.5K');
    assert.equal(view.exact(1500), '1,500');
    assert.equal(view.exact(TOKENS), '33,320,000');
    for (const locale of seen) assert.notEqual(locale, undefined);
    assert.deepEqual([...new Set(seen)], ['en']);
});

test('a ui.language Intl cannot parse does not break the panel', () => {
    // ui.language is an unvalidated string (omlx/admin/routes.py:704), so a
    // hand-edited settings.json can reach <html lang>. Intl throws RangeError
    // on such tags; the counters must still render, in English.
    for (const uiLang of ['en_US', '!!', '', '   ', undefined]) {
        assert.doesNotThrow(() => numberIn(uiLang)(TOKENS));
        assert.equal(numberIn(uiLang)(TOKENS), '33.3M');
        assert.equal(componentIn(uiLang).locale(), 'en');
    }
});

test('counter rendering is unchanged for zero, missing and fractional input', () => {
    for (const uiLang of ['en', 'zh-TW']) {
        const number = numberIn(uiLang);
        assert.equal(number(0), '0');
        assert.equal(number(null), '0');
        assert.equal(number(undefined), '0');
    }
    // Values derived from the ECMA-402 compact spec, not from number() itself:
    // en rounds 0.42 to one decimal and reaches the K unit at 1500, while zh-TW
    // keeps 1500 unscaled because it has no 千 unit.
    assert.equal(numberIn('en')(0.42), '0.4');
    assert.equal(numberIn('en')(999), '999');
    assert.equal(numberIn('en')(1500), '1.5K');
    assert.equal(numberIn('zh-TW')(1500), '1500');
});

test('the Usage template formats tooltips and aria-labels with exact()', () => {
    // The counters, their :title tooltips and the heatmap aria-label must share
    // one locale; a bare toLocaleString() follows the runtime locale instead.
    const template = fs.readFileSync('omlx/admin/templates/dashboard/_usage.html', 'utf8');
    assert.doesNotMatch(template, /toLocaleString/);
    assert.equal((template.match(/exact\(/g) || []).length, 4);
});
