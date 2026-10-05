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

// Load usage.js and return just the number() formatter bound to a UI language.
function numberIn(uiLang, extra = {}) {
    const document = {hidden: false};
    if (uiLang !== undefined) document.documentElement = {lang: uiLang};
    const context = vm.createContext({fetch: () => {}, AbortController, URLSearchParams, Intl,
        window: {t: key => key}, document, setInterval, clearInterval, ...extra});
    vm.runInContext(source, context);
    return context.usageHistory().number;
}

test('UI language, not runtime locale, selects the compact unit', () => {
    // 'en' must give SI units even when the runtime locale is CJK; 'zh-TW' must
    // keep its own units — the CJK UI is the one place where 萬 is correct.
    assert.equal(numberIn('en')(TOKENS), '33.3M');
    assert.equal(numberIn('zh-TW')(TOKENS), '3332萬');
    assert.equal(numberIn('ja')(TOKENS), '3332万');
    assert.equal(numberIn('ko')(TOKENS), '3332만');
    assert.equal(numberIn('fr')(TOKENS), '33,3 M');
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

test('a ui.language Intl cannot parse does not break the panel', () => {
    // ui.language is an unvalidated string (omlx/admin/routes.py:704), so a
    // hand-edited settings.json can reach <html lang>. Intl throws RangeError
    // on such tags; the counters must still render.
    assert.doesNotThrow(() => numberIn('en_US')(TOKENS));
    assert.doesNotThrow(() => numberIn('!!')(TOKENS));
    assert.doesNotThrow(() => numberIn('')(TOKENS));
    assert.doesNotThrow(() => numberIn(undefined)(TOKENS));   // no documentElement at all
    assert.doesNotThrow(() => numberIn('   ')(TOKENS));
});

test('counter rendering is unchanged for zero, missing and fractional input', () => {
    for (const uiLang of ['en', 'zh-TW']) {
        const number = numberIn(uiLang);
        assert.equal(number(0), '0');
        assert.equal(number(null), '0');
        assert.equal(number(undefined), '0');
        assert.equal(number(0.42), number(0.42));
        assert.equal(number(1500), number(1500));
    }
    assert.equal(numberIn('en')(0), '0');
    assert.equal(numberIn('zh-TW')(0), '0');
    assert.equal(numberIn('en')(999), '999');
    assert.equal(numberIn('en')(1500), '1.5K');
});