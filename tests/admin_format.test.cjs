// SPDX-License-Identifier: Apache-2.0
// The console has exactly one count formatter (omlx/admin/static/js/format.js).
// These vectors pin its output per language: the bug this replaces was the
// browser language deciding the scale, so a Chinese screen could show "1.9万"
// next to "1.1M". Run with: node --test tests/admin_format.test.cjs
const assert = require('assert/strict');
const path = require('path');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const { formatCount, formatCountExact, formatParams, formatMetric } = require(
    path.join(root, 'omlx/admin/static/js/format.js')
);

test('Chinese counts use 万/亿 and never K/M', () => {
    assert.equal(formatCount(0, 'zh'), '0');
    assert.equal(formatCount(999, 'zh'), '999');
    assert.equal(formatCount(9999, 'zh'), '9,999');
    assert.equal(formatCount(10000, 'zh'), '1万');
    assert.equal(formatCount(19290, 'zh'), '1.9万');
    assert.equal(formatCount(1070000, 'zh'), '107万');
    assert.equal(formatCount(12345678, 'zh'), '1,234.6万');
    // The mantissa is grouped like any other count: 1,234.5万, not 1234.5万.
    assert.equal(formatCount(12345000, 'zh'), '1,234.5万');
    assert.equal(formatCount(123456789, 'zh'), '1.2亿');
    // The top of the ladder: 万亿, spelled out here rather than left to Intl,
    // because engines disagree about how far the Chinese scale goes.
    assert.equal(formatCount(560000000000, 'zh'), '5,600亿');
    assert.equal(formatCount(1200000000000, 'zh'), '1.2万亿');
    assert.equal(formatCount(34000000000000, 'zh'), '34万亿');
    assert.equal(formatCount(-4200000000000, 'zh'), '-4.2万亿');
});

test('a mantissa that rounds up to 10,000 steps to the next unit', () => {
    // 99,999,999 is 9,999.99999万: at one decimal it rounds to 10,000.0万,
    // which is 1亿 — the unit follows what is printed, not the raw magnitude.
    assert.equal(formatCount(99999999, 'zh'), '1亿');
    assert.equal(formatCount(99999999, 'zh-TW'), '1億');
    assert.equal(formatCount(999999999999, 'zh'), '1万亿');
    assert.equal(formatCount(999999999999, 'zh-TW'), '1兆');
    // Just below the roll-over the mantissa still has room to round down.
    assert.equal(formatCount(99990000, 'zh'), '9,999万');
    assert.equal(formatCount(9999999, 'zh'), '1,000万');
});

test('a value that is not a finite number reads as an em dash', () => {
    // The metric formatter already refuses a non-finite reading; a count
    // must not print "∞万亿" for one.
    for (const value of [Infinity, -Infinity, NaN, undefined, null, '']) {
        assert.equal(formatCount(value, 'zh'), '—');
        assert.equal(formatCount(value, 'en'), '—');
        assert.equal(formatCountExact(value, 'zh'), '—');
        assert.equal(formatParams(value), '—');
    }
    assert.equal(formatCountExact(1234567, 'zh'), '1,234,567');
});

test('Traditional Chinese keeps its own characters', () => {
    assert.equal(formatCount(10000, 'zh-TW'), '1萬');
    assert.equal(formatCount(12345678, 'zh-TW'), '1,234.6萬');
    assert.equal(formatCount(120000000, 'zh-TW'), '1.2億');
    // Taiwan stops the ladder at 兆 rather than 万亿.
    assert.equal(formatCount(1200000000000, 'zh-TW'), '1.2兆');
    assert.equal(formatCount(34000000000000, 'zh-TW'), '34兆');
});

test('English counts use K/M/B/T', () => {
    assert.equal(formatCount(999, 'en'), '999');
    assert.equal(formatCount(1000, 'en'), '1K');
    assert.equal(formatCount(1070000, 'en'), '1.1M');
    assert.equal(formatCount(19290000, 'en'), '19.3M');
    assert.equal(formatCount(2500000000, 'en'), '2.5B');
    assert.equal(formatCount(4000000000000, 'en'), '4T');
});

// The scale is the console language's, not a fixed K/M/B/T: the point of the
// one formatter is that a Russian console reads тыс. and a Japanese one 万,
// which is what the platform's compact notation already knows. The unit is
// asserted rather than the whole string, because the digits' separators are
// the engine's business.
test('every catalogue counts in its own language', () => {
    const units = {
        en: 'K',
        ja: '万',
        ko: '만',
        ru: 'тыс',
        cs: 'tis',
        fr: 'k',
        es: 'mil',
        'pt-BR': 'mil',
    };
    for (const [locale, unit] of Object.entries(units)) {
        const out = formatCount(19290, locale);
        assert.ok(out.includes(unit), `${locale}: ${out} is missing ${unit}`);
    }
    // Chinese and Japanese are different scales spelled with different
    // characters, and neither may fall back to the English letter.
    assert.equal(formatCount(19290, 'ja'), '1.9万');
    assert.equal(formatCount(19290, 'ko'), '1.9만');
    assert.ok(!formatCount(19290, 'ru').includes('K'));
});

test('the console language wins over the browser language', () => {
    // globalThis.__omlxLang is what base.html injects; a Chinese browser must
    // not change what an English console prints.
    globalThis.__omlxLang = 'en';
    try {
        assert.equal(formatCount(19290), '19.3K');
        globalThis.__omlxLang = 'zh';
        assert.equal(formatCount(19290), '1.9万');
    } finally {
        delete globalThis.__omlxLang;
    }
});

test('missing values print an em dash rather than a zero', () => {
    assert.equal(formatCount(null, 'en'), '—');
    assert.equal(formatCount(undefined, 'zh'), '—');
    assert.equal(formatCount(NaN, 'zh'), '—');
    assert.equal(formatCountExact(null, 'zh'), '—');
});

test('negatives keep their sign', () => {
    assert.equal(formatCount(-2500, 'zh'), '-2,500');
    assert.equal(formatCount(-25000, 'zh'), '-2.5万');
    assert.equal(formatCount(-2500, 'en'), '-2.5K');
});

test('exact counts stay exact for titles', () => {
    assert.equal(formatCountExact(19290, 'zh'), '19,290');
    assert.equal(formatCountExact(1070000, 'en'), '1,070,000');
});

test('parameter counts stay on the SI ladder in every language', () => {
    assert.equal(formatParams(8e9), '8B');
    assert.equal(formatParams(7.5e9), '7.5B');
    assert.equal(formatParams(1.5e12), '1.5T');
    assert.equal(formatParams(500e6), '500M');
    assert.equal(formatParams(null), '—');
});

// The chat footer used to print "0.0" for a speed that had not been measured,
// because a fresh stats object holds zeros. A performance figure is now either
// a measurement or an em dash — never a zero standing in for "no data".
test('a metric without a measurement prints an em dash, not a zero', () => {
    assert.equal(formatMetric(0), '—');
    assert.equal(formatMetric('0'), '—');
    assert.equal(formatMetric(null), '—');
    assert.equal(formatMetric(undefined), '—');
    assert.equal(formatMetric(''), '—');
    assert.equal(formatMetric(NaN), '—');
    assert.equal(formatMetric(Infinity), '—');
    assert.equal(formatMetric(-1), '—');
});

test('a measured metric keeps its digits', () => {
    assert.equal(formatMetric(12.3456), '12.3');
    assert.equal(formatMetric(12.3456, 2), '12.35');
    assert.equal(formatMetric(0.4), '0.4');
    assert.equal(formatMetric('7.5'), '7.5');
    assert.equal(formatMetric(1, 0), '1');
});
