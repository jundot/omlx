/* SPDX-License-Identifier: Apache-2.0 */
/* Count formatting shared by dashboard tiles and usage history. */
(function () {
    const EAST_ASIAN_UNITS = {
        zh: ['万', '亿', '万亿'],
        'zh-TW': ['萬', '億', '兆'],
        ja: ['万', '億', '兆'],
        ko: ['만', '억', '조'],
    };
    const SCALE_STORAGE_KEY = 'omlx-number-scale';

    function countLocale() {
        return typeof document !== 'undefined' ? document.documentElement?.lang || 'en' : 'en';
    }

    function groupInteger(value, lang = countLocale()) {
        return Math.round(Number(value)).toLocaleString(lang);
    }

    // The exact grouped figure, with no compact unit.
    function exact(value, lang = countLocale()) {
        const number = Number(value);
        if (!Number.isFinite(number)) return '0';
        return groupInteger(number, lang);
    }

    // East Asian counts in ten-thousands and larger units: 万/亿/万亿 (zh),
    // 萬/億/兆 (zh-TW), 万/億/兆 (ja), and 만/억/조 (ko).
    // A compact unit always keeps one decimal, including 1.0万 and 1.0亿.
    // Returns null for every other language.
    function eastAsian(value, lang = countLocale()) {
        const units = EAST_ASIAN_UNITS[lang];
        if (!units) return null;
        const number = Number(value);
        if (!Number.isFinite(number)) return null;
        const scales = [1e4, 1e8, 1e12];
        const magnitude = Math.abs(number);
        let step = scales.length - 1;
        while (step >= 0 && magnitude < scales[step]) step -= 1;
        if (step < 0) return groupInteger(number, lang);
        let mantissa = (magnitude / scales[step]).toFixed(1);
        // 10,000万 is 1亿: the unit follows the printed mantissa.
        if (Number(mantissa) >= 10000 && step < scales.length - 1) {
            step += 1;
            mantissa = (magnitude / scales[step]).toFixed(1);
        }
        const [whole, fraction] = mantissa.split('.');
        const text = groupInteger(whole, lang) + '.' + (fraction || '0');
        return (number < 0 ? '-' : '') + text + units[step];
    }

    function storedScale() {
        try {
            return typeof localStorage !== 'undefined'
                && localStorage.getItem(SCALE_STORAGE_KEY) === 'exact' ? 'exact' : 'compact';
        } catch (error) {
            return 'compact';
        }
    }

    let scale = storedScale();

    function getScale() {
        return scale;
    }

    function setScale(next) {
        scale = next === 'exact' ? 'exact' : 'compact';
        try {
            if (typeof localStorage !== 'undefined') {
                localStorage.setItem(SCALE_STORAGE_KEY, scale);
            }
        } catch (error) {
            // Storage can be unavailable in private or embedded contexts.
        }
    }

    window.omlxCount = { eastAsian, exact, scale: getScale, setScale };
})();
