/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Saved overrides of the theme.css color tokens (ui.custom_theme), painted from
 * <head> before first paint. This file owns the token list and the contrast
 * floor; the server only checks that each value is #rrggbb.
 */
(function () {
    const STYLE_ID = 'omlx-custom-theme';
    const MODES = ['light', 'dark'];
    const GROUPS = [
        { id: 'surfaces', tokens: ['bg-primary', 'bg-secondary', 'bg-tertiary'] },
        { id: 'text', tokens: ['text-primary', 'text-secondary', 'text-tertiary', 'text-muted'] },
        { id: 'borders', tokens: ['border-faint', 'border-normal'] },
        { id: 'controls', tokens: ['btn-primary', 'btn-primary-hover', 'btn-primary-text', 'switch-on', 'focus-ring-color'] },
        { id: 'status', tokens: ['success', 'warning', 'danger', 'info'] },
        { id: 'accents', tokens: ['timeline-accent', 'link-color', 'code-bg', 'text-danger'] },
    ];
    // Neutral utility channels (tailwind.config.js) drawn from each token.
    const CHANNELS = {
        'bg-primary': ['surface-scale-white'],
        'bg-secondary': ['surface-scale-50'],
        'bg-tertiary': ['surface-scale-100'],
        'text-primary': ['text-scale-900', 'text-scale-800'],
        'text-secondary': ['text-scale-700', 'text-scale-600'],
        // text-neutral-400 draws the 500 step for AA, so it follows tertiary, not muted.
        'text-tertiary': ['text-scale-500', 'text-scale-400'],
        'border-faint': ['border-scale-100', 'border-scale-200'],
        'border-normal': ['border-scale-300'],
    };
    const SURFACES = ['bg-primary', 'bg-secondary', 'bg-tertiary'];
    const STATUS = ['success', 'warning', 'danger', 'info'];
    // How much of a status color its badge fill and line take, and how far its
    // text moves toward black (light) or white (dark). declarations() paints
    // these and badgeColors() computes them, so both read this one table.
    const BADGE_TINT = {
        light: { soft: 0.08, line: 0.3, text: 0.35, toward: 'black' },
        dark: { soft: 0.15, line: 0.3, text: 0.35, toward: 'white' },
    };
    // [foreground, backgrounds, floor]; every pair must reach its floor before
    // Apply. 4.5:1 is WCAG 1.4.3 AA for body text, Apple's minimum up to 17 pt;
    // 3:1 is WCAG 1.4.11 for controls, focus and graphics, and AA for bold or
    // large text. Muted text and borders have no floor: the stock theme draws
    // placeholders and dividers below 3:1, as WCAG allows.
    const CONTRAST_RULES = [
        ['text-primary', SURFACES, 4.5],
        ['text-secondary', SURFACES, 4.5],
        ['link-color', SURFACES, 4.5],
        ['btn-primary-text', ['btn-primary', 'btn-primary-hover'], 4.5],
        ...STATUS.map(status => [`${status}-text`, [`${status}-soft`], 4.5]),
        ['text-tertiary', SURFACES, 3],
        ['text-danger', SURFACES, 3],
        ['focus-ring-color', SURFACES, 3],
        ['switch-on', SURFACES, 3],
        ['timeline-accent', SURFACES, 3],
    ];
    // Not edited directly; resolved too so the previews and contrast checks can
    // use the stock badges.
    const DERIVED = STATUS.flatMap(status => ['soft', 'line', 'text'].map(part => `${status}-${part}`));
    const TOKENS = GROUPS.flatMap(group => group.tokens);
    // Tokens are opaque: a translucent text color would make the contrast
    // floor depend on whatever is drawn behind it.
    const HEX = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i;

    function normalizeHex(value) {
        const match = HEX.exec(String(value ?? '').trim());
        if (!match) return null;
        const digits = match[1].toLowerCase();
        return '#' + (digits.length === 3 ? digits.replace(/./g, c => c + c) : digits);
    }

    function rgb(hex) {
        const digits = normalizeHex(hex)?.slice(1);
        return digits ? [0, 2, 4].map(i => parseInt(digits.slice(i, i + 2), 16)) : null;
    }

    function toHex(r, g, b) {
        const byte = n => Math.round(Math.min(255, Math.max(0, n))).toString(16).padStart(2, '0');
        return '#' + byte(r) + byte(g) + byte(b);
    }

    // WCAG 2 ratio; null when either color is not a valid hex.
    function contrast(fg, bg) {
        const pair = [rgb(fg), rgb(bg)];
        if (!pair[0] || !pair[1]) return null;
        const luminance = channels => {
            const [r, g, b] = channels.map(c => {
                const s = c / 255;
                return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
            });
            return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        };
        const [hi, lo] = pair.map(luminance).sort((a, b) => b - a);
        return (hi + 0.05) / (lo + 0.05);
    }

    // `colors` maps tokens, including the derived badge colors, to opaque hex.
    function contrastChecks(colors) {
        return CONTRAST_RULES.flatMap(([foreground, backgrounds, floor]) => backgrounds.map(background => {
            const ratio = contrast(colors[foreground], colors[background]);
            return { foreground, background, ratio, floor, ok: ratio !== null && ratio >= floor };
        }));
    }

    // A CSS rgb()/rgba() or hex color drawn over an opaque hex, as opaque hex.
    function flatten(color, under) {
        const hex = normalizeHex(color);
        if (hex) return hex;
        const [r, g, b, a = 1] = (String(color).match(/[\d.]+/g) || []).map(Number);
        const base = rgb(under) || [0, 0, 0];
        return toHex(...[r, g, b].map((c, i) => c * a + base[i] * (1 - a)));
    }

    // The badge fill, line and text a custom status color paints (see
    // declarations()), flattened onto the page background.
    function badgeColors(mode, hex, page) {
        const tint = BADGE_TINT[mode];
        const mix = (weight, other) => toHex(...rgb(hex).map((c, i) => c * weight + other[i] * (1 - weight)));
        return {
            soft: mix(tint.soft, rgb(page)),
            line: mix(tint.line, rgb(page)),
            text: mix(1 - tint.text, tint.toward === 'white' ? [255, 255, 255] : [0, 0, 0]),
        };
    }

    function sanitize(theme) {
        const clean = { light: {}, dark: {} };
        for (const mode of MODES) {
            const source = theme && typeof theme === 'object' ? theme[mode] : null;
            if (!source || typeof source !== 'object') continue;
            for (const token of TOKENS) {
                const hex = normalizeHex(source[token]);
                if (hex) clean[mode][token] = hex;
            }
        }
        return clean;
    }

    function isEmpty(theme) {
        return MODES.every(mode => !Object.keys(theme[mode]).length);
    }

    function declarations(mode, token, hex) {
        const lines = [`--${token}: ${hex};`];
        const [r, g, b] = rgb(hex);
        for (const channel of CHANNELS[token] || []) lines.push(`--${channel}: ${r} ${g} ${b};`);
        if (STATUS.includes(token)) {
            // The stock soft fills and lines are tints of the solid colour.
            const tint = BADGE_TINT[mode];
            const percent = value => `${Math.round(value * 100)}%`;
            const base = mode === 'dark' ? 'transparent' : 'var(--bg-primary)';
            lines.push(`--${token}-soft: color-mix(in srgb, ${hex} ${percent(tint.soft)}, ${base});`);
            lines.push(`--${token}-line: color-mix(in srgb, ${hex} ${percent(tint.line)}, ${base});`);
            lines.push(`--${token}-text: color-mix(in srgb, ${hex}, ${tint.toward} ${percent(tint.text)});`);
            // One channel per palette step keeps status.css's shade ladder.
            const pct = (light, dark) => (mode === 'dark' ? dark : light);
            const tintMix = (value, surface) => `color-mix(in srgb, ${hex} ${value}, ${surface})`;
            const steps = {
                'bg-50': tintMix(pct('8%', '15%'), base),
                'bg-100': tintMix(pct('12%', '20%'), base),
                'bg-200': tintMix(pct('25%', '30%'), base),
                'bg-300': tintMix(pct('40%', '45%'), base),
                'bg-400': `color-mix(in srgb, ${hex}, #fff 18%)`,
                'bg-500': hex,
                'bg-600': `color-mix(in srgb, ${hex}, #000 12%)`,
                'bg-700': `color-mix(in srgb, ${hex}, #000 25%)`,
                'bg-800': `color-mix(in srgb, ${hex}, #000 40%)`,
                'bg-900': `color-mix(in srgb, ${hex}, #000 50%)`,
                'fg-400': `color-mix(in srgb, var(--${token}-text), ${hex} 70%)`,
                'fg-500': `color-mix(in srgb, var(--${token}-text), ${hex} 45%)`,
                'fg-600': `color-mix(in srgb, var(--${token}-text), ${hex} 20%)`,
                'fg-700': `var(--${token}-text)`,
                'fg-800': `color-mix(in srgb, var(--${token}-text), #000 18%)`,
                'fg-900': `color-mix(in srgb, var(--${token}-text), #000 30%)`,
                'bd-100': tintMix(pct('20%', '25%'), base),
                'bd-200': tintMix(pct('32%', '35%'), base),
                'bd-300': tintMix(pct('45%', '45%'), base),
            };
            for (const [step, value] of Object.entries(steps)) {
                lines.push(`--${token}-${step}: ${value};`);
            }
        }
        return lines;
    }

    // The memory pressure meter's in-between steps, once a status role is custom.
    const METER = [
        '--meter-high: color-mix(in srgb, var(--danger) 55%, var(--warning));',
        '--meter-mid: color-mix(in srgb, var(--warning) 55%, var(--success));',
    ];

    function css(theme) {
        return MODES.filter(mode => Object.keys(theme[mode]).length)
            .map(mode => {
                const tokens = Object.keys(theme[mode]);
                const lines = tokens.flatMap(token => declarations(mode, token, theme[mode][token]));
                if (tokens.some(token => STATUS.includes(token))) lines.push(...METER);
                return `:root[data-theme="${mode}"] {\n${lines.map(line => `    ${line}\n`).join('')}}\n`;
            })
            .join('');
    }

    function apply(theme) {
        const clean = sanitize(theme);
        let style = document.getElementById(STYLE_ID);
        if (isEmpty(clean)) {
            style?.remove();
            return clean;
        }
        if (!style) {
            style = document.createElement('style');
            style.id = STYLE_ID;
            document.head.appendChild(style);
        }
        style.textContent = css(clean);
        return clean;
    }

    // The stock token colors, read from theme.css itself so they never drift.
    let defaultsCache = null;
    function defaults() {
        if (defaultsCache) return defaultsCache;
        const selectors = { light: ':root', dark: '[data-theme="dark"]' };
        const declared = { light: {}, dark: {} };
        for (const sheet of Array.from(document.styleSheets)) {
            if (!/\/theme\.css(\?|$)/.test(sheet.href || '')) continue;
            let rules;
            try {
                rules = sheet.cssRules;
            } catch (error) {
                continue;
            }
            for (const rule of Array.from(rules)) {
                for (const mode of MODES) {
                    if (rule.selectorText !== selectors[mode]) continue;
                    for (const token of [...TOKENS, ...DERIVED]) {
                        const value = rule.style.getPropertyValue(`--${token}`).trim();
                        if (value) declared[mode][token] = value;
                    }
                }
            }
        }
        // Resolve `rgb(var(--palette-x) / a)` to a concrete color on a probe.
        const probe = document.createElement('span');
        probe.style.display = 'none';
        document.documentElement.appendChild(probe);
        const resolved = { light: {}, dark: {} };
        let complete = true;
        for (const mode of MODES) {
            for (const token of [...TOKENS, ...DERIVED]) {
                probe.style.color = '';
                if (declared[mode][token]) probe.style.setProperty('color', declared[mode][token]);
                const parts = (getComputedStyle(probe).color.match(/[\d.]+/g) || []).map(Number);
                if (!declared[mode][token] || parts.length < 3) {
                    complete = false;
                    resolved[mode][token] = '#000000';
                    continue;
                }
                // Editable tokens are opaque hex; the derived tints keep their
                // computed rgb()/rgba() text, which only the previews read.
                resolved[mode][token] = TOKENS.includes(token)
                    ? toHex(parts[0], parts[1], parts[2])
                    : getComputedStyle(probe).color;
            }
        }
        probe.remove();
        if (complete) defaultsCache = resolved;
        return resolved;
    }

    const api = {
        GROUPS, TOKENS, DERIVED, MODES, STATUS, CONTRAST_RULES,
        normalizeHex, contrast, contrastChecks, flatten, badgeColors, sanitize, isEmpty, apply, defaults, css, declarations,
    };
    if (typeof window !== 'undefined') window.OmlxCustomTheme = api;
    if (typeof module !== 'undefined') module.exports = api;

    if (typeof document !== 'undefined') apply(window._customTheme);
})();
