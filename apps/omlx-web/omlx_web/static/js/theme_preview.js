/* SPDX-License-Identifier: Apache-2.0 */
// Draws the theme preview card for Theme Settings (UI language) and for the
// shareable PNG (fixed English, so the file reads the same anywhere).
(function () {
    const KEYWORD = 'omlx-theme';
    const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace';
    const MODES = ['light', 'dark'];
    const PAD = 32;
    const CARD_ZOOM = 1.5;
    const GAP = 24;
    const HEAD_H = 24;
    const GROUP_H = 18;
    const ROW = 22;
    const NAME_W = 176;
    const ID_W = 131;

    const LABELS = {
        title: 'oMLX theme',
        light: 'Light',
        dark: 'Dark',
        token: 'Token',
        heading: 'Theme Colors',
        subtitle: 'How the dashboard will look',
        port: 'Port',
        link: 'Model card',
        danger: 'Unload',
        hover: 'Hover',
        save: 'Save Settings',
        tabStatus: 'Dashboard',
        tabSettings: 'Settings',
        status: { success: 'Success', warning: 'Warning', danger: 'Error', info: 'Info' },
        hint: 'Every color above is its exact hex value. This PNG also carries the theme as JSON in its "omlx-theme" text chunk: choose Load image in Theme Settings to apply it.',
    };
    const GROUP_LABELS = {
        surfaces: 'Surfaces',
        text: 'Text',
        borders: 'Borders',
        controls: 'Controls',
        status: 'Status',
        accents: 'Accents',
    };
    const TOKEN_LABELS = {
        'bg-primary': 'Page background',
        'bg-secondary': 'Secondary background',
        'bg-tertiary': 'Section header & tab track',
        'text-primary': 'Primary text',
        'text-secondary': 'Secondary text',
        'text-tertiary': 'Tertiary text',
        'text-muted': 'Muted text',
        'border-faint': 'Subtle border',
        'border-normal': 'Strong border',
        'btn-primary': 'Primary button',
        'btn-primary-hover': 'Primary button (hover)',
        'btn-primary-text': 'Primary button text',
        'switch-on': 'Switch on',
        'focus-ring-color': 'Focus ring',
        success: 'Success',
        warning: 'Warning',
        danger: 'Error',
        info: 'Info',
        'timeline-accent': 'Accent',
        'link-color': 'Link',
        'code-bg': 'Code background',
        'text-danger': 'Danger text',
    };

    const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        return c >>> 0;
    });

    function crc32(bytes) {
        let crc = 0xffffffff;
        for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
        return (crc ^ 0xffffffff) >>> 0;
    }

    // Inserts a tEXt chunk after IHDR (8-byte signature + 25-byte IHDR). tEXt is
    // Latin-1; the JSON is ASCII token ids and hex.
    function withTextChunk(png, keyword, text) {
        const data = new TextEncoder().encode(`${keyword}\0${text}`);
        const chunk = new Uint8Array(12 + data.length);
        const view = new DataView(chunk.buffer);
        view.setUint32(0, data.length);
        chunk.set(new TextEncoder().encode('tEXt'), 4);
        chunk.set(data, 8);
        view.setUint32(8 + data.length, crc32(chunk.subarray(4, 8 + data.length)));
        const out = new Uint8Array(png.length + chunk.length);
        out.set(png.subarray(0, 33));
        out.set(chunk, 33);
        out.set(png.subarray(33), 33 + chunk.length);
        return out;
    }

    // Returns null when the bytes are not a PNG or carry no such chunk.
    function readTextChunk(png, keyword) {
        const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
        if (png.length < 8 || signature.some((byte, i) => png[i] !== byte)) return null;
        const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
        const latin1 = new TextDecoder('latin1');
        for (let at = 8; at + 12 <= png.length; at += 12 + view.getUint32(at)) {
            const type = latin1.decode(png.subarray(at + 4, at + 8));
            if (type === 'IEND') break;
            if (type !== 'tEXt') continue;
            const data = png.subarray(at + 8, at + 8 + view.getUint32(at));
            const split = data.indexOf(0);
            if (split > 0 && latin1.decode(data.subarray(0, split)) === keyword) return latin1.decode(data.subarray(split + 1));
        }
        return null;
    }

    function roundRect(ctx, x, y, w, h, r) {
        const radius = Math.min(r, h / 2, w / 2);
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x, y, w, h, radius);
            return;
        }
        // Older Safari/iPadOS (< 16.4) has no roundRect.
        ctx.moveTo(x + radius, y);
        ctx.arcTo(x + w, y, x + w, y + h, radius);
        ctx.arcTo(x + w, y + h, x, y + h, radius);
        ctx.arcTo(x, y + h, x, y, radius);
        ctx.arcTo(x, y, x + w, y, radius);
        ctx.closePath();
    }

    function pill(ctx, x, y, w, h, r, fill, stroke) {
        roundRect(ctx, x, y, w, h, r);
        if (fill) { ctx.fillStyle = fill; ctx.fill(); }
        if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
    }

    function text(ctx, value, x, y, { size, weight = 400, color, font, align = 'left' }) {
        ctx.font = `${weight} ${size}px ${font}`;
        ctx.fillStyle = color;
        ctx.textAlign = align;
        ctx.textBaseline = 'middle';
        ctx.fillText(value, x, y);
        return ctx.measureText(value).width;
    }

    // Measures with the context's current font.
    function ellipsize(ctx, value, max) {
        if (ctx.measureText(value).width <= max) return value;
        let cut = value;
        while (cut && ctx.measureText(`${cut}…`).width > max) cut = cut.slice(0, -1);
        return `${cut}…`;
    }

    // Breaks at spaces; a word wider than the line breaks between characters.
    function wrap(ctx, value, max) {
        const lines = [];
        let line = '';
        for (const ch of value) {
            if (ctx.measureText(line + ch).width <= max) { line += ch; continue; }
            const space = line.lastIndexOf(' ');
            if (space > 0 && ch !== ' ') {
                lines.push(line.slice(0, space));
                line = line.slice(space + 1) + ch;
            } else {
                lines.push(line);
                line = ch === ' ' ? '' : ch;
            }
        }
        if (line) lines.push(line);
        return lines;
    }

    // Card units; the page preview and the image scale the same layout.
    const CARD_W = 288;
    const CARD_H = 236;
    const WIDTH = PAD * 2 + CARD_W * CARD_ZOOM * 2 + GAP;

    // o.regions collects [x, y, w, h, token] per part, topmost last. Keep each
    // hit() box in step with what it draws: click-to-edit reads only these.
    function drawCard(ctx, c, o) {
        const { sans, labels } = o;
        const t = (value, x, y, style) => text(ctx, value, x, y, { font: sans, ...style });
        const hit = (x, y, w, h, token) => o.regions?.push([x, y, w, h, token]);

        roundRect(ctx, 0, 0, CARD_W, CARD_H, 16);
        ctx.save();
        ctx.clip();
        ctx.fillStyle = c['bg-primary'];
        ctx.fillRect(0, 0, CARD_W, CARD_H);
        hit(0, 0, CARD_W, CARD_H, 'bg-primary');
        ctx.fillStyle = c['timeline-accent'];
        ctx.fillRect(0, 0, CARD_W, 3);
        hit(0, 0, CARD_W, 4, 'timeline-accent');

        // Navbar
        const navMid = 3 + 18;
        pill(ctx, 11, navMid - 7, 14, 14, 4.5, c['text-primary']);
        const brand = t('oMLX', 31, navMid, { size: 11, weight: 750, color: c['text-primary'] });
        hit(11, navMid - 8, 20 + brand, 16, 'text-primary');
        const version = t(`v${labels.version}`, 31 + brand + 4, navMid + 0.5, { size: 8, weight: 600, color: c['text-muted'] });
        hit(31 + brand + 3, navMid - 6, version + 2, 12, 'text-muted');
        ctx.font = `650 9px ${sans}`;
        const tabs = [labels.tabStatus, labels.tabSettings].map(label => ctx.measureText(label).width + 12);
        const trackW = tabs[0] + tabs[1] + 6;
        const trackX = CARD_W - 11 - trackW;
        pill(ctx, trackX, navMid - 9, trackW, 18, 7, c['bg-tertiary']);
        hit(trackX, navMid - 9, trackW, 18, 'bg-tertiary');
        pill(ctx, trackX + 2, navMid - 7, tabs[0], 14, 4.5, c['bg-primary']);
        t(labels.tabStatus, trackX + 2 + tabs[0] / 2, navMid, { size: 9, weight: 650, color: c['text-primary'], align: 'center' });
        hit(trackX + 2, navMid - 7, tabs[0], 14, 'text-primary');
        t(labels.tabSettings, trackX + 4 + tabs[0] + tabs[1] / 2, navMid, { size: 9, color: c['text-secondary'], align: 'center' });
        hit(trackX + 4 + tabs[0], navMid - 7, tabs[1], 14, 'text-secondary');
        ctx.fillStyle = c['border-faint'];
        ctx.fillRect(0, 39, CARD_W, 1);

        // Hero
        const left = 11;
        const heroMid = 40 + 11 + 14;
        const aa = t('Aa', left, heroMid + 1, { size: 26, weight: 750, color: c['text-primary'] });
        ctx.font = `650 11px ${sans}`;
        const heading = t(ellipsize(ctx, labels.heading, CARD_W - aa - 40), left + aa + 10, heroMid - 6, { size: 11, weight: 650, color: c['text-primary'] });
        hit(left, heroMid - 14, aa + 10 + heading, 15, 'text-primary');
        ctx.font = `400 9px ${sans}`;
        const subtitle = t(ellipsize(ctx, labels.subtitle, CARD_W - aa - 40), left + aa + 10, heroMid + 7, { size: 9, color: c['text-tertiary'] });
        hit(left + aa + 10, heroMid + 1, subtitle, 12, 'text-tertiary');

        // Settings card: three rows
        const cardY = heroMid + 14 + 10;
        const rowH = 26;
        const innerW = CARD_W - 22;
        ctx.save();
        roundRect(ctx, left, cardY, innerW, rowH * 3, 10);
        ctx.clip();
        ctx.fillStyle = c['bg-primary'];
        ctx.fillRect(left, cardY, innerW, rowH * 3);
        ctx.fillStyle = c['bg-secondary'];
        ctx.fillRect(left, cardY + rowH, innerW, rowH);
        ctx.fillStyle = c['border-faint'];
        ctx.fillRect(left, cardY + rowH, innerW, 1);
        ctx.fillRect(left, cardY + rowH * 2, innerW, 1);
        ctx.restore();
        pill(ctx, left + 0.5, cardY + 0.5, innerW - 1, rowH * 3 - 1, 10, null, c['border-faint']);
        hit(left - 2, cardY - 2, innerW + 4, rowH * 3 + 4, 'border-faint');
        hit(left + 2, cardY + rowH + 1, innerW - 4, rowH - 1, 'bg-secondary');
        const right = left + innerW - 9;
        let mid = cardY + rowH / 2;
        const port = t(labels.port, left + 9, mid, { size: 9.5, color: c['text-secondary'] });
        hit(left + 9, mid - 7, port, 14, 'text-secondary');
        pill(ctx, right - 58, mid - 8, 58, 16, 4.5, c['bg-primary'], c['border-normal']);
        hit(right - 58, mid - 8, 58, 16, 'border-normal');
        t('8000', right - 6, mid, { size: 9.5, color: c['text-primary'], align: 'right' });
        mid += rowH;
        ctx.font = `400 9.5px ${MONO}`;
        const codeW = ctx.measureText('/v1/models').width + 10;
        pill(ctx, left + 9, mid - 8, codeW, 16, 4.5, c['code-bg']);
        hit(left + 9, mid - 8, codeW, 16, 'code-bg');
        text(ctx, '/v1/models', left + 14, mid, { size: 9.5, color: c['text-primary'], font: MONO });
        const link = t(labels.link, right, mid, { size: 9.5, weight: 600, color: c['link-color'], align: 'right' });
        hit(right - link, mid - 7, link, 14, 'link-color');
        mid += rowH;
        ctx.font = `600 9px ${sans}`;
        const dangerW = ctx.measureText(labels.danger).width + 12;
        // The stock chip is red at 10% (light) or 20% (dark); it follows the danger text.
        pill(ctx, right - dangerW, mid - 8, dangerW, 16, 4.5, c['text-danger']);
        ctx.fillStyle = c['bg-primary'];
        ctx.globalAlpha = o.mode === 'dark' ? 0.8 : 0.9;
        ctx.fill();
        ctx.globalAlpha = 1;
        t(labels.danger, right - dangerW / 2, mid, { size: 9, weight: 600, color: c['text-danger'], align: 'center' });
        hit(right - dangerW, mid - 8, dangerW, 16, 'text-danger');
        const meterW = innerW - 18 - dangerW - 8;
        pill(ctx, left + 9, mid - 2.5, meterW, 5, 2.5, c['bg-tertiary']);
        pill(ctx, left + 9, mid - 2.5, meterW * 0.62, 5, 2.5, c['timeline-accent']);
        hit(left + 9, mid - 5, meterW, 10, 'timeline-accent');

        // Status badges
        let x = left;
        const badgeMid = cardY + rowH * 3 + 10 + 8;
        for (const name of ['success', 'warning', 'danger', 'info']) {
            ctx.font = `650 8.5px ${sans}`;
            const w = ctx.measureText(labels.status[name]).width + 14;
            pill(ctx, x + 0.5, badgeMid - 7.5, w, 15, 7.5, c[`${name}-soft`], c[`${name}-line`]);
            t(labels.status[name], x + 0.5 + w / 2, badgeMid, { size: 8.5, weight: 650, color: c[`${name}-text`], align: 'center' });
            hit(x, badgeMid - 8, w + 1, 16, name);
            x += w + 4;
        }

        // Switch and buttons
        const actMid = badgeMid + 8 + 10 + 9;
        pill(ctx, left, actMid - 6, 22, 12, 6, c['switch-on']);
        pill(ctx, left + 12, actMid - 4, 8, 8, 4, '#ffffff');
        hit(left, actMid - 6, 22, 12, 'switch-on');
        ctx.font = `650 9px ${sans}`;
        // Long translations are cut so the two buttons never reach the switch.
        const save = ellipsize(ctx, labels.save, 100);
        const hover = ellipsize(ctx, labels.hover, 70);
        const saveW = ctx.measureText(save).width + 18;
        const hoverW = ctx.measureText(hover).width + 18;
        const saveX = CARD_W - 11 - saveW;
        pill(ctx, saveX - 3, actMid - 12, saveW + 6, 24, 12, c['focus-ring-color']);
        hit(saveX - 3, actMid - 12, saveW + 6, 24, 'focus-ring-color');
        pill(ctx, saveX - 1.5, actMid - 10.5, saveW + 3, 21, 10.5, c['bg-primary']);
        pill(ctx, saveX, actMid - 9, saveW, 18, 9, c['btn-primary']);
        hit(saveX, actMid - 9, saveW, 18, 'btn-primary');
        const saveText = t(save, saveX + saveW / 2, actMid, { size: 9, weight: 650, color: c['btn-primary-text'], align: 'center' });
        hit(saveX + (saveW - saveText) / 2, actMid - 5, saveText, 10, 'btn-primary-text');
        const hoverX = saveX - 8 - hoverW;
        pill(ctx, hoverX, actMid - 9, hoverW, 18, 9, c['btn-primary-hover']);
        hit(hoverX, actMid - 9, hoverW, 18, 'btn-primary-hover');
        t(hover, hoverX + hoverW / 2, actMid, { size: 9, weight: 650, color: c['btn-primary-text'], align: 'center' });
        ctx.restore();
        pill(ctx, 0.5, 0.5, CARD_W - 1, CARD_H - 1, 16, null, c['border-faint']);
    }

    const tableHeight = groups => HEAD_H
        + groups.reduce((height, group) => height + GROUP_H + group.tokens.length * ROW, 0);

    function drawTable(ctx, theme, mode, groups, x, top, o) {
        const c = theme[mode];
        const width = CARD_W * CARD_ZOOM;
        const nameW = NAME_W;
        const idW = ID_W;
        const valueX = x + nameW + idW;
        const valueW = width - nameW - idW;
        const height = tableHeight(groups);
        ctx.save();
        roundRect(ctx, x, top, width, height, 12);
        ctx.clip();
        ctx.fillStyle = c['bg-primary'];
        ctx.fillRect(x, top, width, height);
        ctx.fillStyle = c['bg-tertiary'];
        ctx.fillRect(x, top, width, HEAD_H);
        text(ctx, o.labels.token, x + 14, top + HEAD_H / 2,
            { size: 10, weight: 650, color: c['text-secondary'], font: o.sans });
        text(ctx, o.labels[mode], valueX + valueW / 2, top + HEAD_H / 2,
            { size: 11, weight: 650, color: c['text-secondary'], font: o.sans, align: 'center' });
        let y = top + HEAD_H;
        for (const group of groups) {
            ctx.fillStyle = c['bg-tertiary'];
            ctx.fillRect(x, y, width, GROUP_H);
            ctx.fillStyle = c['border-faint'];
            ctx.fillRect(x, y, width, 1);
            text(ctx, (GROUP_LABELS[group.id] || group.id).toUpperCase(), x + 14, y + GROUP_H / 2,
                { size: 10, weight: 650, color: c['text-secondary'], font: o.sans });
            y += GROUP_H;
            for (const token of group.tokens) {
                ctx.fillStyle = c['border-faint'];
                ctx.fillRect(x, y, width, 1);
                const mid = y + ROW / 2;
                ctx.font = `600 11px ${o.sans}`;
                const label = ellipsize(ctx, TOKEN_LABELS[token] || token, nameW - 28);
                text(ctx, label, x + 14, mid, { size: 11, weight: 600, color: c['text-primary'], font: o.sans });
                text(ctx, `--${token}`, x + nameW + 10, mid, { size: 9, color: c['text-tertiary'], font: MONO });
                pill(ctx, valueX + 14, mid - 7.5, 15, 15, 4, c[token], c['border-normal']);
                text(ctx, c[token], valueX + 38, mid, { size: 11, weight: 500, color: c['text-primary'], font: MONO });
                y += ROW;
            }
        }
        ctx.fillStyle = c['border-faint'];
        ctx.fillRect(valueX, top, 1, height);
        ctx.fillRect(x + nameW, top + HEAD_H, 1, height - HEAD_H);
        ctx.restore();
        pill(ctx, x + 0.5, top + 0.5, width - 1, height - 1, 12, null, c['border-faint']);
    }

    function render(options) {
        const { theme, groups } = options;
        const scale = options.scale || 2;
        const light = theme.light;
        const labels = { ...LABELS, version: options.version };
        // Fixed Latin stack: the image must not change with the dashboard's UI language.
        const o = { labels, sans: 'Inter, system-ui, sans-serif' };
        const measure = document.createElement('canvas').getContext('2d');
        measure.font = `400 11px ${o.sans}`;
        const hint = wrap(measure, labels.hint, WIDTH - PAD * 2);
        const tableY = PAD + 40 + 24 + CARD_H * CARD_ZOOM + 16;
        const height = tableY + tableHeight(groups) + 14 + hint.length * 16 + PAD;

        const canvas = document.createElement('canvas');
        canvas.width = WIDTH * scale;
        canvas.height = Math.ceil(height * scale);
        const ctx = canvas.getContext('2d');
        ctx.scale(scale, scale);
        ctx.fillStyle = light['bg-secondary'];
        ctx.fillRect(0, 0, WIDTH, height);

        text(ctx, labels.title, PAD, PAD + 12, { size: 18, weight: 700, color: light['text-primary'], font: o.sans });
        text(ctx, `oMLX v${labels.version}`, WIDTH - PAD, PAD + 13, { size: 12, color: light['text-secondary'], font: o.sans, align: 'right' });

        MODES.forEach((mode, i) => {
            const x = PAD + i * (CARD_W * CARD_ZOOM + GAP);
            text(ctx, labels[mode], x, PAD + 52, { size: 12, weight: 650, color: light['text-secondary'], font: o.sans });
            ctx.save();
            ctx.translate(x, PAD + 64);
            ctx.scale(CARD_ZOOM, CARD_ZOOM);
            drawCard(ctx, theme[mode], { ...o, mode });
            ctx.restore();
        });

        MODES.forEach((mode, i) => {
            const x = PAD + i * (CARD_W * CARD_ZOOM + GAP);
            drawTable(ctx, theme, mode, groups, x, tableY, o);
        });

        let y = tableY + tableHeight(groups) + 14;
        for (const line of hint) {
            text(ctx, line, PAD, y + 8, { size: 11, color: light['text-secondary'], font: o.sans });
            y += 16;
        }
        return canvas;
    }

    async function png(options) {
        const canvas = render(options);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        const bytes = new Uint8Array(await blob.arrayBuffer());
        // Only the editable tokens, in the shape Theme JSON loads.
        const tokens = options.groups.flatMap(group => group.tokens);
        const json = JSON.stringify(Object.fromEntries(MODES.map(mode =>
            [mode, Object.fromEntries(tokens.map(token => [token, options.theme[mode][token]]))])), null, 2);
        return new Blob([withTextChunk(bytes, KEYWORD, json)], { type: 'image/png' });
    }

    // Returns the click regions in card units.
    function drawPreview(canvas, colors, { mode, labels, sans, hover }) {
        canvas.style.aspectRatio = `${CARD_W} / ${CARD_H}`;
        const scale = canvas.clientWidth / CARD_W;
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.round(CARD_W * scale * ratio);
        canvas.height = Math.round(CARD_H * scale * ratio);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(scale * ratio, 0, 0, scale * ratio, 0, 0);
        const regions = [];
        drawCard(ctx, colors, { mode, labels, sans, regions });
        if (hover) {
            ctx.setLineDash([2, 2]);
            ctx.strokeStyle = colors['timeline-accent'];
            ctx.lineWidth = 1;
            ctx.strokeRect(hover[0] - 1, hover[1] - 1, hover[2] + 2, hover[3] + 2);
        }
        return regions;
    }

    const regionAt = (regions, x, y) => regions.findLast(([rx, ry, w, h]) => x >= rx && x <= rx + w && y >= ry && y <= ry + h);

    const api = {
        KEYWORD, LABELS, GROUP_LABELS, TOKEN_LABELS, CARD_W,
        png, drawCard, drawPreview, regionAt, withTextChunk, readTextChunk, crc32,
    };
    if (typeof window !== 'undefined') window.OmlxThemePreview = api;
    if (typeof module !== 'undefined') module.exports = api;
})();
