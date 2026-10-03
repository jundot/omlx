// Dashboard block layout contract shared by the template, dashboard.js and
// the server validator in admin/routes.py. Keep the constants in sync.
(function (root) {
    'use strict';

    const BLOCK_IDS = [
        'serving_stats',
        'usage_history',
        'active_models',
        'cache_observability',
        'api_endpoints',
        'claude_code',
        'applications',
        'engine_versions',
    ];
    const COLUMNS = 24;
    const MIN_W = 6;
    // The measures are tokens, not classes: tokens.json `layout.widths` generates
    // `--measure-<id>`, and every console page reads the one dashboard.js points
    // `--container-wide` at. So the dashboard's width control is the console's
    // width control, and no page carries a second, private limit.
    const WIDTH_MEASURES = {
        default: 'var(--measure-default)',
        wide: 'var(--measure-wide)',
        wider: 'var(--measure-wider)',
        full: 'var(--measure-full)',
    };
    const WIDTH_IDS = Object.keys(WIDTH_MEASURES);

    function defaultLayout() {
        return {
            version: 1,
            width: 'default',
            blocks: BLOCK_IDS.map((id, index) => ({ id, x: 0, y: index, w: COLUMNS })),
        };
    }

    function toInt(value, fallback) {
        const n = Number(value);
        return Number.isFinite(n) ? Math.trunc(n) : fallback;
    }

    function normalizeBlock(raw, seen) {
        if (!raw || typeof raw !== 'object') return null;
        const id = raw.id;
        if (!BLOCK_IDS.includes(id) || seen.has(id)) return null;
        seen.add(id);
        const w = Math.min(COLUMNS, Math.max(MIN_W, toInt(raw.w, COLUMNS)));
        const x = Math.min(COLUMNS - w, Math.max(0, toInt(raw.x, 0)));
        const y = Math.max(0, toInt(raw.y, 0));
        return { id, x, y, w };
    }

    // Accepts anything the server or a hand-edited settings.json may hold and
    // returns a layout the grid can load. Unknown blocks are dropped, so a
    // layout may legitimately contain fewer than BLOCK_IDS.length blocks.
    function normalizeLayout(raw) {
        if (!raw || typeof raw !== 'object' || !Array.isArray(raw.blocks)) {
            return defaultLayout();
        }
        const seen = new Set();
        const blocks = raw.blocks.map(b => normalizeBlock(b, seen)).filter(Boolean);
        const width = WIDTH_IDS.includes(raw.width) ? raw.width : 'default';
        return { version: 1, width, blocks };
    }

    function widthMeasure(width) {
        return WIDTH_MEASURES[width] || WIDTH_MEASURES.default;
    }

    root.DashboardLayout = {
        BLOCK_IDS,
        COLUMNS,
        MIN_W,
        WIDTH_MEASURES,
        WIDTH_IDS,
        defaultLayout,
        normalizeLayout,
        widthMeasure,
    };
})(typeof window !== 'undefined' ? window : globalThis);
