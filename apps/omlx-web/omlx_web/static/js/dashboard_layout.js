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
    const WIDTH_CLASSES = {
        default: 'max-w-7xl',
        wide: 'max-w-[90rem]',
        wider: 'max-w-[100rem]',
        full: 'max-w-none',
    };
    const WIDTH_IDS = Object.keys(WIDTH_CLASSES);
    // Serving Stats tiles, in picker order. The saved list is the left-to-right order.
    const SERVING_TILE_IDS = [
        'requests',
        'prefill_tokens',
        'cached_tokens',
        'cache_efficiency',
        'generated_tokens',
    ];
    const SERVING_TILE_MAX = 4;
    const DEFAULT_SERVING_TILES = SERVING_TILE_IDS.slice(0, SERVING_TILE_MAX);

    function defaultLayout() {
        return {
            version: 1,
            width: 'default',
            blocks: BLOCK_IDS.map((id, index) => ({ id, x: 0, y: index, w: COLUMNS })),
            serving_stats_tiles: [...DEFAULT_SERVING_TILES],
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

    function normalizeServingTiles(raw) {
        if (!Array.isArray(raw)) return [...DEFAULT_SERVING_TILES];
        const tiles = [...new Set(raw.filter(id => SERVING_TILE_IDS.includes(id)))]
            .slice(0, SERVING_TILE_MAX);
        return tiles.length ? tiles : [...DEFAULT_SERVING_TILES];
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
        const serving_stats_tiles = normalizeServingTiles(raw.serving_stats_tiles);
        return { version: 1, width, blocks, serving_stats_tiles };
    }

    function widthClass(width) {
        return WIDTH_CLASSES[width] || WIDTH_CLASSES.default;
    }

    root.DashboardLayout = {
        BLOCK_IDS,
        COLUMNS,
        MIN_W,
        WIDTH_CLASSES,
        WIDTH_IDS,
        SERVING_TILE_IDS,
        SERVING_TILE_MAX,
        DEFAULT_SERVING_TILES,
        defaultLayout,
        normalizeLayout,
        widthClass,
    };
})(typeof window !== 'undefined' ? window : globalThis);
