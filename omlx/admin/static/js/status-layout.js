/* Status tab: client-side panel layout (flat grid, local patch v2).
 *
 * The Status tab is one CSS grid with fixed 8px auto-rows. Each card is
 * explicitly placed by this script: grid-column from its width mode,
 * grid-row from a masonry cursor algorithm over N configurable columns.
 * Card order lives in the DOM (drag to reorder); placements are derived,
 * never persisted, so any column count works.
 *
 * Persisted in localStorage (client-side only, no server settings):
 *   omlx-status-layout-v4   { order: [cardId...], widths: {cardId: mode} }
 *   omlx-status-settings-v4 { columns: 1-4, maxWidth: px }
 * Layout v3 (two-column left/right/below zones) is dropped on load: the
 * arrangement resets to the template default once.
 *
 * Dragging uses Pointer Events started from the grip handle; window-level
 * capture listeners survive DOM rearrangement (Safari drops pointer capture
 * when the dragged node is re-inserted).
 */
(() => {
    const LAYOUT_KEY = 'omlx-status-layout-v4';
    const SETTINGS_KEY = 'omlx-status-settings-v4';
    const DRAG_THRESHOLD = 4;   // px before a press becomes a drag
    const ROW = 8;              // px per implicit grid row
    const GAP = 32;             // px vertical gap (card margin-bottom, 2rem)
    const DESKTOP_MIN = 1024;   // px viewport width enabling multi-column
    const WIDTH_MODES = ['normal', 'wide', 'full'];
    let wired = false;
    let layout = null;          // {order:[], widths:{}} current state
    let defaultOrder = [];      // template DOM order
    let defaultWidths = {};     // from data-default-span
    let settings = loadSettings();
    let activeDrag = null;

    function loadSettings() {
        try {
            const raw = localStorage.getItem(SETTINGS_KEY);
            const s = raw ? JSON.parse(raw) : null;
            if (s && Number.isInteger(s.columns)) return s;
        } catch (e) { /* fall through to defaults */ }
        return { columns: 2, maxWidth: 1740 };
    }

    function saveSettings() {
        try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); }
        catch (e) { /* private mode: settings just won't persist */ }
    }

    function loadLayout() {
        try {
            // v3 encoded a fixed left/right/below zone model that has no
            // faithful flat-grid equivalent: drop it, start from defaults.
            localStorage.removeItem('omlx-status-layout-v3');
            const raw = localStorage.getItem(LAYOUT_KEY);
            const parsed = raw ? JSON.parse(raw) : null;
            if (parsed && Array.isArray(parsed.order)) return parsed;
        } catch (e) { /* corrupt save: defaults */ }
        return null;
    }

    function saveLayout() {
        try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout)); }
        catch (e) { /* private mode: layout just won't persist */ }
    }

    const root = () => document.getElementById('status-layout');

    const cards = () =>
        [...root().querySelectorAll(':scope > .omlx-card')];

    const effectiveColumns = () =>
        window.innerWidth < DESKTOP_MIN
            ? 1
            : Math.max(1, Math.min(4, settings.columns || 2));

    const spanFor = (card) => {
        const cols = effectiveColumns();
        const mode = (layout && layout.widths[card.getAttribute('data-card')])
            || defaultWidths[card.getAttribute('data-card')] || 'normal';
        if (mode === 'full') return cols;
        if (mode === 'wide') return Math.min(2, cols);
        return 1;
    };

    /* Masonry placement: walk cards in DOM order, drop each into the
       shortest column; full-width cards start at the tallest row. Only
       runs on desktop widths; mobile CSS stacks everything in one column. */
    function place() {
        const r = root();
        if (!r || activeDrag) return;
        const cols = effectiveColumns();
        r.style.setProperty('--omlx-cols', String(cols));
        const all = cards();
        for (const card of all) {
            const span = Math.min(spanFor(card), cols);
            card.style.gridColumn = `1 / span ${span}`;
            card.dataset.span = span >= cols ? 'full' : String(span);
        }
        const heights = new Map(all.map(card => [card, card.getBoundingClientRect().height]));
        const cursor = new Array(cols).fill(1);
        for (const card of all) {
            const span = Math.min(spanFor(card), cols);
            let col;
            if (span >= cols) {
                col = 0;
                const tallest = Math.max(...cursor);
                cursor.fill(tallest);
            } else {
                col = 0;
                for (let i = 1; i <= cols - span; i++) {
                    if (Math.max(...cursor.slice(i, i + span)) <
                        Math.max(...cursor.slice(col, col + span))) col = i;
                }
            }
            const row = Math.max(...cursor.slice(col, col + span));
            const height = heights.get(card);
            const rows = Math.max(1, Math.ceil((height + GAP) / ROW));
            card.style.gridColumn = `${col + 1} / span ${span}`;
            card.style.gridRow = `${row} / span ${rows}`;
            // Expose the effective span to CSS (narrow-card typography fixes)
            card.dataset.span = span >= cols ? 'full' : String(span);
            for (let i = col; i < col + span; i++) cursor[i] = row + rows;
        }
    }

    // Width measurement and placement complete before the browser paints.
    // Never publish an intermediate layout in another animation frame.
    function placeStable() {
        if (activeDrag) return;
        place();
    }

    function applyPageWidth() {
        const shell = document.querySelector('.omlx-page-shell');
        if (shell) {
            shell.style.setProperty('--omlx-max-width',
                (settings.maxWidth || 1740) + 'px');
        }
    }

    function serialize() {
        layout = {
            order: cards().map(el => el.getAttribute('data-card')),
            widths: layout ? layout.widths : {},
        };
    }

    function applyOrder(order) {
        const r = root();
        const byId = {};
        cards().forEach(el => { byId[el.getAttribute('data-card')] = el; });
        for (const id of order) {
            const el = byId[id];
            if (el) { r.appendChild(el); delete byId[id]; }
        }
        // Cards added upstream after this arrangement was saved: append in
        // template order instead of vanishing.
        Object.values(byId).forEach(el => r.appendChild(el));
    }

    /* Per-card width button: cycles normal -> wide -> full. The icon and
       title show the NEXT state (what the click will do): 'minus'/'arrows'
       icons on the current state read as actions and were misread as
       reversed. window.t is defined by base.html before any page script
       runs; fall back to English otherwise. */
    const FALLBACK_LABELS = {
        normal: 'Normal width', wide: 'Wide', full: 'Full width',
    };
    function widthLabels() {
        const label = (key) => {
            try {
                const v = typeof window.t === 'function'
                    ? window.t('status.width_' + key) : key;
                return v && !v.startsWith('status.') ? v : FALLBACK_LABELS[key];
            } catch (e) { return FALLBACK_LABELS[key]; }
        };
        return {
            normal: label('normal'), wide: label('wide'),
            full: label('full'),
        };
    }

    function widthIcon(nextMode) {
        // Icon previews the NEXT state (what clicking produces).
        if (nextMode === 'wide') return 'columns-2';
        if (nextMode === 'full') return 'maximize-2';
        return 'square';
    }

    const NEXT_MODE = { normal: 'wide', wide: 'full', full: 'normal' };

    function refreshWidthButtons() {
        const labels = widthLabels();
        for (const card of cards()) {
            const btn = card.querySelector(':scope > .omlx-width-btn');
            if (!btn) continue;
            const mode = currentMode(card);
            const next = NEXT_MODE[mode];
            btn.dataset.mode = mode;
            btn.title = `${labels[mode]} → ${labels[next]}`;
            // Fixed icon-name template, no user data; rebuilt via DOM API
            // so no HTML ever gets parsed from state.
            btn.textContent = '';
            const icon = document.createElement('i');
            icon.setAttribute('data-lucide', widthIcon(next));
            icon.className = 'w-3.5 h-3.5';
            btn.appendChild(icon);
        }
        if (window.lucide) lucide.createIcons();
    }

    function currentMode(card) {
        const id = card.getAttribute('data-card');
        return (layout.widths[id] || defaultWidths[id] || 'normal');
    }

    function cycleWidth(card) {
        activeDrag?.cancel();
        const id = card.getAttribute('data-card');
        const mode = WIDTH_MODES[
            (WIDTH_MODES.indexOf(currentMode(card)) + 1) % WIDTH_MODES.length];
        layout.widths[id] = mode;
        saveLayout();
        refreshWidthButtons();
        placeStable();
    }

    function injectWidthButtons() {
        for (const card of cards()) {
            if (card.querySelector(':scope > .omlx-width-btn')) continue;
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'omlx-width-btn';
            btn.addEventListener('click', () => cycleWidth(card));
            card.appendChild(btn);
        }
        refreshWidthButtons();
    }

    /* Preview only during a drag. Freeze geometry so live statistics cannot
       change hit targets. Move the original DOM node once, on pointerup.
       A line marks insertion before/after a target, not a promised grid cell. */
    const HYST = 24;
    const startDrag = (evt, card) => {
        if (activeDrag || evt.isPrimary === false) return;
        const pointerId = evt.pointerId;
        const startX = evt.clientX, startY = evt.clientY;
        let dragging = false, candidate = null, indicator = null;
        let x = startX, y = startY, scrollFrame = null;
        let frozen = [], targets = [], rootStyle = null, gridBounds = null;
        const r = root();
        const docRect = el => {
            const b = el.getBoundingClientRect();
            return {left: b.left + window.scrollX, top: b.top + window.scrollY,
                right: b.right + window.scrollX, bottom: b.bottom + window.scrollY,
                width: b.width, height: b.height};
        };
        const restore = (el, props) => {
            for (const [name, value, priority] of props) {
                if (value) el.style.setProperty(name, value, priority);
                else el.style.removeProperty(name);
            }
        };
        const remember = (el, names) => names.map(name =>
            [name, el.style.getPropertyValue(name), el.style.getPropertyPriority(name)]);
        const begin = () => {
            dragging = true;
            const all = cards();
            gridBounds = docRect(r);
            targets = all.filter(el => el !== card).map(el => ({el, box: docRect(el)}));
            frozen = all.map(el => ({el, box: docRect(el),
                props: remember(el, ['height', 'min-height', 'max-height', 'overflow'])}));
            rootStyle = remember(r, ['height', 'overflow-anchor']);
            r.style.height = `${gridBounds.height}px`;
            r.style.overflowAnchor = 'none';
            for (const {el, box} of frozen) {
                el.style.height = `${box.height}px`;
                el.style.minHeight = `${box.height}px`;
                el.style.maxHeight = `${box.height}px`;
                el.style.overflow = 'hidden';
            }
            card.classList.add('omlx-dragging');
            document.documentElement.classList.add('omlx-drag-active');
            indicator = document.createElement('div');
            indicator.setAttribute('data-omlx-drop-indicator', '');
            indicator.setAttribute('aria-hidden', 'true');
            indicator.style.cssText = 'position:fixed;pointer-events:none;z-index:1000;' +
                'background:#3b82f6;border-radius:3px;box-shadow:0 0 0 1px white;display:none';
            document.body.appendChild(indicator);
        };
        const drawIndicator = () => {
            if (!candidate) { indicator.style.display = 'none'; return; }
            const {box:b, vertical, before} = candidate;
            const left = (vertical ? b.left : (before ? b.left : b.right)) - window.scrollX;
            const top = (vertical ? (before ? b.top : b.bottom) : b.top) - window.scrollY;
            Object.assign(indicator.style, {display:'block', left:`${left - 2}px`, top:`${top - 2}px`,
                width:`${vertical ? b.width : 4}px`, height:`${vertical ? 4 : b.height}px`});
        };
        const updateTarget = () => {
            const px = x + window.scrollX, py = y + window.scrollY;
            if (x < 0 || y < 0 || x > window.innerWidth || y > window.innerHeight ||
                px < gridBounds.left || px > gridBounds.right ||
                py < gridBounds.top || py > gridBounds.bottom) {
                candidate = null; indicator.style.display = 'none'; return;
            }
            const target = targets.find(({box:b}) =>
                px >= b.left && px <= b.right && py >= b.top && py <= b.bottom);
            if (!target) {
                // Gaps hold the last valid preview. Keep its line anchored
                // to the frozen target when the page scrolls.
                drawIndicator(); return;
            }
            const b = target.box;
            const dx = px - (b.left + b.width / 2), dy = py - (b.top + b.height / 2);
            const vertical = Math.abs(dx) * 1.5 <= Math.abs(dy);
            const delta = vertical ? dy : dx;
            if (Math.abs(delta) < HYST) { drawIndicator(); return; }
            candidate = {el: target.el, before: delta < 0, box:b, vertical};
            drawIndicator();
        };
        const scroll = () => {
            scrollFrame = null;
            if (!dragging) return;
            if (x >= 0 && x <= window.innerWidth && y >= 0 && y <= window.innerHeight) {
                if (y < 60) window.scrollBy(0, -12);
                else if (y > window.innerHeight - 60) window.scrollBy(0, 12);
                updateTarget();
            }
            scrollFrame = requestAnimationFrame(scroll);
        };
        const onMove = e => {
            if (e.pointerId !== pointerId) return;
            x = e.clientX; y = e.clientY;
            if (!dragging) {
                if ((x-startX)**2 + (y-startY)**2 < DRAG_THRESHOLD**2) return;
                begin();
                scrollFrame = requestAnimationFrame(scroll);
            }
            e.preventDefault();
            updateTarget();
        };
        const cleanup = commit => {
            window.removeEventListener('pointermove', onMove, true);
            window.removeEventListener('pointerup', onUp, true);
            window.removeEventListener('pointercancel', onCancel, true);
            window.removeEventListener('keydown', onKey, true);
            window.removeEventListener('blur', onCancel, true);
            window.removeEventListener('resize', onCancel, true);
            document.removeEventListener('visibilitychange', onVisibility);
            if (scrollFrame !== null) cancelAnimationFrame(scrollFrame);
            const wasDragging = dragging;
            dragging = false;
            indicator?.remove();
            card.classList.remove('omlx-dragging');
            document.documentElement.classList.remove('omlx-drag-active');
            for (const {el, props} of frozen) restore(el, props);
            if (rootStyle) restore(r, rootStyle);
            activeDrag = null;
            if (commit && candidate && candidate.el.parentElement === r) {
                const prior = cards();
                const reference = candidate.before ? candidate.el : candidate.el.nextSibling;
                if (reference !== card) r.insertBefore(card, reference);
                if (cards().some((el,i) => el !== prior[i])) {
                    serialize(); saveLayout();
                }
            }
            if (wasDragging) placeStable();
        };
        const onUp = e => {
            if (e.pointerId !== pointerId) return;
            if (dragging) { x=e.clientX; y=e.clientY; updateTarget(); }
            cleanup(dragging);
        };
        const onCancel = e => {
            if (e?.pointerId !== undefined && e.pointerId !== pointerId) return;
            cleanup(false);
        };
        const onKey = e => { if (e.key === 'Escape') { e.preventDefault(); cleanup(false); } };
        const onVisibility = () => { if (document.hidden) cleanup(false); };
        activeDrag = {cancel: () => cleanup(false)};
        window.addEventListener('pointermove', onMove, true);
        window.addEventListener('pointerup', onUp, true);
        window.addEventListener('pointercancel', onCancel, true);
        window.addEventListener('keydown', onKey, true);
        window.addEventListener('blur', onCancel, true);
        window.addEventListener('resize', onCancel, true);
        document.addEventListener('visibilitychange', onVisibility);
        evt.preventDefault();
    };

    window.resetStatusLayout = () => {
        activeDrag?.cancel();
        try {
            localStorage.removeItem(LAYOUT_KEY);
            localStorage.removeItem(SETTINGS_KEY);
        } catch (e) { /* ignore */ }
        settings = loadSettings();
        layout = { order: [...defaultOrder], widths: { ...defaultWidths } };
        for (const card of cards()) {
            card.style.gridColumn = '';
            card.style.gridRow = '';
        }
        applyOrder(defaultOrder);
        refreshWidthButtons();
        applyPageWidth();
        placeStable();
        window.dispatchEvent(new CustomEvent('omlx-layout-reset'));
    };

    window.applyStatusLayoutSettings = () => {
        activeDrag?.cancel();
        settings = loadSettings();
        applyPageWidth();
        placeStable();
    };

    const wire = () => {
        const r = root();
        if (!r || wired) return;
        wired = true;

        defaultOrder = cards().map(el => el.getAttribute('data-card'));
        for (const card of cards()) {
            defaultWidths[card.getAttribute('data-card')] =
                card.dataset.defaultSpan === 'full' ? 'full' : 'normal';
        }
        layout = loadLayout() || { order: [...defaultOrder], widths: {} };
        // Drop saved modes for cards that no longer exist; unknown cards
        // keep their defaults.
        for (const id of Object.keys(layout.widths)) {
            if (!defaultOrder.includes(id)) delete layout.widths[id];
        }
        applyOrder(layout.order);

        r.querySelectorAll('[data-card]').forEach(card => {
            const handle = card.querySelector('.omlx-drag-handle');
            if (!handle) return;
            handle.addEventListener('pointerdown', (e) => {
                if (e.pointerType === 'mouse' && e.button !== 0) return;
                startDrag(e, card);
            });
        });

        injectWidthButtons();
        applyPageWidth();
        placeStable();

        // Active-models lists, cache bars etc. change card height at runtime.
        let roTimer = null;
        try {
            const ro = new ResizeObserver(() => {
                if (activeDrag) return;
                clearTimeout(roTimer);
                roTimer = setTimeout(placeStable, 300);
            });
            ro.observe(r);
        } catch (e) { /* no ResizeObserver: heights update on next drag */ }

        let resizeTimer = null;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(placeStable, 150);
        });
    };

    if (document.readyState === 'complete') {
        setTimeout(wire, 0);
    } else {
        window.addEventListener('load', () => setTimeout(wire, 50));
    }
})();
