/* Status tab: client-side panel layout.
 *
 * Wraps each Status-tab section in a draggable card. On large screens the
 * two-column grid is created by CSS (.omlx-status-top); this script only
 * reorders/moves DOM nodes between the three containers (left column, right
 * column, below stack) and persists the arrangement in localStorage.
 * DOM nodes are moved with insertBefore/appendChild, so Alpine components
 * and their pollers keep running untouched.
 *
 * Dragging uses Pointer Events (uniform across Safari/Firefox/Chrome)
 * started from the grip handle; HTML5 drag-and-drop is deliberately not
 * used because its handle-restricted semantics differ per browser.
 */
(() => {
    const STORAGE_KEY = 'omlx-sta…t-v3';
    const DRAG_THRESHOLD = 4;   // px before a press becomes a drag
    let wired = false;
    let defaultLayout = null;   // captured at wire() time, before saved layout moves cards

    const containers = () => {
        const root = document.getElementById('status-layout');
        if (!root) return null;
        return {
            left: root.querySelector('[data-column="left"]'),
            right: root.querySelector('[data-column="right"]'),
            below: root.querySelector('[data-zone="below"]'),
        };
    };

    const containerOfPoint = (x, y) => {
        const c = containers();
        if (!c) return null;
        let hit = null;
        for (const key of ['left', 'right', 'below']) {
            const box = c[key].getBoundingClientRect();
            // Generous padding so gaps between cards count as the container.
            const pad = 24;
            if (x >= box.left - pad && x <= box.right + pad &&
                y >= box.top - pad && y <= box.bottom + pad) hit = c[key];
        }
        return hit;
    };

    const serialize = () => {
        const c = containers();
        if (!c) return null;
        const out = {};
        for (const key of ['left', 'right', 'below']) {
            out[key] = [...c[key].querySelectorAll(':scope > .omlx-card')]
                .map(el => el.getAttribute('data-card'));
        }
        return out;
    };

    const captureDefaultLayout = () => {
        // Must reflect the TEMPLATE order, not the current DOM: after drags,
        // querySelectorAll follows the mutated document order and would
        // "restore" the user's arrangement. Captured once at first wire().
        const root = document.getElementById('status-layout');
        const out = { left: [], right: [], below: [] };
        if (!root) return out;
        root.querySelectorAll('[data-card]').forEach(el => {
            const parent = el.getAttribute('data-default-parent');
            if (out[parent]) out[parent].push(el.getAttribute('data-card'));
        });
        return out;
    };

    const allCards = () => {
        const map = {};
        document.querySelectorAll('#status-layout [data-card]').forEach(el => {
            map[el.getAttribute('data-card')] = el;
        });
        return map;
    };

    const applyLayout = (layout) => {
        const c = containers();
        if (!c || !layout) return;
        const cards = allCards();
        for (const key of ['left', 'right', 'below']) {
            for (const id of layout[key] || []) {
                const el = cards[id];
                if (el) c[key].appendChild(el);
            }
        }
    };

    const loadLayout = () => {
        try {
            // Layout-schema version changed (stats default above speed):
            // discard any arrangement saved under the previous default.
            localStorage.removeItem('omlx-sta…t-v2');
            const raw = localStorage.getItem(STORAGE_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch (e) { return null; }
    };

    const saveLayout = () => {
        try {
            const layout = serialize();
            if (layout) localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
        } catch (e) { /* private mode: layout just won't persist */ }
    };

    window.resetStatusLayout = () => {
        try { localStorage.removeItem(STORAGE_KEY); } catch (e) { /* ignore */ }
        applyLayout(defaultLayout || captureDefaultLayout());
    };

    /* Insertion point inside a container: first card whose vertical midpoint
       is below the pointer. */
    const getAfterElement = (container, y) => {
        const cards = [...container.querySelectorAll(':scope > .omlx-card:not(.omlx-dragging)')];
        let closest = null;
        let closestOffset = Number.NEGATIVE_INFINITY;
        for (const el of cards) {
            const box = el.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;
            if (offset < 0 && offset > closestOffset) {
                closestOffset = offset;
                closest = el;
            }
        }
        return closest;
    };

    const autoScroll = (y) => {
        const margin = 60;
        if (y < margin) window.scrollBy(0, -12);
        else if (y > window.innerHeight - margin) window.scrollBy(0, 12);
    };

    const closestContainer = (node) => {
        const c = containers();
        if (!c) return null;
        if (c.left.contains(node)) return c.left;
        if (c.right.contains(node)) return c.right;
        if (c.below.contains(node)) return c.below;
        return null;
    };

    const startDrag = (evt, card) => {
        const startX = evt.clientX, startY = evt.clientY;
        const pointerId = evt.pointerId;
        let dragging = false;

        const onMove = (e) => {
            if (e.pointerId !== pointerId) return;
            if (!dragging) {
                const dx = e.clientX - startX, dy = e.clientY - startY;
                if (dx * dx + dy * dy < DRAG_THRESHOLD * DRAG_THRESHOLD) return;
                dragging = true;
                card.classList.add('omlx-dragging');
                document.documentElement.classList.add('omlx-drag-active');
            }
            e.preventDefault();

            // Hit-test what is under the pointer, ignoring the dragged card.
            const x = e.clientX, y = e.clientY;
            card.style.pointerEvents = 'none';
            const under = document.elementFromPoint(x, y);
            card.style.pointerEvents = '';
            const container = containerOfPoint(x, y) ||
                (under ? closestContainer(under) : null);
            if (!container) return;

            const after = getAfterElement(container, y);
            if (after === card) return;
            if (after) container.insertBefore(card, after);
            else container.appendChild(card);
            autoScroll(y);
        };

        const cleanup = () => {
            window.removeEventListener('pointermove', onMove, true);
            window.removeEventListener('pointerup', onUp, true);
            window.removeEventListener('pointercancel', onUp, true);
            if (!dragging) return;
            dragging = false;
            card.classList.remove('omlx-dragging');
            document.documentElement.classList.remove('omlx-drag-active');
            saveLayout();
        };

        const onUp = (e) => {
            if (e.pointerId !== pointerId) return;
            cleanup();
        };

        // Listeners live on WINDOW, not the handle: Safari drops pointer
        // capture when the dragged node is re-inserted into the DOM, which
        // strands handle-bound listeners mid-drag. Window capture-phase
        // listeners survive any DOM rearrangement, so no capture is needed.
        window.addEventListener('pointermove', onMove, true);
        window.addEventListener('pointerup', onUp, true);
        window.addEventListener('pointercancel', onUp, true);
        evt.preventDefault();
    };

    const wire = () => {
        const root = document.getElementById('status-layout');
        if (!root || wired) return;
        wired = true;

        root.querySelectorAll('[data-card]').forEach(card => {
            const handle = card.querySelector('.omlx-drag-handle');
            if (!handle) return;
            handle.addEventListener('pointerdown', (e) => {
                if (e.pointerType === 'mouse' && e.button !== 0) return;
                startDrag(e, card);
            });
        });

        const saved = loadLayout();
        defaultLayout = captureDefaultLayout();
        if (saved) applyLayout(saved);
    };

    if (document.readyState === 'complete') {
        setTimeout(wire, 0);
    } else {
        window.addEventListener('load', () => setTimeout(wire, 50));
    }
})();
