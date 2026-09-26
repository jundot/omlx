/*
 * Console-wide interaction pieces: one toast, one command palette, one
 * keyboard-shortcut overview, one animated number transition.
 *
 * Everything the console shows as transient feedback goes through omlxToast();
 * the palette and the shortcut overview are the two global overlays. The module
 * is plain DOM + Alpine-free on purpose: it has to work on every page the
 * console ships (dashboard, chat, login), not only inside the dashboard's
 * Alpine root, and the dashboard registers its own commands on top of the
 * built-in "go to" ones.
 *
 * Copy goes through window.t(); the only literal here is the missing-value
 * dash, which format.js also uses.
 */
(function (global) {
    'use strict';

    var DASH = '—';
    var MAX_RESULTS = 12;
    var TOAST_DURATION = 6000;
    var TOAST_LEAVE_MS = 200;
    var COUNT_DURATION = 300;

    function t(key) {
        return typeof global.t === 'function' ? global.t(key) : key;
    }

    function now() {
        if (global.performance && typeof global.performance.now === 'function') {
            return global.performance.now();
        }
        return Date.now();
    }

    function schedule(callback) {
        if (typeof global.requestAnimationFrame === 'function') {
            return global.requestAnimationFrame(callback);
        }
        return setTimeout(function () { callback(now()); }, 16);
    }

    function unschedule(handle) {
        if (handle === null || handle === undefined) return;
        if (typeof global.cancelAnimationFrame === 'function') global.cancelAnimationFrame(handle);
        else clearTimeout(handle);
    }

    function prefersReducedMotion() {
        try {
            return !!(global.matchMedia
                && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
        } catch (error) {
            return false;
        }
    }

    /* ===================== toast ===================== */

    var TOAST_TONES = ['green', 'orange', 'red', 'blue', 'neutral'];
    // Red is the only tone that interrupts. A warning stays polite so it does
    // not talk over whatever the screen reader is reading.
    var ERROR_TONES = ['red'];

    function toastSemantics(tone) {
        var error = ERROR_TONES.indexOf(tone) >= 0;
        return {
            role: error ? 'alert' : 'status',
            ariaLive: error ? 'assertive' : 'polite',
        };
    }

    var openToasts = {};
    var toastSeq = 0;

    function toastHost() {
        var host = global.document.getElementById('omlx-toast-stack');
        if (host) return host;
        host = global.document.createElement('div');
        host.id = 'omlx-toast-stack';
        host.className = 'toast-stack';
        host.setAttribute('data-omlx-toasts', '');
        global.document.body.appendChild(host);
        return host;
    }

    function removeElement(element) {
        if (element && element.parentNode) element.parentNode.removeChild(element);
    }

    function closeToast(entry) {
        if (!entry || entry.closed) return;
        entry.closed = true;
        clearTimeout(entry.timer);
        delete openToasts[entry.id];
        var element = entry.element;
        if (!element || !element.parentNode) return;
        if (prefersReducedMotion() || !global.document.body.contains(element)) {
            removeElement(element);
            return;
        }
        element.classList.add('toast--leaving');
        element.addEventListener('transitionend', function () { removeElement(element); });
        setTimeout(function () { removeElement(element); }, TOAST_LEAVE_MS);
    }

    function restartToastClock(entry) {
        clearTimeout(entry.timer);
        entry.remaining = entry.duration;
        entry.startedAt = now();
        if (!entry.duration) return;
        entry.timer = setTimeout(function () { closeToast(entry); }, entry.remaining);
    }

    function pauseToast(entry) {
        if (!entry.timer) return;
        clearTimeout(entry.timer);
        entry.timer = null;
        entry.remaining = Math.max(0, entry.remaining - (now() - entry.startedAt));
    }

    function resumeToast(entry) {
        if (entry.timer || entry.closed || !entry.remaining) return;
        entry.startedAt = now();
        entry.timer = setTimeout(function () { closeToast(entry); }, entry.remaining);
    }

    function fillToast(entry, tone, title, message) {
        var element = entry.element;
        var semantics = toastSemantics(tone);
        element.className = 'toast toast--' + tone;
        element.setAttribute('role', semantics.role);
        element.setAttribute('aria-live', semantics.ariaLive);
        element.setAttribute('data-toast-tone', tone);
        entry.title.textContent = title || '';
        entry.title.hidden = !title;
        entry.message.textContent = message || '';
        entry.message.hidden = !message;
    }

    /*
     * omlxToast({tone, title, message, id, duration}) → { id, close() }.
     *
     * Tones match the badges (green/orange/red/blue/neutral). Passing the same
     * `id` again updates that toast in place instead of stacking a duplicate,
     * which is what a poll that keeps failing needs. Auto-dismisses, pauses
     * while the pointer or the keyboard focus is inside it, and can be closed
     * by hand.
     */
    function omlxToast(options) {
        var opts = options || {};
        if (!global.document || !global.document.body) return { id: null, close: function () {} };
        var tone = TOAST_TONES.indexOf(opts.tone) >= 0 ? opts.tone : 'neutral';
        var duration = opts.duration === undefined ? TOAST_DURATION : opts.duration;
        var id = opts.id || ('omlx-toast-' + (++toastSeq));
        var entry = openToasts[id];

        if (!entry) {
            var element = global.document.createElement('div');
            var body = global.document.createElement('div');
            body.className = 'toast__body';
            var title = global.document.createElement('p');
            title.className = 'toast__title';
            var message = global.document.createElement('p');
            message.className = 'toast__message';
            body.appendChild(title);
            body.appendChild(message);
            var close = global.document.createElement('button');
            close.type = 'button';
            close.className = 'toast__close';
            close.setAttribute('aria-label', t('toast.dismiss'));
            close.textContent = '×';
            element.appendChild(body);
            element.appendChild(close);
            entry = {
                id: id, element: element, title: title, message: message,
                duration: duration, remaining: duration, startedAt: now(),
                timer: null, closed: false,
            };
            openToasts[id] = entry;
            close.addEventListener('click', function () { closeToast(entry); });
            element.addEventListener('mouseenter', function () { pauseToast(entry); });
            element.addEventListener('mouseleave', function () { resumeToast(entry); });
            element.addEventListener('focusin', function () { pauseToast(entry); });
            element.addEventListener('focusout', function () { resumeToast(entry); });
            toastHost().appendChild(element);
        } else {
            entry.duration = duration;
        }

        fillToast(entry, tone, opts.title, opts.message);
        restartToastClock(entry);
        return { id: id, close: function () { closeToast(entry); } };
    }

    /* ===================== number transition ===================== */

    function easeOutCubic(progress) {
        var clamped = progress < 0 ? 0 : (progress > 1 ? 1 : progress);
        return 1 - Math.pow(1 - clamped, 3);
    }

    /* The value shown mid-flight: counts round to whole units, one-decimal
       figures keep their decimal. */
    function transitionValue(from, to, progress, decimals) {
        var eased = from + (to - from) * easeOutCubic(progress);
        return decimals > 0 ? eased.toFixed(decimals) : String(Math.round(eased));
    }

    function defaultFormat(decimals) {
        return function (value) {
            return decimals > 0 ? Number(value).toFixed(decimals) : String(Math.round(value));
        };
    }

    /*
     * omlxCountUp(element, target, {decimals, format, placeholder}) animates the
     * element's text from the previous value to `target` over ~300ms. Counts
     * round to an integer, one-decimal figures keep the decimal, and the final
     * frame is the exact target rather than an eased remainder. Callers that
     * pass `format` own the unit (window.formatCount, toFixed, …).
     * Reduced motion writes the target straight away.
     */
    function omlxCountUp(element, target, options) {
        if (!element) return;
        var opts = options || {};
        var decimals = opts.decimals || 0;
        var format = opts.format || defaultFormat(decimals);

        if (target === null || target === undefined || target === '' || isNaN(Number(target))) {
            unschedule(element.__omlxFrame);
            element.__omlxFrame = null;
            element.__omlxValue = null;
            element.__omlxCurrent = null;
            element.textContent = opts.placeholder || DASH;
            return;
        }

        var numeric = Number(target);
        var previous = typeof element.__omlxValue === 'number' ? element.__omlxValue : 0;
        if (previous === numeric || prefersReducedMotion()) {
            unschedule(element.__omlxFrame);
            element.__omlxFrame = null;
            element.__omlxValue = numeric;
            element.__omlxCurrent = numeric;
            element.textContent = format(numeric);
            return;
        }

        var from = typeof element.__omlxCurrent === 'number' ? element.__omlxCurrent : previous;
        var startedAt = now();
        unschedule(element.__omlxFrame);

        function step() {
            var progress = (now() - startedAt) / COUNT_DURATION;
            if (progress >= 1) {
                element.__omlxFrame = null;
                element.__omlxCurrent = numeric;
                element.__omlxValue = numeric;
                element.textContent = format(numeric);
                return;
            }
            element.__omlxCurrent = from + (numeric - from) * easeOutCubic(progress);
            element.textContent = format(Number(transitionValue(from, numeric, progress, decimals)));
            element.__omlxFrame = schedule(step);
        }

        element.__omlxFrame = schedule(step);
    }

    /* ===================== command palette ===================== */

    function normalize(value) {
        return String(value === null || value === undefined ? '' : value).toLowerCase().trim();
    }

    /* 0 exact, 1 prefix, 2 word start, 3 substring, 4 subsequence, -1 nothing. */
    function matchScore(text, needle) {
        var haystack = normalize(text);
        if (!needle) return 0;
        if (!haystack) return -1;
        if (haystack === needle) return 0;
        if (haystack.indexOf(needle) === 0) return 1;
        var words = haystack.split(/[\s·/|,:;()[\]-]+/);
        for (var word = 0; word < words.length; word++) {
            if (words[word].indexOf(needle) === 0) return 2;
        }
        if (haystack.indexOf(needle) > 0) return 3;
        var cursor = 0;
        for (var index = 0; index < needle.length; index++) {
            cursor = haystack.indexOf(needle[index], cursor);
            if (cursor < 0) return -1;
            cursor += 1;
        }
        return 4;
    }

    function commandTexts(command) {
        return [command.label, command.group, command.hint].concat(command.keywords || []);
    }

    function filterCommands(commands, query) {
        var needle = normalize(query);
        var matched = [];
        (commands || []).forEach(function (command, index) {
            if (!needle) {
                matched.push({ command: command, score: 0, index: index });
                return;
            }
            var best = -1;
            commandTexts(command).forEach(function (text) {
                var score = matchScore(text, needle);
                if (score >= 0 && (best < 0 || score < best)) best = score;
            });
            if (best >= 0) matched.push({ command: command, score: best, index: index });
        });
        matched.sort(function (a, b) { return a.score - b.score || a.index - b.index; });
        return matched.map(function (entry) { return entry.command; });
    }

    function groupCommands(commands, limit) {
        var groups = [];
        var byName = {};
        commands.slice(0, limit === undefined ? MAX_RESULTS : limit).forEach(function (command) {
            var name = command.group || '';
            if (!byName[name]) {
                byName[name] = { name: name, commands: [] };
                groups.push(byName[name]);
            }
            byName[name].commands.push(command);
        });
        return groups;
    }

    var paletteSources = [];
    var paletteResults = [];
    var paletteItems = [];
    var paletteActive = 0;
    var previousFocus = null;

    function registerPaletteCommands(build) {
        if (typeof build !== 'function') return function () {};
        paletteSources.push(build);
        return function () {
            paletteSources = paletteSources.filter(function (source) { return source !== build; });
        };
    }

    function collectedCommands() {
        var commands = [];
        paletteSources.forEach(function (build) {
            try {
                commands = commands.concat(build() || []);
            } catch (error) {
                // A page source that throws must not take the palette with it.
                if (global.console) global.console.error('palette source failed', error);
            }
        });
        return commands;
    }

    function byId(id) {
        return global.document ? global.document.getElementById(id) : null;
    }

    function paletteDialog() { return byId('omlx-palette'); }
    function shortcutDialog() { return byId('omlx-shortcuts'); }

    function setActive(index) {
        if (!paletteItems.length) return;
        var next = (index + paletteItems.length) % paletteItems.length;
        paletteActive = next;
        paletteItems.forEach(function (item, position) {
            var active = position === next;
            item.classList.toggle('palette__item--active', active);
            item.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        var input = byId('omlx-palette-input');
        if (input) input.setAttribute('aria-activedescendant', paletteItems[next].id);
        if (typeof paletteItems[next].scrollIntoView === 'function') {
            paletteItems[next].scrollIntoView({ block: 'nearest' });
        }
    }

    function moveActive(step) {
        if (paletteItems.length) setActive(paletteActive + step);
    }

    function renderPalette(query) {
        var list = byId('omlx-palette-list');
        if (!list || !global.document) return;
        var empty = byId('omlx-palette-empty');
        var input = byId('omlx-palette-input');
        var commands = filterCommands(collectedCommands(), query).slice(0, MAX_RESULTS);
        paletteResults = commands;
        list.textContent = '';
        paletteItems = [];

        groupCommands(commands).forEach(function (group) {
            if (group.name) {
                var heading = global.document.createElement('div');
                heading.className = 'palette__group';
                heading.setAttribute('aria-hidden', 'true');
                heading.textContent = group.name;
                list.appendChild(heading);
            }
            group.commands.forEach(function (command) {
                var item = global.document.createElement('button');
                item.type = 'button';
                item.className = 'palette__item';
                item.setAttribute('role', 'option');
                item.setAttribute('aria-selected', 'false');
                item.id = 'omlx-palette-item-' + (paletteItems.length + 1);
                var label = global.document.createElement('span');
                label.className = 'palette__item-label';
                label.textContent = command.label;
                item.appendChild(label);
                if (command.hint) {
                    var hint = global.document.createElement('span');
                    hint.className = 'palette__item-hint';
                    hint.textContent = command.hint;
                    item.appendChild(hint);
                }
                item.addEventListener('mousedown', function (event) { event.preventDefault(); });
                item.addEventListener('click', function () { runCommand(command); });
                item.addEventListener('mousemove', function () {
                    setActive(paletteItems.indexOf(item));
                });
                list.appendChild(item);
                paletteItems.push(item);
            });
        });

        if (empty) empty.hidden = paletteItems.length > 0;
        if (input) input.removeAttribute('aria-activedescendant');
        setActive(0);
    }

    function runCommand(command) {
        closePalette();
        if (typeof command.run !== 'function') return;
        try {
            command.run();
        } catch (error) {
            if (global.console) global.console.error('palette command failed', error);
        }
    }

    function runActive() {
        if (paletteResults[paletteActive]) runCommand(paletteResults[paletteActive]);
    }

    function openPalette() {
        var dialog = paletteDialog();
        var input = byId('omlx-palette-input');
        if (!dialog || !input) return;
        if (!dialog.open) {
            previousFocus = global.document.activeElement;
            if (typeof dialog.showModal === 'function') dialog.showModal();
            else dialog.setAttribute('open', '');
        }
        input.value = '';
        renderPalette('');
        input.focus();
    }

    function restoreFocus() {
        var target = previousFocus;
        previousFocus = null;
        if (target && global.document.contains(target) && typeof target.focus === 'function') {
            target.focus();
        }
    }

    function closePalette() {
        var dialog = paletteDialog();
        if (!dialog) return;
        if (dialog.open && typeof dialog.close === 'function') dialog.close();
        else dialog.removeAttribute('open');
        restoreFocus();
    }

    function isPaletteOpen() {
        var dialog = paletteDialog();
        return !!(dialog && dialog.open);
    }

    function togglePalette() {
        if (isPaletteOpen()) closePalette();
        else openPalette();
    }

    function openShortcuts() {
        var dialog = shortcutDialog();
        if (!dialog) return false;
        if (!dialog.open) {
            previousFocus = global.document.activeElement;
            if (typeof dialog.showModal === 'function') dialog.showModal();
            else dialog.setAttribute('open', '');
        }
        var close = dialog.querySelector('[data-shortcuts-close]');
        if (close) close.focus();
        return true;
    }

    function closeShortcuts() {
        var dialog = shortcutDialog();
        if (!dialog) return;
        if (dialog.open && typeof dialog.close === 'function') dialog.close();
        else dialog.removeAttribute('open');
        restoreFocus();
    }

    function isShortcutsOpen() {
        var dialog = shortcutDialog();
        return !!(dialog && dialog.open);
    }

    function toggleShortcuts() {
        if (isShortcutsOpen()) closeShortcuts();
        else openShortcuts();
    }

    /* ===================== wiring ===================== */

    function isTypingTarget(target) {
        if (!target) return false;
        var tag = target.tagName;
        return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
            || target.isContentEditable === true;
    }

    /* Tab must not escape a modal, and it must not land on the page behind it.
       Native <dialog> already inerts the rest of the document; this keeps the
       promise on browsers that only half-implement it. */
    function trapFocus(event) {
        if (event.key !== 'Tab') return;
        var dialog = event.currentTarget;
        var controls = Array.prototype.slice.call(
            dialog.querySelectorAll('a[href], button, input, select, textarea, [tabindex]')
        ).filter(function (element) {
            return element.tabIndex >= 0 && !element.matches(':disabled')
                && element.getClientRects().length
                && global.getComputedStyle(element).visibility !== 'hidden';
        });
        if (!controls.length) return;
        var index = controls.indexOf(global.document.activeElement);
        var last = controls.length - 1;
        if (event.shiftKey ? index <= 0 : (index === last || index === -1)) {
            event.preventDefault();
            (event.shiftKey ? controls[last] : controls[0]).focus();
        }
    }

    function handleGlobalKeydown(event) {
        if (event.defaultPrevented) return;
        var mod = event.metaKey || event.ctrlKey;
        var key = event.key;

        if (mod && (key === 'k' || key === 'K')) {
            event.preventDefault();
            togglePalette();
            return;
        }
        if (mod && key === '/') {
            if (shortcutDialog()) {
                event.preventDefault();
                toggleShortcuts();
            }
            return;
        }
        if (isTypingTarget(event.target)) return;
        if (key === '?') {
            // A page that already owns a shortcut overview (chat) keeps it.
            if (shortcutDialog()) {
                event.preventDefault();
                toggleShortcuts();
            }
            return;
        }
        if (key === '/' && !mod && !event.altKey) {
            event.preventDefault();
            togglePalette();
        }
    }

    function wireDialog(dialog) {
        dialog.addEventListener('keydown', trapFocus);
        dialog.addEventListener('cancel', function (event) {
            event.preventDefault();
            if (dialog === paletteDialog()) closePalette();
            else closeShortcuts();
        });
        dialog.addEventListener('close', restoreFocus);
        dialog.addEventListener('click', function (event) {
            if (event.target === dialog) {
                if (dialog === paletteDialog()) closePalette();
                else closeShortcuts();
            }
        });
    }

    function wireOverlays() {
        var palette = paletteDialog();
        var input = byId('omlx-palette-input');
        if (palette && input) {
            wireDialog(palette);
            input.addEventListener('input', function () { renderPalette(input.value); });
            input.addEventListener('keydown', function (event) {
                if (event.key === 'ArrowDown') { event.preventDefault(); moveActive(1); }
                else if (event.key === 'ArrowUp') { event.preventDefault(); moveActive(-1); }
                else if (event.key === 'Home') { event.preventDefault(); setActive(0); }
                else if (event.key === 'End') { event.preventDefault(); setActive(paletteItems.length - 1); }
                else if (event.key === 'Enter') { event.preventDefault(); runActive(); }
            });
        }
        var shortcuts = shortcutDialog();
        if (shortcuts) {
            wireDialog(shortcuts);
            Array.prototype.forEach.call(
                shortcuts.querySelectorAll('[data-shortcuts-close]'),
                function (button) { button.addEventListener('click', closeShortcuts); }
            );
        }
    }

    /* What every page can offer on its own: the two destinations the console
       has, and its own shortcut overview. The dashboard registers the tabs,
       blocks and settings sections on top (and skips the one command that
       would be a duplicate). */
    function baseCommands() {
        var commands = [];
        var path = typeof global.location !== 'undefined'
            ? String(global.location.pathname || '') : '';
        if (path.indexOf('/chat') < 0) {
            commands.push({
                id: 'goto-chat',
                label: t('navbar.tab.chat'),
                group: t('palette.group_pages'),
                keywords: ['chat'],
                run: function () { global.location.href = '/admin/chat'; },
            });
        }
        var onDashboard = !!(global.document
            && global.document.querySelector('[x-data^="dashboard("]'));
        if (!onDashboard) {
            commands.push({
                id: 'goto-dashboard',
                label: t('navbar.tab.status'),
                group: t('palette.group_pages'),
                keywords: ['dashboard', 'status'],
                run: function () { global.location.href = '/admin/dashboard'; },
            });
        }
        if (shortcutDialog()) {
            commands.push({
                id: 'show-shortcuts',
                label: t('shortcuts.title'),
                group: t('palette.group_pages'),
                keywords: ['keyboard', 'keys', 'help'],
                run: function () { openShortcuts(); },
            });
        }
        return commands;
    }

    function init() {
        if (!global.document) return;
        wireOverlays();
        registerPaletteCommands(baseCommands);
        global.document.addEventListener('keydown', handleGlobalKeydown);
    }

    /* ===================== exports ===================== */

    global.omlxToast = omlxToast;
    global.omlxPrefersReducedMotion = prefersReducedMotion;
    global.omlxCountUp = omlxCountUp;
    global.omlxPalette = {
        open: openPalette,
        close: closePalette,
        toggle: togglePalette,
        isOpen: isPaletteOpen,
        register: registerPaletteCommands,
        commands: collectedCommands,
        filter: filterCommands,
    };
    global.omlxShortcuts = {
        open: openShortcuts,
        close: closeShortcuts,
        toggle: toggleShortcuts,
        isOpen: isShortcutsOpen,
    };

    if (global.document) {
        if (global.document.readyState === 'loading') {
            global.document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
    }

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = {
            DASH: DASH,
            MAX_RESULTS: MAX_RESULTS,
            TOAST_TONES: TOAST_TONES,
            ERROR_TONES: ERROR_TONES,
            toastSemantics: toastSemantics,
            easeOutCubic: easeOutCubic,
            transitionValue: transitionValue,
            normalize: normalize,
            matchScore: matchScore,
            filterCommands: filterCommands,
            groupCommands: groupCommands,
        };
    }
})(typeof window !== 'undefined' ? window : globalThis);
