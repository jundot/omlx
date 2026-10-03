// SPDX-License-Identifier: Apache-2.0
/*
 * The Settings tab's in-page navigation: which sections a sub-tab has, what
 * the deep link for a section is, and which one the reader is currently
 * looking at.
 *
 * The section list itself is data (the template renders it as JSON and reads
 * it back, so the titles come from the i18n catalogue). This file holds the
 * three pure rules behind the rail so they can be tested without a browser —
 * tests/admin_settings_nav.test.cjs — and so dashboard.js stays a view model.
 */

(function (global) {
    'use strict';

    // Fallback for the line below which a section counts as "current": the
    // caller passes the sticky row's own bottom edge (see dashboard.js), which
    // is what a clicked section lands on. This is only used when that cannot be
    // measured.
    const ACTIVE_OFFSET = 120;

    /* The shareable link to a section: origin + path + its anchor. */
    function sectionAnchor(origin, pathname, sectionId) {
        return origin + pathname + '#' + sectionId;
    }

    /*
     * The section a scroll position is inside, from the measured top offsets
     * (viewport coordinates, in the same order as `sections`). `clearance` is
     * the line a section has to reach to count as current — the bottom edge of
     * the sticky controls above it; it defaults to ACTIVE_OFFSET. Before the
     * first section starts, the first one stays current so the rail is never
     * empty; past the last one, the last stays current.
     */
    function activeSection(offsets, scrollTop, sections, clearance) {
        if (!sections.length) return null;
        const line = clearance === undefined || clearance === null ? ACTIVE_OFFSET : clearance;
        // A section the browser scrolled to sits on the line, sub-pixel and all,
        // so the comparison allows the last pixel: without it the section the
        // reader just clicked stayed one px below the line and never won.
        const threshold = scrollTop + line + 1;
        let current = sections[0].id;
        for (let index = 0; index < sections.length; index += 1) {
            if (offsets[index] <= threshold) current = sections[index].id;
            else break;
        }
        return current;
    }

    /*
     * Where that line sits, in viewport coordinates.
     *
     * A section the rail scrolls to lands at its own `scroll-margin-top`, so
     * that is the line: reading it means the clicked section is exactly the one
     * that counts, whatever the sticky row above measures. Failing that, the
     * row's bottom edge, and failing that the constant.
     */
    function activeClearance(row, section) {
        if (section) {
            const styles = global.getComputedStyle ? global.getComputedStyle(section) : null;
            const margin = styles ? parseFloat(styles.scrollMarginTop) : NaN;
            if (Number.isFinite(margin) && margin > 0) return margin;
        }
        if (!row || !row.getBoundingClientRect) return ACTIVE_OFFSET;
        const rect = row.getBoundingClientRect();
        return rect.height > 0 ? rect.bottom : ACTIVE_OFFSET;
    }

    global.OMLXSettingsNav = {
        ACTIVE_OFFSET,
        sectionAnchor,
        activeSection,
        activeClearance,
    };
})(typeof window === 'undefined' ? globalThis : window);
