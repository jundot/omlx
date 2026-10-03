# SPDX-License-Identifier: Apache-2.0
"""Structure of the Logs tab after the console refactor.

The tab used to be one dark ``<textarea>`` of raw text with a client-side
minimum-level filter. It is now a four-column viewer (time / level / module /
message) that keeps continuation lines with their record, collapses consecutive
identical warnings and errors behind a ×N counter, parses the memory-guard lines
into chips with the two remedies they suggest as inline actions, appends each
poll instead of rebuilding the list, and mounts only the rows near the viewport.

These are static assertions over the template and the scripts; the parser, the
aggregator and the windowing math are covered by tests/admin_logs.test.cjs.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "omlx" / "admin"
LOGS = (ADMIN / "templates" / "dashboard" / "_logs.html").read_text(encoding="utf-8")
SETTINGS = (ADMIN / "templates" / "dashboard" / "_settings.html").read_text(encoding="utf-8")
DASHBOARD = (ADMIN / "templates" / "dashboard.html").read_text(encoding="utf-8")
DASHBOARD_JS = (ADMIN / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
LOGS_JS = (ADMIN / "static" / "js" / "logs.js").read_text(encoding="utf-8")
COMPONENTS_CSS = (ADMIN / "static" / "css" / "components.css").read_text(encoding="utf-8")
TOKENS_CSS = (ADMIN / "static" / "css" / "tokens.css").read_text(encoding="utf-8")
EN = json.loads((ADMIN / "i18n" / "en.json").read_text(encoding="utf-8"))


def test_four_structured_columns():
    for key in ("logs.col.time", "logs.col.level", "logs.col.module", "logs.col.message"):
        assert f"t('{key}')" in LOGS
        assert key in EN
    # One grid aligns the header with the rows; the message column is the
    # flexible one.
    assert "log-grid" in LOGS
    assert "grid-template-columns" in COMPONENTS_CSS
    for name in (".log-head", ".log-row", ".log-viewport"):
        assert f"{name} {{" in COMPONENTS_CSS


def test_every_row_sits_inside_a_grid():
    """ARIA's required-parent rule: ``cell``/``columnheader`` -> ``row`` ->
    ``rowgroup`` -> ``grid``.

    The rebuilt viewer declared the leaf roles and no table-like ancestor
    anywhere, so assistive tech fell back to plain divs and the screen read
    as an unlabelled pile of text.
    """
    from html.parser import HTMLParser

    offenders = []
    seen = {"grid": 0, "row": 0}
    required = {"rowgroup": "grid", "row": "rowgroup",
                "columnheader": "rowgroup", "cell": "rowgroup"}
    void = {"br", "img", "input", "hr", "meta", "link"}

    class Scanner(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack = []  # (tag, role or None) for every open element

        def handle_starttag(self, tag, attrs):
            if tag in void:
                return
            role = dict(attrs).get("role")
            roles = [r for _, r in self.stack if r]
            if role in required and required[role] not in roles:
                offenders.append((self.getpos()[0], role, roles))
            if role in seen:
                seen[role] += 1
            self.stack.append((tag, role))

        def handle_endtag(self, tag):
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    return

    Scanner().feed(LOGS)
    assert seen["grid"] >= 1 and seen["row"] >= 1, seen
    assert not offenders, f"a role without the ancestor ARIA requires: {offenders}"


def test_a_row_is_the_parsed_record():
    for field in ("row.time", "row.level", "row.module", "row.message"):
        assert field in LOGS
    assert "parseLogText" in LOGS_JS
    assert "continuation" in LOGS_JS, "continuation lines stay with their record"
    assert "requestId" in LOGS_JS


def test_the_raw_textarea_is_gone():
    assert "<textarea" not in LOGS
    assert "filteredLogContent" not in LOGS
    assert "filteredLogContent" not in DASHBOARD_JS


def test_each_level_has_its_own_badge_tone():
    """Six levels, six tones: a level that shares a colour with another level
    states nothing. The order is the severity order the parser ranks with, and
    the app draws the same six (LogPalette in LogsScreen.swift)."""
    body = DASHBOARD_JS[DASHBOARD_JS.index("logLevelTone(level)") :][:800]
    tones = re.findall(r"return '(badge--[a-z]+)'", body)
    assert tones == [
        "badge--purple",  # CRITICAL
        "badge--red",     # ERROR
        "badge--orange",  # WARNING
        "badge--blue",    # INFO
        "badge--teal",    # DEBUG
    ], "one tone per ranked level, worst first"
    assert len(set(tones)) == len(tones), "two levels may not share a tone"
    assert "return ''" in body, "TRACE keeps the neutral badge"
    for tone in tones:
        assert f".{tone} {{" in COMPONENTS_CSS, tone
    assert "levelRank" in LOGS_JS, "the ranking comes from the parser module"


def test_consecutive_repeats_collapse_behind_a_counter():
    assert "aggregateLogRows" in LOGS_JS
    assert "AGGREGATE_FROM" in LOGS_JS, "only warnings and above collapse"
    assert "'×' + row.count" in LOGS
    assert "row.count > 1" in LOGS
    assert "logs.repeat_tooltip" in EN
    # Every occurrence stays reachable, with its own timestamp.
    assert "occurrences" in LOGS_JS
    assert "logOccurrences.shown" in LOGS
    assert "logs.detail.occurrences" in EN


def test_a_long_run_of_repeats_is_windowed():
    """One element per occurrence: a warning that repeats 20,000 times mounted
    20,000 of them and froze the tab. The panel lists the first window and counts
    the rest; the ×N badge still carries the full number."""
    assert "OCCURRENCE_WINDOW" in LOGS_JS
    assert "occurrenceWindow" in LOGS_JS
    assert "logOccurrences" in DASHBOARD_JS
    assert "logOccurrences.hidden" in LOGS
    assert "logs.detail.occurrences_more" in EN
    assert ".log-occurrences__more {" in COMPONENTS_CSS
    # The windowed list scrolls in its own box: without one, 200 occurrences
    # pushed the viewer off the page (the panel grew by ~4,000px).
    assert 'class="log-occurrences__list"' in LOGS
    box = COMPONENTS_CSS[COMPONENTS_CSS.index(".log-occurrences__list {"):]
    box = box[: box.index("}")]
    assert "max-height:" in box and "overflow-y: auto" in box


def test_memory_guard_chips_and_inline_actions():
    assert "memoryGuardFor" in LOGS_JS
    assert "memoryGuardChips" in LOGS_JS
    assert "logMemoryChips(row)" in LOGS
    assert "log-chips" in LOGS
    for chip in ("usage", "watermark", "ceiling"):
        assert f"logs.memory.{chip}" in LOGS or f"logs.memory.{chip}" in DASHBOARD_JS
        assert f"logs.memory.{chip}" in EN
    # The two remedies switch to Settings and scroll to the section that holds
    # the lever.
    assert "logs.action.raise_tier" in EN
    assert "logs.action.reduce_context" in EN
    assert "anchor: 'memory-guard'" in DASHBOARD_JS
    assert "anchor: 'context-window'" in DASHBOARD_JS
    assert "openLogAction(action.anchor)" in LOGS
    assert 'data-anchor="memory-guard"' in SETTINGS
    assert "setSettingsTab('global')" in DASHBOARD_JS
    assert "data-anchor=" in DASHBOARD_JS, "the fallback target is looked up by anchor"


def test_incremental_refresh_appends_instead_of_rebuilding():
    assert "mergeLogText" in LOGS_JS
    for helper in ("ingestLogText", "commitLogRecords", "dropLeadingLogRecords", "rebuildLogRows"):
        assert helper in DASHBOARD_JS
    # The rows that leave the top of the window must not drag the view with
    # them: the row under the offset is remembered across the poll.
    assert "logAnchor" in DASHBOARD_JS
    assert "restoreLogAnchor" in DASHBOARD_JS
    # The endpoint is untouched: same route, same parameters.
    assert "/admin/api/logs?" in DASHBOARD_JS
    assert "lines: this.logLines.toString()" in DASHBOARD_JS
    assert "fetch(" not in LOGS_JS, "the parser fetches nothing"
    # Auto-scroll survives and now drives the viewport.
    assert "logAutoScroll" in DASHBOARD_JS
    assert "scrollLogToBottom" in DASHBOARD_JS
    assert "logs.auto_scroll" in LOGS and "logs.auto_scroll" in EN


def test_virtual_scrolling_keeps_only_the_window_mounted():
    assert "visibleRange" in LOGS_JS
    assert "logWindow" in DASHBOARD_JS
    assert "visibleLogRows" in DASHBOARD_JS
    assert 'x-for="(row, i) in visibleLogRows"' in LOGS
    assert "translateY(" in LOGS and "logRowHeight" in LOGS
    assert "log-spacer" in LOGS
    assert ".log-spacer {" in COMPONENTS_CSS
    assert "--log-row-height" in TOKENS_CSS
    assert "height: var(--log-row-height)" in COMPONENTS_CSS, "uniform rows keep the window exact"
    for hook in ("onLogScroll", "measureLogViewport", "measureLogRowHeight"):
        assert hook in DASHBOARD_JS


def test_controls_are_the_six_that_were_here_on_shared_components():
    assert "{% call ui.card(" in LOGS
    assert "ui.button(" in LOGS
    assert "ui.segmented(" in LOGS
    assert "ui.empty_state(" in LOGS
    assert "ui.badge(" in LOGS
    for key in (
        "logs.lines_label",
        "logs.refresh_label",
        "logs.file_label",
        "logs.refresh_button",
        "logs.auto_scroll",
        "logs.level_label",
    ):
        assert key in LOGS
        assert key in EN
    assert "setLogMinLevel('{value}')" in LOGS
    assert "restartLogRefresh()" in LOGS
    assert "changeLogFile()" in LOGS


def test_logs_js_loads_before_the_dashboard_script():
    assert "js/logs.js" in DASHBOARD
    assert DASHBOARD.index("js/logs.js") < DASHBOARD.index("js/dashboard.js")


def test_the_counts_in_the_logs_view_are_grouped():
    """A log line count is an exact quantity: it reads 8,405, not 8405 or 0.8万."""
    for expected in (
        "window.formatCountExact(logLines)",
        "window.formatCountExact(logTotalLines)",
        "window.formatCountExact(logRows.length)",
        "window.formatCountExact(logSelectedRow ? logSelectedRow.count : 0)",
        "window.formatCountExact(logOccurrences.hidden)",
        "window.formatCountExact(row.count)",
        "window.formatCountExact(row.lines)",
    ):
        assert expected in LOGS, f"{expected} — the logs view prints a raw count"
