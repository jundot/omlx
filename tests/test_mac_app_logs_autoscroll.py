# SPDX-License-Identifier: Apache-2.0
"""The log screen's auto-scroll follows the window, not the row index.

``LogParser.parse`` numbers records from 0 in every window, so the last row's
id is the window's record count. A log that yields a constant number of header
lines per poll — any homogeneous log fetched with a fixed ``lines`` — therefore
never changes that id, and keying the scroll on it left the list parked after
the first render: the feature stopped firing exactly when it was needed. The
window's own text changes whenever there is anything new to scroll to, and it
does not change when there is not.

CI never runs xcodebuild, so the contract is pinned here, against the source.
"""

from pathlib import Path

SCREEN = (
    Path(__file__).resolve().parents[1]
    / "apps"
    / "omlx-mac"
    / "Sources"
    / "AppView"
    / "Screens"
    / "LogsScreen.swift"
).read_text(encoding="utf-8")


def test_the_auto_scroll_watches_the_window_not_the_row_index():
    assert ".onChange(of: vm.logText)" in SCREEN, (
        "auto-scroll must key on the window: a row id only changes when the "
        "window's record count does"
    )
    assert ".onChange(of: vm.rows.last?.id)" not in SCREEN, (
        "a parse index is not a refresh signal"
    )


def test_the_scroll_still_lands_on_the_newest_row():
    assert "proxy.scrollTo(id, anchor: .bottom)" in SCREEN
    # Following the tail stays the reader's choice, not an assumption.
    assert "guard vm.autoScroll" in SCREEN
