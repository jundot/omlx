# SPDX-License-Identifier: Apache-2.0
"""Pin the menubar refresh cadence: ``defaultRefreshInterval`` = 0.5 s in the
constant, the poller fallback, and the picker's ``@AppStorage`` default, with the
derived ``historyCapacity`` holding the graphs' one-minute window. CI runs pytest
and never xcodebuild, so this file is what gates a drift back to a literal ``1.0``.
"""

import re
from pathlib import Path

SOURCES = Path(__file__).resolve().parents[1] / "apps" / "omlx-mac" / "Sources"
PREFS = (SOURCES / "Menubar" / "MenubarMetricPrefs.swift").read_text(encoding="utf-8")
STORE = (SOURCES / "Menubar" / "MenubarMetricsStore.swift").read_text(encoding="utf-8")
SCREEN = (
    SOURCES / "AppView" / "Screens" / "AppearanceScreen.swift"
).read_text(encoding="utf-8")

DEFAULT = re.search(
    r"defaultRefreshInterval\s*:\s*TimeInterval\s*=\s*([0-9.]+)", PREFS
)
CAPACITY = re.search(
    r"historyCapacity\s*=\s*Int\(\s*([\d.]+)\s*/\s*"
    r"MenubarMetricPrefs\.defaultRefreshInterval\s*\)",
    STORE,
)


def test_the_fallback_constant_is_a_half_second():
    assert DEFAULT, "MenubarMetricPrefs must declare defaultRefreshInterval"
    assert DEFAULT.group(1) == "0.5", (
        "the menubar items are opt-in: the cadence they fall back to is the "
        f"fastest choice in the picker, not {DEFAULT.group(1)}s"
    )
    # The poller reads through this getter, so its fallback has to be the
    # constant — a second literal beside it would drift apart unnoticed.
    getter = re.search(r"static var refreshInterval[\s\S]*?\n    \}", PREFS)
    assert getter, "MenubarMetricPrefs.refreshInterval is the poller's read path"
    assert "return defaultRefreshInterval" in getter.group(0), (
        "an absent or out-of-set pref must fall back to the shared constant"
    )
    assert "?? 1.0" not in getter.group(0), (
        "the old one-second fallback is back"
    )


def test_the_appearance_picker_shows_that_same_default():
    match = re.search(
        r"@AppStorage\(MenubarMetricPrefs\.refreshIntervalKey\)\s+"
        r"private var refreshInterval = (\S+)",
        SCREEN,
    )
    assert match, "AppearanceScreen must default its stored refresh interval"
    assert match.group(1) == "MenubarMetricPrefs.defaultRefreshInterval", (
        "the picker must open on the constant the poller falls back to, not a "
        f"literal ({match.group(1)})"
    )


def test_the_graph_window_is_still_a_minute_at_the_default():
    assert DEFAULT and CAPACITY, (
        "the capacity must be derived from the shared default interval, not hand-written"
    )
    # Re-evaluate the derivation the way the store does, then measure the window.
    capacity = int(float(CAPACITY.group(1)) / float(DEFAULT.group(1)))
    window = capacity * float(DEFAULT.group(1))
    assert window == 60, (
        f"the activity graphs promise a minute, not {window:g}s: "
        f"{capacity} samples × {DEFAULT.group(1)}s"
    )
