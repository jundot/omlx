# SPDX-License-Identifier: Apache-2.0
"""WCAG contrast audit of the console's token palette.

``omlx/admin/tokens.json`` is the only place the console's colours live, so the
contrast of every text/surface pair the console actually uses can be computed
here instead of eyeballed per screen. ``tests/test_admin_accessibility.py``
keeps the focus-ring part of this honest; this file does the palette.

What is asserted:

* **Text** — every text token against every surface the console puts text on
  (page, group/inset, code block, the two control fills, the six tone badges).
  Body copy is 10-17px in this console, so the 4.5:1 threshold applies to all
  of it; the large-text relaxation (3:1 at 18.66px bold / 24px) never applies
  and is not used to excuse anything.
* **UI edges** — the focus ring against the two surfaces it can land on, and the
  destructive button's fill against the page behind it, at 3:1.
* **The tinted action and the selected row** — the accent fill's label at 4.5:1,
  and the settings rail's selected item: a grey one step deeper than the rail
  itself (``text.primary`` at 10% of itself over ``bgSecondary``), because where
  you are is not an action. The system blue stays the tint for things you can
  act on; the six system colours stay status marks.

Not asserted, deliberately:

* Hairline borders (``borderFaint``/``borderNormal``): measured at 1.23:1 and
  1.48:1 over the light page and 1.61:1 / 2.18:1 over the dark one. Every
  control the console draws carries a text label and a fill of its own, so the
  hairline is decoration rather than the only thing identifying the control.
* Data-visualisation fills: the KPI sparkline (4.02:1 light), the hourly-trend
  bars and the memory watermark's blue track, orange guard mark and 22% blue
  overlap tint. WCAG 1.4.11 does not cover graphical objects, and each of these
  is redundant with a legend entry that spells the number out.
* The selected rail row's fill: it is the row's own surface at 10% of the text
  colour, i.e. a grey one step deeper than the rail, so it measures ~1.2:1
  against the rail. That is the point of it (review 6 asked for a grey, not the
  accent), and the row is not identified by the fill alone -- the label is
  bolder and the row carries the rail's left rule. The label's own contrast is
  asserted above.
* The Apple system colours as marks: over the light page they measure green
  2.22:1, orange 2.20:1, red 3.55:1, blue 4.02:1, teal 2.57:1 and purple 4.13:1,
  and 4.61-8.45:1 over the dark one. They still paint dots, bars and fills; the *labels* on a tone badge
  are the ``badge`` tokens below, which are held to 4.5:1.

Every token this pass changed is pinned by ``test_the_fixed_pairs_used_to_fail``
so a revert cannot quietly come back.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKENS = json.loads(
    (ROOT / "omlx" / "admin" / "tokens.json").read_text(encoding="utf-8")
)

BODY_TEXT_MIN = 4.5
NON_TEXT_MIN = 3.0
# What `color-mix(in srgb, <tone> 14%, transparent)` resolves to over a solid
# surface: the tone at 14% alpha, composited in sRGB.
BADGE_TINT_ALPHA = 0.14
# What `color-mix(in srgb, var(--text-primary) 10%, transparent)` resolves to
# over the rail's own surface: how the settings rail paints a selected section.
SELECTION_TINT_ALPHA = 0.10
TONES = ("green", "orange", "red", "blue", "teal", "purple")
APPEARANCES = ("light", "dark")


def _channels(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _relative_luminance(value: str) -> float:
    linear = []
    for channel in _channels(value):
        scaled = channel / 255
        linear.append(
            scaled / 12.92 if scaled <= 0.04045 else ((scaled + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(foreground: str, background: str) -> float:
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _composite(foreground: str, background: str, alpha: float) -> str:
    """`foreground` at `alpha` over `background`, as color-mix() resolves it."""
    mixed = [
        round(channel * alpha + base * (1 - alpha))
        for channel, base in zip(_channels(foreground), _channels(background))
    ]
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


def _text(appearance: str, name: str) -> str:
    return TOKENS["text"][appearance][name]


def _surface(appearance: str, name: str) -> str:
    return TOKENS["surface"][appearance][name]


def _control(appearance: str, name: str) -> str:
    return TOKENS["control"][appearance][name]


def _badge_tint(appearance: str, tone: str) -> str:
    return _composite(
        TOKENS["semantic"][appearance][tone],
        _surface(appearance, "bgPrimary"),
        BADGE_TINT_ALPHA,
    )


def _selection_tint(appearance: str) -> str:
    """The rail paints `text.primary` at 10% over its own grey surface."""
    return _composite(
        _text(appearance, "primary"),
        _surface(appearance, "bgSecondary"),
        SELECTION_TINT_ALPHA,
    )


def _text_pairs() -> list[tuple[str, str, str]]:
    """(label, foreground, background) for every pair the console renders."""
    pairs: list[tuple[str, str, str]] = []
    for appearance in APPEARANCES:
        for token in ("primary", "secondary", "tertiary", "danger", "link"):
            for surface in ("bgPrimary", "bgSecondary", "bgTertiary"):
                pairs.append(
                    (
                        f"{appearance} text.{token} on {surface}",
                        _text(appearance, token),
                        _surface(appearance, surface),
                    )
                )
        for token in ("primary", "secondary", "tertiary"):
            pairs.append(
                (
                    f"{appearance} text.{token} on control.codeBg",
                    _text(appearance, token),
                    _control(appearance, "codeBg"),
                )
            )
        pairs.append(
            (
                f"{appearance} control.primaryText on control.primary",
                _control(appearance, "primaryText"),
                _control(appearance, "primary"),
            )
        )
        pairs.append(
            (
                f"{appearance} control.dangerText on control.danger",
                _control(appearance, "dangerText"),
                _control(appearance, "danger"),
            )
        )
        pairs.append(
            (
                f"{appearance} accent label on the accent fill",
                _control(appearance, "accentText"),
                _control(appearance, "accent"),
            )
        )
        pairs.append(
            (
                f"{appearance} accent label on its hover fill",
                _control(appearance, "accentText"),
                _control(appearance, "accentHover"),
            )
        )
        pairs.append(
            (
                f"{appearance} label on the selected rail row",
                _text(appearance, "primary"),
                _selection_tint(appearance),
            )
        )
        for tone in TONES:
            pairs.append(
                (
                    f"{appearance} badge.{tone} label on its own tint",
                    TOKENS["badge"][appearance][tone],
                    _badge_tint(appearance, tone),
                )
            )
        pairs.append(
            (
                f"{appearance} neutral badge label on bgTertiary",
                _text(appearance, "secondary"),
                _surface(appearance, "bgTertiary"),
            )
        )
    return pairs


def _non_text_pairs() -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for appearance in APPEARANCES:
        for surface in ("bgPrimary", "bgSecondary"):
            pairs.append(
                (
                    f"{appearance} focus ring on {surface}",
                    _control(appearance, "focusRing"),
                    _surface(appearance, surface),
                )
            )
        pairs.append(
            (
                f"{appearance} destructive fill on bgPrimary",
                _control(appearance, "danger"),
                _surface(appearance, "bgPrimary"),
            )
        )
        pairs.append(
            (
                f"{appearance} accent fill on bgPrimary",
                _control(appearance, "accent"),
                _surface(appearance, "bgPrimary"),
            )
        )
        pairs.append(
            (
                f"{appearance} switch fill on bgPrimary",
                _control(appearance, "switchOn"),
                _surface(appearance, "bgPrimary"),
            )
        )

    return pairs


def _fixed_pairs() -> list[tuple[str, str, str, str]]:
    """(label, failing value, background, current value)."""
    pairs = [
        (
            f"{appearance} text.tertiary on bgTertiary",
            {"light": "#64748b", "dark": "#71717a"}[appearance],
            _surface(appearance, "bgTertiary"),
            _text(appearance, "tertiary"),
        )
        for appearance in APPEARANCES
    ]
    pairs.append(
        (
            "light text.danger on bgTertiary",
            "#ef4444",
            _surface("light", "bgTertiary"),
            _text("light", "danger"),
        )
    )
    for tone in TONES:
        pairs.append(
            (
                f"light badge.{tone} label on its own tint",
                TOKENS["semantic"]["light"][tone],
                _badge_tint("light", tone),
                TOKENS["badge"]["light"][tone],
            )
        )
    for tone in ("red", "blue"):
        pairs.append(
            (
                f"dark badge.{tone} label on its own tint",
                TOKENS["semantic"]["dark"][tone],
                _badge_tint("dark", tone),
                TOKENS["badge"]["dark"][tone],
            )
        )
    for appearance in APPEARANCES:
        pairs.append(
            (
                f"{appearance} destructive button label",
                TOKENS["semantic"][appearance]["red"],
                "#ffffff",
                _control(appearance, "danger"),
            )
        )
    return pairs


TEXT_PAIRS = _text_pairs()
NON_TEXT_PAIRS = _non_text_pairs()
FIXED_PAIRS = _fixed_pairs()
TEXT_IDS = [pair[0] for pair in TEXT_PAIRS]
NON_TEXT_IDS = [pair[0] for pair in NON_TEXT_PAIRS]
FIXED_IDS = [pair[0] for pair in FIXED_PAIRS]


def test_the_pairing_table_covers_the_palette():
    # 5 text tokens x 3 surfaces x 2 themes, plus code (3), the two control
    # fills, one per tone badge (`TONES`) and the neutral badge, per theme.
    assert len(TEXT_PAIRS) == len(APPEARANCES) * (5 * 3 + 3 + 2 + len(TONES) + 1 + 3)
    assert len(NON_TEXT_PAIRS) == len(APPEARANCES) * 5
    assert len(FIXED_PAIRS) == 2 + 1 + len(TONES) + 2 + 2


@pytest.mark.parametrize("label,foreground,background", TEXT_PAIRS, ids=TEXT_IDS)
def test_body_text_reaches_four_and_a_half_to_one(label, foreground, background):
    ratio = _contrast_ratio(foreground, background)
    assert (
        ratio >= BODY_TEXT_MIN
    ), f"{label}: {ratio:.2f}:1 ({foreground} on {background})"


@pytest.mark.parametrize(
    "label,foreground,background", NON_TEXT_PAIRS, ids=NON_TEXT_IDS
)
def test_ui_edges_reach_three_to_one(label, foreground, background):
    ratio = _contrast_ratio(foreground, background)
    assert (
        ratio >= NON_TEXT_MIN
    ), f"{label}: {ratio:.2f}:1 ({foreground} on {background})"


@pytest.mark.parametrize(
    "label,previous,background,current", FIXED_PAIRS, ids=FIXED_IDS
)
def test_the_fixed_pairs_used_to_fail(label, previous, background, current):
    ratio = _contrast_ratio(previous, background)
    assert ratio < BODY_TEXT_MIN, (
        f"{label}: the previous value {previous} already reached {ratio:.2f}:1, "
        "so this entry does not document a fix"
    )
    assert previous != current, f"{label}: the failing value is still in tokens.json"


@pytest.mark.parametrize(
    "label,previous,background,current", FIXED_PAIRS, ids=FIXED_IDS
)
def test_the_replacement_passes(label, previous, background, current):
    ratio = _contrast_ratio(current, background)
    assert ratio >= BODY_TEXT_MIN, f"{label}: {ratio:.2f}:1"


def test_the_destructive_fill_is_not_the_system_red():
    for appearance in APPEARANCES:
        fill = _control(appearance, "danger")
        assert (
            _contrast_ratio(_control(appearance, "dangerText"), fill) >= BODY_TEXT_MIN
        )
        assert fill != TOKENS["semantic"][appearance]["red"], (
            "the system colour stays for dots and marks; the button fill is the "
            "contrast-checked step"
        )
