import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
BASE = (ROOT / "omlx/admin/templates/base.html").read_text(encoding="utf-8")
LOGIN = (ROOT / "omlx/admin/templates/login.html").read_text(encoding="utf-8")


def _css_color(stylesheet: str, selector: str, property_name: str) -> str:
    rule = re.search(rf"{selector}\s*\{{([^}}]*)\}}", stylesheet, re.DOTALL)
    assert rule is not None
    color = re.search(
        rf"{re.escape(property_name)}:\s*(#[0-9a-fA-F]{{6}})", rule.group(1)
    )
    assert color is not None
    return color.group(1)


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    first_luminance = _relative_luminance(first)
    second_luminance = _relative_luminance(second)
    lighter = max(first_luminance, second_luminance)
    darker = min(first_luminance, second_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def test_focus_ring_uses_theme_aware_two_pixel_outline():
    focus_rule = re.search(r":focus-visible\s*\{([^}]*)\}", BASE, re.DOTALL)
    assert focus_rule is not None
    assert "outline: 2px solid var(--focus-ring-color) !important" in focus_rule.group(
        1
    )
    assert "var(--text-primary" not in focus_rule.group(1)


def test_focus_ring_contrasts_with_login_backgrounds():
    light_ring = _css_color(BASE, r":root", "--focus-ring-color")
    dark_ring = _css_color(BASE, r'\[data-theme="dark"\]', "--focus-ring-color")
    dark_page = _css_color(LOGIN, r'\[data-theme="dark"\] body', "background-color")
    dark_control = _css_color(
        LOGIN, r'\[data-theme="dark"\] \.bg-neutral-50', "background-color"
    )

    assert _contrast_ratio(light_ring, "#ffffff") >= 3
    assert _contrast_ratio(dark_ring, dark_page) >= 3
    assert _contrast_ratio(dark_ring, dark_control) >= 3
