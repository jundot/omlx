import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
BASE = (ROOT / "omlx_web/templates/base.html").read_text(encoding="utf-8")
LOGIN = (ROOT / "omlx_web/templates/login.html").read_text(encoding="utf-8")
THEME = (ROOT / "omlx_web/static/css/theme.css").read_text(encoding="utf-8")
TAILWIND = (ROOT / "omlx_web/static/css/tailwind.css").read_text(encoding="utf-8")
PAGES = sorted((ROOT / "omlx_web/templates").rglob("*.html")) + sorted(
    (ROOT / "omlx_web/static/js").glob("*.js")
)


def _palette_hex(name: str) -> str:
    channels = re.search(rf"--palette-{re.escape(name)}:\s*(\d+) (\d+) (\d+);", THEME)
    assert channels is not None
    return "#" + "".join(f"{int(value):02x}" for value in channels.groups())


def _css_color(stylesheet: str, selector: str, property_name: str) -> str:
    rule = re.search(rf"{selector}\s*\{{([^}}]*)\}}", stylesheet, re.DOTALL)
    assert rule is not None
    color = re.search(
        rf"{re.escape(property_name)}:\s*rgb\(var\(--palette-([a-z0-9-]+)\)\)",
        rule.group(1),
    )
    assert color is not None
    return _palette_hex(color.group(1))


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
    light_ring = _css_color(THEME, r":root", "--focus-ring-color")
    dark_ring = _css_color(THEME, r'\[data-theme="dark"\]', "--focus-ring-color")
    dark_page = _css_color(LOGIN, r'\[data-theme="dark"\] body', "background-color")
    dark_control = _css_color(
        LOGIN, r'\[data-theme="dark"\] \.bg-neutral-50', "background-color"
    )

    assert _contrast_ratio(light_ring, "#ffffff") >= 3
    assert _contrast_ratio(dark_ring, dark_page) >= 3
    assert _contrast_ratio(dark_ring, dark_control) >= 3


# docs/web-ui-design.md spells out the ramp: caption 10, xs 12, sm 14, base/lg
# 16/18, xl/2xl 20/24, and it names one arbitrary value — the serving stat
# figures at `lg:text-[32px]`. Everything else must name a step.
ALLOWED_ARBITRARY_SIZES = frozenset({"32px"})
ARBITRARY_SIZE = re.compile(r"text-\[(\d+(?:\.\d+)?px)\]")


def test_no_font_size_outside_the_ramp():
    stray = []
    for page in PAGES:
        source = page.read_text(encoding="utf-8")
        for match in ARBITRARY_SIZE.finditer(source):
            if match.group(1) not in ALLOWED_ARBITRARY_SIZES:
                stray.append(f"{page.relative_to(ROOT)}: text-[{match.group(1)}]")
    assert not stray, "name a step of the ramp instead:\n" + "\n".join(stray)


def test_the_step_below_xs_reaches_the_compiled_stylesheet():
    # `caption` comes from theme.extend.fontSize, so a config change that drops
    # it would silently leave the 10px labels at the browser default.
    assert ".text-caption{font-size:10px}" in TAILWIND
