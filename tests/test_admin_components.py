# SPDX-License-Identifier: Apache-2.0
"""Component-spec contract for the admin console.

`templates/components/ui.html` holds the six shared specs (Button, Badge, Card,
FormRow, Segmented, EmptyState) as Jinja macros; `static/css/components.css`
holds their styling. These tests render the macros and check that every class
they emit is a real component class, so a macro cannot invent markup the
stylesheet does not define.
"""

import html as html_module
import re
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "omlx" / "admin" / "templates"
COMPONENTS_CSS = (ROOT / "omlx" / "admin" / "static" / "css" / "components.css").read_text(
    encoding="utf-8"
)
BASE = (TEMPLATES / "base.html").read_text(encoding="utf-8")

DECLARED_CLASSES = set(re.findall(r"\.([a-z][a-z0-9_-]*)", COMPONENTS_CSS))
# Tailwind utilities the macros use for icon sizing and layout.
UTILITY_PREFIXES = ("w-", "h-", "flex", "items-", "gap-", "justify-")


def _ui():
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    return env.get_template("components/ui.html").module


def _classes(html: str) -> set:
    found = set()
    for value in re.findall(r'(?<![:\w-])class="([^"]*)"', html):
        for name in value.split():
            if name.startswith("{{") or "{" in name:
                continue
            found.add(name)
    return found


def test_base_ships_the_component_stylesheet():
    assert "css/components.css" in BASE


@pytest.mark.parametrize("name", ["button", "badge", "card", "form_row", "segmented", "empty_state"])
def test_macro_exists(name):
    assert hasattr(_ui(), name), f"components/ui.html has no {name} macro"


def test_button_variants_are_declared():
    html = "".join(
        _ui().button("Save", variant=variant, icon="save")
        for variant in ("primary", "secondary", "ghost", "destructive")
    )
    assert "btn--primary" in html and "btn--destructive" in html
    assert "btn--secondary" in html and "btn--ghost" in html
    assert "btn btn--primary" in _ui().button("Save", variant="primary")


def test_button_passes_alpine_expressions_through():
    html = _ui().button("Unload", variant="destructive", click="unload()", disabled="busy", show="loaded")
    assert '@click="unload()"' in html
    assert ':disabled="busy"' in html
    assert 'x-show="loaded"' in html


def test_badge_tones_map_to_system_colours():
    for tone in ("green", "orange", "red", "blue", "neutral"):
        html = _ui().badge("Running", tone=tone, dot=True)
        assert f"badge--{tone}" in html
    assert "badge__dot" in _ui().badge("Live", tone="green", dot=True)


def test_card_renders_header_and_body():
    template = Environment(
        loader=FileSystemLoader(str(TEMPLATES)), autoescape=True
    ).from_string(
        '{% import "components/ui.html" as ui %}'
        "{% call ui.card(title='Models', subtitle='2 loaded') %}BODY{% endcall %}"
    )
    html = template.render()
    assert "card__header" in html and "card__title" in html
    assert "card__subtitle" in html
    assert "BODY" in html


def test_form_row_puts_the_control_on_the_right():
    template = Environment(
        loader=FileSystemLoader(str(TEMPLATES)), autoescape=True
    ).from_string(
        '{% import "components/ui.html" as ui %}'
        "{% call ui.form_row(label='Temperature', description='Higher is more random') %}"
        "<input>{% endcall %}"
    )
    html = template.render()
    assert "form-row__label" in html and "form-row__desc" in html
    assert "form-row__control" in html
    assert "<input>" in html


def test_segmented_marks_the_active_item():
    html = html_module.unescape(
        str(
            _ui().segmented(
                [("overview", "Overview"), ("runtime", "Runtime")],
                active="statusTab",
                setter="statusTab = '{value}'",
            )
        )
    )
    assert html.count('class="segmented__item"') == 2
    assert "segmented__item--active" in html
    assert "statusTab === 'overview'" in html
    assert "statusTab = 'runtime'" in html


def test_card_header_puts_the_icon_on_the_title_line():
    """A heading with a subtitle is two lines tall, and the icon belongs to the
    title's line: centred on the block it hangs between the title and the
    subtitle, belonging to neither."""
    template = Environment(
        loader=FileSystemLoader(str(TEMPLATES)), autoescape=True
    ).from_string(
        '{% import "components/ui.html" as ui %}'
        "{% call ui.card(icon='file-text', title='MarkItDown', subtitle='Attachments') %}"
        "BODY{% endcall %}"
    )
    assert 'class="card__icon w-4 h-4"' in template.render()

    heading = COMPONENTS_CSS[COMPONENTS_CSS.index(".card__heading {"):]
    heading = heading[: heading.index("}")]
    assert "align-items: flex-start" in heading, "the row centres, the icon does not"

    icon = COMPONENTS_CSS[COMPONENTS_CSS.index(".card__icon {"):]
    icon = icon[: icon.index("}")]
    assert "margin-top: calc(" in icon, "the icon is centred on the title's line box"
    assert "--lh-emphasis" in icon and "--fs-emphasis" in icon, "from the type tokens"


# A macro call whose arguments are read as one string, so a setter cannot hide
# behind a nested call or a tuple.
_SEGMENTED_CALL = re.compile(r"ui\.segmented(?:_each)?\(")


def _call_arguments(source: str, open_paren: int) -> str:
    depth = 0
    for index in range(open_paren, len(source)):
        char = source[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return source[open_paren + 1:index]
    raise AssertionError("the segmented call never closes")


def _top_level_arguments(arguments: str) -> list:
    parts, depth, current = [], 0, []
    for char in arguments:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part.strip() for part in parts if part.strip()]


def test_every_segmented_control_sets_the_value_it_shows():
    """A segmented control without a setter renders buttons that do nothing:
    the Stats scope toggle shipped as `ui.segmented(items, 'statsScope')` and
    the tab looked dead."""
    for path in sorted(TEMPLATES.rglob("*.html")):
        source = path.read_text(encoding="utf-8")
        for match in _SEGMENTED_CALL.finditer(source):
            arguments = _top_level_arguments(_call_arguments(source, match.end() - 1))
            named = any(part.startswith("setter=") for part in arguments)
            positional = len(arguments) >= 3 and not any("=" in part for part in arguments[:3])
            assert named or positional, f"{path.name}: {match.group(0)}{arguments} has no setter"


def test_empty_state_carries_icon_title_and_action():
    template = Environment(
        loader=FileSystemLoader(str(TEMPLATES)), autoescape=True
    ).from_string(
        '{% import "components/ui.html" as ui %}'
        "{% call ui.empty_state('inbox', 'Nothing yet', 'Downloads appear here') %}"
        "<button>Browse</button>{% endcall %}"
    )
    html = template.render()
    assert "empty-state__icon" in html
    assert "empty-state__title" in html
    assert "empty-state__desc" in html
    assert "<button>Browse</button>" in html


def test_the_switch_is_one_size_and_the_knob_travels_across_it():
    """The switch is a control with a length and a height, not a capsule: the
    track and the knob take the control corner (the knob one hairline tighter,
    concentric), the knob lands one hairline from the far edge, and there is
    exactly one size — the second one (`--sm`, 36x16) made the same control a
    different control depending on where it stood, and shorter than the badge on
    its own row (19px)."""
    def block(selector):
        assert f"{selector} {{" in COMPONENTS_CSS, f"{selector} has no spec"
        body = COMPONENTS_CSS[COMPONENTS_CSS.index(f"{selector} {{"):]
        return body[: body.index("}")]

    track = block(".switch")
    # 40x20, out of the spacing scale so both sizes it replaced are gone.
    assert "width: calc(var(--space-8) - var(--space-2))" in track
    assert "height: calc(var(--space-5) - var(--space-1))" in track
    assert "border-radius: var(--radius-sm)" in track
    assert "--radius-pill" not in track, "a capsule is what the switch was"

    knob = block(".switch__knob")
    assert "width: var(--space-4)" in knob and "height: var(--space-4)" in knob
    assert "border-radius: calc(var(--radius-sm) - 1px)" in knob
    assert "--radius-pill" not in knob

    # 40 - 2 borders - 16 knob - 2 one-hairline insets = 20.
    travel = block(".switch--on .switch__knob")
    assert "translateX(calc(var(--space-5) - var(--space-1)))" in travel

    # …and the badge beside it is the same box. A status badge is a dot, a gap
    # and two 12px characters: 4 + 4 + 24 plus one --space-1 of padding on each
    # side is 40 — the switch's width — and the badge's height is its height.
    badge = block(".badge")
    assert "height: calc(var(--space-5) - var(--space-1))" in badge
    assert "padding: 0 var(--space-2)" in badge
    assert "justify-content: center" in badge
    # Two characters are 40 whether or not the label carries its dot: the dot and
    # its gap take the place of the padding they displace.
    dotted = block(".chip:has(.badge__dot)")   # the second selector of the pair
    assert "padding: 0 var(--space-1)" in dotted

    # …and every other label wears the same box: `.chip` is the geometry without
    # a colour, so a category tag and a status chip of one length measure alike.
    assert ".chip,\n.badge {" in COMPONENTS_CSS, (
        "the label box is declared once and both names share it"
    )
    # …and the two boxes are centred on each other, not on their own baselines:
    # both are inline-level, so without one shared alignment the switch (a box
    # with no text) and the badge (a box with text) sit at different heights in
    # the same row.
    assert "vertical-align: middle" in badge
    assert "vertical-align: middle" in track

    assert ".switch--sm" not in COMPONENTS_CSS, "one size, not two"
    markup = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "omlx" / "admin" / "templates").rglob("*.html"))
    )
    assert "switch--sm" not in markup, "one size, not two"


# Classes the console builds at render time: the base is written once and the
# variant comes from data, so the full name never appears as a literal.
DYNAMIC_CLASSES = (
    "btn--",      # ui.button(variant=…)
    "badge--",    # ui.badge(tone=…)
    "toast--",    # omlxToast({tone})
    "notice--",   # ui.notice(tone=…)
    "skeleton--", # ui.skeleton_kpis / skeleton_table
)


def test_every_declared_class_is_used_somewhere():
    """Dead CSS is how a component library rots: the console declared
    `.surface-card`, `.surface-group`, `.card-stack`, `.hairline` and
    `.skeleton-table__head` and no template, script or page ever named them."""
    # Everything that can name a class *except* components.css itself, which by
    # definition contains every name it declares.
    sources = [path for path in TEMPLATES.rglob("*.html")]
    sources += [path for path in (ROOT / "omlx" / "admin" / "static" / "js").rglob("*.js")]
    sources += [
        path
        for path in (ROOT / "omlx" / "admin" / "static" / "css").rglob("*.css")
        if path.name not in ("tailwind.css", "components.css", "tokens.css")
    ]
    haystack = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    # The chat page's drawer and shortcut sheet land in the next slice: the
    # classes are declared by the layer and rendered by that page.
    pending = {
        "drawer", "drawer--open", "drawer-scrim", "drawer-scrim--closed",
    }
    dead = [
        name
        for name in sorted(DECLARED_CLASSES)
        if name not in haystack
        and not name.startswith(DYNAMIC_CLASSES)
        and name not in pending
    ]
    assert not dead, f"classes nothing renders: {dead}"


def test_every_class_the_macros_emit_is_declared():
    ui = _ui()
    template = Environment(
        loader=FileSystemLoader(str(TEMPLATES)), autoescape=True
    ).from_string(
        '{% import "components/ui.html" as ui %}'
        "{{ ui.button('Save', variant='primary', icon='save') }}"
        "{{ ui.button('Unload', variant='destructive') }}"
        "{{ ui.button('', variant='ghost', size='sm') }}"
        "{{ ui.badge('Running', tone='green', dot=True, icon='activity') }}"
        "{{ ui.segmented([('a', 'A'), ('b', 'B')], active='tab', setter=\"tab='{value}'\") }}"
        "{% call ui.card(title='T', subtitle='S', header_actions='X') %}b{% endcall %}"
        "{% call ui.form_row(label='L', description='D', badge='B') %}c{% endcall %}"
        "{% call ui.empty_state('inbox', 'Title', 'Desc') %}a{% endcall %}"
    )
    html = template.render()
    unknown = {
        name
        for name in _classes(html)
        if name not in DECLARED_CLASSES and not name.startswith(UTILITY_PREFIXES)
    }
    assert not unknown, f"classes without a spec in components.css: {sorted(unknown)}"


def test_macros_are_strict_about_missing_text():
    module = _ui()
    # No hidden English fallback: a caller that forgets a label gets an empty
    # element rather than a hardcoded string.
    assert "Save" not in module.button("", variant="primary")


def test_no_label_hand_rolls_the_box():
    """A label is `.badge` (a status tone) or `.chip` (its own colours) — never
    padded and rounded on its own. Hand-rolled ones came in three heights
    (21/23.5/20) and three corners (`rounded-full`, `-md`, `-lg`) for the same
    two-character label. A label is a span; buttons, fields, alerts and tooltips
    are controls and keep their own shape."""
    offenders = []
    for path in sorted((ROOT / "omlx" / "admin" / "templates").rglob("*.html")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "<span" not in line:
                continue
            for value in re.findall(r'class="([^"]*)"', line):
                if "chip" in value or "badge" in value:
                    continue
                if "bg-neutral-800" in value or "shadow-lg" in value:
                    continue  # a tooltip bubble
                small = "text-xs" in value or "text-[12px]" in value
                padded = "py-0.5" in value or "py-1" in value
                if small and padded and "rounded" in value:
                    offenders.append(f"{path.relative_to(ROOT)}:{number} — {value}")
    assert not offenders, "labels that hand-roll the box:\n" + "\n".join(offenders)
