# SPDX-License-Identifier: Apache-2.0
"""Design-token contract for the admin console and the macOS app.

``omlx/admin/tokens.json`` is the single source for every colour, spacing,
radius and type size; ``static/css/tokens.css`` and
``apps/omlx-mac/Sources/Theme/DesignTokens.swift`` are generated from it and
committed. These are static assertions (no browser, no server): they pin the
generated artifacts to the source, keep the six-level type scale and its floor
honest on both platforms, and keep literal colours/sizes out of the shared
component stylesheet.
"""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADMIN = ROOT / "omlx" / "admin"
TOKENS_PATH = ADMIN / "tokens.json"
TOKENS = json.loads(TOKENS_PATH.read_text(encoding="utf-8"))

CSS = (ADMIN / "static" / "css" / "tokens.css").read_text(encoding="utf-8")
COMPONENTS_CSS = (ADMIN / "static" / "css" / "components.css").read_text(encoding="utf-8")
TAILWIND_CONFIG = (ADMIN / "tailwind.config.js").read_text(encoding="utf-8")
BASE = (ADMIN / "templates" / "base.html").read_text(encoding="utf-8")
DASHBOARD = (ADMIN / "templates" / "dashboard.html").read_text(encoding="utf-8")
SWIFT = (ROOT / "apps" / "omlx-mac" / "Sources" / "Theme" / "DesignTokens.swift").read_text(
    encoding="utf-8"
)
PBXPROJ = (ROOT / "apps" / "omlx-mac" / "oMLX.xcodeproj" / "project.pbxproj").read_text(
    encoding="utf-8"
)

TEMPLATES = sorted((ADMIN / "templates").rglob("*.html"))
# Only the stylesheets the console authors itself; the rest is vendor JavaScript.
OWN_STYLESHEETS = [ADMIN / "static" / "css" / "dashboard.css", ADMIN / "static" / "css" / "components.css"]
OWN_SCRIPTS = [
    ADMIN / "static" / "js" / name
    for name in ("cluster_v2.js", "dashboard.js", "logs.js", "settings_nav.js", "usage.js")
]
SWIFT_SOURCES = sorted((ROOT / "apps" / "omlx-mac" / "Sources").rglob("*.swift"))

SCALE = TOKENS["typography"]["scale"]
FLOOR = TOKENS["typography"]["floor"]
APP_FONT_MODIFIERS = ("omlxText", "omlxMono", "omlxDisplay")
APP_FONT_CALL = re.compile(rf"(?:{'|'.join(APP_FONT_MODIFIERS)})\(\s*(\d+(?:\.\d+)?)")


def _declares(name: str, value: str, stylesheet: str = CSS) -> bool:
    """True when the stylesheet declares `name: value` (values are aligned)."""
    return re.search(rf"{re.escape(name)}:\s*{re.escape(value)};", stylesheet) is not None


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "omlx_build_tokens", ADMIN / "build_tokens.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# === Generated artifacts match tokens.json ===


def test_generated_css_is_current():
    assert _load_generator().render_css(TOKENS) == CSS, (
        "tokens.css is stale; run python omlx/admin/build_tokens.py"
    )


def test_generated_swift_is_current():
    assert _load_generator().render_swift(TOKENS) == SWIFT, (
        "DesignTokens.swift is stale; run python omlx/admin/build_tokens.py"
    )


def test_swift_tokens_are_compiled_by_the_app():
    assert "DesignTokens.swift" in PBXPROJ
    assert PBXPROJ.count("DesignTokens.swift") >= 4, (
        "a new Swift file needs a PBXBuildFile, a PBXFileReference, a group "
        "child and a Sources-phase entry"
    )


# === One six-level type scale ===


def test_scale_has_exactly_six_levels():
    assert list(SCALE) == ["aux", "body", "emphasis", "section", "page", "kpi"]
    assert [step["px"] for step in SCALE.values()] == [12, 14, 16, 18, 24, 32]


def test_css_exposes_the_scale_and_the_floor():
    for name, step in SCALE.items():
        assert _declares(f"--fs-{name}", f"{step['px']}px")
        assert f".text-{name} {{" in CSS
        assert f"var(--fs-{name})" in CSS
    assert _declares("--fs-floor", f"{FLOOR['aux']}px")
    assert _declares("--fs-floor-body", f"{FLOOR['body']}px")
    assert _declares("--fs-floor-enhanced", f"{FLOOR['enhancedReadability']}px")


def test_tailwind_named_sizes_resolve_to_tokens():
    block = re.search(r"fontSize:\s*\{(.*?)\n    \},", TAILWIND_CONFIG, re.DOTALL)
    assert block is not None, "tailwind.config.js must define the type scale"
    sizes = re.findall(r"'?([a-z0-9]+)'?:\s*\[([^\]]+)\]", block.group(1))
    assert sizes, "no font sizes found in the Tailwind theme"
    allowed = {f"var(--fs-{name})" for name in SCALE}
    for name, value in sizes:
        first = value.split(",")[0].strip().strip("'")
        assert first in allowed, f"text-{name} is not a token step: {first}"
    # Every alias maps onto a step of the ladder.
    aliases = TOKENS["typography"]["tailwindAliases"]
    mapping = {name: value.split(",")[0].strip().strip("'") for name, value in sizes}
    for alias, step in aliases.items():
        if alias in mapping:
            assert mapping[alias] == f"var(--fs-{step})"


def test_off_scale_arbitrary_sizes_snap_to_the_ladder():
    # tokens.css pulls every sub-floor size that existed in the markup onto the
    # ladder — the auxiliary step is the floor, so all three land on it and none
    # of them can come back through a copy-pasted class.
    assert re.search(r"\.text-\\\[9px\\\]", CSS)
    assert re.search(r"\.text-\\\[10px\\\]", CSS)
    assert re.search(r"\.text-\\\[11px\\\]", CSS)
    floor = re.search(
        r"\.text-\\\[9px\\\],[\s\S]*?\{[^}]*font-size:\s*var\(--fs-aux\)\s*!important", CSS
    )
    assert floor, "the sub-floor sizes are not lifted to the auxiliary step"


def test_templates_only_use_scale_sizes():
    allowed = {step["px"] for step in SCALE.values()}
    offenders = {}
    for path in TEMPLATES:
        if path.name == "chat.html":
            continue  # the chat page is rebuilt on the ramp in the next slice
        for size in re.findall(r"text-\[(\d+)px\]", path.read_text(encoding="utf-8")):
            if int(size) not in allowed:
                offenders.setdefault(str(path.relative_to(ROOT)), set()).add(size)
    assert not offenders, f"off-scale Tailwind font sizes: {offenders}"


def test_no_text_below_the_auxiliary_floor():
    offenders = {}
    pattern = re.compile(r"font-size:\s*(\d+(?:\.\d+)?)px")
    for path in TEMPLATES + OWN_STYLESHEETS + OWN_SCRIPTS:
        if path.name == "chat.html":
            continue  # the chat page is rebuilt on the ramp in the next slice
        text = path.read_text(encoding="utf-8")
        if path.name == "base.html":
            # The enhanced-readability block names the sizes it lifts; those are
            # selectors, not rendered sizes.
            text = re.sub(r"<style>\s*\[data-enhanced-readability\][\s\S]*?</style>", "", text)
        for value in pattern.findall(text):
            if float(value) < FLOOR["aux"]:
                offenders.setdefault(str(path.relative_to(ROOT)), set()).add(value)
    assert not offenders, f"text below the {FLOOR['aux']}px floor: {offenders}"


def test_app_text_respects_the_floor():
    offenders = {}
    for path in SWIFT_SOURCES:
        for value in APP_FONT_CALL.findall(path.read_text(encoding="utf-8")):
            if float(value) < FLOOR["aux"]:
                offenders.setdefault(str(path.relative_to(ROOT)), set()).add(value)
    assert not offenders, f"app text below the {FLOOR['aux']}pt floor: {offenders}"


def test_enhanced_readability_floor_matches_the_token():
    assert _declares("--fs-floor-enhanced", f"{FLOOR['enhancedReadability']}px")
    assert "font-size: 12px !important" in BASE, (
        "base.html lifts sub-12px text to the enhanced-readability floor"
    )


# === The token layer owns colour ===


def _without_at_rule_conditions(stylesheet: str) -> str:
    """Drop `@media …` preludes, keeping their normal rules.

    A media condition cannot read a `var()`, so a stacking breakpoint has to be
    a literal there; it is generated as `--settings-rail-stack-below` and
    tests/test_admin_settings.py pins the two together. Everything a media
    query *contains* is still checked, because it is reached after the prelude.
    """
    return re.sub(r"@(?:media|container|supports)[^{}]*\{", "{", stylesheet)


def test_components_use_tokens_only():
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", COMPONENTS_CSS), "literal colour in components.css"
    sizes = set(re.findall(r"(\d+(?:\.\d+)?)px", _without_at_rule_conditions(COMPONENTS_CSS)))
    assert sizes <= {"1"}, f"literal sizes in components.css: {sizes}"
    for value in re.findall(r"(?:font-size|padding|margin|gap|border-radius):[^;]+;", COMPONENTS_CSS):
        assert "var(--" in value or value.rstrip(";").endswith(": 0"), (
            f"value not backed by a token: {value}"
        )


def test_every_colour_token_reaches_the_stylesheet():
    """A colour in tokens.json has to become a variable, or a rule that reads it
    paints nothing. Naming the semantic colours one by one in the generator is
    exactly how `teal` and `purple` reached the badge rules — and the app's
    palette — while never reaching tokens.css: `color-mix()` with an undefined
    variable is invalid, so those two tone badges rendered with no background at
    all and only their foreground token made them look deliberate."""
    tokens = json.loads((ADMIN / "tokens.json").read_text(encoding="utf-8"))
    css = (ADMIN / "static" / "css" / "tokens.css").read_text(encoding="utf-8")
    components = (ADMIN / "static" / "css" / "components.css").read_text(encoding="utf-8")

    missing = []
    for appearance in ("light", "dark"):
        for name in tokens["semantic"][appearance]:
            if name.startswith("_"):
                continue
            variable = "--accent" if name == "accent" else f"--sys-{name}"
            if variable not in css:
                missing.append(variable)
        for tone in tokens["badge"][appearance]:
            if not tone.startswith("_") and f"--badge-{tone}-fg" not in css:
                missing.append(f"--badge-{tone}-fg")
    assert not missing, f"tokens.json declares colours tokens.css never does: {sorted(set(missing))}"

    # And the rules that read them exist and read something: every tone the
    # badge spec has is a rule, and every variable inside it is declared.
    for tone in tokens["badge"]["light"]:
        if tone.startswith("_"):
            continue
        rule = f".badge--{tone} {{"
        assert rule in components, f"{rule} is missing from components.css"
        block = components[components.index(rule):]
        block = block[: block.index("}")]
        variables = re.findall(r"var\((--[a-z-]+)\)", block)
        assert variables, f"{rule} reads no token at all"
        for variable in variables:
            assert variable in css, f"the {tone} badge reads {variable}, which tokens.css never declares"


def test_base_loads_the_token_layer_before_page_styles():
    tailwind = BASE.index("css/tailwind.css")
    tokens = BASE.index("css/tokens.css")
    components = BASE.index("css/components.css")
    assert tailwind < tokens < components
    assert "css/tokens.css" in BASE and "css/components.css" in BASE


def test_base_uses_the_system_font_stack():
    assert "/fonts/inter" not in BASE, "no self-hosted webfonts"
    assert "noto-sans" not in BASE, "no self-hosted CJK webfonts"
    assert "BlinkMacSystemFont" in CSS


# === Layout skeleton ===


def test_the_sticky_layers_can_actually_stick():
    """One sub-tab row, and the settings rail, pin below the top bar.

    Both are `position: sticky`, so the page must not clip them: an
    `overflow: hidden` on <main> makes it their scroll container, and neither
    sticks -- which is what left the settings row 46pt lower than the same row
    on Models until the clipping moved onto the decorations.
    """
    blocks = re.findall(r"\.settings-rail \{[^}]*\}", COMPONENTS_CSS)
    assert any(
        "position: sticky" in block and "top: var(--sticky-offset)" in block for block in blocks
    ), "the rail clears the top bar and the sub-tab row"
    tabs = re.findall(r"\.page-tabs \{[^}]*\}", COMPONENTS_CSS)
    assert len(tabs) == 1, "one sub-tab row spec"
    assert "position: sticky" in tabs[0]
    assert "top: var(--topbar-height)" in tabs[0], "the row pins under the top bar"
    assert _declares(
        "--sticky-offset",
        "calc(var(--topbar-height) + var(--sub-tab-height) + var(--sticky-gap))",
    )
    main_tag = DASHBOARD[DASHBOARD.index("<main") : DASHBOARD.index("<main") + 80].split(">")[0]
    assert "overflow-hidden" not in main_tag, (
        "<main> must not clip the sticky layers; the decorations clip themselves"
    )
    for name in ("_settings.html", "_models.html", "_bench.html"):
        text = (ADMIN / "templates" / "dashboard" / name).read_text(encoding="utf-8")
        assert "page-tabs" in text, f"{name} uses the shared sub-tab row"



def test_one_gutter_and_one_shared_measure():
    layout = TOKENS["layout"]
    assert _declares("--gutter", f"{layout['gutter']}px")
    assert _declares("--container-form", f"{layout['formMaxWidth']}px")
    # The console's measure is the dashboard's, so --container-wide starts at
    # the default width and dashboard.js repoints it at the chosen one.
    assert _declares("--container-wide", "var(--measure-default)")
    for name, value in layout["widths"].items():
        expected = "none" if value == 0 else f"{value}px"
        assert _declares(f"--measure-{name}", expected), name
    assert ".page-gutter {" in CSS
    assert ".page-wide {" in CSS
    assert ".page-narrow {" in CSS
    assert ".page-frame {" in CSS
    assert _declares("padding-block", "var(--space-6) var(--space-10)")
    assert f"scrollbar-gutter: {layout['scrollbarGutter']};" in CSS


def test_every_page_uses_the_one_measure():
    """The width control lives on the Status tab but measures every page."""
    assert "page-frame" in DASHBOARD, "one page frame for every tab"
    assert "page-wide" in DASHBOARD, "one shared measure for every tab"
    for name in ("_models.html", "_logs.html", "_cluster_v2.html",
                 "_settings.html", "_bench.html"):
        text = (ADMIN / "templates" / "dashboard" / name).read_text(encoding="utf-8")
        assert "page-narrow" not in text, f"{name} must not cap itself below the measure"
        # The frame and the measure live on the wrapper in dashboard.html; a
        # second measure inside it is how a page grows its own limit again.
        assert "page-wide" not in text, f"{name} must not carry its own measure"
    status = (ADMIN / "templates" / "dashboard" / "_status.html").read_text(encoding="utf-8")
    assert "page-wide" not in status, "the dashboard width control owns this tab"

    layout = (ADMIN / "static" / "js" / "dashboard_layout.js").read_text(encoding="utf-8")
    assert "WIDTH_MEASURES" in layout, "the width ids are the token measures"
    assert "max-w-7xl" not in layout, "no page measure through a Tailwind class"
    dashboard_js = (ADMIN / "static" / "js" / "dashboard.js").read_text(encoding="utf-8")
    assert "setProperty('--container-wide'" in dashboard_js, "the measure is applied on :root"


def test_topbar_is_the_only_translucent_layer():
    assert ".topbar-material {" in CSS
    assert "backdrop-filter: blur(var(--topbar-blur))" in CSS
    navbar = (ADMIN / "templates" / "dashboard" / "_navbar.html").read_text(encoding="utf-8")
    assert "topbar-material" in navbar
    assert "page-gutter" in navbar


def test_every_label_colour_is_a_token():
    """A label's colour is a token, not a Tailwind literal. The literals could not
    be reached by a theme, which is the point of naming them: every family in
    `label` becomes `--label-<family>` (the fill) and `--label-<family>-fg` (the
    step that reads on it), read by one `.chip--<family>` rule."""
    missing = []
    for appearance in ("light", "dark"):
        for name in TOKENS["label"][appearance]:
            if name.startswith("_"):
                continue
            base = name[:-2] if name.endswith("Fg") else name
            suffix = "-fg" if name.endswith("Fg") else ""
            if f"--label-{base}{suffix}" not in CSS:
                missing.append(f"--label-{base}{suffix}")
    assert not missing, f"tokens.json declares label colours tokens.css never does: {sorted(set(missing))}"

    for name in TOKENS["label"]["light"]:
        if name.startswith("_") or name.endswith("Fg"):
            continue
        rule = f".chip--{name} {{"
        assert rule in COMPONENTS_CSS, f"{rule} is missing from components.css"
        block = COMPONENTS_CSS[COMPONENTS_CSS.index(rule):]
        block = block[: block.index("}")]
        variables = re.findall(r"var\((--[A-Za-z-]+)\)", block)
        assert variables, f"{rule} reads no token at all"
        for variable in variables:
            assert variable in CSS, f"the {name} label reads {variable}, which tokens.css never declares"


def test_no_label_paints_itself_with_a_palette_colour():
    """The label box takes a family token (`.chip--<family>`), never a Tailwind
    colour: a literal cannot follow a theme, and this is the layer the custom
    theme rewrites."""
    pattern = re.compile(
        r"(bg|text)-(emerald|cyan|amber|orange|green|purple|indigo|pink|rose|sky|red|neutral)-\d+"
    )
    offenders = []
    for path in TEMPLATES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for value in re.findall(r'class="([^"]*\bchip\b[^"]*)"', line):
                if pattern.search(value):
                    offenders.append(f"{path.name}:{number} — {value}")
    assert not offenders, "labels painted with palette colours:\n" + "\n".join(offenders)
