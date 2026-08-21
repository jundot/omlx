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


def _switch_button_tags(template: Path) -> list[tuple[int, str]]:
    """Return (line number, opening tag) for every custom toggle switch."""
    source = template.read_text(encoding="utf-8")
    tags = []
    for match in re.finditer(r"<button\b[^>]*>", source):
        tag = match.group(0)
        if "w-11 h-6" not in tag:
            continue
        tags.append((source.count("\n", 0, match.start()) + 1, tag))
    return tags


def test_every_toggle_switch_exposes_state_and_name():
    unlabeled = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        for line, tag in _switch_button_tags(template):
            location = f"{template.relative_to(ROOT)}:{line}"
            if "x-a11y-switch" not in tag:
                unlabeled.append(f"{location} missing x-a11y-switch")
            elif "aria-label" not in tag:
                unlabeled.append(f"{location} missing aria-label")

    assert not unlabeled, "toggle switches without accessible state/name:\n" + "\n".join(
        unlabeled
    )


STATE_ANNOTATIONS = (
    "x-a11y-pressed",
    "aria-pressed",
    "x-a11y-switch",
    "aria-checked",
    "aria-current",
    "aria-expanded",
    "aria-selected",
)

# Conditions that drive styling without representing selection state, so the
# control needs no aria state. Keyed by the condition rather than by line so the
# list survives edits above it.
NON_STATE_CONDITIONS = (
    # Transient "Copied!" feedback on copy-to-clipboard buttons.
    "copied",
    "wiredLimitCopied",
    # Hover-only restyling; these buttons swap their visible label instead.
    "hover",
    # Toggles that rename themselves (title/label) rather than expose pressed.
    "chat.pinned",
    "micActive()",
    "chatSettings.webSearchEnabled",
    # Styling that mirrors an actual `disabled` attribute.
    "promptDirty && activePromptProfile",
    "importingMtplx",
    "aneTuning.running",
    "globalSettings.model.model_dirs.length > 1",
    # Progress/result feedback on a one-shot action button.
    "uploadTokenValidated",
)


def _opening_tags(source: str):
    """Yield (line, tag, tag name) for elements that may be interactive."""
    for match in re.finditer(r"<(button|a|div|label|span)\b", source):
        index, quote = match.end(), None
        while index < len(source):
            char = source[index]
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == ">":
                break
            index += 1
        yield source.count("\n", 0, match.start()) + 1, source[match.start() : index + 1], match.group(1)


def test_selectable_controls_expose_their_state():
    """A control whose :class branches on state must expose that state to AT."""
    unexposed = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = template.read_text(encoding="utf-8")
        for line, tag, name in _opening_tags(source):
            if any(annotation in tag for annotation in STATE_ANNOTATIONS):
                continue
            binding = re.search(r':class="([^"]*)"', tag) or re.search(
                r":class='([^']*)'", tag
            )
            if binding is None or "?" not in binding.group(1):
                continue
            if name not in ("button", "a") and "@click" not in tag:
                continue
            condition = " ".join(binding.group(1).split()).split("?")[0].strip()
            if condition in NON_STATE_CONDITIONS:
                continue
            unexposed.append(
                f"{template.relative_to(ROOT)}:{line} state-styled on `{condition}`"
            )

    assert not unexposed, (
        "controls that restyle on state without exposing it (add an aria state, "
        "or list the condition in NON_STATE_CONDITIONS):\n" + "\n".join(unexposed)
    )


# Pointer-only conveniences: each duplicates an action that keyboard users can
# already reach, so they need no role or tab stop of their own.
POINTER_ONLY_CLICKS = (
    # The chat row is clickable for the mouse; its title is a real button.
    "if (renamingChatId !== chat.id) loadChat(chat.id)",
    # Timeline dots scroll to a message that is reachable by reading the thread.
    "scrollToMessage(dot.index)",
)


def test_clickable_non_button_elements_are_keyboard_operable():
    """@click on a non-interactive element needs a role and a tab stop."""
    unreachable = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = template.read_text(encoding="utf-8")
        for line, tag, name in _opening_tags(source):
            if name in ("button", "a") or "@click" not in tag:
                continue
            # Overlays, dismiss backdrops and stop-propagation wrappers are not
            # controls; they only mirror an action offered elsewhere.
            if "cursor-pointer" not in tag:
                continue
            if 'role="' in tag and "tabindex=" in tag:
                continue
            handler = re.search(r'@click="([^"]*)"', tag)
            if handler and handler.group(1) in POINTER_ONLY_CLICKS:
                continue
            unreachable.append(f"{template.relative_to(ROOT)}:{line} <{name}> @click")

    assert not unreachable, (
        "clickable elements that keyboard users cannot reach:\n"
        + "\n".join(unreachable)
    )
