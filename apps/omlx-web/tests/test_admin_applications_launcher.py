# SPDX-License-Identifier: Apache-2.0
"""The Applications card is a launcher: nine icon tiles, one revealed panel.

The card used to list every tool's launch command at once. It now paints the
tiles only, and a tile click reveals that tool's command, so these tests pin the
structure that makes that true: every tile owns a panel, every panel is gated by
its own tool name, every command binding sits inside a panel, and the panel
wrapper is hidden until something is picked.
"""

import json
from html.parser import HTMLParser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).parents[1]
TEMPLATES = ROOT / "omlx_web/templates"
EN = json.loads((ROOT / "omlx_web/i18n/en.json").read_text(encoding="utf-8"))

# The launcher's tools and the command binding each one reveals.
TOOL_COMMANDS = {
    "claude": ["claudeCommand"],
    "codex": ["codexCommand", "codexAppCommand"],
    "dsh": ["dshCommand"],
    "opencode": ["opencodeCommand"],
    "openclaw": ["openclawCommand"],
    "hermes": ["hermesCommand"],
    "pi": ["piCommand"],
    "vibe": ["vibeCommand"],
    "copilot": ["copilotCommand"],
}

VOID_TAGS = {
    "area",
    "base",
    "br",
    "circle",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "line",
    "link",
    "meta",
    "path",
    "polygon",
    "polyline",
    "rect",
    "source",
    "track",
    "use",
    "wbr",
}


class Node:
    def __init__(self, tag, attrs, parent):
        self.tag = tag
        self.attrs = dict(attrs)
        self.parent = parent
        self.children = []

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def ancestors(self):
        node = self.parent
        while node is not None:
            yield node
            node = node.parent


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#document", {}, None)
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        node = Node(tag, attrs, self.stack[-1])
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return


def render():
    """Render the partial the way the dashboard does, against the en catalog.

    Returns the HTML and every translation key the partial asked for but the
    catalog does not define, so a typo in a key fails here rather than shipping
    an empty string.
    """
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    missing = []
    env.globals.update(t=lambda key: EN.get(key) or missing.append(key) or key)
    html = env.get_template("dashboard/blocks/_applications.html").render()
    return html, missing


def tree():
    parser = Tree()
    parser.feed(render()[0])
    return parser.root


def nodes(root, **attrs):
    return [
        node
        for node in root.walk()
        if all(node.attrs.get(name) == value for name, value in attrs.items())
    ]


def tiles(root):
    return [node for node in nodes(root) if "data-tool" in node.attrs]


def panels(root):
    return {
        node.attrs["data-tool-panel"]: node
        for node in nodes(root)
        if "data-tool-panel" in node.attrs
    }


def command_lines(root):
    """Every ``x-text`` launch-command binding in the rendered card."""
    return [
        node
        for node in nodes(root)
        if node.tag == "code" and node.attrs.get("x-text", "").endswith("Command")
    ]


def test_every_tool_has_one_tile_and_one_panel():
    card = tree()
    assert sorted(tile.attrs["data-tool"] for tile in tiles(card)) == sorted(
        TOOL_COMMANDS
    )
    assert sorted(panels(card)) == sorted(TOOL_COMMANDS)
    assert len(nodes(card, role="group")) == 1


def test_tiles_are_buttons_that_own_their_panel():
    card = tree()
    panel_nodes = [node for node in nodes(card) if "data-tool-panel" in node.attrs]
    panel_ids = [node.attrs["id"] for node in panel_nodes]
    assert len(panel_ids) == len(set(panel_ids)), "panel ids must be unique"
    panel_by_id = {node.attrs["id"]: node for node in panel_nodes}

    for tile in tiles(card):
        tool = tile.attrs["data-tool"]
        assert tile.tag == "button"
        assert tile.attrs["type"] == "button"
        controls = tile.attrs["aria-controls"]
        assert controls in panel_by_id, f"{tool} points at a missing panel"
        assert panel_by_id[controls].attrs["data-tool-panel"] == tool
        # A tile discloses its panel and reports the expanded state to
        # assistive tech.
        assert tool in tile.attrs["@click"]
        assert ":aria-expanded" in tile.attrs


def test_panels_are_gated_by_their_own_tool():
    card = tree()
    for tool, panel in panels(card).items():
        assert panel.attrs["x-show"] == f"tool === '{tool}'"


def test_launch_commands_only_render_inside_their_panel():
    card = tree()
    bindings = {}
    for line in command_lines(card):
        panel = next(
            (
                ancestor.attrs["data-tool-panel"]
                for ancestor in line.ancestors()
                if "data-tool-panel" in ancestor.attrs
            ),
            None,
        )
        bindings.setdefault(line.attrs["x-text"], []).append(panel)

    assert sorted(bindings) == sorted(
        binding for values in TOOL_COMMANDS.values() for binding in values
    )
    for tool, expected in TOOL_COMMANDS.items():
        for binding in expected:
            rendered_in = bindings[binding]
            assert rendered_in == [tool], f"{binding} must render in {tool}'s panel"


def test_every_command_line_has_one_copy_button_for_that_command():
    card = tree()
    for line in command_lines(card):
        bar = line.parent
        buttons = [child for child in bar.children if child.tag == "button"]
        assert (
            len(buttons) == 1
        ), f"{line.attrs['x-text']} needs exactly one copy button"
        assert line.attrs["x-text"] in buttons[0].attrs["@click"]


def test_card_starts_collapsed_and_keeps_tool_state_locally():
    card = tree()
    assert len(nodes(card, **{"x-data": "{ tool: null }"})) == 1

    wrapper = [
        node for node in nodes(card, **{"x-show": "tool"}) if "x-cloak" in node.attrs
    ]
    assert len(wrapper) == 1
    assert sorted(panels(wrapper[0])) == sorted(TOOL_COMMANDS)

    # Nothing outside the collapsed wrapper may bind a launch command.
    for node in nodes(card):
        binding = node.attrs.get("x-text", "")
        if binding.endswith("Command"):
            assert wrapper[0] in list(node.ancestors())


def test_hint_sits_after_the_panel_so_opening_does_not_shift_it():
    card = tree()
    hints = [node for node in nodes(card) if node.attrs.get("x-show") == "!tool"]
    assert len(hints) == 1
    assert hints[0].tag == "p"

    wrapper = nodes(card, **{"x-show": "tool"})[0]
    assert hints[0].parent is wrapper.parent
    siblings = hints[0].parent.children
    assert siblings.index(hints[0]) > siblings.index(wrapper)


def test_every_string_the_card_renders_has_an_english_string():
    _html, missing = render()
    assert missing == []

    # The two strings this card added carry a key the catalogs can fill.
    assert EN["status.integrations.pick_tool"]
    assert EN["status.integrations.tools_profile"]


def test_openclaw_panel_keeps_the_tools_profile_control():
    card = tree()
    selects = nodes(card, id="openclaw-tools-profile")
    assert len(selects) == 1
    select = selects[0]
    assert select.tag == "select"
    assert select.attrs["x-model"] == (
        "globalSettings.integrations.openclaw_tools_profile"
    )
    assert select.attrs["@change"] == "saveIntegrationSettings()"
    values = [
        option.attrs["value"] for option in select.walk() if option.tag == "option"
    ]
    assert values == ["minimal", "coding", "messaging", "full"]

    labels = [node for node in nodes(card, **{"for": "openclaw-tools-profile"})]
    assert len(labels) == 1
