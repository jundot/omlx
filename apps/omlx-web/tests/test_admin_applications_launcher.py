# SPDX-License-Identifier: Apache-2.0
"""Applications card: each launch command renders and copies in its own panel."""

import json
from html.parser import HTMLParser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).parents[1]
EN = json.loads((ROOT / "omlx_web/i18n/en.json").read_text(encoding="utf-8"))

TOOL_COMMANDS = {
    "claude": ["claudeCommand", "claudeCodeCommandDisplay", "claudeDesktopCommand"],
    "codex": ["codexCommand", "codexAppCommand"],
    "dsh": ["dshCommand"],
    "opencode": ["opencodeCommand"],
    "openclaw": ["openclawCommand"],
    "hermes": ["hermesCommand"],
    "pi": ["piCommand"],
    "vibe": ["vibeCommand"],
    "copilot": ["copilotCommand"],
}
DISPLAY_COPIES = {"claudeCodeCommandDisplay": "await claudeCodeCommand()"}


class _Card(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.tiles = []
        self.panels = {}
        self.bars = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-tool" in attrs:
            self.tiles.append(attrs["data-tool"])
        if "data-tool-panel" in attrs:
            self.panels[attrs["data-tool-panel"]] = attrs.get("x-show")
        binding = attrs.get("x-text", "")
        if tag == "code" and binding.endswith(("Command", "Display")):
            panel = next(
                (a["data-tool-panel"] for _, a in self.stack if "data-tool-panel" in a),
                None,
            )
            self.bars.append([binding, panel, None])
        elif tag == "button" and "copyToClipboard(" in attrs.get("@click", ""):
            self.bars[-1][2] = attrs["@click"]
        if tag != "img":
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        if any(open_tag == tag for open_tag, _ in self.stack):
            while self.stack.pop()[0] != tag:
                pass


def test_each_command_renders_and_copies_in_its_own_panel():
    env = Environment(
        loader=FileSystemLoader(ROOT / "omlx_web/templates"), autoescape=True
    )
    missing = []
    env.globals["t"] = lambda key: EN.get(key) or missing.append(key) or key
    card = _Card()
    card.feed(env.get_template("dashboard/blocks/_applications.html").render())

    assert missing == []
    assert sorted(card.tiles) == sorted(TOOL_COMMANDS)
    assert card.panels == {tool: f"tool === '{tool}'" for tool in TOOL_COMMANDS}
    assert sorted((binding, panel) for binding, panel, _ in card.bars) == sorted(
        (binding, tool)
        for tool, bindings in TOOL_COMMANDS.items()
        for binding in bindings
    )
    for binding, _, click in card.bars:
        assert f"copyToClipboard({DISPLAY_COPIES.get(binding, binding)})" in click
