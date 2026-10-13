# SPDX-License-Identifier: Apache-2.0
"""Copy-model-name buttons render real icons in model settings and the queue.

The settings copy button once shipped without the ``>`` that closes its
opening tag, so the browser swallowed the first ``<svg>`` icon as an
attribute and the button collapsed to an invisible dot. These tests render
the production templates and parse the DOM: a working copy button must
carry both state icons (copy and check) as direct children.
"""

import json
from html.parser import HTMLParser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "omlx_web" / "templates"
I18N = ROOT / "omlx_web" / "i18n"

_VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "track",
    "wbr",
}


class _ButtonParser(HTMLParser):
    """Collect finished <button> elements with their direct <svg> children."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.buttons = []

    def _open(self, tag, attrs, closed_immediately):
        node = {"tag": tag, "attrs": dict(attrs), "svgs": 0}
        if self.stack and tag == "svg":
            self.stack[-1]["svgs"] += 1
        if not closed_immediately and tag not in _VOID:
            self.stack.append(node)

    def handle_starttag(self, tag, attrs):
        self._open(tag, attrs, closed_immediately=False)

    def handle_startendtag(self, tag, attrs):
        self._open(tag, attrs, closed_immediately=True)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                node = self.stack.pop(index)
                if tag == "button":
                    self.buttons.append(node)
                break


def _render(template_name: str) -> str:
    locale = json.loads((I18N / "en.json").read_text())
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    env.globals.update(
        t=lambda key: locale.get(key, key),
        static=lambda path: f"/admin/static/{path}",
        locale_json=json.dumps(locale),
        current_lang="en",
        version="test",
    )
    return env.get_template(template_name).render()


def _copy_buttons(html: str, title_key: str) -> list:
    parser = _ButtonParser()
    parser.feed(html)
    return [
        button
        for button in parser.buttons
        if any(
            name.endswith("title") and title_key in value
            for name, value in button["attrs"].items()
        )
    ]


def _assert_icon_pair(button):
    # A broken opening tag swallows the first icon as an attribute, leaving
    # one child; a working button keeps both the copy and the check icon.
    assert button["svgs"] == 2, (
        "copy button must contain both state icons as children; "
        "an unclosed opening tag eats the first <svg>"
    )


def test_settings_copy_button_renders_its_icons():
    buttons = _copy_buttons(
        _render("dashboard/_settings.html"),
        "settings.models.table.copy_model_name_tooltip",
    )
    assert len(buttons) == 1
    _assert_icon_pair(buttons[0])


def test_download_queue_copy_buttons_render_their_icons():
    html = _render("dashboard/_models.html")
    buttons = _copy_buttons(html, "models.queue.copy_model_name_tooltip")
    # One button per queue row template: HuggingFace and ModelScope.
    assert len(buttons) == 2
    for button in buttons:
        _assert_icon_pair(button)


def test_download_queue_copy_buttons_copy_the_repo_id_when_completed():
    html = _render("dashboard/_models.html")
    buttons = _copy_buttons(html, "models.queue.copy_model_name_tooltip")
    assert len(buttons) == 2
    for button in buttons:
        attrs = button["attrs"]
        click = next(
            value
            for name, value in attrs.items()
            if name.startswith("@click") or name.startswith("x-on:click")
        )
        assert "copyToClipboard(task.repo_id)" in click
        assert "task.status === 'completed'" in attrs.get("x-show", "")


def test_queue_copy_tooltip_key_exists_in_every_locale():
    key = "models.queue.copy_model_name_tooltip"
    for locale_path in sorted(I18N.glob("*.json")):
        translations = json.loads(locale_path.read_text())
        assert key in translations, f"{locale_path.name} is missing {key!r}"
