# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the navbar "copy version" button (issue #3478).

Static-template assertions (no browser render, no server), matching the pattern
already used by ``tests/test_admin_enhanced_readability.py`` for the navbar.

The behaviours pinned here:

1. The button exists and reuses the shared ``copyToClipboard()`` helper from
   ``dashboard.js`` -- the navbar must not grow a second clipboard
   implementation (``navigator.clipboard`` / ``execCommand``).
2. The button is a *sibling* of the version link, never a descendant, so
   clicking it cannot also navigate to GitHub.
3. The pre-existing update-availability hover card stays inside the version
   link (still anchored to the version text) and the button carries its own
   ``nav-tooltip`` -- the two affordances must not overlap.
4. Exactly one copy button in the navbar (no duplicated control).
5. The ``navbar.copy_version`` i18n key exists in every locale.
"""

import json
from html.parser import HTMLParser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "omlx/admin/templates"
I18N = ROOT / "omlx/admin/i18n"

NAVBAR = (TEMPLATES / "dashboard/_navbar.html").read_text(encoding="utf-8")
DASHBOARD_JS = (ROOT / "omlx/admin/static/js/dashboard.js").read_text(encoding="utf-8")

COPY_VERSION_I18N_KEY = "navbar.copy_version"

# Start of the version <a> (the one that links to GitHub / the release page).
_VERSION_ANCHOR_START = NAVBAR.index('@mouseenter="versionHover = true"')
# Its closing tag -- everything up to here belongs to the version link.
_VERSION_ANCHOR_END = NAVBAR.index("</a>", _VERSION_ANCHOR_START)


def _copy_button_offset() -> int:
    return NAVBAR.index("copyToClipboard(")


# --- Minimal stdlib DOM, used to prove *structural* claims (index arithmetic on
# raw HTML only proves ordering, not nesting). No third-party parser on purpose:
# BeautifulSoup/lxml are not declared test dependencies.

_VOID_TAGS = {"img", "br", "hr", "input", "meta", "link", "source", "path", "rect"}


class _Dom(HTMLParser):
    """Collects (tag, attrs, parent_index) for every start tag."""

    def __init__(self, markup: str) -> None:
        super().__init__(convert_charrefs=True)
        self.nodes: list[tuple[str, dict, int | None]] = []
        self._stack: list[int] = []
        self.feed(markup)

    def handle_starttag(self, tag, attrs):
        parent = self._stack[-1] if self._stack else None
        self.nodes.append((tag, {k.lower(): v for k, v in attrs}, parent))
        if tag not in _VOID_TAGS:
            self._stack.append(len(self.nodes) - 1)

    def handle_endtag(self, tag):
        for i in range(len(self._stack) - 1, -1, -1):
            if self.nodes[self._stack[i]][0] == tag:
                del self._stack[i:]
                return

    def chain(self, index: int) -> list[str]:
        """Tags from ``index`` up to the root, inclusive of ``index``."""
        out: list[str] = []
        while index is not None:
            out.append(self.nodes[index][0])
            index = self.nodes[index][2]
        return out

    def is_child_of(self, index: int, ancestor_index: int) -> bool:
        """Identity check on the immediate parent.

        ``chain()`` compares tag *names*, which is useless inside the navbar:
        every button there is ``button/div/div/div/div/nav``, so all of them
        match. Node identity is what actually distinguishes them.
        """
        return self.nodes[index][2] == ancestor_index

    def is_descendant_of(self, index: int, ancestor_index: int) -> bool:
        node = index
        while node is not None:
            node = self.nodes[node][2]
            if node == ancestor_index:
                return True
        return False

    def same_parent(self, a: int, b: int) -> bool:
        return self.nodes[a][2] == self.nodes[b][2]

    def find(self, pred):
        return [i for i, (tag, attrs, _) in enumerate(self.nodes) if pred(tag, attrs)]


def _dom():
    return _Dom(NAVBAR)


def _copy_button_index(dom) -> int:
    hits = dom.find(
        lambda tag, attrs: tag == "button"
        and "copyToClipboard(" in (attrs.get("@click.stop") or "")
    )
    assert len(hits) == 1, f"expected exactly one copy button, found {len(hits)}"
    return hits[0]


def _version_link_index(dom) -> int:
    hits = dom.find(
        lambda tag, attrs: tag == "a"
        and "versionHover" in (attrs.get("@mouseenter") or "")
    )
    assert len(hits) == 1, f"expected exactly one version link, found {len(hits)}"
    return hits[0]


def test_navbar_template_compiles():
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True)
    env.get_template("dashboard/_navbar.html")


def test_navbar_exposes_copy_version_button():
    assert "copyToClipboard(" in NAVBAR, "navbar has no copy affordance"
    # Copies the exact string rendered next to it, so the clipboard matches the
    # visible label ("v" prefix included).
    assert "copyToClipboard('v' + '{{ version }}')" in NAVBAR
    # The button is keyboard reachable and labelled for screen readers.
    assert 'type="button"' in NAVBAR
    assert f":aria-label=\"window.t('{COPY_VERSION_I18N_KEY}')\"" in NAVBAR
    # Transient "copied" confirmation, same 2s reset as the other copy buttons.
    assert 'x-data="{ versionCopied: false }"' in NAVBAR
    assert (
        "versionCopied = true; setTimeout(() => versionCopied = false, 2000)" in NAVBAR
    )
    # navbar icon-button convention (see the Customize Dashboard button).
    assert "nav-tooltip" in NAVBAR
    assert f":data-tooltip=\"window.t('{COPY_VERSION_I18N_KEY}')\"" in NAVBAR


def test_copy_version_button_reuses_shared_clipboard_helper():
    # The console already ships copyToClipboard() with a execCommand fallback;
    # the navbar must call it instead of re-implementing clipboard access.
    assert "copyToClipboard(text)" in DASHBOARD_JS
    # Nothing clipboard-related may be added to the navbar template itself.
    assert "navigator.clipboard" not in NAVBAR
    assert "execCommand" not in NAVBAR


def test_copy_version_button_sits_outside_the_version_link():
    """A button nested in the version <a> would also navigate to GitHub."""
    assert _copy_button_offset() > _VERSION_ANCHOR_END, (
        "copy button must be a sibling of the version link, not a descendant"
    )


def test_copy_version_button_does_not_collide_with_update_hover_card():
    # The pre-existing hover card stays inside the version link so it remains
    # anchored to the version text after the button is added next to it.
    assert NAVBAR.index('x-show="versionHover"') < _VERSION_ANCHOR_END
    assert NAVBAR.index("navbar.update_available") < _VERSION_ANCHOR_END
    assert NAVBAR.index("navbar.go_to_github") < _VERSION_ANCHOR_END


def test_navbar_has_exactly_one_copy_version_button():
    assert NAVBAR.count("copyToClipboard(") == 1
    assert NAVBAR.count(f"window.t('{COPY_VERSION_I18N_KEY}')") == 2  # tooltip + aria-label


def test_copy_version_i18n_key_exists_in_every_locale():
    i18n_dir = I18N
    for locale_path in sorted(i18n_dir.glob("*.json")):
        catalog = json.loads(locale_path.read_text(encoding="utf-8"))
        assert COPY_VERSION_I18N_KEY in catalog, f"{locale_path.name} is missing the key"
        label = catalog[COPY_VERSION_I18N_KEY]
        assert label.strip(), f"{locale_path.name} has an empty label"


# --- Structural guarantees (parsed DOM, not index arithmetic) ----------------


def test_copy_button_is_a_sibling_of_the_version_link_not_a_descendant():
    """A button inside the version <a> would navigate to GitHub when clicked."""
    dom = _dom()
    button, link = _copy_button_index(dom), _version_link_index(dom)

    assert not dom.is_descendant_of(button, link), (
        "copy button is nested inside the version link and would navigate to GitHub"
    )
    assert dom.same_parent(button, link), (
        "copy button and version link must share the same parent element"
    )


def test_update_hover_card_stays_anchored_inside_the_version_link():
    """The card is absolutely positioned against the link; the button must not
    become its containing block."""
    dom = _dom()
    link = _version_link_index(dom)
    cards = dom.find(lambda tag, attrs: tag == "span" and attrs.get("x-show") == "versionHover")

    assert len(cards) == 1, f"expected one update hover card, found {len(cards)}"
    assert dom.is_child_of(cards[0], link), "hover card left the version link"


def test_copy_button_has_no_hover_handlers_so_it_cannot_trigger_the_update_card():
    dom = _dom()
    button = _copy_button_index(dom)
    _, attrs, _ = dom.nodes[button]

    assert "@mouseenter" not in attrs and "@mouseleave" not in attrs


def test_copy_button_renders_the_copy_icon_before_any_click():
    """Counter-example guard: the button must not boot already showing 'copied'."""
    dom = _dom()
    button = _copy_button_index(dom)
    icons = {
        dom.nodes[i][1]["data-lucide"]: dom.nodes[i][1]
        for i in dom.find(lambda tag, attrs: tag == "i" and "data-lucide" in attrs)
        if dom.is_child_of(i, button)
    }

    assert set(icons) == {"copy", "check"}, f"unexpected icons on the copy button: {icons}"
    # 'copy' is the resting state and must be visible on first paint.
    assert icons["copy"].get("x-show") == "!versionCopied"
    assert "x-cloak" not in icons["copy"]
    # 'check' is the confirmation state and must stay hidden until copied.
    assert icons["check"].get("x-show") == "versionCopied"
    assert "x-cloak" in icons["check"]
