# SPDX-License-Identifier: Apache-2.0
"""The admin login page must focus the API key field on load.

Both cards (first-run setup and regular login) render inside Alpine
``<template x-if>``, so a plain ``autofocus`` attribute never applies -
the browser only honours it during initial parsing, before Alpine swaps
the card in. Each API key input therefore needs ``x-init="$el.focus()"``.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGIN_TEMPLATE = ROOT / "omlx_web/templates/login.html"

INPUT_RE = re.compile(
    r"<input\b[^>]*id=\"(?P<id>api_key|setup_api_key)\"[^>]*>", re.DOTALL
)


def _inputs() -> dict:
    html = LOGIN_TEMPLATE.read_text(encoding="utf-8")
    return {m.group("id"): m.group(0) for m in INPUT_RE.finditer(html)}


def test_login_page_has_both_api_key_inputs():
    ids = set(_inputs())
    assert ids == {"api_key", "setup_api_key"}, f"unexpected inputs: {ids}"


def test_api_key_inputs_are_focused_on_render():
    for field_id, tag in _inputs().items():
        assert 'x-init="$el.focus()"' in tag, f"{field_id} is not auto-focused: {tag}"
