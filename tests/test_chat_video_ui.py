# SPDX-License-Identifier: Apache-2.0
"""Regression guards for the chat video UI logic in chat.html.

The chat page is a single Alpine.js template with no JS test runner, so
these tests assert on the template source the same way a reviewer would:
every send path that can fire with a video model selected must branch on
isVideoModel() instead of unconditionally calling streamResponse().

Regression context: after cancelling an in-chat video generation, editing
the original prompt and re-running used to POST /v1/chat/completions and
the server answered 400 "is a video generation model. Use POST /v1/videos."
"""
import re
from pathlib import Path

import pytest

CHAT_TEMPLATE = (
    Path(__file__).parent.parent / "omlx" / "admin" / "templates" / "chat.html"
)


def _function_body(source: str, name: str) -> str:
    """Extract the body of an Alpine method by brace matching."""
    match = re.search(rf"async {name}\(", source)
    assert match, f"{name} not found in chat.html"
    start = source.index("{", match.end())
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


@pytest.fixture(scope="module")
def chat_source() -> str:
    return CHAT_TEMPLATE.read_text(encoding="utf-8")


class TestVideoModelRouting:
    """All send paths must route video models to generateVideo()."""

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_send_path_branches_on_video_model(self, chat_source, fn):
        body = _function_body(chat_source, fn)
        assert "isVideoModel()" in body, (
            f"{fn} must branch on isVideoModel(); a video model on "
            "/v1/chat/completions is rejected by the server with 400"
        )
        assert "generateVideo(" in body, f"{fn} must call generateVideo for video models"

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_stream_response_is_inside_else_branch(self, chat_source, fn):
        """streamResponse() may only appear after the isVideoModel() check."""
        body = _function_body(chat_source, fn)
        stream_pos = body.index("streamResponse()")
        branch_pos = body.index("isVideoModel()")
        assert branch_pos < stream_pos, (
            f"{fn}: streamResponse() runs before the isVideoModel() branch"
        )

    @pytest.mark.parametrize("fn", ["sendMessage", "saveEdit", "regenerateMessage"])
    def test_send_path_debounces_video_submit(self, chat_source, fn):
        """Every send path must respect the videoSubmitting debounce flag."""
        body = _function_body(chat_source, fn)
        assert "videoSubmitting" in body, (
            f"{fn} must check videoSubmitting like sendMessage does"
        )

    def test_regenerate_reuses_last_user_prompt(self, chat_source):
        """regenerateMessage must extract the prompt from message history,
        since unlike sendMessage it has no input box text to use."""
        body = _function_body(chat_source, "regenerateMessage")
        assert "getTextContent(" in body
