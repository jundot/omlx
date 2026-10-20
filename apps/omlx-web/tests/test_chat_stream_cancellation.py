# SPDX-License-Identifier: Apache-2.0
"""Regression contract for browser-to-inference stream cancellation."""

from pathlib import Path

CHAT_TEMPLATE = (
    Path(__file__).parents[1] / "omlx_web" / "templates" / "chat.html"
)


def test_stop_cancels_the_open_response_reader_before_aborting_fetch():
    source = CHAT_TEMPLATE.read_text(encoding="utf-8")
    helper_start = source.index("cancelStreamTransport(stream)")
    helper_end = source.index("stopChatStreaming(chatId)", helper_start)
    helper = source[helper_start:helper_end]

    assert "reader.cancel('Stopped by user')" in helper
    assert helper.index("reader.cancel('Stopped by user')") < helper.index(
        "stream.abortController?.abort()"
    )


def test_stream_exposes_its_response_reader_to_every_stop_path():
    source = CHAT_TEMPLATE.read_text(encoding="utf-8")

    assert "stream.responseReader = reader;" in source
    assert "this.cancelStreamTransport(stream);" in source
    assert (
        "this.cancelStreamTransport(this.getStreamSession(chatId, false));" in source
    )


def _stream_error_branch():
    """The ``catch`` of ``streamResponse`` for errors that are not the Stop button.

    A backgrounded WebKit tab kills the fetch mid-stream (``TypeError: Load failed``),
    so this path must keep the tokens that already reached the screen.
    """
    source = CHAT_TEMPLATE.read_text(encoding="utf-8")
    catch_start = source.index("[streamResponse] catch error:")
    catch_end = source.index("} finally {", catch_start)
    catch = source[catch_start:catch_end]
    abort_at = catch.index("if (error.name === 'AbortError')")
    return catch[catch.index("} else {", abort_at) :]


def test_stream_error_keeps_the_text_already_streamed():
    branch = _stream_error_branch()

    assert "stream.streamingContent" in branch
    assert "stream._toolRoundContent" in branch


def test_stream_error_keeps_the_reasoning_already_streamed():
    branch = _stream_error_branch()

    assert "hasVisibleThinking(stream.streamingThinking)" in branch
    assert "this.attachThinkingToAssistantMessage(" in branch


def test_stream_error_still_reports_when_nothing_was_streamed():
    branch = _stream_error_branch()

    assert "Error: ${error.message}" in branch
