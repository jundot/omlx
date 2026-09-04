# SPDX-License-Identifier: Apache-2.0
"""Regression contract for browser-to-inference stream cancellation."""

from pathlib import Path

CHAT_TEMPLATE = (
    Path(__file__).parents[1] / "omlx" / "admin" / "templates" / "chat.html"
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
