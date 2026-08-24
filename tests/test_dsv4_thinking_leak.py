# SPDX-License-Identifier: Apache-2.0
"""Regression tests for DeepSeek V4 thinking leak.

DeepSeek V4 Flash uses the *missing-"-ing"* thinking tags ``\n`` and
``\n`` (confirmed byte-exact against the model tokenizer: ``think_start_id`` /
``think_end_id`` decode to these). The parser accepts both shapes. The leak the
user reports — "thinking content appears inside the chat content after a few
rounds" — happens because, when the model ends its turn without emitting the
closing ``\n`` tag (long multi-turn reasoning, max_tokens truncation, or a tool
call following the reasoning), the ``finish()`` / ``extract_thinking`` recovery
logic re-emits the accumulated thinking as visible content.

A DS4 tool invocation must disable that recovery: the output is a tool call, so
the reasoning has to stay in the thinking channel and never leak into the
visible body.

Tags are written as hex escapes so the on-disk bytes are unambiguous and immune
to any display-layer text mangling:
  open  <thinking> (missing-ing) = \x3c\x74\x68\x69\x6e\x6b\x3e
  close </thinking> (missing-ing) = \x3c\x2f\x74\x68\x69\x6e\x6b\x3e
  DSML tool-call open  = \x3c\xef\xbd\x9c\x44\x53\x4d\x4c\xef\xbd\x9c\x74\x6f\x6f\x6c\x5f\x63\x61\x6c\x6c\x73\x3e
  DSML tool-call close = \x3c\x2f\xef\xbd\x9c\x44\x53\x4d\x4c\xef\xbd\x9c\x74\x6f\x6f\x6c\x5f\x63\x61\x6c\x6c\x73\x3e
"""
from omlx.api.thinking import ThinkingParser, extract_thinking

OPEN = "\x3c\x74\x68\x69\x6e\x6b\x3e"  # <thinking> (missing-ing)
CLOSE = "\x3c\x2f\x74\x68\x69\x6e\x6b\x3e"  # </thinking> (missing-ing)
_DSML_TOKEN = "\uff5cDSML\uff5c"  # U+FF5C fullwidth vertical line (DS4 DSML dialect)
TC_OPEN = "<" + _DSML_TOKEN + "tool_calls>"  # <|DSML|tool_calls|>
TC_CLOSE = "</" + _DSML_TOKEN + "tool_calls>"  # </|DSML|tool_calls|>

TOOL_SUMMARY = (
    "I need to update and list the tasks.\n"
    "Used 3 tools\nUpdated tasks\nUpdated tasks\nListed tasks\n"
)
TOOL_CALL_BLOCK = (
    TC_OPEN
    + "\n<invoke name=\"update_tasks\">\n<parameter name=\"id\" string=\"false\">1</parameter>\n</invoke>\n"
    + TC_CLOSE
)


class TestDeepSeekV4MissingIngTags:
    """DeepSeek V4 emits missing-"-ing" tags; they must be separated."""

    def test_extract_non_streaming_separates(self):
        t, c = extract_thinking(OPEN + "deep reasoning" + CLOSE + "answer")
        assert t == "deep reasoning"
        assert c == "answer"

    def test_feed_streaming_separates(self):
        parser = ThinkingParser()
        tacc, cacc = [], []
        for chunk in [
            OPEN[:3],
            OPEN[3:],
            "deep reason",
            "ing" + CLOSE[:3],
            CLOSE[3:] + "answer",
        ]:
            t, c = parser.feed(chunk)
            if t:
                tacc.append(t)
            if c:
                cacc.append(c)
        assert "".join(tacc) == "deep reasoning"
        assert "".join(cacc) == "answer"

    def test_feed_single_chunk_separates(self):
        parser = ThinkingParser()
        t, c = parser.feed(OPEN + "reasoning" + CLOSE + "answer")
        assert t == "reasoning"
        assert c == "answer"

    def test_start_in_thinking_close_tag(self):
        # Anthropic streaming path: chat template opens thinking, model emits
        # the close tag. It must switch back to content.
        parser = ThinkingParser(start_in_thinking=True)
        t1, c1 = parser.feed("deep reasoning" + CLOSE + "answer")
        assert t1 == "deep reasoning"
        assert c1 == "answer"
        assert not parser._in_thinking


class TestHistoricalMissingIngTags:
    """Gemma/MiniMax-style tags must keep working (back-compat)."""

    def test_extract_non_streaming_still_separates(self):
        t, c = extract_thinking(OPEN + "deep reasoning" + CLOSE + "answer")
        assert t == "deep reasoning"
        assert c == "answer"

    def test_feed_streaming_still_separates(self):
        parser = ThinkingParser()
        t, c = parser.feed(OPEN + "reasoning" + CLOSE + "answer")
        assert t == "reasoning"
        assert c == "answer"

    def test_no_leak_in_content(self):
        parser = ThinkingParser()
        t, c = parser.feed(OPEN + "reasoning" + CLOSE + "answer")
        assert OPEN not in c
        assert CLOSE not in c


class TestUnclosedThinkingWithToolCall:
    """Unclosed thinking followed by a tool call must not leak to content.

    The chat template opens a thinking block in the prompt (start_in_thinking
    = True) and the model sometimes emits reasoning then a tool call WITHOUT
    closing the thinking block. The parser must NOT re-emit that reasoning as
    body content — a tool call following means the output is a tool invocation,
    so the recovery-to-content heuristic must not fire. This is the regression
    for the intermittent "tool names / reasoning leak into the visible answer"
    report.
    """

    def test_finish_does_not_recover_tool_call_thinking_to_content(self):
        parser = ThinkingParser(start_in_thinking=True)
        tacc, cacc = [], []
        for chunk in [TOOL_SUMMARY, TOOL_CALL_BLOCK]:
            td, cd = parser.feed(chunk)
            if td:
                tacc.append(td)
            if cd:
                cacc.append(cd)
        td, cd = parser.finish()
        if td:
            tacc.append(td)
        if cd:
            cacc.append(cd)

        thinking = "".join(tacc)
        content = "".join(cacc)
        # Reasoning (the tool summary) belongs in the thinking panel, never
        # re-emitted as body text just because the thinking block was unclosed.
        assert "Updated tasks" in thinking
        assert "Updated tasks" not in content

    def test_finish_does_not_recover_when_tool_marker_split_across_chunks(self):
        # The DSML envelope can span multiple streamed chunks; even then the
        # open marker must be detected and recovery suppressed.
        parser = ThinkingParser(start_in_thinking=True)
        tacc, cacc = [], []
        for chunk in [
            TOOL_SUMMARY,
            TC_OPEN[:6],
            TC_OPEN[6:],
            "\n<invoke name=\"x\">\n</invoke>\n",
            TC_CLOSE,
        ]:
            td, cd = parser.feed(chunk)
            if td:
                tacc.append(td)
            if cd:
                cacc.append(cd)
        td, cd = parser.finish()
        if td:
            tacc.append(td)
        if cd:
            cacc.append(cd)
        assert "Updated tasks" in "".join(tacc)
        assert "Updated tasks" not in "".join(cacc)
        assert parser._tool_block_seen

    def test_finish_keeps_pure_thinking_recovery(self):
        # Back-compat: when thinking is unclosed and NO tool call follows,
        # recovery still surfaces the body so the answer is not empty.
        parser = ThinkingParser(start_in_thinking=True)
        t, c = parser.feed("plain reasoning without a tool call")
        assert t == "plain reasoning without a tool call"
        assert c == ""
        td, cd = parser.finish()
        assert td == ""
        # finish() recovers the accumulated thinking as content.
        assert cd == "plain reasoning without a tool call"

    def test_extract_non_streaming_keeps_tool_call_thinking(self):
        # Malformed branch of extract_thinking: unclosed thinking followed by
        # a tool call. The reasoning must stay thinking, not be dumped into
        # regular_content (which becomes the visible body).
        raw = OPEN + TOOL_SUMMARY + TOOL_CALL_BLOCK
        t, c = extract_thinking(raw)
        assert "Updated tasks" in t
        assert "Updated tasks" not in c

    def test_extract_non_streaming_split_at_dsml_marker(self):
        # Reasoning before the DSML open marker stays thinking; the tool-call
        # envelope from the open marker onward is content for the tool parser.
        raw = OPEN + "think about it " + TOOL_CALL_BLOCK + " trailing text"
        t, c = extract_thinking(raw)
        assert t == "think about it"
        assert "trailing text" in c
        assert "think about it" not in c


class TestStreamingToolCallRouting:
    """The DS4 DSML envelope is routed to the content channel during streaming."""

    def test_envelope_routed_to_content_not_thinking(self):
        parser = ThinkingParser(start_in_thinking=True)
        t, c = parser.feed(
            "reasoning body " + TC_OPEN + "\n<invoke name=\"x\"/>\n" + TC_CLOSE
        )
        assert "reasoning body" in t
        assert "invoke" in c
        assert "invoke" not in t
        assert parser._tool_block_seen
        assert not parser._in_tool_call  # returned to normal after close

    def test_unclosed_envelope_keeps_reasoning_thinking(self):
        parser = ThinkingParser(start_in_thinking=True)
        t, c = parser.feed("reasoning " + TC_OPEN + "\n<invoke name=\"x\"/>\n")
        # After the open marker the reasoning stays in thinking; the envelope
        # body goes to content, and finish() must not recover reasoning.
        assert "reasoning" in t
        tf, cf = parser.finish()
        assert "reasoning" not in cf

    def test_closed_thinking_then_tool_call_then_text(self):
        parser = ThinkingParser(start_in_thinking=True)
        tacc, cacc = [], []
        for chunk in ["reasoning first", CLOSE, TC_OPEN, "<invoke name=\"a\"/>", TC_CLOSE, "then final text"]:
            td, cd = parser.feed(chunk)
            if td:
                tacc.append(td)
            if cd:
                cacc.append(cd)
        tf, cf = parser.finish()
        if tf:
            tacc.append(tf)
        if cf:
            cacc.append(cf)
        assert "reasoning first" in "".join(tacc)
        assert "final text" in "".join(cacc)
        assert parser._tool_block_seen


class TestRecoveryGating:
    """The unclosed-thinking recovery is gated by finish_reason, has_tools and
    tool-block detection so reasoning never leaks into content on a tool-call
    or truncated turn."""

    def _feed_plain_thinking(self, text="deep unclosed reasoning body"):
        p = ThinkingParser(start_in_thinking=True)
        t, c = p.feed(text)
        assert c == ""
        return p

    def test_normal_stop_recovery_preserved(self):
        # Back-compat: a normal stop with unclosed thinking still recovers to
        # a non-empty answer body.
        p = self._feed_plain_thinking()
        tf, cf = p.finish(finish_reason="stop", has_tools=False)
        assert cf == "deep unclosed reasoning body"

    def test_unknown_finish_reason_recovery_preserved(self):
        p = self._feed_plain_thinking()
        tf, cf = p.finish(finish_reason=None, has_tools=False)
        assert cf == "deep unclosed reasoning body"

    def test_length_truncation_does_not_recover(self):
        # A truncated turn (max_tokens hit) is never an answer: the reasoning
        # stays in the thinking channel instead of leaking into content.
        p = self._feed_plain_thinking()
        tf, cf = p.finish(finish_reason="length", has_tools=False)
        assert cf == ""

    def test_aborted_turn_does_not_recover(self):
        p = self._feed_plain_thinking()
        tf, cf = p.finish(finish_reason="abort", has_tools=False)
        assert cf == ""

    def test_has_tools_does_not_recover(self):
        # A request that supplied tools is a tool-call turn; its reasoning must
        # not be re-emitted as content even on a normal stop.
        p = self._feed_plain_thinking()
        tf, cf = p.finish(finish_reason="stop", has_tools=True)
        assert cf == ""

    def test_tool_block_seen_does_not_recover(self):
        # Even without a tools request, a DSML envelope means tool invocation.
        p = ThinkingParser(start_in_thinking=True)
        p.feed("reasoning " + TC_OPEN + "\n<invoke name=\"x\"/>\n" + TC_CLOSE)
        tf, cf = p.finish(finish_reason="stop", has_tools=False)
        assert cf == ""

    def test_length_truncation_with_tools_does_not_recover(self):
        p = self._feed_plain_thinking()
        tf, cf = p.finish(finish_reason="length", has_tools=True)
        assert cf == ""


class TestPrematureCloseSecondReasoningGuard:
    """DS4 second-reasoning guard: premature close must not leak reasoning.

    DeepSeek V4 sometimes emits the close tag BEFORE opening thinking (long
    multi-turn tool histories). The parser, starting in thinking mode, sees the
    premature close and must NOT flip to content mode — the reasoning that
    follows would leak into the visible content channel. With tools, the guard
    holds the text until a tool call or a second close decides the channel.
    """

    def test_premature_close_reasoning_tool_no_leak(self):
        parser = ThinkingParser(start_in_thinking=True, guard_second_reasoning=True)
        tacc, cacc = [], []
        for chunk in [
            CLOSE,  # premature close
            "Let me trace carefully, step by step... ",
            TC_OPEN + "\n<invoke name=\"x\">",
        ]:
            t, c = parser.feed(chunk)
            if t:
                tacc.append(t)
            if c:
                cacc.append(c)
        t, c = parser.finish(finish_reason="stop", has_tools=True)
        if t:
            tacc.append(t)
        if c:
            cacc.append(c)
        assert "".join(tacc) == "Let me trace carefully, step by step... "
        assert "".join(cacc) == TC_OPEN + "\n<invoke name=\"x\">"
        assert "trace" in "".join(tacc)  # reasoning stayed in the thinking channel

    def test_premature_close_reasoning_no_tool_holds_until_finish(self):
        parser = ThinkingParser(start_in_thinking=True, guard_second_reasoning=True)
        t, c = parser.feed(CLOSE)
        t2, c2 = parser.feed("reasoning text")
        tf, cf = parser.finish(finish_reason="stop", has_tools=True)
        assert t2 == ""  # held
        assert "".join((t, t2, tf)) == "reasoning text"  # stays in thinking
        assert cf == ""

    def test_normal_sequence_guard_off_still_separates(self):
        parser = ThinkingParser(start_in_thinking=True, guard_second_reasoning=False)
        t, c = parser.feed(OPEN + "reasoning")
        t2, c2 = parser.feed(CLOSE + "answer")
        assert "".join((t, t2)) == "reasoning"
        assert "".join((c, c2)) == "answer"

    def test_normal_sequence_guard_on_still_separates(self):
        # With the guard enabled the answer after a legitimate close is held
        # until the stream ends (ds4-server behaviour) and is then routed to the
        # content channel — it must never be re-classified as reasoning.
        parser = ThinkingParser(start_in_thinking=True, guard_second_reasoning=True)
        t, c = parser.feed(OPEN + "reasoning")
        t2, c2 = parser.feed(CLOSE + "answer")
        assert "".join((t, t2)) == "reasoning"
        assert "".join((c, c2)) == ""  # answer held by the guard
        tf, cf = parser.finish(finish_reason="stop", has_tools=True)
        assert tf == ""  # held answer is content, not reasoning
        assert "".join((c, c2, cf)) == "answer"

    def test_second_reasoning_after_legitimate_close_stays_thinking(self):
        # DeepSeek V4 often emits a SECOND reasoning pass after the close tag
        # without re-opening thinking: open-reasoning-close-pass2-close-tool.
        # The second pass must stay in the thinking channel, never leak into the
        # visible content channel. Mirrors ds4-server's reroutes_second_reasoning.
        parser = ThinkingParser(start_in_thinking=True, guard_second_reasoning=True)
        tacc, cacc = [], []
        for chunk in [
            OPEN + "first pass",
            CLOSE,
            "second pass draft ",
            CLOSE,
            TC_OPEN + "\n<invoke name=\"x\">",
        ]:
            t, c = parser.feed(chunk)
            if t:
                tacc.append(t)
            if c:
                cacc.append(c)
        tf, cf = parser.finish(finish_reason="stop", has_tools=True)
        if tf:
            tacc.append(tf)
        if cf:
            cacc.append(cf)
        thinking = "".join(tacc)
        content = "".join(cacc)
        assert "first pass" in thinking
        assert "second pass draft" in thinking  # second reasoning stayed thinking
        assert "second pass draft" not in content  # no leak
        assert "invoke" in content  # tool envelope is content

    def test_second_reasoning_without_second_close_stays_thinking(self):
        # Same second-pass pattern but the model jumps straight to the tool call
        # (no second close). The held reasoning must still stay in the thinking
        # channel when the tool call resolves the guard.
        parser = ThinkingParser(start_in_thinking=True, guard_second_reasoning=True)
        tacc, cacc = [], []
        for chunk in [
            OPEN + "first pass",
            CLOSE,
            "second pass draft ",
            TC_OPEN + "\n<invoke name=\"y\">",
        ]:
            t, c = parser.feed(chunk)
            if t:
                tacc.append(t)
            if c:
                cacc.append(c)
        tf, cf = parser.finish(finish_reason="stop", has_tools=True)
        if tf:
            tacc.append(tf)
        if cf:
            cacc.append(cf)
        thinking = "".join(tacc)
        content = "".join(cacc)
        assert "first pass" in thinking
        assert "second pass draft" in thinking
        assert "second pass draft" not in content
        assert "invoke" in content

    def test_answer_before_tool_after_legitimate_close_never_leaks(self):
        # A text passage between a legitimate close and a tool call is held by
        # the guard. The no-leak rule routes it to the thinking channel (never
        # to content), so no reasoning can ever leak into the visible body. The
        # tradeoff: a rare genuine answer-before-tool is surfaced in the thinking
        # panel instead of the content panel.
        parser = ThinkingParser(start_in_thinking=True, guard_second_reasoning=True)
        tacc, cacc = [], []
        for chunk in [
            OPEN + "reasoning",
            CLOSE,
            "Here is the answer: 42",
            TC_OPEN + "\n<invoke name=\"z\">",
        ]:
            t, c = parser.feed(chunk)
            if t:
                tacc.append(t)
            if c:
                cacc.append(c)
        tf, cf = parser.finish(finish_reason="stop", has_tools=True)
        if tf:
            tacc.append(tf)
        if cf:
            cacc.append(cf)
        content = "".join(cacc)
        thinking = "".join(tacc)
        assert "reasoning" in thinking
        assert "answer: 42" in thinking  # held text stays out of content
        assert "reasoning" not in content  # no thinking leak
        assert "invoke" in content

    def test_premature_close_second_close_answer_tool(self):
        # Premature close -> reasoning -> second close -> answer -> tool. The
        # untagged reasoning ends at the second close and stays in thinking; the
        # answer and tool after it are content.
        parser = ThinkingParser(start_in_thinking=True, guard_second_reasoning=True)
        tacc, cacc = [], []
        for chunk in [
            CLOSE,
            "untagged reasoning ",
            CLOSE,
            "answer text ",
            TC_OPEN + "\n<invoke name=\"w\">",
        ]:
            t, c = parser.feed(chunk)
            if t:
                tacc.append(t)
            if c:
                cacc.append(c)
        tf, cf = parser.finish(finish_reason="stop", has_tools=True)
        if tf:
            tacc.append(tf)
        if cf:
            cacc.append(cf)
        thinking = "".join(tacc)
        content = "".join(cacc)
        assert "untagged reasoning" in thinking
        assert "untagged reasoning" not in content
        assert "answer text" in content
        assert "invoke" in content
