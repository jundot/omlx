# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma 4 message extraction."""

from __future__ import annotations

from omlx.adapter.gemma4 import (
    Gemma4OutputParserSession,
    _strip_thinking,
    extract_gemma4_messages,
)
from omlx.api.openai_models import Message


def _tool_call_dict(id: str, name: str, args: str = "{}") -> dict:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


def _assistant_with_calls(*calls) -> Message:
    return Message(role="assistant", content="", tool_calls=list(calls))


def _tool_result(id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=id)


class TestExtractGemma4Messages:
    def test_plain_messages_pass_through(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        result = extract_gemma4_messages(messages)
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}

    def test_tool_result_folded_onto_model_turn(self):
        """Single tool result is attached to the same assistant message as tool_calls."""
        messages = [
            Message(role="user", content="What's the weather?"),
            _assistant_with_calls(_tool_call_dict("c1", "get_weather")),
            _tool_result("c1", "sunny"),
        ]
        result = extract_gemma4_messages(messages)
        # user + assistant(tool_calls + tool_responses)
        assert len(result) == 2
        tr_msg = result[1]
        assert tr_msg["role"] == "assistant"
        assert "tool_calls" in tr_msg
        assert tr_msg["tool_responses"] == [
            {"name": "get_weather", "response": "sunny"}
        ]

    def test_function_name_resolved_from_tool_call_id(self):
        messages = [
            _assistant_with_calls(
                _tool_call_dict("c1", "search"),
                _tool_call_dict("c2", "calculate"),
            ),
            _tool_result("c2", "42"),
            _tool_result("c1", "results"),
        ]
        result = extract_gemma4_messages(messages)
        # tool_responses attached to the same assistant message
        tr_msg = result[0]
        names = {tr["name"] for tr in tr_msg["tool_responses"]}
        assert names == {"calculate", "search"}

    def test_multiple_tool_results_batched(self):
        """Multiple consecutive tool results land on the same assistant message."""
        messages = [
            _assistant_with_calls(
                _tool_call_dict("c1", "fn_a"),
                _tool_call_dict("c2", "fn_b"),
            ),
            _tool_result("c1", "result_a"),
            _tool_result("c2", "result_b"),
        ]
        result = extract_gemma4_messages(messages)
        # tool_responses on the same message as tool_calls
        tr_msg = result[0]
        assert "tool_calls" in tr_msg
        assert len(tr_msg["tool_responses"]) == 2
        assert tr_msg["tool_responses"][0] == {"name": "fn_a", "response": "result_a"}
        assert tr_msg["tool_responses"][1] == {"name": "fn_b", "response": "result_b"}

    def test_json_response_parsed_to_dict(self):
        """JSON-parseable tool result content becomes a dict."""
        messages = [
            _assistant_with_calls(_tool_call_dict("c1", "fn")),
            _tool_result("c1", '{"value": 42}'),
        ]
        result = extract_gemma4_messages(messages)
        response = result[0]["tool_responses"][0]["response"]
        assert response == {"value": 42}

    def test_non_json_response_stays_string(self):
        messages = [
            _assistant_with_calls(_tool_call_dict("c1", "fn")),
            _tool_result("c1", "plain text result"),
        ]
        result = extract_gemma4_messages(messages)
        assert result[0]["tool_responses"][0]["response"] == "plain text result"

    def test_orphaned_tool_result_fallback_to_tool_call_id_as_name(self):
        """Tool result with no preceding assistant turn uses tool_call_id as name."""
        messages = [_tool_result("call_xyz", "orphaned")]
        result = extract_gemma4_messages(messages)
        assert result[0]["tool_responses"][0]["name"] == "call_xyz"

    def test_tool_calls_preserved_on_assistant_turn(self):
        """Assistant turn tool_calls are kept so the template renders them."""
        messages = [
            _assistant_with_calls(_tool_call_dict("c1", "do_thing", '{"x": 1}'))
        ]
        result = extract_gemma4_messages(messages)
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "do_thing"
        # arguments should be parsed to dict
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}

    def test_multi_turn_agentic_conversation(self):
        """Full agentic loop: user → tool call → result → follow-up answer."""
        messages = [
            Message(role="user", content="Look it up"),
            _assistant_with_calls(_tool_call_dict("c1", "search")),
            _tool_result("c1", "found it"),
            Message(role="assistant", content="Here is what I found."),
        ]
        result = extract_gemma4_messages(messages)
        assert result[0] == {"role": "user", "content": "Look it up"}
        assert "tool_calls" in result[1]
        assert result[1]["tool_responses"][0]["name"] == "search"
        assert result[2] == {"role": "assistant", "content": "Here is what I found."}

    def test_system_message_preserved(self):
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
        ]
        result = extract_gemma4_messages(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_developer_role_normalised_to_system(self):
        messages = [Message(role="developer", content="Be concise.")]
        result = extract_gemma4_messages(messages)
        assert result[0]["role"] == "system"

    def test_image_url_preserved_in_user_message(self):
        """User messages with image_url parts keep them for VLM processing."""
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            ),
        ]
        result = extract_gemma4_messages(messages)
        assert isinstance(result[0]["content"], list)
        types = [p["type"] for p in result[0]["content"]]
        assert "image_url" in types
        assert "text" in types

    def test_text_only_content_list_flattened(self):
        """User messages with text-only content list are flattened to string."""
        messages = [
            Message(
                role="user",
                content=[{"type": "text", "text": "hello world"}],
            ),
        ]
        result = extract_gemma4_messages(messages)
        assert isinstance(result[0]["content"], str)
        assert result[0]["content"] == "hello world"

    def test_consecutive_user_with_image_merges_without_crash(self):
        """Consecutive user messages where one has images should not crash.

        Regression test for GitHub issue #671.
        """
        messages = [
            Message(
                role="user",
                content=[
                    {"type": "text", "text": "Look at this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            ),
            Message(role="user", content="What is it?"),
        ]
        result = extract_gemma4_messages(messages)
        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, list)
        types = [p["type"] for p in content]
        assert "image_url" in types

    def test_multi_turn_rule_strips_prior_thought_block(self):
        """Per Gemma 4 docs: only the final visible answer is kept in chat
        history; prior thought blocks must not be fed back on the next turn.
        """
        messages = [
            Message(role="user", content="What is 1+1?"),
            Message(
                role="assistant",
                content="<think>\nLet me compute.\n</think>\n2",
            ),
            Message(role="user", content="What is 2+2?"),
        ]
        result = extract_gemma4_messages(messages)
        assert result[1] == {"role": "assistant", "content": "2"}

    def test_multi_turn_strips_raw_channel_form(self):
        """Clients that preserve the protocol form (raw channel markers in
        assistant content) also need their thought blocks stripped."""
        messages = [
            Message(
                role="assistant",
                content="<|channel>thought\nreasoning here\n<channel|>final",
            ),
        ]
        result = extract_gemma4_messages(messages)
        assert result[0]["content"] == "final"

    def test_multi_turn_preserves_inline_think_mention(self):
        """An assistant explaining the protocol must not have its examples
        gutted on subsequent turns."""
        explanation = "You write `<think>like this</think>` to enable reasoning."
        messages = [Message(role="assistant", content=explanation)]
        result = extract_gemma4_messages(messages)
        assert result[0]["content"] == explanation

    def test_strip_keeps_tool_calls(self):
        """Stripping leading thought from assistant content must not drop
        the structured ``tool_calls`` field."""
        msg = Message(
            role="assistant",
            content="<think>\nplanning\n</think>\n",
            tool_calls=[_tool_call_dict("c1", "search")],
        )
        result = extract_gemma4_messages([msg])
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"


class TestStripThinking:
    """``_strip_thinking`` removes leading thought blocks only."""

    def test_strips_canonical_channel_block(self):
        """Per Gemma 4 spec: ``<|channel>thought\\n[reasoning]<channel|>[answer]``."""
        text = "<|channel>thought\ninternal reasoning\n<channel|>the answer"
        assert _strip_thinking(text) == "the answer"

    def test_strips_empty_channel_block_per_spec(self):
        """Per Gemma 4 spec: thinking-disabled larger models still emit an
        empty thought block before the final answer."""
        text = "<|channel>thought\n<channel|>the answer"
        assert _strip_thinking(text) == "the answer"

    def test_strips_rendered_think_block(self):
        text = "<think>\nreasoning\n</think>\nthe answer"
        assert _strip_thinking(text) == "the answer"

    def test_strips_multiple_consecutive_blocks(self):
        text = "<think>a</think>\n<think>b</think>\nanswer"
        assert _strip_thinking(text) == "answer"

    def test_strips_mixed_channel_and_think_blocks(self):
        text = "<|channel>x<channel|><think>y</think>answer"
        assert _strip_thinking(text) == "answer"

    def test_preserves_inline_think_mention(self):
        text = "You write `<think>like this</think>` to enable reasoning."
        assert _strip_thinking(text) == text

    def test_preserves_inline_channel_mention(self):
        text = "The `<|channel>` token marks reasoning."
        assert _strip_thinking(text) == text

    def test_handles_leading_whitespace(self):
        assert _strip_thinking("   <think>x</think>answer") == "answer"

    def test_empty_string_unchanged(self):
        assert _strip_thinking("") == ""

    def test_non_string_passthrough(self):
        assert _strip_thinking(None) is None
        assert _strip_thinking(42) == 42


class _FakeTokenizer:
    """Tokenizer stub that bypasses the streaming detokenizer path."""

    detokenizer = None

    def decode(self, ids):
        return ""


class TestGemma4OutputParserSession:
    """Output parser converts Gemma 4 channel markers to ``<think>`` tags."""

    def _make_session(self):
        return Gemma4OutputParserSession(_FakeTokenizer())

    def test_canonical_thought_block_per_spec(self):
        """``<|channel>thought\\n[reasoning]<channel|>[answer]`` per Gemma 4 spec."""
        sess = self._make_session()
        result = sess._consume_text(
            "<|channel>thought\nreasoning\n<channel|>final answer",
            final=True,
        )
        assert result.visible_text == "<think>\nreasoning\n</think>\nfinal answer"
        assert sess._in_thought is False

    def test_empty_thought_block_per_spec(self):
        """Thinking-disabled larger models emit an empty thought block per spec."""
        sess = self._make_session()
        result = sess._consume_text(
            "<|channel>thought\n<channel|>the answer", final=True
        )
        assert result.visible_text == "<think>\n</think>\nthe answer"

    def test_no_newline_open_marker_defensive(self):
        """``<|channel>thought<channel|>`` with no newline is recovered via
        the bare-marker fallback, with the ``thought`` keyword absorbed."""
        sess = self._make_session()
        result = sess._consume_text(
            "<|channel>thought<channel|>after", final=True
        )
        assert result.visible_text == "<think>\n</think>\nafter"

    def test_bare_open_with_non_thought_channel(self):
        """A bare ``<|channel>X<channel|>`` (unknown channel name) is wrapped
        defensively as a thought block rather than leaking the markers."""
        sess = self._make_session()
        result = sess._consume_text(
            "<|channel>X<channel|>after", final=True
        )
        assert result.visible_text == "<think>\nX</think>\nafter"

    def test_streaming_canonical_split_at_marker_boundary(self):
        """Open marker arriving across two chunks must still match canonical."""
        sess = self._make_session()
        r1 = sess._consume_text("<|channel>")
        # Streaming defer: nothing emitted yet because the bare match could
        # still extend to the canonical form.
        assert r1.visible_text == ""
        r2 = sess._consume_text(
            "thought\nreasoning\n<channel|>answer", final=True
        )
        assert r1.visible_text + r2.visible_text == (
            "<think>\nreasoning\n</think>\nanswer"
        )

    def test_streaming_defer_prefers_canonical_over_bare(self):
        """When ``<|channel>thought`` arrives without a newline at the chunk
        boundary, the parser must defer rather than commit the bare match."""
        sess = self._make_session()
        r1 = sess._consume_text("<|channel>thought")
        assert r1.visible_text == ""
        r2 = sess._consume_text("\nreasoning\n<channel|>answer", final=True)
        assert r1.visible_text + r2.visible_text == (
            "<think>\nreasoning\n</think>\nanswer"
        )

    def test_stray_close_marker_dropped(self):
        """A ``<channel|>`` outside a thought block is absorbed silently."""
        sess = self._make_session()
        result = sess._consume_text("hello<channel|>world", final=True)
        assert result.visible_text == "helloworld"

    def test_turn_end_marker_dropped(self):
        sess = self._make_session()
        result = sess._consume_text("done<turn|>", final=True)
        assert result.visible_text == "done"

    def test_tool_response_markers_dropped(self):
        sess = self._make_session()
        result = sess._consume_text(
            "<|tool_response>{\"x\":1}<tool_response|>after", final=True
        )
        assert result.visible_text == '{"x":1}after'

    def test_finalize_closes_orphan_thought(self):
        """A thought block left open at end-of-stream is closed in finalize."""
        sess = self._make_session()
        first = sess._consume_text("<|channel>thought\nincomplete reasoning")
        final = sess.finalize()
        combined = first.visible_text + final.visible_text
        assert "<think>\n" in combined
        assert combined.endswith("</think>\n")
        assert sess._in_thought is False

    def test_double_bare_open_does_not_re_emit_think(self):
        """The user's reported double ``<|channel><|channel>`` artefact: the
        second open while already inside a thought block must not re-emit a
        nested ``<think>`` opener."""
        sess = self._make_session()
        result = sess._consume_text(
            "<|channel>thought\nfirst<|channel>second\n<channel|>after",
            final=True,
        )
        # Exactly one <think> opener despite two <|channel>... opens.
        assert result.visible_text.count("<think>\n") == 1
        assert result.visible_text.count("</think>\n") == 1
        assert result.visible_text.endswith("after")
