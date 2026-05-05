# SPDX-License-Identifier: Apache-2.0
"""Tests for ``render_harmony_prompt`` (omlx.adapter.harmony).

The helper produces a Harmony-format prompt from OpenAI-style chat messages,
used as a fallback when a gpt-oss model's tokenizer lacks a ``chat_template``.
"""

import json

import pytest

from omlx.adapter.harmony import render_harmony_prompt


# ── Basic rendering ───────────────────────────────────────────────────


class TestBasicRendering:
    def test_user_message_only(self):
        prompt = render_harmony_prompt(
            [{"role": "user", "content": "hi"}]
        )
        assert "<|start|>system" in prompt
        assert "<|start|>user<|message|>hi<|end|>" in prompt
        # Generation prompt ends expecting assistant continuation.
        assert prompt.rstrip().endswith("<|start|>assistant")

    def test_system_becomes_developer_instructions(self):
        prompt = render_harmony_prompt(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
            ]
        )
        assert "<|start|>developer" in prompt
        # Instructions section carries the user's system text.
        assert "Be terse." in prompt

    def test_multiple_system_messages_concatenated(self):
        prompt = render_harmony_prompt(
            [
                {"role": "system", "content": "A."},
                {"role": "system", "content": "B."},
                {"role": "user", "content": "hi"},
            ]
        )
        assert "A." in prompt and "B." in prompt

    def test_content_as_list_of_blocks(self):
        """OpenAI clients may send content as ``[{type:'text', text:'...'}]``."""
        prompt = render_harmony_prompt(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ],
                }
            ]
        )
        assert "hello world" in prompt


# ── Reasoning effort + conversation date ──────────────────────────────


class TestTemplateKwargs:
    def test_reasoning_effort_high(self):
        prompt = render_harmony_prompt(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "high"},
        )
        assert "Reasoning: high" in prompt

    def test_reasoning_effort_low(self):
        prompt = render_harmony_prompt(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        assert "Reasoning: low" in prompt

    def test_conversation_start_date(self):
        prompt = render_harmony_prompt(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"conversation_start_date": "2026-01-15"},
        )
        assert "2026-01-15" in prompt

    def test_unknown_reasoning_effort_ignored(self):
        prompt = render_harmony_prompt(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "ludicrous"},
        )
        # Renders without raising; default effort still emitted.
        assert "<|start|>user" in prompt


# ── Tools ─────────────────────────────────────────────────────────────


class TestToolRendering:
    SEARCH_TOOL = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the corpus",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }

    def test_tool_rendered_into_developer_block(self):
        prompt = render_harmony_prompt(
            [{"role": "user", "content": "hi"}],
            tools=[self.SEARCH_TOOL],
        )
        assert "<|start|>developer" in prompt
        # openai-harmony emits a ``namespace functions {`` block and a
        # ``type search = (...) => any;`` entry per tool.  Check both so
        # the test fails if the tool fell out of the developer block.
        assert "namespace functions" in prompt
        assert "type search" in prompt
        assert "Search the corpus" in prompt

    def test_tools_without_system_do_not_emit_instructions_heading(self):
        """Developer block should carry ``# Tools`` but not ``# Instructions``
        when the caller supplied tools but no system message."""
        prompt = render_harmony_prompt(
            [{"role": "user", "content": "hi"}],
            tools=[self.SEARCH_TOOL],
        )
        assert "<|start|>developer" in prompt
        assert "# Tools" in prompt
        assert "# Instructions" not in prompt

    def test_raw_function_spec_accepted(self):
        """Some callers pass ``{name, description, parameters}`` without the
        ``{"type": "function", "function": {...}}`` wrapper."""
        prompt = render_harmony_prompt(
            [{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "grep",
                    "description": "Grep",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"],
                    },
                }
            ],
        )
        assert "grep" in prompt

    def test_tool_without_name_dropped(self):
        prompt = render_harmony_prompt(
            [{"role": "user", "content": "hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "description": "should-not-appear-sentinel-xyz",
                    },
                }
            ],
        )
        # Rendering succeeds and does not inject the bogus tool.
        assert "should-not-appear-sentinel-xyz" not in prompt


# ── Assistant tool_calls + tool responses roundtrip ───────────────────


class TestToolCallRoundtrip:
    def test_assistant_tool_call_and_tool_response(self):
        prompt = render_harmony_prompt(
            [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "get_weather",
                    "tool_call_id": "call_1",
                    "content": json.dumps({"temp_c": 14}),
                },
            ]
        )
        # Assistant commentary channel addressed to the function.
        assert "to=functions.get_weather" in prompt
        assert "commentary" in prompt
        assert '{"city": "Paris"}' in prompt
        # Tool response is emitted as a commentary message authored by the tool.
        assert "functions.get_weather" in prompt
        assert '"temp_c": 14' in prompt

    def test_tool_response_name_resolved_from_tool_call_id(self):
        """When a ``tool`` message omits ``name``, the function name is
        recovered from the preceding assistant tool_call with the matching id."""
        prompt = render_harmony_prompt(
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_42",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": {"x": 1},
                            },
                        }
                    ],
                },
                # No ``name`` key — must be resolved via tool_call_id.
                {"role": "tool", "tool_call_id": "call_42", "content": "done"},
            ]
        )
        assert "functions.lookup" in prompt
        # The resolved name should appear both on the call and the response.
        assert prompt.count("functions.lookup") >= 2

    def test_tool_response_dropped_when_name_unresolvable(self):
        """A ``tool`` message with no ``name`` and no matching tool_call_id
        is skipped rather than given a fabricated function name."""
        prompt = render_harmony_prompt(
            [
                {"role": "user", "content": "q"},
                {"role": "tool", "tool_call_id": "nowhere", "content": "orphan"},
            ]
        )
        assert "orphan" not in prompt
        assert "functions.nowhere" not in prompt

    def test_tool_call_arguments_accept_string(self):
        """Arguments may arrive pre-stringified (OpenAI wire format)."""
        prompt = render_harmony_prompt(
            [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "x"}',
                            }
                        }
                    ],
                },
            ]
        )
        assert 'to=functions.search' in prompt
        assert '{"query": "x"}' in prompt


# ── Robustness ────────────────────────────────────────────────────────


class TestRobustness:
    def test_empty_messages_does_not_crash(self):
        prompt = render_harmony_prompt([])
        # Minimum viable Harmony prompt: system header + assistant lead-in.
        assert "<|start|>system" in prompt
        assert prompt.rstrip().endswith("<|start|>assistant")

    def test_non_dict_messages_skipped(self):
        prompt = render_harmony_prompt(
            ["garbage", None, {"role": "user", "content": "ok"}]
        )
        assert "ok" in prompt

    def test_unknown_role_skipped(self):
        prompt = render_harmony_prompt(
            [
                {"role": "stranger", "content": "drop me"},
                {"role": "user", "content": "keep me"},
            ]
        )
        assert "drop me" not in prompt
        assert "keep me" in prompt
